# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Benchmark for torchstore weight synchronization (put_state_dict_batch / get_state_dict_batch).

Measures batched put/get state_dict performance using torchstore's public API.

Usage:
    # Default (Qwen3-4B, 5 iterations):
    pytest tests/bench_weight_sync.py -v -s

    # Custom model and iterations via environment variables:
    BENCH_MODEL=Qwen/Qwen3-30B-A3B BENCH_ITER=3 pytest tests/bench_weight_sync.py -v -s

Environment variables:
    BENCH_MODEL:       HuggingFace model name (default: Qwen/Qwen3-4B)
    BENCH_ITER:        Number of timed iterations (default: 5)
    BENCH_WARMUP:      Number of warmup iterations (default: 1)
    BENCH_TRANSPORT:   Transport type (default: SharedMemory)
    BENCH_CONCURRENCY: Max concurrent storage operations (default: 8)
"""

import os
import statistics
import time

import pytest
import torch
import torchstore as ts
from monarch.actor import Actor, current_rank, endpoint
from torch.distributed.checkpoint._nested_dict import flatten_state_dict
from torchstore.transport import TransportType
from torchstore.utils import spawn_actors


class BenchmarkActor(Actor):
    """Actor that creates state dict internally and benchmarks put/get operations."""

    def __init__(self, model_name, iterations, warmup, max_concurrent):
        self.rank = current_rank().rank
        os.environ["LOCAL_RANK"] = str(self.rank)
        self.model_name = model_name
        self.iterations = iterations
        self.warmup = warmup
        self.max_concurrent = max_concurrent

    @endpoint
    async def run_benchmark(self):
        """Run the full benchmark inside the actor process."""
        # Pre-cache the torchstore client before large allocations
        await ts.client()

        # Load real HuggingFace model
        from transformers import AutoModelForCausalLM

        print(f"[Actor] Loading HuggingFace model: {self.model_name}...")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            token=os.environ.get("HF_TOKEN"),
            torch_dtype=torch.bfloat16,
        )
        state_dict = {"model": model.state_dict()}
        del model

        # Calculate size info
        flat_sd, _ = flatten_state_dict(state_dict)
        num_params = len(flat_sd)
        total_bytes = sum(
            t.numel() * t.element_size()
            for t in flat_sd.values()
            if isinstance(t, torch.Tensor)
        )
        total_gb = total_bytes / (1024**3)
        print(f"[Actor] State dict: {num_params} parameters, {total_gb:.2f} GB")

        # Warmup
        print(f"[Actor] Running {self.warmup} warmup iteration(s)...")
        for i in range(self.warmup):
            await ts.put_state_dict_batch(
                state_dict, "v0", max_concurrent=self.max_concurrent
            )
            await ts.get_state_dict_batch(
                "v0", state_dict, max_concurrent=self.max_concurrent
            )
        print("[Actor] Warmup complete.")

        # Timed iterations
        print(f"[Actor] Running {self.iterations} timed iteration(s)...")
        put_times = []
        get_times = []

        for i in range(self.iterations):
            key = "v0"

            put_start = time.perf_counter()
            await ts.put_state_dict_batch(
                state_dict, key, max_concurrent=self.max_concurrent
            )
            put_elapsed = time.perf_counter() - put_start
            put_times.append(put_elapsed)

            get_start = time.perf_counter()
            await ts.get_state_dict_batch(
                key, state_dict, max_concurrent=self.max_concurrent
            )
            get_elapsed = time.perf_counter() - get_start
            get_times.append(get_elapsed)

            put_tp = total_gb / put_elapsed if put_elapsed > 0 else 0
            get_tp = total_gb / get_elapsed if get_elapsed > 0 else 0
            print(
                f"  Iter {i+1}: put={put_elapsed:.3f}s ({put_tp:.2f} GB/s), "
                f"get={get_elapsed:.3f}s ({get_tp:.2f} GB/s), "
                f"total={put_elapsed + get_elapsed:.3f}s"
            )

        # Verification: assert the retrieved state dict matches the original
        print("[Actor] Verifying correctness of last get...")
        await ts.put_state_dict_batch(
            state_dict, "v0", max_concurrent=self.max_concurrent
        )
        retrieved_sd = await ts.get_state_dict_batch(
            "v0", max_concurrent=self.max_concurrent
        )
        flat_original, _ = flatten_state_dict(state_dict)
        flat_retrieved, _ = flatten_state_dict(retrieved_sd)
        assert len(flat_original) == len(
            flat_retrieved
        ), f"Key count mismatch: {len(flat_original)} vs {len(flat_retrieved)}"
        for k in flat_original:
            assert k in flat_retrieved, f"Missing key in retrieved state dict: {k}"
            if isinstance(flat_original[k], torch.Tensor):
                assert torch.equal(
                    flat_original[k], flat_retrieved[k]
                ), f"Tensor mismatch for key {k}"
            else:
                assert (
                    flat_original[k] == flat_retrieved[k]
                ), f"Value mismatch for key {k}"
        print("[Actor] Verification passed.")

        return {
            "num_params": num_params,
            "total_bytes": total_bytes,
            "put_times": put_times,
            "get_times": get_times,
        }


def print_results(results):
    """Print formatted benchmark results."""
    num_params = results["num_params"]
    total_bytes = results["total_bytes"]
    put_times = results["put_times"]
    get_times = results["get_times"]
    total_mb = total_bytes / (1024 * 1024)
    total_gb = total_bytes / (1024**3)

    round_trip_times = [p + g for p, g in zip(put_times, get_times)]

    def calc_stats(times):
        return {
            "mean": statistics.mean(times),
            "median": statistics.median(times),
            "stdev": statistics.stdev(times) if len(times) > 1 else 0,
            "min": min(times),
            "max": max(times),
        }

    put_stats = calc_stats(put_times)
    get_stats = calc_stats(get_times)
    rt_stats = calc_stats(round_trip_times)

    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"Parameters:       {num_params}")
    print(f"State dict size:  {total_mb:.1f} MB ({total_gb:.2f} GB)")
    print(f"{'='*70}")
    print(
        f"{'Metric':<20} {'Mean':>10} {'Median':>10} "
        f"{'Stdev':>10} {'Min':>10} {'Max':>10}"
    )
    print(f"{'-'*70}")

    for name, s in [
        ("put_state_dict_batch", put_stats),
        ("get_state_dict_batch", get_stats),
        ("round_trip", rt_stats),
    ]:
        print(
            f"{name:<20} {s['mean']:>9.3f}s {s['median']:>9.3f}s "
            f"{s['stdev']:>9.3f}s {s['min']:>9.3f}s {s['max']:>9.3f}s"
        )

    print(f"\n{'Throughput':<20} {'Mean':>10} {'Peak':>10}")
    print(f"{'-'*40}")
    print(
        f"{'put (GB/s)':<20} {total_gb/put_stats['mean']:>9.2f}  "
        f"{total_gb/put_stats['min']:>9.2f}"
    )
    print(
        f"{'get (GB/s)':<20} {total_gb/get_stats['mean']:>9.2f}  "
        f"{total_gb/get_stats['min']:>9.2f}"
    )
    print(
        f"{'round_trip (GB/s)':<20} {total_gb/rt_stats['mean']:>9.2f}  "
        f"{total_gb/rt_stats['min']:>9.2f}"
    )
    print(f"{'='*70}")


@pytest.mark.asyncio
async def test_benchmark_weight_sync():
    """Benchmark put_state_dict_batch / get_state_dict_batch performance."""
    ts.init_logging()

    model_name = os.environ.get("BENCH_MODEL", "Qwen/Qwen3-4B")
    iterations = int(os.environ.get("BENCH_ITER", "5"))
    warmup = int(os.environ.get("BENCH_WARMUP", "1"))
    transport_name = os.environ.get("BENCH_TRANSPORT", "SharedMemory")
    max_concurrent = int(os.environ.get("BENCH_CONCURRENCY", "8"))

    transport_map = {t.name: t for t in TransportType}
    assert transport_name in transport_map, (
        f"Unknown transport: {transport_name}. "
        f"Available: {', '.join(t.name for t in TransportType if t.name != 'Unset')}"
    )
    transport_type = transport_map[transport_name]

    print(f"\n{'='*70}")
    print("WEIGHT SYNC BENCHMARK")
    print(f"{'='*70}")
    print(f"Model:            {model_name}")
    print(f"Transport:        {transport_name}")
    print(f"Max concurrent:   {max_concurrent}")
    print(f"Iterations:       {iterations} (+ {warmup} warmup)")
    print(f"{'='*70}\n")

    await ts.initialize(
        num_storage_volumes=1,
        strategy=ts.LocalRankStrategy(transport_type),
    )

    try:
        actor = await spawn_actors(
            1,
            BenchmarkActor,
            "bench_actor",
            model_name=model_name,
            iterations=iterations,
            warmup=warmup,
            max_concurrent=max_concurrent,
        )

        results = await actor.run_benchmark.call_one()
        print_results(results)

    finally:
        await ts.shutdown()


if __name__ == "__main__":
    ts.init_logging()
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"] + sys.argv[1:]))
