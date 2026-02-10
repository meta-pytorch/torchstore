# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Benchmark for torchstore weight synchronization (put_state_dict / get_state_dict).

Measures put_state_dict and get_state_dict performance using torchstore's public API.
Works on both the feature/auto-batch-state-dict branch (batching) and main (sequential).

Usage:
    # Default (synthetic-small / Qwen3-4B dimensions, 5 iterations):
    pytest benchmarks/bench_weight_sync.py -v -s

    # Custom model and iterations via environment variables:
    BENCH_MODEL=synthetic-large BENCH_ITER=3 pytest benchmarks/bench_weight_sync.py -v -s

    # With real HuggingFace model (requires HF_TOKEN):
    BENCH_MODEL=qwen3-4b BENCH_USE_HF=1 pytest benchmarks/bench_weight_sync.py -v -s

Environment variables:
    BENCH_MODEL:   synthetic-small (default), synthetic-large, qwen3-4b, qwen3-30b
    BENCH_ITER:    Number of timed iterations (default: 5)
    BENCH_WARMUP:  Number of warmup iterations (default: 1)
    BENCH_USE_HF:  Set to 1 to load real HuggingFace model (requires HF_TOKEN)
"""

import os
import statistics
import time
from logging import getLogger

import pytest
import torch
import torchstore as ts
from monarch.actor import Actor, current_rank, endpoint
from torch.distributed.checkpoint._nested_dict import flatten_state_dict
from torchstore.transport import TransportType
from torchstore.utils import spawn_actors

logger = getLogger(__name__)

# Model configs: (hidden_size, intermediate_size, vocab_size, num_layers)
MODEL_CONFIGS = {
    "synthetic-small": (2560, 8960, 151936, 36),
    "qwen3-4b": (2560, 8960, 151936, 36),
    "synthetic-large": (4096, 11008, 151936, 48),
    "qwen3-30b": (4096, 11008, 151936, 48),
}

HF_MODEL_MAP = {
    "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen3-30b": "Qwen/Qwen3-30B-A3B",
}


def create_synthetic_state_dict(model_type, dtype=torch.bfloat16):
    """Create a synthetic state_dict matching real model structure."""
    hidden, intermediate, vocab, num_layers = MODEL_CONFIGS[model_type]
    state_dict = {}

    state_dict["model.embed_tokens.weight"] = torch.randn(vocab, hidden, dtype=dtype)

    for i in range(num_layers):
        prefix = f"model.layers.{i}"
        state_dict[f"{prefix}.self_attn.q_proj.weight"] = torch.randn(
            hidden, hidden, dtype=dtype
        )
        state_dict[f"{prefix}.self_attn.k_proj.weight"] = torch.randn(
            hidden, hidden, dtype=dtype
        )
        state_dict[f"{prefix}.self_attn.v_proj.weight"] = torch.randn(
            hidden, hidden, dtype=dtype
        )
        state_dict[f"{prefix}.self_attn.o_proj.weight"] = torch.randn(
            hidden, hidden, dtype=dtype
        )
        state_dict[f"{prefix}.mlp.gate_proj.weight"] = torch.randn(
            intermediate, hidden, dtype=dtype
        )
        state_dict[f"{prefix}.mlp.up_proj.weight"] = torch.randn(
            intermediate, hidden, dtype=dtype
        )
        state_dict[f"{prefix}.mlp.down_proj.weight"] = torch.randn(
            hidden, intermediate, dtype=dtype
        )
        state_dict[f"{prefix}.input_layernorm.weight"] = torch.randn(
            hidden, dtype=dtype
        )
        state_dict[f"{prefix}.post_attention_layernorm.weight"] = torch.randn(
            hidden, dtype=dtype
        )

    state_dict["model.norm.weight"] = torch.randn(hidden, dtype=dtype)
    state_dict["lm_head.weight"] = torch.randn(vocab, hidden, dtype=dtype)

    return {"model": state_dict}


class BenchmarkActor(Actor):
    """Actor that creates state dict internally and benchmarks put/get operations."""

    def __init__(self, model_type, use_hf, iterations, warmup):
        self.rank = current_rank().rank
        os.environ["LOCAL_RANK"] = str(self.rank)
        self.model_type = model_type
        self.use_hf = use_hf
        self.iterations = iterations
        self.warmup = warmup

    @endpoint
    async def run_benchmark(self):
        """Run the full benchmark inside the actor process."""
        # Pre-cache the torchstore client before large allocations
        await ts.client()

        # Create state dict inside the actor (avoids serializing large tensors)
        if self.use_hf and self.model_type in HF_MODEL_MAP:
            from transformers import AutoModelForCausalLM

            model_name = HF_MODEL_MAP[self.model_type]
            print(f"[Actor] Loading HuggingFace model: {model_name}...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=os.environ["HF_TOKEN"],
                torch_dtype=torch.bfloat16,
            )
            state_dict = {"model": model.state_dict()}
            del model
        else:
            print(f"[Actor] Creating synthetic state dict for: {self.model_type}...")
            state_dict = create_synthetic_state_dict(self.model_type)

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
            await ts.put_state_dict(state_dict, f"warmup_{i}")
            await ts.get_state_dict(f"warmup_{i}", state_dict)
        print("[Actor] Warmup complete.")

        # Timed iterations
        print(f"[Actor] Running {self.iterations} timed iteration(s)...")
        put_times = []
        get_times = []

        for i in range(self.iterations):
            key = f"bench_{i}"

            put_start = time.perf_counter()
            await ts.put_state_dict(state_dict, key)
            put_elapsed = time.perf_counter() - put_start
            put_times.append(put_elapsed)

            get_start = time.perf_counter()
            await ts.get_state_dict(key, state_dict)
            get_elapsed = time.perf_counter() - get_start
            get_times.append(get_elapsed)

            put_tp = total_gb / put_elapsed if put_elapsed > 0 else 0
            get_tp = total_gb / get_elapsed if get_elapsed > 0 else 0
            print(
                f"  Iter {i+1}: put={put_elapsed:.3f}s ({put_tp:.2f} GB/s), "
                f"get={get_elapsed:.3f}s ({get_tp:.2f} GB/s), "
                f"total={put_elapsed + get_elapsed:.3f}s"
            )

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
        ("put_state_dict", put_stats),
        ("get_state_dict", get_stats),
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
    """Benchmark put_state_dict / get_state_dict performance."""
    ts.init_logging()

    model_type = os.environ.get("BENCH_MODEL", "synthetic-small")
    iterations = int(os.environ.get("BENCH_ITER", "5"))
    warmup = int(os.environ.get("BENCH_WARMUP", "1"))
    use_hf = os.environ.get("BENCH_USE_HF", "0") == "1"

    assert model_type in MODEL_CONFIGS, f"Unknown model: {model_type}"

    # Disable transports that may not be available
    os.environ["TORCHSTORE_RDMA_ENABLED"] = "0"
    os.environ["TORCHSTORE_GLOO_ENABLED"] = "0"
    os.environ["TORCHSTORE_SHARED_MEMORY_ENABLED"] = "0"

    hidden, intermediate, vocab, num_layers = MODEL_CONFIGS[model_type]
    print(
        f"\nModel config: hidden={hidden}, intermediate={intermediate}, "
        f"vocab={vocab}, layers={num_layers}"
    )

    print(f"\n{'='*70}")
    print("WEIGHT SYNC BENCHMARK")
    print(f"{'='*70}")
    print(
        f"Model:            {model_type} ({'HuggingFace' if use_hf else 'synthetic'})"
    )
    print("Transport:        MonarchRPC")
    print(f"Iterations:       {iterations} (+ {warmup} warmup)")
    print(f"{'='*70}\n")

    await ts.initialize(
        num_storage_volumes=1,
        strategy=ts.LocalRankStrategy(TransportType.MonarchRPC),
    )

    try:
        actor = await spawn_actors(
            1,
            BenchmarkActor,
            "bench_actor",
            model_type=model_type,
            use_hf=use_hf,
            iterations=iterations,
            warmup=warmup,
        )

        results = await actor.run_benchmark.call_one()
        print_results(results)

    finally:
        await ts.shutdown()


if __name__ == "__main__":
    ts.init_logging()
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"] + sys.argv[1:]))
