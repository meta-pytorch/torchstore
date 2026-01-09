# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tensor Send Benchmark Example (Two Nodes via Slurm)
====================================================

This example demonstrates sending a 1GB tensor between actors on two different
nodes allocated via Slurm, and timing the operation.

Usage:
    python example/send_tensor_monarch_rpc_multinode.py
"""
import os 
import argparse
import asyncio
import atexit
import time

import torch
from monarch.actor import Actor, endpoint, HostMesh, ProcMesh
from monarch.job import SlurmJob


class Sender(Actor):
    """Actor that holds a 1GB tensor and can send it to another actor."""

    def __init__(self):
        # Create a 1GB tensor (float32: 4 bytes per element)
        num_elements = 1024 * 1024 * 1024 // 4  # 1GB / 4 bytes
        self.tensor = torch.ones(num_elements, dtype=torch.float32)

    @endpoint
    def get_tensor(self) -> torch.Tensor:
        """Return the tensor to be sent to another actor."""
        print(f"sender.get tensor master addr: {os.environ['MASTER_ADDR']}")
        return self.tensor


class Receiver(Actor):
    """Actor that receives tensors from a sender and measures latency."""

    def __init__(self, sender: Sender):
        self.sender = sender
        self.received_tensor = None

    @endpoint
    def receive_tensor(self) -> float:
        """Fetch tensor from sender and return the time taken in seconds."""
        print(f"master addr receive: {os.environ['MASTER_ADDR']}")
        start = time.perf_counter()
        self.received_tensor = self.sender.get_tensor.call_one().get()
        elapsed = time.perf_counter() - start
        return elapsed

    @endpoint
    def get_tensor_shape(self) -> tuple[int, ...]:
        """Return the shape of the received tensor."""
        if self.received_tensor is None:
            return ()
        return tuple(self.received_tensor.shape)


async def benchmark_tensor_send(
    num_iterations: int = 10,
    warmup_iterations: int = 3,
) -> None:
    """
    Benchmark sending a 1GB tensor between two actors on different nodes.

    Args:
        num_iterations: Number of iterations to measure
        warmup_iterations: Number of warmup iterations (not counted)
    """
    tensor_size_gb = 1.0
    print(f"Tensor size: {tensor_size_gb} GB")
    print()
    # Create a single Slurm job with both sender and receiver meshes
    print("Allocating Slurm job with sender and receiver nodes...")
    job = SlurmJob(
        meshes={"sender": 1, "receiver": 1},  # 1 node each
        gpus_per_node=1,
        job_name="monarch-rpc-benchmark",
        slurm_args=[
            "--account=agentic-models",
            "--qos=h200_agentic-models_high",
        ],
    )
    job.apply()
    atexit.register(job.kill)

    # Get host meshes from the job state
    # This waits for the Slurm job to start and connects to workers
    print("Waiting for Slurm job to start and connecting to workers...")
    job_state = job.state(cached_path=None)
    sender_host: HostMesh = job_state.sender
    receiver_host: HostMesh = job_state.receiver
    print("Connected to workers!")

    # Spawn process meshes on each host
    print("Spawning process meshes...")
    sender_procs = sender_host.spawn_procs({"gpus": 1})
    receiver_procs = receiver_host.spawn_procs({"gpus": 1})

    # Use async with for proper initialization and cleanup
    async with sender_procs:
        print("  sender_procs initialized")
        await sender_procs.logging_option(stream_to_client=True)

        async with receiver_procs:
            print("  receiver_procs initialized")
            await receiver_procs.logging_option(stream_to_client=True)

            # Spawn actors
            print("Spawning actors...")
            sender = sender_procs.spawn("sender", Sender)
            receiver = receiver_procs.spawn("receiver", Receiver, sender)

            # Warmup iterations
            print(f"Running {warmup_iterations} warmup iterations...")
            for _ in range(warmup_iterations):
                await receiver.receive_tensor.call_one()

            # Benchmark iterations
            print(f"Running {num_iterations} benchmark iterations...")
            latencies = []

            for i in range(num_iterations):
                elapsed = await receiver.receive_tensor.call_one()
                latencies.append(elapsed)
                throughput = tensor_size_gb / elapsed
                print(f"  Iteration {i + 1}: {elapsed * 1000:.3f} ms ({throughput:.2f} GB/s)")

            # Verify the tensor was received correctly
            shape = await receiver.get_tensor_shape.call_one()
            print(f"\nReceived tensor shape: {shape}")

            # Compute statistics
            avg_latency = sum(latencies) / len(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)

            # Calculate throughput in GB/s
            avg_throughput = tensor_size_gb / avg_latency
            max_throughput = tensor_size_gb / min_latency
            min_throughput = tensor_size_gb / max_latency

            print(f"\n{'=' * 50}")
            print("Results:")
            print(f"{'=' * 50}")
            print(f"  Average latency:    {avg_latency * 1000:.3f} ms")
            print(f"  Min latency:        {min_latency * 1000:.3f} ms")
            print(f"  Max latency:        {max_latency * 1000:.3f} ms")
            print(f"  Average throughput: {avg_throughput:.2f} GB/s")
            print(f"  Max throughput:     {max_throughput:.2f} GB/s")
            print(f"  Min throughput:     {min_throughput:.2f} GB/s")
            print(f"{'=' * 50}")


async def main():
    parser = argparse.ArgumentParser(
        description="Benchmark tensor send between actors on two Slurm nodes"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of benchmark iterations (default: 10)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup iterations (default: 3)",
    )
    args = parser.parse_args()

    print(f"{'#' * 60}")
    print("# Benchmarking 1GB tensor send between actors on two nodes")
    print(f"{'#' * 60}\n")
    await benchmark_tensor_send(
        num_iterations=args.iterations,
        warmup_iterations=args.warmup,
    )


if __name__ == "__main__":
    asyncio.run(main())
