"""
Tensor Send Benchmark Example
=============================

This example demonstrates sending a 1GB tensor between actors on a single host
and timing the operation.

Usage:
    python example/send_tensor_monarch_rpc.py
"""

import asyncio
import time

import torch
from monarch.actor import Actor, endpoint, this_host


class Sender(Actor):
    """Actor that holds a 1GB tensor and can send it to another actor."""

    def __init__(self):
        # Create a 1GB tensor (float32: 4 bytes per element)
        num_elements = 1024 * 1024 * 1024 // 4  # 1GB / 4 bytes
        self.tensor = torch.ones(num_elements, dtype=torch.float32)

    @endpoint
    def get_tensor(self) -> torch.Tensor:
        """Return the tensor to be sent to another actor."""
        return self.tensor


class Receiver(Actor):
    """Actor that receives tensors from a sender and measures latency."""

    def __init__(self, sender: Sender):
        self.sender = sender
        self.received_tensor = None

    @endpoint
    def receive_tensor(self) -> float:
        """Fetch tensor from sender and return the time taken in seconds."""
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
    Benchmark sending a 1GB tensor between two actors on the same host.

    Args:
        num_iterations: Number of iterations to measure
        warmup_iterations: Number of warmup iterations (not counted)
    """
    tensor_size_gb = 1.0
    print(f"Tensor size: {tensor_size_gb} GB")
    print()

    # Create two separate process meshes on the same host
    sender_mesh = this_host().spawn_procs(per_host={"sender": 1})
    receiver_mesh = this_host().spawn_procs(per_host={"receiver": 1})

    # Spawn actors
    sender = sender_mesh.spawn("sender", Sender)
    receiver = receiver_mesh.spawn("receiver", Receiver, sender)

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
    print(f"{'#' * 60}")
    print("# Benchmarking 1GB tensor send between actors")
    print(f"{'#' * 60}\n")
    await benchmark_tensor_send(
        num_iterations=10,
        warmup_iterations=3,
    )


if __name__ == "__main__":
    asyncio.run(main())
