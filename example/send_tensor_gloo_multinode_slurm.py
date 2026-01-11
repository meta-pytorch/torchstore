#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Gloo Tensor Send Benchmark Example (Cross node via Slurm)
=========================================================

This example demonstrates sending a 1GB tensor between two processes on different
nodes using PyTorch's Gloo backend, allocated via Slurm.

Usage:
    python example/send_tensor_gloo_multinode_slurm.py
"""

import asyncio
import atexit
import time

from monarch.actor import Actor, endpoint, HostMesh
from monarch.job import SlurmJob


class GlooWorker(Actor):
    """Actor that runs the gloo tensor send/receive logic."""

    def __init__(self):
        self.rank = None
        self.result = None

    @endpoint
    def get_hostname(self) -> str:
        """Return the hostname of this worker."""
        import socket

        return socket.gethostname()

    @endpoint
    def run_gloo_script(
        self,
        rank: int,
        world_size: int,
        master_addr: str,
        master_port: str,
    ) -> dict[str, float | str]:
        """
        Run the gloo tensor send/receive script.

        Args:
            rank: Process rank
            world_size: Total number of processes
            master_addr: IP address of the master node
            master_port: Port for communication

        Returns:
            Dictionary with timing results
        """
        import torch
        import torch.distributed as dist

        self.rank = rank

        print(f"Rank {rank}: Initializing process group")

        # Initialize the process group using TCP init method
        init_method = f"tcp://{master_addr}:{master_port}"
        dist.init_process_group(
            backend="gloo",
            init_method=init_method,
            rank=rank,
            world_size=world_size,
        )

        dist.barrier()
        print(f"Rank {rank}: Process group initialized")

        # Create a 1GB tensor (float32: 4 bytes per element)
        num_elements = 1024 * 1024 * 1024 // 4  # 1GB / 4 bytes
        tensor = torch.ones(num_elements, dtype=torch.float32)
        print(f"Rank {rank}: Created tensor")

        result = {}

        if rank == 0:
            # Send tensor to rank 1
            print("Rank 0: Sending 1GB tensor to rank 1...")
            start_time = time.time()
            dist.send(tensor=tensor, dst=1)
            end_time = time.time()
            elapsed = end_time - start_time
            throughput = 1.0 / elapsed
            print(
                f"Rank 0: Sent 1GB tensor to rank 1 in {elapsed:.4f} seconds ({throughput:.2f} GB/s)"
            )
            result = {
                "rank": rank,
                "role": "sender",
                "elapsed": elapsed,
                "throughput": throughput,
            }
        elif rank == 1:
            # Prepare empty tensor to receive
            recv_tensor = torch.empty(num_elements, dtype=torch.float32)
            print("Rank 1: Waiting to receive 1GB tensor from rank 0...")
            start_time = time.time()
            dist.recv(tensor=recv_tensor, src=0)
            end_time = time.time()
            elapsed = end_time - start_time
            throughput = 1.0 / elapsed
            print(
                f"Rank 1: Received 1GB tensor from rank 0 in {elapsed:.4f} seconds ({throughput:.2f} GB/s)"
            )

            # Verify the tensor was received correctly
            verified = bool(torch.all(recv_tensor == 1.0))
            if verified:
                print("Rank 1: Tensor verified successfully!")
            else:
                print("Rank 1: Warning - tensor values don't match expected!")

            result = {
                "rank": rank,
                "role": "receiver",
                "elapsed": elapsed,
                "throughput": throughput,
                "verified": verified,
            }

        dist.destroy_process_group()
        print(f"Rank {rank}: Process group destroyed")

        self.result = result
        return result


async def benchmark_gloo_tensor_send() -> None:
    """
    Benchmark sending a 1GB tensor between two processes on different nodes using Gloo.
    """
    tensor_size_gb = 1.0
    print(f"Tensor size: {tensor_size_gb} GB")
    print()

    # Create a Slurm job with 2 nodes
    print("Allocating Slurm job with 2 nodes...")
    job = SlurmJob(
        meshes={"node0": 1, "node1": 1},  # 1 node each
        gpus_per_node=1,
        job_name="gloo-tensor-benchmark",
        slurm_args=[
            "--account=agentic-models",
            "--qos=h200_agentic-models_high",
        ],
    )
    job.apply()
    atexit.register(job.kill)

    # Get host meshes from the job state
    print("Waiting for Slurm job to start and connecting to workers...")
    job_state = job.state(cached_path=None)
    node0_host: HostMesh = job_state.node0
    node1_host: HostMesh = job_state.node1
    print("Connected to workers!")

    # Spawn process meshes on each host
    print("Spawning process meshes...")
    node0_procs = node0_host.spawn_procs({"gpus": 1})
    node1_procs = node1_host.spawn_procs({"gpus": 1})

    # Use async with for proper initialization and cleanup
    async with node0_procs:
        print("  node0_procs initialized")
        await node0_procs.logging_option(stream_to_client=True)

        async with node1_procs:
            print("  node1_procs initialized")
            await node1_procs.logging_option(stream_to_client=True)

            # Spawn actors
            print("Spawning actors...")
            worker0 = node0_procs.spawn("worker0", GlooWorker)
            worker1 = node1_procs.spawn("worker1", GlooWorker)

            # Get the master node's address (node0) from the actor
            print("Getting master node hostname...")
            master_addr = await worker0.get_hostname.call_one()
            master_port = "12355"
            print(f"Master address: {master_addr}:{master_port}")

            # Run the gloo benchmark on both workers concurrently
            print("\nStarting benchmark...")
            print(f"{'=' * 50}")

            results = await asyncio.gather(
                worker0.run_gloo_script.call_one(
                    rank=0,
                    world_size=2,
                    master_addr=master_addr,
                    master_port=master_port,
                ),
                worker1.run_gloo_script.call_one(
                    rank=1,
                    world_size=2,
                    master_addr=master_addr,
                    master_port=master_port,
                ),
            )

            # Print results
            print(f"\n{'=' * 50}")
            print("Results:")
            print(f"{'=' * 50}")

            for result in results:
                rank = result["rank"]
                role = result["role"]
                elapsed = result["elapsed"]
                throughput = result["throughput"]

                print(f"Rank {rank} ({role}):")
                print(f"  Latency:    {elapsed * 1000:.3f} ms")
                print(f"  Throughput: {throughput:.2f} GB/s")

                if "verified" in result:
                    print(f"  Verified:   {result['verified']}")

            print(f"{'=' * 50}")


async def main():
    print(f"{'#' * 60}")
    print("# Benchmarking 1GB tensor send using Gloo on two nodes")
    print(f"{'#' * 60}\n")
    await benchmark_gloo_tensor_send()


if __name__ == "__main__":
    asyncio.run(main())
