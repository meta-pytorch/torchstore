# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import math
import os
import tempfile
import time
from logging import getLogger

import torch
import torchstore as ts
from monarch.actor import Actor, current_rank, endpoint
from torch.distributed._tensor import distribute_tensor, Shard
from torch.distributed.device_mesh import init_device_mesh
from torchstore.logging import init_logging
from torchstore.utils import spawn_actors

# ANSI escape codes for colored output
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_yellow(text):
    """Print text in yellow color"""
    print(f"{YELLOW}{BOLD}{text}{RESET}")


init_logging()
logger = getLogger(__name__)


class DTensorActor(Actor):
    """Test class used to verify correctness of resharding across different shardings.
    Currently only supports a single tensor
    """

    shared_key = "test_key"

    def __init__(
        self,
        mesh_shape,
        original_tensor,
        placements,
        file_store_name,
        visible_devices="0,1,2,3,4,5,6,7",
    ):
        self.rank = current_rank().rank
        self.mesh_shape = mesh_shape
        self.world_size = math.prod(mesh_shape)
        self.original_tensor = original_tensor
        self.placements = placements
        self.file_store_name = file_store_name

        # torchstore will fail without this (see LocalRankStrategy)
        os.environ["LOCAL_RANK"] = str(self.rank)

        # this is only necessary for nccl, but we're not using it in this test.
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices

    def rlog(self, msg):
        # TODO: set to 'info' once this is fixed in monarch (which currently is hiding logs :/)
        logger.info(f"rank: {self.rank} {msg}")

    def initialize_distributed(self):
        self.rlog(f"Initialize process group using {self.file_store_name=} ")
        torch.distributed.init_process_group(
            backend="gloo",
            rank=self.rank,
            world_size=self.world_size,
            init_method=f"file://{self.file_store_name}",
        )

        # this barrier is more to make sure torch.distibuted is working
        self.rlog("barrrer")
        torch.distributed.barrier()

    @endpoint
    async def do_put(self):
        self.initialize_distributed()

        self.rlog("Create device mesh")
        device_mesh = init_device_mesh("cpu", self.mesh_shape)

        self.rlog("distributing dtensor")
        tensor = self.original_tensor.to("cpu")
        dtensor = distribute_tensor(tensor, device_mesh, placements=self.placements)

        self.rlog(f"calling put with {dtensor=}")
        await ts.put(self.shared_key, dtensor)

    @endpoint
    async def do_get(self):
        self.initialize_distributed()

        self.rlog("Create device mesh")
        # TODO: nccl is giving me a weird error on process group split for 2d mesh
        device_mesh = init_device_mesh("cpu", self.mesh_shape)

        self.rlog("distributing dtensor")
        tensor = self.original_tensor.to("cpu")
        dtensor = distribute_tensor(tensor, device_mesh, placements=self.placements)

        fetched_tensor = await ts.get(self.shared_key, dtensor)
        assert torch.equal(dtensor, fetched_tensor)

        return fetched_tensor, device_mesh.get_coordinate()

    @endpoint
    async def destroy_process_group(self):
        torch.distributed.destroy_process_group()


async def dtensor_put_get_example():
    """
    Example demonstrating DTensor resharding between different mesh configurations.
    Creates a tensor of shape (size * n_put_actors, size * n_get_actors),
    puts it with Shard(0) and gets it with Shard(1).
    """
    # Configuration variables
    size = 1  # 100 unit size => 2.4 MB Tensor Size
    n_put_actors = 2
    n_get_actors = 1

    print("Starting DTensor put/get example with:")
    print(f"  size = {size}")
    print(f"  n_put_actors = {n_put_actors}")
    print(f"  n_get_actors = {n_get_actors}")

    # Initialize TorchStore
    await ts.initialize(
        num_storage_volumes=max(n_put_actors, n_get_actors),
        strategy=ts.LocalRankStrategy(),
    )

    # Create tensor with shape (size * n_put_actors, size * n_get_actors)
    tensor_shape = (size * n_put_actors, size * n_get_actors)
    original_tensor = (
        torch.arange(tensor_shape[0] * tensor_shape[1]).reshape(tensor_shape).float()
    )
    print(f"Original tensor shape: {tensor_shape}")
    print(f"Original tensor:\n{original_tensor}") if size == 1 else None

    with tempfile.TemporaryDirectory() as filesystem_store_dir:
        put_mesh = None
        get_mesh = None
        try:
            print(
                f"\n--- Phase 1: Putting tensor with Shard(0) in ({n_put_actors},) mesh ---"
            )
            # Create first mesh for putting the tensor with Shard(0)
            put_mesh = await spawn_actors(
                n_put_actors,
                DTensorActor,
                "dtensor_put_mesh",
                mesh_shape=(n_put_actors,),
                original_tensor=original_tensor,
                placements=[Shard(0)],  # Shard along dimension 0
                file_store_name=os.path.join(filesystem_store_dir, "put_test"),
                visible_devices=",".join(str(i) for i in range(n_put_actors)),
            )

            # Put the tensor using the first mesh with timing
            put_start_time = time.perf_counter()
            await put_mesh.do_put.call()
            put_end_time = time.perf_counter()
            put_duration = put_end_time - put_start_time

            print("Successfully put tensor using first mesh")
            print_yellow(f"⏱️  PUT operation took: {put_duration:.4f} seconds")

            print(
                f"\n--- Phase 2: Getting tensor with Shard(1) in ({n_get_actors},) mesh ---"
            )
            # Create second mesh for getting the tensor with Shard(1)
            get_mesh = await spawn_actors(
                n_get_actors,
                DTensorActor,
                "dtensor_get_mesh",
                mesh_shape=(n_get_actors,),
                original_tensor=torch.zeros_like(original_tensor),  # Placeholder
                placements=[Shard(1)],  # Shard along dimension 1
                file_store_name=os.path.join(filesystem_store_dir, "get_test"),
                visible_devices=",".join(
                    str(i) for i in range(n_put_actors, n_put_actors + n_get_actors)
                ),
            )

            # Get the tensor using the second mesh with timing
            get_start_time = time.perf_counter()
            results = await get_mesh.do_get.call()
            get_end_time = time.perf_counter()
            get_duration = get_end_time - get_start_time

            print("Successfully retrieved tensor using second mesh")
            print_yellow(f"⏱️  GET operation took: {get_duration:.4f} seconds")

            # Print results from each rank in the get mesh
            for proc_info, (fetched_tensor, mesh_coord) in results:
                print(
                    f"Get mesh rank {proc_info.rank} (mesh coord {mesh_coord}): "
                    f"Retrieved tensor shape {fetched_tensor.shape}"
                )
                print(f" with values:\n{fetched_tensor}") if size == 1 else None

            print("\n--- Phase 3: Verifying full tensor ---")
            # Also verify we can get the full tensor directly
            fetched_tensor = await ts.get("test_key")
            assert torch.equal(original_tensor, fetched_tensor)
            print(f"Full tensor retrieved directly:\n{fetched_tensor}")
            print("✓ Full tensor matches original!")

            # Calculate tensor size in MB
            total_elements = tensor_shape[0] * tensor_shape[1]
            tensor_size_bytes = total_elements * 4  # float32 = 4 bytes per element
            tensor_size_mb = tensor_size_bytes / (1024 * 1024)

            # Print timing summary
            print("\n" + "=" * 50)
            print_yellow("⏱️  TIMING SUMMARY:")
            print_yellow(f"   Tensor size:   {tensor_size_mb:.4f} MB ({tensor_shape})")
            print_yellow(f"   PUT operation: {put_duration:.4f} seconds")
            print_yellow(f"   GET operation: {get_duration:.4f} seconds")
            print("=" * 50)

        finally:
            # Clean up process groups and meshes
            if put_mesh is not None:
                await put_mesh.destroy_process_group.call()
                await put_mesh._proc_mesh.stop()
            if get_mesh is not None:
                await get_mesh.destroy_process_group.call()
                await get_mesh._proc_mesh.stop()
            await ts.shutdown()

    print("\nDTensor put/get example completed successfully!")


if __name__ == "__main__":
    asyncio.run(dtensor_put_get_example())
