# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import math
import os
import tempfile
from logging import getLogger

import torch
import torchstore as ts
from monarch.actor import Actor, current_rank, endpoint
from torch.distributed._tensor import distribute_tensor, Shard
from torch.distributed.device_mesh import init_device_mesh
from torchstore.logging import init_logging
from torchstore.utils import spawn_actors

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

        self.rlog(f"calling get with {dtensor=}")
        fetched_tensor = await ts.get(self.shared_key, dtensor)
        self.rlog(f"after fetch: {dtensor=}")
        assert torch.equal(dtensor, fetched_tensor)

        return fetched_tensor, device_mesh.get_coordinate()

    @endpoint
    async def destroy_process_group(self):
        torch.distributed.destroy_process_group()


async def dtensor_put_get_example():
    """
    Example demonstrating how to put a 4x4 tensor in a (2,) mesh
    and then get it in another (2,) mesh.
    """
    print("Starting DTensor put/get example...")

    # Initialize TorchStore with 2 storage volumes
    await ts.initialize(num_storage_volumes=2, strategy=ts.LocalRankStrategy())

    # Create a 4x4 tensor to work with
    original_tensor = torch.arange(16).reshape(4, 4).float()
    print(f"Original tensor:\n{original_tensor}")

    with tempfile.TemporaryDirectory() as filesystem_store_dir:
        put_mesh = None
        get_mesh = None
        try:
            print("\n--- Phase 1: Putting tensor in first (2,) mesh ---")
            # Create first mesh for putting the tensor
            put_mesh = await spawn_actors(
                2,
                DTensorActor,
                "dtensor_put_mesh",
                mesh_shape=(2,),
                original_tensor=original_tensor,
                placements=[Shard(0)],
                file_store_name=os.path.join(filesystem_store_dir, "put_test"),
                visible_devices="0,1",
            )

            # Put the tensor using the first mesh
            await put_mesh.do_put.call()
            print("Successfully put tensor using first mesh")

            print("\n--- Phase 2: Getting tensor in second (2,) mesh ---")
            # Create second mesh for getting the tensor
            get_mesh = await spawn_actors(
                2,
                DTensorActor,
                "dtensor_get_mesh",
                mesh_shape=(2,),
                original_tensor=torch.zeros_like(original_tensor),  # Placeholder
                placements=[Shard(1)],  # Same sharding pattern
                file_store_name=os.path.join(filesystem_store_dir, "get_test"),
                visible_devices="2,3",  # Different devices to simulate different mesh
            )

            # Get the tensor using the second mesh
            results = await get_mesh.do_get.call()
            print("Successfully retrieved tensor using second mesh")

            # Print results from each rank in the get mesh
            for proc_info, (fetched_tensor, mesh_coord) in results:
                print(
                    f"Get mesh rank {proc_info.rank} (mesh coord {mesh_coord}): "
                    f"Retrieved tensor shape {fetched_tensor.shape}"
                    f" with values {fetched_tensor}"
                )

            print("\n--- Phase 3: Verifying full tensor ---")
            # Also verify we can get the full tensor directly
            fetched_tensor = await ts.get("test_key")
            assert torch.equal(original_tensor, fetched_tensor)
            print(f"Full tensor retrieved directly:\n{fetched_tensor}")
            print("✓ Full tensor matches original!")

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
