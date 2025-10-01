# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import multiprocessing as mp
import os
import tempfile
import time

import pytest
import torch

import torchstore as ts
from monarch.actor import Actor, current_rank, endpoint

# DTensor imports for DTensor slice testing
from torch.distributed._tensor import Shard
from torchstore.logging import init_logging
from torchstore.transport.pipe import TensorSlice
from torchstore.utils import spawn_actors

from .utils import DTensorActor, main, transport_plus_strategy_params


class FileSync:
    """Simple synchronization using file existence checks."""

    def __init__(self, file_path):
        self.file_path = file_path

    def wait(self, timeout=30):
        """Wait for a file to be created (blocking)."""
        start_time = time.time()
        while not os.path.exists(self.file_path):
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Timeout waiting for file: {self.file_path}")
            time.sleep(0.01)  # Poll every 10ms

    def signal(self):
        """Create the file to signal completion."""
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        # Create an empty file
        with open(self.file_path, 'w') as f:
            f.write('1')

    def cleanup(self):
        """Clean up the file."""
        if os.path.exists(self.file_path):
            os.remove(self.file_path)


@pytest.mark.parametrize(*transport_plus_strategy_params())
@pytest.mark.asyncio
async def test_get_tensor_slice(strategy_params, use_rdma):
    """Test tensor slice API functionality"""
    os.environ["TORCHSTORE_RDMA_ENABLED"] = "1" if use_rdma else "0"

    class TensorSlicePutActor(Actor):
        """Actor for putting tensors."""

        def __init__(self, world_size):
            init_logging()
            self.world_size = world_size
            self.rank = current_rank().rank
            # required by LocalRankStrategy
            os.environ["LOCAL_RANK"] = str(self.rank)

        @endpoint
        async def put(self, key, tensor):
            await ts.put(key, tensor)

    volume_world_size, strategy = strategy_params
    await ts.initialize(num_storage_volumes=volume_world_size, strategy=strategy)

    # Spawn test actors - separate meshes for put and get to test cross-process communication
    put_actor_mesh = await spawn_actors(
        volume_world_size,
        TensorSlicePutActor,
        "tensor_slice_put_actors",
        world_size=volume_world_size,
    )

    try:
        test_tensor = torch.randn(1000, 2000)
        key = "test_tensor"

        # Store the tensor using put actor mesh
        put_actor = put_actor_mesh.slice(gpus=0)
        await put_actor.put.call(key, test_tensor)

        # Test full tensor retrieval using get actor mesh
        retrieved_tensor = await ts.get(key)
        assert torch.equal(test_tensor, retrieved_tensor)

        # Test slice retrieval using get actor mesh
        tensor_slice_spec = TensorSlice(
            offsets=(100, 200),
            coordinates=(),
            global_shape=(1000, 2000),
            local_shape=(50, 100),
            mesh_shape=(),
        )

        tensor_slice = await ts.get(key, tensor_slice_spec=tensor_slice_spec)
        expected_slice = test_tensor[100:150, 200:300]
        assert torch.equal(tensor_slice, expected_slice)
        assert tensor_slice.shape == (50, 100)

    finally:
        await put_actor_mesh._proc_mesh.stop()
        await ts.shutdown()


@pytest.mark.asyncio
async def test_tensor_slice_inplace():
    """Test tensor slice API with in-place operations"""
    await ts.initialize(num_storage_volumes=1)

    try:
        # Store a test tensor
        test_tensor = torch.randn(100, 200)
        await ts.put("inplace_test", test_tensor)

        # Test in-place retrieval with slice
        slice_spec = TensorSlice(
            offsets=(10, 20),
            coordinates=(),
            global_shape=(100, 200),
            local_shape=(30, 40),
            mesh_shape=(),
        )

        # Create pre-allocated buffer
        slice_buffer = torch.empty(30, 40)
        result = await ts.get(
            "inplace_test", inplace_tensor=slice_buffer, tensor_slice_spec=slice_spec
        )

        # Verify in-place operation
        assert result is slice_buffer
        expected_slice = test_tensor[10:40, 20:60]
        assert torch.equal(slice_buffer, expected_slice)

    finally:
        await ts.shutdown()


@pytest.mark.asyncio
async def test_put_dtensor_get_full_tensor():
    """Test basic DTensor put/get functionality with separate put and get meshes using shared DTensorActor"""
    await ts.initialize(num_storage_volumes=2, strategy=ts.LocalRankStrategy())

    original_tensor = torch.arange(16).reshape(4, 4).float()

    with tempfile.TemporaryDirectory() as filesystem_store_dir:
        try:
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

            await put_mesh.do_put.call()

            fetched_tensor = await ts.get("test_key")
            assert torch.equal(original_tensor, fetched_tensor)

        finally:
            # Clean up process groups
            await put_mesh.destroy_process_group.call()
            await put_mesh._proc_mesh.stop()
            await ts.shutdown()


@pytest.mark.asyncio
async def test_partial_put():
    """
    Verify the behavior when a dtensor is partially put.
    1. Create two put actors. Each of them will put half of a DTensor.
    2. Put actor 0 should be able to put the DTensor, but Put actor 1 will wait for
       a signal to do the put.
    3. We wait for Put actor 0 to finish its work and call get(). At this moment,
       get() should raise a KeyError because DTensor is not fully committed.
    4. Then we release the signal so that Put actor 1 also continues to finish put
       and release a signal of finish.
    5. We call get() again to verify that now tensor can be fetched.
    """

    await ts.initialize(num_storage_volumes=2, strategy=ts.LocalRankStrategy())

    original_tensor = torch.arange(16).reshape(4, 4).float()

    with tempfile.TemporaryDirectory() as filesystem_store_dir:
        # Use file-based synchronization for cross-process communication
        sync_dir = os.path.join(filesystem_store_dir, "sync")
        os.makedirs(sync_dir, exist_ok=True)

        # File paths for synchronization
        actor_1_put_file = os.path.join(sync_dir, "actor_1_put.txt")
        rank_0_done_file = os.path.join(sync_dir, "rank_0_done.txt")
        rank_1_done_file = os.path.join(sync_dir, "rank_1_done.txt")

        # Create sync objects for waiting
        rank_0_sync = FileSync(rank_0_done_file)
        rank_1_sync = FileSync(rank_1_done_file)
        actor_1_sync = FileSync(actor_1_put_file)

        try:
            put_mesh = await spawn_actors(
                2,
                DTensorActor,
                "dtensor_put_mesh",
                mesh_shape=(2,),
                original_tensor=original_tensor,
                placements=[Shard(0)],
                file_store_name=os.path.join(filesystem_store_dir, "put_test"),
                visible_devices="0,1",
                put_events=[None, actor_1_put_file],
                get_events=[rank_0_done_file, rank_1_done_file],
            )

            async def put():
                await put_mesh.do_put.call()

            async def get():
                loop = asyncio.get_event_loop()
                print("waiting for rank 0 to complete")
                # Wait for rank 0 to complete
                await loop.run_in_executor(None, rank_0_sync.wait)
                print("starting get after rank 0")

                # Try to get the tensor - should raise KeyError because only rank 0 has committed
                # With PR #40, KeyError is properly raised instead of being wrapped in ActorError
                partial_commit_error_raised = False
                try:
                    fetched_tensor = await ts.get("test_key")
                    print(f"ERROR: Should not have succeeded! Got tensor: {fetched_tensor}")
                except KeyError as e:
                    print(f"Expected KeyError raised: {e}")
                    partial_commit_error_raised = True
                    # Check that the error message mentions partial commit
                    assert "partially committed" in str(e), f"Error message should mention partial commit: {e}"

                assert partial_commit_error_raised, "KeyError should be raised for partially committed DTensor"

                # Signal actor 1 to continue
                await loop.run_in_executor(None, actor_1_sync.signal)
                print("waiting for rank 1 to complete")
                # Wait for rank 1 to complete
                await loop.run_in_executor(None, rank_1_sync.wait)
                print("both ranks completed, getting final tensor")
                return await ts.get("test_key")

            tasks = [
                put(),
                get(),
            ]

            _, fetched_tensor = await asyncio.gather(*tasks)

            assert torch.equal(original_tensor, fetched_tensor)

        finally:
            # Clean up process groups
            await put_mesh.destroy_process_group.call()
            await put_mesh._proc_mesh.stop()
            await ts.shutdown()
            # Clean up sync files
            rank_0_sync.cleanup()
            rank_1_sync.cleanup()
            actor_1_sync.cleanup()


if __name__ == "__main__":
    main(__file__)
