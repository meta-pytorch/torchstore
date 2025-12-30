# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile

import pytest
import torch
import torchstore as ts
from monarch.actor import Actor, current_rank, endpoint

# DTensor imports for DTensor slice testing
from torch.distributed._tensor import Shard
from torchstore.logging import init_logging
from torchstore.transport.pipe import TensorSlice
from torchstore.utils import spawn_actors

from .utils import (
    DTensorActor,
    main,
    set_transport_type,
    transport_plus_strategy_params,
)


@pytest.mark.parametrize(*transport_plus_strategy_params())
@pytest.mark.asyncio
async def test_get_tensor_slice(strategy_params, transport_type):
    """Test tensor slice API functionality"""
    set_transport_type(transport_type)

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

    class TensorSliceGetActor(Actor):
        """Actor for getting tensors."""

        @endpoint
        async def get(self, key, tensor_slice_spec=None):
            return await ts.get(key, tensor_slice_spec=tensor_slice_spec)

    volume_world_size, strategy = strategy_params
    await ts.initialize(num_storage_volumes=volume_world_size, strategy=strategy)

    # Spawn test actors - separate meshes for put and get to test cross-process communication
    put_actor_mesh = await spawn_actors(
        volume_world_size,
        TensorSlicePutActor,
        "tensor_slice_put_actors",
        world_size=volume_world_size,
    )

    get_actor = await spawn_actors(1, TensorSliceGetActor, "tensor_slice_get_actor")

    try:
        # Create a 100x100 tensor filled with sequential values 0-9999
        test_tensor = torch.arange(10000).reshape(100, 100).float()
        key = "test_tensor"

        # Store the tensor using put actor mesh
        put_actor = put_actor_mesh.slice(gpus=0)
        await put_actor.put.call(key, test_tensor)

        # Test full tensor retrieval using get actor mesh
        retrieved_tensor = await get_actor.get.call_one(key)
        assert torch.equal(test_tensor, retrieved_tensor)

        # Test slice retrieval using get actor mesh
        tensor_slice_spec = TensorSlice(
            offsets=(10, 20),
            coordinates=(),
            global_shape=(100, 100),
            local_shape=(5, 10),
            mesh_shape=(),
        )

        tensor_slice = await get_actor.get.call_one(
            key, tensor_slice_spec=tensor_slice_spec
        )
        expected_slice = test_tensor[10:15, 20:30]
        assert torch.equal(tensor_slice, expected_slice)
        assert tensor_slice.shape == (5, 10)

    finally:
        # TODO: Investigate monarch bug with proc_mesh.stop()
        # await put_actor_mesh._proc_mesh.stop()
        await ts.shutdown()


@pytest.mark.asyncio
async def test_tensor_slice_inplace():
    """Test tensor slice API with in-place operations"""

    class TestActor(Actor):
        @endpoint
        async def test(self, test_tensor) -> Exception or None:
            try:
                # Store a test tensor
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
                    "inplace_test",
                    inplace_tensor=slice_buffer,
                    tensor_slice_spec=slice_spec,
                )

                # Verify in-place operation
                assert result is slice_buffer
                expected_slice = test_tensor[10:40, 20:60]
                assert torch.equal(slice_buffer, expected_slice)
            except Exception as e:
                return e

    await ts.initialize(num_storage_volumes=1)

    try:
        test_tensor = torch.randn(100, 200)
        actor = await spawn_actors(1, TestActor, "actor_0")
        err = await actor.test.call_one(test_tensor)

        assert err is None

    finally:
        await ts.shutdown()


@pytest.mark.asyncio
async def test_put_dtensor_get_full_tensor():
    """Test basic DTensor put/get functionality with separate put and get meshes using shared DTensorActor"""

    class GetActor(Actor):
        @endpoint
        async def get_tensor(self, key):
            return await ts.get(key)

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

            get_actor = await spawn_actors(1, GetActor, "get_actor_0")
            fetched_tensor = await get_actor.get_tensor.call_one("test_key")
            assert torch.equal(original_tensor, fetched_tensor)

        finally:
            # Clean up process groups
            await put_mesh.destroy_process_group.call()
            # TODO: Investigate monarch bug with proc_mesh.stop()
            # await put_mesh._proc_mesh.stop()
            await ts.shutdown()


@pytest.mark.asyncio
async def test_dtensor_fetch_slice():
    """
    Test DTensor slice optimization by storing a DTensor across multiple volumes
    and requesting slices that test both cross-volume and single-volume scenarios.

    This test validates that the slice optimization:
    1. Only fetches from relevant storage volumes when requesting slices
    2. Works correctly for slices that span multiple volumes
    3. Works correctly for slices contained within a single volume
    4. Maintains correctness while providing performance benefits
    """
    import tempfile

    class GetActor(Actor):
        @endpoint
        async def get_tensor(self, key, tensor_slice_spec=None):
            return await ts.get(key, tensor_slice_spec=tensor_slice_spec)

    # Use LocalRankStrategy with 2 storage volumes (no RDMA, no parametrization)
    set_transport_type("none")
    os.environ["LOCAL_RANK"] = "0"  # Required by LocalRankStrategy

    await ts.initialize(num_storage_volumes=2, strategy=ts.LocalRankStrategy())

    # Create a tensor that will be sharded across 2 volumes
    # 8x6 tensor, when sharded by 2 actors along dim 0 gives 4x6 slices per volume
    # Volume 0: rows 0-3, Volume 1: rows 4-7
    original_tensor = torch.arange(48).reshape(8, 6).float()

    with tempfile.TemporaryDirectory() as filesystem_store_dir:
        put_mesh = None
        try:
            # Store DTensor across 2 volumes using Shard(0)
            put_mesh = await spawn_actors(
                2,
                DTensorActor,
                "slice_test_mesh",
                mesh_shape=(2,),
                original_tensor=original_tensor,
                placements=[Shard(0)],
                file_store_name=os.path.join(filesystem_store_dir, "slice_test"),
                visible_devices="0,1",
            )

            await put_mesh.do_put.call()

            get_actor = await spawn_actors(1, GetActor, "get_actor_0")

            # Test 1: Cross-volume slice (spans both volumes)
            # Request rows 2-5 (spans volume boundary at row 4)
            cross_volume_slice = TensorSlice(
                offsets=(2, 1),
                coordinates=(),
                global_shape=(8, 6),
                local_shape=(4, 4),  # 4 rows, 4 cols
                mesh_shape=(),
            )

            cross_volume_result = await get_actor.get_tensor.call_one(
                "test_key", tensor_slice_spec=cross_volume_slice
            )
            expected_cross_volume = original_tensor[2:6, 1:5]

            assert torch.equal(cross_volume_result, expected_cross_volume)
            assert cross_volume_result.shape == (4, 4)

            # Test 2: Single-volume slice (contained within volume 0)
            # Request rows 1-2 (only from volume 0)
            single_volume_slice = TensorSlice(
                offsets=(1, 0),
                coordinates=(),
                global_shape=(8, 6),
                local_shape=(2, 3),  # 2 rows, 3 cols
                mesh_shape=(),
            )

            single_volume_result = await get_actor.get_tensor.call_one(
                "test_key", tensor_slice_spec=single_volume_slice
            )
            expected_single_volume = original_tensor[1:3, 0:3]

            assert torch.equal(single_volume_result, expected_single_volume)
            assert single_volume_result.shape == (2, 3)

        finally:
            if put_mesh is not None:
                await put_mesh.destroy_process_group.call()
                # TODO: Investigate monarch bug with proc_mesh.stop()
                # await put_mesh._proc_mesh.stop()
            await ts.shutdown()


@pytest.mark.asyncio
async def test_partial_put():
    """
    Verify the behavior when a dtensor is partially put.
    1. Create two put actors. Each of them should put half of a DTensor.
    2. Rank 1 will skip the put operation (using ranks_to_skip_put=[1]).
    3. After rank 0 completes its put, we call get() which should raise a KeyError
       because the DTensor is not fully committed (only rank 0's shard is stored).
    """

    class TestActor(Actor):
        @endpoint
        async def exists(self, key):
            return await ts.exists(key)

        @endpoint
        async def get(self, key):
            try:
                result = await ts.get(key)
                return {"success": True, "result": result}
            except Exception as e:
                return {"success": False, "error": e, "error_str": str(e)}

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
                ranks_to_skip_put=[1],  # Rank 1 will skip the put
            )

            # Execute the put - rank 0 will put, rank 1 will skip
            await put_mesh.do_put.call()

            test_actor = await spawn_actors(1, TestActor, "test_actor_0")

            assert not await test_actor.exists.call_one("test_key")
            # Try to get the tensor - should raise KeyError because only rank 0 has committed
            result = await test_actor.get.call_one("test_key")

            assert not result["success"], "Expected get to fail but it succeeded"
            assert isinstance(
                result["error"], KeyError
            ), f"Expected KeyError but got {type(result['error'])}"

            # Verify the error message mentions partial commit
            assert (
                "partially committed" in result["error_str"]
            ), f"Error message should mention partial commit: {result['error_str']}"

        finally:
            # Clean up process groups
            await put_mesh.destroy_process_group.call()
            # TODO: Investigate monarch bug with proc_mesh.stop()
            # await put_mesh._proc_mesh.stop()
            await ts.shutdown()


@pytest.mark.asyncio
async def test_fully_local_dtensor_put_get():
    """
    Test that fully local DTensors (Replicate placement) are stored as regular tensors.

    This simulates the MoE use case in torchtitan where individual expert parameters are DTensors
    with Replicate() placement, but each rank puts different expert IDs.
    """
    from torch.distributed.tensor.placement_types import Replicate

    class ExpertPutActor(Actor):
        """Actor simulating MoE expert parameter storage."""

        def __init__(self, world_size, expert_offset):
            init_logging()
            self.world_size = world_size
            self.rank = current_rank().rank
            self.expert_offset = expert_offset  # Different per replica
            os.environ["LOCAL_RANK"] = str(self.rank)

        @endpoint
        async def init_process_group(self, file_store_name, mesh_shape):
            from torch.distributed import init_process_group
            from torch.distributed.device_mesh import init_device_mesh

            init_process_group(
                "gloo",
                rank=self.rank,
                world_size=self.world_size,
                store=torch.distributed.FileStore(file_store_name, self.world_size),
            )
            self.mesh = init_device_mesh("cpu", mesh_shape)

        @endpoint
        async def destroy_process_group(self):
            import torch.distributed as dist

            if dist.is_initialized():
                dist.destroy_process_group()

        @endpoint
        async def put_expert(self, expert_id: int):
            """Put a single expert parameter (DTensor with Replicate placement)."""
            # Create a fully replicated DTensor (simulates individual expert after splitting)
            local_tensor = torch.randn(256, 512)  # Expert weight
            from torch.distributed.tensor import DTensor

            expert_dtensor = DTensor.from_local(
                local_tensor, self.mesh, [Replicate()], run_check=False
            )

            key = f"expert_{expert_id}.weight"
            await ts.put(key, expert_dtensor)
            return key

    class ExpertGetActor(Actor):
        """Actor for retrieving expert parameters."""

        @endpoint
        async def get_expert(self, key):
            return await ts.get(key)

    await ts.initialize(num_storage_volumes=2, strategy=ts.LocalRankStrategy())

    with tempfile.TemporaryDirectory() as filesystem_store_dir:
        try:
            # Create two actor replicas, each will put different expert IDs
            put_mesh = await spawn_actors(
                2,
                ExpertPutActor,
                "expert_put_mesh",
                world_size=2,
                expert_offset=0,  # Not actually used, just for init
            )

            # Initialize process group
            await put_mesh.init_process_group.call(
                file_store_name=os.path.join(filesystem_store_dir, "expert_test"),
                mesh_shape=(2,),
            )

            # Each rank puts different experts (simulating EP sharding)
            # Rank 0 puts expert 0, Rank 1 puts expert 1
            keys = []
            for rank_id in range(2):
                expert_id = rank_id * 16  # Rank 0: expert 0, Rank 1: expert 16
                put_actor = put_mesh.slice(gpus=rank_id)
                key = await put_actor.put_expert.call_one(expert_id=expert_id)
                keys.append(key)

            # Create get actor
            get_actor = await spawn_actors(1, ExpertGetActor, "expert_get_actor")

            # Should be able to retrieve both experts independently
            # (they were stored as regular tensors, not DTensors)
            for key in keys:
                retrieved = await get_actor.get_expert.call_one(key)
                assert retrieved is not None
                assert retrieved.shape == (256, 512)
                assert isinstance(retrieved, torch.Tensor)
                # Should NOT be a DTensor
                from torch.distributed.tensor import DTensor

                assert not isinstance(retrieved, DTensor)

        finally:
            await put_mesh.destroy_process_group.call()
            await ts.shutdown()


if __name__ == "__main__":
    main(__file__)
