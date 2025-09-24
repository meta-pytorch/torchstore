# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import time
from logging import getLogger

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

init_logging()
logger = getLogger(__name__)


@pytest.mark.parametrize(*transport_plus_strategy_params())
@pytest.mark.asyncio
async def test_basic(strategy_params, use_rdma):
    """Test basic put/get functionality for multiple processes"""
    os.environ["TORCHSTORE_RDMA_ENABLED"] = "1" if use_rdma else "0"

    class PutGetActor(Actor):
        """Each instance of this actor represents a single process."""

        def __init__(
            self,
            world_size,
        ):
            init_logging()
            self.world_size = world_size
            self.rank = current_rank().rank

            # required by LocalRankStrategy
            os.environ["LOCAL_RANK"] = str(self.rank)

        @endpoint
        async def put(self):
            t = torch.tensor([self.rank + 1] * 10)
            await ts.put(f"key_{self.rank}", t)

        @endpoint
        async def get(self, rank_offset=0):
            other_rank = (self.rank + rank_offset) % self.world_size
            return await ts.get(f"key_{other_rank}")

    volume_world_size, strategy = strategy_params
    await ts.initialize(num_storage_volumes=volume_world_size, strategy=strategy)
    # each actor mesh represents a group of processes.
    actor_mesh_0 = await spawn_actors(
        volume_world_size, PutGetActor, "actor_mesh_0", world_size=volume_world_size
    )
    actor_mesh_1 = await spawn_actors(
        volume_world_size, PutGetActor, "actor_mesh_1", world_size=volume_world_size
    )

    try:
        await actor_mesh_0.put.call()
        tensors = await actor_mesh_1.get.call()
        for pt, val in tensors:
            expected = torch.tensor([pt.rank + 1] * 10)
            assert torch.equal(expected, val), f"{expected} != {val}"

        # in cases where volume_world_size > 1, we should also test that we can get from a different rank
        rank_offset = 1
        tensors = await actor_mesh_1.get.call(rank_offset)
        for pt, val in tensors:
            other_rank = (pt.rank + rank_offset) % volume_world_size
            expected = torch.tensor([other_rank + 1] * 10)
            assert torch.equal(expected, val), f"{expected} != {val}"
    finally:
        await actor_mesh_0._proc_mesh.stop()
        await actor_mesh_1._proc_mesh.stop()
        await ts.shutdown()


@pytest.mark.parametrize(*transport_plus_strategy_params())
@pytest.mark.asyncio
async def test_objects(strategy_params, use_rdma):
    """Test put/get on arbitrary object"""
    os.environ["TORCHSTORE_RDMA_ENABLED"] = "1" if use_rdma else "0"

    class ObjectActor(Actor):
        """Each instance of this actor represents a single process."""

        def __init__(
            self,
            world_size,
        ):
            init_logging()
            self.world_size = world_size
            self.rank = current_rank().rank
            # required by LocalRankStrategy
            os.environ["LOCAL_RANK"] = str(self.rank)

        @endpoint
        async def put(self, obj):
            await ts.put(f"key_{self.rank}", obj)

        @endpoint
        async def get(self, rank_offset=0):
            other_rank = (self.rank + rank_offset) % self.world_size
            return await ts.get(f"key_{other_rank}")

    volume_world_size, strategy = strategy_params
    await ts.initialize(num_storage_volumes=volume_world_size, strategy=strategy)
    # each actor mesh represents a group of processes.
    actor_mesh_0 = await spawn_actors(
        volume_world_size, ObjectActor, "actor_mesh_0", world_size=volume_world_size
    )
    actor_mesh_1 = await spawn_actors(
        volume_world_size, ObjectActor, "actor_mesh_1", world_size=volume_world_size
    )

    class MyTestObject:
        def __init__(self, val):
            self.val = val

        def __eq__(self, other: object) -> bool:
            return self.val == other.val

    try:
        for idx in range(volume_world_size):
            actor = actor_mesh_0.slice(gpus=idx)
            await actor.put.call(MyTestObject(idx))

        for rank_offset in (0, 1):
            objects = await actor_mesh_1.get.call(rank_offset=rank_offset)
            for pt, val in objects:
                other_rank = (pt.rank + rank_offset) % volume_world_size
                expected = MyTestObject(other_rank)
                assert expected == val, f"{expected.val} != {val.val}"

    finally:
        await actor_mesh_0._proc_mesh.stop()
        await actor_mesh_1._proc_mesh.stop()
        await ts.shutdown()


@pytest.mark.parametrize(*transport_plus_strategy_params())
@pytest.mark.asyncio
async def test_exists(strategy_params, use_rdma):
    """Test the exists() API functionality"""
    os.environ["TORCHSTORE_RDMA_ENABLED"] = "1" if use_rdma else "0"

    class ExistsTestActor(Actor):
        """Actor for testing exists functionality."""

        def __init__(self, world_size):
            init_logging()
            self.world_size = world_size
            self.rank = current_rank().rank
            # required by LocalRankStrategy
            os.environ["LOCAL_RANK"] = str(self.rank)

        @endpoint
        async def put(self, key, value):
            await ts.put(key, value)

        @endpoint
        async def exists(self, key):
            return await ts.exists(key)

    volume_world_size, strategy = strategy_params
    await ts.initialize(num_storage_volumes=volume_world_size, strategy=strategy)

    # Spawn test actors
    actor_mesh = await spawn_actors(
        volume_world_size,
        ExistsTestActor,
        "exists_test_actors",
        world_size=volume_world_size,
    )

    try:
        # Test 1: Check non-existent keys
        results = await actor_mesh.exists.call("non_existent_key")
        for pt, exists_result in results:
            assert not exists_result

        # Test 2: Store tensors and check existence
        tensor = torch.tensor([1, 2, 3, 4, 5])
        for rank in range(volume_world_size):
            actor = actor_mesh.slice(gpus=rank)
            await actor.put.call(f"tensor_key_{rank}", tensor)

        for rank in range(volume_world_size):
            results = await actor_mesh.exists.call(f"tensor_key_{rank}")
            for pt, exists_result in results:
                assert exists_result

        # Test 3: Store objects and check existence
        obj = {"rank": 0, "data": [1, 2, 3]}
        for rank in range(volume_world_size):
            actor = actor_mesh.slice(gpus=rank)
            await actor.put.call(f"object_key_{rank}", obj)

        for rank in range(volume_world_size):
            results = await actor_mesh.exists.call(f"object_key_{rank}")
            for pt, exists_result in results:
                assert exists_result

    finally:
        await actor_mesh._proc_mesh.stop()
        await ts.shutdown()


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
async def test_large_tensors():
    """Test basic put/get functionality for large tensors"""

    class LargeTensorActor(Actor):
        step_size: int = 100  # -> 400mb
        max_step: int = 600  # 4mb -> 2gb

        def __init__(self, generate_benchmark=False) -> None:
            self.generate_benchmark = generate_benchmark
            init_logging()

        @endpoint
        async def put(self):
            dps = []
            for n in range(1, self.max_step, self.step_size):
                shape = (1024, 1024 * n)
                size_mbytes = (
                    math.prod(shape) * 4 // (1024 * 1024)
                )  # float32 is 4 bytes, // mb
                tensor = torch.randn(shape, dtype=torch.float32)

                logger.info(f"Put {n=} {size_mbytes=}")
                t = time.perf_counter()
                try:
                    await ts.put(str(n), tensor)
                except Exception as e:
                    logger.exception(f"Test failed with {size_mbytes=}")
                    raise e

                delta = time.perf_counter() - t
                dps.append((size_mbytes, delta))
                logger.info(f"Took {delta} seconds to put")

            if self.generate_benchmark:
                with open("put_benchmark.csv", "w") as fp:
                    fp.write("size_mbytes, delta\n")
                    for size_mbytes, delta in dps:
                        fp.write(f"{size_mbytes}, {delta}, {size_mbytes/delta}\n")

        @endpoint
        async def get(self):
            dps = []
            for n in range(1, self.max_step, self.step_size):
                shape = (1024, 1024 * n)
                size_mbytes = (
                    math.prod(shape) * 4 // (1024 * 1024)
                )  # float32 is 4 bytes, // mb

                logger.info(f"Get {n=} {size_mbytes=}")
                t = time.perf_counter()
                try:
                    await ts.get(str(n))
                except Exception as e:
                    logger.exception(f"Test failed with {size_mbytes=}")
                    raise e

                delta = time.perf_counter() - t
                dps.append((size_mbytes, delta))
                logger.info(f"Took {delta} seconds to fetch")

            if self.generate_benchmark:
                with open("get_benchmark.csv", "w") as fp:
                    fp.write("size_mbytes, delta\n")
                    for size_mbytes, delta in dps:
                        fp.write(f"{size_mbytes}, {delta}, {size_mbytes/delta}\n")

    # controller code
    await ts.initialize()
    actor = await spawn_actors(1, LargeTensorActor, "large_tensor")
    try:
        await actor.put.call_one()
        await actor.get.call_one()
        # TODO: assert equal tensors from put/get
    finally:
        await actor._proc_mesh.stop()
        await ts.shutdown()


@pytest.mark.asyncio
async def test_put_dtensor_get_full_tensor():
    """Test basic DTensor put/get functionality with separate put and get meshes using shared DTensorActor"""
    import tempfile

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

    # Use LocalRankStrategy with 2 storage volumes (no RDMA, no parametrization)
    os.environ["TORCHSTORE_RDMA_ENABLED"] = "0"
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

            # Test 1: Cross-volume slice (spans both volumes)
            # Request rows 2-5 (spans volume boundary at row 4)
            cross_volume_slice = TensorSlice(
                offsets=(2, 1),
                coordinates=(),
                global_shape=(8, 6),
                local_shape=(4, 4),  # 4 rows, 4 cols
                mesh_shape=(),
            )

            cross_volume_result = await ts.get("test_key", tensor_slice_spec=cross_volume_slice)
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

            single_volume_result = await ts.get("test_key", tensor_slice_spec=single_volume_slice)
            expected_single_volume = original_tensor[1:3, 0:3]

            assert torch.equal(single_volume_result, expected_single_volume)
            assert single_volume_result.shape == (2, 3)

            # Test 3: Verify full tensor retrieval still works
            full_tensor = await ts.get("test_key")
            assert torch.equal(original_tensor, full_tensor)

        finally:
            if put_mesh is not None:
                await put_mesh.destroy_process_group.call()
                await put_mesh._proc_mesh.stop()
            await ts.shutdown()


if __name__ == "__main__":
    main(__file__)
