# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import socket
from logging import getLogger

import pytest
import torch
import torchstore as ts
from monarch.actor import Actor, current_rank, endpoint
from torchstore.logging import init_logging
from torchstore.utils import spawn_actors

from .utils import main, set_transport_type, transport_plus_strategy_params

init_logging()
logger = getLogger(__name__)


# Ensure we have some kind of basic coverage for HostStrategy
@pytest.mark.parametrize(*transport_plus_strategy_params(with_host_strategy=True))
@pytest.mark.asyncio
async def test_basic(strategy_params, transport_type):
    """Test basic put/get functionality for multiple processes"""
    # set_transport_type(transport_type)

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
            # required by HostStrategy
            os.environ["HOSTNAME"] = socket.gethostname()

        @endpoint
        async def put(self):
            t = torch.tensor([self.rank + 1] * 10)
            await ts.put(f"key_{self.rank}", t)

        @endpoint
        async def get(self, rank_offset=0):
            other_rank = (self.rank + rank_offset) % self.world_size
            return await ts.get(f"key_{other_rank}")

    volume_world_size, strategy = strategy_params
    await ts.initialize(
        num_storage_volumes=volume_world_size,
        strategy=strategy(transport_type)
        )
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
        # TODO: Investigate monarch bug with proc_mesh.stop()
        # await actor_mesh_0._proc_mesh.stop()
        # await actor_mesh_1._proc_mesh.stop()
        await ts.shutdown()


@pytest.mark.parametrize(*transport_plus_strategy_params())
@pytest.mark.asyncio
async def test_objects(strategy_params, transport_type):
    """Test put/get on arbitrary object"""
    set_transport_type(transport_type)

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
    await ts.initialize(
        num_storage_volumes=volume_world_size, #TODO: volume_world_size should potentially be in strategy.
        strategy=strategy(transport_type)
    )
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
        # TODO: Investigate monarch bug with proc_mesh.stop()
        # await actor_mesh_0._proc_mesh.stop()
        # await actor_mesh_1._proc_mesh.stop()
        await ts.shutdown()


@pytest.mark.parametrize(*transport_plus_strategy_params())
@pytest.mark.asyncio
async def test_exists(strategy_params, transport_type):
    """Test the exists() API functionality"""
    set_transport_type(transport_type)

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
    await ts.initialize(num_storage_volumes=volume_world_size, strategy=strategy(transport_type))

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
        # TODO: Investigate monarch bug with proc_mesh.stop()
        # await actor_mesh._proc_mesh.stop()
        await ts.shutdown()


@pytest.mark.parametrize(*transport_plus_strategy_params())
@pytest.mark.asyncio
async def test_delete(strategy_params, transport_type):
    """Test the delete() API functionality"""
    set_transport_type(transport_type) # TODO: BOOOOOOO

    class DeleteTestActor(Actor):
        """Actor for testing delete functionality."""

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
        async def delete(self, key):
            await ts.delete(key)

        @endpoint
        async def exists(self, key):
            return await ts.exists(key)

        @endpoint
        async def get(self, key):
            return await ts.get(key)

    volume_world_size, strategy = strategy_params
    await ts.initialize(num_storage_volumes=volume_world_size, strategy=strategy)

    # Spawn test actors
    actor_mesh = await spawn_actors(
        volume_world_size,
        DeleteTestActor,
        "delete_test_actors",
        world_size=volume_world_size,
    )

    try:
        # Test 1: Store tensors, verify they exist, then delete them
        tensor = torch.tensor([1, 2, 3, 4, 5])
        for rank in range(volume_world_size):
            actor = actor_mesh.slice(gpus=rank)
            await actor.put.call(f"tensor_key_{rank}", tensor)

        # Verify all tensors exist
        for rank in range(volume_world_size):
            results = await actor_mesh.exists.call(f"tensor_key_{rank}")
            for _, exists_result in results:
                assert exists_result

        # Delete tensors one at a time and verify each deletion
        for rank in range(volume_world_size):
            actor = actor_mesh.slice(gpus=rank)
            await actor.delete.call(f"tensor_key_{rank}")

            # Verify this specific tensor no longer exists
            results = await actor_mesh.exists.call(f"tensor_key_{rank}")
            for _, exists_result in results:
                assert not exists_result

            # Verify other tensors still exist (if any remain)
            for other_rank in range(rank + 1, volume_world_size):
                results = await actor_mesh.exists.call(f"tensor_key_{other_rank}")
                for _, exists_result in results:
                    assert exists_result

        # Test 2: Try to get deleted tensor (should raise exception)
        with pytest.raises(Exception):
            await actor_mesh.get.call("tensor_key_0")

    finally:
        # TODO: Investigate monarch bug with proc_mesh.stop()
        # await actor_mesh._proc_mesh.stop()
        await ts.shutdown()


@pytest.mark.asyncio
async def test_key_miss():
    """Test the behavior of get() when the key is missing."""

    class TestActor(Actor):
        @endpoint
        async def test(self) -> Exception or None:
            try:
                key = "foo"
                value = torch.tensor([1, 2, 3])
                await ts.put(key, value)

                # # Get the value back
                # retrieved_value = await ts.get(key)
                # assert torch.equal(value, retrieved_value)

                # # Get a missing key
                # with pytest.raises(KeyError):
                #     await ts.get("bar")
            except Exception as e:
                return e

    await ts.initialize()

    actor = await spawn_actors(1, TestActor, "actor_0")
    err = await actor.test.call_one()

    assert err is None

    await ts.shutdown()


if __name__ == "__main__":
    main(__file__)
