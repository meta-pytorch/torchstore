import os
import unittest
from logging import getLogger

import torch

from monarch.actor import Actor, current_rank, endpoint

from torchstore import MultiProcessStore
from torchstore.utils import spawn_actors

logger = getLogger(__name__)


class TestActor(Actor):
    """Each instance of this actor represents a single process."""

    def __init__(self, store):
        self.rank = current_rank().rank
        self.store = store

    def rlog(self, msg):
        logger.warning(f"rank: {self.rank} {msg}")

    @endpoint
    async def do_put(self):
        self.rlog("do_put")
        t = torch.tensor([self.rank] * 10)
        await self.store.put(f"key_{self.rank}", t)

    @endpoint
    async def do_get(self):
        self.rlog("do_get")
        return await self.store.get(f"key_{self.rank}")


class TestStore(unittest.IsolatedAsyncioTestCase):
    async def test_basic(self):
        """Test basic put/get functionality for multiple processes"""
        store = await MultiProcessStore.create_store()

        # each actor mesh represents a group of processes.
        actor_mesh_0 = await spawn_actors(2, TestActor, "actor_mesh_0", store=store)
        actor_mesh_1 = await spawn_actors(2, TestActor, "actor_mesh_1", store=store)

        await actor_mesh_0.do_put.call()
        tensors = await actor_mesh_1.do_get.call()
        for pt, val in tensors:
            assert torch.equal(torch.tensor([pt.rank] * 10), val)

    async def test_get_slice(self):
        """Test get_slice functionality with offsets and local_shape"""
        store = await MultiProcessStore.create_store()

        test_tensor = torch.arange(24).reshape(4, 6)
        global_shape = (4, 6)

        await store.put("test_tensor", test_tensor)

        # Test 1: Get first 2 rows (all columns)
        # Equivalent to [:2, :]
        offsets = (0, 0)
        local_shape = (2, 6)
        slice_result = await store.get_slice("test_tensor", offsets, local_shape)
        expected = test_tensor[:2, :]
        assert torch.equal(
            slice_result, expected
        ), f"Expected {expected}, got {slice_result}"

        # Test 2: Get specific columns from all rows
        # Equivalent to [:, 1:4]
        offsets = (0, 1)
        local_shape = (4, 3)
        slice_result = await store.get_slice("test_tensor", offsets, local_shape)
        expected = test_tensor[:, 1:4]
        assert torch.equal(
            slice_result, expected
        ), f"Expected {expected}, got {slice_result}"

        # Test 3: Get a specific subregion
        # Equivalent to [1:3, 2:5]
        offsets = (1, 2)
        local_shape = (2, 3)
        slice_result = await store.get_slice("test_tensor", offsets, local_shape)
        expected = test_tensor[1:3, 2:5]
        assert torch.equal(
            slice_result, expected
        ), f"Expected {expected}, got {slice_result}"

        # Test 4: Get single row
        # Equivalent to [2:3, :]
        offsets = (2, 0)
        local_shape = (1, 6)
        slice_result = await store.get_slice("test_tensor", offsets, local_shape)
        expected = test_tensor[2:3, :]
        assert torch.equal(
            slice_result, expected
        ), f"Expected {expected}, got {slice_result}"

        # Test 5: Get single element region
        # Equivalent to [1:2, 3:4]
        offsets = (1, 3)
        local_shape = (1, 1)
        slice_result = await store.get_slice("test_tensor", offsets, local_shape)
        expected = test_tensor[1:2, 3:4]
        assert torch.equal(
            slice_result, expected
        ), f"Expected {expected}, got {slice_result}"

        # Test 6: Test last 2 rows
        # Equivalent to [-2:, :] which is [2:, :]
        offsets = (2, 0)
        local_shape = (2, 6)
        slice_result = await store.get_slice("test_tensor", offsets, local_shape)
        expected = test_tensor[-2:, :]
        assert torch.equal(
            slice_result, expected
        ), f"Expected {expected}, got {slice_result}"

        # Test 7: Get middle section
        # Equivalent to [1:3, 1:5]
        offsets = (1, 1)
        local_shape = (2, 4)
        slice_result = await store.get_slice("test_tensor", offsets, local_shape)
        expected = test_tensor[1:3, 1:5]
        assert torch.equal(
            slice_result, expected
        ), f"Expected {expected}, got {slice_result}"

        print("All get_slice tests passed!")


if __name__ == "__main__":
    unittest.main()
