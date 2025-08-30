import os
import unittest
from logging import getLogger

import torch
import math

from monarch.actor import Actor, current_rank, endpoint

from torchstore import MultiProcessStore
from torchstore.logging import init_logging
from torchstore.utils import spawn_actors

init_logging()
logger = getLogger(__name__)


class TestActor(Actor):
    """Each instance of this actor represents a single process."""

    def __init__(self, store):
        init_logging()
        self.rank = current_rank().rank
        self.store = store

    def rlog(self, msg):
        logger.info(f"rank: {self.rank} {msg}")

    @endpoint
    async def do_put(self):
        self.rlog("do_put")
        t = torch.tensor([self.rank+1] * 10)
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
            expected = torch.tensor([pt.rank+1] * 10)
            assert torch.equal(expected, val), f"{expected} != {val}"

    async def test_large_tensors(self):
        """Test basic put/get functionality for multiple processes"""
        class LargeTensorActor(Actor):
            def __init__(self, store) -> None:
                self.store=store
                self.rank = current_rank().rank
                init_logging()

            @endpoint
            async def put(self):

                for n in range(1, 600, 100):
                    shape = (1024, 1024 * n)                      
                    size_mbytes = math.prod(shape) * 4 // (1024 * 1024)  # float32 is 4 bytes, // mb
                    # logger.info(f"Testing {n=} {size_mbytes=}")
                    print(f"Testing {n=} {size_mbytes=}", flush=True)
                    try:
                        t = torch.randn(shape, dtype=torch.float32) 
                        await self.store.put(self.rank, t)
                    except Exception as e:
                        logger.exception(f"Test failed with {size_mbytes=}")
                        raise e

        store = await MultiProcessStore.create_store()
        actor = await spawn_actors(1, LargeTensorActor, "large_tensor", store=store)
        await actor.put.call_one()


    async def test_scalar(self):
        """Test basic put/get functionality for multiple processes"""
        store = await MultiProcessStore.create_store()

        class ScalarTest(Actor):
            def __init__(self, store) -> None:
                self.store=store
            @endpoint
            async def put(self, val):
                await self.store.put("key", val)
            @endpoint
            async def get(self, inplace):
                t = torch.tensor(0.) if inplace else None
                fetched = await self.store.get("key", t if inplace else None)
                return t if inplace else fetched

        # each actor mesh represents a group of processes.
        test_actor = await spawn_actors(1, ScalarTest, "scalar", store=store)
        
        # inplace
        t = torch.tensor(42.)
        await test_actor.put.call_one(t)        
        for inplace in [True, False]:
            fetched = await test_actor.get.call_one(inplace)
            self.assertTrue(
                torch.equal(t, fetched), f"{t} != {fetched} {inplace=}"
            )

if __name__ == "__main__":
    unittest.main()
