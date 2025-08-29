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


import torch
import monarch
from torchstore.utils import spawn_actors


class Foo(Actor):

    def __init__(self) -> None:
        self.tensor = torch.rand(10)

    @endpoint
    async def get_tensor(self):
        return self.tensor

    @endpoint
    async def source(self, rdma_buffer):
        await rdma_buffer.write_from(self.tensor.view(torch.uint8).flatten())

    @endpoint
    async def destination(self, other_actor):
        tensor = torch.rand(10)
        rdma_buffer = monarch.tensor_engine.RDMABuffer(
            tensor.view(torch.uint8).flatten()
        )
        await rdma_buffer.write_from(tensor.view(torch.uint8).flatten())

        await other_actor.source.call_one(rdma_buffer)
        other_tensor = await other_actor.get_tensor.call_one()

        assert torch.equal(tensor, other_tensor)


async def main():
    actor_0 = await spawn_actors(1, Foo, "foo0")    
    actor_1 = await spawn_actors(1, Foo, "foo1")

    await actor_0.destination.call_one(actor_1)


if __name__ == "__main__":
    import asyncio
    # asyncio.run(main())
    unittest.main()
