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
        store = MultiProcessStore()

        # each actor mesh represents a group of processes.
        actor_mesh_0 = await spawn_actors(2, TestActor, "actor_mesh_0", store=store)
        actor_mesh_1 = await spawn_actors(2, TestActor, "actor_mesh_1", store=store)

        await actor_mesh_0.do_put.call()
        tensors = await actor_mesh_1.do_get.call()
        for pt, val in tensors:
            assert torch.equal(torch.tensor([pt.rank] * 10), val)


if __name__ == "__main__":
    unittest.main()
