import os
import unittest
from logging import getLogger

import torch

from monarch.actor import Actor, current_rank, endpoint

import torchstore as ts
from torchstore.logging import init_logging
from torchstore.utils import spawn_actors
from torchstore.controller import LocalRankStrategy

init_logging()
logger = getLogger(__name__)


class TestActor(Actor):
    """Each instance of this actor represents a single process."""

    def __init__(self, actor_init, world_size):
        init_logging()
        actor_init()
        self.rank = current_rank().rank
        self.world_size = world_size

    def rlog(self, msg):
        logger.info(f"rank: {self.rank} {msg}")

    @endpoint
    async def do_put(self):
        self.rlog("do_put")
        t = torch.tensor([self.rank+1] * 10)
        await ts.put(f"key_{self.rank}", t)

    @endpoint
    async def do_get(self):
        self.rlog("do_get")
        return await ts.get(f"key_{self.rank}")
        
    async def do_get_other_rank(self):
        self.rlog("do_get_other_rank")
        other_rank = (self.rank + 1) % self.world_size
        return await ts.get(f"key_{other_rank}")

class TestStore(unittest.IsolatedAsyncioTestCase):
    async def test_local_rank(self):
        """Test basic put/get functionality for multiple processes"""
        await ts.initialize_store(
            num_storage_volumes=2,
            strategy=LocalRankStrategy()
        )

        def actor_init():
            os.environ["LOCAL_RANK"] = str(current_rank().rank)
            
        world_size = 2
        # each actor mesh represents a group of processes.
        actor_mesh_0 = await spawn_actors(2, TestActor, "actor_mesh_0", actor_init=actor_init, world_size=world_size)
        actor_mesh_1 = await spawn_actors(2, TestActor, "actor_mesh_1", actor_init=actor_init, world_size=world_size)

        await actor_mesh_0.do_put.call()
        tensors = await actor_mesh_1.do_get.call()
        for pt, val in tensors:
            expected = torch.tensor([pt.rank+1] * 10)
            assert torch.equal(expected, val), f"{expected} != {val}"

        tensors = await actor_mesh_1.do_get.call()

if __name__ == "__main__":
    unittest.main()
