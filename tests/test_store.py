import os
import time
import math
import pytest
from logging import getLogger

import torch
from monarch.actor import Actor, current_rank, endpoint

import torchstore
import torchstore as ts
from torchstore.logging import init_logging
from torchstore.utils import spawn_actors

init_logging()
logger = getLogger(__name__)



def strategy_params():
    return "volume_world_size, strategy", [
        (2, ts.LocalRankStrategy()),
        (1, None), # Singleton
    ]

@pytest.mark.parametrize(
    "volume_world_size, strategy", [
        (2, ts.LocalRankStrategy()),
        (1, None), # Singleton
    ]
)
@pytest.mark.asyncio
async def test_basic(volume_world_size, strategy):
    """Test basic put/get functionality for multiple processes"""
    
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

        def rlog(self, msg):
            logger.info(f"rank: {self.rank} {msg}")

        @endpoint
        async def put(self):
            self.rlog("pyt")
            t = torch.tensor([self.rank+1] * 10)
            await ts.put(f"key_{self.rank}", t)
        @endpoint
        async def get(self, rank_offset=0):
            other_rank = (self.rank + rank_offset) % self.world_size
            self.rlog(f"get {other_rank=}")
            return await ts.get(f"key_{other_rank}")

    await torchstore.initialize(
        num_storage_volumes=volume_world_size,
        strategy=strategy
    )
    # each actor mesh represents a group of processes.
    actor_mesh_0 = await spawn_actors(volume_world_size, PutGetActor, "actor_mesh_0", world_size=volume_world_size)
    actor_mesh_1 = await spawn_actors(volume_world_size, PutGetActor, "actor_mesh_1", world_size=volume_world_size)

    try:
        await actor_mesh_0.put.call()
        tensors = await actor_mesh_1.get.call()
        for pt, val in tensors:
            expected = torch.tensor([pt.rank+1] * 10)
            assert torch.equal(expected, val), f"{expected} != {val}"

        # in cases where volume_world_size > 1, we should also test that we can get from a different rank
        rank_offset = 1
        tensors = await actor_mesh_1.get.call(rank_offset)
        for pt, val in tensors:
            other_rank = (pt.rank + rank_offset) % volume_world_size
            expected = torch.tensor([other_rank+1] * 10)
            assert torch.equal(expected, val), f"{expected} != {val}"
    finally:
        await actor_mesh_0._proc_mesh.stop()
        await actor_mesh_1._proc_mesh.stop()
        await ts.teardown_store()

@pytest.mark.asyncio
async def test_large_tensors():
    """Test basic put/get functionality for large tensors"""
    class LargeTensorActor(Actor):
        step_size: int = 100 # -> 400mb
        max_step: int = 600 # 4mb -> 2gb

        def __init__(self, store, generate_benchmark=False) -> None:
            self.store=store
            self.generate_benchmark = generate_benchmark
            init_logging()

        @endpoint
        async def put(self):
            dps = []
            for n in range(1, self.max_step, self.step_size):
                shape = (1024, 1024 * n)                      
                size_mbytes = math.prod(shape) * 4 // (1024 * 1024)  # float32 is 4 bytes, // mb                                        
                tensor = torch.randn(shape, dtype=torch.float32) 
                
                logger.info(f"Put {n=} {size_mbytes=}")
                t = time.perf_counter()
                try:                        
                    await self.store.put(str(n), tensor)
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
                size_mbytes = math.prod(shape) * 4 // (1024 * 1024)  # float32 is 4 bytes, // mb
                
                logger.info(f"Get {n=} {size_mbytes=}") 
                t = time.perf_counter()
                try:
                    await self.store.get(str(n))
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
    store = await ts.initialize()
    actor = await spawn_actors(1, LargeTensorActor, "large_tensor", store=store)
    await actor.put.call_one()
    await actor.get.call_one()
    #TODO: assert equal tensors from put/get



if __name__ == "__main__":
    init_logging()
    pytest.main([__file__])
