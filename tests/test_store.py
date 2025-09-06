import os
import time
import math
import pytest
from logging import getLogger

import torch
from monarch.actor import Actor, current_rank, endpoint

import torchstore as ts
from torchstore.logging import init_logging
from torchstore.utils import spawn_actors

from .utils import main, transport_plus_strategy_params

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
            t = torch.tensor([self.rank+1] * 10)
            await ts.put(f"key_{self.rank}", t)
        @endpoint
        async def get(self, rank_offset=0):
            other_rank = (self.rank + rank_offset) % self.world_size
            return await ts.get(f"key_{other_rank}")

    volume_world_size, strategy = strategy_params
    await ts.initialize(
        num_storage_volumes=volume_world_size,
        strategy=strategy
    )
    # each actor mesh represents a group of processes.
    actor_mesh_0 = await spawn_actors(volume_world_size, PutGetActor, "actor_mesh_0", world_size=volume_world_size)
    actor_mesh_1 = await spawn_actors(volume_world_size, PutGetActor, "actor_mesh_1", world_size=volume_world_size)

<<<<<<< HEAD
    try:
        await actor_mesh_0.put.call()
        tensors = await actor_mesh_1.get.call()
=======
    @endpoint
    async def do_put(self):
        self.rlog("do_put")
        t = torch.tensor([self.rank + 1] * 10)
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
>>>>>>> main
        for pt, val in tensors:
            expected = torch.tensor([pt.rank + 1] * 10)
            assert torch.equal(expected, val), f"{expected} != {val}"

<<<<<<< HEAD
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
        await ts.shutdown()

=======
    async def test_large_tensors(self):
        """Test basic put/get functionality for multiple processes"""

        class LargeTensorActor(Actor):
            step_size: int = 100
            max_step: int = 600  # 4mb -> 2gb

            def __init__(self, store, generate_benchmark=False) -> None:
                self.store = store
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
                            fp.write(f"{size_mbytes}, {delta}, {size_mbytes / delta}\n")

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
                            fp.write(f"{size_mbytes}, {delta}, {size_mbytes / delta}\n")

        store = await MultiProcessStore.create_store()
        actor = await spawn_actors(1, LargeTensorActor, "large_tensor", store=store)
        await actor.put.call_one()
        await actor.get.call_one()

        # TODO: assert equal tensors from put/get
>>>>>>> main

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

<<<<<<< HEAD
    volume_world_size, strategy = strategy_params
    await ts.initialize(
        num_storage_volumes=volume_world_size,
        strategy=strategy
    )
    # each actor mesh represents a group of processes.
    actor_mesh_0 = await spawn_actors(volume_world_size, ObjectActor, "actor_mesh_0", world_size=volume_world_size)
    actor_mesh_1 = await spawn_actors(volume_world_size, ObjectActor, "actor_mesh_1", world_size=volume_world_size)

    class MyTestObject:
        def __init__(self, val):
            self.val = val

    try:        
        for idx in range(volume_world_size):
            actor = actor_mesh_0.slice(**{"hosts":0, "gpus": idx})
            await actor.put(
                MyTestObject(idx)
            )
=======
        class ScalarTest(Actor):
            def __init__(self, store) -> None:
                self.store = store

            @endpoint
            async def put(self, val):
                await self.store.put("key", val)

            @endpoint
            async def get(self, inplace):
                t = torch.tensor(0.0) if inplace else None
                fetched = await self.store.get("key", t if inplace else None)
                return t if inplace else fetched

        test_actor = await spawn_actors(1, ScalarTest, "scalar", store=store)

        t = torch.tensor(42.0)
        await test_actor.put.call_one(t)
        for inplace in [True, False]:
            fetched = await test_actor.get.call_one(inplace)
            self.assertTrue(torch.equal(t, fetched), f"{t} != {fetched} {inplace=}")

>>>>>>> main

        for rank_offset in (0, 1):
            objects = await actor_mesh_1.get.call(rank_offset=rank_offset)
            for pt, val in objects:
                other_rank = (pt.rank + rank_offset) % volume_world_size
                expected = MyTestObject(other_rank)
                assert torch.equal(expected, val), f"{expected} != {val}"

    finally:
        await actor_mesh_0._proc_mesh.stop()
        await actor_mesh_1._proc_mesh.stop()
        await ts.shutdown()

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
    main(__file__)
