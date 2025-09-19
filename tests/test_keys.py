# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the keys() api"""

import os

import pytest
import torch

import torchstore as ts
from monarch.actor import Actor, current_rank, endpoint
from pytest_unordered import unordered
from torchstore.logging import init_logging
from torchstore.utils import spawn_actors


@pytest.mark.asyncio
async def test_keys_basic():
    """Test basic put/get functionality"""
    await ts.initialize()

    await ts.put("", torch.tensor([1, 2, 3]))
    await ts.put(".x", torch.tensor([1, 2, 3]))
    await ts.put("v0.x", torch.tensor([1, 2, 3]))
    await ts.put("v0.y", torch.tensor([4, 5, 6]))
    await ts.put("v0.x.z", torch.tensor([7, 8, 9]))
    await ts.put("v1.x", torch.tensor([7, 8, 9]))
    await ts.put("v1.y", torch.tensor([10, 11, 12]))

    assert await ts.keys() == unordered(
        ["", ".x", "v0.x", "v0.y", "v0.x.z", "v1.x", "v1.y"]
    )
    assert await ts.keys("v0") == unordered(["v0.x", "v0.y", "v0.x.z"])
    assert await ts.keys("v0.x") == unordered(["v0.x", "v0.x.z"])
    assert await ts.keys("v0.x.z") == unordered(["v0.x.z"])
    assert await ts.keys("") == unordered(["", ".x"])
    assert await ts.keys("v1") == unordered(["v1.x", "v1.y"])

    await ts.shutdown()


@pytest.mark.asyncio
async def test_keys_multi_process():
    class PutActor(Actor):
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
            t1 = torch.tensor([self.rank + 1] * 10)
            await ts.put(f"key_{self.rank:05d}.t1", t1)
            t2 = torch.tensor([self.rank + 2] * 10)
            await ts.put(f"key_{self.rank:05d}.t2", t2)

    volume_world_size, strategy = 4, ts.LocalRankStrategy()
    await ts.initialize(num_storage_volumes=volume_world_size, strategy=strategy)

    actor_mesh = await spawn_actors(
        volume_world_size, PutActor, "actor_mesh_0", world_size=volume_world_size
    )
    await actor_mesh.put.call()
    assert len(await ts.keys()) == volume_world_size * 2
    for rank in range(volume_world_size):
        assert await ts.keys(f"key_{rank:05d}") == unordered(
            [f"key_{rank:05d}.t1", f"key_{rank:05d}.t2"]
        )

    await ts.shutdown()


if __name__ == "__main__":
    pytest.main([__file__])
