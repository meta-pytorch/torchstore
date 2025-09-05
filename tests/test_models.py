# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import tempfile
import time
import unittest
from logging import getLogger

import pytest

import torch

from monarch.actor import Actor, current_rank, endpoint
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

from transformers import AutoModelForCausalLM

from torchstore import MultiProcessStore
from torchstore._state_dict_utils import get_state_dict, push_state_dict
from torchstore.utils import spawn_actors
from torchstore.logging import init_logging

logger = getLogger(__name__)

needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)


assert os.environ.get("HF_TOKEN", None) is not None, "HF_TOKEN must be set"
TEST_MODEL = "Qwen/Qwen3-1.7B"  # ~4GB
# TEST_MODEL = "meta-llama/Llama-3.1-8B" # ~ 16GB


class ModelTest(Actor):
    def __init__(self, store, mesh_shape, file_store_name):
        init_logging()
        self.rank = current_rank().rank
        self.store = store
        self.mesh_shape = mesh_shape
        self.world_size = math.prod(mesh_shape)
        self.file_store_name = file_store_name

    def initialize_distributed(self):
        self.rlog(f"Initialize process group using {self.file_store_name=} ")
        torch.distributed.init_process_group(
            backend="gloo",
            rank=self.rank,
            world_size=self.world_size,
            init_method=f"file://{self.file_store_name}",
        )

        # this barrier is more to make sure torch.distibuted is working
        self.rlog("barrrer")
        torch.distributed.barrier()

    def build_model(self):
        self.rlog("building model")
        model = AutoModelForCausalLM.from_pretrained(
            TEST_MODEL, token=os.environ["HF_TOKEN"]
        )
        if self.world_size > 1:
            self.initialize_distributed()
            self.rlog("sharding")
            mesh_dim_names = ["dp", "tp"] if len(self.mesh_shape) == 2 else None
            device_mesh = init_device_mesh(
                "cpu", self.mesh_shape, mesh_dim_names=mesh_dim_names
            )
            model = fully_shard(model, mesh=device_mesh)

        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
        return model, optimizer

    def rlog(self, msg):
        logger.info(f"rank: {self.rank} {msg}")

    @endpoint
    async def do_push(self):
        model, optimizer = self.build_model()
        state_dict = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

        if self.world_size > 1:
            torch.distributed.barrier()

        self.rlog("pushing state dict")
        t = time.perf_counter()
        await push_state_dict(self.store, state_dict, "v0")
        self.rlog(f"pushed state dict in {time.perf_counter()-t} seconds")

    @endpoint
    async def do_get(self):
        model, optimizer = self.build_model()
        state_dict = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

        if self.world_size > 1:
            torch.distributed.barrier()
        self.rlog("getting state dict")
        t = time.perf_counter()
        await get_state_dict(self.store, "v0", state_dict)
        self.rlog(f"got state dict in {time.perf_counter() - t} seconds")


@needs_cuda
class TestHFModel(unittest.IsolatedAsyncioTestCase):
    async def test_basic(self):
        # FSDP
        put_mesh_shape = (1,)
        get_mesh_shape = (1,)
        await self._do_test(put_mesh_shape, get_mesh_shape)

    async def test_resharding(self):
        # FSDP
        put_mesh_shape = (4,)
        get_mesh_shape = (8,)
        await self._do_test(put_mesh_shape, get_mesh_shape)

    async def _do_test(self, put_mesh_shape, get_mesh_shape):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = await MultiProcessStore.create_store()

            put_world_size = math.prod(put_mesh_shape)
            put_world = await spawn_actors(
                put_world_size,
                ModelTest,
                "save_world",
                store=store,
                mesh_shape=put_mesh_shape,
                file_store_name=os.path.join(tmpdir, "save_world"),
            )

            get_world_size = math.prod(get_mesh_shape)
            get_world = await spawn_actors(
                get_world_size,
                ModelTest,
                "get_world",
                store=store,
                mesh_shape=get_mesh_shape,
                file_store_name=os.path.join(tmpdir, "get_world"),
            )

            logger.info("pushing state dict")
            t = time.perf_counter()
            await put_world.do_push.call()
            logger.info(f"pushing state dict took: {time.perf_counter()-t} seconds")

            logger.info("fetching state dict")
            t = time.perf_counter()
            await get_world.do_get.call()
            logger.info(f"getting state dict took: {time.perf_counter()-t} seconds")


if __name__ == "__main__":
    init_logging()
    unittest.main()
