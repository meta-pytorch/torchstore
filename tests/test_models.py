# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import tempfile
import time
from logging import getLogger

import pytest
import torch

import torchstore as ts
from monarch.actor import Actor, current_rank, endpoint
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torchstore.state_dict_utils import _state_dict_size
from torchstore.utils import spawn_actors
from transformers import AutoModelForCausalLM

from .utils import main, transport_plus_strategy_params

logger = getLogger(__name__)

needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)


# Skip all tests in this module if HF_TOKEN is not available
pytestmark = pytest.mark.skipif(
    os.environ.get("HF_TOKEN", None) is None,
    reason="HF_TOKEN not available - skipping Transformers model tests",
)

TEST_MODEL = "Qwen/Qwen3-1.7B"  # ~4GB
# TEST_MODEL = "meta-llama/Llama-3.1-8B" # ~ 16GB


class ModelTest(Actor):
    def __init__(self, mesh_shape, file_store_name):
        ts.init_logging()
        self.rank = current_rank().rank
        self.mesh_shape = mesh_shape
        self.world_size = math.prod(mesh_shape)
        self.file_store_name = file_store_name

        os.environ["LOCAL_RANK"] = str(self.rank)

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
        self.rlog(f"State dict size: {_state_dict_size(model.state_dict())}")
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
        print(f"rank: {self.rank} {msg}")
        self.logger.info(f"rank: {self.rank} {msg}")
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
        await ts.put_state_dict(state_dict, "v0")
        self.rlog(f"pushed state dict in {time.perf_counter() - t} seconds")

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
        await ts.get_state_dict("v0", state_dict)
        self.rlog(f"got state dict in {time.perf_counter() - t} seconds")


@pytest.mark.parametrize(*transport_plus_strategy_params())
@pytest.mark.asyncio
async def test_basic(strategy_params, use_rdma):
    # FSDP
    put_mesh_shape = (1,)
    get_mesh_shape = (1,)
    await _do_test(put_mesh_shape, get_mesh_shape, strategy_params[1], use_rdma)


@pytest.mark.parametrize(*transport_plus_strategy_params())
@pytest.mark.asyncio
async def test_resharding(strategy_params, use_rdma):
    # FSDP
    put_mesh_shape = (4,)
    get_mesh_shape = (8,)
    await _do_test(put_mesh_shape, get_mesh_shape, strategy_params[1], use_rdma)


async def _do_test(put_mesh_shape, get_mesh_shape, strategy, use_rdma):
    os.environ["TORCHSTORE_RDMA_ENABLED"] = "1" if use_rdma else "0"

    ts.init_logging()
    logger.info(f"Testing with strategy: {strategy}")

    put_world_size = math.prod(put_mesh_shape)
    await ts.initialize(
        num_storage_volumes=put_world_size if strategy is not None else 1,
        strategy=strategy,
    )
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            put_world_size = math.prod(put_mesh_shape)
            put_world = await spawn_actors(
                put_world_size,
                ModelTest,
                "save_world",
                mesh_shape=put_mesh_shape,
                file_store_name=os.path.join(tmpdir, "save_world"),
            )

            get_world_size = math.prod(get_mesh_shape)
            get_world = await spawn_actors(
                get_world_size,
                ModelTest,
                "get_world",
                mesh_shape=get_mesh_shape,
                file_store_name=os.path.join(tmpdir, "get_world"),
            )

            logger.info("do_push ")
            await put_world.do_push.call()

            await get_world.do_get.call()
    finally:
        await ts.shutdown()


if __name__ == "__main__":
    main([__file__])
