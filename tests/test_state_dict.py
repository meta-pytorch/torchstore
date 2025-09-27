# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import math
import os
import tempfile
from logging import getLogger
from typing import Union

import pytest

import torch
import torch.distributed.checkpoint as dcp
import torch.nn as nn

import torchstore as ts

from monarch.actor import Actor, current_rank, endpoint
from torch.distributed.checkpoint._nested_dict import flatten_state_dict
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor
from torchstore.utils import spawn_actors

from .utils import main, transport_plus_strategy_params

logger = getLogger(__name__)


MODEL_LINER_LENGTH = 10


class UnitModule(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.l1 = nn.Linear(MODEL_LINER_LENGTH, MODEL_LINER_LENGTH, device=device)
        self.seq = nn.Sequential(
            nn.ReLU(),
            nn.Linear(MODEL_LINER_LENGTH, MODEL_LINER_LENGTH, device=device),
            nn.ReLU(),
        )
        self.l2 = nn.Linear(MODEL_LINER_LENGTH, MODEL_LINER_LENGTH, device=device)

    def forward(self, x):
        return self.l2(self.seq(self.l1(x)))


class CompositeParamModel(nn.Module):
    """
    ref:
    https://github.com/pytorch/pytorch/blob/e2c9d8d6414927ce754bbc40b767edf103cf16da/torch/testing/_internal/common_dist_composable.py#L52
    """

    def __init__(self, device: Union[torch.device, str] = "cpu"):
        super().__init__()
        if isinstance(device, str):
            device = torch.device(device)

        self.l = nn.Linear(MODEL_LINER_LENGTH, MODEL_LINER_LENGTH, device=device)
        self.u1 = UnitModule(device)
        self.u2 = UnitModule(device)
        self.p = nn.Parameter(
            torch.randn((MODEL_LINER_LENGTH, MODEL_LINER_LENGTH), device=device)
        )
        # TODO: buffers are failing atm, because they are not DTensors and thus
        # have unique values on each rank. This isn't necessarily a bug,
        # but it makes it a little harder to compare against a DCP checkpoint directly
        # self.register_buffer(
        #     "buffer", torch.randn((MODEL_LINER_LENGTH, MODEL_LINER_LENGTH), device=device), persistent=True
        # )

    def forward(self, x):
        a = self.u2(self.u1(self.l(x)))
        b = self.p
        return torch.mm(a, b)


class DCPParityTest(Actor):
    """Since DCP is known to have resharding support, this test uses DCP as a proxy to confirm
    correctness of torchstore resharding.
    """

    torchstore_checkpoint_fn: str = "torchstore_checkpoint.pt"

    def __init__(self, mesh_shape, dcp_checkpoint_fn, file_store_name):
        self.mesh_shape = mesh_shape
        self.world_size = math.prod(mesh_shape)
        self.dcp_checkpoint_fn = dcp_checkpoint_fn
        self.file_store_name = file_store_name
        self.rank = current_rank().rank
        # needed for LocalRankStrategy
        os.environ["LOCAL_RANK"] = str(self.rank)

    def rlog(self, msg):
        logger.info(f"rank: {self.rank} {msg}")

    def build_model_optimizer(self):
        mesh_dim_names = ["dp", "tp"] if len(self.mesh_shape) == 2 else None
        device_mesh = init_device_mesh(
            "cpu", self.mesh_shape, mesh_dim_names=mesh_dim_names
        )

        model = CompositeParamModel()
        model = fully_shard(model, mesh=device_mesh)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        return model, optimizer

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

    @endpoint
    async def do_put(self):
        self.initialize_distributed()

        torch.manual_seed(0)

        model, optimizer = self.build_model_optimizer()
        for _ in range(5):
            optimizer.zero_grad()
            loss = model(torch.randn(8, MODEL_LINER_LENGTH)).sum()
            loss.backward()
            optimizer.step()

        state_dict = {
            "model": get_model_state_dict(model),
            "optimizer": get_optimizer_state_dict(model, optimizer),
        }

        dcp.save(state_dict, checkpoint_id=self.dcp_checkpoint_fn)
        await ts.put_state_dict(state_dict, "v0")

    @endpoint
    async def do_get(self):
        self.initialize_distributed()

        model, optimizer = self.build_model_optimizer()
        state_dict = {
            "model": get_model_state_dict(model),
            "optimizer": get_optimizer_state_dict(model, optimizer),
        }
        dcp_state_dict = copy.deepcopy(state_dict)
        dcp.load(dcp_state_dict, checkpoint_id=self.dcp_checkpoint_fn)

        torchstore_state_dict = copy.deepcopy(state_dict)
        await ts.get_state_dict("v0", torchstore_state_dict)

        return dcp_state_dict, torchstore_state_dict


@pytest.mark.parametrize(*transport_plus_strategy_params())
@pytest.mark.asyncio
async def test_state_dict(strategy_params, use_rdma):
    os.environ["TORCHSTORE_RDMA_ENABLED"] = "1" if use_rdma else "0"

    class Trainer(Actor):
        # Monarch RDMA does not work outside of an actor, so we need
        # to wrapp this test first
        # TODO: assert this within rdma buffer
        def __init__(self) -> None:
            self.rank = current_rank().rank
            # needed for LocalRankStrategy
            os.environ["LOCAL_RANK"] = str(self.rank)

        @endpoint
        async def do_test(self):
            model = CompositeParamModel()
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

            for _ in range(5):
                optimizer.zero_grad()
                loss = model(torch.randn(8, MODEL_LINER_LENGTH)).sum()
                loss.backward()
                optimizer.step()

            state_dict = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            await ts.put_state_dict(state_dict, "v0")

            fetched_state_dict = await ts.get_state_dict("v0")
            return state_dict, fetched_state_dict

    _, strategy = strategy_params
    await ts.initialize(num_storage_volumes=1, strategy=strategy)
    trainer = await spawn_actors(1, Trainer, "trainer")
    try:
        state_dict, fetched_state_dict = await trainer.do_test.call_one()
    finally:
        await ts.shutdown()
    _assert_equal_state_dict(state_dict, fetched_state_dict)


@pytest.mark.skip("TODO(kaiyuan-li@): fix this test")
@pytest.mark.parametrize(*transport_plus_strategy_params())
@pytest.mark.asyncio
async def test_dcp_sharding_parity(strategy_params, use_rdma):
    os.environ["TORCHSTORE_RDMA_ENABLED"] = "1" if use_rdma else "0"

    for save_mesh_shape, get_mesh_shape in [
        ((2,), (4,)),
        ((4,), (2,)),
        ((2, 2), (4,)),
        ((2,), (2, 4)),
        ((4, 2), (2, 4)),
    ]:
        save_world_size = math.prod(save_mesh_shape)
        get_world_size = math.prod(get_mesh_shape)
        logger.info(
            f"Testing -- save_mesh_shape: {save_mesh_shape} get_mesh_shape: {get_mesh_shape}"
        )

        _, strategy = strategy_params
        await ts.initialize(
            num_storage_volumes=save_world_size if strategy is not None else 1,
            strategy=strategy,
        )
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                dcp_checkpoint_fn = os.path.join(tmpdir, "dcp_checkpoint.pt")

                save_world = await spawn_actors(
                    save_world_size,
                    DCPParityTest,
                    "save_world",
                    mesh_shape=save_mesh_shape,
                    dcp_checkpoint_fn=dcp_checkpoint_fn,
                    file_store_name=os.path.join(tmpdir, "save_world"),
                )
                await save_world.do_put.call()

                get_world = await spawn_actors(
                    get_world_size,
                    DCPParityTest,
                    "get_world",
                    mesh_shape=get_mesh_shape,
                    dcp_checkpoint_fn=dcp_checkpoint_fn,
                    file_store_name=os.path.join(tmpdir, "get_world"),
                )
                value_mesh = await get_world.do_get.call()
                for coord, val in value_mesh:
                    try:
                        dcp_state_dict, torchstore_state_dict = val
                        _assert_equal_state_dict(dcp_state_dict, torchstore_state_dict)
                    except Exception as e:
                        raise AssertionError(
                            f"Assertion failed on rank {coord.rank} ({save_mesh_shape=} {get_mesh_shape=}): {e}"
                        ) from e
        finally:
            await save_world._proc_mesh.stop()
            await get_world._proc_mesh.stop()
            await ts.shutdown()


def _assert_equal_state_dict(state_dict1, state_dict2):
    flattened_state_dict_1, _ = flatten_state_dict(state_dict1)
    flattened_state_dict_2, _ = flatten_state_dict(state_dict2)

    assert len(flattened_state_dict_1) == len(
        flattened_state_dict_2
    ), f"{flattened_state_dict_1.keys()=}\n{flattened_state_dict_2.keys()=}"
    for key in flattened_state_dict_1:

        assert key in flattened_state_dict_2
        if isinstance(flattened_state_dict_1[key], torch.Tensor):
            t1, t2 = flattened_state_dict_1[key], flattened_state_dict_2[key]
            if isinstance(t1, DTensor):
                t1 = t1._local_tensor
            if isinstance(t2, DTensor):
                t2 = t2._local_tensor

            assert torch.equal(t1, t2), (
                f"{key=} {flattened_state_dict_1[key]=} {t1.shape=} {flattened_state_dict_2[key]=} {t2.shape=}",
            )
        else:
            assert (
                flattened_state_dict_1[key] == flattened_state_dict_2[key]
            ), f"{key=} {flattened_state_dict_1[key]=} {flattened_state_dict_2[key]=}"


if __name__ == "__main__":
    main(__file__)
