# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
from itertools import product
from logging import getLogger

from typing import List

import pytest
import torch
import torchstore as ts
from monarch.actor import Actor, current_rank, endpoint
from torch.distributed._tensor import distribute_tensor
from torch.distributed.device_mesh import init_device_mesh

logger = getLogger(__name__)


def main(file):
    ts.init_logging()
    pytest.main([file])


def transport_plus_strategy_params():
    strategies = [
        (2, ts.LocalRankStrategy()),
        (1, None),  # ts.SingletonStrategy
        (1, ts.ControllerStorageVolumes()),
    ]
    rdma_options = (
        [True, False]
        if os.environ.get("TORCHSTORE_RDMA_ENABLED", "0") == "1"
        else [False]
    )

    return "strategy_params, use_rdma", list(product(strategies, rdma_options))


class DTensorActor(Actor):
    """Test class used to verify correctness of resharding across different shardings.
    Currently only supports a single tensor
    """

    shared_key = "test_key"

    def __init__(
        self,
        mesh_shape,
        original_tensor,
        placements,
        file_store_name,
        visible_devices="0,1,2,3,4,5,6,7",
        ranks_to_skip_put: (
            List[int] | None
        ) = None,  # ranks that should skip put operation
    ):
        self.rank = current_rank().rank
        self.mesh_shape = mesh_shape
        self.world_size = math.prod(mesh_shape)
        self.original_tensor = original_tensor
        self.placements = placements
        self.file_store_name = file_store_name
        self.ranks_to_skip_put = ranks_to_skip_put or []

        # torchstore will fail without this (see LocalRankStrategy)
        os.environ["LOCAL_RANK"] = str(self.rank)

        # this is only necessary for nccl, but we're not using it in this test.
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices

    def rlog(self, msg):
        # TODO: set to 'info' once this is fixed in monarch (which currently is hiding logs :/)
        logger.info(f"rank: {self.rank} {msg}")

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

        self.rlog("Create device mesh")
        device_mesh = init_device_mesh("cpu", self.mesh_shape)

        self.rlog("distributing dtensor")
        tensor = self.original_tensor.to("cpu")
        dtensor = distribute_tensor(tensor, device_mesh, placements=self.placements)

        # Skip put if this rank is in the skip list
        if self.rank in self.ranks_to_skip_put:
            self.rlog(f"Skipping put for rank {self.rank}")
            return

        self.rlog(f"calling put with {dtensor=}")
        await ts.put(self.shared_key, dtensor)

    @endpoint
    async def do_get(self):
        self.initialize_distributed()

        self.rlog("Create device mesh")
        # TODO: nccl is giving me a weird error on process group split for 2d mesh
        device_mesh = init_device_mesh("cpu", self.mesh_shape)

        self.rlog("distributing dtensor")
        tensor = self.original_tensor.to("cpu")
        dtensor = distribute_tensor(tensor, device_mesh, placements=self.placements)

        self.rlog(f"calling get with {dtensor=}")
        fetched_tensor = await ts.get(self.shared_key, dtensor)
        self.rlog(f"after fetch: {dtensor=}")
        assert torch.equal(dtensor, fetched_tensor)

        return fetched_tensor, device_mesh.get_coordinate()

    @endpoint
    async def destroy_process_group(self):
        torch.distributed.destroy_process_group()
