# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import tempfile
from logging import getLogger
from typing import List, Tuple, Union

import pytest

import torch

import torchstore as ts

from monarch.actor import Actor, current_rank, endpoint
from torch.distributed._tensor import distribute_tensor, Replicate, Shard
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor._utils import _compute_local_shape_and_global_offset
from torchstore.utils import get_local_tensor, spawn_actors

from .utils import main, transport_plus_strategy_params

logger = getLogger(__name__)


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
    ):
        self.rank = current_rank().rank
        self.mesh_shape = mesh_shape
        self.world_size = math.prod(mesh_shape)
        self.original_tensor = original_tensor
        self.placements = placements
        self.file_store_name = file_store_name

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


@pytest.mark.parametrize(*transport_plus_strategy_params())
@pytest.mark.asyncio
async def test_1d_resharding(strategy_params, use_rdma):
    _, strategy = strategy_params

    for put_mesh_shape, get_mesh_shape in [
        ((4,), (2,)),  # shrink
        ((2,), (4,)),  # grow
    ]:
        for put_sharding_dim, get_sharding_dim in [
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
        ]:
            # TODO: test Replicate as well, which is likely not working
            await _test_resharding(
                put_mesh_shape=put_mesh_shape,
                put_placements=[Shard(put_sharding_dim)],
                get_mesh_shape=get_mesh_shape,
                get_placements=[Shard(get_sharding_dim)],
                strategy=strategy,
                use_rdma=use_rdma,
            )


@pytest.mark.parametrize(*transport_plus_strategy_params())
@pytest.mark.asyncio
async def test_2d_to_2d_resharding(strategy_params, use_rdma):
    _, strategy = strategy_params

    put_mesh_shape = get_mesh_shape = (2, 2)
    for put_sharding_dims, get_sharding_dims in [
        ((1, 1), (0, 1)),
        ((1, 0), (1, 0)),
        ((0, 0), (0, 1)),
        ((1, 1), (0, 0)),
    ]:
        await _test_resharding(
            put_mesh_shape=put_mesh_shape,
            put_placements=[Shard(dim) for dim in put_sharding_dims],
            get_mesh_shape=get_mesh_shape,
            get_placements=[Shard(dim) for dim in get_sharding_dims],
            strategy=strategy,
            use_rdma=use_rdma,
        )


@pytest.mark.parametrize(*transport_plus_strategy_params())
@pytest.mark.asyncio
async def test_1d_to_2d_resharding(strategy_params, use_rdma):
    _, strategy = strategy_params

    put_mesh_shape = (4,)
    get_mesh_shape = (2, 2)
    for put_sharding_dims, get_sharding_dims in [
        ((0,), (0, 1)),
        ((1,), (1, 0)),
        ((0,), (0, 0)),
        ((1,), (1, 1)),
    ]:
        await _test_resharding(
            put_mesh_shape=put_mesh_shape,
            put_placements=[Shard(dim) for dim in put_sharding_dims],
            get_mesh_shape=get_mesh_shape,
            get_placements=[Shard(dim) for dim in get_sharding_dims],
            strategy=strategy,
            use_rdma=use_rdma,
        )


@pytest.mark.parametrize(*transport_plus_strategy_params())
@pytest.mark.asyncio
async def test_2d_to_1d_resharding(strategy_params, use_rdma):
    _, strategy = strategy_params

    put_mesh_shape = (2, 2)
    get_mesh_shape = (4,)
    for put_sharding_dims, get_sharding_dims in [
        ((0, 0), (0,)),
        ((1, 0), (1,)),
        ((0, 1), (0,)),
        ((1, 1), (1,)),
    ]:
        await _test_resharding(
            put_mesh_shape=put_mesh_shape,
            put_placements=[Shard(dim) for dim in put_sharding_dims],
            get_mesh_shape=get_mesh_shape,
            get_placements=[Shard(dim) for dim in get_sharding_dims],
            strategy=strategy,
            use_rdma=use_rdma,
        )


@pytest.mark.parametrize(*transport_plus_strategy_params())
@pytest.mark.asyncio
async def test_data_parallel(strategy_params, use_rdma):
    _, strategy = strategy_params

    # # 1d
    put_mesh_shape = (2,)
    get_mesh_shape = (4,)
    placements = [Replicate()]
    await _test_resharding(
        put_mesh_shape=put_mesh_shape,
        put_placements=placements,
        get_mesh_shape=get_mesh_shape,
        get_placements=placements,
        strategy=strategy,
        use_rdma=use_rdma,
    )

    # 2d -> 1d
    put_mesh_shape = (2, 2)
    get_mesh_shape = (4,)
    await _test_resharding(
        put_mesh_shape=put_mesh_shape,
        put_placements=[
            Replicate(),
            Shard(0),
        ],  # maps to default for fsdp's fully_shard
        get_mesh_shape=get_mesh_shape,
        get_placements=[Shard(1)],
        strategy=strategy,
        use_rdma=use_rdma,
    )


async def _test_resharding(
    put_mesh_shape: Tuple[int],
    put_placements: List[Union[Replicate, Shard]],
    get_mesh_shape: Tuple[int],
    get_placements: List[Union[Replicate, Shard]],
    strategy: ts.TorchStoreStrategy,
    use_rdma: bool,
):
    """Given a "put" mesh shape and a "get" mesh shape.
    1. Create separate worlds for each mesh shape, running on different devices /PGs.
    2. Each rank in 'put' world will create a DTensor, and call self.store.put(key="test_key", value=dtensor)
    3. Each rank in 'get' world will create a DTensor (with a different sharding, and seeded with torch.zero),
        and call self.store.get(key="test_key", value=dtensor)
    4. The result of the above operation should be the original DTensor, but resharded between putter/getter worlds

    Example:
    #Our "put" world starts with something like this:
    original_tensor = [0,1,2,3], world_size=4
    dtensor = distribute_tensor(original_tensor)
    # Rank0: dtensor._local_tensor == [0], Rank1: dtensor._local_tensor == [1], Rank2: dtensor._local_tensor == [2], ...
    self.store.put("shared_key", dtensor)

    #Our "put" world starts with something like this:
    original_Tensor = [0, 0, 0, 0], world_size=2
    dtensor = distribute_tensor(original_tensor)
    # Rank0: dtensor._local_tensor == [0,0], Rank1: dtensor._local_tensor == [0,0]
    self.store.get("shared_key", dtensor)

    # Rank0: dtensor._local_tensor == [0,1], Rank1: dtensor._local_tensor == [2,3]
    """
    os.environ["TORCHSTORE_RDMA_ENABLED"] = "1" if use_rdma else "0"

    put_world_size = math.prod(put_mesh_shape)
    get_world_size = math.prod(get_mesh_shape)
    assert (
        put_world_size + get_world_size <= 8
    ), f"{put_world_size} + {get_world_size} > 8!"
    assert len(put_mesh_shape) == len(
        put_placements
    ), f"{put_mesh_shape=}, {put_placements=}"
    assert len(get_mesh_shape) == len(
        get_placements
    ), f"{get_mesh_shape=}, {get_placements=}"

    logger.warn(
        f"Testing {put_mesh_shape=} {put_placements=} {get_mesh_shape=} {get_placements=}"
    )

    original_tensor = torch.arange(8**2).reshape(
        8, 8
    )  # 8x8 square, with ([[0...7],[8...15],[...]])
    await ts.initialize(
        num_storage_volumes=put_world_size if strategy is not None else 1,
        strategy=strategy,
    )
    with tempfile.TemporaryDirectory() as filesystem_store_dir:
        # each actor mesh represents a group of processes.
        # e.g., two different islands running spmd
        put_visible_devices = ",".join(
            str(d) for d in range(put_world_size)
        )  # e.g. put_world_size=4, put_visible_devices="0,1,2,3"
        put_mesh = await spawn_actors(
            put_world_size,
            DTensorActor,
            "put_mesh",
            original_tensor=original_tensor,
            placements=put_placements,
            mesh_shape=put_mesh_shape,
            file_store_name=os.path.join(filesystem_store_dir, "put_test"),
            visible_devices=put_visible_devices,
        )
        # This call places the local tensor from each rank into TorchStore
        await put_mesh.do_put.call()

        get_visible_devices = ",".join(
            str(d) for d in range(put_world_size, put_world_size + get_world_size)
        )  # e.g. put_world_size=4, get_world_size=2, get_visible_devices="4,5"
        get_mesh = await spawn_actors(
            get_world_size,
            DTensorActor,
            "get_mesh",
            original_tensor=torch.zeros(
                8, 8, dtype=original_tensor.dtype
            ),  # these values get replaced with values from original_tensor after fetching
            placements=get_placements,
            mesh_shape=get_mesh_shape,
            file_store_name=os.path.join(filesystem_store_dir, "get_test"),
            visible_devices=get_visible_devices,
        )
        # This call fetches the tensor from TorchStore into the local DTensor shards
        value_mesh = await get_mesh.do_get.call()

        # assert the correct value is found here
        for _, val in value_mesh:
            sharded_tensor, coord = val
            _assert_correct_sharded_tensor(
                original_tensor, sharded_tensor, get_placements, coord
            )

        # teardown distributed or the next test will complain
        await put_mesh.destroy_process_group.call()
        await put_mesh._proc_mesh.stop()
        await get_mesh.destroy_process_group.call()
        await get_mesh._proc_mesh.stop()
        await ts.shutdown()


def _assert_correct_sharded_tensor(
    full_tensor, sharded_tensor, get_placements, coordinate
):
    local_shape, global_offsets = _compute_local_shape_and_global_offset(
        sharded_tensor.shape,
        mesh_shape=sharded_tensor.device_mesh.shape,
        my_coordinate=coordinate,
        placements=get_placements,
    )
    expected_local_tensor = get_local_tensor(full_tensor, local_shape, global_offsets)

    assert torch.equal(
        expected_local_tensor, sharded_tensor._local_tensor.cpu()
    ), f"{expected_local_tensor=} {sharded_tensor._local_tensor.cpu()=}"


if __name__ == "__main__":
    main(__file__)
