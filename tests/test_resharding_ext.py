# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from logging import getLogger

import pytest

from torch.distributed._tensor import Shard

from .test_resharding_basic import _test_resharding

from .utils import main, transport_plus_strategy_params

logger = getLogger(__name__)


@pytest.mark.parametrize(*transport_plus_strategy_params())
@pytest.mark.parametrize(
    "put_mesh_shape,get_mesh_shape,put_sharding_dim,get_sharding_dim",
    [
        # shrink
        ((4,), (2,), 0, 0),
        ((4,), (2,), 0, 1),
        ((4,), (2,), 1, 0),
        ((4,), (2,), 1, 1),
        # grow
        ((2,), (4,), 0, 0),
        ((2,), (4,), 0, 1),
        ((2,), (4,), 1, 0),
        ((2,), (4,), 1, 1),
    ],
)
@pytest.mark.asyncio
async def test_1d_resharding(
    strategy_params,
    use_rdma,
    put_mesh_shape,
    get_mesh_shape,
    put_sharding_dim,
    get_sharding_dim,
):
    _, strategy = strategy_params

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


if __name__ == "__main__":
    main(__file__)
