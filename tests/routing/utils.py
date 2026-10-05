# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Mapping
from typing import Any

import torchstore as ts
from monarch.actor import get_or_spawn_controller, this_host
from torchstore.routing._model import RankRole
from torchstore.routing.client import RoutingClient
from torchstore.routing.coordinator import RoutingCoordinator
from torchstore.transport import TransportType
from torchstore.transport.types import TensorSlice


def tensor_slice(
    offsets: tuple[int, ...],
    local_shape: tuple[int, ...],
    *,
    global_shape: tuple[int, ...] = (8, 6),
    coordinates: tuple[int, ...] = (),
    mesh_shape: tuple[int, ...] = (),
) -> TensorSlice:
    """Build tensor-slice metadata with empty placement by default."""
    return TensorSlice(
        offsets=offsets,
        coordinates=coordinates,
        global_shape=global_shape,
        local_shape=local_shape,
        mesh_shape=mesh_shape,
    )


async def routing_clients(
    publishers: Mapping[str, Mapping[str, Any]],
    requesters: Mapping[str, Mapping[str, Any]],
) -> dict[str, RoutingClient]:
    """Spawn routing infrastructure without registering state-dict layouts."""
    publisher_mesh = this_host().spawn_procs(per_host={"procs": len(publishers)})
    store_name = f"routing-test-{uuid.uuid4()}"
    strategy = ts.LocalRankStrategy(TransportType.MonarchRPC)
    await ts.initialize(
        mesh=publisher_mesh,
        strategy=strategy,
        client_type=ts.ClientType.ROUTING,
        store_name=store_name,
    )
    coordinator = await get_or_spawn_controller(store_name, RoutingCoordinator)
    roles = {
        **{rank: RankRole.PUBLISHER for rank in publishers},
        **{rank: RankRole.REQUESTER for rank in requesters},
    }
    clients = {
        rank: RoutingClient(rank, role, coordinator, strategy)
        for rank, role in roles.items()
    }
    return clients


async def registered_clients(
    publishers: Mapping[str, Mapping[str, Any]],
    requesters: Mapping[str, Mapping[str, Any]],
    namespace: str = "model",
) -> dict[str, RoutingClient]:
    """Spawn clients and explicitly register layouts for low-level API tests."""
    clients = await routing_clients(publishers, requesters)
    await asyncio.gather(
        *(
            clients[rank].register_layout(state_dict, namespace)
            for rank, state_dict in publishers.items()
        ),
        *(
            clients[rank].get_layouts(state_dict, namespace)
            for rank, state_dict in requesters.items()
        ),
    )
    return clients
