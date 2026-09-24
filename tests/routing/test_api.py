# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import torchstore.api as store_api
from torchstore.routing._model import RankRole
from torchstore.routing.client import RoutingClient
from torchstore.routing.coordinator import RoutingCoordinator
from torchstore.storage_volume import StorageVolume


def test_routing_namespaces_distinguish_participant_meshes(monkeypatch) -> None:
    """Qualify equal local ranks by role and requester-mesh index."""
    # Use one local rank to verify how each participant namespace is composed.
    monkeypatch.setattr(store_api, "current_rank", lambda: SimpleNamespace(rank=3))

    assert (
        store_api._routing_volume_id(
            store_api._routing_namespace(RankRole.PUBLISHER, None)
        )
        == "publisher/3"
    )
    assert (
        store_api._routing_volume_id(
            store_api._routing_namespace(RankRole.REQUESTER, 1)
        )
        == "requester/1/3"
    )

    # Reject role/group combinations that cannot identify a participant mesh.
    with pytest.raises(ValueError, match="publishers .* take no group"):
        store_api._routing_namespace(RankRole.PUBLISHER, 0)
    with pytest.raises(ValueError, match="requesters must pass"):
        store_api._routing_namespace(RankRole.REQUESTER, None)


@pytest.mark.parametrize(
    ("mesh", "relay_meshes", "strategy", "message"),
    [
        (None, [object()], object(), "both mesh and strategy"),
        (object(), [object()], None, "both mesh and strategy"),
        (object(), [], object(), "at least one relay mesh"),
    ],
)
def test_routing_initialization_validates_required_arguments(
    mesh, relay_meshes, strategy, message: str
) -> None:
    """Reject incomplete routing-mode initialization before spawning actors."""
    with pytest.raises(RuntimeError, match=message):
        asyncio.run(
            store_api.initialize(
                mesh=mesh,
                relay_meshes=relay_meshes,
                strategy=strategy,
                store_name="routing-test",
            )
        )


def test_routing_initialization_spawns_publisher_volumes_and_coordinator(
    monkeypatch,
) -> None:
    """Initialize publisher storage and register every participating rank."""

    # 1. Replace actor creation with async mocks so initialization stays local.
    class Mesh:
        def __init__(self, name: str, size: int) -> None:
            self.name = name
            self._size = size

        def size(self) -> int:
            return self._size

    strategy = SimpleNamespace(set_storage_volumes=AsyncMock())
    coordinator = SimpleNamespace(init=SimpleNamespace(call_one=AsyncMock()))
    volumes = object()
    spawn_volumes = AsyncMock(return_value=volumes)
    spawn_coordinator = AsyncMock(return_value=coordinator)

    monkeypatch.setattr(StorageVolume, "spawn", spawn_volumes)
    monkeypatch.setattr(store_api, "get_or_spawn_controller", spawn_coordinator)
    monkeypatch.setattr(store_api, "current_rank", lambda: SimpleNamespace(rank=0))
    publisher = Mesh("publisher-mesh", 2)
    requester_0 = Mesh("requester-mesh-0", 1)
    requester_1 = Mesh("requester-mesh-1", 2)

    # 2. Initialize one publisher mesh and two requester meshes.
    asyncio.run(
        store_api.initialize(
            mesh=publisher,
            relay_meshes=[requester_0, requester_1],
            strategy=strategy,
            store_name="routing-test",
        )
    )

    # 3. Verify the publisher volume and the complete participant roster.
    spawn_volumes.assert_awaited_once()
    args, kwargs = spawn_volumes.await_args
    assert args == (1, publisher)
    assert kwargs["id_func"]() == "publisher/0"
    strategy.set_storage_volumes.assert_awaited_once_with(volumes)
    spawn_coordinator.assert_awaited_once_with("routing-test", RoutingCoordinator)
    coordinator.init.call_one.assert_awaited_once_with(
        ranks={
            "publisher/0",
            "publisher/1",
            "requester/0/0",
            "requester/1/0",
            "requester/1/1",
        },
        strategy=strategy,
    )


def test_client_constructs_and_caches_a_routing_client(monkeypatch) -> None:
    """Create one rank-local routing client and reuse it for later API calls."""
    # 1. Supply a local coordinator handle and a known calling rank.
    strategy = object()
    coordinator = SimpleNamespace(
        strategy=SimpleNamespace(call_one=AsyncMock(return_value=strategy))
    )
    spawn_coordinator = AsyncMock(return_value=coordinator)

    monkeypatch.setattr(store_api, "get_or_spawn_controller", spawn_coordinator)
    monkeypatch.setattr(store_api, "current_rank", lambda: SimpleNamespace(rank=2))

    # 2. Request a routed client, then access the same store without role metadata.
    async def run():
        first = await store_api.client("routing-test", role=RankRole.REQUESTER, group=1)
        second = await store_api.client("routing-test")
        return first, second

    try:
        first, second = asyncio.run(run())
    finally:
        store_api.reset_client("routing-test")

    # 3. Verify the participant identity and cache reuse.
    assert isinstance(first, RoutingClient)
    assert first._controller.rank == "requester/1/2"
    assert first._role == RankRole.REQUESTER
    assert first.strategy is strategy
    assert second is first
    spawn_coordinator.assert_awaited_once_with("routing-test", RoutingCoordinator)
    coordinator.strategy.call_one.assert_awaited_once_with()
