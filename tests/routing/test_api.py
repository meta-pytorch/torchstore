# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import logging
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock, call, MagicMock

import pytest
import torchstore as ts
import torchstore.api as store_api
from torchstore import RankRole
from torchstore.client import ClientType, LocalClient
from torchstore.monarch.coordinator import MonarchControlPlane
from torchstore.routing.client import RoutingClient

from ..backend_stub import install_backend_stub


@pytest.fixture
def backend(monkeypatch) -> MagicMock:
    """A stand-in for the Monarch backend whose actors are mocks, at rank 0."""
    return install_backend_stub(monkeypatch)


def test_routing_namespaces_distinguish_participant_meshes(backend) -> None:
    """Qualify equal local ranks by role and requester-mesh index."""
    # Use one local rank to verify how each participant namespace is composed.
    backend.rank.return_value = 3

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


def test_local_rank_strategy_names_volumes_by_the_backend_rank(backend) -> None:
    """Name each ``LocalRankStrategy`` volume by the rank the backend reports."""
    backend.rank.return_value = 4
    assert ts.LocalRankStrategy.get_volume_id() == "4"


@pytest.mark.parametrize(
    ("mesh", "strategy"),
    [
        (None, object()),
        (object(), None),
    ],
)
def test_routing_initialization_validates_required_arguments(mesh, strategy) -> None:
    """Reject incomplete routing-mode initialization before spawning actors."""
    with pytest.raises(RuntimeError, match="both mesh and strategy"):
        asyncio.run(
            store_api.initialize(
                mesh=mesh,
                strategy=strategy,
                client_type=ClientType.ROUTING,
                store_name="routing-test",
            )
        )


def test_routing_initialization_spawns_publisher_volumes_and_coordinator(
    backend, caplog
) -> None:
    """Initialize publisher storage and register every rank through the Monarch adapter."""

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
    backend.spawn_volumes.return_value = volumes
    backend.coordinator.return_value = MonarchControlPlane(coordinator)
    publisher = Mesh("publisher-mesh", 2)
    caplog.set_level(logging.INFO, logger="torchstore.routing.coordinator")
    # 2. Initialize the publisher mesh.
    asyncio.run(
        store_api.initialize(
            mesh=publisher,
            strategy=strategy,
            client_type=ClientType.ROUTING,
            store_name="routing-test",
        )
    )

    # 3. Verify the publisher volume and publisher roster.
    backend.spawn_volumes.assert_awaited_once()
    mesh, num_volumes, id_func = backend.spawn_volumes.await_args.args
    assert (mesh, num_volumes) == (publisher, 1)
    assert id_func() == "publisher/0"
    strategy.set_storage_volumes.assert_awaited_once_with(volumes)
    backend.coordinator.assert_awaited_once_with("routing-test")
    coordinator.init.call_one.assert_awaited_once_with(
        publishers={
            "publisher/0",
            "publisher/1",
        },
        strategy=strategy,
    )
    # The adapter logs each call it forwards.
    assert [
        match.group(1)
        for record in caplog.records
        if record.name == "torchstore.routing.coordinator"
        and (
            match := re.search(r"control_plane=monarch (.+) took ", record.getMessage())
        )
    ] == ["init"]


def test_client_constructs_and_caches_a_routing_client(backend, caplog) -> None:
    """Create one rank-local routing client and reuse it for later API calls."""
    caplog.set_level(logging.INFO, logger="torchstore.routing.coordinator")
    # 1. Supply a local coordinator handle and a known calling rank.
    strategy = object()
    coordinator = SimpleNamespace(
        strategy=SimpleNamespace(call_one=AsyncMock(return_value=strategy))
    )
    backend.coordinator.return_value = MonarchControlPlane(coordinator)
    backend.rank.return_value = 2

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
    assert first._routing_directory.rank == "requester/1/2"
    assert first._role == RankRole.REQUESTER
    assert first._control_plane is backend.coordinator.return_value
    assert first.strategy is strategy
    assert second is first
    backend.coordinator.assert_awaited_once_with("routing-test")
    coordinator.strategy.call_one.assert_awaited_once_with()
    # The strategy came through the adapter, which logs each call it forwards.
    assert [
        record.getMessage().split(" took ")[0]
        for record in caplog.records
        if record.name == "torchstore.routing.coordinator"
    ] == ["routing control_plane=monarch strategy"]


def _controller(strategy: object = None) -> SimpleNamespace:
    """A controller handle whose endpoints the standard path awaits."""
    return SimpleNamespace(
        init=SimpleNamespace(call=AsyncMock()),
        teardown=SimpleNamespace(call=AsyncMock()),
        get_controller_strategy=SimpleNamespace(
            call_one=AsyncMock(return_value=strategy)
        ),
    )


def test_standard_initialize_spawns_volumes_on_the_callers_mesh(backend) -> None:
    """Spawn the volumes on ``mesh`` through the backend and hand them to the store's controller."""
    controller = backend.controller.return_value = _controller()
    strategy = ts.LocalRankStrategy()
    mesh = object()

    asyncio.run(
        ts.initialize(
            num_storage_volumes=2, strategy=strategy, store_name="standard", mesh=mesh
        )
    )

    backend.spawn_volumes.assert_awaited_once_with(mesh, 2, strategy.get_volume_id)
    backend.controller_volume.assert_not_awaited()
    backend.controller.assert_awaited_once_with("standard")
    controller.init.call.assert_awaited_once_with(
        strategy=strategy,
        num_storage_volumes=2,
        storage_volumes=backend.spawn_volumes.return_value,
    )


def test_standard_initialize_reuses_the_controller_volume(backend) -> None:
    """Find the one controller volume through the backend instead of spawning volumes."""
    controller = backend.controller.return_value = _controller()
    strategy = ts.ControllerStorageVolumes()

    asyncio.run(ts.initialize(strategy=strategy, store_name="singleton"))

    backend.controller_volume.assert_awaited_once_with(strategy.get_volume_id)
    backend.spawn_volumes.assert_not_awaited()
    assert (
        controller.init.call.await_args.kwargs["storage_volumes"]
        is backend.controller_volume.return_value
    )


def test_standard_client_and_shutdown_find_the_stores_own_controller(backend) -> None:
    """Look each store's controller up by its own name for ``client`` and ``shutdown``."""
    strategy = object()
    controller = backend.controller.return_value = _controller(strategy)

    async def run() -> LocalClient:
        try:
            return await ts.client("named-store")
        finally:
            await ts.shutdown("named-store")

    client = asyncio.run(run())

    assert backend.controller.await_args_list == [call("named-store")] * 2
    assert client.strategy is strategy
    backend.directory.assert_called_once_with(controller)
    controller.teardown.call.assert_awaited_once_with()
