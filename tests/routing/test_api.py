# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import inspect
import logging
import os
import pickle
import re
import subprocess
import sys
import threading
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, call, MagicMock, Mock, patch

import pytest
import torch
import torchstore as ts
import torchstore.api as store_api
import torchstore.monarch.backend as monarch_backend
from monarch.actor import ActorError
from torchstore import RankRole
from torchstore.client import ClientType, LocalClient
from torchstore.monarch.coordinator import MonarchControlPlane, RoutingCoordinator
from torchstore.monarch.storage_volume import StorageVolume
from torchstore.routing.client import RoutingClient
from torchstore.routing.control_plane import ControlPlane

from ..backend_stub import install_backend_stub
from .utils import HostedRegistry, registry_store

# Each child process imports the tree that pytest imported.
_ROOT = str(Path(importlib.util.find_spec("torchstore").origin).parents[1])
_SELECTION_LINE = re.compile(
    r"routing control_plane=(\S+) address=(\S+) store=(\S+) rank=(\S+) pid=(\d+)"
)
# Registry hosts that test addresses dial, by host id.
_HOSTS: dict[str, Callable[..., Awaitable[HostedRegistry]]] = {}


@pytest.fixture(autouse=True)
def _fresh_process_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test with no cached client and no recorded or released store."""
    monkeypatch.setattr(store_api, "_local_clent_map", {})
    monkeypatch.setattr(store_api, "_control_plane_map", {})
    monkeypatch.setattr(store_api, "_released_stores", set())
    monkeypatch.setattr(store_api, "_forked_stores", {})


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


@dataclass(frozen=True)
class _RegistryAddress(ts.ControlPlaneAddress):
    """Test-only: dials a ``LayoutRegistry`` hosted on a loop thread in this process."""

    host: str

    async def connect(self) -> ControlPlane:
        return await _HOSTS[self.host]()

    def __str__(self) -> str:
        return f"registry:{self.host}"


@contextlib.asynccontextmanager
async def _hosted_registry() -> AsyncIterator[str]:
    """Host one registry for ``_RegistryAddress`` to dial, and yield its host id."""
    async with registry_store() as connect:
        host = uuid.uuid4().hex
        _HOSTS[host] = connect
        try:
            yield host
        finally:
            _HOSTS.pop(host, None)


class _Mesh:
    def __init__(self, size: int) -> None:
        self._size = size

    def size(self) -> int:
        return self._size


def _store_name() -> str:
    return f"routing-test-{uuid.uuid4()}"


def _strategy() -> SimpleNamespace:
    # get_volume_id lets a standard initialize get as far as spawning, so its checks show.
    return SimpleNamespace(set_storage_volumes=AsyncMock(), get_volume_id=lambda: "0")


async def _initialize(
    store_name: str,
    control_plane: str | ControlPlane = "monarch",
    *,
    publishers: int = 1,
    strategy: Any = None,
) -> None:
    """Initialize a routing store on a stand-in mesh."""
    return await ts.initialize(
        mesh=_Mesh(publishers),
        strategy=strategy or _strategy(),
        client_type=ClientType.ROUTING,
        store_name=store_name,
        control_plane=control_plane,
    )


def _mock_coordinator() -> SimpleNamespace:
    return SimpleNamespace(
        init=SimpleNamespace(call_one=AsyncMock()),
        strategy=SimpleNamespace(call_one=AsyncMock(return_value=object())),
    )


def _mock_controller() -> SimpleNamespace:
    return SimpleNamespace(
        init=SimpleNamespace(call=AsyncMock()),
        get_controller_strategy=SimpleNamespace(
            call_one=AsyncMock(return_value=object())
        ),
        teardown=SimpleNamespace(call=AsyncMock()),
    )


def _patch_actors(monkeypatch: pytest.MonkeyPatch, rank: int = 0) -> SimpleNamespace:
    """Stub volume spawning and the rank, and fail any Monarch actor lookup."""
    actors = SimpleNamespace(
        spawn=AsyncMock(return_value=object()),
        lookup=AsyncMock(side_effect=AssertionError("looked up a Monarch actor")),
    )
    monkeypatch.setattr(StorageVolume, "spawn", actors.spawn)
    monkeypatch.setattr(monarch_backend, "get_or_spawn_controller", actors.lookup)
    monkeypatch.setattr(
        monarch_backend, "current_rank", lambda: SimpleNamespace(rank=rank)
    )
    return actors


def _selection_lines(caplog: pytest.LogCaptureFixture) -> list[tuple[str, ...]]:
    # Search: once Monarch's runtime starts, its log filter prefixes the actor's name.
    return [
        match.groups()
        for record in caplog.records
        if record.levelno == logging.INFO
        and (match := _SELECTION_LINE.search(record.getMessage()))
    ]


class _EqualRegistry(HostedRegistry):
    """Equal to other such handles on its registry, as a dataclass control plane is."""

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _EqualRegistry) and other._registry is self._registry


class _NamedRegistry(HostedRegistry):
    """A registry handle that names itself in log lines."""

    kind = "byo"


class _RecordedWaits:
    """``asyncio`` for the API module: records each close wait, then cuts it to 0.2s."""

    def __init__(self) -> None:
        self.timeouts: list[float] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(asyncio, name)

    async def wait(self, tasks: Any, *, timeout: float) -> Any:
        self.timeouts.append(timeout)
        return await asyncio.wait(tasks, timeout=min(timeout, 0.2))


@pytest.mark.parametrize("client_type", [ClientType.STANDARD, ClientType.ROUTING])
@pytest.mark.parametrize(
    ("control_plane", "error"),
    [("tcpstore", ValueError), ("Monarch", ValueError), (None, TypeError)],
)
def test_initialize_rejects_unknown_control_planes(
    monkeypatch, client_type, control_plane, error
) -> None:
    """Name the accepted control planes before spawning anything."""
    actors = _patch_actors(monkeypatch)
    parameter = inspect.signature(ts.initialize).parameters["control_plane"]
    assert (parameter.kind, parameter.default) == (parameter.KEYWORD_ONLY, "monarch")

    message = (
        "control_plane must be 'monarch' or a ControlPlane instance, "
        f"got {control_plane!r}"
    )
    with pytest.raises(error, match=re.escape(message)):
        asyncio.run(
            ts.initialize(
                mesh=_Mesh(2),
                strategy=_strategy(),
                client_type=client_type,
                store_name=_store_name(),
                control_plane=control_plane,
            )
        )
    actors.spawn.assert_not_called()
    actors.lookup.assert_not_called()


def test_standard_initialize_takes_only_monarch(monkeypatch) -> None:
    """Reject a caller's control plane for a standard store before spawning anything."""
    actors = _patch_actors(monkeypatch)
    control_plane = Mock(spec=ControlPlane)

    with pytest.raises(ValueError, match="standard stores take only"):
        asyncio.run(
            ts.initialize(
                strategy=_strategy(),
                store_name=_store_name(),
                control_plane=control_plane,
            )
        )
    actors.spawn.assert_not_called()
    actors.lookup.assert_not_called()
    assert control_plane.mock_calls == []


@pytest.mark.timeout(60)
def test_initialize_with_a_control_plane_skips_the_coordinator(monkeypatch) -> None:
    """Initialize the caller's control plane once, after the volumes, and record it."""
    actors = _patch_actors(monkeypatch, rank=1)
    store_name, strategy, order = _store_name(), _strategy(), []
    actors.spawn.side_effect = lambda *args, **kwargs: order.append("spawn")
    strategy.set_storage_volumes.side_effect = lambda volumes: order.append("volumes")

    async def run() -> None:
        async with registry_store() as connect:
            control_plane = await connect()
            init = control_plane.init

            async def init_in_order(**kwargs: Any) -> None:
                order.append("init")
                await init(**kwargs)

            control_plane.init = AsyncMock(side_effect=init_in_order)
            assert (
                await _initialize(
                    store_name, control_plane, publishers=2, strategy=strategy
                )
                is None
            )
            assert order == ["spawn", "volumes", "init"]
            control_plane.init.assert_awaited_once_with(
                publishers={"publisher/0", "publisher/1"}, strategy=strategy
            )
            missing = r"missing publishers: \['publisher/0', 'publisher/1'\]"
            with pytest.raises(TimeoutError, match=missing):
                await control_plane.get_layouts("model", timeout_s=0)

            # A later client in this process uses the recorded control plane.
            client = await ts.client(store_name, role=RankRole.PUBLISHER)
            assert isinstance(client, RoutingClient)
            assert client._control_plane is control_plane
            assert client.strategy is strategy
            assert client._routing_directory.rank == "publisher/1"

    asyncio.run(run())
    actors.lookup.assert_not_called()


@pytest.mark.timeout(60)
def test_client_uses_a_given_control_plane(monkeypatch) -> None:
    """Build a client on a given control plane, and reuse it after a client reset."""
    actors = _patch_actors(monkeypatch, rank=2)
    store_name, strategy = _store_name(), _strategy()

    async def run() -> None:
        async with registry_store() as connect:
            control_plane = await connect()
            await control_plane.init({"publisher/0"}, strategy)
            client = await ts.client(
                store_name,
                role=RankRole.REQUESTER,
                group=1,
                control_plane=control_plane,
            )
            assert client._control_plane is control_plane
            assert client.strategy is strategy
            assert client._routing_directory.rank == "requester/1/2"
            assert await ts.client(store_name) is client
            assert (
                await ts.client(
                    store_name,
                    role=RankRole.REQUESTER,
                    group=1,
                    control_plane=control_plane,
                )
                is client
            )
            ts.reset_client(store_name)
            again = await ts.client(store_name, role=RankRole.REQUESTER, group=1)
            assert again is not client
            assert again._control_plane is control_plane

    asyncio.run(run())
    actors.lookup.assert_not_called()


@pytest.mark.timeout(60)
def test_client_takes_an_instance_then_an_address_then_the_record(
    monkeypatch,
) -> None:
    """Pick a given control plane over an address, and an address over the record."""
    actors = _patch_actors(monkeypatch)
    store_name = _store_name()
    connect = patch.object(
        _RegistryAddress,
        "connect",
        autospec=True,
        side_effect=_RegistryAddress.connect,
    )

    async def run() -> None:
        async with _hosted_registry() as host:
            address = _RegistryAddress(store_name, host)
            recorded, given = await _HOSTS[host](), await _HOSTS[host]()
            await _initialize(store_name, recorded)
            records = store_api._control_plane_map
            with connect as connects:
                from_record = await ts.client(store_name, role=RankRole.PUBLISHER)
                assert records[store_name] == (recorded, None)
                ts.reset_client(store_name)
                # Nothing has dialed the address yet, so only precedence stops a dial.
                from_instance = await ts.client(
                    store_name,
                    role=RankRole.PUBLISHER,
                    address=address,
                    control_plane=given,
                )
                # The instance's record names no address, so shutdown never closes it.
                assert records[store_name] == (given, None)
                # Repeating the call gets that client back, as the instance wins again.
                assert (
                    await ts.client(
                        store_name,
                        role=RankRole.PUBLISHER,
                        address=address,
                        control_plane=given,
                    )
                    is from_instance
                )
                assert connects.call_count == 0
                ts.reset_client(store_name)
                from_address = await ts.client(
                    store_name, role=RankRole.PUBLISHER, address=address
                )
                assert records[store_name] == (from_address._control_plane, address)
            assert from_record._control_plane is recorded
            assert from_instance._control_plane is given
            assert isinstance(from_address._control_plane, HostedRegistry)
            assert from_address._control_plane not in (recorded, given)
            assert connects.call_count == 1

    asyncio.run(run())
    actors.lookup.assert_not_called()


@pytest.mark.timeout(60)
def test_client_connects_through_an_address(monkeypatch) -> None:
    """Publish through the registry that a pickled address reaches."""
    actors = _patch_actors(monkeypatch)
    store_name, strategy = _store_name(), _strategy()

    async def run() -> None:
        async with _hosted_registry() as host:
            driver = await _HOSTS[host]()
            await driver.init({"publisher/0"}, strategy)
            address = pickle.loads(pickle.dumps(_RegistryAddress(store_name, host)))

            client = await ts.client(
                store_name, role=RankRole.PUBLISHER, address=address
            )
            assert isinstance(client._control_plane, HostedRegistry)
            assert client.strategy is strategy
            assert client._routing_directory.rank == "publisher/0"
            await client.register_layout({"w": torch.ones(2, 3)}, "model")
            layouts = await driver.get_layouts("model", timeout_s=0)
            [registration] = layouts["publisher/0"].values()
            assert registration.tensor_slice.global_shape == (2, 3)
            assert await ts.client(store_name) is client

    asyncio.run(run())
    actors.lookup.assert_not_called()


@pytest.mark.timeout(60)
def test_client_checks_the_address_store_and_role(monkeypatch) -> None:
    """Need role with an address or control plane, and an address for this store."""
    actors = _patch_actors(monkeypatch)
    store_name = _store_name()
    # No such host: dialing it would raise KeyError, so these fail before connecting.
    address = _RegistryAddress(store_name, "unreachable")
    control_plane = Mock(spec=ControlPlane)

    async def run() -> None:
        other = _RegistryAddress("other-store", "unreachable")
        with pytest.raises(
            ValueError,
            match=f"address is for store 'other-store', not '{store_name}'",
        ):
            await ts.client(store_name, role=RankRole.PUBLISHER, address=other)
        for kwargs in ({"address": address}, {"control_plane": control_plane}):
            with pytest.raises(ValueError, match="for routing clients: pass role"):
                await ts.client(store_name, **kwargs)

    asyncio.run(run())
    assert store_api._local_clent_map == store_api._control_plane_map == {}
    assert control_plane.mock_calls == []
    actors.lookup.assert_not_called()


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    ("role", "group", "message"),
    [
        (RankRole.PUBLISHER, 0, "publishers are a single mesh and take no group"),
        (RankRole.REQUESTER, None, "requesters must pass the index"),
    ],
    ids=["publisher-with-group", "requester-without-group"],
)
def test_client_checks_role_and_group_before_connecting(
    monkeypatch, role, group, message
) -> None:
    """Refuse a bad role and group before dialing, calling an instance or looking up Monarch."""
    actors = _patch_actors(monkeypatch)
    store_name = _store_name()
    control_plane = Mock(spec=ControlPlane)
    connect = AsyncMock(return_value=control_plane)

    async def run() -> None:
        with patch.object(_RegistryAddress, "connect", connect):
            for kwargs in (
                {"address": _RegistryAddress(store_name, "unreachable")},
                {"control_plane": control_plane},
                {},
            ):
                with pytest.raises(ValueError, match=message):
                    await ts.client(store_name, role=role, group=group, **kwargs)

    asyncio.run(run())
    connect.assert_not_awaited()
    assert control_plane.mock_calls == []
    assert store_api._local_clent_map == store_api._control_plane_map == {}
    actors.lookup.assert_not_called()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"control_plane": "monarch"},
            "control_plane must be a ControlPlane instance, got 'monarch'",
        ),
        ({"address": "host:1"}, "address must be a ControlPlaneAddress, got 'host:1'"),
    ],
    ids=["control_plane", "address"],
)
def test_client_rejects_a_control_plane_or_address_of_another_type(
    monkeypatch, kwargs, message
) -> None:
    """Refuse strings, such as initialize's ``"monarch"``, before any lookup."""
    actors = _patch_actors(monkeypatch)
    with pytest.raises(TypeError, match=re.escape(message)):
        asyncio.run(ts.client(_store_name(), role=RankRole.PUBLISHER, **kwargs))
    assert store_api._local_clent_map == store_api._control_plane_map == {}
    actors.lookup.assert_not_called()


@pytest.mark.timeout(60)
def test_first_client_of_a_recorded_store_needs_a_role(monkeypatch) -> None:
    """Refuse a standard client for a store with a recorded control plane."""
    actors = _patch_actors(monkeypatch)
    store_name = _store_name()

    async def run() -> None:
        async with registry_store() as connect:
            await _initialize(store_name, await connect())
            with pytest.raises(ValueError, match="needs role on its first ts.client"):
                await ts.client(store_name)

    asyncio.run(run())
    assert store_api._local_clent_map == {}
    actors.lookup.assert_not_called()


@pytest.mark.timeout(60)
def test_one_stores_record_leaves_other_stores_clients_alone(monkeypatch) -> None:
    """Build Monarch and standard clients for other stores while one has a record."""
    actors = _patch_actors(monkeypatch)
    coordinator, controller = _mock_coordinator(), _mock_controller()
    routed, monarch, standard = _store_name(), _store_name(), _store_name()
    actors.lookup.side_effect = lambda name, *args, **kwargs: (
        coordinator if name == monarch else controller
    )

    async def run() -> None:
        async with registry_store() as connect:
            await _initialize(routed, await connect())
            client = await ts.client(monarch, role=RankRole.PUBLISHER)
            assert isinstance(client._control_plane, MonarchControlPlane)
            assert not isinstance(await ts.client(standard), RoutingClient)
            assert set(store_api._control_plane_map) == {routed}

    asyncio.run(run())


@pytest.mark.timeout(60)
def test_a_control_plane_serves_one_store(monkeypatch) -> None:
    """Refuse another store's instance in a later call, before spawning or calling it.

    Once that store shuts down, the instance may serve another store.
    """
    actors = _patch_actors(monkeypatch)
    weights, optimizer = _store_name(), _store_name()

    def refusal(store_name: str) -> str:
        return re.escape(f"this control plane already serves store '{store_name}'")

    async def run() -> None:
        async with registry_store() as connect:
            control_plane = await connect()
            await _initialize(weights, control_plane)
            control_plane.init = AsyncMock()
            control_plane.strategy = AsyncMock()
            with pytest.raises(ValueError, match=refusal(weights)):
                await _initialize(optimizer, control_plane)
            with pytest.raises(ValueError, match=refusal(weights)):
                await ts.client(
                    optimizer, role=RankRole.PUBLISHER, control_plane=control_plane
                )
            control_plane.init.assert_not_called()
            control_plane.strategy.assert_not_called()
            actors.spawn.assert_awaited_once()
            # Once that store shuts down, another store may take the instance.
            await ts.shutdown(weights)
            await _initialize(optimizer, control_plane)
            control_plane.init.assert_awaited_once()
            assert (
                store_api._control_plane_map[optimizer].control_plane is control_plane
            )

    asyncio.run(run())
    actors.lookup.assert_not_called()


@pytest.mark.timeout(60)
def test_cached_client_refuses_another_address_or_control_plane(monkeypatch) -> None:
    """Return the cached client only for its own address or control plane."""
    _patch_actors(monkeypatch)
    store_name = _store_name()

    async def run() -> None:
        async with _hosted_registry() as host, _hosted_registry() as other_host:
            driver = await _HOSTS[host]()
            await driver.init({"publisher/0"}, _strategy())
            address = _RegistryAddress(store_name, host)
            client = await ts.client(
                store_name, role=RankRole.PUBLISHER, address=address
            )
            # Actor endpoints deliver an equal copy of the address, not the object.
            copy = pickle.loads(pickle.dumps(address))
            assert (
                await ts.client(store_name, role=RankRole.PUBLISHER, address=copy)
                is client
            )
            other = _RegistryAddress(store_name, other_host)
            with pytest.raises(
                ValueError, match=f"did not connect through registry:{other_host}"
            ):
                await ts.client(store_name, role=RankRole.PUBLISHER, address=other)
            with pytest.raises(ValueError, match="uses another control plane"):
                await ts.client(
                    store_name, role=RankRole.PUBLISHER, control_plane=driver
                )
            assert await ts.client(store_name) is client
            # After a reset, the other address wins over the recorded connection.
            ts.reset_client(store_name)
            await (await _HOSTS[other_host]()).init({"publisher/0"}, _strategy())
            moved = await ts.client(store_name, role=RankRole.PUBLISHER, address=other)
            assert moved._control_plane._registry is not client._control_plane._registry
            assert store_api._control_plane_map[store_name].address == other

    asyncio.run(run())


@pytest.mark.timeout(60)
def test_a_selection_that_replaces_a_dialed_connection_closes_it(monkeypatch) -> None:
    """Close a dialed connection once another selection replaces its record.

    Another address after ``reset_client`` and a caller's instance each close the
    connection dialed before. A caller's instance stays open.
    """
    _patch_actors(monkeypatch)
    store_name = _store_name()
    dialed: list[ControlPlane] = []

    async def connect(address: _RegistryAddress) -> ControlPlane:
        connection = await _HOSTS[address.host]()
        connection.close = AsyncMock()
        dialed.append(connection)
        return connection

    async def run() -> None:
        async with _hosted_registry() as first, _hosted_registry() as second:
            for host in (first, second):
                await (await _HOSTS[host]()).init({"publisher/0"}, _strategy())

            async def dial(host: str) -> ControlPlane:
                ts.reset_client(store_name)
                address = _RegistryAddress(store_name, host)
                with patch.object(_RegistryAddress, "connect", connect):
                    client = await ts.client(
                        store_name, role=RankRole.PUBLISHER, address=address
                    )
                return client._control_plane

            replaced = await dial(first)
            await dial(second)
            replaced.close.assert_awaited_once_with()
            instance = await _HOSTS[first]()
            instance.close = AsyncMock()
            ts.reset_client(store_name)
            await ts.client(store_name, role=RankRole.PUBLISHER, control_plane=instance)
            dialed[1].close.assert_awaited_once_with()
            await dial(first)
            instance.close.assert_not_called()
            assert [connection.close.await_count for connection in dialed] == [1, 1, 0]
            # Choosing the dialed connection itself as an instance keeps it open.
            ts.reset_client(store_name)
            await ts.client(
                store_name, role=RankRole.PUBLISHER, control_plane=dialed[2]
            )
            assert [connection.close.await_count for connection in dialed] == [1, 1, 0]

    asyncio.run(run())


@pytest.mark.timeout(60)
def test_cached_client_on_an_instance_refuses_an_equal_one_or_an_address(
    monkeypatch,
) -> None:
    """Refuse an equal but different instance, and any address, for this client."""
    actors = _patch_actors(monkeypatch)
    store_name = _store_name()

    async def run() -> None:
        async with _hosted_registry() as host:
            old, new = await _HOSTS[host](_EqualRegistry), await _HOSTS[host](
                _EqualRegistry
            )
            await _initialize(store_name, old)
            client = await ts.client(store_name, role=RankRole.PUBLISHER)
            # An actor call delivers a copy, so the error says to omit control_plane=.
            refusal = (
                "uses another control plane instance. Later calls in this process "
                "can omit control_plane="
            )
            with pytest.raises(ValueError, match=refusal):
                await ts.client(store_name, role=RankRole.PUBLISHER, control_plane=new)
            with pytest.raises(
                ValueError, match=f"did not connect through registry:{host}"
            ):
                await ts.client(
                    store_name,
                    role=RankRole.PUBLISHER,
                    address=_RegistryAddress(store_name, host),
                )
            assert (
                await ts.client(store_name, role=RankRole.PUBLISHER, control_plane=old)
                is client
            )

    asyncio.run(run())
    actors.lookup.assert_not_called()


def test_cached_monarch_client_refuses_an_address_or_control_plane(
    monkeypatch,
) -> None:
    """Refuse either once this process has a client on the Monarch coordinator."""
    actors = _patch_actors(monkeypatch)
    actors.lookup.side_effect = None
    actors.lookup.return_value = _mock_coordinator()
    store_name = _store_name()

    async def run() -> None:
        await ts.client(store_name, role=RankRole.PUBLISHER)
        with pytest.raises(ValueError, match="did not connect through registry:x"):
            await ts.client(
                store_name,
                role=RankRole.PUBLISHER,
                address=_RegistryAddress(store_name, "x"),
            )
        with pytest.raises(ValueError, match="uses another control plane"):
            await ts.client(
                store_name,
                role=RankRole.PUBLISHER,
                control_plane=Mock(spec=ControlPlane),
            )

    asyncio.run(run())


@pytest.mark.timeout(60)
def test_a_cached_client_still_checks_its_arguments(monkeypatch) -> None:
    """Check that address and control_plane come with a role, their types and the address's store.

    These run before a cached client is returned, which takes any role value or group.
    """
    _patch_actors(monkeypatch)
    store_name = _store_name()

    async def run() -> None:
        async with _hosted_registry() as host:
            await (await _HOSTS[host]()).init({"publisher/0"}, _strategy())
            address = _RegistryAddress(store_name, host)
            client = await ts.client(
                store_name, role=RankRole.PUBLISHER, address=address
            )
            for kwargs in (
                {"address": address},
                {"control_plane": client._control_plane},
            ):
                with pytest.raises(ValueError, match="for routing clients: pass role"):
                    await ts.client(store_name, **kwargs)
            for kwargs in ({"address": "host:1"}, {"control_plane": "monarch"}):
                with pytest.raises(TypeError, match="must be a ControlPlane"):
                    await ts.client(store_name, role=RankRole.PUBLISHER, **kwargs)
            other = _RegistryAddress("other-store", host)
            with pytest.raises(ValueError, match="address is for store 'other-store'"):
                await ts.client(store_name, role=RankRole.PUBLISHER, address=other)
            assert await ts.client(store_name) is client

    asyncio.run(run())


def test_monarch_lookup_passes_other_actor_errors_through(monkeypatch) -> None:
    """Leave any other coordinator error as it was."""
    error = ActorError(RuntimeError("something else"))
    message = error.message
    coordinator = SimpleNamespace(
        strategy=SimpleNamespace(call_one=AsyncMock(side_effect=error))
    )
    monkeypatch.setattr(
        monarch_backend, "get_or_spawn_controller", AsyncMock(return_value=coordinator)
    )
    monkeypatch.setattr(
        monarch_backend, "current_rank", lambda: SimpleNamespace(rank=0)
    )

    with pytest.raises(ActorError) as raised:
        asyncio.run(ts.client(_store_name(), role=RankRole.PUBLISHER))
    assert raised.value is error
    assert raised.value.message == message


@pytest.mark.timeout(60)
def test_shutdown_releases_one_store_and_closes_only_what_it_dialed(
    monkeypatch,
) -> None:
    """Drop one store's record and client, and close only a connection it dialed."""
    actors = _patch_actors(monkeypatch)
    initialized, connected = _store_name(), _store_name()
    dialed: list[ControlPlane] = []

    async def connect(address: _RegistryAddress) -> ControlPlane:
        connection = await _HOSTS[address.host]()
        connection.close = AsyncMock()
        dialed.append(connection)
        return connection

    async def run() -> None:
        async with _hosted_registry() as host:
            control_plane = await _HOSTS[host]()
            control_plane.close = AsyncMock()
            await _initialize(initialized, control_plane)
            await ts.client(initialized, role=RankRole.PUBLISHER)
            with patch.object(_RegistryAddress, "connect", connect):
                connected_client = await ts.client(
                    connected,
                    role=RankRole.PUBLISHER,
                    address=_RegistryAddress(connected, host),
                )
            await ts.shutdown(initialized)
            assert initialized not in store_api._local_clent_map
            assert initialized not in store_api._control_plane_map
            # The other store keeps its record and client.
            assert await ts.client(connected) is connected_client
            await ts.shutdown(connected)
            control_plane.close.assert_not_called()
            [connection] = dialed
            connection.close.assert_awaited_once_with()

    asyncio.run(run())
    actors.lookup.assert_not_called()
    assert store_api._local_clent_map == store_api._control_plane_map == {}


@pytest.mark.timeout(60)
def test_shutdown_releases_state_when_close_fails(monkeypatch) -> None:
    """Drop the record and the client first, then raise the close's error."""
    _patch_actors(monkeypatch)
    store_name = _store_name()
    dialed: list[ControlPlane] = []

    async def connect(address: _RegistryAddress) -> ControlPlane:
        connection = await _HOSTS[address.host]()
        connection.close = AsyncMock(side_effect=ConnectionResetError("gone"))
        dialed.append(connection)
        return connection

    async def run() -> None:
        async with _hosted_registry() as host:
            await (await _HOSTS[host]()).init({"publisher/0"}, _strategy())
            address = _RegistryAddress(store_name, host)
            with patch.object(_RegistryAddress, "connect", connect):
                await ts.client(store_name, role=RankRole.PUBLISHER, address=address)
            with pytest.raises(ConnectionResetError, match="gone"):
                await ts.shutdown(store_name)
            assert store_api._control_plane_map == store_api._local_clent_map == {}
            [connection] = dialed
            connection.close.assert_awaited_once_with()

    asyncio.run(run())


@pytest.mark.timeout(60)
@pytest.mark.parametrize("selection", ["initialized", "dialed", "spmd", "standard"])
def test_a_released_store_spawns_no_monarch_controller(monkeypatch, selection) -> None:
    """After a shutdown releases a store, role-less calls and shutdown spawn nothing.

    A standard initialize, or a routing one on Monarch's coordinator, takes the name back.
    """
    actors = _patch_actors(monkeypatch)
    store_name = _store_name()

    async def run() -> None:
        async with _hosted_registry() as host:
            control_plane = await _HOSTS[host]()
            if selection == "dialed":
                await control_plane.init({"publisher/0"}, _strategy())
                address = _RegistryAddress(store_name, host)
                await ts.client(store_name, role=RankRole.PUBLISHER, address=address)
            else:
                await _initialize(store_name, control_plane)
                await ts.client(store_name, role=RankRole.PUBLISHER)
            await ts.shutdown(store_name)
            if selection == "spmd":
                # An SPMD session for the name still takes role-less calls and shutdown.
                session = SimpleNamespace(
                    controller=_mock_controller(), shutdown=AsyncMock()
                )
                monkeypatch.setitem(store_api._spmd_state_map, store_name, session)
                client = await ts.client(store_name)
                assert client._controller is session.controller
                await ts.shutdown(store_name)
                session.shutdown.assert_awaited_once_with()
                return
            if selection == "standard":
                controller = _mock_controller()
                actors.lookup.side_effect = None
                actors.lookup.return_value = controller
                await ts.initialize(store_name=store_name)
                assert (await ts.client(store_name))._controller is controller
                await ts.shutdown(store_name)
                controller.teardown.call.assert_awaited_once_with()
                return
            await ts.shutdown(store_name)
            with pytest.raises(ValueError, match="needs role on its first ts.client"):
                await ts.client(store_name)
            actors.lookup.assert_not_called()
            # A Monarch initialize finds no stray controller under the name.
            actors.lookup.side_effect = None
            actors.lookup.return_value = _mock_coordinator()
            await _initialize(store_name)
            actors.lookup.assert_awaited_once_with(store_name, RoutingCoordinator)
            # The initialize took the name back, so role-less calls look up Monarch again.
            controller = _mock_controller()
            actors.lookup.return_value = controller
            assert (await ts.client(store_name))._controller is controller

    asyncio.run(run())


@pytest.mark.timeout(60)
def test_an_address_connects_once_until_shutdown(monkeypatch) -> None:
    """Reuse the connection after a reset or a failed or cancelled reuse, until shutdown."""
    actors = _patch_actors(monkeypatch)
    store_name = _store_name()
    connect = patch.object(
        _RegistryAddress,
        "connect",
        autospec=True,
        side_effect=_RegistryAddress.connect,
    )

    async def run() -> None:
        async with _hosted_registry() as host:
            await (await _HOSTS[host]()).init({"publisher/0"}, _strategy())
            address = _RegistryAddress(store_name, host)
            with connect as connects:
                first = await ts.client(
                    store_name, role=RankRole.PUBLISHER, address=address
                )
                ts.reset_client(store_name)
                again = await ts.client(
                    store_name,
                    role=RankRole.PUBLISHER,
                    address=pickle.loads(pickle.dumps(address)),
                )
                assert again is not first
                assert again._control_plane is first._control_plane
                assert connects.call_count == 1
                # A failed reuse keeps the connection recorded and open.
                ts.reset_client(store_name)
                connection = first._control_plane
                connection.close = AsyncMock()
                connection.strategy = AsyncMock(side_effect=RuntimeError("down"))
                with pytest.raises(RuntimeError, match="down"):
                    await ts.client(
                        store_name, role=RankRole.PUBLISHER, address=address
                    )
                connection.close.assert_not_awaited()
                assert store_api._control_plane_map == {
                    store_name: (connection, address)
                }
                # So does a cancelled one.
                entered, blocked = asyncio.Event(), asyncio.Event()

                async def strategy() -> None:
                    entered.set()
                    await blocked.wait()

                connection.strategy = strategy
                reuse = asyncio.ensure_future(
                    ts.client(store_name, role=RankRole.PUBLISHER, address=address)
                )
                await entered.wait()
                reuse.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await reuse
                assert store_api._closing == set()
                assert store_api._control_plane_map == {
                    store_name: (connection, address)
                }
                await ts.shutdown(store_name)
                connection.close.assert_awaited_once_with()
                redialed = await ts.client(
                    store_name, role=RankRole.PUBLISHER, address=address
                )
                assert redialed._control_plane is not first._control_plane
                assert connects.call_count == 2

    asyncio.run(run())
    actors.lookup.assert_not_called()


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "call", ["client", "slow_client", "cancelled", "shutdown", "cancelled_shutdown"]
)
def test_a_failing_dialed_close_is_only_logged(monkeypatch, caplog, call) -> None:
    """Log a failing close of a dialed connection, never in place of the call's own outcome.

    A failed client keeps the record it found and waits 5s for its close, logging one that
    outlasts the wait, a cancelled one never waits, and a shutdown stops waiting at 5s or
    when cancelled, leaving the close to run on.
    """
    caplog.set_level(logging.WARNING, logger=store_api.logger.name)
    _patch_actors(monkeypatch)
    waits = _RecordedWaits()
    monkeypatch.setattr(store_api, "asyncio", waits)
    store_name = _store_name()

    async def run() -> _RegistryAddress:
        async with _hosted_registry() as host:
            address = _RegistryAddress(store_name, host)
            dialed, peer = await _HOSTS[host](), asyncio.Event()
            entered = asyncio.Event()

            async def close() -> None:
                if call != "client":
                    entered.set()
                    await peer.wait()
                raise ConnectionResetError("peer gone")

            dialed.close = close
            connect = patch.object(
                _RegistryAddress, "connect", AsyncMock(return_value=dialed)
            )
            if call in ("client", "slow_client"):
                recorded = await _HOSTS[host]()
                await _initialize(store_name, recorded)
                dialed.strategy = AsyncMock(side_effect=RuntimeError("strategy failed"))
                with connect, pytest.raises(RuntimeError, match="strategy failed"):
                    await ts.client(
                        store_name, role=RankRole.PUBLISHER, address=address
                    )
                assert store_api._control_plane_map == {store_name: (recorded, None)}
                assert store_api._local_clent_map == {}
                client = await ts.client(store_name, role=RankRole.PUBLISHER)
                assert client._control_plane is recorded
                if call == "slow_client":
                    # The close outlasted its wait and fails only now.
                    [closing] = store_api._closing
                    peer.set()
                    await asyncio.wait_for(
                        asyncio.gather(closing, return_exceptions=True), timeout=30
                    )
            elif call == "cancelled":
                strategy_entered = asyncio.Event()

                async def strategy() -> None:
                    strategy_entered.set()
                    await peer.wait()

                dialed.strategy = strategy
                with connect:
                    task = asyncio.ensure_future(
                        ts.client(store_name, role=RankRole.PUBLISHER, address=address)
                    )
                    await strategy_entered.wait()
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task
                assert store_api._control_plane_map == store_api._local_clent_map == {}
                # The close started without the call waiting for it, and fails later.
                [closing] = store_api._closing
                peer.set()
                await asyncio.wait_for(
                    asyncio.gather(closing, return_exceptions=True), timeout=30
                )
            else:
                await (await _HOSTS[host]()).init({"publisher/0"}, _strategy())
                with connect:
                    await ts.client(
                        store_name, role=RankRole.PUBLISHER, address=address
                    )
                if call == "shutdown":
                    with pytest.raises(TimeoutError, match="took over"):
                        await ts.shutdown(store_name)
                else:
                    # The caller's own cancel comes through, not a timeout of the close.
                    task = asyncio.ensure_future(ts.shutdown(store_name))
                    await entered.wait()
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task
                assert store_api._control_plane_map == store_api._local_clent_map == {}
                # The close runs on and fails once the shutdown stopped waiting.
                [closing] = store_api._closing
                peer.set()
                await asyncio.wait_for(
                    asyncio.gather(closing, return_exceptions=True), timeout=30
                )
            return address

    address = asyncio.run(run())
    assert waits.timeouts == ([] if call == "cancelled" else [5.0])
    *slow, failed = (
        r
        for r in caplog.records
        if r.name == store_api.logger.name and r.levelno == logging.WARNING
    )
    still_running = f"closing the connection dialed through {address} is still running"
    assert [r.getMessage() for r in slow] == (
        [f"{still_running} after 5s"] if call == "slow_client" else []
    )
    assert (
        failed.getMessage() == f"closing the connection dialed through {address} failed"
    )
    assert isinstance(failed.exc_info[1], ConnectionResetError)


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    ("ending", "message"),
    [
        (asyncio.CancelledError, "was cancelled"),
        (GeneratorExit, "raised GeneratorExit"),
    ],
)
def test_a_dialed_close_that_ends_cancelled_counts_as_failed(
    monkeypatch, caplog, ending, message
) -> None:
    """Raise or log a close's own cancel or ``GeneratorExit`` as a failure, never as the caller's."""
    caplog.set_level(logging.WARNING, logger=store_api.logger.name)
    _patch_actors(monkeypatch)
    store_name = _store_name()

    async def run() -> None:
        async with _hosted_registry() as host:
            await (await _HOSTS[host]()).init({"publisher/0"}, _strategy())
            address = _RegistryAddress(store_name, host)
            dialed = await _HOSTS[host]()

            async def close() -> None:
                raise ending

            dialed.close = close
            connect = patch.object(
                _RegistryAddress, "connect", AsyncMock(return_value=dialed)
            )
            with connect:
                await ts.client(store_name, role=RankRole.PUBLISHER, address=address)
            with pytest.raises(RuntimeError, match=message):
                await ts.shutdown(store_name)
            dialed.strategy = AsyncMock(side_effect=ValueError("strategy failed"))
            with connect, pytest.raises(ValueError, match="strategy failed"):
                await ts.client(store_name, role=RankRole.PUBLISHER, address=address)

    asyncio.run(run())
    [failed] = [
        r
        for r in caplog.records
        if r.name == store_api.logger.name and r.levelno == logging.WARNING
    ]
    assert failed.getMessage().endswith("failed")
    assert isinstance(failed.exc_info[1], RuntimeError)


@pytest.mark.timeout(60)
def test_a_close_that_fails_as_its_shutdown_is_cancelled_is_logged(
    monkeypatch, caplog
) -> None:
    """Log a close that fails in the loop step that cancels the shutdown waiting for it."""
    caplog.set_level(logging.WARNING, logger=store_api.logger.name)
    _patch_actors(monkeypatch)
    store_name = _store_name()

    async def run() -> _RegistryAddress:
        async with _hosted_registry() as host:
            await (await _HOSTS[host]()).init({"publisher/0"}, _strategy())
            address = _RegistryAddress(store_name, host)
            dialed = await _HOSTS[host]()
            waiting: list[asyncio.Task] = []

            async def close() -> None:
                waiting[0].cancel()
                raise ConnectionResetError("peer gone")

            dialed.close = close
            with patch.object(
                _RegistryAddress, "connect", AsyncMock(return_value=dialed)
            ):
                await ts.client(store_name, role=RankRole.PUBLISHER, address=address)
            waiting.append(asyncio.ensure_future(ts.shutdown(store_name)))
            with pytest.raises(asyncio.CancelledError):
                await waiting[0]
            await asyncio.sleep(0)
            return address

    address = asyncio.run(run())
    [failed] = [
        r
        for r in caplog.records
        if r.name == store_api.logger.name and r.levelno == logging.WARNING
    ]
    assert (
        failed.getMessage() == f"closing the connection dialed through {address} failed"
    )
    assert isinstance(failed.exc_info[1], ConnectionResetError)


def test_a_close_wait_closed_after_its_close_ended_schedules_nothing() -> None:
    """Let a waiting coroutine closed once its loop is gone end quietly, its close already done."""
    selected = store_api._SelectedControlPlane(
        SimpleNamespace(close=AsyncMock()), _RegistryAddress("store", "host")
    )
    loop = asyncio.new_event_loop()

    async def start_and_abandon() -> Any:
        waiting = store_api._wait_for_close(selected)
        waiting.send(None)
        # The close runs and ends, but nothing resumes the waiting coroutine.
        await asyncio.gather(*store_api._closing)
        return waiting

    try:
        waiting = loop.run_until_complete(start_and_abandon())
    finally:
        loop.close()
    waiting.close()
    selected.control_plane.close.assert_awaited_once_with()


@pytest.mark.timeout(60)
def test_a_close_wait_closed_while_its_close_runs_still_logs_its_failure(
    caplog,
) -> None:
    """Leave a close running when its waiting coroutine is closed, and log how it fails."""
    caplog.set_level(logging.WARNING, logger=store_api.logger.name)
    address = _RegistryAddress("store", "host")

    async def run() -> None:
        peer = asyncio.Event()

        async def close() -> None:
            await peer.wait()
            raise ConnectionResetError("peer gone")

        waiting = store_api._wait_for_close(
            store_api._SelectedControlPlane(SimpleNamespace(close=close), address)
        )
        waiting.send(None)
        waiting.close()
        [closing] = store_api._closing
        peer.set()
        await asyncio.gather(closing, return_exceptions=True)

    asyncio.run(run())
    [failed] = [
        r
        for r in caplog.records
        if r.name == store_api.logger.name and r.levelno == logging.WARNING
    ]
    assert (
        failed.getMessage() == f"closing the connection dialed through {address} failed"
    )
    assert isinstance(failed.exc_info[1], ConnectionResetError)


@pytest.mark.timeout(60)
def test_a_failed_connect_records_nothing(monkeypatch) -> None:
    """Raise a failed connect's error, record and cache nothing and look up no Monarch actor."""
    actors = _patch_actors(monkeypatch)
    store_name = _store_name()
    refused = patch.object(
        _RegistryAddress,
        "connect",
        AsyncMock(side_effect=ConnectionRefusedError("refused")),
    )

    async def run() -> None:
        with refused, pytest.raises(ConnectionRefusedError, match="refused"):
            await ts.client(
                store_name,
                role=RankRole.PUBLISHER,
                address=_RegistryAddress(store_name, "down"),
            )

    asyncio.run(run())
    assert store_api._control_plane_map == store_api._local_clent_map == {}
    actors.lookup.assert_not_called()


@pytest.mark.timeout(60)
def test_initialize_closes_a_dialed_record_and_drops_the_client(monkeypatch) -> None:
    """Close the connection this process dialed for the store and drop its cached client."""
    _patch_actors(monkeypatch)
    store_name = _store_name()

    async def run() -> None:
        async with _hosted_registry() as host, _hosted_registry() as other:
            await (await _HOSTS[host]()).init({"publisher/0"}, _strategy())
            dialed = await _HOSTS[host]()
            dialed.close = AsyncMock()
            address = _RegistryAddress(store_name, host)
            with patch.object(
                _RegistryAddress, "connect", AsyncMock(return_value=dialed)
            ):
                old = await ts.client(
                    store_name, role=RankRole.PUBLISHER, address=address
                )
            instance = await _HOSTS[other]()
            await _initialize(store_name, instance)
            dialed.close.assert_awaited_once_with()
            new = await ts.client(
                store_name, role=RankRole.PUBLISHER, control_plane=instance
            )
            assert new is not old and new._control_plane is instance

    asyncio.run(run())


@dataclass(frozen=True)
class _TaggedAddress(_RegistryAddress):
    """An address with a field its equality leaves out, as a log label would be."""

    tag: str = field(default="", compare=False)


@pytest.mark.timeout(60)
def test_addresses_match_by_value(monkeypatch) -> None:
    """Treat addresses equal by their dataclass as one, so a copy reuses the connection."""
    _patch_actors(monkeypatch)
    store_name = _store_name()
    connect = patch.object(
        _TaggedAddress, "connect", autospec=True, side_effect=_RegistryAddress.connect
    )

    async def run() -> None:
        async with _hosted_registry() as host:
            await (await _HOSTS[host]()).init({"publisher/0"}, _strategy())

            def tagged(tag: str) -> _TaggedAddress:
                return _TaggedAddress(store_name, host, tag)

            with connect as connects:
                first = await ts.client(
                    store_name, role=RankRole.PUBLISHER, address=tagged("a")
                )
                cached = await ts.client(
                    store_name, role=RankRole.PUBLISHER, address=tagged("b")
                )
                ts.reset_client(store_name)
                again = await ts.client(
                    store_name, role=RankRole.PUBLISHER, address=tagged("c")
                )
            assert cached is first
            assert again is not first and again._control_plane is first._control_plane
            assert connects.call_count == 1

    asyncio.run(run())


@pytest.mark.timeout(60)
def test_a_connect_that_returns_no_control_plane_records_nothing(monkeypatch) -> None:
    """Refuse a connect that returns no ``ControlPlane``, keeping the dialed record and its connection."""
    _patch_actors(monkeypatch)
    store_name = _store_name()

    async def run() -> None:
        async with _hosted_registry() as host:
            await (await _HOSTS[host]()).init({"publisher/0"}, _strategy())
            address = _RegistryAddress(store_name, host)
            first = await ts.client(
                store_name, role=RankRole.PUBLISHER, address=address
            )
            first._control_plane.close = AsyncMock()
            ts.reset_client(store_name)
            other = _RegistryAddress(store_name, "elsewhere")
            with patch.object(
                _RegistryAddress, "connect", AsyncMock(return_value=object())
            ):
                with pytest.raises(TypeError, match="not a ControlPlane"):
                    await ts.client(store_name, role=RankRole.PUBLISHER, address=other)
            assert store_api._control_plane_map == {
                store_name: (first._control_plane, address)
            }
            assert store_api._local_clent_map == {}
            first._control_plane.close.assert_not_awaited()

    asyncio.run(run())


@pytest.mark.timeout(60)
@pytest.mark.parametrize("passed_to", ["client", "initialize"])
def test_a_dialed_connection_passed_back_in_stays_torchstores_to_close(
    monkeypatch, caplog, passed_to
) -> None:
    """Keep and log the address of a dialed connection passed back in, so shutdown closes it once."""
    caplog.set_level(logging.INFO)
    _patch_actors(monkeypatch)
    store_name = _store_name()

    async def run() -> None:
        async with _hosted_registry() as host:
            await (await _HOSTS[host]()).init({"publisher/0"}, _strategy())
            address = _RegistryAddress(store_name, host)
            dialed = (
                await ts.client(store_name, role=RankRole.PUBLISHER, address=address)
            )._control_plane
            dialed.close = AsyncMock()
            ts.reset_client(store_name)
            if passed_to == "client":
                await ts.client(
                    store_name, role=RankRole.PUBLISHER, control_plane=dialed
                )
            else:
                # The registry ran init already, and only the record matters here.
                dialed.init = AsyncMock()
                await _initialize(store_name, dialed)
            assert store_api._control_plane_map == {store_name: (dialed, address)}
            # The dial's line and the line of the call that passed it back both name it.
            addresses = [line[1] for line in _selection_lines(caplog)]
            assert addresses == [str(address)] * 2
            dialed.close.assert_not_awaited()
            await ts.shutdown(store_name)
            dialed.close.assert_awaited_once_with()

    asyncio.run(run())


@pytest.mark.timeout(60)
def test_a_monarch_initialize_drops_and_closes_a_dialed_record(monkeypatch) -> None:
    """Start the store over on Monarch's coordinator, closing the connection dialed before."""
    actors = _patch_actors(monkeypatch)
    actors.lookup.side_effect = None
    actors.lookup.return_value = _mock_coordinator()
    store_name = _store_name()

    async def run() -> None:
        async with _hosted_registry() as host:
            await (await _HOSTS[host]()).init({"publisher/0"}, _strategy())
            address = _RegistryAddress(store_name, host)
            dialed = (
                await ts.client(store_name, role=RankRole.PUBLISHER, address=address)
            )._control_plane
            dialed.close = AsyncMock()
            await _initialize(store_name)
            assert store_api._control_plane_map == store_api._local_clent_map == {}
            dialed.close.assert_awaited_once_with()

    asyncio.run(run())


@pytest.mark.timeout(60)
def test_a_refused_initialize_keeps_the_record_and_the_client(monkeypatch) -> None:
    """Keep the recorded plane and the cached client when a new plane refuses init."""
    _patch_actors(monkeypatch)
    store_name = _store_name()

    async def run() -> None:
        async with registry_store() as connect:
            recorded = await connect()
            await _initialize(store_name, recorded)
            client = await ts.client(store_name, role=RankRole.PUBLISHER)
            refusing = await connect()
            refusing.init = AsyncMock(side_effect=RuntimeError("refused"))
            with pytest.raises(RuntimeError, match="refused"):
                await _initialize(store_name, refusing)
            assert store_api._control_plane_map == {store_name: (recorded, None)}
            assert await ts.client(store_name) is client

    asyncio.run(run())


@pytest.mark.timeout(60)
def test_equal_control_planes_may_serve_two_stores(monkeypatch) -> None:
    """Refuse only the same instance for another store, not an equal one."""
    _patch_actors(monkeypatch)
    stores = (_store_name(), _store_name())

    async def run() -> list[ControlPlane]:
        async with _hosted_registry() as host:
            planes = [await _HOSTS[host](_EqualRegistry) for _ in stores]
            for store_name, control_plane in zip(stores, planes):
                # Both share one registry, whose init runs once.
                control_plane.init = AsyncMock()
                await _initialize(store_name, control_plane)
            return planes

    planes = asyncio.run(run())
    recorded = [store_api._control_plane_map[name].control_plane for name in stores]
    assert all(plane is mine for plane, mine in zip(recorded, planes, strict=True))


@pytest.mark.timeout(60)
def test_another_shutdown_of_a_released_store_drops_only_its_client(
    monkeypatch,
) -> None:
    """Drop a client built after the release without any Monarch lookup."""
    actors = _patch_actors(monkeypatch)
    store_name = _store_name()

    async def run() -> None:
        async with registry_store() as connect:
            await _initialize(store_name, await connect())
            await ts.shutdown(store_name)
            actors.lookup.side_effect = None
            actors.lookup.return_value = _mock_coordinator()
            client = await ts.client(store_name, role=RankRole.PUBLISHER)
            assert store_api._local_clent_map == {store_name: client}
            actors.lookup.reset_mock()
            await ts.shutdown(store_name)
            assert store_api._local_clent_map == {}
            actors.lookup.assert_not_called()

    asyncio.run(run())


@pytest.mark.timeout(60)
def test_shutting_down_a_recorded_store_leaves_an_spmd_session_of_its_name(
    monkeypatch,
) -> None:
    """Drop only a recorded store's record and client, leaving an SPMD session of its name."""
    _patch_actors(monkeypatch)
    store_name = _store_name()
    session = SimpleNamespace(shutdown=AsyncMock())

    async def run() -> None:
        async with registry_store() as connect:
            await _initialize(store_name, await connect())
            with patch.dict(store_api._spmd_state_map, {store_name: session}):
                await ts.shutdown(store_name)
                assert store_api._spmd_state_map[store_name] is session
            assert store_api._control_plane_map == {}

    asyncio.run(run())
    session.shutdown.assert_not_awaited()


@pytest.mark.timeout(60)
def test_a_caller_cancelled_while_a_replaced_connection_closes_keeps_its_cancel(
    monkeypatch,
) -> None:
    """Keep a cancelled initialize's ``CancelledError`` while it waits for a dialed close."""
    _patch_actors(monkeypatch)
    store_name = _store_name()

    async def run() -> None:
        async with _hosted_registry() as host, _hosted_registry() as other:
            await (await _HOSTS[host]()).init({"publisher/0"}, _strategy())
            dialed, entered, peer = (
                await _HOSTS[host](),
                asyncio.Event(),
                asyncio.Event(),
            )

            async def close() -> None:
                entered.set()
                await peer.wait()

            dialed.close = close
            address = _RegistryAddress(store_name, host)
            with patch.object(
                _RegistryAddress, "connect", AsyncMock(return_value=dialed)
            ):
                await ts.client(store_name, role=RankRole.PUBLISHER, address=address)
            task = asyncio.ensure_future(_initialize(store_name, await _HOSTS[other]()))
            await entered.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            peer.set()

    asyncio.run(run())


@pytest.mark.timeout(60)
def test_a_close_that_raises_when_called_leaves_a_cancel_alone(
    monkeypatch, caplog
) -> None:
    """Keep a cancelled client's ``CancelledError`` and log its dial's close that raised."""
    caplog.set_level(logging.WARNING, logger=store_api.logger.name)
    _patch_actors(monkeypatch)
    store_name = _store_name()

    async def run() -> None:
        async with _hosted_registry() as host:
            dialed, entered, blocked = (
                await _HOSTS[host](),
                asyncio.Event(),
                asyncio.Event(),
            )

            async def strategy() -> None:
                entered.set()
                await blocked.wait()

            def close() -> None:
                raise ConnectionResetError("closed at once")

            dialed.strategy, dialed.close = strategy, close
            address = _RegistryAddress(store_name, host)
            with patch.object(
                _RegistryAddress, "connect", AsyncMock(return_value=dialed)
            ):
                task = asyncio.ensure_future(
                    ts.client(store_name, role=RankRole.PUBLISHER, address=address)
                )
                await entered.wait()
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            await asyncio.gather(*store_api._closing, return_exceptions=True)

    asyncio.run(run())
    [failed] = [r for r in caplog.records if r.name == store_api.logger.name]
    assert isinstance(failed.exc_info[1], ConnectionResetError)


@pytest.mark.timeout(60)
@pytest.mark.parametrize("ending", ["cancelled", "succeeds"])
def test_a_close_left_running_logs_only_a_failure(monkeypatch, caplog, ending) -> None:
    """Log nothing for a close that ``asyncio.run`` cancels or that succeeds after the wait."""
    caplog.set_level(logging.WARNING)
    _patch_actors(monkeypatch)
    monkeypatch.setattr(store_api, "asyncio", _RecordedWaits())
    store_name = _store_name()

    async def run() -> None:
        async with _hosted_registry() as host:
            await (await _HOSTS[host]()).init({"publisher/0"}, _strategy())
            dialed, peer = await _HOSTS[host](), asyncio.Event()
            dialed.close = peer.wait
            address = _RegistryAddress(store_name, host)
            with patch.object(
                _RegistryAddress, "connect", AsyncMock(return_value=dialed)
            ):
                await ts.client(store_name, role=RankRole.PUBLISHER, address=address)
            with pytest.raises(TimeoutError, match="took over"):
                await ts.shutdown(store_name)
            [closing] = store_api._closing
            if ending == "succeeds":
                peer.set()
                await asyncio.wait_for(closing, timeout=30)
            # Otherwise asyncio.run cancels the close as it returns.

    asyncio.run(run())
    assert [
        r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING
    ] == []


@pytest.mark.timeout(60)
def test_a_failed_initialize_records_nothing(monkeypatch, caplog) -> None:
    """Record a caller's control plane only once its initialize succeeds."""
    caplog.set_level(logging.INFO)
    _patch_actors(monkeypatch)
    store_name = _store_name()
    failure = RuntimeError("plane-init failed")

    async def run() -> None:
        async with registry_store() as connect:
            control_plane = await connect()
            control_plane.init = AsyncMock(side_effect=failure)
            with pytest.raises(RuntimeError, match=str(failure)):
                await _initialize(store_name, control_plane)
            assert store_api._control_plane_map == store_api._local_clent_map == {}
            assert _selection_lines(caplog) == []

    asyncio.run(run())


@pytest.mark.timeout(60)
def test_racing_first_clients_on_one_address_share_one_connection(
    monkeypatch, caplog
) -> None:
    """Keep the client and connection recorded first, and close the racing call's own dial.

    Each call gets its own equal copy of the address, as each actor call does. Only the
    call that builds the client logs a selection line.
    """
    caplog.set_level(logging.INFO)
    actors = _patch_actors(monkeypatch)
    store_name = _store_name()
    dialed: list[ControlPlane] = []
    both_dialed = asyncio.Event()

    async def connect(address: _RegistryAddress) -> ControlPlane:
        connection = await _HOSTS[address.host]()
        connection.close = AsyncMock()
        strategy = connection.strategy

        async def strategy_once_both_dialed() -> Any:
            await both_dialed.wait()
            return await strategy()

        connection.strategy = strategy_once_both_dialed
        dialed.append(connection)
        if len(dialed) == 2:
            both_dialed.set()
        return connection

    async def run() -> None:
        async with _hosted_registry() as host:
            await (await _HOSTS[host]()).init({"publisher/0"}, _strategy())
            address = _RegistryAddress(store_name, host)
            copies = [address, pickle.loads(pickle.dumps(address))]
            with patch.object(_RegistryAddress, "connect", connect):
                clients = await asyncio.wait_for(
                    asyncio.gather(
                        *(
                            ts.client(store_name, role=RankRole.PUBLISHER, address=copy)
                            for copy in copies
                        )
                    ),
                    timeout=30,
                )
            kept, address_kept = store_api._control_plane_map[store_name]
            assert address_kept == address
            assert clients[0] is clients[1] and clients[0]._control_plane is kept
            [dropped] = [connection for connection in dialed if connection is not kept]
            dropped.close.assert_awaited_once_with()
            kept.close.assert_not_awaited()
            await ts.shutdown(store_name)
            kept.close.assert_awaited_once_with()

    asyncio.run(run())
    actors.lookup.assert_not_called()
    assert len(_selection_lines(caplog)) == 1


@pytest.mark.timeout(60)
@pytest.mark.parametrize("reset", ["before", "between"])
def test_racing_clients_around_a_reset_share_the_recorded_connection(
    monkeypatch, reset
) -> None:
    """Share the recorded connection when ``reset_client`` comes before or between races.

    Before: both calls reuse the record, and the second takes the first's client.
    Between: the second fresh dial closes itself and builds a new client on the first's
    connection, since the reset dropped the first's client.
    """
    actors = _patch_actors(monkeypatch)
    store_name = _store_name()
    dialed: list[ControlPlane] = []

    async def run() -> None:
        async with _hosted_registry() as host:
            await (await _HOSTS[host]()).init({"publisher/0"}, _strategy())
            address = _RegistryAddress(store_name, host)
            gates: list[asyncio.Event] = []
            entered: list[None] = []
            both_entered = asyncio.Event()

            def gate(connection: ControlPlane) -> None:
                strategy, opened = connection.strategy, asyncio.Event()
                gates.append(opened)

                async def gated() -> Any:
                    entered.append(None)
                    if len(entered) == 2:
                        both_entered.set()
                    await opened.wait()
                    return await strategy()

                connection.strategy = gated

            async def connect(_: _RegistryAddress) -> ControlPlane:
                connection = await _HOSTS[host]()
                connection.close = AsyncMock()
                dialed.append(connection)
                if reset == "between":
                    gate(connection)
                return connection

            def call() -> Any:
                return ts.client(store_name, role=RankRole.PUBLISHER, address=address)

            with patch.object(_RegistryAddress, "connect", connect):
                if reset == "before":
                    gate((await call())._control_plane)
                    ts.reset_client(store_name)
                calls = [asyncio.ensure_future(call()) for _ in range(2)]
                await asyncio.wait_for(both_entered.wait(), timeout=30)
                if reset == "before":
                    gates[0].set()
                    clients = await asyncio.wait_for(asyncio.gather(*calls), timeout=30)
                    assert clients[0] is clients[1] and len(dialed) == 1
                    return
                gates[0].set()
                done, pending = await asyncio.wait(
                    calls, timeout=30, return_when=asyncio.FIRST_COMPLETED
                )
                ts.reset_client(store_name)
                gates[1].set()
                [first], [second] = done, pending
                first, second = first.result(), await asyncio.wait_for(second, 30)
            kept, _ = store_api._control_plane_map[store_name]
            assert first._control_plane is second._control_plane is kept is dialed[0]
            assert first is not second
            dialed[0].close.assert_not_awaited()
            dialed[1].close.assert_awaited_once_with()

    asyncio.run(run())
    actors.lookup.assert_not_called()


@pytest.mark.timeout(60)
def test_racing_first_clients_of_two_stores_cannot_share_an_instance(
    monkeypatch,
) -> None:
    """Let only one of two stores whose first clients race record the same instance."""
    _patch_actors(monkeypatch)
    stores = [_store_name(), _store_name()]

    async def run() -> list[Any]:
        async with _hosted_registry() as host:
            await (await _HOSTS[host]()).init({"publisher/0"}, _strategy())
            instance = await _HOSTS[host]()
            strategy, entered, both = instance.strategy, [], asyncio.Event()

            async def strategy_once_both_entered() -> Any:
                entered.append(None)
                if len(entered) == 2:
                    both.set()
                await both.wait()
                return await strategy()

            instance.strategy = strategy_once_both_entered
            return await asyncio.wait_for(
                asyncio.gather(
                    *(
                        ts.client(name, role=RankRole.PUBLISHER, control_plane=instance)
                        for name in stores
                    ),
                    return_exceptions=True,
                ),
                timeout=30,
            )

    results = asyncio.run(run())
    assert sorted(type(result).__name__ for result in results) == [
        "RoutingClient",
        "ValueError",
    ]
    [error] = [result for result in results if isinstance(result, ValueError)]
    assert "already serves store" in str(error)


@pytest.mark.timeout(60)
def test_initialize_and_new_clients_each_log_one_selection_line(
    monkeypatch, caplog
) -> None:
    """Log the control plane once on the driver and once per new client."""
    caplog.set_level(logging.INFO)
    actors = _patch_actors(monkeypatch)
    actors.lookup.side_effect = None
    actors.lookup.return_value = _mock_coordinator()
    monarch, initialized, connected = _store_name(), _store_name(), _store_name()
    given = _store_name()
    pid = str(os.getpid())

    async def run() -> str:
        async with _hosted_registry() as host:
            await _initialize(monarch)
            await ts.client(monarch, role=RankRole.REQUESTER, group=1)
            # Stores on Monarch's coordinator are never recorded.
            assert monarch not in store_api._control_plane_map
            control_plane = await _HOSTS[host](_NamedRegistry)
            await _initialize(initialized, control_plane)
            await ts.client(initialized, role=RankRole.PUBLISHER)
            await ts.client(
                connected,
                role=RankRole.PUBLISHER,
                address=_RegistryAddress(connected, host),
            )
            # A client built on the recorded connection names its address too.
            ts.reset_client(connected)
            await ts.client(connected, role=RankRole.REQUESTER, group=0)
            given_plane = await _HOSTS[host](_NamedRegistry)
            await ts.client(given, role=RankRole.PUBLISHER, control_plane=given_plane)
            # Cached clients log nothing more.
            for store_name in (monarch, initialized, connected, given):
                await ts.client(store_name)
            return host

    host = asyncio.run(run())
    assert _selection_lines(caplog) == [
        ("monarch", "-", monarch, "-", pid),
        ("monarch", "-", monarch, "requester/1/0", pid),
        ("byo", "-", initialized, "-", pid),
        ("byo", "-", initialized, "publisher/0", pid),
        ("HostedRegistry", f"registry:{host}", connected, "publisher/0", pid),
        ("HostedRegistry", f"registry:{host}", connected, "requester/0/0", pid),
        ("byo", "-", given, "publisher/0", pid),
    ]


@pytest.mark.timeout(60)
def test_a_forked_child_dials_its_own_and_keeps_given_planes(monkeypatch) -> None:
    """After the fork hook, refuse calls without the address and share a caller's plane.

    A shutdown, a ``control_plane=`` call and a standard initialize each take a name back.
    """
    actors = _patch_actors(monkeypatch)
    monkeypatch.setattr(store_api, "_selection_lock", threading.Lock())
    monkeypatch.setattr(store_api, "_inherited", [])
    redial, released, given_store, replaced, standard = (
        _store_name() for _ in range(5)
    )

    async def run() -> None:
        async with _hosted_registry() as host:
            await (await _HOSTS[host]()).init({"publisher/0"}, _strategy())
            addresses = {
                name: _RegistryAddress(name, host)
                for name in (redial, released, replaced, standard)
            }
            for name, address in addresses.items():
                await ts.client(name, role=RankRole.PUBLISHER, address=address)
            parents = {
                name: store_api._control_plane_map[name].control_plane
                for name in addresses
            }
            given = await _HOSTS[host]()
            shared = await ts.client(
                given_store, role=RankRole.PUBLISHER, control_plane=given
            )

            # The hook frees the lock the fork held, here one held for it.
            held = threading.Lock()
            held.acquire()
            store_api._after_fork_in_child(held)
            assert not held.locked()

            for kwargs in ({"role": RankRole.PUBLISHER}, {}):
                with pytest.raises(RuntimeError, match="inherited across fork"):
                    await ts.client(redial, **kwargs)
            assert await ts.client(given_store) is shared
            await ts.shutdown(released)
            replacement = await _HOSTS[host]()
            await ts.client(
                replaced, role=RankRole.PUBLISHER, control_plane=replacement
            )
            actors.lookup.side_effect = None
            actors.lookup.return_value = _mock_controller()
            await ts.initialize(store_name=standard)
            assert store_api._forked_stores == {redial: addresses[redial]}
            assert store_api._released_stores == {released}
            actors.lookup.reset_mock()
            child = await ts.client(
                redial, role=RankRole.PUBLISHER, address=addresses[redial]
            )
            assert child._control_plane is not parents[redial]
            ts.reset_client(redial)
            with pytest.raises(ValueError, match="needs role"):
                await ts.client(redial)
        actors.lookup.assert_not_called()

    asyncio.run(run())


class _Crash(BaseException):
    """A crash that is no ``Exception``, as pyo3's ``PanicException`` is."""


@pytest.mark.timeout(60)
def test_a_dial_whose_strategy_crashes_is_closed_and_not_recorded(monkeypatch) -> None:
    """Start closing a dialed connection whose ``strategy()`` raises a non-``Exception``."""
    _patch_actors(monkeypatch)
    waits = _RecordedWaits()
    monkeypatch.setattr(store_api, "asyncio", waits)
    monkeypatch.setattr(store_api, "_closing", set())
    store_name = _store_name()
    closes: list[None] = []

    async def run() -> None:
        async with _hosted_registry() as host:
            await (await _HOSTS[host]()).init({"publisher/0"}, _strategy())
            release = asyncio.Event()

            async def close() -> None:
                closes.append(None)
                await release.wait()

            async def connect(_: _RegistryAddress) -> ControlPlane:
                connection = await _HOSTS[host]()
                connection.strategy = AsyncMock(side_effect=_Crash("panic"))
                connection.close = close
                return connection

            address = _RegistryAddress(store_name, host)
            with patch.object(_RegistryAddress, "connect", connect):
                with pytest.raises(_Crash):
                    await ts.client(
                        store_name, role=RankRole.PUBLISHER, address=address
                    )
            # The close runs as its own task, which the call does not wait for.
            [closing] = store_api._closing
            assert (waits.timeouts, closing.done()) == ([], False)
            release.set()
            await asyncio.wait_for(closing, timeout=30)
            assert store_api._control_plane_map == store_api._local_clent_map == {}

    asyncio.run(run())
    assert closes == [None]


@pytest.mark.timeout(60)
def test_a_replaced_dial_whose_close_crashes_is_only_logged(
    monkeypatch, caplog
) -> None:
    """Only log a replaced dialed connection whose ``close`` raises a non-``Exception``."""
    _patch_actors(monkeypatch)
    store_name = _store_name()

    async def run() -> None:
        async with _hosted_registry() as host:
            await (await _HOSTS[host]()).init({"publisher/0"}, _strategy())
            address = _RegistryAddress(store_name, host)
            dialed = (
                await ts.client(store_name, role=RankRole.PUBLISHER, address=address)
            )._control_plane
            dialed.close = AsyncMock(side_effect=_Crash("panic"))
            ts.reset_client(store_name)
            given = await _HOSTS[host]()
            client = await ts.client(
                store_name, role=RankRole.PUBLISHER, control_plane=given
            )
            assert client._control_plane is given
            dialed.close.assert_awaited_once_with()

    asyncio.run(run())
    assert [
        record.exc_info[0]
        for record in caplog.records
        if record.getMessage().endswith("failed") and record.exc_info
    ] == [_Crash]


@pytest.mark.timeout(60)
@pytest.mark.parametrize("strategy_fails", [False, True], ids=["strategy", "close"])
def test_a_closed_dialing_client_starts_no_close_and_logs_nothing(
    monkeypatch, caplog, strategy_fails
) -> None:
    """Start no close for a ``ts.client`` closed while its dial's ``strategy()`` waits.

    One closed while a failed ``strategy()``'s close waits leaves it running and logs nothing.
    """
    caplog.set_level(logging.WARNING, logger=store_api.logger.name)
    _patch_actors(monkeypatch)
    monkeypatch.setattr(store_api, "_closing", set())
    store_name = _store_name()
    closes: list[None] = []

    async def run() -> None:
        async with _hosted_registry() as host:
            dialed, release = await _HOSTS[host](), asyncio.Event()

            async def strategy() -> None:
                if strategy_fails:
                    raise RuntimeError("strategy failed")
                await release.wait()

            async def close() -> None:
                closes.append(None)
                await release.wait()

            dialed.strategy, dialed.close = strategy, close
            address = _RegistryAddress(store_name, host)
            with patch.object(
                _RegistryAddress, "connect", AsyncMock(return_value=dialed)
            ):
                call = ts.client(store_name, role=RankRole.PUBLISHER, address=address)
                # Run the call to its first wait, then close it as the collector would.
                call.send(None)
                call.close()
            release.set()
            await asyncio.wait_for(asyncio.gather(*store_api._closing), timeout=30)
            assert store_api._control_plane_map == store_api._local_clent_map == {}

    asyncio.run(run())
    assert closes == ([None] if strategy_fails else [])
    assert [
        r.getMessage() for r in caplog.records if r.name == store_api.logger.name
    ] == []


@pytest.mark.timeout(60)
@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
def test_an_interrupt_while_a_failed_dial_closes_propagates(
    monkeypatch, caplog, interrupt
) -> None:
    """Let ``KeyboardInterrupt`` and ``SystemExit`` out of a failed dial's close wait, logging nothing."""
    caplog.set_level(logging.WARNING, logger=store_api.logger.name)
    _patch_actors(monkeypatch)
    monkeypatch.setattr(store_api, "_closing", set())
    store_name = _store_name()

    async def run() -> None:
        async with _hosted_registry() as host:
            dialed, release = await _HOSTS[host](), asyncio.Event()
            dialed.strategy = AsyncMock(side_effect=RuntimeError("strategy failed"))

            async def close() -> None:
                await release.wait()

            dialed.close = close
            address = _RegistryAddress(store_name, host)
            with patch.object(
                _RegistryAddress, "connect", AsyncMock(return_value=dialed)
            ):
                call = ts.client(store_name, role=RankRole.PUBLISHER, address=address)
                # Run the call to its wait for the close, then interrupt it there.
                call.send(None)
                with pytest.raises(interrupt):
                    call.throw(interrupt())
            release.set()
            await asyncio.wait_for(asyncio.gather(*store_api._closing), timeout=30)

    asyncio.run(run())
    assert [
        r.getMessage() for r in caplog.records if r.name == store_api.logger.name
    ] == []


class _SkippedAddress(_RegistryAddress):
    """Test-only: an address subclass the dataclass decorator skipped."""

    def __init__(self, store_name: str, host: str, port: int) -> None:
        super().__init__(store_name, host)
        object.__setattr__(self, "port", port)


class _OwnEqualityAddress(_SkippedAddress):
    """Test-only: an address subclass the decorator skipped, with an ``__eq__`` of its own."""

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _OwnEqualityAddress) and vars(self) == vars(other)


@dataclass(frozen=True, eq=False)
class _BaseEqualityAddress(_RegistryAddress):
    """Test-only: an address dataclass given ``eq=False``, which inherits its base's ``__eq__``."""

    port: int


@pytest.mark.parametrize(
    "address_class",
    [_SkippedAddress, _OwnEqualityAddress, _BaseEqualityAddress],
    ids=["skipped", "skipped_own_eq", "eq_false"],
)
def test_client_refuses_an_address_class_that_is_no_dataclass(
    monkeypatch, address_class
) -> None:
    """Refuse an address class the decorator skipped, or gave ``eq=False`` without its own ``__eq__``."""
    _patch_actors(monkeypatch)
    store_name = _store_name()
    address = address_class(store_name, "host", 1)
    with pytest.raises(TypeError, match="must be a dataclass with eq=True"):
        asyncio.run(ts.client(store_name, role=RankRole.PUBLISHER, address=address))


class _RecordingLock:
    """A stand-in for the selection lock that records whether a caller holds it."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.held = False

    def __enter__(self) -> _RecordingLock:
        self._lock.acquire()
        self.held = True
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.held = False
        self._lock.release()


class _CheckedDict(dict):
    """A record map that notes, at each change, whether the selection lock is held."""

    def __init__(self, name: str, lock: _RecordingLock, notes: list) -> None:
        super().__init__()
        self._name, self._lock, self._notes = name, lock, notes

    def __setitem__(self, key: Any, value: Any) -> None:
        self._notes.append((f"{self._name} set", self._lock.held))
        super().__setitem__(key, value)

    def pop(self, *args: Any) -> Any:
        self._notes.append((f"{self._name} pop", self._lock.held))
        return super().pop(*args)


class _CheckedSet(set):
    """A set of store marks that notes, at each change, whether the selection lock is held."""

    def __init__(self, name: str, lock: _RecordingLock, notes: list) -> None:
        super().__init__()
        self._name, self._lock, self._notes = name, lock, notes

    def add(self, item: Any) -> None:
        self._notes.append((f"{self._name} add", self._lock.held))
        super().add(item)

    def discard(self, item: Any) -> None:
        self._notes.append((f"{self._name} discard", self._lock.held))
        super().discard(item)


@pytest.mark.timeout(60)
def test_records_change_only_under_the_selection_lock(monkeypatch) -> None:
    """Change records and marks, and check a plane's store, only under the selection lock.

    Every close of a dialed connection runs with the lock free.
    """
    actors = _patch_actors(monkeypatch)
    lock = _RecordingLock()
    monkeypatch.setattr(store_api, "_selection_lock", lock)
    notes: list[tuple[str, bool]] = []
    for name in ("_control_plane_map", "_forked_stores"):
        monkeypatch.setattr(store_api, name, _CheckedDict(name, lock, notes))
    released = _CheckedSet("_released_stores", lock, notes)
    monkeypatch.setattr(store_api, "_released_stores", released)
    for name in ("_check_one_store", "_record_client", "_select_control_plane"):

        def checked(
            *args: Any, _name: str = name, _original: Any = getattr(store_api, name)
        ) -> Any:
            notes.append((_name, lock.held))
            return _original(*args)

        monkeypatch.setattr(store_api, name, checked)
    closes: list[bool] = []
    by_client, by_initialize, by_shutdown, standard = (_store_name() for _ in range(4))

    async def run() -> None:
        async with _hosted_registry() as host:
            await (await _HOSTS[host]()).init({"publisher/0"}, _strategy())

            async def connect(_: _RegistryAddress) -> ControlPlane:
                connection = await _HOSTS[host]()
                connection.close = AsyncMock(
                    side_effect=lambda: closes.append(lock.held)
                )
                return connection

            async def given() -> ControlPlane:
                control_plane = await _HOSTS[host]()
                control_plane.init = AsyncMock()
                return control_plane

            with patch.object(_RegistryAddress, "connect", connect):
                for store_name in (by_client, by_initialize, by_shutdown):
                    address = _RegistryAddress(store_name, host)
                    await ts.client(
                        store_name, role=RankRole.PUBLISHER, address=address
                    )
            # A client's plane, an initialize's plane and a shutdown each replace a dial.
            ts.reset_client(by_client)
            await ts.client(
                by_client, role=RankRole.PUBLISHER, control_plane=await given()
            )
            await _initialize(by_initialize, await given())
            await ts.shutdown(by_shutdown)
            actors.lookup.side_effect = None
            actors.lookup.return_value = _mock_controller()
            await ts.initialize(store_name=standard)

    asyncio.run(run())
    assert [note for note in notes if not note[1]] == []
    assert {name for name, _ in notes} == {
        "_check_one_store",
        "_record_client",
        "_select_control_plane",
        "_control_plane_map set",
        "_control_plane_map pop",
        "_forked_stores pop",
        "_released_stores add",
        "_released_stores discard",
    }
    assert closes == [False, False, False]


@pytest.mark.timeout(60)
def test_an_equal_plane_given_after_a_dial_replaces_and_closes_it(monkeypatch) -> None:
    """Take a caller's plane equal to the dialed one but not it, with no address, closing the dial."""
    _patch_actors(monkeypatch)
    store_name = _store_name()

    async def run() -> None:
        async with _hosted_registry() as host:
            await (await _HOSTS[host]()).init({"publisher/0"}, _strategy())

            async def connect(_: _RegistryAddress) -> ControlPlane:
                return await _HOSTS[host](_EqualRegistry)

            address = _RegistryAddress(store_name, host)
            with patch.object(_RegistryAddress, "connect", connect):
                dialed = (
                    await ts.client(
                        store_name, role=RankRole.PUBLISHER, address=address
                    )
                )._control_plane
            dialed.close = AsyncMock()
            ts.reset_client(store_name)
            equal = await _HOSTS[host](_EqualRegistry)
            assert equal == dialed and equal is not dialed
            client = await ts.client(
                store_name, role=RankRole.PUBLISHER, control_plane=equal
            )
            recorded = store_api._control_plane_map[store_name]
            assert client._control_plane is equal and recorded.control_plane is equal
            assert recorded.address is None
            dialed.close.assert_awaited_once_with()

    asyncio.run(run())


@pytest.mark.timeout(60)
def test_a_connect_that_returns_the_recorded_plane_keeps_it_open(monkeypatch) -> None:
    """Keep a recorded connection open when another address's ``connect`` returns that plane."""
    _patch_actors(monkeypatch)
    store_name = _store_name()

    async def run() -> None:
        async with _hosted_registry() as host:
            await (await _HOSTS[host]()).init({"publisher/0"}, _strategy())
            first = _RegistryAddress(store_name, host)
            dialed = (
                await ts.client(store_name, role=RankRole.PUBLISHER, address=first)
            )._control_plane
            dialed.close = AsyncMock()
            ts.reset_client(store_name)

            async def connect(_: _RegistryAddress) -> ControlPlane:
                return dialed

            second = _RegistryAddress(store_name, "another-host")
            with patch.object(_RegistryAddress, "connect", connect):
                client = await ts.client(
                    store_name, role=RankRole.PUBLISHER, address=second
                )
            assert client._control_plane is dialed
            assert store_api._control_plane_map[store_name].address == second
            dialed.close.assert_not_awaited()

    asyncio.run(run())


_FORK_CHILD = """
import asyncio, importlib, os, signal, sys
from dataclasses import dataclass
from types import SimpleNamespace

root, store_name = sys.argv[1:]
sys.path.insert(0, root)
# Registered before TorchStore's hooks, so in the child it runs first and sees the lock as
# the fork left it.
held_at_fork = []
os.register_at_fork(
    after_in_child=lambda: held_at_fork.append(
        sys.modules["torchstore.api"]._selection_lock.locked()
    )
)
import torchstore as ts
import torchstore.api as store_api
import torchstore.backend as backend_module

# A reload registers TorchStore's fork hooks again, with a lock of their own, and the forks
# below must still finish and leave both locks free.
first_lock = store_api._selection_lock
importlib.reload(store_api)


class Connection(ts.ControlPlane):
    def __init__(self):
        self.pid = os.getpid()

    def __del__(self):
        if os.getpid() != self.pid:
            print("finalizer ran in the child", flush=True)

    async def init(self, publishers, strategy):
        pass

    async def strategy(self):
        return None

    async def register_layout(self, rank, key, registrations):
        pass

    async def get_layouts(self, key, *, timeout_s=None):
        return {}


@dataclass(frozen=True)
class Address(ts.ControlPlaneAddress):
    async def connect(self):
        return Connection()

    def __str__(self):
        return "driver:1"


async def no_monarch(*args, **kwargs):
    raise LookupError("looked up a Monarch actor")


# A rank-0 stand-in for the Monarch backend that refuses every actor lookup.
backend_module._backends["monarch"] = SimpleNamespace(
    rank=lambda: 0, controller=no_monarch, coordinator=no_monarch
)
address = Address(store_name)
asyncio.run(ts.client(store_name, role=ts.RankRole.PUBLISHER, address=address))
lock = store_api._selection_lock
pid = os.fork()
if pid == 0:
    import traceback

    signal.alarm(60)  # ends a child stuck on a lock it inherited
    try:
        # The fork held the lock, so the child copied it held and its hook freed it.
        assert held_at_fork == [True], "the fork did not hold the selection lock"
        assert store_api._selection_lock is lock and not lock.locked()
        assert lock.acquire(timeout=5)
        lock.release()
        client = asyncio.run(
            ts.client(store_name, role=ts.RankRole.PUBLISHER, address=address)
        )
        assert client._control_plane.pid == os.getpid(), "the child reused the parent's"
        print("the child dialed its own", flush=True)
    except BaseException:
        traceback.print_exc()
        os._exit(1)
    os._exit(0)
assert not lock.locked(), "the parent kept the selection lock after the fork"
assert not first_lock.locked(), "the parent kept the first import's lock after the fork"
status = os.waitstatus_to_exitcode(os.waitpid(pid, 0)[1])
# A second fork would wait forever on a lock the first left held.
signal.alarm(60)
second = os.fork()
if second == 0:
    os._exit(0)
assert os.waitstatus_to_exitcode(os.waitpid(second, 0)[1]) == 0
sys.exit(status)
"""


@pytest.mark.timeout(240)
def test_fork_holds_the_lock_and_the_child_dials_its_own() -> None:
    """Hold the selection lock across fork, also after a reload, then free it in the child.

    The child dials its own connection and, exiting through ``os._exit``, runs no finalizer
    of the parent's.
    """
    # A set variable skips TorchStore's logging setup, so its INFO lines stay off stdout.
    env = {**os.environ, "HYPERACTOR_CODEC_MAX_FRAME_LENGTH": "910737418240"}
    child = subprocess.run(
        [sys.executable, "-c", _FORK_CHILD, _ROOT, _store_name()],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    assert child.stdout.splitlines() == ["the child dialed its own"], child.stderr
