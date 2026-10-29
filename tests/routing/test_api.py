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
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
import torch
import torchstore as ts
import torchstore.api as store_api
from monarch.actor import ActorError
from torchstore import RankRole
from torchstore.client import ClientType
from torchstore.routing.client import RoutingClient
from torchstore.routing.control_plane import ControlPlane
from torchstore.routing.coordinator import MonarchControlPlane, RoutingCoordinator
from torchstore.storage_volume import StorageVolume

from .utils import HostedRegistry, registry_store

# Each child process imports the tree that pytest imported.
_ROOT = str(Path(importlib.util.find_spec("torchstore").origin).parents[1])
_SELECTION_LINE = re.compile(
    r"routing control_plane=(\S+) address=(\S+) store=(\S+) rank=(\S+) pid=(\d+)"
)
# Registry hosts that test addresses dial, by host id.
_HOSTS: dict[str, Callable[[], Awaitable[ControlPlane]]] = {}


@pytest.fixture(autouse=True)
def _fresh_process_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test with no cached client and no recorded or released store."""
    monkeypatch.setattr(store_api, "_local_clent_map", {})
    monkeypatch.setattr(store_api, "_control_plane_map", {})
    monkeypatch.setattr(store_api, "_released_stores", set())
    monkeypatch.setattr(store_api, "_forked_stores", {})


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
    monkeypatch, caplog
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
    spawn_volumes = AsyncMock(return_value=volumes)
    spawn_coordinator = AsyncMock(return_value=coordinator)

    monkeypatch.setattr(StorageVolume, "spawn", spawn_volumes)
    monkeypatch.setattr(store_api, "get_or_spawn_controller", spawn_coordinator)
    monkeypatch.setattr(store_api, "current_rank", lambda: SimpleNamespace(rank=0))
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
    spawn_volumes.assert_awaited_once()
    args, kwargs = spawn_volumes.await_args
    assert args == (1, publisher)
    assert kwargs["id_func"]() == "publisher/0"
    strategy.set_storage_volumes.assert_awaited_once_with(volumes)
    spawn_coordinator.assert_awaited_once_with("routing-test", RoutingCoordinator)
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


def test_client_constructs_and_caches_a_routing_client(monkeypatch, caplog) -> None:
    """Create one rank-local routing client and reuse it for later API calls."""
    caplog.set_level(logging.INFO, logger="torchstore.routing.coordinator")
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
    assert first._routing_directory.rank == "requester/1/2"
    assert first._role == RankRole.REQUESTER
    assert isinstance(first._control_plane, MonarchControlPlane)
    assert first.strategy is strategy
    assert second is first
    spawn_coordinator.assert_awaited_once_with("routing-test", RoutingCoordinator)
    coordinator.strategy.call_one.assert_awaited_once_with()
    # The strategy came through the adapter, which logs each call it forwards.
    assert [
        record.getMessage().split(" took ")[0]
        for record in caplog.records
        if record.name == "torchstore.routing.coordinator"
    ] == ["routing control_plane=monarch strategy"]


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
            del _HOSTS[host]


class _Mesh:
    def __init__(self, size: int) -> None:
        self._size = size

    def size(self) -> int:
        return self._size


def _store_name() -> str:
    return f"routing-test-{uuid.uuid4()}"


def _strategy() -> SimpleNamespace:
    return SimpleNamespace(set_storage_volumes=AsyncMock())


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
    monkeypatch.setattr(store_api, "get_or_spawn_controller", actors.lookup)
    monkeypatch.setattr(store_api, "current_rank", lambda: SimpleNamespace(rank=rank))
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
    """``asyncio`` for the API module: records each close wait, then cuts it to 0.2 s."""

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
            assert address == _RegistryAddress(store_name, host)

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
    """Refuse a bad role and group before dialing an address or calling an instance."""
    actors = _patch_actors(monkeypatch)
    store_name = _store_name()
    control_plane = Mock(spec=ControlPlane)
    connect = AsyncMock(return_value=control_plane)

    async def run() -> None:
        with patch.object(_RegistryAddress, "connect", connect):
            for kwargs in (
                {"address": _RegistryAddress(store_name, "unreachable")},
                {"control_plane": control_plane},
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
            assert copy is not address
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
            base = await _HOSTS[host]()
            old, new = (_EqualRegistry(base._registry, base._loop) for _ in range(2))
            assert old == new
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
    """Check role, types and the address's store before returning a cached client."""
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


@pytest.mark.timeout(60)
def test_monarch_lookup_before_init_hints_at_the_address(monkeypatch, caplog) -> None:
    """Add an address hint to the coordinator's ``ActorError`` before init."""
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(store_api, "current_rank", lambda: SimpleNamespace(rank=0))
    store_name = _store_name()

    # Monarch spawns an uninitialized coordinator for this new store name.
    with pytest.raises(ActorError) as raised:
        asyncio.run(ts.client(store_name, role=RankRole.REQUESTER, group=0))
    assert "RoutingCoordinator.init has not run" in str(raised.value)
    hint = (
        f" If ts.initialize gave store '{store_name}' another control plane, pass "
        "ts.client that control plane (control_plane=) or its address (address=)."
    )
    # The hint follows Monarch's own message instead of replacing it.
    assert raised.value.message.endswith(hint)
    assert len(raised.value.message) > len(hint)
    assert hint in str(raised.value)
    assert store_api._local_clent_map == store_api._control_plane_map == {}
    assert _selection_lines(caplog) == []


def test_monarch_lookup_passes_other_actor_errors_through(monkeypatch) -> None:
    """Leave any other coordinator error as it was."""
    error = ActorError(RuntimeError("something else"))
    message = error.message
    coordinator = SimpleNamespace(
        strategy=SimpleNamespace(call_one=AsyncMock(side_effect=error))
    )
    monkeypatch.setattr(
        store_api, "get_or_spawn_controller", AsyncMock(return_value=coordinator)
    )
    monkeypatch.setattr(store_api, "current_rank", lambda: SimpleNamespace(rank=0))

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
            # The caller's control plane still serves calls.
            await control_plane.strategy()

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

    A selection, or a standard or Monarch initialize, takes the name back.
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
                blocked = asyncio.Event()
                connection.strategy = blocked.wait
                with pytest.raises(asyncio.TimeoutError):
                    await asyncio.wait_for(
                        ts.client(store_name, role=RankRole.PUBLISHER, address=address),
                        timeout=0.2,
                    )
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

    A failed client keeps the record it found and waits 5 s for its close, logging one that
    outlasts the wait, a cancelled one never waits, and a shutdown stops waiting at 5 s or
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

            async def close() -> None:
                if call != "client":
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
                dialed.strategy = peer.wait
                with connect, pytest.raises(asyncio.TimeoutError):
                    await asyncio.wait_for(
                        ts.client(store_name, role=RankRole.PUBLISHER, address=address),
                        timeout=0.2,
                    )
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
                    with pytest.raises(asyncio.TimeoutError):
                        await asyncio.wait_for(ts.shutdown(store_name), timeout=0.1)
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
        [f"{still_running} after 5 s"] if call == "slow_client" else []
    )
    assert (
        failed.getMessage() == f"closing the connection dialed through {address} failed"
    )
    assert isinstance(failed.exc_info[1], ConnectionResetError)


@pytest.mark.timeout(60)
def test_a_dialed_close_that_ends_cancelled_counts_as_failed(
    monkeypatch, caplog
) -> None:
    """Raise or log a close that cancels itself as a failure, never as the caller's cancel."""
    caplog.set_level(logging.WARNING, logger=store_api.logger.name)
    _patch_actors(monkeypatch)
    store_name = _store_name()

    async def run() -> None:
        async with _hosted_registry() as host:
            await (await _HOSTS[host]()).init({"publisher/0"}, _strategy())
            address = _RegistryAddress(store_name, host)
            dialed = await _HOSTS[host]()

            async def close() -> None:
                raise asyncio.CancelledError

            dialed.close = close
            connect = patch.object(
                _RegistryAddress, "connect", AsyncMock(return_value=dialed)
            )
            with connect:
                await ts.client(store_name, role=RankRole.PUBLISHER, address=address)
            with pytest.raises(RuntimeError, match="was cancelled"):
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


@pytest.mark.timeout(60)
def test_addresses_compare_by_every_attribute(monkeypatch) -> None:
    """Tell apart addresses of a subclass that sets attributes outside its fields."""
    _patch_actors(monkeypatch)
    store_name = _store_name()

    class _RegionAddress(_RegistryAddress):
        def __init__(self, store_name: str, host: str, region: str) -> None:
            super().__init__(store_name, host)
            object.__setattr__(self, "region", region)

    async def run() -> None:
        async with _hosted_registry() as host:
            await (await _HOSTS[host]()).init({"publisher/0"}, _strategy())
            east = _RegionAddress(store_name, host, "east")
            west = _RegionAddress(store_name, host, "west")
            # The inherited dataclass equality reads only the dataclass fields.
            assert east == west
            await ts.client(store_name, role=RankRole.PUBLISHER, address=east)
            with pytest.raises(ValueError, match="did not connect through"):
                await ts.client(store_name, role=RankRole.PUBLISHER, address=west)

    asyncio.run(run())


@pytest.mark.timeout(60)
@pytest.mark.parametrize("step", ["plane-init", "volumes"])
def test_a_failed_initialize_records_nothing(monkeypatch, caplog, step) -> None:
    """Record a caller's control plane only once its initialize succeeds."""
    caplog.set_level(logging.INFO)
    actors = _patch_actors(monkeypatch)
    store_name = _store_name()
    failure = RuntimeError(f"{step} failed")

    async def run() -> None:
        async with registry_store() as connect:
            control_plane = await connect()
            if step == "plane-init":
                control_plane.init = AsyncMock(side_effect=failure)
            else:
                actors.spawn.side_effect = failure
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
            assert copies[0] == copies[1] and copies[0] is not copies[1]
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
    """Share one client and connection when ``reset_client`` comes before or between races.

    Before: both calls reuse the record, and the second takes the first's client.
    Between: the second fresh dial closes itself and builds on the first's connection.
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

            def gate(connection: ControlPlane) -> None:
                strategy, opened = connection.strategy, asyncio.Event()
                gates.append(opened)

                async def gated() -> Any:
                    entered.append(None)
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
                while len(entered) < 2:
                    await asyncio.sleep(0)
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
            base = await _HOSTS[host]()
            control_plane = _NamedRegistry(base._registry, base._loop)
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
            given_plane = _NamedRegistry(base._registry, base._loop)
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


_FORK_HELD_LOCK_CHILD = """
import asyncio, os, sys, threading
from dataclasses import dataclass
from types import SimpleNamespace

root, store_name = sys.argv[1:]
sys.path.insert(0, root)
import torchstore as ts
import torchstore.api as store_api


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


store_api.current_rank = lambda: SimpleNamespace(rank=0)
store_api.get_or_spawn_controller = no_monarch
address, other, given = Address(store_name), Address(store_name + "-b"), Connection()
for name, kwargs in (
    (store_name, {"address": address}),
    (other.store_name, {"address": other}),
    (store_name + "-c", {"control_plane": given}),
):
    asyncio.run(ts.client(name, role=ts.RankRole.PUBLISHER, **kwargs))
held, release = threading.Event(), threading.Event()


def hold():
    with store_api._selection_lock:
        held.set()
        release.wait()


threading.Thread(target=hold, daemon=True).start()
held.wait()
pid = os.fork()
if pid == 0:
    import gc, signal, traceback

    signal.alarm(60)  # ends a child stuck on the lock it inherited
    try:
        gc.collect()
        for kwargs in ({"role": ts.RankRole.PUBLISHER}, {}):
            try:
                asyncio.run(ts.client(store_name, **kwargs))
            except RuntimeError as error:
                assert "inherited across fork" in str(error), error
            else:
                raise AssertionError("a call without address kept the parent's record")
        # A shutdown of a dropped record needs no Monarch lookup.
        asyncio.run(ts.shutdown(other.store_name))
        # A caller's own plane stays recorded, so the child shares it and its client.
        shared = asyncio.run(ts.client(store_name + "-c"))
        assert shared._control_plane is given, "the fork dropped a caller's plane"
        client = asyncio.run(
            ts.client(store_name, role=ts.RankRole.PUBLISHER, address=address)
        )
        assert client._control_plane.pid == os.getpid(), "the child reused the parent's"
        ts.reset_client(store_name)
        try:
            asyncio.run(ts.client(store_name))
        except ValueError as error:
            assert "needs role" in str(error), error
        else:
            raise AssertionError("a role-less call ignored the child's new record")
        print("the child built a client", flush=True)
    except BaseException:
        traceback.print_exc()
        os._exit(1)
    os._exit(0)
release.set()
sys.exit(os.waitstatus_to_exitcode(os.waitpid(pid, 0)[1]))
"""


@pytest.mark.timeout(240)
def test_a_child_forked_while_a_parent_thread_holds_the_lock_dials_its_own() -> None:
    """Give a forked child a fresh selection lock, and make it dial its own connection.

    Another parent thread holds the lock across the fork. Calls without the address raise,
    shutdown needs no Monarch, a caller's plane stays shared and the os._exit child runs
    no finalizer of the parent's connections.
    """
    # A set variable skips TorchStore's logging setup, so its INFO lines stay off stdout.
    env = {**os.environ, "HYPERACTOR_CODEC_MAX_FRAME_LENGTH": "910737418240"}
    child = subprocess.run(
        [sys.executable, "-c", _FORK_HELD_LOCK_CHILD, _ROOT, _store_name()],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    assert child.stdout.splitlines() == ["the child built a client"], child.stderr
