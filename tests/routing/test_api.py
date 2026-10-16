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
    monkeypatch.setattr(store_api, "_initializing", [])
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
    assert first._routing_directory.rank == "requester/1/2"
    assert first._role == RankRole.REQUESTER
    assert first.strategy is strategy
    assert second is first
    spawn_coordinator.assert_awaited_once_with("routing-test", RoutingCoordinator)
    coordinator.strategy.call_one.assert_awaited_once_with()


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


def _hold_strategy(
    owner: Any, name: str = "strategy"
) -> tuple[asyncio.Event, asyncio.Event]:
    """Hold the strategy call: set the first event, answer once the second is set."""
    asked, answer = asyncio.Event(), asyncio.Event()
    strategy = getattr(owner, name)

    async def held_strategy() -> Any:
        asked.set()
        await answer.wait()
        return await strategy()

    setattr(owner, name, held_strategy)
    return asked, answer


class _EqualRegistry(HostedRegistry):
    """Equal to other such handles on its registry, as a dataclass control plane is."""

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _EqualRegistry) and other._registry is self._registry


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
def test_a_roleless_client_refuses_a_record_made_while_it_builds(monkeypatch) -> None:
    """Cache no standard client over a control plane recorded during its lookup."""
    actors = _patch_actors(monkeypatch)
    controller = _mock_controller()
    actors.lookup.side_effect = None
    actors.lookup.return_value = controller
    store_name = _store_name()

    async def run() -> None:
        async with registry_store() as connect:
            asked, answer = _hold_strategy(
                controller.get_controller_strategy, "call_one"
            )
            building = asyncio.ensure_future(ts.client(store_name))
            await asyncio.wait_for(asked.wait(), timeout=30)
            control_plane = await connect()
            await _initialize(store_name, control_plane)
            answer.set()
            with pytest.raises(ValueError, match="needs role on its first ts.client"):
                await building
            assert store_api._local_clent_map == {}
            client = await ts.client(store_name, role=RankRole.PUBLISHER)
            assert client._control_plane is control_plane

    asyncio.run(run())
    actors.lookup.assert_awaited_once()


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
def test_initialize_replaces_the_recorded_control_plane(monkeypatch) -> None:
    """Keep the latest initialize's control plane, dropping clients on older ones."""
    actors = _patch_actors(monkeypatch)
    store_name = _store_name()

    async def run() -> None:
        async with registry_store() as connect:
            first, second = await connect(), await connect()
            await _initialize(store_name, first)
            client = await ts.client(store_name, role=RankRole.PUBLISHER)
            assert client._control_plane is first
            # Initializing the same instance again keeps the client.
            await _initialize(store_name, first)
            assert await ts.client(store_name) is client
            await _initialize(store_name, second)
            client = await ts.client(store_name, role=RankRole.PUBLISHER)
            assert client._control_plane is second
            # An equal instance is another control plane all the same.
            old, new = (
                _EqualRegistry(second._registry, second._loop) for _ in range(2)
            )
            assert old == new
            await _initialize(store_name, old)
            await ts.client(store_name, role=RankRole.PUBLISHER)
            await _initialize(store_name, new)
            client = await ts.client(
                store_name, role=RankRole.PUBLISHER, control_plane=new
            )
            assert client._control_plane is new
            # Back on Monarch: the record and the client go, and lookups find the actor.
            actors.lookup.side_effect = None
            actors.lookup.return_value = _mock_coordinator()
            await _initialize(store_name)
            assert store_api._control_plane_map == {}
            client = await ts.client(store_name, role=RankRole.PUBLISHER)
            assert isinstance(client._control_plane, MonarchControlPlane)
            # Re-initializing on Monarch keeps this process's Monarch client.
            await _initialize(store_name)
            assert await ts.client(store_name) is client
            # A later control plane drops the Monarch client in turn.
            await _initialize(store_name, first)
            client = await ts.client(store_name, role=RankRole.PUBLISHER)
            assert client._control_plane is first

    asyncio.run(run())
    assert actors.lookup.await_count == 3


@pytest.mark.timeout(60)
def test_a_control_plane_serves_one_store(monkeypatch) -> None:
    """Refuse another store's instance before spawning or calling it, or as it records."""
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
            # Other stores are refused while one store's initialize on a new instance
            # spawns volumes and while it runs init, so only its init runs.
            shared, strategy = await connect(), _strategy()
            init = shared.init
            reached = {stage: asyncio.Event() for stage in ("spawn", "init")}
            resume = {stage: asyncio.Event() for stage in ("spawn", "init")}

            async def first_spawn(*args: Any, **kwargs: Any) -> object:
                if not reached["spawn"].is_set():
                    reached["spawn"].set()
                    await resume["spawn"].wait()
                return object()

            async def first_init(**kwargs: Any) -> None:
                if kwargs["strategy"] is strategy:
                    reached["init"].set()
                    await resume["init"].wait()
                await init(**kwargs)

            actors.spawn.side_effect = first_spawn
            shared.init = AsyncMock(side_effect=first_init)
            first, second = _store_name(), _store_name()
            running = asyncio.ensure_future(
                _initialize(first, shared, strategy=strategy)
            )
            for stage in ("spawn", "init"):
                await asyncio.wait_for(reached[stage].wait(), timeout=30)
                with pytest.raises(ValueError, match=refusal(first)):
                    await _initialize(second, shared)
                with pytest.raises(ValueError, match=refusal(first)):
                    await ts.client(
                        second, role=RankRole.PUBLISHER, control_plane=shared
                    )
                resume[stage].set()
            await asyncio.wait_for(running, timeout=30)
            shared.init.assert_awaited_once()
            assert actors.spawn.await_count == 2
            assert set(store_api._control_plane_map) == {weights, first}
            assert await shared.strategy() is strategy
            # Once that store shuts down, another store may take the instance.
            await ts.shutdown(first)
            await _initialize(second, shared)
            assert store_api._control_plane_map[second].control_plane is shared
            # A client given an instance that another store records meanwhile is
            # refused as it records.
            late, third, fourth = await connect(), _store_name(), _store_name()
            asked, answer = _hold_strategy(late)
            building = asyncio.ensure_future(
                ts.client(third, role=RankRole.PUBLISHER, control_plane=late)
            )
            await asyncio.wait_for(asked.wait(), timeout=30)
            await _initialize(fourth, late)
            answer.set()
            with pytest.raises(ValueError, match=refusal(fourth)):
                await building
            assert third not in store_api._control_plane_map
            assert third not in store_api._local_clent_map

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

    Another address after ``reset_client``, a caller's instance and a routing or standard
    initialize each close the connection dialed before. A caller's instance stays open.
    """
    actors = _patch_actors(monkeypatch)
    actors.lookup.side_effect = None
    actors.lookup.return_value = _mock_controller()
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
            await _initialize(store_name, await _HOSTS[second]())
            dialed[2].close.assert_awaited_once_with()
            await dial(second)
            await ts.initialize(store_name=store_name)
            assert [connection.close.await_count for connection in dialed] == [1] * 4
            assert store_api._control_plane_map == {}

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
    assert raised.value.message == "A remote actor call has failed."


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
@pytest.mark.parametrize("selection", ["initialized", "dialed", "spmd"])
def test_a_released_store_spawns_no_monarch_controller(monkeypatch, selection) -> None:
    """After a shutdown releases a store, spawn nothing for its name until re-selected."""
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
            await ts.shutdown(store_name)
            with pytest.raises(ValueError, match="needs role on its first ts.client"):
                await ts.client(store_name)
            actors.lookup.assert_not_called()
            # A Monarch initialize finds no stray controller under the name.
            actors.lookup.side_effect = None
            actors.lookup.return_value = _mock_coordinator()
            await _initialize(store_name)
            actors.lookup.assert_awaited_once_with(store_name, RoutingCoordinator)
            # A selection takes the name back, so role-less clients work again.
            controller = _mock_controller()
            actors.lookup.return_value = controller
            await ts.initialize(store_name=store_name)
            assert (await ts.client(store_name))._controller is controller

    asyncio.run(run())


@pytest.mark.timeout(60)
def test_an_address_connects_once_until_shutdown(monkeypatch) -> None:
    """Reuse the connection after a reset or failed reuse, and redial after shutdown."""
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
                await ts.shutdown(store_name)
                connection.close.assert_awaited_once_with()
                redialed = await ts.client(
                    store_name, role=RankRole.PUBLISHER, address=address
                )
                assert redialed._control_plane is not first._control_plane
                assert connects.call_count == 2

    asyncio.run(run())
    actors.lookup.assert_not_called()


def _stalled_close(connection: ControlPlane) -> asyncio.Event:
    """Make the connection's close wait for the returned event, like a stopped peer."""
    peer_answers = asyncio.Event()

    async def close() -> None:
        await peer_answers.wait()
        connection.closed = True

    connection.close = close
    return peer_answers


@pytest.mark.timeout(60)
@pytest.mark.parametrize("wait", ["cancelled", "too-long"])
def test_a_shutdown_leaves_a_slow_close_running(monkeypatch, caplog, wait) -> None:
    """Let a close finish after its shutdown is cancelled or stops waiting for it."""
    _patch_actors(monkeypatch)
    monkeypatch.setattr(store_api, "_CLOSE_WAIT_S", 0.5)
    store_name = _store_name()

    async def run() -> None:
        async with _hosted_registry() as host:
            await (await _HOSTS[host]()).init({"publisher/0"}, _strategy())
            address = _RegistryAddress(store_name, host)
            client = await ts.client(
                store_name, role=RankRole.PUBLISHER, address=address
            )
            connection = client._control_plane
            peer_answers = _stalled_close(connection)
            if wait == "cancelled":
                with pytest.raises(asyncio.TimeoutError):
                    await asyncio.wait_for(ts.shutdown(store_name), timeout=0.1)
            else:
                message = f"closing the connection dialed through {address} took over"
                with pytest.raises(TimeoutError, match=message):
                    await ts.shutdown(store_name)
            assert store_api._control_plane_map == store_api._local_clent_map == {}
            [closing] = store_api._closing
            peer_answers.set()
            await asyncio.wait_for(closing, timeout=30)
            assert connection.closed

    asyncio.run(run())
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_a_dialed_close_is_waited_for_5_seconds() -> None:
    """Keep the 5 s that callers wait for a dialed connection's close."""
    assert store_api._CLOSE_WAIT_S == 5.0


@pytest.mark.timeout(60)
def test_a_close_that_fails_after_shutdown_stops_waiting_is_logged(
    monkeypatch, caplog
) -> None:
    """Log a close that fails once ``ts.shutdown`` has stopped waiting for it."""
    _patch_actors(monkeypatch)
    monkeypatch.setattr(store_api, "_CLOSE_WAIT_S", 0.2)
    store_name = _store_name()

    async def run() -> None:
        async with _hosted_registry() as host:
            await (await _HOSTS[host]()).init({"publisher/0"}, _strategy())
            address = _RegistryAddress(store_name, host)
            client = await ts.client(
                store_name, role=RankRole.PUBLISHER, address=address
            )
            peer = asyncio.Event()

            async def close() -> None:
                await peer.wait()
                raise ConnectionResetError("peer gone")

            client._control_plane.close = close
            with pytest.raises(TimeoutError, match="took over"):
                await ts.shutdown(store_name)
            [closing] = store_api._closing
            peer.set()
            await asyncio.wait_for(
                asyncio.gather(closing, return_exceptions=True), timeout=30
            )

    asyncio.run(run())
    [record] = [
        r
        for r in caplog.records
        if r.name == store_api.logger.name and r.levelno == logging.WARNING
    ]
    assert record.getMessage().startswith("closing the connection dialed through")
    assert isinstance(record.exc_info[1], ConnectionResetError)


@pytest.mark.timeout(60)
def test_a_cancelled_client_never_waits_for_its_dialed_close(
    monkeypatch, caplog
) -> None:
    """Return at the caller's deadline, and log a close that fails afterwards."""
    _patch_actors(monkeypatch)
    store_name = _store_name()

    async def run() -> float:
        stopped = asyncio.Event()

        async def connect(address: _RegistryAddress) -> ControlPlane:
            connection = await _HOSTS[address.host]()
            connection.strategy = stopped.wait

            async def close() -> None:
                await stopped.wait()
                raise ConnectionResetError("peer gone")

            connection.close = close
            return connection

        async with _hosted_registry() as host:
            address = _RegistryAddress(store_name, host)
            start = asyncio.get_running_loop().time()
            with patch.object(_RegistryAddress, "connect", connect):
                with pytest.raises(asyncio.TimeoutError):
                    await asyncio.wait_for(
                        ts.client(store_name, role=RankRole.PUBLISHER, address=address),
                        timeout=0.2,
                    )
            elapsed = asyncio.get_running_loop().time() - start
            [closing] = store_api._closing
            stopped.set()
            await asyncio.wait_for(
                asyncio.gather(closing, return_exceptions=True), timeout=30
            )
            return elapsed

    # The call returns at its 0.2 s deadline, without the close's 5 s wait.
    assert asyncio.run(asyncio.wait_for(run(), timeout=30)) < store_api._CLOSE_WAIT_S
    [record] = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert record.getMessage().startswith("closing the connection dialed through")
    assert isinstance(record.exc_info[1], ConnectionResetError)


@pytest.mark.timeout(60)
def test_a_cancelled_reuse_keeps_the_recorded_connection_open(monkeypatch) -> None:
    """Leave the recorded connection open and recorded when a call reusing it is cancelled."""
    actors = _patch_actors(monkeypatch)
    store_name = _store_name()

    async def run() -> None:
        async with _hosted_registry() as host:
            await (await _HOSTS[host]()).init({"publisher/0"}, _strategy())
            address = _RegistryAddress(store_name, host)
            first = await ts.client(
                store_name, role=RankRole.PUBLISHER, address=address
            )
            connection = first._control_plane
            connection.close = AsyncMock()
            ts.reset_client(store_name)
            connection.strategy = asyncio.Event().wait
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(
                    ts.client(store_name, role=RankRole.PUBLISHER, address=address),
                    timeout=0.2,
                )
            await asyncio.gather(*store_api._closing, return_exceptions=True)
            connection.close.assert_not_awaited()
            assert store_api._control_plane_map[store_name].control_plane is connection

    asyncio.run(run())
    actors.lookup.assert_not_called()


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "step",
    [
        "init",
        "connect",
        "strategy",
        "dialed-strategy",
        "dialed-cancel",
        "dialed-strategy-close-fails",
        "dialed-cancel-close-fails",
        "dialed-interrupt",
    ],
)
def test_a_failed_selection_keeps_the_record_and_logs_nothing(
    monkeypatch, caplog, step
) -> None:
    """Leave the record, the client cache and the log as they were when a step fails.

    A connection dialed for the call is closed on an error or cancellation, not on
    an interrupt, and a failing close only logs, so the call's exception propagates.
    """
    caplog.set_level(logging.INFO)
    actors = _patch_actors(monkeypatch)
    store_name = _store_name()
    failure: BaseException = RuntimeError(f"{step} failed")
    if "cancel" in step:
        failure = asyncio.CancelledError()
    elif "interrupt" in step:
        failure = KeyboardInterrupt()
    close_error = ConnectionResetError("gone") if step.endswith("close-fails") else None

    async def run() -> None:
        async with _hosted_registry() as host:
            recorded, other = await _HOSTS[host](), await _HOSTS[host]()
            other.close = AsyncMock(side_effect=close_error)
            address = _RegistryAddress(store_name, host)
            await _initialize(store_name, recorded)
            # Only a failed init can find a cached client, and it keeps that client.
            cached = (
                await ts.client(store_name, role=RankRole.PUBLISHER)
                if step == "init"
                else None
            )
            lines = _selection_lines(caplog)
            with pytest.raises(type(failure), match=str(failure) or None):
                if step == "init":
                    other.init = AsyncMock(side_effect=failure)
                    await _initialize(store_name, other)
                elif step == "connect":
                    with patch.object(
                        _RegistryAddress, "connect", AsyncMock(side_effect=failure)
                    ):
                        await ts.client(
                            store_name, role=RankRole.PUBLISHER, address=address
                        )
                elif step == "strategy":
                    other.strategy = AsyncMock(side_effect=failure)
                    await ts.client(
                        store_name, role=RankRole.PUBLISHER, control_plane=other
                    )
                else:
                    # The address dials ``other``, whose strategy call then fails.
                    other.strategy = AsyncMock(side_effect=failure)
                    with patch.object(
                        _RegistryAddress, "connect", AsyncMock(return_value=other)
                    ):
                        await ts.client(
                            store_name, role=RankRole.PUBLISHER, address=address
                        )
            # A cancelled call leaves the close running on its own. Let it finish.
            await asyncio.wait_for(
                asyncio.gather(*store_api._closing, return_exceptions=True), timeout=30
            )
            assert store_api._control_plane_map == {store_name: (recorded, None)}
            assert store_api._initializing == []
            assert store_api._local_clent_map == (
                {} if cached is None else {store_name: cached}
            )
            assert _selection_lines(caplog) == lines
            # Only a connection this call dialed is closed, never a caller's own.
            if step.startswith("dialed") and step != "dialed-interrupt":
                other.close.assert_awaited_once_with()
            else:
                other.close.assert_not_awaited()
            close_logs = [
                (record.getMessage(), record.exc_info[1])
                for record in caplog.records
                if record.name == store_api.logger.name
                and record.levelno == logging.WARNING
            ]
            message = f"closing the connection dialed through {address} failed"
            assert close_logs == (
                [] if close_error is None else [(message, close_error)]
            )
            client = await ts.client(store_name, role=RankRole.PUBLISHER)
            assert client._control_plane is recorded

    asyncio.run(run())
    actors.lookup.assert_not_called()


@pytest.mark.timeout(60)
@pytest.mark.parametrize("step", ["standard-init", "monarch-init", "volumes"])
def test_a_failed_initialize_keeps_the_record_and_client(
    monkeypatch, caplog, step
) -> None:
    """Replace the recorded control plane only once an initialize succeeds."""
    caplog.set_level(logging.INFO)
    actors = _patch_actors(monkeypatch)
    store_name = _store_name()
    failure = RuntimeError(f"{step} failed")

    async def run() -> None:
        async with registry_store() as connect:
            recorded = await connect()
            await _initialize(store_name, recorded)
            client = await ts.client(store_name, role=RankRole.PUBLISHER)
            lines = _selection_lines(caplog)
            actors.lookup.side_effect = None
            with pytest.raises(RuntimeError, match=str(failure)):
                if step == "standard-init":
                    controller = _mock_controller()
                    controller.init.call.side_effect = failure
                    actors.lookup.return_value = controller
                    await ts.initialize(store_name=store_name)
                elif step == "monarch-init":
                    coordinator = _mock_coordinator()
                    coordinator.init.call_one.side_effect = failure
                    actors.lookup.return_value = coordinator
                    await _initialize(store_name)
                else:
                    actors.spawn.side_effect = failure
                    await _initialize(store_name)
            assert store_api._control_plane_map == {store_name: (recorded, None)}
            assert await ts.client(store_name) is client
            assert _selection_lines(caplog) == lines

    asyncio.run(run())


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "case", ["instance", "equal", "monarch", "monarch-lookup", "monarch-uninitialized"]
)
def test_a_client_follows_a_record_replaced_while_it_builds(
    monkeypatch, caplog, case
) -> None:
    """Build on, keep and log only the control plane of an initialize mid-build."""
    caplog.set_level(logging.INFO)
    actors = _patch_actors(monkeypatch)
    coordinator = _mock_coordinator()
    actors.lookup.side_effect = None
    actors.lookup.return_value = coordinator
    store_name = _store_name()

    async def run() -> None:
        async with registry_store() as connect:
            first, second = await connect(), await connect()
            if case == "equal":
                first, second = (
                    _EqualRegistry(first._registry, first._loop) for _ in range(2)
                )
            first.kind, second.kind = "first", "second"
            if case.startswith("monarch-"):
                # The client looks up Monarch's coordinator, as nothing is recorded.
                if case == "monarch-lookup":
                    await _initialize(store_name)
                else:
                    # The lookup finds an uninitialized coordinator, as on a new name.
                    coordinator.strategy.call_one.side_effect = ActorError(
                        RuntimeError("RoutingCoordinator.init has not run")
                    )
                asked, answer = _hold_strategy(coordinator.strategy, "call_one")
            else:
                await _initialize(store_name, first)
                asked, answer = _hold_strategy(first)
            building = asyncio.ensure_future(
                ts.client(store_name, role=RankRole.PUBLISHER)
            )
            await asyncio.wait_for(asked.wait(), timeout=30)
            # A Monarch initialize drops the record, as a shutdown does.
            await _initialize(store_name, "monarch" if case == "monarch" else second)
            answer.set()
            client = await building
            if case == "monarch":
                assert store_api._control_plane_map == {}
                assert isinstance(client._control_plane, MonarchControlPlane)
            else:
                assert store_api._control_plane_map == {store_name: (second, None)}
                assert client._control_plane is second
                assert (
                    await ts.client(
                        store_name, role=RankRole.PUBLISHER, control_plane=second
                    )
                    is client
                )
            assert await ts.client(store_name) is client

    asyncio.run(run())
    lookups = {"monarch": 2, "monarch-lookup": 2, "monarch-uninitialized": 1}
    assert actors.lookup.await_count == lookups.get(case, 0)
    kind = "monarch" if case == "monarch" else "second"
    client_lines = [line for line in _selection_lines(caplog) if line[3] != "-"]
    assert client_lines == [(kind, "-", store_name, "publisher/0", str(os.getpid()))]


@pytest.mark.timeout(60)
@pytest.mark.parametrize("close", ["keeps-calls", "fails-calls"])
def test_a_shutdown_while_an_address_reuses_its_connection_redials(
    monkeypatch, caplog, close
) -> None:
    """Dial again rather than keep, record or fail on a connection shut down mid-build."""
    caplog.set_level(logging.INFO)
    actors = _patch_actors(monkeypatch)
    store_name = _store_name()
    dialed: list[ControlPlane] = []
    closed: list[ControlPlane] = []

    async def connect(address: _RegistryAddress) -> ControlPlane:
        connection = await _HOSTS[address.host]()
        connection.close = AsyncMock(side_effect=lambda: closed.append(connection))
        dialed.append(connection)
        return connection

    async def run() -> str:
        async with _hosted_registry() as host:
            await (await _HOSTS[host]()).init({"publisher/0"}, _strategy())
            address = _RegistryAddress(store_name, host)
            with patch.object(_RegistryAddress, "connect", connect):
                await ts.client(store_name, role=RankRole.PUBLISHER, address=address)
                ts.reset_client(store_name)
                asked, answer = _hold_strategy(dialed[0])
                if close == "fails-calls":
                    held = dialed[0].strategy

                    async def strategy() -> Any:
                        # Like a real connection, closing fails the call in flight.
                        answered = await held()
                        if dialed[0] in closed:
                            raise ConnectionError("closed during the call")
                        return answered

                    dialed[0].strategy = strategy
                building = asyncio.ensure_future(
                    ts.client(store_name, role=RankRole.PUBLISHER, address=address)
                )
                await asyncio.wait_for(asked.wait(), timeout=30)
                await ts.shutdown(store_name)
                answer.set()
                client = await building
                ts.reset_client(store_name)
                again = await ts.client(
                    store_name, role=RankRole.PUBLISHER, address=address
                )
            first, second = dialed
            first.close.assert_awaited_once_with()
            assert client._control_plane is again._control_plane is second
            assert store_api._control_plane_map == {store_name: (second, address)}
            return host

    host = asyncio.run(run())
    actors.lookup.assert_not_called()
    # One line per client: the first dial, the redial and the reuse.
    line = ("HostedRegistry", f"registry:{host}", store_name, "publisher/0")
    assert _selection_lines(caplog) == [(*line, str(os.getpid()))] * 3


@pytest.mark.timeout(60)
@pytest.mark.parametrize("change", ["initialize", "shutdown"])
@pytest.mark.parametrize("given", ["address", "instance"])
def test_a_client_keeps_its_own_choice_when_the_record_changes_mid_build(
    monkeypatch, given, change
) -> None:
    """Dial and ask for the strategy once, and keep that choice, as the record changes."""
    actors = _patch_actors(monkeypatch)
    store_name = _store_name()

    async def run() -> None:
        async with _hosted_registry() as host:
            first, second = await _HOSTS[host](), await _HOSTS[host]()
            chosen = await _HOSTS[host]()
            chosen.close = AsyncMock()
            asked, answer = _hold_strategy(chosen)
            chosen.strategy = AsyncMock(side_effect=chosen.strategy)
            await _initialize(store_name, first)
            address = _RegistryAddress(store_name, host) if given == "address" else None
            kwargs = (
                {"control_plane": chosen} if address is None else {"address": address}
            )
            # A second dial raises StopAsyncIteration.
            connect = AsyncMock(side_effect=[chosen])
            with patch.object(_RegistryAddress, "connect", connect):
                building = asyncio.ensure_future(
                    ts.client(store_name, role=RankRole.PUBLISHER, **kwargs)
                )
                await asyncio.wait_for(asked.wait(), timeout=30)
                if change == "initialize":
                    await _initialize(store_name, second)
                else:
                    await ts.shutdown(store_name)
                answer.set()
                client = await building
            assert connect.await_count == (0 if address is None else 1)
            chosen.strategy.assert_awaited_once_with()
            assert client._control_plane is chosen
            assert store_api._control_plane_map == {store_name: (chosen, address)}
            chosen.close.assert_not_awaited()

    asyncio.run(run())
    actors.lookup.assert_not_called()


@pytest.mark.timeout(60)
def test_racing_first_clients_on_one_address_share_one_connection(monkeypatch) -> None:
    """Keep the connection recorded first, and close the racing call's own dial.

    Each call gets its own equal copy of the address, as each actor call does.
    """
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
            assert {client._control_plane for client in clients} == {kept}
            [dropped] = [connection for connection in dialed if connection is not kept]
            dropped.close.assert_awaited_once_with()
            kept.close.assert_not_awaited()
            await ts.shutdown(store_name)
            kept.close.assert_awaited_once_with()

    asyncio.run(run())
    actors.lookup.assert_not_called()


@pytest.mark.timeout(60)
def test_a_racing_dial_that_returns_the_recorded_plane_leaves_it_open(
    monkeypatch,
) -> None:
    """Close no connection when the racing call's ``connect()`` returned the one recorded."""
    actors = _patch_actors(monkeypatch)
    store_name = _store_name()

    async def run() -> None:
        async with _hosted_registry() as host:
            await (await _HOSTS[host]()).init({"publisher/0"}, _strategy())
            connection = await _HOSTS[host]()
            connection.close = AsyncMock()
            strategy = connection.strategy
            both_dialed = asyncio.Event()
            dials: list[None] = []

            async def strategy_once_both_dialed() -> Any:
                await both_dialed.wait()
                return await strategy()

            async def connect(address: _RegistryAddress) -> ControlPlane:
                dials.append(None)
                if len(dials) == 2:
                    both_dialed.set()
                return connection

            connection.strategy = strategy_once_both_dialed
            address = _RegistryAddress(store_name, host)
            with patch.object(_RegistryAddress, "connect", connect):
                clients = await asyncio.wait_for(
                    asyncio.gather(
                        *(
                            ts.client(
                                store_name, role=RankRole.PUBLISHER, address=address
                            )
                            for _ in range(2)
                        )
                    ),
                    timeout=30,
                )
            assert {client._control_plane for client in clients} == {connection}
            connection.close.assert_not_awaited()
            await ts.shutdown(store_name)
            connection.close.assert_awaited_once_with()

    asyncio.run(run())
    actors.lookup.assert_not_called()


@pytest.mark.timeout(60)
def test_a_shutdown_on_another_thread_waits_for_a_client_being_recorded(
    monkeypatch,
) -> None:
    """Never leave a connection a shutdown closed in the record or the client cache."""
    _patch_actors(monkeypatch)
    store_name = _store_name()
    shutdown_waits = threading.Event()
    outcomes: dict[str, Any] = {}

    class Lock:
        """The selection lock, telling the test when the shutdown waits for it."""

        def __init__(self) -> None:
            self._lock = threading.Lock()

        def __enter__(self) -> None:
            if threading.current_thread() is shutting_down:
                shutdown_waits.set()
            self._lock.acquire()

        def __exit__(self, *exc_info: object) -> None:
            self._lock.release()

    def shut_down() -> None:
        try:
            outcomes["shutdown"] = asyncio.run(ts.shutdown(store_name))
        except BaseException as error:
            outcomes["shutdown"] = error
        finally:
            shutdown_waits.set()

    shutting_down = threading.Thread(target=shut_down)
    routing_client = store_api.RoutingClient

    def recording_client(*args: Any) -> RoutingClient:
        if threading.current_thread() is building:
            # The build is recording its client: start the shutdown here.
            shutting_down.start()
            assert shutdown_waits.wait(timeout=30)
        return routing_client(*args)

    monkeypatch.setattr(store_api, "_selection_lock", Lock(), raising=False)
    monkeypatch.setattr(store_api, "RoutingClient", recording_client)

    async def run() -> None:
        nonlocal building
        async with _hosted_registry() as host:
            await (await _HOSTS[host]()).init({"publisher/0"}, _strategy())
            address = _RegistryAddress(store_name, host)
            first = await ts.client(
                store_name, role=RankRole.PUBLISHER, address=address
            )
            connection = first._control_plane
            connection.close = AsyncMock()
            ts.reset_client(store_name)
            reuse = ts.client(store_name, role=RankRole.PUBLISHER, address=address)

            def build() -> None:
                try:
                    outcomes["build"] = asyncio.run(reuse)
                except BaseException as error:
                    outcomes["build"] = error

            building = threading.Thread(target=build)
            building.start()
            for thread in (building, shutting_down):
                await asyncio.to_thread(thread.join, 30)
                assert not thread.is_alive()
            assert isinstance(outcomes["build"], RoutingClient), outcomes
            assert outcomes["shutdown"] is None, outcomes
            connection.close.assert_awaited_once_with()
            recorded = store_api._control_plane_map.get(store_name)
            assert recorded is None or recorded.control_plane is not connection
            cached = store_api._local_clent_map.get(store_name)
            assert cached is None or cached._control_plane is not connection

    building: threading.Thread | None = None
    asyncio.run(run())


@pytest.mark.timeout(60)
def test_a_standard_initialize_drops_the_routing_record(monkeypatch) -> None:
    """Hand a store name back to a standard store, and its controller's teardown."""
    actors = _patch_actors(monkeypatch)
    controller = _mock_controller()
    actors.lookup.side_effect = None
    actors.lookup.return_value = controller
    store_name = _store_name()

    async def run() -> None:
        async with registry_store() as connect:
            await _initialize(store_name, await connect())
            await ts.client(store_name, role=RankRole.PUBLISHER)
            await ts.initialize(store_name=store_name)
            controller.init.call.assert_awaited_once()
            assert store_api._control_plane_map == store_api._local_clent_map == {}
            client = await ts.client(store_name)
            assert not isinstance(client, RoutingClient)
            assert client._controller is controller
            await ts.shutdown(store_name)
            controller.teardown.call.assert_awaited_once_with()
            assert store_api._local_clent_map == {}
            # A later control plane drops the standard client in turn.
            assert not isinstance(await ts.client(store_name), RoutingClient)
            control_plane = await connect()
            await _initialize(store_name, control_plane)
            client = await ts.client(store_name, role=RankRole.PUBLISHER)
            assert client._control_plane is control_plane

    asyncio.run(run())


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
            control_plane = await _HOSTS[host]()
            control_plane.kind = "byo"
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
            given_plane = await _HOSTS[host]()
            given_plane.kind = "byo"
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


_LOGGING_CHILD = """
import asyncio, sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

root, store_name = sys.argv[1:]
sys.path.insert(0, root)
import torchstore as ts
import torchstore.api as store_api
from torchstore.storage_volume import StorageVolume


class Plane(ts.ControlPlane):
    async def init(self, publishers, strategy):
        self._strategy = strategy

    async def strategy(self):
        return self._strategy

    async def register_layout(self, rank, key, registrations):
        pass

    async def get_layouts(self, key, *, timeout_s=None):
        return {}


class Mesh:
    def size(self):
        return 1


async def main():
    StorageVolume.spawn = AsyncMock()
    store_api.current_rank = lambda: SimpleNamespace(rank=0)
    await ts.initialize(
        mesh=Mesh(),
        strategy=SimpleNamespace(set_storage_volumes=AsyncMock()),
        client_type=ts.ClientType.ROUTING,
        store_name=store_name,
        control_plane=Plane(),
    )
    await ts.client(store_name, role=ts.RankRole.PUBLISHER)
    await ts.client(store_name)


asyncio.run(main())
"""


@pytest.mark.timeout(240)
def test_selection_lines_show_under_torchstore_logging() -> None:
    """Print the selection lines through TorchStore's own logging setup."""
    store_name = _store_name()
    # Unset both, so the child's import installs TorchStore's handler at INFO.
    unset = ("HYPERACTOR_CODEC_MAX_FRAME_LENGTH", "TORCHSTORE_LOG_LEVEL")
    with subprocess.Popen(
        [sys.executable, "-c", _LOGGING_CHILD, _ROOT, store_name],
        env={k: v for k, v in os.environ.items() if k not in unset},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as child:
        try:
            stdout, stderr = child.communicate(timeout=180)
        except subprocess.TimeoutExpired:
            child.kill()
            stdout, stderr = child.communicate()
            pytest.fail(f"child timed out:\n{stdout}{stderr}")
    assert child.returncode == 0, stdout + stderr
    lines = [line for line in stdout.splitlines() if "routing control_plane=" in line]
    assert lines == [
        f"routing control_plane=Plane address=- store={store_name} rank={rank} "
        f"pid={child.pid}"
        for rank in ("-", "publisher/0")
    ]


_FORK_CHILD = """
import asyncio, os, sys
from dataclasses import dataclass
from types import SimpleNamespace

root, store_name = sys.argv[1:]
sys.path.insert(0, root)
import torchstore as ts
import torchstore.api as store_api

dials = []


class Connection(ts.ControlPlane):
    def __init__(self):
        self.pid = os.getpid()

    async def init(self, publishers, strategy):
        pass

    async def strategy(self):
        if os.getpid() != self.pid:
            raise RuntimeError("inherited across fork")
        return None

    async def register_layout(self, rank, key, registrations):
        pass

    async def get_layouts(self, key, *, timeout_s=None):
        return {}


@dataclass(frozen=True)
class Address(ts.ControlPlaneAddress):
    async def connect(self):
        dials.append(os.getpid())
        return Connection()

    def __str__(self):
        return "driver:1"


async def client(**kwargs):
    try:
        built = await ts.client(store_name, **kwargs)
    except Exception as error:
        return f"{type(error).__name__}: {error}"
    return "dialed here" if built._control_plane.pid == os.getpid() else "inherited"


store_api.current_rank = lambda: SimpleNamespace(rank=0)
publisher = ts.RankRole.PUBLISHER
print("parent", asyncio.run(client(role=publisher, address=Address(store_name))))
sys.stdout.flush()
if os.fork() == 0:
    print("child", asyncio.run(client()))
    print("child", asyncio.run(client(role=publisher)))
    print("child", asyncio.run(client(role=publisher, address=Address(store_name))))
    print("child dials", dials.count(os.getpid()))
    sys.stdout.flush()
    os._exit(0)
os.wait()
print("parent", asyncio.run(client()))
"""


@pytest.mark.timeout(240)
def test_a_forked_child_dials_its_own_connection() -> None:
    """Leave the parent's connection to the parent: a forked child dials its own."""
    store_name = _store_name()
    # A set variable keeps Monarch, and its threads, out of the child until used.
    env = {**os.environ, "HYPERACTOR_CODEC_MAX_FRAME_LENGTH": "910737418240"}
    with subprocess.Popen(
        [sys.executable, "-c", _FORK_CHILD, _ROOT, store_name],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as child:
        try:
            stdout, stderr = child.communicate(timeout=180)
        except subprocess.TimeoutExpired:
            child.kill()
            stdout, stderr = child.communicate()
            pytest.fail(f"child timed out:\n{stdout}{stderr}")
    assert child.returncode == 0, stdout + stderr
    inherited = (
        f"RuntimeError: routing store '{store_name}' on driver:1: its control plane "
        "was inherited across fork. Connect again from the address (address=)"
    )
    assert stdout.splitlines() == [
        "parent dialed here",
        f"child {inherited}",
        f"child {inherited}",
        "child dialed here",
        "child dials 1",
        "parent dialed here",
    ], stderr


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


store_api.current_rank = lambda: SimpleNamespace(rank=0)
address = Address(store_name)
asyncio.run(ts.client(store_name, role=ts.RankRole.PUBLISHER, address=address))
held, release = threading.Event(), threading.Event()


def hold():
    with store_api._selection_lock:
        held.set()
        release.wait()


threading.Thread(target=hold, daemon=True).start()
held.wait()
if os.fork() == 0:
    import gc, signal

    signal.alarm(60)  # ends a child stuck on the lock it inherited
    gc.collect()
    asyncio.run(ts.client(store_name, role=ts.RankRole.PUBLISHER, address=address))
    print("the child built a client", flush=True)
    os._exit(0)
release.set()
os.wait()
"""


@pytest.mark.timeout(240)
def test_a_child_forked_while_a_parent_thread_holds_the_lock_dials_its_own() -> None:
    """Give a forked child a fresh selection lock and keep its inherited connection alive.

    Another parent thread holds the lock across the fork, and no finalizer of the parent's
    connection runs in the child.
    """
    # A set variable keeps Monarch, and its threads, out of the child until used.
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
