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
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Iterator,
    MutableMapping,
    MutableSet,
)
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NoReturn
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import torch
import torchstore as ts
import torchstore.api as store_api
import torchstore.backend as backend_module
import torchstore.monarch.backend as monarch_backend
from monarch.actor import ActorError
from torchstore import RankRole
from torchstore.backend import ServiceBackend
from torchstore.carrier import CarrierAddress, PeerLostError
from torchstore.client import ClientType, LocalClient
from torchstore.monarch.coordinator import MonarchControlPlane, RoutingCoordinator
from torchstore.monarch.storage_volume import StorageVolume
from torchstore.routing import service as service_module, tcpstore
from torchstore.routing._model import KeyRegistration
from torchstore.routing.client import RoutingClient
from torchstore.routing.control_plane import ControlPlane
from torchstore.routing.service import (
    CoordinatorService,
    ServiceAddress,
    ServiceControlPlane,
)
from torchstore.transport import TransportType
from torchstore.transport.types import TensorSlice

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
def _fresh_process_state(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Start every test with no cached client, recorded or released store, owned service, forked store or close.

    Coordinator services and handles that a failed test left in this process's records are closed after it.
    """
    monkeypatch.setattr(store_api, "_local_clent_map", {})
    monkeypatch.setattr(store_api, "_control_plane_map", {})
    monkeypatch.setattr(store_api, "_released_stores", set())
    monkeypatch.setattr(store_api, "_owned_service_map", {})
    monkeypatch.setattr(store_api, "_forked_stores", {})
    monkeypatch.setattr(store_api, "_closing", set())
    yield
    left = [
        *store_api._owned_service_map.values(),
        *(
            selected.control_plane
            for selected in store_api._control_plane_map.values()
            if isinstance(selected.control_plane, ServiceControlPlane)
        ),
    ]
    if left:
        asyncio.run(_close_all(left))


async def _close_all(closeable: Iterable[Any], failing: bool = False) -> None:
    """Close each of these even after another's close fails and then raise the first error unless ``failing``."""
    results = await asyncio.gather(
        *(item.close() for item in closeable), return_exceptions=True
    )
    errors = [result for result in results if isinstance(result, BaseException)]
    if errors and not failing:
        raise errors[0]


async def _clean_up(*steps: Callable[[], Awaitable[Any]]) -> None:
    """Run each step in order from a ``finally`` block even after another fails.

    The first error propagates only when no error is already leaving that block. So the error of a failing body stays
    the one reported.
    """
    failing = sys.exc_info()[1] is not None
    first: BaseException | None = None
    for step in steps:
        try:
            await step()
        except BaseException as error:
            first = first or error
    if first is not None and not failing:
        raise first


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

    asyncio.run(ts.initialize(store_name="singleton"))

    backend.controller.assert_awaited_once_with("singleton")
    backend.controller_volume.assert_awaited_once_with(
        ts.ControllerStorageVolumes.get_volume_id
    )
    backend.spawn_volumes.assert_not_awaited()
    assert (
        controller.init.call.await_args.kwargs["storage_volumes"]
        is backend.controller_volume.return_value
    )


def test_standard_client_finds_the_stores_own_controller(backend, monkeypatch) -> None:
    """Look the store's controller up by its own name and build the client on it."""
    monkeypatch.setattr(store_api, "_local_clent_map", {})
    strategy = object()
    controller = backend.controller.return_value = _controller(strategy)

    client = asyncio.run(ts.client("named-store"))

    backend.controller.assert_awaited_once_with("named-store")
    assert client.strategy is strategy
    backend.directory.assert_called_once_with(controller)


def test_standard_shutdown_finds_the_stores_own_controller(backend) -> None:
    """Look the store's controller up by its own name and tear it down."""
    controller = backend.controller.return_value = _controller()

    asyncio.run(ts.shutdown("named-store"))

    backend.controller.assert_awaited_once_with("named-store")
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
) -> ts.ControlPlaneAddress | None:
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
    """Equal to other such handles on its registry, as a dataclass control plane is.

    Comparing it with anything but a control plane raises, as a strict ``__eq__`` may.
    """

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ControlPlane):
            raise TypeError(f"cannot compare a control plane with {other!r}")
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


class _ClosingEqualRegistry(_EqualRegistry):
    """An ``_EqualRegistry`` that counts its closes, as a dialed connection closes."""

    closes = 0

    async def close(self) -> None:
        self.closes += 1


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
        "control_plane must be 'monarch', 'service' or a ControlPlane instance, "
        f"got {control_plane!r}"
    )
    with pytest.raises(error, match=re.escape(message)):
        asyncio.run(
            ts.initialize(
                client_type=client_type,
                store_name=_store_name(),
                control_plane=control_plane,
            )
        )
    actors.spawn.assert_not_called()
    actors.lookup.assert_not_called()


@pytest.mark.parametrize("kind", ["instance", "service"])
def test_standard_initialize_takes_only_monarch(monkeypatch, kind) -> None:
    """Reject a caller's control plane or the service for a standard store, spawning nothing."""
    actors = _patch_actors(monkeypatch)
    start = AsyncMock(side_effect=AssertionError("started a coordinator service"))
    monkeypatch.setattr(CoordinatorService, "start", start)
    control_plane = Mock(spec=ControlPlane)

    with pytest.raises(ValueError, match="standard stores take only"):
        asyncio.run(
            ts.initialize(
                store_name=_store_name(),
                control_plane=control_plane if kind == "instance" else kind,
            )
        )
    actors.spawn.assert_not_called()
    actors.lookup.assert_not_called()
    start.assert_not_called()
    assert control_plane.mock_calls == []
    assert store_api._owned_service_map == {}


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


@pytest.mark.timeout(60)
def test_a_client_given_an_equal_instance_records_that_instance(monkeypatch) -> None:
    """Record the instance a client is given, though an equal one is recorded, and keep using it."""
    actors = _patch_actors(monkeypatch)
    store_name = _store_name()

    async def run() -> None:
        async with registry_store() as connect:
            base = await connect()
            old, new = (_EqualRegistry(base._registry, base._loop) for _ in range(2))
            await _initialize(store_name, old)
            client = await ts.client(
                store_name, role=RankRole.PUBLISHER, control_plane=new
            )
            assert client._control_plane is new
            assert store_api._control_plane_map[store_name].control_plane is new
            assert (
                await ts.client(store_name, role=RankRole.PUBLISHER, control_plane=new)
                is client
            )
            ts.reset_client(store_name)
            again = await ts.client(store_name, role=RankRole.PUBLISHER)
            assert again._control_plane is new

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
            # Neither would build a client, yet a cached one comes back for either.
            assert await ts.client(store_name, role=RankRole.REQUESTER) is client
            assert (
                await ts.client(store_name, role=RankRole.PUBLISHER, group=1) is client
            )

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
    during_close: list[tuple[bool, bool]] = []

    def close() -> None:
        during_close.append(
            (
                store_name in store_api._control_plane_map,
                store_name in store_api._local_clent_map,
            )
        )
        raise ConnectionResetError("gone")

    async def connect(address: _RegistryAddress) -> ControlPlane:
        connection = await _HOSTS[address.host]()
        connection.close = AsyncMock(side_effect=close)
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
            assert during_close == [(False, False)]
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
                assert await ts.initialize(store_name=store_name) is None
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


@pytest.fixture
def ended_close_wait():
    """Start close waits whose closes each end on a new loop that then closes.

    Teardown closes each wait and its loop on every outcome.
    """
    with contextlib.ExitStack() as cleanup:

        def start(selected: store_api._SelectedControlPlane) -> Any:
            loop = asyncio.new_event_loop()
            cleanup.callback(loop.close)
            waiting = store_api._wait_for_close(selected)
            cleanup.callback(waiting.close)

            async def end_the_close() -> None:
                waiting.send(None)
                # The close runs and ends, but nothing resumes the waiting coroutine.
                await asyncio.wait(store_api._closing)

            loop.run_until_complete(end_the_close())
            loop.close()
            return waiting

        yield start


@pytest.mark.parametrize(
    "ending, logged",
    [
        (None, False),
        (asyncio.CancelledError(), False),
        (ConnectionResetError("peer gone"), True),
    ],
    ids=["done", "cancelled", "failed"],
)
def test_a_close_wait_closed_after_its_close_and_loop_ended_logs_only_a_failure(
    ended_close_wait, caplog, ending, logged
) -> None:
    """Let a waiting coroutine closed after its close and loop ended log a failed close and only that."""
    caplog.set_level(logging.WARNING, logger=store_api.logger.name)
    address = _RegistryAddress("store", "host")
    close = AsyncMock(side_effect=ending)
    waiting = ended_close_wait(
        store_api._SelectedControlPlane(SimpleNamespace(close=close), address)
    )
    waiting.close()

    close.assert_awaited_once_with()
    message = f"closing the connection dialed through {address} failed"
    assert [
        (r.name, r.getMessage(), r.exc_info and r.exc_info[1])
        for r in caplog.records
        if r.levelno >= logging.WARNING
    ] == ([(store_api.logger.name, message, ending)] if logged else [])


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


def test_a_monarch_initialize_drops_a_client_cached_on_monarchs_coordinator(
    monkeypatch,
) -> None:
    """Drop a client on Monarch's coordinator, which no record holds, when initialize starts over."""
    actors = _patch_actors(monkeypatch)
    actors.lookup.side_effect = None
    actors.lookup.return_value = _mock_coordinator()
    store_name = _store_name()

    async def run() -> None:
        await _initialize(store_name)
        client = await ts.client(store_name, role=RankRole.PUBLISHER)
        assert store_api._local_clent_map == {store_name: client}
        assert store_api._control_plane_map == {}
        await _initialize(store_name)
        assert store_api._local_clent_map == {}

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
    """Drop only a recorded store's record, leaving an SPMD session of its name."""
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
@pytest.mark.parametrize("caller", ["initialize", "client"])
def test_a_caller_cancelled_while_a_replaced_connection_closes_keeps_its_cancel(
    monkeypatch, caplog, caller
) -> None:
    """Keep a cancelled call's ``CancelledError`` while it waits for a dialed close it replaced.

    The call's new record, its client from ``client`` and its selection line all stay.
    """
    caplog.set_level(logging.INFO)
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
            replacement = await _HOSTS[other]()
            caplog.clear()
            if caller == "initialize":
                task = asyncio.ensure_future(_initialize(store_name, replacement))
            else:
                ts.reset_client(store_name)
                await (await _HOSTS[other]()).init({"publisher/0"}, _strategy())
                task = asyncio.ensure_future(
                    ts.client(
                        store_name, role=RankRole.PUBLISHER, control_plane=replacement
                    )
                )
            await entered.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert store_api._control_plane_map == {store_name: (replacement, None)}
            cached = store_api._local_clent_map.get(store_name)
            if caller == "initialize":
                assert cached is None
            else:
                assert cached._control_plane is replacement
            assert len(_selection_lines(caplog)) == 1

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
    [winner] = [
        name
        for name, result in zip(stores, results)
        if isinstance(result, RoutingClient)
    ]
    assert list(store_api._control_plane_map) == [winner]


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
    """A map that notes, at each item set, ``del`` or ``pop``, whether the selection lock is held.

    These are the only changes the API module makes to its maps.
    """

    def __init__(self, name: str, lock: _RecordingLock, notes: list) -> None:
        super().__init__()
        self._name, self._lock, self._notes = name, lock, notes

    def __setitem__(self, key: Any, value: Any) -> None:
        self._notes.append((f"{self._name} set", self._lock.held))
        super().__setitem__(key, value)

    def __delitem__(self, key: Any) -> None:
        self._notes.append((f"{self._name} del", self._lock.held))
        super().__delitem__(key)

    def pop(self, *args: Any) -> Any:
        self._notes.append((f"{self._name} pop", self._lock.held))
        return super().pop(*args)


class _CheckedSet(set):
    """A set of store marks that notes, at each ``add`` or ``discard``, whether the lock is held.

    These are the only changes the API module makes to its sets.
    """

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
    clients = _CheckedDict("_local_clent_map", lock, notes)
    monkeypatch.setattr(store_api, "_local_clent_map", clients)
    for name in ("_check_one_store", "_record_client", "_select_control_plane"):

        def checked(
            *args: Any, _name: str = name, _original: Any = getattr(store_api, name)
        ) -> Any:
            notes.append((_name, lock.held))
            return _original(*args)

        monkeypatch.setattr(store_api, name, checked)
    closes: list[bool] = []
    by_client, by_initialize, by_shutdown, standard, failing = (
        _store_name() for _ in range(5)
    )

    async def run() -> None:
        async with _hosted_registry() as host:
            await (await _HOSTS[host]()).init({"publisher/0"}, _strategy())

            async def connect(address: _RegistryAddress) -> ControlPlane:
                connection = await _HOSTS[host]()
                connection.close = AsyncMock(
                    side_effect=lambda: closes.append(lock.held)
                )
                if address.store_name == failing:
                    connection.strategy = AsyncMock(side_effect=RuntimeError("no plan"))
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
                # A dial whose strategy fails is closed.
                with pytest.raises(RuntimeError, match="no plan"):
                    await ts.client(
                        failing,
                        role=RankRole.PUBLISHER,
                        address=_RegistryAddress(failing, host),
                    )
            # A client's plane, an initialize's plane and a shutdown each replace a dial.
            # The test's own reset_client, which takes no lock, holds it here.
            with lock:
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
        "_local_clent_map set",
        "_local_clent_map pop",
    }
    assert closes == [False, False, False, False]


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
# below must still finish and leave both locks free in the parent and the child.
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
        assert not first_lock.locked(), "the child kept the first import's lock"
        grandchild = os.fork()
        if grandchild == 0:
            os._exit(0)
        assert os.waitstatus_to_exitcode(os.waitpid(grandchild, 0)[1]) == 0
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
    """Hold the selection lock across fork, also after a reload, then free both in the child.

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


_LOCALHOST = "127.0.0.1"
# The one tensor slice in each layout that service cases register.
_LAYOUT = TensorSlice(
    offsets=(0,), coordinates=(), global_shape=(4,), local_shape=(4,), mesh_shape=()
)


@pytest.fixture
def service_strategy(monkeypatch: pytest.MonkeyPatch) -> ts.TorchStoreStrategy:
    """A strategy the coordinator service can pickle, served on localhost.

    Volume setup is stubbed on the class, since ``_patch_actors`` spawns no volume.
    """
    monkeypatch.setattr(ts.TorchStoreStrategy, "set_storage_volumes", AsyncMock())
    monkeypatch.setenv("TORCHSTORE_TCPSTORE_HOST", _LOCALHOST)
    return ts.TorchStoreStrategy(TransportType.MonarchRPC)


@pytest.fixture
def started_services(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[list[CoordinatorService]]:
    """Record each coordinator service the test starts, and stop every one after the test however it ends."""
    started: list[CoordinatorService] = []
    start, close = CoordinatorService.start, CoordinatorService.close

    async def recording_start(*args: Any, **kwargs: Any) -> CoordinatorService:
        started.append(await start(*args, **kwargs))
        return started[-1]

    async def stop_all() -> list[Any]:
        # The class's own close, which a test may patch. A second close returns at once.
        return await asyncio.gather(
            *(close(service) for service in started), return_exceptions=True
        )

    monkeypatch.setattr(CoordinatorService, "start", recording_start)
    yield started
    errors = [r for r in asyncio.run(stop_all()) if isinstance(r, BaseException)]
    assert not errors, errors


async def _release_all(store_names: Iterable[str], failing: bool) -> None:
    """Release each store, then raise the first error unless ``failing``."""
    first: BaseException | None = None
    for store_name in store_names:
        try:
            await _release(store_name)
        except BaseException as error:
            first = first or error
    if first is not None and not failing:
        raise first


async def _until_set(event: asyncio.Event, call: asyncio.Future[Any]) -> None:
    """Wait for ``event``, failing at once if ``call`` ends first."""
    waiting = asyncio.ensure_future(event.wait())
    try:
        await asyncio.wait({waiting, call}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        waiting.cancel()
    assert event.is_set(), call


async def _initialize_service(
    store_name: str, strategy: ts.TorchStoreStrategy, *, publishers: int = 1
) -> ServiceAddress:
    """Initialize a routing store on the coordinator service and return its address."""
    address = await _initialize(
        store_name, "service", publishers=publishers, strategy=strategy
    )
    assert isinstance(address, ServiceAddress)
    return address


def _layout() -> dict[str, KeyRegistration]:
    return {"w": KeyRegistration(_LAYOUT, dtype=torch.float32)}


def _service_handle(store_name: str) -> ServiceControlPlane:
    """The coordinator handle that this process's record holds for a store."""
    handle = store_api._control_plane_map[store_name].control_plane
    assert isinstance(handle, ServiceControlPlane)
    return handle


async def _release(store_name: str) -> None:
    """Shut a store down if this process still records or owns anything for it.

    A deadlocked test leaves the selection lock held, so this fails rather than wait on it.
    """
    if not store_api._selection_lock.acquire(timeout=10):
        pytest.fail("the selection lock is still held")
    store_api._selection_lock.release()
    if (
        store_name in store_api._control_plane_map
        or store_name in store_api._owned_service_map
    ):
        await ts.shutdown(store_name)


def _controller_stub() -> SimpleNamespace:
    """Answers the Monarch coordinator's and a standard controller's calls."""
    return SimpleNamespace(
        init=SimpleNamespace(call=AsyncMock(), call_one=AsyncMock()),
        strategy=SimpleNamespace(call_one=AsyncMock(return_value=object())),
        get_controller_strategy=SimpleNamespace(
            call_one=AsyncMock(return_value=object())
        ),
        teardown=SimpleNamespace(call=AsyncMock()),
    )


@contextlib.asynccontextmanager
async def _other_service(
    store_name: str, strategy: ts.TorchStoreStrategy
) -> AsyncIterator[CoordinatorService]:
    """Serve another initialized coordinator for a store, outside this process's records."""
    service = await CoordinatorService.start(store_name, host=_LOCALHOST)
    driver = None
    try:
        driver = await service.connect()
        await driver.init(publishers={"publisher/0"}, strategy=strategy)
        yield service
    finally:
        closing = [] if driver is None else [driver.close]
        await _clean_up(*closing, service.close)


class _Dials:
    """Records each connection dialed through a ``ServiceAddress`` and counts its closes.

    While ``holding`` is set, a close waits for ``release`` before it runs.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.dialed: list[ServiceControlPlane] = []
        self.closes: dict[ServiceControlPlane, int] = {}
        self.held: list[ServiceControlPlane] = []
        self.holding = False
        self.release = asyncio.Event()
        self._held_changed = asyncio.Condition()
        connect = ServiceAddress.connect

        async def dial(address: ServiceAddress) -> ServiceControlPlane:
            handle = await connect(address)
            self.dialed.append(handle)
            self.count_closes(handle)
            return handle

        monkeypatch.setattr(ServiceAddress, "connect", dial)

    def count_closes(self, handle: ServiceControlPlane) -> None:
        close = handle.close
        self.closes[handle] = 0

        async def counted() -> None:
            self.closes[handle] += 1
            if self.holding:
                async with self._held_changed:
                    self.held.append(handle)
                    self._held_changed.notify_all()
                await self.release.wait()
            await close()

        handle.close = counted

    async def until_held(self, count: int) -> None:
        """Return once ``count`` closes wait for ``release``."""
        async with self._held_changed:
            await asyncio.wait_for(
                self._held_changed.wait_for(lambda: len(self.held) >= count), 30
            )


# What calls through a stopped service's address raise. Each names its host:port.
_STOPPED: tuple[type[Exception], ...] = (ConnectionError, TimeoutError, RuntimeError)


async def _fails_naming(
    call: Callable[[], Awaitable[Any]], address: ServiceAddress
) -> BaseException:
    """Assert that ``call`` fails naming ``address`` within the 2 s TCPStore deadline plus slack."""
    start = time.monotonic()
    with pytest.raises(_STOPPED) as raised:
        await call()
    assert time.monotonic() - start < 7
    assert str(address) in str(raised.value), raised.value
    return raised.value


async def _until(condition: Callable[[], object], call: asyncio.Future[Any]) -> None:
    """Poll ``condition`` until a deadline, failing at once if ``call`` ends first."""
    deadline = time.monotonic() + 30
    while not condition():
        assert time.monotonic() < deadline, "the condition did not hold in time"
        done, _ = await asyncio.wait({call}, timeout=0.01)
        assert not done, call


async def _connect_and_read(address: ServiceAddress) -> None:
    handle = await address.connect()
    try:
        await handle.strategy()
    finally:
        await _clean_up(handle.close)


@pytest.mark.timeout(60)
def test_service_initialize_serves_the_coordinator_without_an_actor(
    monkeypatch, caplog, service_strategy
) -> None:
    """Spawn the volumes once, serve the coordinator here, record it and return its address."""
    caplog.set_level(logging.INFO)
    actors = _patch_actors(monkeypatch)
    store_name = _store_name()

    async def run() -> ServiceAddress:
        address = await _initialize_service(store_name, service_strategy, publishers=2)
        try:
            assert type(address) is ServiceAddress
            assert (address.store_name, address.host) == (store_name, _LOCALHOST)
            selected = store_api._control_plane_map[store_name]
            assert type(selected.control_plane) is ServiceControlPlane
            assert selected.address == selected.control_plane.address == address
            assert store_api._owned_service_map[store_name].address == address
            # Ranks read the pickled strategy and wait on both publishers.
            strategy = await selected.control_plane.strategy()
            assert type(strategy) is ts.TorchStoreStrategy
            assert strategy.default_transport_type == TransportType.MonarchRPC
            missing = r"missing publishers: \['publisher/0', 'publisher/1'\]"
            with pytest.raises(TimeoutError, match=missing):
                await selected.control_plane.get_layouts("model", timeout_s=0)
            return address
        finally:
            await _clean_up(lambda: _release(store_name))

    address = asyncio.run(run())
    actors.spawn.assert_awaited_once()
    assert actors.spawn.await_args.args[0] == 1
    set_storage_volumes = ts.TorchStoreStrategy.set_storage_volumes
    assert isinstance(set_storage_volumes, AsyncMock)
    set_storage_volumes.assert_awaited_once_with(actors.spawn.return_value)
    actors.lookup.assert_not_called()
    pid = str(os.getpid())
    assert _selection_lines(caplog) == [("service", str(address), store_name, "-", pid)]


@pytest.mark.timeout(60)
@pytest.mark.parametrize("host", ["localhost", None], ids=["variable", "hostname"])
def test_service_initialize_advertises_the_host_ranks_dial(
    monkeypatch, service_strategy, host
) -> None:
    """Advertise ``$TORCHSTORE_TCPSTORE_HOST``, else this host's name, for TCPStore and the coordinator."""
    _patch_actors(monkeypatch)
    if host is None:
        monkeypatch.delenv("TORCHSTORE_TCPSTORE_HOST")
        host = socket.gethostname()
        try:
            socket.getaddrinfo(host, None)
        except OSError:
            pytest.skip("this host's name does not resolve")
    else:
        monkeypatch.setenv("TORCHSTORE_TCPSTORE_HOST", host)
    store_name = _store_name()

    async def run() -> None:
        address = await _initialize_service(store_name, service_strategy)
        try:
            rendezvous = _service_handle(store_name)._rendezvous
            assert rendezvous is not None
            published = CarrierAddress.from_json(await rendezvous.read_coordinator())
            assert (address.host, published.host) == (host, host)
            # A rank dials both servers there.
            handle = await address.connect()
            try:
                await handle.register_layout("publisher/0", "model", _layout())
                layouts = await handle.get_layouts("model", timeout_s=0)
                assert layouts == {"publisher/0": _layout()}
            finally:
                await _clean_up(handle.close)
        finally:
            await _clean_up(lambda: _release(store_name))

    asyncio.run(run())


@pytest.mark.timeout(60)
def test_overlapping_service_initializes_of_two_stores_both_succeed(
    monkeypatch, service_strategy
) -> None:
    """Start two stores' services at once: a store checks only a caller's own control plane."""
    _patch_actors(monkeypatch)
    start = CoordinatorService.start
    entered: list[str] = []
    both = asyncio.Event()

    async def held_start(
        store_name: str, *args: Any, **kwargs: Any
    ) -> CoordinatorService:
        # Each start waits for the other, so both initializes run their checks first.
        entered.append(store_name)
        if len(entered) == 2:
            both.set()
        await asyncio.wait_for(both.wait(), 30)
        return await start(store_name, *args, **kwargs)

    monkeypatch.setattr(CoordinatorService, "start", held_start)
    names = [_store_name(), _store_name()]

    async def run() -> None:
        try:
            results = await asyncio.gather(
                *(_initialize_service(name, service_strategy) for name in names),
                return_exceptions=True,
            )
            errors = [r for r in results if isinstance(r, BaseException)]
            assert not errors, errors
            assert sorted(store_api._owned_service_map) == sorted(names)
        finally:
            await _release_all(names, sys.exc_info()[1] is not None)

    asyncio.run(run())


@pytest.mark.timeout(60)
def test_calls_to_a_stopped_service_fail_naming_its_address(
    monkeypatch, service_strategy
) -> None:
    """Fail a waiting lookup when the grace ends, then calls through the old address.

    They fail at once, or new connects within the TCPStore deadline, each naming host:port.
    """
    _patch_actors(monkeypatch)
    monkeypatch.setattr(tcpstore, "_DEFAULT_TIMEOUT_S", 2)
    monkeypatch.setattr(service_module, "_CLOSE_GRACE_S", 1)
    store_name = _store_name()

    async def run() -> None:
        address = await _initialize_service(store_name, service_strategy)
        handle = _service_handle(store_name)
        server = store_api._owned_service_map[store_name]._server
        assert server is not None
        # A rank's handle, outside this process's records, that resolved the coordinator.
        rank = await address.connect()
        try:
            await rank.register_layout("publisher/0", "model", _layout())
            waiting = asyncio.ensure_future(
                rank.get_layouts("unpublished", timeout_s=60)
            )
            await _until(lambda: server.stats()["running"] == 1, waiting)
            start = time.monotonic()
            await ts.shutdown(store_name)
            assert store_api._control_plane_map == store_api._owned_service_map == {}
            stop_s = time.monotonic() - start
            coordinator = f"on {re.escape(str(address))}: coordinator call failed"
            with pytest.raises(PeerLostError, match=coordinator):
                await asyncio.wait_for(waiting, 5)
            assert 0.9 < stop_s < 1 + 2
            with pytest.raises(
                RuntimeError, match=f"{re.escape(str(address))}: .* is closed"
            ):
                await handle.get_layouts("model", timeout_s=0)
            lost = await _fails_naming(
                lambda: rank.get_layouts("model", timeout_s=0), address
            )
            assert isinstance(lost, PeerLostError)
            stale = await _fails_naming(rank.strategy, address)
            # At EOF on the rank's open TCPStore connection, not at the TCPStore deadline.
            assert isinstance(stale, ConnectionError), stale
            # Refused, timed out, or a port another process took after the stop.
            await _fails_naming(lambda: _connect_and_read(address), address)
        finally:
            await _clean_up(rank.close, lambda: _release(store_name))

    asyncio.run(run())


@pytest.mark.timeout(60)
def test_a_dialed_connection_replaced_by_an_equal_instance_closes(monkeypatch) -> None:
    """Close a dialed connection that an equal but different instance replaces."""
    actors = _patch_actors(monkeypatch)
    store_name = _store_name()
    dialed: list[_ClosingEqualRegistry] = []

    async def run() -> None:
        async with _hosted_registry() as host:
            base = await _HOSTS[host]()
            assert isinstance(base, HostedRegistry)
            await base.init(
                {"publisher/0"}, ts.TorchStoreStrategy(TransportType.MonarchRPC)
            )
            registry, loop = base._registry, base._loop

            async def connect(address: _RegistryAddress) -> ControlPlane:
                dialed.append(_ClosingEqualRegistry(registry, loop))
                return dialed[-1]

            with patch.object(_RegistryAddress, "connect", connect):
                await ts.client(
                    store_name,
                    role=RankRole.PUBLISHER,
                    address=_RegistryAddress(store_name, host),
                )
            [connection] = dialed
            equal = _EqualRegistry(registry, loop)
            assert equal == connection
            ts.reset_client(store_name)
            await ts.client(store_name, role=RankRole.PUBLISHER, control_plane=equal)
            assert store_api._control_plane_map[store_name] == (equal, None)
            assert connection.closes == 1

    asyncio.run(run())
    actors.lookup.assert_not_called()


@pytest.mark.timeout(60)
def test_a_second_shutdown_of_a_service_store_looks_up_no_controller(
    monkeypatch, service_strategy
) -> None:
    """Find the store released on a second shutdown, which drops only a client."""
    actors = _patch_actors(monkeypatch)
    store_name = _store_name()

    async def run() -> None:
        await _initialize_service(store_name, service_strategy)
        await ts.shutdown(store_name)
        await ts.shutdown(store_name)
        assert store_name in store_api._released_stores

    asyncio.run(run())
    actors.lookup.assert_not_called()


@pytest.mark.timeout(60)
def test_an_owned_service_stops_after_a_caller_replaced_its_record(
    monkeypatch, service_strategy
) -> None:
    """Stop the owned service on shutdown, though an instance holds the record."""
    _patch_actors(monkeypatch)
    store_name = _store_name()

    async def run() -> None:
        async with registry_store() as connect:
            await _initialize_service(store_name, service_strategy)
            try:
                service = store_api._owned_service_map[store_name]
                plane = await connect()
                await plane.init({"publisher/0"}, service_strategy)
                await ts.client(
                    store_name, role=RankRole.PUBLISHER, control_plane=plane
                )
                assert store_api._control_plane_map[store_name] == (plane, None)
                assert store_api._owned_service_map[store_name] is service
                await ts.shutdown(store_name)
                assert (service._server, service._rendezvous) == (None, None)
                assert store_api._owned_service_map == {}
            finally:
                await _clean_up(lambda: _release(store_name))

    asyncio.run(run())


@pytest.mark.timeout(60)
def test_initialize_given_a_coordinator_handle_returns_none_and_never_closes_it(
    monkeypatch, service_strategy
) -> None:
    """Record a caller's coordinator handle with no address and return None.

    A shutdown never closes the handle.
    """
    _patch_actors(monkeypatch)
    store_name = _store_name()

    async def run() -> None:
        service = await CoordinatorService.start(store_name, host=_LOCALHOST)
        closing = [service.close]
        try:
            handle = await service.connect()
            closing.insert(0, handle.close)
            address = await _initialize(store_name, handle, strategy=service_strategy)
            assert address is None
            assert store_api._control_plane_map[store_name] == (handle, None)
            await ts.shutdown(store_name)
            assert not handle._closed
            await handle.strategy()
        finally:
            await _clean_up(lambda: _release(store_name), *closing)

    asyncio.run(run())


@pytest.mark.timeout(60)
def test_shutdown_while_replaced_and_redundant_connections_close(
    monkeypatch, service_strategy
) -> None:
    """Shut down while a replaced handle and a redundant dial wait in their closes.

    No record, cached client or owned service remains, and each connection and the
    service close exactly once.
    """
    _patch_actors(monkeypatch)
    store_name = _store_name()
    dials = _Dials(monkeypatch)

    async def run() -> None:
        async with _other_service(store_name, service_strategy) as other:
            await _initialize_service(store_name, service_strategy)
            try:
                handle = _service_handle(store_name)
                dials.count_closes(handle)
                service = store_api._owned_service_map[store_name]
                service_closes = AsyncMock(side_effect=service.close)
                service.close = service_closes
                dials.holding = True
                # One racing call replaces the driver's record and the other adopts that dial.
                racing = [
                    asyncio.ensure_future(
                        ts.client(
                            store_name, role=RankRole.PUBLISHER, address=other.address
                        )
                    )
                    for _ in range(2)
                ]
                await dials.until_held(2)
                assert not any(call.done() for call in racing)
                recorded = _service_handle(store_name)
                [redundant] = [d for d in dials.dialed if d is not recorded]
                assert dials.held == [handle, redundant]
                assert store_api._control_plane_map[store_name] == (
                    recorded,
                    other.address,
                )
                dials.holding = False
                await ts.shutdown(store_name)
                state = (
                    store_api._control_plane_map,
                    store_api._local_clent_map,
                    store_api._owned_service_map,
                )
                assert state == ({}, {}, {})
                dials.release.set()
                clients = await asyncio.gather(*racing)
                assert [c._control_plane for c in clients] == [recorded, recorded]
                assert state == ({}, {}, {})
                assert dials.closes == {recorded: 1, redundant: 1, handle: 1}
                service_closes.assert_awaited_once_with()
            finally:
                dials.release.set()
                await _clean_up(lambda: _release(store_name))

    asyncio.run(run())


class _ShutdownOnRecord(dict):
    """A record map that starts ``ts.shutdown`` as a coordinator handle is recorded.

    The shutdown runs at the first await after the selection's synchronous step.
    """

    task: asyncio.Future[Any] | None = None

    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key, value)
        if self.task is None and isinstance(value.control_plane, ServiceControlPlane):
            self.task = asyncio.ensure_future(ts.shutdown(key))


@pytest.mark.timeout(60)
def test_a_shutdown_due_right_after_a_dialed_selection_leaves_nothing(
    monkeypatch, service_strategy
) -> None:
    """Leave no record or cached client, and close the dial, when shutdown follows its step."""
    _patch_actors(monkeypatch)
    store_name = _store_name()
    records = _ShutdownOnRecord()
    monkeypatch.setattr(store_api, "_control_plane_map", records)

    async def run() -> None:
        async with _other_service(store_name, service_strategy) as other:
            client = await ts.client(
                store_name, role=RankRole.PUBLISHER, address=other.address
            )
            assert records.task is not None
            await records.task
            assert store_api._control_plane_map == store_api._local_clent_map == {}
            with pytest.raises(RuntimeError, match="is closed"):
                await client._control_plane.strategy()

    asyncio.run(run())


@pytest.mark.timeout(60)
def test_a_shutdown_due_right_after_an_instance_client_step_leaves_nothing(
    monkeypatch, service_strategy
) -> None:
    """Leave no record or cached client when shutdown follows a step on a caller's instance."""
    _patch_actors(monkeypatch)
    store_name = _store_name()
    records = _ShutdownOnRecord()
    monkeypatch.setattr(store_api, "_control_plane_map", records)

    async def run() -> None:
        async with _other_service(store_name, service_strategy) as other:
            handle = await other.address.connect()
            try:
                await ts.client(
                    store_name, role=RankRole.PUBLISHER, control_plane=handle
                )
                assert records.task is not None
                await records.task
                assert store_api._control_plane_map == store_api._local_clent_map == {}
                # Shutdown never closes a caller's own control plane.
                await handle.strategy()
            finally:
                await _clean_up(handle.close)

    asyncio.run(run())


@pytest.mark.timeout(60)
def test_a_shutdown_due_right_after_a_reused_record_step_leaves_nothing(
    monkeypatch, service_strategy
) -> None:
    """Leave no record, client or owned service when shutdown follows a step on the record."""
    _patch_actors(monkeypatch)
    store_name = _store_name()

    async def run() -> None:
        await _initialize_service(store_name, service_strategy)
        handle = _service_handle(store_name)
        strategy, shutting_down = handle.strategy, []

        async def strategy_then_shutdown() -> ts.TorchStoreStrategy:
            result = await strategy()
            # The shutdown runs at the first await after this call's selection step.
            shutting_down.append(asyncio.ensure_future(ts.shutdown(store_name)))
            return result

        handle.strategy = strategy_then_shutdown
        client = await ts.client(store_name, role=RankRole.PUBLISHER)
        await shutting_down[0]
        assert store_api._control_plane_map == store_api._local_clent_map == {}
        assert store_api._owned_service_map == {}
        with pytest.raises(RuntimeError, match="is closed"):
            await client._control_plane.get_layouts("model", timeout_s=0)

    asyncio.run(run())


@pytest.mark.timeout(60)
def test_a_shutdown_due_right_after_an_adopting_step_leaves_nothing(
    monkeypatch, service_strategy
) -> None:
    """Leave no record or client when shutdown follows a dial that adopts the recorded one."""
    _patch_actors(monkeypatch)
    store_name = _store_name()
    dial = ServiceAddress.connect

    async def run() -> None:
        first_done = asyncio.Event()
        shutting_down: list[asyncio.Future[Any]] = []
        started: list[int] = []

        async def dial_in_order(address: ServiceAddress) -> ServiceControlPlane:
            index = len(started)
            started.append(index)
            handle = await dial(address)
            if index == 1:
                strategy = handle.strategy

                async def strategy_then_shutdown() -> ts.TorchStoreStrategy:
                    await first_done.wait()
                    result = await strategy()
                    shutting_down.append(asyncio.ensure_future(ts.shutdown(store_name)))
                    return result

                handle.strategy = strategy_then_shutdown
            return handle

        monkeypatch.setattr(ServiceAddress, "connect", dial_in_order)
        async with _other_service(store_name, service_strategy) as other:
            calls = [
                asyncio.ensure_future(
                    ts.client(
                        store_name, role=RankRole.PUBLISHER, address=other.address
                    )
                )
                for _ in range(2)
            ]
            first = await calls[0]
            first_done.set()
            second = await calls[1]
            assert second._control_plane is first._control_plane
            await shutting_down[0]
            assert store_api._control_plane_map == store_api._local_clent_map == {}

    asyncio.run(run())


@pytest.mark.timeout(60)
def test_a_shutdown_due_right_after_a_service_initialize_leaves_nothing(
    monkeypatch, service_strategy
) -> None:
    """Leave no record or owned service, and stop the service, when shutdown follows its step."""
    _patch_actors(monkeypatch)
    store_name = _store_name()
    records = _ShutdownOnRecord()
    monkeypatch.setattr(store_api, "_control_plane_map", records)

    async def run() -> None:
        await _initialize_service(store_name, service_strategy)
        service = store_api._owned_service_map[store_name]
        assert records.task is not None
        await records.task
        assert store_api._control_plane_map == store_api._owned_service_map == {}
        assert (service._server, service._rendezvous) == (None, None)

    asyncio.run(run())


class _StartOnPop(dict):
    """A map that starts ``start(store)`` when a store first leaves it or a sibling.

    The siblings share ``started``, which holds that call's task.
    """

    def __init__(
        self,
        start: Callable[[str], Awaitable[Any]],
        started: list[asyncio.Future[Any]],
    ) -> None:
        super().__init__()
        self.start, self.started = start, started

    def pop(self, key: str, *default: Any) -> Any:
        if not self.started and key in self:
            self.started.append(asyncio.ensure_future(self.start(key)))
        return super().pop(key, *default)


@pytest.mark.timeout(60)
def test_a_client_due_right_after_a_shutdown_step_keeps_its_record_and_cache(
    monkeypatch, service_strategy
) -> None:
    """Keep a record and client selected right after a shutdown's step, which still stops the service."""
    _patch_actors(monkeypatch)
    store_name = _store_name()
    plane = Mock(spec=ControlPlane)
    plane.strategy = AsyncMock(return_value=object())
    started: list[asyncio.Future[Any]] = []

    def select(key: str) -> Awaitable[LocalClient]:
        return ts.client(key, role=RankRole.PUBLISHER, control_plane=plane)

    # The call starts at whichever of the two the shutdown releases first.
    records = _StartOnPop(select, started)
    monkeypatch.setattr(store_api, "_control_plane_map", records)
    monkeypatch.setattr(store_api, "_owned_service_map", _StartOnPop(select, started))

    async def run() -> None:
        await _initialize_service(store_name, service_strategy)
        service = store_api._owned_service_map[store_name]
        await ts.shutdown(store_name)
        [call] = started
        client = await call
        assert dict(records) == {store_name: (plane, None)}
        assert store_api._local_clent_map == {store_name: client}
        assert store_api._owned_service_map == {}
        assert (service._server, service._rendezvous) == (None, None)

    asyncio.run(run())


class _ShutdownOnRead(dict):
    """A client cache that starts ``ts.shutdown`` at its first read once armed."""

    armed = False
    task: asyncio.Future[Any] | None = None

    def get(self, key: str, *default: Any) -> Any:
        if self.armed and self.task is None:
            self.task = asyncio.ensure_future(ts.shutdown(key))
        return super().get(key, *default)


@pytest.mark.timeout(60)
def test_a_shutdown_due_right_after_a_cached_read_returns_the_cached_client(
    monkeypatch, service_strategy
) -> None:
    """Return the cached client when a shutdown is due right after the cached read."""
    _patch_actors(monkeypatch)
    store_name = _store_name()
    clients = _ShutdownOnRead()
    monkeypatch.setattr(store_api, "_local_clent_map", clients)

    async def run() -> None:
        address = await _initialize_service(store_name, service_strategy)
        first = await ts.client(store_name, role=RankRole.PUBLISHER, address=address)
        clients.armed = True
        again = await ts.client(store_name, role=RankRole.PUBLISHER, address=address)
        assert again is first
        assert clients.task is not None
        await clients.task
        assert store_api._control_plane_map == clients == {}
        assert store_api._owned_service_map == {}

    asyncio.run(run())


@pytest.mark.timeout(60)
def test_dialed_connections_close_when_replaced_or_redundant(
    monkeypatch, service_strategy
) -> None:
    """Share one connection among racing first calls, and close a record another replaces."""
    _patch_actors(monkeypatch)
    store_name = _store_name()
    dials = _Dials(monkeypatch)

    async def run() -> None:
        async with _other_service(
            store_name, service_strategy
        ) as first, _other_service(store_name, service_strategy) as second:
            try:
                # Each call gets its own copy of the address, as Monarch endpoints do.
                clients = await asyncio.gather(
                    *(
                        ts.client(
                            store_name,
                            role=RankRole.PUBLISHER,
                            address=pickle.loads(pickle.dumps(first.address)),
                        )
                        for _ in range(3)
                    )
                )
                recorded = _service_handle(store_name)
                assert store_api._control_plane_map[store_name].address == first.address
                assert len(dials.dialed) == 3
                assert all(c._control_plane is recorded for c in clients)
                assert dials.closes == {
                    handle: int(handle is not recorded) for handle in dials.dialed
                }
                for client in clients:
                    await client._control_plane.register_layout(
                        "publisher/0", "model", _layout()
                    )
                ts.reset_client(store_name)
                await ts.client(
                    store_name, role=RankRole.PUBLISHER, address=second.address
                )
                assert dials.closes[recorded] == 1
                with pytest.raises(RuntimeError, match="is closed"):
                    await recorded.strategy()
                connection = _service_handle(store_name)
                assert store_api._control_plane_map[store_name] == (
                    connection,
                    second.address,
                )
                # The dialed connection passed back in keeps its record, open.
                ts.reset_client(store_name)
                await ts.client(
                    store_name, role=RankRole.PUBLISHER, control_plane=connection
                )
                assert store_api._control_plane_map[store_name] == (
                    connection,
                    second.address,
                )
                assert dials.closes[connection] == 0
                await connection.strategy()
            finally:
                await _clean_up(
                    lambda: _release(store_name),
                    lambda: _close_all(dials.dialed, failing=True),
                )

    asyncio.run(run())


@dataclass(frozen=True)
class _IncarnationAddress(_RegistryAddress):
    """Test-only: prints as its host whatever its incarnation, as a service's nonce stays out of ``str``."""

    incarnation: int = 0


@pytest.mark.timeout(60)
def test_a_dial_through_an_address_that_only_prints_like_the_record_replaces_it(
    monkeypatch,
) -> None:
    """Record and keep a dial whose address prints like the recorded one but differs, closing the old."""
    actors = _patch_actors(monkeypatch)
    store_name = _store_name()
    dialed: list[ControlPlane] = []

    async def connect(address: _RegistryAddress) -> ControlPlane:
        connection = await _HOSTS[address.host]()
        connection.close = AsyncMock()
        dialed.append(connection)
        return connection

    async def run() -> None:
        async with _hosted_registry() as host:
            await (await _HOSTS[host]()).init({"publisher/0"}, _strategy())
            old, new = (_IncarnationAddress(store_name, host, n) for n in (0, 1))
            assert str(old) == str(new) and old != new
            with patch.object(_RegistryAddress, "connect", connect):
                await ts.client(store_name, role=RankRole.PUBLISHER, address=old)
                ts.reset_client(store_name)
                client = await ts.client(
                    store_name, role=RankRole.PUBLISHER, address=new
                )
            first, second = dialed
            assert client._control_plane is second
            assert store_api._control_plane_map == {store_name: (second, new)}
            first.close.assert_awaited_once_with()
            second.close.assert_not_awaited()

    asyncio.run(run())
    actors.lookup.assert_not_called()


class _Recording:
    """Records each access's caller and whether the selection lock was held.

    The caller is the first frame outside the container's own methods and the ``collections.abc`` mixins, and for
    ``reset_client``'s pop the function that called ``reset_client``.
    """

    def __init__(self, name: str, calls: list[tuple[str, str, str, bool]]) -> None:
        self.name, self.calls = name, calls

    def _record(self, op: str) -> None:
        frame = sys._getframe(1)
        while frame.f_back is not None and (
            frame.f_locals.get("self") is self
            or frame.f_globals.get("__name__") == "_collections_abc"
        ):
            frame = frame.f_back
        if frame.f_code.co_name == "reset_client" and frame.f_back is not None:
            frame = frame.f_back
        locked = store_api._selection_lock.locked()
        self.calls.append((self.name, op, frame.f_code.co_name, locked))


class _LockChecked(_Recording, MutableMapping[str, Any]):
    """A map whose every read, write, membership check, size and iteration runs a method below or a mixin on them."""

    def __init__(self, name: str, calls: list[tuple[str, str, str, bool]]) -> None:
        super().__init__(name, calls)
        self.data: dict[str, Any] = {}

    def __getitem__(self, key: str) -> Any:
        self._record("getitem")
        return self.data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._record("set")
        self.data[key] = value

    def __delitem__(self, key: str) -> None:
        self._record("del")
        del self.data[key]

    def __iter__(self) -> Iterator[str]:
        self._record("iter")
        return iter(self.data)

    def __len__(self) -> int:
        self._record("len")
        return len(self.data)

    def __contains__(self, key: object) -> bool:
        self._record("in")
        return key in self.data

    def get(self, key: str, default: Any = None) -> Any:
        self._record("get")
        return self.data.get(key, default)


class _LockCheckedSet(_Recording, MutableSet[str]):
    """A set whose every access runs a method below or a mixin on them."""

    def __init__(self, name: str, calls: list[tuple[str, str, str, bool]]) -> None:
        super().__init__(name, calls)
        self.data: set[str] = set()

    def __contains__(self, x: object) -> bool:
        self._record("in")
        return x in self.data

    def __iter__(self) -> Iterator[str]:
        self._record("iter")
        return iter(self.data)

    def __len__(self) -> int:
        self._record("len")
        return len(self.data)

    def add(self, value: str) -> None:
        self._record("add")
        self.data.add(value)

    def discard(self, value: str) -> None:
        self._record("discard")
        self.data.discard(value)


@pytest.mark.timeout(60)
def test_every_selection_step_holds_the_selection_lock(
    monkeypatch, service_strategy
) -> None:
    """Hold the lock wherever a selection step this case drives changes, reads or checks the selection maps or released stores.

    A client's reads before it builds or looks up a controller hold it too. So none sees a step half done. The lock's
    comment lets a later shutdown's read of the released stores and ``reset_client``'s own drop of a client take none.
    """
    actors = _patch_actors(monkeypatch)
    actors.lookup.side_effect = None
    actors.lookup.return_value = _controller_stub()
    calls: list[tuple[str, str, str, bool]] = []
    for attribute, name in (
        ("_control_plane_map", "records"),
        ("_owned_service_map", "services"),
        ("_local_clent_map", "clients"),
        ("_forked_stores", "forked"),
    ):
        monkeypatch.setattr(store_api, attribute, _LockChecked(name, calls))
    monkeypatch.setattr(
        store_api, "_released_stores", _LockCheckedSet("released", calls)
    )
    service, monarch, standard, initialized = (_store_name() for _ in range(4))
    dialed, given = _store_name(), _store_name()
    plane = Mock(spec=ControlPlane)
    plane.strategy = AsyncMock(return_value=object())

    async def run() -> None:
        await _initialize_service(service, service_strategy)
        await ts.client(service, role=RankRole.PUBLISHER)
        await ts.client(service, role=RankRole.PUBLISHER)
        await ts.client(monarch, role=RankRole.PUBLISHER)
        await ts.client(standard)
        # A standard store has no address for its ranks.
        assert await ts.initialize(store_name=initialized) is None
        await ts.shutdown(service)
        # The client step's other branches: a dial through an address, a caller's instance.
        async with _other_service(dialed, service_strategy) as other:
            await ts.client(dialed, role=RankRole.PUBLISHER, address=other.address)
            await ts.shutdown(dialed)
        await ts.client(given, role=RankRole.PUBLISHER, control_plane=plane)
        await ts.shutdown(given)

    asyncio.run(run())
    assert {call[:3] for call in calls if not call[3]} == set()
    assert {
        ("records", "get", "_routing_client"),
        ("records", "get", "_check_cached_client"),
        ("records", "in", "_check_unrecorded"),
        ("clients", "getitem", "_shutdown_selected"),
        ("forked", "get", "_check_not_forked"),
        ("forked", "getitem", "_select_control_plane"),
        ("forked", "getitem", "_shutdown_selected"),
        ("released", "discard", "initialize"),
        ("released", "add", "_shutdown_selected"),
        ("released", "in", "_check_unrecorded"),
    } <= {call[:3] for call in calls if call[3]}


class _CountingLock:
    """The selection lock, counting its acquisitions, which tells one locked step from two.

    ``changes`` counts only the steps that change the selection of the store named ``store``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.count = 0
        self.changes = 0
        self.store = ""
        self._before: tuple[object, ...] = ()

    def _selection(self) -> tuple[object, ...]:
        """The store's record, owned service, cached client, released mark and fork marker."""
        return (
            store_api._control_plane_map.get(self.store),
            store_api._owned_service_map.get(self.store),
            store_api._local_clent_map.get(self.store),
            self.store in store_api._released_stores,
            store_api._forked_stores.get(self.store),
        )

    def acquire(self, *args: Any, **kwargs: Any) -> bool:
        acquired = self._lock.acquire(*args, **kwargs)
        if acquired:
            self.count += 1
            self._before = self._selection()
        return acquired

    def release(self) -> None:
        after = self._selection()
        if any(old is not new for old, new in zip(self._before, after)):
            self.changes += 1
        self._lock.release()

    def locked(self) -> bool:
        return self._lock.locked()

    def __enter__(self) -> bool:
        return self.acquire()

    def __exit__(self, *exc: object) -> None:
        self.release()

    async def taken_by(self, call: Awaitable[Any]) -> int:
        """Return how many times ``call`` takes the lock."""
        before = self.count
        await call
        return self.count - before

    async def changes_by(self, call: Awaitable[Any]) -> int:
        """Return how many of ``call``'s locked steps change the store's selection."""
        before = self.changes
        await call
        return self.changes - before


@pytest.mark.timeout(60)
def test_each_initialize_and_shutdown_takes_the_selection_lock_once(
    monkeypatch, service_strategy
) -> None:
    """Swap or release a store's record, owned service and client in one locked step.

    The lock's other step, the check of a caller's own plane, changes none of it.
    """
    actors = _patch_actors(monkeypatch)
    actors.lookup.side_effect = None
    actors.lookup.return_value = _controller_stub()
    lock = _CountingLock()
    monkeypatch.setattr(store_api, "_selection_lock", lock)
    store_name = _store_name()
    lock.store = store_name

    async def run() -> list[int]:
        async with registry_store() as connect:
            plane = await connect()
            # Service, then a caller's instance and service again, each after a shutdown.
            calls: list[Callable[[], Awaitable[Any]]] = [
                lambda: _initialize_service(store_name, service_strategy),
                lambda: ts.shutdown(store_name),
                lambda: _initialize(store_name, plane),
                lambda: ts.shutdown(store_name),
                lambda: _initialize_service(store_name, service_strategy),
                lambda: ts.shutdown(store_name),
            ]
            try:
                return [await lock.changes_by(call()) for call in calls]
            finally:
                await _clean_up(lambda: _release(store_name))

    assert asyncio.run(run()) == [1] * 6


@pytest.mark.timeout(60)
def test_each_client_takes_the_selection_lock_once_per_step(
    monkeypatch, service_strategy
) -> None:
    """Take the lock once per step: twice for a standard client and three times for a routing one."""
    actors = _patch_actors(monkeypatch)
    actors.lookup.side_effect = None
    actors.lookup.return_value = _controller_stub()
    lock = _CountingLock()
    monkeypatch.setattr(store_api, "_selection_lock", lock)
    standard, monarch, service = (_store_name() for _ in range(3))
    role = RankRole.PUBLISHER

    async def run() -> list[int]:
        taken = [
            # The first check, then the step that caches the client.
            await lock.taken_by(ts.client(standard)),
            # The first check, the record's read, then the step that caches the client.
            await lock.taken_by(ts.client(monarch, role=role)),
        ]
        address = await _initialize_service(service, service_strategy)
        try:
            # On the driver's record, then through the address once the client is reset.
            taken.append(await lock.taken_by(ts.client(service, role=role)))
            ts.reset_client(service)
            taken.append(
                await lock.taken_by(ts.client(service, role=role, address=address))
            )
        finally:
            await _clean_up(lambda: _release(service))
        return taken

    assert asyncio.run(run()) == [2, 3, 3, 3]


@pytest.mark.timeout(60)
def test_closes_run_after_the_selection_lock_is_released(
    monkeypatch, service_strategy
) -> None:
    """Close what each selection or shutdown replaced only after its step releases the lock."""
    actors = _patch_actors(monkeypatch)
    actors.lookup.side_effect = None
    actors.lookup.return_value = _controller_stub()
    closes: list[tuple[str, bool]] = []
    for owner in (ServiceControlPlane, CoordinatorService):

        async def recorded_close(self: Any, close: Any = owner.close) -> None:
            closes.append((type(self).__name__, store_api._selection_lock.locked()))
            await close(self)

        monkeypatch.setattr(owner, "close", recorded_close)
    store_name = _store_name()
    # A handle's and a service's close, each with the lock released.
    handle, service = ("ServiceControlPlane", False), ("CoordinatorService", False)

    async def closed_by(call: Awaitable[Any]) -> list[tuple[str, bool]]:
        closes.clear()
        await call
        return list(closes)

    async def run() -> None:
        async with _other_service(store_name, service_strategy) as other:
            try:
                initialized = _initialize_service(store_name, service_strategy)
                assert await closed_by(initialized) == []
                dialed = ts.client(
                    store_name, role=RankRole.PUBLISHER, address=other.address
                )
                assert await closed_by(dialed) == [handle]
                assert await closed_by(ts.shutdown(store_name)) == [handle, service]
            finally:
                await _clean_up(lambda: _release(store_name))

    asyncio.run(run())


@pytest.mark.timeout(60)
def test_a_cancelled_build_on_the_record_keeps_it_open(
    monkeypatch, service_strategy
) -> None:
    """Keep the record, the driver's own handle, open when a build that reuses it is cancelled."""
    actors = _patch_actors(monkeypatch)
    store_name = _store_name()

    async def run() -> None:
        await _initialize_service(store_name, service_strategy)
        handle = _service_handle(store_name)
        try:
            asked, answer = _hold_strategy(handle)
            building = asyncio.ensure_future(
                ts.client(store_name, role=RankRole.PUBLISHER)
            )
            await asyncio.wait_for(asked.wait(), 30)
            building.cancel()
            with pytest.raises(asyncio.CancelledError):
                await building
            answer.set()
            assert not handle._closed
            assert _service_handle(store_name) is handle
            client = await ts.client(store_name, role=RankRole.PUBLISHER)
            assert client._control_plane is handle
        finally:
            await _clean_up(lambda: _release(store_name))

    asyncio.run(run())
    actors.lookup.assert_not_called()


@pytest.mark.timeout(60)
def test_a_failed_log_after_a_selection_still_closes_what_it_replaced(
    monkeypatch, service_strategy
) -> None:
    """Close the replaced handle even when the step after the client's selection fails."""
    _patch_actors(monkeypatch)
    store_name = _store_name()

    async def run() -> None:
        async with _other_service(store_name, service_strategy) as other:
            await _initialize_service(store_name, service_strategy)
            try:
                with patch.object(
                    store_api,
                    "_log_control_plane",
                    side_effect=RuntimeError("log failed"),
                ):
                    replaced = _service_handle(store_name)
                    with pytest.raises(RuntimeError, match="log failed"):
                        await ts.client(
                            store_name, role=RankRole.PUBLISHER, address=other.address
                        )
                    with pytest.raises(RuntimeError, match="is closed"):
                        await replaced.strategy()
                assert store_api._control_plane_map[store_name].address == other.address
            finally:
                await _clean_up(lambda: _release(store_name))

    asyncio.run(run())


@pytest.mark.timeout(60)
@pytest.mark.parametrize("error", [ConnectionResetError, RuntimeError])
def test_a_failing_close_after_a_client_selection_only_logs(
    monkeypatch, caplog, service_strategy, error
) -> None:
    """Keep a client's selection when closing the record it replaced fails, and log it."""
    caplog.set_level(logging.WARNING, logger=store_api.logger.name)
    _patch_actors(monkeypatch)
    store_name = _store_name()

    async def run() -> None:
        async with _other_service(store_name, service_strategy) as other:
            old = await _initialize_service(store_name, service_strategy)
            handle = _service_handle(store_name)
            close = handle.close
            try:
                handle.close = AsyncMock(side_effect=error("gone"))
                client = await ts.client(
                    store_name, role=RankRole.PUBLISHER, address=other.address
                )
                assert store_api._control_plane_map[store_name].address == other.address
                assert store_api._local_clent_map[store_name] is client
                handle.close.assert_awaited_once_with()
                warnings = [
                    (record.getMessage(), record.exc_info[1])
                    for record in caplog.records
                    if record.name == store_api.logger.name
                    and record.levelno == logging.WARNING
                ]
                assert warnings == [
                    (
                        f"closing the connection dialed through {old} failed",
                        handle.close.side_effect,
                    )
                ]
            finally:
                await _clean_up(close, lambda: _release(store_name))

    asyncio.run(run())


@pytest.mark.timeout(60)
def test_a_shutdown_cancelled_closing_the_handle_still_stops_its_service(
    monkeypatch, service_strategy
) -> None:
    """Release the store and stop its service when cancelled in the driver handle's close."""
    _patch_actors(monkeypatch)
    store_name = _store_name()

    async def run() -> None:
        await _initialize_service(store_name, service_strategy)
        handle = _service_handle(store_name)
        service = store_api._owned_service_map[store_name]
        rendezvous = service._rendezvous
        assert rendezvous is not None
        close, entered = handle.close, asyncio.Event()

        async def held_close() -> None:
            entered.set()
            await asyncio.Event().wait()

        handle.close = held_close
        try:
            stopping = asyncio.ensure_future(ts.shutdown(store_name))
            await asyncio.wait_for(entered.wait(), 30)
            stopping.cancel()
            with pytest.raises(asyncio.CancelledError):
                await stopping
            assert store_api._control_plane_map == store_api._owned_service_map == {}
            assert (service._server, service._rendezvous) == (None, None)
            assert rendezvous._store is None
        finally:
            await _clean_up(close, service.close)

    asyncio.run(run())


@pytest.mark.timeout(60)
@pytest.mark.parametrize("failing", ["handle", "service"])
def test_shutdown_releases_a_service_store_before_a_close_fails(
    monkeypatch, service_strategy, failing
) -> None:
    """Drop the record, client and owned service first, close both, then raise the failure.

    A failing handle close still stops the service.
    """
    _patch_actors(monkeypatch)
    store_name = _store_name()

    async def run() -> None:
        await _initialize_service(store_name, service_strategy)
        handle = _service_handle(store_name)
        service = store_api._owned_service_map[store_name]
        close, stop = handle.close, service.close

        async def failing_stop() -> None:
            await stop()
            raise OSError("service failed")

        try:
            await ts.client(store_name, role=RankRole.PUBLISHER)
            if failing == "handle":
                failure = ConnectionResetError("handle failed")
                handle.close = AsyncMock(side_effect=failure)
            else:
                service.close = failing_stop
            with pytest.raises(OSError, match=f"{failing} failed"):
                await ts.shutdown(store_name)
            state = (
                store_api._control_plane_map,
                store_api._local_clent_map,
                store_api._owned_service_map,
            )
            assert state == ({}, {}, {})
            assert (service._server, service._rendezvous) == (None, None)
            assert handle._closed == (failing == "service")
        finally:
            await _clean_up(close, lambda: _release(store_name))

    asyncio.run(run())


@pytest.mark.timeout(60)
def test_service_selection_lines(monkeypatch, caplog, service_strategy) -> None:
    """Log the driver's line with the address, one per new client and none when cached.

    The driver builds its clients on its record, also when given its own address.
    """
    caplog.set_level(logging.INFO)
    _patch_actors(monkeypatch)
    store_name = _store_name()
    dial = patch.object(
        ServiceAddress, "connect", autospec=True, side_effect=ServiceAddress.connect
    )

    async def run() -> ServiceAddress:
        address = await _initialize_service(store_name, service_strategy)
        try:
            with dial as dials:
                await ts.client(store_name, role=RankRole.PUBLISHER)
                ts.reset_client(store_name)
                copy = pickle.loads(pickle.dumps(address))
                await ts.client(
                    store_name, role=RankRole.REQUESTER, group=0, address=copy
                )
                await ts.client(store_name)
            assert dials.call_count == 0
            return address
        finally:
            await _clean_up(lambda: _release(store_name))

    address = asyncio.run(run())
    pid = str(os.getpid())
    assert _selection_lines(caplog) == [
        ("service", str(address), store_name, rank, pid)
        for rank in ("-", "publisher/0", "requester/0/0")
    ]


@pytest.mark.timeout(60)
def test_the_fork_hook_names_a_caller_held_handle_and_skips_standard_clients(
    monkeypatch, service_strategy
) -> None:
    """Run the API's fork hook here: name a caller's coordinator handle, keep standard clients.

    The handle's record has no address and its store no client, so only the handle can name
    the store, and a standard client cached before the service stores stays.
    """
    actors = _patch_actors(monkeypatch)
    actors.lookup.side_effect = None
    actors.lookup.return_value = _controller_stub()
    # The hook keeps what it drops in this list, which is restored.
    monkeypatch.setattr(store_api, "_selection_lock", threading.Lock())
    monkeypatch.setattr(store_api, "_inherited", [])
    store_name, standard, held = _store_name(), _store_name(), _store_name()

    async def run() -> None:
        standard_client = await ts.client(standard)
        address = await _initialize_service(store_name, service_strategy)
        driver = _service_handle(store_name)
        service = store_api._owned_service_map[store_name]
        async with _other_service(held, service_strategy) as other:
            handle = await other.address.connect()
            try:
                await ts.client(held, role=RankRole.PUBLISHER, control_plane=handle)
                ts.reset_client(held)
                # The hook frees the lock the fork held, here one held for it.
                fork_lock = threading.Lock()
                fork_lock.acquire()
                store_api._after_fork_in_child(fork_lock)
                assert store_api._forked_stores == {
                    store_name: address,
                    held: other.address,
                }
                assert store_api._control_plane_map == {}
                assert store_api._owned_service_map == {}
                # Kept: it must never be freed before the rendezvous module's hook keeps its TCPStore.
                assert any(kept is service for kept in store_api._inherited)
                assert store_api._local_clent_map == {standard: standard_client}
            finally:
                failing = sys.exc_info()[1] is not None
                await _close_all([handle, driver, service], failing)

    asyncio.run(run())


@pytest.mark.timeout(60)
def test_a_forked_store_selected_again_without_its_address_drops_the_marker(
    monkeypatch, service_strategy
) -> None:
    """Re-select a store the fork hook dropped through a Monarch initialize, which records no plane."""
    actors = _patch_actors(monkeypatch)
    actors.lookup.side_effect = None
    actors.lookup.return_value = _controller_stub()
    monkeypatch.setattr(store_api, "_selection_lock", threading.Lock())
    monkeypatch.setattr(store_api, "_inherited", [])
    store_name = _store_name()

    async def run() -> None:
        address = await _initialize_service(store_name, service_strategy)
        driver = _service_handle(store_name)
        service = store_api._owned_service_map[store_name]
        try:
            fork_lock = threading.Lock()
            fork_lock.acquire()
            store_api._after_fork_in_child(fork_lock)
            assert store_api._forked_stores == {store_name: address}
            await _initialize(store_name)
            assert store_api._forked_stores == {}
            await ts.client(store_name, role=RankRole.PUBLISHER)
        finally:
            await _clean_up(lambda: _release(store_name), driver.close, service.close)

    asyncio.run(run())


def test_the_service_backend_loads_once_and_names_only_its_handles(
    monkeypatch,
) -> None:
    """Return one service backend, which names a coordinator handle's address and nothing else's.

    A stand-in installed before the backend registers stays.
    """
    backend = backend_module.get_backend("service")
    assert isinstance(backend, service_module.CarrierServiceBackend)
    assert backend_module.get_backend("service") is backend
    assert backend_module.loaded_backend("service") is backend
    address = ServiceAddress("store", "127.0.0.1", 1, "nonce")
    handle = MagicMock(spec=ServiceControlPlane)
    handle.address = address
    assert backend.service_address(handle) is address
    assert backend.service_address(SimpleNamespace(address=address)) is None
    assert backend.service_address(None) is None
    stand_in = MagicMock(spec=ServiceBackend)
    monkeypatch.setitem(backend_module._backends, "service", stand_in)
    backend_module.register_backend("service", backend)
    assert backend_module.loaded_backend("service") is stand_in


def test_a_backend_stands_again_once_a_stand_in_before_it_goes(monkeypatch) -> None:
    """Return a module's own backend once a stand-in installed before the module registered is gone.

    Before backends registered, such a lookup also returned the module's backend. A reload's registration leaves the
    first one standing, and the fork hook still names service handles.
    """
    backend = backend_module.get_backend("service")
    # As if a stand-in had come first, kept the module's registration out of the table, and gone.
    monkeypatch.delitem(backend_module._backends, "service")
    assert backend_module.loaded_backend("service") is backend
    assert backend_module.get_backend("service") is backend
    backend_module.register_backend("service", service_module.CarrierServiceBackend())
    assert backend_module.loaded_backend("service") is backend
    monkeypatch.delitem(backend_module._backends, "service")
    monkeypatch.setattr(store_api, "_selection_lock", threading.Lock())
    monkeypatch.setattr(store_api, "_inherited", [])
    store_name = _store_name()
    handle = MagicMock(spec=ServiceControlPlane)
    handle.address = ServiceAddress(store_name, "127.0.0.1", 1, "nonce")
    store_api._control_plane_map[store_name] = store_api._SelectedControlPlane(
        handle, None
    )
    fork_lock = threading.Lock()
    fork_lock.acquire()
    store_api._after_fork_in_child(fork_lock)
    assert store_api._control_plane_map == {}
    assert store_api._forked_stores == {store_name: handle.address}


@pytest.mark.timeout(60)
def test_a_store_running_its_service_refuses_another_initialize(
    monkeypatch, service_strategy
) -> None:
    """Refuse to initialize a store again, on any plane and before spawning anything, while it runs its coordinator service.

    Another store initializes meanwhile, and the store itself once shut down.
    """
    actors = _patch_actors(monkeypatch)
    store_name, other = _store_name(), _store_name()
    plane = Mock(spec=ControlPlane)

    async def run() -> None:
        address = await _initialize_service(store_name, service_strategy)
        try:
            spawned = actors.spawn.await_count
            for control_plane in ("service", "monarch", plane):
                with pytest.raises(RuntimeError, match="call ts.shutdown"):
                    await _initialize(store_name, control_plane)
            assert actors.spawn.await_count == spawned and not plane.init.called
            assert store_api._owned_service_map[store_name].address == address
            await _initialize_service(other, service_strategy)
        finally:
            await _release_all([store_name, other], sys.exc_info()[1] is not None)
        assert await _initialize_service(store_name, service_strategy) != address
        await _release(store_name)

    asyncio.run(run())


@pytest.mark.timeout(60)
def test_the_later_of_overlapping_service_initializes_of_a_store_stops_its_service(
    monkeypatch, service_strategy, started_services
) -> None:
    """Refuse the later of two overlapping service initializes of one store, stopping the service and handle it made.

    The docstring forbids the overlap, but no service may outlive ``ts.shutdown``.
    """
    _patch_actors(monkeypatch)
    store_name = _store_name()
    handles: list[ServiceControlPlane] = []
    init = ServiceControlPlane.init

    async def run() -> None:
        arrived, gate = asyncio.Event(), asyncio.Event()

        async def ordered_init(handle: ServiceControlPlane, **kwargs: Any) -> None:
            handles.append(handle)
            if len(handles) == 1:
                # The first to arrive waits: the other one records its service first.
                arrived.set()
                await gate.wait()
            await init(handle, **kwargs)

        monkeypatch.setattr(ServiceControlPlane, "init", ordered_init)
        later = asyncio.ensure_future(
            _initialize(store_name, "service", strategy=service_strategy)
        )
        await _until_set(arrived, later)
        try:
            await _initialize_service(store_name, service_strategy)
        finally:
            gate.set()
        with pytest.raises(RuntimeError, match="call ts.shutdown"):
            await later
        mine, kept = started_services
        assert (mine._server, mine._rendezvous) == (None, None) and handles[0]._closed
        assert store_api._owned_service_map[store_name] is kept
        await _release(store_name)
        assert (kept._server, kept._rendezvous) == (None, None)

    asyncio.run(run())


@pytest.mark.timeout(60)
@pytest.mark.parametrize("plane", ["monarch", "caller"])
def test_an_initialize_recording_after_an_overlapping_service_initialize_is_refused(
    monkeypatch, service_strategy, started_services, plane: str
) -> None:
    """Refuse an initialize on another plane that began before a service initialize of its store but records after it.

    The docstring forbids the overlap. The service keeps its record until ``ts.shutdown`` stops it.
    """
    actors = _patch_actors(monkeypatch)
    store_name = _store_name()
    coordinator = _controller_stub()
    actors.lookup.side_effect = None
    actors.lookup.return_value = coordinator
    caller = Mock(spec=ControlPlane)

    async def run() -> None:
        entered, release = asyncio.Event(), asyncio.Event()

        async def held_init(**kwargs: Any) -> None:
            entered.set()
            await release.wait()

        if plane == "monarch":
            coordinator.init.call_one = held_init
        else:
            caller.init = held_init
        other = asyncio.ensure_future(
            _initialize(store_name, "monarch" if plane == "monarch" else caller)
        )
        await _until_set(entered, other)
        await _initialize_service(store_name, service_strategy)
        handle = _service_handle(store_name)
        release.set()
        with pytest.raises(RuntimeError, match="call ts.shutdown"):
            await other
        [service] = started_services
        assert _service_handle(store_name) is handle and not handle._closed
        assert store_api._owned_service_map[store_name] is service
        await ts.shutdown(store_name)
        assert (service._server, service._rendezvous) == (None, None)
        assert store_name not in store_api._owned_service_map

    asyncio.run(run())


@pytest.mark.timeout(60)
def test_a_service_initialize_raises_the_refusal_of_a_strategy_class_defined_in_a_function(
    monkeypatch, service_strategy, started_services
) -> None:
    """Raise the TCPStore rendezvous's refusal of a strategy class without an importable name, and keep no service."""
    _patch_actors(monkeypatch)
    store_name = _store_name()

    class LocalStrategy(ts.TorchStoreStrategy):
        pass

    async def run() -> None:
        refused: tuple[type[Exception], ...] = (AttributeError, pickle.PicklingError)
        with pytest.raises(refused, match="local"):
            await _initialize(
                store_name, "service", strategy=LocalStrategy(TransportType.MonarchRPC)
            )
        [service] = started_services
        assert (service._server, service._rendezvous) == (None, None)
        assert store_api._control_plane_map == store_api._owned_service_map == {}

    asyncio.run(run())


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "failure",
    ["interrupted", "cancelled", "connect-fails", "handle-close-fails", "stop-fails"],
)
def test_a_failed_service_initialize_leaves_no_server(
    monkeypatch, service_strategy, started_services, caplog, failure: str
) -> None:
    """Close the new handle and stop the new service when its connect or init fails, also interrupted or cancelled.

    A close or stop that fails meanwhile only logs, and the plane recorded before and its cached client stay.
    """
    _patch_actors(monkeypatch)
    store_name = _store_name()
    handles: list[ServiceControlPlane] = []
    connect, stop = CoordinatorService.connect, CoordinatorService.close
    raised = {
        "interrupted": KeyboardInterrupt,
        "cancelled": asyncio.CancelledError,
    }.get(failure, OSError)

    async def failing_connect(service: CoordinatorService) -> ServiceControlPlane:
        if failure == "connect-fails":
            raise OSError("connect failed")
        handle = await connect(service)
        if failure == "handle-close-fails":

            async def failing_close() -> NoReturn:
                raise RuntimeError("handle close failed")

            handle.close = failing_close  # type: ignore[method-assign]
        return handle

    async def failing_init(handle: ServiceControlPlane, **kwargs: Any) -> None:
        handles.append(handle)
        raise raised("init failed")

    async def failing_stop(service: CoordinatorService) -> None:
        await stop(service)
        if failure == "stop-fails":
            raise RuntimeError("stop failed")

    caplog.set_level(logging.WARNING, logger=store_api.__name__)

    async def run() -> None:
        async with registry_store() as connect_registry:
            recorded = await connect_registry()
            await _initialize(store_name, recorded)
            client = await ts.client(store_name, role=RankRole.PUBLISHER)
            # Undone however the body ends. Cleanup then stops each service through the class's own close.
            with monkeypatch.context() as patched:
                patched.setattr(CoordinatorService, "connect", failing_connect)
                patched.setattr(CoordinatorService, "close", failing_stop)
                patched.setattr(ServiceControlPlane, "init", failing_init)
                # Awaited in this task: the interrupt reaches pytest.raises and not the loop.
                with pytest.raises(raised):
                    await _initialize(store_name, "service", strategy=service_strategy)
            [service] = started_services
            assert (service._server, service._rendezvous) == (None, None)
            if failure != "connect-fails":
                [handle] = handles
                assert handle._closed or failure == "handle-close-fails"
            assert store_api._control_plane_map == {store_name: (recorded, None)}
            assert store_api._owned_service_map == {}
            assert await ts.client(store_name) is client
        warned = "\n".join(r.getMessage() for r in caplog.records)
        if failure == "handle-close-fails":
            assert "closing the connection dialed through" in warned
        if failure == "stop-fails":
            assert "stopping the coordinator service on" in warned

    asyncio.run(run())
