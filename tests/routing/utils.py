# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import functools
import logging
import threading
import uuid
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Collection,
    Coroutine,
    Mapping,
)
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import monarch.actor as monarch_actor
import pytest
import pytest_asyncio
import torchstore as ts
import torchstore.api as store_api
import torchstore.monarch.backend as monarch_backend
from monarch.actor import get_or_spawn_controller, ProcMesh, this_host
from torchstore import carrier_volumes
from torchstore.monarch.coordinator import MonarchControlPlane, RoutingCoordinator
from torchstore.routing._model import KeyRegistration, RankRole, Registrations
from torchstore.routing.client import RoutingClient
from torchstore.routing.control_plane import ControlPlane, ControlPlaneAddress
from torchstore.routing.layout_registry import LayoutRegistry
from torchstore.routing.service import ServiceControlPlane
from torchstore.strategy import _installed_volume_host
from torchstore.transport import TransportType
from torchstore.transport.types import TensorSlice

from .tcpstore_fixtures import _initialize_service

logger = logging.getLogger(__name__)

# Makes routing clients from publishers' and requesters' state dicts by rank.
MakeClients = Callable[..., Awaitable[dict[str, RoutingClient]]]
# One rank's control plane, and the strategy that rank read from it.
_RankPlane = tuple[ControlPlane, ts.TorchStoreStrategy]
# Live volumes answer in milliseconds, and a stalled one gives no other sign.
_VOLUME_ANSWER_S = 10.0


class _FaultLog:
    """Monarch's fault hook from a session's first routing case to its end, which logs each fault while a case runs.

    Once a volume check failed it logs later faults at INFO between cases and at WARNING within a later case. A test
    that installs its own hook gets them. Otherwise the hook it replaced (Monarch's ends the session) gets each fault.
    """

    # The hook this one replaced, read when the session's first case installs it.
    _previous: Callable[..., None]

    def __init__(self) -> None:
        self._installed = False
        self._running = 0
        # Set when a volume check fails, as that case already failed for those faults.
        self.late = False

    def __call__(self, failure: monarch_actor.MeshFailure) -> None:
        if self._running:
            logger.warning("Monarch reported a mesh fault: %s", failure.report())
        elif self.late:
            # Below WARNING so later tests that check their own WARNINGs see none of these.
            logger.info(
                "Monarch reported a mesh fault after its case: %s", failure.report()
            )
        else:
            self._previous(failure)

    @contextlib.asynccontextmanager
    async def case(
        self, config: pytest.Config
    ) -> AsyncIterator[contextlib.AsyncExitStack]:
        """Run one case: install this hook once per session, and undo the case's setup on exit."""
        if not self._installed:
            self._installed = True
            self._previous = monarch_actor.unhandled_fault_hook
            monarch_actor.unhandled_fault_hook = self
            config.add_cleanup(self._uninstall)
        self._running += 1
        try:
            async with contextlib.AsyncExitStack() as exit_stack:
                yield exit_stack
        finally:
            self._running -= 1

    def _uninstall(self) -> None:
        monarch_actor.unhandled_fault_hook = self._previous
        # So a later session in this interpreter installs it again, with no failed check.
        self._installed = self.late = False


_fault_log = _FaultLog()


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


@pytest_asyncio.fixture(params=["monarch", "service"])
async def make_clients(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[MakeClients]:
    """Make routing clients on the ``ts.initialize`` value that the test's id names.

    ``registered=True`` makes them with ``registered_clients``, else ``routing_clients``. Each setup is undone after
    the test, apart from Monarch's coordinators and the fault log, which stays Monarch's hook for the session.
    """
    lookup = Mock(wraps=monarch_backend.get_or_spawn_controller)
    constructed: list[str] = []
    init = RoutingCoordinator.__init__
    spawned: list[ProcMesh] = []
    spawn_host = this_host

    def looked_up() -> list[str]:
        return [call.args[1].__name__ for call in lookup.call_args_list]

    def recording_init(self: Any, *args: Any, **kwargs: Any) -> None:
        constructed.append(type(self).__name__)
        init(self, *args, **kwargs)

    def recording_host() -> Any:
        host = spawn_host()

        def spawn_procs(*args: Any, **kwargs: Any) -> ProcMesh:
            spawned.append(host.spawn_procs(*args, **kwargs))
            return spawned[-1]

        return SimpleNamespace(spawn_procs=spawn_procs)

    # torchstore.api looks Monarch's actors up through the Monarch backend so this records each of
    # its lookups. The Monarch arm's own lookup, like ts.initialize_spmd's, goes to Monarch directly.
    monkeypatch.setattr(monarch_backend, "get_or_spawn_controller", lookup)
    monkeypatch.setattr(RoutingCoordinator, "__init__", recording_init)
    monkeypatch.setattr(f"{__name__}.this_host", recording_host)
    # ts.shutdown adds each service store's name here, and this copy goes after the case.
    monkeypatch.setattr(store_api, "_released_stores", set(store_api._released_stores))

    def held() -> set[str]:
        return {*store_api._control_plane_map, *store_api._owned_service_map}

    before = held()
    made = 0
    async with _fault_log.case(request.config) as exit_stack:

        async def make(
            publishers: Mapping[str, Mapping[str, Any]],
            requesters: Mapping[str, Mapping[str, Any]],
            *,
            registered: bool = False,
        ) -> dict[str, RoutingClient]:
            nonlocal made
            # Only the Monarch arm spawns a publisher mesh: on the service backend publishers serve their own volumes.
            if request.param == "monarch":
                made += 1
            seen = len(lookup.call_args_list), len(constructed)
            helper = registered_clients if registered else routing_clients
            clients = await helper(publishers, requesters, request.param, exit_stack)
            if request.param == "monarch":
                # ts.initialize looks the new store's coordinator up once, which builds it.
                built = looked_up()[seen[0] :], constructed[seen[1] :]
                assert built == (["RoutingCoordinator"], ["RoutingCoordinator"]), built
            return clients

        yield make
    # After the stack, so a service store whose teardown never ran fails too.
    await _assert_torn_down(
        request.param, held() - before, spawned, made, looked_up(), constructed
    )


async def _assert_torn_down(
    control_plane: str,
    left: Collection[str],
    spawned: Collection[Any],
    made: int,
    looked_up: list[str],
    constructed: list[str],
) -> None:
    """Fail on what a case's stack left: a store recorded or owned, a mesh running or never recorded, a Monarch lookup.

    ``made`` counts the ``make_clients`` calls that spawn one publisher mesh each, those on the Monarch arm. Neither the
    service arm nor its shutdown looks up or builds Monarch's coordinator.
    """
    assert not left, f"left recorded or owned: {sorted(left)}"
    assert len(spawned) == made, f"{made} meshes due but {len(spawned)} meshes recorded"
    for mesh in spawned:
        # A stopped mesh refuses to start again, and this stops one the stack left running.
        with pytest.raises(RuntimeError):
            async with mesh:
                pass
    if control_plane == "service":
        reached = (looked_up, constructed)
        assert reached == ([], []), f"the service arm reached Monarch: {reached}"


async def routing_clients(
    publishers: Mapping[str, Mapping[str, Any]],
    requesters: Mapping[str, Mapping[str, Any]],
    control_plane: str,
    exit_stack: contextlib.AsyncExitStack,
) -> dict[str, RoutingClient]:
    """Make one client per rank on a new store initialized on ``control_plane``.

    Each client gets its rank's control plane and strategy, as ``ts.client`` does. No layout
    is registered. ``exit_stack`` undoes the setup, apart from Monarch's coordinator, which
    callers cannot stop.
    """
    roles = {
        **{rank: RankRole.PUBLISHER for rank in publishers},
        **{rank: RankRole.REQUESTER for rank in requesters},
    }
    if control_plane == "monarch":
        ranks = await _monarch_ranks(len(publishers), len(roles), exit_stack)
    elif control_plane == "service":
        ranks = await _service_ranks(list(publishers), len(roles), exit_stack)
    else:
        raise ValueError(f"no routing test backend {control_plane!r}")
    clients = {}
    for (rank, role), (plane, strategy) in zip(roles.items(), ranks, strict=True):
        clients[rank] = RoutingClient(rank, role, plane, strategy)
    if control_plane == "service":
        _assert_own_handles(clients.values())
    return clients


def _assert_own_handles(clients: Collection[RoutingClient]) -> None:
    """Fail unless each client calls a service handle of its own, never a driver's."""
    planes = [c._control_plane for c in clients]
    kinds = [type(plane).__name__ for plane in planes]
    assert all(type(plane) is ServiceControlPlane for plane in planes), kinds
    handles = {id(plane) for plane in planes}
    assert len(handles) == len(planes), f"{len(planes)} ranks, {len(handles)} handles"
    drivers = {id(r.control_plane) for r in store_api._control_plane_map.values()}
    assert not handles & drivers, "a rank calls the driver's handle"


async def _initialize(
    publisher_mesh: ProcMesh, control_plane: str, exit_stack: contextlib.AsyncExitStack
) -> tuple[str, ControlPlaneAddress | None]:
    """Initialize a routing store on ``control_plane`` with its volumes on ``publisher_mesh``.

    Returns the store's name and the address ``ts.initialize`` returned. ``exit_stack`` asks
    each volume to answer, then stops the mesh.
    """
    await exit_stack.enter_async_context(publisher_mesh)
    store_name = f"routing-test-{uuid.uuid4()}"
    strategy = ts.TorchStoreStrategy(TransportType.MonarchRPC)
    address = await ts.initialize(
        mesh=publisher_mesh,
        strategy=strategy,
        client_type=ts.ClientType.ROUTING,
        store_name=store_name,
        control_plane=control_plane,
    )
    # Before the mesh stops so that in a make_clients case a publisher's death or stall fails that case.
    exit_stack.push_async_callback(_assert_volumes_answer, strategy)
    return store_name, address


async def _assert_volumes_answer(strategy: ts.TorchStoreStrategy) -> None:
    """Fail unless each of the store's volumes answers within ``_VOLUME_ANSWER_S``.

    One that fails or stalls also has the fault log, once a ``make_clients`` case installed it, keep Monarch's
    later faults from ending the session.
    """
    host = _installed_volume_host(strategy)
    volume_ids = list(
        strategy.volume_id_to_coord if host is None else host.volume_ids()
    )
    assert volume_ids, "the store has no volumes"
    calls = [
        strategy.get_storage_volume(i).volume.get_id.call_one() for i in volume_ids
    ]
    answers = asyncio.gather(*calls)
    try:
        await asyncio.wait_for(answers, _VOLUME_ANSWER_S)
    except asyncio.TimeoutError:
        _fault_log.late = True
        pytest.fail(f"a publisher's volume did not answer within {_VOLUME_ANSWER_S}s")
    except BaseException:
        _fault_log.late = True
        raise


async def _monarch_ranks(
    num_publishers: int, num_ranks: int, exit_stack: contextlib.AsyncExitStack
) -> list[_RankPlane]:
    """Every rank on the one handle to the store's Monarch coordinator.

    No ``ts.shutdown``: on a routing store it would look up a standard controller. The
    coordinator stays up, since callers cannot stop it.
    """
    mesh = this_host().spawn_procs(per_host={"procs": num_publishers})
    store_name, _ = await _initialize(mesh, "monarch", exit_stack)
    plane = MonarchControlPlane(
        await get_or_spawn_controller(store_name, RoutingCoordinator)
    )
    return [(plane, await plane.strategy()) for _ in range(num_ranks)]


async def _service_ranks(
    publishers: list[str], num_ranks: int, exit_stack: contextlib.AsyncExitStack
) -> list[_RankPlane]:
    """Give each rank its own handle from the coordinator service's address, as a rank process does.

    Handles in one process share its one connection to the coordinator. Each publisher serves its
    volume here, as its ``ts.client`` does, with no proc mesh. Teardown checks that each volume
    answers, closes every host and handle, then shuts the store down.
    """
    patch = exit_stack.enter_context(pytest.MonkeyPatch.context())
    # The coordinator service and the volumes advertise this host, which this process reaches.
    patch.setenv("TORCHSTORE_TCPSTORE_HOST", "127.0.0.1")
    patch.setenv("TORCHSTORE_CARRIER_HOST", "127.0.0.1")
    # ts.shutdown records the store as released, and the stack drops this copy after it.
    patch.setattr(store_api, "_released_stores", set(store_api._released_stores))
    store_name = f"routing-test-{uuid.uuid4()}"
    address = await _initialize_service(
        store_name,
        ts.TorchStoreStrategy(TransportType.MonarchRPC),
        publishers=len(publishers),
    )
    handles: list[ServiceControlPlane] = []
    exit_stack.push(functools.partial(_assert_released, store_name, handles))
    exit_stack.push_async_callback(ts.shutdown, store_name)
    for _ in range(num_ranks):
        handle = await address.connect()
        assert type(handle) is ServiceControlPlane, type(handle)
        exit_stack.push_async_callback(handle.close)
        handles.append(handle)
    ranks: list[_RankPlane] = []
    for handle in handles:
        strategy = await handle.strategy()
        host = _installed_volume_host(strategy)
        assert isinstance(host, carrier_volumes.CarrierVolumeHost), type(host)
        exit_stack.push_async_callback(host.close)
        ranks.append((handle, strategy))
    for volume_id, handle, (_, strategy) in zip(publishers, handles, ranks):
        volume = await carrier_volumes.start_publisher_volume(
            volume_id,
            store_name=store_name,
            selection=(handle, address),
            register=handle.register_volume,
        )
        assert volume.record is not None, volume_id
        host = _installed_volume_host(strategy)
        assert isinstance(host, carrier_volumes.CarrierVolumeHost), type(host)
        host.adopt(volume_id, volume.record)
    # The driver's handle, which no rank uses, resolves every volume for the check.
    driver = store_api._control_plane_map[store_name].control_plane
    checking = await driver.strategy()
    await checking._prepare_volumes(publishers)
    host = _installed_volume_host(checking)
    assert isinstance(host, carrier_volumes.CarrierVolumeHost), type(host)
    exit_stack.push_async_callback(host.close)
    # Before the store shuts down, so a volume's stall surfaces in its own case.
    exit_stack.push_async_callback(_assert_volumes_answer, checking)
    return ranks


def _assert_released(
    store_name: str,
    handles: Collection[ServiceControlPlane],
    exc_type: type[BaseException] | None,
    *_: Any,
) -> None:
    """Fail if the store stays recorded, owned or served, or a rank's handle was never closed.

    Silent once a teardown step has raised, so that step's error is the one reported.
    """
    if exc_type is not None:
        return
    assert store_name not in store_api._control_plane_map, f"{store_name!r} recorded"
    assert store_name not in store_api._owned_service_map, f"{store_name!r} owned"
    served = [
        volume_id
        for name, volume_id in carrier_volumes._publishers
        if name == store_name
    ]
    assert not served, f"{store_name!r} still serves {served}"
    unclosed = sum(not handle._closed for handle in handles)
    assert not unclosed, f"{unclosed} of {len(handles)} rank handles never closed"


async def registered_clients(
    publishers: Mapping[str, Mapping[str, Any]],
    requesters: Mapping[str, Mapping[str, Any]],
    control_plane: str,
    exit_stack: contextlib.AsyncExitStack,
    namespace: str = "model",
) -> dict[str, RoutingClient]:
    """Make clients as ``routing_clients`` does, then exchange their layouts.

    Publishers register their layouts. Requesters fetch them and plan their routes.
    """
    clients = await routing_clients(publishers, requesters, control_plane, exit_stack)
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


class HostedRegistry(ControlPlane):
    """Runs each call as a task on the registry's loop, like a concurrent endpoint."""

    def __init__(
        self, registry: LayoutRegistry, loop: asyncio.AbstractEventLoop
    ) -> None:
        self._registry = registry
        self._loop = loop

    async def _on_host(self, coroutine: Coroutine[Any, Any, Any]) -> Any:
        return await asyncio.wrap_future(
            asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        )

    async def init(
        self, publishers: Collection[str], strategy: ts.TorchStoreStrategy
    ) -> None:
        await self._on_host(self._registry.init(publishers, strategy))

    async def strategy(self) -> ts.TorchStoreStrategy:
        return await self._on_host(self._registry.strategy())

    async def register_layout(
        self, rank: str, key: str, registrations: Mapping[str, KeyRegistration]
    ) -> None:
        await self._on_host(self._registry.register_layout(rank, key, registrations))

    async def get_layouts(
        self, key: str, *, timeout_s: float | None = None
    ) -> Registrations:
        layouts = await self._on_host(
            self._registry.get_layouts(key, timeout_s=timeout_s)
        )
        # Callers own their layouts, as with Monarch's pickled replies.
        return {rank: dict(registrations) for rank, registrations in layouts.items()}


@contextlib.asynccontextmanager
async def registry_store() -> AsyncIterator[Callable[..., Awaitable[HostedRegistry]]]:
    """Host one ``LayoutRegistry`` on a thread's own event loop, without Monarch.

    ``connect`` takes a ``HostedRegistry`` subclass to build. Exit fails if the host loop
    has not stopped within 10s, unless the body already failed.
    """
    hosting: concurrent.futures.Future = concurrent.futures.Future()

    async def host() -> None:
        stop = asyncio.Event()
        hosting.set_result((asyncio.get_running_loop(), stop))
        await stop.wait()

    thread = threading.Thread(target=asyncio.run, args=(host(),), daemon=True)
    thread.start()
    loop, stop = await asyncio.wrap_future(hosting)
    registry = LayoutRegistry()

    async def connect(cls: type[HostedRegistry] = HostedRegistry) -> HostedRegistry:
        return cls(registry, loop)

    failed = False
    try:
        yield connect
    except BaseException:
        failed = True
        raise
    finally:
        # A failing body stays the reported failure.
        with contextlib.suppress(Exception) if failed else contextlib.nullcontext():
            try:
                # ``asyncio.run`` cancels any call still running on the host loop.
                loop.call_soon_threadsafe(stop.set)
            finally:
                await asyncio.to_thread(thread.join, 10)
            assert not thread.is_alive(), "the registry's host loop did not stop"
