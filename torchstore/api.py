# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import contextlib
import os
import sys
import threading
import weakref
from collections.abc import Awaitable, Callable
from functools import partial
from logging import getLogger
from typing import Any, Literal, NamedTuple, overload, TYPE_CHECKING

import torch
from torch.distributed.tensor import DTensor

import torchstore.state_dict_utils
from torchstore.backend import get_backend, loaded_backend, ServiceBackend
from torchstore.client import ClientType, LocalClient
from torchstore.routing._model import RankRole
from torchstore.routing.client import RoutingClient
from torchstore.routing.control_plane import ControlPlane, ControlPlaneAddress
from torchstore.routing.volume_records import VolumeRecords
from torchstore.strategy import (
    _installed_volume_host,
    ControllerStorageVolumes,
    TorchStoreStrategy,
)
from torchstore.transport.types import TensorSlice
from torchstore.utils import _off_task, _store_address, start_daemon_thread

if TYPE_CHECKING:
    from monarch.actor import ProcMesh

    from torchstore.carrier_volumes import CarrierVolumeHost, PublisherVolume
    from torchstore.monarch.controller import Controller
    from torchstore.monarch.spmd import _SPMDSession
    from torchstore.routing.service import CoordinatorService

logger = getLogger(__name__)

# I need to keep this somewhere, so here we go
DEFAULT_TORCHSTORE_NAME: str = "torchstore"

# cache for local clients
_local_clent_map: dict[str, LocalClient] = {}


# SPMD-initialized stores register their session here
_spmd_state_map: dict[str, "_SPMDSession"] = {}


class _SelectedControlPlane(NamedTuple):
    """A routing control plane passed in or dialed, and the address it came from."""

    control_plane: ControlPlane
    address: ControlPlaneAddress | None


# Control planes this process initialized, was given or connected to. It survives
# reset_client, and stores selected with control_plane="monarch" never enter it.
_control_plane_map: dict[str, _SelectedControlPlane] = {}

# Coordinator services that ts.initialize started here, until shut down.
_owned_service_map: dict[str, "CoordinatorService"] = {}

# Held while initialize, client and shutdown update the control-plane records, owned services,
# fork markers, released stores and dropped clients with the client cache, for a client build's lone
# cache write, and across fork. Other reads and reset_client rely on the overlaps their docstrings
# allow. Never held across an await: dialed connections and services close after it is released.
_selection_lock = threading.Lock()

# Stores whose record ts.shutdown released here, until a routing initialize on Monarch's
# coordinator or a standard initialize takes the name back, so that a role-less call or
# another shutdown spawns no Monarch actor.
_released_stores: set[str] = set()

# A forked child's stores whose dialed records the fork dropped, with their addresses,
# until a selection, a standard initialize or a shutdown. Without address= or
# control_plane=, calls for them raise instead of looking Monarch up.
_forked_stores: dict[str, ControlPlaneAddress] = {}

# Stores this process builds a client for through an address or a plane, until one such build
# records the store or fails into _attempted. Shutdown releases a marked store without Monarch until
# a selection, a standard initialize, an SPMD session, or a Monarch or standard client takes the name.
_building: set[str] = set()
_attempted: set[str] = set()
# Standard stores initialized here until shut down, whose teardown no such mark skips.
_standard_stores: set[str] = set()
# Per store, the rank IDs that builds in flight were given. A racing build given another is refused.
_building_ranks: dict[str, list[str]] = {}

# The most publishers a store on "service" takes, four times the design's largest scale. The
# rendezvous writes their keys well within its deadline even beside busy threads.
_MAX_SERVICE_PUBLISHERS = 16_384

# Volume hosts of clients that reset_client or a newer client dropped, beside weak refs to
# their strategies. Shutdown closes them, and a build the collected ones but the newest.
# Both take entries out of a store's list in place because reset_client appends without the lock.
_dropped_carriers: dict[
    str, list[tuple[weakref.ref[TorchStoreStrategy], "CarrierVolumeHost"]]
] = {}

# How long a caller waits for a dialed connection's close, which then runs on alone
# until its event loop ends.
_CLOSE_WAIT_S = 5.0

# Closes still running, kept so that none is collected before it finishes.
_closing: set["asyncio.Task[None]"] = set()


def _routing_volume_id(namespace: str) -> str:
    """Qualify the caller's rank from the backend with its mesh's namespace.

    A call made through a slice of the mesh gets the caller's position in that slice.
    """
    return f"{namespace}/{get_backend('monarch').rank()}"


def _actor_rank() -> int:
    """This call's position in its Monarch mesh (``current_rank``), from the Monarch backend.

    It is the actor's rank only for an endpoint called on the whole mesh, not through a slice.
    Raises ValueError outside an actor's context, as in a Monarch driver or without torchmonarch.
    """
    try:
        rank = get_backend("monarch").actor_rank()
    except ModuleNotFoundError as error:
        if error.name is None or error.name.split(".")[0] != "monarch":
            raise
        rank = None
    if rank is None:
        raise ValueError(
            "pass rank=: this call has no rank of its own. No Monarch actor context is current"
            " outside an actor or on a thread that threading.Thread or run_in_executor started"
        )
    return rank


def _routing_namespace(role: RankRole, group: int | None) -> str:
    """The namespace of one participating mesh."""
    if role == RankRole.PUBLISHER:
        if group is not None:
            raise ValueError("publishers are a single mesh and take no group")
        return role.value
    if group is None:
        raise ValueError("requesters must pass the index of their requester mesh")
    return f"{role.value}/{group}"


def _client_rank(
    namespace: str, rank: int | None, control_plane: ControlPlane | None
) -> str:
    """This process's rank ID in a routing store, such as ``publisher/0``.

    Where publishers serve their own volumes it is ``rank``, or without one the call's position
    in its Monarch mesh. On other control planes Monarch's ``current_rank`` names it, and
    ``rank`` raises.
    """
    if not _hosts_volumes(control_plane):
        if rank is not None:
            raise _hosted_only("rank")
        return _routing_volume_id(namespace)
    return f"{namespace}/{_actor_rank() if rank is None else rank}"


def _hosted_only(argument: str) -> ValueError:
    """The error for ``rank`` or ``num_publishers`` where Monarch spawns the volumes."""
    return ValueError(
        f"{argument} is only for control planes whose publishers serve their own"
        " volumes, such as 'service'. On others, volumes are Monarch actors and ranks"
        " come from Monarch"
    )


def _check_int(name: str, value: object, minimum: int) -> None:
    """Refuse anything but an int of at least ``minimum``. A bool is not one."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an int, got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")


def _caller_control_plane(
    control_plane: str | ControlPlane,
) -> ControlPlane | Literal["service"] | None:
    """The caller's own control plane, ``"service"``, or None to keep Monarch's coordinator."""
    if isinstance(control_plane, ControlPlane):
        return control_plane
    if control_plane == "monarch":
        return None
    if control_plane == "service":
        return "service"
    error = ValueError if isinstance(control_plane, str) else TypeError
    raise error(
        "control_plane must be 'monarch', 'service' or a ControlPlane instance, "
        f"got {control_plane!r}"
    )


def _hosts_volumes(control_plane: ControlPlane | Literal["service"] | None) -> bool:
    """Whether publishers serve their own volumes: on a control plane sharing volume records."""
    return isinstance(control_plane, str) or isinstance(control_plane, VolumeRecords)


def _check_one_store(store_name: str, control_plane: ControlPlane) -> None:
    """Refuse an instance another store recorded.

    Its methods name no store. The caller holds ``_selection_lock``.
    """
    for other, selected in _control_plane_map.items():
        if other != store_name and selected.control_plane is control_plane:
            raise ValueError(
                f"this control plane already serves store {other!r}. Give each "
                "routing store its own ControlPlane instance"
            )


def _select_control_plane(
    store_name: str, selected: _SelectedControlPlane | None
) -> _SelectedControlPlane | None:
    """Record this process's control plane for a store (None for Monarch's) and return the old.

    The caller holds ``_selection_lock``.
    """
    if selected is not None:
        _check_one_store(store_name, selected.control_plane)
    _forked_stores.pop(store_name, None)
    _attempted.discard(store_name)
    recorded = _control_plane_map.get(store_name)
    if selected is None:
        _released_stores.discard(store_name)
        _control_plane_map.pop(store_name, None)
    else:
        _control_plane_map[store_name] = selected
    return recorded


def _given(store_name: str, control_plane: ControlPlane) -> _SelectedControlPlane:
    """Select a caller's plane, keeping the address of one this process dialed for the store.

    A plane that ``connect`` returned stays TorchStore's to close.
    """
    recorded = _control_plane_map.get(store_name)
    if recorded is not None and recorded.control_plane is control_plane:
        return recorded
    return _SelectedControlPlane(control_plane, None)


def _control_plane_kind(control_plane: ControlPlane) -> str:
    """The log line's kind: the plane's ``kind``, else its class name."""
    return control_plane.kind or type(control_plane).__name__


async def _run_close(close: Callable[[], Awaitable[object]]) -> None:
    """Call ``close`` in the close task, so that an error raised at once lands there."""
    await close()


def _start_close(selected: _SelectedControlPlane) -> "asyncio.Task[None] | None":
    """Start closing a connection dialed through an address. A caller's own stays open.

    The close runs as its own task, so cancelling a caller never interrupts it. ``asyncio.run``
    cancels it when it returns unless it started while ``asyncio.run`` was cancelling the
    caller: such a close may finish in that cleanup and stays pending only if it outlasts it.
    """
    if selected.address is None:
        return None
    task = asyncio.ensure_future(_run_close(selected.control_plane.close))
    _closing.add(task)
    task.add_done_callback(_closing.discard)
    return task


def _log_close_failure(
    address: ControlPlaneAddress | None, task: "asyncio.Task[None]"
) -> None:
    """Log the failure of a close that no caller waits for any more, unless it ended cancelled."""
    if not task.cancelled() and task.exception() is not None:
        logger.warning(
            "closing the connection dialed through %s failed",
            address,
            exc_info=task.exception(),
        )


def _leave_running(
    task: "asyncio.Task[None]", address: ControlPlaneAddress | None
) -> None:
    """Let a close that no caller waits for run on, logging a failure unless it ends cancelled."""
    task.add_done_callback(partial(_log_close_failure, address))


async def _wait_for_close(selected: _SelectedControlPlane) -> bool:
    """Close a dialed connection, waiting for it at most ``_CLOSE_WAIT_S``.

    Returns False if the close outlasts the wait. Such a close, or one whose caller is cancelled (also in the loop step
    it ends in) or whose waiting coroutine is closed, runs on while its event loop runs and logs a failure unless it
    ends cancelled. One ending cancelled or in ``GeneratorExit`` while its caller waits raises ``RuntimeError``.
    """
    task = _start_close(selected)
    if task is None:
        return True
    try:
        await asyncio.wait({task}, timeout=_CLOSE_WAIT_S)
    except GeneratorExit:
        # A coroutine closed unfinished may have no loop left to run a done task's callback.
        if task.done():
            _log_close_failure(selected.address, task)
        else:
            _leave_running(task, selected.address)
        raise
    except BaseException:
        _leave_running(task, selected.address)
        raise
    if not task.done():
        _leave_running(task, selected.address)
        return False
    # The close's own cancel or GeneratorExit is not this caller's: report a failed close.
    if task.cancelled():
        raise RuntimeError(
            f"closing the connection dialed through {selected.address} was cancelled"
        )
    if isinstance(task.exception(), GeneratorExit):
        raise RuntimeError(
            f"closing the connection dialed through {selected.address} raised GeneratorExit"
        ) from task.exception()
    task.result()
    return True


async def _close_dialed(selected: _SelectedControlPlane) -> None:
    """Close a dialed connection, raising if the close outlasts its wait."""
    if not await _wait_for_close(selected):
        raise TimeoutError(
            f"closing the connection dialed through {selected.address} took over "
            f"{_CLOSE_WAIT_S:g}s"
        )


async def _close_dialed_quietly(selected: _SelectedControlPlane) -> None:
    """Close a dialed connection, only logging a slow close or a failure.

    The caller's own cancel or ``GeneratorExit`` and any ``KeyboardInterrupt`` or ``SystemExit``
    still propagate.
    """
    try:
        if not await _wait_for_close(selected):
            logger.warning(
                "closing the connection dialed through %s is still running after %gs",
                selected.address,
                _CLOSE_WAIT_S,
            )
    except (asyncio.CancelledError, GeneratorExit, KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        logger.warning(
            "closing the connection dialed through %s failed",
            selected.address,
            exc_info=True,
        )


async def _close_quietly(
    dialed: _SelectedControlPlane | None,
    kept: _SelectedControlPlane | None,
    service: "CoordinatorService | None" = None,
) -> None:
    """Close ``dialed`` unless ``kept`` holds its connection, then stop ``service``.

    Each runs even if an earlier one is cancelled, and failures only log, so the caller's
    selection, or its own error, stands.
    """
    async with contextlib.AsyncExitStack() as closing:
        if service is not None:
            closing.push_async_callback(_stop_service_quietly, service)
        if dialed is not None and (
            kept is None or dialed.control_plane is not kept.control_plane
        ):
            closing.push_async_callback(_close_dialed_quietly, dialed)


async def _stop_service_quietly(service: "CoordinatorService") -> None:
    """Stop a coordinator service, logging a failure so the caller's outcome stands."""
    try:
        await service.close()
    except Exception:
        logger.warning(
            "stopping the coordinator service on %s failed",
            service.address,
            exc_info=True,
        )


async def _close_carrier_quietly(
    strategy: TorchStoreStrategy | None, host: "CarrierVolumeHost | None", whose: str
) -> None:
    """Close a dropped client's volume handles and only log a failure so the caller's outcome stands."""
    try:
        if host is not None:
            await _close_carrier(strategy, host)
    except Exception:
        logger.warning("closing %s volume handles failed", whose, exc_info=True)


async def _close_volume_quietly(volume: "PublisherVolume") -> None:
    """Stop a publisher volume that no client took and only log a failure so the caller's error stands."""
    try:
        await volume.close()
    except Exception:
        logger.warning(
            "stopping routing volume=%s failed", volume.volume_id, exc_info=True
        )


def _log_control_plane(
    kind: str, address: ControlPlaneAddress | None, store_name: str, rank: str
) -> None:
    """Log which control plane this process uses, once per initialize or new client."""
    logger.info(
        "routing control_plane=%s address=%s store=%s rank=%s pid=%d",
        kind,
        "-" if address is None else address,
        store_name,
        rank,
        os.getpid(),
    )


async def initialize(
    num_storage_volumes: int = 1,
    strategy: TorchStoreStrategy | None = None,
    store_name: str = DEFAULT_TORCHSTORE_NAME,
    mesh: "ProcMesh | None" = None,
    client_type: ClientType = ClientType.STANDARD,
    *,
    control_plane: str | ControlPlane = "monarch",
    num_publishers: int | None = None,
) -> ControlPlaneAddress | None:
    """Initialize the TorchStore distributed storage system.

    Sets up storage volumes and controller. Must be called before any put/get operations.
    A routing store starts over on each call that its control plane's ``init`` accepts,
    as Monarch's coordinator does, dropping this process's cached client and closing a
    connection it dialed for the store, unless that connection is the plane passed in.
    If the plane refuses, this call raises and keeps both.
    Initialize other stores, and a store running a coordinator service, again only after ``shutdown``.
    Never overlap it with another ``initialize``, ``client`` or ``shutdown`` call for the
    same store in this process, or with an ``initialize`` or ``client`` given the same
    control plane.

    With ``control_plane="service"`` ranks dial ``$TORCHSTORE_TCPSTORE_HOST``, else this
    host's name, and each publisher's volume at ``$TORCHSTORE_CARRIER_HOST``, else its host's.
    The coordinator and volumes, like TCPStore, listen on every interface unauthenticated: any
    TCPStore client can read what their JSON hellos check before unpickling and can rewrite the
    addresses whose replies ranks unpickle, so trust the network.

    Args:
        num_storage_volumes (int): Number of storage volumes to create. Defaults to 1.
        strategy (TorchStoreStrategy, optional): Strategy for distributing tensors across volumes.
            Uses ControllerStorageVolumes if None and num_storage_volumes=1.
        store_name (str): Unique name for this store instance. Defaults to DEFAULT_TORCHSTORE_NAME.
        mesh (ProcMesh, optional): Monarch ProcMesh on which to spawn StorageVolumes.
            Where publishers serve their own volumes, as on ``"service"``, only its size is read.
        client_type: Client implementation to initialize. Routing clients lazily
            exchange state-dict layouts and construct routes.
        control_plane: Where a routing store's ranks share publishers and layouts.
            ``"monarch"`` spawns the coordinator actor, which ``client`` then finds
            with no ``control_plane``, and ``"service"`` serves it from this process
            beside a TCPStore. An instance, which serves only this store, is
            initialized here for later clients in this process, and log lines name
            it by ``kind`` or class name.
        num_publishers: The publisher count, in place of ``mesh``, where publishers serve
            their own volumes, so any launcher can start them: publisher ``r`` passes
            ``rank=r`` to ``ts.client``, for ``r`` from 0 to ``num_publishers - 1``.

    Returns:
        For ``"service"`` the address that ranks in other processes pass to ``ts.client``,
        otherwise None.

    Raises:
        RuntimeError: If num_storage_volumes > 1 but no strategy is provided, or a routing
            store gets no strategy, or neither mesh nor num_publishers, or, for a routing
            initialize in the process whose earlier call started it, the store still runs
            that coordinator service (checked before anything spawns and as it records).
            Also if control_plane is a plane dialed from a service's address.
        ConnectionError: If the service's advertised host does not answer, about
            60 s in.
        TypeError: If control_plane is neither a string nor a ``ControlPlane``, or
            publishers serve their own volumes and strategy is not a plain one, or
            num_publishers is not an int.
        ValueError: If control_plane is a string other than ``"monarch"`` or
            ``"service"``, anything but ``"monarch"`` for a standard store, an
            instance that serves another store, or publishers serve their own volumes
            over a transport they do not carry. Also if num_publishers is below 1, comes
            with mesh, or comes where Monarch spawns the volumes or for a standard store,
            or ``"service"`` gets more than 16,384 publishers, four times the design's
            largest scale, or a ``store_name`` too long for the TCPStore keys it publishes
            (over about 8,100 UTF-8 bytes).

    Example:
        >>> import torchstore as ts
        >>> await ts.initialize(num_storage_volumes=4, strategy=LocalRankStrategy()) # uses default namespace.
        >>> >>> await ts.initialize("my_custom_store")
    """
    caller_control_plane = _caller_control_plane(control_plane)
    if num_publishers is not None:
        _check_int("num_publishers", num_publishers, 1)
    if client_type == ClientType.ROUTING:
        if num_publishers is not None:
            if not _hosts_volumes(caller_control_plane):
                raise _hosted_only("num_publishers")
            if mesh is not None:
                raise ValueError("pass mesh or num_publishers, not both")
        if strategy is None or (mesh is None and num_publishers is None):
            # Upstream's text comes first for callers that match it.
            raise RuntimeError(
                "routing mode requires both mesh and strategy, or num_publishers in place"
                " of mesh where publishers serve their own volumes"
            )
        return await _initialize_routing(
            mesh, strategy, store_name, caller_control_plane, num_publishers
        )
    if caller_control_plane is not None:
        raise ValueError("standard stores take only control_plane='monarch'")
    if num_publishers is not None:
        raise ValueError("standard stores take no num_publishers")

    if num_storage_volumes == 1 and strategy is None:
        strategy = ControllerStorageVolumes()
    elif strategy is None:
        raise RuntimeError(
            "Must specify controller strategy if num_storage_volumes > 1"
        )

    backend = get_backend("monarch")
    # TODO: monarch doesn't support nested actors yet, so we need to spawn storage volumes here
    # ideally this is done in the controller.init
    if isinstance(strategy, ControllerStorageVolumes):
        storage_volumes = await backend.controller_volume(strategy.get_volume_id)
    else:
        storage_volumes = await backend.spawn_volumes(
            mesh, num_storage_volumes, strategy.get_volume_id
        )

    controller = await _controller(store_name)
    await controller.init.call(
        strategy=strategy,
        num_storage_volumes=num_storage_volumes,
        storage_volumes=storage_volumes,
    )
    with _selection_lock:
        # Client and shutdown reach this controller, unless a routing record of the name stays.
        _released_stores.discard(store_name)
        _forked_stores.pop(store_name, None)
        _attempted.discard(store_name)
        _standard_stores.add(store_name)
    return None


async def _initialize_routing(
    mesh: "ProcMesh | None",
    strategy: TorchStoreStrategy,
    store_name: str,
    control_plane: ControlPlane | Literal["service"] | None,
    num_publishers: int | None,
) -> ControlPlaneAddress | None:
    """Spawn trainer volumes unless publishers serve their own, then initialize the control plane."""
    with _selection_lock:
        if store_name in _owned_service_map:
            raise _service_still_runs(store_name)
        if isinstance(control_plane, ControlPlane):
            _check_one_store(store_name, control_plane)
    namespace = _routing_namespace(RankRole.PUBLISHER, None)
    if _hosts_volumes(control_plane):
        # Before anything starts, as the volume servers that ranks start must honor it.
        get_backend("service").check_strategy(strategy)
    else:
        # Volumes are Monarch actors here, also under a caller's own control plane.
        volumes = await get_backend("monarch").spawn_volumes(
            mesh, 1, partial(_routing_volume_id, namespace)
        )
        await strategy.set_storage_volumes(volumes)

    if num_publishers is None:
        assert mesh is not None
        num_publishers = mesh.size()
    if isinstance(control_plane, str) and num_publishers > _MAX_SERVICE_PUBLISHERS:
        raise ValueError(
            f"'service' takes at most {_MAX_SERVICE_PUBLISHERS} publishers, got"
            f" {num_publishers}: four times the design's largest scale"
        )
    publishers = {f"{namespace}/{rank}" for rank in range(num_publishers)}

    selected: _SelectedControlPlane | None = None
    service: "CoordinatorService | None" = None
    if isinstance(control_plane, str):
        selected, service = await _start_service(store_name, publishers, strategy)
        kind = "service"
    elif control_plane is None:
        coordinator = await get_backend("monarch").coordinator(store_name)
        await coordinator.init(
            publishers=publishers,
            strategy=strategy,
        )
        kind = _control_plane_kind(coordinator)
    else:
        await control_plane.init(publishers=publishers, strategy=strategy)
        kind = _control_plane_kind(control_plane)
    address = None if selected is None else selected.address
    replaced: _SelectedControlPlane | None = None
    with _selection_lock:
        # Only an initialize overlapping a service initialize of the store, which the docstring forbids, finds one.
        overlapped = store_name in _owned_service_map
        if not overlapped:
            if isinstance(control_plane, ControlPlane):
                selected = _given(store_name, control_plane)
            replaced = _select_control_plane(store_name, selected)
            if service is not None:
                _owned_service_map[store_name] = service
            # The store starts over, so a client cached on its old state goes too.
            _drop_client(store_name)
    if overlapped:
        await _close_quietly(selected, None, service)
        raise _service_still_runs(store_name)
    _log_control_plane(
        kind, None if selected is None else selected.address, store_name, "-"
    )
    await _close_quietly(replaced, selected)
    return address


def _service_still_runs(store_name: str) -> RuntimeError:
    return RuntimeError(
        f"routing store {store_name!r} still runs its coordinator service:"
        f" call ts.shutdown({store_name!r}) before initializing it again"
    )


async def _start_service(
    store_name: str, publishers: set[str], strategy: TorchStoreStrategy
) -> tuple[_SelectedControlPlane, "CoordinatorService"]:
    """Start a coordinator service and initialize it through this process's own handle.

    A failure, cancellation included, closes both and leaves every record unchanged.
    """
    service = await get_backend("service").start_coordinator(store_name)
    selected = None
    try:
        handle = await service.connect()
        selected = _SelectedControlPlane(handle, service.address)
        await handle.init(publishers=publishers, strategy=strategy)
    except BaseException:
        await _close_quietly(selected, None, service)
        raise
    return selected, service


async def shutdown(store_name: str = DEFAULT_TORCHSTORE_NAME) -> None:
    """Shutdown and cleanup a TorchStore instance.

    Gracefully shuts down all storage volumes and controllers associated with the
    store. For SPMD-initialized stores, this delegates to the session cleanup
    path created by ``torchstore.spmd.initialize()``, which also stops Job API
    sidecars and tears down the Monarch host mesh.

    For a routing store whose control plane this process initialized, was given or
    connected to, or whose dialed record a fork dropped, it drops only that record and
    the client, closes its clients' volume handles and a connection made through an
    address, and stops a service ``initialize`` started here and the volume this process
    serves, giving a call already running 30s. A call stuck in a Gloo transfer runs past
    that grace, and this process's exit waits for it up to ``TORCHSTORE_GLOO_INIT_TIMEOUT``
    (120s by default). A handshake its volume runs after the requester gave up holds exit only
    for the exit hooks' bounded waits, after which it can abort the process if it returns while
    the interpreter finalizes. Another shutdown then drops only a client, unless an SPMD session of
    the name tears its store down. A caller's control plane stays open, and Monarch volumes
    stay with their mesh. A store whose only ``ts.client`` here is still connecting through
    an address or a plane, or failed once it began to, asks Monarch nothing however often it
    is shut down. Never overlap it with another ``initialize``, ``client`` or ``shutdown``
    call for the same store in this process except with such a connecting ``client``. That
    client raises ``RuntimeError`` if this shutdown released the store's record it connects through.

    Args:
        store_name (str): Name of the store to shutdown. Defaults to DEFAULT_TORCHSTORE_NAME.

    Raises:
        TimeoutError: If closing a connection made through an address takes over
            5s. The record and client are dropped by then, and the close runs on as a
            task of this event loop.
        RuntimeError: If that close ends cancelled or in ``GeneratorExit``, after the same drop.
        Exception: Whatever stopping this process's publisher volume, closing a client's
            volume handles, that connection or the service raises, once every close ran.

    Example:
        >>> import torchstore as ts
        >>> await ts.shutdown()  # Shutdown default store
        >>> await ts.shutdown("my_custom_store")
    """
    if await _shutdown_selected(store_name):
        return

    session = _spmd_state_map.pop(store_name, None)
    if session is not None:
        try:
            await session.shutdown()
        finally:
            reset_client(store_name)
        return
    if store_name in _released_stores:
        # Released here before: Monarch's lookup would spawn a controller for the name.
        reset_client(store_name)
        return

    controller = await _controller(store_name)
    task = _current_task()
    try:
        await controller.teardown.call()
    finally:
        _locked_cleanup(partial(_standard_stores.discard, store_name), _off_task(task))
        reset_client(store_name)


async def _shutdown_selected(store_name: str) -> bool:
    """Release a store's record, client, owned service and publisher volume, then close them.

    Volumes stop, then every client's volume handles, the connection and the service, each even
    if an earlier close fails. Returns False, logging any failure, if it found only dropped clients
    and no client this process builds, or failed to build, through an address or a plane.
    """
    # None before the service backend loads: this process serves no volume yet.
    backend = loaded_backend("service")
    # The caller's own error if it shuts down while handling one. AsyncExitStack drops it.
    outer = sys.exc_info()[1]
    with _selection_lock:
        selected = _control_plane_map.pop(store_name, None)
        # Owned only beside a record of the store, which no step removes while the service stays.
        service = _owned_service_map.pop(store_name, None)
        forked = _forked_stores.pop(store_name, None)
        volumes = [] if backend is None else backend.stop_publisher_volumes(store_name)
        # A session or a standard store here outranks the marks, as for released stores.
        attempted = (
            store_name not in _spmd_state_map
            and store_name not in _standard_stores
            and (store_name in _building or store_name in _attempted)
        )
        found = not (selected is None and forked is None and not volumes)
        released = found or attempted
        carriers = []
        # A build's mark stays for later shutdowns. One that alone recorded nothing records no
        # released store either, and a role-less call then still looks Monarch up.
        if found:
            _released_stores.add(store_name)
        if released:
            cached = _client_carrier(_local_clent_map.pop(store_name, None))
            if cached is not None:
                carriers.append(cached)
        # Taken after the client and in place. No host is then lost to a reset_client that moved it
        # here from a finalizer or appends from another thread meanwhile.
        kept = _dropped_carriers.get(store_name, [])
        taken = kept[:]
        del kept[: len(taken)]
        carriers += [(strategy_ref(), host) for strategy_ref, host in taken]
    try:
        async with contextlib.AsyncExitStack() as closing:
            if service is not None:
                closing.push_async_callback(service.close)
            if selected is not None:
                closing.push_async_callback(_close_dialed, selected)
            for strategy, host in carriers:
                closing.push_async_callback(_close_carrier, strategy, host)
            for volume in volumes:
                closing.push_async_callback(volume.close)
    except BaseException as error:
        _restore_context(error, outer)
        if released or not isinstance(error, Exception):
            raise
        # The store's own teardown runs next, which a failed close must not skip.
        logger.warning(
            "closing dropped clients of store %r failed", store_name, exc_info=True
        )
    return released


def _restore_context(error: BaseException, outer: BaseException | None) -> None:
    """Link the end of ``error``'s context chain to ``outer`` like a raise outside AsyncExitStack.

    A chain that cycles (as AsyncExitStack makes when two callbacks raise one error) stays as it is.
    """
    seen = {id(error)}
    while error.__context__ is not None:
        error = error.__context__
        if id(error) in seen:
            return
        seen.add(id(error))
    if outer is not None and error is not outer:
        error.__context__ = outer


async def _close_carrier(
    strategy: TorchStoreStrategy | None, host: "CarrierVolumeHost"
) -> None:
    """Close a released client's handles to carrier volumes, then forget its Gloo pairs.

    A collected client's strategy is None, and its pairs' groups went with it.
    """
    volume_ids = list(host.volume_ids())
    try:
        await host.close()
    finally:
        if strategy is not None:
            get_backend("service").forget_context_pairs(
                volume_ids, strategy.transport_context
            )


def _volume_host(strategy: object) -> "CarrierVolumeHost | None":
    """A strategy's carrier volume host, if it resolves volumes through one.

    None before the service backend loads, which a client sharing volume records loads first.
    """
    service = loaded_backend("service")
    return None if service is None else service.volume_host(strategy)


def _client_carrier(
    client: LocalClient | None,
) -> "tuple[TorchStoreStrategy, CarrierVolumeHost] | None":
    """A client's strategy and its carrier volume host, if it resolves volumes through one."""
    strategy = getattr(client, "strategy", None)
    host = _volume_host(strategy)
    if strategy is None or host is None:
        return None
    return strategy, host


def _keep_dropped(store_name: str, client: LocalClient | None) -> None:
    """Hold a dropped client's carrier volume host in one append until ``shutdown`` closes it.

    Its strategy is held weakly, so collecting the client frees what it read, and a later
    client's build closes the host once a newer dropped client is collected too.
    """
    carrier = _client_carrier(client)
    if carrier is not None:
        strategy, host = carrier
        _dropped_carriers.setdefault(store_name, []).append(
            (weakref.ref(strategy), host)
        )


async def _close_collected_carriers(store_name: str) -> None:
    """Close the volume hosts of the store's dropped clients that were collected, but the newest.

    The newest keeps its connections for the client being built to share. Each close runs even
    if an earlier one fails or is cancelled, and a failure only logs.
    """
    with _selection_lock:
        kept = _dropped_carriers.get(store_name, [])
        collected = [entry for entry in kept if entry[0]() is None][:-1]
        for entry in collected:
            kept.remove(entry)
    async with contextlib.AsyncExitStack() as closing:
        for _, host in collected:
            closing.push_async_callback(
                _close_carrier_quietly, None, host, "a collected client's"
            )


def _drop_client(store_name: str) -> None:
    """Drop the store's cached client and forget its Gloo pairs to carrier volumes.

    It needs no selection lock: each step changes one map at once or locks its own state, and shutdown
    takes what it finds in place. A shutdown on another thread between its steps leaves this host to the next.
    """
    dropped = _local_clent_map.pop(store_name, None)
    carrier = _client_carrier(dropped)
    if carrier is not None:
        strategy, host = carrier
        get_backend("service").forget_context_pairs(
            host.volume_ids(), strategy.transport_context
        )
    _keep_dropped(store_name, dropped)


def reset_client(store_name: str = DEFAULT_TORCHSTORE_NAME) -> None:
    """Reset the cached or installed client for a given store and forget its Gloo pairs to carrier volumes.

    Its volume handles close at the store's next ``shutdown`` or at a later build once it and a newer dropped
    client are collected. A finalizer or a signal handler may call it during ``initialize``, ``client`` or ``shutdown``.
    """
    _drop_client(store_name)


# What the hook below drops in a forked child, kept alive until the child exits so that the
# rendezvous module's hook (which runs after it) still finds each served TCPStore and keeps it.
# The service module's hook still frees each handle's TCPStore connection, which leaves the
# parent's connection working.
_inherited: list[object] = []


def _after_fork_in_child(lock: threading.Lock) -> None:
    """Free the fork's selection lock and drop dialed records, service handles' records, owned services and dropped clients.

    The parent's stay the parent's. Until a selection, a standard initialize or a shutdown of such a store, ``client``
    raises for it without ``address`` or ``control_plane``. Other planes a caller passed stay recorded with their clients.
    """
    lock.release()
    for carriers in _dropped_carriers.values():
        _inherited.extend(carriers)
    _dropped_carriers.clear()
    _building.clear()
    _building_ranks.clear()
    # None before the service backend loads: no handle exists yet.
    service = loaded_backend("service")
    for store_name, selected in list(_control_plane_map.items()):
        address = selected.address
        if address is None and service is not None:
            address = service.service_address(selected.control_plane)
        if address is not None:
            del _control_plane_map[store_name]
            _inherited.append((selected, _local_clent_map.pop(store_name, None)))
            _forked_stores[store_name] = address
    _inherited.extend(_owned_service_map.values())
    _owned_service_map.clear()


# Hold the lock across fork so a child copies no half-made update unless a signal handler raises
# while the before hook waits: CPython then forks anyway and the after hooks free a lock the before
# hook never took. Otherwise a fork from code under the lock on its holder's thread waits for good.
# Each hook takes or frees the lock it was registered with so a reload that registers them again uses two.
os.register_at_fork(
    before=_selection_lock.acquire,
    after_in_parent=_selection_lock.release,
    after_in_child=partial(_after_fork_in_child, _selection_lock),
)


def _start_spmd_session(store_name: str, session: "_SPMDSession") -> None:
    """Register an SPMD session, which takes the name as a standard initialize does.

    It outranks a failed routing client's mark only while it runs. This step drops the mark too.
    """
    with _selection_lock:
        _attempted.discard(store_name)
        _spmd_state_map[store_name] = session


async def _controller(store_name: str = DEFAULT_TORCHSTORE_NAME) -> "Controller":
    """Get a controller handle for interacting with the store."""
    session = _spmd_state_map.get(store_name)
    if session is not None:
        return session.controller
    return await get_backend("monarch").controller(store_name)


async def client(
    store_name: str = DEFAULT_TORCHSTORE_NAME,
    *,
    role: RankRole | str | None = None,
    group: int | None = None,
    address: ControlPlaneAddress | None = None,
    control_plane: ControlPlane | None = None,
    rank: int | None = None,
) -> LocalClient:
    """Get a local client handle for interacting with the store.

    Returns a cached LocalClient instance that provides the interface for put/get operations.
    Never overlap it with an ``initialize`` or ``shutdown`` call for the same store in this
    process other than a ``shutdown`` while it is the store's only client here and connects
    through an address or a plane. It then raises ``RuntimeError`` if that shutdown released
    the store's record it connects through. Nor overlap a first call with ``address`` or
    ``control_plane`` with another ``client`` call for the store unless both pass one
    address and neither passes ``control_plane``. Those share one connection and its client
    (unless ``reset_client`` drops it in between), and of two for different rank IDs one
    raises. Concurrent first calls without a role share one client.

    Args:
        store_name (str): Name of the store to get a client for. Defaults to DEFAULT_TORCHSTORE_NAME.
        role: Set on a rank of a routing store to get a routing client:
            ``RankRole.PUBLISHER`` for ranks on the publisher mesh and
            ``RankRole.REQUESTER`` for ranks on a requester mesh. State-dict
            put/get operations register its layout lazily.
        group: Requester-mesh index used to namespace requester ranks. Required
            for requesters, rejected for publishers.
        address: Reaches a routing store's control plane from any rank, as the
            driver hands it out. It must name this store and needs ``role``. Later
            clients in this process reuse its connection, through this address or an
            equal one, until ``shutdown`` or another selection replaces and closes it.
            A forked child dials its own.
        control_plane: A routing store's own control plane instance, which serves only
            that store and needs ``role``. It wins over ``address``, then this process's
            record (kept across ``reset_client``), then the Monarch lookup, which needs no
            argument: unlike ``initialize`` this takes no ``"monarch"``. A cached client
            refuses other instances, even a copy that an actor call delivers.
        rank: This process's rank on its mesh, from 0, so any launcher can start ranks
            where publishers serve their own volumes. It needs ``role``. In a Monarch endpoint it
            defaults to the call's position (``current_rank``), the actor's rank only on a
            whole-mesh call, so pass it otherwise, a driver included. Other planes refuse it.

    Returns:
        LocalClient: A client instance for performing storage operations.

    Raises:
        TypeError: If address or control_plane has the wrong type, the address's class
            is not a dataclass of its own or has ``eq=False`` without an ``__eq__`` of its
            own, its ``connect`` returns something other than a ``ControlPlane``, rank is
            not an int, or a publisher's control plane shares volume records but resolves
            volumes another way.
        ValueError: If address, control_plane or rank comes without role, the address names
            another store, control_plane (else address) or rank differs from the cached
            client's, control_plane serves another store, or the first client of a store with a
            recorded control plane, or one shut down here with no SPMD session of its name, has
            no role. Also if rank is negative or comes where ranks come from Monarch, another
            call in this process builds the store's client for another rank or this process
            serves another publisher rank's volume of the store, or publishers serve their own
            volumes and a call outside an actor's context passes no rank. A connection this call
            dialed closes.
        RuntimeError: In a forked child, if a call without address or control_plane
            needs a store whose connection the fork dropped. With any selection: if a
            shutdown stops a publisher's volume as it starts, another
            thread's start of it fails, or the store holds another record for that
            volume, which closes the new one.
        KeyError: If a publisher's rank is not among the store's publishers. The
            volume it started closes.
        TimeoutError: If a publisher cannot write its volume's record in time, which
            closes the volume as a cancel during that write does, or another call's start
            of that volume takes over 150 s, which closes that start only if its event loop
            has closed.
        ConnectionError: If the TCPStore connection drops as a publisher writes its
            volume's record. The volume closes. That write, like a timed-out or
            cancelled one, can still land and refuse a retry: start the store over.

    Example:
        >>> store_client = await client()
        >>> await store_client.put("my_key", tensor)
    """
    if control_plane is not None and not isinstance(control_plane, ControlPlane):
        raise TypeError(
            f"control_plane must be a ControlPlane instance, got {control_plane!r}"
        )
    if address is not None and not isinstance(address, ControlPlaneAddress):
        raise TypeError(f"address must be a ControlPlaneAddress, got {address!r}")
    if address is not None and not {"__dataclass_fields__", "__eq__"} <= set(
        vars(type(address))
    ):
        # The decorator skipped this class, or gave it eq=False without an __eq__ of its own.
        raise TypeError(
            f"{type(address).__name__} must be a dataclass with eq=True, as "
            "ControlPlaneAddress is, so that addresses compare by their own fields"
        )
    if role is None and (address is not None or control_plane is not None):
        raise ValueError("address and control_plane are for routing clients: pass role")
    wanted = None
    if rank is not None:
        if role is None:
            raise ValueError("rank is for routing clients: pass role")
        _check_int("rank", rank, 0)
        wanted = f"{_routing_namespace(RankRole(role), group)}/{rank}"
    if address is not None and address.store_name != store_name:
        raise ValueError(
            f"address is for store {address.store_name!r}, not {store_name!r}"
        )
    with _selection_lock:
        if control_plane is not None:
            _check_one_store(store_name, control_plane)
        cached = _local_clent_map.get(store_name)
        if cached is not None:
            _check_cached_client(store_name, address, control_plane, wanted)
            return cached
        if role is None:
            _check_unrecorded(store_name)
        elif wanted is not None:
            _reserve_rank(store_name, wanted)
        # The build fails if a shutdown on another thread releases it before the build reads it.
        seen = _control_plane_map.get(store_name)

    if role is not None:
        task = asyncio.current_task()
        # One through an address or a plane is marked for shutdown while it builds.
        build = (
            _routing_client
            if address is None and control_plane is None
            else _building_client
        )
        try:
            return await build(
                store_name, RankRole(role), group, address, control_plane, rank, seen
            )
        finally:
            if wanted is not None:
                _locked_cleanup(
                    partial(_release_rank, store_name, wanted), _off_task(task)
                )
    controller = await _controller(store_name)
    controller_strategy = await controller.get_controller_strategy.call_one()
    local_client = LocalClient(
        directory=get_backend("monarch").directory(controller),
        strategy=controller_strategy,
        controller=controller,
    )
    with _selection_lock:
        # Racing first calls share the client cached first: a process's Gloo pair to a volume ID
        # serves one live client, so a second would be refused its tensor transfers.
        local_client = _local_clent_map.setdefault(store_name, local_client)
        _attempted.discard(store_name)
    return local_client


async def _building_client(
    store_name: str,
    role: RankRole,
    group: int | None,
    address: ControlPlaneAddress | None,
    control_plane: ControlPlane | None,
    rank: int | None,
    seen: _SelectedControlPlane | None,
) -> LocalClient:
    """Build a routing client through an address or a plane, marked as running for ``shutdown``.

    A failed one marks the store in ``_attempted``.
    """
    # Checked first, so a call refused for its arguments marks nothing.
    _routing_namespace(role, group)
    if (
        rank is not None
        and control_plane is not None
        and not _hosts_volumes(control_plane)
    ):
        raise _hosted_only("rank")
    task = asyncio.current_task()
    with _selection_lock:
        _building.add(store_name)
    try:
        return await _routing_client(
            store_name, role, group, address, control_plane, rank, seen
        )
    except BaseException:
        _locked_cleanup(partial(_attempted.add, store_name), _off_task(task))
        raise
    finally:
        _locked_cleanup(partial(_building.discard, store_name), _off_task(task))


def _locked_cleanup(update: Callable[[], None], closed: bool) -> None:
    """Run ``update`` under ``_selection_lock`` and never wait for it if the GC closed the call.

    The collector can close a build or a shutdown inside a locked step of this thread. Then ``update`` runs
    here if the lock is free, else on a thread once it frees, and not at all where no thread starts (as the
    interpreter finalizes or the OS refuses one), which leaves a build running and its rank claimed.
    """
    if not closed:
        with _selection_lock:
            update()
    elif _selection_lock.acquire(blocking=False):
        try:
            update()
        finally:
            _selection_lock.release()
    else:
        start_daemon_thread(
            partial(_locked_cleanup, update, False), "torchstore-selection-update"
        )


def _reserve_rank(store_name: str, wanted: str) -> None:
    """Count a build of the store's client as rank ID ``wanted``, refusing one while another runs.

    The caller holds ``_selection_lock``. Two builds for different ranks would each serve a volume
    or share one client of the wrong rank.
    """
    building = _building_ranks.setdefault(store_name, [])
    other = next((rank_id for rank_id in building if rank_id != wanted), None)
    if other is not None:
        raise ValueError(
            f"another ts.client call in this process builds store {store_name!r} as"
            f" {other}, not {wanted}"
        )
    building.append(wanted)


def _release_rank(store_name: str, wanted: str) -> None:
    """Drop one count of builds as ``wanted``. The caller holds ``_selection_lock``."""
    building = _building_ranks.get(store_name)
    if building is not None and wanted in building:
        building.remove(wanted)
        if not building:
            del _building_ranks[store_name]


def _check_unrecorded(store_name: str) -> None:
    """Refuse a role-less first client for a store whose plane this process selected.

    The caller holds ``_selection_lock``.
    """
    _check_not_forked(store_name)
    released = store_name in _released_stores and store_name not in _spmd_state_map
    if store_name in _control_plane_map or released:
        raise ValueError(
            f"routing store {store_name!r} needs role on its first ts.client call"
        )


def _check_not_forked(store_name: str) -> None:
    """Refuse a store whose dialed record a fork dropped, before any Monarch lookup.

    The caller holds ``_selection_lock``.
    """
    address = _forked_stores.get(store_name)
    if address is not None:
        raise RuntimeError(
            f"routing store {store_name!r} on {address}: its control plane was "
            "inherited across fork. Connect again from the address (address=)"
        )


def _check_cached_client(
    store_name: str,
    address: ControlPlaneAddress | None,
    control_plane: ControlPlane | None,
    rank: str | None,
) -> None:
    """Refuse a control plane other than the cached client's, else another address or rank ID."""
    selected = _control_plane_map.get(store_name)
    if control_plane is not None:
        if selected is None or selected.control_plane is not control_plane:
            raise ValueError(
                f"the cached client for store {store_name!r} uses another control "
                "plane instance. Later calls in this process can omit control_plane= "
                "(an actor call delivers a copy), or call ts.reset_client to switch"
            )
    elif address is not None and (selected is None or selected.address != address):
        raise ValueError(
            f"the cached client for store {store_name!r} did not connect through "
            f"{address}. Call ts.reset_client to replace it"
        )
    if rank is not None:
        cached = _local_clent_map[store_name]
        if not isinstance(cached, RoutingClient) or not _hosts_volumes(
            cached.control_plane
        ):
            raise _hosted_only("rank")
        if cached.rank_id != rank:
            # A publisher volume outlives reset_client, so only shutdown lets this process serve another.
            remedy = (
                "Call ts.shutdown to release it"
                if rank.split("/")[0] == RankRole.PUBLISHER.value
                and get_backend("service").serves_another_publisher(store_name, rank)
                else "Call ts.reset_client to replace it"
            )
            raise ValueError(
                f"the cached client for store {store_name!r} is {cached.rank_id}, not"
                f" {rank}. {remedy}"
            )


async def _routing_client(
    store_name: str,
    role: RankRole,
    group: int | None,
    address: ControlPlaneAddress | None,
    control_plane: ControlPlane | None,
    given_rank: int | None,
    seen: _SelectedControlPlane | None,
) -> LocalClient:
    """Build and cache a routing client on the first plane in ``client``'s order.

    ``seen`` is the store's record as ``client`` began. The build fails once another thread's shutdown released it.
    """
    with _selection_lock:
        recorded = _control_plane_map.get(store_name)
        if seen is not None and recorded is not seen:
            raise _lost_record(store_name, seen)
        # Checked in the read's locked step: a selection drops the marker as it records the store.
        if control_plane is None and address is None and recorded is None:
            _check_not_forked(store_name)
        given = None if control_plane is None else _given(store_name, control_plane)
    if control_plane is None and address is None and recorded is None:
        if given_rank is not None:
            raise ValueError(
                "rank needs the store's address (address=) or control plane"
                f" (control_plane=): store {store_name!r} has no control plane in this"
                " process, and on Monarch's coordinator ranks come from Monarch"
            )
        return await _monarch_routing_client(store_name, role, group)
    # Check role and group first, so a bad call dials nothing.
    namespace = _routing_namespace(role, group)
    if given is not None:
        selected = given
    elif address is not None and (recorded is None or recorded.address != address):
        dialed = await address.connect()
        if not isinstance(dialed, ControlPlane):
            raise TypeError(
                f"{type(address).__name__}.connect() returned {dialed!r}, "
                "not a ControlPlane"
            )
        selected = _SelectedControlPlane(dialed, address)
    else:
        # The record, also when this process already connected through ``address``.
        assert recorded is not None
        selected = recorded
    try:
        rank = _client_rank(namespace, given_rank, selected.control_plane)
        if given_rank is None:
            # Claimed once the plane is known (after any dial) so a racing build refuses before it reads the strategy.
            with _selection_lock:
                _check_racing_rank(store_name, rank)
                _reserve_rank(store_name, rank)
    except Exception:
        if selected is not recorded:
            await _close_dialed_quietly(selected)
        raise
    if given_rank is not None:
        return await _build_routing_client(store_name, role, rank, selected, recorded)
    task = asyncio.current_task()
    try:
        return await _build_routing_client(store_name, role, rank, selected, recorded)
    finally:
        _locked_cleanup(partial(_release_rank, store_name, rank), _off_task(task))


def _check_racing_rank(store_name: str, rank: str) -> None:
    """Refuse a build as ``rank`` once the store's client of another rank ID is cached.

    The caller holds ``_selection_lock``.
    """
    cached = _local_clent_map.get(store_name)
    if isinstance(cached, RoutingClient) and cached.rank_id != rank:
        raise ValueError(
            f"another ts.client call in this process built store {store_name!r} as"
            f" {cached.rank_id}, not {rank}"
        )


async def _build_routing_client(
    store_name: str,
    role: RankRole,
    rank: str,
    selected: _SelectedControlPlane,
    recorded: _SelectedControlPlane | None,
) -> LocalClient:
    """Read the strategy and start a publisher's volume where the plane shares volume records.

    Then cache a client on ``selected``, or share a racing call's cached client or record as ``_record_client`` does.
    """
    strategy: TorchStoreStrategy | None = None
    volume = None
    task = asyncio.current_task()
    try:
        await _close_collected_carriers(store_name)
        # A plane sharing volume records reaches volumes through the service backend's hosts.
        service = (
            get_backend("service") if _hosts_volumes(selected.control_plane) else None
        )
        strategy = await selected.control_plane.strategy()
        if role == RankRole.PUBLISHER and service is not None:
            volume = await _start_volume(service, store_name, rank, selected, strategy)
    except Exception:
        if _off_task(task):
            # The GC is closing an abandoned build, maybe in a locked step of this thread.
            raise
        if selected is not recorded:
            # The call's own error propagates, not the close's.
            await _close_dialed_quietly(selected)
        raise
    except GeneratorExit:
        # A coroutine closed unfinished may have no running loop to close on.
        raise
    except BaseException:
        if _off_task(task):
            raise
        if selected is not recorded:
            # A cancel or a crash: waiting for the close would hold the caller past it.
            task = _start_close(selected)
            if task is not None:
                _leave_running(task, selected.address)
        raise
    with _selection_lock:
        # A shutdown, or a start for another store of this name, stopped the volume. A shutdown
        # also releases the record this build read.
        step = None
        if (volume is None or volume.current()) and (
            selected is not recorded or _control_plane_map.get(store_name) is recorded
        ):
            step = _record_client(store_name, rank, role, selected, recorded, strategy)
    if step is None:
        async with contextlib.AsyncExitStack() as discarding:
            if selected is not recorded:
                discarding.push_async_callback(_close_dialed_quietly, selected)
            if strategy is not None:
                # No client took it, and its host's handles close nothing when collected.
                discarding.push_async_callback(
                    _close_carrier_quietly,
                    strategy,
                    _volume_host(strategy),
                    "a discarded client's",
                )
            if volume is not None and volume.stop():
                # It started after a shutdown released the record this build read, so no stop took it.
                discarding.push_async_callback(_close_volume_quietly, volume)
        if selected is not recorded:
            # A shutdown or a start for another store of this name stopped its volume.
            raise _lost_build(
                store_name,
                _store_address(selected.control_plane, selected.address),
                "Call ts.client again, or restart the store if it refuses the volume"
                " as already registered",
            )
        # Through a record a shutdown released: it stopped the volume, or the stack above did for one started later.
        raise _lost_record(store_name, selected)
    routing_client, selected, closing, built = step
    try:
        if built:
            _log_control_plane(
                _control_plane_kind(selected.control_plane),
                selected.address,
                store_name,
                rank,
            )
            if volume is not None:
                _log_volume(volume)
    finally:
        async with contextlib.AsyncExitStack() as closes:
            if strategy is not None and routing_client.strategy is not strategy:
                # A racing call's client is shared, and collecting this host closes nothing.
                closes.push_async_callback(
                    _close_carrier_quietly,
                    strategy,
                    _volume_host(strategy),
                    "a discarded client's",
                )
            closes.push_async_callback(_close_quietly, closing, selected)
    return routing_client


def _current_task() -> "asyncio.Task[Any] | None":
    """The running task, or None for a coroutine stepped outside any event loop."""
    try:
        return asyncio.current_task()
    except RuntimeError:
        return None


def _lost_record(store_name: str, selected: _SelectedControlPlane) -> RuntimeError:
    """The error of a build through the store's record that a shutdown released."""
    return _lost_build(
        store_name,
        _store_address(selected.control_plane, selected.address),
        "Call ts.initialize again or pass address=, and restart the store if it"
        " refuses the volume as already registered",
    )


def _lost_build(
    store_name: str, address: ControlPlaneAddress | None, remedy: str
) -> RuntimeError:
    """The error of a build whose publisher volume a shutdown or another store's start stopped."""
    where = "" if address is None else f" on {address}"
    return RuntimeError(
        f"routing store {store_name!r}{where} was shut down or replaced while this "
        f"client was being built. {remedy}"
    )


async def _start_volume(
    service: ServiceBackend,
    store_name: str,
    volume_id: str,
    selected: _SelectedControlPlane,
    strategy: TorchStoreStrategy,
) -> "PublisherVolume":
    """Serve this publisher's volume, or adopt the one this process serves, into ``strategy``."""
    plane = selected.control_plane
    assert isinstance(plane, VolumeRecords)
    host = service.volume_host(strategy)
    if host is None:
        raise TypeError(
            f"{type(plane).__name__} shares volume records, so its strategy() must"
            " resolve volumes through a CarrierVolumeHost, not"
            f" {type(_installed_volume_host(strategy)).__name__}"
        )
    volume = await service.start_publisher_volume(
        volume_id,
        store_name=store_name,
        selection=selected,
        register=plane.register_volume,
    )
    # One that a shutdown stopped as it served has no record, and the step fails the build.
    if volume.record is not None:
        host.adopt(volume_id, volume.record)
    return volume


def _log_volume(volume: "PublisherVolume") -> None:
    """Log where this publisher's volume listens, once per new client."""
    address = volume.address
    assert address is not None
    logger.info(
        "routing volume=%s carrier=%s:%d pid=%d",
        volume.volume_id,
        address.host,
        address.port,
        os.getpid(),
    )


def _record_client(
    store_name: str,
    rank: str,
    role: RankRole,
    selected: _SelectedControlPlane,
    recorded: _SelectedControlPlane | None,
    strategy: TorchStoreStrategy,
) -> tuple[LocalClient, _SelectedControlPlane, _SelectedControlPlane | None, bool]:
    """Record ``selected`` and cache a client on it, sharing a racing call's client.

    Returns the client, its record, what to close after the step (a replaced record or
    this call's own dial) and whether this call built the client. The caller holds the lock.
    """
    closing = None
    cached = _local_clent_map.get(store_name)
    if selected is recorded:
        # A racing call built its client first, on the record or on a plane it was given.
        if cached is not None:
            return cached, selected, None, False
    else:
        current = _control_plane_map.get(store_name)
        if (
            selected.address is not None
            and current is not None
            and current.address == selected.address
        ):
            # A racing call connected through this address first. Share its connection, and
            # its client when one is cached.
            if cached is not None:
                return cached, current, selected, False
            closing, selected = selected, current
            # The strategy came through this call's own dial, which closes after the step.
            host = _volume_host(strategy)
            if host is not None and isinstance(current.control_plane, VolumeRecords):
                host.rebind(current.control_plane.volume_records)
        else:
            closing = _select_control_plane(store_name, selected)
    routing_client = RoutingClient(rank, role, selected.control_plane, strategy)
    # A racing call's client stays with its caller, so shutdown closes its handles too.
    _keep_dropped(store_name, _local_clent_map.get(store_name))
    _local_clent_map[store_name] = routing_client
    return routing_client, selected, closing, True


async def _monarch_routing_client(
    store_name: str, role: RankRole, group: int | None
) -> RoutingClient:
    """Build and cache a routing client on the Monarch coordinator found by name."""
    # Check role and group first, so a bad call looks nothing up.
    rank = _client_rank(_routing_namespace(role, group), None, None)
    control_plane = await get_backend("monarch").coordinator(store_name)
    strategy = await control_plane.strategy()
    routing_client = RoutingClient(rank, role, control_plane, strategy)
    with _selection_lock:
        _local_clent_map[store_name] = routing_client
        _attempted.discard(store_name)
    _log_control_plane(_control_plane_kind(control_plane), None, store_name, rank)
    return routing_client


async def put(
    key: str, value: torch.Tensor | Any, store_name: str = DEFAULT_TORCHSTORE_NAME
) -> None:
    """Store a tensor or object in the distributed store.

    Args:
        key (str): Unique identifier for the stored value.
        value (torch.Tensor or Any): Tensor or object to store.
        store_name (str): Name of the store to use. Defaults to DEFAULT_TORCHSTORE_NAME.

    Example:
        >>> tensor = torch.randn(100, 100)
        >>> await put("my_tensor", tensor)
        >>> await put("my_object", {"data": [1, 2, 3]})
    """
    cl = await client(store_name)
    return await cl.put(key, value)


async def put_batch(
    entries: dict[str, torch.Tensor | Any],
    store_name: str = DEFAULT_TORCHSTORE_NAME,
) -> None:
    """Store multiple key-value pairs in a single batched operation.

    Args:
        entries: Dict mapping keys to values to store.
        store_name (str): Name of the store to use. Defaults to DEFAULT_TORCHSTORE_NAME.

    Example:
        >>> t1 = torch.randn(100, 100)
        >>> t2 = torch.randn(50, 50)
        >>> await put_batch({"key1": t1, "key2": t2})
    """
    cl = await client(store_name)
    return await cl.put_batch(entries)


async def get(
    key: str,
    inplace_tensor: torch.Tensor | None = None,
    tensor_slice_spec: TensorSlice | None = None,
    store_name: str = DEFAULT_TORCHSTORE_NAME,
) -> torch.Tensor | Any:
    """Retrieve a tensor or object from the distributed store.

    Args:
        key (str): Unique identifier of the value to retrieve.
        inplace_tensor (torch.Tensor, optional): Pre-allocated tensor for in-place retrieval.
        tensor_slice_spec (TensorSlice, optional): Specification for which slice of the tensor to retrieve.
        store_name (str): Name of the store to use. Defaults to DEFAULT_TORCHSTORE_NAME.

    Returns:
        The stored tensor, tensor slice, or object.

    Raises:
        KeyError: If the key does not exist.

    Example:
        >>> # Get full tensor
        >>> tensor = await get("my_tensor")

        >>> # Get specific slice
        >>> from torchstore.transport.pipe import TensorSlice
        >>> slice_spec = TensorSlice(
        ...     offsets=(10, 20),
        ...     coordinates=(),
        ...     global_shape=(1000, 1000),
        ...     local_shape=(50, 100),
        ...     mesh_shape=()
        ... )
        >>> tensor_slice = await get("my_tensor", tensor_slice_spec=slice_spec)

        >>> # In-place retrieval
        >>> pre_allocated_tensor = torch.empty(100, 100)
        >>> await get("my_tensor", inplace_tensor=pre_allocated_tensor)

        >>> # In-place slice retrieval (copies slice into pre-allocated tensor)
        >>> slice_buffer = torch.empty(50, 100)
        >>> await get("my_tensor", inplace_tensor=slice_buffer, tensor_slice_spec=slice_spec)
    """
    cl = await client(store_name)
    return await cl.get(key, inplace_tensor, tensor_slice_spec)


@overload
async def get_batch(
    keys: list[str],
    store_name: str = DEFAULT_TORCHSTORE_NAME,
) -> dict[str, Any]:
    ...


@overload
async def get_batch(
    keys: dict[str, torch.Tensor | DTensor | None],
    store_name: str = DEFAULT_TORCHSTORE_NAME,
) -> dict[str, Any]:
    ...


async def get_batch(
    keys: list[str] | dict[str, torch.Tensor | DTensor | None],
    store_name: str = DEFAULT_TORCHSTORE_NAME,
) -> dict[str, Any]:
    """Retrieve multiple keys from the distributed store in a single batched operation.

    All-or-nothing: if any key is missing, the entire batch raises
    and no partial results are returned.

    Args:
        keys: Either a list of keys to retrieve, or a dict mapping keys to
            optional pre-allocated tensors for in-place retrieval.
        store_name (str): Name of the store to use. Defaults to DEFAULT_TORCHSTORE_NAME.

    Returns:
        dict mapping each key to its fetched data.

    Raises:
        KeyError: If any key does not exist.
    """
    cl = await client(store_name)
    return await cl.get_batch(keys)


async def delete(
    key: str,
    *,
    store_name: str = DEFAULT_TORCHSTORE_NAME,
) -> None:
    """Delete a key from the distributed store.

    Args:
        key (str): Unique identifier of the value to delete.

    Keyword Args:
        store_name (str): Name of the store to use. Defaults to DEFAULT_TORCHSTORE_NAME.

    Example:
        >>> await delete("my_tensor")
    """
    cl = await client(store_name=store_name)
    return await cl.delete(key)


async def delete_batch(
    keys: list[str],
    *,
    store_name: str = DEFAULT_TORCHSTORE_NAME,
) -> None:
    """Delete multiple keys from the distributed store.

    Missing keys are ignored to make cleanup retries idempotent.

    Args:
        keys: Unique identifiers of the values to delete.

    Keyword Args:
        store_name (str): Name of the store to use. Defaults to DEFAULT_TORCHSTORE_NAME.

    Example:
        >>> await delete_batch(["tensor_a", "tensor_b"])
    """
    cl = await client(store_name=store_name)
    return await cl.delete_batch(keys)


async def keys(
    prefix: str | None = None,
    *,
    store_name: str = DEFAULT_TORCHSTORE_NAME,
) -> list[str]:
    """
    Get all keys that match the given prefix.

    This method retrieves all keys from the storage that start with the specified prefix.

    Args:
        prefix (str): The prefix to match against stored keys.


    Returns:
        List[str]: A list of keys that match the given prefix.

    Example:
        >>> keys = await keys("my_prefix")
    """
    cl = await client(store_name=store_name)
    return await cl.keys(prefix)


async def exists(key: str, store_name: str = DEFAULT_TORCHSTORE_NAME) -> bool:
    """Check if a key exists in the distributed store.

    Args:
        key (str): Unique identifier to check for existence.
        store_name (str): Name of the store to use. Defaults to DEFAULT_TORCHSTORE_NAME.

    Returns:
        bool: True if the key exists, False otherwise.

    Example:
        >>> if await exists("my_tensor"):
        ...     tensor = await get("my_tensor")
        >>>
        >>> # Check before storing
        >>> if not await exists("checkpoint_1"):
        ...     await put("checkpoint_1", model.state_dict())
    """
    cl = await client(store_name)
    return await cl.exists(key)


async def put_state_dict(
    state_dict: dict[str, Any] | None,
    key: str,
    store_name: str = DEFAULT_TORCHSTORE_NAME,
    direct_rdma: bool = False,
    transfer_dtype: "torch.dtype | None" = None,
    *,
    preserve_dtype_keys: frozenset[str] = frozenset(),
) -> None:
    """Store a PyTorch model state_dict in the distributed store.

    Args:
        state_dict (dict or None): Model state_dict to store.  Required on the
            first call.  When ``direct_rdma=True``, may be ``None`` on
            subsequent calls to skip the (potentially expensive)
            ``model.state_dict()`` construction — only staging-buffer refresh
            is performed.
        key (str): Unique identifier for the state_dict.
        store_name (str): Name of the store to use. Defaults to DEFAULT_TORCHSTORE_NAME.
        direct_rdma (bool): If True, register RDMA handles pointing at the
            caller's GPU memory instead of copying data to a StorageVolume.
            First call registers handles; subsequent calls refresh staging
            buffers for non-contiguous params.
        transfer_dtype (torch.dtype, optional): If set, cast floating-point
            weights to this dtype for transfer. Allows the source to keep
            higher-precision master weights (e.g. float32) while transferring
            in a lower precision (e.g. bfloat16). Without ``direct_rdma``,
            SharedMemory casts while copying into StorageVolume memory and
            other transports cast each dispatched batch, so tensors must not
            change until the put returns.
        preserve_dtype_keys (frozenset[str]): Flattened state-dict keys (e.g.
            buffers) that keep their dtype when ``transfer_dtype`` is set.
            Ignored unless ``transfer_dtype`` is set; then raises
            ``ValueError`` if a key is missing, and ``NotImplementedError``
            with ``direct_rdma=True``.

    Example:
        >>> model = torch.nn.Linear(10, 5)
        >>> await put_state_dict(model.state_dict(), "model_checkpoint")
    """
    cl = await client(store_name)
    await torchstore.state_dict_utils.put_state_dict(
        store=cl,
        state_dict=state_dict,
        key=key,
        direct_rdma=direct_rdma,
        transfer_dtype=transfer_dtype,
        preserve_dtype_keys=preserve_dtype_keys,
    )


async def get_state_dict(
    key: str,
    user_state_dict: dict[str, Any] | None = None,
    strict: bool = True,
    store_name: str = DEFAULT_TORCHSTORE_NAME,
    direct_rdma: bool = False,
) -> dict[str, Any]:
    """Retrieve a PyTorch model state_dict from the distributed store.

    A routing requester's call that passes entries no earlier call through its client planned, unless
    ``strict=False`` and no publisher has them, turns automatic garbage collection off for the whole process
    while it plans, then back on if it was on, which can undo a change another thread makes meanwhile. A
    call in another thread that turned it off first turns it back on as it finishes, maybe mid-plan.

    Args:
        key (str): Unique identifier of the state_dict to retrieve.
        user_state_dict (dict, optional): Pre-existing state_dict to merge with.
            Required when ``direct_rdma=True`` (destination tensors for in-place writes).
        strict (bool): Whether to enforce strict loading. Defaults to True.
        store_name (str): Name of the store to use. Defaults to DEFAULT_TORCHSTORE_NAME.
        direct_rdma (bool): If True, pull weights directly from the source's
            GPU memory via one-sided RDMA reads. Handles are fetched from
            TorchStore on the first call and cached for subsequent calls.

    Returns:
        dict: The retrieved state_dict.

    Example:
        >>> state_dict = await get_state_dict("model_checkpoint")
        >>> model.load_state_dict(state_dict)
    """
    cl = await client(store_name)
    return await torchstore.state_dict_utils.get_state_dict(
        cl, key, user_state_dict, strict, direct_rdma=direct_rdma
    )
