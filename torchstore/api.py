# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import os
import threading
from collections.abc import Awaitable, Callable
from functools import partial
from logging import getLogger
from typing import Any, NamedTuple, overload, TYPE_CHECKING, TypeVar

import torch
from torch.distributed.tensor import DTensor

import torchstore.state_dict_utils
from torchstore.client import ClientType, LocalClient
from torchstore.routing._model import RankRole
from torchstore.routing.client import RoutingClient
from torchstore.routing.control_plane import ControlPlane, ControlPlaneAddress
from torchstore.strategy import ControllerStorageVolumes, TorchStoreStrategy
from torchstore.transport.types import TensorSlice

if TYPE_CHECKING:
    from monarch.actor import Actor, Future, Point, ProcMesh

    from torchstore.controller import Controller
    from torchstore.spmd import _SPMDSession

logger = getLogger(__name__)

# I need to keep this somewhere, so here we go
DEFAULT_TORCHSTORE_NAME: str = "torchstore"

# cache for local clients
_local_clent_map: dict[str, LocalClient] = {}


# SPMD-initialized stores register their session here
_spmd_state_map: dict[str, "_SPMDSession"] = {}


class _SelectedControlPlane(NamedTuple):
    """A routing control plane other than Monarch's, and the address it came from."""

    control_plane: ControlPlane
    address: ControlPlaneAddress | None


# Control planes this process initialized, was given or connected to. It survives
# reset_client, and stores on Monarch's coordinator never enter it.
_control_plane_map: dict[str, _SelectedControlPlane] = {}

# Guards each check-and-swap of the record and the client cache. Never held across an
# await: a dialed connection closes after it is released.
_selection_lock = threading.Lock()

# Stores whose record ts.shutdown released here, until a selection takes the name back,
# so that a role-less call or another shutdown spawns no Monarch controller for it.
_released_stores: set[str] = set()

# A forked child's stores whose dialed records the fork dropped, with their addresses,
# until a selection or shutdown. Calls without an address raise, not look up Monarch.
_forked_stores: dict[str, ControlPlaneAddress] = {}

# How long a caller waits for a dialed connection's close, which then runs on alone
# until its event loop ends.
_CLOSE_WAIT_S = 5.0

# Closes still running, kept so that none is collected before it finishes.
_closing: set["asyncio.Task[None]"] = set()

_TActor = TypeVar("_TActor", bound="Actor")


def current_rank() -> "Point":
    """Monarch's ``current_rank``, imported when first called."""
    from monarch.actor import current_rank as monarch_current_rank

    return monarch_current_rank()


def get_or_spawn_controller(
    name: str, Class: type[_TActor], *args: Any, **kwargs: Any
) -> "Future[_TActor]":
    """Monarch's ``get_or_spawn_controller``, imported when first called."""
    from monarch.actor import get_or_spawn_controller as monarch_get_or_spawn_controller

    return monarch_get_or_spawn_controller(name, Class, *args, **kwargs)


def _routing_volume_id(namespace: str) -> str:
    """Qualify the monarch rank with its mesh's namespace."""
    return f"{namespace}/{current_rank().rank}"


def _routing_namespace(role: RankRole, group: int | None) -> str:
    """The namespace of one participating mesh."""
    if role == RankRole.PUBLISHER:
        if group is not None:
            raise ValueError("publishers are a single mesh and take no group")
        return role.value
    if group is None:
        raise ValueError("requesters must pass the index of their requester mesh")
    return f"{role.value}/{group}"


def _caller_control_plane(control_plane: str | ControlPlane) -> ControlPlane | None:
    """The caller's own control plane, or None to keep Monarch's coordinator."""
    if isinstance(control_plane, ControlPlane):
        return control_plane
    if control_plane == "monarch":
        return None
    error = ValueError if isinstance(control_plane, str) else TypeError
    raise error(
        "control_plane must be 'monarch' or a ControlPlane instance, "
        f"got {control_plane!r}"
    )


def _check_one_store(store_name: str, control_plane: ControlPlane) -> None:
    """Refuse an instance another store recorded.

    Its methods name no store. The caller holds ``_selection_lock``.
    """
    held = [(other, plane.control_plane) for other, plane in _control_plane_map.items()]
    for other, instance in held:
        if other != store_name and instance is control_plane:
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
    _released_stores.discard(store_name)
    _forked_stores.pop(store_name, None)
    recorded = _control_plane_map.get(store_name)
    if selected is None:
        _control_plane_map.pop(store_name, None)
    else:
        _control_plane_map[store_name] = selected
    return recorded


def _same_address(
    first: ControlPlaneAddress | None, second: ControlPlaneAddress | None
) -> bool:
    """Compare by type and every attribute, also on a subclass that is no dataclass."""
    return (
        first is not None
        and type(first) is type(second)
        and first == second
        and getattr(first, "__dict__", None) == getattr(second, "__dict__", None)
    )


def _control_plane_kind(control_plane: ControlPlane) -> str:
    """The log line's kind: the plane's ``kind``, else its class name."""
    return control_plane.kind or type(control_plane).__name__


async def _run_close(close: Callable[[], Awaitable[object]]) -> None:
    """Call ``close`` in the close task, so that an error raised at once lands there."""
    await close()


def _start_close(selected: _SelectedControlPlane) -> "asyncio.Task[None] | None":
    """Start closing a connection dialed through an address. A caller's own stays open.

    The close runs as its own task, so cancelling a caller never interrupts it. Its event
    loop's end does.
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
    """Log the failure of a close that no caller waits for any more."""
    if not task.cancelled() and task.exception() is not None:
        logger.warning(
            "closing the connection dialed through %s failed",
            address,
            exc_info=task.exception(),
        )


def _leave_running(
    task: "asyncio.Task[None]", address: ControlPlaneAddress | None
) -> None:
    """Let a close that no caller waits for run on, logging a failure it ends with."""
    task.add_done_callback(partial(_log_close_failure, address))


async def _wait_for_close(selected: _SelectedControlPlane) -> bool:
    """Close a dialed connection, waiting for it at most ``_CLOSE_WAIT_S``.

    Returns False if the close outlasts the wait. Such a close, or one whose caller is
    cancelled, runs on while its event loop runs and logs a failure. A close that ends
    cancelled while its caller waits raises ``RuntimeError``.
    """
    task = _start_close(selected)
    if task is None:
        return True
    try:
        await asyncio.wait({task}, timeout=_CLOSE_WAIT_S)
    finally:
        if not task.done():
            _leave_running(task, selected.address)
    if not task.done():
        return False
    if task.cancelled():
        # The close cancelled itself, not this caller: report it as a failed close.
        raise RuntimeError(
            f"closing the connection dialed through {selected.address} was cancelled"
        )
    task.result()
    return True


async def _close_dialed(selected: _SelectedControlPlane) -> None:
    """Close a dialed connection, raising if the close outlasts its wait."""
    if not await _wait_for_close(selected):
        raise TimeoutError(
            f"closing the connection dialed through {selected.address} took over "
            f"{_CLOSE_WAIT_S:g} s"
        )


async def _close_dialed_quietly(selected: _SelectedControlPlane) -> None:
    """Close a dialed connection, only logging a failure or a slow close."""
    try:
        if not await _wait_for_close(selected):
            logger.warning(
                "closing the connection dialed through %s is still running after %g s",
                selected.address,
                _CLOSE_WAIT_S,
            )
    except Exception:
        logger.warning(
            "closing the connection dialed through %s failed",
            selected.address,
            exc_info=True,
        )


async def _close_quietly(
    dialed: _SelectedControlPlane | None, kept: _SelectedControlPlane | None
) -> None:
    """Close ``dialed`` unless ``kept`` holds its connection, logging a failure."""
    if dialed is not None and (
        kept is None or dialed.control_plane is not kept.control_plane
    ):
        await _close_dialed_quietly(dialed)


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
) -> None:
    """Initialize the TorchStore distributed storage system.

    Sets up storage volumes and controller. Must be called before any put/get operations.
    A routing store starts over on each call that its control plane's ``init`` accepts,
    as Monarch's coordinator does, dropping this process's cached client and closing a
    connection it dialed. If the plane refuses, this call raises and keeps both.
    Initialize other stores again only after ``shutdown``.
    Never overlap it with another ``initialize``, ``client`` or ``shutdown`` call for the
    same store in this process, or with an ``initialize`` given the same control plane.

    Args:
        num_storage_volumes (int): Number of storage volumes to create. Defaults to 1.
        strategy (TorchStoreStrategy, optional): Strategy for distributing tensors across volumes.
            Uses ControllerStorageVolumes if None and num_storage_volumes=1.
        store_name (str): Unique name for this store instance. Defaults to DEFAULT_TORCHSTORE_NAME.
        mesh (ProcMesh, optional): Monarch ProcMesh on which to spawn StorageVolumes
        client_type: Client implementation to initialize. Routing clients lazily
            exchange state-dict layouts and construct routes.
        control_plane: Where a routing store's ranks share publishers and layouts.
            ``"monarch"`` spawns the routing coordinator actor. A ``ControlPlane``
            instance, which serves only this store, is initialized here instead for
            later clients in this process. Log lines name it by ``kind`` or class name.

    Raises:
        RuntimeError: If num_storage_volumes > 1 but no strategy is provided.
        TypeError: If control_plane is neither a string nor a ``ControlPlane``.
        ValueError: If control_plane is a string other than ``"monarch"``, or an
            instance for a standard store or one that serves another store.

    Example:
        >>> import torchstore as ts
        >>> await ts.initialize(num_storage_volumes=4, strategy=LocalRankStrategy()) # uses default namespace.
        >>> >>> await ts.initialize("my_custom_store")
    """
    caller_control_plane = _caller_control_plane(control_plane)
    if client_type == ClientType.ROUTING:
        if strategy is None or mesh is None:
            raise RuntimeError("routing mode requires both mesh and strategy")
        await _initialize_routing(mesh, strategy, store_name, caller_control_plane)
        return
    if caller_control_plane is not None:
        raise ValueError("standard stores take only control_plane='monarch'")

    if num_storage_volumes == 1 and strategy is None:
        strategy = ControllerStorageVolumes()
    elif strategy is None:
        raise RuntimeError(
            "Must specify controller strategy if num_storage_volumes > 1"
        )

    from torchstore.storage_volume import StorageVolume

    # TODO: monarch doesn't support nested actors yet, so we need to spawn storage volumes here
    # ideally this is done in the controller.init
    if isinstance(strategy, ControllerStorageVolumes):
        storage_volumes = await get_or_spawn_controller(
            "storage_volume_controller",
            StorageVolume,
            id_func=strategy.get_volume_id,
        )
    else:
        storage_volumes = await StorageVolume.spawn(
            num_volumes=num_storage_volumes,
            mesh=mesh,
            id_func=strategy.get_volume_id,
        )

    controller = await _controller(store_name)
    await controller.init.call(
        strategy=strategy,
        num_storage_volumes=num_storage_volumes,
        storage_volumes=storage_volumes,
    )
    with _selection_lock:
        # The name is this controller's again, so client and shutdown reach it.
        _released_stores.discard(store_name)
        _forked_stores.pop(store_name, None)


async def _initialize_routing(
    mesh: "ProcMesh",
    strategy: TorchStoreStrategy,
    store_name: str,
    control_plane: ControlPlane | None,
) -> None:
    """Spawn trainer volumes, then initialize the given control plane or Monarch's."""
    from torchstore.storage_volume import StorageVolume

    if control_plane is not None:
        with _selection_lock:
            _check_one_store(store_name, control_plane)
    namespace = _routing_namespace(RankRole.PUBLISHER, None)
    volumes = await StorageVolume.spawn(
        1,
        mesh,
        id_func=partial(_routing_volume_id, namespace),
    )
    await strategy.set_storage_volumes(volumes)

    publishers = {f"{namespace}/{rank}" for rank in range(mesh.size())}

    selected: _SelectedControlPlane | None = None
    if control_plane is None:
        from torchstore.routing.coordinator import (
            MonarchControlPlane,
            RoutingCoordinator,
        )

        coordinator = await get_or_spawn_controller(store_name, RoutingCoordinator)
        await MonarchControlPlane(coordinator).init(
            publishers=publishers,
            strategy=strategy,
        )
        kind = "monarch"
    else:
        await control_plane.init(publishers=publishers, strategy=strategy)
        selected = _SelectedControlPlane(control_plane, None)
        kind = _control_plane_kind(control_plane)
    with _selection_lock:
        replaced = _select_control_plane(store_name, selected)
        # The store starts over, so a client cached on its old state goes too.
        reset_client(store_name)
    _log_control_plane(kind, None, store_name, "-")
    await _close_quietly(replaced, selected)


async def shutdown(store_name: str = DEFAULT_TORCHSTORE_NAME) -> None:
    """Shutdown and cleanup a TorchStore instance.

    Gracefully shuts down all storage volumes and controllers associated with the
    store. For SPMD-initialized stores, this delegates to the session cleanup
    path created by ``torchstore.spmd.initialize()``, which also stops Job API
    sidecars and tears down the Monarch host mesh.

    For a routing store whose control plane this process initialized, was given
    or connected to, it drops only that record and the client, and closes a
    connection made through an address. Another shutdown then drops only a client.
    A caller's control plane stays open, and the store's volumes stay with their mesh.
    Never overlap it with another ``initialize``, ``client`` or ``shutdown`` call for the
    same store in this process.

    Args:
        store_name (str): Name of the store to shutdown. Defaults to DEFAULT_TORCHSTORE_NAME.

    Raises:
        TimeoutError: If closing a connection made through an address takes over
            5 s. The record and client are dropped by then, and the close runs on
            until this event loop ends.
        Exception: Whatever that connection's ``close`` raises, after the same drop.

    Example:
        >>> import torchstore as ts
        >>> await ts.shutdown()  # Shutdown default store
        >>> await ts.shutdown("my_custom_store")
    """
    if await _shutdown_selected(store_name):
        return

    session = _spmd_state_map.get(store_name)
    if session is not None:
        await session.shutdown()
        return
    if store_name in _released_stores:
        # Released here before: Monarch's lookup would spawn a controller for the name.
        reset_client(store_name)
        return

    controller = await _controller(store_name)
    try:
        await controller.teardown.call()
    finally:
        reset_client(store_name)


async def _shutdown_selected(store_name: str) -> bool:
    """Release a store's record and client, then close a connection it dialed.

    Returns False if there was nothing to release. A close's error propagates.
    """
    with _selection_lock:
        selected = _control_plane_map.pop(store_name, None)
        forked = _forked_stores.pop(store_name, None)
        if selected is None and forked is None:
            return False
        _released_stores.add(store_name)
        reset_client(store_name)
    if selected is not None:
        await _close_dialed(selected)
    return True


def reset_client(store_name: str = DEFAULT_TORCHSTORE_NAME) -> None:
    """Reset the cached or installed client for a given store."""
    _local_clent_map.pop(store_name, None)


# What the hook below drops in a forked child, kept alive so that no finalizer runs on
# the parent's connections from a child that exits through os._exit, as multiprocessing's do.
_inherited: list[object] = []


def _after_fork_in_child() -> None:
    """Replace the selection lock, which a parent thread may hold, and drop dialed records.

    The parent's connections stay the parent's, so the child dials its own. A caller's
    own plane stays recorded, so the child shares it and its cached client.
    """
    global _selection_lock
    _selection_lock = threading.Lock()
    for store_name, selected in list(_control_plane_map.items()):
        if selected.address is not None:
            del _control_plane_map[store_name]
            _inherited.append((selected, _local_clent_map.pop(store_name, None)))
            _forked_stores[store_name] = selected.address


os.register_at_fork(after_in_child=_after_fork_in_child)


async def _controller(store_name: str = DEFAULT_TORCHSTORE_NAME) -> "Controller":
    """Get a controller handle for interacting with the store."""
    from torchstore.controller import Controller

    session = _spmd_state_map.get(store_name)
    if session is not None:
        return session.controller
    return await get_or_spawn_controller(store_name, Controller)


async def client(
    store_name: str = DEFAULT_TORCHSTORE_NAME,
    *,
    role: RankRole | str | None = None,
    group: int | None = None,
    address: ControlPlaneAddress | None = None,
    control_plane: ControlPlane | None = None,
) -> LocalClient:
    """Get a local client handle for interacting with the store.

    Returns a cached LocalClient instance that provides the interface for put/get operations.
    Never overlap it with another ``initialize``, ``client`` or ``shutdown`` call for the
    same store in this process, except concurrent first calls through one address, which
    share one client and its connection.

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
            clients in this process reuse its connection until ``shutdown`` or
            another selection replaces and closes it. A forked child dials its own.
        control_plane: A routing store's own control plane, which serves only that
            store and needs ``role``. It wins over ``address``, then this process's record
            (kept across ``reset_client``), then the Monarch lookup. A cached client
            refuses other instances, even a copy that an actor call delivers.

    Returns:
        LocalClient: A client instance for performing storage operations.

    Raises:
        TypeError: If address or control_plane has the wrong type.
        ValueError: If address or control_plane comes without role, the address
            names another store, control_plane (else address) differs from the cached
            client's, control_plane serves another store, or the first client of a
            store with a recorded or shut down control plane has no role.
        RuntimeError: In a forked child, if a call without address or control_plane
            needs a store whose connection the fork dropped.

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
    if role is None and (address is not None or control_plane is not None):
        raise ValueError("address and control_plane are for routing clients: pass role")
    if address is not None and address.store_name != store_name:
        raise ValueError(
            f"address is for store {address.store_name!r}, not {store_name!r}"
        )
    with _selection_lock:
        if control_plane is not None:
            _check_one_store(store_name, control_plane)
        cached = _local_clent_map.get(store_name)
        if cached is not None:
            _check_cached_client(store_name, address, control_plane)
            return cached

    if role is not None:
        return await _routing_client(
            store_name, RankRole(role), group, address, control_plane
        )
    _check_unrecorded(store_name)
    from torchstore.controller import ControllerDirectory

    controller = await _controller(store_name)
    controller_strategy = await controller.get_controller_strategy.call_one()
    local_client = LocalClient(
        directory=ControllerDirectory(controller),
        strategy=controller_strategy,
        controller=controller,
    )
    with _selection_lock:
        _local_clent_map[store_name] = local_client
    return local_client


def _check_unrecorded(store_name: str) -> None:
    """Refuse a role-less first client for a store whose plane this process selected."""
    _check_not_forked(store_name)
    released = store_name in _released_stores and store_name not in _spmd_state_map
    if store_name in _control_plane_map or released:
        raise ValueError(
            f"routing store {store_name!r} needs role on its first ts.client call"
        )


def _check_not_forked(store_name: str) -> None:
    """Refuse a store whose dialed record a fork dropped, before any Monarch lookup."""
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
) -> None:
    """Refuse a control plane other than the cached client's, else another address."""
    selected = _control_plane_map.get(store_name)
    if control_plane is not None:
        if selected is None or selected.control_plane is not control_plane:
            raise ValueError(
                f"the cached client for store {store_name!r} uses another control "
                "plane instance. Later calls in this process can omit control_plane= "
                "(an actor call delivers a copy), or call ts.reset_client to switch"
            )
    elif address is not None and (
        selected is None or not _same_address(selected.address, address)
    ):
        raise ValueError(
            f"the cached client for store {store_name!r} did not connect through "
            f"{address}. Call ts.reset_client to replace it"
        )


async def _routing_client(
    store_name: str,
    role: RankRole,
    group: int | None,
    address: ControlPlaneAddress | None,
    control_plane: ControlPlane | None,
) -> RoutingClient:
    """Build and cache a routing client on the first plane in ``client``'s order."""
    recorded = _control_plane_map.get(store_name)
    if control_plane is None and address is None and recorded is None:
        _check_not_forked(store_name)
        return await _monarch_routing_client(store_name, role, group)
    # Check role and group first, so a bad call dials nothing.
    rank = _routing_volume_id(_routing_namespace(role, group))
    if control_plane is not None:
        selected = _SelectedControlPlane(control_plane, None)
    elif address is not None and (
        recorded is None or not _same_address(recorded.address, address)
    ):
        selected = _SelectedControlPlane(await address.connect(), address)
    else:
        # The record, also when this process already connected through ``address``.
        assert recorded is not None
        selected = recorded
    try:
        strategy = await selected.control_plane.strategy()
    except asyncio.CancelledError:
        if selected is not recorded:
            # Waiting for the close would hold the caller past its deadline or Ctrl-C.
            task = _start_close(selected)
            if task is not None:
                _leave_running(task, selected.address)
        raise
    except Exception:
        if selected is not recorded:
            # The call's own error propagates, not the close's.
            await _close_dialed_quietly(selected)
        raise
    with _selection_lock:
        routing_client, selected, closing, built = _record_client(
            store_name, rank, role, selected, recorded, strategy
        )
    try:
        if built:
            _log_control_plane(
                _control_plane_kind(selected.control_plane),
                selected.address,
                store_name,
                rank,
            )
    finally:
        await _close_quietly(closing, selected)
    return routing_client


def _record_client(
    store_name: str,
    rank: str,
    role: RankRole,
    selected: _SelectedControlPlane,
    recorded: _SelectedControlPlane | None,
    strategy: TorchStoreStrategy,
) -> tuple[RoutingClient, _SelectedControlPlane, _SelectedControlPlane | None, bool]:
    """Record ``selected`` and cache a client on it, sharing a racing call's client.

    Returns the client, its record, what to close after the step (a replaced record or
    this call's own dial) and whether this call built the client. The caller holds the lock.
    """
    closing = None
    cached = _local_clent_map.get(store_name)
    if selected is recorded:
        # A racing call that also reused the record built its client first.
        if (
            isinstance(cached, RoutingClient)
            and cached._control_plane is selected.control_plane
        ):
            return cached, selected, None, False
    else:
        current = _control_plane_map.get(store_name)
        if current is not None and _same_address(selected.address, current.address):
            # A racing call connected through this address first. Share its client.
            if isinstance(cached, RoutingClient):
                return cached, current, selected, False
            closing, selected = selected, current
        else:
            closing = _select_control_plane(store_name, selected)
    routing_client = RoutingClient(rank, role, selected.control_plane, strategy)
    _local_clent_map[store_name] = routing_client
    return routing_client, selected, closing, True


async def _monarch_routing_client(
    store_name: str, role: RankRole, group: int | None
) -> RoutingClient:
    """Build and cache a routing client on the Monarch coordinator found by name."""
    from monarch.actor import ActorError

    from torchstore.routing.coordinator import MonarchControlPlane, RoutingCoordinator

    coordinator = await get_or_spawn_controller(store_name, RoutingCoordinator)
    control_plane = MonarchControlPlane(coordinator)
    rank = _routing_volume_id(_routing_namespace(role, group))
    try:
        strategy = await control_plane.strategy()
    except ActorError as error:
        if "init has not run" in str(error):
            # ``add_note`` needs Python 3.11, and this keeps the error an ActorError.
            error.message = (
                f"{error.message} If ts.initialize gave store {store_name!r} another "
                "control plane, pass ts.client that control plane (control_plane=) "
                "or its address (address=)."
            )
        raise
    routing_client = RoutingClient(rank, role, control_plane, strategy)
    with _selection_lock:
        _local_clent_map[store_name] = routing_client
    _log_control_plane("monarch", None, store_name, rank)
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
            in a lower precision (e.g. bfloat16).

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
    )


async def get_state_dict(
    key: str,
    user_state_dict: dict[str, Any] | None = None,
    strict: bool = True,
    store_name: str = DEFAULT_TORCHSTORE_NAME,
    direct_rdma: bool = False,
) -> dict[str, Any]:
    """Retrieve a PyTorch model state_dict from the distributed store.

    A routing requester's first call for a key pauses automatic garbage collection for
    every thread while it plans its routes, then resumes it if it was on.

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
