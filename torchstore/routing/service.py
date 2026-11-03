# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The routing coordinator served over the carrier, beside its store's TCPStore rendezvous.

Internal and may change: users never import it. They reach it through ``ControlPlane``, the ``initialize`` and
``client`` keywords, ``shutdown`` and the address ``initialize`` returns, whose unpickling imports it.
"""

from __future__ import annotations

import asyncio
import io
import itertools
import logging
import os
import pickle
import threading
import weakref
import zlib
from collections.abc import Callable, Collection, Mapping
from concurrent import futures
from dataclasses import dataclass, field
from typing import Any, cast, NoReturn, TYPE_CHECKING

from torchstore.backend import register_backend, ServiceBackend

from torchstore.carrier import (
    CarrierAddress,
    CarrierClient,
    CarrierError,
    CarrierServer,
    PeerLostError,
    RemoteError,
    VOLUME_CLOSED,
    wait_future,
)
from torchstore.logging import LatencyTracker
from . import layout_registry
from ._gc import call_gc_paused
from .control_plane import ControlPlane, ControlPlaneAddress
from .layout_registry import LayoutRegistry
from .tcpstore import TCPStoreAddress, TCPStoreRendezvous

if TYPE_CHECKING:
    from torchstore.strategy import TorchStoreStrategy
    from ._model import KeyRegistration, Registrations

__all__ = [
    "CarrierServiceBackend",
    "CoordinatorService",
    "ServiceAddress",
    "ServiceControlPlane",
]

logger: logging.Logger = logging.getLogger(__name__)

_SERVICE = "coordinator"
_ENDPOINTS = ("register_layout", "get_layouts")
# Read when used, so tests can shorten them.
_CLOSE_GRACE_S = 5.0
# A lookup's call deadline outlasts its wait by this, so a timeout names missing publishers.
_REPLY_MARGIN_S = 60.0
# On Llama 3 70B and Kimi K2.5 layouts: 7 to 10% smaller than level 6, and level 9's bytes in less time.
_PART_ZLIB_LEVEL = 8
# The empty last block that ends a raw deflate stream of parts.
_LAST_BLOCK = b"\x03\x00"
# How every part ends so that parts join into one stream: a sync flush's empty stored block.
_SYNC_FLUSH = b"\x00\x00\xff\xff"
# Deflate's largest ratio of inflated to deflated bytes.
_DEFLATE_MAX_RATIO = 1032

# What a forked child lets go of. Weak, so they never keep a handle or service alive.
_handles: weakref.WeakSet[ServiceControlPlane] = weakref.WeakSet()
_services: weakref.WeakSet[CoordinatorService] = weakref.WeakSet()


def _latency_tracker(operation: str) -> LatencyTracker:
    return LatencyTracker(
        f"routing control_plane=service {operation}", level=logging.INFO
    )


def _wait_s(timeout_s: float | None) -> float:
    """Resolve a lookup's wait: ``None`` is this process's registry default."""
    return (
        layout_registry._LAYOUT_REGISTRATION_TIMEOUT_S
        if timeout_s is None
        else timeout_s
    )


def _compress_part(layout: Mapping[str, Any]) -> tuple[bytes, int]:
    """Pickle and compress a layout into a part and its pickle's size.

    A part is a raw deflate stream's blocks up to a sync flush. Parts thus join into one stream.
    """
    # @lint-ignore PYTHONPICKLEISBAD Layouts must reach ranks in other processes.
    pickled = pickle.dumps(layout, protocol=5)
    compressor = zlib.compressobj(_PART_ZLIB_LEVEL, zlib.DEFLATED, -zlib.MAX_WBITS)
    part = compressor.compress(pickled) + compressor.flush(zlib.Z_SYNC_FLUSH)
    return part, len(pickled)


def _inflate(stream: bytes, size: int) -> bytes:
    """Inflate a raw deflate stream into one buffer of its ``size`` inflated bytes, which zlib returns uncopied.

    Its first call (more past 4GiB) fills that buffer and reaches the stream's end. The full buffer makes CPython call
    once more (with no input) into a spare 64KiB block it drops. Each call yields the GIL once.
    """
    return zlib.decompress(stream, wbits=-zlib.MAX_WBITS, bufsize=size)


def _inflate_part(part: bytes, size: int) -> bytes:
    """Inflate one part of ``size`` inflated bytes in one call, which yields the GIL once.

    A byte to spare in its buffer spares CPython the empty call a full buffer makes. zlib then copies the part once.
    """
    return zlib.decompress(part + _LAST_BLOCK, wbits=-zlib.MAX_WBITS, bufsize=size + 1)


def _encode_layouts(parts: Mapping[str, tuple[bytes, int]]) -> bytes:
    """Pickle a reply of the ranks, their pickles' sizes and their parts, which a reader joins into one stream."""
    ranks, pieces = list(parts), list(parts.values())
    buffer = io.BytesIO()
    # @lint-ignore PYTHONPICKLEISBAD Layouts must reach ranks in other processes.
    pickler = pickle.Pickler(buffer, protocol=5)
    # Safe without a memo: the reply holds only rank names, sizes and parts, none shared.
    pickler.fast = True
    pickler.dump((ranks, [size for _, size in pieces], [part for part, _ in pieces]))
    return buffer.getvalue()


def _decode_layouts(data: Any) -> dict[str, Any]:
    """Load a reply as each rank's layout: inflate its parts at once, then unpickle each in one pause of collection.

    ``dict`` takes each part's ``pickle.loads`` through ``map`` in C and holds the GIL from one part to the next.
    """
    # Unpaused: of a reply's objects collection tracks only its tuple and three lists.
    # @lint-ignore PYTHONPICKLEISBAD Replies come from the store's own coordinator.
    ranks, sizes, parts = pickle.loads(data)
    # zlib yields the GIL as it inflates, so it runs before the pause, where other threads would run with collection off.
    pickled = memoryview(_inflate(b"".join([*parts, _LAST_BLOCK]), sum(sizes)))
    views = [
        pickled[end - size : end]
        for size, end in zip(sizes, itertools.accumulate(sizes))
    ]
    # A large layout's new objects set off collections that more than double Python 3.10's decode
    # time. Pickling leaves few new objects alive, so only decoding pauses collection.
    # @lint-ignore PYTHONPICKLEISBAD Layouts come from the store's own ranks.
    return call_gc_paused(dict, zip(ranks, map(pickle.loads, views)))


def _named(address: ServiceAddress, problem: str) -> str:
    return f"routing store {address.store_name!r} on {address}: {problem}"


@dataclass(frozen=True)
class ServiceAddress(ControlPlaneAddress):
    """Where ranks reach a coordinator service: the host and port of its store's TCPStore.

    The nonce selects one ``CoordinatorService.start``'s keys and stays out of logs.
    """

    host: str
    port: int
    nonce: str = field(repr=False)

    async def connect(self) -> ServiceControlPlane:
        """Connect to the TCPStore. The handle reads the rest when its calls need it."""
        rendezvous = await TCPStoreRendezvous.connect(self._tcpstore_address())
        return ServiceControlPlane(self, rendezvous)

    def __str__(self) -> str:
        return f"{self.host}:{self.port}"

    def _tcpstore_address(self) -> TCPStoreAddress:
        return TCPStoreAddress(self.host, self.port, self.store_name, self.nonce)


class ServiceControlPlane(ControlPlane):
    """Reads a coordinator service's strategy from TCPStore and calls it for layouts.

    Each handle holds its own TCPStore connection and a share of this process's one connection to
    the coordinator. A forked child's copy dials the TCPStore again when it next reads it: planes must work after fork.
    """

    # TorchStore's selection log lines name a control plane by its ``kind``.
    kind = "service"

    def __init__(
        self,
        address: ServiceAddress,
        rendezvous: TCPStoreRendezvous,
        service: CoordinatorService | None = None,
    ) -> None:
        self.address = address
        # Set on handles from ``CoordinatorService.connect``, the only ones that init.
        self._service = service
        self._pid = os.getpid()
        # Guards the fields below, and is never held across an await.
        self._lock = threading.Lock()
        self._rendezvous: TCPStoreRendezvous | None = rendezvous
        self._coordinator: CarrierClient | None = None
        # One per coordinator call in flight, which close sets to fail that call.
        self._stops: set[futures.Future[None]] = set()
        self._closed = False
        _handles.add(self)

    def __reduce_ex__(self, protocol: Any) -> NoReturn:
        raise TypeError(f"{type(self).__name__} cannot be pickled: pass its address")

    async def init(
        self,
        publishers: Collection[str],
        strategy: TorchStoreStrategy,
    ) -> None:
        """Initialize the coordinator once, through a handle from ``CoordinatorService.connect``."""
        tracker = _latency_tracker("init")
        try:
            # This check makes no TCPStore call: init never uses this handle's connection.
            with self._lock:
                if self._closed:
                    raise self._error("this control plane is closed")
            if self._service is None:
                raise self._error(
                    "init runs only on a handle from CoordinatorService.connect(),"
                    " in the process that started the service"
                )
            await self._service._init(publishers, strategy)
        finally:
            tracker.track_e2e()

    async def strategy(self) -> TorchStoreStrategy:
        """Decode a fresh copy of the strategy from TCPStore on every call."""
        tracker = _latency_tracker("strategy")
        try:
            rendezvous = await self._state()
            return await rendezvous.read_strategy()
        finally:
            tracker.track_e2e()

    async def register_layout(
        self,
        rank: str,
        key: str,
        registrations: Mapping[str, KeyRegistration],
    ) -> None:
        tracker = _latency_tracker(f"register_layout key={key}")
        try:
            # Pickled and compressed here, in parallel with other publishers, so the coordinator's
            # one loop only decodes a layout and keeps these bytes as the rank's part.
            part, size = _compress_part(dict(registrations))
            await self._call("register_layout", rank, key, part, size)
        finally:
            tracker.track_e2e()

    async def get_layouts(
        self,
        key: str,
        *,
        timeout_s: float | None = None,
    ) -> Registrations:
        """Wait for every publisher's layout for ``key``, then decode a fresh copy.

        ``timeout_s=None`` waits the caller's ``_LAYOUT_REGISTRATION_TIMEOUT_S``, 300 s by default. The deadline is any wait
        above 0 plus ``_REPLY_MARGIN_S``, after a first call's 60 s address read and in a forked child a 60 s dial before it.
        """
        tracker = _latency_tracker(f"get_layouts key={key}")
        try:
            wait = _wait_s(timeout_s)
            # Written so that a NaN wait, which the registry checks once, gets the margin too.
            deadline_s = (wait if wait > 0 else 0) + _REPLY_MARGIN_S
            data = await self._call("get_layouts", key, wait, timeout_s=deadline_s)
            return _decode_layouts(data)
        finally:
            tracker.track_e2e()

    async def close(self) -> None:
        """Drop this handle's TCPStore connection and its share of the coordinator's.

        Its coordinator calls in flight fail with ``PeerLostError``, and the process's last share closes that connection.
        """
        with self._lock:
            self._closed = True
            rendezvous, self._rendezvous = self._rendezvous, None
            coordinator, self._coordinator = self._coordinator, None
            stops, self._stops = self._stops, set()
        for stop in stops:
            stop.set_result(None)
        try:
            if coordinator is not None:
                await coordinator.close()
        finally:
            if rendezvous is not None:
                await rendezvous.close()

    def _error(self, problem: str) -> RuntimeError:
        return RuntimeError(_named(self.address, problem))

    async def _state(self) -> TCPStoreRendezvous:
        """Return this handle's TCPStore connection (dialed again in a forked child).

        The child's fork hook dropped its copy of the parent's connection, and an inherited carrier client still calls.
        """
        with self._lock:
            if self._closed:
                raise self._error("this control plane is closed")
            if self._pid == os.getpid():
                return cast(TCPStoreRendezvous, self._rendezvous)
        dialed: TCPStoreRendezvous | None = await TCPStoreRendezvous.connect(
            self.address._tcpstore_address()
        )
        with self._lock:
            closed = self._closed
            if not closed and self._pid != os.getpid():
                self._rendezvous, self._pid, dialed = dialed, os.getpid(), None
            rendezvous = self._rendezvous
        if dialed is not None:
            # A close, or a racing first call, came first.
            await dialed.close()
        if closed:
            raise self._error("this control plane is closed")
        return cast(TCPStoreRendezvous, rendezvous)

    async def _coordinator_client(self) -> CarrierClient:
        """Return this handle's client, made once from an address read after ``ready``.

        An inherited client still calls so a forked child's handle dials the TCPStore here only to read the address.
        """
        with self._lock:
            # Close clears the client so a closed handle goes on to the closed check in _state.
            if self._coordinator is not None:
                return self._coordinator
        rendezvous = await self._state()
        data = await rendezvous.read_coordinator()
        address = CarrierAddress.from_json(data)
        with self._lock:
            # A close during the read must not leave a client that no close releases.
            if self._closed:
                raise self._error("this control plane is closed")
            # Racing first calls each read the address, and the first here makes the one client.
            if self._coordinator is None:
                self._coordinator = CarrierClient(address)
            return self._coordinator

    async def _call(
        self, endpoint: str, *args: Any, timeout_s: float | None = None
    ) -> Any:
        """Call the coordinator, reading its address too, or fail with ``PeerLostError`` once ``close`` takes this call's stop.

        Other handles may share the connection, so closing this handle's client alone would not end the call.
        """
        stop: futures.Future[None] = futures.Future()
        with self._lock:
            # A closed handle's call fails in the task: its first step checks.
            self._stops.add(stop)
        call = asyncio.ensure_future(self._call_coordinator(endpoint, args, timeout_s))
        stopped = asyncio.ensure_future(wait_future(stop, None))
        try:
            await asyncio.wait({call, stopped}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            with self._lock:
                # A close takes the stop and then wins over the call's own end even before it sets the stop.
                stopped_by_close = stop not in self._stops
                self._stops.discard(stop)
            stopped.cancel()
            # Lets the carrier abandon a call still running, and marks an ended call's error as seen.
            call.cancel()
        if stopped_by_close:
            raise PeerLostError(
                _named(
                    self.address, "coordinator call failed: this control plane closed"
                )
            )
        return call.result()

    async def _call_coordinator(
        self,
        endpoint: str,
        args: tuple[Any, ...],
        timeout_s: float | None,
    ) -> Any:
        try:
            coordinator = await self._coordinator_client()
            return await coordinator.call(endpoint, *args, timeout_s=timeout_s)
        except CarrierError as error:
            # A RemoteError is the coordinator's own error, which the carrier could not rebuild here.
            if isinstance(error, RemoteError):
                raise
            # Name the coordinator in transport errors.
            problem = str(error)
            if isinstance(error, PeerLostError) and problem.startswith(VOLUME_CLOSED):
                problem = (
                    "the coordinator service closed" + problem[len(VOLUME_CLOSED) :]
                )
            raise type(error)(
                _named(self.address, f"coordinator call failed: {problem}")
            ) from error


class _CoordinatorAdapter:
    """What the carrier serves: the registry's layout calls, as tasks on one dispatcher loop."""

    def __init__(self, registry: LayoutRegistry) -> None:
        self.registry = registry
        # Each rank's first layout per key, as its publisher sent it and its pickle's size, until the reply is encoded.
        self._parts: dict[str, dict[str, tuple[bytes, int]]] = {}
        # A rank's first layout wins, so complete layouts are encoded once for all requesters.
        self._encoded: dict[str, bytes] = {}

    async def init(
        self, publishers: Collection[str], strategy: TorchStoreStrategy
    ) -> None:
        """Initialize the registry, which refuses a second call."""
        await self.registry.init(publishers, strategy)

    def drop_encodings(self) -> None:
        """Drop every key's parts and reply."""
        self._parts, self._encoded = {}, {}

    def drop_layouts(self) -> None:
        """Drop every layout with its parts and replies, once no call can reach the registry."""
        self.registry = LayoutRegistry()
        self.drop_encodings()

    async def register_layout(
        self, rank: str, key: str, part: bytes, size: int
    ) -> None:
        # An honest size never asks for more than deflate can give so none takes unbounded memory.
        if type(size) is not int or not 0 <= size <= _DEFLATE_MAX_RATIO * len(part):
            raise ValueError(
                f"a layout part of {len(part)} bytes cannot inflate to {size!r} bytes"
            )
        pickled = _inflate_part(part, size)
        # zlib stops at a part's own last block, which would end every reader's joined stream there.
        if part[-len(_SYNC_FLUSH) :] != _SYNC_FLUSH:
            raise ValueError("a layout part must end with a sync flush")
        # @lint-ignore PYTHONPICKLEISBAD Layouts come from the store's own ranks.
        registrations = call_gc_paused(pickle.loads, pickled)
        await self.registry.register_layout(rank, key, registrations)
        if key in self._encoded:
            return
        # The registry keeps a rank's first layout, and no task runs between its return and here.
        ranks = self._parts.setdefault(key, {})
        if rank not in ranks:
            # Its pickle's own size: a reply then splits its stream right whatever size came with it.
            ranks[rank] = part, len(pickled)

    async def get_layouts(self, key: str, wait: float | None) -> pickle.PickleBuffer:
        """Return ``key``'s complete layouts as a buffer that the carrier sends uncopied."""
        encoded = self._encoded.get(key)
        if encoded is None:
            layouts = await self.registry.get_layouts(key, timeout_s=wait)
            # Lookups that one registration woke find the first one's encoding.
            encoded = self._encoded.get(key)
            if encoded is None:
                parts = self._parts.get(key, {})
                # A layout whose part went (as with drop_encodings) is compressed again here.
                encoded = _encode_layouts(
                    {
                        rank: parts.get(rank) or _compress_part(layout)
                        for rank, layout in layouts.items()
                    }
                )
                self._encoded[key] = encoded
                # Only now so that a failed encode leaves the parts for the next lookup.
                self._parts.pop(key, None)
        return pickle.PickleBuffer(encoded)


class CoordinatorService:
    """Serves one routing store's coordinator over the carrier, beside the store's TCPStore.

    This process initializes it through a handle from ``connect``, ranks reach it through ``address`` and
    ``close`` stops both servers. Decoding each layout a publisher registers pauses this process's cyclic collection.
    """

    def __init__(
        self, rendezvous: TCPStoreRendezvous, adapter: _CoordinatorAdapter
    ) -> None:
        served = rendezvous.address
        self.address = ServiceAddress(
            served.store_name, served.host, served.port, served.nonce
        )
        self._adapter = adapter
        self._pid = os.getpid()
        # Guards the fields below, and is never held across an await.
        self._lock = threading.Lock()
        self._rendezvous: TCPStoreRendezvous | None = rendezvous
        # Set once ``start`` serves the coordinator.
        self._server: CarrierServer | None = None
        _services.add(self)

    def __reduce_ex__(self, protocol: Any) -> NoReturn:
        raise TypeError(f"{type(self).__name__} cannot be pickled: pass its address")

    @classmethod
    async def start(
        cls, store_name: str, *, host: str | None = None
    ) -> CoordinatorService:
        """Serve a TCPStore under a fresh nonce, then the coordinator at the same host.

        Neither server authenticates, and any TCPStore client can list the nonce, so the
        job's network is trusted. Ranks dial ``host``, else ``$TORCHSTORE_TCPSTORE_HOST``,
        else the hostname. Nothing is written until ``init``.
        """
        adapter = _CoordinatorAdapter(LayoutRegistry())
        rendezvous = await TCPStoreRendezvous.serve(store_name, host=host)
        # Made at once, so a failed start closes the TCPStore it served.
        service = cls(rendezvous, adapter)
        try:
            served = rendezvous.address
            server = await CarrierServer.serve(
                adapter,
                _ENDPOINTS,
                service=_SERVICE,
                scope=f"{store_name}/{served.nonce}",
                host=served.host,
                concurrent=True,
            )
        except BaseException:
            await service.close()
            raise
        with service._lock:
            service._server = server
        return service

    async def connect(self) -> ServiceControlPlane:
        """Return a handle with a TCPStore connection of its own, which may call ``init``."""
        if os.getpid() != self._pid:
            raise self._error(
                "this coordinator service was inherited across fork."
                " Connect again from the address"
            )
        with self._lock:
            if self._rendezvous is None:
                raise self._error("this coordinator service is closed")
        rendezvous = await TCPStoreRendezvous.connect(self.address._tcpstore_address())
        return ServiceControlPlane(self.address, rendezvous, self)

    async def close(self) -> None:
        """Stop the coordinator, giving running calls ``_CLOSE_GRACE_S``, then drop its layouts and stop the TCPStore.

        A cancelled close finishes these steps in order on a thread of its own. If no thread can start, it stops the
        TCPStore at once and keeps the layouts until this service is freed. Later calls return at once.
        """
        with self._lock:
            server, self._server = self._server, None
            rendezvous, self._rendezvous = self._rendezvous, None
        try:
            if server is not None:
                await server.close(grace_s=_CLOSE_GRACE_S)
                self._adapter.drop_layouts()
        except asyncio.CancelledError:
            # The carrier finishes a close whose wait is cancelled, and calls in its grace still
            # read the layouts and the TCPStore. Only a finished close may drop them.
            if self._finish_close(cast(CarrierServer, server), rendezvous):
                rendezvous = None
            raise
        finally:
            if rendezvous is not None:
                await rendezvous.close()

    def _finish_close(
        self, server: CarrierServer, rendezvous: TCPStoreRendezvous | None
    ) -> bool:
        """Wait on a daemon thread for ``server``'s close, then drop the layouts and stop the TCPStore.

        Returns False if no thread starts, leaving the TCPStore to the caller and the layouts until this service is freed.
        """
        adapter = self._adapter

        async def finish() -> None:
            # A second close returns once the first, which the cancel left running, ends.
            await server.close(grace_s=_CLOSE_GRACE_S)
            adapter.drop_layouts()
            if rendezvous is not None:
                await rendezvous.close()

        started = True
        try:
            _start_thread(lambda: asyncio.run(finish()))
        except RuntimeError as error:
            started = False
            problem = (
                "a cancelled close stops the TCPStore at once and keeps the coordinator's layouts"
                f" until the service is freed, as no thread can wait for the coordinator: {error!r}"
            )
            logger.warning("%s", _named(self.address, problem))
        return started

    def _error(self, problem: str) -> RuntimeError:
        return RuntimeError(_named(self.address, problem))

    async def _init(
        self, publishers: Collection[str], strategy: TorchStoreStrategy
    ) -> None:
        """Set up the registry, then write publishers, strategy, coordinator, a key per publisher and ``ready``."""
        if os.getpid() != self._pid:
            raise self._error(
                "this coordinator service was inherited across fork: only the process"
                " that started it initializes it"
            )
        with self._lock:
            rendezvous, server = self._rendezvous, self._server
        # Close clears both, and no handle exists until start has served the coordinator.
        if rendezvous is None:
            raise self._error("this coordinator service is closed")
        # Before the registry: a publish the TCPStore would refuse then leaves init to a later call.
        try:
            prepared = rendezvous.prepare_publish(
                publishers,
                strategy,
                coordinator=cast(CarrierServer, server).address.to_json(),
            )
        except ValueError as error:
            raise ValueError(
                _named(self.address, f"cannot publish: {error}")
            ) from error
        # The one registry call off the dispatcher loop: init only sets attributes,
        # and no client reaches the registry before ``ready``.
        await self._adapter.init(publishers, strategy)
        await rendezvous.publish_prepared(prepared)


def _start_thread(target: Callable[[], object]) -> None:
    threading.Thread(
        target=target, name="torchstore-coordinator-close", daemon=True
    ).start()


class CarrierServiceBackend(ServiceBackend):
    """The service backend, which ``torchstore.backend.get_backend("service")`` returns."""

    async def start_coordinator(self, store_name: str) -> CoordinatorService:
        return await CoordinatorService.start(store_name)

    def service_address(self, control_plane: object) -> ServiceAddress | None:
        if isinstance(control_plane, ServiceControlPlane):
            return control_plane.address
        return None


def _after_fork_in_child() -> None:
    """Replace locks that a parent thread may hold, and drop inherited handles' TCPStore connections.

    Handles dial the TCPStore again when they next read it since an inherited carrier client still calls. The carrier's hook
    leaves servers inert, and the rendezvous module's keeps each served TCPStore and closes this child's copies of its sockets.
    """
    for handle in list(_handles):
        handle._lock = threading.Lock()
        handle._rendezvous = None
        # The parent's calls in flight, whose futures' locks a parent thread may hold.
        handle._stops = set()
    for service in list(_services):
        service._lock = threading.Lock()


os.register_at_fork(after_in_child=_after_fork_in_child)
# Last, so a load still running has not registered the backend yet.
register_backend("service", CarrierServiceBackend())
