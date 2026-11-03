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
import struct
import threading
import time
import weakref
import zlib
from collections.abc import Awaitable, Callable, Collection, Iterable, Mapping
from concurrent import futures
from dataclasses import dataclass, field
from typing import Any, cast, NoReturn, TYPE_CHECKING

from torchstore.backend import register_backend, ServiceBackend

from torchstore.carrier import (
    CarrierAddress,
    CarrierClient,
    CarrierError,
    CarrierProtocolError,
    CarrierServer,
    PeerLostError,
    RemoteError,
    VOLUME_CLOSED,
    wait_future,
)
from torchstore.carrier_volumes import (
    CarrierVolumeHost,
    check_strategy,
    forget_context_pairs,
    PublisherVolume,
    serves_another_publisher,
    start_publisher_volume,
    stop_publisher_volumes,
)
from torchstore.logging import LatencyTracker
from torchstore.strategy import _installed_volume_host
from . import layout_registry, tcpstore
from ._gc import call_gc_paused
from .control_plane import ControlPlane, ControlPlaneAddress
from .layout_registry import LayoutRegistry
from .tcpstore import TCPStoreAddress, TCPStoreRendezvous
from .volume_records import VolumeRecords

if TYPE_CHECKING:
    from torchstore.strategy import TorchStoreStrategy
    from torchstore.transport.buffers import TransportContext
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
# Records reads start at this many registrations, or at any lookup in a store this large. A later pass starts
# once this many publishers that registered after the previous pass began have no record found, or at the last
# registration. A smaller store reads once, at its last registration.
_RECORDS_PASS = 16
# On Llama 3 70B and Kimi K2.5 layouts: 7 to 10% smaller than level 6, and level 9's bytes in less time.
_PART_ZLIB_LEVEL = 8
# The empty last block that ends a raw deflate stream of parts.
_LAST_BLOCK = b"\x03\x00"
# How every part ends so that parts join into one stream: a sync flush's empty stored block.
_SYNC_FLUSH = b"\x00\x00\xff\xff"
# Deflate's largest ratio of inflated to deflated bytes.
_DEFLATE_MAX_RATIO = 1032
# A reply is its layouts' pickle, which ``pickle.loads`` reads alone, then the store's volume
# records, pickled and compressed, then this trailer: their size and the reply's version.
_REPLY_TRAILER = struct.Struct("<IB")
_REPLY_VERSION = 1

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


def _encode_layouts(
    parts: Mapping[str, tuple[bytes, int]], *, tail: bytes = b""
) -> bytes:
    """Pickle a reply of the ranks, their pickles' sizes and their parts, which a reader joins into one stream.

    ``tail`` follows the pickle in the same buffer, so appending it never copies the reply.
    """
    ranks, pieces = list(parts), list(parts.values())
    buffer = io.BytesIO()
    # @lint-ignore PYTHONPICKLEISBAD Layouts must reach ranks in other processes.
    pickler = pickle.Pickler(buffer, protocol=5)
    # Safe without a memo: the reply holds only rank names, sizes and parts, none shared.
    pickler.fast = True
    pickler.dump((ranks, [size for _, size in pieces], [part for part, _ in pieces]))
    buffer.write(tail)
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


def _encode_records(records: Mapping[str, bytes]) -> bytes:
    """Pickle and compress volume records into the section that ends a reply, with its trailer."""
    # zlib's default level: the parts' level 8 takes 3 to 4 times as long for about 1% fewer bytes of records.
    # @lint-ignore PYTHONPICKLEISBAD Records reach ranks in other processes with the layouts.
    section = zlib.compress(pickle.dumps(dict(records), protocol=5))
    return section + _REPLY_TRAILER.pack(len(section), _REPLY_VERSION)


def _split_reply(
    address: ServiceAddress, data: Any
) -> tuple[memoryview, dict[str, bytes]]:
    """Split a reply into its layouts' pickle and the volume records that follow it.

    Raises:
        CarrierProtocolError: If it is shorter than a trailer, its trailer has another version or leaves no room for
            layouts, or its records do not load as bytes per id.
    """
    view = memoryview(data)
    end = view.nbytes - _REPLY_TRAILER.size
    size, version = _REPLY_TRAILER.unpack_from(view, end) if end >= 0 else (0, None)
    if version != _REPLY_VERSION:
        problem = (
            f"the coordinator's reply does not end in a version {_REPLY_VERSION} trailer. The driver may run"
            " another TorchStore version: run one TorchStore version in the driver and every rank"
        )
        raise CarrierProtocolError(_named(address, problem))
    if size >= end:
        problem = (
            f"the coordinator's reply trailer names {size} bytes of volume records,"
            f" which leaves no layouts in the {end} bytes before it"
        )
        raise CarrierProtocolError(_named(address, problem))
    malformed = "the coordinator's reply holds malformed volume records"
    try:
        # Unpaused: zlib yields the GIL as it inflates, and records map ids to bytes, which collection does not track.
        # @lint-ignore PYTHONPICKLEISBAD Records come from the store's own coordinator.
        records = pickle.loads(zlib.decompress(view[end - size : end]))
    except Exception as error:
        raise CarrierProtocolError(
            _named(address, f"{malformed}: {error!r}")
        ) from error
    if type(records) is not dict:
        raise CarrierProtocolError(_named(address, malformed))
    for volume_id, record in records.items():
        if type(volume_id) is not str or type(record) is not bytes or not record:
            problem = f"the coordinator's reply holds a malformed record for volume {volume_id!r}"
            raise CarrierProtocolError(_named(address, problem))
    return view[: end - size], records


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


class ServiceControlPlane(ControlPlane, VolumeRecords):
    """Reads a store's strategy from TCPStore and calls its coordinator, whose replies carry volume records.

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
        # Volume records from the coordinator's replies.
        self._records: dict[str, bytes] = {}
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
        """Decode a fresh strategy from TCPStore on every call, resolving volumes from records.

        A config naming a transport volume servers do not carry fails the read naming the store.
        """
        tracker = _latency_tracker("strategy")
        try:
            rendezvous = await self._state()
            strategy = await rendezvous.read_strategy(check=check_strategy)
            strategy.set_volume_host(CarrierVolumeHost(self.volume_records))
            return strategy
        finally:
            tracker.track_e2e()

    async def register_volume(self, volume_id: str, record: bytes, /) -> None:
        tracker = _latency_tracker("register_volume")
        try:
            rendezvous = await self._state()
            await rendezvous.register_volume(volume_id, record)
        finally:
            tracker.track_e2e()

    async def volume_records(self, volume_ids: Collection[str], /) -> dict[str, bytes]:
        """Return records the coordinator's replies carried, reading TCPStore only for the ids they lack."""
        tracker = _latency_tracker("volume_records")
        try:
            rendezvous = await self._state()
            ids = list(dict.fromkeys(volume_ids))
            with self._lock:
                held = {
                    volume_id: self._records[volume_id]
                    for volume_id in ids
                    if volume_id in self._records
                }
            lacking = [volume_id for volume_id in ids if volume_id not in held]
            read = await rendezvous.volume_records(lacking)
            return {
                volume_id: held[volume_id] if volume_id in held else read[volume_id]
                for volume_id in ids
            }
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
        """Wait for every publisher's layout for ``key``, then decode a fresh copy and keep the reply's records.

        ``timeout_s=None`` waits the caller's ``_LAYOUT_REGISTRATION_TIMEOUT_S``, 300 s by default. The deadline is any wait
        above 0 plus ``_REPLY_MARGIN_S``, after a first call's 60 s address read and in a forked child a 60 s dial before it.
        """
        tracker = _latency_tracker(f"get_layouts key={key}")
        try:
            wait = _wait_s(timeout_s)
            # Written so that a NaN wait, which the registry checks once, gets the margin too.
            deadline_s = (wait if wait > 0 else 0) + _REPLY_MARGIN_S
            data = await self._call("get_layouts", key, wait, timeout_s=deadline_s)
            layouts, records = _split_reply(self.address, data)
            registrations = _decode_layouts(layouts)
            with self._lock:
                if not self._closed:
                    self._records.update(records)
            return registrations
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
            self._records = {}
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


class _ReplyRecords:
    """A store's volume records as the section that ends every reply, read in passes as its publishers register.

    The first pass asks for every record, at the 16th registration or a first lookup, or at a smaller store's last
    registration. Later ones ask for publishers that registered since the previous pass began and lack a found
    record. Their own thread makes them with no turn on the dispatcher loop, but decodes there still hold the GIL.
    """

    def __init__(
        self, rendezvous: TCPStoreRendezvous, publishers: Collection[str] = ()
    ) -> None:
        self.rendezvous = rendezvous
        self._publishers = sorted(publishers)
        # The records in hand once the passes end or fail or the closing service stops them, set under the
        # lock below. Lookups wait for it.
        self._section: futures.Future[bytes] = futures.Future()
        # Guards every field below, which registrations, lookups, a closing service and the passes share.
        self._lock = threading.Lock()
        # Wakes the passes' thread once a pass falls due or the section is set.
        self._wake = threading.Condition(self._lock)
        # Publishers with no layout yet. Each registers its record before its first layout.
        self._unregistered = set(self._publishers)
        # Records read so far, and what the next pass asks for: every publisher for the first, then registered
        # publishers lacking one that no pass has asked for since they registered.
        self._found: dict[str, bytes] = {}
        self._due: set[str] = set()
        # Whether ``_start`` ran, though it may have started no thread, and the task of a pass in flight.
        self._started = False
        self._task: asyncio.Task[tuple[float, bool]] | None = None
        # Half a lookup's reply margin after the last registration, by when the passes after it must end.
        self._deadline: float | None = None
        # Lookups waiting for the section, and whether the service is closing.
        self._waiting = 0
        self._closing = False

    def registered(self, rank: str) -> None:
        """Count a publisher's first layout. The 16th, or a smaller store's last, starts the passes."""
        with self._lock:
            if rank not in self._unregistered:
                return
            self._unregistered.remove(rank)
            if rank not in self._found:
                self._due.add(rank)
            if not self._unregistered:
                self._deadline = time.monotonic() + _REPLY_MARGIN_S / 2
            if not self._started:
                if (
                    self._unregistered
                    and len(self._publishers) - len(self._unregistered) < _RECORDS_PASS
                ):
                    return
                self._start()
            elif self._pass_due():
                # Only then, so the passes' thread takes the GIL once per pass, not once per registration.
                self._wake.notify()

    def wanted(self) -> None:
        """Start the passes for a lookup in a store of 16 or more publishers, before its registrations end."""
        with self._lock:
            if not self._started and len(self._publishers) >= _RECORDS_PASS:
                self._start()

    def _start(self) -> None:
        """Start the passes' thread, whose first pass asks for every record. Call with the lock held."""
        self._started = True
        self._due = set(self._publishers)
        if self._section.done():
            # The closing service stopped the reads before they began.
            return
        try:
            _start_thread("torchstore-coordinator-records", self._run)
        except RuntimeError as error:
            # The OS or the interpreter refused the thread.
            self._section.set_result(_encode_records({}))
            logger.warning(
                "routing store %r: replies carry no volume records, as no thread could read them: %r",
                self.rendezvous.address.store_name,
                error,
            )

    def _pass_due(self) -> bool:
        """Whether 16 ids are due or no publisher is left to register. Call with the lock held."""
        return len(self._due) >= _RECORDS_PASS or not self._unregistered

    async def section(self) -> bytes:
        """Return the section once the reads end, waiting without cancelling them."""
        section = self._section
        with self._lock:
            self._waiting += 1
        try:
            await wait_future(section, None)
        finally:
            with self._lock:
                self._waiting -= 1
                # A closing server cancels the last lookup before it stops this loop.
                if self._closing and not self._waiting:
                    self._stop()
        return section.result()

    def cancel(self) -> None:
        """Stop the reads as the service closes, at once unless a lookup waits for them, else as the last one leaves."""
        with self._lock:
            self._closing = True
            if not self._waiting:
                self._stop()

    def _stop(self) -> None:
        """Set the section with the records in hand and cancel a pass in flight. Call with the lock held."""
        if self._section.done():
            return
        self._section.set_result(_encode_records(self._found))
        task = self._task
        if task is not None:
            task.get_loop().call_soon_threadsafe(task.cancel)
        self._wake.notify()

    def _run(self) -> None:
        """Make passes as they fall due, waiting between them, then set the section unless the closing service has."""
        seconds = 0.0
        while True:
            with self._wake:
                self._wake.wait_for(lambda: self._section.done() or self._pass_due())
            try:
                took, ended = asyncio.run(self._passes())
            except asyncio.CancelledError:
                # The closing service stopped the reads and set the section.
                return
            except Exception as error:
                self._finish(seconds, error)
                return
            seconds += took
            if ended:
                self._finish(seconds, None)
                return

    async def _passes(self) -> tuple[float, bool]:
        """Make passes while one is due, each reading the records that exist among the ids due.

        Returns their seconds, and whether no record can come any more.

        Raises:
            asyncio.CancelledError: Once the closing service has stopped the reads, before a pass or during one.
        """
        took = 0.0
        while True:
            with self._lock:
                if self._section.done():
                    raise asyncio.CancelledError
                if len(self._found) == len(self._publishers) or not (
                    self._unregistered or self._due
                ):
                    return took, True
                if not self._pass_due():
                    return took, False
                asked, self._due = self._due, set()
                # Half a lookup's reply margin, so a reply sent after a slow read meets its deadline, counted from
                # the last registration once that has come.
                timeout_s = (
                    _REPLY_MARGIN_S / 2
                    if self._deadline is None
                    else self._deadline - time.monotonic()
                )
                # Under the lock that a stop takes, so a stop from here on cancels this pass.
                self._task = asyncio.current_task()
            start = time.monotonic()
            try:
                records = await self.rendezvous.existing_volume_records(
                    sorted(asked), timeout_s=timeout_s
                )
            finally:
                with self._lock:
                    self._task = None
            took += time.monotonic() - start
            with self._lock:
                self._found.update(records)
                self._due -= self._found.keys()

    def _finish(self, seconds: float, failure: Exception | None) -> None:
        """Log what replies carry, then set the section with the records in hand unless the closing service has.

        No pass runs any more, so a section the closing service set holds the same records. A read refused as the driver
        exits logs nothing, and lookups that still come get the records found before it.
        """
        with self._lock:
            found = dict(self._found)
        section = _encode_records(found)
        store, total = self.rendezvous.address.store_name, len(self._publishers)
        if failure is None:
            logger.info(
                "routing store %r: replies carry %d of %d volume records, read in %.3fs",
                store,
                len(found),
                total,
                seconds,
            )
        elif not isinstance(failure, tcpstore.ExitingError):
            logger.warning(
                "routing store %r: replies carry %d of %d volume records, as a read failed: %r",
                store,
                len(found),
                total,
                failure,
            )
        with self._lock:
            if not self._section.done():
                self._section.set_result(section)


class _CoordinatorAdapter:
    """What the carrier serves: the registry's layout calls, as tasks on one dispatcher loop."""

    def __init__(
        self, registry: LayoutRegistry, rendezvous: TCPStoreRendezvous
    ) -> None:
        self.registry = registry
        # Where replies' volume records are read.
        self._rendezvous = rendezvous
        # The publishers from ``init``, whose registrations start the store's records reads.
        self._publishers: frozenset[str] = frozenset()
        # Each rank's first layout per key, as its publisher sent it and its pickle's size, until the reply is encoded.
        self._parts: dict[str, dict[str, tuple[bytes, int]]] = {}
        # A rank's first layout wins, so complete layouts are encoded once for all requesters.
        self._encoded: dict[str, bytes] = {}
        self._records = _ReplyRecords(rendezvous)

    async def init(
        self, publishers: Collection[str], strategy: TorchStoreStrategy
    ) -> None:
        """Initialize the registry, which refuses a second call, then set up the store's records for its publishers."""
        await self.registry.init(publishers, strategy)
        self._publishers = frozenset(publishers)
        self.drop_encodings()

    def drop_encodings(self) -> None:
        """Drop every key's parts and reply, and the store's records."""
        self._parts, self._encoded = {}, {}
        self._records = _ReplyRecords(self._rendezvous, self._publishers)

    def stop_reading(self) -> None:
        """Stop the store's records reads, at once unless a lookup waits for them."""
        self._records.cancel()

    def drop_layouts(self) -> None:
        """Drop every layout with its parts, replies and records, once no call can reach the registry."""
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
        self._records.registered(rank)
        if key in self._encoded:
            return
        # The registry keeps a rank's first layout, and no task runs between its return and here.
        ranks = self._parts.setdefault(key, {})
        if rank not in ranks:
            # Its pickle's own size: a reply then splits its stream right whatever size came with it.
            ranks[rank] = part, len(pickled)

    async def get_layouts(self, key: str, wait: float | None) -> pickle.PickleBuffer:
        """Return ``key``'s complete layouts and the store's records as a buffer that the carrier sends uncopied."""
        encoded = self._encoded.get(key)
        if encoded is None:
            # In a store of 16 or more publishers, any lookup starts the reads before its registrations end.
            self._records.wanted()
            layouts = await self.registry.get_layouts(key, timeout_s=wait)
            section = await self._records.section()
            # Lookups that one registration woke find the first one's encoding.
            encoded = self._encoded.get(key)
            if encoded is None:
                parts = self._parts.get(key, {})
                # A layout whose part went (as with drop_encodings) is compressed again here.
                encoded = _encode_layouts(
                    {
                        rank: parts.get(rank) or _compress_part(layout)
                        for rank, layout in layouts.items()
                    },
                    tail=section,
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
        rendezvous = await TCPStoreRendezvous.serve(store_name, host=host)
        adapter = _CoordinatorAdapter(LayoutRegistry(), rendezvous)
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

        A store call that the records reads still make holds the TCPStore until it returns. A cancelled close finishes
        these steps in order on a thread of its own. If no thread can start, it stops the TCPStore at once and keeps
        the layouts until this service is freed. Later calls return at once.
        """
        with self._lock:
            server, self._server = self._server, None
            rendezvous, self._rendezvous = self._rendezvous, None
        try:
            if server is not None:
                # First: the reads stop now unless a lookup waits for them, and lookups after the stop reply
                # with the records read before it.
                self._adapter.stop_reading()
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
            _start_thread("torchstore-coordinator-close", lambda: asyncio.run(finish()))
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
        """Set up the registry, then write publishers, config, coordinator, a key per publisher and ``ready``."""
        if os.getpid() != self._pid:
            raise self._error(
                "this coordinator service was inherited across fork: only the process"
                " that started it initializes it"
            )
        # First, so that a strategy volume servers cannot honor leaves init to a later call.
        check_strategy(strategy)
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


def _start_thread(name: str, target: Callable[[], object]) -> None:
    threading.Thread(target=target, name=name, daemon=True).start()


class CarrierServiceBackend(ServiceBackend):
    """The service backend, which ``torchstore.backend.get_backend("service")`` returns."""

    async def start_coordinator(self, store_name: str) -> CoordinatorService:
        return await CoordinatorService.start(store_name)

    def service_address(self, control_plane: object) -> ServiceAddress | None:
        if isinstance(control_plane, ServiceControlPlane):
            return control_plane.address
        return None

    def check_strategy(self, strategy: TorchStoreStrategy) -> None:
        check_strategy(strategy)

    def volume_host(self, strategy: object) -> CarrierVolumeHost | None:
        host = _installed_volume_host(strategy)
        return host if isinstance(host, CarrierVolumeHost) else None

    def forget_context_pairs(
        self, volume_ids: Iterable[str], transport_context: TransportContext
    ) -> None:
        forget_context_pairs(volume_ids, transport_context)

    async def start_publisher_volume(
        self,
        volume_id: str,
        *,
        store_name: str,
        selection: tuple[Any, Any],
        register: Callable[[str, bytes], Awaitable[None]],
    ) -> PublisherVolume:
        return await start_publisher_volume(
            volume_id,
            store_name=store_name,
            selection=selection,
            register=register,
            sole=True,
        )

    def stop_publisher_volumes(self, store_name: str) -> list[PublisherVolume]:
        return stop_publisher_volumes(store_name)

    def serves_another_publisher(self, store_name: str, volume_id: str) -> bool:
        return serves_another_publisher(store_name, volume_id)


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
        # Its close then stops fresh records and not the parent's reads. A parent thread may hold their locks.
        service._adapter.drop_encodings()


os.register_at_fork(after_in_child=_after_fork_in_child)
# Last, so a load still running has not registered the backend yet.
register_backend("service", CarrierServiceBackend())
