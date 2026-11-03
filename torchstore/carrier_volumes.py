# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Storage volumes served over the carrier, and the host and handles that reach them from clients.

Internal and may change.
"""

from __future__ import annotations

import asyncio
import atexit
import collections
import itertools
import logging
import os
import threading
import time
import uuid
import weakref
from collections.abc import Awaitable, Callable, Collection, Iterable, Mapping
from concurrent import futures
from dataclasses import dataclass, field
from typing import Any, NamedTuple, NoReturn, TYPE_CHECKING

from torchstore import transport
from torchstore.carrier import (
    CarrierAddress,
    CarrierClient,
    CarrierProtocolError,
    CarrierServer,
    JsonRecord,
    wait_future,
)
from torchstore.storage import StorageVolumeCore
from torchstore.strategy import TorchStoreStrategy
from torchstore.transport import (
    gloo,
    GlooTransportBuffer,
    MonarchRPCTransportBuffer,
    SharedMemoryTransportBuffer,
    TransportType,
)
from torchstore.utils import _off_task, _store_address, start_daemon_thread
from torchstore.volume_host import VolumeHost

if TYPE_CHECKING:
    from torchstore.strategy import StorageVolumeRef
    from torchstore.transport.buffers import TransportContext

__all__ = [
    "CARRIER_TRANSPORTS",
    "CarrierVolumeHost",
    "check_strategy",
    "decode_volume_record",
    "encode_volume_record",
    "forget_context_pairs",
    "PublisherVolume",
    "serve_volume",
    "serves_another_publisher",
    "start_publisher_volume",
    "stop_publisher_volumes",
    "VOLUME_ENDPOINTS",
    "VolumeLocation",
    "VolumeProxy",
]

logger = logging.getLogger(__name__)

# Routed syncs call handshake, put, get and get_meta. STANDARD also deletes and resets.
VOLUME_ENDPOINTS = (
    "get_id",
    "handshake",
    "put",
    "get",
    "get_meta",
    "delete",
    "delete_batch",
    "reset",
)
# A failed one can leave a Gloo half waiting, and a failed put leaves the client's mappings of
# keys it already held, which the volume may no longer hold.
_TRANSFERS = frozenset(("handshake", "put", "get"))
# Keywords the proxy owns: CarrierClient.call's parameters and the pairs it retires.
_RESERVED_KEYWORDS = frozenset(("endpoint", "timeout_s", "retired_store_keys"))
# What a proxy carries and each transport's buffer. NIXL and TorchComms wait until a proxy keeps the caller
# memory their volumes read and write, which a failed call frees at once. MonarchRDMA stays on Monarch's
# backend.
_CARRIED = {
    TransportType.SharedMemory: SharedMemoryTransportBuffer,
    TransportType.Gloo: GlooTransportBuffer,
    TransportType.MonarchRPC: MonarchRPCTransportBuffer,
}
CARRIER_TRANSPORTS = frozenset(_CARRIED)
_CARRIED_BUFFERS = tuple(_CARRIED.values())
_RECORD_VERSION = 1
# Served volumes for the exit hook, held weakly so a dropped one frees its core and tensors.
_served: weakref.WeakSet[CarrierServer] = weakref.WeakSet()
# Carrier volume hosts for the fork hook, held weakly so a dropped one frees its proxies.
_hosts: weakref.WeakSet[CarrierVolumeHost] = weakref.WeakSet()
# Store keys of pairs that a proxy's handshake made and this process gave up, by the volume server
# holding their half, each with its age, oldest server first. The next call there drops those halves.
_retired: dict[CarrierAddress, list[tuple[int, str]]] = {}
# Keys that wait for the next call to queue them under their servers, with the age a failed call gave one back with.
# Appended without a lock because keys are given up under Gloo's pair-cache lock and in GC finalizers.
_retiring: collections.deque[
    tuple[CarrierAddress, str, int | None]
] = collections.deque()
# Ages for keys in the order calls queue them. Used under _retired_lock.
_ages = itertools.count()
# The volume server each pair that a proxy's handshake made reaches, so its key goes there alone.
_pair_servers: dict[str, CarrierAddress] = {}
# Keys a call has queued wait for at most this many servers, the server whose keys have waited longest going first.
# A process that re-makes stores leaves keys toward servers it never calls again. A 65536-rank job's volumes fit.
_RETIRED_SERVERS = 65536
# Held while a call takes its server's keys. Nothing that gives a key up takes it.
_retired_lock = threading.Lock()
# How long a stopping publisher volume gives its running call. Read when used so tests can
# shorten it.
_VOLUME_CLOSE_GRACE_S = 30.0
# How long a caller waits for another task's start of its volume before it raises. Read when
# used so tests can shorten it.
_START_WAIT_S = 150.0
# This process's publisher volumes by store name and volume ID. The lock guards the map and
# each volume's server and core. It is never held across an await.
_publisher_lock = threading.Lock()
_publishers: dict[tuple[str, str], PublisherVolume] = {}
# A forked child's inherited publisher volumes, kept so that none of their destructors runs there.
_inherited: list[PublisherVolume] = []


class VolumeLocation(NamedTuple):
    """A decoded volume record: where the volume's server listens, and its host's name.

    Clients may dial the host by another name, such as an IP address, so shared memory
    checks locality against ``hostname``.
    """

    address: CarrierAddress
    hostname: str


@dataclass(frozen=True)
class VolumeRecord(JsonRecord):
    """A volume record's fields: its version, its host's name and its server's address."""

    v: int
    hostname: str
    host: str
    port: int
    service: str
    scope: str = field(repr=False)
    incarnation: str


def check_strategy(strategy: TorchStoreStrategy) -> None:
    """Refuse a strategy that volume servers cannot honor, before anything depends on it.

    Raises:
        TypeError: If ``strategy`` is not exactly a ``TorchStoreStrategy``.
        ValueError: If its transport is neither ``Unset`` nor in ``CARRIER_TRANSPORTS``.
    """
    if type(strategy) is not TorchStoreStrategy:
        raise TypeError(
            "volume servers take a plain TorchStoreStrategy, not"
            f" {type(strategy).__qualname__}"
        )
    transport_type = strategy.default_transport_type
    # Typed first: an unhashable value would fail the membership test with TypeError.
    if not isinstance(transport_type, TransportType) or (
        transport_type is not TransportType.Unset
        and transport_type not in CARRIER_TRANSPORTS
    ):
        name = (
            transport_type.name
            if isinstance(transport_type, TransportType)
            else f"{transport_type!r:.64}"
        )
        raise ValueError(
            "volume servers carry SharedMemory, Gloo and MonarchRPC, or Unset to choose"
            f" among them, not {name}"
        )


def encode_volume_record(address: CarrierAddress, hostname: str) -> bytes:
    """Return a volume's record as versioned JSON: its server's address and host name."""
    record = VolumeRecord(
        _RECORD_VERSION,
        hostname,
        address.host,
        address.port,
        address.service,
        address.scope,
        address.incarnation,
    )
    return record.to_json().encode()


def decode_volume_record(volume_id: str, data: bytes) -> VolumeLocation:
    """Parse the record of volume ``volume_id`` that ``encode_volume_record`` wrote.

    Raises:
        CarrierProtocolError: If ``data`` is not this version's record of a server
            for ``volume_id``.
    """
    try:
        record = VolumeRecord.from_json(data)
    except CarrierProtocolError as e:
        # The codec's error never quotes the record, whose scope is the volume's only credential.
        raise CarrierProtocolError(
            f"volume {volume_id!r} has a malformed record: {e}"
        ) from None
    if record.v != _RECORD_VERSION:
        raise CarrierProtocolError(
            f"volume {volume_id!r} has a version {record.v} record,"
            f" not version {_RECORD_VERSION}"
        )
    if record.service != volume_id:
        raise CarrierProtocolError(
            f"volume {volume_id!r} has a record for service {record.service!r}"
        )
    address = CarrierAddress(
        record.host, record.port, record.service, record.scope, record.incarnation
    )
    return VolumeLocation(address, record.hostname)


async def serve_volume(
    core: StorageVolumeCore, *, scope: str, host: str | None = None
) -> tuple[CarrierServer, bytes]:
    """Serve ``core`` one call at a time, each connection's calls in the order sent. Return the server and record.

    Clients dial ``host``, else ``$TORCHSTORE_CARRIER_HOST``, else this host's name.
    A volume still served when the interpreter exits closes before finalization.
    """
    _, hostname = await core.get_id()
    server = await CarrierServer.serve(
        _ServedVolume(core),
        VOLUME_ENDPOINTS,
        service=core.volume_id,
        scope=scope,
        host=host or os.environ.get("TORCHSTORE_CARRIER_HOST") or None,
    )
    _served.add(server)
    return server, encode_volume_record(server.address, hostname)


class _ServedVolume:
    """A core's endpoints, each first dropping the volume halves of the pairs its caller retired."""

    def __init__(self, core: StorageVolumeCore) -> None:
        self._core = core

    def __getattr__(self, name: str) -> Callable[..., Awaitable[Any]]:
        method = getattr(self._core, name)

        async def endpoint(
            *args: Any, retired_store_keys: Collection[str] = (), **kwargs: Any
        ) -> Any:
            if retired_store_keys:
                # Calls run one at a time, so no handler uses these pairs now.
                context = self._core.store.transport_context
                groups = context.get(gloo.GlooProcessGroupCache)
                for store_key in retired_store_keys:
                    groups.pop(store_key)
            return await method(*args, **kwargs)

        return endpoint


@atexit.register
def _close_at_exit() -> None:
    """Close each volume still served, then park a dispatcher whose handler outlives Gloo's exit deadline.

    A dispatcher returning from a torch call in finalization can abort the process. A park first lets the running
    callback end. A loop of its own leaves the main thread's current loop to the exit hooks after this one.
    """
    deadline = gloo.begin_exit()
    loop = asyncio.new_event_loop()
    try:
        for server in list(_served):
            loop.run_until_complete(server.close(grace_s=0))
            dispatcher = server.dispatcher
            dispatcher.thread.join(max(deadline - time.monotonic(), 0.0))
            if dispatcher.thread.is_alive():
                dispatcher.park()
    finally:
        loop.close()


def forget_context_pairs(
    volume_ids: Iterable[str], transport_context: TransportContext
) -> None:
    """Forget the Gloo pair this process caches for each of ``volume_ids`` if ``transport_context`` holds its group.

    Each volume drops its half at this process's next call to it. Another context's pair stays.
    """
    groups = transport_context.get(gloo.GlooProcessGroupCache)
    for volume_id in volume_ids:
        store_key = gloo.cached_store_key(volume_id)
        if store_key is not None and groups.lookup(store_key) is not None:
            gloo.forget_pair(volume_id, store_key, transport_context)


def _retire(volume_id: str, store_key: str) -> None:
    # Only a pair that a proxy's handshake made has a carrier volume server holding its half.
    where = _pair_servers.pop(store_key, None)
    if where is not None:
        _retiring.append((where, store_key, None))


gloo.add_retired_listener(_retire)


def _take_retired(server: CarrierAddress) -> list[tuple[int, str]]:
    """Queue the keys given up since the last call, take ``server``'s whole, then drop the oldest past the bound.

    Returns ``server``'s keys oldest first, each with the age a failed call gives it back with.
    """
    with _retired_lock:
        given_back = False
        while _retiring:
            where, store_key, age = _retiring.popleft()
            given_back |= age is not None
            _retired.setdefault(where, []).append(
                (next(_ages) if age is None else age, store_key)
            )
        if given_back:
            # A failed call's keys are older than those queued since, so servers go back in order of their oldest.
            for keys in _retired.values():
                keys.sort()
            ordered = sorted(_retired.items(), key=lambda item: item[1][0])
            _retired.clear()
            _retired.update(ordered)
        taken = _retired.pop(server, [])
        while len(_retired) > _RETIRED_SERVERS:
            del _retired[next(iter(_retired))]
        return taken


def _buffer_of(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    return args[0] if args else kwargs.get("transport_buffer")


async def _call(
    client: CarrierClient, endpoint: str, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Any:
    volume_id = client.address.service
    retired: list[tuple[int, str]] = []
    try:
        reserved = sorted(_RESERVED_KEYWORDS.intersection(kwargs))
        if reserved:
            raise TypeError(
                f"volume endpoint {endpoint!r} got reserved keywords {reserved}"
            )
        if endpoint in _TRANSFERS:
            buffer = _buffer_of(args, kwargs)
            if not isinstance(buffer, _CARRIED_BUFFERS):
                raise TypeError(
                    "volume proxies carry SharedMemory, Gloo and MonarchRPC transfers,"
                    f" not {type(buffer).__name__}"
                )
            if endpoint == "handshake" and isinstance(buffer, GlooTransportBuffer):
                _pair_servers[buffer.store_key] = client.address
        retired = _take_retired(client.address)
        if retired:
            return await client.call(
                endpoint,
                *args,
                retired_store_keys=[store_key for _, store_key in retired],
                **kwargs,
            )
        return await client.call(endpoint, *args, **kwargs)
    except BaseException:
        if retired:
            # The volume may not have dropped them, so the next call sends them again. They keep their age.
            _retiring.extend(
                [(client.address, store_key, age) for age, store_key in retired]
            )
        if endpoint in _TRANSFERS:
            buffer = _buffer_of(args, kwargs)
            if isinstance(buffer, _CARRIED_BUFFERS):
                requests = args[1] if len(args) > 1 else kwargs.get("requests")
                _forget_transfer(volume_id, endpoint, buffer, requests)
        raise


def _forget_transfer(
    volume_id: str, endpoint: str, buffer: Any, requests: Iterable[Any] | None
) -> None:
    """Drop what a failed handshake, put or get may have stranded, given its buffer and requests.

    The buffer's own drop then lets go of its halves and the handshake's TCPStore master.
    """
    context = buffer.storage_volume_ref.transport_context
    if isinstance(buffer, GlooTransportBuffer) and buffer.store_key is not None:
        # Only this call's pair, if it used one: a newer one that another call cached is sound.
        gloo.forget_pair(volume_id, buffer.store_key, context)
        if endpoint == "handshake":
            # Cached nowhere here, but a volume that ran it late keeps its half.
            _retire(volume_id, buffer.store_key)
    if endpoint == "put" and requests is not None:
        # The buffer's drop discards new segments: this evicts the keys the client held.
        context.delete(request.key for request in requests)


class _Endpoint:
    """One volume endpoint, called as a Monarch actor endpoint is."""

    # Holds the client, not the proxy, so a dropped proxy frees its handle without a cyclic GC.
    __slots__ = ("_client", "_name")

    def __init__(self, client: CarrierClient, name: str) -> None:
        self._client = client
        self._name = name

    async def call_one(self, *args: Any, **kwargs: Any) -> Any:
        """Call the volume and return its reply."""
        return await _call(self._client, self._name, args, kwargs)

    async def call(self, *args: Any, **kwargs: Any) -> list[Any]:
        """Call the volume and return its reply in a list, one entry per volume."""
        return [await _call(self._client, self._name, args, kwargs)]


class VolumeProxy:
    """Reaches the volume served at ``address`` through endpoints shaped like a Monarch slice's.

    Errors arrive as ``CarrierClient.call`` gives them: an error the volume raised keeps its type
    when this process can rebuild it, else arrives as ``RemoteError``. A failed or cancelled
    handshake, put or get first drops the client state it may have stranded.
    """

    __slots__ = ("_client", *VOLUME_ENDPOINTS)
    # One per name in VOLUME_ENDPOINTS, set in __init__ and declared for type checkers.
    get_id: _Endpoint
    handshake: _Endpoint
    put: _Endpoint
    get: _Endpoint
    get_meta: _Endpoint
    delete: _Endpoint
    delete_batch: _Endpoint
    reset: _Endpoint

    def __init__(self, address: CarrierAddress) -> None:
        self._client = CarrierClient(address)
        for name in VOLUME_ENDPOINTS:
            setattr(self, name, _Endpoint(self._client, name))

    @property
    def volume_id(self) -> str:
        """The ID of the volume this proxy reaches, which its server serves as its service."""
        return self._client.address.service

    def __repr__(self) -> str:
        address = self._client.address
        return f"VolumeProxy({self.volume_id!r} at {address.host}:{address.port})"

    def __reduce_ex__(self, protocol: Any) -> NoReturn:
        raise TypeError("volume handles do not pickle: pass the volume record")

    async def close(self) -> None:
        """Close this proxy's carrier handle. Later calls raise ``PeerLostError``."""
        await self._client.close()


class _Entry(NamedTuple):
    """A volume this host resolves: its decoded record and its one handle."""

    location: VolumeLocation
    proxy: VolumeProxy


class CarrierVolumeHost(VolumeHost):
    """Resolves carrier volumes to ``VolumeProxy`` handles from their records.

    ``prepare`` reads the records it lacks with one ``resolver`` call on the sorted lacking ids,
    such as ``VolumeRecords.volume_records``, which must return each or raise. A volume's first
    record wins. Its handles hold connections. It never pickles, and only ``close`` closes them.
    """

    def __init__(
        self, resolver: Callable[[Collection[str]], Awaitable[Mapping[str, bytes]]], /
    ) -> None:
        self._resolver = resolver
        # Guards the fields below and is never held across an await.
        self._lock = threading.Lock()
        self._entries: dict[str, _Entry] = {}
        # Proxies taken by close. Every close closes them, so one that starts while
        # another runs never returns with a handle still open.
        self._retired: tuple[VolumeProxy, ...] = ()
        self._closed = False
        _hosts.add(self)

    def __repr__(self) -> str:
        state = "closed" if self._closed else f"volumes={len(self._entries)}"
        return f"CarrierVolumeHost({state})"

    def __reduce_ex__(self, protocol: Any) -> NoReturn:
        raise TypeError(
            "a carrier volume host does not pickle: each process builds its own"
        )

    def rebind(
        self, resolver: Callable[[Collection[str]], Awaitable[Mapping[str, bytes]]]
    ) -> None:
        """Read the records this host lacks through ``resolver`` from now on."""
        with self._lock:
            self._resolver = resolver

    def adopt(self, volume_id: str, record: bytes) -> None:
        """Resolve ``volume_id`` through a record this process holds, with no resolver call.

        A closed host raises RuntimeError whatever the record. Another volume's record raises
        CarrierProtocolError, and a record differing from the one held ValueError.
        """
        # Unlocked so that an adopt takes the lock once. A close racing past here fails ``_install``.
        self._check_open()
        location = decode_volume_record(volume_id, record)
        # Read unlocked since a racing install only makes this forget redundant.
        if not self._closed and volume_id not in self._entries:
            _forget_unheld_pairs([volume_id])
        with self._lock:
            held = self._entries.get(volume_id)
            if held is not None and held.location != location:
                raise ValueError(
                    f"volume {volume_id!r} already resolves through another record"
                )
            self._install({volume_id: location})

    async def prepare(self, volume_ids: Iterable[str], /) -> None:
        """Read the records this host lacks among ``volume_ids``, all in one resolver call.

        A closed host raises RuntimeError first, a bare str TypeError, a missing record KeyError, a malformed or
        misfiled one CarrierProtocolError, and a host closed before the records arrive RuntimeError.
        """
        # Unlocked as in adopt. The locked check below catches a close racing past here.
        self._check_open()
        if isinstance(volume_ids, str):
            raise TypeError(
                f"need an iterable of volume IDs, not the str {volume_ids!r}"
            )
        # Read first: a lazy iterable that reads this host would wait on its lock forever.
        wanted = set(volume_ids)
        with self._lock:
            self._check_open()
            lacking = sorted(wanted.difference(self._entries))
        if not lacking:
            return
        records = await self._resolver(lacking)
        missing = [volume_id for volume_id in lacking if volume_id not in records]
        if missing:
            raise KeyError(f"no volume record for {missing}")
        locations = {
            volume_id: decode_volume_record(volume_id, records[volume_id])
            for volume_id in lacking
        }
        if not self._closed:
            _forget_unheld_pairs(lacking)
        with self._lock:
            self._install(locations)

    def volume_ids(self) -> Collection[str]:
        with self._lock:
            return list(self._entries)

    def endpoints(self, volume_id: str) -> VolumeProxy:
        return self._entry(volume_id).proxy

    def hostname(self, volume_id: str) -> str:
        return self._entry(volume_id).location.hostname

    @property
    def transports(self) -> Collection[TransportType]:
        return CARRIER_TRANSPORTS

    def resolve_transport(self, ref: StorageVolumeRef, /) -> TransportType:
        """Keep SharedMemory, Gloo or MonarchRPC and choose among them when none is set. Another raises ValueError.

        ``ref`` must name this host's ``transports`` (``CARRIER_TRANSPORTS``) as a strategy's lookup does. Automatic
        selection then takes the first of them whose probe passes in the transport package's order, else MonarchRPC.
        """
        requested = ref.default_transport_type
        if requested in CARRIER_TRANSPORTS:
            return requested
        if requested is not TransportType.Unset:
            raise ValueError(
                f"volume {ref.volume_id!r} is served over the carrier, which carries"
                f" SharedMemory, Gloo and MonarchRPC, not {requested.name}"
            )
        return transport.get_available_transport(ref)

    async def close(self) -> None:
        """Close every handle and refuse later use.

        The host then lists no volumes, and ``endpoints``, ``hostname``, ``adopt`` and ``prepare``
        raise RuntimeError. Handles close together and a stuck carrier I/O thread delays this by
        one bounded wait. A cancel ends that wait at once with every handle closed.
        """
        with self._lock:
            self._closed = True
            self._retired += tuple(entry.proxy for entry in self._entries.values())
            self._entries = {}
            proxies = self._retired
        # A task's first step closes its handle before a cancel can wake this coroutine.
        # Cancelling the tasks below then only ends their waits on the I/O thread.
        closing = [asyncio.ensure_future(proxy.close()) for proxy in proxies]
        if not closing:
            return
        try:
            await asyncio.wait(closing)
        except asyncio.CancelledError:
            for task in closing:
                task.cancel()
            await asyncio.wait(closing)
            raise

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("this carrier volume host is closed")

    def _install(self, locations: Mapping[str, VolumeLocation]) -> None:
        """Give each new volume its one handle. Call with the lock held."""
        self._check_open()
        for volume_id, location in locations.items():
            if volume_id not in self._entries:
                proxy = VolumeProxy(location.address)
                self._entries[volume_id] = _Entry(location, proxy)

    def _entry(self, volume_id: str) -> _Entry:
        entry = self._entries.get(volume_id)
        if self._closed:
            raise RuntimeError(
                f"volume {volume_id!r} has no handle: its carrier volume host is closed"
            )
        if entry is None:
            raise KeyError(
                f"volume {volume_id!r} has no record here. Records load once a"
                " requester's first layouts return"
            )
        return entry


def _forget_unheld_pairs(volume_ids: Iterable[str]) -> None:
    """Forget this process's cached Gloo pair to each volume a host is about to resolve unless a live client holds it.

    Gloo caches one pair per volume ID for the whole process. Called outside the host's lock since
    ``reset_client`` may take it on a thread that holds Gloo's.
    """
    for volume_id in volume_ids:
        gloo.forget_unheld_pair(volume_id)


class PublisherVolume:
    """A publisher's volume that this process serves for one store, with its record.

    ``start_publisher_volume`` serves and registers it, and ``close`` stops serving it.
    """

    def __init__(
        self, volume_id: str, store_name: str, selection: tuple[Any, Any]
    ) -> None:
        self.volume_id = volume_id
        self.store_name = store_name
        # The (control plane, address) it serves: the address, else the plane's, names the store.
        self.selection = selection
        # Set when the start that made this volume ends: None, or why it failed.
        self.started: futures.Future[str | None] = futures.Future()
        # Both set once served and before ``started`` is. Readers that waited for ``started`` read both unlocked.
        self.record: bytes | None = None
        self.address: CarrierAddress | None = None
        # Taken by close. _publisher_lock guards both.
        self._server: CarrierServer | None = None
        self._core: StorageVolumeCore | None = None
        # Set by a withdraw under _publisher_lock or by _abandon when it finds that lock held.
        self._stopped = False
        # The event loop its start runs on. Held weakly so an abandoned start stays collectable.
        self._start_loop: weakref.ref[asyncio.AbstractEventLoop] | None = None

    def __repr__(self) -> str:
        return f"PublisherVolume({self.volume_id!r} of {self.store_name!r})"

    def current(self) -> bool:
        """Whether this process still serves this volume for its store: no stop took it."""
        with _publisher_lock:
            key = (self.store_name, self.volume_id)
            return _publishers.get(key) is self and not self._stopped

    def stop(self) -> bool:
        """Take this volume out of this process's map, so no start adopts it. True if this call did.

        ``close`` then stops serving it.
        """
        with _publisher_lock:
            return _withdraw(self)

    async def close(self) -> None:
        """Stop this volume and its server: queued calls fail, a running one gets the grace.

        The core resets only once no handler runs: a FIFO handler that outlives the grace,
        and every handler after a cancelled close, keeps running on its dispatcher thread.
        """
        self.stop()
        with _publisher_lock:
            server, self._server = self._server, None
            core, self._core = self._core, None
        if server is None or core is None:
            return
        try:
            await server.close(grace_s=_VOLUME_CLOSE_GRACE_S)
        except BaseException as error:
            cancelled = isinstance(error, asyncio.CancelledError)
            self._keep_core(
                "close was cancelled" if cancelled else f"close failed: {error!r}"
            )
            raise
        if server.stats()["running"]:
            self._keep_core("handler outlived the grace")
        else:
            await core.reset()

    def serves(self, selection: tuple[Any, Any]) -> bool:
        """Whether ``selection`` names this volume's store: the same control plane or an equal store address."""
        store = _store_address(*selection)
        return selection[0] is self.selection[0] or (
            store is not None and store == _store_address(*self.selection)
        )

    def _keep_core(self, why: str) -> None:
        # Its last reference is then the running handler's, which frees it when it returns.
        logger.warning(
            "routing volume=%s %s, so its core stays unreset", self.volume_id, why
        )

    def _abandoned(self) -> bool:
        """Whether its start's event loop closed or went. Such a start never finishes."""
        if self._start_loop is None:
            return False
        loop = self._start_loop()
        return loop is None or loop.is_closed()

    def _abandon(self) -> None:
        """Stop serving without waiting: the GC closed this volume's start, maybe in a locked step here."""
        if _publisher_lock.acquire(blocking=False):
            try:
                _withdraw(self)
            finally:
                _publisher_lock.release()
        else:
            # The next start of its ID takes it out of the map, if the close below has not yet.
            self._stopped = True
        _close_on_thread(self)

    async def _start(
        self,
        register: Callable[[str, bytes], Awaitable[None]],
        stale: PublisherVolume | None,
    ) -> None:
        """Stop ``stale``, then serve and register unless a stop took this volume meanwhile.

        Any failure closes and drops it, also a registration that timed out, lost its connection,
        was cancelled or was abandoned with its task, since no retry reuses its record.
        """
        task = asyncio.current_task()
        self._start_loop = weakref.ref(asyncio.get_running_loop())
        failure = None
        try:
            if stale is not None:
                await stale.close()
            core = StorageVolumeCore(self.volume_id)
            scope = f"{self.store_name}/{uuid.uuid4().hex}"
            server, record = await serve_volume(core, scope=scope)
            with _publisher_lock:
                stopped = self._stopped
                if not stopped:
                    self._server, self._core = server, core
                    self.record, self.address = record, server.address
            if stopped:
                # A stop took it while it served: its caller's build then finds it stopped.
                await server.close(grace_s=0)
                return
            await register(self.volume_id, record)
        except BaseException as error:
            failure = f"starting volume {self.volume_id!r}"
            if isinstance(error, asyncio.CancelledError):
                failure += " was cancelled in another task"
            elif isinstance(error, GeneratorExit):
                failure += " was abandoned in another task"
            else:
                failure += f" failed in another task: {error!r}"
            if _off_task(task):
                self._abandon()
            else:
                await self.close()
            raise
        finally:
            self.started.set_result(failure)


def _close_on_thread(volume: PublisherVolume) -> None:
    """Close ``volume`` on a thread of its own, which may wait for a lock that this thread holds."""

    def close() -> None:
        try:
            asyncio.run(volume.close())
        except Exception:
            logger.warning(
                "stopping routing volume=%s failed", volume.volume_id, exc_info=True
            )

    # A finalizing interpreter starts no thread, and the exit hook closed every served volume.
    start_daemon_thread(close, f"torchstore-volume-close {volume.volume_id}")


async def start_publisher_volume(
    volume_id: str,
    *,
    store_name: str,
    selection: tuple[Any, Any],
    register: Callable[[str, bytes], Awaitable[None]],
    sole: bool = False,
) -> PublisherVolume:
    """Serve this process's volume ``volume_id`` for one store, registered through ``register``.

    ``selection``, a (control plane, address) pair, names the store. Another store's volume of that
    name stops first, and with ``sole`` one of another ID under the store's name raises ValueError.
    Callers on any thread share one start, and one that fails closes its volume.
    """
    key = (store_name, volume_id)
    with _publisher_lock:
        # The service backend passes sole: a process registers as one publisher rank of a store.
        other = _other_publisher(store_name, volume_id) if sole else None
        if other is not None:
            raise ValueError(
                f"this process serves volume {other.volume_id} of routing store"
                f" {store_name!r}, so it cannot serve {volume_id} too. Call ts.shutdown first"
            )
        volume, stale = _publishers.get(key), None
        if volume is not None and volume._stopped:
            # A start the GC closed in a locked step stopped it without taking it out.
            _withdraw(volume)
            volume = None
        if volume is not None and not volume.serves(selection):
            stale, volume = volume, None
            _withdraw(stale)
        starting = volume is None
        if volume is None:
            volume = PublisherVolume(volume_id, store_name, selection)
            _publishers[key] = volume
        # Decided here, since a start that closes its volume leaves the map before it ends.
        waiting = not starting and not volume.started.done()
    if starting:
        await volume._start(register, stale)
        return volume
    if waiting:
        if not await wait_future(volume.started, _START_WAIT_S):
            if volume._abandoned():
                # With its loop gone it would never register or close. A live start still does.
                await volume.close()
            raise TimeoutError(
                f"starting volume {volume_id!r} in another task did not finish within"
                f" {_START_WAIT_S:g} s"
            )
        failure = volume.started.result()
        if failure is not None:
            raise RuntimeError(failure)
    return volume


def serves_another_publisher(store_name: str, volume_id: str) -> bool:
    """Whether this process serves a publisher volume of ``store_name`` other than ``volume_id``."""
    with _publisher_lock:
        return _other_publisher(store_name, volume_id) is not None


def _other_publisher(store_name: str, volume_id: str) -> PublisherVolume | None:
    """A volume of the store's name other than ``volume_id`` that no stop took. Hold the lock.

    ``ts.client`` and ``ts.shutdown`` take a store by its name. So a volume of the store started
    over at another address counts too.
    """
    return next(
        (
            held
            for (name, held_id), held in _publishers.items()
            if name == store_name and held_id != volume_id and not held._stopped
        ),
        None,
    )


def stop_publisher_volumes(store_name: str) -> list[PublisherVolume]:
    """Withdraw this process's publisher volumes of ``store_name``. The caller must ``close`` each."""
    with _publisher_lock:
        taken = [
            volume for (name, _), volume in _publishers.items() if name == store_name
        ]
        for volume in taken:
            _withdraw(volume)
    return taken


def _withdraw(volume: PublisherVolume) -> bool:
    """Mark ``volume`` stopped and take it out of the map. True if it was there. Hold the lock."""
    volume._stopped = True
    key = (volume.store_name, volume.volume_id)
    if _publishers.get(key) is not volume:
        return False
    del _publishers[key]
    return True


def _after_fork_in_child() -> None:
    # A thread that held a host's lock, the retired keys' lock or the publisher lock at the fork never
    # releases the child's copy, and the parent's pairs and servers are not the child's.
    global _retired_lock, _retired, _retiring, _pair_servers, _publisher_lock, _publishers
    for host in list(_hosts):
        host._lock = threading.Lock()
    _retired_lock = threading.Lock()
    _retired, _retiring, _pair_servers = {}, collections.deque(), {}
    _publisher_lock = threading.Lock()
    _inherited.extend(_publishers.values())
    _publishers = {}


os.register_at_fork(after_in_child=_after_fork_in_child)
