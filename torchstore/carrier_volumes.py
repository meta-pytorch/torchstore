# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Storage volumes served over the carrier, and the handles that reach them from clients.

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
import weakref
from collections.abc import Awaitable, Callable, Collection, Iterable
from dataclasses import dataclass, field
from typing import Any, NamedTuple, NoReturn, TYPE_CHECKING

from torchstore.carrier import (
    CarrierAddress,
    CarrierClient,
    CarrierProtocolError,
    CarrierServer,
    JsonRecord,
)
from torchstore.transport import (
    gloo,
    GlooTransportBuffer,
    MonarchRPCTransportBuffer,
    SharedMemoryTransportBuffer,
    TransportType,
)

if TYPE_CHECKING:
    from torchstore.storage import StorageVolumeCore

__all__ = [
    "CARRIER_TRANSPORTS",
    "decode_volume_record",
    "encode_volume_record",
    "serve_volume",
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


def _after_fork_in_child() -> None:
    # A thread that held the lock at the fork never releases the child's copy, and the parent's
    # pairs are not the child's to retire.
    global _retired_lock, _retired, _retiring, _pair_servers
    _retired_lock = threading.Lock()
    _retired, _retiring, _pair_servers = {}, collections.deque(), {}


os.register_at_fork(after_in_child=_after_fork_in_child)
