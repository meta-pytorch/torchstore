# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A TCPStore rendezvous through which routing ranks find their store.

Internal and may change. Loading it registers an exit hook that refuses native calls once it
runs, so register an exit hook that uses a rendezvous after this module loads.
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import ctypes
import json
import logging
import os
import socket
import threading
import time
import uuid
import weakref
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from functools import partial
from typing import Any, Generic, NamedTuple, NoReturn, TypeVar
from urllib.parse import quote

from torch.distributed import DistError, DistNetworkError, PrefixStore, TCPStore

from torchstore import native_calls
from torchstore.routing.volume_records import VolumeRecords
from torchstore.strategy import TorchStoreStrategy
from torchstore.transport import TransportType

__all__ = ["ExitingError", "TCPStoreAddress", "TCPStoreRendezvous"]

logger: logging.Logger = logging.getLogger(__name__)

_T = TypeVar("_T")

_HOST_ENV = "TORCHSTORE_TCPSTORE_HOST"
# The libuv server closes the connection on a longer wire key or a larger value.
_MAX_KEY_BYTES = 8 * 1024
_MAX_VALUE_BYTES = 8 * 1024 * 1024
# Reads go in chunks of at most 16 keys, so a reply holds at most 16 records, and a
# chunk's request (9 bytes, plus 8 and the wire key per key) stays within the server's
# 4 KiB read buffer unless the chunk is one longer key alone. The client still sends
# in 1440 B flushes, so a full chunk's request can take two or three reads.
_READ_CHUNK_KEYS = 16
_READ_CHUNK_BYTES = 4 * 1024
# A store call takes the GIL back as it returns and can wait a switch interval (5 ms by
# default) or more for it while other threads run Python. A native call's thread hop waits too.
# So a read makes its record checks and fetches in one native call, and a publish writes the
# per-publisher keys by one multi_set of up to this many per native call: 16 for 4,096 keys.
_MEMBER_CHUNK_KEYS = 256
# A volume_records KeyError names at most this many missing ids and its checks stop once they are found.
_NAMED_MISSING = 16
# get and multi_get wait on the store timeout internally, so it is never shortened.
_STORE_TIMEOUT = timedelta(seconds=300)
_DEFAULT_TIMEOUT_S = 60.0
_EXIT_DRAIN_S = 5.0
# A native attempt dials a looked-up address and, refused or unreachable, gives up
# within about 3 s, inside the exit drain, unless c10d's reverse lookup of that address
# stalls or the address accepts and never replies, which holds it until it hangs up.
# Connects retry in such attempts until their deadline.
_CONNECT_ATTEMPT_S = 0.5
_CONNECT_RETRY_S = 0.25
_MAX_RETRY_S = 8.0
_WAIT_OUT_POLL_S = 0.05
# A reload keeps the first load's state, which the hooks that load registered go on using.
if "_live_calls" not in globals():
    # Threads running a lane's call (a c10d call or a name lookup), added once they hold its turn.
    # At exit _drain_native_calls sets _exiting under the lock so that no call starts later, then joins them.
    _live_calls: set[threading.Thread] = set()
    _live_calls_lock = threading.Lock()
    _exiting = False
    # Served TCPStores and their PrefixStores, which a forked child must never destroy: a
    # call still holding a PrefixStore keeps a closed store's server alive.
    _masters: weakref.WeakSet[TCPStore | PrefixStore] = weakref.WeakSet()
    # Per name and per dialed address, the lane connects share, kept while one uses it or a
    # call given up on it has not returned.
    _connect_lanes: dict[str, _Lane] = {}
    _connect_lanes_lock = threading.Lock()
    # Listeners a serve holds open, to probe addresses or until its server takes one, which
    # a forked child closes. A fork waits while one opens or closes, so no child gets a copy
    # missing from the set.
    _probe_listeners: set[socket.socket] = set()
    _probe_listeners_lock = threading.Lock()
    # Ports of listeners handed to a native server whose store is not in _masters yet. A
    # fork waits while one is added or removed, and the child closes their sockets too.
    _handed_off_ports: set[int] = set()

_PUBLISHERS = "publishers"
_CONFIG = "config"
_COORDINATOR = "coordinator"
_READY = "ready"
# The value of each publisher's own key. Registration reads only whether the key exists.
_MEMBER = b"1"
# Stores of this schema hold those keys and the config in place of a pickled strategy, and their coordinator's
# replies carry volume records. Calls refuse any other schema at the ready check.
_SCHEMA_VERSION = b"4"
_CONFIG_VERSION = 1
# A config has one brace, and on Python 3.10 json's scanner nests as deep as the recursion limit.
_MAX_CONFIG_BRACKETS = 64
# What keeps a published value under the value limit, for the near-limit WARNING.
_SMALLER_VALUE = {_PUBLISHERS: "fewer or shorter publisher names"}


class ExitingError(RuntimeError):
    """Refuses a store call that would start once this process's exit has begun."""


class _QuietNetworkError(DistNetworkError):
    """A network failure found in Python, before any native attempt c10d would log."""


def _encode_publishers(publishers: Collection[str]) -> bytes:
    return json.dumps(sorted(set(publishers))).encode()


def _decode_publishers(data: bytes) -> frozenset[str]:
    return frozenset(json.loads(data))


def _encode_config(strategy: TorchStoreStrategy) -> bytes:
    """Encode only a plain strategy's transport as versioned JSON.

    Raises TypeError for any other strategy class and ValueError for a transport that is
    not a ``TransportType``. Which transports a store takes is its service's to check.
    """
    if type(strategy) is not TorchStoreStrategy:
        raise TypeError(
            "a config holds only a plain TorchStoreStrategy, not"
            f" {type(strategy).__qualname__}"
        )
    transport_type = strategy.default_transport_type
    if not isinstance(transport_type, TransportType):
        raise ValueError(f"a config names a TransportType, not {transport_type!r:.64}")
    fields = {"v": _CONFIG_VERSION, "transport": transport_type.name}
    return json.dumps(fields, sort_keys=True).encode()


def _decode_config(data: bytes) -> TorchStoreStrategy:
    """Rebuild the strategy ``_encode_config`` wrote. Anything else raises ValueError.

    Over 64 brackets fails before parsing, whatever the version. Otherwise the version is
    checked first, so a later format is reported as such. Errors quote at most 64 characters
    of what they read, since a value can be 8 MiB.
    """
    if (
        isinstance(data, (bytes, bytearray))
        and data.count(b"[") + data.count(b"{") > _MAX_CONFIG_BRACKETS
    ):
        raise ValueError(
            f"malformed config {data!r:.64}: over {_MAX_CONFIG_BRACKETS} brackets"
        )
    try:
        fields = json.loads(data)
        version = fields["v"]
    except (ValueError, TypeError, KeyError, RecursionError) as e:
        raise ValueError(f"malformed config {data!r:.64}: {type(e).__name__}") from e
    if type(version) is not int or version != _CONFIG_VERSION:
        raise ValueError(f"config version {version!r:.64}, not {_CONFIG_VERSION}")
    if fields.keys() != {"v", "transport"}:
        raise ValueError(
            f"malformed config {data!r:.64}: its fields are not v and transport"
        )
    try:
        return TorchStoreStrategy(TransportType[fields["transport"]])
    except (TypeError, KeyError) as e:
        # Not chained: an unknown name's KeyError holds all of it.
        raise ValueError(f"malformed config {data!r:.64}: {type(e).__name__}") from None


@dataclass(frozen=True)
class TCPStoreAddress:
    """Where a routing store's TCPStore listens, and which of its keys to use.

    The nonce is fresh for every ``serve``, so a stale address finds no
    ``ready`` key on a reused or shared store.
    """

    host: str
    port: int
    store_name: str
    nonce: str


class _Publish(NamedTuple):
    """What one publish writes, encoded and size-checked: its values by key, then the publishers' own keys."""

    values: dict[str, bytes]
    members: list[str]


def _endpoint(address: TCPStoreAddress) -> str:
    return f"{address.host}:{address.port}"


def _prefix(address: TCPStoreAddress) -> str:
    return f"torchstore/routing/{address.store_name}/{address.nonce}"


def _check_sizes(prefix: str, key: str, value: bytes = b"") -> int:
    """Reject what the server would drop the connection for, before sending.

    Counts the wire key, ``/{prefix}/{key}`` under a PrefixStore, in UTF-8 bytes.
    """
    key_size = len(f"/{prefix}/{key}".encode())
    if key_size > _MAX_KEY_BYTES:
        raise ValueError(
            f"TCPStore key {key!r} takes {key_size} bytes on the wire "
            f"(the limit is {_MAX_KEY_BYTES})"
        )
    if len(value) > _MAX_VALUE_BYTES:
        raise ValueError(
            f"TCPStore value for {key!r} has {len(value)} bytes "
            f"(the limit is {_MAX_VALUE_BYTES})"
        )
    return key_size


def _volume_key(volume_id: str) -> str:
    # One key segment, so no volume id can reach another key under the prefix.
    return f"volume/{quote(volume_id, safe='')}"


def _member_key(volume_id: str) -> str:
    # Written for each publisher before ready, so membership is one check of one key.
    return f"member/{quote(volume_id, safe='')}"


def _member_keys(prefix: str, publishers: Collection[str]) -> list[str]:
    """Return the publishers' own keys sorted and without repeats.

    A key over the wire cap raises ValueError. An id that UTF-8 cannot encode raises
    UnicodeEncodeError (a ValueError too).
    """
    members = sorted({_member_key(publisher) for publisher in publishers})
    for member in members:
        _check_sizes(prefix, member)
    return members


def _read_chunks(prefix: str, keys: Sequence[str]) -> list[list[str]]:
    """Split ``keys`` into chunks that each fit both read bounds above.

    A key whose request alone is over the byte budget gets a chunk of its own.
    """
    chunks: list[list[str]] = []
    request_size = 0
    for key in keys:
        key_size = 8 + _check_sizes(prefix, key)
        if (
            chunks
            and len(chunks[-1]) < _READ_CHUNK_KEYS
            and request_size + key_size <= _READ_CHUNK_BYTES
        ):
            chunks[-1].append(key)
            request_size += key_size
        else:
            chunks.append([key])
            request_size = 9 + key_size
    return chunks


def _start_store_call(what: str, deadline: float, given_up: Callable[[], bool]) -> None:
    """Refuse a store call named ``what`` in a native call once its deadline passes, its caller gives up or exit begins."""
    if not deadline - time.monotonic() > 0:
        raise TimeoutError(f"{what} was not started: its deadline had passed")
    if given_up():
        raise RuntimeError(f"{what} was not started: its caller gave up")
    if _exiting:
        raise ExitingError(f"{what} was not started: exiting")


def _read_records(
    ids: Mapping[str, str],
    chunks: Sequence[list[str]],
    start: Callable[[str], None],
    store: PrefixStore,
    *,
    existing_under: str | None = None,
) -> tuple[list[str], dict[str, bytes]]:
    """Return the ids under ``chunks``' keys that have no record and the records fetched by id.

    Checks each chunk and each key of a failed chunk right after it, calling ``start`` with each call's name first.
    Stops at ``_NAMED_MISSING`` missing ids and fetches once all exist, or with ``existing_under`` fetches what exists.
    """

    def exists(keys: list[str]) -> bool:
        start(f"check({keys!r})")
        return store.check(keys)

    missing: list[str] = []
    for chunk in chunks:
        if not exists(chunk):
            for key in chunk:
                if not exists([key]):
                    missing.append(ids[key])
                    # A KeyError names this many at most. A read for the records that exist, under their
                    # prefix ``existing_under``, checks every key.
                    if existing_under is None and len(missing) >= _NAMED_MISSING:
                        return missing, {}
    if missing:
        if existing_under is None:
            return missing, {}
        absent = set(missing)
        present = [key for chunk in chunks for key in chunk if ids[key] not in absent]
        chunks = _read_chunks(existing_under, present)
    records: dict[str, bytes] = {}
    for chunk in chunks:
        # multi_get blocks on a missing key. Records are never deleted.
        start(f"multi_get({chunk!r})")
        records.update(zip((ids[key] for key in chunk), store.multi_get(chunk)))
    return missing, records


def _resolve(host: str, port: int, attempt_s: float) -> list[str]:
    """Look ``host`` up in Python, so TCPStore's constructor only gets addresses.

    A stalled lookup that returns in finalization ends its thread quietly here, but
    aborts the process inside the constructor, where no attempt timeout bounds it.
    """
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except (OSError, UnicodeError) as error:  # a failed lookup, or a name IDNA rejects
        raise _QuietNetworkError(f"cannot resolve {host!r}: {error}") from None
    addresses = list(dict.fromkeys(str(info[4][0]) for info in infos))
    if len(addresses) < 2:
        return addresses
    return _answering_first(addresses, port, attempt_s)


def _answering_first(addresses: list[str], port: int, attempt_s: float) -> list[str]:
    """Start ``addresses`` at the first that accepts a connection on ``port``.

    TCPStore skips a refused address at once, but dialing a single one it retries it
    until its attempt times out. A serve, on port 0, probes a listener of its own.
    """
    with contextlib.ExitStack() as listening:
        if not port:
            try:
                port = listening.enter_context(_probe_listener()).getsockname()[1]
            except OSError:
                return addresses
        for index, numeric in enumerate(addresses):
            try:
                socket.create_connection((numeric, port), timeout=attempt_s).close()
            except OSError:
                continue
            return addresses[index:] + addresses[:index]
    return addresses


def _listener() -> socket.socket:
    """Listen on every interface, on a port the OS picks, as the libuv server does."""
    if socket.has_dualstack_ipv6():
        try:
            return socket.create_server(
                ("", 0), family=socket.AF_INET6, dualstack_ipv6=True
            )
        except OSError:
            pass
    return socket.create_server(("", 0))


@contextlib.contextmanager
def _probe_listener() -> Iterator[socket.socket]:
    """Hold a ``_listener`` open where a child forked meanwhile finds and closes it."""
    with _probe_listeners_lock:
        listener = _listener()
        _probe_listeners.add(listener)
    try:
        yield listener
    finally:
        with _probe_listeners_lock:
            listener.close()
            _probe_listeners.discard(listener)


def _check_answers(host: str, port: int, attempt_s: float) -> None:
    """Raise unless ``host`` accepts a connection on ``port``, dialing in Python.

    c10d logs each failed native attempt with a C++ backtrace and reports a refusal as a
    timeout, so serves and connects dial natively only an address that just answered.
    """
    try:
        socket.create_connection((host, port), timeout=attempt_s).close()
    except OSError as error:
        raise _QuietNetworkError(f"cannot connect to {host}:{port}: {error}") from None


def _open_store(host: str, port: int, attempt_s: float, *, is_master: bool) -> TCPStore:
    listen_fd = None
    if is_master:
        # The server takes this listener, so a serve dials its own host in Python too.
        with _probe_listener() as listener:
            port = listener.getsockname()[1]
            _check_answers(host, port, attempt_s)
            # Until the store is in _masters, a forked child finds the listener by its port.
            with _probe_listeners_lock:
                listen_fd = listener.detach()
                _handed_off_ports.add(port)
    else:
        _check_answers(host, port, attempt_s)
    try:
        # A refused attempt retries for up to about twice attempt_s plus a jittered backoff
        # (0.25-0.75 s, then 1.5x per retry). The store timeout bounds the waits in get,
        # multi_get, wait and blocking queue pops, of which this module calls only
        # multi_get: its other calls block until the server replies, hangs up or dies.
        store = TCPStore(
            host,
            port,
            is_master=is_master,
            wait_for_workers=False,
            timeout=timedelta(seconds=attempt_s),
            use_libuv=True,
            master_listen_fd=listen_fd,
        )
        store.set_timeout(_STORE_TIMEOUT)
        if is_master:
            _masters.add(store)
    finally:
        if is_master:
            with _probe_listeners_lock:
                _handed_off_ports.discard(port)
    return store


def _drain_native_calls() -> None:
    """Refuse new native calls, then give live ones a bounded chance to return."""
    global _exiting
    with _live_calls_lock:
        _exiting = True
        live = list(_live_calls)
    deadline = time.monotonic() + _EXIT_DRAIN_S
    for thread in live:
        thread.join(max(deadline - time.monotonic(), 0.0))
    alive = sorted(thread.name for thread in live if thread.is_alive())
    if alive:
        logger.warning(
            "Exiting with TCPStore calls still running after %.1f s, where a c10d call that "
            "returns during finalization aborts the process before Python 3.14: %s",
            _EXIT_DRAIN_S,
            ", ".join(alive),
        )


def _local_port(fd: int) -> int | None:
    """The local port of TCP socket ``fd``, or None for any other descriptor."""
    try:
        # Read through procfs: a stat or a dup's close would wait on a file whose FUSE or NFS server hangs.
        if not os.readlink(f"/proc/self/fd/{fd}").startswith("socket:"):
            return None
        duplicate = os.dup(fd)
    except OSError:
        return None
    try:
        sock = socket.socket(fileno=duplicate)
    except OSError:
        os.close(duplicate)
        return None
    with sock:
        try:
            if (
                sock.family in (socket.AF_INET, socket.AF_INET6)
                and sock.type == socket.SOCK_STREAM
            ):
                return sock.getsockname()[1]
        except OSError:
            pass
    return None


def _close_served_sockets(ports: set[int]) -> None:
    """Close this process's TCP sockets on local ``ports``: listeners and their connections.

    An outbound connection never gets a port that a listener holds, so nothing else matches.
    """
    try:
        fds = [int(name) for name in os.listdir("/proc/self/fd")]
    except OSError:
        return
    # With a default timeout, socket(fileno=) sets O_NONBLOCK, which the parent shares.
    timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(None)
    try:
        for fd in fds:
            if _local_port(fd) in ports:
                os.close(fd)
    finally:
        socket.setdefaulttimeout(timeout)


def _hold_probe_listeners() -> None:
    # Looked up at call time: the child replaces the lock, while its inherited copy stays held.
    _probe_listeners_lock.acquire()


def _release_probe_listeners() -> None:
    _probe_listeners_lock.release()


def _after_fork_in_child() -> None:
    """Forget the parent's calls, close its probe listeners and served sockets, never destroy its servers.

    Destroying a served TCPStore joins its server thread, which only the parent has, and this
    child's copies of its listener and connections would take connects and withhold EOF.
    """
    global _live_calls_lock, _live_calls, _exiting, _connect_lanes_lock
    global _probe_listeners_lock
    _live_calls_lock, _live_calls, _exiting = threading.Lock(), set(), False
    _connect_lanes_lock = threading.Lock()
    _connect_lanes.clear()
    _probe_listeners_lock = threading.Lock()
    ports = set(_handed_off_ports)
    _handed_off_ports.clear()
    for store in list(_masters):
        # Not a module list: finalization clears module globals, which would destroy the store.
        ctypes.pythonapi.Py_IncRef(ctypes.py_object(store))
        server = store.underlying_store if isinstance(store, PrefixStore) else store
        if isinstance(server, TCPStore):
            ports.add(server.port)
    # Those references keep the stores alive, and this child's own forks must not close sockets on their ports.
    _masters.clear()
    for listener in _probe_listeners:
        listener.close()
    _probe_listeners.clear()
    if ports:
        _close_served_sockets(ports)


# Once per module object, which a reload keeps: registering them again would take the probe lock
# twice at fork. A fresh import after the module left sys.modules registers its own.
if not globals().get("_hooks_registered", False):
    _hooks_registered = True
    atexit.register(_drain_native_calls)
    os.register_at_fork(after_in_child=_after_fork_in_child)
    os.register_at_fork(
        before=_hold_probe_listeners, after_in_parent=_release_probe_listeners
    )


async def _run_with_deadline(fn: Callable[[], _T], deadline: float, what: str) -> _T:
    """Run one blocking TCPStore call on a daemon thread, bounded by ``deadline``.

    A stuck call never delays ``asyncio.run``. Exit drains only calls a ``_Lane`` admitted,
    giving them ``_EXIT_DRAIN_S`` to return: before Python 3.14 a c10d call that returns in
    finalization aborts the process, while a name lookup ends quietly.
    """
    try:
        return await native_calls.run_with_deadline(
            fn, deadline, what, f"torchstore-tcpstore {what}"
        )
    except DistError as error:
        detail = str(error).splitlines()[0] if str(error) else type(error).__name__
        raise ConnectionError(f"{what} failed: {detail}") from error


class _NativeCall(Generic[_T]):
    """One native call, which never starts once its caller has given up on it."""

    def __init__(self, fn: Callable[[], _T]) -> None:
        self._fn: Callable[[], _T] | None = fn
        self._lock = threading.Lock()
        # Whether it runs now, and whether it ever began.
        self.running = self.started = False

    def __call__(self) -> _T:
        with self._lock:
            fn = self._fn
            self.running = self.started = fn is not None
        if fn is None:
            raise RuntimeError("its caller gave up on it before it started")
        try:
            return fn()
        finally:
            self.running = False

    def give_up(self) -> bool:
        """Keep the call from starting, let go of it, and return whether it runs."""
        with self._lock:
            self._fn = None
            return self.running

    @property
    def given_up(self) -> bool:
        return self._fn is None


class _Lane:
    """Runs native calls one at a time, and tells whether one given up on still runs.

    A connection's calls block one another natively, as a stuck server blocks every
    connect to it, so callers take turns here and later ones wait out a stuck call.
    """

    def __init__(self) -> None:
        self._gate = threading.Lock()
        # Set only by the holder of _gate, so no give-up on another thread can overwrite it.
        self._last: _NativeCall[Any] | None = None
        # Connects sharing this lane through _connect_lanes.
        self.users = 0

    @property
    def idle(self) -> bool:
        # Calls take the gate in turn, so only the last to take it can still run.
        last = self._last
        return last is None or not (last.running and last.given_up)

    async def run(self, call: _NativeCall[_T], deadline: float, what: str) -> _T:
        try:
            while not self.idle:
                if not deadline - time.monotonic() > 0:
                    raise TimeoutError(
                        f"{what} was not started: an earlier call has not returned"
                    )
                # The stuck call can end on any thread, for any event loop, so poll.
                await asyncio.sleep(_WAIT_OUT_POLL_S)
            admitted = partial(self._admit, call, deadline, what)
            try:
                return await _run_with_deadline(admitted, deadline, what)
            except TimeoutError:
                # Giving up first keeps a call reported as not started from starting.
                if call.give_up() or call.started:
                    raise
            raise TimeoutError(f"{what} was not started: its deadline had passed")
        finally:
            call.give_up()

    def _admit(self, call: _NativeCall[_T], deadline: float, what: str) -> Any:
        """Run ``call`` once no other call of the lane runs, if its deadline allows."""
        # Waits in slices, so a call given up while it waits ends its thread.
        while True:
            wait = min(max(deadline - time.monotonic(), 0.0), _WAIT_OUT_POLL_S)
            if self._gate.acquire(timeout=wait):
                break
            if call.given_up or not deadline - time.monotonic() > 0:
                return native_calls.EXPIRED
        try:
            self._last = call
            current = threading.current_thread()
            # Counted only once it holds the gate, so exit never waits for a queued call.
            with _live_calls_lock:
                if _exiting:
                    raise ExitingError(f"{what} was not started: exiting")
                _live_calls.add(current)
            try:
                return call()
            finally:
                with _live_calls_lock:
                    _live_calls.discard(current)
        finally:
            self._gate.release()


def _lane_busy(endpoint: str) -> bool:
    """Whether connects' lane to ``endpoint`` still runs a call given up on."""
    with _connect_lanes_lock:
        lane = _connect_lanes.get(endpoint)
        return lane is not None and not lane.idle


@contextlib.contextmanager
def _shared_lane(endpoint: str) -> Iterator[_Lane]:
    """The lane connects to ``endpoint`` share. A connect leaving its lane drops every idle lane none uses."""
    with _connect_lanes_lock:
        lane = _connect_lanes.setdefault(endpoint, _Lane())
        lane.users += 1
    try:
        yield lane
    finally:
        with _connect_lanes_lock:
            lane.users -= 1
            # Only a connect using a lane can leave a stuck call on it, so an idle
            # lane that none uses has nothing to keep.
            for kept_endpoint, kept in list(_connect_lanes.items()):
                if kept.idle and not kept.users:
                    del _connect_lanes[kept_endpoint]


async def _open(
    host: str,
    port: int,
    timeout_s: float,
    what: str,
    *,
    is_master: bool,
    lane_for: Callable[[str], contextlib.AbstractContextManager[_Lane]],
) -> TCPStore:
    """Open a TCPStore within ``timeout_s``, in short native attempts.

    A network failure is retried while the next attempt could still fail natively in
    time, so a caller usually gets the failure itself rather than a timeout.
    """
    deadline = time.monotonic() + timeout_s
    attempt_s = min(_CONNECT_ATTEMPT_S, timeout_s / 3)
    addresses: list[str] = []
    attempts = native_failures = 0
    while True:
        limited = False
        try:
            if not addresses:
                resolve = _NativeCall(partial(_resolve, host, port, attempt_s))
                with lane_for(f"{host}:{port}") as lane:
                    addresses = await lane.run(resolve, deadline, what)
            # Each attempt dials the next address, so a failing one holds up no other.
            numeric = addresses[attempts % len(addresses)]
            if not is_master:
                # An address still holding an abandoned attempt would make this one wait it out.
                for skip in range(len(addresses)):
                    candidate = addresses[(attempts + skip) % len(addresses)]
                    if not _lane_busy(f"{candidate}:{port}"):
                        numeric, attempts = candidate, attempts + skip
                        break
            attempts += 1
            limit = deadline
            if not is_master and len(addresses) > 1:
                # TCPStore's handshake has no timeout, so an attempt moves on from an address
                # that accepts and never replies, as a stale one may, while another fits in time.
                bounded = time.monotonic() + 6 * attempt_s
                if deadline - bounded > 3 * attempt_s:
                    limit, limited = bounded, True
            call = _NativeCall(
                partial(_open_store, numeric, port, attempt_s, is_master=is_master)
            )
            with lane_for(f"{numeric}:{port}") as lane:
                return await lane.run(call, limit, what)
        except TimeoutError:
            if not limited:
                raise
            pause = 0.0
        except ConnectionError as error:
            if not isinstance(error.__cause__, DistNetworkError):
                raise
            # c10d logs each failed native attempt, so retries after one pause longer.
            if not isinstance(error.__cause__, _QuietNetworkError):
                native_failures += 1
            # Capping the exponent keeps a long run of failures from overflowing a float.
            pause = min(_CONNECT_RETRY_S * 2 ** min(native_failures, 30), _MAX_RETRY_S)
            if deadline - time.monotonic() < 3 * attempt_s + pause:
                raise
        await asyncio.sleep(pause)


class TCPStoreRendezvous(VolumeRecords):
    """One connection to a routing store's keys on a TCPStore.

    The driver calls ``serve`` and publishes the publisher ranks, the config and a key per publisher once. Other
    ranks ``connect`` to its address to read them back and to register and read volume records. ``timeout_s`` bounds
    a whole call, 60 s if None. TimeoutError and ConnectionError name the server.
    """

    def __init__(self, address: TCPStoreAddress, store: TCPStore) -> None:
        self.address = address
        self._store: TCPStore | None = store
        self._prefixed: PrefixStore | None = PrefixStore(_prefix(address), store)
        if store in _masters:
            _masters.add(self._prefixed)
        # A forked child shares this connection's socket, so it must not use it.
        self._pid = os.getpid()
        self._lane = _Lane()

    def __reduce_ex__(self, protocol: Any) -> NoReturn:
        raise TypeError(f"{type(self).__name__} cannot be pickled: pass the address")

    @classmethod
    async def serve(
        cls,
        store_name: str,
        *,
        host: str | None = None,
        timeout_s: float | None = None,
    ) -> TCPStoreRendezvous:
        """Start a TCPStore for ``store_name`` and connect to it.

        The server has no authentication: it listens on every interface and trusts the job's
        network. Clients and this process dial ``host``, else ``$TORCHSTORE_TCPSTORE_HOST``, else the
        hostname. The server holds a descriptor per client, so this process's open-file limit must exceed its ranks.
        """
        timeout_s = _DEFAULT_TIMEOUT_S if timeout_s is None else timeout_s
        host = host or os.environ.get(_HOST_ENV) or socket.gethostname()
        lane = _Lane()
        store = await _open(
            host,
            0,
            timeout_s,
            f"starting a TCPStore for {host}",
            is_master=True,
            lane_for=lambda _: contextlib.nullcontext(lane),
        )
        address = TCPStoreAddress(host, store.port, store_name, uuid.uuid4().hex)
        logger.info(
            "Serving routing store %r on TCPStore %s", store_name, _endpoint(address)
        )
        return cls(address, store)

    @classmethod
    async def connect(
        cls,
        address: TCPStoreAddress,
        *,
        timeout_s: float | None = None,
    ) -> TCPStoreRendezvous:
        """Connect to the TCPStore that ``address`` names, within ``timeout_s``.

        Refusals and failed lookups are retried until under 2 s of ``timeout_s`` is left, so
        a connect to a server that exited fails only then, about 58 s in by default.
        """
        timeout_s = _DEFAULT_TIMEOUT_S if timeout_s is None else timeout_s
        endpoint = _endpoint(address)
        store = await _open(
            address.host,
            address.port,
            timeout_s,
            f"connecting to TCPStore {endpoint}",
            is_master=False,
            lane_for=_shared_lane,
        )
        return cls(address, store)

    async def register_volume(
        self,
        volume_id: str,
        record: bytes,
        /,
        *,
        timeout_s: float | None = None,
    ) -> None:
        """Write a publisher's volume record once. The same record again returns.

        ``timeout_s=None`` means 60 s. Non-publisher ids raise KeyError and another record RuntimeError. Before
        sending, a record that is not bytes raises TypeError and an empty or oversized record or long id ValueError.
        """
        key = _volume_key(volume_id)
        if not isinstance(record, bytes):
            # A str would be stored and then compare unequal to the bytes read back.
            raise TypeError(
                f"volume {volume_id!r} has a {type(record).__name__} record, not bytes"
            )
        if not record:
            # compare_set would let a later record replace an empty one.
            raise ValueError(f"volume {volume_id!r} has an empty record")
        key_size = _check_sizes(_prefix(self.address), key, record)
        member = _member_key(volume_id)
        timeout_s = _DEFAULT_TIMEOUT_S if timeout_s is None else timeout_s
        deadline = time.monotonic() + timeout_s
        await self._require_ready(deadline)
        # Both are written before ready, so a missing publisher set means a malformed store.
        if not await self._check([member], deadline):
            if not await self._check([_PUBLISHERS], deadline):
                raise self._error("marked ready without its publishers")
            raise KeyError(f"{volume_id!r} is not a publisher rank")
        stored = await self._call(
            f"compare_set({key!r})",
            lambda store: store.compare_set(key, "", record),  # type: ignore[arg-type]
            deadline,
        )
        if stored != record:
            raise self._error(
                f"volume {volume_id!r} is already registered with another record"
            )
        # Warns near either cap as _compare_set does.
        near_limit = 2 * key_size > _MAX_KEY_BYTES or 2 * len(record) > _MAX_VALUE_BYTES
        logger.log(
            logging.WARNING if near_limit else logging.INFO,
            "Volume %r has its record of %d bytes at TCPStore key %r (%d key bytes) on %s%s",
            volume_id,
            len(record),
            key,
            key_size,
            _endpoint(self.address),
            (
                f". A key over {_MAX_KEY_BYTES} bytes or a record over {_MAX_VALUE_BYTES} bytes"
                " fails registration: shorten the store name"
            )
            if near_limit
            else "",
        )

    async def volume_records(
        self,
        volume_ids: Collection[str],
        /,
        *,
        timeout_s: float | None = None,
    ) -> dict[str, bytes]:
        """Return the record of each id in ``volume_ids``, in chunks and never waiting.

        ``timeout_s=None`` means 60 s. A KeyError names the ids without a record, 16 at most, before any is
        fetched, and an id over the key cap raises ValueError before sending. The store calls after the ready
        marker's share one native call.
        """
        ids = {_volume_key(volume_id): volume_id for volume_id in volume_ids}
        if not ids:
            return {}
        chunks = _read_chunks(_prefix(self.address), list(ids))
        timeout_s = _DEFAULT_TIMEOUT_S if timeout_s is None else timeout_s
        deadline = time.monotonic() + timeout_s
        await self._require_ready(deadline)
        missing, records = await self._steps(
            f"reading {len(ids)} volume records",
            partial(_read_records, ids, chunks),
            deadline,
        )
        if missing:
            first = (
                ""
                if len(missing) < _NAMED_MISSING
                else f" (the first {len(missing)} found)"
            )
            raise KeyError(
                self._named(f"no volume record for {sorted(missing)}{first}")
            )
        return records

    async def existing_volume_records(
        self,
        volume_ids: Collection[str],
        /,
        *,
        timeout_s: float | None = None,
    ) -> dict[str, bytes]:
        """Return the records among ``volume_ids`` and leave out ids without one. ``timeout_s=None`` means 60 s.

        Reads through ``volume_records``' reader with the same size checks and store calls in one native call
        after the ready check, but fetches the records that exist in chunks cut afresh from their keys.
        """
        ids = {_volume_key(volume_id): volume_id for volume_id in volume_ids}
        if not ids:
            return {}
        prefix = _prefix(self.address)
        chunks = _read_chunks(prefix, list(ids))
        timeout_s = _DEFAULT_TIMEOUT_S if timeout_s is None else timeout_s
        deadline = time.monotonic() + timeout_s
        await self._require_ready(deadline)
        _, records = await self._steps(
            f"reading {len(ids)} volume records",
            partial(_read_records, ids, chunks, existing_under=prefix),
            deadline,
        )
        return records

    async def publish(
        self,
        publishers: Collection[str],
        strategy: TorchStoreStrategy,
        *,
        coordinator: str | None = None,
        timeout_s: float | None = None,
    ) -> None:
        """Write the publisher ranks, config, any coordinator and a key per publisher, then ``ready``.

        Raises TypeError or ValueError before any write if a config cannot hold the strategy, a key or
        value is over its limit, or UTF-8 cannot encode a publisher id. Raises RuntimeError if the store
        already holds other published content.
        """
        prepared = self.prepare_publish(publishers, strategy, coordinator=coordinator)
        await self.publish_prepared(prepared, timeout_s=timeout_s)

    def prepare_publish(
        self,
        publishers: Collection[str],
        strategy: TorchStoreStrategy,
        /,
        *,
        coordinator: str | None = None,
    ) -> _Publish:
        """Encode and size-check what ``publish`` writes, raising as it would before any store call.

        ``publish_prepared`` then writes it. A caller can thus refuse a publish before work of its own.
        """
        # Walked twice below: a one-shot iterable still gets every publisher's own key.
        publishers = list(publishers)
        values = {
            _PUBLISHERS: _encode_publishers(publishers),
            _CONFIG: _encode_config(strategy),
        }
        if coordinator is not None:
            values[_COORDINATOR] = coordinator.encode()
        members = _member_keys(_prefix(self.address), publishers)
        for key, value in [*values.items(), (_READY, _SCHEMA_VERSION)]:
            _check_sizes(_prefix(self.address), key, value)
        return _Publish(values, members)

    async def publish_prepared(
        self, prepared: _Publish, /, *, timeout_s: float | None = None
    ) -> None:
        """Write what ``prepare_publish`` returned, as ``publish`` does. ``timeout_s=None`` means 60 s."""
        timeout_s = _DEFAULT_TIMEOUT_S if timeout_s is None else timeout_s
        deadline = time.monotonic() + timeout_s
        for key, value in prepared.values.items():
            await self._write_once(key, value, deadline)
        await self._write_members(prepared.members, deadline)
        await self._write_once(_READY, _SCHEMA_VERSION, deadline)

    async def read(
        self,
        *,
        timeout_s: float | None = None,
    ) -> tuple[frozenset[str], TorchStoreStrategy]:
        """Return the publisher ranks and the config's bare strategy, without waiting.

        Raises:
            RuntimeError: If this address has no complete publish under this schema, or
                its config is malformed or of a later version.
        """
        timeout_s = _DEFAULT_TIMEOUT_S if timeout_s is None else timeout_s
        deadline = time.monotonic() + timeout_s
        await self._require_ready(deadline)
        rest = [_PUBLISHERS, _CONFIG]
        # Written before ready, so these are missing only from a malformed store.
        if not await self._check(rest, deadline):
            raise self._error("marked ready without its publishers or config")
        publishers, config = await self._multi_get(rest, deadline)
        try:
            strategy = _decode_config(config)
        except ValueError as e:
            raise self._error(str(e)) from e
        return _decode_publishers(publishers), strategy

    async def read_strategy(
        self,
        *,
        timeout_s: float | None = None,
        check: Callable[[TorchStoreStrategy], None] | None = None,
    ) -> TorchStoreStrategy:
        """Return the config's bare strategy alone, without waiting or reading the publisher ranks.

        Raises:
            RuntimeError: If this address has no complete publish under this schema, its config is
                malformed or of a later version, or ``check`` refuses its strategy with ValueError.
        """
        timeout_s = _DEFAULT_TIMEOUT_S if timeout_s is None else timeout_s
        deadline = time.monotonic() + timeout_s
        await self._require_ready(deadline)
        # Written before ready, so it is missing only from a malformed store.
        if not await self._check([_CONFIG], deadline):
            raise self._error("marked ready without its config")
        (config,) = await self._multi_get([_CONFIG], deadline)
        try:
            strategy = _decode_config(config)
            if check is not None:
                check(strategy)
        except ValueError as e:
            raise self._error(str(e)) from e
        return strategy

    async def read_coordinator(
        self,
        *,
        timeout_s: float | None = None,
    ) -> str:
        """Return the published coordinator address, without waiting.

        Raises:
            RuntimeError: If this address has no ready marker under this schema, or no coordinator address.
        """
        timeout_s = _DEFAULT_TIMEOUT_S if timeout_s is None else timeout_s
        deadline = time.monotonic() + timeout_s
        await self._require_ready(deadline)
        if not await self._check([_COORDINATOR], deadline):
            raise self._error("marked ready without a coordinator")
        (coordinator,) = await self._multi_get([_COORDINATOR], deadline)
        return coordinator.decode()

    async def close(self) -> None:
        """Drop this connection. A served TCPStore stops with its last handle."""
        # The store's destructor holds the GIL, so running it on a thread would
        # still block the loop.
        self._prefixed = self._store = None

    async def _require_ready(self, deadline: float) -> None:
        """Fail at once unless the ready marker exists with this schema version."""
        # multi_get blocks on a missing key for the store timeout but check does not.
        if not await self._check([_READY], deadline):
            # The control-plane interface's text for a store not yet set up, which callers match.
            raise self._error("init has not run")
        (version,) = await self._multi_get([_READY], deadline)
        if version != _SCHEMA_VERSION:
            raise self._error(
                f"schema version {version!r}, expected {_SCHEMA_VERSION!r}"
            )

    def _error(self, problem: str) -> RuntimeError:
        return RuntimeError(self._named(problem))

    def _named(self, problem: str) -> str:
        return (
            f"routing store {self.address.store_name!r} on TCPStore "
            f"{_endpoint(self.address)}: {problem}"
        )

    async def _call(
        self, what: str, fn: Callable[[PrefixStore], _T], deadline: float
    ) -> _T:
        # Only the call holds the store. A kept error holds this frame and must not keep the store.
        call = _NativeCall(partial(fn, self._live_store()))
        return await self._lane.run(
            call,
            deadline,
            f"{what} on TCPStore {_endpoint(self.address)}",
        )

    async def _steps(
        self,
        what: str,
        fn: Callable[[Callable[[str], None], PrefixStore], _T],
        deadline: float,
    ) -> _T:
        """Make ``fn``'s store calls in one native call to spare each a thread hop.

        ``fn`` passes each store call's name to its first argument before making it. That argument
        raises once ``deadline`` has passed, the caller has given up or exit has begun.
        """
        endpoint = _endpoint(self.address)

        def start(step: str) -> None:
            _start_store_call(
                f"{step} on TCPStore {endpoint}", deadline, lambda: call.given_up
            )

        call = _NativeCall(partial(fn, start, self._live_store()))
        return await self._lane.run(call, deadline, f"{what} on TCPStore {endpoint}")

    def _live_store(self) -> PrefixStore:
        """Return this connection's store. A forked child or a closed connection gets RuntimeError."""
        if os.getpid() != self._pid:
            raise RuntimeError(
                f"connection to TCPStore {_endpoint(self.address)} was inherited "
                "across fork. Connect again from the address"
            )
        prefixed = self._prefixed  # read once: close() may run on another thread
        if prefixed is None:
            raise RuntimeError(
                f"connection to TCPStore {_endpoint(self.address)} is closed"
            )
        return prefixed

    async def _write_members(self, members: Sequence[str], deadline: float) -> None:
        """Write the publisher keys by one multi_set per native call of up to ``_MEMBER_CHUNK_KEYS`` keys.

        Only a publish whose publisher list matched the stored one gets here, and each writes ``_MEMBER`` to
        the same keys. multi_set's last write therefore wins harmlessly. A failed chunk ends the publish.
        """
        if not members:
            return
        endpoint = _endpoint(self.address)
        longest = max(members, key=lambda member: len(member.encode()))
        key_size = _check_sizes(_prefix(self.address), longest)
        near_cap = 2 * key_size > _MAX_KEY_BYTES
        # One line per publish rather than per publisher, a warning near the key cap as in _compare_set.
        logger.log(
            logging.WARNING if near_cap else logging.INFO,
            "Writing a key per publisher, %d in all, the longest %r (%d key bytes), to TCPStore %s%s",
            len(members),
            longest,
            key_size,
            endpoint,
            f". A key over {_MAX_KEY_BYTES} bytes fails the publish: shorten the store name"
            if near_cap
            else "",
        )
        for start in range(0, len(members), _MEMBER_CHUNK_KEYS):
            chunk = list(members[start : start + _MEMBER_CHUNK_KEYS])
            # multi_set returns before the server applies it. Ready's compare_set later on this connection
            # returns only once these keys are in: the server applies one connection's requests in order.
            await self._call(
                f"multi_set of {len(chunk)} publisher keys from {chunk[0]!r}",
                lambda store, keys=chunk: store.multi_set(keys, [_MEMBER] * len(keys)),  # type: ignore[arg-type]
                deadline,
            )

    async def _write_once(self, key: str, value: bytes, deadline: float) -> None:
        """Write ``value`` by compare_set, raising RuntimeError if ``key`` holds another non-empty value."""
        if await self._compare_set(key, value, deadline) != value:
            raise self._error(f"store already initialized with a different {key!r}")

    async def _compare_set(self, key: str, value: bytes, deadline: float) -> bytes:
        key_size = _check_sizes(_prefix(self.address), key, value)
        smaller = _SMALLER_VALUE.get(key, "a smaller value")
        near: list[tuple[str, str]] = []
        if 2 * key_size > _MAX_KEY_BYTES:
            near.append((f"{_MAX_KEY_BYTES} key bytes", "a shorter store name"))
        if 2 * len(value) > _MAX_VALUE_BYTES:
            near.append((f"{_MAX_VALUE_BYTES} value bytes", smaller))
        written = (len(value), key, key_size, _endpoint(self.address))
        if near:
            logger.warning(
                "Writing %d bytes to TCPStore key %r (%d key bytes) on %s, over half the %s of %s"
                " past which a write fails: use %s",
                *written,
                "limits" if len(near) > 1 else "limit",
                " and ".join(limit for limit, _ in near),
                " and ".join(remedy for _, remedy in near),
            )
        else:
            logger.info(
                "Writing %d bytes to TCPStore key %r (%d key bytes) on %s", *written
            )
        # set returns before the server applies it but compare_set waits for the
        # reply, keeps the first non-empty value, and returns the stored value.
        return await self._call(
            f"compare_set({key!r})",
            lambda store: store.compare_set(key, "", value),  # type: ignore[arg-type]
            deadline,
        )

    async def _check(self, keys: Sequence[str], deadline: float) -> bool:
        for key in keys:
            _check_sizes(_prefix(self.address), key)
        return await self._call(
            f"check({list(keys)!r})", lambda store: store.check(list(keys)), deadline
        )

    async def _multi_get(self, keys: Sequence[str], deadline: float) -> list[bytes]:
        for key in keys:
            _check_sizes(_prefix(self.address), key)
        return await self._call(
            f"multi_get({list(keys)!r})",
            lambda store: store.multi_get(list(keys)),
            deadline,
        )
