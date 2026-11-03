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
import pickle
import socket
import threading
import time
import uuid
import weakref
import zlib
from collections.abc import Callable, Collection, Iterator, Sequence
from dataclasses import dataclass
from datetime import timedelta
from functools import partial
from typing import Any, Generic, NoReturn, TYPE_CHECKING, TypeVar

from torch.distributed import DistError, DistNetworkError, PrefixStore, TCPStore

from torchstore import native_calls

if TYPE_CHECKING:
    from torchstore.strategy import TorchStoreStrategy

__all__ = ["TCPStoreAddress", "TCPStoreRendezvous"]

logger: logging.Logger = logging.getLogger(__name__)

_T = TypeVar("_T")

_HOST_ENV = "TORCHSTORE_TCPSTORE_HOST"
# The libuv server closes the connection on a longer wire key or a larger value.
_MAX_KEY_BYTES = 8 * 1024
_MAX_VALUE_BYTES = 8 * 1024 * 1024
# Readers stop inflating a strategy past this, so no small value takes unbounded memory.
_MAX_STRATEGY_BYTES = 256 * 1024 * 1024
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
_STRATEGY = "strategy"
_COORDINATOR = "coordinator"
_READY = "ready"
_SCHEMA_VERSION = b"1"
# What keeps a published value under the value limit, for the near-limit WARNING.
_SMALLER_VALUE = {
    _PUBLISHERS: "fewer or shorter publisher names",
    _STRATEGY: "a smaller strategy",
}


class _QuietNetworkError(DistNetworkError):
    """A network failure found in Python, before any native attempt c10d would log."""


def _encode_publishers(publishers: Collection[str]) -> bytes:
    return json.dumps(sorted(set(publishers))).encode()


def _decode_publishers(data: bytes) -> frozenset[str]:
    return frozenset(json.loads(data))


def _encode_strategy(strategy: TorchStoreStrategy) -> bytes:
    # @lint-ignore PYTHONPICKLEISBAD Volume handles must reach other processes.
    pickled = pickle.dumps(strategy)
    if len(pickled) > _MAX_STRATEGY_BYTES:
        raise ValueError(
            f"the strategy pickles to {len(pickled)} bytes "
            f"(the limit is {_MAX_STRATEGY_BYTES})"
        )
    # Its volume handles repeat each volume's ids, so this shrinks it about 14 times.
    return zlib.compress(pickled)


def _decode_strategy(data: bytes) -> TorchStoreStrategy:
    inflate = zlib.decompressobj()
    pickled = inflate.decompress(data, _MAX_STRATEGY_BYTES + 1)
    if len(pickled) > _MAX_STRATEGY_BYTES:
        raise ValueError(f"the strategy inflates past {_MAX_STRATEGY_BYTES} bytes")
    # @lint-ignore PYTHONPICKLEISBAD Volume handles must reach other processes.
    return pickle.loads(pickled)


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
                    raise RuntimeError(f"{what} was not started: exiting")
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


class TCPStoreRendezvous:
    """One connection to a routing store's keys on a TCPStore.

    The driver calls ``serve`` and publishes the publisher ranks and strategy
    once. Other ranks ``connect`` to its address and read them back. ``timeout_s``
    bounds a whole call, 60 s if None. TimeoutError and ConnectionError name the server.
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

    async def publish(
        self,
        publishers: Collection[str],
        strategy: TorchStoreStrategy,
        *,
        coordinator: str | None = None,
        timeout_s: float | None = None,
    ) -> None:
        """Write the publisher ranks, the strategy and any coordinator, then ``ready``.

        Raises:
            ValueError: If a key or value is over its size limit. Nothing is written.
            RuntimeError: If the store was already published with other content.
        """
        timeout_s = _DEFAULT_TIMEOUT_S if timeout_s is None else timeout_s
        deadline = time.monotonic() + timeout_s
        values = {
            _PUBLISHERS: _encode_publishers(publishers),
            _STRATEGY: _encode_strategy(strategy),
        }
        if coordinator is not None:
            values[_COORDINATOR] = coordinator.encode()
        values[_READY] = _SCHEMA_VERSION
        for key, value in values.items():
            _check_sizes(_prefix(self.address), key, value)
        for key, value in values.items():
            if await self._compare_set(key, value, deadline) != value:
                raise self._error(f"store already initialized with a different {key!r}")

    async def read(
        self,
        *,
        timeout_s: float | None = None,
    ) -> tuple[frozenset[str], TorchStoreStrategy]:
        """Return the published publisher ranks and unpickled strategy, without waiting.

        Raises:
            RuntimeError: If this address has no complete publish under this schema.
            ValueError: If the published strategy inflates past its size limit.
        """
        timeout_s = _DEFAULT_TIMEOUT_S if timeout_s is None else timeout_s
        deadline = time.monotonic() + timeout_s
        await self._require_ready(deadline)
        rest = [_PUBLISHERS, _STRATEGY]
        # Written before ready, so these are missing only from a malformed store.
        if not await self._check(rest, deadline):
            raise self._error("marked ready without its publishers or strategy")
        publishers, strategy = await self._multi_get(rest, deadline)
        return _decode_publishers(publishers), _decode_strategy(strategy)

    async def read_strategy(
        self,
        *,
        timeout_s: float | None = None,
    ) -> TorchStoreStrategy:
        """Return the published strategy alone, unpickled, without waiting or reading the publisher ranks.

        Raises:
            RuntimeError: If this address has no ready marker under this schema, or no strategy.
            ValueError: If the published strategy inflates past its size limit.
        """
        timeout_s = _DEFAULT_TIMEOUT_S if timeout_s is None else timeout_s
        deadline = time.monotonic() + timeout_s
        await self._require_ready(deadline)
        # Written before ready, so it is missing only from a malformed store.
        if not await self._check([_STRATEGY], deadline):
            raise self._error("marked ready without its strategy")
        (strategy,) = await self._multi_get([_STRATEGY], deadline)
        return _decode_strategy(strategy)

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
        return RuntimeError(
            f"routing store {self.address.store_name!r} on TCPStore "
            f"{_endpoint(self.address)}: {problem}"
        )

    async def _call(
        self, what: str, fn: Callable[[PrefixStore], _T], deadline: float
    ) -> _T:
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
        call = _NativeCall(partial(fn, prefixed))
        del prefixed  # a kept error holds this frame, which must not keep the store
        return await self._lane.run(
            call,
            deadline,
            f"{what} on TCPStore {_endpoint(self.address)}",
        )

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
