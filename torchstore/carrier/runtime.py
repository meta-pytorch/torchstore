# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The carrier's runtime: one I/O thread and up to eight reply writers per process, one dispatcher per served object.

Callers on any thread or loop pickle their own calls and wait on their own loops, so a call's
deadline holds when the server or the network is stuck, and a stuck I/O thread adds at most 5 s.
"""

from __future__ import annotations

import asyncio
import atexit
import collections
import contextlib
import fcntl
import inspect
import json
import logging
import os
import queue
import secrets
import select
import socket
import struct
import sys
import termios
import threading
import time
import weakref
from collections.abc import Callable, Collection, Coroutine
from concurrent import futures
from functools import partial
from typing import Any, NoReturn

import torch

from torchstore.carrier import codec
from torchstore.carrier.codec import (
    CallTimeoutError,
    CarrierAddress,
    CarrierError,
    CarrierProtocolError,
    Hello,
    Kind,
    PeerLostError,
    Reject,
    RemoteError,
    StaleAddressError,
    Welcome,
)

__all__ = [
    "CarrierClient",
    "CarrierServer",
    "close_all",
    "close_scope",
    "VOLUME_CLOSED",
    "wait_future",
]

logger = logging.getLogger("torchstore.carrier")

# Read when used, so tests can shorten them.
_CALL_TIMEOUT_S = 300.0
_CONNECT_TIMEOUT_S = 30.0
_QUEUE_BOUND = 64
# Unsent reply bytes past which a FIFO server runs none of that connection's queued calls.
_UNSENT_BYTES_PER_CONNECTION = 256 << 20
_STATS_INTERVAL_S = 60.0
# How long a caller or a closing server waits on the I/O or a dispatcher thread.
_RELEASE_WAIT_S = 5.0
# How long exit waits for a close or loop thread already freeing a served object, as a free always ends.
_EXIT_FREE_WAIT_S = 300.0
# When exit logs a WARNING naming a thread it still waits for.
_EXIT_FREE_WARN_S = 30.0
# How often a reader stopped at the queue bound checks whether its client hung up.
_HANGUP_CHECK_S = 1.0
# Python's SOMAXCONN is 128, which a burst of first connects overflows. The kernel caps this.
_LISTEN_BACKLOG = 4096
_KEEPALIVE = (("TCP_KEEPIDLE", 60), ("TCP_KEEPINTVL", 15), ("TCP_KEEPCNT", 4))
# Poll events for a peer that closed or reset its end, set even with its data unread.
_HANGUP_EVENTS = getattr(select, "POLLRDHUP", 0) | select.POLLHUP | select.POLLERR
# Linger on with a zero timeout, so closing resets the connection.
_RESET_ON_CLOSE = struct.pack("ii", 1, 0)
# Linux's SO_TIMESTAMPNS, which Python's socket module does not name, on its common architectures:
# each receive then reports when its last bytes reached the kernel.
_SO_TIMESTAMPNS = (
    35 if sys.platform == "linux" and os.uname().machine in ("x86_64", "aarch64") else 0
)
_STAMP_SPACE = socket.CMSG_SPACE(16) if _SO_TIMESTAMPNS else 0
_SMALL_BYTES = 64 * 1024
# Reply chunks this large go to writer threads while another is in flight, since one thread's
# copy into socket buffers caps a server near 3 GiB/s. With 0 writers they stay on the I/O thread.
_OFFLOAD_BYTES = 1 << 20
_WRITER_THREADS = 8
# What a writer copies of one chunk before the chunks of other connections get a turn.
_WRITE_TURN_BYTES = 64 << 20
# Buffers a reader, or chunks a sender, takes before other callbacks get a turn: each costs the loop, however small.
_BUFFER_TURN = 1024
# How the PeerLostError of a call that a closing server drops begins.
VOLUME_CLOSED = "volume closed"
_LOGGED_STATS = (
    "calls",
    "errors",
    "timeouts",
    "cancels",
    "expired",
    "bytes_in",
    "bytes_out",
    "max_queue_depth",
)

# A call's progress on the I/O thread, and a request's on its server.
_PENDING, _SENDING, _SENT, _DONE = range(4)
_QUEUED, _RUNNING, _FINISHED = range(3)


class _EndOfStream(Exception):
    """The peer closed its end of the connection."""


class _NotStarted(Exception):
    """A call that must not start its handler, carrying its answer's body."""


class _LoopThread:
    """A daemon thread that runs its own event loop, published once the loop runs."""

    def __init__(self, name: str) -> None:
        self.stopped: futures.Future[None] = futures.Future()
        # Set by another exit hook that blocked this thread for good, so the carrier's neither stops nor joins it.
        self.parked = False
        started: futures.Future[asyncio.AbstractEventLoop] = futures.Future()
        self.thread = threading.Thread(
            target=self._run, args=(started,), name=name, daemon=True
        )
        self.thread.start()
        self.loop = started.result()

    def _run(self, started: futures.Future[Any]) -> None:
        try:
            loop = asyncio.new_event_loop()
        except BaseException as e:
            started.set_exception(e)
            return
        self.loop = loop
        with _loop_threads_lock:
            _loop_threads.add(self)
        loop.call_soon(started.set_result, loop)
        try:
            loop.run_forever()
        finally:
            # Closing frees what its callbacks hold, a served object say: exit waits for that as for
            # a close thread's free, and once exit has begun the main thread frees them instead.
            thread = threading.current_thread()
            with _loop_threads_lock:
                park = _exiting
                if not park:
                    _freeing.add(thread)
            try:
                # Before the close because cancelling a run posts its task's cancel to this loop.
                with _loop_threads_lock:
                    left = _runs.pop(loop, ())
                for run in left:
                    run.cancel()
                if park:
                    state = getattr(loop, "__dict__", {})
                    _parked.append(
                        [*state.get("_ready", ()), *state.get("_scheduled", ())]
                    )
                loop.close()
            finally:
                with _loop_threads_lock:
                    _freeing.discard(thread)
                    _loop_threads.discard(self)
                self.stopped.set_result(None)


# Running loop threads, stopped at exit: a daemon thread that returns from torch into a
# finalizing interpreter aborts the process.
_loop_threads: set[_LoopThread] = set()
_loop_threads_lock = threading.Lock()
# Started close threads, which exit lets finish before it stops the loops they use.
_close_threads: weakref.WeakSet[threading.Thread] = weakref.WeakSet()
# Chunks a writer let go of after the I/O loop closed, at exit: never freed on a writer.
_unfreed: list[memoryview] = []
# Set at exit under _loop_threads_lock. A call whose last check comes later never starts. A close or
# loop thread then leaves what it holds here for the main thread to free. Exit first waits for those
# threads already freeing.
_exiting = False
_parked: list[Any] = []
_freeing: set[threading.Thread] = set()
# Loops told to stop, under _loop_threads_lock: _post_call posts to them no more.
_stopped: weakref.WeakSet[asyncio.AbstractEventLoop] = weakref.WeakSet()
# Under _loop_threads_lock, the runs _run_on waits for by loop, which a loop thread cancels once its loop stopped.
_runs: dict[asyncio.AbstractEventLoop, set[futures.Future[Any]]] = {}


@atexit.register
def _stop_loops_at_exit() -> None:
    """Close every client, give started closes 5 s to finish, then stop each carrier loop not parked within 5 s more.

    Clients close while the I/O loop runs, so their calls fail before it stops. It stops once that close ran, or once
    those first 5 s ran out, so their connections' tasks end first unless it was busy that long.
    """
    global _exiting
    with _loop_threads_lock:
        _exiting = True
        freeing = list(_freeing)
    with _lock:
        runtime, peers = _detach_clients(lambda address: True)
    clients_closed = None
    if runtime is not None:
        clients_closed = _post_call(runtime.loop, partial(_close_peers, peers))
    # Cut off by finalization, a free would abort the process.
    for thread in freeing:
        thread.join(_EXIT_FREE_WARN_S)
        if thread.is_alive():
            logger.warning(
                "Exit has waited %g s for %s to free a served object",
                _EXIT_FREE_WARN_S,
                thread.name,
            )
            thread.join(max(_EXIT_FREE_WAIT_S - _EXIT_FREE_WARN_S, 0.0))
            if thread.is_alive():
                logger.warning(
                    "Exit stopped waiting for %s after %g s",
                    thread.name,
                    _EXIT_FREE_WAIT_S,
                )
    deadline = time.monotonic() + _RELEASE_WAIT_S
    with _loop_threads_lock:
        closing = list(_close_threads)
    for thread in closing:
        thread.join(max(deadline - time.monotonic(), 0.0))
    if clients_closed is not None:
        futures.wait([clients_closed], max(deadline - time.monotonic(), 0.0))
    # A wait of their own: a close still running may have used up the first.
    deadline = time.monotonic() + _RELEASE_WAIT_S
    with _loop_threads_lock:
        threads = [
            loop_thread for loop_thread in _loop_threads if not loop_thread.parked
        ]
    for loop_thread in threads:
        _stop_loop(loop_thread.loop)
    for loop_thread in threads:
        loop_thread.thread.join(max(deadline - time.monotonic(), 0.0))
    # A shutdown left to the I/O loop that stopped before it ended never ends so its closes return now.
    for server in list(_servers):
        shutdown = server._shutdown
        if shutdown is not None and not shutdown.done():
            _resolve(server._closed)


class _Runtime:
    """The process's I/O thread and reply writers, and the connections, servers and sockets they run."""

    def __init__(self) -> None:
        self.loop = _LoopThread("torchstore-carrier").loop
        self.peers: dict[CarrierAddress, _Peer] = {}
        self.servers: set[CarrierServer] = set()
        self.sockets: weakref.WeakSet[socket.socket] = weakref.WeakSet()
        self.writers = _Writers()
        # Name lookups still running, by host and port. Used on the I/O thread only.
        self.lookups: dict[tuple[str, int], asyncio.Future[list[Any]]] = {}


# Also held while a carrier socket is made and registered, or closed, so a fork, which
# takes it too, never inherits one beyond its hook's reach.
_lock = threading.Lock()
_runtime: _Runtime | None = None
# Every server made here, so a fork renews the lock of each, one partway through closing included.
_servers: weakref.WeakSet[CarrierServer] = weakref.WeakSet()
# Every client handle, so close_scope and close_all also close one inherited across fork.
_handles: weakref.WeakSet[CarrierClient] = weakref.WeakSet()
# Per thread, as forks from two threads overlap: only a fork that took the lock releases it.
_fork_state = threading.local()


def _current_runtime() -> _Runtime:
    """Return the runtime, starting it on first use. Call with ``_lock`` held.

    Raises:
        PeerLostError: Once the exit hook has begun, as exit closes no handle or server made later.
    """
    global _runtime
    if _exiting:
        raise PeerLostError(
            "the carrier takes no new client or server once exit has begun"
        )
    if _runtime is None:
        _runtime = _Runtime()
    return _runtime


def _before_fork() -> None:
    # Bounded, so a fork never waits long on a thread that holds the lock.
    _fork_state.held = _lock.acquire(timeout=1.0)
    if not _fork_state.held:
        logger.warning(
            "Forking without the carrier lock, which another thread held for 1 s, so the "
            "child may keep a carrier socket made meanwhile"
        )


def _after_fork_in_parent() -> None:
    if getattr(_fork_state, "held", False):
        _fork_state.held = False
        _lock.release()


def _after_fork_in_child() -> None:
    global _exiting, _lock, _loop_threads_lock, _runtime
    # A child forked during or after the parent's exit hook is a live process.
    _exiting = False
    _lock = threading.Lock()
    # Their threads did not survive the fork, and their wakeup pipes are the parent's.
    _loop_threads_lock = threading.Lock()
    _loop_threads.clear()
    _close_threads.clear()
    _freeing.clear()
    _runs.clear()
    runtime, _runtime = _runtime, None
    if runtime is not None:
        # Its loop stays open: closing it would unregister the parent's sockets from the epoll instance both share.
        for sock in list(runtime.sockets):
            try:
                # Never shutdown(), which would end the parent's connections as well.
                sock.close()
            except OSError:
                pass
    for server in list(_servers):
        server._lock = threading.Lock()


os.register_at_fork(
    before=_before_fork,
    after_in_parent=_after_fork_in_parent,
    after_in_child=_after_fork_in_child,
)


def _post(loop: asyncio.AbstractEventLoop, fn: Callable[..., Any], *args: Any) -> bool:
    """Schedule ``fn`` on ``loop`` from any thread. False if the loop has closed."""
    try:
        loop.call_soon_threadsafe(fn, *args)
    except RuntimeError:
        return False
    return True


def _post_unless_stopped(
    loop: asyncio.AbstractEventLoop, fn: Callable[..., Any], *args: Any
) -> bool:
    """Schedule ``fn`` on ``loop`` from any thread. False if the loop is stopping or has closed."""
    # Under the lock _stop_loop posts its stop with: what this posts then runs before that stop.
    with _loop_threads_lock:
        return loop not in _stopped and _post(loop, fn, *args)


def _set_done(waiter: asyncio.Future[None]) -> None:
    if not waiter.done():
        waiter.set_result(None)


async def wait_future(future: futures.Future[Any], timeout: float | None) -> bool:
    """Wait on this loop for another thread's ``future`` up to ``timeout`` and return True if it is done.

    Cancelling one waiter never cancels ``future``, so any number of waiters can share it.
    """
    if future.done():
        return True
    loop = asyncio.get_running_loop()
    waiter = loop.create_future()
    # Delivery to a loop that has closed is skipped.
    future.add_done_callback(lambda _: _post(loop, _set_done, waiter))
    done, _ = await asyncio.wait({waiter}, timeout=timeout)
    return bool(done)


def _post_call(
    loop: asyncio.AbstractEventLoop, fn: Callable[..., Any]
) -> futures.Future[None]:
    """Schedule ``fn`` on ``loop``'s thread. The future is done once ``fn`` ran, or at once if the loop is stopping or closed."""
    done: futures.Future[None] = futures.Future()

    def run() -> None:
        try:
            fn()
        finally:
            done.set_result(None)

    if not _post_unless_stopped(loop, run):
        done.set_result(None)
    return done


def _stop_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Stop ``loop`` from any thread, once each call ``_post_call`` posted to it before now has run."""
    with _loop_threads_lock:
        _stopped.add(loop)
        _post(loop, loop.stop)


async def _call_on(loop: asyncio.AbstractEventLoop, fn: Callable[..., Any]) -> None:
    """Run ``fn`` on ``loop``'s thread and wait for it, up to ``_RELEASE_WAIT_S``."""
    await wait_future(_post_call(loop, fn), _RELEASE_WAIT_S)


async def _run_on(
    loop: asyncio.AbstractEventLoop,
    coroutine: Coroutine[Any, Any, Any],
    timeout: float | None,
) -> None:
    """Run ``coroutine`` on ``loop`` and wait for it up to ``timeout`` or until the loop stops first.

    One posted once the loop is told to stop or has closed is closed unstarted as ``_post_call`` drops a call then.
    """
    future = None
    # Like _post_unless_stopped, this checks under the lock _stop_loop marks a loop stopped with.
    with _loop_threads_lock, contextlib.suppress(RuntimeError):
        if loop not in _stopped:
            future = asyncio.run_coroutine_threadsafe(coroutine, loop)
            _runs.setdefault(loop, set()).add(future)
    if future is None:
        coroutine.close()
        return
    try:
        await wait_future(future, timeout)
    finally:
        with _loop_threads_lock:
            runs = _runs.get(loop)
            if runs is not None:
                runs.discard(future)
                if not runs:
                    del _runs[loop]


def _getaddrinfo(host: str, port: int) -> list[Any]:
    return socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)


def _lookup(runtime: _Runtime, host: str, port: int) -> asyncio.Future[list[Any]]:
    """Resolve on a daemon thread, as the loop's executor would be joined at exit. On the I/O thread.

    Connects share a lookup still running, so one that hangs holds one thread, not one per attempt.
    """
    key = (host, port)
    running = runtime.lookups.get(key)
    if running is not None:
        return running
    loop = runtime.loop
    future = runtime.lookups[key] = loop.create_future()
    # Retrieved here, as every connect waiting on a lookup may have given up by its end.
    future.add_done_callback(lambda done: done.cancelled() or done.exception())

    def settle(infos: list[Any], error: Exception | None) -> None:
        if runtime.lookups.get(key) is future:
            del runtime.lookups[key]
        if error is None:
            future.set_result(infos)
        else:
            future.set_exception(error)

    def run() -> None:
        infos: list[Any] = []
        error: Exception | None = None
        try:
            infos = _getaddrinfo(host, port)
        except Exception as e:
            error = e
        _post(loop, settle, infos, error)

    try:
        threading.Thread(
            target=run, name=f"torchstore-carrier-resolve {host}", daemon=True
        ).start()
    except BaseException:
        del runtime.lookups[key]
        raise
    return future


def _numeric(sockaddr: tuple[Any, ...]) -> tuple[Any, ...]:
    # asyncio resolves again, in its executor, an IPv6 address that keeps its zone.
    return (sockaddr[0].split("%", 1)[0], *sockaddr[1:])


def _listen(port: int) -> socket.socket:
    """Listen on every interface, dual-stack IPv6 first and then IPv4, as TCPStore does."""
    error: OSError | None = None
    for family, host in ((socket.AF_INET6, "::"), (socket.AF_INET, "0.0.0.0")):
        try:
            sock = socket.socket(family, socket.SOCK_STREAM)
        except OSError as e:
            error = e
            continue
        try:
            if family == socket.AF_INET6:
                sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
            sock.listen(max(socket.SOMAXCONN, _LISTEN_BACKLOG))
            sock.setblocking(False)
            return sock
        except OSError as e:
            sock.close()
            error = e
        except BaseException:
            sock.close()
            raise
    assert error is not None
    raise OSError(
        error.errno, f"cannot listen for the carrier on port {port}: {error.strerror}"
    ) from error


def _configure(sock: socket.socket) -> None:
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    for name, value in _KEEPALIVE:
        if hasattr(socket, name):
            sock.setsockopt(socket.IPPROTO_TCP, getattr(socket, name), value)
    # TCP_USER_TIMEOUT stays unset: it also aborts a healthy connection whose reader
    # is merely busy (a zero window), and call deadlines already bound partitions.


def _close_socket(loop: asyncio.AbstractEventLoop, sock: socket.socket) -> None:
    """Close a socket on its loop's thread, unregistering it first so no stale callback runs."""
    fd = sock.fileno()
    if fd >= 0:
        loop.remove_reader(fd)
        loop.remove_writer(fd)
    with _lock:
        sock.close()


def _hung_up(sock: socket.socket) -> bool:
    """Whether the peer closed or reset the connection, even with data it sent still unread."""
    poller = select.poll()
    poller.register(sock, _HANGUP_EVENTS)
    return bool(poller.poll(0))


def _unread_bytes(sock: socket.socket) -> int:
    """Bytes that reached the kernel for ``sock`` and wait to be read, or 0 if it cannot tell."""
    try:
        data = fcntl.ioctl(sock.fileno(), termios.FIONREAD, bytes(4))
    except (OSError, ValueError):
        return 0
    return struct.unpack("i", data)[0]


def _drop_traceback(error: BaseException) -> None:
    """Drop a caught error's traceback: a future that still holds the error would keep its frames alive."""
    error.__traceback__ = None


def _peer_name() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


def _frame_size(chunks: list[bytes | memoryview]) -> int:
    return sum(len(chunk) for chunk in chunks)


async def _send_chunks(
    loop: asyncio.AbstractEventLoop,
    sock: socket.socket,
    chunks: list[bytes | memoryview],
    gate: asyncio.Event | None = None,
    send_large: Callable[[bytes | memoryview], Coroutine[Any, Any, None]] | None = None,
) -> int:
    """Send a frame's chunks in order, the header and a small body in one write.

    ``send_large``, if given, sends each chunk of ``_OFFLOAD_BYTES`` or more in its place.
    """
    if len(chunks) > 1 and len(chunks[1]) <= _SMALL_BYTES:
        chunks = [bytes(chunks[0]) + bytes(chunks[1]), *chunks[2:]]
    sent = 0
    for index, chunk in enumerate(chunks):
        if index % _BUFFER_TURN == _BUFFER_TURN - 1:
            await asyncio.sleep(0)
        if not len(chunk):
            continue
        if gate is not None:
            await gate.wait()
        if send_large is not None and len(chunk) >= _OFFLOAD_BYTES:
            await send_large(chunk)
        else:
            await _sendall(loop, sock, chunk)
        sent += len(chunk)
    return sent


async def _sendall(
    loop: asyncio.AbstractEventLoop, sock: socket.socket, data: bytes | memoryview
) -> None:
    """Send all of ``data``, waiting for room on ``loop``.

    Unlike ``loop.sock_sendall``, whose failed send keeps its error and ``data`` in a cycle through a
    future its frame holds, an error raises here.
    """
    view = memoryview(data).cast("B")
    sent = 0
    while True:
        try:
            sent += sock.send(view[sent:])
        except BlockingIOError:
            pass
        if sent == len(view):
            return
        await _writable(loop, sock)


async def _writable(loop: asyncio.AbstractEventLoop, sock: socket.socket) -> None:
    """Wait until a full socket has room to send again."""
    await _ready(loop, sock, loop.add_writer, loop.remove_writer)


async def _readable(loop: asyncio.AbstractEventLoop, sock: socket.socket) -> None:
    """Wait until a drained socket has bytes, or its end, to read again."""
    await _ready(loop, sock, loop.add_reader, loop.remove_reader)


async def _ready(
    loop: asyncio.AbstractEventLoop,
    sock: socket.socket,
    add: Callable[..., Any],
    remove: Callable[[int], Any],
) -> None:
    ready = loop.create_future()
    add(sock.fileno(), _set_done, ready)
    try:
        await ready
    finally:
        # A socket closed meanwhile has no number left to unregister, and its old one may be another socket's.
        if sock.fileno() >= 0:
            remove(sock.fileno())


def _stamp_arrivals(sock: socket.socket) -> bool:
    """Have each receive on ``sock`` report when its bytes reached the kernel. False if it cannot."""
    if not _SO_TIMESTAMPNS:
        return False
    try:
        sock.setsockopt(socket.SOL_SOCKET, _SO_TIMESTAMPNS, 1)
    except OSError:
        return False
    return True


def _arrival(ancdata: list[tuple[int, int, bytes]]) -> float | None:
    """When a receive's last bytes reached the kernel, on the monotonic clock, or None if unstamped."""
    for level, kind, data in ancdata:
        if level == socket.SOL_SOCKET and kind == _SO_TIMESTAMPNS and len(data) >= 16:
            seconds, nanoseconds = struct.unpack_from("qq", data)
            if seconds:
                age = time.time() - seconds - nanoseconds / 1e9
                return time.monotonic() - max(age, 0.0)
    return None


class _SendJob:
    """One turn of a large chunk, copied into its socket on a writer thread without blocking.

    The socket is never closed during a copy: a close that comes then is left to the writer.
    """

    __slots__ = (
        "cancelled",
        "copying",
        "done",
        "lock",
        "loop",
        "pending_close",
        "sock",
        "view",
    )

    def __init__(
        self, loop: asyncio.AbstractEventLoop, sock: socket.socket, view: memoryview
    ) -> None:
        self.loop, self.sock = loop, sock
        self.view: memoryview | None = view
        self.lock = threading.Lock()
        self.cancelled = self.copying = False
        # A close that came during a copy, which the writer posts to the loop after it.
        self.pending_close: Callable[[], None] | None = None
        # The bytes sent, and whether the socket filled up. Set on the loop.
        self.done: asyncio.Future[tuple[int, bool]] = loop.create_future()

    def run(self) -> None:
        """Copy until the socket is full, the turn ends or the job is cancelled. On a writer thread."""
        view, self.view = self.view, None
        assert view is not None
        sent, full = 0, False
        error: Exception | None = None
        try:
            end = min(len(view), _WRITE_TURN_BYTES)
            while sent < end and not full and error is None:
                with self.lock:
                    if self.cancelled:
                        break
                    self.copying = True
                try:
                    sent += self.sock.send(view[sent:end])
                except BlockingIOError:
                    full = True
                except Exception as e:
                    error = e
                finally:
                    with self.lock:
                        self.copying = False
                        close, self.pending_close = self.pending_close, None
                if close is not None:
                    _post(self.loop, close)
        finally:
            # Freed on the I/O thread, which exit waits for: torch's free retakes the GIL, which a
            # daemon thread cannot do once the interpreter finalizes.
            held = [view]
            del view
            if not _post(self.loop, self._settle, sent, full, error, held):
                _unfreed.extend(held)

    def defer_close(self, close: Callable[[], None]) -> bool:
        """Cancel, leaving ``close`` to the writer if a copy runs now. True if one does."""
        with self.lock:
            self.cancelled = True
            if self.copying:
                self.pending_close = close
            return self.copying

    def _settle(
        self, sent: int, full: bool, error: Exception | None, held: list[memoryview]
    ) -> None:
        held.clear()
        if self.done.done():
            return
        if error is None:
            self.done.set_result((sent, full))
        else:
            self.done.set_exception(error)


class _Writers:
    """The runtime's reply writer threads, started on first use, and the jobs they take in turn."""

    def __init__(self) -> None:
        self.jobs: queue.SimpleQueue[_SendJob | None] = queue.SimpleQueue()
        # Each writer thread, and a future set once it returns.
        self.threads: dict[threading.Thread, futures.Future[None]] = {}
        # Large chunks being sent, counted on the I/O thread, which sends a lone one itself.
        self.sending = 0

    def start(self) -> bool:
        """Start the writers unless some run. False if none can, so sends stay on the I/O thread."""
        if self.threads:
            return True
        for index in range(_WRITER_THREADS):
            stopped: futures.Future[None] = futures.Future()
            thread = threading.Thread(
                target=self._run,
                args=(self.jobs, stopped),
                name=f"torchstore-carrier-writer {index}",
                daemon=True,
            )
            try:
                thread.start()
            except RuntimeError:
                break
            self.threads[thread] = stopped
        return bool(self.threads)

    def stop(self) -> dict[threading.Thread, futures.Future[None]]:
        """Stop each writer once the jobs queued before now have run. Return them and their futures.

        Call on the I/O thread. Later replies start new writers.
        """
        threads, self.threads = self.threads, {}
        jobs, self.jobs = self.jobs, queue.SimpleQueue()
        for _ in threads:
            jobs.put(None)
        return threads

    @staticmethod
    def _run(
        jobs: queue.SimpleQueue[_SendJob | None], stopped: futures.Future[None]
    ) -> None:
        try:
            while True:
                job = jobs.get()
                if job is None:
                    return
                job.run()
                # Dropped now, or an idle writer would hold its last job's socket.
                del job
        finally:
            stopped.set_result(None)


class _Reader:
    """Exact reads from a socket: small ones through a buffer, large ones straight in.

    A stamped reader also knows when the newest bytes of the receive that took a read's first byte reached the kernel.
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        sock: socket.socket,
        gate: asyncio.Event | None = None,
        stamped: bool = False,
    ) -> None:
        self._loop, self._sock, self._gate = loop, sock, gate
        self._buffer = memoryview(bytearray(_SMALL_BYTES))
        self._start = self._end = 0
        self._stamped = stamped
        # When the last receive's newest bytes reached the kernel, on the monotonic clock. Only that receive's bytes
        # ever wait in the buffer.
        self._stamp: float | None = None
        # For the stamped read under way: the stamp of the receive that took its first byte, and whether the next
        # receive takes that byte.
        self._first: float | None = None
        self._taking_first = False
        # Bytes received since this reader last took a turn of its own.
        self._turn = 0
        # Bytes taken from the socket in all, and whether it is giving the loop a turn mid-read.
        self.received = 0
        self.turning = False

    def consumed(self) -> int:
        """Bytes read out of this reader so far."""
        return self.received - (self._end - self._start)

    def unread(self) -> int:
        """Bytes that have arrived but are not read yet, in its buffer or the kernel's."""
        return self._end - self._start + _unread_bytes(self._sock)

    async def take_turn(self) -> None:
        """Let other callbacks run mid-read. A held request's release waits for the bytes that came with it."""
        self.turning = True
        try:
            await asyncio.sleep(0)
        finally:
            self.turning = False

    async def read_stamped(self, size: int) -> tuple[bytes, float]:
        """Read ``size`` bytes and when the receive that took the first got its newest bytes, or now if unstamped."""
        self._taking_first = self._end == self._start
        self._first = self._stamp
        data = await self.read(size)
        now = time.monotonic()
        return data, now if self._first is None else min(self._first, now)

    async def read(self, size: int) -> bytes:
        if self._end - self._start >= size:
            self._start += size
            return bytes(self._buffer[self._start - size : self._start])
        data = bytearray(size)
        await self.read_into(memoryview(data))
        return bytes(data)

    async def read_tensor(self, size: int) -> torch.Tensor:
        tensor = torch.empty(size, dtype=torch.uint8)
        await self.read_into(codec.byte_view(tensor))
        return tensor

    async def read_body(self, size: int) -> bytes | memoryview:
        """Read a body. A large one lands in an unzeroed tensor and stays a view of it."""
        if size <= _SMALL_BYTES:
            return await self.read(size)
        return codec.byte_view(await self.read_tensor(size))

    async def read_into(self, target: memoryview) -> None:
        size = len(target)
        filled = min(size, self._end - self._start)
        target[:filled] = self._buffer[self._start : self._start + filled]
        self._start += filled
        while filled < size:
            # A sender that keeps up never makes a receive wait, so other callbacks get turns.
            if self._turn >= _WRITE_TURN_BYTES:
                self._turn = 0
                await self.take_turn()
            if self._gate is not None:
                await self._gate.wait()
            if size - filled >= len(self._buffer):
                count = taken = await self._recv_into(target[filled:])
            else:
                count = await self._recv_into(self._buffer)
                taken = min(count, size - filled)
                target[filled : filled + taken] = self._buffer[:taken]
                self._start, self._end = taken, count
            if not count:
                raise _EndOfStream()
            filled += taken

    async def _recv_into(self, view: memoryview) -> int:
        while True:
            try:
                if self._stamped:
                    count, ancdata, _, _ = self._sock.recvmsg_into([view], _STAMP_SPACE)
                    self._stamp = _arrival(ancdata)
                else:
                    count = self._sock.recv_into(view)
            except (BlockingIOError, InterruptedError):
                await _readable(self._loop, self._sock)
                continue
            if self._taking_first:
                self._taking_first = False
                self._first = self._stamp
            self._turn += count
            self.received += count
            return count


async def _read_frame(
    reader: _Reader,
) -> tuple[codec.FrameHeader, float, bytes | memoryview, list[torch.Tensor], int]:
    """Read one frame after the handshake: header, stamp of the receive with its first byte, body, buffers and size."""
    data, arrived = await reader.read_stamped(codec.HEADER_SIZE)
    header = codec.decode_header(data, handshake=False)
    lengths = codec.decode_lengths(
        header, await reader.read(codec.LENGTH_SIZE * header.buffer_count)
    )
    body = await reader.read_body(header.body_len)
    buffers = []
    for index, length in enumerate(lengths):
        # Small buffers come from the reader's own buffer, which takes no turn of its own.
        if index % _BUFFER_TURN == _BUFFER_TURN - 1:
            await reader.take_turn()
        buffers.append(await reader.read_tensor(length))
    size = codec.HEADER_SIZE + codec.LENGTH_SIZE * len(lengths)
    return header, arrived, body, buffers, size + len(body) + sum(lengths)


def _copy_mutable(buffers: list[memoryview]) -> list[memoryview]:
    """Copy writable reply buffers, so a later call cannot change a reply still being sent.

    The codec's own copies, of non-contiguous or CUDA tensors for example, are private already.
    """
    return [
        (
            buffer
            if buffer.readonly
            or not buffer.nbytes
            or getattr(buffer.obj, "_private", False)
            else codec.byte_view(torch.frombuffer(buffer, dtype=torch.uint8).clone())
        )
        for buffer in buffers
    ]


class _Call:
    """A caller's request: its frame's parts, its deadline, and its outcome."""

    __slots__ = (
        "abandoned",
        "body",
        "buffers",
        "conn",
        "deadline",
        "lock",
        "outcome",
        "released",
        "request_id",
        "state",
        "timeout_message",
    )

    def __init__(
        self,
        body: bytes,
        buffers: list[memoryview],
        deadline: float,
        timeout_message: str,
    ) -> None:
        self.body: bytes | None = body
        self.buffers: list[memoryview] | None = buffers
        self.deadline = deadline
        # This call's own timeouts raise with this text on either thread. A server's queue expiry and the connect
        # deadline raise texts of their own.
        self.timeout_message = timeout_message
        self.lock = threading.Lock()
        self.state = _PENDING
        self.abandoned = False
        self.conn: _ClientConnection | None = None
        self.request_id = 0
        # Set once the I/O thread stops reading an abandoned call's memory.
        self.released: futures.Future[None] | None = None
        self.outcome: futures.Future[
            tuple[Kind, bytes | memoryview, list[torch.Tensor]]
        ] = futures.Future()

    def abandon(self) -> bool:
        """Give up on the call. True if its frame is being sent, so the caller must wait."""
        with self.lock:
            self.abandoned = True
            if self.state != _SENDING:
                # Never read once abandoned, so the caller's memory is let go at once.
                self.body = self.buffers = None
                return False
            self.released = futures.Future()
            return True

    def start_send(self) -> bool:
        with self.lock:
            if self.abandoned:
                self.state = _DONE
                return False
            self.state = _SENDING
            return True

    def end_send(self) -> bool:
        """Mark the frame sent and drop its parts. True if the caller gave up meanwhile."""
        self.body = self.buffers = None
        with self.lock:
            self.state = _SENT
            released, abandoned = self.released, self.abandoned
        if released is not None:
            released.set_result(None)
        return abandoned

    def settle(
        self,
        outcome: tuple[Kind, bytes | memoryview, list[torch.Tensor]] | None = None,
        error: Exception | None = None,
    ) -> None:
        """Deliver a reply or an error, unless the caller gave up."""
        self.body = self.buffers = None
        with self.lock:
            self.state = _DONE
            released, abandoned = self.released, self.abandoned
        if released is not None and not released.done():
            released.set_result(None)
        if abandoned or self.outcome.done():
            return
        if error is None:
            self.outcome.set_result(outcome)  # type: ignore[arg-type]
        else:
            self.outcome.set_exception(error)


class _Peer:
    """This process's one connection to an address, run on the I/O thread."""

    def __init__(self, runtime: _Runtime, address: CarrierAddress) -> None:
        self.runtime = runtime
        self.address = address
        self.where = f"carrier {address.service} at {address.host}:{address.port}"
        # Guarded by _lock, like closed's writes.
        self.handles: weakref.WeakSet[CarrierClient] = weakref.WeakSet()
        self.closed = False
        self.conn: _ClientConnection | None = None
        self.connecting: asyncio.Task[None] | None = None
        self.waiting: list[_Call] = []

    def submit(self, call: _Call) -> None:
        if call.abandoned:
            return
        if self.closed:
            call.settle(error=self._closed_error())
        elif self.conn is not None:
            self.conn.send_call(call)
        else:
            self.waiting.append(call)
            if self.connecting is None:
                self.connecting = self.runtime.loop.create_task(self._connect())

    def abandoned(self, call: _Call) -> None:
        if call.conn is not None:
            call.conn.abandoned(call)

    def close(self) -> None:
        if self.connecting is not None:
            self.connecting.cancel()
        self._fail_waiting(self._closed_error)
        if self.conn is not None:
            self.conn.close(self._closed_error)

    def _closed_error(self) -> PeerLostError:
        return PeerLostError(f"{self.where}: its last client handle closed")

    def _fail_waiting(self, error: Callable[[], Exception]) -> None:
        waiting, self.waiting = self.waiting, []
        for call in waiting:
            call.settle(error=error())

    async def _connect(self) -> None:
        loop = self.runtime.loop
        timeout = _CONNECT_TIMEOUT_S
        task = asyncio.current_task()
        expired: list[bool] = []

        def expire() -> None:
            expired.append(True)
            task.cancel()  # type: ignore[union-attr]

        timer = loop.call_at(loop.time() + timeout, expire)
        try:
            sock, reader, window = await self._open()
        except asyncio.CancelledError:
            error: Callable[[], Exception] = self._closed_error
            if expired:
                message = (
                    f"connecting to {self.where} did not finish within its"
                    f" {timeout:g} s connect deadline"
                )
                error = partial(CallTimeoutError, message)
            self._fail_waiting(error)
            return
        except CarrierError as e:
            self._fail_waiting(partial(type(e), *e.args))
            return
        except Exception as e:
            message = f"cannot connect to {self.where}: {e}"
            self._fail_waiting(partial(PeerLostError, message))
            return
        finally:
            timer.cancel()
            self.connecting = None
        if self.closed:
            # Marked closed by a close still on its way to this thread, so the waiting calls must not go out.
            _close_socket(loop, sock)
            self._fail_waiting(self._closed_error)
            return
        waiting, self.waiting = self.waiting, []
        logger.info(
            "carrier connect service=%s to=%s:%d from=%s",
            self.address.service,
            self.address.host,
            self.address.port,
            _peer_name(),
        )
        self.conn = _ClientConnection(self, sock, reader, window)
        for call in waiting:
            self.conn.send_call(call)

    async def _open(self) -> tuple[socket.socket, _Reader, int]:
        loop, address = self.runtime.loop, self.address
        try:
            # Shielded, as a connect deadline must not cancel a lookup other connects share.
            infos = await asyncio.shield(
                _lookup(self.runtime, address.host, address.port)
            )
        except Exception as e:
            raise PeerLostError(f"cannot resolve {self.where}: {e}") from e
        error: OSError | None = None
        for family, _, proto, _, sockaddr in infos:
            try:
                with _lock:
                    sock = socket.socket(family, socket.SOCK_STREAM, proto)
                    self.runtime.sockets.add(sock)
            except OSError as e:
                error = e
                continue
            try:
                sock.setblocking(False)
                await loop.sock_connect(sock, _numeric(sockaddr))
                _configure(sock)
                # Every close resets, a killed process's too: a FIN would wait behind
                # calls a server stopped at its queue bound never reads.
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, _RESET_ON_CLOSE)
                reader = _Reader(loop, sock)
                window = await self._handshake(sock, reader)
            except CarrierError:
                _close_socket(loop, sock)
                raise
            except OSError as e:
                _close_socket(loop, sock)
                error = e
                continue
            except BaseException:
                _close_socket(loop, sock)
                raise
            return sock, reader, window
        raise PeerLostError(f"cannot connect to {self.where}: {error}")

    async def _handshake(self, sock: socket.socket, reader: _Reader) -> int:
        """Say hello and check the answer. Return the server's window."""
        address = self.address
        hello = Hello(
            scope=address.scope,
            service=address.service,
            incarnation=address.incarnation,
            peer=_peer_name(),
        )
        body = hello.to_json().encode()
        loop = self.runtime.loop
        await _send_chunks(loop, sock, codec.encode_frame(Kind.HELLO, body=body))
        try:
            data = await reader.read(codec.HEADER_SIZE)
            header = codec.decode_header(data, handshake=True)
            body = await reader.read(header.body_len)
        except _EndOfStream:
            raise PeerLostError(
                f"{self.where} closed the connection during the handshake"
            ) from None
        except CarrierProtocolError as e:
            raise _not_a_carrier(self.where, e) from e
        # Read before the rest: a server of another version may answer in another shape.
        fields = _json_object(body)
        version = fields.get("version")
        if header.kind in (Kind.REJECT, Kind.WELCOME) and type(version) is int:
            if version != codec.PROTOCOL_VERSION:
                reason = fields.get("reason") if header.kind == Kind.REJECT else None
                raise CarrierProtocolError(
                    f"{self.where} speaks carrier protocol version {version},"
                    f" not {codec.PROTOCOL_VERSION}"
                    + (f": {reason}" if type(reason) is str else "")
                )
        try:
            reject = Reject.from_json(body) if header.kind == Kind.REJECT else None
            welcome = Welcome.from_json(body) if header.kind == Kind.WELCOME else None
        except CarrierProtocolError as e:
            raise _not_a_carrier(self.where, e) from e
        if reject is not None:
            raise StaleAddressError(
                f"{self.where} refused the connection: {reject.reason}"
            )
        if welcome is None:
            raise CarrierProtocolError(
                f"{self.where} answered a hello with a {header.kind.name} frame"
            )
        expected = (address.scope, address.service, address.incarnation)
        if (welcome.scope, welcome.service, welcome.incarnation) != expected:
            raise StaleAddressError(
                f"{self.where} serves a different scope, service or incarnation"
            )
        return max(welcome.window, 0)


def _json_object(body: bytes) -> dict[str, Any]:
    """The JSON object a handshake body holds, or an empty dict for any other body."""
    fields: Any = None
    try:
        text = body.decode()
        codec.check_record_brackets(text)
        fields = json.loads(text)
    except ValueError:  # a UnicodeDecodeError or JSONDecodeError too
        pass
    return fields if isinstance(fields, dict) else {}


def _not_a_carrier(where: str, error: CarrierProtocolError) -> StaleAddressError:
    # Not a carrier's answer: some other listener holds the address's port now.
    return StaleAddressError(
        f"{where} did not answer the hello as a carrier server: {error}"
    )


class _ClientConnection:
    """A connection's pending calls and outbox, and its reader and writer tasks."""

    def __init__(
        self, peer: _Peer, sock: socket.socket, reader: _Reader, window: int
    ) -> None:
        self.peer, self.sock, self.reader = peer, sock, reader
        self.loop = peer.runtime.loop
        self.pending: dict[int, _Call] = {}
        # Calls to send, and ids of sent calls to cancel.
        self.outbox: collections.deque[_Call | int] = collections.deque()
        self.ready = asyncio.Event()
        # Sent requests not yet answered, held to the server's window so it reads every frame.
        self.window, self.unanswered = window, 0
        self.last_id = 0
        self.sending: _Call | None = None
        self.closed = False
        self.tasks: tuple[asyncio.Task[None], ...] = (
            self.loop.create_task(self._read()),
            self.loop.create_task(self._write()),
        )

    def send_call(self, call: _Call) -> None:
        call.conn = self
        self.outbox.append(call)
        self.ready.set()

    def abandoned(self, call: _Call) -> None:
        if self.closed:
            return
        if call is self.sending:
            # The frame references the caller's memory, so the rest is never sent.
            self.close(
                partial(
                    PeerLostError,
                    f"{self.peer.where}: the connection closed when a call was"
                    " abandoned while it was being sent",
                )
            )
        elif self.pending.get(call.request_id) is call:
            del self.pending[call.request_id]
            self.outbox.appendleft(call.request_id)
            self.ready.set()

    def close(self, error: Callable[[], Exception]) -> None:
        if self.closed:
            return
        self.closed = True
        self.sending = None
        if self.peer.conn is self:
            self.peer.conn = None
        _close_socket(self.loop, self.sock)
        current = asyncio.current_task()
        for task in self.tasks:
            if task is not current:
                task.cancel()
        # A cancelled task keeps a traceback whose frames hold this connection.
        self.tasks = ()
        pending, self.pending = self.pending, {}
        for call in pending.values():
            call.settle(error=error())
        unsent = [item for item in self.outbox if isinstance(item, _Call)]
        self.outbox.clear()
        for call in unsent:
            # Never sent, so it may go out on a new connection.
            call.conn = None
            self.peer.submit(call)

    async def _next(self) -> _Call | int:
        """Return the next call or cancel to send, holding calls while the window is full."""
        while True:
            if self.outbox and (
                isinstance(self.outbox[0], int)
                or not self.window
                or self.unanswered < self.window
            ):
                return self.outbox.popleft()
            self.ready.clear()
            await self.ready.wait()

    async def _write(self) -> None:
        try:
            while True:
                item = await self._next()
                if isinstance(item, int):
                    frame = codec.encode_frame(Kind.CANCEL, item)
                    await _send_chunks(self.loop, self.sock, frame)
                else:
                    await self._send(item)
                # Dropped now, or an idle connection would hold the last call and its reply.
                del item
        except Exception as e:
            message = f"lost the connection to {self.peer.where}: {e!r}"
            self.close(partial(PeerLostError, message))

    async def _send(self, call: _Call) -> None:
        if not call.start_send():
            return
        remaining = call.deadline - time.monotonic()
        if remaining <= 0:
            call.settle(error=CallTimeoutError(call.timeout_message))
            return
        try:
            chunks = codec.encode_frame(
                Kind.REQUEST,
                self.last_id + 1,
                deadline_us=int(min(remaining, 1e9) * 1e6),
                body=call.body,  # type: ignore[arg-type]
                buffers=call.buffers,  # type: ignore[arg-type]
            )
        except (CarrierProtocolError, ValueError) as e:
            # Its traceback's frames hold the call, which holds the error once settled.
            _drop_traceback(e)
            call.settle(error=e)
            return
        self.last_id += 1
        call.request_id = self.last_id
        self.pending[call.request_id] = call
        self.unanswered += 1
        self.sending = call
        await _send_chunks(self.loop, self.sock, chunks)
        self.sending = None
        del chunks
        if call.end_send() and self.pending.pop(call.request_id, None) is call:
            self.outbox.appendleft(call.request_id)

    async def _read(self) -> None:
        try:
            while True:
                header, _, body, buffers, _ = await _read_frame(self.reader)
                if header.kind not in (Kind.REPLY, Kind.ERROR):
                    raise CarrierProtocolError(
                        f"a {header.kind.name} frame from the server"
                    )
                call = self.pending.pop(header.request_id, None)
                if call is None and header.request_id > self.last_id:
                    raise CarrierProtocolError(
                        f"a reply to request {header.request_id}, which was never sent"
                    )
                # The server answers every request once, abandoned ones included.
                if self.unanswered:
                    self.unanswered -= 1
                    if self.outbox:
                        self.ready.set()
                if call is not None:
                    call.settle((header.kind, body, buffers))
                # Dropped now, or an idle connection would hold the last reply.
                del body, buffers, call
        except _EndOfStream:
            message = f"{self.peer.where} closed the connection"
            self.close(partial(PeerLostError, message))
        except Exception as e:
            message = f"lost the connection to {self.peer.where}: {e!r}"
            self.close(partial(PeerLostError, message))


class CarrierClient:
    """A handle for calls to one served object. Handles to one address share a connection.

    Calls may come from any thread and loop. Each waits on its caller's loop.

    Raises:
        TypeError: If ``address`` is not a ``CarrierAddress``.
        PeerLostError: Once the carrier's exit hook has begun.
    """

    def __init__(self, address: CarrierAddress) -> None:
        if not isinstance(address, CarrierAddress):
            raise TypeError(f"need a CarrierAddress, not {type(address).__name__}")
        self.address = address
        self._closed = False
        with _lock:
            self._peer = self._attach()
            _handles.add(self)

    def __repr__(self) -> str:
        return f"CarrierClient({self.address!r})"

    def __reduce_ex__(self, protocol: Any) -> NoReturn:
        raise TypeError("CarrierClient cannot be pickled: pass its CarrierAddress")

    async def call(
        self,
        endpoint: str,
        /,
        *args: Any,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> Any:
        """Call ``endpoint`` with all but ``timeout_s``. Return its result, or raise its error as its own type or ``RemoteError``.

        A reply that does not unpickle raises unpickling's own error, and a handler's own carrier errors arrive as ``RemoteError``.

        Raises:
            CallTimeoutError: If ``timeout_s`` (300 s, queue time included) or 30 s to connect runs out.
            CarrierError: ``PeerLostError`` if the server is gone, restarted or breaks the protocol, or another caller
                abandoned a call mid-send on this connection. ``CarrierProtocolError`` if the server speaks another protocol
                version or answers a hello with a HELLO frame, if the call or its reply passes a frame limit, or if an error
                reply is malformed.
        """
        timeout = _CALL_TIMEOUT_S if timeout_s is None else timeout_s
        deadline = time.monotonic() + timeout
        peer = self._live_peer()
        message = (
            f"call {endpoint!r} to {peer.where} did not finish within its"
            f" {timeout:g} s deadline"
        )
        body, buffers = codec.dumps((endpoint, args, kwargs))
        call = _Call(body, buffers, deadline, message)
        del body, buffers
        if not _post_unless_stopped(peer.runtime.loop, peer.submit, call):
            raise PeerLostError(f"{peer.where}: the carrier's I/O loop has stopped")
        try:
            done = await wait_future(
                call.outcome, max(deadline - time.monotonic(), 0.0)
            )
        except BaseException:
            await self._abandon(peer, call)
            raise
        if not done:
            await self._abandon(peer, call)
            if not call.outcome.done():
                raise CallTimeoutError(message)
        try:
            kind, body, received = call.outcome.result()
        finally:
            # Its outcome keeps an error raised here, whose traceback would hold this frame.
            del call
        if kind == Kind.ERROR:
            raise codec.decode_error(body)
        return codec.loads(body, received)

    async def close(self) -> None:
        """Close this handle. The shared connection closes, failing its calls, with the last one."""
        with _lock:
            if self._closed:
                return
            self._closed = True
            peer = self._peer
            peer.handles.discard(self)
            last = not peer.handles and not peer.closed and peer.runtime is _runtime
            if last:
                peer.closed = True
                if peer.runtime.peers.get(self.address) is peer:
                    del peer.runtime.peers[self.address]
        if last:
            await _call_on(peer.runtime.loop, peer.close)

    def _attach(self) -> _Peer:
        """Join this process's connection to the address. Call with ``_lock`` held."""
        runtime = _current_runtime()
        peer = runtime.peers.get(self.address)
        if peer is None:
            peer = runtime.peers[self.address] = _Peer(runtime, self.address)
        peer.handles.add(self)
        return peer

    def _live_peer(self) -> _Peer:
        with _lock:
            if self._closed:
                where = f"{self.address.host}:{self.address.port}"
                raise PeerLostError(f"carrier client for {where} is closed")
            # A handle inherited across fork joins this process's runtime.
            if self._peer.runtime is not _runtime:
                self._peer = self._attach()
            return self._peer

    async def _abandon(self, peer: _Peer, call: _Call) -> None:
        """Give up on ``call`` and return once the I/O thread no longer reads its memory.

        If that thread still holds its frame after ``_RELEASE_WAIT_S``, first shut its connection's sending side down.
        """
        sending = call.abandon()
        _post(peer.runtime.loop, peer.abandoned, call)
        if not sending or call.released is None:
            return
        # A shielded plain future: neither another cancel nor a loop's teardown, which
        # cancels every task, ends the wait while the frame reads caller memory.
        loop = asyncio.get_running_loop()
        waiter = loop.create_future()
        call.released.add_done_callback(lambda _: _post(loop, _set_done, waiter))
        timer = loop.call_later(_RELEASE_WAIT_S, _set_done, waiter)
        cancelled: asyncio.CancelledError | None = None
        try:
            while not waiter.done():
                try:
                    await asyncio.shield(waiter)
                except asyncio.CancelledError as e:
                    cancelled = e
        finally:
            timer.cancel()
        if not call.released.done():
            conn = call.conn
            assert conn is not None
            # Under the lock carrier sockets close under, so the number is still this socket's. No later send reads the frame.
            with _lock:
                try:
                    conn.sock.shutdown(socket.SHUT_WR)
                except OSError:
                    pass
            logger.warning(
                "%s: an abandoned call returns after %g s with the I/O thread still holding its frame,"
                " so its connection stops sending",
                peer.where,
                _RELEASE_WAIT_S,
            )
        if cancelled is not None:
            raise cancelled


class _Request:
    """A received call, its deadline and parts, and where it stands on its server."""

    __slots__ = (
        "abandoned",
        "answered",
        "body",
        "buffers",
        "conn",
        "deadline",
        "held",
        "received",
        "request_id",
        "state",
        "task",
    )

    def __init__(
        self,
        conn: _ServerConnection,
        request_id: int,
        received: float,
        deadline: float,
        body: bytes | memoryview,
        buffers: list[torch.Tensor],
    ) -> None:
        self.conn, self.request_id = conn, request_id
        self.received, self.deadline = received, deadline
        self.body: bytes | memoryview | None = body
        self.buffers: list[torch.Tensor] | None = buffers
        self.state = _QUEUED
        # Its caller gave up. It still gets one answer, which frees its place in the window.
        self.abandoned = False
        # Its one answer is on its way, so no other goes out.
        self.answered = False
        # Not dispatched until its reader has read what arrived with it, its CANCEL say, or stopped at the
        # queue bound, which only a concurrent server's reader reaches: a CANCEL read later cancels its task.
        self.held = False
        self.task: asyncio.Task[None] | None = None


class _ServerConnection:
    """One client's connection to a server, run on the I/O thread."""

    def __init__(self, server: CarrierServer, sock: socket.socket) -> None:
        self.server, self.sock = server, sock
        self.loop = server._runtime.loop
        self.reader = _Reader(self.loop, sock, server._reading, _stamp_arrivals(sock))
        # Guarded by the server's lock: requests not yet answered, and the queued ones and
        # answers not yet written, which together count against the queue bound.
        self.requests: dict[int, _Request] = {}
        self.queued = self.unsent = self.unsent_bytes = 0
        self.room = asyncio.Event()
        self.outbox: collections.deque[
            tuple[list[bytes | memoryview], int]
        ] = collections.deque()
        self.ready = asyncio.Event()
        self.flushed = asyncio.Event()
        self.flushed.set()
        self.closed = False
        # Half closed by a closing server, which reads on until its client's EOF.
        self.draining = False
        self.expired = False
        self.peer = "an unidentified peer"
        self.writer: asyncio.Task[None] | None = None
        # A writer thread's turn at a large reply chunk, kept until it settles or the close.
        self.job: _SendJob | None = None
        # Guarded by the server's lock: requests read but held.
        self.held: list[_Request] = []
        # Used on the I/O thread only: whether a release is due.
        self.releasing = False
        # All the bytes that had arrived at the latest hold, which a turn of the reader's releases only once read.
        self.release_after = 0
        self.task: asyncio.Task[None] | None = self.loop.create_task(self._serve())

    def send(self, chunks: list[bytes | memoryview]) -> None:
        """Queue a frame from any thread that does not hold the server's lock."""
        size = _frame_size(chunks)
        with self.server._lock:
            self.add_unsent(size)
        if not _post_unless_stopped(self.loop, self._queue_frame, chunks, size):
            self.sent(size)

    def add_unsent(self, size: int) -> None:
        """Count a frame on its way to the outbox. Call with the server's lock held."""
        self.unsent += 1
        self.unsent_bytes += size

    def sent(self, size: int) -> None:
        """Uncount a frame written or dropped, waking the reader or the dispatcher it held."""
        server = self.server
        with server._lock:
            over = self.unsent_bytes > _UNSENT_BYTES_PER_CONNECTION
            self.unsent -= 1
            self.unsent_bytes -= size
            room = self.queued + self.unsent == server._bound - 1
            resume = over and self.unsent_bytes <= _UNSENT_BYTES_PER_CONNECTION
        if room:
            _post(self.loop, self.room.set)
        if resume and not server._concurrent:
            _post(server._dloop, server._dispatch)

    def drain(self) -> bool:
        """Half-close a flushed connection, so its replies arrive before any reset. True if so."""
        if self.closed or not self.flushed.is_set():
            return False
        self.draining = True
        try:
            self.sock.shutdown(socket.SHUT_WR)
        except OSError:
            return False
        return True

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.server._connections.discard(self)
        close_socket = partial(_close_socket, self.loop, self.sock)
        # A writer thread copying into the socket closes it once that copy returns.
        if self.job is None or not self.job.defer_close(close_socket):
            close_socket()
        current = asyncio.current_task()
        for task in (self.task, self.writer):
            if task is not None and task is not current:
                task.cancel()
        # A cancelled task keeps a traceback whose frames hold this connection and its server.
        self.task = self.writer = None
        frames, self.outbox = self.outbox, collections.deque()
        for _, size in frames:
            self.sent(size)
        self.flushed.set()

    def release_soon(self) -> None:
        """Release the held requests once the reader has read the frames already here, or stops to wait."""
        self.release_after = self.reader.consumed() + self.reader.unread()
        if not self.releasing:
            self.releasing = True
            self.loop.call_soon(self._release)

    def _release(self) -> None:
        reader = self.reader
        if reader.turning and reader.consumed() < self.release_after:
            # A frame that came with the held requests, their CANCEL say, is still being read.
            self.loop.call_soon(self._release)
            return
        self.releasing = False
        self.server._unhold(self)

    def _queue_frame(self, chunks: list[bytes | memoryview], size: int) -> None:
        if self.closed or self.draining:
            self.sent(size)
            return
        self.outbox.append((chunks, size))
        self.flushed.clear()
        self.ready.set()

    def _expire(self) -> None:
        self.expired = True
        if self.task is not None:
            self.task.cancel()

    async def _serve(self) -> None:
        server = self.server
        service = server.address.service
        timer = self.loop.call_at(self.loop.time() + _CONNECT_TIMEOUT_S, self._expire)
        try:
            welcomed = await self._handshake()
            timer.cancel()
            if welcomed:
                self.writer = self.loop.create_task(self._write())
                await self._read()
        except asyncio.CancelledError:
            if not self.expired:
                raise
            logger.info("carrier service=%s: a handshake timed out", service)
        except _EndOfStream:
            pass
        except CarrierProtocolError as e:
            logger.warning(
                "carrier service=%s dropped a connection from %r: %s",
                service,
                self.peer,
                e,
            )
        except OSError as e:
            # Clients close with a reset, so this is how one leaves.
            if not isinstance(e, ConnectionResetError):
                logger.info("carrier service=%s lost %r: %r", service, self.peer, e)
        finally:
            timer.cancel()
            server._lost(self)
            self.close()

    async def _handshake(self) -> bool:
        server = self.server
        try:
            data = await self.reader.read(codec.HEADER_SIZE)
            header = codec.decode_header(data, handshake=True)
            body = await self.reader.read(header.body_len)
            if header.kind != Kind.HELLO:
                raise CarrierProtocolError(f"a {header.kind.name} frame before a hello")
            hello = Hello.from_json(body)
        except CarrierProtocolError as e:
            await self._reject(str(e))
            raise
        reason = server._refusal(hello)
        if reason is not None:
            await self._reject(reason)
            return False
        # Taken only from a peer that proved the scope, as logs show it.
        self.peer = hello.peer
        address = server.address
        welcome = Welcome(
            scope=address.scope,
            service=address.service,
            incarnation=address.incarnation,
            peer=_peer_name(),
            window=server._window(),
        )
        frame = codec.encode_frame(Kind.WELCOME, body=welcome.to_json().encode())
        await _send_chunks(self.loop, self.sock, frame)
        server._count("connections")
        return True

    async def _reject(self, reason: str) -> None:
        self.server._count("rejected")
        frame = codec.encode_frame(Kind.REJECT, body=Reject(reason).to_json().encode())
        await _send_chunks(self.loop, self.sock, frame)

    async def _read(self) -> None:
        server = self.server
        while True:
            await self._wait_for_room()
            header, arrived, body, buffers, size = await _read_frame(self.reader)
            if header.kind == Kind.REQUEST:
                server._enqueue(self, header, arrived, body, buffers, size)
            elif header.kind == Kind.CANCEL:
                server._cancel(self, header.request_id, size)
            else:
                raise CarrierProtocolError(f"a {header.kind.name} frame from a client")
            # Dropped now, or an idle connection would hold the last request.
            del body, buffers

    async def _wait_for_room(self) -> None:
        """Wait while the connection holds its bound of queued requests and unsent answers.

        A stopped reader cannot see its client go, so after each wake it checks for a hangup,
        raising ``_EndOfStream`` before it reads any more of that client's calls.
        """
        server = self.server
        stopped = False
        while True:
            with server._lock:
                full = self.queued + self.unsent >= server._bound
                if full:
                    self.room.clear()
            if stopped and _hung_up(self.sock):
                raise _EndOfStream()
            if not full:
                return
            stopped = True
            timer = self.loop.call_later(_HANGUP_CHECK_S, self.room.set)
            try:
                await self.room.wait()
            finally:
                timer.cancel()

    async def _write(self) -> None:
        server = self.server
        try:
            while True:
                while not self.outbox:
                    self.flushed.set()
                    self.ready.clear()
                    await self.ready.wait()
                chunks, size = self.outbox.popleft()
                try:
                    sent = await _send_chunks(
                        self.loop, self.sock, chunks, server._sending, self._send_large
                    )
                finally:
                    del chunks
                    self.sent(size)
                server._count("bytes_out", sent)
        except Exception as e:
            _drop_traceback(e)
            # Clients close with a reset, which a send can meet before the reader does.
            if not isinstance(e, (ConnectionResetError, BrokenPipeError)):
                logger.info(
                    "carrier service=%s lost %r: %r",
                    server.address.service,
                    self.peer,
                    e,
                )
            self.close()

    async def _send_large(self, chunk: bytes | memoryview) -> None:
        """Send a large chunk in steps: here while it is the only one in flight, else on writer threads."""
        writers = self.server._runtime.writers
        writers.sending += 1
        try:
            view = memoryview(chunk).cast("B")
            offset = 0
            while offset < len(view):
                if writers.sending == 1 or not writers.start():
                    try:
                        offset += self.sock.send(
                            view[offset : offset + _WRITE_TURN_BYTES]
                        )
                    except BlockingIOError:
                        pass
                    if offset < len(view):
                        await _writable(self.loop, self.sock)
                    continue
                job = _SendJob(self.loop, self.sock, view[offset:])
                # A close cancels the job through this before it cancels the wait below.
                self.job = job
                writers.jobs.put(job)
                sent, full = await job.done
                self.job = None
                offset += sent
                if offset < len(view) and (full or not sent):
                    await _writable(self.loop, self.sock)
        finally:
            writers.sending -= 1


class CarrierServer:
    """Serves one object's methods over the carrier on every interface, on a dispatcher thread of its own.

    Calls run one at a time, each connection's in the order sent, but a call held for what came with it, or whose connection
    has too many reply bytes unsent, lets other connections' later calls go first. With ``concurrent=True`` each is a task
    that a CANCEL, its client's death or ``close``'s grace ending may cancel at any await. Writable reply buffers are copied.
    """

    def __init__(
        self,
        address: CarrierAddress,
        endpoints: dict[str, Callable[..., Any]],
        listener: socket.socket,
        concurrent: bool,
        runtime: _Runtime,
    ) -> None:
        self.address = address
        self._endpoints = endpoints
        self._listener = listener
        self._concurrent = concurrent
        self._bound = _QUEUE_BOUND
        self._runtime = runtime
        self._lock = threading.Lock()
        self._queue: collections.deque[_Request] = collections.deque()
        self._running: set[_Request] = set()
        self._stats = dict.fromkeys((*_LOGGED_STATS, "connections", "rejected"), 0)
        self._max_queue_wait = 0.0
        self._stats_logged = time.monotonic()
        self._closing = False
        self._stop_when_idle = False
        # Replies on their way to a connection's outbox, which close waits for.
        self._unsent = 0
        # Set once no reply is on its way after the grace ends, so none arrives after a drain.
        self._handed_off: futures.Future[None] | None = None
        self._idle: futures.Future[None] = futures.Future()
        self._closed: futures.Future[None] = futures.Future()
        # A shutdown left to the I/O loop, held because asyncio holds tasks weakly.
        self._shutdown: futures.Future[None] | None = None
        # The answers to calls a close fails and to expired calls, naming this server.
        where = f"carrier {address.service} at {address.host}:{address.port}"
        self._closed_body = codec.encode_error(
            PeerLostError(f"{VOLUME_CLOSED}: {where}")
        )
        # The answer to a call that exit stopped from starting, which no close made.
        self._exit_body = codec.encode_error(PeerLostError(f"process exiting: {where}"))
        self._expired_body = codec.encode_error(
            CallTimeoutError(
                f"{where} dropped a call whose deadline passed in its queue"
            )
        )
        # Used on the I/O thread only. The events are test hooks.
        self._connections: set[_ServerConnection] = set()
        self._accepting: asyncio.Task[None] | None = None
        self._reading = asyncio.Event()
        self._reading.set()
        self._sending = asyncio.Event()
        self._sending.set()
        self._dispatcher = _LoopThread(f"torchstore-carrier-dispatch {address.service}")
        self._dloop = self._dispatcher.loop
        with _lock:
            # Again, under the lock exit sets its flag with: exit may have begun since serve's check.
            with _loop_threads_lock:
                exiting = _exiting
            if not exiting:
                runtime.servers.add(self)
                _servers.add(self)
        if exiting:
            _stop_loop(self._dloop)
            raise PeerLostError(
                "the carrier takes no new client or server once exit has begun"
            )
        _post(runtime.loop, self._start_accepting)

    @classmethod
    async def serve(
        cls,
        target: Any,
        endpoints: Collection[str],
        /,
        *,
        service: str,
        scope: str,
        host: str | None = None,
        port: int = 0,
        concurrent: bool = False,
    ) -> CarrierServer:
        """Serve ``target``'s ``endpoints`` on every interface. Clients dial ``host``, or else this host's name.

        ``host`` never limits where it listens. The scope keeps out stale clients, not attackers: the network is trusted.

        Raises:
            TypeError: If ``service``, ``scope`` or ``host`` is not a str, ``port`` not an int, or ``endpoints`` a single str.
            ValueError: If ``port`` is outside 0-65535, or ``target`` has no method for an endpoint.
            OSError: If no port can be bound on any interface.
            PeerLostError: Once the carrier's exit hook has begun. It is a ``ConnectionError`` so an ``OSError``
                handler catches it too.
        """
        if isinstance(endpoints, str):
            raise TypeError(
                f"endpoints must be a collection of names, not the str {endpoints!r}"
            )
        for name, value in (("service", service), ("scope", scope)):
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a str, not {type(value).__name__}")
        if host is not None and not isinstance(host, str):
            raise TypeError(f"host must be a str, not {type(host).__name__}")
        if isinstance(port, bool) or not isinstance(port, int):
            raise TypeError(f"port must be an int, not {type(port).__name__}")
        if not 0 <= port <= 65535:
            raise ValueError(f"port must be from 0 to 65535, not {port}")
        handlers = {name: getattr(target, name, None) for name in endpoints}
        missing = sorted(
            name for name, handler in handlers.items() if not callable(handler)
        )
        if missing:
            raise ValueError(f"{type(target).__name__} has no methods {missing}")
        with _lock:
            runtime = _current_runtime()
            listener = _listen(port)
            runtime.sockets.add(listener)
        try:
            address = CarrierAddress(
                host=host or socket.gethostname(),
                port=listener.getsockname()[1],
                service=service,
                scope=scope,
                incarnation=secrets.token_hex(8),
            )
            server = cls(address, handlers, listener, concurrent, runtime)  # type: ignore[arg-type]
        except BaseException:
            with _lock:
                listener.close()
            raise
        logger.info(
            "carrier serve service=%s address=%s:%d incarnation=%s pid=%d",
            service,
            address.host,
            address.port,
            address.incarnation,
            os.getpid(),
        )
        return server

    def __repr__(self) -> str:
        return f"CarrierServer({self.address!r})"

    def __reduce_ex__(self, protocol: Any) -> NoReturn:
        raise TypeError("CarrierServer cannot be pickled: pass its CarrierAddress")

    @property
    def dispatcher(self) -> _LoopThread:
        """This server's dispatcher, with its ``thread``, its ``loop`` and a ``stopped`` future.

        Another exit hook that blocks that thread for good sets its ``parked``, so the carrier's exit hook skips it.
        """
        return self._dispatcher

    def stats(self) -> dict[str, int]:
        """Return the counters, including connections and rejected hellos, and how many calls are queued and running."""
        with self._lock:
            return {
                **self._stats,
                "max_queue_wait_ms": int(self._max_queue_wait * 1000),
                "queued": len(self._queue),
                "running": len(self._running),
            }

    async def close(self, *, grace_s: float = 30.0) -> None:
        """Stop serving. Queued calls fail at once. Running ones get ``grace_s`` to finish, or the first close's grace.

        Callers left over get ``PeerLostError`` reading "volume closed" and naming the server. Concurrent mode cancels their
        tasks, and a FIFO handler runs on until it ends, as ``stats()["running"]`` shows. A cancelled close ends only its wait.
        """
        if self._begin_close(grace_s):
            await wait_future(self._closed, None)

    def _begin_close(self, grace_s: float) -> bool:
        """Start the shutdown once, on a thread of its own or else the I/O loop.

        Returns False if the server was inherited across fork, which leaves nothing to close.
        """
        if self._runtime is not _runtime:
            return False
        deadline = time.monotonic() + max(grace_s, 0.0)
        with self._lock:
            first, self._closing = not self._closing, True
            queued = list(self._queue) if first else []
            self._queue.clear()
            for request in queued:
                self._drop(request)
            self._check_idle()
        # Only where each error goes, so the requests and their parts are freed here.
        failed = [(request.conn, request.request_id) for request in queued]
        del queued
        if first:
            # The thread holds this server only through ``work``, which it empties first.
            work: list[Any] = [self, failed, deadline]
            thread = threading.Thread(
                target=_shut_down_on_thread,
                args=(work,),
                name=f"torchstore-carrier-close {self.address.service}",
                daemon=True,
            )
            try:
                thread.start()
            except RuntimeError as e:
                work.clear()
                self._shut_down_on_io_loop(self._shut_down_here(failed, deadline), e)
            else:
                with _loop_threads_lock:
                    _close_threads.add(thread)
        return True

    def _shut_down_on_io_loop(
        self, shutdown: Coroutine[Any, Any, None], error: Exception
    ) -> None:
        closed, loop = self._closed, self._runtime.loop
        posted = None
        # Checked as _post_call checks: under the lock that _stop_loop marks a loop stopped with.
        with _loop_threads_lock, contextlib.suppress(RuntimeError):
            if loop not in _stopped:
                posted = asyncio.run_coroutine_threadsafe(shutdown, loop)
                self._shutdown = posted
        if posted is None:
            # Exit stopped or closed the I/O loop, which ends what this close would.
            shutdown.close()
            _resolve(closed)
        else:
            # At a thread or file limit, for example. Unlike the caller's loop, the I/O loop runs until exit.
            logger.warning(
                "carrier service=%s closes on the I/O thread: %r",
                self.address.service,
                error,
            )
            # Called once the finished shutdown has let go of this server.
            posted.add_done_callback(lambda _: _resolve(closed))

    async def _shut_down_here(
        self, failed: list[tuple[_ServerConnection, int]], deadline: float
    ) -> None:
        try:
            await self._shut_down(failed, deadline)
        except Exception:
            logger.exception("carrier service=%s failed to close", self.address.service)

    async def _shut_down(
        self, failed: list[tuple[_ServerConnection, int]], deadline: float
    ) -> None:
        runtime = self._runtime
        await _call_on(runtime.loop, self._stop_accepting)
        for conn, request_id in failed:
            conn.send(self._closed_frame(request_id))
        if not await wait_future(self._idle, max(deadline - time.monotonic(), 0.0)):
            with self._lock:
                late = [request for request in self._running if not request.answered]
                for request in late:
                    request.answered = True
                handed_off = self._handed_off = futures.Future()
                if not self._unsent:
                    handed_off.set_result(None)
            for request in late:
                request.conn.send(self._closed_frame(request.request_id))
            if self._concurrent:
                # Now, not once the connections close: their callers were told the volume closed.
                await _run_on(self._dloop, self._drain(), 2 * _RELEASE_WAIT_S)
            # Replies whose handlers finished within the grace, but which have not reached their outbox.
            await wait_future(handed_off, _RELEASE_WAIT_S)
        # Replies still being sent get the rest of the grace, and at least the release wait.
        flush_s = max(deadline - time.monotonic(), _RELEASE_WAIT_S)
        await _run_on(
            runtime.loop,
            self._close_connections(flush_s),
            flush_s + 2 * _RELEASE_WAIT_S,
        )
        await self._stop_dispatcher()
        with _lock:
            runtime.servers.discard(self)
        await self._stop_writers()
        self._log_stats()

    async def _stop_writers(self) -> None:
        """Stop the runtime's reply writers once it serves nothing, and wait for them within a bound."""
        runtime = self._runtime
        stopped: dict[threading.Thread, futures.Future[None]] = {}

        def stop() -> None:
            # On the I/O thread, so no reply hands a job to the writers being stopped.
            with _lock:
                idle = not runtime.servers
            if idle:
                stopped.update(runtime.writers.stop())

        await _call_on(runtime.loop, stop)
        deadline = time.monotonic() + _RELEASE_WAIT_S
        for thread, done in stopped.items():
            if await wait_future(done, max(deadline - time.monotonic(), 0.0)):
                thread.join(max(deadline - time.monotonic(), 0.0))

    async def _set_reading(self, on: bool) -> None:
        """Test hook: stop or resume reading from every connection."""
        await _call_on(
            self._runtime.loop, self._reading.set if on else self._reading.clear
        )

    async def _set_sending(self, on: bool) -> None:
        """Test hook: stop or resume sending replies on every connection."""
        await _call_on(
            self._runtime.loop, self._sending.set if on else self._sending.clear
        )

    def _window(self) -> int:
        """The most requests a client may leave unanswered on a connection, 0 for any.

        One below a FIFO server's bound but at least 1: with a bound of 2 or more the server reads every frame (cancels
        included) as it arrives. Concurrent waits never count, so they cannot hold back the call that releases them.
        """
        return 0 if self._concurrent else max(self._bound - 1, 1)

    def _closed_frame(self, request_id: int) -> list[bytes | memoryview]:
        return codec.encode_frame(Kind.ERROR, request_id, body=self._closed_body)

    def _check_idle(self) -> None:
        """Mark a closing server idle once no call runs and every reply is queued to send."""
        if self._closing and not self._running and not self._unsent:
            if not self._idle.done():
                self._idle.set_result(None)

    def _count(self, name: str, value: int = 1) -> None:
        with self._lock:
            self._stats[name] += value

    def _refusal(self, hello: Hello) -> str | None:
        address = self.address
        # First, so a peer that has not proved the scope learns only that. Others get 64 characters of a field back.
        if not secrets.compare_digest(hello.scope.encode(), address.scope.encode()):
            return "the scope does not match"
        if hello.version != codec.PROTOCOL_VERSION:
            return f"carrier protocol version {str(hello.version)[:64]} is not {codec.PROTOCOL_VERSION}"
        if hello.service != address.service:
            return f"this server serves {address.service!r}, not {hello.service[:64]!r}"
        if hello.incarnation != address.incarnation:
            return (
                f"incarnation {hello.incarnation[:64]!r} is not {address.incarnation}"
            )
        return None

    def _start_accepting(self) -> None:
        if not self._closing:
            self._accepting = self._runtime.loop.create_task(self._accept())

    def _stop_accepting(self) -> None:
        if self._accepting is not None:
            self._accepting.cancel()
            # Its traceback's frames would hold this server.
            self._accepting = None
        _close_socket(self._runtime.loop, self._listener)

    def _accept_now(self) -> socket.socket | None:
        """Accept a waiting connection, registering it under ``_lock`` as fork requires. None if none waits."""
        with _lock:
            try:
                sock, _ = self._listener.accept()
            except (BlockingIOError, InterruptedError):
                return None
            self._runtime.sockets.add(sock)
        sock.setblocking(False)
        return sock

    async def _accept_one(self, loop: asyncio.AbstractEventLoop) -> socket.socket:
        while True:
            sock = self._accept_now()
            if sock is not None:
                return sock
            await _readable(loop, self._listener)

    async def _accept(self) -> None:
        loop = self._runtime.loop
        failing = False
        while True:
            try:
                sock = await self._accept_one(loop)
            except OSError as e:
                # Out of file descriptors, for example, which lasts: retrying at once would
                # spin, and a warning per retry would flood the log.
                if not failing:
                    logger.warning(
                        "carrier service=%s cannot accept: %r", self.address.service, e
                    )
                failing = True
                await asyncio.sleep(0.1)
                continue
            if failing:
                logger.info("carrier service=%s accepts again", self.address.service)
                failing = False
            try:
                _configure(sock)
            except OSError:
                _close_socket(loop, sock)
                continue
            self._connections.add(_ServerConnection(self, sock))

    def _enqueue(
        self,
        conn: _ServerConnection,
        header: codec.FrameHeader,
        arrived: float,
        body: bytes | memoryview,
        buffers: list[torch.Tensor],
        size: int,
    ) -> None:
        # The deadline runs from the kernel's stamp on the receive that took the header's first byte: the newest
        # arrival among that receive's bytes, which include those Linux merged with it while they waited unread.
        # The queue wait runs from the frame's end.
        deadline = arrived + header.deadline_us / 1e6
        request = _Request(
            conn, header.request_id, time.monotonic(), deadline, body, buffers
        )
        with self._lock:
            self._stats["calls"] += 1
            self._stats["bytes_in"] += size
            if header.request_id in conn.requests:
                raise CarrierProtocolError(f"request id {header.request_id} is in use")
            closing = self._closing
            if not closing:
                conn.requests[request.request_id] = request
                conn.queued += 1
                request.held = True
                conn.held.append(request)
                self._queue.append(request)
                depth = max(self._stats["max_queue_depth"], len(self._queue))
                self._stats["max_queue_depth"] = depth
        if closing:
            conn.send(self._closed_frame(request.request_id))
        else:
            conn.release_soon()

    def _unhold(self, conn: _ServerConnection) -> None:
        """Let the dispatcher run the requests held on ``conn``."""
        with self._lock:
            held, conn.held = conn.held, []
            for request in held:
                request.held = False
        if held:
            _post(self._dloop, self._dispatch)

    def _cancel(self, conn: _ServerConnection, request_id: int, size: int) -> None:
        with self._lock:
            self._stats["bytes_in"] += size
            request = conn.requests.get(request_id)
            answer, cancel = (
                (False, False) if request is None else self._abandon(request)
            )
        if answer:
            conn.send(self._closed_frame(request_id))
        if cancel:
            _post(self._dloop, self._cancel_task, request)

    def _lost(self, conn: _ServerConnection) -> None:
        """Abandon a gone client's calls: queued ones drop, running ones finish or are cancelled."""
        with self._lock:
            requests = list(conn.requests.values())
            cancel = [request for request in requests if self._abandon(request)[1]]
        for request in cancel:
            _post(self._dloop, self._cancel_task, request)

    def _abandon(self, request: _Request) -> tuple[bool, bool]:
        """Abandon a request. Return whether to answer it now and whether to cancel its task.

        Call with the lock held. A queued request drops, and its answer frees its client's window.
        """
        if request.abandoned or request.answered:
            return False, False
        request.abandoned = True
        self._stats["cancels"] += 1
        if request.state != _QUEUED:
            return False, self._concurrent
        self._queue.remove(request)
        self._drop(request)
        return True, False

    def _drop(self, request: _Request) -> None:
        """Finish a request that never ran, freeing its parts. Call with the lock held."""
        request.state, request.answered = _FINISHED, True
        request.body = request.buffers = None
        del request.conn.requests[request.request_id]
        self._release(request.conn)

    def _release(self, conn: _ServerConnection) -> None:
        """Take a request off its connection's queue, waking its reader. Call with the lock held."""
        conn.queued -= 1
        if conn.queued + conn.unsent == self._bound - 1:
            _post(conn.loop, conn.room.set)

    def _cancel_task(self, request: _Request) -> None:
        if request.task is not None:
            request.task.cancel()

    def _dispatch(self) -> None:
        while True:
            # A call started at exit could still be in torch once exit stops waiting.
            if _exiting:
                return
            expired: list[_Request] = []
            with self._lock:
                request = None
                if self._concurrent or not self._running:
                    request = self._next(expired)
                if request is not None:
                    request.state = _RUNNING
                    self._running.add(request)
                    wait = time.monotonic() - request.received
                    self._max_queue_wait = max(self._max_queue_wait, wait)
                    self._release(request.conn)
                    # Created under the lock, so a cancel never reaches a task before it starts.
                    request.task = self._dloop.create_task(self._run(request))
            for gone in expired:
                frame = codec.encode_frame(
                    Kind.ERROR, gone.request_id, body=self._expired_body
                )
                gone.conn.send(frame)
            if request is None or not self._concurrent:
                return

    def _next(self, expired: list[_Request]) -> _Request | None:
        """Take the next request to run, moving expired ones to ``expired``. Call with the lock held.

        FIFO mode passes over a connection with more unsent reply bytes than its share.
        """
        now = time.monotonic()
        for request in list(self._queue):
            if now >= request.deadline:
                self._queue.remove(request)
                self._drop(request)
                self._stats["expired"] += 1
                expired.append(request)
            elif request.held:
                continue
            elif (
                self._concurrent
                or request.conn.unsent_bytes <= _UNSENT_BYTES_PER_CONNECTION
            ):
                self._queue.remove(request)
                return request
        return None

    def _refused_start(self, request: _Request) -> bytes | None:
        """The answer to a call whose handler must not start because exit began, its caller gave up or its time ran out."""
        # Under the lock exit sets its flag with: exit may have begun since the dispatch.
        with _loop_threads_lock:
            exiting = _exiting
        if exiting:
            return self._exit_body
        if request.abandoned:
            return self._closed_body
        if time.monotonic() >= request.deadline:
            return self._expired_body
        return None

    async def _run(self, request: _Request) -> None:
        refused = self._refused_start(request)
        if refused is not None:
            request.body = request.buffers = None
            chunks = codec.encode_frame(Kind.ERROR, request.request_id, body=refused)
            self._finish(request, chunks, False)
            return
        failed = True
        try:
            endpoint, args, kwargs = codec.loads(request.body, request.buffers)  # type: ignore[arg-type]
            request.body = request.buffers = None
            # Again right before the handler: a CANCEL, the deadline or exit may have come while it unpickled.
            refused = self._refused_start(request)
            if refused is not None:
                raise _NotStarted(refused)
            handler = self._endpoints.get(endpoint)
            if handler is None:
                raise AttributeError(
                    f"carrier service {self.address.service!r} has no endpoint {endpoint!r}"
                )
            try:
                result = handler(*args, **kwargs)
                if inspect.isawaitable(result):
                    result = await result
            except CarrierError as e:
                if isinstance(e, RemoteError):
                    raise
                # From the handler's own calls: its caller would take it for a failure of the call it made.
                raise RemoteError(codec._type_name(type(e)), str(e)) from e
            body, buffers = codec.dumps(result)
            del result
            chunks = codec.encode_frame(
                Kind.REPLY,
                request.request_id,
                body=body,
                buffers=_copy_mutable(buffers),
            )
            failed = False
        except _NotStarted as refusal:
            chunks = codec.encode_frame(
                Kind.ERROR, request.request_id, body=refusal.args[0]
            )
            failed = False
        except Exception as e:
            try:
                body = codec.encode_error(e)
                chunks = codec.encode_frame(Kind.ERROR, request.request_id, body=body)
            except BaseException as failure:
                # An error's own hooks may raise past the codec. Its caller still gets an answer.
                logger.error(
                    "carrier service=%s cannot encode the %s a call raised: %r",
                    self.address.service,
                    type(e).__name__,
                    failure,
                )
                chunks = self._unencodable_frame(request.request_id, e, failure)
        except BaseException as e:
            # The GC closes a task left pending on the closed dispatcher loop: no one is left to answer.
            if isinstance(e, GeneratorExit) and self._dloop.is_closed():
                raise
            # Caught, so a handler cannot stop the loop. Its caller gets a RemoteError naming the type.
            if isinstance(e, asyncio.CancelledError):
                failed = False
            else:
                logger.error(
                    "carrier service=%s handler raised %r", self.address.service, e
                )
            try:
                chunks = codec.encode_frame(
                    Kind.ERROR, request.request_id, body=codec.encode_error(e)
                )
            except BaseException as failure:
                chunks = self._unencodable_frame(request.request_id, e, failure)
        self._finish(request, chunks, failed)

    def _unencodable_frame(
        self, request_id: int, error: BaseException, failure: BaseException
    ) -> list[bytes | memoryview]:
        """Answer with a ``RemoteError`` naming ``error``'s type, as encoding ``error`` raised ``failure``."""
        text = f"its encoding raised {codec._type_name(type(failure))}"
        body = codec.encode_error(RemoteError(codec._type_name(type(error)), text))
        return codec.encode_frame(Kind.ERROR, request_id, body=body)

    def _finish(
        self, request: _Request, chunks: list[bytes | memoryview], failed: bool
    ) -> None:
        now = time.monotonic()
        conn = request.conn
        with self._lock:
            self._running.discard(request)
            request.state = _FINISHED
            conn.requests.pop(request.request_id, None)
            reply = not request.answered
            request.answered = True
            if reply and request.abandoned:
                # Its caller gave up, so a short answer only frees its place in the window.
                chunks = self._closed_frame(request.request_id)
            size = _frame_size(chunks)
            if reply:
                conn.add_unsent(size)
                self._unsent += 1
            self._stats["errors"] += failed
            self._stats["timeouts"] += now > request.deadline
            self._check_idle()
            stop = self._stop_when_idle and not self._running
            due = now - self._stats_logged >= _STATS_INTERVAL_S
            if due:
                self._stats_logged = now
        if reply and not _post_unless_stopped(
            conn.loop, self._queue_reply, conn, chunks, size
        ):
            self._queue_reply(conn, None, size)
        if stop:
            _stop_loop(self._dloop)
        elif not self._concurrent:
            self._dispatch()
        if due:
            self._log_stats()

    def _queue_reply(
        self,
        conn: _ServerConnection,
        chunks: list[bytes | memoryview] | None,
        size: int,
    ) -> None:
        if chunks is None:
            conn.sent(size)
        else:
            conn._queue_frame(chunks, size)
        with self._lock:
            self._unsent -= 1
            self._check_idle()
            handed_off = self._handed_off
            # Set once: no reply comes after the grace, as every call still running then is answered.
            if not self._unsent and handed_off is not None:
                handed_off.set_result(None)

    def _log_stats(self) -> None:
        stats = self.stats()
        logger.info(
            "carrier stats service=%s calls=%d errors=%d timeouts=%d cancels=%d"
            " expired=%d bytes_in=%d bytes_out=%d max_queue_depth=%d"
            " max_queue_wait_ms=%d",
            self.address.service,
            *(stats[name] for name in _LOGGED_STATS),
            stats["max_queue_wait_ms"],
        )

    async def _close_connections(self, flush_s: float) -> None:
        conns = list(self._connections)
        flushes = [
            asyncio.ensure_future(conn.flushed.wait())
            for conn in conns
            if not conn.closed
        ]
        if flushes:
            _, pending = await asyncio.wait(flushes, timeout=flush_s)
            for flush in pending:
                flush.cancel()
        # Closing with bytes unread resets a connection and drops the replies still in the
        # kernel, so flushed ones half-close first and read on until their client's EOF.
        draining = [conn.task for conn in conns if conn.drain() and conn.task]
        if draining:
            await asyncio.wait(draining, timeout=_RELEASE_WAIT_S)
        for conn in conns:
            conn.close()

    async def _stop_dispatcher(self) -> None:
        """Stop the dispatcher's loop and join its thread, within bounds.

        Concurrent mode cancelled its tasks once the grace ended, and a task that ignored that is left on the closed
        loop. In FIFO mode an abandoned handler keeps the loop until it returns so the thread is not joined.
        """
        if not self._concurrent:
            with self._lock:
                self._stop_when_idle = bool(self._running)
            if self._stop_when_idle:
                return
        _stop_loop(self._dloop)
        if await wait_future(self._dispatcher.stopped, _RELEASE_WAIT_S):
            self._dispatcher.thread.join(_RELEASE_WAIT_S)

    async def _drain(self) -> None:
        with self._lock:
            tasks = [request.task for request in self._running if request.task]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.wait(tasks, timeout=_RELEASE_WAIT_S)


async def close_scope(scope: str, *, grace_s: float = 30.0) -> None:
    """Close this process's carrier clients and servers whose address carries ``scope``, each server as its ``close`` does.

    Raises:
        TypeError: If ``scope`` is not a str.
    """
    if not isinstance(scope, str):
        raise TypeError(f"scope must be a str, not {type(scope).__name__}")
    await _close_where(lambda address: address.scope == scope, grace_s)


async def close_all(*, grace_s: float = 30.0) -> None:
    """Close every carrier client and server in this process, each server as its ``close`` does."""
    await _close_where(lambda address: True, grace_s)


async def _close_where(match: Callable[[CarrierAddress], bool], grace_s: float) -> None:
    with _lock:
        runtime, peers = _detach_clients(match)
        if runtime is None:
            return
        servers = [server for server in runtime.servers if match(server.address)]
    # All of it starts before the first await, so a cancelled caller still closes everything.
    peers_closed = _post_call(runtime.loop, partial(_close_peers, peers))
    closing = [server for server in servers if server._begin_close(grace_s)]
    await wait_future(peers_closed, _RELEASE_WAIT_S)
    for server in closing:
        await wait_future(server._closed, None)


def _detach_clients(
    match: Callable[[CarrierAddress], bool],
) -> tuple[_Runtime | None, list[_Peer]]:
    """Close the matching handles, and take their peers out of the runtime for ``_close_peers``. Call with ``_lock`` held."""
    # Handles inherited across fork join this runtime only on their next call.
    for handle in list(_handles):
        if match(handle.address):
            handle._closed = True
            handle._peer.handles.discard(handle)
    runtime = _runtime
    if runtime is None:
        return None, []
    peers = [peer for address, peer in runtime.peers.items() if match(address)]
    for peer in peers:
        peer.handles.clear()
        peer.closed = True
        del runtime.peers[peer.address]
    return runtime, peers


def _close_peers(peers: list[_Peer]) -> None:
    for peer in peers:
        try:
            peer.close()
        except Exception:
            logger.exception("carrier close failed for %s", peer.where)


def _shut_down_on_thread(work: list[Any]) -> None:
    """Run a server's shutdown on this thread's own loop, then mark the server closed.

    It lets go of the server before its closer can return, as freeing the served object's tensors in a finalizing
    interpreter aborts the process. Once exit has begun it leaves the server to the main thread, as exit may stop waiting.
    """
    server, failed, deadline = work
    work.clear()
    closed = server._closed
    shutdown = server._shut_down_here(failed, deadline)
    del failed
    handed_over = False
    try:
        asyncio.run(shutdown)
    except Exception as e:
        if inspect.getcoroutinestate(shutdown) != inspect.CORO_CREATED:
            raise
        # No event loop could start, so the shutdown has not begun.
        server._shut_down_on_io_loop(shutdown, e)
        handed_over = True
    finally:
        thread = threading.current_thread()
        with _loop_threads_lock:
            park = _exiting
            if not park:
                _freeing.add(thread)
        if park:
            _parked.append(server)
        try:
            del server, shutdown
        finally:
            with _loop_threads_lock:
                _freeing.discard(thread)
        if not handed_over:
            _resolve(closed)


def _resolve(future: futures.Future[None]) -> None:
    if not future.done():
        future.set_result(None)
