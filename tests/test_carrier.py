# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the carrier's runtime: calls, deadlines, failures, closing and fork."""

import asyncio
import concurrent.futures
import contextlib
import errno
import fcntl
import functools
import gc
import inspect
import json
import logging
import os
import pickle
import re
import select
import signal
import socket
import struct
import subprocess
import sys
import termios
import threading
import time
import types
import uuid
import warnings
import weakref
from collections.abc import AsyncIterator, Callable, Coroutine, Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
import torchstore
from torchstore.carrier import (
    CallTimeoutError,
    CarrierAddress,
    CarrierClient,
    CarrierProtocolError,
    CarrierServer,
    close_all,
    close_scope,
    codec,
    PeerLostError,
    RemoteError,
    runtime,
    StaleAddressError,
)
from torchstore.carrier.codec import Hello, Kind, Reject, Welcome
from torchstore.routing.layout_registry import LayoutRegistry

pytestmark = [
    pytest.mark.timeout(120),
    pytest.mark.filterwarnings("error::pytest.PytestUnhandledThreadExceptionWarning"),
]

# Child processes import the tree that pytest imported.
_ROOT = str(Path(torchstore.__file__).parents[1])
_PUBLISHERS = ("publisher/0", "publisher/1")
# Layout lookups never read the registry's strategy.
_NO_STRATEGY: Any = None
_ENDPOINTS = ("echo", "hold", "record", "fail", "exit")

_CHILD = """
import asyncio, json, os, sys, threading, time, traceback

# The carrier never needs Monarch, so importing it here fails.
sys.modules["monarch"] = None
sys.path.insert(0, sys.argv[1])
from torchstore.carrier import CarrierAddress, CarrierClient, CarrierServer, runtime


class Target:
    def echo(self, value):
        return value

    def hold(self, label=None):
        threading.Event().wait()
"""

_SERVE_CHILD = (
    _CHILD
    + """
server = asyncio.run(
    CarrierServer.serve(Target(), ["echo", "hold"], service="child", scope=sys.argv[2])
)
print(server.address.to_json(), flush=True)
sys.stdin.read()
"""
)

_CALL_CHILD = (
    _CHILD
    + """
async def main():
    client = CarrierClient(CarrierAddress.from_json(sys.argv[2]))
    calls = [client.call(endpoint, *args) for endpoint, args in json.loads(sys.argv[3])]
    await asyncio.wait([asyncio.ensure_future(call) for call in calls])


asyncio.run(main())
"""
)

_THREADS_CHILD = (
    _CHILD
    + """
client = CarrierClient(CarrierAddress.from_json(sys.argv[2]))
asyncio.run(client.call("record", os.getpid(), -1, -1))
print("ready", flush=True)
# Every child starts its calls together, so calls on different connections overlap.
sys.stdin.readline()
errors = []


def work(thread):
    try:
        for i in range(int(sys.argv[3])):
            reply = asyncio.run(client.call("record", os.getpid(), thread, i))
            assert reply == [os.getpid(), thread, i], reply
    except BaseException as e:
        errors.append(repr(e))


threads = [threading.Thread(target=work, args=(thread,)) for thread in range(4)]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
print(json.dumps(errors), flush=True)
"""
)

_FORK_LOCK_CHILD = (
    _CHILD
    + """
import signal

server = asyncio.run(CarrierServer.serve(Target(), ["echo"], service="parent", scope="f/2"))
client = CarrierClient(server.address)
assert asyncio.run(client.call("echo", 1)) == 1
# Another name for the server, so this handle is its connection's only one.
a = server.address
unused = CarrierClient(CarrierAddress("127.0.0.1", a.port, a.service, a.scope, a.incarnation))
# An ordinary fork takes the carrier lock, which the parent gets back.
pid = os.fork()
if pid == 0:
    os._exit(0)
os.waitpid(pid, 0)
assert runtime._lock.acquire(timeout=10), "the parent kept no carrier lock after a fork"
runtime._lock.release()
fds = [server._listener.fileno(), client._peer.conn.sock.fileno()]
fds += [conn.sock.fileno() for conn in server._connections]
held, release = threading.Event(), threading.Event()


def hold_locks():
    # As a serve, a connect or a close holds the first, a loop thread starting or stopping the second, a call the last.
    with runtime._lock, runtime._loop_threads_lock, server._lock:
        held.set()
        release.wait()


threading.Thread(target=hold_locks, daemon=True).start()
held.wait()
# As if the parent's exit hook had begun.
runtime._exiting = True
pid = os.fork()
if pid == 0:
    # A child stuck on a lock its parent's other thread held, or on a loop of its parent's, dies at the alarm.
    signal.alarm(30)
    runtime._RELEASE_WAIT_S = 3600
    try:
        for fd in fds:
            try:
                os.fstat(fd)
            except OSError:
                continue
            raise AssertionError(f"fd {fd} survived the fork")
        assert not runtime._loop_threads and server.stats()["calls"] == 1
        # The inherited server and an inherited handle never used here close at once.
        asyncio.run(server.close())
        asyncio.run(unused.close())
        # A used one joins this process's runtime, so it reaches the parent's server.
        assert asyncio.run(client.call("echo", 2, timeout_s=10)) == 2
        child = asyncio.run(CarrierServer.serve(Target(), ["echo"], service="child", scope="l/1"))
        assert asyncio.run(CarrierClient(child.address).call("echo", 1)) == 1
        os._exit(0)
    except BaseException:
        traceback.print_exc()
        os._exit(1)
runtime._exiting = False
release.set()
_, status = os.waitpid(pid, 0)
assert asyncio.run(client.call("echo", 3)) == 3
print(os.waitstatus_to_exitcode(status), flush=True)
"""
)

_fired = threading.Event()


def _fire() -> None:
    _fired.set()


class _Marker:
    """Sets ``_fired`` when unpickled."""

    def __reduce__(self) -> tuple[Any, ...]:
        return _fire, ()


class _Target:
    """A served object that records its calls and can hold them until released."""

    def __init__(self) -> None:
        self.calls: list[Any] = []
        self.finished: list[Any] = []
        self.release = threading.Event()
        self.active = 0
        self.max_active = 0

    def echo(self, value: Any) -> Any:
        self.calls.append(value)
        return value

    def hold(self, label: Any = None) -> Any:
        """Block the dispatcher thread until released."""
        self.calls.append(label)
        self.release.wait()
        return label

    async def wait(self, label: Any = None) -> Any:
        """Await release off the dispatcher thread, as a volume's async handlers await."""
        self.calls.append(label)
        await asyncio.get_running_loop().run_in_executor(None, self.release.wait)
        self.finished.append(label)
        return label

    async def record(self, *value: Any) -> list[Any]:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        # Any call running alongside this one would start during these yields.
        for _ in range(3):
            await asyncio.sleep(0)
        self.calls.append(value)
        self.active -= 1
        return list(value)

    def fail(self, error: BaseException) -> None:
        raise error

    def exit(self) -> None:
        raise SystemExit("handler exits")

    def first(self, items: Any) -> Any:
        return next(iter(items))

    async def nap(self, seconds: float) -> None:
        """Take a while, as a slow handler does, before its answer goes out."""
        self.calls.append("nap")
        await asyncio.sleep(seconds)

    def where(
        self, endpoint: str = "default", timeout_s: float | None = None
    ) -> tuple[str, float | None]:
        """Return its own keywords, which share the names of ``CarrierClient.call``'s parameters."""
        return endpoint, timeout_s


class _Store:
    """Hands out its own tensor, like a volume's get, and a reply pickled once."""

    def __init__(self) -> None:
        self.tensor = torch.zeros(1 << 16)
        self.data = bytes(range(256)) * 1024

    async def get(self) -> torch.Tensor:
        return self.tensor

    async def fill(self, value: float) -> float:
        self.tensor.fill_(value)
        return value

    def payload(self) -> pickle.PickleBuffer:
        return pickle.PickleBuffer(self.data)


class _RegistryHost:
    """Serves the routing ``LayoutRegistry``, passing a lookup's own wait as an argument."""

    def __init__(self) -> None:
        self.registry = LayoutRegistry()
        self.started = 0
        self.cancelled = 0

    async def register_layout(self, rank: str, key: str, layout: Any) -> None:
        await self.registry.register_layout(rank, key, layout)

    async def get_layouts(self, key: str, wait_s: float) -> Any:
        self.started += 1
        try:
            return await self.registry.get_layouts(key, timeout_s=wait_s)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise


class _Big:
    """Hands out a fresh 4 MiB tensor once released, and a column or row of its own tensor, which it fills."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.tensor = torch.zeros(1024, 4096)

    def big(self) -> torch.Tensor:
        self.release.wait()
        return torch.zeros(4 << 20, dtype=torch.uint8)

    def column(self) -> torch.Tensor:
        return self.tensor.narrow(1, 0, 2048)

    def row(self) -> torch.Tensor:
        return self.tensor.narrow(0, 0, 512)

    def fill(self, value: float) -> float:
        self.tensor.fill_(value)
        return value


class _Sink:
    """Hands out fresh gets of a given size, and echoes."""

    def __init__(self, size: int) -> None:
        self.size = size

    def get(self) -> torch.Tensor:
        return torch.ones(self.size, dtype=torch.uint8)

    def echo(self, value: Any) -> Any:
        return value


class _Stubborn:
    """A concurrent-mode object whose wait ignores every cancel (as a misbehaving handler might) and counts them."""

    def __init__(self) -> None:
        self.started = self.cancels = 0

    async def wait(self) -> None:
        self.started += 1
        while True:
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                self.cancels += 1


class _Unpicklable:
    def __reduce__(self) -> tuple[Any, ...]:
        raise TypeError("this reply does not pickle")


class _BadReplies:
    """Returns replies, and raises errors, that fail to encode."""

    def bad(self) -> Any:
        return _Unpicklable()

    def many(self) -> list[torch.Tensor]:
        return [torch.ones(1) for _ in range(3)]

    def exit_on_reduce(self) -> None:
        raise _ExitOnReduce("bad")

    def base_exit_on_reduce(self) -> None:
        raise _BaseExitOnReduce("bad")

    def local_text(self) -> None:
        class Text(str):
            """Defined here, so pickle cannot find it to send the error."""

        raise _TextError(Text("local text"))

    def echo(self, value: Any) -> Any:
        return value


def _explode() -> None:
    raise ValueError("this request does not unpickle")


class _Boom:
    """Pickles, but raises when unpickled."""

    def __reduce__(self) -> tuple[Any, ...]:
        return _explode, ()


class _ExitOnReduce(Exception):
    """An error whose pickling raises a ``BaseException``, past every codec fallback."""

    def __reduce__(self) -> tuple[Any, ...]:
        raise SystemExit("from __reduce__")


class _BaseExitOnReduce(BaseException):
    """A ``BaseException`` whose pickling raises one too."""

    def __reduce__(self) -> tuple[Any, ...]:
        raise SystemExit("from __reduce__")


class _TextError(Exception):
    """An error whose ``str()`` returns its argument as it is, which may be a str subclass."""

    def __str__(self) -> str:
        return self.args[0]


class _Children:
    """Child Python processes, each in a process group of its own, killed when the test ends."""

    def __init__(self) -> None:
        self.processes: list[subprocess.Popen] = []

    def spawn(self, script: str, *args: str) -> subprocess.Popen:
        child = subprocess.Popen(
            [sys.executable, "-c", script, _ROOT, *args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        self.processes.append(child)
        return child

    def kill_all(self) -> None:
        """Kill and reap every child, though killing or reaping one raises, then raise the first such error."""
        errors: list[BaseException] = []
        for child in self.processes:
            try:
                _kill_group(child)
                child.communicate(timeout=30)
            except BaseException as e:
                errors.append(e)
        if errors:
            raise errors[0]


def _kill_group(child: subprocess.Popen) -> None:
    """SIGKILL a child and the processes it forked, while its unreaped pid still holds the group id."""
    if child.returncode is None:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _read_line(child: subprocess.Popen, timeout_s: float = 60) -> str:
    """Read the child's next line, failing with its stderr if none comes in time."""
    stdout = child.stdout
    assert stdout is not None
    ready, _, _ = select.select([stdout], [], [], timeout_s)
    line = stdout.readline() if ready else ""
    if not line:
        _kill_group(child)
        pytest.fail(f"child reported nothing: {child.communicate(timeout=30)[1]}")
    return line.strip()


def _in_thread(coroutine: Coroutine[Any, Any, Any]) -> None:
    """Run a coroutine to completion on a thread of its own, outside any test's loop."""
    errors: list[BaseException] = []

    def run() -> None:
        try:
            asyncio.run(coroutine)
        except BaseException as e:
            errors.append(e)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(60)
    assert not thread.is_alive() and not errors, errors


async def _until(condition: Callable[[], bool], timeout_s: float = 10) -> None:
    deadline = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline, "the condition did not hold in time"
        await asyncio.sleep(0.01)


async def _until_child(
    child: subprocess.Popen, condition: Callable[[], bool], timeout_s: float = 60
) -> None:
    """Wait for a child's calls to land, failing with its stderr once it exits or time runs out."""
    deadline = time.monotonic() + timeout_s
    while not condition():
        status = child.poll()
        if status is not None or time.monotonic() >= deadline:
            _kill_group(child)
            stderr = child.communicate(timeout=30)[1]
            pytest.fail(
                f"the child's calls did not land (exit status {status}): {stderr}"
            )
        await asyncio.sleep(0.01)


def _stopped(child: subprocess.Popen) -> bool:
    """Whether every thread of the child has stopped. A stop signal takes effect only later."""
    flags = os.WSTOPPED | os.WNOHANG | os.WNOWAIT
    return os.waitid(os.P_PID, child.pid, flags) is not None


def _counts(server: CarrierServer) -> tuple[int, int, int]:
    """The server's arrived, running and queued call counts, read in one snapshot."""
    stats = server.stats()
    return stats["calls"], stats["running"], stats["queued"]


def _scope() -> str:
    return f"test/{uuid.uuid4().hex}"


async def _serve(target: Any, *endpoints: str, **kwargs: Any) -> CarrierServer:
    return await CarrierServer.serve(
        target, endpoints or _ENDPOINTS, service="test", scope=_scope(), **kwargs
    )


def _sending(client: CarrierClient) -> bool:
    """Whether the client's connection is partway through sending a frame."""
    conn = client._peer.conn
    return conn is not None and conn.sending is not None


def _pending(client: CarrierClient) -> bool:
    """Whether the client's connection holds a call that has no reply yet."""
    conn = client._peer.conn
    return conn is not None and bool(conn.pending)


def _unsent(client: CarrierClient) -> bool:
    """Whether the client's connection holds a call it has not started to send."""
    conn = client._peer.conn
    return conn is not None and any(
        isinstance(item, runtime._Call) for item in list(conn.outbox)
    )


async def _close(server: CarrierServer, grace_s: float, cancelled: bool) -> None:
    """Close the server, after cancelling a first ``close`` once it has started if ``cancelled``."""
    if cancelled:
        first = asyncio.ensure_future(server.close(grace_s=grace_s))
        await _until(lambda: server._closing)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
    await server.close(grace_s=grace_s)


def _raw_connection(address: CarrierAddress) -> socket.socket:
    return socket.create_connection((address.host, address.port), timeout=10)


def _send_frame(
    sock: socket.socket, kind: Kind, request_id: int = 0, **kwargs: Any
) -> None:
    chunks = codec.encode_frame(kind, request_id, **kwargs)
    sock.sendall(b"".join(bytes(chunk) for chunk in chunks))


def _hello(
    address: CarrierAddress, record: type[Hello] = Hello, **changes: Any
) -> bytes:
    fields: dict[str, Any] = dict(
        scope=address.scope,
        service=address.service,
        incarnation=address.incarnation,
        peer="raw:0",
    )
    return record(**{**fields, **changes}).to_json().encode()


def _closed(server: CarrierServer) -> str:
    """Match the ``PeerLostError`` text of a call ``server`` failed as it closed."""
    a = server.address
    text = f"volume closed: carrier {a.service} at {a.host}:{a.port}"
    return f"^{re.escape(text)}$"


def _exiting(server: CarrierServer) -> str:
    """Match the ``PeerLostError`` text of a call that exit kept from starting on ``server``."""
    a = server.address
    return (
        f"^{re.escape(f'process exiting: carrier {a.service} at {a.host}:{a.port}')}$"
    )


def _no_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let carrier clients fill a FIFO server's queue, as otherwise only raw clients can."""
    monkeypatch.setattr(runtime.CarrierServer, "_window", lambda self: 0)


def _unsent_bytes(sock: socket.socket) -> int:
    """Bytes in a socket's send queue that TCP has not sent yet (Linux's SIOCOUTQNSD)."""
    return struct.unpack("i", fcntl.ioctl(sock.fileno(), 0x894B, struct.pack("i", 0)))[
        0
    ]


def _unacknowledged_bytes(sock: socket.socket) -> int:
    """Bytes in a socket's send queue that its peer has not acknowledged yet (Linux's SIOCOUTQ)."""
    data = fcntl.ioctl(sock.fileno(), termios.TIOCOUTQ, struct.pack("i", 0))
    return struct.unpack("i", data)[0]


def _receive_into(sock: socket.socket, size: int) -> bytearray:
    data = bytearray(size)
    view = memoryview(data)
    got = 0
    while got < size:
        count = sock.recv_into(view[got:])
        assert count, "the connection closed"
        got += count
    return data


def _receive(sock: socket.socket, size: int) -> bytes:
    return bytes(_receive_into(sock, size))


def _answers(sock: socket.socket, count: int) -> dict[int, Kind]:
    """Read ``count`` answers with their buffers, returning each one's kind by request id."""
    answers = {}
    for _ in range(count):
        header = codec.decode_header(_receive(sock, codec.HEADER_SIZE), handshake=False)
        size = codec.LENGTH_SIZE * header.buffer_count
        lengths = codec.decode_lengths(header, _receive(sock, size))
        _receive_into(sock, header.body_len + sum(lengths))
        answers[header.request_id] = header.kind
    return answers


def _handshake_reply(sock: socket.socket) -> tuple[Kind, bytes]:
    header = codec.decode_header(_receive(sock, codec.HEADER_SIZE), handshake=True)
    return header.kind, _receive(sock, header.body_len)


def _closed_by_peer(sock: socket.socket) -> bool:
    """Read until the server closes the connection. False if the socket's timeout passes first."""
    try:
        while sock.recv(1 << 16):
            pass
    except ConnectionResetError:
        pass
    except TimeoutError:
        return False
    return True


def _hold_large_send(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[threading.Event, threading.Event, list[str]]:
    """Hold the I/O thread inside the next send of 1 MiB or more until released. Return (held, release, outcome).

    ``outcome`` then records whether that send went out or which error it raised.
    """
    held, release = threading.Event(), threading.Event()
    outcome: list[str] = []
    sendall = runtime._sendall

    async def held_sendall(
        loop: asyncio.AbstractEventLoop, sock: socket.socket, data: Any
    ) -> None:
        if len(data) < 1 << 20 or held.is_set():
            return await sendall(loop, sock, data)
        held.set()
        release.wait(30)  # The stuck I/O thread under test.
        try:
            await sendall(loop, sock, data)
        except OSError as e:
            outcome.append(type(e).__name__)
            raise
        outcome.append("sent")

    monkeypatch.setattr(runtime, "_sendall", held_sendall)
    return held, release, outcome


class _FreedOn:
    """Records the name of the thread that frees it."""

    def __init__(self, freed: list[str]) -> None:
        self.freed = freed

    def __del__(self) -> None:
        self.freed.append(threading.current_thread().name)


class _TakesAll:
    """A socket stand-in that takes every send whole, as one whose reader keeps up, and counts them."""

    def __init__(self) -> None:
        self.sends: list[int] = []

    def send(self, data: Any) -> int:
        self.sends.append(len(data))
        return len(data)


def _task_errors(caplog: pytest.LogCaptureFixture, coroutine: str) -> list[str]:
    """Errors asyncio logged about tasks running ``coroutine``, destroyed while pending or with an exception unretrieved."""
    gc.collect()
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == "asyncio" and coroutine in r.getMessage()
    ]


@pytest.fixture
def fresh_runtime(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Give the test a runtime and exit state of its own, so an exit hook it runs stops only its threads."""
    for name, value in (
        ("_runtime", None),
        ("_exiting", False),
        ("_loop_threads", set()),
        ("_close_threads", weakref.WeakSet()),
        ("_freeing", set()),
        ("_parked", []),
        ("_stopped", weakref.WeakSet()),
        ("_runs", {}),
        ("_handles", weakref.WeakSet()),
        ("_servers", weakref.WeakSet()),
    ):
        monkeypatch.setattr(runtime, name, value)
    yield
    # Then no thread of that runtime outlives the test.
    _in_thread(close_all(grace_s=0))
    for loop_thread in list(runtime._loop_threads):
        runtime._stop_loop(loop_thread.loop)
        loop_thread.thread.join(10)


@pytest.fixture(autouse=True)
def _close_carriers() -> Iterator[None]:
    yield
    _in_thread(close_all(grace_s=0))


@pytest.fixture
def target() -> Iterator[_Target]:
    target = _Target()
    yield target
    target.release.set()


@pytest.fixture
def children() -> Iterator[_Children]:
    children = _Children()
    yield children
    children.kill_all()


@pytest.mark.asyncio
async def test_calls_return_results_and_raise_remote_errors(target: _Target) -> None:
    """Return tensors, containers and bytes. Re-raise remote errors with their type and text."""
    server = await _serve(target)
    client = CarrierClient(server.address)
    tensor = torch.arange(6.0).reshape(2, 3)
    reply = await client.call("echo", {"t": tensor, "b": b"\x00\xff", "n": None})
    assert torch.equal(reply["t"], tensor) and reply["b"] == b"\x00\xff"
    assert await client.call("echo", value=[1, 2]) == [1, 2]
    with pytest.raises(KeyError, match="'missing'"):
        await client.call("fail", KeyError("missing"))
    with pytest.raises(AttributeError, match="no endpoint 'nope'"):
        await client.call("nope")
    with pytest.raises(TypeError, match="cannot be pickled"):
        pickle.dumps(client)


@pytest.mark.asyncio
async def test_a_handlers_own_carrier_errors_arrive_as_remote_errors(
    target: _Target,
) -> None:
    """Raise a handler's own carrier error as ``RemoteError`` naming its type, so a ``PeerLostError`` is always the call's own.

    A ``RemoteError`` the handler raises arrives as it is.
    """
    server = await _serve(target)
    client = CarrierClient(server.address)
    for error in (
        PeerLostError("volume closed: carrier v at h:1"),
        CallTimeoutError("late"),
    ):
        with pytest.raises(RemoteError) as raised:
            await client.call("fail", error)
        assert raised.value.remote_type == codec._type_name(type(error))
        assert raised.value.args[1] == str(error)
    with pytest.raises(RemoteError) as raised:
        await client.call("fail", RemoteError("KeyError", "'k'"))
    assert raised.value.args == ("KeyError", "'k'")
    assert await client.call("echo", 1) == 1


@pytest.mark.asyncio
async def test_a_handlers_endpoint_keyword_arrives_while_timeout_s_stays_the_calls(
    target: _Target,
) -> None:
    """Pass a handler its own ``endpoint`` keyword, as the endpoint's name goes by position, but never ``timeout_s``."""
    server = await _serve(target, "where")
    client = CarrierClient(server.address)
    assert await client.call("where", endpoint="mine") == ("mine", None)
    assert await client.call("where", "mine", 7) == ("mine", 7)
    assert await client.call("where", timeout_s=7) == ("default", None)


@pytest.mark.asyncio
async def test_empty_tensors_round_trip_through_a_server(target: _Target) -> None:
    """Return empty tensors, whose reply buffers hold no bytes to copy."""
    server = await _serve(target)
    client = CarrierClient(server.address)
    for empty in (torch.empty(0), torch.zeros(0, 3, dtype=torch.bfloat16)):
        reply = await client.call("echo", empty)
        assert reply.shape == empty.shape and reply.dtype == empty.dtype


@pytest.mark.asyncio
@pytest.mark.parametrize("concurrent", [False, True])
async def test_a_received_pickle_buffer_can_be_sent_back(concurrent: bool) -> None:
    """Echo a ``PickleBuffer`` over ``bytes``, which arrives as a read-only ``memoryview``, and echo that."""
    payload = bytes(range(256)) * 1024
    server = await _serve(_Target(), "echo", concurrent=concurrent)
    client = CarrierClient(server.address)
    reply = await client.call("echo", pickle.PickleBuffer(payload))
    assert isinstance(reply, memoryview) and reply.readonly
    assert bytes(reply) == payload
    again = await client.call("echo", {"reply": reply})
    assert isinstance(again["reply"], memoryview) and bytes(again["reply"]) == payload


@pytest.mark.asyncio
async def test_a_remote_stop_iteration_arrives_as_a_remote_error(
    target: _Target,
) -> None:
    """Raise a handler's ``StopIteration`` as ``RemoteError``, not the ``RuntimeError`` a coroutine would make."""
    server = await _serve(target, "fail", "first", "echo")
    client = CarrierClient(server.address)
    for call, text in (
        (client.call("fail", StopIteration("no more")), "StopIteration: no more"),
        (client.call("first", []), "StopIteration"),
    ):
        with pytest.raises(RemoteError) as raised:
            await call
        assert raised.value.remote_type == "StopIteration" and str(raised.value) == text
    assert await client.call("echo", 1) == 1


@pytest.mark.asyncio
async def test_a_handler_base_exception_loses_only_its_call(target: _Target) -> None:
    """Give a call whose handler raises a BaseException a ``RemoteError`` naming its type, and serve on."""
    server = await _serve(target)
    client = CarrierClient(server.address)
    calls = {
        "SystemExit": client.call("exit"),
        codec._type_name(asyncio.CancelledError): client.call(
            "fail", asyncio.CancelledError()
        ),
    }
    for remote_type, call in calls.items():
        with pytest.raises(RemoteError) as raised:
            await call
        assert raised.value.remote_type == remote_type
    assert await client.call("echo", 1) == 1


@pytest.mark.asyncio
async def test_an_error_that_cannot_be_encoded_fails_only_its_call(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Answer a call whose error, or ``BaseException``, raises one as it pickles with a ``RemoteError`` naming its type.

    So no live server's answer reads as a closed one. One whose text is a str subclass arrives as text.
    """
    caplog.set_level(logging.ERROR, logger="torchstore.carrier")
    endpoints = ("exit_on_reduce", "base_exit_on_reduce", "local_text", "echo")
    server = await _serve(_BadReplies(), *endpoints)
    client = CarrierClient(server.address)
    for endpoint, error in (
        ("exit_on_reduce", _ExitOnReduce),
        ("base_exit_on_reduce", _BaseExitOnReduce),
    ):
        with pytest.raises(RemoteError) as raised:
            await client.call(endpoint, timeout_s=10)
        assert raised.value.remote_type == codec._type_name(error)
        assert raised.value.args[1] == "its encoding raised SystemExit"
    failure = "SystemExit('from __reduce__')"
    assert f"cannot encode the _ExitOnReduce a call raised: {failure}" in caplog.text
    with pytest.raises(RemoteError, match="_TextError: local text"):
        await client.call("local_text", timeout_s=10)
    assert await client.call("echo", 1, timeout_s=10) == 1
    assert _counts(server)[1:] == (0, 0)


@pytest.mark.asyncio
async def test_a_reply_that_fails_to_encode_fails_only_its_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail a call whose reply does not pickle, or has too many buffers, at once, and keep serving."""
    server = await _serve(_BadReplies(), "bad", "many", "echo")
    client = CarrierClient(server.address)
    with pytest.raises(TypeError, match="does not pickle"):
        await client.call("bad", timeout_s=10)
    assert await client.call("echo", 1, timeout_s=10) == 1
    monkeypatch.setattr(codec, "MAX_BUFFERS", 2)
    with pytest.raises(CarrierProtocolError, match="over the limit of 2"):
        await client.call("many", timeout_s=10)
    assert await client.call("echo", 2, timeout_s=10) == 2


@pytest.mark.asyncio
async def test_a_request_that_fails_to_unpickle_fails_only_its_call(
    target: _Target,
) -> None:
    """Fail a call whose request does not unpickle on the server at once, and keep serving."""
    server = await _serve(target)
    client = CarrierClient(server.address)
    with pytest.raises(ValueError, match="does not unpickle"):
        await client.call("echo", _Boom(), timeout_s=10)
    assert await client.call("echo", 1, timeout_s=10) == 1


@pytest.mark.asyncio
async def test_a_call_over_the_frame_limits_fails_alone(
    monkeypatch: pytest.MonkeyPatch, target: _Target
) -> None:
    """Fail a call whose frame has too many buffers, and keep its connection for the next."""
    server = await _serve(target)
    client = CarrierClient(server.address)
    assert await client.call("echo", 1) == 1
    monkeypatch.setattr(codec, "MAX_BUFFERS", 2)
    with pytest.raises(CarrierProtocolError, match="over the limit of 2"):
        await client.call("echo", [torch.ones(1) for _ in range(3)], timeout_s=10)
    assert await client.call("echo", 2) == 2
    assert target.calls == [1, 2] and server.stats()["connections"] == 1


@pytest.mark.asyncio
async def test_running_and_queued_calls_time_out(target: _Target) -> None:
    """Time out a running and a queued call. The queued one never runs and the connection stays."""
    server = await _serve(target)
    client = CarrierClient(server.address)
    start = time.monotonic()
    calls = [
        client.call("hold", "running", timeout_s=1),
        client.call("echo", "queued", timeout_s=1),
    ]
    for outcome in await asyncio.gather(*calls, return_exceptions=True):
        assert isinstance(outcome, CallTimeoutError), outcome
    assert 1 <= time.monotonic() - start < 6
    await _until(lambda: server.stats()["cancels"] == 2)
    with server._lock:
        [request] = server._running
    # The server's deadline trails the caller's by the frame's transit time.
    await _until(lambda: time.monotonic() > request.deadline)
    target.release.set()
    assert await client.call("echo", "after") == "after"
    assert target.calls == ["running", "after"]
    stats = server.stats()
    assert (stats["connections"], stats["cancels"], stats["timeouts"]) == (1, 2, 1)


@pytest.mark.asyncio
async def test_an_expired_request_never_runs(
    monkeypatch: pytest.MonkeyPatch, target: _Target
) -> None:
    """Skip a queued request whose deadline passed, even when its caller's cancel never comes."""
    monkeypatch.setattr(runtime._ClientConnection, "abandoned", lambda self, call: None)
    server = await _serve(target)
    client = CarrierClient(server.address)
    running = asyncio.ensure_future(client.call("hold", "running"))
    await _until(lambda: server.stats()["running"] == 1)
    expired = asyncio.ensure_future(client.call("echo", "expired", timeout_s=0.5))
    await _until(lambda: server.stats()["queued"] == 1)
    with server._lock:
        [request] = server._queue
    with pytest.raises(CallTimeoutError, match="within its 0.5 s deadline$"):
        await expired
    # The server's deadline trails the caller's by the frame's transit time.
    await _until(lambda: time.monotonic() >= request.deadline)
    target.release.set()
    assert await running == "running"
    assert await client.call("echo", "after") == "after"
    assert target.calls == ["running", "after"]
    assert server.stats()["expired"] == 1
    # The server answered the expired request too, freeing its place in the window.
    conn = client._peer.conn
    assert conn is not None and conn.unanswered == 0


@pytest.mark.asyncio
async def test_a_call_queued_behind_an_expired_one_still_runs(
    monkeypatch: pytest.MonkeyPatch, target: _Target
) -> None:
    """Run a call queued behind an expired request, though no later call arrives to dispatch it."""
    monkeypatch.setattr(runtime._ClientConnection, "abandoned", lambda self, call: None)
    # An async handler, as a volume's are, leaves the dispatcher loop free meanwhile.
    server = await _serve(target, "wait", "echo")
    client = CarrierClient(server.address)
    running = asyncio.ensure_future(client.call("wait", "running"))
    await _until(lambda: server.stats()["running"] == 1)
    expired = asyncio.ensure_future(client.call("echo", "expired", timeout_s=0.5))
    await _until(lambda: server.stats()["queued"] == 1)
    behind = asyncio.ensure_future(client.call("echo", "behind", timeout_s=30))
    await _until(lambda: server.stats()["queued"] == 2)
    with server._lock:
        request = server._queue[0]
    with pytest.raises(CallTimeoutError, match="within its 0.5 s deadline$"):
        await expired
    await _until(lambda: time.monotonic() >= request.deadline)
    target.release.set()
    assert await running == "running"
    assert await asyncio.wait_for(behind, 10) == "behind"
    assert target.calls == ["running", "behind"] and server.stats()["expired"] == 1
    conn = client._peer.conn
    assert conn is not None and conn.unanswered == 0


@pytest.mark.asyncio
async def test_a_call_expiring_before_it_is_sent_names_its_deadline(
    monkeypatch: pytest.MonkeyPatch, target: _Target
) -> None:
    """Name the endpoint, server and deadline when the I/O thread times out a call it never sent."""
    server = await _serve(target)
    client = CarrierClient(server.address)
    assert await client.call("echo", 0) == 0
    settled = threading.Event()
    settle = runtime._Call.settle

    def spy(call: runtime._Call, *args: Any, **kwargs: Any) -> None:
        settle(call, *args, **kwargs)
        settled.set()

    monkeypatch.setattr(runtime._Call, "settle", spy)
    stuck = threading.Event()
    client._peer.runtime.loop.call_soon_threadsafe(stuck.wait)
    call = asyncio.ensure_future(client.call("echo", 1, timeout_s=1))
    release = threading.Timer(
        1.5, stuck.set
    )  # Past the deadline: the pause under test.
    try:
        # The call has queued its submit behind the held I/O loop.
        await asyncio.sleep(0)
        release.start()
        # This loop is late: it runs again only once the I/O thread timed the call out.
        assert settled.wait(30)
    finally:
        stuck.set()
        release.cancel()
    where = re.escape(f"{server.address.host}:{server.address.port}")
    with pytest.raises(
        CallTimeoutError,
        match=f"^call 'echo' to carrier test at {where} did not finish within its 1 s deadline$",
    ):
        await call


@pytest.mark.asyncio
async def test_a_request_left_unread_past_its_deadline_never_runs(
    target: _Target,
) -> None:
    """Time a request from its arrival in the kernel, so one a stopped reader left unread past its deadline never runs.

    Timed from its header's read, it got its whole deadline again once the reader went on.
    """
    server = await _serve(target)
    plug, _ = codec.dumps(("echo", (b"plug" * 1024,), {}))
    stale, _ = codec.dumps(("echo", ("stale",), {}))
    with _raw_connection(server.address) as sock:
        _send_frame(sock, Kind.HELLO, body=_hello(server.address))
        assert _handshake_reply(sock)[0] == Kind.WELCOME
        [conn] = server._connections
        received = conn.reader.received
        await server._set_reading(False)
        # Its reader, parked past the gate, takes one receive of the plug, so its next receive waits at the gate and
        # the stale request stays unread in the kernel.
        _send_frame(sock, Kind.REQUEST, 1, deadline_us=60_000_000, body=plug)
        await _until(lambda: conn.reader.received > received)
        sent = time.monotonic()
        _send_frame(sock, Kind.REQUEST, 2, deadline_us=500_000, body=stale)
        await _until(lambda: time.monotonic() > sent + 0.6 and _unsent_bytes(sock) == 0)
        await server._set_reading(True)
        answers = {}
        for _ in range(2):
            header = codec.decode_header(
                _receive(sock, codec.HEADER_SIZE), handshake=False
            )
            answers[header.request_id] = (header.kind, _receive(sock, header.body_len))
    assert answers[1][0] == Kind.REPLY
    kind, body = answers[2]
    assert kind == Kind.ERROR and type(codec.decode_error(body)) is CallTimeoutError
    assert target.calls == [b"plug" * 1024] and server.stats()["expired"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case", ["first-byte-alone", "after-idle", "first-byte-with-a-frame"]
)
async def test_a_request_is_timed_from_the_receive_that_took_its_first_byte(
    target: _Target, case: str
) -> None:
    """Time a request from the receive that took its header's first byte and from neither a later nor an earlier one.

    Sending its first byte alone or with the frame before it and the rest after its 1s deadline expires it. Timed from
    the later receive it got that deadline again. Sent whole once the connection idled past it, it runs.
    """
    server = await _serve(target)
    frames = [
        b"".join(
            bytes(chunk)
            for chunk in codec.encode_frame(
                Kind.REQUEST, request_id, deadline_us=deadline_us, body=body
            )
        )
        for request_id, deadline_us, body in (
            (1, 60_000_000, codec.dumps(("echo", ("earlier",), {}))[0]),
            (2, 1_000_000, codec.dumps(("echo", ("late",), {}))[0]),
        )
    ]
    earlier, frame = frames
    with _raw_connection(server.address) as sock:
        _send_frame(sock, Kind.HELLO, body=_hello(server.address))
        assert _handshake_reply(sock)[0] == Kind.WELCOME
        [conn] = server._connections
        received = conn.reader.received
        sent = time.monotonic()
        if case == "after-idle":
            await _until(lambda: time.monotonic() > sent + 1.1)
            sock.sendall(frame)
        else:
            head = (earlier if case == "first-byte-with-a-frame" else b"") + frame[:1]
            sock.sendall(head)
            await _until(lambda: conn.reader.received == received + len(head))
            await _until(lambda: time.monotonic() > sent + 1.1)
            sock.sendall(frame[1:])
        if case == "first-byte-with-a-frame":
            assert _raw_reply(sock) == (1, "earlier")
        if case == "after-idle":
            assert _raw_reply(sock) == (2, "late")
        else:
            header = codec.decode_header(
                _receive(sock, codec.HEADER_SIZE), handshake=False
            )
            assert header.kind == Kind.ERROR
            assert (
                type(codec.decode_error(_receive(sock, header.body_len)))
                is CallTimeoutError
            )
    ran = {
        "first-byte-alone": [],
        "after-idle": ["late"],
        "first-byte-with-a-frame": ["earlier"],
    }
    assert target.calls == ran[case]
    assert server.stats()["expired"] == (case != "after-idle")


@pytest.mark.asyncio
async def test_a_call_whose_cancel_arrived_with_it_never_runs(
    monkeypatch: pytest.MonkeyPatch, target: _Target
) -> None:
    """Drop a call whose CANCEL arrived with it behind a frame of many buffers, and free its parts with the cancel.

    The reader holds the call until it has read the bytes there when it read the call, the kernel's included, so neither
    the running call's end nor a turn of the reader's dispatches it first.
    """
    received = _record_received(monkeypatch)
    enqueue, cancel = runtime.CarrierServer._enqueue, runtime.CarrierServer._cancel
    held, proceed = threading.Event(), threading.Event()
    freed: list[bool] = []

    def recorded_enqueue(self: Any, conn: Any, header: Any, *args: Any) -> None:
        enqueue(self, conn, header, *args)
        if header.request_id == 2:
            # The I/O thread waits here, the call held, while the running call ends.
            held.set()
            assert proceed.wait(30)

    def recorded_cancel(self: Any, conn: Any, request_id: int, size: int) -> None:
        cancel(self, conn, request_id, size)
        # The cancelled call's tensor, the first one received, is freed with the cancel.
        freed.append(received[0]() is None)

    monkeypatch.setattr(runtime.CarrierServer, "_enqueue", recorded_enqueue)
    monkeypatch.setattr(runtime.CarrierServer, "_cancel", recorded_cancel)
    server = await _serve(target)
    # Accepted sockets take this receive buffer, which holds every frame while the I/O loop is held. Linux's
    # default of 128 KiB does not.
    server._listener.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 << 20)
    hold, _ = codec.dumps(("hold", ("held",), {}))
    body, buffers = codec.dumps(("echo", (torch.ones(4),), {}))
    many, many_buffers = codec.dumps(
        ("echo", ([torch.ones(1) for _ in range(8 * runtime._BUFFER_TURN)],), {})
    )
    frames = [
        *codec.encode_frame(
            Kind.REQUEST, 2, deadline_us=60_000_000, body=body, buffers=buffers
        ),
        *codec.encode_frame(
            Kind.REQUEST, 3, deadline_us=60_000_000, body=many, buffers=many_buffers
        ),
        *codec.encode_frame(Kind.CANCEL, 2),
    ]
    stuck = threading.Event()
    with _raw_connection(server.address) as sock:
        _send_frame(sock, Kind.HELLO, body=_hello(server.address))
        assert _handshake_reply(sock)[0] == Kind.WELCOME
        _send_frame(sock, Kind.REQUEST, 1, deadline_us=60_000_000, body=hold)
        await _until(lambda: server.stats()["running"] == 1)
        server._runtime.loop.call_soon_threadsafe(stuck.wait)
        try:
            sock.sendall(b"".join(bytes(chunk) for chunk in frames))
            # Every byte is in the server's kernel before its reader reads the first.
            await _until(lambda: _unacknowledged_bytes(sock) == 0)
            stuck.set()
            await _until(held.is_set)
            # The running call ends, and its dispatcher looks for the next.
            target.release.set()
            await _until(lambda: server.stats()["running"] == 0)
        finally:
            stuck.set()
            proceed.set()
        answers = _answers(sock, 3)
    assert answers == {1: Kind.REPLY, 2: Kind.ERROR, 3: Kind.REPLY} and freed == [True]
    assert len(target.calls) == 2 and server.stats()["cancels"] == 1


@pytest.mark.asyncio
async def test_a_held_call_is_released_at_a_turn_once_what_came_with_it_is_read(
    monkeypatch: pytest.MonkeyPatch, target: _Target
) -> None:
    """Release a held call at the reader's next turn once it has read what arrived with the call.

    So a call never waits for the end of a long frame that its client sent after it.
    """
    enqueue, unhold = runtime.CarrierServer._enqueue, runtime.CarrierServer._unhold
    enqueued: list[int] = []
    released: list[list[int]] = []
    first, proceed = threading.Event(), threading.Event()

    def recorded_enqueue(self: Any, conn: Any, header: Any, *args: Any) -> None:
        enqueue(self, conn, header, *args)
        enqueued.append(header.request_id)
        if header.request_id == 1:
            # The I/O thread waits here, having held the call, until the long frame is all here.
            first.set()
            assert proceed.wait(30)

    def recorded_unhold(self: Any, conn: Any) -> None:
        released.append(list(enqueued))
        unhold(self, conn)

    monkeypatch.setattr(runtime.CarrierServer, "_enqueue", recorded_enqueue)
    monkeypatch.setattr(runtime.CarrierServer, "_unhold", recorded_unhold)
    server = await _serve(target)
    # Accepted sockets take this receive buffer, which holds the long frame while the I/O thread waits. Some kernels,
    # such as those of CI's runners, give a default buffer too small a window for it.
    server._listener.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 << 20)
    call, _ = codec.dumps(("echo", ("first",), {}))
    middle = [torch.ones(1) for _ in range(4 * runtime._BUFFER_TURN)]
    body, buffers = codec.dumps(("echo", (middle,), {}))
    long_frame = codec.encode_frame(
        Kind.REQUEST, 2, deadline_us=60_000_000, body=body, buffers=buffers
    )
    with _raw_connection(server.address) as sock:
        _send_frame(sock, Kind.HELLO, body=_hello(server.address))
        assert _handshake_reply(sock)[0] == Kind.WELCOME
        try:
            _send_frame(sock, Kind.REQUEST, 1, deadline_us=60_000_000, body=call)
            await _until(first.is_set)
            sock.sendall(b"".join(bytes(chunk) for chunk in long_frame))
            await _until(lambda: _unacknowledged_bytes(sock) == 0)
        finally:
            proceed.set()
        answers = _answers(sock, 2)
    assert answers == {1: Kind.REPLY, 2: Kind.REPLY}
    # Released at the long frame's first turn, before the reader had read all of it.
    assert released[0] == [1], released


_unpickled: list[str] = []


def _record_unpickle(label: str) -> str:
    _unpickled.append(label)
    return label


class _RecordsItsUnpickle:
    """Unpickles through a global that records its label. Cases can thus tell whether a server unpickled its request."""

    def __init__(self, label: str) -> None:
        self.label = label

    def __reduce__(self) -> tuple[Any, ...]:
        return _record_unpickle, (self.label,)


@pytest.fixture
def unpickled(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The labels of each ``_RecordsItsUnpickle`` unpickled while the test runs."""
    labels: list[str] = []
    monkeypatch.setattr(sys.modules[__name__], "_unpickled", labels)
    return labels


@pytest.mark.asyncio
async def test_a_call_abandoned_after_its_dispatch_never_runs(
    monkeypatch: pytest.MonkeyPatch, target: _Target, unpickled: list[str]
) -> None:
    """Skip the handler of a call cancelled after its dispatch but before its task ran.

    Its request is never unpickled and the skip counts no handler error.
    """
    server = await _serve(target)
    client = CarrierClient(server.address)
    assert await client.call("echo", "first") == "first"
    dispatch = server._dispatch
    dispatched, release = threading.Event(), threading.Event()

    def held_dispatch() -> None:
        dispatch()
        # Holds the dispatcher's thread before the task it just made takes its first step. The
        # first call's own last dispatch, which may come after its reply, starts nothing.
        with server._lock:
            started = bool(server._running)
        if started and not dispatched.is_set():
            dispatched.set()
            release.wait(30)

    monkeypatch.setattr(server, "_dispatch", held_dispatch)
    # A deadline that cannot pass here: only the cancel can skip the handler.
    call = asyncio.ensure_future(
        client.call("echo", _RecordsItsUnpickle("late"), timeout_s=60)
    )
    try:
        await _until(dispatched.is_set)
        call.cancel()
        await _until(lambda: server.stats()["cancels"] == 1)
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await call
    # Not undo(), which would also restore the record of unpickles before the late call ran.
    monkeypatch.setattr(server, "_dispatch", dispatch)
    # The FIFO server runs this call only after the late one finished.
    assert await client.call("echo", "after", timeout_s=10) == "after"
    assert target.calls == ["first", "after"] and unpickled == []
    assert server.stats()["errors"] == 0


@pytest.mark.asyncio
async def test_a_call_whose_deadline_passed_after_its_dispatch_never_runs(
    monkeypatch: pytest.MonkeyPatch, target: _Target, unpickled: list[str]
) -> None:
    """Answer with an expiry a dispatched call whose deadline passed before its task's first step.

    Its handler never runs, its request is never unpickled and the expiry counts no handler error.
    """
    server = await _serve(target)
    dispatch = server._dispatch

    def dispatch_then_expire() -> None:
        dispatch()
        # Its deadline passes between the dispatch and the task's first step.
        with server._lock:
            for request in server._running:
                request.deadline = 0.0

    monkeypatch.setattr(server, "_dispatch", dispatch_then_expire)
    with pytest.raises(CallTimeoutError, match="dropped a call whose deadline passed"):
        await CarrierClient(server.address).call(
            "echo", _RecordsItsUnpickle("late"), timeout_s=60
        )
    assert target.calls == [] and unpickled == []
    assert server.stats()["errors"] == 0


@pytest.mark.asyncio
async def test_default_deadline_is_read_at_call_time(
    monkeypatch: pytest.MonkeyPatch, target: _Target
) -> None:
    """Read the default deadline when a call starts, and name it in the timeout."""
    server = await _serve(target)
    client = CarrierClient(server.address)
    monkeypatch.setattr(runtime, "_CALL_TIMEOUT_S", 0.5)
    with pytest.raises(CallTimeoutError, match="'hold' .* within its 0.5 s deadline"):
        await client.call("hold")


@pytest.mark.asyncio
async def test_a_blocked_handler_stalls_only_its_own_server(target: _Target) -> None:
    """Time out a call to a handler blocked in ``Event.wait()`` while another server answers."""
    blocked = await _serve(target)
    other = await _serve(_Target())
    start = time.monotonic()
    call = asyncio.ensure_future(
        CarrierClient(blocked.address).call("hold", timeout_s=0.5)
    )
    await _until(lambda: blocked.stats()["running"] == 1)
    other_client = CarrierClient(other.address)
    assert await other_client.call("echo", "other", timeout_s=10) == "other"
    with pytest.raises(CallTimeoutError, match="within its 0.5 s deadline$"):
        await call
    assert 0.5 <= time.monotonic() - start < 5.5


@pytest.mark.asyncio
async def test_a_stuck_io_thread_does_not_hold_a_call(target: _Target) -> None:
    """Time out on the caller's loop while the I/O thread is stuck."""
    server = await _serve(target)
    client = CarrierClient(server.address)
    stuck = threading.Event()
    client._peer.runtime.loop.call_soon_threadsafe(stuck.wait)
    try:
        start = time.monotonic()
        with pytest.raises(CallTimeoutError, match="within its 2 s deadline$"):
            await client.call("echo", 1, timeout_s=2)
        assert 2 <= time.monotonic() - start < 7
    finally:
        stuck.set()
    assert await client.call("echo", 2) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "how", ["timeout", "cancel", "cancel-twice", "timeout-then-cancel"]
)
async def test_a_stuck_io_thread_holding_a_send_shuts_its_connection_after_the_release_wait(
    monkeypatch: pytest.MonkeyPatch,
    target: _Target,
    caplog: pytest.LogCaptureFixture,
    how: str,
) -> None:
    """Return a call abandoned while a stuck I/O thread holds its send only after the release wait, its connection shut.

    So no byte its caller then overwrites goes out. A second cancel, or one during the wait after a timeout, ends the wait
    no sooner and raises ``CancelledError``. A WARNING names the server.
    """
    caplog.set_level(logging.WARNING, logger="torchstore.carrier")
    monkeypatch.setattr(runtime, "_RELEASE_WAIT_S", 1.0)
    held, release, outcome = _hold_large_send(monkeypatch)
    server = await _serve(target)
    client = CarrierClient(server.address)
    tensor = torch.zeros(4 << 20, dtype=torch.uint8)
    timeout_s = 0.5 if how.startswith("timeout") else 60
    call = asyncio.ensure_future(client.call("echo", tensor, timeout_s=timeout_s))
    try:
        await _until(held.is_set)
        stuck_at = time.monotonic()
        conn = client._peer.conn
        assert conn is not None and conn.sending is not None
        sending = conn.sending
        if how.startswith("cancel"):
            call.cancel()
        if how != "cancel" and how != "timeout":
            # Once its caller waits for the frame: a cancel then ends that wait no sooner.
            await _until(lambda: sending.abandoned)
            call.cancel()
        with pytest.raises(
            CallTimeoutError if how == "timeout" else asyncio.CancelledError
        ):
            await call
        # The send stays held: only the release wait's end lets the call return.
        waited = time.monotonic() - stuck_at
        tensor.fill_(7)
    finally:
        release.set()
    await _until(lambda: bool(outcome))
    assert outcome == ["BrokenPipeError"] and waited >= 1.0, (outcome, waited)
    where = f"{server.address.host}:{server.address.port}"
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert (
        len(warnings) == 1 and "abandoned call" in warnings[0] and where in warnings[0]
    )
    assert not [r for r in caplog.records if r.levelno > logging.WARNING]
    assert target.calls == []


@pytest.mark.asyncio
async def test_a_call_given_up_as_its_send_ends_returns_at_once_and_never_runs(
    monkeypatch: pytest.MonkeyPatch,
    target: _Target,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Return a call cancelled as its frame's send ends once the I/O thread marks it sent, with no WARNING, and cancel it.

    Its whole frame went out, so its CANCEL follows it and the call never runs, and the connection serves on.
    """
    caplog.set_level(logging.WARNING, logger="torchstore.carrier")
    end_send = runtime._Call.end_send
    sent, gave_up = threading.Event(), threading.Event()

    def held_end_send(call: runtime._Call) -> bool:
        if call.buffers:
            # The frame is all sent, and its caller gives up before the I/O thread marks it so.
            sent.set()
            gave_up.wait(30)
        return end_send(call)

    monkeypatch.setattr(runtime._Call, "end_send", held_end_send)
    server = await _serve(target)
    a = server.address
    # On a connection of its own, so the call waits behind it.
    other = CarrierAddress("127.0.0.1", a.port, a.service, a.scope, a.incarnation)
    hold = asyncio.ensure_future(CarrierClient(other).call("hold", "held"))
    await _until(lambda: server.stats()["running"] == 1)
    client = CarrierClient(a)
    call = asyncio.ensure_future(client.call("echo", torch.ones(4), timeout_s=60))
    try:
        await _until(sent.is_set)
        conn = client._peer.conn
        assert conn is not None
        [sending] = conn.pending.values()
        call.cancel()
        await _until(lambda: sending.abandoned)
    finally:
        gave_up.set()
    with pytest.raises(asyncio.CancelledError):
        await call
    await _until(lambda: server.stats()["cancels"] == 1)
    target.release.set()
    assert await hold == "held" and await client.call("echo", "after") == "after"
    assert target.calls == ["held", "after"] and server.stats()["connections"] == 2
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


@pytest.mark.asyncio
async def test_the_cancel_of_a_call_given_up_as_its_send_ends_goes_out_next(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Send the CANCEL of a call given up as its frame's send ends right after that frame, ahead of a call queued behind it.

    So it reaches the server as soon as it can.
    """
    end_send = runtime._Call.end_send
    ended, gave_up, reading = threading.Event(), threading.Event(), threading.Event()

    def held_end_send(call: runtime._Call) -> bool:
        if call.buffers:
            ended.set()
            gave_up.wait(30)
        return end_send(call)

    monkeypatch.setattr(runtime._Call, "end_send", held_end_send)
    frames: list[tuple[Kind, int]] = []
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.settimeout(30)
        port = listener.getsockname()[1]
        address = CarrierAddress("127.0.0.1", port, "test", _scope(), "a" * 16)

        def serve_one() -> None:
            # Records the first three frames, reading none until told to.
            conn, _ = listener.accept()
            with conn:
                conn.settimeout(30)
                _handshake_reply(conn)
                _send_frame(conn, Kind.WELCOME, body=_hello(address, Welcome))
                reading.wait(30)
                while len(frames) < 3:
                    header = codec.decode_header(
                        _receive(conn, codec.HEADER_SIZE), handshake=False
                    )
                    lengths = codec.decode_lengths(
                        header, _receive(conn, codec.LENGTH_SIZE * header.buffer_count)
                    )
                    _receive_into(conn, header.body_len + sum(lengths))
                    frames.append((header.kind, header.request_id))
                _closed_by_peer(conn)

        thread = threading.Thread(target=serve_one, daemon=True)
        thread.start()
        client = CarrierClient(address)
        big = torch.zeros(64 << 20, dtype=torch.uint8)
        first = asyncio.ensure_future(client.call("echo", big, timeout_s=60))
        second = asyncio.ensure_future(client.call("echo", 2, timeout_s=60))
        try:
            await _until(lambda: _sending(client) and _unsent(client))
            conn = client._peer.conn
            assert conn is not None
            [given_up] = [call for call in conn.pending.values() if call.buffers]
            reading.set()
            await _until(ended.is_set)
            first.cancel()
            await _until(lambda: given_up.abandoned)
        finally:
            reading.set()
            gave_up.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        await _until(lambda: len(frames) == 3)
        second.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second
        await client.close()
        thread.join(10)
    assert frames == [(Kind.REQUEST, 1), (Kind.CANCEL, 1), (Kind.REQUEST, 2)]


@pytest.mark.asyncio
async def test_a_cancel_goes_out_past_a_full_window(
    monkeypatch: pytest.MonkeyPatch, target: _Target
) -> None:
    """Send a sent call's CANCEL at once past a full window and ahead of a call it holds back.

    The server drops the cancelled call and the call ahead runs meanwhile.
    """
    monkeypatch.setattr(runtime, "_QUEUE_BOUND", 4)
    server = await _serve(target)
    client = CarrierClient(server.address)
    running = asyncio.ensure_future(client.call("hold", "running"))
    queued = [asyncio.ensure_future(client.call("echo", label)) for label in "abc"]
    await _until(lambda: _counts(server) == (3, 1, 2))
    conn = client._peer.conn
    assert conn is not None and conn.unanswered == conn.window == 3
    # The third call waits in the outbox, where a CANCEL behind it would wait too.
    assert _unsent(client)
    queued[0].cancel()
    with pytest.raises(asyncio.CancelledError):
        await queued[0]
    await _until(lambda: server.stats()["cancels"] == 1)
    target.release.set()
    assert await queued[1] == "b" and await queued[2] == "c"
    assert await running == "running" and target.calls == ["running", "b", "c"]


@pytest.mark.asyncio
async def test_a_call_abandoned_as_the_io_thread_takes_it_is_never_sent(
    monkeypatch: pytest.MonkeyPatch, target: _Target
) -> None:
    """Send nothing for a call its caller gives up on after the I/O thread took it to send, before its send started."""
    server = await _serve(target)
    client = CarrierClient(server.address)
    assert await client.call("echo", "first") == "first"
    taken, abandoned = threading.Event(), threading.Event()
    start_send = runtime._Call.start_send

    def racing(call: runtime._Call) -> bool:
        if not taken.is_set():
            taken.set()
            abandoned.wait(
                30
            )  # The I/O thread loses the CPU here: the pause under test.
        return start_send(call)

    monkeypatch.setattr(runtime._Call, "start_send", racing)
    call = asyncio.ensure_future(client.call("echo", "late"))
    try:
        await _until(taken.is_set)
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call
    finally:
        abandoned.set()
    assert await client.call("echo", "after") == "after"
    assert target.calls == ["first", "after"] and server.stats()["connections"] == 1


@pytest.mark.asyncio
async def test_a_call_abandoned_before_the_io_thread_takes_it_never_connects(
    monkeypatch: pytest.MonkeyPatch, target: _Target
) -> None:
    """Open no connection for a call its caller gave up on before the I/O thread took it."""
    server = await _serve(target)
    client = CarrierClient(server.address)
    stuck, posted = threading.Event(), threading.Event()
    loop = client._peer.runtime.loop
    post = runtime._post_unless_stopped

    def recorded(
        loop: asyncio.AbstractEventLoop, fn: Callable[..., Any], *args: Any
    ) -> bool:
        result = post(loop, fn, *args)
        if getattr(fn, "__name__", "") == "submit":
            posted.set()
        return result

    monkeypatch.setattr(runtime, "_post_unless_stopped", recorded)
    loop.call_soon_threadsafe(stuck.wait, 30)
    try:
        call = asyncio.ensure_future(client.call("echo", 1))
        # Its first step posts it to the stuck I/O thread and waits.
        await _until(posted.is_set)
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call
    finally:
        stuck.set()
    await runtime._call_on(loop, lambda: None)
    peer = client._peer
    assert peer.conn is None and peer.connecting is None and not peer.waiting
    assert server.stats()["connections"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("loop_state", ["stopped", "closed"])
async def test_a_call_on_an_io_loop_that_stopped_or_closed_fails_at_once(
    fresh_runtime: None,
    monkeypatch: pytest.MonkeyPatch,
    target: _Target,
    loop_state: str,
) -> None:
    """Fail a call through a live handle at once when the I/O loop has stopped or closed (as exit's may have since its check)."""
    server = await _serve(target)
    client = CarrierClient(server.address)
    assert await client.call("echo", 1, timeout_s=10) == 1
    loop, post = server._runtime.loop, runtime._post
    if loop_state == "stopped":
        monkeypatch.setattr(runtime, "_stopped", weakref.WeakSet([loop]))
    else:
        monkeypatch.setattr(runtime, "_post", lambda *args: False)
    start = time.monotonic()
    try:
        with pytest.raises(PeerLostError, match="the carrier's I/O loop has stopped$"):
            await client.call("echo", 2, timeout_s=60)
    finally:
        # The fixture's teardown stops this runtime's loops through them.
        monkeypatch.setattr(runtime, "_post", post)
        monkeypatch.setattr(runtime, "_stopped", weakref.WeakSet())
    assert time.monotonic() - start < 10 and target.calls == [1]


@pytest.mark.asyncio
async def test_a_reader_that_never_stopped_reads_what_came_before_a_hangup(
    target: _Target,
) -> None:
    """Read a frame that arrived before its client shut its sending side, as only a stopped reader checks for a hangup."""
    server = await _serve(target)
    body, _ = codec.dumps(("echo", ("sent",), {}))
    stuck = threading.Event()
    server._runtime.loop.call_soon_threadsafe(stuck.wait, 30)
    try:
        with _raw_connection(server.address) as sock:
            # All of it waits in the server's kernel, the hangup included, until the I/O thread reads.
            _send_frame(sock, Kind.HELLO, body=_hello(server.address))
            _send_frame(sock, Kind.REQUEST, 1, deadline_us=10**8, body=body)
            sock.shutdown(socket.SHUT_WR)
            await _until(lambda: _unacknowledged_bytes(sock) == 0)
            stuck.set()
            await _until(lambda: not server._connections and server.stats()["calls"])
    finally:
        stuck.set()
    assert server.stats()["calls"] == 1


@pytest.mark.asyncio
async def test_a_reply_that_settles_a_call_as_its_deadline_passes_is_returned(
    monkeypatch: pytest.MonkeyPatch, target: _Target
) -> None:
    """Return a reply that settled a call after its deadline ran out but before its caller gave up on it."""
    abandon = runtime._Call.abandon

    def late(call: runtime._Call) -> bool:
        # The caller loses the CPU between its deadline and giving up, until the reply settles the call.
        call.outcome.exception(timeout=30)
        return abandon(call)

    monkeypatch.setattr(runtime._Call, "abandon", late)
    server = await _serve(target, "nap")
    assert await CarrierClient(server.address).call("nap", 0.5, timeout_s=0.1) is None
    assert target.calls == ["nap"]


@pytest.mark.asyncio
async def test_a_killed_server_gives_peer_lost(children: _Children) -> None:
    """Fail a running call with ``PeerLostError`` once its server is SIGKILLed."""
    child = children.spawn(_SERVE_CHILD, _scope())
    client = CarrierClient(CarrierAddress.from_json(_read_line(child)))
    assert await client.call("echo", 1) == 1
    call = asyncio.ensure_future(client.call("hold"))
    await _until(lambda: _sending(client) or _pending(client))
    start = time.monotonic()
    child.send_signal(signal.SIGKILL)
    a = client.address
    with pytest.raises(
        PeerLostError, match=re.escape(f"carrier child at {a.host}:{a.port}")
    ):
        await call
    assert time.monotonic() - start < 10


@pytest.mark.asyncio
async def test_a_stopped_server_times_out(children: _Children) -> None:
    """Time out a call to a SIGSTOPped server at the call's deadline."""
    child = children.spawn(_SERVE_CHILD, _scope())
    client = CarrierClient(CarrierAddress.from_json(_read_line(child)))
    assert await client.call("echo", 1) == 1
    child.send_signal(signal.SIGSTOP)
    try:
        await _until(lambda: _stopped(child))
        start = time.monotonic()
        with pytest.raises(CallTimeoutError, match="within its 2 s deadline$"):
            await client.call("echo", 2, timeout_s=2)
        assert 2 <= time.monotonic() - start < 7
    finally:
        child.send_signal(signal.SIGCONT)


@pytest.mark.asyncio
async def test_a_closed_port_or_unknown_name_names_the_address() -> None:
    """Fail at once with ``PeerLostError`` naming host:port when nothing listens or resolves."""
    # Bound but not listening, so the port refuses connections and stays ours.
    with socket.socket() as bound:
        bound.bind(("127.0.0.1", 0))
        port = bound.getsockname()[1]
        # IDNA cannot encode the last name, so its lookup raises UnicodeError, not OSError.
        for host in ("127.0.0.1", "nonexistent.invalid", "node..invalid"):
            client = CarrierClient(CarrierAddress(host, port, "test", _scope(), "0"))
            start = time.monotonic()
            with pytest.raises(PeerLostError, match=re.escape(f"{host}:{port}")):
                await client.call("echo", 1)
            assert time.monotonic() - start < 10


@pytest.mark.asyncio
@pytest.mark.parametrize("how", ["refused", "stale"])
async def test_a_failed_connect_closes_its_socket(target: _Target, how: str) -> None:
    """Close the socket of a connect that a closed port refused or a stale address failed, rather than leaving it to the GC."""
    server = await _serve(target)
    with socket.socket() as bound:
        bound.bind(("127.0.0.1", 0))
        a = server.address
        if how == "refused":
            address = CarrierAddress(
                "127.0.0.1", bound.getsockname()[1], "test", _scope(), "0"
            )
        else:
            address = CarrierAddress("127.0.0.1", a.port, a.service, a.scope, "0" * 16)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ResourceWarning)
            with pytest.raises(PeerLostError):
                await CarrierClient(address).call("echo", 1, timeout_s=10)
            await runtime._call_on(server._runtime.loop, lambda: None)
            gc.collect()
    unclosed = [
        str(w.message) for w in caught if issubclass(w.category, ResourceWarning)
    ]
    assert not unclosed, unclosed


@pytest.mark.asyncio
async def test_a_wrong_scope_service_or_incarnation_is_stale(target: _Target) -> None:
    """Refuse a wrong scope, incarnation or service with one handshake per call."""
    server = await _serve(target)
    address = server.address
    stale = [
        CarrierAddress(
            address.host, address.port, "test", _scope(), address.incarnation
        ),
        CarrierAddress(address.host, address.port, "test", address.scope, "0" * 16),
        CarrierAddress(
            address.host, address.port, "other", address.scope, address.incarnation
        ),
    ]
    for count, wrong in enumerate(stale, start=1):
        with pytest.raises(StaleAddressError, match="refused the connection"):
            await CarrierClient(wrong).call("echo", 1)
        assert server.stats()["rejected"] == count
    assert target.calls == []


@pytest.mark.asyncio
async def test_a_silent_listener_times_out_at_the_connect_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Time out a connection to a listener that never answers at the connect deadline."""
    monkeypatch.setattr(runtime, "_CONNECT_TIMEOUT_S", 2)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        client = CarrierClient(CarrierAddress("127.0.0.1", port, "test", _scope(), "0"))
        start = time.monotonic()
        with pytest.raises(CallTimeoutError, match="2 s connect deadline"):
            await client.call("echo", 1)
        assert 2 <= time.monotonic() - start < 7


@pytest.mark.asyncio
async def test_connects_share_a_name_lookup_that_hangs(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Start one resolver thread for a name whose lookup hangs, however many connects and handles try it.

    A lookup that fails once every connect gave up on it logs nothing either.
    """
    blocked = threading.Event()

    def hang(host: str, port: int) -> list[Any]:
        blocked.wait()
        raise socket.gaierror(socket.EAI_NONAME, "no such name")

    caplog.set_level(logging.ERROR, logger="asyncio")
    monkeypatch.setattr(runtime, "_CONNECT_TIMEOUT_S", 0.2)
    monkeypatch.setattr(runtime, "_getaddrinfo", hang)
    name = "torchstore-carrier-resolve shared.invalid"
    address = CarrierAddress("shared.invalid", 1, "test", _scope(), "0")
    try:
        client = CarrierClient(address)
        for _ in range(3):
            with pytest.raises(CallTimeoutError, match="connect deadline"):
                await client.call("echo", 1, timeout_s=10)
        await client.close()
        # A fresh handle connects anew, but shares the lookup still running.
        for _ in range(2):
            fresh = CarrierClient(address)
            with pytest.raises(CallTimeoutError, match="connect deadline"):
                await fresh.call("echo", 1, timeout_s=10)
            await fresh.close()
        assert [thread.name for thread in threading.enumerate()].count(name) == 1
        blocked.set()
        await _until(lambda: name not in [t.name for t in threading.enumerate()])
        # Its result reaches the I/O loop before this runs there.
        await runtime._call_on(client._peer.runtime.loop, lambda: None)
        gc.collect()
        assert not [r for r in caplog.records if "never retrieved" in r.getMessage()]
    finally:
        blocked.set()


@pytest.mark.asyncio
async def test_a_resolver_thread_that_cannot_start_leaves_no_lookup_behind(
    monkeypatch: pytest.MonkeyPatch, target: _Target
) -> None:
    """Fail a connect whose resolver thread cannot start, and let the next connect start a lookup of its own."""
    server = await _serve(target)
    start_thread = threading.Thread.start
    failing = True

    def start(thread: threading.Thread) -> None:
        if failing and thread.name.startswith("torchstore-carrier-resolve"):
            raise RuntimeError("can't start new thread")
        start_thread(thread)

    monkeypatch.setattr(threading.Thread, "start", start)
    # A lookup left behind would hold the next connect until this deadline.
    monkeypatch.setattr(runtime, "_CONNECT_TIMEOUT_S", 3)
    client = CarrierClient(server.address)
    with pytest.raises(PeerLostError, match="can't start new thread"):
        await client.call("echo", 1, timeout_s=10)
    failing = False
    assert await client.call("echo", 2, timeout_s=10) == 2


@pytest.mark.asyncio
async def test_a_name_lookup_that_failed_is_tried_again(
    monkeypatch: pytest.MonkeyPatch, target: _Target
) -> None:
    """Resolve a name again on the next connect after its lookup failed, as a resolver's transient error makes it."""
    server = await _serve(target)
    a = server.address
    lookups: list[str] = []

    def flaky(host: str, port: int) -> list[Any]:
        lookups.append(host)
        if len(lookups) == 1:
            raise socket.gaierror(
                socket.EAI_AGAIN, "Temporary failure in name resolution"
            )
        return socket.getaddrinfo("127.0.0.1", port, type=socket.SOCK_STREAM)

    monkeypatch.setattr(runtime, "_getaddrinfo", flaky)
    address = CarrierAddress("flaky.invalid", a.port, a.service, a.scope, a.incarnation)
    client = CarrierClient(address)
    with pytest.raises(PeerLostError, match="cannot resolve"):
        await client.call("echo", 1, timeout_s=10)
    assert await client.call("echo", 2, timeout_s=10) == 2
    assert lookups == ["flaky.invalid", "flaky.invalid"]


@pytest.mark.asyncio
async def test_close_fails_queued_calls_and_lets_the_running_one_finish(
    target: _Target,
) -> None:
    """Fail queued calls with a ``PeerLostError`` naming the closed volume, and finish the running one."""
    server = await _serve(target)
    client = CarrierClient(server.address)
    running = asyncio.ensure_future(client.call("hold", "running"))
    await _until(lambda: server.stats()["running"] == 1)
    queued = [asyncio.ensure_future(client.call("echo", label)) for label in "ab"]
    await _until(lambda: server.stats()["queued"] == 2)
    closing = asyncio.ensure_future(server.close(grace_s=10))
    for call in queued:
        with pytest.raises(PeerLostError, match=_closed(server)):
            await asyncio.wait_for(call, 5)
    assert not running.done()
    target.release.set()
    assert await running == "running"
    await closing
    assert target.calls == ["running"]


@pytest.mark.asyncio
@pytest.mark.parametrize("concurrent", [False, True])
async def test_close_returns_once_the_running_call_finishes(
    target: _Target, concurrent: bool
) -> None:
    """Return from a close soon after its running call finishes, well before its 30 s grace ends."""
    server = await _serve(target, concurrent=concurrent)
    call = asyncio.ensure_future(
        CarrierClient(server.address).call("hold", "x", timeout_s=60)
    )
    await _until(lambda: server.stats()["running"] == 1)
    closing = asyncio.ensure_future(server.close(grace_s=30))
    await _until(lambda: server._closing)
    target.release.set()
    start = time.monotonic()
    assert await call == "x"
    await closing
    assert time.monotonic() - start < 10


@pytest.mark.asyncio
async def test_close_waits_for_a_reply_still_on_its_way_to_the_outbox(
    monkeypatch: pytest.MonkeyPatch, target: _Target
) -> None:
    """Deliver a reply a close finds after its handler ended but before its connection's outbox has it.

    No call runs then and the outbox is empty, so only the count of replies on their way holds the close.
    """
    handoff, release = threading.Event(), threading.Event()
    post = runtime._post

    def held_post(
        loop: asyncio.AbstractEventLoop, fn: Callable[..., Any], *args: Any
    ) -> bool:
        if getattr(fn, "__name__", "") == "_queue_reply":
            handoff.set()
            release.wait(30)
        return post(loop, fn, *args)

    monkeypatch.setattr(runtime, "_post", held_post)
    server = await _serve(target)
    call = asyncio.ensure_future(
        CarrierClient(server.address).call("hold", "x", timeout_s=60)
    )
    try:
        await _until(lambda: server.stats()["running"] == 1)
        closing = asyncio.ensure_future(server.close(grace_s=30))
        await _until(lambda: server._closing)
        target.release.set()
        await _until(handoff.is_set)
        assert not server._idle.done()
    finally:
        release.set()
    assert await call == "x"
    await closing


@pytest.mark.asyncio
async def test_close_delivers_a_reply_that_reaches_its_connection_after_the_grace(
    monkeypatch: pytest.MonkeyPatch, target: _Target
) -> None:
    """Deliver each reply whose handler finished inside the grace but reached its connection after it ended.

    Two are on their way when the grace ends, and close waits for the second after the first arrives.
    """
    late: list[
        tuple[asyncio.AbstractEventLoop, Callable[..., Any], tuple[Any, ...]]
    ] = []
    post = runtime._post

    def late_post(
        loop: asyncio.AbstractEventLoop, fn: Callable[..., Any], *args: Any
    ) -> bool:
        if getattr(fn, "__name__", "") != "_queue_reply":
            return post(loop, fn, *args)
        late.append(
            (loop, fn, args)
        )  # as if the dispatcher thread lost the CPU right after it finished
        return True

    monkeypatch.setattr(runtime, "_post", late_post)
    # So a close that waited out its bound for the replies, rather than for the replies, would show.
    monkeypatch.setattr(runtime, "_RELEASE_WAIT_S", 30)
    server = await _serve(target, "wait", concurrent=True)
    client = CarrierClient(server.address)
    calls = [asyncio.ensure_future(client.call("wait", x, timeout_s=60)) for x in "ab"]
    await _until(lambda: server.stats()["running"] == 2)
    closing = asyncio.ensure_future(server.close(grace_s=3))
    await _until(lambda: server._closing)
    target.release.set()
    # Both handlers finished inside the grace, and their replies are still on their way as it ends.
    await _until(lambda: len(late) == 2 and server.stats()["running"] == 0)
    await _until(lambda: server._handed_off is not None)
    (first_loop, first, first_args), (last_loop, last, last_args) = late
    post(first_loop, first, *first_args)
    await _until(lambda: sum(call.done() for call in calls) == 1)
    # The first reply that arrives leaves close waiting for the other.
    handed_off = server._handed_off
    assert handed_off is not None and not handed_off.done()
    post(last_loop, last, *last_args)
    assert await asyncio.wait_for(asyncio.gather(*calls), 20) == ["a", "b"]
    await asyncio.wait_for(closing, 15)


@pytest.mark.asyncio
async def test_a_handler_outliving_the_grace_period_loses_its_caller(
    target: _Target,
) -> None:
    """Fail a handler's caller with a ``PeerLostError`` naming the volume when the grace ends.

    A cancelled close still shuts the server down, and a later close returns once it has.
    """
    server = await _serve(target)
    client = CarrierClient(server.address)
    call = asyncio.ensure_future(client.call("hold", timeout_s=10))
    await _until(lambda: server.stats()["running"] == 1)
    start = time.monotonic()
    await _close(server, 1, cancelled=True)
    with pytest.raises(PeerLostError, match=_closed(server)):
        await call
    assert 1 <= time.monotonic() - start < 6
    assert server not in server._runtime.servers and not server._connections


async def _held_reply(
    timeout_s: float | None = None,
) -> tuple[CarrierServer, CarrierClient, asyncio.Future[Any]]:
    """Start a ``get`` whose reply its server holds unsent. Return the server, client and call."""
    server = await _serve(_Store(), "get")
    client = CarrierClient(server.address)
    await server._set_sending(False)
    get = asyncio.ensure_future(client.call("get", timeout_s=timeout_s))
    await _until(lambda: _counts(server) == (1, 0, 0))
    return server, client, get


@pytest.mark.asyncio
async def test_a_close_with_no_grace_lets_a_reply_finish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Give a reply still being sent at least the release wait, even when the grace is 0."""
    monkeypatch.setattr(runtime, "_RELEASE_WAIT_S", 3)
    server, _, get = await _held_reply()
    closing = asyncio.ensure_future(server.close(grace_s=0))
    await asyncio.sleep(1)  # A slow reply, shorter than the release wait.
    assert not get.done()
    await server._set_sending(True)
    assert torch.equal(await get, torch.zeros(1 << 16))
    await closing


@pytest.mark.asyncio
async def test_a_connection_whose_reply_is_unsent_closes_without_draining(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Close a connection whose reply is still unsent when the flush wait ends at once, never half-closing it to read on."""
    monkeypatch.setattr(runtime, "_RELEASE_WAIT_S", 0.5)
    server, _, get = await _held_reply()
    [conn] = server._connections
    await server.close(grace_s=0)
    with pytest.raises(PeerLostError):
        await get
    assert conn.closed and not conn.draining


@pytest.mark.asyncio
async def test_close_stops_waiting_for_a_client_that_left(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End a closing server's wait to flush a reply once that reply's client disconnects."""
    started = threading.Event()
    close_connections = CarrierServer._close_connections

    async def spy(self: CarrierServer, flush_s: float) -> None:
        started.set()
        await close_connections(self, flush_s)

    monkeypatch.setattr(CarrierServer, "_close_connections", spy)
    server, client, get = await _held_reply()
    closing = asyncio.ensure_future(server.close(grace_s=30))
    await _until(started.is_set)
    await client.close()
    with pytest.raises(PeerLostError):
        await get
    start = time.monotonic()
    await asyncio.wait_for(closing, 15)
    assert time.monotonic() - start < 5


def _slow_tensor_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make each received tensor take 1 s longer to arrive, as over a slow link."""
    read_tensor = runtime._Reader.read_tensor

    async def slow(reader: runtime._Reader, size: int) -> torch.Tensor:
        await asyncio.sleep(1)  # A slow link, not synchronization.
        return await read_tensor(reader, size)

    monkeypatch.setattr(runtime._Reader, "read_tensor", slow)


def _record_received(
    monkeypatch: pytest.MonkeyPatch,
) -> list[weakref.ref[torch.Tensor]]:
    """Return weak references to the tensors received from now on, in arrival order."""
    received: list[weakref.ref[torch.Tensor]] = []
    read_tensor = runtime._Reader.read_tensor

    async def record(reader: runtime._Reader, size: int) -> torch.Tensor:
        tensor = await read_tensor(reader, size)
        received.append(weakref.ref(tensor))
        return tensor

    monkeypatch.setattr(runtime._Reader, "read_tensor", record)
    return received


@pytest.mark.asyncio
async def test_close_delivers_a_flushed_reply_to_a_client_still_sending() -> None:
    """Deliver a whole reply flushed before the close, though its client sends calls after it.

    Closing a socket with bytes unread resets it, dropping what the kernel still holds to send,
    so a half-closed connection reads on and drops the answers it can no longer send.
    """
    size, held_back = 16 << 20, 1 << 20
    server = await _serve(_Sink(size), "get")
    a = server.address
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 128 << 10)
        sock.settimeout(30)
        sock.connect(("127.0.0.1", a.port))
        _send_frame(sock, Kind.HELLO, body=_hello(a))
        assert _handshake_reply(sock)[0] == Kind.WELCOME
        body, _ = codec.dumps(("get", (), {}))
        _send_frame(sock, Kind.REQUEST, 1, deadline_us=10**8, body=body)
        header = codec.decode_header(_receive(sock, codec.HEADER_SIZE), handshake=False)
        lengths = _receive(sock, codec.LENGTH_SIZE * header.buffer_count)
        assert codec.decode_lengths(header, lengths) == (size,)
        _receive(sock, header.body_len)
        # Stop short of the end, so the rest is still in the kernel when the server closes.
        _receive_into(sock, size - held_back)
        await _until(lambda: server.stats()["bytes_out"] >= size)
        [conn] = server._connections
        closing = asyncio.ensure_future(server.close(grace_s=30))
        await _until(lambda: conn.closed or conn.draining)
        # Writing the second call's answer would fail and close the socket with the
        # third, too big to arrive at once, still unread.
        tensor = torch.zeros(8 << 20, dtype=torch.uint8)
        big_body, big = codec.dumps(("get", (tensor,), {}))
        _send_frame(sock, Kind.REQUEST, 2, deadline_us=10**8, body=body)
        _send_frame(
            sock, Kind.REQUEST, 3, deadline_us=10**8, body=big_body, buffers=big
        )
        await _until(lambda: server.stats()["calls"] == 3)
        _receive_into(sock, held_back)
        assert sock.recv(1) == b""
    await asyncio.wait_for(closing, 30)


@pytest.mark.asyncio
async def test_close_frees_the_queued_calls_it_fails_at_once(
    monkeypatch: pytest.MonkeyPatch, target: _Target
) -> None:
    """Free the received tensors of queued calls a close fails, while a running call holds the grace."""
    received = _record_received(monkeypatch)
    server = await _serve(target)
    client = CarrierClient(server.address)
    running = asyncio.ensure_future(client.call("hold", "running", timeout_s=60))
    await _until(lambda: server.stats()["running"] == 1)
    queued = [
        asyncio.ensure_future(client.call("echo", torch.ones(1 << 18), timeout_s=60))
        for _ in range(4)
    ]
    await _until(lambda: server.stats()["queued"] == 4)
    closing = asyncio.ensure_future(server.close(grace_s=30))
    for call in queued:
        with pytest.raises(PeerLostError, match=_closed(server)):
            await call
    assert [ref() for ref in received] == [None] * 4
    assert not closing.done()
    target.release.set()
    assert await running == "running"
    await closing


@pytest.mark.asyncio
async def test_answered_calls_and_closed_servers_leave_nothing_behind(
    target: _Target,
) -> None:
    """Drop answered calls from both ends' tables, and stop a closed server's dispatcher, also after a late handler."""
    server = await _serve(target)
    client = CarrierClient(server.address)
    for i in range(5):
        value = torch.full((4,), float(i))
        assert torch.equal(await client.call("echo", value), value)
    [accepted] = server._connections
    conn = client._peer.conn
    assert conn is not None and not conn.pending and not conn.unanswered
    await _until(lambda: not accepted.requests and not accepted.unsent)
    assert (accepted.queued, accepted.unsent_bytes) == (0, 0)
    await server.close(grace_s=0)
    assert not server._dispatcher.thread.is_alive()
    late = await _serve(target)
    call = asyncio.ensure_future(CarrierClient(late.address).call("hold", timeout_s=30))
    await _until(lambda: late.stats()["running"] == 1)
    await late.close(grace_s=0.5)
    with pytest.raises(PeerLostError, match=_closed(late)):
        await call
    assert late.stats()["running"] == 1
    target.release.set()
    await _until(lambda: not late._dispatcher.thread.is_alive())
    assert late.stats()["running"] == 0


@pytest.mark.asyncio
async def test_a_close_is_done_only_once_its_thread_lets_go_of_the_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mark a close done only after its thread drops the server, which then frees at once with the cyclic GC off.

    A thread still holding it once its closer returned could free the served object's tensors at exit.
    """
    run, resolve = asyncio.run, runtime._resolve
    ran, resolved = threading.Event(), threading.Event()
    run_released, resolve_released = threading.Event(), threading.Event()

    def on_close_thread() -> bool:
        return threading.current_thread().name.startswith("torchstore-carrier-close")

    def held_run(coroutine: Coroutine[Any, Any, Any]) -> Any:
        try:
            return run(coroutine)
        finally:
            if on_close_thread():
                ran.set()
                run_released.wait(30)

    def held_resolve(future: Any) -> None:
        held = on_close_thread()
        if held:
            resolved.set()
        resolve(future)
        if held:
            resolve_released.wait(30)

    monkeypatch.setattr(asyncio, "run", held_run)
    monkeypatch.setattr(runtime, "_resolve", held_resolve)
    gc.disable()
    try:
        store = _Store()
        server = await _serve(store, "get")
        assert torch.equal(
            await CarrierClient(server.address).call("get"), store.tensor
        )
        freed = weakref.ref(store)
        del store
        closing = asyncio.ensure_future(server.close(grace_s=0))
        await _until(ran.is_set)
        # Its shutdown has ended, but its thread still holds the server.
        assert not server._closed.done()
        run_released.set()
        await closing
        # Its thread is still held in the close's last step.
        assert resolved.is_set()
        del server
        assert freed() is None
    finally:
        run_released.set()
        resolve_released.set()
        gc.enable()


@pytest.mark.asyncio
async def test_a_close_that_failed_a_queued_call_lets_go_of_the_server_before_it_is_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Free the served object, with the cyclic GC off, once a close that failed a queued call is done.

    The failed call holds the server through its connection, so the close thread drops the call first.
    """
    resolve = runtime._resolve
    resolved, release = threading.Event(), threading.Event()

    def held_resolve(future: Any) -> None:
        held = threading.current_thread().name.startswith("torchstore-carrier-close")
        if held:
            resolved.set()
        resolve(future)
        if held:
            release.wait(30)

    monkeypatch.setattr(runtime, "_resolve", held_resolve)
    gc.disable()
    try:
        target = _Target()
        server = await _serve(target, "wait", "echo")
        client = CarrierClient(server.address)
        running = asyncio.ensure_future(client.call("wait", "running", timeout_s=60))
        await _until(lambda: server.stats()["running"] == 1)
        queued = asyncio.ensure_future(client.call("echo", "queued", timeout_s=60))
        await _until(lambda: server.stats()["queued"] == 1)
        freed = weakref.ref(target)
        closing = asyncio.ensure_future(server.close(grace_s=30))
        with pytest.raises(PeerLostError, match=_closed(server)):
            await queued
        target.release.set()
        assert await running == "running"
        await closing
        # Its thread is still held in the close's last step.
        assert resolved.is_set()
        await client.close()
        del target, server, client, running, queued, closing
        await _until(lambda: freed() is None, 5)
    finally:
        release.set()
        gc.enable()


@pytest.mark.asyncio
async def test_close_fails_a_call_behind_a_full_queue_at_once(
    monkeypatch: pytest.MonkeyPatch, target: _Target
) -> None:
    """Fail queued calls, and one left unread behind the full queue, at once when the server closes."""
    _no_window(monkeypatch)
    monkeypatch.setattr(runtime, "_QUEUE_BOUND", 2)
    server = await _serve(target)
    client = CarrierClient(server.address)
    running = asyncio.ensure_future(client.call("hold", "running", timeout_s=60))
    await _until(lambda: server.stats()["running"] == 1)
    queued = [
        asyncio.ensure_future(client.call("echo", i, timeout_s=60)) for i in range(2)
    ]
    await _until(lambda: server.stats()["queued"] == 2)
    unread = asyncio.ensure_future(client.call("echo", "unread", timeout_s=60))
    conn = client._peer.conn
    assert conn is not None
    await _until(lambda: len(conn.pending) == 4 and conn.sending is None)
    start = time.monotonic()
    closing = asyncio.ensure_future(server.close(grace_s=30))
    for call in [*queued, unread]:
        with pytest.raises(PeerLostError, match=_closed(server)):
            await asyncio.wait_for(call, 10)
    assert time.monotonic() - start < 10
    target.release.set()
    assert await running == "running"
    await closing
    assert target.calls == ["running"]


@pytest.mark.asyncio
async def test_a_closed_server_stops_listening(
    target: _Target, caplog: pytest.LogCaptureFixture
) -> None:
    """Refuse connections to a closed server's port, where nothing listens any more.

    Its accept loop, whose listener closed under its wait, ends quietly.
    """
    caplog.set_level(logging.ERROR, logger="asyncio")
    server = await _serve(target)
    address = server.address
    await server.close(grace_s=0)
    with pytest.raises(PeerLostError, match="cannot connect"):
        await CarrierClient(address).call("echo", 1, timeout_s=10)
    assert _task_errors(caplog, "CarrierServer._accept") == []


@pytest.mark.asyncio
async def test_a_port_serves_again_right_after_its_server_closed(
    target: _Target,
) -> None:
    """Serve again on a port whose closed server's connections still linger in TIME_WAIT."""
    first = await _serve(target)
    a = first.address
    with _raw_connection(a) as sock:
        _send_frame(sock, Kind.HELLO, body=_hello(a))
        assert _handshake_reply(sock)[0] == Kind.WELCOME
        closing = asyncio.ensure_future(first.close(grace_s=0))
        # The shutdown runs off this loop, which the read below blocks.
        await _until(lambda: first._closing)
        # The server's FIN came first, so its end waits in TIME_WAIT once this one closes.
        assert _closed_by_peer(sock)
    await closing
    second = await _serve(target, port=a.port)
    assert await CarrierClient(second.address).call("echo", 2, timeout_s=10) == 2


@pytest.mark.asyncio
async def test_a_restarted_server_refuses_its_old_address(target: _Target) -> None:
    """Refuse the old address of a server served again on the same port, scope and service."""
    first = await _serve(target)
    old = first.address
    await first.close(grace_s=0)
    second = await CarrierServer.serve(
        target, ["echo"], service=old.service, scope=old.scope, port=old.port
    )
    assert second.address.incarnation != old.incarnation
    with pytest.raises(StaleAddressError, match="refused the connection: incarnation"):
        await CarrierClient(old).call("echo", 1, timeout_s=10)
    assert await CarrierClient(second.address).call("echo", 2, timeout_s=10) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
@pytest.mark.parametrize("fails", ["thread", "loop"])
async def test_close_finishes_when_its_thread_cannot_start(
    monkeypatch: pytest.MonkeyPatch,
    target: _Target,
    fails: str,
    cancelled: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Shut down on the I/O thread, with a WARNING, when no close thread, or no loop on it, can start.

    A close returns once that shutdown ends, later closes then return, and a close cancelled
    on a loop that ends, as ``asyncio.run`` ends, still shuts down.
    """
    caplog.set_level(logging.WARNING, logger="torchstore.carrier")
    start_thread = threading.Thread.start
    new_event_loop = asyncio.events.new_event_loop

    def start(thread: threading.Thread) -> None:
        if thread.name.startswith("torchstore-carrier-close"):
            raise RuntimeError("can't start new thread")
        start_thread(thread)

    def new_loop() -> asyncio.AbstractEventLoop:
        if threading.current_thread().name.startswith("torchstore-carrier-close"):
            raise OSError(errno.EMFILE, "Too many open files")
        return new_event_loop()

    async def cancel_close(server: CarrierServer) -> None:
        closing = asyncio.ensure_future(server.close(grace_s=1))
        await _until(lambda: server._closing)
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closing

    server = await _serve(target)
    client = CarrierClient(server.address)
    running = asyncio.ensure_future(client.call("hold", "running", timeout_s=10))
    await _until(lambda: server.stats()["running"] == 1)
    queued = asyncio.ensure_future(client.call("echo", "queued", timeout_s=10))
    await _until(lambda: server.stats()["queued"] == 1)
    if fails == "thread":
        monkeypatch.setattr(threading.Thread, "start", start)
    else:
        monkeypatch.setattr(asyncio.events, "new_event_loop", new_loop)
    begin = time.monotonic()
    if cancelled:
        _in_thread(cancel_close(server))
    else:
        await asyncio.wait_for(server.close(grace_s=1), 15)
        assert server not in server._runtime.servers
    for call in (queued, running):
        with pytest.raises(PeerLostError, match=_closed(server)):
            await call
    assert time.monotonic() - begin < 6
    await asyncio.wait_for(server.close(grace_s=1), 5)
    await asyncio.wait_for(close_all(grace_s=0), 5)
    assert server not in server._runtime.servers and not server._connections
    assert target.calls == ["running"]
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "closes on the I/O thread" in warnings[0], warnings


async def _stop_accepting(server: CarrierServer) -> None:
    """End a server's accept task and close its listener, which a shutdown that never runs leaves behind."""

    async def stop() -> None:
        accepting = server._accepting
        server._stop_accepting()
        if accepting is not None:
            await asyncio.wait([accepting])

    await asyncio.wrap_future(
        asyncio.run_coroutine_threadsafe(stop(), server._runtime.loop)
    )


@pytest.mark.asyncio
async def test_a_close_left_to_the_io_loop_after_exit_returns(
    fresh_runtime: None,
    monkeypatch: pytest.MonkeyPatch,
    target: _Target,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Return from a close whose thread cannot start once exit has closed the I/O loop, and from every later close."""
    server = await _serve(target)
    await _stop_accepting(server)
    caplog.set_level(logging.WARNING, logger="torchstore.carrier")
    runtime._stop_loops_at_exit()
    start_thread = threading.Thread.start

    def start(thread: threading.Thread) -> None:
        if thread.name.startswith("torchstore-carrier-close"):
            raise RuntimeError("can't create new thread at interpreter shutdown")
        start_thread(thread)

    monkeypatch.setattr(threading.Thread, "start", start)
    await asyncio.wait_for(server.close(grace_s=0), 10)
    await asyncio.wait_for(server.close(grace_s=0), 10)
    # Nothing closes on the I/O thread so nothing says it does.
    assert not [
        r for r in caplog.records if "closes on the I/O thread" in r.getMessage()
    ]


@pytest.mark.asyncio
async def test_a_close_left_to_an_io_loop_that_exit_stopped_returns(
    fresh_runtime: None,
    monkeypatch: pytest.MonkeyPatch,
    target: _Target,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Return from a close whose thread cannot start once exit has stopped the I/O loop but not closed it yet.

    The shutdown could never run there: the close closes it unstarted, and every later close returns too.
    """
    server = await _serve(target)
    await _stop_accepting(server)
    loop = server._runtime.loop
    shutdowns: list[Coroutine[Any, Any, None]] = []
    shut_down_here = server._shut_down_here

    def recorded(*args: Any) -> Coroutine[Any, Any, None]:
        shutdowns.append(shut_down_here(*args))
        return shutdowns[-1]

    monkeypatch.setattr(server, "_shut_down_here", recorded)
    start_thread = threading.Thread.start

    def start(thread: threading.Thread) -> None:
        if thread.name.startswith("torchstore-carrier-close"):
            raise RuntimeError("can't create new thread at interpreter shutdown")
        start_thread(thread)

    caplog.set_level(logging.WARNING, logger="torchstore.carrier")
    # So exit leaves the loop stopped but open: its thread's close does nothing.
    loop.close = lambda: None  # type: ignore[method-assign]
    try:
        runtime._stop_loops_at_exit()
        monkeypatch.setattr(threading.Thread, "start", start)
        await asyncio.wait_for(server.close(grace_s=0), 10)
        await asyncio.wait_for(server.close(grace_s=0), 10)
    finally:
        del loop.close
        loop.close()
    [shutdown] = shutdowns
    assert inspect.getcoroutinestate(shutdown) == inspect.CORO_CLOSED
    assert not [
        r for r in caplog.records if "closes on the I/O thread" in r.getMessage()
    ]


@pytest.mark.asyncio
async def test_a_close_left_to_an_io_loop_that_ended_without_exit_returns(
    fresh_runtime: None, monkeypatch: pytest.MonkeyPatch, target: _Target
) -> None:
    """Return from a close whose thread cannot start once the I/O loop closed without exit stopping it, and from a later close.

    A BaseException that escapes one of the loop's callbacks ends it that way.
    """
    server = await _serve(target)
    await _stop_accepting(server)
    loop = server._runtime.loop
    [loop_thread] = [thread for thread in runtime._loop_threads if thread.loop is loop]
    loop.call_soon_threadsafe(loop.stop)
    await asyncio.to_thread(loop_thread.thread.join, 10)
    assert loop.is_closed() and loop not in runtime._stopped
    start_thread = threading.Thread.start

    def start(thread: threading.Thread) -> None:
        if thread.name.startswith("torchstore-carrier-close"):
            raise RuntimeError("can't start new thread")
        start_thread(thread)

    monkeypatch.setattr(threading.Thread, "start", start)
    await asyncio.wait_for(server.close(grace_s=0), 10)
    await asyncio.wait_for(server.close(grace_s=0), 10)


@pytest.mark.asyncio
async def test_a_close_left_running_on_the_io_loop_as_exit_stops_it_returns(
    fresh_runtime: None, monkeypatch: pytest.MonkeyPatch, target: _Target
) -> None:
    """Return from a close whose shutdown, left to the I/O loop, is still running when exit stops that loop."""
    server = await _serve(target)
    await _stop_accepting(server)
    loop = server._runtime.loop
    [loop_thread] = [thread for thread in runtime._loop_threads if thread.loop is loop]
    started, release, done = threading.Event(), asyncio.Event(), asyncio.Event()

    async def stuck_shutdown(*args: Any) -> None:
        started.set()
        await release.wait()
        done.set()

    async def end_shutdown() -> None:
        release.set()
        await done.wait()

    monkeypatch.setattr(server, "_shut_down_here", stuck_shutdown)
    start_thread = threading.Thread.start

    def start(thread: threading.Thread) -> None:
        if thread.name.startswith("torchstore-carrier-close"):
            raise RuntimeError("can't start new thread")
        start_thread(thread)

    monkeypatch.setattr(threading.Thread, "start", start)
    closing = asyncio.ensure_future(server.close(grace_s=0))
    await _until(started.is_set)
    monkeypatch.setattr(threading.Thread, "start", start_thread)
    # Left open past exit's stop for the shutdown to end after the checks.
    loop.close = lambda: None  # type: ignore[method-assign]
    try:
        await asyncio.to_thread(runtime._stop_loops_at_exit)
        await asyncio.wait_for(closing, 10)
        await asyncio.wait_for(server.close(grace_s=0), 10)
    finally:
        del loop.close
        await asyncio.to_thread(loop_thread.thread.join, 10)
        if not loop.is_running():
            await asyncio.to_thread(loop.run_until_complete, end_shutdown())
            loop.close()


def _new_threads(before: list[threading.Thread], prefix: str) -> list[threading.Thread]:
    return [
        thread
        for thread in threading.enumerate()
        if thread.name.startswith(prefix) and thread not in before
    ]


@pytest.mark.asyncio
async def test_a_serve_whose_io_thread_cannot_start_leaves_nothing_behind(
    monkeypatch: pytest.MonkeyPatch, target: _Target
) -> None:
    """Start no dispatcher thread and keep no descriptor when the process's I/O thread cannot start."""
    start_thread = threading.Thread.start

    def start(thread: threading.Thread) -> None:
        if thread.name == "torchstore-carrier":
            raise RuntimeError("can't start new thread")
        start_thread(thread)

    before, fds = threading.enumerate(), len(os.listdir("/proc/self/fd"))
    monkeypatch.setattr(runtime, "_runtime", None)
    monkeypatch.setattr(threading.Thread, "start", start)
    for _ in range(3):
        with pytest.raises(RuntimeError, match="can't start new thread"):
            await _serve(target)
    monkeypatch.undo()
    assert not _new_threads(before, "torchstore-carrier-dispatch")
    assert len(os.listdir("/proc/self/fd")) <= fds


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "thread", ["torchstore-carrier-dispatch", "torchstore-carrier"]
)
async def test_a_loop_thread_that_cannot_make_its_loop_fails_serve(
    monkeypatch: pytest.MonkeyPatch, target: _Target, thread: str
) -> None:
    """Raise from serve, not hang, when a dispatcher or the I/O thread cannot create its event loop."""
    new_event_loop = asyncio.new_event_loop

    def failing() -> asyncio.AbstractEventLoop:
        name = threading.current_thread().name
        if name == thread or name.startswith(f"{thread} "):
            raise OSError(errno.EMFILE, "Too many open files")
        return new_event_loop()

    if thread == "torchstore-carrier":
        monkeypatch.setattr(runtime, "_runtime", None)
    monkeypatch.setattr(asyncio, "new_event_loop", failing)
    outcome: list[BaseException] = []

    def run() -> None:
        try:
            asyncio.run(_serve(target))
        except BaseException as e:
            outcome.append(e)

    serving = threading.Thread(target=run, daemon=True)
    serving.start()
    serving.join(10)
    assert not serving.is_alive(), "serve hung"
    [error] = outcome
    assert isinstance(error, OSError) and error.errno == errno.EMFILE


@pytest.mark.asyncio
async def test_a_failed_serve_closes_its_listener(
    monkeypatch: pytest.MonkeyPatch, target: _Target
) -> None:
    """Close the listening socket when a server cannot start its dispatcher thread."""
    listen = runtime._listen
    opened: list[socket.socket] = []

    def spy(port: int) -> socket.socket:
        sock = listen(port)
        opened.append(sock)
        return sock

    start_thread = threading.Thread.start

    def start(thread: threading.Thread) -> None:
        if thread.name.startswith("torchstore-carrier-dispatch"):
            raise RuntimeError("can't start new thread")
        start_thread(thread)

    monkeypatch.setattr(runtime, "_listen", spy)
    monkeypatch.setattr(threading.Thread, "start", start)
    with pytest.raises(RuntimeError, match="can't start new thread"):
        await _serve(target)
    monkeypatch.undo()
    [sock] = opened
    assert sock.fileno() == -1


@pytest.mark.asyncio
async def test_listeners_take_a_burst_of_first_connects(
    monkeypatch: pytest.MonkeyPatch, target: _Target
) -> None:
    """Listen with a backlog of at least 4096, past Python's SOMAXCONN, for many ranks' first connects."""
    backlogs: list[int] = []

    class Recording(socket.socket):
        def listen(self, backlog: int = 0) -> None:
            backlogs.append(backlog)
            super().listen(backlog)

    monkeypatch.setattr(socket, "socket", Recording)
    server = await _serve(target)
    monkeypatch.undo()
    assert backlogs == [max(socket.SOMAXCONN, 4096)]
    assert await CarrierClient(server.address).call("echo", 1, timeout_s=10) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("how", ["cancel", "timeout"])
async def test_abandoning_a_call_mid_send_closes_the_connection(
    target: _Target, how: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Close the connection on a call abandoned mid-send at once, with no WARNING. Nothing runs, and unsent calls go anew."""
    caplog.set_level(logging.WARNING, logger="torchstore.carrier")
    server = await _serve(target)
    client = CarrierClient(server.address)
    assert await client.call("echo", "first") == "first"
    await server._set_reading(False)
    big = torch.zeros(64 << 20, dtype=torch.uint8)
    call = asyncio.ensure_future(client.call("echo", big, timeout_s=2))
    await _until(lambda: _sending(client))
    unsent = asyncio.ensure_future(client.call("echo", "unsent", timeout_s=30))
    await _until(lambda: _unsent(client))
    if how == "cancel":
        call.cancel()
    with pytest.raises(asyncio.CancelledError if how == "cancel" else CallTimeoutError):
        await call
    assert client._peer.conn is None
    await server._set_reading(True)
    assert await unsent == "unsent"
    assert await client.call("echo", "after") == "after"
    assert target.calls == ["first", "unsent", "after"]
    assert server.stats()["connections"] == 2
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


@pytest.mark.asyncio
async def test_a_call_cancelled_before_it_is_sent_is_never_sent(
    target: _Target,
) -> None:
    """Drop a call cancelled while it waits behind a stalled send, so its memory is never read."""
    server = await _serve(target)
    client = CarrierClient(server.address)
    assert await client.call("echo", "first") == "first"
    await server._set_reading(False)
    big = torch.zeros(64 << 20, dtype=torch.uint8)
    sending = asyncio.ensure_future(client.call("echo", big, timeout_s=60))
    await _until(lambda: _sending(client))
    small = torch.zeros(4)
    unsent = asyncio.ensure_future(client.call("echo", small, timeout_s=60))
    await _until(lambda: _unsent(client))
    conn = client._peer.conn
    assert conn is not None
    [held] = [item for item in list(conn.outbox) if isinstance(item, runtime._Call)]
    unsent.cancel()
    with pytest.raises(asyncio.CancelledError):
        await unsent
    # Its parts, which hold the caller's tensor, are let go as the call returns.
    assert held.body is None and held.buffers is None
    await server._set_reading(True)
    assert torch.equal(await sending, big)
    assert await client.call("echo", "after") == "after"
    stats = server.stats()
    assert (stats["calls"], stats["cancels"]) == (3, 0)
    labels = [call for call in target.calls if isinstance(call, str)]
    assert labels == ["first", "after"]


@pytest.mark.asyncio
async def test_a_full_queue_stops_reading_and_resumes_without_loss(
    monkeypatch: pytest.MonkeyPatch, target: _Target
) -> None:
    """Stop reading a connection at the queue bound, then deliver every call in order.

    The hangup check is off, so only the wakes as calls leave the queue or answers go out resume the reader.
    """
    _no_window(monkeypatch)
    monkeypatch.setattr(runtime, "_QUEUE_BOUND", 4)
    monkeypatch.setattr(runtime, "_HANGUP_CHECK_S", 3600)
    server = await _serve(target)
    client = CarrierClient(server.address)
    running = asyncio.ensure_future(client.call("hold", "running"))
    await _until(lambda: server.stats()["running"] == 1)
    queued = [asyncio.ensure_future(client.call("echo", i)) for i in range(10)]
    await _until(lambda: server.stats()["queued"] == 4)
    target.release.set()
    gathered: asyncio.Future[list[Any]] = asyncio.gather(*queued)
    assert await asyncio.wait_for(gathered, 10) == list(range(10))
    assert await running == "running"
    assert target.calls == ["running", *range(10)]
    assert server.stats()["max_queue_depth"] == 4


@pytest.mark.asyncio
async def test_a_call_leaving_a_full_queue_wakes_its_stopped_reader(
    monkeypatch: pytest.MonkeyPatch, target: _Target
) -> None:
    """Resume a reader stopped at the queue bound as a call leaves its queue, though no answer goes out.

    That call holds the dispatcher so it answers nothing, and the hangup check is off: only that wake can resume it.
    """
    _no_window(monkeypatch)
    monkeypatch.setattr(runtime, "_QUEUE_BOUND", 4)
    monkeypatch.setattr(runtime, "_HANGUP_CHECK_S", 3600)
    server = await _serve(target)
    a = server.address
    # Another name for the server, so the hold runs on a connection of its own.
    other = CarrierAddress("127.0.0.1", a.port, a.service, a.scope, a.incarnation)
    held = asyncio.ensure_future(CarrierClient(other).call("hold", "held"))
    await _until(lambda: server.stats()["running"] == 1)
    client = CarrierClient(a)
    # The first to leave the queue holds the dispatcher. The slot it frees thus takes no answer.
    calls = [asyncio.ensure_future(client.call("hold", "next"))]
    calls += [asyncio.ensure_future(client.call("echo", i)) for i in range(4)]
    # Four fill the queue, and the fifth waits unread in the stopped reader's socket.
    await _until(lambda: server.stats()["queued"] == 4)
    conn = client._peer.conn
    assert conn is not None
    await _until(lambda: len(conn.pending) == 5 and conn.sending is None)
    first, target.release = target.release, threading.Event()
    first.set()
    await _until(lambda: server.stats()["calls"] == 6)
    target.release.set()
    assert await asyncio.gather(*calls) == ["next", 0, 1, 2, 3]
    assert await held == "held"


@pytest.mark.asyncio
async def test_the_window_keeps_a_client_below_the_queue_bound(
    monkeypatch: pytest.MonkeyPatch, target: _Target
) -> None:
    """Hold a client's calls beyond the server's window, one below the bound, and send each as an answer frees room."""
    monkeypatch.setattr(runtime, "_QUEUE_BOUND", 4)
    server = await _serve(target)
    client = CarrierClient(server.address)
    running = asyncio.ensure_future(client.call("hold", "running"))
    queued = [asyncio.ensure_future(client.call("echo", i)) for i in range(10)]
    await _until(lambda: _counts(server) == (3, 1, 2))
    conn = client._peer.conn
    assert conn is not None and (conn.window, conn.unanswered) == (3, 3)
    target.release.set()
    assert await asyncio.gather(*queued) == list(range(10))
    assert await running == "running"
    assert target.calls == ["running", *range(10)]
    assert server.stats()["max_queue_depth"] < 4 and conn.unanswered == 0


@pytest.mark.asyncio
async def test_a_client_that_never_reads_stops_its_reader_at_the_bound(
    monkeypatch: pytest.MonkeyPatch, target: _Target
) -> None:
    """Stop reading a client whose answers pile up unsent, as they count against its queue bound.

    The hangup check is off: only the wake as answers go out resumes the reader.
    """
    monkeypatch.setattr(runtime, "_QUEUE_BOUND", 4)
    monkeypatch.setattr(runtime, "_HANGUP_CHECK_S", 3600)
    server = await _serve(target)
    await server._set_sending(False)
    body, _ = codec.dumps(("echo", (1,), {}))
    with _raw_connection(server.address) as sock:
        _send_frame(sock, Kind.HELLO, body=_hello(server.address))
        assert _handshake_reply(sock)[0] == Kind.WELCOME
        for request_id in range(1, 21):
            _send_frame(sock, Kind.REQUEST, request_id, deadline_us=10**8, body=body)
        [conn] = server._connections
        # Nothing queued or running and unsent answers at the bound.
        await _until(
            lambda: _counts(server)[1:] == (0, 0) and conn.unsent >= server._bound
        )
        # At the bound, or one past it for a call that was running then.
        calls = server.stats()["calls"]
        assert calls in (4, 5)
        # A wake as the hangup check would make leaves it stopped: the unsent answers still count.
        loop = server._runtime.loop
        await runtime._call_on(loop, conn.room.set)
        for _ in range(3):
            await runtime._call_on(loop, lambda: None)
        assert server.stats()["calls"] == calls
        await server._set_sending(True)
        for request_id in range(1, 21):
            header = codec.decode_header(
                _receive(sock, codec.HEADER_SIZE), handshake=False
            )
            _receive(sock, header.body_len)
            assert (header.kind, header.request_id) == (Kind.REPLY, request_id)
    assert server.stats()["calls"] == 20


@pytest.mark.asyncio
async def test_unsent_replies_hold_back_a_connections_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run none of a connection's queued calls while its unsent replies pass their share of bytes."""
    monkeypatch.setattr(runtime, "_UNSENT_BYTES_PER_CONNECTION", 600 << 10)
    store = _Store()
    server = await _serve(store, "get")
    client = CarrierClient(server.address)
    await server._set_sending(False)
    gets = [asyncio.ensure_future(client.call("get", timeout_s=60)) for _ in range(10)]
    # The third 256 KiB reply took the connection past its share, so seven calls wait.
    # Waiting for fewer counts could see the moment between two calls with none running.
    await _until(lambda: _counts(server) == (10, 0, 7))
    # Every dispatch the arrivals asked for has run, and none started another call.
    await runtime._call_on(server._dloop, lambda: None)
    assert _counts(server) == (10, 0, 7)
    await server._set_sending(True)
    for get in gets:
        assert torch.equal(await get, store.tensor)


@pytest.mark.asyncio
@pytest.mark.parametrize("how", ["abandon", "close"])
async def test_a_call_left_unread_on_a_closed_connection_never_runs(
    monkeypatch: pytest.MonkeyPatch, target: _Target, how: str
) -> None:
    """Never read a call left behind a full queue when its connection closes, so it cannot run after a later call.

    The connection closes on a call abandoned mid-send or with its last handle, and once a queue slot frees, the
    reader finds the connection reset before it reads the call. The hangup check is off so no timer wakes that reader.
    """
    _no_window(monkeypatch)
    monkeypatch.setattr(runtime, "_QUEUE_BOUND", 4)
    monkeypatch.setattr(runtime, "_HANGUP_CHECK_S", 3600)
    server = await _serve(target)
    a = server.address
    # Another name for the server, so the hold runs on a connection of its own.
    other = CarrierAddress("127.0.0.1", a.port, a.service, a.scope, a.incarnation)
    held = asyncio.ensure_future(CarrierClient(other).call("hold", "held"))
    await _until(lambda: server.stats()["running"] == 1)
    client = CarrierClient(a)
    queued = [asyncio.ensure_future(client.call("echo", i)) for i in range(4)]
    await _until(lambda: server.stats()["queued"] == 4)
    stale = asyncio.ensure_future(client.call("echo", "stale"))
    conn = client._peer.conn
    assert conn is not None
    # Sent in full, so it waits in the stopped reader's socket buffer.
    await _until(lambda: len(conn.pending) == 5 and conn.sending is None)
    if how == "abandon":
        big = torch.zeros(64 << 20, dtype=torch.uint8)
        with pytest.raises(CallTimeoutError, match="within its 1 s deadline$"):
            await client.call("echo", big, timeout_s=1)
    else:
        await client.close()
        client = CarrierClient(a)
    for call in [*queued, stale]:
        with pytest.raises(PeerLostError):
            await call
    later = asyncio.ensure_future(client.call("echo", "later"))
    await _until(lambda: server.stats()["queued"] == 5)
    target.release.set()
    assert await later == "later"
    assert await held == "held"
    await _until(lambda: len(server._connections) == 2)
    assert "stale" not in target.calls and target.calls[-1] == "later"
    # The hold, the 4 queued calls and the later one: the stale call was never read.
    assert server.stats()["calls"] == 6


@pytest.mark.asyncio
async def test_a_stopped_reader_finds_its_client_gone_though_no_slot_frees(
    monkeypatch: pytest.MonkeyPatch, target: _Target
) -> None:
    """Drop the queued calls of a client that reset while its reader was stopped at the bound, though no slot frees."""
    monkeypatch.setattr(runtime, "_QUEUE_BOUND", 4)
    monkeypatch.setattr(runtime, "_HANGUP_CHECK_S", 0.1)
    server = await _serve(target)
    a = server.address
    # Another name for the server, so the hold runs on a connection of its own.
    other = CarrierAddress("127.0.0.1", a.port, a.service, a.scope, a.incarnation)
    held = asyncio.ensure_future(CarrierClient(other).call("hold", "held"))
    await _until(lambda: server.stats()["running"] == 1)
    body, _ = codec.dumps(("echo", (1,), {}))
    with _raw_connection(a) as sock:
        _send_frame(sock, Kind.HELLO, body=_hello(a))
        assert _handshake_reply(sock)[0] == Kind.WELCOME
        for request_id in range(1, 5):
            _send_frame(sock, Kind.REQUEST, request_id, deadline_us=10**8, body=body)
        await _until(lambda: server.stats()["queued"] == 4)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    await _until(lambda: _counts(server)[1:] == (1, 0))
    assert server.stats()["cancels"] == 4
    target.release.set()
    assert await held == "held" and target.calls == ["held"]


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["sendall", "send_large"])
async def test_a_send_that_starts_on_a_full_socket_waits_for_room(path: str) -> None:
    """Send a whole chunk though the socket is full when its send starts: a caller's path, and a server's lone chunk."""
    loop = asyncio.get_running_loop()
    with socket.create_server(("127.0.0.1", 0)) as listener:
        sender = socket.create_connection(listener.getsockname())
        receiver, _ = listener.accept()
    with sender, receiver:
        sender.setblocking(False)
        filled = 0
        with contextlib.suppress(BlockingIOError):
            while True:
                filled += sender.send(bytes(1 << 16))
        # 8 MiB, past the offload threshold.
        payload = bytes(range(256)) * (8 << 12)
        received = bytearray()

        def drain() -> None:
            receiver.settimeout(30)
            while len(received) < filled + len(payload):
                chunk = receiver.recv(1 << 16)
                assert chunk, "the connection closed"
                received.extend(chunk)

        reader = threading.Thread(target=drain, daemon=True)
        # Started once the send has begun, so its first attempt finds the socket full.
        loop.call_soon(reader.start)
        if path == "sendall":
            await runtime._sendall(loop, sender, payload)
        else:
            owner = types.SimpleNamespace(writers=runtime._Writers())
            server = types.SimpleNamespace(_runtime=owner)
            conn = types.SimpleNamespace(
                server=server, sock=sender, loop=loop, job=None
            )
            await runtime._ServerConnection._send_large(conn, payload)  # type: ignore[arg-type]
        await asyncio.to_thread(reader.join, 30)
        assert bytes(received[filled:]) == payload


@pytest.mark.asyncio
async def test_a_paused_reader_delays_but_does_not_break_a_call(
    target: _Target,
) -> None:
    """Finish a 64 MiB call whose server stopped reading for 5 s, with TCP_USER_TIMEOUT unset."""
    server = await _serve(target)
    client = CarrierClient(server.address)
    assert await client.call("echo", "first") == "first"
    await server._set_reading(False)
    big = torch.randint(0, 256, (64 << 20,), dtype=torch.uint8)
    call = asyncio.ensure_future(client.call("echo", big, timeout_s=30))
    await _until(lambda: _sending(client))
    await asyncio.sleep(5)  # The pause under test, not synchronization.
    await server._set_reading(True)
    assert torch.equal(await call, big)
    assert server.stats()["connections"] == 1


@pytest.mark.asyncio
async def test_queue_wait_leaves_out_the_time_a_call_takes_to_arrive(
    monkeypatch: pytest.MonkeyPatch, target: _Target
) -> None:
    """Count a call's queue wait from the end of its frame, not from its header."""
    _slow_tensor_reads(monkeypatch)
    server = await _serve(target)
    client = CarrierClient(server.address)
    assert torch.equal(await client.call("echo", torch.ones(4)), torch.ones(4))
    assert server.stats()["max_queue_wait_ms"] < 500


@pytest.mark.asyncio
async def test_stats_count_bytes_and_queue_wait(target: _Target) -> None:
    """Count the bytes of a 1 MiB echo both ways, and its wait behind a running call."""
    server = await _serve(target)
    client = CarrierClient(server.address)
    held = asyncio.ensure_future(client.call("hold"))
    await _until(lambda: server.stats()["running"] == 1)
    queued = asyncio.ensure_future(client.call("echo", torch.ones(1 << 18)))
    await _until(lambda: server.stats()["queued"] == 1)
    start = time.monotonic()
    await asyncio.sleep(0.5)  # Queue time to measure, not synchronization.
    waited_ms = int((time.monotonic() - start) * 1000)
    target.release.set()
    await held
    assert torch.equal(await queued, torch.ones(1 << 18))
    stats = server.stats()
    assert stats["max_queue_wait_ms"] >= waited_ms and stats["bytes_in"] >= 1 << 20
    # Counted once the send returns, which may trail the reply's arrival.
    await _until(lambda: server.stats()["bytes_out"] >= 1 << 20)


@pytest.mark.asyncio
async def test_stats_count_an_in_band_body(target: _Target) -> None:
    """Count a call's in-band body in bytes_in, not only its out-of-band buffers."""
    server = await _serve(target)
    client = CarrierClient(server.address)
    payload = b"x" * (1 << 20)
    assert await client.call("echo", payload) == payload
    assert server.stats()["bytes_in"] >= 1 << 20


@pytest.mark.asyncio
async def test_a_large_in_band_body_arrives_unzeroed_and_uncopied(
    monkeypatch: pytest.MonkeyPatch, target: _Target
) -> None:
    """Receive in-band bodies over 64KiB in requests and replies into unzeroed tensors that they stay views of.

    Zeroing or copying one would keep the I/O thread busy for the whole body and stall every connection. No torch
    function runs on a landing tensor. Zeroing it with ``zero_`` or ``torch.zeros`` would run one. Each body stays a
    view of the very tensor it landed in. A copy would not.
    """
    sizes: list[int] = []
    make = bytearray

    def spy(*args: Any) -> bytearray:
        made = make(*args)
        sizes.append(len(made))
        return made

    landed: list[torch.Tensor] = []
    touched: list[Any] = []

    class Untouched(torch.Tensor):
        @classmethod
        def __torch_function__(  # type: ignore[override]
            cls, func: Any, types: Any, args: Any = (), kwargs: Any = None
        ) -> Any:
            touched.append(func)
            with torch._C.DisableTorchFunction():
                return func(*args, **(kwargs or {}))

    torch_names = types.ModuleType("torch")
    torch_names.__dict__.update(vars(torch))

    def empty(size: int, **kwargs: Any) -> torch.Tensor:
        landed.append(torch.empty(size, **kwargs).as_subclass(Untouched))
        return landed[-1]

    torch_names.empty = empty  # type: ignore[attr-defined]

    bodies: list[tuple[type, int, bool]] = []
    read_body = runtime._Reader.read_body

    async def recorded(reader: runtime._Reader, size: int) -> bytes | memoryview:
        body = await read_body(reader, size)
        held = getattr(getattr(body, "obj", None), "_tensor", None)
        bodies.append((type(body), len(body), any(held is t for t in landed)))
        return body

    monkeypatch.setattr(runtime, "bytearray", spy, raising=False)
    monkeypatch.setattr(runtime, "torch", torch_names)
    monkeypatch.setattr(runtime._Reader, "read_body", recorded)
    server = await _serve(target)
    client = CarrierClient(server.address)
    payload = {
        "blob": b"x" * (1 << 20),
        "items": list(range(20_000)),
        "text": "y" * 70_000,
    }
    assert await client.call("echo", payload) == payload
    assert await client.call("echo", b"z" * ((64 << 10) + 1)) == b"z" * ((64 << 10) + 1)
    assert not [size for size in sizes if size > runtime._SMALL_BYTES], sizes
    # Each large body of a request or a reply stays a view of the very tensor torch.empty made for it.
    large = [body for body in bodies if body[1] > runtime._SMALL_BYTES]
    assert (
        large == [(memoryview, size, True) for _, size, _ in large] and len(large) == 4
    )
    assert touched == [], touched


@pytest.mark.asyncio
async def test_a_dead_clients_queued_calls_are_dropped(
    target: _Target, children: _Children
) -> None:
    """Drop a killed client's queued calls, finish its running one, and keep serving the others."""
    server = await _serve(target)
    calls = [["hold", ["running"]], ["echo", ["a"]], ["echo", ["b"]]]
    child = children.spawn(_CALL_CHILD, server.address.to_json(), json.dumps(calls))
    await _until_child(child, lambda: _counts(server) == (3, 1, 2))
    survivor = asyncio.ensure_future(
        CarrierClient(server.address).call("echo", "survivor", timeout_s=30)
    )
    await _until(lambda: _counts(server) == (4, 1, 3))
    child.send_signal(signal.SIGKILL)
    await _until(lambda: _counts(server) == (4, 1, 1))
    target.release.set()
    assert await survivor == "survivor"
    await _until(lambda: server.stats()["running"] == 0)
    assert await CarrierClient(server.address).call("echo", "after") == "after"
    assert target.calls == ["running", "survivor", "after"]
    assert server.stats()["cancels"] == 3


def _no_tensor_after(calls: list[Any], label: str) -> bool:
    """Whether no tensor echo ran after the call labeled ``label``."""
    return not any(
        isinstance(call, torch.Tensor) for call in calls[calls.index(label) :]
    )


@pytest.mark.asyncio
async def test_calls_abandoned_before_the_last_handle_closes_never_run_later(
    monkeypatch: pytest.MonkeyPatch, target: _Target
) -> None:
    """Never run, after a later call, the calls a closed handle left in a stopped reader's full window.

    Every client socket closes with a reset, which reaches a server whose window is full. The
    first call naps, so its slot frees before the server writes anything that would reset it.
    The hangup check is off so no timer wakes the stopped reader.
    """
    _no_window(monkeypatch)
    monkeypatch.setattr(runtime, "_QUEUE_BOUND", 1)
    monkeypatch.setattr(runtime, "_HANGUP_CHECK_S", 3600)
    server = await _serve(target, *_ENDPOINTS, "nap")
    a = server.address
    other = CarrierAddress("127.0.0.1", a.port, a.service, a.scope, a.incarnation)
    held = asyncio.ensure_future(CarrierClient(other).call("hold", "held"))
    await _until(lambda: server.stats()["running"] == 1)
    [holding] = server._connections
    client = CarrierClient(a)
    calls = [asyncio.ensure_future(client.call("nap", 1.0, timeout_s=300))]
    await _until(lambda: server.stats()["queued"] == 1)
    conn = client._peer.conn
    assert conn is not None
    [filled] = [
        accepted for accepted in list(server._connections) if accepted is not holding
    ]
    # Whole frames until the window is full, so a FIN would wait behind them.
    while not _unsent_bytes(conn.sock):
        tensor = torch.zeros(64 << 10, dtype=torch.uint8)
        calls.append(asyncio.ensure_future(client.call("echo", tensor, timeout_s=301)))
        await _until(
            lambda: len(conn.pending) == len(calls)
            and conn.sending is None
            and not conn.outbox
        )
    for call in calls:
        call.cancel()
    for call in calls:
        with pytest.raises(asyncio.CancelledError):
            await call
    await _until(lambda: not conn.pending)
    await client.close()
    later = asyncio.ensure_future(CarrierClient(a).call("echo", "later"))
    await _until(lambda: server.stats()["queued"] == 2)
    target.release.set()
    assert await later == "later" and await held == "held"
    # The closed connection's reader is gone, so every frame it would read has run.
    await _until(lambda: filled.closed)
    await _until(lambda: _counts(server)[1:] == (0, 0))
    assert _no_tensor_after(target.calls, "later"), target.calls


@pytest.mark.asyncio
async def test_processes_threads_and_loops_run_one_call_at_a_time(
    target: _Target, children: _Children
) -> None:
    """Serve 3 processes x 4 threads x one ``asyncio.run`` per call, one call at a time."""
    server = await _serve(target)
    per_thread = 10
    spawned = [
        children.spawn(_THREADS_CHILD, server.address.to_json(), str(per_thread))
        for _ in range(3)
    ]
    for child in spawned:
        assert _read_line(child, timeout_s=90) == "ready"
    for child in spawned:
        stdin = child.stdin
        assert stdin is not None
        stdin.write("go\n")
        stdin.flush()
    for child in spawned:
        assert json.loads(_read_line(child, timeout_s=90)) == []
    calls = [call for call in target.calls if call[1] >= 0]
    assert len(calls) == 3 * 4 * per_thread and target.max_active == 1
    order: dict[tuple[int, int], list[int]] = {}
    for pid, thread, i in calls:
        order.setdefault((pid, thread), []).append(i)
    assert list(order.values()) == [list(range(per_thread))] * 12
    assert server.stats()["connections"] == 3


@pytest.mark.asyncio
async def test_calls_on_one_connection_run_in_the_order_sent(target: _Target) -> None:
    """Run calls sent on one connection in the order they were sent."""
    server = await _serve(target)
    client = CarrierClient(server.address)
    replies = await asyncio.gather(*(client.call("echo", i) for i in range(20)))
    assert replies == list(range(20))
    assert target.calls == list(range(20))


@pytest.mark.asyncio
async def test_one_connection_per_address_per_process(
    target: _Target, caplog: pytest.LogCaptureFixture
) -> None:
    """Share one connection among the handles that eight threads make for one address."""
    caplog.set_level(logging.INFO, logger="torchstore.carrier")
    server = await _serve(target)

    def call(i: int) -> Any:
        return asyncio.run(CarrierClient(server.address).call("echo", i))

    replies = await asyncio.gather(*(asyncio.to_thread(call, i) for i in range(8)))
    assert sorted(replies) == list(range(8))
    assert server.stats()["connections"] == 1
    connects = [
        r for r in caplog.records if r.getMessage().startswith("carrier connect")
    ]
    assert len(connects) == 1


@pytest.mark.asyncio
async def test_both_ends_set_nodelay_and_keepalive(target: _Target) -> None:
    """Set TCP_NODELAY and keepalive, with its timings, on both ends of a connection, but not TCP_USER_TIMEOUT."""
    server = await _serve(target)
    client = CarrierClient(server.address)
    assert await client.call("echo", 1) == 1
    conn = client._peer.conn
    assert conn is not None
    [accepted] = server._connections
    for sock in (conn.sock, accepted.sock):
        assert sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY)
        assert sock.getsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE)
        for name, value in runtime._KEEPALIVE:
            if hasattr(socket, name):
                option = getattr(socket, name)
                assert sock.getsockopt(socket.IPPROTO_TCP, option) == value
        if hasattr(socket, "TCP_USER_TIMEOUT"):
            assert not sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_USER_TIMEOUT)


def _can_bind(family: socket.AddressFamily, host: str) -> bool:
    """Whether this host has the address ``host``."""
    try:
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            sock.bind((host, 0))
    except OSError:
        return False
    return True


@pytest.mark.asyncio
async def test_a_server_answers_on_each_loopback_family(target: _Target) -> None:
    """Answer calls dialed at the IPv4 and the IPv6 loopback address, as a dual-stack listener does."""
    server = await _serve(target)
    a = server.address
    loopbacks = ((socket.AF_INET, "127.0.0.1"), (socket.AF_INET6, "::1"))
    hosts = [host for family, host in loopbacks if _can_bind(family, host)]
    assert hosts
    for host in hosts:
        dialed = CarrierAddress(host, a.port, a.service, a.scope, a.incarnation)
        assert await CarrierClient(dialed).call("echo", host, timeout_s=10) == host


@pytest.mark.asyncio
async def test_a_server_listens_on_every_interface_whatever_its_host(
    target: _Target,
) -> None:
    """Listen on the wildcard address though the server advertises a loopback host, so its name reaches it too."""
    server = await _serve(target, host="127.0.0.1")
    assert server._listener.getsockname()[0] in ("::", "0.0.0.0")
    a = server.address
    dialed = CarrierAddress(
        socket.gethostname(), a.port, a.service, a.scope, a.incarnation
    )
    assert await CarrierClient(dialed).call("echo", "named", timeout_s=10) == "named"


@pytest.mark.asyncio
async def test_ipv4_dials_reach_a_server_where_ipv6_sockets_default_to_ipv6_only(
    monkeypatch: pytest.MonkeyPatch, target: _Target
) -> None:
    """Answer an IPv4 dial where new IPv6 sockets start IPv6-only, as net.ipv6.bindv6only=1 makes them."""
    plain = socket.socket

    class V6OnlyByDefault(plain):  # type: ignore[valid-type, misc]
        def __init__(
            self, family: int = -1, kind: int = -1, proto: int = -1, fileno: Any = None
        ) -> None:
            super().__init__(family, kind, proto, fileno)
            if fileno is None and self.family == socket.AF_INET6:
                self.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)

    monkeypatch.setattr(socket, "socket", V6OnlyByDefault)
    server = await _serve(target)
    a = server.address
    dialed = CarrierAddress("127.0.0.1", a.port, a.service, a.scope, a.incarnation)
    assert await CarrierClient(dialed).call("echo", 1, timeout_s=10) == 1


@pytest.mark.asyncio
async def test_serve_falls_back_to_ipv4(
    monkeypatch: pytest.MonkeyPatch, target: _Target
) -> None:
    """Serve on IPv4 alone where this host cannot make IPv6 sockets."""
    plain = socket.socket

    class NoIPv6(plain):  # type: ignore[valid-type, misc]
        def __init__(
            self, family: int = -1, kind: int = -1, proto: int = -1, fileno: Any = None
        ) -> None:
            if fileno is None and family == socket.AF_INET6:
                raise OSError(errno.EAFNOSUPPORT, "Address family not supported")
            super().__init__(family, kind, proto, fileno)

    monkeypatch.setattr(socket, "socket", NoIPv6)
    server = await _serve(target)
    monkeypatch.undo()
    a = server.address
    dialed = CarrierAddress("127.0.0.1", a.port, a.service, a.scope, a.incarnation)
    assert await CarrierClient(dialed).call("echo", 1, timeout_s=10) == 1


@pytest.mark.asyncio
async def test_connecting_tries_each_resolved_address(
    monkeypatch: pytest.MonkeyPatch, target: _Target
) -> None:
    """Connect through the next resolved address when the first refuses."""
    server = await _serve(target)
    a = server.address
    # Bound but not listening, so the port refuses connections and stays ours.
    with socket.socket() as bound:
        bound.bind(("127.0.0.1", 0))
        refusing = bound.getsockname()[1]

        def two(host: str, port: int) -> list[Any]:
            first = socket.getaddrinfo("127.0.0.1", refusing, type=socket.SOCK_STREAM)
            return first + socket.getaddrinfo(
                "127.0.0.1", port, type=socket.SOCK_STREAM
            )

        monkeypatch.setattr(runtime, "_getaddrinfo", two)
        dialed = CarrierAddress(
            "two.invalid", a.port, a.service, a.scope, a.incarnation
        )
        assert await CarrierClient(dialed).call("echo", 1, timeout_s=10) == 1


@pytest.mark.asyncio
async def test_connecting_skips_an_address_whose_socket_cannot_be_made(
    monkeypatch: pytest.MonkeyPatch, target: _Target
) -> None:
    """Connect over IPv4 where policy refuses IPv6 stream sockets yet the resolver lists IPv6 first.

    An AppArmor rule refusing ``inet6 stream``, say: the resolver's datagram probe passes.
    """
    server = await _serve(target)
    a = server.address
    plain = socket.socket

    class NoIPv6Streams(plain):  # type: ignore[valid-type, misc]
        def __init__(
            self, family: int = -1, kind: int = -1, proto: int = -1, fileno: Any = None
        ) -> None:
            if fileno is None and (family, kind) == (
                socket.AF_INET6,
                socket.SOCK_STREAM,
            ):
                raise PermissionError(errno.EACCES, "Permission denied")
            super().__init__(family, kind, proto, fileno)

    def both(host: str, port: int) -> list[Any]:
        tcp = socket.IPPROTO_TCP
        return [
            (socket.AF_INET6, socket.SOCK_STREAM, tcp, "", ("::1", port, 0, 0)),
            (socket.AF_INET, socket.SOCK_STREAM, tcp, "", ("127.0.0.1", port)),
        ]

    monkeypatch.setattr(runtime, "_getaddrinfo", both)
    monkeypatch.setattr(socket, "socket", NoIPv6Streams)
    dialed = CarrierAddress("dual.invalid", a.port, a.service, a.scope, a.incarnation)
    assert await CarrierClient(dialed).call("echo", 1, timeout_s=10) == 1


@pytest.mark.asyncio
async def test_a_served_host_is_resolved_once_on_the_resolver_thread(
    monkeypatch: pytest.MonkeyPatch, target: _Target
) -> None:
    """Advertise the given host, and connect to its numeric address, so no executor thread resolves it again."""
    real = socket.getaddrinfo
    elsewhere: list[str] = []

    def resolve(host: Any, *args: Any, **kwargs: Any) -> Any:
        if host == "carrier.invalid":
            name = threading.current_thread().name
            if not name.startswith("torchstore-carrier-resolve"):
                elsewhere.append(name)
            host = "127.0.0.1"
        return real(host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    server = await _serve(target, host="carrier.invalid")
    assert server.address.host == "carrier.invalid"
    assert await CarrierClient(server.address).call("echo", 1, timeout_s=10) == 1
    assert elsewhere == []


@pytest.mark.asyncio
async def test_serve_advertises_this_hosts_name_unless_given_a_host(
    target: _Target,
) -> None:
    """Advertise this host's name, which clients on other hosts dial, unless serve is given a host."""
    default = await _serve(target)
    named = await _serve(target, host="10.0.0.1")
    assert default.address.host == socket.gethostname()
    assert named.address.host == "10.0.0.1"


@pytest.mark.asyncio
async def test_serve_reports_a_taken_port_and_a_missing_method(target: _Target) -> None:
    """Raise ``OSError`` with ``EADDRINUSE`` for a port in use, and ``ValueError`` for a missing method."""
    server = await _serve(target)
    port = server.address.port
    with pytest.raises(OSError, match=f"on port {port}") as error:
        await _serve(target, port=port)
    assert error.value.errno == errno.EADDRINUSE
    with pytest.raises(ValueError, match=r"has no methods \['nope'\]"):
        await _serve(target, "echo", "nope")


@pytest.mark.asyncio
async def test_serve_refuses_a_non_str_service_scope_or_host_and_a_str_of_endpoints(
    monkeypatch: pytest.MonkeyPatch, target: _Target, caplog: pytest.LogCaptureFixture
) -> None:
    """Refuse a non-str ``service``, ``scope`` or ``host``, a bad ``port``, and endpoints given as one str, before listening.

    Unchecked, a non-str service or scope served, but each dial raised ``StaleAddressError``, the error for a restarted server.
    """
    caplog.set_level(logging.INFO, logger="torchstore.carrier")

    def listen(port: int) -> socket.socket:
        raise AssertionError("serve listened")

    monkeypatch.setattr(runtime, "_listen", listen)
    for changes, error, message in (
        ({"service": 0}, TypeError, "^service must be a str, not int$"),
        ({"scope": 7}, TypeError, "^scope must be a str, not int$"),
        ({"host": 123}, TypeError, "^host must be a str, not int$"),
        ({"port": "1"}, TypeError, "^port must be an int, not str$"),
        ({"port": True}, TypeError, "^port must be an int, not bool$"),
        ({"port": 65536}, ValueError, "^port must be from 0 to 65535, not 65536$"),
        ({"port": -1}, ValueError, "^port must be from 0 to 65535, not -1$"),
        # Both ends pass the check and serve goes on to listen.
        ({"port": 0}, AssertionError, "^serve listened$"),
        ({"port": 65535}, AssertionError, "^serve listened$"),
    ):
        kwargs: dict[str, Any] = {"service": "test", "scope": _scope()}
        kwargs.update(changes)
        with pytest.raises(error, match=message):
            await CarrierServer.serve(target, ["echo"], **kwargs)
    with pytest.raises(
        TypeError, match="^endpoints must be a collection of names, not the str 'echo'$"
    ):
        await CarrierServer.serve(target, "echo", service="test", scope=_scope())  # type: ignore[arg-type]
    assert not [r for r in caplog.records if r.getMessage().startswith("carrier serve")]


def test_listening_closes_its_socket_when_binding_raises_any_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Close each socket a listen made when its bind raises, an ``OSError`` for a port in use or another error for a bad port."""
    made: list[socket.socket] = []

    class Recording(socket.socket):
        def bind(self, address: Any) -> None:
            made.append(self)
            super().bind(address)

    with runtime._listen(0) as busy:
        monkeypatch.setattr(socket, "socket", Recording)
        with pytest.raises(OSError):
            runtime._listen(busy.getsockname()[1])
        assert made and all(sock.fileno() == -1 for sock in made)
        made.clear()
        with pytest.raises(OverflowError):
            runtime._listen(70000)
        assert made and all(sock.fileno() == -1 for sock in made)


@pytest.mark.asyncio
async def test_handles_and_servers_refuse_misuse(target: _Target) -> None:
    """Refuse a client for anything but a ``CarrierAddress``, and pickling a server."""
    with pytest.raises(TypeError, match="need a CarrierAddress, not tuple"):
        CarrierClient(("host", 1))  # type: ignore[arg-type]
    server = await _serve(target)
    with pytest.raises(TypeError, match="cannot be pickled"):
        pickle.dumps(server)


@pytest.mark.asyncio
@pytest.mark.parametrize("concurrent", [False, True])
async def test_replies_are_copied_before_the_next_call_runs(concurrent: bool) -> None:
    """Keep a reply's bytes when the next call changes its tensor before the reply is sent."""
    store = _Store()
    server = await _serve(store, "get", "fill", concurrent=concurrent)
    client = CarrierClient(server.address)
    await server._set_sending(False)
    get = asyncio.ensure_future(client.call("get"))
    await _until(lambda: _counts(server) == (1, 0, 0))
    fill = asyncio.ensure_future(client.call("fill", 7.0))
    await _until(lambda: bool(store.tensor[0] == 7.0))
    await server._set_sending(True)
    assert torch.equal(await get, torch.zeros(1 << 16))
    assert await fill == 7.0


@pytest.mark.asyncio
async def test_a_bytes_reply_goes_to_every_caller_uncopied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Send a ``PickleBuffer`` over ``bytes`` without a copy, while tensor replies are copied."""
    copy_mutable = runtime._copy_mutable
    sent: list[list[tuple[Any, Any]]] = []

    def spy(buffers: list[memoryview]) -> list[memoryview]:
        copies = copy_mutable(buffers)
        sent.append([(buffer.obj, copy.obj) for buffer, copy in zip(buffers, copies)])
        return copies

    monkeypatch.setattr(runtime, "_copy_mutable", spy)
    store = _Store()
    server = await _serve(store, "get", "payload")
    client = CarrierClient(server.address)
    for reply in await asyncio.gather(*(client.call("payload") for _ in range(3))):
        assert isinstance(reply, memoryview) and reply.readonly
        assert bytes(reply) == store.data
    assert [[copy for _, copy in buffers] for buffers in sent] == [[store.data]] * 3
    assert all(copy is store.data for buffers in sent for _, copy in buffers)
    assert torch.equal(await client.call("get"), store.tensor)
    [(original, copy)] = sent[-1]
    assert copy is not original


@pytest.mark.asyncio
async def test_the_codecs_own_copies_skip_the_reply_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Copy a reply that views the served object's tensor, but not the codec's copy of a column.

    Either reply keeps its values when the next call fills the tensor before the reply is sent.
    """
    copy_mutable = runtime._copy_mutable
    copied: list[list[bool]] = []

    def spy(buffers: list[memoryview]) -> list[memoryview]:
        copies = copy_mutable(buffers)
        copied.append([copy is not buffer for buffer, copy in zip(buffers, copies)])
        return copies

    monkeypatch.setattr(runtime, "_copy_mutable", spy)
    big = _Big()
    server = await _serve(big, "column", "row", "fill")
    client = CarrierClient(server.address)
    for value, endpoint in ((1.0, "column"), (2.0, "row")):
        before, made = getattr(big, endpoint)().clone(), len(copied)
        await server._set_sending(False)
        reply = asyncio.ensure_future(client.call(endpoint))
        await _until(lambda: len(copied) > made)
        fill = asyncio.ensure_future(client.call("fill", value))
        await _until(lambda: bool(big.tensor[0, 0] == value))
        await server._set_sending(True)
        assert torch.equal(await reply, before) and await fill == value
    assert [buffers for buffers in copied if buffers] == [[False], [True]]


def _writer_jobs(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Return the names of the threads that run writer jobs from now on, one per job."""
    ran: list[str] = []
    run = runtime._SendJob.run

    def spy(job: runtime._SendJob) -> None:
        ran.append(threading.current_thread().name)
        run(job)

    monkeypatch.setattr(runtime._SendJob, "run", spy)
    return ran


@pytest.fixture
def held_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[threading.Event, threading.Event]]:
    """Hold the first ``send()`` a writer thread makes until released. Yield (copying, release)."""
    copying, release = threading.Event(), threading.Event()
    send = socket.socket.send

    def held_send(sock: socket.socket, data: Any, *args: Any) -> int:
        writer = threading.current_thread().name.startswith("torchstore-carrier-writer")
        if writer and not copying.is_set():
            copying.set()
            release.wait(30)
        return send(sock, data, *args)

    monkeypatch.setattr(socket.socket, "send", held_send)
    yield copying, release
    release.set()


def _raw_reply(sock: socket.socket) -> tuple[int, Any]:
    """Read a reply frame from a raw connection. Return its request id and its value."""
    header = codec.decode_header(_receive(sock, codec.HEADER_SIZE), handshake=False)
    assert header.kind == Kind.REPLY, header
    lengths = codec.decode_lengths(
        header, _receive(sock, codec.LENGTH_SIZE * header.buffer_count)
    )
    body = _receive(sock, header.body_len)
    buffers = [
        torch.frombuffer(_receive_into(sock, n), dtype=torch.uint8) for n in lengths
    ]
    return header.request_id, codec.loads(body, buffers)


@contextlib.asynccontextmanager
async def _stalled_get(
    server: CarrierServer,
) -> AsyncIterator[tuple[Any, socket.socket]]:
    """Keep a raw client's ``get`` reply unread, so it fills its socket, until the body ends.

    Yields the server's end of that connection and the client's socket, then reads the whole
    reply and checks it, unless the body closed that socket.
    """
    a = server.address
    before = set(server._connections)
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 128 << 10)
        sock.settimeout(30)
        sock.connect(("127.0.0.1", a.port))
        _send_frame(sock, Kind.HELLO, body=_hello(a))
        assert _handshake_reply(sock)[0] == Kind.WELCOME
        body, _ = codec.dumps(("get", (), {}))
        _send_frame(sock, Kind.REQUEST, 1, deadline_us=10**8, body=body)
        [conn] = set(server._connections) - before
        await _until(lambda: conn.unsent == 1 and _unsent_bytes(conn.sock) > 0)
        yield conn, sock
        if sock.fileno() >= 0:
            request_id, reply = _raw_reply(sock)
            assert request_id == 1 and bool((reply == 1).all())


@pytest.mark.asyncio
@pytest.mark.parametrize("concurrent", [False, True])
async def test_overlapping_large_replies_go_out_on_writer_threads(
    monkeypatch: pytest.MonkeyPatch, concurrent: bool
) -> None:
    """Send a lone large reply on the I/O thread, and large replies that overlap another on writers, intact."""
    ran = _writer_jobs(monkeypatch)
    server = await _serve(_Sink(64 << 20), "get", "echo", concurrent=concurrent)
    client = CarrierClient(server.address)
    big = torch.randint(0, 256, (3 << 20,), dtype=torch.uint8)
    assert torch.equal(await client.call("echo", big), big)
    assert ran == []
    async with _stalled_get(server):
        small = torch.ones(1 << 18, dtype=torch.uint8)
        assert torch.equal(await client.call("echo", small), small)
        assert ran == []
        values = [torch.full((3 << 20,), i, dtype=torch.uint8) for i in range(8)]
        replies = await asyncio.gather(*(client.call("echo", v) for v in values))
        assert all(torch.equal(reply, v) for reply, v in zip(replies, values))
    assert len(ran) >= len(values)
    assert all(name.startswith("torchstore-carrier-writer ") for name in ran)


@pytest.mark.asyncio
@pytest.mark.parametrize("concurrent", [False, True])
async def test_one_connections_large_replies_go_out_whole_and_in_order(
    monkeypatch: pytest.MonkeyPatch, concurrent: bool
) -> None:
    """Send one connection's large replies on writer threads one frame at a time, in request order."""
    ran = _writer_jobs(monkeypatch)
    server = await _serve(_Sink(64 << 20), "get", "echo", concurrent=concurrent)
    values = [torch.full((3 << 20,), i, dtype=torch.uint8) for i in range(8)]
    async with _stalled_get(server):
        with _raw_connection(server.address) as sock:
            _send_frame(sock, Kind.HELLO, body=_hello(server.address))
            assert _handshake_reply(sock)[0] == Kind.WELCOME
            for request_id, value in enumerate(values, 1):
                body, buffers = codec.dumps(("echo", (value,), {}))
                _send_frame(
                    sock,
                    Kind.REQUEST,
                    request_id,
                    deadline_us=10**8,
                    body=body,
                    buffers=buffers,
                )
            replies = [_raw_reply(sock) for _ in values]
    assert [request_id for request_id, _ in replies] == list(range(1, len(values) + 1))
    assert all(torch.equal(reply, value) for (_, reply), value in zip(replies, values))
    assert len(ran) >= len(values)


@pytest.mark.asyncio
async def test_overlapping_replies_stay_on_the_io_thread_when_no_writer_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Send large replies from the I/O thread, as without writers, when no writer thread can start."""
    start_thread = threading.Thread.start

    def start(thread: threading.Thread) -> None:
        if thread.name.startswith("torchstore-carrier-writer"):
            raise RuntimeError("can't start new thread")
        start_thread(thread)

    ran = _writer_jobs(monkeypatch)
    server = await _serve(_Sink(64 << 20), "get", "echo")
    monkeypatch.setattr(threading.Thread, "start", start)
    client = CarrierClient(server.address)
    async with _stalled_get(server):
        big = torch.randint(0, 256, (8 << 20,), dtype=torch.uint8)
        assert torch.equal(await client.call("echo", big), big)
    assert ran == [] and not server._runtime.writers.threads


@pytest.mark.asyncio
async def test_a_client_that_stops_reading_holds_no_writer_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Send a reply on the only writer while another client's reply, also on writers, fills its socket.

    A writer copies only while a socket has room. A full socket waits for room on the I/O thread.
    """
    monkeypatch.setattr(runtime, "_WRITER_THREADS", 1)
    size = 64 << 20
    server = await _serve(_Sink(size), "get")
    # The first stalled reply is on the I/O thread, so the second one goes to the writer.
    async with _stalled_get(server), _stalled_get(server) as (stalled, _):
        # The second setup can return before its reply's writer thread has started.
        await _until(lambda: len(server._runtime.writers.threads) == 1)
        await _until(lambda: stalled.job is None)
        reply = await CarrierClient(server.address).call("get", timeout_s=30)
        assert reply.numel() == size and bool((reply == 1).all())
        assert stalled.unsent == 1


def _waits_for_room(monkeypatch: pytest.MonkeyPatch) -> list[socket.socket]:
    """Return the sockets whose senders wait for room now, as the list of each wait still in progress."""
    waiting: list[socket.socket] = []
    writable = runtime._writable

    async def recorded(loop: asyncio.AbstractEventLoop, sock: socket.socket) -> None:
        waiting.append(sock)
        try:
            await writable(loop, sock)
        finally:
            waiting.remove(sock)

    monkeypatch.setattr(runtime, "_writable", recorded)
    return waiting


@pytest.mark.asyncio
@pytest.mark.parametrize("stalled", [1, 2])
async def test_a_reply_left_unread_waits_for_room_without_retrying(
    monkeypatch: pytest.MonkeyPatch, stalled: int
) -> None:
    """Send no more into a full socket until it has room: a lone reply on the I/O thread, a second on writers."""
    sends: list[int] = []
    send = socket.socket.send

    def counted(sock: socket.socket, data: Any, *args: Any) -> int:
        # Not the event loops' wakeup pipes.
        if sock.family != socket.AF_UNIX:
            sends.append(len(data))
        return send(sock, data, *args)

    waiting = _waits_for_room(monkeypatch)
    monkeypatch.setattr(socket.socket, "send", counted)
    server = await _serve(_Sink(64 << 20), "get")
    async with contextlib.AsyncExitStack() as stack:
        for _ in range(stalled):
            await stack.enter_async_context(_stalled_get(server))
        # Each stalled reply's sender waits for room on the I/O thread.
        await _until(lambda: len(waiting) == stalled)
        before = len(sends)
        # A sender that retried a full socket at every turn would send again at this turn of the I/O loop.
        await runtime._call_on(server._runtime.loop, lambda: None)
        assert len(sends) == before and len(waiting) == stalled


@pytest.mark.asyncio
async def test_a_request_left_unread_waits_for_room_without_retrying(
    monkeypatch: pytest.MonkeyPatch, target: _Target
) -> None:
    """Send no more of a request into a full socket until it has room, as a stopped reader leaves it."""
    _no_window(monkeypatch)
    monkeypatch.setattr(runtime, "_QUEUE_BOUND", 1)
    waiting = _waits_for_room(monkeypatch)
    sends: list[int] = []
    send = socket.socket.send

    def counted(sock: socket.socket, data: Any, *args: Any) -> int:
        # Not the event loops' wakeup pipes.
        if sock.family != socket.AF_UNIX:
            sends.append(len(data))
        return send(sock, data, *args)

    monkeypatch.setattr(socket.socket, "send", counted)
    server = await _serve(target)
    client = CarrierClient(server.address)
    held = asyncio.ensure_future(client.call("hold", "held", timeout_s=60))
    await _until(lambda: server.stats()["running"] == 1)
    queued = asyncio.ensure_future(client.call("echo", 0, timeout_s=60))
    await _until(lambda: server.stats()["queued"] == 1)
    tensor = torch.ones(64 << 20, dtype=torch.uint8)
    large = asyncio.ensure_future(client.call("echo", tensor, timeout_s=60))
    conn = client._peer.conn
    assert conn is not None
    await _until(lambda: conn.sending is not None and len(waiting) == 1)
    before = len(sends)
    # A sender that retried a full socket at every turn would send again at this turn of the I/O loop.
    await runtime._call_on(server._runtime.loop, lambda: None)
    assert len(sends) == before and len(waiting) == 1
    target.release.set()
    assert await held == "held" and await queued == 0
    assert torch.equal(await large, tensor)


class _AlwaysReady:
    """A socket stand-in that always has bytes ready, as from a sender that keeps up, ``step`` at most per receive."""

    def __init__(self, total: int, step: int = 1 << 20) -> None:
        self.left, self.step = total, step

    def recv_into(self, view: memoryview) -> int:
        count = min(len(view), self.step, self.left)
        self.left -= count
        return count


@pytest.mark.asyncio
@pytest.mark.parametrize("reads", [1, 32])
async def test_a_reader_gives_the_loop_turns_while_bytes_keep_arriving(
    monkeypatch: pytest.MonkeyPatch, reads: int
) -> None:
    """Let other callbacks run after each turn's bytes though no receive ever waits, in one read or across many smaller ones."""
    monkeypatch.setattr(runtime, "_WRITE_TURN_BYTES", 1 << 20)
    reader = runtime._Reader(asyncio.get_running_loop(), _AlwaysReady(16 << 20))  # type: ignore[arg-type]

    async def read() -> None:
        for _ in range(reads):
            await reader.read_into(memoryview(bytearray((16 << 20) // reads)))

    beats = await _beats_during(read())
    assert beats >= 15, beats


@pytest.mark.asyncio
async def test_a_reader_takes_one_turn_per_turns_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Take a turn once per turn's bytes, though the bytes come in smaller receives, not at every receive after the first."""
    monkeypatch.setattr(runtime, "_WRITE_TURN_BYTES", 1 << 20)
    ready = _AlwaysReady(16 << 20, step=256 << 10)
    reader = runtime._Reader(asyncio.get_running_loop(), ready)  # type: ignore[arg-type]
    turns: list[int] = []
    take_turn = reader.take_turn

    async def counted() -> None:
        turns.append(reader.received)
        await take_turn()

    reader.take_turn = counted  # type: ignore[method-assign]
    await reader.read_into(memoryview(bytearray(16 << 20)))
    assert len(turns) == 15, turns


class _Frame:
    """A socket stand-in holding one frame's bytes, all ready at once, as from a sender that keeps up."""

    def __init__(self, data: bytes) -> None:
        self.data, self.start = data, 0

    def recv_into(self, view: memoryview) -> int:
        count = min(len(view), len(self.data) - self.start)
        view[:count] = self.data[self.start : self.start + count]
        self.start += count
        return count


async def _beats_during(work: Coroutine[Any, Any, Any]) -> int:
    """Run ``work`` and return how many turns other callbacks got meanwhile."""
    loop = asyncio.get_running_loop()
    beats, beating = [0], [True]

    def beat() -> None:
        if beating[0]:
            beats[0] += 1
            loop.call_soon(beat)

    loop.call_soon(beat)
    await work
    beating[0] = False
    return beats[0]


@pytest.mark.asyncio
async def test_a_reader_gives_the_loop_turns_while_it_reads_many_small_buffers() -> None:
    """Let other callbacks run while a frame of many empty buffers is read, though no buffer needs a receive."""
    count = 8 * runtime._BUFFER_TURN
    data = b"".join(
        bytes(chunk)
        for chunk in codec.encode_frame(Kind.REPLY, buffers=[memoryview(b"")] * count)
    )
    reader = runtime._Reader(asyncio.get_running_loop(), _Frame(data))  # type: ignore[arg-type]
    frame = runtime._read_frame(reader)
    assert await _beats_during(frame) >= 7


@pytest.mark.asyncio
async def test_a_sender_gives_the_loop_turns_while_it_sends_many_small_buffers() -> None:
    """Let other callbacks run while a frame of many small buffers goes out, though no send waits for room."""
    chunks = codec.encode_frame(
        Kind.REPLY, buffers=[memoryview(b"x")] * (8 * runtime._BUFFER_TURN)
    )
    sending = runtime._send_chunks(asyncio.get_running_loop(), _TakesAll(), chunks)  # type: ignore[arg-type]
    assert await _beats_during(sending) >= 7


@pytest.mark.asyncio
async def test_a_small_frame_goes_out_in_one_send() -> None:
    """Send a frame's header and small body in one write, as a second would cost a second system call."""
    sock = _TakesAll()
    chunks = codec.encode_frame(Kind.REPLY, 1, body=b"x" * 100)
    await runtime._send_chunks(asyncio.get_running_loop(), sock, chunks)  # type: ignore[arg-type]
    assert sock.sends == [runtime._frame_size(chunks)]


def test_a_resolved_address_loses_its_ipv6_zone() -> None:
    """Strip the zone from a resolved IPv6 address, which asyncio would otherwise resolve again in its executor."""
    assert runtime._numeric(("fe80::1%eth0", 80, 0, 2)) == ("fe80::1", 80, 0, 2)
    assert runtime._numeric(("127.0.0.1", 80)) == ("127.0.0.1", 80)


@pytest.mark.skipif(not _can_bind(socket.AF_INET6, "::1"), reason="no IPv6 loopback")
@pytest.mark.asyncio
async def test_a_connect_to_a_zoned_address_starts_no_executor_thread(
    monkeypatch: pytest.MonkeyPatch, target: _Target
) -> None:
    """Dial a looked-up IPv6 address that keeps its zone without asyncio resolving it again on an executor thread."""
    server = await _serve(target)
    zoned = [
        (
            socket.AF_INET6,
            socket.SOCK_STREAM,
            6,
            "",
            ("::1%lo", server.address.port, 0, 1),
        )
    ]
    monkeypatch.setattr(runtime, "_getaddrinfo", lambda host, port: zoned)
    loop = server._runtime.loop
    resolved_again: list[Any] = []

    async def getaddrinfo(*args: Any, **kwargs: Any) -> Any:
        resolved_again.append(args)
        return await type(loop).getaddrinfo(loop, *args, **kwargs)

    monkeypatch.setattr(loop, "getaddrinfo", getaddrinfo)
    assert await CarrierClient(server.address).call("echo", 1, timeout_s=10) == 1
    assert resolved_again == []


@pytest.mark.asyncio
async def test_a_writer_job_copies_at_most_one_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hand a large reply to the writers in turns of ``_WRITE_TURN_BYTES``, though its reader keeps up."""
    monkeypatch.setattr(runtime, "_WRITE_TURN_BYTES", 64 << 10)
    turns: list[int] = []
    settle = runtime._SendJob._settle

    def spy(
        job: runtime._SendJob,
        sent: int,
        full: bool,
        error: Exception | None,
        held: list[memoryview],
    ) -> None:
        turns.append(sent)
        settle(job, sent, full, error, held)

    monkeypatch.setattr(runtime._SendJob, "_settle", spy)
    server = await _serve(_Sink(16 << 20), "get")
    async with _stalled_get(server):
        reply = await CarrierClient(server.address).call("get", timeout_s=30)
    assert reply.numel() == 16 << 20 and bool((reply == 1).all())
    assert turns and max(turns) <= 64 << 10, max(turns)


def test_stopped_writers_leave_later_jobs_a_queue_of_their_own() -> None:
    """Stop each writer once the jobs before the stop ran, and give writers started later a queue the stopped ones never read.

    Every writer is busy as the stop comes so the jobs queued behind them are still waiting then.
    """
    writers = runtime._Writers()
    assert writers.start()
    busy, release = threading.Barrier(len(writers.threads) + 1), threading.Event()
    ran: list[str] = []

    class Job:
        def __init__(self, label: str) -> None:
            self.label = label

        def run(self) -> None:
            if self.label == "busy":
                busy.wait(10)
                release.wait(10)
            ran.append(self.label)

    old_jobs = writers.jobs
    for _ in writers.threads:
        old_jobs.put(Job("busy"))  # type: ignore[arg-type]
    busy.wait(10)
    for label in ("first", "second"):
        old_jobs.put(Job(label))  # type: ignore[arg-type]
    stopped = writers.stop()
    release.set()
    assert stopped and writers.jobs is not old_jobs
    for thread, done in stopped.items():
        done.result(10)
        thread.join(10)
    assert sorted(ran) == ["busy"] * len(stopped) + ["first", "second"]
    assert writers.start() and old_jobs.empty()
    for thread, done in writers.stop().items():
        done.result(10)
        thread.join(10)


@pytest.mark.asyncio
async def test_an_idle_writer_holds_no_job(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop each job a writer ran, so an idle writer holds no connection's socket through it."""
    done: list[weakref.ref[asyncio.Future[Any]]] = []
    run = runtime._SendJob.run

    def recorded(job: runtime._SendJob) -> None:
        # Jobs take no weak references, but their futures do.
        done.append(weakref.ref(job.done))
        run(job)

    monkeypatch.setattr(runtime._SendJob, "run", recorded)
    server = await _serve(_Sink(64 << 20), "get", "echo")
    async with _stalled_get(server):
        value = torch.full((3 << 20,), 5, dtype=torch.uint8)
        assert torch.equal(
            await CarrierClient(server.address).call("echo", value), value
        )
        assert done
        await _until(lambda: all(future() is None for future in done))


@pytest.mark.asyncio
async def test_a_connection_closed_during_a_copy_closes_its_socket_after_the_copy(
    held_copy: tuple[threading.Event, threading.Event],
) -> None:
    """Keep a closed connection's socket open while a writer copies into it, then close it.

    Closed under the copy, its number could pass to a new socket, which the copy would then write.
    """
    copying, release = held_copy
    server = await _serve(_Sink(64 << 20), "get")
    client = CarrierClient(server.address)
    async with _stalled_get(server):
        get = asyncio.ensure_future(client.call("get", timeout_s=30))
        await _until(copying.is_set)
        [conn] = [c for c in server._connections if c.job is not None]
        sock = conn.sock
        await client.close()
        with pytest.raises(PeerLostError):
            await get
        await _until(lambda: conn.closed)
        assert sock.fileno() >= 0
        release.set()
        await _until(lambda: sock.fileno() == -1)


@pytest.mark.asyncio
@pytest.mark.parametrize("loop_closed", [False, True])
async def test_a_writer_never_frees_a_reply_copy(
    monkeypatch: pytest.MonkeyPatch, loop_closed: bool
) -> None:
    """Free a reply copy whose connection closed under a writer's copy on the I/O thread, even if it settles first.

    Torch's free retakes the GIL, which a writer, never waited for at exit, cannot do while Python finalizes.
    Once the I/O loop has closed, as at exit, the writer keeps the copy instead.
    """
    copying, release, settled = threading.Event(), threading.Event(), threading.Event()
    send, post, settle = socket.socket.send, runtime._post, runtime._SendJob._settle

    def settle_spy(job: runtime._SendJob, *args: Any) -> None:
        settle(job, *args)
        settled.set()

    def post_and_wait(loop: asyncio.AbstractEventLoop, fn: Any, *args: Any) -> bool:
        if getattr(fn, "__func__", None) is not settle_spy:
            return post(loop, fn, *args)
        if loop_closed:
            return False
        # The writer moves on only once the I/O thread has settled its turn.
        assert post(loop, fn, *args) and settled.wait(30)
        return True

    def held_send(sock: socket.socket, data: Any, *args: Any) -> int:
        writer = threading.current_thread().name.startswith("torchstore-carrier-writer")
        if writer and not copying.is_set():
            copying.set()
            release.wait(30)
        try:
            return send(sock, data, *args)
        finally:
            # A raised error's traceback would otherwise keep the reply copy alive.
            del data

    freed: list[str] = []
    copy_mutable = runtime._copy_mutable

    def tracked(buffers: list[memoryview]) -> list[memoryview]:
        copies = copy_mutable(buffers)
        for buffer, copy in zip(buffers, copies):
            if copy is not buffer:
                weakref.finalize(
                    copy.obj._tensor,  # type: ignore[attr-defined]
                    lambda: freed.append(threading.current_thread().name),
                )
        return copies

    monkeypatch.setattr(socket.socket, "send", held_send)
    monkeypatch.setattr(runtime, "_copy_mutable", tracked)
    monkeypatch.setattr(runtime, "_post", post_and_wait)
    monkeypatch.setattr(runtime._SendJob, "_settle", settle_spy)
    server = await _serve(_Sink(64 << 20), "get")
    client = CarrierClient(server.address)
    try:
        async with _stalled_get(server):
            get = asyncio.ensure_future(client.call("get", timeout_s=30))
            await _until(copying.is_set)
            [conn] = [c for c in server._connections if c.job is not None]
            writer = conn.writer
            assert writer is not None
            await client.close()
            with pytest.raises(PeerLostError):
                await get
            await _until(writer.done)
            del conn, writer
            unfreed = len(runtime._unfreed)
            release.set()
            if loop_closed:
                await _until(lambda: len(runtime._unfreed) == unfreed + 1)
                assert freed == []
                del runtime._unfreed[unfreed:]
            else:
                await _until(lambda: bool(freed))
                assert freed == ["torchstore-carrier"], freed
    finally:
        release.set()


@pytest.mark.asyncio
async def test_close_lets_a_writer_finish_its_reply_and_the_last_close_stops_the_writers(
    monkeypatch: pytest.MonkeyPatch,
    held_copy: tuple[threading.Event, threading.Event],
) -> None:
    """Deliver a reply a writer is copying as its server closes, and stop the writers with the last server.

    Writers serve every server in the process, so they outlive a close while another server is open.
    """
    copying, release = held_copy
    server = await _serve(_Sink(64 << 20), "get")
    other = await _serve(_Target())
    client = CarrierClient(server.address)
    writers = server._runtime.writers
    async with _stalled_get(server):
        get = asyncio.ensure_future(client.call("get", timeout_s=60))
        await _until(copying.is_set)
        threads = list(writers.threads)
        closing = asyncio.ensure_future(server.close(grace_s=30))
        await _until(lambda: server._closing)
        release.set()
        reply = await get
        assert reply.numel() == 64 << 20 and bool((reply == 1).all())
    await closing
    assert list(writers.threads) == threads and all(t.is_alive() for t in threads)
    # Idle writers exit at once, so the last close never waits out the release wait.
    monkeypatch.setattr(runtime, "_RELEASE_WAIT_S", 60)
    start = time.monotonic()
    await other.close()
    assert time.monotonic() - start < 30
    assert not writers.threads and not any(t.is_alive() for t in threads)
    again = await _serve(_Sink(64 << 20), "get")
    async with _stalled_get(again):
        reply = await CarrierClient(again.address).call("get", timeout_s=60)
        assert reply.numel() == 64 << 20
    assert writers.threads and not set(writers.threads) & set(threads)


@pytest.mark.asyncio
async def test_a_writer_sends_no_more_once_its_connection_closes(
    monkeypatch: pytest.MonkeyPatch,
    held_copy: tuple[threading.Event, threading.Event],
) -> None:
    """Stop a writer at the copy a close found it in, though the socket still has room.

    A later copy would write the socket while the I/O thread closes it.
    """
    monkeypatch.setattr(runtime, "_RELEASE_WAIT_S", 1)
    copying, release = held_copy
    sends: list[socket.socket] = []
    held_send = socket.socket.send

    def spy(sock: socket.socket, data: Any, *args: Any) -> int:
        if threading.current_thread().name.startswith("torchstore-carrier-writer"):
            sends.append(sock)
        return held_send(sock, data, *args)

    monkeypatch.setattr(socket.socket, "send", spy)
    server = await _serve(_Sink(64 << 20), "get")
    client = CarrierClient(server.address)
    async with _stalled_get(server) as (_, stalled):
        get = asyncio.ensure_future(client.call("get", timeout_s=60))
        await _until(copying.is_set)
        [conn] = [c for c in server._connections if c.job is not None]
        sock, job = conn.sock, conn.job
        closing = asyncio.ensure_future(server.close(grace_s=0))
        await _until(lambda: job is not None and job.pending_close is not None)
        release.set()
        with pytest.raises(PeerLostError):
            await get
        await closing
        stalled.close()
    assert sends.count(sock) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("window", [True, False])
async def test_replies_on_writers_keep_a_bounded_connection_moving(
    monkeypatch: pytest.MonkeyPatch, window: bool
) -> None:
    """Finish calls past the queue bound and the unsent-byte share while writers send their replies.

    Without a window the reader stops at the bound, and with the hangup check off only the
    wakes as replies go out and calls start resume it.
    """
    monkeypatch.setattr(runtime, "_QUEUE_BOUND", 4)
    monkeypatch.setattr(runtime, "_UNSENT_BYTES_PER_CONNECTION", 4 << 20)
    monkeypatch.setattr(runtime, "_HANGUP_CHECK_S", 3600)
    if not window:
        _no_window(monkeypatch)
    ran = _writer_jobs(monkeypatch)
    server = await _serve(_Sink(64 << 20), "get", "echo")
    client = CarrierClient(server.address)
    values = [torch.full((3 << 20,), i, dtype=torch.uint8) for i in range(24)]
    async with _stalled_get(server):
        calls = (client.call("echo", value, timeout_s=60) for value in values)
        gathered: asyncio.Future[list[Any]] = asyncio.gather(*calls)
        replies = await asyncio.wait_for(gathered, 30)
    assert all(torch.equal(reply, value) for reply, value in zip(replies, values))
    assert len(ran) >= len(values)


@pytest.mark.asyncio
@pytest.mark.parametrize("concurrent", [False, True])
async def test_dispatch_passes_over_held_calls_and_in_fifo_mode_crowded_connections(
    target: _Target, concurrent: bool
) -> None:
    """Take the first call not held, passing over one whose connection has too many reply bytes unsent in FIFO mode alone."""
    server = await _serve(target, concurrent=concurrent)
    share = runtime._UNSENT_BYTES_PER_CONNECTION
    over = types.SimpleNamespace(unsent_bytes=share + 1)
    full = types.SimpleNamespace(unsent_bytes=share)
    under = types.SimpleNamespace(unsent_bytes=0)
    far = time.monotonic() + 3600
    held, crowded, at_share, ready = (
        runtime._Request(conn, request_id, 0.0, far, b"", [])  # type: ignore[arg-type]
        for request_id, conn in enumerate((under, over, full, under), 1)
    )
    held.held = True
    with server._lock:
        server._queue.extend([held, crowded, at_share, ready])
        taken = [server._next([]) for _ in range(3)]
        server._queue.clear()
    assert taken == (
        [crowded, at_share, ready] if concurrent else [at_share, ready, None]
    )


@pytest.mark.asyncio
async def test_fifo_mode_keeps_a_release_behind_the_waiting_call() -> None:
    """Time out a waiting ``get_layouts`` at its deadline when calls run one at a time."""
    host = _RegistryHost()
    await host.registry.init(_PUBLISHERS, strategy=_NO_STRATEGY)
    server = await _serve(host, "register_layout", "get_layouts")
    client = CarrierClient(server.address)
    start = time.monotonic()
    wait = asyncio.ensure_future(client.call("get_layouts", "k", 3, timeout_s=2))
    await _until(lambda: server.stats()["running"] == 1)
    registers = [
        client.call("register_layout", rank, "k", {"w": rank}) for rank in _PUBLISHERS
    ]
    registered = asyncio.gather(*registers)
    # Both releases reach the server and wait behind the running call.
    await _until(lambda: _counts(server) == (3, 1, 2))
    with pytest.raises(CallTimeoutError, match="within its 2 s deadline$"):
        await wait
    assert 2 <= time.monotonic() - start < 7
    # They run once the waiting handler's own 3 s wait ends.
    await registered


@pytest.mark.asyncio
@pytest.mark.parametrize("bound, window", [(64, 63), (2, 1), (1, 1)])
async def test_a_fifo_window_stays_below_the_bound_but_never_drops_to_0(
    monkeypatch: pytest.MonkeyPatch, target: _Target, bound: int, window: int
) -> None:
    """Welcome FIFO clients with a window one below the bound, but at least 1, as 0 would mean unlimited."""
    monkeypatch.setattr(runtime, "_QUEUE_BOUND", bound)
    server = await _serve(target)
    assert server._window() == window
    assert (await _serve(target, concurrent=True))._window() == 0


@pytest.mark.asyncio
async def test_concurrent_waits_never_fill_the_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Release the queue bound plus one waiting calls with one call on the same connection."""
    monkeypatch.setattr(runtime, "_QUEUE_BOUND", 4)
    host = _RegistryHost()
    await host.registry.init(_PUBLISHERS, strategy=_NO_STRATEGY)
    server = await _serve(host, "register_layout", "get_layouts", concurrent=True)
    client = CarrierClient(server.address)
    first, last = _PUBLISHERS
    await client.call("register_layout", first, "k", {"w": first})
    waits = [
        asyncio.ensure_future(client.call("get_layouts", "k", 60)) for _ in range(5)
    ]
    await _until(lambda: server.stats()["running"] == 5)
    await client.call("register_layout", last, "k", {"w": last})
    for wait in waits:
        assert await wait == {rank: {"w": rank} for rank in _PUBLISHERS}


@pytest.mark.asyncio
async def test_a_cancel_or_its_clients_death_cancels_a_running_concurrent_call() -> None:
    """Cancel a waiting lookup's task on its CANCEL frame and when its client's connection resets."""
    host = _RegistryHost()
    await host.registry.init(_PUBLISHERS, strategy=_NO_STRATEGY)
    server = await _serve(host, "register_layout", "get_layouts", concurrent=True)
    waiting, _ = codec.dumps(("get_layouts", ("k", 60), {}))
    with _raw_connection(server.address) as sock:
        _send_frame(sock, Kind.HELLO, body=_hello(server.address))
        assert _handshake_reply(sock)[0] == Kind.WELCOME
        _send_frame(sock, Kind.REQUEST, 1, deadline_us=60_000_000, body=waiting)
        # A call counts as running from its dispatch, and one abandoned before its task starts skips its handler.
        await _until(lambda: host.started == 1)
        _send_frame(sock, Kind.CANCEL, 1)
        await _until(lambda: host.cancelled == 1)
        _send_frame(sock, Kind.REQUEST, 2, deadline_us=60_000_000, body=waiting)
        await _until(lambda: host.started == 2)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    await _until(lambda: host.cancelled == 2)
    # A cancelled handler is no handler error.
    assert server.stats()["cancels"] == 2 and server.stats()["errors"] == 0


@pytest.mark.asyncio
async def test_the_end_of_a_close_grace_cancels_a_concurrent_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancel a waiting concurrent call's task when its server's close grace ends, and fail its caller."""
    host = _RegistryHost()
    await host.registry.init(_PUBLISHERS, strategy=_NO_STRATEGY)
    server = await _serve(host, "get_layouts", concurrent=True)
    thread, joins = server.dispatcher.thread, []
    join = thread.join

    def joined(timeout: float) -> None:
        joins.append(timeout)
        join(timeout)

    monkeypatch.setattr(thread, "join", joined)
    client = CarrierClient(server.address)
    call = asyncio.ensure_future(client.call("get_layouts", "k", 60, timeout_s=60))
    await _until(lambda: host.started == 1)
    await server.close(grace_s=0.2)
    assert host.cancelled == 1 and server.stats()["errors"] == 0
    # Joined within the close, as concurrent mode cancels every task first.
    assert joins == [runtime._RELEASE_WAIT_S] and not thread.is_alive()
    with pytest.raises(PeerLostError, match=_closed(server)):
        await call


@pytest.mark.asyncio
async def test_a_close_cancels_a_concurrent_call_before_it_closes_connections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancel a concurrent call's task as its close grace ends and before any connection closes. Its handler stops then.

    Its caller has been told the volume closed, and the connections can take seconds to close.
    """
    reached: concurrent.futures.Future[None] = concurrent.futures.Future()
    close_connections = CarrierServer._close_connections
    side_effects: list[str] = []
    cancelled = threading.Event()

    async def closing(self: CarrierServer, flush_s: float) -> None:
        reached.set_result(None)
        await close_connections(self, flush_s)

    class Late:
        async def wait(self) -> None:
            try:
                await runtime.wait_future(reached, None)
            except asyncio.CancelledError:
                cancelled.set()
                raise
            side_effects.append("ran on")

        def quick(self) -> str:
            return "quick"

    # A reply whose handler finished inside the grace stays on its way until the test lets it go.
    late: list[
        tuple[asyncio.AbstractEventLoop, Callable[..., Any], tuple[Any, ...]]
    ] = []
    post = runtime._post

    def late_post(
        loop: asyncio.AbstractEventLoop, fn: Callable[..., Any], *args: Any
    ) -> bool:
        if getattr(fn, "__name__", "") != "_queue_reply":
            return post(loop, fn, *args)
        late.append((loop, fn, args))
        return True

    monkeypatch.setattr(CarrierServer, "_close_connections", closing)
    monkeypatch.setattr(runtime, "_post", late_post)
    # So a close that cancels only after it waited for that reply would wait 30 s for it.
    monkeypatch.setattr(runtime, "_RELEASE_WAIT_S", 30)
    server = await _serve(Late(), "wait", "quick", concurrent=True)
    client = CarrierClient(server.address)
    call = asyncio.ensure_future(client.call("wait", timeout_s=60))
    await _until(lambda: server.stats()["running"] == 1)
    quick = asyncio.ensure_future(client.call("quick", timeout_s=60))
    await _until(lambda: len(late) == 1)
    closing_server = asyncio.ensure_future(server.close(grace_s=0.2))
    # Cancelled as the grace ends, while the quick reply is still on its way.
    await _until(cancelled.is_set, 10)
    [(loop, fn, args)] = late
    post(loop, fn, *args)
    assert await asyncio.wait_for(quick, 20) == "quick"
    await asyncio.wait_for(closing_server, 20)
    with pytest.raises(PeerLostError, match=_closed(server)):
        await call
    assert reached.done() and side_effects == []


@pytest.mark.asyncio
async def test_a_shutdown_that_raises_is_logged_and_the_close_still_returns(
    fresh_runtime: None,
    monkeypatch: pytest.MonkeyPatch,
    target: _Target,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Log a shutdown step that raises with its traceback, and still mark the server closed.

    Its own runtime keeps the server and dispatcher thread that the failed step left.
    """
    caplog.set_level(logging.ERROR, logger="torchstore.carrier")

    async def failing(self: CarrierServer) -> None:
        raise RuntimeError("dispatcher would not stop")

    monkeypatch.setattr(CarrierServer, "_stop_dispatcher", failing)
    server = await _serve(target)
    await asyncio.wait_for(server.close(grace_s=0), 10)
    [record] = [r for r in caplog.records if "failed to close" in r.getMessage()]
    assert record.exc_info is not None and "would not stop" in str(record.exc_info[1])


@pytest.mark.asyncio
async def test_a_caller_that_gave_up_gets_a_short_answer() -> None:
    """Answer a call whose caller gave up while it ran with a short error, not the large reply its handler made."""
    big = _Big()
    server = await _serve(big, "big")
    client = CarrierClient(server.address)
    call = asyncio.ensure_future(client.call("big", timeout_s=60))
    await _until(lambda: server.stats()["running"] == 1)
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call
    await _until(lambda: server.stats()["cancels"] == 1)
    big.release.set()
    conn = client._peer.conn
    assert conn is not None
    # The answer, short or not, reached the client.
    await _until(lambda: conn.unanswered == 0)
    await _until(lambda: server.stats()["bytes_out"] > 0)
    assert server.stats()["bytes_out"] < 1 << 20


@pytest.mark.asyncio
async def test_a_handler_ending_after_the_grace_sends_no_second_answer(
    monkeypatch: pytest.MonkeyPatch, target: _Target
) -> None:
    """Send only the closed answer for a call whose handler ends after the grace but before its connection closes."""
    drain: concurrent.futures.Future[None] = concurrent.futures.Future()
    close_connections = CarrierServer._close_connections

    async def held(self: CarrierServer, flush_s: float) -> None:
        # The connections close only once the test lets them.
        await runtime.wait_future(drain, 30)
        await close_connections(self, flush_s)

    monkeypatch.setattr(CarrierServer, "_close_connections", held)
    server = await _serve(target)
    body, _ = codec.dumps(("hold", ("held",), {}))
    with _raw_connection(server.address) as sock:
        _send_frame(sock, Kind.HELLO, body=_hello(server.address))
        assert _handshake_reply(sock)[0] == Kind.WELCOME
        _send_frame(sock, Kind.REQUEST, 1, deadline_us=10**8, body=body)
        await _until(lambda: server.stats()["running"] == 1)
        closing = asyncio.ensure_future(server.close(grace_s=0.2))
        await _until(lambda: server._closing)
        try:
            assert _answers(sock, 1) == {1: Kind.ERROR}
            target.release.set()
            await _until(lambda: server.stats()["running"] == 0)
        finally:
            drain.set_result(None)
        # The server half-closes once flushed, so the client reads to its end.
        rest = bytearray()
        while chunk := sock.recv(1 << 16):
            rest += chunk
    await closing
    assert rest == b""


@pytest.mark.asyncio
async def test_a_call_cancelled_and_then_lost_counts_one_cancel(
    target: _Target,
) -> None:
    """Count one cancel for a running call whose client sends its CANCEL and then resets."""
    server = await _serve(target)
    body, _ = codec.dumps(("hold", ("held",), {}))
    with _raw_connection(server.address) as sock:
        _send_frame(sock, Kind.HELLO, body=_hello(server.address))
        assert _handshake_reply(sock)[0] == Kind.WELCOME
        _send_frame(sock, Kind.REQUEST, 1, deadline_us=10**8, body=body)
        await _until(lambda: server.stats()["running"] == 1)
        received = server.stats()["bytes_in"]
        _send_frame(sock, Kind.CANCEL, 1)
        await _until(lambda: server.stats()["cancels"] == 1)
        # A CANCEL's bytes count as received too.
        assert server.stats()["bytes_in"] == received + codec.HEADER_SIZE
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    await _until(lambda: not server._connections)
    target.release.set()
    await _until(lambda: server.stats()["running"] == 0)
    assert server.stats()["cancels"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("concurrent", [False, True])
async def test_a_cancel_of_an_answered_call_counts_and_cancels_nothing(
    target: _Target, concurrent: bool
) -> None:
    """Leave a call alone and uncounted when its CANCEL comes after a closing server answered it."""
    server = await _serve(target, concurrent=concurrent)
    request = runtime._Request(None, 1, 0.0, time.monotonic() + 3600, b"", [])  # type: ignore[arg-type]
    request.answered = True
    with server._lock:
        outcome = server._abandon(request)
    assert outcome == (False, False) and not request.abandoned
    assert server.stats()["cancels"] == 0


@pytest.mark.asyncio
async def test_a_reply_the_io_loop_cannot_take_is_uncounted(
    monkeypatch: pytest.MonkeyPatch, target: _Target
) -> None:
    """Uncount a reply that cannot be posted to the I/O loop (as once it closed) so that no close waits for it."""
    server = await _serve(target)
    post = runtime._post

    def refusing(loop: asyncio.AbstractEventLoop, fn: Any, *args: Any) -> bool:
        if getattr(fn, "__func__", None) is CarrierServer._queue_reply:
            return False
        return post(loop, fn, *args)

    monkeypatch.setattr(runtime, "_post", refusing)
    call = asyncio.ensure_future(
        CarrierClient(server.address).call("echo", 1, timeout_s=2)
    )
    await _until(lambda: target.calls == [1] and server.stats()["running"] == 0)
    [conn] = server._connections
    await _until(lambda: (conn.unsent, conn.unsent_bytes, server._unsent) == (0, 0, 0))
    with pytest.raises(CallTimeoutError):
        await call


@pytest.mark.asyncio
async def test_a_new_connection_starts_flushed(target: _Target) -> None:
    """Count a connection that has queued nothing as flushed before its hello: a close never waits on a silent one."""
    server = await _serve(target)
    with _raw_connection(server.address):
        await _until(lambda: bool(server._connections))
        [conn] = server._connections
        assert conn.writer is None and conn.flushed.is_set()


@pytest.mark.asyncio
async def test_no_queued_call_starts_once_exit_has_begun(
    monkeypatch: pytest.MonkeyPatch, target: _Target
) -> None:
    """Start no queued call after the exit hook began, since it could still be in torch at exit."""
    server = await _serve(target)
    client = CarrierClient(server.address)
    held = asyncio.ensure_future(client.call("hold", "held"))
    await _until(lambda: server.stats()["running"] == 1)
    queued = asyncio.ensure_future(client.call("echo", "queued"))
    await _until(lambda: server.stats()["queued"] == 1)
    monkeypatch.setattr(runtime, "_exiting", True)
    target.release.set()
    assert await held == "held"
    # The held call's dispatcher step, which would start the queued one, ends before this runs.
    ran = asyncio.run_coroutine_threadsafe(asyncio.sleep(0), server._dloop)
    await asyncio.wrap_future(ran)
    assert _counts(server)[1:] == (0, 1) and target.calls == ["held"]
    monkeypatch.setattr(runtime, "_exiting", False)
    await server.close(grace_s=0)
    with pytest.raises(PeerLostError, match=_closed(server)):
        await queued


@pytest.mark.asyncio
async def test_a_call_dispatched_as_exit_begins_never_starts(
    monkeypatch: pytest.MonkeyPatch, target: _Target, unpickled: list[str]
) -> None:
    """Skip the handler of a call dispatched just before the exit hook began: its task checks again.

    Its caller hears that the process is exiting rather than that the server closed. Its request is never unpickled
    and the skip counts no handler error.
    """
    server = await _serve(target)
    client = CarrierClient(server.address)
    dispatch = server._dispatch

    def dispatch_then_exit() -> None:
        dispatch()
        # The hook sets its flag between the dispatch and the task's first step.
        monkeypatch.setattr(runtime, "_exiting", True)

    monkeypatch.setattr(server, "_dispatch", dispatch_then_exit)
    with pytest.raises(PeerLostError, match=_exiting(server)):
        await client.call("echo", _RecordsItsUnpickle("late"), timeout_s=60)
    assert target.calls == [] and unpickled == []
    assert server.stats()["errors"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("what", ["abandoned", "expired", "exit"])
async def test_a_call_overtaken_while_its_request_unpickles_never_starts(
    monkeypatch: pytest.MonkeyPatch, target: _Target, what: str
) -> None:
    """Skip the handler of a call whose CANCEL, deadline or process exit came while its request unpickled.

    Once the request is unpickled the task checks again right before its handler starts. The skip counts no handler error.
    """
    server = await _serve(target)
    client = CarrierClient(server.address)
    loads = codec.loads

    def overtaking(body: Any, buffers: Any = ()) -> Any:
        decoded = loads(body, buffers)
        if threading.current_thread() is server.dispatcher.thread:
            with server._lock:
                [request] = server._running
                if what == "abandoned":
                    request.abandoned = True
                elif what == "expired":
                    request.deadline = 0.0
            if what == "exit":
                monkeypatch.setattr(runtime, "_exiting", True)
        return decoded

    monkeypatch.setattr(codec, "loads", overtaking)
    answers = {
        "abandoned": (PeerLostError, _closed(server)),
        "expired": (CallTimeoutError, "dropped a call whose deadline passed"),
        "exit": (PeerLostError, _exiting(server)),
    }
    error, message = answers[what]
    with pytest.raises(error, match=message):
        await client.call("echo", "late", timeout_s=60)
    assert target.calls == [] and server.stats()["errors"] == 0


@pytest.mark.asyncio
async def test_exit_leaves_a_close_waiting_while_its_thread_holds_the_server(
    fresh_runtime: None, monkeypatch: pytest.MonkeyPatch, target: _Target
) -> None:
    """Keep a close waiting through exit while its thread still holds the server, and return once that thread let go."""
    monkeypatch.setattr(runtime, "_RELEASE_WAIT_S", 0.5)
    inside, release = threading.Event(), threading.Event()
    stop_writers = CarrierServer._stop_writers

    async def held(self: CarrierServer) -> None:
        inside.set()
        await asyncio.to_thread(release.wait, 30)
        await stop_writers(self)

    monkeypatch.setattr(CarrierServer, "_stop_writers", held)
    server = await _serve(target)
    closing = asyncio.ensure_future(server.close(grace_s=0))
    try:
        await _until(inside.is_set)
        await asyncio.to_thread(runtime._stop_loops_at_exit)
        assert not server._closed.done()
    finally:
        release.set()
    await asyncio.wait_for(closing, 10)


@pytest.mark.asyncio
async def test_exit_lets_started_closes_finish_then_stops_every_loop_not_parked(
    fresh_runtime: None, monkeypatch: pytest.MonkeyPatch, target: _Target
) -> None:
    """Join a started close before stopping the loops it uses, then stop and join each carrier loop but a parked one."""
    stop_writers = CarrierServer._stop_writers

    async def slow(self: CarrierServer) -> None:
        await asyncio.sleep(
            0.3
        )  # The close's last steps take a while: the pause under test.
        await stop_writers(self)

    monkeypatch.setattr(CarrierServer, "_stop_writers", slow)
    server = await _serve(target)
    closing = asyncio.ensure_future(server.close(grace_s=0))
    await _until(lambda: server._closing)
    [close_thread] = [
        t for t in threading.enumerate() if t.name == "torchstore-carrier-close test"
    ]
    parked = runtime._LoopThread("torchstore-carrier test-parked")
    parked.parked = True
    stopping = [t.thread for t in runtime._loop_threads if t is not parked]
    try:
        runtime._stop_loops_at_exit()
        assert (
            runtime._exiting and not close_thread.is_alive() and server._closed.done()
        )
        # The I/O and dispatcher threads stopped and were joined, and only the parked one runs on.
        assert stopping and not any(thread.is_alive() for thread in stopping)
        assert runtime._loop_threads == {parked} and parked.loop.is_running()
    finally:
        runtime._stop_loop(parked.loop)
        parked.thread.join(10)
    await closing


@pytest.mark.asyncio
async def test_exit_closes_every_client_while_the_io_loop_runs(
    fresh_runtime: None, children: _Children, caplog: pytest.LogCaptureFixture
) -> None:
    """Fail every client's calls and end its connection's tasks at exit, so later calls and closes fail or return at once.

    The tasks end though the I/O loop is busy as exit begins, and no handle can be made after exit began.
    """
    caplog.set_level(logging.ERROR, logger="asyncio")
    child = children.spawn(_SERVE_CHILD, _scope())
    address = CarrierAddress.from_json(_read_line(child))
    client = CarrierClient(address)
    assert await client.call("echo", 1) == 1
    call = asyncio.ensure_future(client.call("hold", timeout_s=10))
    await _until(lambda: _pending(client))
    conn = client._peer.conn
    assert conn is not None
    tasks = conn.tasks
    # Busy as exit begins: the clients' close and a stop posted without waiting for it would then run in one
    # turn of the loop and leave the tasks pending.
    client._peer.runtime.loop.call_soon_threadsafe(threading.Event().wait, 0.5)
    runtime._stop_loops_at_exit()
    assert tasks and all(task.done() for task in tasks)
    del conn, tasks
    # Closed at exit, so closing it now drops no connection with tasks still pending.
    await client.close()
    assert _task_errors(caplog, "_ClientConnection") == []
    with pytest.raises(PeerLostError, match="its last client handle closed$"):
        await call
    with pytest.raises(PeerLostError, match="is closed$"):
        await client.call("echo", 2)
    with pytest.raises(PeerLostError, match="once exit has begun$"):
        CarrierClient(address)


@pytest.mark.asyncio
async def test_waits_on_a_stuck_or_closed_loop_end_within_their_bounds(
    fresh_runtime: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stop waiting on a stuck loop after the release wait or a given timeout, and return at once for a closed one."""
    monkeypatch.setattr(runtime, "_RELEASE_WAIT_S", 0.2)
    loop_thread = runtime._LoopThread("torchstore-carrier test-stuck")
    stuck, ran = threading.Event(), []
    loop_thread.loop.call_soon_threadsafe(stuck.wait, 30)
    closed = asyncio.new_event_loop()
    closed.close()
    held = asyncio.sleep(0)

    def finish_held() -> None:
        # Its task runs on the freed loop before the loop stops so none is left pending there.
        for _ in range(10):
            if inspect.getcoroutinestate(held) == inspect.CORO_CLOSED:
                break
            runtime._post_call(loop_thread.loop, lambda: None).result(10)

    with contextlib.ExitStack() as cleanup:
        cleanup.callback(loop_thread.stopped.result, 10)
        cleanup.callback(runtime._stop_loop, loop_thread.loop)
        cleanup.callback(finish_held)
        cleanup.callback(stuck.set)
        start = time.monotonic()
        await asyncio.wait_for(
            runtime._call_on(loop_thread.loop, lambda: ran.append(1)), 10
        )
        await asyncio.wait_for(runtime._run_on(loop_thread.loop, held, 0.2), 10)
        # Each wait ended at its 0.2 s bound, not the 10 s of wait_for.
        assert ran == [] and time.monotonic() - start < 1.0
        never = asyncio.sleep(0)
        await asyncio.wait_for(runtime._run_on(closed, never, None), 10)
        # Closed at once: no warning that it was never awaited comes later.
        assert inspect.getcoroutinestate(never) == inspect.CORO_CLOSED


@pytest.mark.asyncio
async def test_a_wait_for_a_ready_socket_unregisters_it() -> None:
    """Unregister a socket once its wait to read or write ends so that the loop stops watching it."""
    loop = asyncio.get_running_loop()
    first, second = socket.socketpair()
    with first, second:
        first.setblocking(False)
        second.send(b"x")
        await asyncio.wait_for(runtime._readable(loop, first), 10)
        await asyncio.wait_for(runtime._writable(loop, first), 10)
        assert not loop.remove_reader(first.fileno())
        assert not loop.remove_writer(first.fileno())


def test_a_loop_thread_that_stopped_is_not_left_freeing(fresh_runtime: None) -> None:
    """Take a loop thread out of the threads exit waits for once its loop has closed."""
    loop_thread = runtime._LoopThread("torchstore-carrier test-freeing")
    runtime._stop_loop(loop_thread.loop)
    loop_thread.stopped.result(10)
    assert not runtime._freeing and loop_thread not in runtime._loop_threads


def test_a_call_posted_before_a_loop_stops_runs_and_one_after_is_done_at_once(
    fresh_runtime: None,
) -> None:
    """Run a call posted to a loop before its stop, and mark one posted after the stop done at once without running it."""
    loop_thread = runtime._LoopThread("torchstore-carrier test-stop")
    stuck, ran = threading.Event(), []
    loop_thread.loop.call_soon_threadsafe(stuck.wait)
    try:
        before = runtime._post_call(loop_thread.loop, lambda: ran.append("before"))
        runtime._stop_loop(loop_thread.loop)
        after = runtime._post_call(loop_thread.loop, lambda: ran.append("after"))
        assert after.done() and not before.done()
    finally:
        stuck.set()
    before.result(10)
    loop_thread.stopped.result(10)
    assert ran == ["before"]


@pytest.mark.asyncio
async def test_a_coroutine_run_on_a_stopping_loop_is_waited_for_only_while_it_can_run(
    fresh_runtime: None,
) -> None:
    """Close a coroutine run on a loop after its stop unstarted, and stop waiting on one posted before once the loop stops.

    That one never starts: its first step comes after the stop.
    """
    loop_thread = runtime._LoopThread("torchstore-carrier test-run-on")
    # Its task is destroyed pending with the loop, which this handler takes instead of asyncio's logger.
    loop_thread.loop.set_exception_handler(lambda loop, context: None)
    entered, stuck = threading.Event(), threading.Event()

    def hold() -> None:
        entered.set()
        stuck.wait(30)

    loop_thread.loop.call_soon_threadsafe(hold)
    before, after = asyncio.sleep(0), asyncio.sleep(0)
    try:
        # A coroutine posted before the loop thread is inside hold can join hold's turn. Its first step then shares
        # the stop's turn and runs.
        await _until(entered.is_set)
        waiting = asyncio.ensure_future(runtime._run_on(loop_thread.loop, before, 3600))
        # One turn of this loop lets that run post its coroutine ahead of the stop.
        await asyncio.sleep(0)
        runtime._stop_loop(loop_thread.loop)
        await asyncio.wait_for(runtime._run_on(loop_thread.loop, after, 3600), 10)
        assert inspect.getcoroutinestate(after) == inspect.CORO_CLOSED
        assert not waiting.done()
    finally:
        stuck.set()
    await asyncio.wait_for(waiting, 10)
    loop_thread.stopped.result(10)
    assert inspect.getcoroutinestate(before) == inspect.CORO_CREATED
    assert not runtime._runs
    before.close()


@pytest.mark.asyncio
async def test_a_reply_or_frame_left_for_a_stopping_io_loop_is_uncounted_at_once(
    monkeypatch: pytest.MonkeyPatch, fresh_runtime: None, target: _Target
) -> None:
    """Uncount a reply or frame at once once the I/O loop is marked stopping.

    A close thus never waits for one that never leaves. The loop runs on here as a marked loop does until the stop
    posted with the mark comes up. Even so neither reaches its outbox.
    """
    server = await _serve(target)
    client = CarrierClient(server.address)
    call = asyncio.ensure_future(client.call("hold", "x", timeout_s=60))
    await _until(lambda: target.calls == ["x"])
    [conn] = server._connections
    queued: list[int] = []
    queue_frame = conn._queue_frame

    def recorded(chunks: Any, size: int) -> None:
        queued.append(size)
        queue_frame(chunks, size)

    monkeypatch.setattr(conn, "_queue_frame", recorded)
    loop = server._runtime.loop
    runtime._stopped.add(loop)
    try:
        target.release.set()
        await _until(lambda: not server._running and not server._unsent)
        conn.send(server._closed_frame(99))
        assert conn.unsent == 0 and queued == []
    finally:
        runtime._stopped.discard(loop)
    call.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await call


def test_a_loop_stopped_at_exit_leaves_what_it_holds_to_the_main_thread(
    fresh_runtime: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep what a carrier loop's callbacks hold when it stops after exit began, for the main thread to free.

    A free on that daemon thread could then return into a finalizing interpreter.
    """
    freed: list[str] = []
    loop_thread = runtime._LoopThread("torchstore-carrier test-park")

    def schedule() -> None:
        loop_thread.loop.call_later(3600, print, _FreedOn(freed))

    runtime._post_call(loop_thread.loop, schedule).result(10)
    monkeypatch.setattr(runtime, "_exiting", True)
    runtime._stop_loop(loop_thread.loop)
    loop_thread.stopped.result(10)
    assert freed == [] and runtime._parked
    runtime._parked.clear()
    assert freed == [threading.current_thread().name]


@pytest.mark.asyncio
async def test_a_close_that_ends_once_exit_began_leaves_the_server_to_the_main_thread(
    fresh_runtime: None, monkeypatch: pytest.MonkeyPatch, target: _Target
) -> None:
    """Hand the server to the main thread to free when its close ends after exit began, as exit may stop waiting for it."""
    server = await _serve(target)
    monkeypatch.setattr(runtime, "_exiting", True)
    await server.close(grace_s=0)
    assert any(item is server for item in runtime._parked)


@pytest.mark.asyncio
@pytest.mark.parametrize("started", [False, True])
async def test_no_client_or_server_joins_the_carrier_once_exit_has_begun(
    monkeypatch: pytest.MonkeyPatch, target: _Target, started: bool
) -> None:
    """Refuse a new client or server after the exit hook began, the runtime started or not, as exit closes neither."""
    if started:
        await _serve(target)
    else:
        monkeypatch.setattr(runtime, "_runtime", None)
    monkeypatch.setattr(runtime, "_exiting", True)
    address = CarrierAddress("127.0.0.1", 1, "test", _scope(), "incarnation")
    refused = "^the carrier takes no new client or server once exit has begun$"
    with pytest.raises(PeerLostError, match=refused):
        CarrierClient(address)
    with pytest.raises(PeerLostError, match=refused):
        await _serve(target)


@pytest.mark.asyncio
async def test_a_serve_overtaken_by_the_exit_hook_attaches_nothing(
    fresh_runtime: None, monkeypatch: pytest.MonkeyPatch, target: _Target
) -> None:
    """Refuse a serve whose exit check passed just before the exit hook began, stopping its dispatcher and listener."""
    listen = runtime._listen
    listeners: list[socket.socket] = []

    def listen_then_exit(port: int) -> socket.socket:
        listeners.append(listen(port))
        # The hook sets its flag right after serve's own check.
        monkeypatch.setattr(runtime, "_exiting", True)
        return listeners[-1]

    monkeypatch.setattr(runtime, "_listen", listen_then_exit)
    before = threading.enumerate()
    refused = "^the carrier takes no new client or server once exit has begun$"
    with pytest.raises(PeerLostError, match=refused):
        await _serve(target)
    runtime_ = runtime._runtime
    assert runtime_ is not None and not runtime_.servers and not list(runtime._servers)
    [listener] = listeners
    assert listener.fileno() == -1
    for thread in _new_threads(before, "torchstore-carrier-dispatch"):
        thread.join(10)
        assert not thread.is_alive()


@pytest.mark.asyncio
async def test_a_task_left_on_a_closed_dispatcher_is_collected_quietly(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Log no handler error, and count none, when the GC closes a task that ignored close's cancel."""
    caplog.set_level(logging.INFO, logger="torchstore.carrier")
    monkeypatch.setattr(runtime, "_RELEASE_WAIT_S", 1.0)
    stubborn = _Stubborn()
    server = await _serve(stubborn, "wait", concurrent=True)
    call = asyncio.ensure_future(
        CarrierClient(server.address).call("wait", timeout_s=30)
    )
    await _until(lambda: stubborn.started == 1)
    await server.close(grace_s=0.5)
    with pytest.raises(PeerLostError, match=_closed(server)):
        await call
    # Cancelled once when the grace ended: stopping the dispatcher cancels it no more.
    assert server.stats()["running"] == 1 and server._dloop.is_closed()
    assert stubborn.cancels == 1
    stats, gone = server._stats, weakref.ref(server)
    del server
    deadline = time.monotonic() + 30
    while gone() is not None:
        assert time.monotonic() < deadline, "the closed server was never collected"
        gc.collect()
        await asyncio.sleep(0.05)
    assert stats["errors"] == 0
    assert not [r for r in caplog.records if "handler raised" in r.getMessage()]


@pytest.mark.asyncio
async def test_closing_the_last_handle_closes_the_shared_connection(
    target: _Target,
) -> None:
    """Keep a call running while another handle remains, and fail it when the last one closes."""
    server = await _serve(target)
    other = await _serve(_Target())
    first, second = CarrierClient(server.address), CarrierClient(server.address)
    call = asyncio.ensure_future(first.call("hold", "held"))
    await _until(lambda: server.stats()["running"] == 1)
    await first.close()
    with pytest.raises(PeerLostError, match="is closed"):
        await first.call("echo", 1)
    assert await CarrierClient(other.address).call("echo", 2) == 2
    assert not call.done()
    start = time.monotonic()
    await second.close()
    address = server.address
    with pytest.raises(
        PeerLostError, match=re.escape(f"{address.host}:{address.port}")
    ):
        await call
    assert time.monotonic() - start < 5
    target.release.set()
    assert await CarrierClient(server.address).call("echo", 3) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("close", ["handle", "scope"])
async def test_closing_the_last_handle_fails_its_unsent_calls_at_once(
    target: _Target, close: str
) -> None:
    """Fail a call still unsent behind a stalled send at once when the last handle or its scope closes, connecting no more."""
    server = await _serve(target)
    client = CarrierClient(server.address)
    assert await client.call("echo", "first") == "first"
    await server._set_reading(False)
    big = torch.zeros(64 << 20, dtype=torch.uint8)
    sending = asyncio.ensure_future(client.call("echo", big, timeout_s=30))
    await _until(lambda: _sending(client))
    unsent = asyncio.ensure_future(client.call("echo", "unsent", timeout_s=30))
    await _until(lambda: _unsent(client))
    peer = client._peer
    start = time.monotonic()
    if close == "handle":
        await client.close()
    else:
        await close_scope(server.address.scope, grace_s=0)
    for call in (sending, unsent):
        with pytest.raises(PeerLostError, match="its last client handle closed"):
            await asyncio.wait_for(call, 5)
    assert time.monotonic() - start < 5 and peer.connecting is None
    await server._set_reading(True)


@pytest.mark.asyncio
async def test_closing_the_last_handle_fails_a_call_still_connecting() -> None:
    """Fail a call waiting for its connection at once when its last handle closes, and stop connecting."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        client = CarrierClient(CarrierAddress("127.0.0.1", port, "test", _scope(), "0"))
        call = asyncio.ensure_future(client.call("echo", 1))
        await _until(lambda: client._peer.connecting is not None)
        await client.close()
        with pytest.raises(PeerLostError, match="its last client handle closed"):
            await asyncio.wait_for(call, 5)
        await _until(lambda: client._peer.connecting is None)


@pytest.mark.asyncio
async def test_closing_the_last_handle_fails_a_call_whose_connect_ends_meanwhile(
    monkeypatch: pytest.MonkeyPatch, target: _Target
) -> None:
    """Send no call whose connect ends after the last handle's close began but before that close reached the I/O thread.

    A thread switch inside that close can open the window, which holding the close until the call ends stands for.
    """
    server = await _serve(target)
    client = CarrierClient(server.address)
    handshake, call_on = runtime._Peer._handshake, runtime._call_on

    async def handshake_once_closed(peer: runtime._Peer, *args: Any) -> int:
        await _until(lambda: peer.closed)
        return await handshake(peer, *args)

    async def call_on_once_the_call_ends(
        loop: asyncio.AbstractEventLoop, fn: Callable[[], Any]
    ) -> None:
        await asyncio.wait([call], timeout=10)
        await call_on(loop, fn)

    monkeypatch.setattr(runtime._Peer, "_handshake", handshake_once_closed)
    monkeypatch.setattr(runtime, "_call_on", call_on_once_the_call_ends)
    call = asyncio.ensure_future(client.call("echo", "raced", timeout_s=10))
    await _until(lambda: bool(client._peer.waiting))
    await client.close()
    with pytest.raises(PeerLostError, match="its last client handle closed$"):
        await call
    assert target.calls == [] and server.stats()["calls"] == 0


def test_a_fork_while_another_thread_holds_the_lock_gets_a_working_runtime(
    children: _Children,
) -> None:
    """Serve and call in a child forked after exit began, while another thread held the runtime's locks.

    The child keeps none of the parent's carrier sockets or threads, closes inherited servers and handles at once and
    reaches the parent's server anew. The parent gets the lock back after a fork and warns when it forks without it.
    """
    child = children.spawn(_FORK_LOCK_CHILD)
    assert _read_line(child, timeout_s=90) == "0"
    _, stderr = child.communicate(timeout=60)
    assert "Forking without the carrier lock" in stderr, stderr


def test_the_fork_hooks_take_the_lock_and_reset_a_childs_state(
    fresh_runtime: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Take the carrier lock across fork and give it back in the parent. In the child, reset the exit and thread state.

    The child also gets fresh locks for the runtime and each server, and closes the runtime's sockets. It runs here on stand-ins.
    """
    monkeypatch.setattr(runtime, "_lock", threading.Lock())
    monkeypatch.setattr(runtime, "_loop_threads_lock", threading.Lock())
    runtime._before_fork()
    assert runtime._lock.locked() and runtime._fork_state.held
    runtime._after_fork_in_parent()
    assert not runtime._lock.locked() and not runtime._fork_state.held
    held = [runtime._lock, runtime._loop_threads_lock]
    for lock in held:
        lock.acquire()  # as another thread of the parent may hold it at fork
    sock = socket.socket()
    server_lock = threading.Lock()
    server_lock.acquire()
    server = types.SimpleNamespace(_lock=server_lock)
    stand_in = types.SimpleNamespace(sockets=weakref.WeakSet([sock]))
    monkeypatch.setattr(runtime, "_runtime", stand_in)
    monkeypatch.setattr(runtime, "_servers", [server])
    monkeypatch.setattr(runtime, "_exiting", True)
    runtime._loop_threads.add(object())  # type: ignore[arg-type]
    runtime._close_threads.add(threading.current_thread())
    runtime._freeing.add(threading.current_thread())
    runtime._runs[asyncio.new_event_loop()] = {concurrent.futures.Future()}
    with contextlib.ExitStack() as cleanup:
        cleanup.callback(sock.close)
        for lock in held:
            cleanup.callback(lock.release)
        for loop in runtime._runs:
            cleanup.callback(loop.close)
        runtime._after_fork_in_child()
        assert runtime._runtime is None and not runtime._exiting
        assert not (runtime._loop_threads or runtime._close_threads or runtime._freeing)
        assert not runtime._runs
        assert sock.fileno() == -1
        assert runtime._lock not in held and not runtime._lock.locked()
        assert runtime._loop_threads_lock not in held
        assert not runtime._loop_threads_lock.locked()
        assert server._lock is not server_lock and not server._lock.locked()


@pytest.mark.parametrize("thread", ["loop", "close"])
def test_exit_waits_for_a_thread_freeing_a_served_object(
    fresh_runtime: None,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    thread: str,
) -> None:
    """Wait at exit for a loop or close thread freeing a served object, naming it after 30 s and when exit stops waiting."""
    caplog.set_level(logging.WARNING, logger="torchstore.carrier")
    for name, value in (
        ("_EXIT_FREE_WARN_S", 0.05),
        ("_EXIT_FREE_WAIT_S", 0.1),
        ("_RELEASE_WAIT_S", 0.1),
    ):
        monkeypatch.setattr(runtime, name, value)
    freeing: list[str] = []
    release = threading.Event()

    class SlowToFree:
        def echo(self, value: Any) -> Any:
            return value

        def __del__(self) -> None:
            # Never on the main thread, which then could not release it.
            if threading.current_thread() is not threading.main_thread():
                freeing.append(threading.current_thread().name)
                release.wait(30)

    try:
        if thread == "loop":
            loop_thread = runtime._LoopThread("torchstore-carrier test-free")

            def schedule() -> None:
                loop_thread.loop.call_later(3600, print, SlowToFree())

            runtime._post_call(loop_thread.loop, schedule).result(10)
            runtime._stop_loop(loop_thread.loop)
        else:
            serve = CarrierServer.serve(
                SlowToFree(), ["echo"], service="test", scope=_scope()
            )
            begin_close = asyncio.run(serve)._begin_close
            # The close thread then holds the server's last reference.
            begin_close(0)
            del begin_close
        deadline = time.monotonic() + 10
        while not freeing:
            assert (
                time.monotonic() < deadline
            ), "no thread began to free the served object"
            time.sleep(0.01)
        runtime._stop_loops_at_exit()
    finally:
        release.set()
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 2 and all(freeing[0] in w for w in warnings), warnings


@pytest.mark.asyncio
async def test_close_scope_closes_only_matching_clients_and_servers(
    target: _Target,
) -> None:
    """Close the clients and servers of one scope, leaving another scope's alone."""
    servers = [
        await CarrierServer.serve(target, ["echo"], service="test", scope=_scope())
        for _ in range(2)
    ]
    clients = [CarrierClient(server.address) for server in servers]
    assert [await client.call("echo", i) for i, client in enumerate(clients)] == [0, 1]
    await close_scope(servers[0].address.scope)
    with pytest.raises(PeerLostError, match="is closed"):
        await clients[0].call("echo", 2)
    with pytest.raises(PeerLostError, match="cannot connect"):
        await CarrierClient(servers[0].address).call("echo", 3)
    assert await clients[1].call("echo", 4) == 4


@pytest.mark.asyncio
async def test_close_scope_starts_every_close_first_and_waits_for_the_clients(
    monkeypatch: pytest.MonkeyPatch, target: _Target, children: _Children
) -> None:
    """Start every matching close before ``close_scope``'s first await and wait for the clients.

    A cancel at that await thus stops no close.
    The clients' close runs on the I/O thread, held here until the test lets it go.
    """
    server = await _serve(target)
    remote_scope = _scope()
    child = children.spawn(_SERVE_CHILD, remote_scope)
    remote = CarrierClient(CarrierAddress.from_json(_read_line(child)))
    assert await remote.call("echo", 1) == 1
    conn = remote._peer.conn
    assert conn is not None
    # The addresses of the peers each posted client close takes.
    posted: list[CarrierAddress] = []
    post_call = runtime._post_call

    def recorded(loop: asyncio.AbstractEventLoop, fn: Callable[..., Any]) -> Any:
        if isinstance(fn, functools.partial) and fn.func is runtime._close_peers:
            posted.extend(peer.address for peer in fn.args[0])
        return post_call(loop, fn)

    monkeypatch.setattr(runtime, "_post_call", recorded)
    stuck = threading.Event()
    server._runtime.loop.call_soon_threadsafe(stuck.wait, 30)
    try:
        local = asyncio.ensure_future(close_scope(server.address.scope, grace_s=0))
        clients = asyncio.ensure_future(close_scope(remote_scope, grace_s=0))
        # One turn of this loop runs each to its first await.
        await asyncio.sleep(0)
        # One started its server's close, and the other posted its client's close and waits.
        assert server._closing and not clients.done() and posted == [remote.address]
        for closing in (local, clients):
            closing.cancel()
            with pytest.raises(asyncio.CancelledError):
                await closing
    finally:
        stuck.set()
    await _until(lambda: conn.closed)
    await asyncio.wait_for(server.close(grace_s=0), 10)
    with pytest.raises(PeerLostError, match="is closed$"):
        await remote.call("echo", 2)


@pytest.mark.asyncio
async def test_closing_clients_goes_on_past_one_whose_close_raises(
    monkeypatch: pytest.MonkeyPatch, target: _Target, caplog: pytest.LogCaptureFixture
) -> None:
    """Close the other clients of a scope, and log the failure, when closing one of them raises."""
    caplog.set_level(logging.ERROR, logger="torchstore.carrier")
    server = await _serve(target)
    a = server.address
    failing = CarrierClient(a)
    other = CarrierClient(
        CarrierAddress("127.0.0.1", a.port, a.service, a.scope, a.incarnation)
    )
    close = runtime._Peer.close

    def raising(peer: runtime._Peer) -> None:
        if peer is failing._peer:
            raise RuntimeError("close failed")
        close(peer)

    monkeypatch.setattr(runtime._Peer, "close", raising)
    call = asyncio.ensure_future(other.call("hold", "held"))
    await _until(lambda: server.stats()["running"] == 1)
    await close_scope(a.scope, grace_s=0)
    # Failed by its client's close, not by the server's, which answers only after the grace.
    with pytest.raises(PeerLostError, match="its last client handle closed$"):
        await call
    assert f"carrier close failed for carrier test at {a.host}:{a.port}" in caplog.text


@pytest.mark.asyncio
async def test_close_scope_refuses_a_scope_that_is_not_a_str(target: _Target) -> None:
    """Refuse a scope that is not a str, such as the address itself, closing nothing.

    Unchecked, it matched nothing and returned, leaving the server listening.
    """
    server = await _serve(target)
    client = CarrierClient(server.address)
    for scope in (server.address, None, server.address.scope.encode()):
        with pytest.raises(TypeError, match="^scope must be a str, not "):
            await close_scope(scope)  # type: ignore[arg-type]
    assert await client.call("echo", 1) == 1 and not server._closing


@pytest.mark.asyncio
async def test_a_new_handle_after_close_scope_reaches_a_live_server(
    children: _Children,
) -> None:
    """Dial a server in another process again after closing this process's clients of its scope."""
    scope = _scope()
    child = children.spawn(_SERVE_CHILD, scope)
    address = CarrierAddress.from_json(_read_line(child))
    assert await CarrierClient(address).call("echo", 1) == 1
    await close_scope(scope)
    assert await CarrierClient(address).call("echo", 2, timeout_s=30) == 2


@pytest.mark.asyncio
async def test_a_malformed_frame_closes_only_its_connection(target: _Target) -> None:
    """Close only the connection that sent garbage."""
    server = await _serve(target)
    client = CarrierClient(server.address)
    assert await client.call("echo", 1) == 1
    with _raw_connection(server.address) as sock:
        _send_frame(sock, Kind.HELLO, body=_hello(server.address))
        assert _handshake_reply(sock)[0] == Kind.WELCOME
        sock.sendall(b"x" * codec.HEADER_SIZE)
        assert _closed_by_peer(sock)
    assert await client.call("echo", 2) == 2
    assert target.calls == [1, 2]


@pytest.mark.asyncio
@pytest.mark.parametrize("frame", ["reply", "reused-id"])
async def test_a_wrong_frame_after_the_hello_closes_only_its_connection(
    target: _Target, frame: str
) -> None:
    """Close the connection that sends a reply, or reuses a running request's id, after its hello."""
    server = await _serve(target)
    client = CarrierClient(server.address)
    body, _ = codec.dumps(("hold", ("raw",), {}))
    with _raw_connection(server.address) as sock:
        _send_frame(sock, Kind.HELLO, body=_hello(server.address))
        assert _handshake_reply(sock)[0] == Kind.WELCOME
        if frame == "reply":
            _send_frame(sock, Kind.REPLY, 1)
        else:
            for _ in range(2):
                _send_frame(sock, Kind.REQUEST, 1, deadline_us=10**8, body=body)
        assert _closed_by_peer(sock)
    target.release.set()
    assert await client.call("echo", 1) == 1


@pytest.mark.asyncio
async def test_a_request_before_the_hello_is_rejected_unread(target: _Target) -> None:
    """Reject a request sent before the hello without unpickling it."""
    server = await _serve(target)
    body, buffers = codec.dumps(("echo", (_Marker(),), {}))
    with _raw_connection(server.address) as sock:
        _send_frame(
            sock, Kind.REQUEST, 1, deadline_us=10**6, body=body, buffers=buffers
        )
        kind, reply = _handshake_reply(sock)
        reason = Reject.from_json(reply).reason
        assert kind == Kind.REJECT and "before the handshake" in reason
        assert _closed_by_peer(sock)
    assert not _fired.is_set() and target.calls == []


@pytest.mark.asyncio
async def test_a_welcome_before_the_hello_is_rejected(target: _Target) -> None:
    """Reject a connection whose first frame is a handshake frame other than a hello."""
    server = await _serve(target)
    with _raw_connection(server.address) as sock:
        _send_frame(sock, Kind.WELCOME, body=_hello(server.address))
        kind, reply = _handshake_reply(sock)
        assert kind == Kind.REJECT
        assert Reject.from_json(reply).reason == "a WELCOME frame before a hello"
        assert _closed_by_peer(sock)
    assert server.stats()["rejected"] == 1


@pytest.mark.asyncio
async def test_a_client_that_never_says_hello_is_dropped(
    monkeypatch: pytest.MonkeyPatch, target: _Target, caplog: pytest.LogCaptureFixture
) -> None:
    """Close a connection whose client sends no hello once the connect deadline passes, with an INFO line."""
    caplog.set_level(logging.INFO, logger="torchstore.carrier")
    monkeypatch.setattr(runtime, "_CONNECT_TIMEOUT_S", 1)
    server = await _serve(target)
    # Taken before connecting, as the server arms its deadline only once it accepts.
    start = time.monotonic()
    with _raw_connection(server.address) as sock:
        assert _closed_by_peer(sock)
    assert 1 <= time.monotonic() - start < 6
    await _until(lambda: "carrier service=test: a handshake timed out" in caplog.text)


@pytest.mark.asyncio
async def test_failed_accepts_back_off(
    monkeypatch: pytest.MonkeyPatch, target: _Target, caplog: pytest.LogCaptureFixture
) -> None:
    """Wait between failed accepts, as at the file limit, off the I/O thread, which serves another server meanwhile.

    A run of failures logs one warning, and the first accept after it one line.
    """
    caplog.set_level(logging.INFO, logger="torchstore.carrier")
    server = await _serve(target)
    other = CarrierClient((await _serve(_Target())).address)
    assert await other.call("echo", 0) == 0
    attempts: list[float] = []
    accept_now = server._accept_now

    def failing() -> socket.socket | None:
        attempts.append(time.monotonic())
        if len(attempts) <= 20:
            raise OSError(errno.EMFILE, "Too many open files")
        return accept_now()

    monkeypatch.setattr(server, "_accept_now", failing)
    call = asyncio.ensure_future(
        CarrierClient(server.address).call("echo", 1, timeout_s=30)
    )
    await _until(lambda: len(attempts) >= 2)
    # 20 waits of 0.1 s: a wait that blocked the I/O thread would hold this call past its deadline.
    assert await other.call("echo", 2, timeout_s=1) == 2
    assert len(attempts) <= 20
    assert await call == 1
    gaps = [later - earlier for earlier, later in zip(attempts, attempts[1:6])]
    assert len(gaps) == 5 and min(gaps) >= 0.09, gaps
    lines = [(r.levelno, r.getMessage()) for r in caplog.records]
    failures = [line for line in lines if " cannot accept: " in line[1]]
    assert failures == [(logging.WARNING, failures[0][1])], lines
    assert lines.count((logging.INFO, "carrier service=test accepts again")) == 1


@pytest.mark.asyncio
async def test_a_later_run_of_failed_accepts_warns_again(
    monkeypatch: pytest.MonkeyPatch, target: _Target, caplog: pytest.LogCaptureFixture
) -> None:
    """Warn once per run of failed accepts (again after they recover) and say each time they recover."""
    caplog.set_level(logging.INFO, logger="torchstore.carrier")
    server = await _serve(target)
    fails = [0]
    accept_now = server._accept_now

    def failing() -> socket.socket | None:
        if fails[0]:
            fails[0] -= 1
            raise OSError(errno.EMFILE, "Too many open files")
        return accept_now()

    monkeypatch.setattr(server, "_accept_now", failing)
    a = server.address
    for host in ("127.0.0.1", "localhost"):
        # Two failures as each connect arrives, a run of its own each.
        fails[0] = 2
        # A name of its own each: each dials a connection of its own.
        address = CarrierAddress(host, a.port, a.service, a.scope, a.incarnation)
        assert await CarrierClient(address).call("echo", host, timeout_s=30) == host
        assert fails[0] == 0
    messages = [(r.levelno, r.getMessage()) for r in caplog.records]
    warned = [m for m in messages if " cannot accept: " in m[1]]
    assert [level for level, _ in warned] == [logging.WARNING] * 2, messages
    assert messages.count((logging.INFO, "carrier service=test accepts again")) == 2


@pytest.mark.asyncio
async def test_a_connection_that_cannot_be_configured_closes_alone(
    monkeypatch: pytest.MonkeyPatch, target: _Target
) -> None:
    """Close an accepted connection whose socket options cannot be set, and accept the next one."""
    monkeypatch.setattr(runtime, "_CONNECT_TIMEOUT_S", 5)
    server = await _serve(target)
    configure = runtime._configure
    failed: list[bool] = []

    def failing(sock: socket.socket) -> None:
        # The server's end of a connection, the first one only.
        if sock.getsockname()[1] == server.address.port and not failed:
            failed.append(True)
            raise OSError(errno.EINVAL, "Invalid argument")
        configure(sock)

    monkeypatch.setattr(runtime, "_configure", failing)
    a = server.address
    # One address, so the failed connect tries no other.
    client = CarrierClient(
        CarrierAddress("127.0.0.1", a.port, a.service, a.scope, a.incarnation)
    )
    with pytest.raises(PeerLostError):
        await client.call("echo", 1, timeout_s=30)
    assert await client.call("echo", 2, timeout_s=30) == 2
    assert target.calls == [2]


@pytest.mark.asyncio
async def test_a_protocol_version_mismatch_is_rejected(target: _Target) -> None:
    """Reject a hello of another protocol version, and fail a client a server rejects so."""
    server = await _serve(target)
    with _raw_connection(server.address) as sock:
        _send_frame(sock, Kind.HELLO, body=_hello(server.address, version=2))
        kind, reply = _handshake_reply(sock)
        assert kind == Kind.REJECT and "version 2" in Reject.from_json(reply).reason
        assert _closed_by_peer(sock)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.settimeout(30)

        def reject() -> None:
            conn, _ = listener.accept()
            with conn:
                conn.settimeout(30)
                _handshake_reply(conn)
                body = Reject("old", version=2).to_json().encode()
                _send_frame(conn, Kind.REJECT, body=body)

        thread = threading.Thread(target=reject, daemon=True)
        thread.start()
        address = CarrierAddress(
            "127.0.0.1", listener.getsockname()[1], "test", _scope(), "0"
        )
        with pytest.raises(CarrierProtocolError, match="protocol version 2"):
            await CarrierClient(address).call("echo", 1)
        thread.join(10)


@pytest.mark.asyncio
async def test_a_welcomed_connection_outlives_the_handshake_deadline(
    monkeypatch: pytest.MonkeyPatch, target: _Target
) -> None:
    """Keep a welcomed connection past the handshake's deadline, which no longer applies once the hello is answered."""
    monkeypatch.setattr(runtime, "_CONNECT_TIMEOUT_S", 0.2)
    server = await _serve(target)
    client = CarrierClient(server.address)
    assert await client.call("echo", 1) == 1
    conn = client._peer.conn
    await asyncio.sleep(0.5)  # Past the deadline under test, not synchronization.
    assert await client.call("echo", 2) == 2
    assert client._peer.conn is conn and server.stats()["connections"] == 1


@pytest.mark.asyncio
async def test_a_peer_without_the_scope_never_writes_to_the_log(
    target: _Target, caplog: pytest.LogCaptureFixture
) -> None:
    """Refuse a hello without the scope, though its other fields are right, saying only that and logging none of it.

    Log peers escaped once they prove the scope.
    """
    caplog.set_level(logging.INFO, logger="torchstore.carrier")
    server = await _serve(target)
    forged = (
        "x\ncarrier serve service=coord address=evil.example:1 incarnation=00 pid=1"
    )
    refused = _hello(server.address, scope="wrong", peer=forged)
    with _raw_connection(server.address) as sock:
        _send_frame(sock, Kind.HELLO, body=refused)
        kind, reply = _handshake_reply(sock)
        assert kind == Kind.REJECT
        assert Reject.from_json(reply).reason == "the scope does not match"
        assert _closed_by_peer(sock)
    assert not any("evil.example" in record.getMessage() for record in caplog.records)
    # A peer that proved the scope is logged, escaped so that it cannot forge a line.
    with _raw_connection(server.address) as sock:
        _send_frame(sock, Kind.HELLO, body=_hello(server.address, peer=forged))
        assert _handshake_reply(sock)[0] == Kind.WELCOME
        sock.sendall(b"x" * codec.HEADER_SIZE)
        assert _closed_by_peer(sock)
    await _until(lambda: "dropped a connection" in caplog.text)
    assert "\ncarrier serve" not in caplog.text


@pytest.mark.asyncio
async def test_a_refused_peer_whose_reject_fails_to_send_is_logged_unnamed(
    monkeypatch: pytest.MonkeyPatch, target: _Target, caplog: pytest.LogCaptureFixture
) -> None:
    """Log a lost connection without the name a peer that never proved the scope gave, as its reject failed to send."""
    caplog.set_level(logging.INFO, logger="torchstore.carrier")
    send_chunks = runtime._send_chunks

    async def failing(loop: Any, sock: Any, chunks: Any, *args: Any) -> int:
        if bytes(chunks[0][4:5]) == bytes([Kind.REJECT]):
            raise BrokenPipeError(errno.EPIPE, "Broken pipe")
        return await send_chunks(loop, sock, chunks, *args)

    monkeypatch.setattr(runtime, "_send_chunks", failing)
    server = await _serve(target)
    with _raw_connection(server.address) as sock:
        _send_frame(
            sock, Kind.HELLO, body=_hello(server.address, scope="x", peer="claimed")
        )
        assert _closed_by_peer(sock)
    await _until(lambda: "carrier service=test lost" in caplog.text)
    assert "claimed" not in caplog.text and "an unidentified peer" in caplog.text


@pytest.mark.asyncio
async def test_a_refusal_echoes_at_most_64_characters_of_each_hello_field(
    target: _Target,
) -> None:
    """Refuse hellos with oversized fields, the version included, echoing at most 64 characters of each.

    A peer without the scope learns only that, whatever else it sent.
    """
    server = await _serve(target)
    # Within the 4,300 digits Python parses into an int by default.
    address, long = server.address, "7" * 4000
    others: dict[str, Any] = dict(service=long, incarnation=long, peer=long)
    # The server checks the scope, version, service and incarnation in that order.
    refusals: dict[str, dict[str, Any]] = {
        "the scope does not match": {**others, "scope": long, "version": int(long)},
        f"carrier protocol version {long[:64]} is not {codec.PROTOCOL_VERSION}": {
            **others,
            "version": int(long),
        },
        f"this server serves 'test', not {long[:64]!r}": others,
        f"incarnation {long[:64]!r} is not {address.incarnation}": dict(
            incarnation=long, peer=long
        ),
    }
    for reason, fields in refusals.items():
        with _raw_connection(address) as sock:
            _send_frame(sock, Kind.HELLO, body=_hello(address, **fields))
            kind, reply = _handshake_reply(sock)
            assert kind == Kind.REJECT
            assert Reject.from_json(reply).reason == reason
            assert _closed_by_peer(sock)
    assert server.stats()["rejected"] == len(refusals)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer, error, message",
    [
        ("stale", StaleAddressError, "different scope, service or incarnation"),
        ("hello", CarrierProtocolError, "answered a hello with a HELLO frame"),
        ("eof", PeerLostError, "closed the connection during the handshake"),
        (
            "http",
            StaleAddressError,
            "did not answer the hello as a carrier server: bad frame magic b'HTTP'",
        ),
        (
            "garbled",
            StaleAddressError,
            "did not answer the hello as a carrier server: not a Welcome object",
        ),
        ("version", CarrierProtocolError, "speaks carrier protocol version 2, not 1$"),
        ("reshaped", CarrierProtocolError, "speaks carrier protocol version 2, not 1$"),
        (
            "reshaped-reject",
            CarrierProtocolError,
            "speaks carrier protocol version 2, not 1: old$",
        ),
        ("scope", StaleAddressError, "different scope, service or incarnation"),
        ("service", StaleAddressError, "different scope, service or incarnation"),
    ],
    ids=[
        "stale",
        "hello",
        "eof",
        "http",
        "garbled",
        "version",
        "reshaped",
        "reshaped-reject",
        "scope",
        "service",
    ],
)
async def test_a_client_checks_the_answer_to_its_hello(
    answer: str, error: type[Exception], message: str
) -> None:
    """Fail a call answered with a stale or other version's welcome, another frame, other bytes or EOF, naming host:port.

    Send no request.
    """
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.settimeout(30)
        port = listener.getsockname()[1]
        address = CarrierAddress("127.0.0.1", port, "test", _scope(), "a" * 16)
        after: list[bytes] = []

        def answer_hello() -> None:
            conn, _ = listener.accept()
            with conn:
                conn.settimeout(30)
                _handshake_reply(conn)
                if answer == "eof":
                    return
                if answer == "http":
                    conn.sendall(
                        b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n"
                    )
                elif answer == "garbled":
                    _send_frame(conn, Kind.WELCOME, body=b"not json")
                elif answer == "reshaped":
                    # Another version's welcome without this version's window.
                    _send_frame(
                        conn, Kind.WELCOME, body=b'{"version": 2, "scope": "x"}'
                    )
                elif answer == "reshaped-reject":
                    _send_frame(
                        conn,
                        Kind.REJECT,
                        body=b'{"version": 2, "why": "x", "reason": "old"}',
                    )
                elif answer in ("version", "scope", "service"):
                    other = {"version": 2, "scope": "x/y", "service": "other"}[answer]
                    welcome = _hello(address, Welcome, **{answer: other})
                    _send_frame(conn, Kind.WELCOME, body=welcome)
                else:
                    kind = Kind.WELCOME if answer == "stale" else Kind.HELLO
                    stale = _hello(address, Welcome, incarnation="b" * 16)
                    _send_frame(conn, kind, body=stale)
                try:
                    after.append(conn.recv(1 << 16))
                except ConnectionResetError:
                    after.append(b"")

        thread = threading.Thread(target=answer_hello, daemon=True)
        thread.start()
        with pytest.raises(error, match=message) as raised:
            await CarrierClient(address).call("echo", 1, timeout_s=10)
        assert f"carrier test at 127.0.0.1:{port}" in str(raised.value)
        thread.join(10)
        assert not thread.is_alive()
        assert after == ([] if answer == "eof" else [b""])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer, message",
    [
        ("request", "a REQUEST frame from the server"),
        ("unsent", "which was never sent"),
    ],
    ids=["request", "unsent"],
)
async def test_a_client_drops_a_server_that_breaks_the_protocol(
    answer: str, message: str
) -> None:
    """Fail a call with ``PeerLostError`` when the server answers it with a request, or a reply to an unsent id."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.settimeout(30)
        port = listener.getsockname()[1]
        address = CarrierAddress("127.0.0.1", port, "test", _scope(), "a" * 16)

        def serve_one() -> None:
            conn, _ = listener.accept()
            with conn:
                conn.settimeout(30)
                _handshake_reply(conn)
                _send_frame(conn, Kind.WELCOME, body=_hello(address, Welcome))
                data = _receive(conn, codec.HEADER_SIZE)
                request_id = codec.decode_header(data, handshake=False).request_id
                if answer == "request":
                    _send_frame(conn, Kind.REQUEST, request_id, deadline_us=10**6)
                else:
                    _send_frame(conn, Kind.REPLY, request_id + 1)
                _closed_by_peer(conn)

        thread = threading.Thread(target=serve_one, daemon=True)
        thread.start()
        with pytest.raises(PeerLostError, match=message):
            await CarrierClient(address).call("echo", 1, timeout_s=10)
        thread.join(10)
        assert not thread.is_alive()


@pytest.mark.asyncio
async def test_a_client_takes_a_negative_window_as_none(target: _Target) -> None:
    """Send calls to a server that welcomes with a negative window, as one of 0, unlimited."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.settimeout(30)
        port = listener.getsockname()[1]
        address = CarrierAddress("127.0.0.1", port, "test", _scope(), "a" * 16)

        def serve_one() -> None:
            conn, _ = listener.accept()
            with conn:
                conn.settimeout(30)
                _handshake_reply(conn)
                _send_frame(
                    conn, Kind.WELCOME, body=_hello(address, Welcome, window=-1)
                )
                header = codec.decode_header(
                    _receive(conn, codec.HEADER_SIZE), handshake=False
                )
                _receive(conn, header.body_len)
                body, _ = codec.dumps("answered")
                _send_frame(conn, Kind.REPLY, header.request_id, body=body)
                _closed_by_peer(conn)

        thread = threading.Thread(target=serve_one, daemon=True)
        thread.start()
        client = CarrierClient(address)
        assert await client.call("echo", 1, timeout_s=10) == "answered"
        await client.close()
        thread.join(10)
        assert not thread.is_alive()


@pytest.mark.asyncio
async def test_a_client_send_that_fails_fails_its_call_at_once(
    monkeypatch: pytest.MonkeyPatch, target: _Target
) -> None:
    """Fail a call with ``PeerLostError`` at once when its request's send fails, not at its deadline."""
    server = await _serve(target)
    client = CarrierClient(server.address)
    send_chunks = runtime._send_chunks

    async def failing(loop: Any, sock: Any, chunks: Any, *args: Any) -> int:
        if bytes(chunks[0][4:5]) == bytes([Kind.REQUEST]):
            raise ConnectionResetError(errno.ECONNRESET, "Connection reset by peer")
        return await send_chunks(loop, sock, chunks, *args)

    monkeypatch.setattr(runtime, "_send_chunks", failing)
    start = time.monotonic()
    with pytest.raises(PeerLostError, match="lost the connection to carrier test"):
        await client.call("echo", 2, timeout_s=10)
    assert time.monotonic() - start < 5


@pytest.mark.asyncio
async def test_log_lines_match_their_pinned_formats(
    monkeypatch: pytest.MonkeyPatch, target: _Target, caplog: pytest.LogCaptureFixture
) -> None:
    """Log the serve, connect and stats lines in their pinned formats, never the scope."""
    caplog.set_level(logging.INFO, logger="torchstore.carrier")
    monkeypatch.setattr(runtime, "_STATS_INTERVAL_S", 3600)
    server = await _serve(target)
    client = CarrierClient(server.address)
    assert await client.call("echo", 1) == 1
    with pytest.raises(ValueError):
        await client.call("fail", ValueError("bad"))
    await client.close()
    await server.close()
    lines = [record.getMessage() for record in caplog.records]
    patterns = [
        r"carrier serve service=test address=\S+:\d+ incarnation=[0-9a-f]{16} pid=\d+",
        r"carrier connect service=test to=\S+:\d+ from=\S+:\d+",
        r"carrier stats service=test calls=2 errors=1 timeouts=0 cancels=0 expired=0"
        r" bytes_in=\d+ bytes_out=\d+ max_queue_depth=1 max_queue_wait_ms=\d+",
    ]
    for pattern in patterns:
        assert len([line for line in lines if re.fullmatch(pattern, line)]) == 1, lines
    assert not any(server.address.scope in line for line in lines)


@pytest.mark.asyncio
@pytest.mark.parametrize("error", ["reset", "unreachable"])
async def test_a_failed_reply_send_logs_a_lost_line_unless_it_met_a_reset(
    monkeypatch: pytest.MonkeyPatch,
    target: _Target,
    caplog: pytest.LogCaptureFixture,
    error: str,
) -> None:
    """Log no lost connection for a client's reset, which its reader or a reply's send meets, and one for another failure."""
    caplog.set_level(logging.INFO, logger="torchstore.carrier")
    server = await _serve(target)
    # A client close resets the connection, which the server's reader meets.
    leaving = CarrierClient(server.address)
    assert await leaving.call("echo", 0) == 0
    await leaving.close()
    await _until(lambda: not server._connections)
    failure = (
        ConnectionResetError()
        if error == "reset"
        else OSError(errno.EHOSTUNREACH, "No route to host")
    )
    send_chunks = runtime._send_chunks

    async def failing(
        loop: asyncio.AbstractEventLoop,
        sock: socket.socket,
        chunks: list[bytes | memoryview],
        gate: asyncio.Event | None = None,
        send_large: Any = None,
    ) -> int:
        # Only a server connection's writer passes the server's sending gate.
        if gate is server._sending:
            raise failure
        return await send_chunks(loop, sock, chunks, gate, send_large)

    monkeypatch.setattr(runtime, "_send_chunks", failing)
    with pytest.raises(PeerLostError):
        await CarrierClient(server.address).call("echo", 1, timeout_s=30)
    await _until(lambda: not server._connections)
    lost = [r.getMessage() for r in caplog.records if " lost " in r.getMessage()]
    if error == "reset":
        assert lost == []
    else:
        assert len(lost) == 1 and "No route to host" in lost[0], lost


@pytest.mark.asyncio
async def test_a_failed_call_frees_its_arguments_without_the_cyclic_gc(
    target: _Target,
) -> None:
    """Free a call's arguments once it failed and its caller dropped the error, with the cyclic GC off."""
    server = await _serve(target)
    client = CarrierClient(server.address)
    gc.disable()
    try:
        tensor = torch.zeros(1 << 20, dtype=torch.uint8)
        freed = weakref.ref(tensor)
        call = asyncio.ensure_future(client.call("hold", tensor))
        await _until(lambda: server.stats()["running"] == 1)
        await client.close()
        # Awaiting a call that ends after close returns resumes this test from the task's done
        # callback, which keeps the task and its error's frames alive until the assert has run.
        await _until(call.done)
        with pytest.raises(PeerLostError):
            call.result()
        del tensor, call
        assert freed() is None
    finally:
        gc.enable()


@pytest.mark.asyncio
async def test_an_idle_client_connection_holds_no_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Free a call's received buffers without the cyclic GC once its caller drops the result on an idle connection."""
    received: list[weakref.ref[torch.Tensor]] = []
    read_tensor = runtime._Reader.read_tensor

    async def recorded(reader: runtime._Reader, size: int) -> torch.Tensor:
        tensor = await read_tensor(reader, size)
        received.append(weakref.ref(tensor))
        return tensor

    monkeypatch.setattr(runtime._Reader, "read_tensor", recorded)
    server = await _serve(_Store(), "get")
    client = CarrierClient(server.address)
    gc.disable()
    try:
        result = await client.call("get")
        # Once the result is here, the I/O thread drops its own references before its next read.
        await runtime._call_on(client._peer.runtime.loop, lambda: None)
        del result
        assert received and all(ref() is None for ref in received)
    finally:
        gc.enable()


@pytest.mark.asyncio
async def test_a_send_that_fails_while_it_waits_for_room_frees_its_data_without_the_cyclic_gc() -> None:
    """Free the data of a send whose socket fails while it waits for room, with the cyclic GC off.

    ``loop.sock_sendall`` keeps such a send's error in a future that its frame holds, with the data.
    """
    loop = asyncio.get_running_loop()
    with socket.create_server(("127.0.0.1", 0)) as listener:
        sender = socket.create_connection(listener.getsockname())
        receiver, _ = listener.accept()
    with sender, receiver:
        sender.setblocking(False)
        tensors = [torch.zeros(64 << 20, dtype=torch.uint8)]
        freed = weakref.ref(tensors[0])

        async def send() -> str:
            try:
                await runtime._sendall(loop, sender, codec.byte_view(tensors.pop()))
            except OSError as e:
                return type(e).__name__
            return "sent"

        gc.disable()
        try:
            sending = asyncio.ensure_future(send())
            await _until(lambda: _unsent_bytes(sender) > 0)
            receiver.setsockopt(
                socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
            )
            receiver.close()
            assert await sending in ("ConnectionResetError", "BrokenPipeError")
            assert freed() is None
        finally:
            gc.enable()


@pytest.mark.asyncio
async def test_a_closed_client_connection_is_freed_without_the_cyclic_gc(
    target: _Target,
) -> None:
    """Free a client's connection once its last handle closed and its tasks ended, with the cyclic GC off."""
    server = await _serve(target)
    client = CarrierClient(server.address)
    assert await client.call("echo", 1) == 1
    gc.disable()
    try:
        conn = client._peer.conn
        assert conn is not None
        freed, tasks = weakref.ref(conn), conn.tasks
        del conn
        await client.close()
        await _until(lambda: all(task.done() for task in tasks))
        del tasks
        assert freed() is None
    finally:
        gc.enable()


@pytest.mark.asyncio
async def test_a_server_connection_whose_reply_failed_on_a_writer_is_freed_without_the_cyclic_gc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Free a server's connection whose reply failed on a writer thread once it closed, with the cyclic GC off.

    The writer's future keeps the error, so the connection drops its traceback.
    """
    failed = threading.Event()
    send = socket.socket.send

    def failing(sock: socket.socket, data: Any, *args: Any) -> int:
        writer = threading.current_thread().name.startswith("torchstore-carrier-writer")
        if writer and not failed.is_set():
            failed.set()
            raise ConnectionResetError()
        return send(sock, data, *args)

    conns: list[weakref.ref[runtime._ServerConnection]] = []
    send_large = runtime._ServerConnection._send_large

    async def recorded(self: runtime._ServerConnection, chunk: Any) -> None:
        conns.append(weakref.ref(self))
        await send_large(self, chunk)

    monkeypatch.setattr(socket.socket, "send", failing)
    monkeypatch.setattr(runtime._ServerConnection, "_send_large", recorded)
    server = await _serve(_Sink(64 << 20), "get")
    async with _stalled_get(server) as (stalled, _):
        gc.disable()
        try:
            with pytest.raises(PeerLostError):
                await CarrierClient(server.address).call("get", timeout_s=30)
            assert failed.is_set()
            refs = [ref for ref in conns if ref() is not stalled]
            await _until(lambda: all(ref() is None for ref in refs))
        finally:
            gc.enable()


@pytest.mark.asyncio
async def test_stats_log_while_serving_once_the_interval_passes(
    monkeypatch: pytest.MonkeyPatch, target: _Target, caplog: pytest.LogCaptureFixture
) -> None:
    """Log the stats line when a call finishes after the interval, with no close."""
    caplog.set_level(logging.INFO, logger="torchstore.carrier")
    monkeypatch.setattr(runtime, "_STATS_INTERVAL_S", 0)
    server = await _serve(target)
    assert await CarrierClient(server.address).call("echo", 1) == 1
    # Logged on the dispatcher thread once the reply is queued, so it may trail the reply.
    await _until(
        lambda: any(
            record.getMessage().startswith("carrier stats service=test calls=1 ")
            for record in caplog.records
        )
    )
    assert not server._closing


@pytest.mark.asyncio
async def test_stats_log_at_most_once_per_interval(
    monkeypatch: pytest.MonkeyPatch, target: _Target, caplog: pytest.LogCaptureFixture
) -> None:
    """Log the stats line once when the interval has passed, and not again for calls within the next."""
    caplog.set_level(logging.INFO, logger="torchstore.carrier")
    monkeypatch.setattr(runtime, "_STATS_INTERVAL_S", 30)
    server = await _serve(target)
    client = CarrierClient(server.address)
    with server._lock:
        server._stats_logged -= 60
    for i in range(5):
        assert await client.call("echo", i) == i
    stats = "carrier stats service=test "
    # Logged on the dispatcher thread once the first reply is queued.
    await _until(lambda: stats in caplog.text)
    await _until(lambda: server.stats()["running"] == 0)
    await runtime._call_on(server._dloop, lambda: None)
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith(stats)]
    assert len(lines) == 1, lines
