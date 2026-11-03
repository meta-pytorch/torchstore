# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the coordinator service: its address, handles, replies, failures and fork."""

from __future__ import annotations

import asyncio
import collections
import contextlib
import dataclasses
import gc
import json
import logging
import math
import os
import pickle
import pickletools
import re
import select
import signal
import socket
import subprocess
import sys
import threading
import time
import types
import uuid
import weakref
import zlib
from collections.abc import AsyncIterator, Awaitable, Callable, Collection, Iterator
from pathlib import Path
from typing import Any, cast, NoReturn

import pytest
import torch
import torch.multiprocessing as mp
import torchstore as ts
import torchstore.api as store_api
import torchstore.backend as backend_module
from torchstore.carrier import (
    CallTimeoutError,
    CarrierAddress,
    CarrierClient,
    CarrierError,
    CarrierServer,
    PeerLostError,
    RemoteError,
    runtime,
    StaleAddressError,
)
from torchstore.routing import _gc, layout_registry, service as service_module, tcpstore
from torchstore.routing._model import KeyRegistration
from torchstore.routing.service import (
    CoordinatorService,
    ServiceAddress,
    ServiceControlPlane,
)
from torchstore.routing.tcpstore import TCPStoreAddress, TCPStoreRendezvous
from torchstore.transport import TransportType
from torchstore.transport.types import TensorSlice

pytestmark = [
    pytest.mark.timeout(120),
    # A native-call or carrier thread that raises fails its test.
    pytest.mark.filterwarnings("error::pytest.PytestUnhandledThreadExceptionWarning"),
]

_LOCALHOST = "127.0.0.1"
_PUBLISHERS = ("publisher/0", "publisher/1")
# Child processes import the tree that pytest imported.
_ROOT = str(Path(ts.__file__).parents[1])
# A lookup that does not wait returns well within this. A correct one cannot.
_STILL_PENDING_S = 0.5
_TIMING_LINE = re.compile(r"routing control_plane=service ([a-z_]+( key=[^ ]+)?) took")
# What a TCPStore call raises when its server is gone or silent.
_UNREACHABLE: tuple[type[Exception], ...] = (ConnectionError, TimeoutError)
# A TCPStore call on a connection its server closed fails within this, well before 9 s.
_AT_EOF_S = 5

_CHILD = """
import asyncio, json, os, signal, socket, sys, threading, time, traceback

# The service never needs Monarch, and the child starts faster without it. With it, a
# forked child that started a thread would crash in Monarch's atexit shutdown at exit.
sys.modules["monarch"] = None
sys.path.insert(0, sys.argv[1])
from torchstore.routing import service as service_module, tcpstore
from torchstore.routing.service import CoordinatorService
from torchstore.strategy import TorchStoreStrategy
from torchstore.transport import TransportType


async def start(store_name, publishers):
    service = await CoordinatorService.start(store_name, host="127.0.0.1")
    driver = await service.connect()
    await driver.init(publishers, TorchStoreStrategy(TransportType.MonarchRPC))
    return service, driver
"""

# Serves a store, prints its addresses, then runs commands (argv: store, publishers, grace).
_SERVE_CHILD = (
    _CHILD
    + """
store_name, publishers = sys.argv[2], json.loads(sys.argv[3])
service_module._CLOSE_GRACE_S = float(sys.argv[4])


async def main():
    service, driver = await start(store_name, publishers)
    address = service.address
    fields = [address.store_name, address.host, address.port, address.nonce]
    print(json.dumps([fields, service._server.address.to_json()]), flush=True)
    while True:
        command = (await asyncio.to_thread(sys.stdin.readline)).strip()
        if command == "stats":
            print(json.dumps(service._server.stats()), flush=True)
        elif command == "close":
            await service.close()
            print("done", flush=True)
        elif command == "stop-tcpstore":
            rendezvous, service._rendezvous = service._rendezvous, None
            await rendezvous.close()
            print("done", flush=True)
        else:
            # "exit", or stdin's end: leave without closing anything.
            return


asyncio.run(main())
"""
)

# A forked child calls a handle and the service it inherited, whose locks a parent thread held at the
# fork and whose call was in flight, under a default socket timeout (argv: store). The handle dials again.
_FORK_CHILD = (
    _CHILD
    + """
# Under a default timeout, a socket made from a descriptor turns it non-blocking for both processes.
socket.setdefaulttimeout(30)
service, driver = asyncio.run(start(sys.argv[2], ["publisher/0"]))


def socket_inodes():
    # Read through /proc, since a socket object would turn a descriptor non-blocking here.
    found = {}
    for name in os.listdir("/proc/self/fd"):
        try:
            if os.readlink(f"/proc/self/fd/{name}").startswith("socket:"):
                found[int(name)] = os.stat(f"/proc/self/fd/{name}").st_ino
        except OSError:
            pass
    return found


before = socket_inodes()
handle = asyncio.run(service.address.connect())
# The handle's TCPStore connection and the served TCPStore's end of it.
dropped = [(fd, inode) for fd, inode in socket_inodes().items() if fd not in before]
assert dropped, "connect opened no socket"
asyncio.run(handle.register_layout("publisher/0", "model", {}))
# A lookup in flight on its own thread, which waits until the parent registers its layout.
looked_up = {}
looking = threading.Thread(
    target=lambda: looked_up.update(layouts=asyncio.run(handle.get_layouts("pending", timeout_s=30))),
    daemon=True,
)
looking.start()
deadline = time.monotonic() + 10
while not handle._stops:
    assert time.monotonic() < deadline, "the lookup never got under way"
    time.sleep(0.01)
# Sockets the hook keeps: another listener, UDP on the served port, a client of it.
port = service.address.port
other = socket.socket()
other.bind(("127.0.0.1", 0))
other.listen()
udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp.bind(("127.0.0.1", port))
raw = socket.create_connection(("127.0.0.1", port), timeout=10)
raw.settimeout(None)
kept = [(sock.fileno(), os.fstat(sock.fileno()).st_ino) for sock in (other, udp, raw)]
held, release = threading.Event(), threading.Event()


def hold():
    with handle._lock, driver._lock, service._lock:
        held.set()
        release.wait(30)


threading.Thread(target=hold, daemon=True).start()
assert held.wait(10)
pid = os.fork()
if pid == 0:
    signal.alarm(30)  # a child stuck on an inherited lock dies by SIGALRM
    try:
        # First, before any new descriptor could take a closed one's number.
        for fd, inode in kept:
            assert os.fstat(fd).st_ino == inode, f"the fork hook closed fd {fd}"
        assert socket.getdefaulttimeout() == 30, "the fork hook changed the default socket timeout"
        for fd, inode in dropped:
            assert socket_inodes().get(fd) != inode, f"the fork hook kept the handle's fd {fd}"
        # Its close must not touch the parent's call, whose future a parent thread may lock.
        assert not handle._stops, "the fork hook kept the parent's call in flight"
        # The inherited client calls the coordinator with no TCPStore connection, which a read dials again.
        assert asyncio.run(handle.get_layouts("model", timeout_s=0)) == {"publisher/0": {}}
        assert handle._rendezvous is None, "a call through the inherited client dialed the TCPStore"
        assert type(asyncio.run(handle.strategy())).__name__ == "TorchStoreStrategy"
        # The service stays the parent's: only the parent inits or connects through it.
        calls = (
            (
                lambda: driver.init(["publisher/0"], TorchStoreStrategy(TransportType.MonarchRPC)),
                "inherited across fork: only the process that started it initializes it",
            ),
            (service.connect, "inherited across fork. Connect again from the address"),
        )
        for call, problem in calls:
            try:
                asyncio.run(call())
            except RuntimeError as error:
                assert problem in str(error) and str(service.address) in str(error), error
            else:
                raise AssertionError("an inherited service served a call")
        # Closing what the child inherited leaves the parent's connections and servers.
        for inherited in (handle, driver, service):
            asyncio.run(inherited.close())
        os._exit(0)
    except BaseException:
        traceback.print_exc()
        os._exit(1)
release.set()
_, status = os.waitpid(pid, 0)
assert os.get_blocking(raw.fileno()), "the child's fork hook made a parent socket non-blocking"
# The served TCPStore's own client, which the child kept, still reads.
assert asyncio.run(service._rendezvous.read(timeout_s=10))[0] == {"publisher/0"}
for sock in (other, udp, raw):
    sock.close()
assert asyncio.run(handle.get_layouts("model", timeout_s=0)) == {"publisher/0": {}}
assert type(asyncio.run(handle.strategy())).__name__ == "TorchStoreStrategy"
# The child's hook leaves this process's TCPStore listener taking new connects.
fresh = asyncio.run(service.address.connect())
assert type(asyncio.run(fresh.strategy())).__name__ == "TorchStoreStrategy"
# The child's close left the parent's lookup waiting, and it returns once its layout arrives.
asyncio.run(fresh.register_layout("publisher/0", "pending", {}))
looking.join(30)
assert looked_up == {"layouts": {"publisher/0": {}}}, looked_up
print(os.waitstatus_to_exitcode(status), flush=True)
"""
)


def _store_name() -> str:
    return f"service-test-{uuid.uuid4()}"


def _key() -> str:
    return f"model-{uuid.uuid4().hex[:8]}"


def _strategy() -> ts.TorchStoreStrategy:
    return ts.TorchStoreStrategy(TransportType.MonarchRPC)


def _layout(name: str) -> dict[str, KeyRegistration]:
    tensor_slice = TensorSlice(
        offsets=(0,), coordinates=(), global_shape=(4,), local_shape=(4,), mesh_shape=()
    )
    return {name: KeyRegistration(tensor_slice, dtype=torch.float32)}


def _raw_part(data: bytes, level: int = service_module._PART_ZLIB_LEVEL) -> bytes:
    """``data`` deflated raw at ``level`` up to a sync flush, as a publisher ends its part."""
    compressor = zlib.compressobj(level, zlib.DEFLATED, -zlib.MAX_WBITS)
    return compressor.compress(data) + compressor.flush(zlib.Z_SYNC_FLUSH)


def _part(name: str) -> tuple[bytes, int]:
    """``_layout(name)`` as a publisher sends it to the coordinator: its compressed pickle and the pickle's size."""
    data = pickle.dumps(_layout(name), protocol=5)
    return _raw_part(data), len(data)


def _layouts(key: str) -> dict[str, dict[str, KeyRegistration]]:
    """What a lookup of ``key`` returns once every publisher registered its layout."""
    return {rank: _layout(f"{key}/{rank}") for rank in _PUBLISHERS}


def _llama3_70b_layout(rank: int) -> dict[str, KeyRegistration]:
    """One of 64 FSDP ranks' layouts of TorchTitan's Llama 3 70B, as TitanRL registers its state dict.

    Parameters are bfloat16 under ``model_state_dict``, wqkv and w13 fused, all evenly sharded.
    """
    world = 64
    shapes: list[tuple[str, tuple[int, ...], int]] = [
        ("tok_embeddings.weight", (128256, 8192), 0)
    ]
    for layer in range(80):
        prefix = f"layers.{layer}."
        shapes += [
            (f"{prefix}attention.qkv_linear.wqkv.weight", (10240, 8192), 0),
            (f"{prefix}attention.wo.weight", (8192, 8192), 0),
            (f"{prefix}feed_forward.w13.weight", (2, 28672, 8192), 1),
            (f"{prefix}feed_forward.w2.weight", (8192, 28672), 0),
            (f"{prefix}attention_norm.weight", (8192,), 0),
            (f"{prefix}ffn_norm.weight", (8192,), 0),
        ]
    shapes += [("norm.weight", (8192,), 0), ("lm_head.weight", (128256, 8192), 0)]
    # A device mesh hands every DTensor the same coordinate tuple.
    coordinate = (rank,)
    layout = {}
    for name, shape, dim in shapes:
        local, offsets = list(shape), [0] * len(shape)
        local[dim] = shape[dim] // world
        offsets[dim] = local[dim] * rank
        tensor_slice = TensorSlice(
            tuple(offsets), coordinate, torch.Size(shape), torch.Size(local), (world,)
        )
        layout[f"model_state_dict/{name}"] = KeyRegistration(
            tensor_slice, dtype=torch.bfloat16
        )
    return layout


async def _register(handle: ServiceControlPlane, key: str, *ranks: str) -> None:
    # By keyword, as RoutingClient registers.
    for rank in ranks or _PUBLISHERS:
        await handle.register_layout(
            rank=rank, key=key, registrations=_layout(f"{key}/{rank}")
        )


async def _until(condition: Callable[[], bool], timeout_s: float = 10) -> None:
    deadline = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline, "the condition did not hold in time"
        await asyncio.sleep(0.01)


async def _assert_pending(*calls: asyncio.Future[Any]) -> None:
    done, _ = await asyncio.wait(calls, timeout=_STILL_PENDING_S)
    assert (
        not done
    ), f"returned early: {[call.exception() or call.result() for call in done]}"


class _WatchedLock:
    """A lock that counts the acquires that had to wait."""

    def __init__(self) -> None:
        self.lock, self.waited = threading.Lock(), 0

    def __enter__(self) -> None:
        if not self.lock.acquire(blocking=False):
            self.waited += 1
            self.lock.acquire()

    def __exit__(self, *exc: object) -> None:
        self.lock.release()


def _names_coordinator(address: ServiceAddress) -> str:
    """A pattern for errors that name the store's address and its coordinator."""
    return f"on {re.escape(str(address))}: coordinator call failed"


async def _fails_at_eof(
    call: Callable[[], Awaitable[Any]], address: ServiceAddress
) -> None:
    """Assert that a TCPStore call whose server closed its connection fails at once."""
    start = time.monotonic()
    with pytest.raises(ConnectionError, match=re.escape(str(address))) as raised:
        await call()
    assert not isinstance(raised.value, CarrierError)
    assert "coordinator call failed" not in str(raised.value), raised.value
    assert time.monotonic() - start < _AT_EOF_S


async def _close_each(
    closes: Collection[Callable[[], Awaitable[object]]], failing: bool
) -> None:
    """Run every close in turn, then raise the first one's error unless the test's own is on its way."""
    first: BaseException | None = None
    for close in closes:
        try:
            await close()
        except BaseException as error:
            first = first or error
    if first is not None and not failing:
        raise first


class _Served:
    """A service in this process, its driver's handle and the handles dialed from its address."""

    def __init__(
        self, service: CoordinatorService, driver: ServiceControlPlane
    ) -> None:
        assert service._server is not None
        self.service = service
        self.server: CarrierServer = service._server
        self.driver = driver
        self.handles = [driver]

    async def connect(self) -> ServiceControlPlane:
        self.handles.append(await self.service.address.connect())
        return self.handles[-1]


@contextlib.asynccontextmanager
async def _served(
    publishers: Collection[str] | None = _PUBLISHERS,
    host: str | None = _LOCALHOST,
    store_name: str | None = None,
) -> AsyncIterator[_Served]:
    """Start a service here, initialized unless ``publishers`` is None. Close it all at exit."""
    service = await CoordinatorService.start(store_name or _store_name(), host=host)
    served: _Served | None = None
    try:
        served = _Served(service, await service.connect())
        if publishers is not None:
            # By keyword, as ts.initialize does.
            await served.driver.init(publishers=publishers, strategy=_strategy())
        yield served
    finally:
        # Handles first, so no call left waiting holds the service's close grace.
        failing = sys.exc_info()[1] is not None
        handles = [] if served is None else served.handles
        await _close_each(
            [*(handle.close for handle in handles), service.close], failing
        )


@contextlib.asynccontextmanager
async def _dialed(
    address: ServiceAddress,
) -> AsyncIterator[Callable[[], Awaitable[ServiceControlPlane]]]:
    """Dial ``address`` once per call, and close every handle at exit."""
    handles: list[ServiceControlPlane] = []

    async def connect() -> ServiceControlPlane:
        handles.append(await address.connect())
        return handles[-1]

    try:
        yield connect
    finally:
        await _close_each(
            [handle.close for handle in handles], sys.exc_info()[1] is not None
        )


class _Child:
    """A child Python process in a process group of its own."""

    def __init__(self, script: str, *args: str) -> None:
        self.process = subprocess.Popen(
            [sys.executable, "-c", script, _ROOT, *args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )

    def read_line(self, timeout_s: float = 60) -> str:
        """Read the child's next line, failing with its stderr if none comes in time."""
        stdout = self.process.stdout
        assert stdout is not None
        ready, _, _ = select.select([stdout], [], [], timeout_s)
        line = stdout.readline() if ready else ""
        if not line:
            pytest.fail(f"the child reported nothing: {self.kill()}")
        return line.strip()

    def kill(self) -> str:
        """SIGKILL the child and anything it forked, reap it, and return its stderr."""
        if self.process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.process.pid, signal.SIGKILL)
        return self.process.communicate(timeout=30)[1] or ""


class _ChildService(_Child):
    """A coordinator service in a child process, which the test commands, stops or kills."""

    def __init__(self, grace_s: float = 1.0) -> None:
        publishers = json.dumps(list(_PUBLISHERS))
        super().__init__(_SERVE_CHILD, _store_name(), publishers, str(grace_s))
        fields, carrier = json.loads(self.read_line())
        self.address = ServiceAddress(*fields)
        self.carrier = CarrierAddress.from_json(carrier)

    def send(self, command: str) -> None:
        stdin = self.process.stdin
        assert stdin is not None
        stdin.write(f"{command}\n")
        stdin.flush()

    def command(self, command: str) -> str:
        self.send(command)
        return self.read_line()

    def stats(self) -> dict[str, int]:
        return json.loads(self.command("stats"))

    def stop(self) -> None:
        """SIGSTOP the child and return once the kernel reports it stopped."""
        self.process.send_signal(signal.SIGSTOP)
        # WNOWAIT leaves the stop to be reported again, so the child is not reaped.
        os.waitid(os.P_PID, self.process.pid, os.WSTOPPED | os.WNOWAIT)


def _native_calls() -> set[threading.Thread]:
    return {
        thread
        for thread in threading.enumerate()
        if thread.name.startswith("torchstore-tcpstore ")
    }


@pytest.fixture(autouse=True)
def _joins_its_native_calls() -> Iterator[None]:
    """Fail the test that leaves a native TCPStore call running, not a later one.

    Autouse fixtures set up first, so this joins after ``children`` kills the services.
    """
    earlier = _native_calls()
    yield
    mine = _native_calls() - earlier
    for thread in mine:
        thread.join(timeout=30)
    alive = [thread.name for thread in mine if thread.is_alive()]
    assert not alive, f"native TCPStore calls outlived the test: {alive}"
    assert not tcpstore._live_calls - earlier, "finished calls stayed registered"


@pytest.fixture
def children() -> Iterator[list[_Child]]:
    started: list[_Child] = []
    yield started
    for child in started:
        child.kill()


@pytest.fixture(scope="module")
def shared_service() -> Iterator[_ChildService]:
    """A child service for cases that only read it or use keys of their own."""
    child = _ChildService()
    try:
        yield child
    finally:
        child.kill()


def _service_child(children: list[_Child], grace_s: float = 1.0) -> _ChildService:
    child = _ChildService(grace_s)
    children.append(child)
    return child


@pytest.mark.asyncio
async def test_address_pickles_compares_by_value_and_hides_its_nonce() -> None:
    """Round-trip the address by value, show it as host:port, and keep the nonce out of logs."""
    cloudpickle = pytest.importorskip("cloudpickle")
    async with _served(publishers=None) as served:
        address = served.service.address
        for dumps, loads in (
            (pickle.dumps, pickle.loads),
            (cloudpickle.dumps, cloudpickle.loads),
        ):
            copy = loads(dumps(address))
            assert type(copy) is ServiceAddress
            assert copy == address and hash(copy) == hash(address)
        assert dataclasses.replace(address, nonce=uuid.uuid4().hex) != address
        assert isinstance(address, ts.ControlPlaneAddress)
        assert str(address) == f"{_LOCALHOST}:{address.port}"
        assert address.nonce not in str(address)
        assert address.nonce not in repr(address)
        with pytest.raises(dataclasses.FrozenInstanceError):
            address.nonce = "stale"  # type: ignore[misc]
        for live in (served.driver, served.service):
            with pytest.raises(TypeError, match="pass its address"):
                pickle.dumps(live)


@pytest.mark.asyncio
async def test_only_the_serving_process_initializes_once_writing_ready_last() -> None:
    """Refuse init through an address and a second init. Write the four keys, ready last."""
    async with _served(publishers=None) as served:
        rendezvous = served.service._rendezvous
        assert rendezvous is not None
        prefixed = rendezvous._prefixed
        assert prefixed is not None
        registry = served.service._adapter.registry
        written: list[str] = []
        publishers_at_ready: list[frozenset[str] | None] = []

        class Recorder:
            def __init__(self, store: Any) -> None:
                self.store = store

            def compare_set(self, key: str, expected: str, value: bytes) -> bytes:
                written.append(key)
                if key == "ready":
                    publishers_at_ready.append(registry._publishers)
                return self.store.compare_set(key, expected, value)

        rendezvous._prefixed = Recorder(prefixed)  # type: ignore[assignment]
        try:
            handle = await served.connect()
            with pytest.raises(
                RuntimeError,
                match=r"only on a handle from CoordinatorService\.connect\(\)",
            ):
                await handle.init(_PUBLISHERS, _strategy())
            await served.driver.init(_PUBLISHERS, _strategy())
            with pytest.raises(RuntimeError, match="init already ran"):
                await served.driver.init(_PUBLISHERS, _strategy())
        finally:
            rendezvous._prefixed = prefixed
        assert written == ["publishers", "strategy", "coordinator", "ready"]
        # A rank that sees ``ready`` finds the registry already set up.
        assert publishers_at_ready == [frozenset(_PUBLISHERS)]
        coordinator = await rendezvous.read_coordinator()
        assert CarrierAddress.from_json(coordinator) == served.server.address


@pytest.mark.asyncio
async def test_init_refuses_a_closed_service() -> None:
    """Refuse init after close, leaving the registry unset."""
    service = await CoordinatorService.start(_store_name(), host=_LOCALHOST)
    driver = await service.connect()
    try:
        await service.close()
        with pytest.raises(RuntimeError, match="this coordinator service is closed"):
            await driver.init(_PUBLISHERS, _strategy())
        assert service._adapter.registry._publishers is None
    finally:
        await driver.close()
        await service.close()


@pytest.mark.asyncio
async def test_a_closed_driver_handle_refuses_init() -> None:
    """Refuse init on the driver's own handle once it is closed, leaving the registry unset."""
    async with _served(publishers=None) as served:
        await served.driver.close()
        address = re.escape(str(served.service.address))
        with pytest.raises(RuntimeError, match=f"{address}: .* is closed"):
            await served.driver.init(_PUBLISHERS, _strategy())
        assert served.service._adapter.registry._publishers is None


@pytest.mark.asyncio
async def test_a_failed_init_is_not_retried_and_keeps_layouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refuse an init after one whose publish failed late, keeping the layouts ranks registered."""
    async with _served(publishers=None) as served:
        rendezvous = served.service._rendezvous
        assert rendezvous is not None
        publish = rendezvous.publish

        async def late(*args: Any, **kwargs: Any) -> None:
            # Every key is written, ready included, before the caller gives up.
            await publish(*args, **kwargs)
            raise TimeoutError("publish timed out")

        monkeypatch.setattr(rendezvous, "publish", late)
        with pytest.raises(TimeoutError, match="publish timed out"):
            await served.driver.init(_PUBLISHERS, _strategy())
        monkeypatch.setattr(rendezvous, "publish", publish)
        key = _key()
        handle = await served.connect()
        await _register(handle, key, "publisher/0")
        with pytest.raises(RuntimeError, match="init already ran"):
            await served.driver.init(_PUBLISHERS, _strategy())
        await _register(handle, key, "publisher/1")
        assert await handle.get_layouts(key, timeout_s=0) == _layouts(key)


@pytest.mark.asyncio
async def test_the_coordinator_is_published_at_the_tcpstores_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Publish the coordinator at the host that ranks dial for TCPStore, and reach it there.

    The host comes from the environment, and it is not the loopback literal that a service
    advertising 127.0.0.1 regardless would also match.
    """
    monkeypatch.setenv("TORCHSTORE_TCPSTORE_HOST", "localhost")
    async with _served(host=None) as served:
        rendezvous = served.driver._rendezvous
        assert rendezvous is not None
        published = CarrierAddress.from_json(await rendezvous.read_coordinator())
        assert served.service.address.host == published.host == "localhost"
        key, rank = _key(), await served.connect()
        await _register(rank, key)
        assert await rank.get_layouts(key, timeout_s=0) == _layouts(key)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [OSError("no port"), asyncio.CancelledError()],
    ids=["oserror", "cancelled"],
)
async def test_a_start_that_cannot_serve_the_coordinator_stops_its_tcpstore(
    monkeypatch: pytest.MonkeyPatch, error: BaseException
) -> None:
    """Stop the TCPStore that start served when serving the coordinator fails or is cancelled."""
    served: list[TCPStoreRendezvous] = []

    async def recording(*args: Any, **kwargs: Any) -> TCPStoreRendezvous:
        served.append(await TCPStoreRendezvous.serve(*args, **kwargs))
        return served[-1]

    async def failing(*args: Any, **kwargs: Any) -> NoReturn:
        raise error

    rendezvous_class = types.SimpleNamespace(serve=recording)
    monkeypatch.setattr(service_module, "TCPStoreRendezvous", rendezvous_class)
    server_class = types.SimpleNamespace(serve=failing)
    monkeypatch.setattr(service_module, "CarrierServer", server_class)
    with pytest.raises(type(error)) as raised:
        await CoordinatorService.start(_store_name(), host=_LOCALHOST)
    assert raised.value is error
    assert len(served) == 1 and served[0]._store is None


@pytest.mark.asyncio
async def test_a_failed_read_is_not_cached() -> None:
    """Fail a layout call before init with the store's error, then succeed after init."""
    async with _served(publishers=None) as served:
        handle = await served.connect()
        key = _key()
        with pytest.raises(RuntimeError, match="init has not run"):
            await asyncio.wait_for(_register(handle, key, "publisher/0"), 1)
        await served.driver.init(_PUBLISHERS, _strategy())
        await _register(handle, key)
        assert await handle.get_layouts(key, timeout_s=0) == _layouts(key)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("ready", "error"),
    [
        (tcpstore._SCHEMA_VERSION, "marked ready without a coordinator"),
        (b"2", "schema version b'2'"),
    ],
    ids=["no_coordinator", "other_schema"],
)
async def test_a_store_without_a_coordinator_under_this_schema_fails_fast(
    ready: bytes, error: str
) -> None:
    """Fail a first layout call at once, making no client, under another schema or with no coordinator key."""
    store_name = _store_name()
    rendezvous = await TCPStoreRendezvous.serve(store_name, host=_LOCALHOST)
    try:
        served = rendezvous.address
        values = [
            (tcpstore._PUBLISHERS, tcpstore._encode_publishers(_PUBLISHERS)),
            (tcpstore._STRATEGY, tcpstore._encode_strategy(_strategy())),
            (tcpstore._READY, ready),
        ]
        deadline = time.monotonic() + 10
        for key, value in values:
            await rendezvous._compare_set(key, value, deadline)
        address = ServiceAddress(store_name, _LOCALHOST, served.port, served.nonce)
        async with _dialed(address) as connect:
            handle = await connect()
            with pytest.raises(RuntimeError, match=re.escape(error)):
                await asyncio.wait_for(_register(handle, _key(), "publisher/0"), 1)
            assert handle._coordinator is None
            if ready == tcpstore._SCHEMA_VERSION:
                assert type(await handle.strategy()) is ts.TorchStoreStrategy
        with pytest.raises(RuntimeError, match=re.escape(error)):
            await rendezvous.read_coordinator(timeout_s=5)
    finally:
        await rendezvous.close()


@pytest.mark.asyncio
async def test_a_handle_close_cancelled_in_its_coordinator_close_still_drops_its_tcpstore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drop a handle's TCPStore connection when its close is cancelled while it closes its coordinator client."""
    async with _served() as served:
        handle = await served.connect()
        await _register(handle, _key(), "publisher/0")
        rendezvous, coordinator = handle._rendezvous, handle._coordinator
        assert rendezvous is not None and coordinator is not None
        close, entered = coordinator.close, asyncio.Event()

        async def held() -> None:
            entered.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(coordinator, "close", held)
        closing = asyncio.ensure_future(handle.close())
        await asyncio.wait_for(entered.wait(), 10)
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closing
        assert rendezvous._store is None
        await close()


@pytest.mark.asyncio
async def test_closing_the_drivers_handle_leaves_the_tcpstore_serving() -> None:
    """Keep the TCPStore serving ranks after the driver's handle closes its own connection."""
    async with _served() as served:
        rank = await served.connect()
        await served.driver.close()
        assert type(await rank.strategy()) is ts.TorchStoreStrategy
        fresh = await served.connect()
        assert type(await fresh.strategy()) is ts.TorchStoreStrategy


@pytest.mark.asyncio
async def test_strategy_never_reads_the_publisher_ranks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read a rank's strategy without the publisher ranks, whose size grows with the job."""

    async def full_read(**kwargs: Any) -> NoReturn:
        raise AssertionError("strategy() read the publisher ranks")

    async with _served() as served:
        handle = await served.connect()
        monkeypatch.setattr(handle._rendezvous, "read", full_read)
        assert type(await handle.strategy()) is ts.TorchStoreStrategy


@pytest.mark.asyncio
async def test_strategy_decodes_a_new_object_on_every_call() -> None:
    """Return a fresh strategy per call, with the registered type and transport."""
    async with _served() as served:
        handle = await served.connect()
        first, second = await handle.strategy(), await handle.strategy()
        assert first is not second
        for strategy in (first, second):
            assert type(strategy) is ts.TorchStoreStrategy
            assert strategy.default_transport_type == TransportType.MonarchRPC


@pytest.mark.asyncio
async def test_complete_layouts_are_pickled_once_and_sent_from_one_buffer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pickle a key's layouts once for four waiting and later lookups, sending one buffer.

    The waiting lookups' reply joins the parts kept as each rank registered, then drops them.
    """
    encoded: list[Any] = []
    encode = service_module._encode_layouts
    monkeypatch.setattr(
        service_module,
        "_encode_layouts",
        lambda layouts: encoded.append(layouts) or encode(layouts),
    )
    sent: list[list[Any]] = []
    copy_mutable = runtime._copy_mutable

    def spy(buffers: list[memoryview]) -> list[memoryview]:
        copies = copy_mutable(buffers)
        sent.append([copy.obj for copy in copies])
        return copies

    monkeypatch.setattr(runtime, "_copy_mutable", spy)
    key = _key()
    async with _served() as served:
        requesters = [await served.connect() for _ in range(4)]
        lookups = [
            asyncio.ensure_future(requester.get_layouts(key, timeout_s=30))
            for requester in requesters
        ]
        await _until(lambda: served.server.stats()["running"] == 4)
        await _register(await served.connect(), key)
        _, pending = await asyncio.wait(lookups, timeout=10)
        assert not pending
        assert [lookup.result() for lookup in lookups] == 4 * [_layouts(key)]
        later = await requesters[0].get_layouts(key, timeout_s=0)
        assert later == _layouts(key) and len(encoded) == 1
        parts = encoded[0].values()
        assert all(type(part) is bytes and type(size) is int for part, size in parts)
        assert key not in served.service._adapter._parts
        later["publisher/0"] = {}
        del later["publisher/1"]
        assert await requesters[1].get_layouts(key, timeout_s=0) == _layouts(key)
        cached = served.service._adapter._encoded[key]
    replies = [objects for objects in sent if objects]
    assert len(replies) == 6
    assert all(objects == [cached] and objects[0] is cached for objects in replies)
    assert type(cached) is bytes


@pytest.mark.asyncio
async def test_each_layout_is_pickled_as_it_registers_and_lookups_only_join_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep each rank's first layout as the compressed bytes its publisher sent, and join replies from them.

    The registry decodes the same first layouts, which later registrations change in neither.
    """
    encoded: list[dict[str, Any]] = []
    encode = service_module._encode_layouts
    monkeypatch.setattr(
        service_module,
        "_encode_layouts",
        lambda layouts: encoded.append(dict(layouts)) or encode(layouts),
    )
    key = _key()
    async with _served() as served:
        handle = await served.connect()
        await _register(handle, key, "publisher/0")
        # A rank's second layout changes neither its part nor the reply.
        await handle.register_layout("publisher/0", key, _layout(f"{key}/other"))
        await _register(handle, key, "publisher/1")
        assert await handle.get_layouts(key, timeout_s=0) == _layouts(key)
        await _register(handle, key)
        assert await handle.get_layouts(key, timeout_s=0) == _layouts(key)
        adapter = served.service._adapter
        assert key not in adapter._parts and type(adapter._encoded[key]) is bytes
        assert adapter.registry._layouts[key].publishers == _layouts(key)
    assert len(encoded) == 1 and list(encoded[0]) == list(_PUBLISHERS)
    assert encoded[0] == {rank: _part(f"{key}/{rank}") for rank in _PUBLISHERS}


@pytest.mark.asyncio
async def test_a_lookup_drops_only_its_own_keys_parts() -> None:
    """Keep another key's parts through a lookup, so a rank's retried layout never reaches its reply."""
    first, other = _key(), _key()
    async with _served() as served:
        handle = await served.connect()
        await _register(handle, other, "publisher/0")
        await _register(handle, first)
        assert await handle.get_layouts(first, timeout_s=0) == _layouts(first)
        assert list(served.service._adapter._parts) == [other]
        await handle.register_layout("publisher/0", other, _layout(f"{other}/later"))
        await _register(handle, other, "publisher/1")
        assert await handle.get_layouts(other, timeout_s=0) == _layouts(other)


class _WatchedPickle:
    """The service module's ``pickle``, recording what each ``loads`` gets and collection's state as a reply dumps."""

    def __init__(self, seen: list[tuple[str, bool]]) -> None:
        self.loaded: list[Any] = []

        class Pickler(pickle.Pickler):
            def dump(self, obj: Any) -> None:
                seen.append(("dumped", gc.isenabled()))
                super().dump(obj)

        self.Pickler = Pickler

    def __getattr__(self, name: str) -> Any:
        return getattr(pickle, name)

    def loads(self, data: Any, /) -> Any:
        self.loaded.append(data)
        return pickle.loads(data)


@pytest.mark.asyncio
async def test_a_reply_holds_parts_that_join_into_one_stream_naming_no_global(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Send the parts as publishers sent them, which join into one raw deflate stream, in a reply with no memo or global.

    Plain ``pickle.loads`` runs no function on it, and it lists each rank with its pickle's own size, whatever
    size its publisher sent. A client loads each part from a view of the one buffer it inflated, never a copy.
    """
    key = _key()
    adapter = service_module._CoordinatorAdapter(layout_registry.LayoutRegistry())
    await adapter.init(_PUBLISHERS, _strategy())
    layouts = _layouts(key)
    parts = {rank: service_module._compress_part(layouts[rank]) for rank in layouts}
    for rank, (part, size) in parts.items():
        # publisher/1's size, below its pickle's, only makes its inflate grow and copy its buffer.
        await adapter.register_layout(
            rank, key, part, size if rank == "publisher/0" else 1
        )
    reply = bytes(await adapter.get_layouts(key, 0))
    ops = {op.name for op, _, _ in pickletools.genops(reply)}
    assert not ops & {"MEMOIZE", "PUT", "BINPUT", "LONG_BINPUT"}
    assert not ops & {"GLOBAL", "STACK_GLOBAL", "INST", "OBJ", "REDUCE", "NEWOBJ"}
    ranks, sizes, sent = pickle.loads(reply)
    assert (ranks, sizes, sent) == (
        list(_PUBLISHERS),
        [size for _, size in parts.values()],
        [part for part, _ in parts.values()],
    )
    # The reply's own parts, then the empty last block that ends the stream.
    pickled = [pickle.dumps(layouts[rank], protocol=5) for rank in _PUBLISHERS]
    joined = b"".join([*sent, b"\x03\x00"])
    assert zlib.decompress(joined, -zlib.MAX_WBITS) == b"".join(pickled)
    watched = _WatchedPickle([])
    monkeypatch.setattr(service_module, "pickle", watched)
    inflated: list[bytes] = []
    inflate = service_module._inflate

    def recording(stream: bytes, size: int) -> bytes:
        inflated.append(inflate(stream, size))
        return inflated[-1]

    monkeypatch.setattr(service_module, "_inflate", recording)
    assert service_module._decode_layouts(reply) == layouts
    _, *views = watched.loaded
    [buffer] = inflated
    assert len(views) == 2
    assert all(type(view) is memoryview and view.obj is buffer for view in views)


@pytest.mark.asyncio
async def test_a_client_inflates_a_reply_once_then_loads_its_parts_in_one_pause_of_c(
    collecting: None,
) -> None:
    """Inflate a reply's stream once with collection on, then pause collection once and call only ``gc.enable`` inside.

    ``dict`` takes each part's ``pickle.loads`` through ``map`` in C. So only the two ends of ``call_gc_paused``'s own
    frame run. Other threads may run there too, and the decode-window case checks that the parts load inside the pause.
    """
    key = _key()
    adapter = service_module._CoordinatorAdapter(layout_registry.LayoutRegistry())
    await adapter.init(_PUBLISHERS, _strategy())
    for rank in _PUBLISHERS:
        await adapter.register_layout(rank, key, *_part(f"{key}/{rank}"))
    reply = bytes(await adapter.get_layouts(key, 0))
    # Each Python function and each C function that Python code calls, with whether collection was on.
    called: list[tuple[str, bool]] = []

    def profile(frame: types.FrameType, event: str, arg: Any) -> None:
        if event == "call":
            called.append((frame.f_code.co_name, gc.isenabled()))
        elif event == "c_call":
            called.append((getattr(arg, "__name__", ""), gc.isenabled()))

    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        decoded = service_module._decode_layouts(reply)
    finally:
        sys.setprofile(previous)
    assert decoded == _layouts(key)
    # Within the one pause no Python function runs, and of C calls made from Python code only its own gc.enable.
    assert [name for name, on in called if not on] == ["enable"], called
    assert [
        on for name, on in called if name in ("disable", "decompress", "loads")
    ] == [
        True,  # the reply's own load
        True,  # its stream's one inflate
        True,  # the pause
    ], called


def test_a_decode_builds_its_dict_before_collection_resumes(collecting: None) -> None:
    """Build a decode's dict inside its pause so that no collection over its loaded layouts runs before it returns."""
    layouts = {"publisher/0": _llama3_70b_layout(0)}
    parts = {
        rank: service_module._compress_part(layout) for rank, layout in layouts.items()
    }
    reply = service_module._encode_layouts(parts)
    started: list[int] = []

    def watch(phase: str, info: dict[str, int]) -> None:
        if phase == "start":
            started.append(info["generation"])

    # From empty counts and with no settrace tracer (such as coverage's) whose own allocations would set it off.
    tracer = sys.gettrace()
    sys.settrace(None)
    try:
        gc.collect()
        gc.callbacks.append(watch)
        try:
            decoded = service_module._decode_layouts(reply)
        finally:
            gc.callbacks.remove(watch)
    finally:
        sys.settrace(tracer)
    assert started == [] and decoded == layouts


def _fail_to_load() -> NoReturn:
    raise ValueError("this layout does not load")


class _LoadFails:
    """A layout entry that pickles, but whose load in the service raises."""

    def __reduce__(self) -> tuple[Any, ...]:
        return _fail_to_load, ()


@pytest.mark.asyncio
async def test_a_layout_that_does_not_pickle_or_load_fails_only_its_registration(
    collecting: None,
) -> None:
    """Fail a registration whose layout its publisher cannot pickle or the service cannot load.

    The publisher's fails with pickle's error before any call. Neither reaches the registry, so the
    rank's next layout is its first, and collection resumes as the load raises, not once it is freed.
    """
    key = _key()
    async with _served(publishers={"publisher"}) as served:
        handle = await served.connect()
        calls = served.server.stats()["calls"]
        unpicklable: Any = {"lock": threading.Lock()}
        with pytest.raises(TypeError, match="^cannot pickle '_thread.lock' object$"):
            await handle.register_layout("publisher", key, unpicklable)
        unloadable: Any = {"entry": _LoadFails()}
        with pytest.raises(ValueError, match="^this layout does not load$"):
            await handle.register_layout("publisher", key, unloadable)
        # The service's own call, whose error keeps the frames of the window it raised in.
        adapter = served.service._adapter
        part = _raw_part(b"not a pickle")
        with pytest.raises(pickle.UnpicklingError, match="^invalid load key") as raised:
            await adapter.register_layout("publisher", key, part, 12)
        assert gc.isenabled(), raised
        await handle.register_layout("publisher", key, _layout(key))
        assert served.server.stats()["calls"] == calls + 2
        assert await handle.get_layouts(key, timeout_s=0) == {"publisher": _layout(key)}


@pytest.mark.asyncio
async def test_a_part_that_does_not_decompress_fails_only_its_registration() -> None:
    """Fail a registration whose part does not decompress, as a plain pickle does, with zlib's error.

    It raises in the service and across the carrier, and a part ended by a last block fails too. Nothing reaches the
    registry, and other ranks' parts stay.
    """
    key = _key()
    plain = pickle.dumps(_layout(f"{key}/publisher/0"), protocol=5)
    async with _served() as served:
        handle = await served.connect()
        await _register(handle, key, "publisher/1")
        adapter = served.service._adapter
        with pytest.raises(zlib.error, match="invalid stored block lengths$"):
            await adapter.register_layout("publisher/0", key, plain, len(plain))
        with pytest.raises(zlib.error, match="invalid stored block lengths$"):
            await handle._call("register_layout", "publisher/0", key, plain, len(plain))
        # Ended by its own last block, it would end every reader's joined stream there.
        finishing = zlib.compressobj(8, zlib.DEFLATED, -zlib.MAX_WBITS)
        final = finishing.compress(plain) + finishing.flush()
        with pytest.raises(ValueError, match="must end with a sync flush$"):
            await adapter.register_layout("publisher/0", key, final, len(plain))
        assert list(adapter._parts[key]) == ["publisher/1"]
        assert list(adapter.registry._layouts[key].publishers) == ["publisher/1"]


@pytest.mark.asyncio
async def test_llama_3_70b_layouts_round_trip_in_a_sixteenth_of_their_bytes() -> None:
    """Register and look up two of 64 FSDP ranks' Llama 3 70B layouts in a reply under 1/16 of their pickles.

    The registry decodes every publisher's layout unchanged, and so does the client.
    """
    layouts = {f"publisher/{rank}": _llama3_70b_layout(rank) for rank in range(2)}
    key = _key()
    async with _served(publishers=list(layouts)) as served:
        handle = await served.connect()
        for rank, layout in layouts.items():
            await handle.register_layout(rank, key, layout)
        adapter = served.service._adapter
        # Level 8's bytes. Levels 1 to 7 give others that also pass the size bound below, and level 9
        # gives these same bytes more slowly, so no byte check tells it from level 8.
        pickled = {
            rank: pickle.dumps(layout, protocol=5) for rank, layout in layouts.items()
        }
        assert adapter._parts[key] == {
            rank: (_raw_part(data, 8), len(data)) for rank, data in pickled.items()
        }
        assert await handle.get_layouts(key, timeout_s=0) == layouts
        assert adapter.registry._layouts[key].publishers == layouts
        reply = adapter._encoded[key]
    pickled = sum(len(pickle.dumps(layout, protocol=5)) for layout in layouts.values())
    assert len(reply) * 16 < pickled, (len(reply), pickled)


@pytest.mark.asyncio
async def test_a_key_with_a_reply_still_refuses_a_nonpublisher_and_an_unloadable_layout() -> None:
    """Check each registration with the decode and the registry, also once the key's reply exists."""
    key = _key()
    async with _served() as served:
        handle = await served.connect()
        await _register(handle, key)
        assert await handle.get_layouts(key, timeout_s=0) == _layouts(key)
        with pytest.raises(KeyError, match="'requester' is not a publisher rank"):
            await handle.register_layout("requester", key, _layout(key))
        with pytest.raises(pickle.UnpicklingError, match="^invalid load key"):
            await served.service._adapter.register_layout(
                "publisher/0", key, _raw_part(b"not a pickle"), 12
            )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mapping",
    [types.MappingProxyType, collections.OrderedDict],
    ids=["mappingproxy", "ordereddict"],
)
async def test_a_layout_in_any_mapping_returns_as_a_plain_dict(
    mapping: Callable[[Any], Any],
) -> None:
    """Send a layout given as any ``Mapping`` as the plain dict that the registry keeps."""
    key = _key()
    async with _served(publishers={"publisher"}) as served:
        handle = await served.connect()
        await handle.register_layout("publisher", key, mapping(_layout(key)))
        reply = await handle.get_layouts(key, timeout_s=0)
        assert reply == {"publisher": _layout(key)}
        assert type(reply["publisher"]) is dict


@pytest.mark.asyncio
async def test_a_closed_service_drops_its_parts_and_replies() -> None:
    """Drop every part and reply at close. The store's next service sends its own layouts only."""
    complete, partial = _key(), _key()
    async with _served() as served:
        handle = await served.connect()
        await _register(handle, complete)
        assert await handle.get_layouts(complete, timeout_s=0) == _layouts(complete)
        await _register(handle, partial, "publisher/0")
        adapter = served.service._adapter
        assert list(adapter._encoded) == [complete]
        assert list(adapter._parts) == [partial]
        store_name = served.service.address.store_name
    assert adapter._parts == {} and adapter._encoded == {}
    async with _served(publishers={"other/0"}, store_name=store_name) as again:
        handle = await again.connect()
        for key in (complete, partial):
            await handle.register_layout("other/0", key, _layout(f"{key}/other"))
            reply = await handle.get_layouts(key, timeout_s=0)
            assert reply == {"other/0": _layout(f"{key}/other")}


@pytest.mark.asyncio
async def test_a_closed_service_keeps_no_layouts_for_its_handles() -> None:
    """Drop every layout at close, so a closed service or driver handle pins none of them."""
    key = _key()
    async with _served() as served:
        await _register(await served.connect(), key)
        assert await served.driver.get_layouts(key, timeout_s=0) == _layouts(key)
        layouts = weakref.ref(served.service._adapter.registry._layouts[key])
    gc.collect()
    assert layouts() is None, "a closed service keeps every rank's decoded layout"


_HELD: dict[str, tuple[threading.Event, threading.Event]] = {}


def _held_loaded(name: str) -> str:
    entered, release = _HELD[name]
    entered.set()
    assert release.wait(30)
    return name


class _Held:
    """A layout entry whose load in the service waits until the test releases it."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __reduce__(self) -> tuple[Any, ...]:
        return _held_loaded, (self.name,)


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["returned", "cancelled"])
async def test_a_close_through_only_the_carriers_public_names_drops_every_layout(
    ending: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Serve and close the coordinator through a carrier server that offers only its public names.

    Whether the close returns or is cancelled in its grace, the TCPStore serves until the server
    stops, then every layout drops and the TCPStore stops.
    """
    monkeypatch.setattr(service_module, "_CLOSE_GRACE_S", 30)
    monkeypatch.setitem(_HELD, "held", (threading.Event(), threading.Event()))
    servers: list[CarrierServer] = []

    async def serve(*args: Any, **kwargs: Any) -> types.SimpleNamespace:
        server = await CarrierServer.serve(*args, **kwargs)
        servers.append(server)
        return types.SimpleNamespace(address=server.address, close=server.close)

    monkeypatch.setattr(
        service_module, "CarrierServer", types.SimpleNamespace(serve=serve)
    )
    try:
        async with _served() as served:
            (server,) = servers
            handle = await served.connect()
            await _register(handle, _key())
            # Its load holds the dispatcher, so the close stays in its grace until the test releases it.
            entry: Any = {"x": _Held("held")}
            held = asyncio.ensure_future(
                handle.register_layout("publisher/0", _key(), entry)
            )
            await _until(_HELD["held"][0].is_set)
            adapter, rendezvous = served.service._adapter, served.service._rendezvous
            assert rendezvous is not None
            closing = asyncio.ensure_future(served.service.close())
            await _until(lambda: server._closing)
            if ending == "cancelled":
                closing.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await closing
            assert adapter.registry._layouts, "dropped before the server stopped"
            # Ranks still read the TCPStore while calls in the grace run.
            assert type(await handle.strategy()) is ts.TorchStoreStrategy
            _HELD["held"][1].set()
            await held
            if ending == "returned":
                await closing
                assert server._closed.done() and not adapter.registry._layouts
            await _until(lambda: server._closed.done(), timeout_s=30)
            await _until(
                lambda: not (
                    adapter.registry._layouts or adapter._parts or adapter._encoded
                )
            )
            await _until(lambda: rendezvous._store is None)
    finally:
        _HELD["held"][1].set()
        await asyncio.gather(*(server.close(grace_s=0) for server in servers))


def test_a_close_its_loops_end_cancels_drops_every_layout_once_the_server_stops(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Drop every layout and part, then stop the TCPStore, once the server stops, though asyncio.run ended in the grace.

    Its end cancels the close and closes the loop, so no task on that loop could finish the close.
    """
    monkeypatch.setattr(service_module, "_CLOSE_GRACE_S", 30)
    monkeypatch.setitem(_HELD, "held", (threading.Event(), threading.Event()))
    held: dict[str, Any] = {}

    async def main() -> None:
        service = await CoordinatorService.start(_store_name(), host=_LOCALHOST)
        held.update(service=service, adapter=service._adapter, server=service._server)
        held.update(rendezvous=service._rendezvous)
        driver = held["driver"] = await service.connect()
        await driver.init(publishers=_PUBLISHERS, strategy=_strategy())
        await _register(driver, _key())
        # Its load holds the dispatcher, so the close stays in its grace past this loop's end.
        entry: Any = {"x": _Held("held")}
        held["registering"] = asyncio.ensure_future(
            driver.register_layout("publisher/0", _key(), entry)
        )
        await _until(_HELD["held"][0].is_set)
        held["closing"] = asyncio.ensure_future(service.close())
        await _until(lambda: held["server"]._closing)

    caplog.set_level(logging.ERROR, logger="asyncio")
    try:
        asyncio.run(main())
        adapter, rendezvous = held["adapter"], held["rendezvous"]
        assert adapter.registry._layouts, "dropped before the server stopped"
        assert rendezvous._store is not None, "stopped the TCPStore before the server"
        _HELD["held"][1].set()
        deadline = time.monotonic() + 10
        while adapter.registry._layouts or adapter._parts or adapter._encoded:
            assert time.monotonic() < deadline, "the layouts outlived the close's grace"
            time.sleep(0.01)
        while rendezvous._store is not None:
            assert time.monotonic() < deadline, "the TCPStore outlived the grace"
            time.sleep(0.01)
    finally:
        _HELD["held"][1].set()
        if "driver" in held:
            asyncio.run(held["driver"].close())
        held.clear()
    gc.collect()
    assert "Task was destroyed but it is pending" not in caplog.text


@pytest.fixture
def collecting() -> Iterator[None]:
    """Start with collection enabled, end with no window recorded, and enable it again."""
    assert gc.isenabled()
    try:
        yield
        assert _gc._resuming == set()
    finally:
        gc.enable()


class _GcSeen:
    """A layout entry that records whether collection is enabled as it pickles and loads."""

    seen: list[tuple[str, bool]] = []

    def __reduce__(self) -> tuple[Any, ...]:
        _GcSeen.seen.append(("pickled", gc.isenabled()))
        return _gc_seen_loaded, ()


def _gc_seen_loaded() -> _GcSeen:
    _GcSeen.seen.append(("loaded", gc.isenabled()))
    return _GcSeen()


class _WatchedZlib:
    """The service module's ``zlib``, recording each decompress's buffer and whether collection is enabled."""

    bufsizes: list[object] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(zlib, name)

    @staticmethod
    def decompress(*args: Any, **kwargs: Any) -> bytes:
        _WatchedZlib.bufsizes.append(kwargs.get("bufsize"))
        _GcSeen.seen.append(("decompressed", gc.isenabled()))
        return zlib.decompress(*args, **kwargs)


@pytest.mark.asyncio
async def test_decodes_run_with_collection_paused_and_nothing_else_does(
    collecting: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pause collection only while the service loads a layout and a client loads a reply's layouts.

    A publisher's pickle, each inflate, the registry's calls and the reply's dump run with it enabled. A client
    inflates a reply of two parts once, before the one pause in which it loads them.
    """
    seen: list[tuple[str, bool]] = []
    monkeypatch.setattr(_GcSeen, "seen", seen)
    monkeypatch.setattr(_WatchedZlib, "bufsizes", [])
    paused = service_module.call_gc_paused

    def pausing(fn: Callable[..., Any], *args: Any) -> Any:
        seen.append(("pause", gc.isenabled()))
        return paused(fn, *args)

    monkeypatch.setattr(service_module, "call_gc_paused", pausing)
    monkeypatch.setattr(service_module, "pickle", _WatchedPickle(seen))
    monkeypatch.setattr(service_module, "zlib", _WatchedZlib())
    key = _key()
    async with _served() as served:
        registry = served.service._adapter.registry

        def watching(name: str) -> Callable[..., Awaitable[Any]]:
            call = getattr(registry, name)

            async def watched(*args: Any, **kwargs: Any) -> Any:
                seen.append((name, gc.isenabled()))
                return await call(*args, **kwargs)

            return watched

        for name in ("register_layout", "get_layouts"):
            monkeypatch.setattr(registry, name, watching(name))
        handle = await served.connect()
        for rank in _PUBLISHERS:
            entry: Any = _GcSeen()
            await handle.register_layout(rank, key, {"entry": entry})
        reply = await handle.get_layouts(key, timeout_s=0)
        assert [type(layout["entry"]) for layout in reply.values()] == [_GcSeen] * 2
        assert gc.isenabled()
    registration = [
        ("pickled", True),  # the publisher pickles its layout, which becomes its part
        ("decompressed", True),  # the service decodes it for the registry
        ("pause", True),
        ("loaded", False),
        ("register_layout", True),
    ]
    assert seen == [
        *registration,
        *registration,
        ("get_layouts", True),
        ("dumped", True),  # the reply, from the parts
        ("decompressed", True),  # the client's one inflate of the reply's stream
        ("pause", True),
        ("loaded", False),
        ("loaded", False),
    ]


@pytest.mark.asyncio
async def test_a_part_inflates_with_a_byte_to_spare_and_a_reply_into_exactly_its_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inflate a part into a buffer with a byte to spare and a reply's stream into one of exactly its pickles' size.

    Each publisher sends its pickle's size, which sets its part's buffer in the service.
    """
    bufsizes: list[object] = []
    monkeypatch.setattr(_WatchedZlib, "bufsizes", bufsizes)
    monkeypatch.setattr(_GcSeen, "seen", [])
    monkeypatch.setattr(service_module, "zlib", _WatchedZlib())
    key = _key()
    layouts = _layouts(key)
    sizes = [len(pickle.dumps(layout, protocol=5)) for layout in layouts.values()]
    async with _served() as served:
        handle = await served.connect()
        await _register(handle, key)
        assert await handle.get_layouts(key, timeout_s=0) == layouts
    assert bufsizes == [*(size + 1 for size in sizes), sum(sizes)]


@pytest.mark.asyncio
async def test_a_publishers_size_beyond_what_its_part_can_inflate_to_fails_its_registration() -> None:
    """Refuse a size that is not an int or asks for more than deflate's largest ratio, before any inflate."""
    key = _key()
    adapter = service_module._CoordinatorAdapter(layout_registry.LayoutRegistry())
    await adapter.init(_PUBLISHERS, _strategy())
    part, size = _part(f"{key}/publisher/0")
    for wrong in (-1, 1033 * len(part), 1.5, "12", True):
        with pytest.raises(ValueError, match="cannot inflate to"):
            await adapter.register_layout("publisher/0", key, part, wrong)
    await adapter.register_layout("publisher/0", key, part, size)
    assert list(adapter.registry._layouts[key].publishers) == ["publisher/0"]


@pytest.mark.asyncio
async def test_a_key_answers_again_after_a_failed_encode_or_dropped_parts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Answer a key from its kept parts after its reply's encode raised once, and once they are dropped from its layouts."""
    key = _key()
    adapter = service_module._CoordinatorAdapter(layout_registry.LayoutRegistry())
    await adapter.init(_PUBLISHERS, _strategy())
    for rank in _PUBLISHERS:
        await adapter.register_layout(rank, key, *_part(f"{key}/{rank}"))
    encode, failures = service_module._encode_layouts, [MemoryError()]
    compress, compressed = service_module._compress_part, []

    def failing_once(parts: Any) -> bytes:
        if failures:
            raise failures.pop()
        return encode(parts)

    def compressing(layout: Any) -> tuple[bytes, int]:
        compressed.append(layout)
        return compress(layout)

    monkeypatch.setattr(service_module, "_encode_layouts", failing_once)
    monkeypatch.setattr(service_module, "_compress_part", compressing)
    with pytest.raises(MemoryError):
        await adapter.get_layouts(key, 0)
    reply = bytes(await adapter.get_layouts(key, 0))
    assert service_module._decode_layouts(reply) == _layouts(key) and compressed == []
    adapter.drop_encodings()
    assert bytes(await adapter.get_layouts(key, 0)) == reply
    assert compressed == list(_layouts(key).values())


@pytest.mark.asyncio
async def test_a_reply_that_fails_to_decode_resumes_collection_as_it_raises(
    collecting: None,
) -> None:
    """Resume collection as a client's decode of a reply raises, not once its traceback is freed.

    A part that the client cannot load, as after a version skew, ends the decode.
    """
    key = _key()
    async with _served(publishers={"publisher"}) as served:
        handle = await served.connect()
        await handle.register_layout("publisher", key, _layout(key))
        served.service._adapter._parts[key]["publisher"] = (
            _raw_part(b"not a pickle"),
            12,
        )
        with pytest.raises(pickle.UnpicklingError, match="^invalid load key") as raised:
            await handle.get_layouts(key, timeout_s=0)
        # The held error keeps its frames, and with them any window they left open.
        assert gc.isenabled(), raised


@pytest.mark.asyncio
async def test_two_services_in_one_process_keep_their_own_layouts() -> None:
    """Keep each store's publishers and layouts apart while two services run in one process."""
    key = _key()
    async with _served() as first, _served(publishers={"other/0"}) as second:
        await _register(await first.connect(), key)
        await (await second.connect()).register_layout("other/0", key, _layout(key))
        reply = await (await first.connect()).get_layouts(key, timeout_s=0)
        assert reply == _layouts(key)
        reply = await (await second.connect()).get_layouts(key, timeout_s=0)
        assert reply == {"other/0": _layout(key)}


@pytest.mark.asyncio
async def test_an_infinite_wait_ends_on_the_last_registration() -> None:
    """Wait without end for inf, and check once for any negative or NaN wait."""
    key = _key()
    async with _served() as served:
        handle = await served.connect()
        await _register(handle, key, "publisher/0")
        # The call's deadline stays the margin however far below zero the wait is.
        checks = (-2 * service_module._REPLY_MARGIN_S, math.nan)
        for wait in checks:
            with pytest.raises(TimeoutError, match="missing publishers") as raised:
                await handle.get_layouts(key, timeout_s=wait)
            assert not isinstance(raised.value, CallTimeoutError)
        lookup = asyncio.ensure_future(handle.get_layouts(key, timeout_s=math.inf))
        await _until(lambda: served.server.stats()["running"] == 1)
        await _assert_pending(lookup)
        await _register(handle, key, "publisher/1")
        assert await asyncio.wait_for(lookup, 10) == _layouts(key)
        for wait in checks:
            assert await handle.get_layouts(key, timeout_s=wait) == _layouts(key)


@pytest.mark.asyncio
async def test_the_registry_waits_each_lookups_own_wait_however_long(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hand the registry the wait each lookup asked for, beyond 300 s and infinite included.

    A lookup of a key whose reply is already encoded sends that reply without the registry.
    """
    waits: list[float | None] = []
    async with _served() as served:
        registry = served.service._adapter.registry
        look_up = registry.get_layouts

        async def recording(key: str, *, timeout_s: float | None = None) -> Any:
            waits.append(timeout_s)
            return await look_up(key, timeout_s=timeout_s)

        monkeypatch.setattr(registry, "get_layouts", recording)
        handle = await served.connect()
        for wait in (900, math.inf, None):
            key = _key()
            await _register(handle, key)
            assert await handle.get_layouts(key, timeout_s=wait) == _layouts(key)
            assert await handle.get_layouts(key, timeout_s=0) == _layouts(key)
    assert waits == [900, math.inf, layout_registry._LAYOUT_REGISTRATION_TIMEOUT_S]


@pytest.mark.asyncio
async def test_a_lookup_without_a_timeout_waits_the_callers_default(
    shared_service: _ChildService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Send the caller's ``_LAYOUT_REGISTRATION_TIMEOUT_S`` as the wait, not the service process's own."""
    monkeypatch.setattr(layout_registry, "_LAYOUT_REGISTRATION_TIMEOUT_S", 0.25)
    async with _dialed(shared_service.address) as connect:
        handle = await connect()
        missing = r"^timed out after 0\.25 seconds waiting for publisher layouts for "
        with pytest.raises(TimeoutError, match=missing):
            await asyncio.wait_for(handle.get_layouts(_key()), 10)


@pytest.mark.asyncio
async def test_a_lookups_deadline_is_its_wait_plus_the_margin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Give each lookup's carrier call its wait, or 0 for one not above 0, plus 60 s.

    The default wait, 900 s, an infinite wait and NaN included.
    """
    recorded: list[tuple[str, Any]] = []

    class Recording(CarrierClient):
        async def call(
            self, endpoint: str, /, *args: Any, timeout_s: Any = None, **kwargs: Any
        ) -> Any:
            recorded.append((endpoint, timeout_s))
            return await super().call(endpoint, *args, timeout_s=timeout_s, **kwargs)

    monkeypatch.setattr(service_module, "CarrierClient", Recording)
    key = _key()
    async with _served() as served:
        handle = await served.connect()
        await _register(handle, key)
        for wait in (0, 10, 300, -5, 900, math.inf, math.nan):
            await handle.get_layouts(key, timeout_s=wait)
        await handle.get_layouts(key)
    deadlines = [
        timeout_s for endpoint, timeout_s in recorded if endpoint == "get_layouts"
    ]
    assert deadlines == [60, 70, 360, 60, 960, math.inf, 60, 360]


class _Unpicklable(Exception):
    """An error that cannot travel pickled, so the carrier delivers it as a RemoteError."""

    def __reduce__(self) -> NoReturn:
        raise TypeError("this error does not pickle")


class _Unencodable(Exception):
    """An error whose pickling raises past every fallback of the carrier's error encoding."""

    def __reduce__(self) -> NoReturn:
        raise SystemExit("from __reduce__")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "text"),
    [
        (_Unpicklable, "the registry broke"),
        (_Unencodable, "its encoding raised SystemExit"),
    ],
    ids=["unpicklable", "unencodable"],
)
async def test_a_coordinator_error_that_cannot_be_rebuilt_arrives_as_a_remote_error(
    monkeypatch: pytest.MonkeyPatch, error: type[Exception], text: str
) -> None:
    """Raise a RemoteError with the coordinator's error type, its text or the carrier's note on its encoding, no transport prefix.

    The live coordinator keeps serving, so one it cannot encode never reads as its closing.
    """
    async with _served() as served:

        async def fail(*args: Any) -> NoReturn:
            raise error("the registry broke")

        registry = served.service._adapter.registry
        monkeypatch.setattr(registry, "register_layout", fail)
        with pytest.raises(RemoteError) as raised:
            await _register(served.driver, _key(), "publisher/0")
        assert raised.value.remote_type.endswith(f".{error.__name__}")
        assert raised.value.args[1] == text
        assert "coordinator call failed" not in str(raised.value)
        monkeypatch.undo()
        key = _key()
        await _register(served.driver, key)
        assert await served.driver.get_layouts(key, timeout_s=0) == _layouts(key)


@pytest.mark.asyncio
async def test_registry_and_tcpstore_errors_keep_their_types_and_texts() -> None:
    """Deliver the registry's and TCPStore's own errors unchanged, with no coordinator prefix.

    The service keeps no part of a layout that the registry refuses.
    """
    key = _key()
    async with _served(publishers=None) as served:
        address = served.service.address
        handle = await served.connect()
        with pytest.raises(RuntimeError) as uninitialized:
            await handle.register_layout("publisher/0", key, {})
        assert type(uninitialized.value) is RuntimeError
        assert str(uninitialized.value) == (
            f"routing store {address.store_name!r} on TCPStore {address}: init has not run"
        )
        await served.driver.init(_PUBLISHERS, _strategy())
        with pytest.raises(KeyError) as rejected:
            await handle.register_layout("requester", key, {})
        assert type(rejected.value) is KeyError
        assert rejected.value.args == ("'requester' is not a publisher rank",)
        assert key not in served.service._adapter._parts
        with pytest.raises(TimeoutError) as timed_out:
            await handle.get_layouts(key, timeout_s=0)
        assert type(timed_out.value) is TimeoutError
        assert str(timed_out.value) == (
            f"timed out after 0 seconds waiting for publisher layouts for {key!r}; "
            "missing publishers: ['publisher/0', 'publisher/1']"
        )


@pytest.mark.asyncio
async def test_threads_with_their_own_loops_share_one_handle() -> None:
    """Serve 4 threads, each with ``asyncio.run`` per call, through one handle."""
    errors: list[BaseException] = []
    async with _served(publishers={"publisher"}) as served:
        handle = await served.connect()

        def work(thread: int) -> None:
            try:
                for i in range(50):
                    key = f"thread-{thread}/{i}"
                    layout = _layout(key)
                    asyncio.run(handle.register_layout("publisher", key, layout))
                    reply = asyncio.run(handle.get_layouts(key, timeout_s=0))
                    assert reply == {"publisher": layout}, reply
            except BaseException as e:
                errors.append(e)

        threads = [threading.Thread(target=work, args=(i,)) for i in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            await asyncio.to_thread(thread.join, 60)
        assert not errors and not any(thread.is_alive() for thread in threads)


@pytest.mark.asyncio
async def test_threads_racing_a_first_call_make_one_coordinator_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Make one coordinator client when a second thread's first call arrives mid-construction."""
    constructing, release = threading.Event(), threading.Event()
    made: list[CarrierAddress] = []

    class HeldClient(CarrierClient):
        def __init__(self, address: CarrierAddress) -> None:
            made.append(address)
            constructing.set()
            assert release.wait(10)
            super().__init__(address)

    async with _served(publishers={"publisher"}) as served:
        handle = await served.connect()
        watched = _WatchedLock()
        handle._lock = watched  # type: ignore[assignment]
        monkeypatch.setattr(service_module, "CarrierClient", HeldClient)
        errors: list[BaseException] = []

        def first_call(key: str) -> None:
            try:
                asyncio.run(handle.register_layout("publisher", key, _layout(key)))
            except BaseException as e:
                errors.append(e)

        threads = [
            threading.Thread(target=first_call, args=(f"k{i}",), daemon=True)
            for i in range(2)
        ]
        threads[0].start()
        try:
            assert await asyncio.to_thread(constructing.wait, 10)
            threads[1].start()
            # The second thread waits on the handle's lock or makes a client of its own.
            await _until(lambda: watched.waited == 1 or len(made) == 2)
            assert len(made) == 1
        finally:
            release.set()
            for thread in threads:
                if thread.ident is not None:
                    await asyncio.to_thread(thread.join, 30)
        assert not errors and len(made) == 1
        assert not any(thread.is_alive() for thread in threads)


@pytest.mark.asyncio
async def test_a_layout_call_leaves_no_task_or_stop_behind() -> None:
    """Leave none of a call's tasks pending, and no stop for a close to set, once each call ends."""
    key = _key()
    async with _served() as served:
        handle = await served.connect()
        tasks = asyncio.all_tasks()
        await _register(handle, key)
        assert await handle.get_layouts(key, timeout_s=0) == _layouts(key)
        with pytest.raises(TimeoutError, match="missing publishers"):
            await handle.get_layouts(_key(), timeout_s=0)
        await _until(lambda: asyncio.all_tasks() <= tasks)
        assert not handle._stops


@pytest.mark.asyncio
async def test_first_calls_that_both_read_the_coordinator_key_make_one_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Make one coordinator client when two first calls have each read the coordinator's address."""
    made: list[CarrierAddress] = []

    class Recording(CarrierClient):
        def __init__(self, address: CarrierAddress) -> None:
            made.append(address)
            super().__init__(address)

    async with _served(publishers={"publisher"}) as served:
        handle = await served.connect()
        rendezvous = handle._rendezvous
        assert rendezvous is not None
        read, reads, both = rendezvous.read_coordinator, [], asyncio.Event()

        async def together(**kwargs: Any) -> str:
            reads.append(await read(**kwargs))
            if len(reads) == 2:
                both.set()
            await both.wait()
            return reads[-1]

        monkeypatch.setattr(rendezvous, "read_coordinator", together)
        monkeypatch.setattr(service_module, "CarrierClient", Recording)
        keys = (_key(), _key())
        first, second = (
            handle.register_layout("publisher", key, _layout(key)) for key in keys
        )
        await asyncio.wait_for(asyncio.gather(first, second), 10)
    assert len(reads) == 2 and len(made) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("read_returns", [False, True])
async def test_a_close_during_the_first_read_makes_no_coordinator_client(
    monkeypatch: pytest.MonkeyPatch, read_returns: bool
) -> None:
    """Fail a first layout call with ``PeerLostError`` as its handle closes while it reads the coordinator's address.

    A read still held is cancelled, and one that returns as the close lands makes no client, which no close would release.
    """
    made: list[CarrierAddress] = []

    class Recording(CarrierClient):
        def __init__(self, address: CarrierAddress) -> None:
            made.append(address)
            super().__init__(address)

    monkeypatch.setattr(service_module, "CarrierClient", Recording)
    async with _served() as served:
        handle = await served.connect()
        rendezvous = handle._rendezvous
        assert rendezvous is not None
        read = rendezvous.read_coordinator
        entered, release = asyncio.Event(), asyncio.Event()
        cancelled: list[bool] = []

        async def held(**kwargs: Any) -> str:
            data = await read(**kwargs)
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.append(True)
                raise
            return data

        monkeypatch.setattr(rendezvous, "read_coordinator", held)
        first = asyncio.ensure_future(_register(handle, _key(), "publisher/0"))
        await asyncio.wait_for(entered.wait(), 10)
        if read_returns:
            # The read resumes before the close's stop reaches the call so the call ends on its own first.
            release.set()
        await handle.close()
        try:
            with pytest.raises(PeerLostError, match="this control plane closed$"):
                await asyncio.wait_for(first, 10)
        finally:
            release.set()
        assert cancelled == ([] if read_returns else [True])
    assert made == []


@pytest.mark.asyncio
async def test_a_close_on_another_thread_fails_a_call_that_saw_it_before_its_stop_was_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail a call with ``PeerLostError`` once a close on another thread has taken its stop, before that close sets it."""
    async with _served() as served:
        handle = await served.connect()
        rendezvous = handle._rendezvous
        assert rendezvous is not None
        read = rendezvous.read_coordinator
        entered, release, go = asyncio.Event(), asyncio.Event(), threading.Event()

        async def held(**kwargs: Any) -> str:
            data = await read(**kwargs)
            entered.set()
            await release.wait()
            return data

        monkeypatch.setattr(rendezvous, "read_coordinator", held)
        first = asyncio.ensure_future(_register(handle, _key(), "publisher/0"))
        await asyncio.wait_for(entered.wait(), 10)
        (stop,) = handle._stops
        set_result = stop.set_result

        def set_late(result: None) -> None:
            go.wait(10)
            set_result(result)

        monkeypatch.setattr(stop, "set_result", set_late)
        closing = threading.Thread(target=lambda: asyncio.run(handle.close()))
        closing.start()
        try:
            await _until(lambda: handle._closed)
            # The call reaches the closed check after its read while the close holds its stop unset.
            release.set()
            with pytest.raises(PeerLostError, match="this control plane closed$"):
                await asyncio.wait_for(first, 10)
        finally:
            go.set()
            release.set()
            await asyncio.to_thread(closing.join, 30)
        assert not closing.is_alive()


@pytest.mark.asyncio
async def test_a_close_on_another_thread_fails_a_call_whose_reply_came_after_it_took_the_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail a call with ``PeerLostError`` when its reply comes after a close on another thread took its stop.

    The close wins though the call ended with a reply and the close has yet to set the stop.
    """
    async with _served() as served:
        handle = await served.connect()
        await _register(handle, _key(), "publisher/0")
        client = handle._coordinator
        assert client is not None
        call = client.call
        entered, release, go = asyncio.Event(), asyncio.Event(), threading.Event()

        async def held(*args: Any, **kwargs: Any) -> Any:
            reply = await call(*args, **kwargs)
            entered.set()
            await release.wait()
            return reply

        monkeypatch.setattr(client, "call", held)
        second = asyncio.ensure_future(_register(handle, _key(), "publisher/0"))
        await asyncio.wait_for(entered.wait(), 10)
        (stop,) = handle._stops
        set_result = stop.set_result

        def set_late(result: None) -> None:
            go.wait(10)
            set_result(result)

        monkeypatch.setattr(stop, "set_result", set_late)
        closing = threading.Thread(target=lambda: asyncio.run(handle.close()))
        closing.start()
        try:
            await _until(lambda: handle._closed)
            # The reply reaches the call after the close took its stop and before that close set it.
            release.set()
            with pytest.raises(PeerLostError, match="this control plane closed$"):
                await asyncio.wait_for(second, 10)
        finally:
            go.set()
            release.set()
            await asyncio.to_thread(closing.join, 30)
        assert not closing.is_alive()


@pytest.mark.asyncio
async def test_a_call_whose_handle_closes_before_it_is_sent_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail a layout call with ``PeerLostError`` when its handle closes once it has the client but before the call is sent."""
    async with _served() as served:
        handle = await served.connect()
        await _register(handle, _key(), "publisher/0")
        get_client = handle._coordinator_client

        async def closing_first() -> CarrierClient:
            client = await get_client()
            await handle.close()
            return client

        monkeypatch.setattr(handle, "_coordinator_client", closing_first)
        with pytest.raises(
            PeerLostError, match="coordinator call failed: this control plane closed$"
        ):
            await handle.get_layouts(_key(), timeout_s=0)


@pytest.mark.asyncio
async def test_a_first_call_once_exit_has_begun_fails_naming_the_coordinator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Name the coordinator when the carrier refuses a first call's client because exit has begun."""
    async with _served() as served:
        handle = await served.connect()
        refused = _names_coordinator(served.service.address)
        with monkeypatch.context() as exiting:
            exiting.setattr(runtime, "_exiting", True)
            with pytest.raises(
                PeerLostError, match=f"{refused}: the carrier takes no new client"
            ):
                await handle.get_layouts(_key(), timeout_s=0)


@pytest.mark.asyncio
async def test_waiting_lookups_beyond_the_queue_bound_never_block_their_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Release the queue bound plus one waiting lookups with one registration, on one connection."""
    monkeypatch.setattr(runtime, "_QUEUE_BOUND", 4)
    key = _key()
    async with _served() as served:
        handle = await served.connect()
        await _register(handle, key, "publisher/0")
        lookups = [
            asyncio.ensure_future(handle.get_layouts(key, timeout_s=60))
            for _ in range(5)
        ]
        await _until(lambda: served.server.stats()["running"] == 5)
        await _register(handle, key, "publisher/1")
        for lookup in lookups:
            assert await asyncio.wait_for(lookup, 10) == _layouts(key)


@pytest.mark.asyncio
async def test_closing_the_service_fails_waiting_lookups_and_joins_its_dispatcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail 4 waiting lookups once the grace ends, naming the coordinator. Join the dispatcher."""
    # Close gives running calls 5 s by default. This case shortens that.
    assert service_module._CLOSE_GRACE_S == 5
    monkeypatch.setattr(service_module, "_CLOSE_GRACE_S", 1)
    key = _key()
    async with _served() as served:
        lookups = [
            asyncio.ensure_future(handle.get_layouts(key, timeout_s=60))
            for handle in [await served.connect() for _ in range(4)]
        ]
        await _until(lambda: served.server.stats()["running"] == 4)
        start = time.monotonic()
        await served.service.close()
        closed = _names_coordinator(served.service.address)
        for lookup in lookups:
            with pytest.raises(
                PeerLostError, match=f"{closed}: the coordinator service closed: "
            ):
                await asyncio.wait_for(lookup, 5)
        assert time.monotonic() - start < 1 + 1
        assert served.server.dispatcher.thread not in threading.enumerate()
        with pytest.raises(RuntimeError, match="coordinator service is closed"):
            await served.service.connect()


@pytest.mark.asyncio
async def test_a_cancelled_lookup_leaves_no_task_and_a_cancelled_close_stops_both(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End a cancelled lookup's task within 1 s. A close cancelled in its grace still stops both, in order."""
    monkeypatch.setattr(service_module, "_CLOSE_GRACE_S", 1)
    key = _key()
    async with _served() as served:
        handle = await served.connect()
        cancelled = asyncio.ensure_future(handle.get_layouts(key, timeout_s=60))
        await _until(lambda: served.server.stats()["running"] == 1)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        await _until(lambda: served.server.stats()["running"] == 0, timeout_s=1)

        waiting = asyncio.ensure_future(handle.get_layouts(key, timeout_s=60))
        await _until(lambda: served.server.stats()["running"] == 1)
        rendezvous = served.service._rendezvous
        assert rendezvous is not None
        closing = asyncio.ensure_future(served.service.close())
        await _until(lambda: served.server._closing)
        # The TCPStore keeps serving through the coordinator's grace, also once the close is cancelled.
        assert type(await handle.strategy()) is ts.TorchStoreStrategy
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closing
        assert type(await handle.strategy()) is ts.TorchStoreStrategy
        with pytest.raises(
            PeerLostError, match=_names_coordinator(served.service.address)
        ):
            await asyncio.wait_for(waiting, 5)
        await _until(lambda: served.server._closed.done())
        await _until(lambda: rendezvous._store is None)
        # The served TCPStore is freed, so a fork from here on leaves its old port alone.
        servers = [getattr(s, "underlying_store", s) for s in tcpstore._masters]
        assert served.service.address.port not in {
            server.port for server in servers if isinstance(server, tcpstore.TCPStore)
        }


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
async def test_a_close_drops_the_layouts_before_it_stops_the_tcpstore(
    monkeypatch: pytest.MonkeyPatch, cancelled: bool
) -> None:
    """Drop the layouts once the coordinator has stopped and only then stop the TCPStore, also when the close is cancelled.

    A cancelled close finishes on a daemon thread, which interpreter exit never waits for.
    """
    monkeypatch.setattr(service_module, "_CLOSE_GRACE_S", 1)
    steps: list[str] = []
    closes: list[str] = []
    dropped_on: list[threading.Thread] = []
    async with _served() as served:
        service, server = served.service, served.server
        adapter = service._adapter
        rendezvous = service._rendezvous
        assert rendezvous is not None
        drop_layouts, close_rendezvous = adapter.drop_layouts, rendezvous.close
        close_server = server.close

        async def closing_server(**kwargs: Any) -> None:
            closes.append("begun")
            await close_server(**kwargs)
            closes.append("returned")

        def dropping() -> None:
            dropped_on.append(threading.current_thread())
            steps.append(
                f"layouts dropped, coordinator stopped: {'returned' in closes}"
            )
            drop_layouts()

        async def closing_rendezvous() -> None:
            steps.append("tcpstore stopped")
            await close_rendezvous()

        monkeypatch.setattr(server, "close", closing_server)
        monkeypatch.setattr(adapter, "drop_layouts", dropping)
        monkeypatch.setattr(rendezvous, "close", closing_rendezvous)
        if cancelled:
            # A lookup in flight holds the close in its grace, where the cancel lands.
            waiting = asyncio.ensure_future(
                served.driver.get_layouts(_key(), timeout_s=60)
            )
            await _until(lambda: server.stats()["running"] == 1)
            closing = asyncio.ensure_future(service.close())
            await _until(lambda: bool(closes))
            closing.cancel()
            with pytest.raises(asyncio.CancelledError):
                await closing
            closed = _names_coordinator(service.address)
            with pytest.raises(
                PeerLostError, match=f"{closed}: the coordinator service closed: "
            ):
                await waiting
            await _until(lambda: len(steps) == 2)
        else:
            await service.close()
    assert steps == ["layouts dropped, coordinator stopped: True", "tcpstore stopped"]
    assert dropped_on[0].daemon is cancelled


@pytest.mark.asyncio
async def test_a_cancelled_close_that_cannot_start_a_thread_keeps_the_layouts(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep every layout for the calls in the grace, and stop the TCPStore at once, when no thread can finish the close.

    A WARNING says so, as the layouts then stay until the service is freed.
    """
    monkeypatch.setattr(service_module, "_CLOSE_GRACE_S", 30)
    monkeypatch.setitem(_HELD, "held", (threading.Event(), threading.Event()))

    def no_thread(target: Callable[[], object]) -> NoReturn:
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(service_module, "_start_thread", no_thread)
    caplog.set_level(logging.WARNING, logger=service_module.__name__)
    key = _key()
    try:
        async with _served() as served:
            handle = await served.connect()
            await _register(handle, key)
            # Its load holds the dispatcher, so the close stays in its grace until the test releases it.
            entry: Any = {"x": _Held("held")}
            held = asyncio.ensure_future(
                handle.register_layout("publisher/0", _key(), entry)
            )
            await _until(_HELD["held"][0].is_set)
            rendezvous = served.service._rendezvous
            assert rendezvous is not None
            closing = asyncio.ensure_future(served.service.close())
            await _until(lambda: served.server._closing)
            closing.cancel()
            with pytest.raises(asyncio.CancelledError):
                await closing
            assert rendezvous._store is None
            _HELD["held"][1].set()
            # The call in the grace still finds the registry it registers into.
            await held
            await _until(lambda: served.server._closed.done())
            assert served.service._adapter.registry._layouts[key].complete.is_set()
    finally:
        _HELD["held"][1].set()
    assert "keeps the coordinator's layouts until the service is freed" in caplog.text


@pytest.mark.asyncio
async def test_each_call_logs_one_timing_line(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Log one INFO line per call, failed or not, naming layout keys, as the requester's lines do.

    A successful lookup logs its line once its reply is decoded.
    """
    caplog.set_level(logging.INFO)
    key = _key()
    decode = service_module._decode_layouts
    # How many timing lines were logged as each decode began.
    decoded_after: list[int] = []

    def counted(data: Any) -> Any:
        lines = [r for r in caplog.records if _TIMING_LINE.search(r.getMessage())]
        decoded_after.append(len(lines))
        return decode(data)

    async with _served(publishers=None) as served:
        handle = await served.connect()
        with pytest.raises(RuntimeError, match="init has not run"):
            await handle.strategy()
        with pytest.raises(RuntimeError, match="init has not run"):
            await handle.register_layout("publisher/0", key, {})
        await served.driver.init(_PUBLISHERS, _strategy())
        with pytest.raises(RuntimeError, match="init already ran"):
            await served.driver.init(_PUBLISHERS, _strategy())
        await handle.strategy()
        await _register(handle, key, "publisher/0")
        unpicklable: Any = {"lock": threading.Lock()}
        with pytest.raises(TypeError, match="cannot pickle"):
            await handle.register_layout("publisher/1", key, unpicklable)
        with pytest.raises(TimeoutError, match="missing publishers"):
            await handle.get_layouts(key, timeout_s=0)
        await _register(handle, key, "publisher/1")
        monkeypatch.setattr(service_module, "_decode_layouts", counted)
        assert await handle.get_layouts(key, timeout_s=0) == _layouts(key)
    timed = [
        match.group(1)
        for record in caplog.records
        if record.levelno == logging.INFO
        and (match := _TIMING_LINE.search(record.getMessage()))
    ]
    # Failed calls log too: two before init, a second init and a layout that fails to pickle.
    assert timed == [
        "strategy",
        f"register_layout key={key}",
        "init",
        "init",
        "strategy",
        f"register_layout key={key}",
        f"register_layout key={key}",
        f"get_layouts key={key}",
        f"register_layout key={key}",
        f"get_layouts key={key}",
    ]
    assert decoded_after == [len(timed) - 1]


@pytest.mark.asyncio
async def test_a_client_dialed_through_the_address_names_the_service_kind(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Name a handle ``service`` in TorchStore's selection line, by its ``kind``."""
    monarch = cast(backend_module.Backend, types.SimpleNamespace(rank=lambda: 0))
    monkeypatch.setitem(backend_module._backends, "monarch", monarch)
    caplog.set_level(logging.INFO)
    async with _served() as served:
        address = served.service.address
        try:
            await ts.client(
                address.store_name, role=ts.RankRole.PUBLISHER, address=address
            )
        finally:
            # Shutting down a store with no record would look up Monarch's controller.
            if address.store_name in store_api._control_plane_map:
                await ts.shutdown(address.store_name)
    selection = re.compile(
        rf"routing control_plane=(\S+) address={re.escape(str(address))} "
    )
    kinds = [
        match.group(1)
        for record in caplog.records
        if (match := selection.search(record.getMessage()))
    ]
    assert kinds == ["service"], kinds


def _requester(address: ServiceAddress, key: str, results: Any) -> None:
    async def look_up() -> Any:
        handle = await address.connect()
        try:
            return await handle.get_layouts(key, timeout_s=60)
        finally:
            await handle.close()

    try:
        results.put(("requester", asyncio.run(look_up())))
    except BaseException as e:
        results.put(("error", repr(e)))


@pytest.mark.asyncio
async def test_a_rank_in_another_process_looks_up_layouts_through_the_pickled_address() -> None:
    """Release a waiting requester in a spawned process, which dialed the pickled address, once both publishers register."""
    context = mp.get_context("spawn")
    results = context.Queue()
    key = _key()
    async with _served() as served:
        requester = context.Process(
            target=_requester, args=(served.service.address, key, results)
        )
        requester.start()
        try:
            await _until(
                lambda: served.server.stats()["running"] == 1 or not results.empty(),
                timeout_s=90,
            )
            assert results.empty(), results.get()
            await _register(await served.connect(), key)
            reported = await asyncio.to_thread(results.get, True, 60)
        finally:
            await asyncio.to_thread(requester.join, 60)
            if requester.is_alive():
                requester.kill()
                await asyncio.to_thread(requester.join, 30)
    assert reported == ("requester", _layouts(key))
    assert requester.exitcode == 0


def test_a_forked_child_dials_again_through_an_inherited_handle() -> None:
    """Serve a forked child's lookup through its inherited client, dial again only for the strategy, and refuse its service.

    The fork hook drops the child's copy of the handle's TCPStore connection, keeps unrelated sockets, their
    blocking mode and the default socket timeout, and replaces locks a parent thread held. Closing what it inherited
    leaves the parent's handles and TCPStore working.
    """
    child = _Child(_FORK_CHILD, _store_name())
    try:
        status = child.read_line(timeout_s=90)
    finally:
        stderr = child.kill()
    assert status == "0", stderr[-4000:]


async def _as_inherited(handle: ServiceControlPlane) -> None:
    """Leave ``handle`` as a forked child sees its copy: another pid's, whose TCPStore connection no longer works."""
    handle._pid = -1
    assert handle._rendezvous is not None
    await handle._rendezvous.close()


def _dial_through(
    monkeypatch: pytest.MonkeyPatch,
    dial: Callable[..., Awaitable[TCPStoreRendezvous]],
) -> None:
    """Send the service module's TCPStore dials through ``dial``, leaving every other connect alone."""

    class Dialing(TCPStoreRendezvous):
        @classmethod
        async def connect(
            cls, address: TCPStoreAddress, *, timeout_s: float | None = None
        ) -> TCPStoreRendezvous:
            return await dial(address, timeout_s=timeout_s)

    monkeypatch.setattr(service_module, "TCPStoreRendezvous", Dialing)


@pytest.mark.asyncio
async def test_an_inherited_handle_dials_again_only_to_read_its_tcpstore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dial an inherited handle's TCPStore again only for a read since its inherited client still calls the coordinator.

    Init's refusals come before any dial because init never uses the handle's TCPStore.
    """
    dialed: list[TCPStoreRendezvous] = []

    async def recorded(address: TCPStoreAddress, **kwargs: Any) -> TCPStoreRendezvous:
        dialed.append(await TCPStoreRendezvous.connect(address, **kwargs))
        return dialed[-1]

    key = _key()
    async with _served() as served:
        resolved, unresolved = await served.connect(), await served.connect()
        await _register(resolved, key)
        for handle in (served.driver, resolved, unresolved):
            await _as_inherited(handle)
        monkeypatch.setattr(served.service, "_pid", -1)
        _dial_through(monkeypatch, recorded)
        refusals = (
            (served.driver, "inherited across fork: only the process that started it"),
            (resolved, "init runs only on a handle from CoordinatorService.connect()"),
        )
        for handle, problem in refusals:
            with pytest.raises(RuntimeError, match=re.escape(problem)):
                await handle.init(_PUBLISHERS, _strategy())
        assert await resolved.get_layouts(key, timeout_s=0) == _layouts(key)
        assert dialed == []
        assert type(await resolved.strategy()) is ts.TorchStoreStrategy
        assert len(dialed) == 1 and resolved._rendezvous is dialed[0]
        # Without a client, a first lookup dials to read the coordinator's address.
        assert await unresolved.get_layouts(key, timeout_s=0) == _layouts(key)
        assert len(dialed) == 2 and unresolved._rendezvous is dialed[1]


@pytest.mark.asyncio
async def test_a_close_during_an_inherited_handles_dial_fails_its_call_and_drops_the_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail an inherited handle's first call when a close comes during its dial, closing what it dialed, then dial no more."""
    dialing, proceed = asyncio.Event(), asyncio.Event()
    dialed: list[TCPStoreRendezvous] = []

    async def held(address: TCPStoreAddress, **kwargs: Any) -> TCPStoreRendezvous:
        dialing.set()
        await proceed.wait()
        dialed.append(await TCPStoreRendezvous.connect(address, **kwargs))
        return dialed[-1]

    async with _served() as served:
        handle = await served.connect()
        await _as_inherited(handle)
        _dial_through(monkeypatch, held)
        call = asyncio.ensure_future(handle.strategy())
        await asyncio.wait_for(dialing.wait(), 10)
        await handle.close()
        proceed.set()
        with pytest.raises(RuntimeError, match="this control plane is closed$"):
            await call
        with pytest.raises(RuntimeError, match="this control plane is closed$"):
            await handle.strategy()
    assert len(dialed) == 1 and dialed[0]._store is None


@pytest.mark.asyncio
async def test_racing_first_calls_on_an_inherited_handle_keep_one_new_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the connection of the dial that finished first when first calls on an inherited handle race, closing the other."""
    dialed: list[TCPStoreRendezvous] = []

    async def raced(address: TCPStoreAddress, **kwargs: Any) -> TCPStoreRendezvous:
        dialed.append(await TCPStoreRendezvous.connect(address, **kwargs))
        return dialed[-1]

    async with _served() as served:
        handle = await served.connect()
        await _as_inherited(handle)
        _dial_through(monkeypatch, raced)
        strategies = await asyncio.gather(handle.strategy(), handle.strategy())
        assert {type(strategy) for strategy in strategies} == {ts.TorchStoreStrategy}
        assert handle._rendezvous is dialed[0] and dialed[1]._store is None


@pytest.mark.asyncio
async def test_the_fork_hooks_sets_keep_nothing_alive() -> None:
    """Collect a closed handle and service once dropped: the fork hook's sets hold them weakly."""
    service = await CoordinatorService.start(_store_name(), host=_LOCALHOST)
    handle = await service.address.connect()
    await handle.close()
    await service.close()
    refs = [weakref.ref(handle), weakref.ref(service)]
    del handle, service
    gc.collect()
    assert [ref() for ref in refs] == [None, None]


@pytest.mark.asyncio
async def test_a_killed_service_fails_lookups_reads_and_connects(
    children: list[_Child], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail a waiting lookup within 10 s of a SIGKILL, then reads and connects naming host:port."""
    child = _service_child(children)
    address, key = child.address, _key()
    async with _dialed(address) as connect:
        resolved, unresolved = await connect(), await connect()
        await _register(resolved, key, "publisher/0")
        lookup = asyncio.ensure_future(resolved.get_layouts(key, timeout_s=60))
        await _until(lambda: child.stats()["running"] == 1)
        start = time.monotonic()
        child.kill()
        with pytest.raises(PeerLostError, match=_names_coordinator(address)) as lost:
            await asyncio.wait_for(lookup, 10)
        assert time.monotonic() - start < 10
        # Only a closing coordinator's drop is reworded. A lost connection keeps the carrier's text.
        assert str(lost.value).endswith(
            f": coordinator call failed: {lost.value.__cause__}"
        )
        monkeypatch.setattr(tcpstore, "_DEFAULT_TIMEOUT_S", 9)
        await _fails_at_eof(lambda: _register(unresolved, key, "publisher/1"), address)
        monkeypatch.setattr(tcpstore, "_DEFAULT_TIMEOUT_S", 2)
        with pytest.raises(_UNREACHABLE, match=re.escape(str(address))):
            await address.connect()


@pytest.mark.asyncio
async def test_an_exited_service_fails_waiting_lookups(children: list[_Child]) -> None:
    """Fail a waiting lookup naming the coordinator once the service's process exits, closing nothing first."""
    child = _service_child(children)
    address, key = child.address, _key()
    async with _dialed(address) as connect:
        handle = await connect()
        await _register(handle, key, "publisher/0")
        lookup = asyncio.ensure_future(handle.get_layouts(key, timeout_s=60))
        await _until(lambda: child.stats()["running"] == 1)
        child.send("exit")
        with pytest.raises(PeerLostError, match=_names_coordinator(address)):
            await asyncio.wait_for(lookup, 10)


@pytest.mark.asyncio
async def test_a_stopped_service_times_out_lookups_registrations_and_reads(
    children: list[_Child], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Time out lookups at their wait plus margin, registrations and reads at their deadlines."""
    # A lookup's deadline is its wait plus 60 s by default. This case shortens that.
    assert service_module._REPLY_MARGIN_S == 60
    child = _service_child(children)
    address, key = child.address, _key()
    async with _dialed(address) as connect:
        resolved, unresolved = await connect(), await connect()
        # Caches this handle's coordinator address, so its layout calls below are carrier calls.
        await _register(resolved, key, "publisher/0")
        child.stop()
        monkeypatch.setattr(service_module, "_REPLY_MARGIN_S", 0.25)
        start = time.monotonic()
        with pytest.raises(
            CallTimeoutError, match=_names_coordinator(address)
        ) as raised:
            await resolved.get_layouts(key, timeout_s=0.25)
        assert 0.45 < time.monotonic() - start < 6
        # The deadline is the wait plus the margin.
        assert "within its 0.5 s deadline" in str(raised.value)
        # A registration has the carrier's default deadline, 300 s unless shortened, never
        # one from the margin, which would outlast the bounds below.
        monkeypatch.setattr(service_module, "_REPLY_MARGIN_S", 30)
        monkeypatch.setattr(runtime, "_CALL_TIMEOUT_S", 0.5)
        start = time.monotonic()
        with pytest.raises(CallTimeoutError, match=_names_coordinator(address)):
            await asyncio.wait_for(_register(resolved, key, "publisher/1"), 20)
        assert 0.45 < time.monotonic() - start < 6
        monkeypatch.setattr(tcpstore, "_DEFAULT_TIMEOUT_S", 0.5)
        for call in (resolved.strategy, lambda: unresolved.get_layouts(key)):
            start = time.monotonic()
            with pytest.raises(TimeoutError, match=re.escape(str(address))) as raised:
                await call()
            assert not isinstance(raised.value, CallTimeoutError)
            assert 0.45 < time.monotonic() - start < 6


@pytest.mark.asyncio
async def test_a_stopped_tcpstore_leaves_resolved_handles_working(
    children: list[_Child], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep resolved handles registering and looking up. Fail store reads naming host:port."""
    child = _service_child(children)
    address, key = child.address, _key()
    async with _dialed(address) as connect:
        resolved, unresolved = await connect(), await connect()
        await _register(resolved, key, "publisher/0")
        assert child.command("stop-tcpstore") == "done"
        await _register(resolved, key, "publisher/1")
        assert await resolved.get_layouts(key, timeout_s=0) == _layouts(key)
        for call in (lambda: _register(unresolved, key), resolved.strategy):
            with pytest.raises(_UNREACHABLE) as raised:
                await call()
            assert str(address) in str(raised.value)
            assert not isinstance(raised.value, CarrierError)
            assert "coordinator call failed" not in str(raised.value), raised.value
        monkeypatch.setattr(tcpstore, "_DEFAULT_TIMEOUT_S", 2)
        with pytest.raises(_UNREACHABLE, match=re.escape(str(address))):
            await address.connect()


@pytest.mark.asyncio
async def test_closing_a_service_fails_its_callers_and_new_connects(
    children: list[_Child], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail a waiting lookup within the grace, later calls at once and connects naming host:port."""
    child = _service_child(children, grace_s=1)
    address, key = child.address, _key()
    async with _dialed(address) as connect:
        handle = await connect()
        await _register(handle, key, "publisher/0")
        lookup = asyncio.ensure_future(handle.get_layouts(key, timeout_s=60))
        await _until(lambda: child.stats()["running"] == 1)
        start = time.monotonic()
        child.send("close")
        with pytest.raises(PeerLostError, match=_names_coordinator(address)):
            await asyncio.wait_for(lookup, 10)
        assert time.monotonic() - start < 1 + 2
        assert child.read_line() == "done"
        with pytest.raises(PeerLostError, match=_names_coordinator(address)):
            await asyncio.wait_for(_register(handle, key, "publisher/1"), 5)
        monkeypatch.setattr(tcpstore, "_DEFAULT_TIMEOUT_S", 9)
        await _fails_at_eof(handle.strategy, address)
        with pytest.raises(ConnectionRefusedError):
            socket.create_connection((address.host, address.port), timeout=10).close()
        monkeypatch.setattr(tcpstore, "_DEFAULT_TIMEOUT_S", 2)
        with pytest.raises(_UNREACHABLE, match=re.escape(str(address))):
            await address.connect()


@pytest.mark.asyncio
async def test_closing_a_handle_fails_its_calls_in_flight_and_leaves_other_handles() -> None:
    """Fail a closed handle's lookup at once though another handle shares the connection, and keep that one's.

    The process's last share fails its lookups too, with ``PeerLostError`` naming the coordinator.
    """
    shared, last = _key(), _key()
    async with _served() as served:
        address = served.service.address
        closed, other = await served.connect(), await served.connect()
        for handle in (closed, other):
            await _register(handle, shared, "publisher/0")
        lookups = [
            asyncio.ensure_future(handle.get_layouts(shared, timeout_s=math.inf))
            for handle in (closed, other)
        ]
        await _until(lambda: served.server.stats()["running"] == 2)
        await closed.close()
        closing = f"{_names_coordinator(address)}: this control plane closed$"
        with pytest.raises(PeerLostError, match=closing):
            await asyncio.wait_for(lookups[0], 5)
        # The closed handle's call was abandoned, so the coordinator ended its task.
        await _until(lambda: served.server.stats()["running"] == 1)
        await _register(other, shared, "publisher/1")
        assert await asyncio.wait_for(lookups[1], 10) == _layouts(shared)
        await _register(other, last, "publisher/0")
        lookup = asyncio.ensure_future(other.get_layouts(last, timeout_s=math.inf))
        await _until(lambda: served.server.stats()["running"] == 1)
        await other.close()
        with pytest.raises(PeerLostError, match=_names_coordinator(address)):
            await asyncio.wait_for(lookup, 5)


@pytest.mark.asyncio
async def test_a_closed_handle_refuses_calls_and_a_racing_read_ends() -> None:
    """Refuse every call on a closed handle at once, naming host:port, and end a strategy read that races its close."""
    key = _key()
    async with _served() as served:
        address = served.service.address
        handle = await served.connect()
        await _register(handle, key, "publisher/0")
        rendezvous = handle._rendezvous
        await handle.close()
        await handle.close()
        assert rendezvous is not None and rendezvous._store is None
        calls: list[Callable[[], Awaitable[Any]]] = [
            handle.strategy,
            lambda: _register(handle, key, "publisher/1"),
            lambda: handle.get_layouts(key),
            lambda: handle.init(_PUBLISHERS, _strategy()),
        ]
        for call in calls:
            with pytest.raises(
                RuntimeError, match=f"{re.escape(str(address))}: .* is closed"
            ):
                await asyncio.wait_for(call(), 1)
        for steps in range(4):
            racing = await served.connect()
            read = asyncio.ensure_future(racing.strategy())
            for _ in range(steps):
                await asyncio.sleep(0)
            await racing.close()
            try:
                strategy = await asyncio.wait_for(read, 1)
            except RuntimeError as error:
                assert "is closed" in str(error)
            else:
                assert type(strategy) is ts.TorchStoreStrategy


@pytest.mark.asyncio
async def test_a_stale_nonce_fails_reads_at_once(shared_service: _ChildService) -> None:
    """Report an address from another start as uninitialized at once, for reads and lookups."""
    stale = dataclasses.replace(shared_service.address, nonce=uuid.uuid4().hex)
    async with _dialed(stale) as connect:
        handle = await connect()
        for call in (handle.strategy, lambda: handle.get_layouts(_key())):
            start = time.monotonic()
            with pytest.raises(RuntimeError, match="init has not run"):
                await call()
            assert time.monotonic() - start < 1


@pytest.mark.asyncio
async def test_a_coordinator_key_naming_another_servers_scope_is_stale_once(
    shared_service: _ChildService,
) -> None:
    """Raise ``StaleAddressError`` after one refused hello when the key names another scope."""
    store_name = _store_name()
    rendezvous = await TCPStoreRendezvous.serve(store_name, host=_LOCALHOST)
    try:
        nonce = rendezvous.address.nonce
        # This store's own scope, at a port whose coordinator serves another scope.
        named = dataclasses.replace(
            shared_service.carrier, scope=f"{store_name}/{nonce}"
        )
        await rendezvous.publish(_PUBLISHERS, _strategy(), coordinator=named.to_json())
        address = ServiceAddress(store_name, _LOCALHOST, rendezvous.address.port, nonce)
        rejected = shared_service.stats()["rejected"]
        async with _dialed(address) as connect:
            handle = await connect()
            with pytest.raises(StaleAddressError, match=_names_coordinator(address)):
                await asyncio.wait_for(_register(handle, _key(), "publisher/0"), 5)
        assert shared_service.stats()["rejected"] == rejected + 1
    finally:
        await rendezvous.close()


@pytest.mark.asyncio
async def test_the_coordinator_takes_only_its_nonce_scope_and_its_two_layout_calls() -> None:
    """Refuse a hello without this start's nonce, and a peer's ``init``, ``drop_encodings`` or ``drop_layouts``.

    The nonce keeps out clients of other starts, not attackers, as the network is trusted. Served, ``drop_layouts``
    would drop every layout, ``drop_encodings`` every part and reply, and an early ``init`` would fix the publishers.
    """
    key = _key()
    async with _served() as served:
        await _register(served.driver, key)
        address, store_name = served.server.address, served.service.address.store_name
        assert address.scope == f"{store_name}/{served.service.address.nonce}"
        for scope in (store_name, f"{store_name}/{uuid.uuid4().hex}"):
            stranger = CarrierClient(dataclasses.replace(address, scope=scope))
            try:
                with pytest.raises(StaleAddressError, match="the scope does not match"):
                    await stranger.call("get_layouts", key, 0)
            finally:
                await stranger.close()
        raw = CarrierClient(address)
        try:
            for endpoint, args in (
                ("init", (["intruder"], None)),
                ("drop_encodings", ()),
                ("drop_layouts", ()),
            ):
                with pytest.raises(
                    AttributeError, match=f"has no endpoint {endpoint!r}$"
                ):
                    await raw.call(endpoint, *args)
        finally:
            await raw.close()
        assert await served.driver.get_layouts(key, timeout_s=0) == _layouts(key)
