# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import contextlib
import copy
import dataclasses
import gc
import ipaddress
import json
import logging
import math
import os
import pickle
import re
import select
import signal
import socket
import subprocess
import sys
import threading
import time
import tracemalloc
import types
import uuid
import weakref
import zlib
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from datetime import timedelta
from typing import Any, cast, NoReturn

import cloudpickle
import pytest
import torchstore as ts
from monarch.actor import Actor, endpoint, this_host
from torch.distributed import DistNetworkError, DistStoreError, TCPStore
from torchstore.monarch.storage_volume import StorageVolume
from torchstore.routing import tcpstore
from torchstore.routing.tcpstore import TCPStoreAddress, TCPStoreRendezvous
from torchstore.transport import TransportType

pytestmark = [
    pytest.mark.timeout(120),
    # A native-call thread that raises, say on a closed loop, fails its test.
    pytest.mark.filterwarnings("error::pytest.PytestUnhandledThreadExceptionWarning"),
]

_LOCALHOST = "127.0.0.1"
_UNREACHABLE: tuple[type[OSError], ...] = (ConnectionError, TimeoutError)
_PUBLISHER_RANKS = frozenset({"publisher/0", "publisher/1"})
# A TCPStore server the tests can stop or kill. It exits when its stdin closes.
_SERVER = (
    "import sys\n"
    "from torch.distributed import TCPStore\n"
    f"store = TCPStore({_LOCALHOST!r}, 0, is_master=True, wait_for_workers=False)\n"
    "print(store.port, flush=True)\n"
    "sys.stdin.read()\n"
)
# Connects within a deadline, then exits. argv: port, timeout_s, teardown_s.
# Finalization waits up to teardown_s while a native call's thread exists, so one
# that outlives the drain returns into it.
_CONNECT_THEN_EXIT = (
    "import asyncio, os, sys, threading, time\n"
    "from torchstore.routing.tcpstore import TCPStoreAddress, TCPStoreRendezvous\n"
    "port = int(sys.argv[1])\n"
    "timeout_s, teardown_s = map(float, sys.argv[2:])\n"
    "class SlowTeardown:\n"
    "    def __init__(self, threads, seconds):\n"
    "        self.tasks = [f'/proc/self/task/{t.native_id}' for t in threads]\n"
    "        self.seconds = seconds\n"
    "    def __del__(self, sleep=time.sleep, now=time.monotonic, has=os.path.exists):\n"
    "        deadline = now() + self.seconds  # holds finalization open\n"
    "        while now() < deadline and any(map(has, self.tasks)):\n"
    "            sleep(0.01)\n"
    "async def connect():\n"
    f"    address = TCPStoreAddress({_LOCALHOST!r}, port, 'store', 'nonce')\n"
    "    try:\n"
    "        await TCPStoreRendezvous.connect(address, timeout_s=timeout_s)\n"
    "    except TimeoutError:\n"
    "        print('timed out', flush=True)\n"
    "asyncio.run(connect())\n"
    "prefix = 'torchstore-tcpstore'\n"
    "native = [t for t in threading.enumerate() if t.name.startswith(prefix)]\n"
    "teardown = SlowTeardown(native, teardown_s)\n"
    "print('exiting at', time.monotonic(), flush=True)\n"
)
# Skips the Python dial before each native connect attempt, so attempts still dial a
# refusing address natively, as after a server stops listening once it answered.
_DIAL_NATIVELY = (
    "from torchstore.routing import tcpstore\n"
    "tcpstore._check_answers = lambda *args: None\n"
)
# Imports the module alone, without torchstore's package and so without Monarch.
# argv[1] is the repository root.
_IMPORT_ALONE = (
    "import sys, types\n"
    "for name in ('torchstore', 'torchstore.routing'):\n"
    "    package = sys.modules[name] = types.ModuleType(name)\n"
    "    package.__path__ = [sys.argv[1] + '/' + name.replace('.', '/')]\n"
    "from torchstore.routing import tcpstore\n"
)
# Checks the module loaded no Monarch or cloudpickle module, even through a guarded
# import, then decodes with Monarch blocked. argv: root, hexes.
_DECODE_WITHOUT_MONARCH = _IMPORT_ALONE + (
    "loaded = [m for m in sys.modules if m.split('.')[0] in ('monarch', 'cloudpickle')]\n"
    "assert not loaded, loaded\n"
    "sys.modules['monarch'] = None\n"
    "for encoded in sys.argv[2:]:\n"
    "    try:\n"
    "        decoded = tcpstore._decode_strategy(bytes.fromhex(encoded))\n"
    "        print(type(decoded).__name__, flush=True)\n"
    "    except ModuleNotFoundError as error:\n"
    "        print(error.name.split('.')[0], flush=True)\n"
)
# Serves, connects, writes and checks with Monarch and cloudpickle blocked. argv[1]
# is the repository root.
_SERVE_WITHOUT_CLOUDPICKLE = _IMPORT_ALONE + (
    "import asyncio, time\n"
    "sys.modules['monarch'] = sys.modules['cloudpickle'] = None\n"
    "from torchstore.routing.tcpstore import TCPStoreRendezvous\n"
    "async def main():\n"
    "    served = await TCPStoreRendezvous.serve('plain', host='127.0.0.1')\n"
    "    other = await TCPStoreRendezvous.connect(served.address)\n"
    "    await served._compare_set('k', b'v', time.monotonic() + 30)\n"
    "    print(await other._check(['k'], time.monotonic() + 30), flush=True)\n"
    "    await other.close()\n"
    "    await served.close()\n"
    "asyncio.run(main())\n"
)
# Forks a child in four cases and records each child's exit status, or "hung" for a child left hanging.
# The children of cases 1 and 3 exit with how many descriptors they kept on a served store's port, counted
# by their own scan. Case 2's adds 1 if its grandchild failed, and case 4's exits 1 if it was forked while
# the holder held the lock. Prints the error the first child's inherited handle raised, the results, then
# the parent's own read. argv[1] is the root.
_FORK_CASES = _IMPORT_ALONE + (
    "import asyncio, gc, json, os, socket, threading, time\n"
    "from torchstore.routing.tcpstore import TCPStoreRendezvous\n"
    "# Under a default timeout, wrapping a descriptor turns it non-blocking in both processes.\n"
    "socket.setdefaulttimeout(30)\n"
    "results = {}\n"
    "def kept(port):\n"
    "    socket.setdefaulttimeout(None)\n"
    "    count = 0\n"
    "    for fd in map(int, os.listdir('/proc/self/fd')):\n"
    "        try:\n"
    "            if os.readlink(f'/proc/self/fd/{fd}').startswith('socket:'):\n"
    "                with socket.socket(fileno=os.dup(fd)) as sock:\n"
    "                    tcp = sock.family in (socket.AF_INET, socket.AF_INET6)\n"
    "                    count += tcp and sock.getsockname()[1] == port\n"
    "        except OSError:\n"
    "            pass\n"
    "    return count\n"
    "def waited(pid):\n"
    "    deadline = time.monotonic() + 30\n"
    "    while time.monotonic() < deadline:\n"
    "        done, status = os.waitpid(pid, os.WNOHANG)\n"
    "        if done:\n"
    "            return os.waitstatus_to_exitcode(status)\n"
    "        time.sleep(0.05)\n"
    "    os.kill(pid, 9)\n"
    "    os.waitpid(pid, 0)\n"
    "    return 'hung'\n"
    "# 1. As a second serve's native server holds its handed listener, before serve records it.\n"
    "class ForkingStore(tcpstore.TCPStore):\n"
    "    def set_timeout(self, timeout):\n"
    "        super().set_timeout(timeout)\n"
    "        pid = os.fork()\n"
    "        if pid == 0:\n"
    "            try:\n"
    "                asyncio.run(inherited._check(['k'], time.monotonic() + 30))\n"
    "                print('used the inherited connection', flush=True)\n"
    "            except RuntimeError as error:\n"
    "                print(error, flush=True)\n"
    "            os._exit(kept(self.port))\n"
    "        results['handoff'] = waited(pid)\n"
    "inherited = asyncio.run(TCPStoreRendezvous.serve('inherited', host='127.0.0.1'))\n"
    "tcpstore.TCPStore = ForkingStore\n"
    "asyncio.run(TCPStoreRendezvous.serve('fork', host='127.0.0.1'))\n"
    "tcpstore.TCPStore = ForkingStore.__base__\n"
    "# 2. After both serves: the child serves, forks and drops the inherited store.\n"
    "pid = os.fork()\n"
    "if pid == 0:\n"
    "    count = kept(inherited.address.port)\n"
    "    served = asyncio.run(\n"
    "        TCPStoreRendezvous.serve('child', host='127.0.0.1', timeout_s=10)\n"
    "    )\n"
    "    grandchild = os.fork()\n"
    "    if grandchild == 0:\n"
    "        os._exit(0)\n"
    "    count += waited(grandchild) != 0\n"
    "    asyncio.run(served.close())\n"
    "    del inherited\n"
    "    gc.collect()\n"
    "    sys.exit(count)\n"
    "results['after_serves'] = waited(pid)\n"
    "# 3. After close(), as a call's result lands: only that call's PrefixStore holds the server.\n"
    "closing = asyncio.run(TCPStoreRendezvous.serve('closing', host='127.0.0.1'))\n"
    "closing_port = closing.address.port\n"
    "settle = tcpstore._settle\n"
    "def fork_as_the_result_lands(future, result, error):\n"
    "    settle(future, result, error)\n"
    "    tcpstore._settle = settle\n"
    "    pid = os.fork()\n"
    "    if pid == 0:\n"
    "        raise SystemExit(kept(closing_port))  # through the loop and finalization\n"
    "    results['closed'] = waited(pid)\n"
    "async def call_through_close():\n"
    "    entered, closed = threading.Event(), threading.Event()\n"
    "    def held(store):\n"
    "        entered.set()\n"
    "        closed.wait(30)\n"
    "        return store.check(['k'])\n"
    "    call = asyncio.ensure_future(closing._call('held', held, time.monotonic() + 30))\n"
    "    while not entered.is_set():\n"
    "        await asyncio.sleep(0.01)\n"
    "    await closing.close()\n"
    "    gc.collect()\n"
    "    tcpstore._settle = fork_as_the_result_lands\n"
    "    closed.set()\n"
    "    await call\n"
    "asyncio.run(call_through_close())\n"
    "# 4. While another thread holds the probe-listener lock: the child exits 1 if it was forked while the\n"
    "# holder held it. The holder lets go once the fork has begun and either went ahead or waited 0.5 s.\n"
    "holding, entering, forked = threading.Event(), threading.Event(), threading.Event()\n"
    "inside = [False]\n"
    "os.register_at_fork(before=entering.set)  # runs before the module's own hook\n"
    "def hold_probe_lock():\n"
    "    with tcpstore._probe_listeners_lock:\n"
    "        inside[0] = True\n"
    "        holding.set()\n"
    "        entering.wait(30)\n"
    "        forked.wait(0.5)\n"
    "        inside[0] = False\n"
    "def fork_from_a_thread():\n"
    "    pid = os.fork()\n"
    "    if pid == 0:\n"
    "        os._exit(inside[0])\n"
    "    forked.set()\n"
    "    results['probe_lock'] = waited(pid)\n"
    "holder = threading.Thread(target=hold_probe_lock)\n"
    "holder.start()\n"
    "holding.wait(30)\n"
    "forker = threading.Thread(target=fork_from_a_thread)\n"
    "forker.start()\n"
    "holder.join(30)\n"
    "forker.join(30)\n"
    "print(json.dumps(results), flush=True)\n"
    "print(asyncio.run(inherited._check(['k'], time.monotonic() + 30)), flush=True)\n"
)


def _store_name() -> str:
    return f"tcpstore-test-{uuid.uuid4()}"


def _strategy() -> ts.TorchStoreStrategy:
    return ts.TorchStoreStrategy(TransportType.MonarchRPC)


def _deadline() -> float:
    return time.monotonic() + 60


def _native_calls(name: str = "") -> list[threading.Thread]:
    prefix = f"torchstore-tcpstore {name}"
    return [t for t in threading.enumerate() if t.name.startswith(prefix)]


_earlier_calls: set[threading.Thread] = set()


def _join_native_calls() -> None:
    """Let this test's native calls finish, so none outlives the test."""
    mine = set(_native_calls()) - _earlier_calls
    for thread in mine:
        thread.join(timeout=60)
    alive = [thread for thread in mine if thread.is_alive()]
    assert not alive, "native TCPStore calls outlived the test"


@pytest.fixture(autouse=True)
def _joins_its_native_calls() -> Iterator[None]:
    """Fail the test that leaves a native call running, not a later one."""
    _earlier_calls.clear()
    _earlier_calls.update(_native_calls())
    yield
    _join_native_calls()
    assert not tcpstore._live_calls - _earlier_calls, "finished calls stayed registered"


def _stop(process: subprocess.Popen[bytes]) -> None:
    """Stop the server and wait until the kernel reports it stopped."""
    process.send_signal(signal.SIGSTOP)
    os.waitpid(process.pid, os.WUNTRACED)


def _connect_then_exit(
    port: int,
    timeout_s: float,
    teardown_s: float = 0,
    prefix: str = "",
) -> bytes:
    """Connect from a fresh interpreter that times out, then exits with status 0."""
    args = (port, timeout_s, teardown_s)
    child = subprocess.run(
        [sys.executable, "-c", prefix + _CONNECT_THEN_EXIT, *map(str, args)],
        capture_output=True,
        timeout=60,
        cwd=os.path.dirname(os.path.dirname(ts.__file__)),
    )
    assert b"timed out" in child.stdout
    assert child.returncode == 0, child.stderr.decode()[-2000:]
    # CLOCK_MONOTONIC is system-wide, so this bounds the exit drain and the teardown.
    assert time.monotonic() - float(child.stdout.split()[-1]) < 20
    return child.stdout


async def _until(condition: Callable[[], object], timeout_s: float = 30) -> None:
    deadline = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline, "condition not met before the deadline"
        await asyncio.sleep(0.01)


def _two_addresses(
    first: dict[str, str], *, localhost_first: bool = False
) -> Callable[..., Any]:
    """Resolve each name in ``first`` to its address and 127.0.0.1, in that order unless ``localhost_first``."""
    lookup = socket.getaddrinfo

    def resolve(host: str, *args: Any, **kwargs: Any) -> Any:
        if host not in first:
            return lookup(host, *args, **kwargs)
        stream = (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "")
        numerics = (first[host], _LOCALHOST)
        return [
            (*stream, (numeric, 0))
            for numeric in (numerics[::-1] if localhost_first else numerics)
        ]

    return resolve


def _ipv4_only_store() -> TCPStore:
    """Serve a libuv TCPStore that listens on 127.0.0.1 alone."""
    listener = socket.create_server((_LOCALHOST, 0))
    return TCPStore(
        _LOCALHOST,
        listener.getsockname()[1],
        is_master=True,
        wait_for_workers=False,
        master_listen_fd=listener.detach(),
        use_libuv=True,
    )


def _has_ipv6_loopback() -> bool:
    """Whether servers here listen dual-stack and the IPv6 loopback address exists."""
    if not socket.has_dualstack_ipv6():
        return False
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as probe:
            probe.bind(("::1", 0))
    except OSError:
        return False
    return True


@contextlib.asynccontextmanager
async def _served(
    host: str | None = _LOCALHOST, store_name: str | None = None
) -> AsyncIterator[TCPStoreRendezvous]:
    rendezvous = await TCPStoreRendezvous.serve(store_name or _store_name(), host=host)
    try:
        yield rendezvous
    finally:
        await rendezvous.close()


@contextlib.asynccontextmanager
async def _connected(address: TCPStoreAddress) -> AsyncIterator[TCPStoreRendezvous]:
    rendezvous = await TCPStoreRendezvous.connect(address)
    try:
        yield rendezvous
    finally:
        await rendezvous.close()


@pytest.fixture
def server() -> Iterator[tuple[subprocess.Popen[bytes], TCPStoreAddress]]:
    """Serve a TCPStore from a subprocess so a test can stop or kill it."""
    with subprocess.Popen(
        [sys.executable, "-c", _SERVER], stdin=subprocess.PIPE, stdout=subprocess.PIPE
    ) as process:
        try:
            assert process.stdout is not None
            readable, _, _ = select.select([process.stdout], [], [], 60)
            line = process.stdout.readline() if readable else b""
            if not line:
                process.kill()  # a no-op once it has exited
                pytest.fail(
                    "the TCPStore server subprocess did not report its port "
                    f"(exit status {process.wait(timeout=60)})"
                )
            nonce = uuid.uuid4().hex
            yield process, TCPStoreAddress(_LOCALHOST, int(line), _store_name(), nonce)
        finally:
            process.kill()
            process.wait(timeout=60)


class _StrategyReader(Actor):
    """Reads a published strategy in its own process and calls its volumes."""

    @endpoint
    async def read_volumes(self, address: TCPStoreAddress) -> tuple[str, dict, dict]:
        async with _connected(address) as rendezvous:
            publishers, strategy = await rendezvous.read()
        replies = {}
        for rank in sorted(publishers):
            ref = strategy.get_storage_volume(rank)
            reply = tuple(await ref.volume.get_id.call_one())
            replies[rank] = (ref.volume_hostname, reply)
        return type(strategy).__name__, dict(strategy.volume_id_to_hostname), replies


@pytest.mark.asyncio
async def test_size_guards_reject_oversized_requests_before_sending(caplog) -> None:
    """Accept requests at the server's limits and reject larger ones locally."""
    caplog.set_level(logging.INFO, logger=tcpstore.__name__)
    # A two-byte character in the store name makes UTF-8 bytes and characters differ.
    async with _served(store_name=f"tcpstore-test-é-{uuid.uuid4()}") as rendezvous:
        assert rendezvous._store.libuvBackend  # these are the libuv server's limits
        # 1. A value of exactly the limit is stored. One more byte is rejected.
        at_limit = b"x" * (8 * 1024 * 1024)
        assert await rendezvous._compare_set("value", at_limit, _deadline()) == at_limit
        with pytest.raises(ValueError, match="value for 'over' has"):
            await rendezvous._compare_set("over", at_limit + b"x", _deadline())

        # 2. The same holds for the wire key ("/" + prefix + "/" + key), on reads too.
        prefix = tcpstore._prefix(rendezvous.address)
        longest = "k" * (8 * 1024 - len(f"/{prefix}/".encode()))
        assert tcpstore._check_sizes(prefix, longest) == 8 * 1024  # Store-only callers
        assert await rendezvous._compare_set(longest, b"v", _deadline()) == b"v"
        with pytest.raises(ValueError, match="bytes on the wire"):
            await rendezvous._compare_set(longest + "k", b"v", _deadline())
        for read in (rendezvous._check, rendezvous._multi_get):
            with pytest.raises(ValueError, match="bytes on the wire"):
                await read([longest + "k"], _deadline())

        # 3. Nothing rejected reached the server, so the same handle still works.
        assert not await rendezvous._check(["over"], _deadline())
        half = tcpstore._MAX_VALUE_BYTES // 2
        await rendezvous._compare_set("half", b"x" * half, _deadline())
        await rendezvous._compare_set("over_half", b"x" * (half + 1), _deadline())
        half_key = "h" * (4 * 1024 - len(f"/{prefix}/".encode()))
        await rendezvous._compare_set(half_key, b"v", _deadline())
        await rendezvous._compare_set(half_key + "h", b"v", _deadline())

        # 4. publish checks every size before it writes anything. Random hex stays
        # over the limit once the strategy is compressed.
        oversized = _strategy()
        oversized.volume_id_to_coord = {"publisher/0": {"procs": 0}}
        oversized.volume_id_to_hostname = {"publisher/0": os.urandom(half * 2).hex()}
        with pytest.raises(ValueError, match="value for 'strategy' has"):
            await rendezvous.publish(_PUBLISHER_RANKS, oversized)
        assert not await rendezvous._check([tcpstore._PUBLISHERS], _deadline())
        over_half = b"x" * (half + 1)
        for key in (tcpstore._PUBLISHERS, tcpstore._STRATEGY, half_key + "hh"):
            await rendezvous._compare_set(key, over_half, _deadline())

    # 5. Sizes are logged, as warnings once over half a limit.
    def logged(level: int) -> str:
        return "\n".join(r.getMessage() for r in caplog.records if r.levelno == level)

    # Each WARNING names the limits it is over half of and what keeps that key under them.
    warned = logged(logging.WARNING)
    fails = "past which a write fails: use"
    value_limit = f"over half the limit of 8388608 value bytes {fails}"
    for key, remedy in [
        ("value", "a smaller value"),
        (tcpstore._PUBLISHERS, "fewer or shorter publisher names"),
        (tcpstore._STRATEGY, "a smaller strategy"),
    ]:
        pattern = rf"key {key!r} \(\d+ key bytes\) on \S+, {value_limit} {remedy}$"
        assert re.search(pattern, warned, re.M), warned
    key_limit = f"over half the limit of 8192 key bytes {fails} a shorter store name"
    both = (
        "over half the limits of 8192 key bytes and 8388608 value bytes "
        f"{fails} a shorter store name and a smaller value"
    )
    assert re.search(rf"key {longest!r} .*, {key_limit}$", warned, re.M), warned
    assert re.search(rf"key {half_key + 'hh'!r} .*, {both}$", warned, re.M), warned
    assert "over half" not in logged(logging.INFO)
    assert "key 'over_half'" in logged(logging.WARNING)
    assert "key 'half'" in logged(logging.INFO)
    assert f"key {half_key + 'h'!r}" in logged(logging.WARNING)
    assert f"key {half_key!r}" in logged(logging.INFO)


def test_a_strategy_over_the_inflate_limit_is_neither_encoded_nor_decoded(
    monkeypatch,
) -> None:
    """Refuse to encode or decode a strategy whose pickle is over the inflate limit.

    Decoding stops inflating just past the limit, so a small value never inflates far beyond it.
    """
    strategy = _strategy()
    size = len(pickle.dumps(strategy))
    monkeypatch.setattr(tcpstore, "_MAX_STRATEGY_BYTES", size)
    encoded = tcpstore._encode_strategy(strategy)
    assert type(tcpstore._decode_strategy(encoded)) is ts.TorchStoreStrategy
    monkeypatch.setattr(tcpstore, "_MAX_STRATEGY_BYTES", size - 1)
    with pytest.raises(ValueError, match=f"pickles to {size} bytes"):
        tcpstore._encode_strategy(strategy)
    with pytest.raises(ValueError, match=f"inflates past {size - 1} bytes"):
        tcpstore._decode_strategy(encoded)

    deflate = zlib.compressobj()
    zeros = [deflate.compress(bytes(1 << 20)) for _ in range(64)]
    bomb = b"".join([*zeros, deflate.flush()])  # about 64 KB that inflate to 64 MiB
    tracemalloc.start()
    try:
        with pytest.raises(ValueError, match="inflates past"):
            tcpstore._decode_strategy(bomb)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 8 << 20, peak


def test_exit_drains_for_5_s_and_readers_inflate_up_to_256_mib() -> None:
    """Keep the documented bounds: 5 s for the exit drain and 256 MiB for an inflated strategy.

    Other cases shrink both. Only this one fails if either grows.
    """
    assert tcpstore._EXIT_DRAIN_S == 5.0
    assert tcpstore._MAX_STRATEGY_BYTES == 256 << 20


@pytest.mark.asyncio
async def test_first_publish_wins_and_different_content_raises() -> None:
    """Keep the first published content and reject a conflicting publish."""
    published = _strategy()
    async with _served() as rendezvous, _connected(rendezvous.address) as other:
        # Neither connect deadline became the timeout of the hidden waits.
        for handle in (rendezvous, other):
            assert handle._store.timeout == timedelta(seconds=300)
        await rendezvous.publish([*_PUBLISHER_RANKS, "publisher/0"], published)
        await other.publish(_PUBLISHER_RANKS, published)  # the same set and strategy

        conflict = "already initialized .* 'publishers'"
        # The second set encodes to as many bytes as the published one.
        for conflicting in ({"publisher/0"}, {"publisher/2", "publisher/3"}):
            with pytest.raises(RuntimeError, match=conflict):
                await other.publish(conflicting, published)

        publishers, _ = await other.read()
        assert publishers == _PUBLISHER_RANKS


@pytest.mark.asyncio
async def test_publish_over_another_schemas_ready_marker_raises() -> None:
    """Refuse to publish where a ready marker of another schema already sits."""
    async with _served() as rendezvous:
        other_schema = tcpstore._SCHEMA_VERSION + b"x"
        await rendezvous._compare_set(tcpstore._READY, other_schema, _deadline())
        with pytest.raises(RuntimeError, match="different 'ready'"):
            await rendezvous.publish(_PUBLISHER_RANKS, _strategy())


@pytest.mark.asyncio
async def test_publish_writes_ready_last_and_read_reads_it_first(monkeypatch) -> None:
    """Write ready last with compare_set and read ready alone first, one deadline each.

    The strategy is decoded on the calling thread, inside its Monarch context.
    """
    calls: list[tuple[str, object]] = []
    deadlines: list[float] = []
    decoded_on: list[int] = []
    run, decode = tcpstore._run_with_deadline, tcpstore._decode_strategy

    async def record(fn: Callable[[], object], deadline: float, what: str) -> object:
        deadlines.append(deadline)
        return await run(fn, deadline, what)

    def record_decode(data: bytes) -> object:
        decoded_on.append(threading.get_ident())
        return decode(data)

    async with _served() as rendezvous, _connected(rendezvous.address) as other:
        prefixed = rendezvous._prefixed

        def recorded(name: str) -> Callable[..., object]:
            def call(*args: object) -> object:
                calls.append((name, args[0]))
                return getattr(prefixed, name)(*args)

            return call

        # No set or multi_set: they return before the server applies them.
        rendezvous._prefixed = types.SimpleNamespace(  # type: ignore[assignment]
            **{name: recorded(name) for name in ("compare_set", "check", "multi_get")}
        )
        monkeypatch.setattr(tcpstore, "_run_with_deadline", record)
        monkeypatch.setattr(tcpstore, "_decode_strategy", record_decode)
        await rendezvous.publish(_PUBLISHER_RANKS, _strategy())
        await rendezvous.read()
        monkeypatch.undo()
        rendezvous._prefixed = prefixed
        assert await other._check([tcpstore._READY], _deadline())

    ready, rest = [tcpstore._READY], [tcpstore._PUBLISHERS, tcpstore._STRATEGY]
    assert calls == [
        *[("compare_set", key) for key in (*rest, tcpstore._READY)],
        ("check", ready),
        ("multi_get", ready),
        ("check", rest),
        ("multi_get", rest),
    ]
    assert deadlines == [deadlines[0]] * 3 + [deadlines[3]] * 4, deadlines
    assert decoded_on == [threading.get_ident()]


@pytest.mark.asyncio
async def test_the_strategy_read_reads_ready_and_the_strategy_alone(
    monkeypatch,
) -> None:
    """Read the strategy in four store calls under one deadline, never touching the publisher ranks."""
    seen: list[tuple[str, tuple[str, ...]]] = []
    deadlines: set[float] = set()
    check, multi_get = TCPStoreRendezvous._check, TCPStoreRendezvous._multi_get

    async def counted_check(self: TCPStoreRendezvous, keys, deadline: float) -> bool:
        seen.append(("check", tuple(keys)))
        deadlines.add(deadline)
        return await check(self, keys, deadline)

    async def counted_multi_get(
        self: TCPStoreRendezvous, keys, deadline: float
    ) -> list[bytes]:
        seen.append(("multi_get", tuple(keys)))
        deadlines.add(deadline)
        return await multi_get(self, keys, deadline)

    monkeypatch.setattr(TCPStoreRendezvous, "_check", counted_check)
    monkeypatch.setattr(TCPStoreRendezvous, "_multi_get", counted_multi_get)
    served = await TCPStoreRendezvous.serve(_store_name(), host=_LOCALHOST)
    try:
        await served.publish({f"publisher/{i}" for i in range(512)}, _strategy())
        seen.clear()
        deadlines.clear()
        assert type(await served.read_strategy()) is ts.TorchStoreStrategy
        assert len(deadlines) == 1, deadlines
        ready, strategy = (tcpstore._READY,), (tcpstore._STRATEGY,)
        assert seen == [
            ("check", ready),
            ("multi_get", ready),
            ("check", strategy),
            ("multi_get", strategy),
        ]
    finally:
        await served.close()


class _ModuleStrategy(ts.TorchStoreStrategy):
    """Defined at module level, so the standard pickler encodes it by name."""


@pytest.mark.asyncio
async def test_keys_sit_under_the_store_name_and_nonce() -> None:
    """Pin the wire keys and values under torchstore/routing/{store_name}/{nonce}."""
    published = _ModuleStrategy(TransportType.MonarchRPC)
    store_name = _store_name()
    async with _served(store_name=store_name) as rendezvous:
        await rendezvous.publish(_PUBLISHER_RANKS, published)
        nonce, unprefixed = rendezvous.address.nonce, rendezvous._store
        prefix = f"torchstore/routing/{store_name}/{nonce}"
        keys = [f"{prefix}/{key}" for key in ("publishers", "strategy", "ready")]
        # check never waits, unlike multi_get.
        assert unprefixed.check(keys), f"the published keys are not under {prefix!r}"
        publishers, strategy, ready = unprefixed.multi_get(keys)
        _, decoded = await rendezvous.read()
    assert json.loads(publishers) == sorted(_PUBLISHER_RANKS)
    ranks = [f"publisher/{i}" for i in range(16)]
    encoded = tcpstore._encode_publishers(ranks[::-1])
    assert encoded == json.dumps(sorted(ranks)).encode()
    assert strategy == tcpstore._encode_strategy(published)
    assert type(decoded) is _ModuleStrategy
    assert ready == b"1"


@pytest.mark.asyncio
async def test_a_strategy_class_defined_in_a_function_is_refused_before_any_write() -> None:
    """Refuse a strategy whose class has no importable name, and write nothing.

    Only a by-value pickler such as cloudpickle encodes it, and the rendezvous pickles by name.
    """

    class LocalStrategy(ts.TorchStoreStrategy):
        pass

    refused: tuple[type[Exception], ...] = (AttributeError, pickle.PicklingError)
    async with _served() as rendezvous:
        with pytest.raises(refused, match="local"):
            await rendezvous.publish(_PUBLISHER_RANKS, LocalStrategy())
        with pytest.raises(RuntimeError, match="init has not run"):
            await rendezvous.read()


@pytest.mark.asyncio
async def test_reads_fail_fast_before_publish_and_under_a_stale_nonce() -> None:
    """Report an unpublished or failed store at once instead of blocking in a wait."""
    store_name = _store_name()
    async with _served(store_name=store_name) as earlier:
        stale_nonce = earlier.address.nonce
        start = time.monotonic()
        with pytest.raises(RuntimeError, match="init has not run"):
            await earlier.read()
        assert time.monotonic() - start < 10

        # A publish that fails part way never writes the ready marker.
        await earlier._compare_set(tcpstore._STRATEGY, b"other", _deadline())
        with pytest.raises(RuntimeError, match="different 'strategy'"):
            await earlier.publish(_PUBLISHER_RANKS, _strategy())
        with pytest.raises(RuntimeError, match="init has not run"):
            await earlier.read()
    async with _served(store_name=store_name) as rendezvous:
        # An address from an earlier serve finds nothing, even on this store.
        await rendezvous.publish(_PUBLISHER_RANKS, _strategy())
        stale = dataclasses.replace(rendezvous.address, nonce=stale_nonce)
        async with _connected(stale) as reader:
            with pytest.raises(RuntimeError, match="init has not run"):
                await reader.read()


@pytest.mark.parametrize(
    ("ready", "error", "strategy_error"),
    [
        (b"2", "schema version b'2'", "schema version b'2'"),
        (
            tcpstore._SCHEMA_VERSION + b"0",
            f"schema version {tcpstore._SCHEMA_VERSION + b'0'!r}",
            f"schema version {tcpstore._SCHEMA_VERSION + b'0'!r}",
        ),
        (tcpstore._SCHEMA_VERSION, "without its publishers", "without its strategy"),
    ],
    ids=["other_schema", "longer_schema", "ready_without_strategy"],
)
@pytest.mark.asyncio
async def test_reads_reject_unknown_schemas_and_missing_keys_at_once(
    ready: bytes, error: str, strategy_error: str
) -> None:
    """Refuse a ready marker of another schema, or one missing keys, at once, on both reads."""
    publishers = tcpstore._encode_publishers(())
    async with _served() as rendezvous:
        await rendezvous._compare_set(tcpstore._PUBLISHERS, publishers, _deadline())
        await rendezvous._compare_set(tcpstore._READY, ready, _deadline())

        # A hidden multi_get wait would time out first and block the handle.
        with pytest.raises(RuntimeError, match=re.escape(error)):
            await rendezvous.read(timeout_s=5)
        with pytest.raises(RuntimeError, match=re.escape(strategy_error)):
            await rendezvous.read_strategy(timeout_s=5)
        assert await rendezvous._check([tcpstore._READY], _deadline())


@pytest.mark.asyncio
async def test_published_volume_handles_work_from_another_process() -> None:
    """Call storage volumes through a strategy read back in another process."""
    store_name = _store_name()
    strategy = _strategy()
    meshes = [this_host().spawn_procs(per_host={"procs": len(_PUBLISHER_RANKS)})]
    try:
        meshes.append(this_host().spawn_procs(per_host={"procs": 1}))
        await ts.initialize(
            mesh=meshes[0],
            strategy=strategy,
            client_type=ts.ClientType.ROUTING,
            store_name=store_name,
        )
        reader = meshes[1].spawn("strategy_reader", _StrategyReader)
        driver_hostnames = dict(strategy.volume_id_to_hostname)
        # Hostnames only drive a same-host check, so handles work with fake ones.
        # Each volume gets its own, so a hostname read for the wrong volume shows.
        fake_names = {v: v.replace("/", "-") + ".invalid" for v in driver_hostnames}
        fake = copy.copy(strategy)
        fake.volume_id_to_hostname = fake_names
        reads = []
        for published in (strategy, fake):
            async with _served(store_name=store_name) as rendezvous:
                await rendezvous.publish(_PUBLISHER_RANKS, published)
                reads.append(await reader.read_volumes.call_one(rendezvous.address))
    finally:
        await asyncio.gather(*(mesh.stop() for mesh in meshes))

    (kind, hostnames, replies), (fake_kind, fake_hostnames, fake_replies) = reads
    assert kind == fake_kind == type(strategy).__name__
    assert hostnames == driver_hostnames and set(hostnames) == _PUBLISHER_RANKS
    # Each reference carries its volume's hostname and reaches that volume.
    assert replies == {r: (hostnames[r], (r, hostnames[r])) for r in _PUBLISHER_RANKS}
    assert fake_hostnames == fake_names
    assert fake_replies == {
        r: (fake_names[r], reply) for r, (_, reply) in replies.items()
    }


@pytest.mark.asyncio
async def test_two_store_names_on_one_tcpstore_keep_separate_keys() -> None:
    """Keep each routing store's keys apart on one served TCPStore."""
    async with _served() as first:
        other = dataclasses.replace(first.address, store_name=_store_name())
        async with _connected(other) as second:
            await first.publish({"publisher/0"}, _strategy())
            with pytest.raises(RuntimeError, match="init has not run"):
                await second.read()
            await second.publish(_PUBLISHER_RANKS, _strategy())
            assert (await first.read())[0] == {"publisher/0"}
            assert (await second.read())[0] == _PUBLISHER_RANKS


@pytest.mark.asyncio
async def test_strategy_key_needs_monarch_to_decode() -> None:
    """Import without Monarch or cloudpickle. Decoding the strategy key needs Monarch.

    Its volume handles need Monarch: a strategy without volumes decodes without it.
    """
    strategy = _strategy()
    mesh = this_host().spawn_procs(per_host={"procs": 1})
    try:
        volumes = await StorageVolume.spawn(
            1, mesh, id_func=ts.LocalRankStrategy.get_volume_id
        )
        await strategy.set_storage_volumes(volumes)
        with_volumes = tcpstore._encode_strategy(strategy).hex()
    finally:
        await mesh.stop()
    root = os.path.dirname(os.path.dirname(ts.__file__))
    no_volumes = tcpstore._encode_strategy(_strategy()).hex()
    child = subprocess.run(
        [sys.executable, "-c", _DECODE_WITHOUT_MONARCH, root, no_volumes, with_volumes],
        capture_output=True,
        timeout=60,
    )
    assert child.returncode == 0, child.stderr.decode()[-2000:]
    decoded = child.stdout.split()[-2:]
    assert decoded == [b"TorchStoreStrategy", b"monarch"], child.stdout


@pytest.mark.asyncio
async def test_the_strategy_key_grows_under_100_bytes_per_volume() -> None:
    """Compress the strategy, whose pickle grows about 600 B per volume.

    Its volume handles repeat each volume's ids, and every rank reads the whole key.
    """
    sizes: dict[int, int] = {}
    for count in (2, 8):
        strategy = _strategy()
        mesh = this_host().spawn_procs(per_host={"procs": count})
        try:
            volumes = await StorageVolume.spawn(
                count, mesh, id_func=ts.LocalRankStrategy.get_volume_id
            )
            await strategy.set_storage_volumes(volumes)
            assert len(strategy.volume_id_to_coord) == count
            sizes[count] = len(tcpstore._encode_strategy(strategy))
        finally:
            await mesh.stop()
    assert (sizes[8] - sizes[2]) / 6 < 100, sizes


def test_serve_and_connect_need_neither_monarch_nor_cloudpickle() -> None:
    """Serve, connect, write and check without Monarch or cloudpickle."""
    root = os.path.dirname(os.path.dirname(ts.__file__))
    child = subprocess.run(
        [sys.executable, "-c", _SERVE_WITHOUT_CLOUDPICKLE, root],
        capture_output=True,
        timeout=60,
    )
    assert child.returncode == 0, child.stderr.decode()[-2000:]
    assert child.stdout.split() == [b"True"], child.stdout


@pytest.mark.asyncio
async def test_killed_server_fails_an_outstanding_call_promptly(server) -> None:
    """Turn a server killed during a call into a prompt error naming it."""
    process, address = server
    async with _connected(address) as rendezvous:
        _stop(process)
        publish = asyncio.create_task(rendezvous.publish(_PUBLISHER_RANKS, _strategy()))
        call = f"compare_set('publishers') on TCPStore {tcpstore._endpoint(address)}"
        await _until(lambda: _native_calls(call))
        assert not publish.done()

        process.kill()
        start = time.monotonic()
        with pytest.raises(
            ConnectionError, match=re.escape(tcpstore._endpoint(address))
        ):
            await publish
        assert time.monotonic() - start < 10


@pytest.mark.asyncio
async def test_stopped_server_times_out_calls_and_connects(server, monkeypatch) -> None:
    """Bound a write, a read and a connect that a stopped server never answers."""
    process, address = server
    async with _connected(address) as writer, _connected(address) as reader:
        await writer.publish(_PUBLISHER_RANKS, _strategy())
        _stop(process)
        assert tcpstore._DEFAULT_TIMEOUT_S == 60.0  # the documented default
        start = time.monotonic()
        with pytest.raises(TimeoutError, match=re.escape(tcpstore._endpoint(address))):
            await reader.read_strategy(timeout_s=0.5)
        assert 0.45 < time.monotonic() - start < 10
        monkeypatch.setattr(tcpstore, "_DEFAULT_TIMEOUT_S", 0.5)  # read when called
        for call in (
            lambda: writer.publish(_PUBLISHER_RANKS, _strategy(), timeout_s=0.5),
            lambda: reader.read(),
            lambda: reader.read_strategy(),
            lambda: TCPStoreRendezvous.connect(address, timeout_s=0.5),
        ):
            start = time.monotonic()
            with pytest.raises(
                TimeoutError, match=re.escape(tcpstore._endpoint(address))
            ):
                await call()
            assert 0.45 < time.monotonic() - start < 10


@pytest.mark.asyncio
async def test_calls_given_no_timeout_take_the_default(monkeypatch) -> None:
    """Bound serve, connect, publish and both reads by the default when given no timeout."""
    monkeypatch.setattr(tcpstore, "_DEFAULT_TIMEOUT_S", 7.0)  # read when called
    opened: list[float] = []
    left: list[float] = []
    open_, run = tcpstore._open, tcpstore._run_with_deadline

    async def record_open(host: str, port: int, timeout_s: float, *args, **kwargs):
        opened.append(timeout_s)
        return await open_(host, port, timeout_s, *args, **kwargs)

    async def record_run(
        fn: Callable[[], object], deadline: float, what: str
    ) -> object:
        left.append(deadline - time.monotonic())
        return await run(fn, deadline, what)

    monkeypatch.setattr(tcpstore, "_open", record_open)
    async with _served() as served, _connected(served.address) as other:
        monkeypatch.setattr(tcpstore, "_run_with_deadline", record_run)
        for call in (
            lambda: served.publish(_PUBLISHER_RANKS, _strategy()),
            other.read,
            other.read_strategy,
        ):
            left.clear()
            await call()
            assert left and all(5 < seconds <= 7 for seconds in left), left
    assert opened == [7.0, 7.0]


@pytest.mark.asyncio
async def test_calls_behind_a_stuck_call_wait_it_out_without_threads(server) -> None:
    """Start no native call behind one stuck on a stopped server: later ones time out.

    A connection runs its calls one at a time, and the server holds every connect to
    it, so each later call would leave one more thread blocked behind the first.
    """
    process, address = server
    endpoint = tcpstore._endpoint(address)
    async with _connected(address) as rendezvous:
        _stop(process)
        reads = [lambda: rendezvous.read(timeout_s=1)] * 2
        connects = [lambda: TCPStoreRendezvous.connect(address, timeout_s=1)] * 2
        for i, call in enumerate(reads + connects):
            start = time.monotonic()
            with pytest.raises(TimeoutError, match=re.escape(endpoint)) as caught:
                await call()
            assert time.monotonic() - start < 1.9  # waiting keeps the deadline
            started = i in (0, len(reads))
            expected = "did not finish" if started else "an earlier call has not"
            assert expected in str(caught.value)
        with pytest.raises(TimeoutError, match="an earlier call has not returned"):
            await asyncio.wait_for(rendezvous._check(["k"], math.nan), 10)
        stuck = set(_native_calls()) - _earlier_calls
        assert len(stuck) == 2, stuck  # one read and one connect
        assert tcpstore._live_calls - _earlier_calls == stuck
        # Neither another connection's calls nor connects to another server here wait.
        async with _served() as healthy:
            other = await TCPStoreRendezvous.connect(healthy.address, timeout_s=5)
            try:
                assert not await other._check(["k"], time.monotonic() + 5)
            finally:
                await other.close()


@pytest.mark.asyncio
async def test_calls_made_together_leave_one_thread_on_a_stopped_server(server) -> None:
    """Run one native call per lane, so calls made together take turns or time out.

    Reads on one connection, or connects to one server, would otherwise each leave one
    more thread blocked behind the first.
    """
    process, address = server
    async with _connected(address) as rendezvous:
        _stop(process)
        reads = [lambda: rendezvous.read(timeout_s=2)] * 50
        connects = [lambda: TCPStoreRendezvous.connect(address, timeout_s=2)] * 20
        for calls in (reads, connects):
            started: list[Awaitable[object]] = [call() for call in calls]
            results = await asyncio.gather(*started, return_exceptions=True)
            assert {type(result) for result in results} == {TimeoutError}, results
            # Only the call that reached the server may still land, and only it says so.
            texts = [str(result) for result in results]
            assert sum("did not finish" in text for text in texts) == 1, texts
            assert sum("was not started" in text for text in texts) == len(calls) - 1
        # The calls left waiting end at their deadline. One read and one connect stay.
        await _until(lambda: len(set(_native_calls()) - _earlier_calls) == 2)
        assert len(tcpstore._live_calls - _earlier_calls) == 2


@pytest.mark.parametrize("timeout_s", [math.inf, 60])
@pytest.mark.asyncio
async def test_calls_given_up_while_waiting_their_turn_end_their_threads(
    server, timeout_s: float
) -> None:
    """End a call's thread once its caller gives up while it waits for its lane."""
    process, address = server
    async with _connected(address) as rendezvous:
        _stop(process)
        calls = [rendezvous.read(timeout_s=timeout_s) for _ in range(20)]
        results = await asyncio.gather(
            *(asyncio.wait_for(call, 1) for call in calls), return_exceptions=True
        )
        assert {type(result) for result in results} == {asyncio.TimeoutError}
        # Only the read stuck on the stopped server keeps its thread.
        await _until(lambda: len(set(_native_calls()) - _earlier_calls) == 1, 5)
        assert len(tcpstore._live_calls - _earlier_calls) == 1


class _RecordingGate:
    """A lane's gate that hands each acquire's timeout to ``record``, then takes it."""

    def __init__(self, gate: Any, record: Callable[[float], object]) -> None:
        self.gate, self.record = gate, record

    def acquire(self, timeout: float = -1) -> bool:
        self.record(timeout)
        return self.gate.acquire(timeout=timeout)

    def release(self) -> None:
        self.gate.release()


def test_a_call_waiting_for_its_turn_polls_in_slices_instead_of_spinning() -> None:
    """Wait for a lane's turn in slices of at most 0.05s rather than spinning or waiting out the deadline at once.

    The wait here lasts 0.3s. A call thus checks at most 8 times however busy the host is.
    """
    acquires: list[float] = []
    lane = tcpstore._Lane()
    gate = lane._gate
    lane._gate = _RecordingGate(gate, acquires.append)
    with gate:  # another call's turn
        expired = lane._admit(
            tcpstore._NativeCall(lambda: None), time.monotonic() + 0.3, "waiting"
        )
    assert expired is tcpstore._EXPIRED and 1 < len(acquires) <= 8, acquires
    assert max(acquires) <= tcpstore._WAIT_OUT_POLL_S == 0.05, acquires


@pytest.mark.asyncio
async def test_a_call_behind_a_stuck_call_polls_until_that_call_returns(
    monkeypatch,
) -> None:
    """Wait out a stuck call in 0.05s sleeps and run once it returns, long before the deadline.

    The stuck call returns during the third sleep so neither spinning nor sleeping to the deadline passes.
    """
    release = threading.Event()
    sleeps: list[float] = []

    async def recording_sleep(delay: float) -> None:
        sleeps.append(delay)
        if len(sleeps) == 3:
            release.set()
        await asyncio.sleep(delay)

    lane = tcpstore._Lane()
    try:
        with pytest.raises(TimeoutError, match="stuck did not finish"):
            stuck = tcpstore._NativeCall(lambda: release.wait(30))
            await lane.run(stuck, time.monotonic() + 0.2, "stuck")
        polled = types.SimpleNamespace(**{**vars(asyncio), "sleep": recording_sleep})
        monkeypatch.setattr(tcpstore, "asyncio", polled)
        behind = lane.run(tcpstore._NativeCall(lambda: "ran"), _deadline(), "behind")
        assert await asyncio.wait_for(behind, 10) == "ran"
    finally:
        release.set()
    assert len(sleeps) >= 3 and set(sleeps) == {tcpstore._WAIT_OUT_POLL_S}, sleeps


@pytest.mark.asyncio
async def test_a_call_not_begun_by_its_deadline_was_not_started() -> None:
    """Report a call whose thread had not begun it by the deadline as not started, and never run it.

    The thread is held where a call that took its lane's turn registers.
    """
    ran: list[int] = []
    with tcpstore._live_calls_lock:
        with pytest.raises(TimeoutError, match="held back was not started"):
            await tcpstore._Lane().run(
                tcpstore._NativeCall(lambda: ran.append(1)),
                time.monotonic() + 0.2,
                "held back",
            )
    _join_native_calls()
    assert not ran


@pytest.mark.asyncio
async def test_a_call_reported_as_not_started_was_given_up_first() -> None:
    """Give a timed-out call up before reporting it as not started so that a turn taken just then never runs it.

    Here the caller's own thread tries the call just as it reads whether the call started. It stands in
    for the waiting thread taking its turn just then.
    """
    ran: list[int] = []

    class Raced(tcpstore._NativeCall):
        raced = False

        @property
        def started(self) -> bool:
            if not self.raced:
                self.raced = True
                with contextlib.suppress(RuntimeError):
                    self()  # as the waiting thread's turn would, on the caller's thread
            return self._started

        @started.setter
        def started(self, value: bool) -> None:
            self._started = value

    lane = tcpstore._Lane()
    with lane._gate:  # another call's turn, past this call's deadline
        with pytest.raises(TimeoutError, match="raced was not started"):
            await lane.run(
                Raced(lambda: ran.append(1)), time.monotonic() + 0.2, "raced"
            )
    _join_native_calls()
    assert not ran


@pytest.mark.asyncio
async def test_a_call_returning_after_its_deadline_did_not_finish(monkeypatch) -> None:
    """Report a call that ran past its deadline as not finished, though it returned before its caller looked.

    Its late result then settles nothing, so no error reaches the loop's exception handler.
    """
    loop = asyncio.get_running_loop()
    handled: list[dict[str, Any]] = []
    loop.set_exception_handler(lambda _, context: handled.append(context))
    release, returned = threading.Event(), threading.Event()

    class Recorded(tcpstore._NativeCall):
        def __call__(self) -> Any:
            try:
                return super().__call__()
            finally:
                returned.set()

    settle = tcpstore._settle

    def return_at_the_deadline(
        future: asyncio.Future[object], result: object, error: Exception | None
    ) -> None:
        settle(future, result, error)
        if result is tcpstore._EXPIRED:
            release.set()
            assert returned.wait(30)  # before its caller sees the deadline pass

    monkeypatch.setattr(tcpstore, "_settle", return_at_the_deadline)
    try:
        with pytest.raises(TimeoutError, match="late did not finish"):
            await tcpstore._Lane().run(
                Recorded(lambda: release.wait(30)), time.monotonic() + 0.2, "late"
            )
        _join_native_calls()
        await asyncio.sleep(0)  # runs the late result's settle
    finally:
        loop.set_exception_handler(None)
    assert not handled, handled


def test_a_call_waiting_its_turn_stops_at_its_deadline_on_its_own() -> None:
    """End a call's wait for its lane at its deadline though its caller never gives up, as a blocked loop would not."""
    lane = tcpstore._Lane()
    call = tcpstore._NativeCall(lambda: "ran")
    admitted: list[object] = []
    with lane._gate:  # another call's turn
        waiting = threading.Thread(
            target=lambda: admitted.append(
                lane._admit(call, time.monotonic() + 0.1, "waiting")
            ),
            daemon=True,
        )
        waiting.start()
        waiting.join(10)
    assert admitted == [tcpstore._EXPIRED] and not call.started


@pytest.mark.asyncio
async def test_a_late_give_up_of_an_earlier_call_keeps_the_lane_busy() -> None:
    """Keep a lane busy while its stuck call runs, though a give-up of an earlier call lands after it.

    Callers on other threads and loops share connect lanes, so their give-ups can interleave.
    """
    lane = tcpstore._Lane()
    release_earlier, release_stuck, paused, resume = (
        threading.Event() for _ in range(4)
    )

    class PausedGiveUp(tcpstore._NativeCall):
        def give_up(self) -> bool:
            running = super().give_up()
            if running and not paused.is_set():
                paused.set()  # as if its thread were switched out here
                resume.wait(30)
            return running

    earlier = PausedGiveUp(lambda: release_earlier.wait(30))

    def give_up_on_earlier() -> None:
        with contextlib.suppress(TimeoutError):
            asyncio.run(lane.run(earlier, time.monotonic() + 0.2, "earlier"))

    other = threading.Thread(target=give_up_on_earlier)
    other.start()
    try:
        await _until(paused.is_set)
        release_earlier.set()
        await _until(lambda: not earlier.running)
        stuck = tcpstore._NativeCall(lambda: release_stuck.wait(30))
        with pytest.raises(TimeoutError, match="stuck did not finish"):
            await lane.run(stuck, time.monotonic() + 0.2, "stuck")
        resume.set()
        other.join(30)
        assert not lane.idle
    finally:
        for event in (release_earlier, resume, release_stuck):
            event.set()
        other.join(30)


@pytest.mark.asyncio
async def test_a_lane_runs_calls_again_once_its_stuck_call_returns() -> None:
    """Run calls on a lane again once its stuck call returns, and drop the shared lane then."""
    endpoint = f"stuck-{uuid.uuid4()}:1"
    release = threading.Event()
    try:
        with tcpstore._shared_lane(endpoint) as lane:
            with pytest.raises(TimeoutError, match="stuck did not finish"):
                stuck = tcpstore._NativeCall(lambda: release.wait(30))
                await lane.run(stuck, time.monotonic() + 0.2, "stuck")
        assert tcpstore._lane_busy(endpoint)  # kept while its stuck call runs
        with tcpstore._shared_lane(endpoint) as lane:
            with pytest.raises(TimeoutError, match="an earlier call has not returned"):
                behind = tcpstore._NativeCall(lambda: None)
                await lane.run(behind, time.monotonic() + 0.2, "behind")
    finally:
        release.set()
    _join_native_calls()
    with tcpstore._shared_lane(endpoint) as lane:
        again = tcpstore._NativeCall(lambda: "again")
        assert await lane.run(again, time.monotonic() + 5, "again") == "again"
    assert endpoint not in tcpstore._connect_lanes


def _lane_with(*, running: bool, given_up: bool, users: int = 0) -> tcpstore._Lane:
    """A lane whose last call runs or has returned and was or was not given up on."""
    last = tcpstore._NativeCall(lambda: None)
    last.running, last.started = running, True
    if given_up:
        last.give_up()
    lane = tcpstore._Lane()
    lane._last, lane.users = last, users
    return lane


def test_a_connect_lane_is_busy_only_while_a_call_given_up_on_runs(
    monkeypatch,
) -> None:
    """Skip an address only for a stuck call, not for a lane another connect uses or that none has.

    The lane in use runs a call its caller still waits for.
    """
    busy = _lane_with(running=True, given_up=True)
    in_use = _lane_with(running=True, given_up=False, users=1)
    lanes = {"busy:1": busy, "in-use:1": in_use}
    monkeypatch.setattr(tcpstore, "_connect_lanes", lanes)
    busy_now = [tcpstore._lane_busy(e) for e in ("busy:1", "in-use:1", "absent:1")]
    assert busy_now == [True, False, False]


def test_a_connect_leaving_its_lane_drops_every_idle_lane_none_uses(
    monkeypatch,
) -> None:
    """Drop every idle lane that no connect uses when a connect leaves its lane, not only that one.

    A lane whose stuck call returned after its connect left stays until then.
    """
    kept = {
        "busy:1": _lane_with(running=True, given_up=True),
        "in-use:1": _lane_with(running=False, given_up=True, users=1),
    }
    lanes = {**kept, "returned:1": _lane_with(running=False, given_up=True)}
    monkeypatch.setattr(tcpstore, "_connect_lanes", lanes)
    with tcpstore._shared_lane("leaving:1") as lane:
        assert lanes["leaving:1"] is lane and lane.users == 1
    assert lanes == kept


@pytest.mark.asyncio
async def test_serves_for_one_host_never_take_turns(monkeypatch) -> None:
    """Open each serve's store on a lane of its own, so serves for one host never wait on each other."""
    both = threading.Barrier(2, timeout=5)
    open_store = tcpstore._open_store

    def meet(*args: Any, **kwargs: Any) -> TCPStore:
        both.wait()  # each opens only once the other is opening too
        return open_store(*args, **kwargs)

    monkeypatch.setattr(tcpstore, "_open_store", meet)
    served = await asyncio.gather(
        *(
            TCPStoreRendezvous.serve(_store_name(), host=_LOCALHOST, timeout_s=10)
            for _ in range(2)
        )
    )
    for rendezvous in served:
        await rendezvous.close()


@pytest.mark.asyncio
async def test_a_nan_deadline_starts_no_call_on_an_idle_lane() -> None:
    """Refuse a NaN deadline before starting a thread, as a passed one is refused."""
    ran: list[int] = []
    with pytest.raises(TimeoutError, match="nan was not started"):
        call = tcpstore._NativeCall(lambda: ran.append(1))
        await tcpstore._Lane().run(call, math.nan, "nan")
    assert not ran and not _native_calls("nan")


@pytest.mark.parametrize(
    ("message", "detail"), [("first\nsecond", "first"), ("", "DistStoreError")]
)
@pytest.mark.asyncio
async def test_a_native_failure_names_its_first_line_or_its_type(
    message: str, detail: str
) -> None:
    """Report a native failure by its message's first line, or by its type when it has no message."""

    def fail() -> NoReturn:
        raise DistStoreError(message)

    with pytest.raises(ConnectionError) as caught:
        await tcpstore._run_with_deadline(fail, _deadline(), "call")
    assert str(caught.value) == f"call failed: {detail}"


@pytest.mark.asyncio
async def test_a_write_given_up_on_before_it_starts_never_lands() -> None:
    """Never run a native call whose caller gave up before its thread started it."""
    async with _served() as rendezvous:
        with tcpstore._live_calls_lock:  # holds the native thread before the call
            write = asyncio.create_task(rendezvous._compare_set("k", b"v", _deadline()))
            await _until(lambda: _native_calls("compare_set('k')"))
            write.cancel()
            with pytest.raises(asyncio.CancelledError):
                await write
        _join_native_calls()
        assert not await rendezvous._check(["k"], _deadline())


def test_exit_waits_for_an_abandoned_call_that_returns_within_its_budget() -> None:
    """Drain an abandoned call at exit, before it can return into finalization."""
    with socket.socket() as probe:
        probe.bind((_LOCALHOST, 0))  # bound but not listening, so connects are refused
        port = probe.getsockname()[1]
        _connect_then_exit(port, timeout_s=0.3, teardown_s=3, prefix=_DIAL_NATIVELY)


# Registers an exit hook that calls a connection before the module loads and one after,
# then serves and connects. Each hook prints what its call returned or raised.
_EXIT_HOOKS = (
    (
        "import asyncio, atexit, time\n"
        "def call_at_exit(name):\n"
        "    try:\n"
        "        print(name, asyncio.run(handle._check(['k'], time.monotonic() + 30)), flush=True)\n"
        "    except RuntimeError as error:\n"
        "        print(name, error, flush=True)\n"
        "atexit.register(call_at_exit, 'before:')\n"
    )
    + _IMPORT_ALONE
    + (
        "from torchstore.routing.tcpstore import TCPStoreRendezvous\n"
        "atexit.register(call_at_exit, 'after:')\n"
        "async def main():\n"
        "    global handle, served\n"
        "    served = await TCPStoreRendezvous.serve('hooks', host='127.0.0.1')\n"
        "    handle = await TCPStoreRendezvous.connect(served.address)\n"
        "asyncio.run(main())\n"
    )
)


def test_an_exit_hook_registered_after_the_module_loads_can_still_call() -> None:
    """Run an exit hook registered after the module loads before the exit drain, which refuses later calls."""
    root = os.path.dirname(os.path.dirname(ts.__file__))
    child = subprocess.run(
        [sys.executable, "-c", _EXIT_HOOKS, root],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert child.returncode == 0, child.stderr[-2000:]
    after, before = child.stdout.splitlines()[-2:]
    assert after == "after: False", child.stdout
    refused = r"before: check\(\['k'\]\) on TCPStore 127\.0\.0\.1:\d+ was not started: exiting"
    assert re.fullmatch(refused, before), child.stdout


@pytest.mark.asyncio
async def test_exit_drain_shares_one_budget_then_refuses_calls(
    monkeypatch, caplog
) -> None:
    """Refuse native calls at exit, then give every live call one shared budget.

    A WARNING names the calls still running when the budget ends, and none that returned.
    """
    caplog.set_level(logging.WARNING, logger=tcpstore.__name__)
    release = threading.Event()
    joins: list[tuple[int, bool]] = []

    class Live(threading.Thread):
        def join(self, timeout: float | None = None) -> None:
            joins.append((id(self), tcpstore._exiting))
            super().join(timeout)

    live = [
        Live(target=release.wait, args=(60,), name=f"check on TCPStore host:{port}")
        for port in range(4)
    ]
    returned = threading.Thread(target=lambda: None, name="check on TCPStore returned")
    for thread in (*live, returned):
        thread.start()
    returned.join()
    monkeypatch.setattr(tcpstore, "_live_calls", {*live, returned})
    monkeypatch.setattr(tcpstore, "_exiting", False)
    monkeypatch.setattr(tcpstore, "_EXIT_DRAIN_S", 0.5)
    start = time.monotonic()
    try:
        tcpstore._drain_native_calls()
        elapsed = time.monotonic() - start
    finally:
        release.set()
    assert elapsed < 1.0, elapsed  # 0.5 s in all, not 0.5 s for each of the 4 calls
    # Each live call is joined once, with calls that start meanwhile already refused.
    assert sorted(joins) == sorted((id(thread), True) for thread in live)
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and all(t.name in warnings[0] for t in live), warnings
    assert returned.name not in warnings[0]
    aborts = "a c10d call that returns during finalization aborts the process before Python 3.14"
    assert aborts in warnings[0]
    with pytest.raises(RuntimeError, match="was not started: exiting"):
        await tcpstore._Lane().run(
            tcpstore._NativeCall(lambda: None), _deadline(), "call"
        )


def test_an_exit_drain_with_no_call_left_running_logs_nothing(
    monkeypatch, caplog
) -> None:
    """Log nothing at exit, not even at INFO, once every native call has returned."""
    caplog.set_level(logging.INFO, logger=tcpstore.__name__)
    returned = threading.Thread(target=lambda: None, name="check on TCPStore returned")
    returned.start()
    returned.join()
    monkeypatch.setattr(tcpstore, "_live_calls", {returned})
    monkeypatch.setattr(tcpstore, "_exiting", False)
    tcpstore._drain_native_calls()
    assert tcpstore._exiting and not caplog.records, caplog.records


@pytest.mark.asyncio
async def test_a_call_waiting_for_its_turn_at_exit_never_starts(
    monkeypatch, caplog
) -> None:
    """Refuse a call still waiting for its lane's turn when the exit hook runs.

    The exit WARNING names the running call alone, since the waiting one can only be refused.
    """
    caplog.set_level(logging.WARNING, logger=tcpstore.__name__)
    entered, release = threading.Event(), threading.Event()
    ran: list[str] = []
    at_gate: list[str] = []

    def held(store) -> bool:
        entered.set()
        release.wait(timeout=60)
        return store.check(["k"])

    def queued(store) -> bool:
        ran.append("queued")
        return store.check(["k"])

    def record_thread(timeout: float) -> None:
        at_gate.append(threading.current_thread().name)

    monkeypatch.setattr(tcpstore, "_exiting", False)
    monkeypatch.setattr(tcpstore, "_EXIT_DRAIN_S", 0.1)
    async with _served() as rendezvous:
        what = f"queued on TCPStore {tcpstore._endpoint(rendezvous.address)}"
        gate = _RecordingGate(rendezvous._lane._gate, record_thread)
        monkeypatch.setattr(rendezvous._lane, "_gate", gate)
        first = asyncio.create_task(rendezvous._call("held", held, _deadline()))
        await _until(entered.is_set)
        second = asyncio.create_task(rendezvous._call("queued", queued, _deadline()))
        await _until(lambda: any(name.endswith(what) for name in at_gate))
        try:
            tcpstore._drain_native_calls()
        finally:
            release.set()
        assert await first is False
        with pytest.raises(
            RuntimeError, match=re.escape(f"{what} was not started: exiting")
        ):
            await second
    assert not ran
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "held on TCPStore" in warnings[0], warnings
    assert "queued" not in warnings[0]


@pytest.mark.asyncio
async def test_native_call_errors_reach_the_caller_at_once() -> None:
    """Raise a native call's own error at once, not a TimeoutError at the deadline."""

    def fail() -> NoReturn:
        raise LookupError("not a store error")

    start = time.monotonic()
    with pytest.raises(LookupError, match="not a store error"):
        await tcpstore._run_with_deadline(fail, time.monotonic() + 20, "call")
    assert time.monotonic() - start < 10


@pytest.mark.asyncio
async def test_native_calls_run_on_daemon_threads() -> None:
    """Run each native call on a daemon thread, which threading's own shutdown never joins.

    Only the exit drain waits for one, and only for a call a lane admitted, up to its budget.
    """
    daemon = await tcpstore._run_with_deadline(
        lambda: threading.current_thread().daemon, _deadline(), "daemon"
    )
    assert daemon is True


@pytest.mark.asyncio
async def test_call_whose_deadline_has_passed_is_not_started() -> None:
    """Refuse to start a native call once its deadline has passed."""
    async with _served() as rendezvous:
        endpoint = tcpstore._endpoint(rendezvous.address)
        late = f"compare_set('late') on TCPStore {endpoint} was not started"
        with pytest.raises(TimeoutError, match=re.escape(late)):
            await rendezvous._compare_set("late", b"v", time.monotonic())
        _join_native_calls()  # a call that started anyway has landed
        assert not await rendezvous._check(["late"], _deadline())


@pytest.mark.parametrize("host", ["no-such-host.invalid", "bad..name"])
@pytest.mark.asyncio
async def test_serve_fails_when_its_own_host_does_not_resolve(host: str) -> None:
    """Dial the advertised host from the server too, so a bad name fails at serve.

    A name that IDNA rejects fails the same way, not with a ``UnicodeError``.
    """
    with pytest.raises(_UNREACHABLE, match=re.escape(host)):
        await TCPStoreRendezvous.serve(_store_name(), host=host, timeout_s=2)


@pytest.mark.asyncio
async def test_a_lookup_that_keeps_failing_is_retried_after_a_pause(
    monkeypatch,
) -> None:
    """Pause between retries of a failing lookup, so an outage sees no lookup storm."""
    lookup = socket.getaddrinfo
    looked_up: list[float] = []

    def dns_down(host: str, *args: Any, **kwargs: Any) -> Any:
        if host == "dns-down.invalid":
            looked_up.append(time.monotonic())
            error = "Temporary failure in name resolution"
            raise socket.gaierror(socket.EAI_AGAIN, error)
        return lookup(host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", dns_down)
    address = TCPStoreAddress("dns-down.invalid", 1, _store_name(), "nonce")
    with pytest.raises(ConnectionError, match="cannot resolve") as caught:
        await TCPStoreRendezvous.connect(address, timeout_s=5)
    assert isinstance(caught.value.__cause__, tcpstore._QuietNetworkError)  # no backoff
    gaps = [later - earlier for earlier, later in zip(looked_up, looked_up[1:])]
    assert gaps and min(gaps) >= 0.2 and len(looked_up) <= 20, looked_up


@pytest.mark.asyncio
async def test_connects_dial_the_address_in_short_attempts_that_retry(
    monkeypatch,
) -> None:
    """Dial the host's addresses in turn, in attempts short enough for the exit drain.

    An attempt gets a third of the deadline at most. Network failures retry on the next
    address while one more attempt could still fail natively in time. Others never do.
    """
    dialed: list[tuple[str, int, timedelta]] = []
    answered: list[tuple[str, int, timedelta]] = []
    listening: list[int] = []
    open_store = tcpstore.TCPStore

    def refuse_first(
        host: str, port: int, *, timeout: timedelta, is_master: bool, **kwargs: Any
    ) -> TCPStore:
        dialed.append((host, port, timeout))
        if host == "unusable.invalid":
            raise DistStoreError("unusable")
        if is_master:
            # Closes the listener it was handed, as the server it fails to start would.
            with socket.socket(fileno=kwargs["master_listen_fd"]) as listener:
                listening.append(listener.getsockname()[1])
        if is_master or len(dialed) == 1:
            raise DistNetworkError("refused")
        return open_store(_LOCALHOST, port, timeout=timeout, is_master=False, **kwargs)

    def record(host: str, port: int, attempt_s: float) -> None:
        answered.append((host, port, timedelta(seconds=attempt_s)))

    async with _served() as served:
        monkeypatch.setattr(tcpstore, "TCPStore", refuse_first)
        monkeypatch.setattr(
            tcpstore, "_resolve", lambda host, *_: [host, f"next.{host}"]
        )
        monkeypatch.setattr(tcpstore, "_check_answers", record)
        address = dataclasses.replace(served.address, host="store-host.invalid")
        async with _connected(address) as connected:
            assert await connected._compare_set("k", b"v", _deadline()) == b"v"
        unusable = dataclasses.replace(address, host="unusable.invalid")
        with pytest.raises(ConnectionError, match="unusable"):
            await TCPStoreRendezvous.connect(unusable, timeout_s=9)
        serve_error = "starting a TCPStore for serve-host.invalid failed: refused"
        with pytest.raises(ConnectionError, match=re.escape(serve_error)):
            await TCPStoreRendezvous.serve(
                _store_name(), host="serve-host.invalid", timeout_s=1.2
            )
    attempt = timedelta(seconds=0.5)
    assert dialed == [
        ("store-host.invalid", address.port, attempt),
        ("next.store-host.invalid", address.port, attempt),
        ("unusable.invalid", address.port, attempt),
        ("serve-host.invalid", listening[0], timedelta(seconds=0.4)),
    ]
    # Each attempt first dials in Python, a serve's at the listener its server takes.
    assert answered == dialed


@pytest.mark.asyncio
async def test_a_connect_retries_a_python_dial_that_failed(monkeypatch) -> None:
    """Retry a connect whose Python dial fails once, as a lost SYN makes it fail.

    The retry is a second attempt at the address already looked up, without a second lookup.
    """
    looked_up: list[str] = []
    attempted: list[str] = []
    lost: list[object] = []
    resolve, create = tcpstore._resolve, socket.create_connection
    open_store = tcpstore._open_store

    def record(host: str, *args: Any) -> list[str]:
        looked_up.append(host)
        return resolve(host, *args)

    def record_attempt(host: str, *args: Any, **kwargs: Any) -> TCPStore:
        attempted.append(host)
        return open_store(host, *args, **kwargs)

    def lose_first_syn(address: Any, *args: Any, **kwargs: Any) -> socket.socket:
        if not lost:
            lost.append(address)
            raise TimeoutError("timed out")
        return create(address, *args, **kwargs)

    async with _served() as served:
        monkeypatch.setattr(tcpstore, "_resolve", record)
        monkeypatch.setattr(tcpstore, "_open_store", record_attempt)
        monkeypatch.setattr(socket, "create_connection", lose_first_syn)
        # A 3 s connect still has room for a retry after a lost dial.
        connected = await TCPStoreRendezvous.connect(served.address, timeout_s=3)
        try:
            assert await connected._compare_set("k", b"v", _deadline()) == b"v"
        finally:
            await connected.close()
    assert lost and looked_up == [_LOCALHOST], (lost, looked_up)
    assert attempted == [_LOCALHOST, _LOCALHOST]


@pytest.mark.asyncio
async def test_serves_and_connects_dial_first_an_address_that_answers(
    monkeypatch,
) -> None:
    """Dial first the address that accepts a connection, as TCPStore itself falls back.

    TCP never connects to a broadcast address, and a server listening on 127.0.0.1 alone
    refuses 127.0.0.2. A serve, its port not known yet, probes a listener of its own.
    """
    first = {"serve.invalid": "255.255.255.255", "connect.invalid": "127.0.0.2"}
    dialed: list[tuple[str, bool]] = []
    answered: list[tuple[str, int]] = []
    open_store, check = tcpstore.TCPStore, tcpstore._check_answers

    def record(host: str, port: int, *, is_master: bool, **kwargs: Any) -> TCPStore:
        dialed.append((host, is_master))
        return open_store(host, port, is_master=is_master, **kwargs)

    def record_check(host: str, port: int, attempt_s: float) -> None:
        answered.append((host, port))
        check(host, port, attempt_s)

    ipv4_only = _ipv4_only_store()
    port = ipv4_only.port
    monkeypatch.setattr(socket, "getaddrinfo", _two_addresses(first))
    monkeypatch.setattr(tcpstore, "TCPStore", record)
    monkeypatch.setattr(tcpstore, "_check_answers", record_check)
    async with _served(host="serve.invalid") as served:
        assert served.address.host == "serve.invalid"  # clients still dial the name
        served_port = served.address.port
    address = TCPStoreAddress("connect.invalid", port, _store_name(), uuid.uuid4().hex)
    connected = await TCPStoreRendezvous.connect(address, timeout_s=2)
    await connected.close()
    del ipv4_only
    assert dialed == [(_LOCALHOST, True), (_LOCALHOST, False)]
    # Each first attempt dials in Python too, a serve's where its server then listens.
    assert answered == [(_LOCALHOST, served_port), (_LOCALHOST, port)]


@pytest.mark.asyncio
async def test_a_first_address_that_drops_connects_costs_one_short_probe(
    monkeypatch,
) -> None:
    """Dial the next address once a probe of one that drops connects times out.

    A listener whose accept queue is full drops new connects, as a firewall would.
    """
    ipv4_only = _ipv4_only_store()
    port = ipv4_only.port
    with socket.create_server(("127.0.0.2", port), backlog=0) as full:
        with socket.create_connection(full.getsockname(), timeout=10):  # fills it
            name = "dropping.invalid"
            monkeypatch.setattr(
                socket, "getaddrinfo", _two_addresses({name: "127.0.0.2"})
            )
            address = TCPStoreAddress(name, port, _store_name(), uuid.uuid4().hex)
            start = time.monotonic()
            connected = await TCPStoreRendezvous.connect(address, timeout_s=5)
            await connected.close()
            assert time.monotonic() - start < 3  # a 0.5 s probe, not a SYN timeout
    del ipv4_only


@pytest.mark.asyncio
async def test_a_connect_to_an_address_that_drops_connects_names_the_cause() -> None:
    """Bound each Python dial by its attempt, so dropped connects fail with that cause.

    The dial leaves no call stuck on the server's lane, so a second connect fails alike.
    """
    with socket.create_server(("127.0.0.2", 0), backlog=0) as full:
        with socket.create_connection(full.getsockname(), timeout=10):  # fills it
            port = full.getsockname()[1]
            address = TCPStoreAddress("127.0.0.2", port, _store_name(), "nonce")
            for _ in range(2):
                with pytest.raises(ConnectionError, match="timed out") as caught:
                    await TCPStoreRendezvous.connect(address, timeout_s=3)
                assert isinstance(caught.value.__cause__, tcpstore._QuietNetworkError)
    _join_native_calls()


@pytest.mark.asyncio
async def test_retries_reach_an_address_after_the_one_that_answered(
    monkeypatch,
) -> None:
    """Retry the next looked-up address once the one that answered hangs up on TCPStore.

    A stale address can reach another service listening on the store's port.
    """
    ipv4_only = _ipv4_only_store()
    port = ipv4_only.port
    dialed: list[str] = []
    open_store = tcpstore.TCPStore

    def record(host: str, port: int, **kwargs: Any) -> TCPStore:
        dialed.append(host)
        return open_store(host, port, **kwargs)

    stop = threading.Event()
    with socket.create_server(("127.0.0.2", port)) as other:
        other.settimeout(0.1)

        def hang_up() -> None:
            while not stop.is_set():
                with contextlib.suppress(OSError):
                    other.accept()[0].close()

        hanging_up = threading.Thread(target=hang_up, daemon=True)
        hanging_up.start()
        try:
            name = "stale.invalid"
            monkeypatch.setattr(
                socket, "getaddrinfo", _two_addresses({name: "127.0.0.2"})
            )
            monkeypatch.setattr(tcpstore, "TCPStore", record)
            address = TCPStoreAddress(name, port, _store_name(), uuid.uuid4().hex)
            connected = await TCPStoreRendezvous.connect(address, timeout_s=10)
            await connected.close()
        finally:
            stop.set()
            hanging_up.join(timeout=10)
    del ipv4_only
    assert dialed[0] == "127.0.0.2" and dialed[-1] == _LOCALHOST, dialed


@pytest.mark.asyncio
async def test_retries_reach_an_address_before_the_one_that_answered(
    monkeypatch,
) -> None:
    """Retry an address whose probe was lost once the one that answered hangs up.

    A lost SYN makes the store's own address miss the probe.
    """
    ipv4_only = _ipv4_only_store()
    port = ipv4_only.port
    dialed: list[str] = []
    lost: list[object] = []
    open_store, create = tcpstore.TCPStore, socket.create_connection
    store_first = _two_addresses({"lossy.invalid": "127.0.0.2"}, localhost_first=True)

    def record(host: str, port: int, **kwargs: Any) -> TCPStore:
        dialed.append(host)
        return open_store(host, port, **kwargs)

    def lose_first_syn(address: Any, *args: Any, **kwargs: Any) -> socket.socket:
        if address == (_LOCALHOST, port) and not lost:
            lost.append(address)
            raise TimeoutError("timed out")
        return create(address, *args, **kwargs)

    stop = threading.Event()
    with socket.create_server(("127.0.0.2", port)) as other:
        other.settimeout(0.1)

        def hang_up() -> None:
            while not stop.is_set():
                with contextlib.suppress(OSError):
                    other.accept()[0].close()

        hanging_up = threading.Thread(target=hang_up, daemon=True)
        hanging_up.start()
        try:
            monkeypatch.setattr(socket, "getaddrinfo", store_first)
            monkeypatch.setattr(socket, "create_connection", lose_first_syn)
            monkeypatch.setattr(tcpstore, "TCPStore", record)
            address = TCPStoreAddress("lossy.invalid", port, _store_name(), "nonce")
            connected = await TCPStoreRendezvous.connect(address, timeout_s=10)
            await connected.close()
        finally:
            stop.set()
            hanging_up.join(timeout=10)
    del ipv4_only
    assert lost and dialed[0] == "127.0.0.2" and dialed[-1] == _LOCALHOST, dialed


@pytest.mark.parametrize("silent_first", [True, False])
@pytest.mark.asyncio
async def test_retries_reach_an_address_after_one_that_never_replies(
    monkeypatch, silent_first: bool
) -> None:
    """Move on from a looked-up address that accepts and never replies, as a stale one's new owner may.

    Silent first, a retry after the live address's failed dial and the next connect both skip it
    while its abandoned attempt runs. Live first, its attempt after that failed dial is bounded too.
    """
    lost: list[str] = []
    check = tcpstore._check_answers

    def lose_first_live_dial(host: str, *args: Any) -> None:
        if host == _LOCALHOST and not lost:
            lost.append(host)
            raise tcpstore._QuietNetworkError(f"cannot connect to {host}: timed out")
        check(host, *args)

    monkeypatch.setattr(tcpstore, "_check_answers", lose_first_live_dial)
    ipv4_only = _ipv4_only_store()
    port = ipv4_only.port
    held: list[socket.socket] = []
    stop = threading.Event()
    with socket.create_server(("127.0.0.2", port)) as other:
        other.settimeout(0.1)

        def hold() -> None:
            while not stop.is_set():
                with contextlib.suppress(OSError):
                    held.append(other.accept()[0])

        holding = threading.Thread(target=hold, daemon=True)
        holding.start()
        try:
            name = "silent.invalid"
            resolve = _two_addresses(
                {name: "127.0.0.2"}, localhost_first=not silent_first
            )
            monkeypatch.setattr(socket, "getaddrinfo", resolve)
            address = TCPStoreAddress(name, port, _store_name(), uuid.uuid4().hex)
            for _ in range(2):  # the second while the first's abandoned attempt runs
                start = time.monotonic()
                connected = await TCPStoreRendezvous.connect(address, timeout_s=10)
                await connected.close()
                assert time.monotonic() - start < 6
            assert lost == [_LOCALHOST]
        finally:
            stop.set()
            holding.join(timeout=10)
            for conn in held:
                conn.close()  # ends the attempt left on the silent address
    del ipv4_only


@pytest.mark.asyncio
async def test_native_attempts_that_fail_back_off(monkeypatch) -> None:
    """Double the pause after each failed native attempt, as at a port that hangs up.

    Python's dial passes there, but c10d logs each native failure with a backtrace.
    """
    attempts: list[tuple[float, float]] = []
    open_store = tcpstore.TCPStore

    def record(*args: Any, **kwargs: Any) -> TCPStore:
        start = time.monotonic()
        try:
            return open_store(*args, **kwargs)
        finally:
            attempts.append((start, time.monotonic()))

    stop = threading.Event()
    with socket.create_server((_LOCALHOST, 0)) as other:
        other.settimeout(0.1)

        def hang_up() -> None:
            while not stop.is_set():
                with contextlib.suppress(OSError):
                    other.accept()[0].close()

        hanging_up = threading.Thread(target=hang_up, daemon=True)
        hanging_up.start()
        try:
            monkeypatch.setattr(tcpstore, "TCPStore", record)
            port = other.getsockname()[1]
            address = TCPStoreAddress(_LOCALHOST, port, _store_name(), "nonce")
            failed = re.escape(f"TCPStore {_LOCALHOST}:{port} failed")
            with pytest.raises(ConnectionError, match=failed) as caught:
                await TCPStoreRendezvous.connect(address, timeout_s=10)
        finally:
            stop.set()
            hanging_up.join(timeout=10)
    cause = caught.value.__cause__
    assert isinstance(cause, DistNetworkError), cause  # the native attempt's own error
    assert not isinstance(cause, tcpstore._QuietNetworkError), cause
    gaps = [later[0] - earlier[1] for earlier, later in zip(attempts, attempts[1:])]
    assert 2 <= len(attempts) <= 5, attempts
    assert all(gap >= 0.45 * 2**i for i, gap in enumerate(gaps)), gaps


@pytest.mark.asyncio
async def test_retry_pauses_double_per_failed_native_attempt_up_to_8_s(
    monkeypatch,
) -> None:
    """Pause 0.25 s after quiet failures, doubling per failed native attempt up to 8 s.

    A failure that is not a network failure ends the retries.
    """
    quiet, native = tcpstore._QuietNetworkError, DistNetworkError
    failures = [quiet, native, quiet, native, native, native, native, native]
    pauses: list[float] = []
    sleep = asyncio.sleep

    def fail(*args: Any, **kwargs: Any) -> NoReturn:
        raise (failures.pop(0) if failures else DistStoreError)("failed")

    async def record(delay: float) -> None:
        pauses.append(delay)
        await sleep(0)

    monkeypatch.setattr(tcpstore, "_open_store", fail)
    monkeypatch.setattr(asyncio, "sleep", record)
    address = TCPStoreAddress(_LOCALHOST, 1, _store_name(), "nonce")
    with pytest.raises(ConnectionError) as caught:
        await TCPStoreRendezvous.connect(address, timeout_s=60)
    assert isinstance(caught.value.__cause__, DistStoreError)
    assert pauses == [0.25, 0.5, 0.5, 1, 2, 4, 8, 8], pauses


@pytest.mark.asyncio
async def test_retries_outlast_a_thousand_failed_native_attempts(monkeypatch) -> None:
    """Keep retrying under an unbounded deadline, with pauses that stay at 8 s."""
    failures = [DistNetworkError] * 1100
    pauses: list[float] = []
    sleep = asyncio.sleep

    def fail(*args: Any, **kwargs: Any) -> NoReturn:
        raise (failures.pop() if failures else DistStoreError)("failed")

    async def record(delay: float) -> None:
        pauses.append(delay)
        await sleep(0)

    monkeypatch.setattr(tcpstore, "_open_store", fail)
    monkeypatch.setattr(asyncio, "sleep", record)
    address = TCPStoreAddress(_LOCALHOST, 1, _store_name(), "nonce")
    with pytest.raises(ConnectionError) as caught:
        await TCPStoreRendezvous.connect(address, timeout_s=math.inf)
    assert isinstance(caught.value.__cause__, DistStoreError)
    assert len(pauses) == 1100 and max(pauses) == 8, pauses[-3:]


@pytest.mark.parametrize(
    ("timeout_s", "expected_left"), [(2.01, [2.01, 1.51]), (1.99, [1.99])]
)
@pytest.mark.asyncio
async def test_a_retry_starts_only_while_3_attempt_timeouts_and_its_pause_fit(
    monkeypatch, timeout_s: float, expected_left: list[float]
) -> None:
    """Retry a failed native attempt only while 3 attempt timeouts beyond its pause are left.

    A native failure can take about that long. The caller thus gets it rather than a timeout.
    The clock here moves by the pauses. With 0.5s attempts and a first pause of 0.5s a connect
    thus retries with 2.01s left and fails with 1.99s left.
    """
    # Far ahead of the event loop's clock so no real deadline timer fires during the test.
    clock = [time.monotonic() + 1000]
    left: list[float] = []

    async def sleep(delay: float) -> None:
        clock[0] += delay

    def fail(*args: Any, **kwargs: Any) -> NoReturn:
        left.append(deadline - clock[0])
        raise DistNetworkError("refused")

    fake_time = types.SimpleNamespace(monotonic=lambda: clock[0])
    monkeypatch.setattr(tcpstore, "time", fake_time)
    fake_asyncio = types.SimpleNamespace(**{**vars(asyncio), "sleep": sleep})
    monkeypatch.setattr(tcpstore, "asyncio", fake_asyncio)
    monkeypatch.setattr(tcpstore, "_resolve", lambda *_: [_LOCALHOST])
    monkeypatch.setattr(tcpstore, "_open_store", fail)
    deadline = clock[0] + timeout_s
    lane = tcpstore._Lane()
    with pytest.raises(ConnectionError, match="opening failed: refused"):
        await tcpstore._open(
            "one.invalid",
            1,
            timeout_s,
            "opening",
            is_master=False,
            lane_for=lambda _: contextlib.nullcontext(lane),
        )
    assert left == pytest.approx(expected_left), left


def _never_replies(
    started: list[tuple[str, float]], release: threading.Event
) -> Callable[..., NoReturn]:
    """An attempt that records its host and start time, then holds until ``release`` as a silent peer would."""

    def never_replies(host: str, *args: Any, **kwargs: Any) -> NoReturn:
        started.append((host, time.monotonic()))
        release.wait(60)
        raise DistNetworkError("released")

    return never_replies


@pytest.mark.parametrize(("timeout_s", "attempts"), [(1.0, 2), (0.8, 1)])
@pytest.mark.asyncio
async def test_a_connect_to_silent_addresses_ends_at_its_deadline(
    monkeypatch, timeout_s: float, attempts: int
) -> None:
    """Bound an attempt at 6 attempt timeouts only while 3 more still fit, then end at the deadline.

    With the attempt timeout cut to 0.1 s, a 1 s connect bounds its first attempt and an 0.8 s one does not.
    """
    release = threading.Event()
    silent = [f"silent-{i}-{uuid.uuid4().hex}.invalid" for i in range(2)]
    started: list[tuple[str, float]] = []
    monkeypatch.setattr(tcpstore, "_CONNECT_ATTEMPT_S", 0.1)
    monkeypatch.setattr(tcpstore, "_resolve", lambda *_: silent)
    monkeypatch.setattr(tcpstore, "_open_store", _never_replies(started, release))
    start = time.monotonic()
    try:
        connect = tcpstore._open(
            "silent.invalid",
            1,
            timeout_s,
            "connecting",
            is_master=False,
            lane_for=tcpstore._shared_lane,
        )
        with pytest.raises(TimeoutError, match="connecting did not finish"):
            await asyncio.wait_for(connect, 10)
        elapsed = time.monotonic() - start
    finally:
        release.set()
    assert [host for host, _ in started] == silent[:attempts]
    assert timeout_s - 0.01 < elapsed < timeout_s + 0.5, elapsed
    if attempts == 2:
        assert 0.55 < started[1][1] - started[0][1] < 0.8, started


@pytest.mark.parametrize(
    ("addresses", "is_master"),
    [(["one.invalid"], False), (["first.invalid", "second.invalid"], True)],
    ids=["connect to one address", "serve"],
)
@pytest.mark.asyncio
async def test_a_serve_and_a_connect_to_one_address_never_bound_an_attempt(
    monkeypatch, addresses: list[str], is_master: bool
) -> None:
    """Leave an attempt that never replies to the deadline on a serve and on a connect to one address.

    Only a connect to several addresses moves on from it.
    With the attempt timeout cut to 0.1s these thus end as not finished.
    """
    release = threading.Event()
    started: list[tuple[str, float]] = []
    monkeypatch.setattr(tcpstore, "_CONNECT_ATTEMPT_S", 0.1)
    monkeypatch.setattr(tcpstore, "_open_store", _never_replies(started, release))
    monkeypatch.setattr(tcpstore, "_resolve", lambda *_: addresses)
    lane = tcpstore._Lane()
    try:
        opening = tcpstore._open(
            "name.invalid",
            0 if is_master else 1,
            1.0,
            "opening",
            is_master=is_master,
            lane_for=lambda _: contextlib.nullcontext(lane),
        )
        with pytest.raises(TimeoutError, match=r"opening did not finish within"):
            await asyncio.wait_for(opening, 10)
    finally:
        release.set()
    assert [host for host, _ in started] == addresses[:1]


@pytest.mark.asyncio
async def test_a_serve_whose_first_address_fails_serves_on_the_next(
    monkeypatch,
) -> None:
    """Retry a serve on the next looked-up address once the first fails."""
    dialed: list[str] = []
    open_store = tcpstore._open_store

    def first_fails(host: str, *args: Any, **kwargs: Any) -> TCPStore:
        dialed.append(host)
        if host != _LOCALHOST:
            raise tcpstore._QuietNetworkError(f"cannot connect to {host}")
        return open_store(host, *args, **kwargs)

    monkeypatch.setattr(tcpstore, "_resolve", lambda *_: ["first.invalid", _LOCALHOST])
    monkeypatch.setattr(tcpstore, "_open_store", first_fails)
    served = await TCPStoreRendezvous.serve(
        _store_name(), host="two.invalid", timeout_s=5
    )
    await served.close()
    assert served.address.host == "two.invalid"
    assert dialed == ["first.invalid", _LOCALHOST]


@pytest.mark.asyncio
async def test_retries_skip_a_busy_address_and_still_take_the_rest_in_turn(
    monkeypatch,
) -> None:
    """Skip an address whose abandoned attempt still runs without dialing another twice in a row.

    The first of three addresses never replies, the second always fails and the third answers its second dial.
    """
    release = threading.Event()
    first, second, third = (f"{i}-{uuid.uuid4().hex}.invalid" for i in range(3))
    dialed: list[str] = []
    store = object()

    def dial(host: str, *args: Any, **kwargs: Any) -> object:
        dialed.append(host)
        if host == first:
            release.wait(60)
            raise DistNetworkError("released")
        if host == second or dialed.count(third) == 1:
            raise tcpstore._QuietNetworkError(f"cannot connect to {host}")
        return store

    monkeypatch.setattr(tcpstore, "_CONNECT_ATTEMPT_S", 0.1)
    monkeypatch.setattr(tcpstore, "_CONNECT_RETRY_S", 0.01)
    monkeypatch.setattr(tcpstore, "_resolve", lambda *_: [first, second, third])
    monkeypatch.setattr(tcpstore, "_open_store", dial)
    try:
        opened = await tcpstore._open(
            "three.invalid",
            1,
            10,
            "connecting",
            is_master=False,
            lane_for=tcpstore._shared_lane,
        )
    finally:
        release.set()
    assert opened is store
    assert dialed == [first, second, third, second, third]


def test_a_lookup_of_one_address_probes_none_and_counts_duplicates_once(
    monkeypatch,
) -> None:
    """Return a name's single address without probing it, counting a duplicate of it once."""
    dialed: list[object] = []
    stream = (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "")

    def refuse(address: Any, *args: Any, **kwargs: Any) -> NoReturn:
        dialed.append(address)
        raise ConnectionRefusedError("refused")

    monkeypatch.setattr(
        socket, "getaddrinfo", lambda *args, **kwargs: [(*stream, (_LOCALHOST, 0))] * 2
    )
    monkeypatch.setattr(socket, "create_connection", refuse)
    assert tcpstore._resolve("twice.invalid", 1, 0.1) == [_LOCALHOST]
    assert not dialed


def test_probes_fall_back_when_a_dual_stack_bind_fails_or_descriptors_run_out(
    monkeypatch,
) -> None:
    """Probe on IPv4 when a dual-stack bind fails, and keep the looked-up order when no probe can listen."""
    create_server = socket.create_server

    def no_ipv6(
        address: Any, *, family: int = socket.AF_INET, **kwargs: Any
    ) -> socket.socket:
        if family == socket.AF_INET6:
            raise OSError("IPv6 is disabled")
        return create_server(address, family=family, **kwargs)

    def no_descriptors() -> NoReturn:
        raise OSError(24, "Too many open files")

    monkeypatch.setattr(socket, "has_dualstack_ipv6", lambda: True)
    monkeypatch.setattr(socket, "create_server", no_ipv6)
    with tcpstore._listener() as listener:
        assert listener.family == socket.AF_INET
    monkeypatch.setattr(tcpstore, "_listener", no_descriptors)
    addresses = ["first.invalid", "second.invalid"]
    assert tcpstore._answering_first(addresses, 0, 0.1) == addresses


@pytest.mark.asyncio
async def test_a_serve_records_and_forgets_its_listener_under_the_probe_lock(
    monkeypatch,
) -> None:
    """Open, hand off and close a serve's listener and record its port under the lock a fork takes.

    The listener leaves its set once closed, and the port only once the store is recorded.
    """
    lock = tcpstore._probe_listeners_lock
    changes: list[tuple[str, bool | None]] = []
    listen = tcpstore._listener

    class Detached(socket.socket):
        def detach(self) -> int:
            changes.append(("detached", lock.locked()))
            return super().detach()

        def close(self) -> None:
            changes.append(("closed", lock.locked()))
            super().close()

    def listener() -> socket.socket:
        opened = listen()
        changes.append(("opened", lock.locked()))
        return Detached(opened.family, opened.type, opened.proto, opened.detach())

    class Checked(set):
        """Records each change under its name and whether the probe lock was held."""

        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name

        def add(self, item: Any) -> None:
            changes.append((f"{self.name} added", lock.locked()))
            super().add(item)

        def discard(self, item: Any) -> None:
            changes.append((f"{self.name} dropped", lock.locked()))
            super().discard(item)

    class Masters(weakref.WeakSet):
        def add(self, item: Any) -> None:
            changes.append(("store recorded", None))
            super().add(item)

    listeners, ports = Checked("listener"), Checked("port")
    monkeypatch.setattr(tcpstore, "_listener", listener)
    monkeypatch.setattr(tcpstore, "_probe_listeners", listeners)
    monkeypatch.setattr(tcpstore, "_handed_off_ports", ports)
    monkeypatch.setattr(tcpstore, "_masters", Masters())
    async with _served() as rendezvous:
        assert rendezvous._store in tcpstore._masters
        assert not listeners and not ports
    # The served TCPStore, then its PrefixStore, are recorded.
    assert changes == [
        ("opened", True),
        ("listener added", True),
        ("detached", True),
        ("port added", True),
        ("closed", True),
        ("listener dropped", True),
        ("store recorded", None),
        ("port dropped", True),
        ("store recorded", None),
    ], changes


@pytest.mark.asyncio
async def test_a_name_whose_addresses_all_refuse_fails_naming_host_and_port(
    monkeypatch,
) -> None:
    """Dial anyway, then fail naming host:port, when no looked-up address answers."""
    with socket.socket() as probe:
        probe.bind((_LOCALHOST, 0))  # bound but not listening, so connects are refused
        port = probe.getsockname()[1]
        name = "refusing.invalid"
        monkeypatch.setattr(socket, "getaddrinfo", _two_addresses({name: "127.0.0.2"}))
        address = TCPStoreAddress(name, port, _store_name(), uuid.uuid4().hex)
        with pytest.raises(_UNREACHABLE, match=re.escape(f"{name}:{port}")):
            await TCPStoreRendezvous.connect(address, timeout_s=2)
        _join_native_calls()  # before the port is released for reuse


@pytest.mark.asyncio
async def test_a_refused_connect_names_its_cause_and_logs_nothing(capfd) -> None:
    """Dial in Python before each native attempt, so a refused port fails quietly.

    c10d would log each failed attempt with a C++ backtrace and report a timeout.
    """
    with socket.socket() as probe:
        probe.bind((_LOCALHOST, 0))  # bound but not listening, so connects are refused
        port = probe.getsockname()[1]
        address = TCPStoreAddress(_LOCALHOST, port, _store_name(), uuid.uuid4().hex)
        with pytest.raises(ConnectionError, match="Connection refused") as caught:
            await TCPStoreRendezvous.connect(address, timeout_s=3)
        _join_native_calls()  # before the port is released for reuse
    assert "c10d" not in capfd.readouterr().err
    assert isinstance(caught.value.__cause__, tcpstore._QuietNetworkError)  # no backoff


@pytest.mark.asyncio
async def test_a_serve_whose_host_misses_this_machine_names_the_cause_quietly(
    monkeypatch, capfd
) -> None:
    """Dial a serve's own host in Python before each native attempt, as connects do.

    A host that does not reach this machine then fails with that cause, never in c10d.
    """
    name = "elsewhere.invalid"
    lookup = socket.getaddrinfo

    def elsewhere(host: str, *args: Any, **kwargs: Any) -> Any:
        if host != name:
            return lookup(host, *args, **kwargs)
        stream = (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "")
        return [(*stream, ("255.255.255.255", 0))]  # TCP never connects to it

    monkeypatch.setattr(socket, "getaddrinfo", elsewhere)
    cause = f"TCPStore for {name} failed: cannot connect to 255.255.255.255:"
    with pytest.raises(ConnectionError, match=re.escape(cause)):
        await TCPStoreRendezvous.serve(_store_name(), host=name, timeout_s=3)
    assert "c10d" not in capfd.readouterr().err


@pytest.mark.asyncio
async def test_native_calls_dial_numeric_hosts_only(monkeypatch) -> None:
    """Look names up in Python, so no lookup can return into TCPStore's constructor."""
    dialed: list[tuple[str, bool]] = []
    open_store = tcpstore.TCPStore

    def record(host: str, port: int, *, is_master: bool, **kwargs: Any) -> TCPStore:
        dialed.append((host, is_master))
        return open_store(host, port, is_master=is_master, **kwargs)

    monkeypatch.setattr(tcpstore, "TCPStore", record)
    async with _served(host="localhost") as served, _connected(served.address):
        assert served.address.host == "localhost"  # clients still dial the name
    assert {is_master for _, is_master in dialed} == {True, False}, dialed
    for host, _ in dialed:
        ipaddress.ip_address(host)


@pytest.mark.asyncio
async def test_a_serve_cancelled_as_its_store_opens_keeps_no_server(
    monkeypatch,
) -> None:
    """Stop a cancelled serve's server, even while its CancelledError is kept."""
    serving: list[asyncio.Task[TCPStoreRendezvous]] = []
    opened: list[int] = []
    settle = tcpstore._settle

    def settle_then_cancel(
        future: asyncio.Future[object], result: object, error: Exception | None
    ) -> None:
        settle(future, result, error)
        if isinstance(result, TCPStore):  # the cancel lands after the result
            opened.append(result.port)
            serving[0].cancel()

    monkeypatch.setattr(tcpstore, "_settle", settle_then_cancel)
    gc.disable()
    try:
        serving.append(
            asyncio.create_task(
                TCPStoreRendezvous.serve(_store_name(), host=_LOCALHOST)
            )
        )
        with pytest.raises(asyncio.CancelledError) as caught:
            await serving[0]
        _join_native_calls()  # a finished call's thread holds its result till it ends
        await asyncio.sleep(0)  # the loop lets go of the last step's wakeup
        with pytest.raises(ConnectionRefusedError):
            socket.create_connection((_LOCALHOST, opened[0]), 10)
    finally:
        gc.enable()


@pytest.mark.parametrize(
    "failed", [None, "in_time", "late", "not_started", "cancelled"]
)
@pytest.mark.asyncio
async def test_close_stops_the_server_without_gc_and_refuses_calls(
    failed: str | None,
) -> None:
    """Stop serving on close, even after a failed or late call, and refuse calls.

    The failed call's error, or its cancelled task, stays referenced throughout.
    """
    entered, release = threading.Event(), threading.Event()
    kept: list[object] = []

    def oversized_check(store) -> bool:
        entered.set()
        release.wait(timeout=60)
        # An oversized key, past the size guard, makes the server drop the socket.
        return store.check(["k" * 10_000])

    gc.disable()
    try:
        async with _served() as rendezvous, _connected(rendezvous.address) as other:
            if failed in ("late", "cancelled"):
                deadline = time.monotonic() + (2 if failed == "late" else 60)
                call = asyncio.create_task(
                    rendezvous._call("check", oversized_check, deadline)
                )
                await _until(entered.is_set)  # given up on while it runs
                if failed == "cancelled":
                    call.cancel()
                error = TimeoutError if failed == "late" else asyncio.CancelledError
                with pytest.raises(error) as caught:
                    await call
                kept += [call, caught.value]
            elif failed == "not_started":
                with pytest.raises(TimeoutError, match="was not started") as caught:
                    await rendezvous.read(timeout_s=0)
                kept.append(caught.value)
            release.set()  # a late or cancelled call fails after its caller gave up
            if failed == "in_time":
                with pytest.raises(ConnectionError) as caught:
                    await rendezvous._call("check", oversized_check, _deadline())
                kept.append(caught.value)
            await rendezvous.close()
            _join_native_calls()  # finished calls' threads hold the store till they end
            await asyncio.sleep(0)  # the loop lets go of the last step's wakeup
            with pytest.raises(ConnectionRefusedError):
                socket.create_connection((_LOCALHOST, rendezvous.address.port), 10)
            with pytest.raises(
                ConnectionError, match=re.escape(tcpstore._endpoint(other.address))
            ):
                await other.read(timeout_s=10)
            with pytest.raises(RuntimeError, match="is closed"):
                await rendezvous.read()
    finally:
        release.set()
        gc.enable()


@pytest.mark.asyncio
async def test_a_close_landing_after_the_closed_check_lets_that_call_run() -> None:
    """Run a call that passed the closed check though close() lands just after it.

    close() may run on another thread. The calls after it are refused.
    """

    class ClosedAfterCheck(TCPStoreRendezvous):
        armed = False

        def __getattribute__(self, name: str) -> Any:
            value = super().__getattribute__(name)
            if name == "_prefixed" and self.armed:  # close() lands just after this read
                self.armed, self._prefixed = False, None
            return value

    async with _served() as served:
        racing = ClosedAfterCheck(served.address, cast(TCPStore, served._store))
        racing.armed = True
        assert not await racing._check(["k"], _deadline())
        with pytest.raises(RuntimeError, match="is closed"):
            await racing._check(["k"], _deadline())
        await racing.close()


@pytest.mark.asyncio
async def test_address_pickles_and_the_rendezvous_refuses() -> None:
    """Send addresses between processes, never live connections."""
    async with _served() as rendezvous:
        assert pickle.loads(pickle.dumps(rendezvous.address)) == rendezvous.address
        with pytest.raises(dataclasses.FrozenInstanceError):
            rendezvous.address.nonce = "stale"  # type: ignore[misc]
        for dumps in (pickle.dumps, cloudpickle.dumps):
            with pytest.raises(TypeError, match="pass the address"):
                dumps(rendezvous)


def test_a_forked_child_keeps_no_copy_of_a_served_listener() -> None:
    """Close a forked child's copies of served sockets, keep its inherited servers alive, and refuse inherited handles.

    Children fork during a serve's handoff, after two serves, after close() while a call holds the
    PrefixStore, and while the probe-listener lock is held. The parent's sockets stay blocking.
    """
    root = os.path.dirname(os.path.dirname(ts.__file__))
    child = subprocess.run(
        [sys.executable, "-c", _FORK_CASES, root],
        capture_output=True,
        text=True,
        timeout=100,
    )
    assert child.returncode == 0, child.stderr[-2000:]
    inherited, results, answered = child.stdout.splitlines()[-3:]
    assert re.fullmatch(
        r"connection to TCPStore 127\.0\.0\.1:\d+ was inherited across fork\. "
        r"Connect again from the address",
        inherited,
    ), child.stdout
    expected = {"handoff": 0, "after_serves": 0, "closed": 0, "probe_lock": 0}
    assert json.loads(results) == expected, child.stderr[-2000:]
    assert answered == "False", child.stderr[-2000:]


# Reloads the module and prints whether it kept each name its state guard keeps and the deadline sentinel,
# forks under an alarm that ends a deadlocked fork and prints the child's status, then leaves a call
# running so each exit drain logs. _exiting holds a marker a reload would replace. argv[1] is the root.
_RELOAD = _IMPORT_ALONE + (
    "import importlib, os, signal, threading\n"
    "names = ('_EXPIRED', '_live_calls', '_live_calls_lock', '_exiting', '_masters', '_connect_lanes',\n"
    "         '_connect_lanes_lock', '_probe_listeners', '_probe_listeners_lock', '_handed_off_ports')\n"
    "tcpstore._exiting = object()\n"
    "state = [getattr(tcpstore, name) for name in names]\n"
    "importlib.reload(tcpstore)\n"
    "print(all(getattr(tcpstore, n) is s for n, s in zip(names, state)), flush=True)\n"
    "tcpstore._exiting = False\n"
    "signal.alarm(30)\n"
    "pid = os.fork()\n"
    "if pid == 0:\n"
    "    os._exit(0)\n"
    "print(os.waitstatus_to_exitcode(os.waitpid(pid, 0)[1]), flush=True)\n"
    "signal.alarm(0)\n"
    "tcpstore._EXIT_DRAIN_S = 0.1\n"
    "stuck = threading.Thread(target=threading.Event().wait, name='stuck', daemon=True)\n"
    "stuck.start()\n"
    "tcpstore._live_calls.add(stuck)\n"
)


def test_a_reload_keeps_the_state_and_registers_no_hooks_again() -> None:
    """Keep the module's state across a reload, and its fork and exit hooks registered once.

    A second registration of the fork hooks would take the probe-listener lock twice and every fork would hang.
    """
    root = os.path.dirname(os.path.dirname(ts.__file__))
    child = subprocess.run(
        [sys.executable, "-c", _RELOAD, root],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert child.returncode == 0, child.stderr[-2000:]
    assert child.stdout.splitlines()[-2:] == ["True", "0"], child.stdout
    assert child.stderr.count("Exiting with TCPStore calls still running") == 1


def test_the_fork_hook_resets_bookkeeping_and_closes_inherited_sockets(
    monkeypatch,
) -> None:
    """Reset a child's calls, lanes, locks and served stores, and close its probe listeners and handed-off sockets.

    It runs here on stand-in state, under a default socket timeout it keeps and that turns no socket non-blocking.
    """
    held = [threading.Lock() for _ in range(3)]
    for lock in held:
        lock.acquire()  # as another thread of the parent may hold it at fork
    lanes = {"endpoint": tcpstore._Lane()}
    handed_off: set[int] = set()
    monkeypatch.setattr(tcpstore, "_live_calls", {threading.current_thread()})
    monkeypatch.setattr(tcpstore, "_exiting", True)
    monkeypatch.setattr(tcpstore, "_connect_lanes", lanes)
    monkeypatch.setattr(tcpstore, "_probe_listeners", set())
    monkeypatch.setattr(tcpstore, "_handed_off_ports", handed_off)
    # Stands in for a served store, which the hook keeps alive.
    served = threading.Event()
    monkeypatch.setattr(tcpstore, "_masters", type(tcpstore._masters)([served]))
    default = socket.getdefaulttimeout()
    with contextlib.ExitStack() as stack:
        probe = stack.enter_context(tcpstore._probe_listener())
        assert probe in tcpstore._probe_listeners
        handed = stack.enter_context(socket.create_server((_LOCALHOST, 0)))
        client = stack.enter_context(socket.create_connection(handed.getsockname()))
        accepted = stack.enter_context(handed.accept()[0])
        client.setblocking(True)
        handed_off.add(handed.getsockname()[1])
        inodes = [os.fstat(sock.fileno()).st_ino for sock in (handed, accepted)]
        monkeypatch.setattr(tcpstore, "_live_calls_lock", held[0])
        monkeypatch.setattr(tcpstore, "_connect_lanes_lock", held[1])
        monkeypatch.setattr(tcpstore, "_probe_listeners_lock", held[2])
        socket.setdefaulttimeout(30)
        try:
            tcpstore._after_fork_in_child()
            kept_default = socket.getdefaulttimeout()
        finally:
            socket.setdefaulttimeout(default)
        # The hook closed these descriptors, which the socket objects must not close again.
        fds = [handed.detach(), accepted.detach()]
        assert kept_default == 30 and os.get_blocking(client.fileno())
        for fd, inode in zip(fds, inodes):
            with contextlib.suppress(OSError):
                assert os.fstat(fd).st_ino != inode, "a handed-off socket stayed open"
        assert probe.fileno() == -1 and not tcpstore._probe_listeners
        assert not handed_off and not lanes
        # Forgotten: this child's own forks must never close sockets on the parent's ports.
        assert not tcpstore._masters
        assert tcpstore._live_calls == set() and tcpstore._exiting is False
        for name, lock in zip(
            ("_live_calls_lock", "_connect_lanes_lock", "_probe_listeners_lock"), held
        ):
            replaced = getattr(tcpstore, name)
            assert replaced is not lock and not replaced.locked(), name
    for lock in held:
        lock.release()


def test_local_port_names_tcp_sockets_and_dups_only_sockets(
    monkeypatch, tmp_path
) -> None:
    """Report the local port of TCP sockets and None for any other descriptor, duplicating only sockets.

    Closing a duplicate of a file on a hung FUSE or NFS mount would wait on its server.
    """
    duplicated: list[int] = []
    dup = os.dup

    def recording_dup(fd: int) -> int:
        duplicated.append(fd)
        return dup(fd)

    reading, writing = os.pipe()
    with contextlib.ExitStack() as stack:
        stack.callback(os.close, reading)
        stack.callback(os.close, writing)
        listener = stack.enter_context(socket.create_server((_LOCALHOST, 0)))
        client = stack.enter_context(socket.create_connection(listener.getsockname()))
        accepted = stack.enter_context(listener.accept()[0])
        unix = [stack.enter_context(sock) for sock in socket.socketpair()]
        udp = stack.enter_context(socket.socket(socket.AF_INET, socket.SOCK_DGRAM))
        udp.bind((_LOCALHOST, 0))
        regular = stack.enter_context(open(tmp_path / "regular", "w"))
        port = listener.getsockname()[1]
        monkeypatch.setattr(os, "dup", recording_dup)
        assert tcpstore._local_port(listener.fileno()) == port
        assert tcpstore._local_port(accepted.fileno()) == port
        assert tcpstore._local_port(client.fileno()) == client.getsockname()[1]
        others = [unix[0].fileno(), udp.fileno(), reading, writing, regular.fileno()]
        assert [tcpstore._local_port(fd) for fd in others] == [None] * 5
        monkeypatch.undo()
        assert not {reading, writing, regular.fileno()} & set(duplicated), duplicated


def _refused(*args: Any, **kwargs: Any) -> NoReturn:
    raise OSError("refused")


def test_the_fork_hooks_scan_closes_nothing_when_it_cannot_list_descriptors(
    monkeypatch,
) -> None:
    """Close nothing when the descriptor list cannot be read."""
    with socket.create_server((_LOCALHOST, 0)) as listener:
        inode = os.fstat(listener.fileno()).st_ino
        unlisted = types.SimpleNamespace(**{**vars(os), "listdir": _refused})
        monkeypatch.setattr(tcpstore, "os", unlisted)
        tcpstore._close_served_sockets({listener.getsockname()[1]})
        monkeypatch.undo()
        # os.close leaves the socket object's number so only the descriptor's inode tells.
        assert os.fstat(listener.fileno()).st_ino == inode


class _Unnamed:
    """A wrapped socket whose name cannot be read."""

    family, type = socket.AF_INET, socket.SOCK_STREAM

    def __init__(self, fileno: int) -> None:
        self.fd = fileno

    def __enter__(self) -> _Unnamed:
        return self

    def __exit__(self, *exc: object) -> None:
        os.close(self.fd)

    def getsockname(self) -> NoReturn:
        raise OSError("not connected")


@pytest.mark.parametrize("wrap", [_refused, _Unnamed], ids=["unwrapped", "unnamed"])
def test_the_fork_hooks_scan_skips_a_socket_it_cannot_wrap_or_name(
    monkeypatch, wrap: Callable[..., Any]
) -> None:
    """Skip a socket that cannot be wrapped or named, and close the duplicate it made."""
    with socket.create_server((_LOCALHOST, 0)) as listener:
        before = set(os.listdir("/proc/self/fd"))
        wrapping = types.SimpleNamespace(**{**vars(socket), "socket": wrap})
        monkeypatch.setattr(tcpstore, "socket", wrapping)
        assert tcpstore._local_port(listener.fileno()) is None
        monkeypatch.undo()
        assert set(os.listdir("/proc/self/fd")) == before


@pytest.mark.asyncio
async def test_serve_advertises_the_argument_then_environment_then_hostname(
    monkeypatch,
) -> None:
    """Advertise the host argument, else the environment, else the hostname.

    An empty argument or environment variable counts as unset.
    """
    monkeypatch.setenv("TORCHSTORE_TCPSTORE_HOST", _LOCALHOST)
    async with _served(host=None) as from_environment, _served(
        host="localhost"
    ) as from_argument, _served(host="") as from_empty_argument:
        assert from_environment.address.host == _LOCALHOST
        assert from_argument.address.host == "localhost"
        assert from_empty_argument.address.host == _LOCALHOST

    monkeypatch.delenv("TORCHSTORE_TCPSTORE_HOST")
    try:
        socket.getaddrinfo(socket.gethostname(), None)
    except OSError:
        pytest.skip("the hostname does not resolve")
    # Clients on other hosts dial the hostname.
    async with _served(host=None) as rendezvous, _connected(
        rendezvous.address
    ) as other:
        assert rendezvous.address.host == socket.gethostname()
        await rendezvous.publish(_PUBLISHER_RANKS, _strategy())
        publishers, _ = await other.read()
        assert publishers == _PUBLISHER_RANKS

    monkeypatch.setenv("TORCHSTORE_TCPSTORE_HOST", "")
    async with _served(host=None) as from_empty_environment:
        assert from_empty_environment.address.host == socket.gethostname()


@pytest.mark.skipif(not _has_ipv6_loopback(), reason="no dual-stack IPv6 loopback")
@pytest.mark.asyncio
async def test_a_served_store_answers_on_ipv6_and_ipv4() -> None:
    """Serve on either loopback family and answer on both, as a dual-stack listener does.

    The server takes the listener a serve binds, so that bind decides who reaches it.
    """
    for host, other_family in (("::1", _LOCALHOST), (_LOCALHOST, "::1")):
        served = await TCPStoreRendezvous.serve(_store_name(), host=host, timeout_s=10)
        try:
            other = await TCPStoreRendezvous.connect(served.address, timeout_s=10)
            try:
                await served._compare_set("k", b"v", _deadline())
                assert await other._check(["k"], _deadline())
            finally:
                await other.close()
            socket.create_connection((other_family, served.address.port), 10).close()
        finally:
            await served.close()
