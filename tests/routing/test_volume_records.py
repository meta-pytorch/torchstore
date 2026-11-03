# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Volume records on the TCPStore rendezvous: contract, keys, chunks and failures."""

from __future__ import annotations

import ast
import asyncio
import contextlib
import dataclasses
import inspect
import itertools
import json
import logging
import re
import socket
import subprocess
import sys
import threading
import time
import types
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Collection,
    Coroutine,
    Iterator,
)
from pathlib import Path
from typing import Any, cast, NoReturn
from urllib.parse import quote

import pytest
import pytest_asyncio
from torchstore import native_calls
from torchstore.routing import tcpstore, volume_records as volume_records_module
from torchstore.routing.layout_registry import LayoutRegistry
from torchstore.routing.tcpstore import TCPStoreRendezvous
from torchstore.routing.volume_records import VolumeRecords

# The shared TCPStore tests import cloudpickle and Monarch when they load.
pytest.importorskip("cloudpickle")
pytest.importorskip("monarch")

from . import test_tcpstore as tcpstore_tests
from .test_control_plane import _imported
from .test_tcpstore import (
    _connected,
    _deadline,
    _LOCALHOST,
    _native_calls,
    _PUBLISHER_RANKS,
    _served,
    _stop,
    _store_name,
    _strategy,
    _until,
)

pytestmark = [
    pytest.mark.timeout(120),
    # A native-call thread that raises, say on a closed loop, fails its test.
    pytest.mark.filterwarnings("error::pytest.PytestUnhandledThreadExceptionWarning"),
]

# The TCPStore tests' fixtures, which pytest finds here under these names.
server = tcpstore_tests.server
_joins_its_native_calls = tcpstore_tests._joins_its_native_calls

# The design's interface. The rendezvous adds only a keyword-only timeout to each.
_DESIGN = {
    "register_volume": "(self, volume_id: 'str', record: 'bytes', /) -> 'None'",
    "volume_records": "(self, volume_ids: 'Collection[str]', /) -> 'dict[str, bytes]'",
}
# Slashes, percents, an id that looks encoded, non-ASCII and another key's name.
_ODD_IDS = ["publisher/0", "50%", "a/b", "a%2Fb", "卷/数据", "ready"]
_METHODS = ("register_volume", "volume_records")
# Each method's last store call in ``_call`` and how the thread name of the native call making it starts.
_LAST_STORE_CALL = {
    "register_volume": ("compare_set", "volume/publisher%2F1"),
    "volume_records": ("multi_get", ["volume/publisher%2F0"]),
}
_LAST_NATIVE_CALL = {
    "register_volume": "compare_set('volume/publisher%2F1')",
    "volume_records": "reading 1 volume records",
}
# 40 publishers that reads take in chunks of 16. The 3 without a record sit in chunks 2, 1 and 1.
_FORTY = [f"publisher/{i}" for i in range(40)]
_MISSING = ["publisher/21", "publisher/5", "publisher/7"]
# Imports the interface with Monarch and cloudpickle blocked and the packages empty,
# then lists every TorchStore module loaded. argv: the root of the tree.
_IMPORT_ALONE = (
    "import sys, types\n"
    "sys.modules['monarch'] = sys.modules['cloudpickle'] = None\n"
    "for name in ('torchstore', 'torchstore.routing'):\n"
    "    package = sys.modules[name] = types.ModuleType(name)\n"
    "    package.__path__ = [sys.argv[1] + '/' + name.replace('.', '/')]\n"
    "from torchstore.routing.volume_records import VolumeRecords\n"
    "print(sorted(m for m in sys.modules if m.split('.')[0] == 'torchstore'))\n"
)
# Registers publisher/1's record from a fresh interpreter. argv: the address as JSON, then the record.
_REGISTER_IN_CHILD = (
    "import asyncio, json, sys\n"
    "from torchstore.routing.tcpstore import TCPStoreAddress, TCPStoreRendezvous\n"
    "async def register():\n"
    "    address = TCPStoreAddress(**json.loads(sys.argv[1]))\n"
    "    rendezvous = await TCPStoreRendezvous.connect(address)\n"
    "    try:\n"
    "        await rendezvous.register_volume('publisher/1', sys.argv[2].encode())\n"
    "    finally:\n"
    "        await rendezvous.close()\n"
    "asyncio.run(register())\n"
)


def _call(
    records: VolumeRecords, method: str, **kwargs: Any
) -> Coroutine[Any, Any, Any]:
    """Register ``publisher/1``'s record, or read ``publisher/0``'s."""
    if method == "register_volume":
        return records.register_volume("publisher/1", b"record", **kwargs)
    return records.volume_records(["publisher/0"], **kwargs)


def _request_size(prefix: str, keys: list[str]) -> int:
    """Count a check or multi_get request: query type, key count, then each wire key."""
    return 9 + sum(8 + len(f"/{prefix}/{key}".encode()) for key in keys)


def _moved(name: str, args: tuple[Any, ...], result: Any) -> int:
    """The value bytes a store call wrote and read back."""
    if name == "compare_set":
        return len(args[2]) + len(result)
    if name == "multi_set":
        return sum(map(len, args[1]))
    return sum(map(len, result)) if name == "multi_get" else 0


_Hook = Callable[[str, Any], None]


@contextlib.contextmanager
def _then(cleanup: Callable[[], object]) -> Iterator[None]:
    """Run ``cleanup`` on the way out. A failing body stays the reported failure though ``cleanup`` raises too."""
    try:
        yield
    except BaseException:
        with contextlib.suppress(Exception):
            cleanup()
        raise
    cleanup()


@contextlib.contextmanager
def _store_calls(
    rendezvous: TCPStoreRendezvous,
    whats: list[str] | None = None,
    *,
    before: _Hook | None = None,
    after: _Hook | None = None,
) -> Iterator[tuple[list[tuple[str, Any]], list[tuple[str, Any, int]], list[float]]]:
    """Record the handle's store calls and each native call's deadline and name (the name into ``whats``).

    Yields the store calls as they start, the finished ones with the value bytes each moved, and the deadlines.
    ``before`` and ``after`` get each store call's name and key or keys and run around it on the native call's thread.
    """
    calls: list[tuple[str, Any]] = []
    counted: list[tuple[str, Any, int]] = []
    deadlines: list[float] = []
    store, run = rendezvous._prefixed, tcpstore._run_with_deadline
    assert store is not None
    prefixed = store  # a new name: type checkers do not narrow into closures

    def recorder(name: str) -> Callable[..., Any]:
        def call(*args: Any) -> Any:
            calls.append((name, args[0]))
            if before is not None:
                before(name, args[0])
            result = getattr(prefixed, name)(*args)
            counted.append((name, args[0], _moved(name, args, result)))
            if after is not None:
                after(name, args[0])
            return result

        return call

    async def record(fn: Callable[[], Any], deadline: float, what: str) -> Any:
        deadlines.append(deadline)
        if whats is not None:
            whats.append(what)
        return await run(fn, deadline, what)

    patch = pytest.MonkeyPatch()
    with _then(patch.undo):
        # The store calls the rendezvous makes. Any other call such as set fails the test.
        names = ("compare_set", "check", "multi_get", "multi_set")
        patch.setattr(
            rendezvous,
            "_prefixed",
            types.SimpleNamespace(**{name: recorder(name) for name in names}),
        )
        patch.setattr(tcpstore, "_run_with_deadline", record)
        yield calls, counted, deadlines


@contextlib.contextmanager
def _recording(
    rendezvous: TCPStoreRendezvous,
    whats: list[str] | None = None,
    *,
    before: _Hook | None = None,
    after: _Hook | None = None,
) -> Iterator[tuple[list[tuple[str, Any]], list[float]]]:
    """Record each store call as it starts and each native call's deadline and name (the name into ``whats``)."""
    with _store_calls(rendezvous, whats, before=before, after=after) as (
        calls,
        _,
        deadlines,
    ):
        yield calls, deadlines


def _at(step: tuple[str, Any], then: Callable[[], None]) -> _Hook:
    """A store-call hook that runs ``then`` at ``step``: a store call's name with its key or keys."""

    def hook(name: str, keys: Any) -> None:
        if (name, keys) == step:
            then()

    return hook


async def _done_within_loop_steps(task: asyncio.Task[Any], steps: int = 5) -> None:
    """Require ``task`` done once the loop has run ``steps`` times with no wait between."""
    for _ in range(steps):
        await asyncio.sleep(0)
    assert task.done(), f"not done within {steps} loop steps"


class _Clock:
    """A module's clock that runs an hour ahead once ``late`` holds anything."""

    def __init__(self, late: list[bool]) -> None:
        self._late = late

    def __getattr__(self, name: str) -> Any:
        return getattr(time, name)

    def monotonic(self) -> float:
        return time.monotonic() + (3600.0 if self._late else 0.0)


async def _register_own_ids(
    rendezvous: TCPStoreRendezvous, volume_ids: list[str]
) -> None:
    """Publish ``volume_ids`` as the publishers, then register each one with its id as its record."""
    await rendezvous.publish(volume_ids, _strategy())
    for volume_id in volume_ids:
        await rendezvous.register_volume(volume_id, volume_id.encode())


@contextlib.asynccontextmanager
async def _tcpstore_records(
    publishers: Collection[str] | None,
) -> AsyncIterator[Callable[[], Awaitable[VolumeRecords]]]:
    """Serve a fresh store, publish ``publishers`` unless None, and connect handles."""
    async with _served() as driver, contextlib.AsyncExitStack() as handles:
        if publishers is not None:
            await driver.publish(publishers, _strategy())

        async def connect() -> VolumeRecords:
            return await handles.enter_async_context(_connected(driver.address))

        yield connect


@pytest_asyncio.fixture(params=[pytest.param(_tcpstore_records, id="tcpstore")])
async def make_records(
    request: pytest.FixtureRequest,
) -> AsyncIterator[Callable[..., Awaitable[Any]]]:
    """Open a fresh store per call and return a handle, or ``handles`` of them.

    Handles connect on their own, as ranks do. ``publishers=None`` skips init.
    """
    new_store = request.param
    async with contextlib.AsyncExitStack() as stack:

        async def make(
            publishers: Collection[str] | None = _PUBLISHER_RANKS, *, handles: int = 1
        ) -> Any:
            connect = await stack.enter_async_context(new_store(publishers))
            clients = [await connect() for _ in range(handles)]
            return clients[0] if handles == 1 else clients

        yield make


async def _forty_but_three_registered(
    make_records: Callable[..., Awaitable[Any]],
) -> VolumeRecords:
    """Register each of ``_FORTY`` but ``_MISSING`` through one handle and return another."""
    writer, reader = await make_records(_FORTY, handles=2)
    for rank in _FORTY:
        if rank not in _MISSING:
            await writer.register_volume(rank, f"record of {rank}".encode())
    return reader


@pytest.mark.asyncio
async def test_first_record_wins_and_the_same_record_again_returns(
    make_records,
) -> None:
    """Keep the first record, accept it again, and reject a different one."""
    records = await make_records()
    await records.register_volume("publisher/0", b"first")
    await records.register_volume("publisher/0", b"first")
    with pytest.raises(RuntimeError, match="'publisher/0' is already registered"):
        await records.register_volume("publisher/0", b"second")
    assert await records.volume_records(["publisher/0"]) == {"publisher/0": b"first"}


@pytest.mark.asyncio
async def test_only_publisher_ranks_register(make_records) -> None:
    """Reject an id that is not a publisher rank with the registry's KeyError, and read no record for it."""
    registry = LayoutRegistry()
    await registry.init(_PUBLISHER_RANKS, _strategy())
    records = await make_records()
    with pytest.raises(KeyError) as expected:
        await registry.register_layout("publisher/2", "model", {})
    with pytest.raises(KeyError) as raised:
        await records.register_volume("publisher/2", b"record")
    assert raised.value.args == expected.value.args
    with pytest.raises(KeyError, match=re.escape("'publisher/2'")):
        await records.volume_records(["publisher/2"])


@pytest.mark.asyncio
async def test_calls_before_init_fail(make_records) -> None:
    """Fail both methods before init, naming it. An empty read still returns {}."""
    records = await make_records(None)
    for method in _METHODS:
        with pytest.raises(RuntimeError, match="init has not run"):
            await _call(records, method)
    assert await records.volume_records([]) == {}


@pytest.mark.asyncio
async def test_each_missing_id_is_named_once(make_records) -> None:
    """Name every id without a record once in the KeyError even when one repeats in the read."""
    reader = await _forty_but_three_registered(make_records)
    with pytest.raises(KeyError, match=re.escape(f"record for {_MISSING}")):
        await reader.volume_records([*_FORTY, _MISSING[0]])


@pytest.mark.asyncio
async def test_every_record_comes_back(make_records) -> None:
    """Return the record of every id read that has one."""
    reader = await _forty_but_three_registered(make_records)
    present = [rank for rank in _FORTY if rank not in _MISSING]
    expected = {rank: f"record of {rank}".encode() for rank in present}
    assert await reader.volume_records(present) == expected


@pytest.mark.asyncio
async def test_one_handle_takes_calls_from_any_thread_or_loop(
    make_records, monkeypatch
) -> None:
    """Serve one handle's registrations and then its reads from this loop and a worker thread's own loop at once.

    Both reads pass their ready checks first. The other read asks for its turn on the handle only once the first
    holds it, and the first holds it until the other waits there. So the two reads always overlap.
    """
    records = await make_records()
    await asyncio.gather(
        asyncio.to_thread(
            asyncio.run, records.register_volume("publisher/0", b"from a thread")
        ),
        records.register_volume("publisher/1", b"from this loop"),
    )
    steps, admit = TCPStoreRendezvous._steps, tcpstore._Lane._admit
    ready, first, holding, waiting = (
        threading.Barrier(2, timeout=30),
        itertools.count(),
        threading.Event(),
        threading.Event(),
    )

    async def overlapping(
        self: TCPStoreRendezvous, what: str, fn: Any, deadline: float
    ) -> Any:
        await asyncio.to_thread(ready.wait)
        if next(first):
            assert await asyncio.to_thread(
                holding.wait, 30
            ), "the first read never took its turn"
            return await steps(self, what, fn, deadline)

        def held(*args: Any) -> Any:
            # Runs once this read holds the handle's turn, which it keeps until the other waits for it.
            holding.set()
            assert waiting.wait(30), "the other read never waited for its turn"
            return fn(*args)

        return await steps(self, what, held, deadline)

    def noted_admit(self: Any, call: Any, deadline: float, what: str) -> Any:
        if "volume records" in what and self._gate.locked():
            waiting.set()
        return admit(self, call, deadline, what)

    monkeypatch.setattr(TCPStoreRendezvous, "_steps", overlapping)
    monkeypatch.setattr(tcpstore._Lane, "_admit", noted_admit)
    ranks = sorted(_PUBLISHER_RANKS)
    reads = await asyncio.gather(
        asyncio.to_thread(
            asyncio.run, asyncio.wait_for(records.volume_records(ranks), 60)
        ),
        asyncio.wait_for(records.volume_records(ranks), 60),
    )
    expected = {"publisher/0": b"from a thread", "publisher/1": b"from this loop"}
    assert reads == [expected, expected]


@pytest.mark.asyncio
async def test_ids_with_slashes_percents_and_non_ascii_round_trip(
    make_records,
) -> None:
    """Keep each odd id's opaque record apart, including an id that looks encoded."""
    writer, reader = await make_records(_ODD_IDS, handles=2)
    records = {volume_id: b"\xff\x00" + volume_id.encode() for volume_id in _ODD_IDS}
    for volume_id, record in records.items():
        await writer.register_volume(volume_id, record)
    assert await reader.volume_records(_ODD_IDS) == records


@pytest.mark.asyncio
async def test_each_record_is_one_key_under_the_store_prefix() -> None:
    """Store each record at ``volume/{quoted id}``, one key segment under the prefix."""
    segments = [quote(volume_id, safe="") for volume_id in _ODD_IDS]
    async with _served() as rendezvous:
        await rendezvous.publish(_ODD_IDS, _strategy())
        for volume_id in _ODD_IDS:
            await rendezvous.register_volume(volume_id, volume_id.encode())
        prefix = tcpstore._prefix(rendezvous.address)
        keys = [f"{prefix}/volume/{segment}" for segment in segments]
        store = rendezvous._store
        assert store is not None
        # check never waits, unlike multi_get.
        assert store.check(keys)
        assert store.multi_get(keys) == [v.encode() for v in _ODD_IDS]
        # Each publisher's own key quotes its id the same way, so no id reads another's.
        members = [f"{prefix}/member/{segment}" for segment in segments]
        assert all(store.check([member]) for member in members)


@pytest.mark.asyncio
async def test_registration_reads_ready_then_its_own_publisher_key_then_writes_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read ready, then check the id's own publisher key, then write once. A conflict names the server.

    The default timeout is read at call time.
    """
    monkeypatch.setattr(tcpstore, "_DEFAULT_TIMEOUT_S", 30.0)
    async with _served() as rendezvous:
        await rendezvous.publish(_PUBLISHER_RANKS, _strategy())
        with _recording(rendezvous) as (calls, deadlines):
            before = time.monotonic()
            await rendezvous.register_volume("publisher/0", b"record")
            after = time.monotonic()
        with _recording(rendezvous) as (conflict, _):
            with pytest.raises(RuntimeError) as raised:
                await rendezvous.register_volume("publisher/0", b"other")
    assert conflict == calls
    ready = [tcpstore._READY]
    assert calls == [
        ("check", ready),
        ("multi_get", ready),
        ("check", ["member/publisher%2F0"]),
        ("compare_set", "volume/publisher%2F0"),
    ]
    assert deadlines == [deadlines[0]] * 4
    default = tcpstore._DEFAULT_TIMEOUT_S
    assert before + default <= deadlines[0] <= after + default
    address = rendezvous.address
    assert str(raised.value) == (
        f"routing store {address.store_name!r} on TCPStore "
        f"{tcpstore._endpoint(address)}: volume 'publisher/0' is already registered "
        "with another record"
    )


@pytest.mark.asyncio
async def test_a_registration_logs_one_line_once_its_record_is_stored(caplog) -> None:
    """Log one INFO line for a stored record and again for the same record but none for a refused one."""
    caplog.set_level(logging.DEBUG, logger=tcpstore.logger.name)
    async with _served() as rendezvous:
        await rendezvous.publish(_PUBLISHER_RANKS, _strategy())
        caplog.clear()
        for record in (b"record", b"record", b"other"):
            with contextlib.suppress(RuntimeError):
                await rendezvous.register_volume("publisher/0", record)
        prefix = tcpstore._prefix(rendezvous.address)
        endpoint = tcpstore._endpoint(rendezvous.address)
    key_bytes = len(f"/{prefix}/volume/publisher%2F0".encode())
    line = (
        "Volume 'publisher/0' has its record of 6 bytes at TCPStore key "
        f"'volume/publisher%2F0' ({key_bytes} key bytes) on {endpoint}"
    )
    logged = [r for r in caplog.records if r.name == tcpstore.logger.name]
    assert [(r.levelno, r.getMessage()) for r in logged] == [(logging.INFO, line)] * 2


@pytest.mark.parametrize(
    "key_bytes, record_bytes, level",
    [
        pytest.param(4096, 4 * 1024 * 1024, logging.INFO, id="both_at_half"),
        pytest.param(4097, 6, logging.WARNING, id="key_over_half"),
        pytest.param(None, 4 * 1024 * 1024 + 1, logging.WARNING, id="record_over_half"),
    ],
)
@pytest.mark.asyncio
async def test_a_registration_warns_once_its_key_or_record_is_over_half_its_cap(
    caplog, key_bytes: int | None, record_bytes: int, level: int
) -> None:
    """Log a registration at INFO up to half the 8 KiB key and 8 MiB value caps and at WARNING past either."""
    caplog.set_level(logging.DEBUG, logger=tcpstore.logger.name)
    async with _served() as rendezvous:
        volume_id = "publisher/0"
        if key_bytes is not None:
            prefix = tcpstore._prefix(rendezvous.address)
            volume_id = "p" * (key_bytes - len(f"/{prefix}/volume/"))
        await rendezvous.publish([volume_id], _strategy())
        caplog.clear()
        await rendezvous.register_volume(volume_id, b"x" * record_bytes)
    logged = [r for r in caplog.records if r.name == tcpstore.logger.name]
    assert [r.levelno for r in logged] == [level]
    message = logged[0].getMessage()
    assert f"record of {record_bytes} bytes" in message
    advice = (
        f". A key over {8 * 1024} bytes or a record over {8 * 1024 * 1024} bytes"
        " fails registration: shorten the store name"
    )
    assert message.endswith(advice) == (level == logging.WARNING)


@pytest.mark.asyncio
async def test_a_registration_reads_its_own_publisher_key_and_never_the_set() -> None:
    """Register with four store calls that move the record and the schema alone.

    publish writes one small key per publisher, so a registration never reads the set.
    """
    async with _served() as rendezvous:
        await rendezvous.publish(["publisher/0", "publisher/1"], _strategy())
        with _store_calls(rendezvous) as (_, calls, _):
            await rendezvous.register_volume("publisher/1", b"record")
    assert calls == [
        ("check", [tcpstore._READY], 0),
        ("multi_get", [tcpstore._READY], len(tcpstore._SCHEMA_VERSION)),
        ("check", ["member/publisher%2F1"], 0),
        ("compare_set", "volume/publisher%2F1", 2 * len(b"record")),
    ]


@pytest.mark.asyncio
async def test_a_publish_writes_its_publisher_keys_over_any_value() -> None:
    """Publish over another value at a publisher's key: only a publish of this publisher list gets there."""
    async with _served() as rendezvous:
        key = "member/publisher%2F1"
        assert await rendezvous._compare_set(key, b"other", _deadline()) == b"other"
        await rendezvous.publish(_PUBLISHER_RANKS, _strategy())
        assert await rendezvous._multi_get([key], _deadline()) == [tcpstore._MEMBER]
        assert await rendezvous._check([tcpstore._READY], _deadline())


@pytest.mark.parametrize(
    "publisher, error",
    [
        pytest.param(None, "bytes on the wire", id="key_over_the_cap"),
        pytest.param("\ud800", "surrogates not allowed", id="not_utf8"),
    ],
)
@pytest.mark.asyncio
async def test_a_publisher_id_without_its_own_key_fails_the_publish_first(
    publisher: str | None, error: str
) -> None:
    """Fail a publish before it writes any key when a publisher's own key is over the wire cap.

    A publisher id that UTF-8 cannot encode fails it the same way.
    """
    async with _served() as rendezvous:
        prefix = tcpstore._prefix(rendezvous.address)
        if publisher is None:
            publisher = "k" * (8192 - len(f"/{prefix}/member/") + 1)
        with _recording(rendezvous) as (calls, deadlines):
            with pytest.raises(ValueError, match=error):
                await rendezvous.publish([*_PUBLISHER_RANKS, publisher], _strategy())
        assert calls == [] and deadlines == []
        await rendezvous.publish(_PUBLISHER_RANKS, _strategy())


@pytest.mark.asyncio
async def test_publisher_keys_go_one_multi_set_per_chunk_and_native_call() -> None:
    """Write the publisher keys by one multi_set per native call of ``_MEMBER_CHUNK_KEYS`` keys.

    4,096 publishers take 16 such calls. Each is named for its first key.
    """
    publishers = [f"publisher/{i}" for i in range(4096)]
    members = sorted(tcpstore._member_key(publisher) for publisher in publishers)
    chunks = [members[start : start + 256] for start in range(0, 4096, 256)]
    async with _served() as rendezvous:
        whats: list[str] = []
        with _recording(rendezvous, whats) as (calls, _):
            await rendezvous.publish(publishers, _strategy())
        endpoint = tcpstore._endpoint(rendezvous.address)
    # The publisher list and strategy, a multi_set per chunk and then ready take one native call each.
    assert calls[2:] == [
        *(("multi_set", chunk) for chunk in chunks),
        ("compare_set", tcpstore._READY),
    ]
    assert whats[2:-1] == [
        f"multi_set of 256 publisher keys from {chunk[0]!r} on TCPStore {endpoint}"
        for chunk in chunks
    ]
    assert len(whats) == len(calls)


@pytest.mark.asyncio
async def test_a_publish_stops_before_its_next_chunk_at_its_deadline(
    monkeypatch,
) -> None:
    """Start no later chunk of publisher keys and no ready marker once a publish's deadline passes.

    The error names the first key of the chunk it did not start and the server.
    """
    monkeypatch.setattr(tcpstore, "_MEMBER_CHUNK_KEYS", 1)
    late: list[bool] = []
    async with _served() as rendezvous:
        endpoint = tcpstore._endpoint(rendezvous.address)
        what = re.escape(
            f"multi_set of 1 publisher keys from 'member/publisher%2F1' on TCPStore {endpoint}"
        )
        first = ("multi_set", ["member/publisher%2F0"])
        with _recording(rendezvous, after=_at(first, lambda: late.append(True))):
            for module in (tcpstore, native_calls):
                monkeypatch.setattr(module, "time", _Clock(late))
            with pytest.raises(
                TimeoutError, match=f"^{what} was not started: its deadline had passed$"
            ):
                await rendezvous.publish(["publisher/0", "publisher/1"], _strategy())
            monkeypatch.undo()
        assert await rendezvous._check(["member/publisher%2F0"], _deadline())
        for key in ("member/publisher%2F1", tcpstore._READY):
            assert not await rendezvous._check([key], _deadline()), key


@pytest.mark.asyncio
async def test_a_publish_logs_one_line_for_its_publisher_keys(caplog) -> None:
    """Log one INFO line for all publisher keys, naming the longest, and write and count a repeated id once."""
    caplog.set_level(logging.DEBUG, logger=tcpstore.logger.name)
    async with _served() as rendezvous:
        with _recording(rendezvous) as (calls, _):
            await rendezvous.publish(["a", "bbb", "cc", "a"], _strategy())
        prefix = tcpstore._prefix(rendezvous.address)
        endpoint = tcpstore._endpoint(rendezvous.address)
    assert ("multi_set", ["member/a", "member/bbb", "member/cc"]) in calls
    lines = [r for r in caplog.records if "member/" in r.getMessage()]
    key_bytes = len(f"/{prefix}/member/bbb".encode())
    assert [(r.levelno, r.getMessage()) for r in lines] == [
        (
            logging.INFO,
            f"Writing a key per publisher, 3 in all, the longest 'member/bbb' ({key_bytes} key bytes), "
            f"to TCPStore {endpoint}",
        )
    ]


@pytest.mark.asyncio
async def test_a_publish_without_publishers_writes_and_logs_no_publisher_key(
    caplog,
) -> None:
    """Publish an empty publisher set with no publisher key and no line for them."""
    caplog.set_level(logging.DEBUG, logger=tcpstore.logger.name)
    async with _served() as rendezvous:
        with _recording(rendezvous) as (calls, _):
            await rendezvous.publish([], _strategy())
    published = [tcpstore._PUBLISHERS, tcpstore._STRATEGY, tcpstore._READY]
    assert calls == [("compare_set", key) for key in published]
    assert not [r for r in caplog.records if "per publisher" in r.getMessage()]


@pytest.mark.asyncio
async def test_publisher_keys_near_the_key_cap_log_one_warning(caplog) -> None:
    """Warn once per publish when the longest publisher key is over half the cap.

    The line names that key and what to shorten.
    """
    caplog.set_level(logging.DEBUG, logger=tcpstore.logger.name)
    # Each key's wire form is then over half the 8 KiB cap.
    async with _served(store_name=f"{'s' * 4100}-{_store_name()}") as rendezvous:
        await rendezvous.publish([f"publisher/{i}" for i in range(12)], _strategy())
    lines = [r for r in caplog.records if "member/" in r.getMessage()]
    assert [r.levelno for r in lines] == [logging.WARNING]
    assert "12 in all, the longest 'member/publisher%2F10' (" in lines[0].getMessage()
    assert (
        lines[0]
        .getMessage()
        .endswith(". A key over 8192 bytes fails the publish: shorten the store name")
    )


@pytest.mark.parametrize(
    "key_bytes, level", [(4096, logging.INFO), (4097, logging.WARNING)]
)
@pytest.mark.asyncio
async def test_publisher_keys_warn_from_one_byte_over_half_the_key_cap(
    caplog, key_bytes: int, level: int
) -> None:
    """Log the publisher keys' line at INFO while the longest takes half the 8 KiB key cap and at WARNING past it."""
    caplog.set_level(logging.DEBUG, logger=tcpstore.logger.name)
    async with _served() as rendezvous:
        prefix = tcpstore._prefix(rendezvous.address)
        longest = "p" * (key_bytes - len(f"/{prefix}/member/"))
        await rendezvous.publish(["publisher/0", longest], _strategy())
    lines = [r for r in caplog.records if "per publisher" in r.getMessage()]
    assert [r.levelno for r in lines] == [level]
    assert f"'member/{longest}' ({key_bytes} key bytes)" in lines[0].getMessage()


async def _read_in_chunks(
    rendezvous: TCPStoreRendezvous, volume_ids: list[str]
) -> list[list[str]]:
    """Read ``volume_ids``, check the calls and deadline, and return the chunks' keys.

    The store calls after the ready marker's share one native call.
    """
    with _recording(rendezvous) as (calls, deadlines):
        before = time.monotonic()
        records = await rendezvous.volume_records(volume_ids)
        after = time.monotonic()
    assert records == {v: v.encode() for v in volume_ids}
    ready = [tcpstore._READY]
    assert calls[:2] == [("check", ready), ("multi_get", ready)]
    chunks = [keys for name, keys in calls[2:] if name == "multi_get"]
    # One check per chunk, all before the first multi_get.
    assert calls[2:] == [("check", c) for c in chunks] + [
        ("multi_get", c) for c in chunks
    ]
    assert [k for c in chunks for k in c] == [
        tcpstore._volume_key(v) for v in volume_ids
    ]
    # Two native calls for the ready marker, then one for all the chunks.
    assert deadlines == [deadlines[0]] * 3
    default = tcpstore._DEFAULT_TIMEOUT_S
    assert before + default <= deadlines[0] <= after + default
    return chunks


@pytest.mark.asyncio
async def test_short_ids_read_in_chunks_of_16_keys() -> None:
    """Read 40 records in one native call in 3 chunks that the key bound sets."""
    async with _served() as rendezvous:
        await _register_own_ids(rendezvous, _FORTY)
        chunks = await _read_in_chunks(rendezvous, _FORTY)
    assert [len(chunk) for chunk in chunks] == [16, 16, 8]


@pytest.mark.asyncio
async def test_long_ids_read_in_chunks_that_the_byte_bound_fills() -> None:
    """Read long ids in chunks that the byte bound fills before the key bound.

    No chunk but the last could take one more key.
    """
    long = [f"publisher/{i}/{'x' * 600}" for i in range(12)]
    async with _served() as rendezvous:
        await _register_own_ids(rendezvous, long)
        chunks = await _read_in_chunks(rendezvous, long)
        prefix = tcpstore._prefix(rendezvous.address)
    assert all(_request_size(prefix, chunk) <= 4096 for chunk in chunks)
    assert 1 < len(chunks[0]) < 16
    for chunk, after in zip(chunks, chunks[1:]):
        assert _request_size(prefix, chunk + after[:1]) > 4096


@pytest.mark.asyncio
async def test_an_empty_read_makes_no_call() -> None:
    """Answer an empty read with {} and no store or native call."""
    async with _served() as rendezvous:
        with _recording(rendezvous) as (calls, deadlines):
            assert await rendezvous.volume_records([]) == {}
    assert calls == [] and deadlines == []


@pytest.mark.asyncio
async def test_read_bounds_and_the_default_timeout_are_read_at_call_time(
    monkeypatch,
) -> None:
    """Chunk by the bounds and time out by the default set at call time. A key over the byte bound reads alone."""
    async with _served() as rendezvous:
        await _register_own_ids(rendezvous, _FORTY)
        monkeypatch.setattr(tcpstore, "_DEFAULT_TIMEOUT_S", 30.0)
        monkeypatch.setattr(tcpstore, "_READ_CHUNK_KEYS", 5)
        chunks = await _read_in_chunks(rendezvous, _FORTY)
        assert [len(chunk) for chunk in chunks] == [5] * 8
        monkeypatch.setattr(tcpstore, "_READ_CHUNK_BYTES", 100)
        chunks = await _read_in_chunks(rendezvous, _FORTY[:3])
        assert [len(chunk) for chunk in chunks] == [1] * 3


def test_chunks_fill_the_byte_budget_exactly() -> None:
    """Count 9 bytes, plus 8 and the wire key per key, up to exactly 4,096 bytes."""
    # Under prefix "p" a key's wire form is "/p/" + key: 3 more bytes, then 8 more.
    keys = ["a" * 997, "b" * 997, "c" * 997]  # 1,008 request bytes each
    assert tcpstore._read_chunks("p", [*keys, "d" * 1052]) == [[*keys, "d" * 1052]]
    assert tcpstore._read_chunks("p", [*keys, "d" * 1053]) == [keys, ["d" * 1053]]
    # A key over the budget on its own reads alone, and a key over the cap raises.
    alone = "e" * 5000
    assert tcpstore._read_chunks("p", ["a", alone, "b"]) == [["a"], [alone], ["b"]]
    with pytest.raises(ValueError, match="bytes on the wire"):
        tcpstore._read_chunks("p", ["a", "f" * 8190])


@pytest.mark.asyncio
async def test_missing_ids_are_found_per_key_only_in_failing_chunks() -> None:
    """Check each key only in chunks with a missing record, once each though an id repeats, and fetch nothing then."""
    async with _served() as rendezvous:
        await rendezvous.publish(_FORTY, _strategy())
        for rank in _FORTY:
            if rank not in _MISSING:
                await rendezvous.register_volume(rank, rank.encode())
        with _recording(rendezvous) as (calls, deadlines):
            with pytest.raises(KeyError) as raised:
                await rendezvous.volume_records([*_FORTY, *_FORTY[:21]])
    keys = [tcpstore._volume_key(rank) for rank in _FORTY]
    chunks = [keys[:16], keys[16:32], keys[32:]]
    ready = [tcpstore._READY]
    assert calls == [
        ("check", ready),
        ("multi_get", ready),
        ("check", chunks[0]),
        *[("check", [key]) for key in chunks[0]],
        ("check", chunks[1]),
        *[("check", [key]) for key in chunks[1]],
        ("check", chunks[2]),
    ]
    assert deadlines == [deadlines[0]] * 3
    address = rendezvous.address
    assert raised.value.args == (
        f"routing store {address.store_name!r} on TCPStore "
        f"{address.host}:{address.port}: no volume record for {_MISSING}",
    )


@pytest.mark.asyncio
async def test_a_read_names_the_first_16_missing_ids_and_checks_no_further() -> None:
    """Name the first 16 ids found without a record and stop checking there mid-way through a chunk."""
    async with _served() as rendezvous:
        await rendezvous.publish(_FORTY, _strategy())
        # Then the first chunk lacks 10 records and the 16th missing id is the second chunk's 6th key.
        for rank in _FORTY[:6]:
            await rendezvous.register_volume(rank, rank.encode())
        with _recording(rendezvous) as (calls, _):
            with pytest.raises(KeyError) as raised:
                await rendezvous.volume_records(_FORTY)
    keys = [tcpstore._volume_key(rank) for rank in _FORTY]
    ready = [tcpstore._READY]
    assert calls == [
        ("check", ready),
        ("multi_get", ready),
        ("check", keys[:16]),
        *[("check", [key]) for key in keys[:16]],
        ("check", keys[16:32]),
        *[("check", [key]) for key in keys[16:22]],
    ]
    address = rendezvous.address
    first = sorted(_FORTY[6:22])
    assert raised.value.args == (
        f"routing store {address.store_name!r} on TCPStore {address.host}:"
        f"{address.port}: no volume record for {first} (the first 16 found)",
    )


@pytest.mark.asyncio
async def test_a_failed_chunk_whose_records_turn_up_is_fetched() -> None:
    """Fetch a failed chunk whose missing record lands before its per-key checks."""
    ranks = sorted(_PUBLISHER_RANKS)
    chunk = [tcpstore._volume_key(rank) for rank in ranks]
    async with _served() as rendezvous:
        await rendezvous.publish(ranks, _strategy())
        await rendezvous.register_volume(ranks[0], b"early")
        store = rendezvous._prefixed
        assert store is not None
        writer = store  # a new name: type checkers do not narrow into closures

        def land() -> None:
            writer.compare_set(chunk[1], "", b"late")  # type: ignore[arg-type]

        with _recording(rendezvous, after=_at(("check", chunk), land)) as (calls, _):
            records = await rendezvous.volume_records(ranks)
    assert calls[2:] == [
        ("check", chunk),
        ("check", chunk[:1]),
        ("check", chunk[1:]),
        ("multi_get", chunk),
    ]
    assert records == {ranks[0]: b"early", ranks[1]: b"late"}


@pytest.mark.parametrize(
    "case",
    [
        "record_over_the_value_cap",
        "empty_record",
        "str_record",
        "id_over_the_key_cap",
        "id_over_the_key_cap_once_quoted",
        "id_not_utf8",
        "read_with_an_id_over_the_key_cap",
        "read_of_an_id_over_the_key_cap_once_quoted",
        "read_with_an_id_not_utf8",
    ],
)
@pytest.mark.asyncio
async def test_oversized_and_empty_records_are_rejected_before_sending(
    case: str,
) -> None:
    """Reject before sending a record or key over the libuv caps, an empty record or one that is not bytes.

    An id that UTF-8 cannot encode is rejected the same way.
    """
    async with _served() as rendezvous:
        await rendezvous.publish(_PUBLISHER_RANKS, _strategy())
        over_key = "k" * 8192  # its wire key alone is over the 8 KiB cap
        # Over the cap only once quoted ("/" is "%2F") and under the store's prefix.
        prefix = tcpstore._prefix(rendezvous.address)
        edge_key = "/" * ((8192 - len(f"/{prefix}/volume/")) // 3 + 1)
        register, read = rendezvous.register_volume, rendezvous.volume_records
        cases: dict[str, tuple[Callable[[], Awaitable[Any]], str]] = {
            "record_over_the_value_cap": (
                lambda: register("publisher/0", b"x" * (8 * 1024 * 1024 + 1)),
                "value for 'volume/publisher%2F0' has",
            ),
            "empty_record": (
                lambda: register("publisher/0", b""),
                "'publisher/0' has an empty record",
            ),
            "str_record": (
                lambda: register("publisher/0", cast(Any, "record")),
                "'publisher/0' has a str record, not bytes",
            ),
            "id_over_the_key_cap": (
                lambda: register(over_key, b"record"),
                "bytes on the wire",
            ),
            "id_over_the_key_cap_once_quoted": (
                lambda: register(edge_key, b"record"),
                "bytes on the wire",
            ),
            "id_not_utf8": (
                lambda: register("\ud800", b"record"),
                "surrogates not allowed",
            ),
            "read_with_an_id_over_the_key_cap": (
                lambda: read(["publisher/0", over_key]),
                "bytes on the wire",
            ),
            "read_of_an_id_over_the_key_cap_once_quoted": (
                lambda: read([edge_key]),
                "bytes on the wire",
            ),
            "read_with_an_id_not_utf8": (
                lambda: read(["publisher/0", "\ud800"]),
                "surrogates not allowed",
            ),
        }
        call, error = cases[case]
        refused = TypeError if case == "str_record" else ValueError
        with _recording(rendezvous) as (calls, deadlines):
            with pytest.raises(refused, match=error):
                await call()
        assert calls == [] and deadlines == []


@pytest.mark.asyncio
async def test_a_record_of_exactly_the_value_cap_round_trips() -> None:
    """Register and read back a record of exactly the libuv server's 8 MiB value cap."""
    async with _served() as rendezvous:
        await rendezvous.publish(_PUBLISHER_RANKS, _strategy())
        at_limit = b"x" * (8 * 1024 * 1024)
        await rendezvous.register_volume("publisher/0", at_limit)
        records = await rendezvous.volume_records(["publisher/0"])
        assert records == {"publisher/0": at_limit}


@pytest.mark.asyncio
async def test_another_schema_fails_both_methods_at_once() -> None:
    """Refuse a ready marker of another schema, naming both, before reading further."""
    foreign = tcpstore._SCHEMA_VERSION + b"x"
    async with _served() as rendezvous:
        await rendezvous._compare_set(tcpstore._READY, foreign, _deadline())
        error = f"schema version {foreign!r}, expected {tcpstore._SCHEMA_VERSION!r}"
        for method in _METHODS:
            with _recording(rendezvous) as (calls, _):
                with pytest.raises(RuntimeError, match=re.escape(error)):
                    await _call(rendezvous, method, timeout_s=5)
            ready = [tcpstore._READY]
            assert calls == [("check", ready), ("multi_get", ready)]


@pytest.mark.asyncio
async def test_a_schema_1_store_fails_registration_on_its_schema() -> None:
    """Refuse to register on a schema 1 store with an error naming both schemas. Such a store has no publisher keys."""
    async with _served() as rendezvous:
        await rendezvous._compare_set(tcpstore._READY, b"1", _deadline())
        error = f"schema version b'1', expected {tcpstore._SCHEMA_VERSION!r}"
        with pytest.raises(RuntimeError, match=re.escape(error)):
            await rendezvous.register_volume("publisher/0", b"record")


@pytest.mark.asyncio
async def test_ready_without_publishers_fails_registration_at_once() -> None:
    """Refuse to register on a malformed store instead of blocking in a hidden wait."""
    async with _served() as rendezvous:
        await rendezvous._compare_set(
            tcpstore._READY, tcpstore._SCHEMA_VERSION, _deadline()
        )
        # A hidden multi_get wait would time out first and name no problem.
        with _recording(rendezvous) as (calls, _):
            with pytest.raises(RuntimeError) as raised:
                await rendezvous.register_volume("publisher/0", b"record")
        ready = [tcpstore._READY]
        assert calls == [
            ("check", ready),
            ("multi_get", ready),
            ("check", ["member/publisher%2F0"]),
            ("check", [tcpstore._PUBLISHERS]),
        ]
        address = rendezvous.address
        assert str(raised.value) == (
            f"routing store {address.store_name!r} on TCPStore "
            f"{tcpstore._endpoint(address)}: marked ready without its publishers"
        )
        assert not await rendezvous._check(["volume/publisher%2F0"], _deadline())


@pytest.mark.parametrize("method", _METHODS)
@pytest.mark.asyncio
async def test_calls_before_init_fail_on_one_ready_check(method: str) -> None:
    """Fail on one ready check before init."""
    async with _served() as rendezvous:
        with _recording(rendezvous) as (calls, _):
            with pytest.raises(RuntimeError, match="init has not run"):
                await _call(rendezvous, method)
    assert calls == [("check", [tcpstore._READY])]


@pytest.mark.parametrize("method", _METHODS)
@pytest.mark.asyncio
async def test_calls_under_a_stale_nonce_fail_on_one_ready_check(method: str) -> None:
    """Fail on one ready check under the nonce of an earlier serve of the store name.

    The new serve has published by then.
    """
    store_name = _store_name()
    async with _served(store_name=store_name) as earlier:
        stale_nonce = earlier.address.nonce
    async with _served(store_name=store_name) as rendezvous:
        await rendezvous.publish(_PUBLISHER_RANKS, _strategy())
        stale = dataclasses.replace(rendezvous.address, nonce=stale_nonce)
        async with _connected(stale) as reader:
            with _recording(reader) as (calls, _):
                with pytest.raises(RuntimeError, match="init has not run"):
                    await _call(reader, method)
    assert calls == [("check", [tcpstore._READY])]


@pytest.mark.parametrize("method", _METHODS)
@pytest.mark.asyncio
async def test_calls_on_a_closed_handle_fail(method: str) -> None:
    """Refuse a call on a closed handle."""
    async with _served() as rendezvous:
        async with _connected(rendezvous.address) as reader:
            pass
        with pytest.raises(RuntimeError, match="is closed"):
            await _call(reader, method)


@pytest.mark.asyncio
async def test_records_leave_the_published_keys_unchanged() -> None:
    """Leave the publishers, strategy, coordinator and ready marker alone, even under ids named after them."""
    names = (
        tcpstore._PUBLISHERS,
        tcpstore._STRATEGY,
        tcpstore._COORDINATOR,
        tcpstore._READY,
    )
    async with _served() as rendezvous, _connected(rendezvous.address) as reader:
        await rendezvous.publish(names, _strategy(), coordinator="the coordinator")
        prefix = tcpstore._prefix(rendezvous.address)
        keys = [f"{prefix}/{name}" for name in names]
        store = rendezvous._store
        assert store is not None
        published = store.multi_get(keys)
        for name in names:
            await reader.register_volume(name, name.encode())
        assert store.multi_get(keys) == published
        records = await reader.volume_records(names)
    assert records == {name: name.encode() for name in names}


@pytest.mark.asyncio
async def test_a_record_registered_in_another_process_reads_back() -> None:
    """Read back the record a fresh interpreter registered through the store's address."""
    async with _served() as driver:
        await driver.publish(_PUBLISHER_RANKS, _strategy())
        address = json.dumps(dataclasses.asdict(driver.address))
        # Run from the tree's root, which ``-c`` puts first on the child's path.
        child = await asyncio.to_thread(
            subprocess.run,
            [sys.executable, "-c", _REGISTER_IN_CHILD, address, "host-1:29501"],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=Path(tcpstore.__file__).parents[2],
        )
        assert child.returncode == 0, child.stderr[-2000:]
        records = await driver.volume_records(["publisher/1"])
    assert records == {"publisher/1": b"host-1:29501"}


@pytest.mark.asyncio
async def test_stopped_server_times_out_a_registration_at_its_deadline(
    server,
) -> None:
    """Time out at the caller's deadline and not the default while the registration's last store call is stuck.

    Every native call of the registration shares that deadline.
    """
    process, address = server
    async with _connected(address) as rendezvous:
        await rendezvous.publish(_PUBLISHER_RANKS, _strategy())
        stop = _at(_LAST_STORE_CALL["register_volume"], lambda: _stop(process))
        with _recording(rendezvous, before=stop) as (_, deadlines):
            before = time.monotonic()
            with pytest.raises(
                TimeoutError,
                match=re.escape(
                    f"{_LAST_NATIVE_CALL['register_volume']} on TCPStore {tcpstore._endpoint(address)}"
                ),
            ):
                await _call(rendezvous, "register_volume", timeout_s=2)
            raised = time.monotonic()
    assert len(deadlines) == 4
    [deadline] = set(deadlines)
    assert before + 2 <= deadline <= raised
    assert deadline < before + 30  # the caller's 2 s and not the 60 s default


@pytest.mark.asyncio
async def test_stopped_server_times_out_a_read_in_its_per_key_checks(server) -> None:
    """Time out at the caller's deadline with a read stuck in the per-key checks that missing records cause."""
    process, address = server
    async with _connected(address) as rendezvous:
        await rendezvous.publish(_PUBLISHER_RANKS, _strategy())
        # With no record the chunk check fails and each key is checked.
        stop = _at(("check", ["volume/publisher%2F0"]), lambda: _stop(process))
        with _recording(rendezvous, before=stop) as (calls, deadlines):
            before = time.monotonic()
            with pytest.raises(
                TimeoutError,
                match=re.escape(
                    f"reading 2 volume records on TCPStore {tcpstore._endpoint(address)}"
                ),
            ):
                await rendezvous.volume_records(sorted(_PUBLISHER_RANKS), timeout_s=2)
            raised = time.monotonic()
    assert calls[-1] == ("check", ["volume/publisher%2F0"])
    [deadline] = set(deadlines)
    assert before + 2 <= deadline <= raised
    assert deadline < before + 30  # the caller's 2 s and not the 60 s default


@pytest.mark.parametrize("method", _METHODS)
@pytest.mark.asyncio
async def test_zero_timeout_fails_before_any_native_call(method: str) -> None:
    """Spend an explicit timeout_s on the whole call, so 0 fails before any native call."""
    async with _served() as rendezvous:
        await rendezvous.publish(_PUBLISHER_RANKS, _strategy())
        if method == "volume_records":
            # With any budget left, the read would succeed.
            await rendezvous.register_volume("publisher/0", b"record")
        with _recording(rendezvous) as (calls, deadlines):
            with pytest.raises(TimeoutError, match="was not started"):
                await _call(rendezvous, method, timeout_s=0)
        assert calls == [] and len(deadlines) == 1
        with _recording(rendezvous) as (calls, deadlines):
            before = time.monotonic()
            await _call(rendezvous, method, timeout_s=7)
            after = time.monotonic()
        assert calls and deadlines == [deadlines[0]] * len(deadlines)
        assert before + 7 <= deadlines[0] <= after + 7


@pytest.mark.parametrize("method", _METHODS)
@pytest.mark.asyncio
async def test_killed_server_fails_each_method_once_its_native_call_ends(
    server, method: str
) -> None:
    """Turn a server killed in a method's last store call into a ConnectionError as soon as the native call ends."""
    process, address = server
    stopped = threading.Event()

    def stop() -> None:
        _stop(process)
        stopped.set()

    async with _connected(address) as rendezvous:
        await rendezvous.publish(_PUBLISHER_RANKS, _strategy())
        await rendezvous.register_volume("publisher/0", b"record")
        with _recording(rendezvous, before=_at(_LAST_STORE_CALL[method], stop)):
            call = asyncio.create_task(_call(rendezvous, method))
            await _until(stopped.is_set)
            [native] = _native_calls(_LAST_NATIVE_CALL[method])
            assert not call.done()
            process.kill()
            await _until(lambda: not native.is_alive())
            await _done_within_loop_steps(call)
            with pytest.raises(
                ConnectionError, match=re.escape(tcpstore._endpoint(address))
            ):
                call.result()


@pytest.mark.parametrize("method", _METHODS)
@pytest.mark.asyncio
async def test_a_cancel_returns_while_a_stopped_server_holds_the_native_call(
    server, method: str
) -> None:
    """Raise CancelledError within a few loop steps of a cancel while the method's last store call stays stuck."""
    process, address = server
    stopped = threading.Event()

    def stop() -> None:
        _stop(process)
        stopped.set()

    async with _connected(address) as rendezvous:
        await rendezvous.publish(_PUBLISHER_RANKS, _strategy())
        await rendezvous.register_volume("publisher/0", b"record")
        with _recording(rendezvous, before=_at(_LAST_STORE_CALL[method], stop)):
            call = asyncio.create_task(_call(rendezvous, method))
            await _until(stopped.is_set)
            call.cancel()
            await _done_within_loop_steps(call)
            assert call.cancelled()
            # The fixture's kill ends it.
            assert _native_calls(_LAST_NATIVE_CALL[method])


async def _read_after_its_records_check(
    rendezvous: TCPStoreRendezvous, ranks: list[str], registered: bool
) -> str:
    """Publish ``ranks`` (registered if ``registered``) and name the store call after a read's check of their records.

    That is the fetch of every record, or without records a check of the first key alone.
    """
    keys = [tcpstore._volume_key(rank) for rank in ranks]
    if registered:
        await _register_own_ids(rendezvous, ranks)
        return f"multi_get({keys!r})"
    await rendezvous.publish(ranks, _strategy())
    return f"check({keys[:1]!r})"


_REGISTERED = pytest.mark.parametrize(
    "registered", [True, False], ids=["before_its_fetch", "before_a_later_check"]
)


@_REGISTERED
@pytest.mark.asyncio
async def test_a_read_stops_inside_its_native_call_at_its_deadline(
    monkeypatch, registered: bool
) -> None:
    """Start no store call in a read's native call once the deadline passes (a later check included).

    The error names that call and the server.
    """
    late: list[bool] = []
    ranks = sorted(_PUBLISHER_RANKS)
    keys = [tcpstore._volume_key(rank) for rank in ranks]
    async with _served() as rendezvous:
        refused = await _read_after_its_records_check(rendezvous, ranks, registered)
        endpoint = tcpstore._endpoint(rendezvous.address)
        what = re.escape(f"{refused} on TCPStore {endpoint}")
        with _recording(
            rendezvous, after=_at(("check", keys), lambda: late.append(True))
        ) as (calls, _):
            monkeypatch.setattr(tcpstore, "time", _Clock(late))
            with pytest.raises(
                TimeoutError, match=f"^{what} was not started: its deadline had passed$"
            ):
                await rendezvous.volume_records(ranks)
            monkeypatch.undo()
    assert calls[-1] == ("check", keys)


@_REGISTERED
@pytest.mark.asyncio
async def test_a_read_stops_inside_its_native_call_once_exit_begins(
    monkeypatch, registered: bool
) -> None:
    """Start no store call in a read's native call once exit begins (a later check included).

    The error names that call and the server.
    """

    def begin_exit() -> None:
        # As the exit hook does, after the native call has started.
        monkeypatch.setattr(tcpstore, "_exiting", True)

    ranks = sorted(_PUBLISHER_RANKS)
    keys = [tcpstore._volume_key(rank) for rank in ranks]
    async with _served() as rendezvous:
        refused = await _read_after_its_records_check(rendezvous, ranks, registered)
        endpoint = tcpstore._endpoint(rendezvous.address)
        what = re.escape(f"{refused} on TCPStore {endpoint}")
        with _recording(rendezvous, after=_at(("check", keys), begin_exit)) as (
            calls,
            _,
        ):
            with pytest.raises(
                RuntimeError, match=f"^{what} was not started: exiting$"
            ):
                await rendezvous.volume_records(ranks)
        monkeypatch.undo()
    assert calls[-1] == ("check", keys)


@_REGISTERED
@pytest.mark.asyncio
async def test_a_cancelled_read_starts_no_further_store_call(registered: bool) -> None:
    """Start no store call in a read's native call once the read is cancelled even while that native call still runs.

    Without records the call refused is a later check.
    """
    checked, release = threading.Event(), threading.Event()

    def hold() -> None:
        checked.set()
        release.wait(60)

    ranks = sorted(_PUBLISHER_RANKS)
    keys = [tcpstore._volume_key(rank) for rank in ranks]
    async with _served() as rendezvous:
        await _read_after_its_records_check(rendezvous, ranks, registered)
        with _recording(rendezvous, after=_at(("check", keys), hold)) as (calls, _):
            read = asyncio.create_task(rendezvous.volume_records(ranks))
            with _then(release.set):
                await _until(checked.is_set)
                read.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await read
            await _until(lambda: not _native_calls("reading 2 volume records"))
    assert calls[-1] == ("check", keys)


@pytest.mark.asyncio
async def test_a_read_on_a_handle_closed_after_its_ready_check_names_the_closed_connection(
    monkeypatch,
) -> None:
    """Refuse a read's store calls on a handle closed once its ready check returned with an error naming the connection."""
    ranks = sorted(_PUBLISHER_RANKS)
    async with _served() as rendezvous:
        await _register_own_ids(rendezvous, ranks)
        require_ready = rendezvous._require_ready

        async def ready_then_closed(deadline: float) -> None:
            await require_ready(deadline)
            await rendezvous.close()

        monkeypatch.setattr(rendezvous, "_require_ready", ready_then_closed)
        endpoint = re.escape(tcpstore._endpoint(rendezvous.address))
        with pytest.raises(
            RuntimeError, match=f"^connection to TCPStore {endpoint} is closed$"
        ):
            await rendezvous.volume_records(ranks)


@pytest.mark.asyncio
async def test_a_kept_read_error_holds_no_store_so_close_stops_the_server(
    monkeypatch,
) -> None:
    """Stop serving on close while an error that a read's native call raised stays referenced."""

    def failing_read(*args: Any, **kwargs: Any) -> NoReturn:
        raise LookupError("the read failed")

    ranks = sorted(_PUBLISHER_RANKS)
    async with _served() as rendezvous:
        await rendezvous.publish(ranks, _strategy())
        monkeypatch.setattr(tcpstore, "_read_records", failing_read)
        # Bound so the error and the frames its traceback holds stay referenced through the close.
        with pytest.raises(LookupError, match="the read failed") as kept:
            await rendezvous.volume_records(ranks)
        await rendezvous.close()
        with pytest.raises(ConnectionRefusedError):
            socket.create_connection((_LOCALHOST, rendezvous.address.port), 10)


def test_interface_is_the_design_and_the_rendezvous_implements_it() -> None:
    """Keep the design's two coroutines. The rendezvous adds a keyword-only timeout."""
    assert VolumeRecords.__abstractmethods__ == set(_DESIGN)
    public = {name for name in dir(VolumeRecords) if not name.startswith("_")}
    assert public == set(_DESIGN)
    assert issubclass(TCPStoreRendezvous, VolumeRecords)
    for name, signature in _DESIGN.items():
        assert inspect.iscoroutinefunction(getattr(VolumeRecords, name)), name
        assert str(inspect.signature(getattr(VolumeRecords, name))) == signature
        implemented = str(inspect.signature(getattr(TCPStoreRendezvous, name)))
        timeout = ", *, timeout_s: 'float | None' = None)"
        assert implemented == signature.replace(")", timeout, 1), name


def test_interface_loads_only_the_standard_library() -> None:
    """Import only the standard library, so the interface loads with Monarch blocked."""
    path = Path(volume_records_module.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assert _imported(ast.walk(tree)) == {"__future__", "abc", "collections.abc"}
    root = str(path.parents[2])
    child = subprocess.run(
        [sys.executable, "-c", _IMPORT_ALONE, root],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert child.returncode == 0, child.stderr[-2000:]
    loaded = ["torchstore", "torchstore.routing", "torchstore.routing.volume_records"]
    assert child.stdout.strip().splitlines()[-1] == str(loaded)
