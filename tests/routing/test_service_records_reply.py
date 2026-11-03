# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Volume records in the coordinator service's layout replies: reads, fallback, errors, and stores."""

from __future__ import annotations

import asyncio
import contextlib
import gc
import logging
import os
import pickle
import re
import socket
import threading
import time
import tracemalloc
import types
import zlib
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Collection,
    Iterable,
    Iterator,
    Mapping,
)
from typing import Any, cast, NoReturn

import pytest
import torchstore as ts
from torchstore.carrier import CarrierAddress, CarrierProtocolError, PeerLostError
from torchstore.carrier_volumes import CarrierVolumeHost, encode_volume_record
from torchstore.routing import layout_registry, service as service_module, tcpstore
from torchstore.routing._model import RankRole
from torchstore.routing.client import RoutingClient
from torchstore.routing.service import ServiceAddress, ServiceControlPlane
from torchstore.routing.tcpstore import TCPStoreRendezvous
from torchstore.state_dict_utils import put_state_dict
from torchstore.storage import StorageVolumeCore
from torchstore.strategy import _installed_volume_host

from ..test_carrier_volumes import _Volumes
from . import routed_sync_data as data, tcpstore_fixtures
from .tcpstore_fixtures import _Clock, _LOCALHOST, _Reads, _store_name, _until
from .test_record_strategy import _PUBLISHERS, _Requester, _REQUESTER_RANKS
from .test_service import _key, _layouts, _part, _register, _Served, _served
from .test_service_volumes import _timed
from .test_volume_records import _at, _recording, _store_calls

pytestmark = [
    pytest.mark.timeout(120),
    # A native-call, carrier or records thread that raises fails its test.
    pytest.mark.filterwarnings("error::pytest.PytestUnhandledThreadExceptionWarning"),
]

# The shared TCPStore fixture, which pytest finds here under this name.
_joins_its_native_calls = tcpstore_fixtures._joins_its_native_calls

# A coordinator's address, for checks that need no service behind it.
_ADDRESS = ServiceAddress("fake", _LOCALHOST, 1, "nonce")
_MARGIN = service_module._REPLY_MARGIN_S
_THREAD = "torchstore-coordinator-records"


def _publishers(count: int) -> list[str]:
    return [f"publisher/{rank}" for rank in range(count)]


def _record(volume_id: str, port: int) -> bytes:
    """A record of ``volume_id`` served at ``port`` on a host named for the port, which no test dials."""
    address = CarrierAddress(_LOCALHOST, port, volume_id, "scope", "incarnation")
    return encode_volume_record(address, f"host-{port}")


def _records(publishers: Collection[str], port: int) -> dict[str, bytes]:
    """A record for each publisher, at consecutive ports from ``port``."""
    return {
        volume_id: _record(volume_id, port + i)
        for i, volume_id in enumerate(publishers)
    }


@contextlib.contextmanager
def _counted(handle: ServiceControlPlane) -> Iterator[list[tuple[str, Any, int]]]:
    """Record each call that ``handle``'s own TCPStore connection makes while the block runs."""
    rendezvous = handle._rendezvous
    assert rendezvous is not None
    with _store_calls(rendezvous) as (_, counted, _):
        yield counted


def _record_keys(calls: list[tuple[str, Any, int]]) -> list[str]:
    """Every volume record key that the counted store reads named, in order."""
    return [
        key
        for name, keys, _ in calls
        if name in ("check", "multi_get")
        for key in keys
        if key.startswith("volume/")
    ]


def _section(payload: bytes, version: int = service_module._REPLY_VERSION) -> bytes:
    """A reply's records section holding ``payload``, with its trailer."""
    return payload + service_module._REPLY_TRAILER.pack(len(payload), version)


def _records_section(
    records: object, version: int = service_module._REPLY_VERSION
) -> bytes:
    """A records section holding ``records`` pickled and compressed, with its trailer."""
    return _section(zlib.compress(pickle.dumps(records)), version)


def _carried(section: bytes) -> dict[str, bytes]:
    """The records that a reply ending in ``section`` carries."""
    layouts = service_module._encode_layouts(
        {"publisher/0": service_module._compress_part({})}
    )
    return service_module._split_reply(_ADDRESS, layouts + section)[1]


@contextlib.asynccontextmanager
async def _store(
    publishers: Collection[str],
    records: Mapping[str, bytes] | None = None,
    store_name: str | None = None,
) -> AsyncIterator[_Served]:
    """Serve and initialize a store, register ``records`` in it, and close it all at exit."""
    async with _served(publishers, store_name=store_name or _store_name()) as store:
        for volume_id, record in (records or {}).items():
            await store.driver.register_volume(volume_id, record)
        yield store


@pytest.fixture
def coordinator_reads(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """The ids of each read that a coordinator here makes for its replies' records."""
    reads: list[list[str]] = []
    read = TCPStoreRendezvous.existing_volume_records

    async def recorded(
        rendezvous: TCPStoreRendezvous, volume_ids: Collection[str], **kwargs: Any
    ) -> dict[str, bytes]:
        reads.append(list(volume_ids))
        return await read(rendezvous, volume_ids, **kwargs)

    monkeypatch.setattr(TCPStoreRendezvous, "existing_volume_records", recorded)
    return reads


@pytest.mark.asyncio
async def test_requesters_read_no_record_from_tcpstore_once_their_reply_carries_them(
    coordinator_reads: list[list[str]],
) -> None:
    """Serve a requester's records from its first reply, with no TCPStore record read of its own.

    The coordinator reads the records once for the store, as its last publisher registers, so the reply carries
    a record registered while a lookup waited, and a later key's lookup reads nothing more.
    """
    publishers = _publishers(8)
    records = _records(publishers, 1000)
    early = {volume_id: records[volume_id] for volume_id in publishers[1:]}
    first, later = _key(), _key()
    async with _store(publishers, early) as store:
        for key in (first, later):
            await _register(store.driver, key, *publishers[1:])
        requester = await store.connect()
        with _counted(requester) as calls:
            lookup = asyncio.ensure_future(requester.get_layouts(first, timeout_s=30))
            try:
                registry = store.service._adapter.registry
                await _until(lambda: registry._layouts[first].lookups == 1)
                # As a publisher's ts.client does: its record, then its layouts.
                await store.driver.register_volume(
                    publishers[0], records[publishers[0]]
                )
                for key in (first, later):
                    await _register(store.driver, key, *publishers[:1])
                assert await asyncio.wait_for(lookup, 30) == _layouts(first, publishers)
            finally:
                lookup.cancel()
                await asyncio.gather(lookup, return_exceptions=True)
            assert await requester.get_layouts(later) == _layouts(later, publishers)
            made = len(calls)
            assert await requester.volume_records(publishers) == records
            assert calls[made:] == []
        assert _record_keys(calls) == []
    assert coordinator_reads == [publishers]


@pytest.mark.asyncio
async def test_a_requesters_carrier_host_resolves_every_volume_from_its_reply(
    caplog: pytest.LogCaptureFixture, coordinator_reads: list[list[str]]
) -> None:
    """Resolve every volume through each requester's strategy from its reply, reading no record from TCPStore.

    A later key's layouts load nothing more, so each requester logs one ``volume_records`` line.
    """
    caplog.set_level(logging.INFO)
    publishers = _publishers(4)
    records = _records(publishers, 2000)
    keys = [_key(), _key()]
    async with _store(
        publishers, records
    ) as store, contextlib.AsyncExitStack() as stack:
        for key in keys:
            await _register(store.driver, key, *publishers)
        for _ in range(2):
            handle = await store.connect()
            calls = stack.enter_context(_counted(handle))
            strategy = await handle.strategy()
            host = _installed_volume_host(strategy)
            assert isinstance(host, CarrierVolumeHost)
            stack.push_async_callback(host.close)
            for key in keys:
                await strategy._prepare_volumes(await handle.get_layouts(key))
            assert sorted(host.volume_ids()) == publishers
            for port, volume_id in enumerate(publishers, 2000):
                ref = strategy.get_storage_volume(volume_id)
                assert ref.volume_hostname == f"host-{port}"
                assert (
                    repr(ref.volume)
                    == f"VolumeProxy({volume_id!r} at {_LOCALHOST}:{port})"
                )
            assert _record_keys(calls) == []
    assert _timed(caplog).count("volume_records") == 2
    assert coordinator_reads == [publishers]


@pytest.mark.asyncio
async def test_routed_syncs_on_the_service_read_no_record_on_any_requester(
    caplog: pytest.LogCaptureFixture, coordinator_reads: list[list[str]]
) -> None:
    """Sync two served publishers into three requesters bitwise, twice, with no requester reading a record.

    Each requester logs one ``volume_records`` line, at its first layouts.
    """
    caplog.set_level(logging.INFO)
    volumes = _Volumes()
    try:
        records = {}
        for volume_id in _PUBLISHERS:
            _, _, records[volume_id] = await volumes.serve(StorageVolumeCore(volume_id))
        async with _store(
            _PUBLISHERS, records
        ) as store, contextlib.AsyncExitStack() as stack:

            async def client(
                name: str, role: RankRole
            ) -> tuple[RoutingClient, CarrierVolumeHost, list[tuple[str, Any, int]]]:
                handle = await store.connect()
                calls = stack.enter_context(_counted(handle))
                strategy = await handle.strategy()
                host = _installed_volume_host(strategy)
                assert isinstance(host, CarrierVolumeHost)
                stack.push_async_callback(host.close)
                return RoutingClient(name, role, handle, strategy), host, calls

            senders = []
            for volume_id in _PUBLISHERS:
                sender, host, _ = await client(volume_id, RankRole.PUBLISHER)
                host.adopt(volume_id, records[volume_id])
                senders.append(sender)
            readers = []
            for group, rank in _REQUESTER_RANKS:
                name = f"requester/{group}/{rank}"
                routing, _, calls = await client(name, RankRole.REQUESTER)
                readers.append((_Requester(group, rank, routing), calls))
            for round_ in range(2):
                for rank, sender in enumerate(senders):
                    state_dict = data.place(
                        data.weights(round_),
                        data.PUBLISHER_DIMS,
                        rank,
                        data.PUBLISHER_RANKS,
                    )
                    await put_state_dict(sender, state_dict, data.KEY)
                for reader, _ in readers:
                    reader.check(round_, await reader.get())
            assert [_record_keys(calls) for _, calls in readers] == [[]] * len(readers)
    finally:
        volumes.close()
    assert _timed(caplog).count("volume_records") == len(_REQUESTER_RANKS)
    assert coordinator_reads == [_PUBLISHERS]


@pytest.mark.asyncio
async def test_a_record_missing_from_the_reply_is_read_from_tcpstore_alone(
    coordinator_reads: list[list[str]],
) -> None:
    """Read only the ids a reply lacked from TCPStore, which raises KeyError while it lacks one too.

    The coordinator reads once per store, so a later key's reply still lacks the late record.
    """
    publishers = _publishers(3)
    records = _records(publishers, 3000)
    late = "publisher/2"
    first, later = _key(), _key()
    early = {
        volume_id: records[volume_id] for volume_id in publishers if volume_id != late
    }
    async with _store(publishers, early) as store:
        await _register(store.driver, first, *publishers)
        handle = await store.connect()
        assert await handle.get_layouts(first, timeout_s=0) == _layouts(
            first, publishers
        )
        served = store.service.address
        missing = (
            f"routing store {served.store_name!r} on TCPStore {served}: no volume record"
            " for ['publisher/2']"
        )
        with _counted(handle) as calls:
            with pytest.raises(KeyError, match=re.escape(missing)):
                await handle.volume_records(publishers)
        assert _record_keys(calls) == ["volume/publisher%2F2"] * 2
        await store.driver.register_volume(late, records[late])
        with _counted(handle) as calls:
            assert await handle.volume_records(publishers) == records
        assert calls == [
            ("check", [tcpstore._READY], 0),
            ("multi_get", [tcpstore._READY], len(tcpstore._SCHEMA_VERSION)),
            ("check", ["volume/publisher%2F2"], 0),
            ("multi_get", ["volume/publisher%2F2"], len(records[late])),
        ]
        await _register(store.driver, later, *publishers)
        fresh = await store.connect()
        assert await fresh.get_layouts(later, timeout_s=0) == _layouts(
            later, publishers
        )
        with _counted(fresh) as calls:
            assert await fresh.volume_records(publishers) == records
        assert _record_keys(calls) == ["volume/publisher%2F2"] * 2
    assert coordinator_reads == [publishers]


def _corrupting(
    monkeypatch: pytest.MonkeyPatch,
    corrupt: Callable[[dict[str, bytes]], dict[str, Any]],
) -> None:
    """Make the coordinator's record reads return ``corrupt`` of what TCPStore holds."""
    read = TCPStoreRendezvous.existing_volume_records

    async def corrupted(
        rendezvous: TCPStoreRendezvous, volume_ids: Collection[str], **kwargs: Any
    ) -> dict[str, Any]:
        return corrupt(await read(rendezvous, volume_ids, **kwargs))

    monkeypatch.setattr(TCPStoreRendezvous, "existing_volume_records", corrupted)


@pytest.mark.parametrize(
    ("record", "error"),
    [
        (b"not a record", r"^volume 'publisher/1' has a malformed record: "),
        (
            _record("publisher/0", 4000),
            r"^volume 'publisher/1' has a record for service 'publisher/0'$",
        ),
    ],
    ids=["not-a-record", "another-volumes-record"],
)
@pytest.mark.asyncio
async def test_a_record_in_the_reply_that_does_not_decode_fails_naming_its_volume(
    monkeypatch: pytest.MonkeyPatch, record: bytes, error: str
) -> None:
    """Raise ``CarrierProtocolError`` naming the volume whose record in the reply does not decode.

    TCPStore holds a good record for it, which the requester never reads in its place.
    """
    _corrupting(monkeypatch, lambda records: {**records, "publisher/1": record})
    publishers = _publishers(2)
    key = _key()
    async with _store(publishers, _records(publishers, 4000)) as store:
        await _register(store.driver, key, *publishers)
        handle = await store.connect()
        strategy = await handle.strategy()
        host = _installed_volume_host(strategy)
        assert isinstance(host, CarrierVolumeHost)
        try:
            with _counted(handle) as calls:
                layouts = await handle.get_layouts(key)
                with pytest.raises(CarrierProtocolError, match=error):
                    await strategy._prepare_volumes(layouts)
            assert _record_keys(calls) == []
        finally:
            await host.close()


@pytest.mark.asyncio
async def test_a_reply_with_a_malformed_records_section_fails_naming_the_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail a lookup whose reply holds a record that is not bytes, naming the store and the coordinator's address."""
    monkeypatch.setattr(
        service_module,
        "_encode_records",
        lambda records: _records_section({**records, "publisher/1": "r"}),
    )
    publishers = _publishers(1)
    key = _key()
    async with _store(publishers) as store:
        await _register(store.driver, key, *publishers)
        handle = await store.connect()
        served = store.service.address
        expected = (
            f"routing store {served.store_name!r} on {served}: the coordinator's reply holds"
            " a malformed record for volume 'publisher/1'"
        )
        with pytest.raises(CarrierProtocolError, match=f"^{re.escape(expected)}$"):
            await handle.get_layouts(key, timeout_s=0)


def test_a_reply_takes_its_records_without_a_second_copy_of_its_layouts() -> None:
    """Encode a reply and its records within 1.5 times its size, as appending them copies nothing.

    Parts make up nearly all of the reply, so a copy made to append the records doubles the peak.
    """
    if tracemalloc.is_tracing():
        pytest.skip(
            "allocation tracing is already on: this case's peak would hold earlier allocations"
            " and its stop would end that tracing"
        )
    parts = {f"publisher/{rank}": (os.urandom(1 << 20), 1 << 20) for rank in range(16)}
    tail = service_module._encode_records(
        {"publisher/0": _record("publisher/0", 12000)}
    )
    tracemalloc.start()
    try:
        base, _ = tracemalloc.get_traced_memory()
        reply = service_module._encode_layouts(parts, tail=tail)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert reply.endswith(tail)
    assert peak - base < 1.5 * len(reply), (peak - base, len(reply))


@pytest.mark.asyncio
async def test_each_store_replies_with_its_own_records_also_once_initialized_again(
    coordinator_reads: list[list[str]],
) -> None:
    """Reply with each store's own records, though both stores' publishers have the same ids.

    A store started again under its name, as a later ``ts.initialize`` starts it, replies with its new records.
    """
    publishers = _publishers(2)
    records = [_records(publishers, port) for port in (6000, 6010, 6020)]
    key, store_name = _key(), _store_name()
    async with _store(publishers, records[0], store_name) as first, _store(
        publishers, records[1]
    ) as other:
        handles = []
        for store in (first, other):
            await _register(store.driver, key, *publishers)
            handles.append(await store.connect())
            assert await handles[-1].get_layouts(key) == _layouts(key, publishers)
        for handle, expected in zip(handles, records):
            with _counted(handle) as calls:
                assert await handle.volume_records(publishers) == expected
            assert calls == []
    async with _store(publishers, records[2], store_name) as again:
        await _register(again.driver, key, *publishers)
        handle = await again.connect()
        assert await handle.get_layouts(key) == _layouts(key, publishers)
        with _counted(handle) as calls:
            assert await handle.volume_records(publishers) == records[2]
        assert calls == []
    assert coordinator_reads == [publishers] * 3


_LAYOUTS = service_module._encode_layouts(
    {"publisher/0": service_module._compress_part({"w": 1})}
)
_OTHER_VERSION = (
    re.escape(
        "does not end in a version 1 trailer. The driver may run another TorchStore version:"
        " run one TorchStore version in the driver and every rank"
    )
    + "$"
)


@pytest.mark.parametrize(
    ("reply", "problem"),
    [
        (b"abc", _OTHER_VERSION),
        (_LAYOUTS, _OTHER_VERSION),
        (_LAYOUTS + _records_section({}, version=2), _OTHER_VERSION),
        (
            _LAYOUTS
            + service_module._REPLY_TRAILER.pack(
                1 << 30, service_module._REPLY_VERSION
            ),
            r"names 1073741824 bytes of volume records, which leaves no layouts in the"
            rf" {len(_LAYOUTS)} bytes before it$",
        ),
        (
            _records_section({"publisher/0": b"r"}),
            r"bytes of volume records, which leaves no layouts in the \d+ bytes before it$",
        ),
        (_LAYOUTS + _section(b"not zlib"), r"holds malformed volume records: error\("),
        (
            _LAYOUTS + _section(zlib.compress(b"not a pickle")),
            r"holds malformed volume records: UnpicklingError\(",
        ),
        (
            _LAYOUTS + _records_section([("publisher/0", b"r")]),
            r"holds malformed volume records$",
        ),
        (
            _LAYOUTS + _records_section({7: b"r"}),
            r"holds a malformed record for volume 7$",
        ),
        (
            _LAYOUTS + _records_section({"publisher/0": b""}),
            r"for volume 'publisher/0'$",
        ),
    ],
    ids=[
        "shorter-than-a-trailer",
        "untailed",
        "version-2",
        "oversized",
        "no-layouts",
        "not-zlib",
        "not-a-pickle",
        "list",
        "int-id",
        "empty",
    ],
)
def test_a_reply_of_another_format_raises_naming_the_store(
    reply: bytes, problem: str
) -> None:
    """Refuse a reply whose trailer or records are not this version's, as a parent-commit driver's are."""
    with pytest.raises(CarrierProtocolError, match=problem) as raised:
        service_module._split_reply(_ADDRESS, reply)
    prefix = f"routing store 'fake' on {_LOCALHOST}:1: the coordinator's reply "
    assert str(raised.value).startswith(prefix), raised.value


@pytest.mark.asyncio
async def test_a_failed_records_read_leaves_requesters_reading_tcpstore(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Reply without records when the coordinator's read fails, so requesters read the records from TCPStore.

    The store's later keys do not read again, and one warning names the store, the count and the error.
    """
    attempts: list[list[str]] = []

    async def failing(
        rendezvous: TCPStoreRendezvous, volume_ids: Collection[str], **kwargs: Any
    ) -> dict[str, bytes]:
        attempts.append(list(volume_ids))
        raise ConnectionError("TCPStore unreachable")

    monkeypatch.setattr(TCPStoreRendezvous, "existing_volume_records", failing)
    caplog.set_level(logging.INFO, logger=service_module.__name__)
    publishers = _publishers(3)
    records = _records(publishers, 7000)
    keys = [_key(), _key()]
    async with _store(publishers, records) as store:
        handle = await store.connect()
        for key in keys:
            await _register(store.driver, key, *publishers)
            assert await handle.get_layouts(key, timeout_s=0) == _layouts(
                key, publishers
            )
        with _counted(handle) as calls:
            assert await handle.volume_records(publishers) == records
        record_keys = [f"volume/publisher%2F{rank}" for rank in range(3)]
        assert _record_keys(calls) == record_keys * 2
        store_name = store.service.address.store_name
    assert attempts == [publishers]
    logged = [
        (record.levelno, record.getMessage())
        for record in caplog.records
        if record.name == service_module.__name__
    ]
    assert logged == [
        (
            logging.WARNING,
            f"routing store {store_name!r}: replies carry 0 of 3 volume records, as a read"
            " failed: ConnectionError('TCPStore unreachable')",
        )
    ]


def _stall_record_checks(
    store: _Served, reading: threading.Event, release: threading.Event
) -> None:
    """Make the coordinator's record checks wait for ``release``, keeping no reference to its TCPStore here."""
    coordinator = store.service._rendezvous
    assert coordinator is not None and coordinator._prefixed is not None
    native = coordinator._prefixed

    def check(keys: list[str]) -> bool:
        if any(name.startswith("volume/") for name in keys):
            reading.set()
            release.wait(60)
        return native.check(keys)

    stuck = types.SimpleNamespace(check=check, multi_get=native.multi_get)
    coordinator._prefixed = cast(Any, stuck)


@pytest.mark.asyncio
async def test_a_records_read_past_half_the_reply_margin_leaves_the_reply_without_records(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Reply without records once the coordinator's read outlasts its pass's bound of half a lookup's reply margin.

    One warning names the timeout, and the requester reads TCPStore.
    """
    monkeypatch.setattr(service_module, "_REPLY_MARGIN_S", 4.0)
    # Frozen, so the pass's bound is exactly half the margin. The TCPStore call's deadline keeps real time.
    monkeypatch.setattr(service_module, "time", _Clock())
    bounds: list[Any] = []
    read = TCPStoreRendezvous.existing_volume_records

    async def bounded(
        rendezvous: TCPStoreRendezvous, volume_ids: Collection[str], **kwargs: Any
    ) -> dict[str, bytes]:
        bounds.append(kwargs.get("timeout_s"))
        return await read(rendezvous, volume_ids, **kwargs)

    monkeypatch.setattr(TCPStoreRendezvous, "existing_volume_records", bounded)
    caplog.set_level(logging.INFO, logger=service_module.__name__)
    reading, release = threading.Event(), threading.Event()
    publishers = _publishers(2)
    records = _records(publishers, 11000)
    key = _key()
    async with _store(publishers, records) as store:
        handle = await store.connect()
        coordinator = store.service._rendezvous
        assert coordinator is not None
        native = coordinator._prefixed
        try:
            _stall_record_checks(store, reading, release)
            # The last publisher's first layout starts the read.
            await _register(store.driver, key, *publishers)
            # A wait that keeps the lookup's own deadline far from the read's.
            layouts = await handle.get_layouts(key, timeout_s=30)
        finally:
            release.set()
            coordinator._prefixed = native
        assert layouts == _layouts(key, publishers)
        with _counted(handle) as calls:
            assert await handle.volume_records(publishers) == records
        record_keys = ["volume/publisher%2F0", "volume/publisher%2F1"]
        assert _record_keys(calls) == record_keys * 2
        store_name = store.service.address.store_name
    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.name == service_module.__name__ and record.levelno == logging.WARNING
    ]
    expected = (
        f"routing store {store_name!r}: replies carry 0 of 2 volume records, as a read"
        " failed: TimeoutError("
    )
    assert len(warnings) == 1 and warnings[0].startswith(expected), warnings
    assert "did not finish within" in warnings[0], warnings
    assert bounds == [service_module._REPLY_MARGIN_S / 2]


@pytest.mark.asyncio
async def test_a_lookup_cancelled_while_it_reads_records_leaves_the_read_to_the_next(
    coordinator_reads: list[list[str]],
) -> None:
    """Leave the read running when the only lookup awaiting it is cancelled, for the next lookup to await.

    That lookup's reply carries the records of the one read, and its requester reads none from TCPStore.
    """
    reading, release = threading.Event(), threading.Event()
    publishers = _publishers(2)
    records = _records(publishers, 8000)
    key = _key()
    async with _store(publishers, records) as store:
        lookups: list[asyncio.Future[Any]] = []
        try:
            _stall_record_checks(store, reading, release)
            await _register(store.driver, key, *publishers)
            # The last publisher's first layout started the read.
            assert await asyncio.to_thread(reading.wait, 30)
            cancelled, waiting = await store.connect(), await store.connect()
            server = store.server
            lookups.append(
                asyncio.ensure_future(cancelled.get_layouts(key, timeout_s=0))
            )
            await _until(lambda: server.stats()["running"] == 1)
            lookups[0].cancel()
            with pytest.raises(asyncio.CancelledError):
                await lookups[0]
            await _until(lambda: server.stats()["running"] == 0)
            assert not store.service._adapter._records._section.done()
            lookups.append(asyncio.ensure_future(waiting.get_layouts(key, timeout_s=0)))
            await _until(lambda: server.stats()["running"] == 1)
            release.set()
            assert await asyncio.wait_for(lookups[1], 30) == _layouts(key, publishers)
        finally:
            release.set()
            for lookup in lookups:
                lookup.cancel()
            await asyncio.gather(*lookups, return_exceptions=True)
        with _counted(waiting) as calls:
            assert await waiting.volume_records(publishers) == records
        assert calls == []
    assert coordinator_reads == [publishers]


def _answers(port: int) -> bool:
    """Whether a TCPStore still listens at ``port`` on this host."""
    try:
        socket.create_connection((_LOCALHOST, port), timeout=1).close()
    except OSError:
        return False
    return True


@pytest.mark.parametrize("waiting", ["lookup_waiting", "lookup_cancelled"])
@pytest.mark.asyncio
async def test_a_service_closing_during_its_records_read_stops_the_read_and_its_tcpstore(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, waiting: str
) -> None:
    """Stop the store's records read as its service closes, before its server's grace once no lookup awaits it.

    A lookup still waiting at the grace's end fails as the service closes. Once the read's store call returns,
    nothing holds the TCPStore.
    """
    monkeypatch.setattr(service_module, "_CLOSE_GRACE_S", 1.0)
    reading, release = threading.Event(), threading.Event()
    publishers = _publishers(2)
    key = _key()
    lookup: asyncio.Future[Any] | None = None
    try:
        async with _store(publishers, _records(publishers, 13000)) as store:
            _stall_record_checks(store, reading, release)
            await _register(store.driver, key, *publishers)
            assert await asyncio.to_thread(reading.wait, 30)
            records, server = store.service._adapter._records, store.server
            handle = await store.connect()
            lookup = asyncio.ensure_future(handle.get_layouts(key, timeout_s=0))
            await _until(lambda: records._waiting == 1)
            if waiting == "lookup_cancelled":
                lookup.cancel()
                await _until(lambda: server.stats()["running"] == 0)
            port = store.service.address.port
            stopped_at_grace: list[bool] = []
            close = server.close

            async def closing(*args: Any, **kwargs: Any) -> None:
                stopped_at_grace.append(records._section.done())
                await close(*args, **kwargs)

            monkeypatch.setattr(server, "close", closing)
            await store.service.close()
            assert stopped_at_grace == [waiting == "lookup_cancelled"]
            # The read waits for the release below, so only the close's stop can have set the section.
            assert _carried(records._section.result(timeout=10)) == {}
            release.set()
            enabled = gc.isenabled()
            gc.disable()
            try:
                await _until(lambda: not _answers(port), timeout_s=10)
            finally:
                if enabled:
                    gc.enable()
            if waiting == "lookup_cancelled":
                with pytest.raises(asyncio.CancelledError):
                    await lookup
            else:
                with pytest.raises(
                    PeerLostError,
                    match="coordinator call failed: the coordinator service closed",
                ):
                    await lookup
    finally:
        release.set()
        if lookup is not None:
            lookup.cancel()
            await asyncio.gather(lookup, return_exceptions=True)
    gc.collect()
    assert not [r for r in caplog.records if "Task was destroyed" in r.getMessage()]


class _Sleeps(threading.Condition):
    """A condition that counts how often its waiter goes to sleep and how often it is notified.

    So a test can tell that the passes' thread is idle, and that registrations left it asleep.
    """

    def __init__(self, lock: threading.Lock) -> None:
        super().__init__(lock)
        self.count = 0
        self.notified = 0

    def wait(self, timeout: float | None = None) -> bool:
        self.count += 1
        return super().wait(timeout)

    def notify(self, n: int = 1) -> None:
        self.notified += 1
        super().notify(n)


def _held(ranks: Iterable[str]) -> dict[str, bytes]:
    return {rank: rank.encode() for rank in ranks}


_MakeRecords = Callable[
    [Any, Collection[str]], tuple[service_module._ReplyRecords, _Sleeps]
]


def _force_stop(records: service_module._ReplyRecords) -> None:
    """Stop ``records``' reads whatever waits for them, so that no case leaves the passes' thread behind."""
    with records._lock:
        records._stop()


@pytest.fixture
def reply_records() -> Iterator[_MakeRecords]:
    """Make a store's records with a sleep counter, and stop each one's reads at teardown on every outcome.

    Teardown then checks that each case released every hold it made.
    """
    made: list[tuple[service_module._ReplyRecords, _Reads]] = []

    def make(
        rendezvous: _Reads, publishers: Collection[str]
    ) -> tuple[service_module._ReplyRecords, _Sleeps]:
        records = service_module._ReplyRecords(cast(Any, rendezvous), publishers)
        sleeps = _Sleeps(records._lock)
        records._wake = sleeps
        made.append((records, rendezvous))
        return records, sleeps

    yield make
    held = [reads.unreleased_holds() for _, reads in made]
    for records, _ in made:
        _force_stop(records)
    assert held == [[]] * len(made)
    assert [reads.unreleased for _, reads in made] == [[]] * len(made)


def _register_with_records(
    records: service_module._ReplyRecords, rendezvous: _Reads, ranks: Iterable[str]
) -> None:
    """Register each rank as a publisher's ts.client does: its record, then its first layout."""
    for rank in ranks:
        rendezvous.held[rank] = rank.encode()
        records.registered(rank)


async def _settled(records: service_module._ReplyRecords, sleeps: _Sleeps) -> None:
    """Return once the passes' thread has looked for a due pass again and gone back to sleep."""
    with records._lock:
        count = sleeps.count
        sleeps.notify()
    await _until(lambda: sleeps.count > count, timeout_s=10)


def _records_threads(before: Collection[threading.Thread]) -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name == _THREAD and t not in before]


@pytest.mark.asyncio
async def test_the_first_pass_reads_every_record_at_the_16th_registration(
    reply_records: _MakeRecords,
) -> None:
    """Read every record at the 16th registration, so a store whose records all came first has its section by then.

    The registrations after it read nothing more.
    """
    publishers = _publishers(40)
    rendezvous = _Reads(_held(publishers))
    records, _ = reply_records(rendezvous, publishers)
    for rank in publishers[:15]:
        records.registered(rank)
    assert not records._started
    records.registered(publishers[15])
    await _until(records._section.done)
    assert rendezvous.reads == [sorted(publishers)]
    for rank in publishers[16:]:
        records.registered(rank)
    assert _carried(await records.section()) == _held(publishers)
    assert rendezvous.reads == [sorted(publishers)]


@pytest.mark.asyncio
async def test_later_passes_start_at_16_lacking_records_and_at_the_last_registration(
    reply_records: _MakeRecords,
) -> None:
    """Read registered publishers still lacking a record once 16 do, and the rest at the last registration.

    Fifteen lacking ones leave the passes' thread asleep, and the 16th wakes it once.
    """
    publishers = _publishers(40)
    rendezvous = _Reads()
    records, sleeps = reply_records(rendezvous, publishers)
    _register_with_records(records, rendezvous, publishers[:16])
    await _settled(records, sleeps)
    assert rendezvous.reads == [sorted(publishers)]
    notified = sleeps.notified
    _register_with_records(records, rendezvous, publishers[16:31])
    assert sleeps.notified == notified
    await _settled(records, sleeps)
    assert len(rendezvous.reads) == 1
    notified = sleeps.notified
    _register_with_records(records, rendezvous, publishers[31:32])
    assert sleeps.notified == notified + 1
    await _settled(records, sleeps)
    assert rendezvous.reads[1:] == [publishers[16:32]]
    _register_with_records(records, rendezvous, publishers[32:])
    assert _carried(await asyncio.wait_for(records.section(), 30)) == _held(publishers)
    assert rendezvous.reads[2:] == [publishers[32:]]


@pytest.mark.asyncio
async def test_registrations_during_a_pass_are_read_by_the_next_one(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    reply_records: _MakeRecords,
) -> None:
    """Read the records of publishers that register while a later pass runs, also the last ones, in the next pass.

    The INFO line counts every pass's time.
    """
    clock = _Clock()
    monkeypatch.setattr(service_module, "time", clock)
    caplog.set_level(logging.INFO, logger=service_module.__name__)
    publishers = _publishers(40)
    rendezvous = _Reads(holds={1}, clock=clock)
    records, sleeps = reply_records(rendezvous, publishers)
    try:
        _register_with_records(records, rendezvous, publishers[:16])
        await _settled(records, sleeps)
        _register_with_records(records, rendezvous, publishers[16:32])
        await _until(lambda: len(rendezvous.reads) == 2)
        _register_with_records(records, rendezvous, publishers[32:])
    finally:
        rendezvous.release(1)
    assert _carried(await asyncio.wait_for(records.section(), 30)) == _held(publishers)
    assert rendezvous.reads[1:] == [publishers[16:32], publishers[32:]]
    logged = [
        (record.levelno, record.getMessage())
        for record in caplog.records
        if record.name == service_module.__name__
    ]
    assert logged == [
        (
            logging.INFO,
            "routing store 'reads': replies carry 40 of 40 volume records, read in 4.500s",
        )
    ]


@pytest.mark.asyncio
async def test_15_registrations_during_a_pass_wait_for_a_16th_before_another_pass(
    reply_records: _MakeRecords,
) -> None:
    """Start no pass after one during which 15 publishers registered, until a 16th lacks a record."""
    publishers = _publishers(40)
    rendezvous = _Reads(holds={0})
    records, sleeps = reply_records(rendezvous, publishers)
    try:
        _register_with_records(records, rendezvous, publishers[:16])
        await _until(lambda: rendezvous.reads)
        _register_with_records(records, rendezvous, publishers[16:31])
    finally:
        rendezvous.release(0)
    await _settled(records, sleeps)
    assert rendezvous.reads == [sorted(publishers)]
    _register_with_records(records, rendezvous, publishers[31:32])
    await _settled(records, sleeps)
    assert rendezvous.reads[1:] == [publishers[16:32]]


@pytest.mark.parametrize("registering", ["after_that_pass", "during_that_pass"])
@pytest.mark.asyncio
async def test_a_publisher_whose_record_a_pass_found_is_never_read_again(
    reply_records: _MakeRecords, registering: str
) -> None:
    """Leave publishers whose records an earlier pass found out of later passes, which read only lacking ones.

    That holds whether they register after that pass or while it runs.
    """
    publishers = _publishers(40)
    during = registering == "during_that_pass"
    rendezvous = _Reads(_held(publishers[:30]), holds={0} if during else ())
    records, sleeps = reply_records(rendezvous, publishers)
    # A lookup in a store this large starts the reads before any registration.
    records.wanted()
    if during:
        try:
            await _until(lambda: rendezvous.reads)
            _register_with_records(records, rendezvous, publishers[:30])
        finally:
            rendezvous.release(0)
    else:
        await _settled(records, sleeps)
        _register_with_records(records, rendezvous, publishers[:30])
    await _settled(records, sleeps)
    assert rendezvous.reads == [sorted(publishers)]
    _register_with_records(records, rendezvous, publishers[30:])
    assert _carried(await asyncio.wait_for(records.section(), 30)) == _held(publishers)
    assert rendezvous.reads == [sorted(publishers), publishers[30:]]


@pytest.mark.asyncio
async def test_a_record_that_tcpstore_lacks_ends_the_reads_without_another_pass(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    reply_records: _MakeRecords,
) -> None:
    """End the reads once every publisher registered and none is left to ask for, though a record is missing.

    One INFO line gives the count and the passes' time.
    """
    clock = _Clock()
    monkeypatch.setattr(service_module, "time", clock)
    caplog.set_level(logging.INFO, logger=service_module.__name__)
    publishers = _publishers(3)
    rendezvous = _Reads(_held(publishers[:2]), clock=clock)
    records, _ = reply_records(rendezvous, publishers)
    for rank in publishers:
        records.registered(rank)
    assert _carried(await asyncio.wait_for(records.section(), 30)) == _held(
        publishers[:2]
    )
    assert rendezvous.reads == [publishers]
    logged = [
        (record.levelno, record.getMessage())
        for record in caplog.records
        if record.name == service_module.__name__
    ]
    assert logged == [
        (
            logging.INFO,
            "routing store 'reads': replies carry 2 of 3 volume records, read in 1.500s",
        )
    ]


_REFUSED_AT_EXIT = "check([...]) on TCPStore was not started: exiting"


@pytest.mark.parametrize(
    ("failing", "warned"),
    [
        (None, "ConnectionError('TCPStore unreachable')"),
        (RuntimeError, f"RuntimeError({_REFUSED_AT_EXIT!r})"),
        (tcpstore.ExitingError, None),
    ],
    ids=["unreachable", "plain_runtime_error", "exiting"],
)
@pytest.mark.asyncio
async def test_a_failed_later_pass_keeps_the_records_earlier_passes_found(
    caplog: pytest.LogCaptureFixture,
    reply_records: _MakeRecords,
    failing: type[RuntimeError] | None,
    warned: str | None,
) -> None:
    """Reply with the records found before a failed pass, and read nothing after it.

    One warning says so unless the TCPStore client refused the read as the driver exits. Only its ``ExitingError``
    tells that: a plain ``RuntimeError`` with the same text still warns.
    """
    caplog.set_level(logging.INFO, logger=service_module.__name__)
    publishers = _publishers(40)
    failure = None if failing is None else failing(_REFUSED_AT_EXIT)
    rendezvous = _Reads(fails={1}, failure=failure)
    records, sleeps = reply_records(rendezvous, publishers)
    _register_with_records(records, rendezvous, publishers[:16])
    await _settled(records, sleeps)
    _register_with_records(records, rendezvous, publishers[16:32])
    await _until(records._section.done)
    _register_with_records(records, rendezvous, publishers[32:])
    assert _carried(await records.section()) == _held(publishers[:16])
    assert rendezvous.reads == [sorted(publishers), publishers[16:32]]
    logged = [
        (record.levelno, record.getMessage())
        for record in caplog.records
        if record.name == service_module.__name__
    ]
    warning = (
        logging.WARNING,
        f"routing store 'reads': replies carry 16 of 40 volume records, as a read failed: {warned}",
    )
    assert logged == ([] if warned is None else [warning])


@pytest.mark.parametrize("begins", ["before_the_pass", "inside_its_native_call"])
@pytest.mark.asyncio
async def test_a_records_read_the_tcpstore_client_refuses_at_exit_logs_nothing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, begins: str
) -> None:
    """Log nothing and reply with no records when the TCPStore client refuses a pass's store call as exit begins.

    It refuses the ready check if exit began before the pass and the fetch if exit began inside the pass's native call.
    """
    caplog.set_level(logging.INFO, logger=service_module.__name__)
    monkeypatch.setattr(tcpstore, "_exiting", False)
    publishers = _publishers(2)
    keys = [tcpstore._volume_key(rank) for rank in publishers]
    inside = begins == "inside_its_native_call"

    def begin_exit() -> None:
        # As the exit hook does.
        monkeypatch.setattr(tcpstore, "_exiting", True)

    async with tcpstore_fixtures._served() as rendezvous:
        await rendezvous.publish(publishers, ts.TorchStoreStrategy())
        for volume_id, record in _records(publishers, 14000).items():
            await rendezvous.register_volume(volume_id, record)
        records = service_module._ReplyRecords(rendezvous, publishers)
        hook = _at(("check", keys), begin_exit) if inside else None
        try:
            with _store_calls(rendezvous, after=hook) as (calls, _, _):
                if not inside:
                    begin_exit()
                for rank in publishers:
                    records.registered(rank)
                section = await asyncio.wait_for(records.section(), 30)
        finally:
            _force_stop(records)
            monkeypatch.setattr(tcpstore, "_exiting", False)
    assert _carried(section) == {}
    ready = [("check", [tcpstore._READY]), ("multi_get", [tcpstore._READY])]
    assert calls == ([*ready, ("check", keys)] if inside else [])
    assert not [r for r in caplog.records if r.name == service_module.__name__]


@pytest.mark.asyncio
async def test_passes_get_half_a_reply_margin_counted_from_the_last_registration_once_it_came(
    monkeypatch: pytest.MonkeyPatch,
    reply_records: _MakeRecords,
) -> None:
    """Bound passes by half a reply margin, and those after the last registration by what is left of it.

    Earlier registrations and a later key's registration move that bound in no way.
    """
    clock = _Clock(1000.0)
    monkeypatch.setattr(service_module, "time", clock)
    publishers = _publishers(40)
    rendezvous = _Reads(_held(publishers[:16]), holds={0, 1})
    records, _ = reply_records(rendezvous, publishers)
    try:
        _register_with_records(records, rendezvous, publishers[:16])
        await _until(lambda: len(rendezvous.reads) == 1)
        clock.now = 1005.0
        _register_with_records(records, rendezvous, publishers[16:32])
        clock.now = 1009.0
        rendezvous.release(0)
        await _until(lambda: len(rendezvous.reads) == 2)
        clock.now = 1020.0
        _register_with_records(records, rendezvous, publishers[32:])
        clock.now = 1021.0
        records.registered(publishers[5])  # its layout for another key
        clock.now = 1023.0
    finally:
        for n in (0, 1):
            if not rendezvous._holds[n].done():
                rendezvous.release(n)
    assert _carried(await asyncio.wait_for(records.section(), 30)) == _held(publishers)
    assert rendezvous.reads[1:] == [publishers[16:32], publishers[32:]]
    assert rendezvous.bounds == [_MARGIN / 2, _MARGIN / 2, _MARGIN / 2 - 3.0]


@pytest.mark.asyncio
async def test_waiting_lookups_start_one_first_pass_in_a_store_of_16_or_more(
    reply_records: _MakeRecords,
) -> None:
    """Start one first pass for any number of waiting lookups in a store of 16, before its registrations."""
    publishers = _publishers(16)
    rendezvous = _Reads(_held(publishers), holds={0})
    records, _ = reply_records(rendezvous, publishers)
    before = set(threading.enumerate())
    try:
        records.wanted()
        records.wanted()
        assert len(_records_threads(before)) == 1
        await _until(lambda: rendezvous.reads)
    finally:
        rendezvous.release(0)
    assert _carried(await asyncio.wait_for(records.section(), 30)) == _held(publishers)
    assert rendezvous.reads == [sorted(publishers)]


@pytest.mark.asyncio
async def test_a_waiting_lookup_starts_nothing_in_a_store_of_15(
    reply_records: _MakeRecords,
) -> None:
    """Leave a smaller store's reads to its last registration, though a lookup waits."""
    publishers = _publishers(15)
    rendezvous = _Reads(_held(publishers))
    records, _ = reply_records(rendezvous, publishers)
    records.wanted()
    for rank in publishers[:14]:
        records.registered(rank)
    assert not records._started
    records.registered(publishers[14])
    assert _carried(await asyncio.wait_for(records.section(), 30)) == _held(publishers)
    assert rendezvous.reads == [sorted(publishers)]


@pytest.mark.asyncio
async def test_reads_that_no_thread_can_make_leave_replies_without_records(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Reply without records, logging one warning, when no thread can start for the reads."""

    def no_thread(name: str, target: Callable[[], object]) -> NoReturn:
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(service_module, "_start_thread", no_thread)
    caplog.set_level(logging.INFO, logger=service_module.__name__)
    publishers = _publishers(2)
    rendezvous = _Reads()
    records = service_module._ReplyRecords(cast(Any, rendezvous), publishers)
    for rank in publishers:
        records.registered(rank)
    assert _carried(await asyncio.wait_for(records.section(), 10)) == {}
    assert rendezvous.reads == []
    logged = [
        (record.levelno, record.getMessage())
        for record in caplog.records
        if record.name == service_module.__name__
    ]
    assert logged == [
        (
            logging.WARNING,
            "routing store 'reads': replies carry no volume records, as no thread could"
            ' read them: RuntimeError("can\'t start new thread")',
        )
    ]


@pytest.mark.asyncio
async def test_a_close_with_no_lookup_waiting_stops_the_reads_and_cancels_the_pass_in_flight(
    caplog: pytest.LogCaptureFixture,
    reply_records: _MakeRecords,
) -> None:
    """Set an empty section at once as the service closes, cancelling the pass in flight, whose thread then ends.

    Nothing is logged about the reads.
    """
    caplog.set_level(logging.INFO, logger=service_module.__name__)
    publishers = _publishers(2)
    rendezvous = _Reads(holds={0})
    records, _ = reply_records(rendezvous, publishers)
    before = set(threading.enumerate())
    try:
        for rank in publishers:
            records.registered(rank)
        await _until(lambda: rendezvous.reads)
        (thread,) = _records_threads(before)
        records.cancel()
        assert _carried(records._section.result(timeout=0)) == {}
        thread.join(10)
        assert not thread.is_alive()
        assert rendezvous.cancelled == [0]
    finally:
        if not rendezvous._holds[0].done():
            rendezvous.release(0)
    assert not [r for r in caplog.records if r.name == service_module.__name__]


@pytest.mark.asyncio
async def test_a_close_between_passes_stops_the_reads_and_their_thread(
    reply_records: _MakeRecords,
) -> None:
    """End the passes' thread as the service closes between passes, keeping the records found and reading no more.

    Exit never joins that daemon thread. A store call it has in flight still holds exit for up to 5s.
    """
    publishers = _publishers(40)
    rendezvous = _Reads(_held(publishers[:16]))
    records, sleeps = reply_records(rendezvous, publishers)
    before = set(threading.enumerate())
    for rank in publishers[:16]:
        records.registered(rank)
    await _settled(records, sleeps)
    (thread,) = _records_threads(before)
    assert thread.daemon
    records.cancel()
    thread.join(10)
    assert not thread.is_alive()
    _register_with_records(records, rendezvous, publishers[16:])
    assert _carried(await records.section()) == _held(publishers[:16])
    assert rendezvous.reads == [sorted(publishers)]


@pytest.mark.asyncio
async def test_a_close_lets_lookups_waiting_for_the_reads_leave_with_every_record(
    reply_records: _MakeRecords,
) -> None:
    """Keep the reads going through a close while lookups wait, so one that stays gets every record of the one read."""
    publishers = _publishers(2)
    rendezvous = _Reads(_held(publishers), holds={0})
    records, _ = reply_records(rendezvous, publishers)
    lookups: list[asyncio.Future[bytes]] = []
    try:
        for rank in publishers:
            records.registered(rank)
        await _until(lambda: rendezvous.reads)
        lookups = [asyncio.ensure_future(records.section()) for _ in range(2)]
        await _until(lambda: records._waiting == 2)
        records.cancel()
        lookups[0].cancel()
        await asyncio.gather(lookups[0], return_exceptions=True)
        assert not records._section.done()
        rendezvous.release(0)
        assert _carried(await asyncio.wait_for(lookups[1], 30)) == _held(publishers)
        assert rendezvous.cancelled == []
    finally:
        if not rendezvous._holds[0].done():
            rendezvous.release(0)
        for lookup in lookups:
            lookup.cancel()
        await asyncio.gather(*lookups, return_exceptions=True)


@pytest.mark.asyncio
async def test_a_close_stops_the_reads_as_the_last_waiting_lookup_leaves(
    reply_records: _MakeRecords,
) -> None:
    """Stop the reads of a closing service once the last lookup waiting for them leaves, cancelling the pass in flight."""
    publishers = _publishers(2)
    rendezvous = _Reads(holds={0})
    records, _ = reply_records(rendezvous, publishers)
    lookup: asyncio.Future[bytes] | None = None
    try:
        for rank in publishers:
            records.registered(rank)
        await _until(lambda: rendezvous.reads)
        lookup = asyncio.ensure_future(records.section())
        await _until(lambda: records._waiting == 1)
        records.cancel()
        assert not records._section.done()
        lookup.cancel()
        await asyncio.gather(lookup, return_exceptions=True)
        assert _carried(records._section.result(timeout=0)) == {}
        await _until(lambda: rendezvous.cancelled == [0])
    finally:
        if not rendezvous._holds[0].done():
            rendezvous.release(0)
        if lookup is not None:
            lookup.cancel()
            await asyncio.gather(lookup, return_exceptions=True)


@pytest.mark.asyncio
async def test_a_close_before_the_reads_start_leaves_replies_without_records(
    monkeypatch: pytest.MonkeyPatch, reply_records: _MakeRecords
) -> None:
    """Start no thread for registrations that come after the service begins to close. Replies then carry no records."""
    started: list[str] = []
    monkeypatch.setattr(
        service_module, "_start_thread", lambda name, target: started.append(name)
    )
    publishers = _publishers(2)
    records, _ = reply_records(_Reads(), publishers)
    records.registered(publishers[0])
    records.cancel()
    records.registered(publishers[1])
    assert started == []


@pytest.mark.asyncio
async def test_a_close_after_the_reads_end_keeps_every_record_for_later_lookups(
    reply_records: _MakeRecords,
) -> None:
    """Reply with every record read once the reads ended, though the service then closes."""
    publishers = _publishers(2)
    rendezvous = _Reads(_held(publishers))
    records, _ = reply_records(rendezvous, publishers)
    for rank in publishers:
        records.registered(rank)
    await _until(records._section.done)
    records.cancel()
    assert _carried(await records.section()) == _held(publishers)


@pytest.mark.asyncio
async def test_a_close_that_comes_as_the_reads_encode_keeps_its_own_section(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    reply_records: _MakeRecords,
) -> None:
    """Keep the section a closing service sets as the ended reads encode theirs, which still log what replies carry."""
    caplog.set_level(logging.INFO, logger=service_module.__name__)
    entered, release = threading.Event(), threading.Event()
    encode = service_module._encode_records

    def encoding(records: Mapping[str, bytes]) -> bytes:
        if threading.current_thread().name == _THREAD:
            entered.set()
            release.wait(30)
        return encode(records)

    monkeypatch.setattr(service_module, "_encode_records", encoding)
    publishers = _publishers(2)
    rendezvous = _Reads(_held(publishers))
    records, _ = reply_records(rendezvous, publishers)
    before = set(threading.enumerate())
    try:
        for rank in publishers:
            records.registered(rank)
        (thread,) = _records_threads(before)
        assert await asyncio.to_thread(entered.wait, 30)
        records.cancel()
    finally:
        release.set()
    thread.join(10)
    assert not thread.is_alive()
    assert _carried(records._section.result(timeout=0)) == _held(publishers)
    logged = [
        (record.levelno, record.getMessage())
        for record in caplog.records
        if record.name == service_module.__name__
    ]
    carried = "routing store 'reads': replies carry 2 of 2 volume records, read in "
    assert [level for level, _ in logged] == [logging.INFO], logged
    assert logged[0][1].startswith(carried), logged


@pytest.mark.asyncio
async def test_the_records_reads_end_while_the_coordinators_loop_is_blocked(
    reply_records: _MakeRecords,
) -> None:
    """Set the section while the coordinator's loop is blocked since the reads need no turn on that loop.

    The registrations after it read nothing more.
    """
    publishers = _publishers(40)
    rendezvous = _Reads(_held(publishers))
    records, _ = reply_records(rendezvous, publishers)
    for rank in publishers[:16]:
        records.registered(rank)
    # Blocks this loop until the passes' thread hands the section over.
    section = records._section.result(timeout=10)
    for rank in publishers[16:]:
        records.registered(rank)
    assert _carried(section) == _held(publishers)
    assert rendezvous.reads == [sorted(publishers)]


_MakeAdapter = Callable[
    [_Reads, Collection[str]], Awaitable[service_module._CoordinatorAdapter]
]


@pytest.fixture
def make_adapter() -> Iterator[_MakeAdapter]:
    """Make an initialized adapter over a fake connection, and stop its records' reads at teardown.

    Teardown then checks that each case released every hold it made.
    """
    made: list[tuple[service_module._CoordinatorAdapter, _Reads]] = []

    async def make(
        rendezvous: _Reads, publishers: Collection[str]
    ) -> service_module._CoordinatorAdapter:
        adapter = service_module._CoordinatorAdapter(
            layout_registry.LayoutRegistry(), cast(Any, rendezvous)
        )
        made.append((adapter, rendezvous))
        await adapter.init(publishers, ts.TorchStoreStrategy())
        return adapter

    yield make
    held = [reads.unreleased_holds() for _, reads in made]
    for adapter, _ in made:
        _force_stop(adapter._records)
    assert held == [[]] * len(made)
    assert [reads.unreleased for _, reads in made] == [[]] * len(made)


async def _register_parts(
    adapter: service_module._CoordinatorAdapter, key: str, publishers: Iterable[str]
) -> None:
    for rank in publishers:
        await adapter.register_layout(rank, key, *_part(f"{key}/{rank}"))


@pytest.mark.asyncio
async def test_the_records_are_read_as_the_last_publisher_registers_before_any_lookup(
    make_adapter: _MakeAdapter,
) -> None:
    """Read a small store's records at its last publisher's first layout, before any lookup, which then only encodes them.

    Dropping the layouts at close drops the records too.
    """
    publishers, key = _publishers(2), _key()
    rendezvous = _Reads(_held(publishers))
    adapter = await make_adapter(rendezvous, publishers)
    await _register_parts(adapter, key, publishers)
    await _until(lambda: rendezvous.reads == [publishers], timeout_s=5)
    reply = bytes(await asyncio.wait_for(adapter.get_layouts(key, 0), 30))
    assert service_module._split_reply(_ADDRESS, reply)[1] == _held(publishers)
    assert rendezvous.reads == [publishers]
    adapter.drop_layouts()
    assert adapter._records._found == {}


@pytest.mark.asyncio
async def test_a_waiting_lookup_starts_the_records_reads_before_any_registration(
    make_adapter: _MakeAdapter,
) -> None:
    """Read every record as soon as a lookup waits in a store of 20, before the publishers register.

    Its reply carries every record, and nothing more is read as the publishers register.
    """
    publishers, key = _publishers(20), _key()
    rendezvous = _Reads(_held(publishers))
    adapter = await make_adapter(rendezvous, publishers)
    lookup = asyncio.ensure_future(adapter.get_layouts(key, 30))
    try:
        await _until(lambda: rendezvous.reads, timeout_s=5)
        assert rendezvous.reads == [sorted(publishers)]
        await _register_parts(adapter, key, publishers)
        reply = bytes(await asyncio.wait_for(lookup, 30))
    finally:
        lookup.cancel()
        await asyncio.gather(lookup, return_exceptions=True)
    assert service_module._split_reply(_ADDRESS, reply)[1] == _held(publishers)
    assert rendezvous.reads == [sorted(publishers)]


@pytest.mark.asyncio
async def test_a_closed_handle_drops_its_records_and_keeps_none_from_a_later_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drop a handle's records as it closes and keep none from a reply that lands after its close.

    The closed handle then refuses ``volume_records``.
    """
    publishers = _publishers(2)
    key, late = _key(), _key()
    async with _store(publishers, _records(publishers, 9000)) as store:
        await _register(store.driver, key, *publishers)
        await _register(store.driver, late, *publishers)
        handle = await store.connect()
        await handle.get_layouts(key)
        assert sorted(handle._records) == publishers
        call = handle._call

        async def closing_as_the_reply_lands(*args: Any, **kwargs: Any) -> Any:
            reply = await call(*args, **kwargs)
            await handle.close()
            return reply

        monkeypatch.setattr(handle, "_call", closing_as_the_reply_lands)
        assert await handle.get_layouts(late, timeout_s=0) == _layouts(late, publishers)
        assert handle._records == {}
        with pytest.raises(RuntimeError, match="this control plane is closed$"):
            await handle.volume_records(publishers)


@pytest.mark.asyncio
async def test_a_later_replys_record_replaces_the_one_a_handle_holds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the record of the latest reply that carries a volume's and serve it from ``volume_records``."""
    publishers = _publishers(2)
    key, later = _key(), _key()
    records = _records(publishers, 9100)
    async with _store(publishers, records) as store:
        await _register(store.driver, key, *publishers)
        await _register(store.driver, later, *publishers)
        handle = await store.connect()
        await handle.get_layouts(key)
        call = handle._call
        replaced = {publishers[0]: b"replaced"}

        async def another_record(*args: Any, **kwargs: Any) -> Any:
            layouts, _ = service_module._split_reply(
                handle.address, await call(*args, **kwargs)
            )
            return bytes(layouts) + service_module._encode_records(replaced)

        monkeypatch.setattr(handle, "_call", another_record)
        assert await handle.get_layouts(later, timeout_s=0) == _layouts(
            later, publishers
        )
        with _counted(handle) as calls:
            assert await handle.volume_records(publishers) == {**records, **replaced}
        assert _record_keys(calls) == []


@pytest.mark.asyncio
async def test_existing_volume_records_reads_only_the_records_that_exist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return only the ids with records, never asking ``multi_get`` for another, in chunks of at most 16.

    No ids read nothing, an id over the key cap fails before sending and no ``timeout_s`` takes the class's default
    as in ``volume_records``. After the ready check every store call runs in one native call. The read fetches every
    record that exists past the 16 missing ids where ``volume_records`` stops.
    """
    monkeypatch.setattr(tcpstore, "_DEFAULT_TIMEOUT_S", 3600.0)
    publishers = _publishers(20)
    absent = {"publisher/3", "publisher/17"}
    async with tcpstore_fixtures._served() as rendezvous:
        await rendezvous.publish(publishers, ts.TorchStoreStrategy())
        records = _records([p for p in publishers if p not in absent], 10000)
        for volume_id, record in records.items():
            await rendezvous.register_volume(volume_id, record)
        whats: list[str] = []
        with _store_calls(rendezvous) as (_, calls, _), _recording(
            rendezvous, whats
        ) as (
            _,
            deadlines,
        ):
            found = await rendezvous.existing_volume_records(publishers)
            assert deadlines[0] - time.monotonic() > 3000
            assert found == records
            assert await rendezvous.existing_volume_records([], timeout_s=60) == {}
            with pytest.raises(ValueError, match="takes .* bytes on the wire"):
                await rendezvous.existing_volume_records(["x" * 9000], timeout_s=60)
        endpoint = tcpstore._endpoint(rendezvous.address)
        assert whats == [
            f"check(['ready']) on TCPStore {endpoint}",
            f"multi_get(['ready']) on TCPStore {endpoint}",
            f"reading 20 volume records on TCPStore {endpoint}",
        ]
        gets = [keys for name, keys, _ in calls if name == "multi_get"][1:]
        assert [len(keys) for keys in gets] == [16, 2]
        expected = [
            f"volume/{p.replace('/', '%2F')}" for p in publishers if p not in absent
        ]
        assert [key for keys in gets for key in keys] == expected
        checked = [keys for name, keys, _ in calls if name == "check"][1:]
        assert [len(keys) for keys in checked] == [16] + [1] * 16 + [4] + [1] * 4
        unregistered = [f"unregistered/{rank}" for rank in range(16)]
        asked = [*publishers, *unregistered]
        assert await rendezvous.existing_volume_records(asked, timeout_s=60) == records


@pytest.mark.asyncio
async def test_existing_volume_records_under_a_long_store_name_reads_in_chunks_its_keys_fit() -> None:
    """Cut the checks and the fetch of the records that exist into chunks whose requests fit their long keys.

    Three wire keys of about 1.1KiB each fit one 4KiB request under this store's name.
    """
    publishers = _publishers(8)
    absent = "publisher/3"
    async with tcpstore_fixtures._served(
        store_name=f"{'s' * 1000}-{_store_name()}"
    ) as rendezvous:
        await rendezvous.publish(publishers, ts.TorchStoreStrategy())
        records = _records([p for p in publishers if p != absent], 16000)
        for volume_id, record in records.items():
            await rendezvous.register_volume(volume_id, record)
        with _store_calls(rendezvous) as (_, calls, _):
            found = await rendezvous.existing_volume_records(publishers, timeout_s=60)
    assert found == records
    # After the ready check's own calls. The chunk that holds the absent id is checked again key by key.
    checked = [len(keys) for name, keys, _ in calls if name == "check"][1:]
    assert checked == [3, 3, 1, 1, 1, 2]
    fetched = [len(keys) for name, keys, _ in calls if name == "multi_get"][1:]
    assert fetched == [3, 3, 1]


@pytest.mark.asyncio
async def test_a_records_read_whose_caller_leaves_gives_up_its_native_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Give up the records read's one native call once its caller is cancelled, so it starts no more store calls."""
    publishers = _publishers(2)
    entered, release, started = threading.Event(), threading.Event(), []

    def stalled(*args: Any, **kwargs: Any) -> tuple[list[str], dict[str, bytes]]:
        entered.set()
        release.wait(30)
        start = args[2]
        try:
            start("check(['next'])")
        except RuntimeError as error:
            started.append(str(error))
        else:
            started.append("started")
        return [], {}

    async with tcpstore_fixtures._served() as rendezvous:
        await rendezvous.publish(publishers, ts.TorchStoreStrategy())
        endpoint = tcpstore._endpoint(rendezvous.address)
        monkeypatch.setattr(tcpstore, "_read_records", stalled)
        read = asyncio.ensure_future(
            rendezvous.existing_volume_records(publishers, timeout_s=60)
        )
        try:
            assert await asyncio.to_thread(entered.wait, 30)
            read.cancel()
            await asyncio.gather(read, return_exceptions=True)
        finally:
            release.set()
        await _until(lambda: started)
    refused = (
        f"check(['next']) on TCPStore {endpoint} was not started: its caller gave up"
    )
    assert read.cancelled() and started == [refused]
