# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the carrier volume host: records in, proxies out, and the transport policy."""

import asyncio
import contextlib
import dataclasses
import gc
import itertools
import json
import logging
import pickle
import threading
import weakref
from collections.abc import AsyncIterator, Awaitable, Callable, Collection, Mapping
from functools import partial
from typing import Any, cast

import pytest
import torchstore.transport as transport
from torchstore import carrier_volumes
from torchstore.carrier import (
    CarrierAddress,
    CarrierProtocolError,
    PeerLostError,
    runtime,
)
from torchstore.carrier_volumes import (
    CARRIER_TRANSPORTS,
    CarrierVolumeHost,
    decode_volume_record,
    encode_volume_record,
    VolumeLocation,
    VolumeProxy,
)
from torchstore.strategy import StorageVolumeRef, TorchStoreStrategy
from torchstore.transport import TransportType
from torchstore.transport.buffers import TransportContext
from torchstore.transport.shared_memory import SharedMemoryTransportBuffer
from torchstore.utils import get_local_hostname

from . import test_carrier
from .test_carrier import _Children, _read_line, _scope

# Bound here, the carrier tests' autouse fixture closes every carrier after each test.
_close_carriers = test_carrier._close_carriers

pytestmark = [
    pytest.mark.timeout(120),
    pytest.mark.filterwarnings("error::pytest.PytestUnhandledThreadExceptionWarning"),
]

_IDS = ["publisher/0", "publisher/1"]
# The transports volume servers carry, written out so the module's set cannot shrink unseen.
_CARRIED = [TransportType.Gloo, TransportType.MonarchRPC, TransportType.SharedMemory]
# The probe MonarchRDMA's backend registers in the transport package, which a test replaces.
_MONARCH_RDMA = "monarch_rdma"
# Each transport volume servers refuse, by the probe that offers it to automatic selection.
_REFUSED = {
    "torch_transport_available": TransportType.TorchDistributed,
    "nixl_available": TransportType.NIXL,
    "torchcomms_rdma_available": TransportType.TorchComms,
    _MONARCH_RDMA: TransportType.MonarchRDMA,
}
# Every probe the transport log runs but SharedMemory's, by the transport it offers.
_PROBED = {
    **_REFUSED,
    "torchcomms_uniflow_available": TransportType.TorchComms,
    "gloo_available": TransportType.Gloo,
}
# The probes of transports volume servers refuse.
_OUTSIDE = [name for name, kind in _PROBED.items() if kind not in CARRIER_TRANSPORTS]

_FORK_CHILD = (
    test_carrier._CHILD
    + """
import signal

from torchstore import carrier_volumes
from torchstore.carrier_volumes import CarrierVolumeHost, encode_volume_record

inside, release = threading.Semaphore(0), threading.Event()
make_proxy = carrier_volumes.VolumeProxy


def held_proxy(address):
    # Holds the host's lock, as a first proxy does while the carrier's thread starts.
    inside.release()
    release.wait()
    return make_proxy(address)


def record(volume_id):
    return encode_volume_record(CarrierAddress("dial", 1, volume_id, "fork/1", "0f1e"), "host-a")


async def resolve(volume_ids):
    return {volume_id: record(volume_id) for volume_id in volume_ids}


# A process can hold several hosts, so two, each with its lock held by a thread of its own.
hosts = [CarrierVolumeHost(resolve) for _ in range(2)]
carrier_volumes.VolumeProxy = held_proxy
adopting = [
    threading.Thread(target=host.adopt, args=("publisher/0", record("publisher/0")))
    for host in hosts
]
for thread in adopting:
    thread.start()
for _ in hosts:
    inside.acquire()
pid = os.fork()
if pid == 0:
    # A child stuck on a lock its parent's other threads held dies at the alarm.
    signal.alarm(30)
    try:
        carrier_volumes.VolumeProxy = make_proxy
        for host in hosts:
            assert list(host.volume_ids()) == []
            asyncio.run(host.prepare(["publisher/1"]))
            host.adopt("publisher/2", record("publisher/2"))
            assert sorted(host.volume_ids()) == ["publisher/1", "publisher/2"]
            asyncio.run(host.close())
        os._exit(0)
    except BaseException:
        traceback.print_exc()
        os._exit(1)
# The parent keeps its locks, still held by the adopting threads.
held = all(host._lock.locked() for host in hosts)
release.set()
for thread in adopting:
    thread.join()
for host in hosts:
    asyncio.run(host.close())
_, status = os.waitpid(pid, 0)
print(os.waitstatus_to_exitcode(status) if held else "parent lock replaced", flush=True)
"""
)

_PUT_AND_GET_CHILD = """
import asyncio, importlib.util, json, os, sys

# RDMA on, as by default, so MonarchRDMA's probe can pass where Monarch has RDMA.
os.environ.update(
    TORCHSTORE_RDMA_ENABLED="1", TORCHSTORE_SHARED_MEMORY_ENABLED="1", TORCHSTORE_GLOO_ENABLED="1"
)
sys.path.insert(0, sys.argv[1])
import torch
import torchstore.transport as transport
from torchstore.backend import get_backend
from torchstore.carrier_volumes import (
    CARRIER_TRANSPORTS,
    CarrierVolumeHost,
    decode_volume_record,
    encode_volume_record,
    serve_volume,
)
from torchstore.storage import StorageVolumeCore
from torchstore.strategy import TorchStoreStrategy
from torchstore.transport import TransportType
from torchstore.transport.types import Request

OFFERS = {
    "torch_transport_available": TransportType.TorchDistributed,
    "nixl_available": TransportType.NIXL,
    "torchcomms_uniflow_available": TransportType.TorchComms,
    "torchcomms_rdma_available": TransportType.TorchComms,
}
probed = []


def recorded(name, probe):
    def call():
        probed.append(name)
        return probe()

    return call


# Each probe of a transport outside CARRIER_TRANSPORTS records its calls, then runs.
for name, offered in OFFERS.items():
    if offered not in CARRIER_TRANSPORTS:
        setattr(transport, name, recorded(name, getattr(transport, name)))


async def unread(volume_ids):
    raise AssertionError(f"the host read records for {volume_ids}")


async def put_and_get(record, transport_type, key):
    host = CarrierVolumeHost(unread)
    host.adopt("publisher/0", record)
    strategy = TorchStoreStrategy(transport_type)
    strategy.set_volume_host(host)
    ref = strategy.get_storage_volume("publisher/0")
    value = torch.arange(4.0) + len(key)
    buffer = transport.create_transport_buffer(ref)
    await buffer.put_to_storage_volume([Request.from_tensor(key, value)])
    buffer = transport.create_transport_buffer(ref)
    [fetched] = await buffer.get_from_storage_volume([Request(key=key)])
    assert torch.equal(fetched, value)
    await host.close()
    return ref.default_transport_type.name


async def main():
    monarch = importlib.util.find_spec("monarch") is not None
    if monarch:
        # The Monarch backend registers MonarchRDMA and its probe as it loads.
        get_backend("monarch")
    registered = TransportType.MonarchRDMA in transport._backend_transports
    if registered:
        buffer, available = transport._backend_transports[TransportType.MonarchRDMA]
        transport.register_backend_transport(
            TransportType.MonarchRDMA, buffer, recorded("monarch_rdma", available)
        )
    _, record = await serve_volume(StorageVolumeCore("publisher/0"), scope=sys.argv[2])
    address, _ = decode_volume_record("publisher/0", record)
    remote = encode_volume_record(address, "another-host")
    picked = []
    arms = [(record, TransportType.Unset), (remote, TransportType.Unset), (remote, TransportType.MonarchRPC)]
    for key, (arm, transport_type) in zip(["a", "bb", "ccc"], arms):
        picked.append(await put_and_get(arm, transport_type, key))
    calls = list(probed)
    transport._backend_transport_available(TransportType.MonarchRDMA)
    control = probed[len(calls):] == ["monarch_rdma"]
    report = {
        "picked": picked, "probed": calls, "monarch": monarch, "registered": registered,
        "control": control,
    }
    print(json.dumps(report), flush=True)


asyncio.run(main())
"""


class _Records:
    """A resolver over a dict of records, as ``VolumeRecords.volume_records`` reads a store.

    It keeps each call's IDs, and one KeyError names every ID it has no record for.
    """

    def __init__(self, records: Mapping[str, bytes]) -> None:
        self.records = dict(records)
        self.calls: list[list[str]] = []

    async def __call__(self, volume_ids: Collection[str]) -> dict[str, bytes]:
        self.calls.append(list(volume_ids))
        missing = sorted(set(volume_ids) - self.records.keys())
        if missing:
            raise KeyError(f"no volume record for {missing}")
        return {volume_id: self.records[volume_id] for volume_id in volume_ids}


def _record(volume_id: str, *, host: str = "dial", hostname: str = "host-a") -> bytes:
    """A record whose address nothing listens on: lookups never dial it."""
    address = CarrierAddress(host, 1, volume_id, _scope(), "0f1e")
    return encode_volume_record(address, hostname)


def _address(volume_id: str, record: bytes) -> CarrierAddress:
    return decode_volume_record(volume_id, record).address


def _peers() -> set[CarrierAddress]:
    """The addresses this process's carrier runtime holds a connection entry for."""
    current = runtime._runtime
    return set() if current is None else set(current.peers)


def _ref(transport_type: TransportType, **record_fields: Any) -> StorageVolumeRef:
    """A reference to ``publisher/0`` through a strategy on a carrier host."""
    host = CarrierVolumeHost(_Records({}))
    host.adopt("publisher/0", _record("publisher/0", **record_fields))
    strategy = TorchStoreStrategy(transport_type)
    strategy.set_volume_host(host)
    return strategy.get_storage_volume("publisher/0")


def _unrestricted(ref: StorageVolumeRef) -> StorageVolumeRef:
    """A reference to ``ref``'s volume naming no transport and no ``transports`` (as on a Monarch mesh)."""
    return StorageVolumeRef(
        ref.volume,
        ref.volume_id,
        ref.transport_context,
        TransportType.Unset,
        ref.volume_hostname,
    )


@contextlib.asynccontextmanager
async def _finally(cleanup: Callable[[], Awaitable[object]]) -> AsyncIterator[None]:
    """Await ``cleanup`` on the way out.

    After a failing body an Exception from ``cleanup`` is logged rather than raised.
    """
    try:
        yield
    except BaseException:
        try:
            await cleanup()
        except Exception:
            logging.getLogger(__name__).exception(
                "cleanup failed after the case failed"
            )
        raise
    await cleanup()


def _record_probes(
    monkeypatch: pytest.MonkeyPatch,
    offered: str | Collection[str] | None,
    *,
    gloo_on: bool,
) -> list[str]:
    """Make the probes named in ``offered`` report their transports, Gloo's ``gloo_on``, and others none.

    Returns the list that each later call of those probes appends its name to.
    """
    calls: list[str] = []
    offers = {offered} if isinstance(offered, str) else set(offered or ())

    def probe(name: str) -> bool:
        calls.append(name)
        return gloo_on if name == "gloo_available" else name in offers

    for name in _PROBED:
        if name != _MONARCH_RDMA:
            monkeypatch.setattr(transport, name, partial(probe, name))
    # A copy, which monkeypatch restores, so the stand-in never outlives the test.
    monkeypatch.setattr(
        transport, "_backend_transports", dict(transport._backend_transports)
    )
    transport.register_backend_transport(
        TransportType.MonarchRDMA,
        transport.TransportBuffer,
        partial(probe, _MONARCH_RDMA),
    )
    return calls


class _HandOff:
    """Stands in for a host's lock and runs ``then`` once, right after its next release.

    A call ``then`` starts on another thread lands between two of the host's lock holds.
    ``reached`` is set once a call starts to take the lock.
    """

    def __init__(self, host: CarrierVolumeHost) -> None:
        self._lock = host._lock
        self.then: Callable[[], None] | None = None
        self.reached = threading.Event()
        host._lock = self  # type: ignore[assignment]

    def __enter__(self) -> None:
        self.reached.set()
        self._lock.acquire()

    def __exit__(self, *exc_info: object) -> None:
        self._lock.release()
        then, self.then = self.then, None
        if then is not None:
            then()


def _on_another_thread(
    call: Callable[[], object], errors: list[BaseException]
) -> Callable[[], None]:
    """Return a callback that runs ``call`` to its end on a thread of its own, keeping its error."""

    def run_there() -> None:
        def run() -> None:
            try:
                call()
            except BaseException as e:
                errors.append(e)

        thread = threading.Thread(target=run)
        thread.start()
        thread.join(30)
        assert not thread.is_alive()

    return run_there


@pytest.mark.asyncio
async def test_an_adopted_record_resolves_without_a_resolver_call() -> None:
    """Resolve an adopted volume at once, also once adopted again, and never read its record."""
    record = _record("publisher/0", hostname="host-0")
    records = _Records({})
    host = CarrierVolumeHost(records)
    host.adopt("publisher/0", record)
    # A publisher client created again adopts the same record again.
    host.adopt("publisher/0", record)
    await host.prepare(["publisher/0"])
    assert records.calls == []
    proxy = host.endpoints("publisher/0")
    assert isinstance(proxy, VolumeProxy)
    assert proxy.volume_id == "publisher/0"
    assert host.endpoints("publisher/0") is proxy
    assert host.hostname("publisher/0") == "host-0"
    assert list(host.volume_ids()) == ["publisher/0"]
    assert repr(host) == "CarrierVolumeHost(volumes=1)"
    # The same server under another host name, then another server under the same one.
    address = _address("publisher/0", record)
    for other in (
        encode_volume_record(address, "host-1"),
        encode_volume_record(dataclasses.replace(address, port=2), "host-0"),
    ):
        with pytest.raises(ValueError, match="'publisher/0' already resolves through"):
            host.adopt("publisher/0", other)
    with pytest.raises(CarrierProtocolError, match="for service 'publisher/0'"):
        host.adopt("publisher/1", record)
    assert host.endpoints("publisher/0") is proxy
    assert list(host.volume_ids()) == ["publisher/0"]


@pytest.mark.asyncio
async def test_one_prepare_reads_every_record_the_host_lacks_in_one_call() -> None:
    """Read the records of all lacking volumes in one call, and later none for volumes held."""
    ids = [f"publisher/{rank}" for rank in range(5)]
    records = _Records({v: _record(v, hostname=f"host-{v}") for v in ids})
    host = CarrierVolumeHost(records)
    host.adopt(ids[0], records.records[ids[0]])
    # Routing passes the layouts it got, keyed by the publishers' volume IDs.
    await host.prepare({volume_id: {} for volume_id in reversed(ids)})
    assert records.calls == [ids[1:]]
    await host.prepare(ids)
    await host.prepare([])
    assert records.calls == [ids[1:]]
    assert sorted(host.volume_ids()) == ids
    for volume_id in ids:
        assert host.endpoints(volume_id).volume_id == volume_id
        assert host.hostname(volume_id) == f"host-{volume_id}"


def test_a_lazy_iterable_that_reads_the_host_prepares_its_volumes() -> None:
    """Prepare from a generator that lists the host's volumes, reading only the lacking record.

    The host reads the iterable before it takes the lock a listing also takes.
    """
    records = _Records({v: _record(v) for v in _IDS})
    host = CarrierVolumeHost(records)
    host.adopt(_IDS[0], records.records[_IDS[0]])
    strategy = TorchStoreStrategy(TransportType.MonarchRPC)
    strategy.set_volume_host(host)
    lacking = (v for v in _IDS if v not in host.volume_ids())
    # On a thread of its own, so a prepare stuck on the lock fails here instead of hanging.
    preparing = threading.Thread(
        target=lambda: asyncio.run(strategy._prepare_volumes(lacking)), daemon=True
    )
    preparing.start()
    preparing.join(30)
    assert not preparing.is_alive(), "the prepare waited on the host's lock"
    assert records.calls == [_IDS[1:]]
    assert sorted(host.volume_ids()) == _IDS


@pytest.mark.asyncio
async def test_two_threads_preparing_at_once_end_with_one_proxy_per_volume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Make one proxy per volume though two threads, each on its own loop, read at once.

    One thread makes its first proxy only once the other has decoded both records, its
    last step before the host's lock, so the two installs overlap.
    """
    records = _Records({v: _record(v) for v in _IDS})
    both_reading = threading.Barrier(2, timeout=30)

    async def read_together(volume_ids: Collection[str]) -> dict[str, bytes]:
        await asyncio.to_thread(both_reading.wait)
        return await records(volume_ids)

    decode = carrier_volumes.decode_volume_record
    decoded: list[str] = []
    all_decoded = threading.Event()

    def counted_decode(volume_id: str, record: bytes) -> VolumeLocation:
        location = decode(volume_id, record)
        decoded.append(volume_id)
        if len(decoded) == 2 * len(_IDS):
            all_decoded.set()
        return location

    made: list[str] = []
    proxy_class = carrier_volumes.VolumeProxy

    def counted(address: CarrierAddress) -> VolumeProxy:
        if not made:
            assert all_decoded.wait(30)
        made.append(address.service)
        return proxy_class(address)

    monkeypatch.setattr(carrier_volumes, "decode_volume_record", counted_decode)
    monkeypatch.setattr(carrier_volumes, "VolumeProxy", counted)
    host = CarrierVolumeHost(read_together)
    await asyncio.gather(
        *(asyncio.to_thread(lambda: asyncio.run(host.prepare(_IDS))) for _ in range(2))
    )
    assert records.calls == [_IDS, _IDS]
    assert sorted(made) == _IDS
    assert [host.endpoints(v).volume_id for v in _IDS] == _IDS


@pytest.mark.parametrize("second", ["adopt", "close", "volume_ids", "prepare"])
def test_calls_on_another_thread_wait_for_an_install_in_progress(
    monkeypatch: pytest.MonkeyPatch, second: str
) -> None:
    """Hold off another thread's adopt, close, listing or read while one thread makes a proxy.

    So the volume gets one proxy, none outlives ``close``, a listing waits to include the
    volume, and a read of its record never starts.
    """
    record = _record("publisher/0")
    records = _Records({"publisher/0": record})
    making, release = threading.Event(), threading.Event()
    made: list[str] = []
    listed: list[list[str]] = []
    proxy_class = carrier_volumes.VolumeProxy

    def held(address: CarrierAddress) -> VolumeProxy:
        made.append(address.service)
        if len(made) == 1:
            making.set()
            # The test releases this once its own 30s wait for the other call ends.
            assert release.wait(60)
        return proxy_class(address)

    monkeypatch.setattr(carrier_volumes, "VolumeProxy", held)
    host = CarrierVolumeHost(records)
    first = threading.Thread(target=host.adopt, args=("publisher/0", record))
    first.start()
    assert making.wait(30)
    calls = {
        "adopt": partial(host.adopt, "publisher/0", record),
        "close": lambda: asyncio.run(host.close()),
        "volume_ids": lambda: listed.append(list(host.volume_ids())),
        "prepare": lambda: asyncio.run(host.prepare(["publisher/0"])),
    }
    # The first install holds the host's lock until released. The other call waits for it there.
    hand_off = _HandOff(host)
    other = threading.Thread(target=calls[second])
    other.start()
    reached = hand_off.reached.wait(30)
    release.set()
    for thread in (first, other):
        thread.join(30)
        assert not thread.is_alive()
    assert reached, f"the {second} never took the host's lock"
    assert made == ["publisher/0"]
    assert records.calls == []
    assert listed == ([["publisher/0"]] if second == "volume_ids" else [])
    if second == "close":
        assert list(host.volume_ids()) == []
        assert _address("publisher/0", record) not in _peers()
    else:
        assert list(host.volume_ids()) == ["publisher/0"]


@pytest.mark.asyncio
async def test_an_unknown_volume_or_a_reply_lacking_its_record_names_it() -> None:
    """Raise KeyError naming a volume looked up before its record loads, or left out of a reply."""
    host = CarrierVolumeHost(_Records({}))
    strategy = TorchStoreStrategy(TransportType.MonarchRPC)
    strategy.set_volume_host(host)
    unknown = "'publisher/0' has no record here. Records load once a requester's"
    for lookup in (host.endpoints, host.hostname, strategy.get_storage_volume):
        with pytest.raises(KeyError, match=unknown):
            lookup("publisher/0")

    async def partial_reply(volume_ids: Collection[str]) -> dict[str, bytes]:
        return {"publisher/0": _record("publisher/0")}

    host = CarrierVolumeHost(partial_reply)
    with pytest.raises(KeyError, match=r"no volume record for \['publisher/1'\]"):
        await host.prepare(_IDS)
    assert list(host.volume_ids()) == []


@pytest.mark.asyncio
async def test_a_malformed_record_names_its_volume() -> None:
    """Raise naming the volume whose record is malformed.

    A failed ``prepare`` keeps none of its batch, so the next one reads it all again.
    """
    records = _Records({"publisher/0": _record("publisher/0")})
    records.records["publisher/1"] = b"[1"
    host = CarrierVolumeHost(records)
    for _ in range(2):
        with pytest.raises(
            CarrierProtocolError, match="^volume 'publisher/1' has a malformed record"
        ):
            await host.prepare(_IDS)
        assert list(host.volume_ids()) == []
    assert records.calls == [_IDS, _IDS]


@pytest.mark.asyncio
async def test_prepare_refuses_one_volume_id_passed_as_a_str() -> None:
    """Raise TypeError for a bare str, which would read as one-character IDs, reading nothing."""
    records = _Records({"publisher/10": _record("publisher/10")})
    host = CarrierVolumeHost(records)
    with pytest.raises(
        TypeError,
        match="^need an iterable of volume IDs, not the str 'publisher/10'$",
    ):
        await host.prepare("publisher/10")
    assert records.calls == []
    assert list(host.volume_ids()) == []


@pytest.mark.asyncio
async def test_a_closed_host_closes_its_proxies_then_refuses_lookups() -> None:
    """Close every proxy and its connection entry, then refuse lookups, adopts and reads whatever their arguments."""
    records = _Records({v: _record(v) for v in _IDS})
    host = CarrierVolumeHost(records)
    host.adopt(_IDS[0], records.records[_IDS[0]])
    await host.prepare(_IDS)
    proxies = [host.endpoints(v) for v in _IDS]
    addresses = {_address(v, records.records[v]) for v in _IDS}
    assert addresses <= _peers()
    strategy = TorchStoreStrategy(TransportType.MonarchRPC)
    strategy.set_volume_host(host)

    await host.close()
    assert not addresses & _peers()
    for proxy in proxies:
        with pytest.raises(PeerLostError, match="is closed"):
            await proxy.get_id.call_one()
    closed = "'publisher/0' has no handle: its carrier volume host is closed"
    for lookup in (host.endpoints, host.hostname, strategy.get_storage_volume):
        with pytest.raises(RuntimeError, match=closed):
            lookup("publisher/0")
    # An open host refuses a malformed or misfiled record and a bare str otherwise.
    for record in (records.records[_IDS[0]], b"{not json", _record("publisher/1")):
        with pytest.raises(RuntimeError, match="^this carrier volume host is closed$"):
            host.adopt(_IDS[0], record)
    for volume_ids in (["publisher/2"], "publisher/2"):
        with pytest.raises(RuntimeError, match="^this carrier volume host is closed$"):
            await host.prepare(volume_ids)
    assert records.calls == [_IDS[1:]]
    assert list(host.volume_ids()) == []
    assert repr(host) == "CarrierVolumeHost(closed)"
    await host.close()


@pytest.mark.asyncio
async def test_a_cancelled_close_still_closes_every_proxy() -> None:
    """Close every proxy, and leave no task running, when a cancel lands as ``close`` first waits."""
    records = _Records({v: _record(v) for v in _IDS})
    host = CarrierVolumeHost(records)
    await host.prepare(_IDS)
    proxies = [host.endpoints(v) for v in _IDS]
    addresses = {_address(v, records.records[v]) for v in _IDS}
    before = asyncio.all_tasks()
    left: list[set[asyncio.Task[Any]]] = []

    async def close_then_list_tasks() -> None:
        try:
            await host.close()
        finally:
            left.append(asyncio.all_tasks() - before - {asyncio.current_task()})

    closing = asyncio.ensure_future(close_then_list_tasks())
    # Runs right after close's first step, ahead of the steps that close each proxy.
    asyncio.get_running_loop().call_soon(closing.cancel)
    with pytest.raises(asyncio.CancelledError):
        await closing
    assert left == [set()]
    assert not addresses & _peers()
    for proxy in proxies:
        with pytest.raises(PeerLostError, match="is closed"):
            await proxy.get_id.call_one()


@pytest.mark.asyncio
async def test_a_close_starting_as_another_runs_closes_every_handle_before_it_returns() -> None:
    """Return from a second close, started as the first lets go of the lock, only once every handle closed."""
    records = _Records({v: _record(v) for v in _IDS})
    host = CarrierVolumeHost(records)
    await host.prepare(_IDS)
    addresses = {_address(v, records.records[v]) for v in _IDS}
    errors: list[BaseException] = []
    left: list[set[CarrierAddress]] = []

    def second_close() -> None:
        asyncio.run(host.close())
        left.append(addresses & _peers())

    _HandOff(host).then = _on_another_thread(second_close, errors)
    await host.close()
    assert errors == [] and left == [set()]


@pytest.mark.asyncio
async def test_a_cancel_ends_a_close_stuck_on_a_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End a cancelled close at once, though a handle's close never returns."""
    records = _Records({v: _record(v) for v in _IDS})
    host = CarrierVolumeHost(records)
    await host.prepare(_IDS)
    started = asyncio.Event()

    async def stuck(self: VolumeProxy) -> None:
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(VolumeProxy, "close", stuck)
    closing = asyncio.ensure_future(host.close())
    await started.wait()
    closing.cancel()
    done, _ = await asyncio.wait({closing}, timeout=5)
    assert closing in done and closing.cancelled()


@pytest.mark.asyncio
async def test_a_close_waiting_on_a_proxy_lets_lookups_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """List the volumes from another thread while ``close`` waits on a proxy's close."""
    host = CarrierVolumeHost(_Records({v: _record(v) for v in _IDS}))
    await host.prepare(_IDS)
    proxy_close = VolumeProxy.close
    closing_one, release = asyncio.Event(), asyncio.Event()

    async def held_close(self: VolumeProxy) -> None:
        closing_one.set()
        await release.wait()
        await proxy_close(self)

    monkeypatch.setattr(VolumeProxy, "close", held_close)
    closing = asyncio.ensure_future(host.close())

    async def finish_closing() -> None:
        release.set()
        await closing

    await closing_one.wait()
    async with _finally(finish_closing):
        assert await asyncio.wait_for(asyncio.to_thread(host.volume_ids), 10) == []


@pytest.mark.asyncio
async def test_an_adopt_as_close_releases_the_lock_finds_the_host_closed() -> None:
    """Refuse another thread's adopt that takes the lock the moment ``close`` lets it go.

    ``close`` marks the host closed in the hold that takes its handles, so no proxy slips in.
    """
    record = _record("publisher/1")
    host = CarrierVolumeHost(_Records({}))
    host.adopt("publisher/0", _record("publisher/0"))
    errors: list[BaseException] = []
    adopt = partial(host.adopt, "publisher/1", record)
    _HandOff(host).then = _on_another_thread(adopt, errors)
    await host.close()
    assert [type(e) for e in errors] == [RuntimeError], errors
    assert list(host.volume_ids()) == []
    assert _address("publisher/1", record) not in _peers()


def test_a_differing_record_adopted_as_another_adopt_releases_the_lock_raises() -> None:
    """Raise ValueError for another thread's differing record right after an adopt's one hold."""
    host = CarrierVolumeHost(_Records({}))
    errors: list[BaseException] = []
    adopt = partial(
        host.adopt, "publisher/0", _record("publisher/0", hostname="host-2")
    )
    _HandOff(host).then = _on_another_thread(adopt, errors)
    host.adopt("publisher/0", _record("publisher/0", hostname="host-1"))
    assert [type(e) for e in errors] == [ValueError], errors
    assert host.hostname("publisher/0") == "host-1"


@pytest.mark.asyncio
async def test_a_close_as_prepare_installs_closes_the_new_proxy() -> None:
    """Close the proxy a ``prepare`` made when another thread's close takes the lock right after.

    ``prepare`` checks for a close and installs in one hold, so that close takes the proxy.
    """
    record = _record("publisher/0")
    errors: list[BaseException] = []
    hand_offs: list[_HandOff] = []

    async def reply_then_hand_off(volume_ids: Collection[str]) -> dict[str, bytes]:
        hand_offs[0].then = _on_another_thread(
            lambda: asyncio.run(host.close()), errors
        )
        return {"publisher/0": record}

    host = CarrierVolumeHost(reply_then_hand_off)
    hand_offs.append(_HandOff(host))
    await host.prepare(["publisher/0"])
    assert errors == []
    assert list(host.volume_ids()) == []
    assert _address("publisher/0", record) not in _peers()


@pytest.mark.parametrize("arrival", ["after_close", "while_closing"])
@pytest.mark.asyncio
async def test_records_that_arrive_once_close_starts_make_no_proxy(
    monkeypatch: pytest.MonkeyPatch, arrival: str
) -> None:
    """Raise, and leave no connection entry, when a read's records arrive once ``close`` starts.

    In the ``while_closing`` arm, they arrive while ``close`` waits on another volume's proxy.
    """
    record = _record("publisher/0")
    reading, release = asyncio.Event(), asyncio.Event()
    closing_held, finish_closing = asyncio.Event(), asyncio.Event()

    async def held(volume_ids: Collection[str]) -> dict[str, bytes]:
        reading.set()
        await release.wait()
        return {"publisher/0": record}

    host = CarrierVolumeHost(held)
    if arrival == "while_closing":
        host.adopt("publisher/1", _record("publisher/1"))
        proxy_close = VolumeProxy.close

        async def close_when_told(self: VolumeProxy) -> None:
            closing_held.set()
            await finish_closing.wait()
            await proxy_close(self)

        monkeypatch.setattr(VolumeProxy, "close", close_when_told)
    preparing = asyncio.ensure_future(host.prepare(["publisher/0"]))
    await reading.wait()
    closing = asyncio.ensure_future(host.close())

    async def end_closing() -> None:
        finish_closing.set()
        await closing

    if arrival == "while_closing":
        await closing_held.wait()
    else:
        await closing
    release.set()
    async with _finally(end_closing):
        with pytest.raises(RuntimeError, match="^this carrier volume host is closed$"):
            await preparing
    assert _address("publisher/0", record) not in _peers()
    assert list(host.volume_ids()) == []


@pytest.mark.asyncio
async def test_a_host_dropped_without_close_frees_its_proxies() -> None:
    """Free a dropped host's proxies, so another host's close still ends their connection."""
    record = _record("publisher/0")
    host = CarrierVolumeHost(_Records({}))
    host.adopt("publisher/0", record)
    freed = weakref.ref(host)
    del host
    gc.collect()
    assert freed() is None
    kept = CarrierVolumeHost(_Records({}))
    kept.adopt("publisher/0", record)
    await kept.close()
    assert _address("publisher/0", record) not in _peers()


def test_a_host_and_a_strategy_holding_it_refuse_to_pickle() -> None:
    """Refuse to pickle a host, so no record-backed strategy rides a pickled message."""
    host = CarrierVolumeHost(_Records({}))
    strategy = TorchStoreStrategy()
    strategy.set_volume_host(host)
    for value in (host, strategy):
        with pytest.raises(TypeError, match="a carrier volume host does not pickle"):
            pickle.dumps(value)


def test_a_fork_while_another_thread_holds_the_lock_gets_a_working_host(
    children: _Children,
) -> None:
    """List, prepare, adopt and close in a child forked while other threads held two hosts' locks.

    Every live host's lock is fresh in the child, and the parent's stay with the threads holding them.
    """
    child = children.spawn(_FORK_CHILD)
    assert _read_line(child, timeout_s=90) == "0"


@pytest.mark.parametrize("transport_type", _CARRIED)
def test_a_transport_volume_servers_carry_is_kept(
    transport_type: TransportType,
) -> None:
    """Keep SharedMemory, Gloo or MonarchRPC when the strategy names it, local or not."""
    assert CARRIER_TRANSPORTS == frozenset(_CARRIED)
    assert _ref(transport_type).default_transport_type is transport_type
    local = _ref(transport_type, hostname=get_local_hostname())
    assert local.default_transport_type is transport_type


@pytest.mark.parametrize("gloo_on", [True, False], ids=["gloo", "no_gloo"])
@pytest.mark.parametrize("offered", [*sorted(_REFUSED), None])
def test_automatic_selection_keeps_to_the_transports_volume_servers_carry(
    monkeypatch: pytest.MonkeyPatch, offered: str | None, gloo_on: bool
) -> None:
    """Map a refused transport that automatic selection picks to Gloo, or to MonarchRPC without Gloo."""
    _record_probes(monkeypatch, offered, gloo_on=gloo_on)
    ref = _ref(TransportType.Unset, hostname="another-host")
    chosen = TransportType.Gloo if gloo_on else TransportType.MonarchRPC
    offered_type = chosen if offered is None else _REFUSED[offered]
    assert transport.get_available_transport(_unrestricted(ref)) is offered_type
    assert ref.default_transport_type is chosen


@pytest.mark.parametrize("gloo_on", [True, False], ids=["gloo", "no_gloo"])
def test_one_automatic_strategy_picks_each_volumes_transport_by_its_own_record(
    monkeypatch: pytest.MonkeyPatch, gloo_on: bool
) -> None:
    """Pick SharedMemory for a local volume and Gloo, or MonarchRPC, for a remote one.

    One strategy looks up both, in turn, and its own transport stays Unset throughout.
    """
    monkeypatch.setattr(transport, "SHM_ENABLED", True)
    _record_probes(monkeypatch, _MONARCH_RDMA, gloo_on=gloo_on)
    host = CarrierVolumeHost(_Records({}))
    local, remote = _IDS
    host.adopt(local, _record(local, hostname=get_local_hostname()))
    host.adopt(remote, _record(remote, hostname="another-host"))
    strategy = TorchStoreStrategy()
    strategy.set_volume_host(host)
    remote_type = TransportType.Gloo if gloo_on else TransportType.MonarchRPC
    expected = {local: TransportType.SharedMemory, remote: remote_type}
    for volume_id in (local, remote, local):
        ref = strategy.get_storage_volume(volume_id)
        assert ref.default_transport_type is expected[volume_id], volume_id
        assert strategy.default_transport_type is TransportType.Unset


@pytest.mark.parametrize("requested", sorted(_REFUSED.values(), key=str))
def test_an_explicit_transport_volume_servers_refuse_raises(
    requested: TransportType,
) -> None:
    """Refuse a strategy that names NIXL, TorchComms, MonarchRDMA or TorchDistributed, naming it and the volume."""
    with pytest.raises(
        ValueError,
        match=(
            "^volume 'publisher/0' is served over the carrier, which carries"
            f" SharedMemory, Gloo and MonarchRPC, not {requested.name}$"
        ),
    ):
        _ref(requested)


def test_sharedmemory_follows_the_records_host_name_not_the_address_it_dials(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Choose SharedMemory for a record naming this host, not for one dialing this host's name or a loopback address."""
    monkeypatch.setattr(transport, "SHM_ENABLED", True)
    local = get_local_hostname()
    ref = _ref(TransportType.Unset, hostname=local)
    assert ref.default_transport_type is TransportType.SharedMemory
    for dialed in (local, "127.0.0.1"):
        other = _ref(TransportType.Unset, host=dialed, hostname="another-host")
        assert other.default_transport_type is not TransportType.SharedMemory, dialed
    # Building a buffer still logs the transport it builds.
    caplog.set_level(logging.INFO, logger=transport.__name__)
    assert isinstance(
        transport.create_transport_buffer(ref), SharedMemoryTransportBuffer
    )
    assert "[ts-transport] resolved=SharedMemory" in caplog.text


@pytest.mark.parametrize("gloo_on", [True, False], ids=["gloo", "no_gloo"])
@pytest.mark.parametrize("offered", [*_OUTSIDE, None])
@pytest.mark.parametrize("local", [True, False], ids=["local", "remote"])
@pytest.mark.parametrize("shm_on", [True, False], ids=["shm", "no_shm"])
def test_automatic_selection_probes_only_carried_transports(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    shm_on: bool,
    local: bool,
    offered: str | None,
    gloo_on: bool,
) -> None:
    """Resolve and log an automatic strategy's volume probing only ``CARRIER_TRANSPORTS``.

    It picks what full automatic selection would, with a pick outside the set becoming
    Gloo, or MonarchRPC without Gloo.
    """
    caplog.set_level(logging.INFO, logger=transport.__name__)
    monkeypatch.setattr(transport, "SHM_ENABLED", shm_on)
    calls = _record_probes(monkeypatch, offered, gloo_on=gloo_on)
    hostname = get_local_hostname() if local else "another-host"
    ref = _ref(TransportType.Unset, hostname=hostname)
    transport.create_transport_buffer(ref)
    assert set(calls).isdisjoint(_OUTSIDE)
    picked = ref.default_transport_type
    line = f"[ts-transport] resolved={picked.name} (gloo={gloo_on}, shm={shm_on and local})"
    assert line in caplog.text
    expected = transport.get_available_transport(_unrestricted(ref))
    if expected not in CARRIER_TRANSPORTS:
        expected = TransportType.Gloo if gloo_on else TransportType.MonarchRPC
    assert picked is expected


@pytest.mark.parametrize(
    "offered",
    [
        "torch_transport_available",
        "nixl_available",
        "torchcomms_uniflow_available",
        "torchcomms_rdma_available",
        _MONARCH_RDMA,
    ],
)
def test_the_transport_log_shows_each_probes_result_under_its_own_name(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, offered: str
) -> None:
    """Log every probe for a reference naming no ``transports``. Only ``offered``'s shows True."""
    caplog.set_level(logging.INFO, logger=transport.__name__)
    columns = {
        "torch_transport_available": "torch_distributed",
        "nixl_available": "nixl",
        "torchcomms_uniflow_available": "uniflow",
        "torchcomms_rdma_available": "tc_rdma",
        _MONARCH_RDMA: "monarch_rdma",
    }
    _record_probes(monkeypatch, offered, gloo_on=False)
    plain = StorageVolumeRef(
        cast(Any, object()),
        "publisher/0",
        TransportContext(),
        TransportType.Gloo,
        "another-host",
    )
    transport.create_transport_buffer(plain)
    shown = ", ".join(f"{column}={name == offered}" for name, column in columns.items())
    assert (
        f"[ts-transport] resolved=Gloo ({shown}, gloo=False, shm=False)" in caplog.text
    )


# Every non-empty set of the transports volume servers refuse, as a widening could add.
_ADDED = [
    frozenset(added)
    for count in (1, 2, 3, 4)
    for added in itertools.combinations(
        sorted(set(_PROBED.values()) - CARRIER_TRANSPORTS, key=str), count
    )
]


@pytest.mark.parametrize(
    "added", _ADDED, ids=lambda added: "+".join(sorted(kind.name for kind in added))
)
def test_transports_added_to_the_carried_set_pick_as_automatic_selection_in_every_probe_state(
    monkeypatch: pytest.MonkeyPatch, added: frozenset[TransportType]
) -> None:
    """Pick the first member of the widened set whose probe passes, in automatic selection's order.

    A probe outside the set never runs so its offer does not move the pick. Every probe state's mismatch is reported.
    """
    carried = CARRIER_TRANSPORTS | added
    monkeypatch.setattr(carrier_volumes, "CARRIER_TRANSPORTS", carried)
    offers = [name for name in _PROBED if name != "gloo_available"]
    mismatches = []
    for shm_on, hostname, gloo_on in itertools.product(
        [True, False], [get_local_hostname(), "another-host"], [True, False]
    ):
        for count in range(len(offers) + 1):
            for offered in itertools.combinations(offers, count):
                monkeypatch.setattr(transport, "SHM_ENABLED", shm_on)
                calls = _record_probes(monkeypatch, offered, gloo_on=gloo_on)
                ref = _ref(TransportType.Unset, hostname=hostname)
                outside = [name for name in calls if _PROBED[name] not in carried]
                picked = ref.default_transport_type
                carried_offers = [name for name in offered if _PROBED[name] in carried]
                _record_probes(monkeypatch, carried_offers, gloo_on=gloo_on)
                expected = transport.get_available_transport(_unrestricted(ref))
                if outside or picked is not expected:
                    mismatches.append(
                        f"{shm_on=} {hostname=} {offered=} {gloo_on=}: ran {outside},"
                        f" picked {picked.name} for {expected.name}"
                    )
    assert not mismatches, "\n".join(mismatches)


def test_a_put_and_get_through_carrier_hosts_probe_only_carried_transports(
    children: _Children,
) -> None:
    """Put and get through carrier hosts, automatic and named, probing only ``CARRIER_TRANSPORTS``.

    A fresh child turns RDMA on and loads the Monarch backend where torchmonarch is installed,
    which registers MonarchRDMA's probe: the put and get never call it, and a direct query does.
    """
    child = children.spawn(_PUT_AND_GET_CHILD, _scope())
    report = json.loads(_read_line(child, timeout_s=90))
    assert report["picked"] == ["SharedMemory", "Gloo", "MonarchRPC"]
    assert report["probed"] == []
    assert report["registered"] is report["control"] is report["monarch"]
