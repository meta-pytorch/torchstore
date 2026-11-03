# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Publishers that serve their own volumes on the service backend: lifecycle, errors and faults."""

from __future__ import annotations

import asyncio
import collections
import contextlib
import copy
import gc
import json
import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
import weakref
from collections.abc import AsyncIterator, Callable, Collection, Coroutine, Iterator
from functools import partial
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest
import torchstore as ts
import torchstore.api as store_api
import torchstore.utils as store_utils
from torchstore import carrier_volumes
from torchstore.carrier import (
    CarrierAddress,
    close_all,
    PeerLostError,
    runtime,
    StaleAddressError,
)
from torchstore.carrier_volumes import (
    CarrierVolumeHost,
    decode_volume_record,
    PublisherVolume,
    serve_volume,
    stop_publisher_volumes,
    VolumeProxy,
)
from torchstore.client import LocalClient
from torchstore.routing import tcpstore
from torchstore.routing._model import RankRole
from torchstore.routing.client import RoutingClient
from torchstore.routing.service import (
    CoordinatorService,
    ServiceAddress,
    ServiceControlPlane,
)
from torchstore.state_dict_utils import put_state_dict
from torchstore.storage import StorageVolumeCore
from torchstore.strategy import _installed_volume_host
from torchstore.transport import gloo, TransportType
from torchstore.transport.shared_memory import SharedMemoryCache

from ..test_carrier import _Children, _in_thread, _read_line, _stopped, _until
from ..test_carrier_volume_host import _peers
from ..test_carrier_volumes import _command, _InstrumentedCore, _waiting
from ..test_volume_host import _RecordHost
from . import routed_sync_data as data, tcpstore_fixtures
from .tcpstore_fixtures import (
    _close_each,
    _initialize_service,
    _LOCALHOST,
    _other_service,
)
from .test_record_strategy import _Requester

pytestmark = [
    pytest.mark.timeout(120),
    # A carrier or native-call thread that raises fails its test.
    pytest.mark.filterwarnings("error::pytest.PytestUnhandledThreadExceptionWarning"),
]

# The shared TCPStore fixture, which pytest finds here under this name.
_joins_its_native_calls = tcpstore_fixtures._joins_its_native_calls

_RankClient = Callable[..., Coroutine[Any, Any, LocalClient]]
_TIMING_LINE = re.compile(r"routing control_plane=service ([a-z_]+)( key=[^ ]+)? took")

# One rank of a routing store per process, commanded through stdin. Replies are prefixed
# lines. argv: the tree's root, modules to block, the role, the store name, role arguments.
_RANK = """
import asyncio, json, os, sys, threading, traceback

root, blocked, role, store_name, *rest = sys.argv[1:]
blocked = set(filter(None, blocked.split(",")))
attempts = []


class Blocked:
    # Refuses and records every import of a blocked module, even a guarded one. Monarch, when
    # blocked, gets a None entry instead, so any import of it fails without a record.
    def find_spec(self, name, path=None, target=None):
        if name in blocked:
            attempts.append(name)
            raise ModuleNotFoundError(f"blocked {name}", name=name)


if "monarch" in blocked:
    sys.modules["monarch"] = None
sys.meta_path.insert(0, Blocked())
sys.path.insert(0, root)
import torch
import torchstore as ts
import torchstore.api as store_api
from torch.distributed.tensor import DTensor
from torchstore.carrier import runtime
from torchstore.routing.service import ServiceAddress
from torchstore.storage import InMemoryStore
from torchstore.transport import TransportType
from tests.routing import routed_sync_data as data


def reply(**fields):
    print("reply", json.dumps(fields), flush=True)


def loaded():
    names = (
        "monarch.actor",
        "cloudpickle",
        "torchstore.monarch",
    )
    return [name for name in names if sys.modules.get(name) is not None] + attempts


def servers():
    serving = runtime._runtime.servers if runtime._runtime else ()
    return sorted(server.address.service for server in serving)


async def serve(commands):
    # Each line names a command and its arguments. Every role reports what it loaded.
    commands["loaded"] = lambda: reply(loaded=loaded(), servers=servers())
    while line := await asyncio.to_thread(sys.stdin.readline):
        command, *args = line.split()
        result = commands[command](*args)
        if asyncio.iscoroutine(result):
            await result


async def driver(transport, publishers):
    from tests.routing.tcpstore_fixtures import _initialize_service

    strategy = ts.TorchStoreStrategy(TransportType[transport])
    address = await _initialize_service(store_name, strategy, publishers=int(publishers))
    reply(address=[address.store_name, address.host, address.port, address.nonce])

    async def stop_tcpstore():
        service = store_api._owned_service_map[store_name]
        rendezvous, service._rendezvous = service._rendezvous, None
        await rendezvous.close()
        reply(stopped=True)

    await serve({"stop-tcpstore": stop_tcpstore})


async def client(rank, **kwargs):
    address = ServiceAddress(*json.loads(sys.stdin.readline()))
    await ts.client(store_name, address=address, rank=rank, **kwargs)
    return address


async def publisher(rank, publishers):
    rank, publishers = int(rank), int(publishers)
    try:
        await client(rank, role=ts.RankRole.PUBLISHER)
    except Exception as error:
        reply(error=f"{type(error).__name__}: {error}", servers=servers())
        return
    reply(pid=os.getpid(), servers=servers())
    held, release = threading.Event(), threading.Event()
    get = InMemoryStore.get

    async def held_get(self, buffer, requests):
        # Holds every get but the mapping's, on the volume's own thread. The served
        # core bound its methods when it was served, so its store's is patched.
        if not any(request.key.endswith("MAPPING") for request in requests):
            held.set()
            await asyncio.to_thread(release.wait, 60)
        return await get(self, buffer, requests)

    async def push(round_):
        weights = data.weights(int(round_))
        state_dict = data.place(weights, data.PUBLISHER_DIMS, rank, publishers)
        put = ts.put_state_dict(state_dict, data.KEY, store_name=store_name)
        # A worker thread with a fresh event loop per push, as in TitanRL's trainer.
        await asyncio.to_thread(lambda: asyncio.run(put))
        reply(pushed=int(round_))

    def hold_gets():
        InMemoryStore.get = held_get
        reply(holding=True)

    async def wait_held():
        reply(held=await asyncio.to_thread(held.wait, 60))

    def release_gets():
        release.set()
        reply(released=True)

    await serve(
        {
            "push": push,
            "hold-gets": hold_gets,
            "wait-held": wait_held,
            "release": release_gets,
        }
    )


async def requester(group, rank):
    group, rank = int(group), int(rank)
    role = ts.RankRole.REQUESTER
    address = await client(rank, role=role, group=group)
    num_ranks, dims = data.REQUESTERS[group]
    zeros = {name: torch.zeros_like(t) for name, t in data.weights(0).items()}
    destination = data.place(zeros, dims, rank, num_ranks)
    destination[data.DESTINATION_ONLY] = data.sentinel()
    reply(pid=os.getpid())

    async def pull(round_):
        expected = data.weights(int(round_))
        try:
            result = await ts.get_state_dict(
                data.KEY, destination, strict=False, store_name=store_name
            )
            assert sorted(result) == sorted(expected)
            for name, tensor in expected.items():
                held = destination[name]
                held = held.to_local() if isinstance(held, DTensor) else held
                local = data.local(tensor, dims[name], rank, num_ranks)
                data.assert_bitwise_equal(held, local, name)
            assert torch.equal(destination[data.DESTINATION_ONLY], data.sentinel())
        except Exception as error:
            reply(error="".join(traceback.format_exception_only(type(error), error)))
        else:
            reply(pulled=int(round_))

    async def reset():
        # A new client, with a new strategy and volume host, as a re-created rank makes.
        ts.reset_client(store_name)
        await ts.client(store_name, role=role, group=group, rank=rank, address=address)
        reply(reset=True)

    async def shutdown():
        await ts.shutdown(store_name)
        reply(loaded=loaded(), servers=servers())

    await serve({"pull": pull, "reset": reset, "shutdown": shutdown})


roles = {"driver": driver, "publisher": publisher, "requester": requester}
asyncio.run(roles[role](*rest))
"""
# The modules every rank child blocks, unless a case names its own.
_BLOCKED = "monarch,cloudpickle,torchstore.monarch.storage_volume"


def _store_name() -> str:
    return f"service-volumes-{uuid.uuid4()}"


def _fields(address: ServiceAddress) -> str:
    return json.dumps([address.store_name, address.host, address.port, address.nonce])


def _reply(child: subprocess.Popen, timeout_s: float = 60) -> dict[str, Any]:
    """The child's next reply, skipping any other line it prints."""
    while True:
        line = _read_line(child, timeout_s)
        if line.startswith("reply "):
            return json.loads(line[len("reply ") :])


def _rank(
    children: _Children, role: str, store_name: str, *args: Any, blocked: str = _BLOCKED
) -> subprocess.Popen:
    return children.spawn(_RANK, blocked, role, store_name, *map(str, args))


def _publisher(
    children: _Children, address: ServiceAddress, rank: int, publishers: int
) -> subprocess.Popen:
    child = _rank(children, "publisher", address.store_name, rank, publishers)
    _command(child, _fields(address))
    return child


def _requester(
    children: _Children, address: ServiceAddress, group: int = 0, rank: int = 0
) -> subprocess.Popen:
    child = _rank(children, "requester", address.store_name, group, rank)
    _command(child, _fields(address))
    assert "pid" in _reply(child, timeout_s=90)
    return child


def _driver(
    children: _Children, store_name: str, transport: str, publishers: int
) -> tuple[subprocess.Popen, ServiceAddress]:
    child = _rank(children, "driver", store_name, transport, publishers)
    return child, ServiceAddress(*_reply(child, timeout_s=90)["address"])


def _push(child: subprocess.Popen, round_: int) -> None:
    _command(child, f"push {round_}")
    assert _reply(child) == {"pushed": round_}


def _pull(child: subprocess.Popen, round_: int) -> None:
    _command(child, f"pull {round_}")
    assert _reply(child) == {"pulled": round_}


def _state_dict(round_: int, rank: int = 0, publishers: int = 1) -> dict[str, Any]:
    return data.place(data.weights(round_), data.PUBLISHER_DIMS, rank, publishers)


def _requester_of(client: LocalClient, group: int = 0, rank: int = 0) -> _Requester:
    """The requester rank that reads through a routing client made in this process."""
    assert isinstance(client, RoutingClient)
    return _Requester(group, rank, client)


@contextlib.asynccontextmanager
async def _reader(
    address: ServiceAddress, group: int = 0, rank: int = 0
) -> AsyncIterator[_Requester]:
    """A requester in this process, which a publisher client here leaves to ``RoutingClient``."""
    handle = await address.connect()
    closes = [handle.close]
    try:
        strategy = await handle.strategy()
        host = _installed_volume_host(strategy)
        assert isinstance(host, CarrierVolumeHost)
        closes.insert(0, host.close)
        name = f"requester/{group}/{rank}"
        client = RoutingClient(name, RankRole.REQUESTER, handle, strategy)
        yield _Requester(group, rank, client)
    finally:
        # Every close runs, and a failing case's own error stays the one reported.
        await _close_each(closes, sys.exc_info()[1] is not None)


def _volume(store_name: str, volume_id: str = "publisher/0") -> PublisherVolume:
    return carrier_volumes._publishers[(store_name, volume_id)]


def _serve_lines(caplog: pytest.LogCaptureFixture, volume_id: str) -> list[str]:
    prefix = f"carrier serve service={volume_id} "
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith(prefix)]


def _timed(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        match.group(1)
        for record in caplog.records
        if (match := _TIMING_LINE.search(record.getMessage()))
    ]


def _servers(store_name: str) -> list[str]:
    """The services of this process's live carrier servers for a store, its volumes' and coordinator's."""
    serving = runtime._runtime.servers if runtime._runtime else set()
    return sorted(
        server.address.service
        for server in serving
        if server.address.scope.startswith(f"{store_name}/")
    )


def _gloo_groups(volume: PublisherVolume) -> int:
    """How many Gloo pairs the volume's side holds."""
    core = volume._core
    assert core is not None
    groups = core.store.transport_context.get(gloo.GlooProcessGroupCache)
    return len(groups._process_groups)


def _publisher_servers(store_name: str) -> list[str]:
    return [
        service for service in _servers(store_name) if service.startswith("publisher/")
    ]


def _volume_peers() -> list[CarrierAddress]:
    """This process's connections to ``publisher/0``, which close with their last handle."""
    return [address for address in _peers() if address.service == "publisher/0"]


@pytest.fixture
def start_waits(monkeypatch: pytest.MonkeyPatch) -> mock.AsyncMock:
    """Count the callers that wait on another caller's start of a volume as its ``await_count``."""
    waits = mock.AsyncMock(wraps=carrier_volumes.wait_future)
    monkeypatch.setattr(carrier_volumes, "wait_future", waits)
    return waits


@pytest.fixture(autouse=True)
def _local_servers(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Serve on localhost, give each case its own publisher volumes, then close what it left."""
    monkeypatch.setenv("TORCHSTORE_TCPSTORE_HOST", _LOCALHOST)
    monkeypatch.setenv("TORCHSTORE_CARRIER_HOST", _LOCALHOST)
    monkeypatch.setattr(carrier_volumes, "_publishers", {})
    # Bound Gloo's waits here and in rank children, as the carrier volume tests do.
    monkeypatch.setattr(gloo, "TORCHSTORE_GLOO_INIT_TIMEOUT", 10)
    monkeypatch.setenv("TORCHSTORE_GLOO_INIT_TIMEOUT", "10")
    yield
    try:
        _in_thread(close_all(grace_s=0))
    finally:
        # Tests share this process, and Gloo caches pairs per volume ID.
        gloo._forget_pairs(list(gloo._store_addrs))


@pytest.mark.parametrize(
    "transport_type", [TransportType.MonarchRPC, TransportType.Gloo], ids=str
)
@pytest.mark.asyncio
async def test_a_publisher_client_made_again_adopts_its_running_volume(
    children: _Children,
    caplog: pytest.LogCaptureFixture,
    rank_client: _RankClient,
    transport_type: TransportType,
) -> None:
    """Adopt the running volume after ``reset_client``: one server, one record, one registration.

    The new client and a requester remade in another process pair afresh, and it reads bitwise.
    The volume drops each old pair at its next call, and keeps the requester's after its shutdown.
    """
    caplog.set_level(logging.INFO)
    store_name = _store_name()
    address = await _initialize_service(
        store_name, ts.TorchStoreStrategy(transport_type)
    )
    try:
        first = await rank_client(store_name, role=RankRole.PUBLISHER)
        volume = _volume(store_name)
        record = volume.record
        reader = await asyncio.to_thread(_requester, children, address)
        await put_state_dict(first, _state_dict(0), data.KEY)
        await asyncio.to_thread(_pull, reader, 0)

        ts.reset_client(store_name)
        second = await rank_client(store_name, role=RankRole.PUBLISHER)
        assert second is not first and _volume(store_name) is volume
        assert volume.record == record and volume.current()
        host = _installed_volume_host(second.strategy)
        assert host is not None and list(host.volume_ids()) == ["publisher/0"]
        await put_state_dict(second, _state_dict(1), data.KEY)
        pairs = [_gloo_groups(volume)]
        _command(reader, "reset")
        assert await asyncio.to_thread(_reply, reader) == {"reset": True}
        await asyncio.to_thread(_pull, reader, 1)
        pairs.append(_gloo_groups(volume))
        # Over Gloo, one pair for each live client: the new publisher's and the requester's.
        assert pairs == ([2, 2] if transport_type is TransportType.Gloo else [0, 0])
        _command(reader, "shutdown")
        await asyncio.to_thread(_reply, reader)
        assert _gloo_groups(volume) == pairs[-1]
    finally:
        await ts.shutdown(store_name)
    assert len(_serve_lines(caplog, "publisher/0")) == 1
    assert _timed(caplog).count("register_volume") == 1


@pytest.mark.parametrize(
    ("first", "again"),
    [
        ("handle", "same_handle"),
        ("handle", "handle"),
        ("address", "handle"),
        ("handle", "address"),
        ("record", "handle"),
        ("unaddressed", "same_handle"),
    ],
)
@pytest.mark.asyncio
async def test_a_publisher_client_made_again_on_any_handle_to_its_store_adopts_its_volume(
    caplog: pytest.LogCaptureFixture, rank_client: _RankClient, first: str, again: str
) -> None:
    """Adopt the running volume when the client is made again through another handle to its store.

    A caller's handle, a dial and this process's record name one store, by the selection's address,
    else its plane's, or else the plane itself. A reader then reads the volume bitwise.
    """
    caplog.set_level(logging.INFO)
    store_name = _store_name()
    async with contextlib.AsyncExitStack() as stack:
        if first == "record":
            address = await _initialize_service(store_name)
        else:
            # A store that this process neither initialized nor records yet.
            other = _other_service(store_name, ts.TorchStoreStrategy())
            address = (await stack.enter_async_context(other)).address
        stack.push_async_callback(ts.shutdown, store_name)

        async def selection(kind: str) -> dict[str, Any]:
            if kind == "record":
                return {}
            if kind == "address":
                return {"address": address}
            handle = await address.connect()
            stack.push_async_callback(handle.close)
            if kind == "unaddressed":
                # A caller's plane that shares records need not name an address.
                del handle.address
            return {"control_plane": handle}

        kwargs = await selection(first)
        built = await rank_client(store_name, role=RankRole.PUBLISHER, **kwargs)
        volume = _volume(store_name)
        ts.reset_client(store_name)
        if again != "same_handle":
            kwargs = await selection(again)
        rebuilt = await rank_client(store_name, role=RankRole.PUBLISHER, **kwargs)
        assert rebuilt is not built and _volume(store_name) is volume
        assert volume.current()
        await put_state_dict(rebuilt, _state_dict(0), data.KEY)
        async with _reader(address) as reader:
            reader.check(0, await reader.get())
    assert len(_serve_lines(caplog, "publisher/0")) == 1
    assert _timed(caplog).count("register_volume") == 1


@pytest.mark.parametrize("addressed", [True, False])
@pytest.mark.asyncio
async def test_a_publisher_of_another_store_with_its_name_stops_the_old_volume_before_serving(
    monkeypatch: pytest.MonkeyPatch, rank_client: _RankClient, addressed: bool
) -> None:
    """Stop the volume served for a store another driver replaced, before serving the new store's.

    Both stores have the name, and this process initialized neither, so only the start can tell:
    by their addresses, or by the planes it came through when neither names an address.
    """
    store_name = _store_name()
    serve = carrier_volumes.serve_volume
    old: list[PublisherVolume] = []
    old_served: list[bool] = []

    async def recording_serve(*args: Any, **kwargs: Any) -> Any:
        old_served.append(bool(old) and old[0]._server is not None)
        return await serve(*args, **kwargs)

    monkeypatch.setattr(carrier_volumes, "serve_volume", recording_serve)
    async with contextlib.AsyncExitStack() as stack:
        services, selections = [], []
        for _ in range(2):
            other = _other_service(store_name, ts.TorchStoreStrategy())
            services.append(await stack.enter_async_context(other))
            if addressed:
                selections.append({"address": services[-1].address})
            else:
                handle = await services[-1].connect()
                stack.push_async_callback(handle.close)
                del handle.address
                selections.append({"control_plane": handle})
        stack.push_async_callback(ts.shutdown, store_name)
        await rank_client(store_name, role=RankRole.PUBLISHER, **selections[0])
        old.append(_volume(store_name))
        ts.reset_client(store_name)
        await rank_client(store_name, role=RankRole.PUBLISHER, **selections[1])
        assert old_served == [False, False]
        assert _volume(store_name) is not old[0]
        assert not old[0].current() and old[0]._server is None
        assert _publisher_servers(store_name) == ["publisher/0"]
        # The new store holds the new volume's record.
        reader = await services[1].connect()
        stack.push_async_callback(reader.close)
        records = await reader.volume_records(["publisher/0"])
        assert records == {"publisher/0": _volume(store_name).record}


@pytest.mark.asyncio
async def test_one_stores_shutdown_leaves_another_store_serving(
    rank_client: _RankClient,
) -> None:
    """Leave another store's volume serving, and its dropped client's handles open, in this process.

    Both stores' publishers here have one ID. Only the first store shuts down.
    """
    first, second = _store_name(), _store_name()
    await _initialize_service(first)
    address = await _initialize_service(second)
    try:
        await rank_client(first, role=RankRole.PUBLISHER)
        held = await rank_client(second, role=RankRole.PUBLISHER)
        kept = _volume(second)
        ts.reset_client(second)
        await ts.shutdown(first)
        assert kept.current() and kept._server is not None
        await put_state_dict(held, _state_dict(0), data.KEY)
        async with _reader(address) as reader:
            reader.check(0, await reader.get())
    finally:
        for store_name in (first, second):
            await store_api._shutdown_selected(store_name)


@pytest.mark.asyncio
async def test_a_second_publisher_for_a_volume_id_is_refused_and_closes_its_server(
    children: _Children, rank_client: _RankClient
) -> None:
    """Refuse another process's publisher for an ID this one registered. Its server closes."""
    store_name = _store_name()
    address = await _initialize_service(store_name)
    try:
        await rank_client(store_name, role=RankRole.PUBLISHER)
        refused = await asyncio.to_thread(
            _reply, _publisher(children, address, 0, 1), 90
        )
        assert refused["servers"] == []
        assert refused["error"].startswith("RuntimeError: ")
        registered = "volume 'publisher/0' is already registered with another record"
        assert registered in refused["error"]
        assert _volume(store_name).current()
    finally:
        await ts.shutdown(store_name)


@pytest.mark.asyncio
async def test_a_publisher_outside_the_stores_publishers_is_refused_and_closes_its_server(
    rank_client: _RankClient,
) -> None:
    """Refuse a publisher whose rank the store does not list with a KeyError. Its server closes."""
    store_name = _store_name()
    await _initialize_service(store_name)
    try:
        with pytest.raises(KeyError, match="'publisher/1' is not a publisher rank"):
            await rank_client(store_name, rank=1, role=RankRole.PUBLISHER)
        assert carrier_volumes._publishers == {}
        assert _publisher_servers(store_name) == []
    finally:
        await ts.shutdown(store_name)


@pytest.mark.parametrize(
    "lost", [TimeoutError, ConnectionError], ids=["timeout", "connection"]
)
@pytest.mark.asyncio
async def test_a_registration_that_times_out_or_loses_its_connection_closes_its_volume(
    monkeypatch: pytest.MonkeyPatch, rank_client: _RankClient, lost: type[OSError]
) -> None:
    """Close the volume of a registration that timed out or lost its connection, and raise that error.

    Its write may still land, but no retry reuses the record, so no entry or server stays.
    """

    async def lose(self: ServiceControlPlane, volume_id: str, record: bytes) -> None:
        raise lost(f"registering {volume_id!r} lost its wait")

    monkeypatch.setattr(ServiceControlPlane, "register_volume", lose)
    store_name = _store_name()
    await _initialize_service(store_name)
    try:
        with pytest.raises(lost, match="lost its wait"):
            await rank_client(store_name, role=RankRole.PUBLISHER)
        assert carrier_volumes._publishers == {}
        assert _publisher_servers(store_name) == []
    finally:
        await ts.shutdown(store_name)


@pytest.mark.asyncio
async def test_a_store_after_a_shut_down_one_with_the_same_ids_syncs_over_gloo(
    children: _Children, rank_client: _RankClient
) -> None:
    """Sync a Gloo store, shut it down, then sync another with the same IDs from this process.

    Shutdown forgets the client's pairs and lets go of their groups.
    """
    for round_ in range(2):
        store_name = _store_name()
        address = await _initialize_service(
            store_name, ts.TorchStoreStrategy(TransportType.Gloo)
        )
        try:
            publisher = await rank_client(store_name, role=RankRole.PUBLISHER)
            reader = await asyncio.to_thread(_requester, children, address)
            await put_state_dict(publisher, _state_dict(round_), data.KEY)
            await asyncio.to_thread(_pull, reader, round_)
            assert "publisher/0" in gloo._store_addrs
        finally:
            await ts.shutdown(store_name)
        assert stop_publisher_volumes(store_name) == []
        assert "publisher/0" not in gloo._store_addrs
        groups = publisher.strategy.transport_context.get(gloo.GlooProcessGroupCache)
        assert groups._process_groups == {}


@pytest.mark.asyncio
async def test_a_host_forgets_a_stale_gloo_pair_before_a_volume_first_resolves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forget the process's Gloo pair to a volume before a host's first adopt or read shows it.

    So no lookup meets the stale pair and its volume half retires. A pair cached once it resolves is
    the host's own and stays.
    """
    server, record = await serve_volume(
        StorageVolumeCore("publisher/0"), scope=f"{_store_name()}/scope"
    )
    hosts: list[CarrierVolumeHost] = []
    forgotten: list[tuple[str, bool, bool]] = []
    forget = gloo.forget_unheld_pair

    def recording_forget(volume_id: str) -> None:
        if hosts:
            # The forget runs before the install and outside the host's lock.
            last = hosts[-1]
            forgotten.append(
                (volume_id, volume_id in last._entries, last._lock.locked())
            )
        forget(volume_id)

    async def records(volume_ids: Collection[str]) -> dict[str, bytes]:
        return {volume_id: record for volume_id in volume_ids}

    monkeypatch.setattr(gloo, "forget_unheld_pair", recording_forget)
    monkeypatch.setattr(carrier_volumes, "_retired", {})
    monkeypatch.setattr(carrier_volumes, "_retiring", collections.deque())
    monkeypatch.setattr(carrier_volumes, "_pair_servers", {})
    try:
        for first in ("adopt", "prepare"):
            host = CarrierVolumeHost(records)
            hosts.append(host)
            gloo._store_addrs["publisher/0"] = ("stale", 1, f"{first}-stale", True)
            # Recorded as a proxy's handshake records it so that the stale key retires to that server.
            carrier_volumes._pair_servers[f"{first}-stale"] = server.address
            if first == "adopt":
                host.adopt("publisher/0", record)
            else:
                await host.prepare(["publisher/0"])
            assert "publisher/0" not in gloo._store_addrs
            own = ("own", 2, f"{first}-own", True)
            gloo._store_addrs["publisher/0"] = own
            host.adopt("publisher/0", record)
            await host.prepare(["publisher/0"])
            assert gloo._store_addrs["publisher/0"] == own
    finally:
        for host in hosts:
            await host.close()
    assert forgotten == [("publisher/0", False, False)] * 2
    assert _waiting() == {server.address: ["adopt-stale", "prepare-stale"]}


@pytest.mark.asyncio
async def test_calls_to_a_shut_down_volumes_record_fail(
    rank_client: _RankClient,
) -> None:
    """Fail a call through the record of a volume that ``shutdown`` stopped, naming its host:port."""
    store_name = _store_name()
    await _initialize_service(store_name)
    try:
        await rank_client(store_name, role=RankRole.PUBLISHER)
        record = _volume(store_name).record
        assert record is not None
    finally:
        await ts.shutdown(store_name)
    address = decode_volume_record("publisher/0", record).address
    proxy = VolumeProxy(address)
    try:
        start = time.monotonic()
        where = re.escape(f"{address.host}:{address.port}")
        with pytest.raises((PeerLostError, StaleAddressError), match=where):
            await proxy.get_id.call_one()
        assert time.monotonic() - start < 10
    finally:
        await proxy.close()


@pytest.mark.asyncio
async def test_a_publisher_volume_serves_under_a_fresh_nonce_scope_its_only_credential(
    rank_client: _RankClient,
) -> None:
    """Serve each publisher volume under its store's name and a fresh nonce, refusing a dial that guesses it.

    A peer that knows the store, the volume ID, the port and the incarnation still lacks the scope.
    """
    store_name = _store_name()
    scopes: list[str] = []
    for _ in range(2):
        await _initialize_service(store_name)
        try:
            await rank_client(store_name, role=RankRole.PUBLISHER)
            served = _volume(store_name).address
            assert served is not None
            assert re.fullmatch(
                rf"{re.escape(store_name)}/[0-9a-f]{{32}}", served.scope
            )
            scopes.append(served.scope)
            guessed = f"{store_name}/{served.service}"
            proxy = VolumeProxy(
                CarrierAddress(
                    served.host,
                    served.port,
                    served.service,
                    guessed,
                    served.incarnation,
                )
            )
            try:
                with pytest.raises(StaleAddressError, match="the scope does not match"):
                    await proxy.get_id.call_one()
            finally:
                await proxy.close()
        finally:
            await ts.shutdown(store_name)
    assert scopes[0] != scopes[1]


@pytest.mark.parametrize(
    "case", ["idle", "finished", "outlived", "cancelled", "failed"]
)
@pytest.mark.asyncio
async def test_shutdown_resets_the_core_only_once_no_handler_runs(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    rank_client: _RankClient,
    case: str,
) -> None:
    """Reset the core once its last call ends, idle or within the grace, and the caller's put lands.

    A put that outlives the grace, a shutdown cancelled in it and a failing server close, which
    shutdown raises, leave the core unreset. No calls overlap, and shutdown ends within 11 s.
    """
    caplog.set_level(logging.WARNING, logger=carrier_volumes.__name__)
    # Shutdown gives a running call 30 s by default, which the finished put runs on.
    assert carrier_volumes._VOLUME_CLOSE_GRACE_S == 30
    if case != "finished":
        monkeypatch.setattr(carrier_volumes, "_VOLUME_CLOSE_GRACE_S", 1)
    cores: list[_InstrumentedCore] = []

    def core(volume_id: str) -> _InstrumentedCore:
        cores.append(_InstrumentedCore(volume_id=volume_id))
        return cores[-1]

    monkeypatch.setattr(carrier_volumes, "StorageVolumeCore", core)
    store_name = _store_name()
    await _initialize_service(store_name)
    service = store_api._owned_service_map[store_name]
    handle = store_api._control_plane_map[store_name].control_plane
    assert isinstance(handle, ServiceControlPlane)
    try:
        publisher = await rank_client(store_name, role=RankRole.PUBLISHER)
        await put_state_dict(publisher, _state_dict(0), data.KEY)
        [made] = cores
        server = _volume(store_name)._server
        assert server is not None
        if case == "failed":
            close = server.close

            async def failing_close(grace_s: float = 30) -> None:
                await close(grace_s=grace_s)
                raise OSError("volume server close failed")

            monkeypatch.setattr(server, "close", failing_close)
        pushing: asyncio.Future[Any] | None = None
        if case not in ("idle", "failed"):
            made.held.add("put")
            norm = {f"{data.KEY}/norm.weight": data.weights(1)["norm.weight"]}
            pushing = asyncio.ensure_future(publisher.put_batch(norm))
            await _until(lambda: made.active == 1)
        start = time.monotonic()
        stopping = asyncio.ensure_future(ts.shutdown(store_name))
        if case in ("finished", "cancelled"):
            await _until(lambda: server._closing)
        if case == "finished":
            made.release.set()
        if case == "cancelled":
            stopping.cancel()
            with pytest.raises(asyncio.CancelledError):
                await stopping
        elif case == "failed":
            with pytest.raises(OSError, match="volume server close failed"):
                await stopping
        else:
            await stopping
        assert time.monotonic() - start < 1 + 10
        # Every later close ran: the connection, the service and its TCPStore. The volume
        # server's own shutdown, which a cancel leaves running, ends once its call returns or
        # its grace passes.
        assert handle._closed
        assert (service._server, service._rendezvous) == (None, None)
        await _until(lambda: _servers(store_name) == [], timeout_s=10)
        if case == "finished":
            # The running call ended within the grace, so its put landed.
            assert pushing is not None
            await pushing
        elif pushing is not None:
            assert made.active == 1 and "reset" not in made.names()
            made.release.set()
            # A cancelled close goes on to close this client's handles at once.
            closed = "volume closed" if case == "outlived" else "last client handle"
            with pytest.raises(PeerLostError, match=closed):
                await pushing
    finally:
        for held in cores:
            held.release.set()
        await store_api._shutdown_selected(store_name)
    assert made.max_active == 1
    assert ("reset" in made.names()) is (case in ("idle", "finished"))
    why = {
        "outlived": "handler outlived the grace",
        "cancelled": "close was cancelled",
        "failed": "close failed: OSError('volume server close failed')",
    }
    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.name == carrier_volumes.__name__
    ]
    expected = [
        f"routing volume=publisher/0 {why.get(case)}, so its core stays unreset"
    ]
    assert warnings == (expected if case in why else [])


@pytest.mark.asyncio
async def test_shutdown_runs_every_later_close_even_if_the_volumes_close_fails(
    monkeypatch: pytest.MonkeyPatch, rank_client: _RankClient
) -> None:
    """Close the client's handles, the connection and the service even if the volume's close fails.

    The raised close error keeps the error its caller was handling as its context.
    """
    close = PublisherVolume.close

    async def failing_close(self: PublisherVolume) -> None:
        await close(self)
        raise OSError("volume close failed")

    monkeypatch.setattr(PublisherVolume, "close", failing_close)
    store_name = _store_name()
    await _initialize_service(store_name)
    service = store_api._owned_service_map[store_name]
    handle = store_api._control_plane_map[store_name].control_plane
    assert isinstance(handle, ServiceControlPlane)
    try:
        publisher = await rank_client(store_name, role=RankRole.PUBLISHER)
        with pytest.raises(OSError, match="volume close failed") as raised:
            try:
                raise ValueError("the caller's own error")
            except ValueError:
                await ts.shutdown(store_name)
        # Chained to the error the caller was handling, like a raise outside AsyncExitStack.
        assert type(raised.value.__context__) is ValueError
        with pytest.raises(RuntimeError, match="its carrier volume host is closed"):
            publisher.strategy.get_storage_volume("publisher/0")
        assert handle._closed
        assert (service._server, service._rendezvous) == (None, None)
        assert _servers(store_name) == []
    finally:
        await store_api._shutdown_selected(store_name)


@pytest.mark.parametrize("via", ["address", "handle", "failing_host_close"])
@pytest.mark.asyncio
async def test_a_shutdown_while_a_publisher_client_builds_leaves_no_volume(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    rank_client: _RankClient,
    via: str,
) -> None:
    """Fail a build on a store that shuts down while its volume serves. No server, record or client stays.

    The error names the store's address even through a caller's handle whose address it reads. A failing
    close of the build's volume handles only logs. A dial closes and the store serves on.
    """
    caplog.set_level(logging.WARNING, logger=store_api.__name__)
    reached, release = asyncio.Event(), asyncio.Event()
    serve, read = carrier_volumes.serve_volume, ServiceControlPlane.strategy

    async def held_after_serve(*args: Any, **kwargs: Any) -> Any:
        result = await serve(*args, **kwargs)
        reached.set()
        await release.wait()
        return result

    store_name = _store_name()
    async with contextlib.AsyncExitStack() as stack:
        # A store that this process neither initialized nor records.
        other = await stack.enter_async_context(
            _other_service(store_name, ts.TorchStoreStrategy())
        )
        selection: dict[str, Any] = {"address": other.address}
        if via == "handle":
            handle = await other.address.connect()
            stack.push_async_callback(handle.close)
            selection = {"control_plane": handle}
        dials: list[ServiceControlPlane] = []
        connect = ServiceAddress.connect

        async def recorded_connect(self: ServiceAddress) -> ServiceControlPlane:
            dials.append(await connect(self))
            return dials[-1]

        async def failing_host(self: ServiceControlPlane) -> ts.TorchStoreStrategy:
            strategy = await read(self)
            host = _installed_volume_host(strategy)
            assert isinstance(host, CarrierVolumeHost)
            close = host.close

            async def close_then_fail() -> None:
                await close()
                raise OSError("host close failed")

            monkeypatch.setattr(host, "close", close_then_fail)
            return strategy

        # Only the build's own dials, so the setup's connections above are not counted.
        monkeypatch.setattr(ServiceAddress, "connect", recorded_connect)
        monkeypatch.setattr(carrier_volumes, "serve_volume", held_after_serve)
        if via == "failing_host_close":
            monkeypatch.setattr(ServiceControlPlane, "strategy", failing_host)
        building = asyncio.ensure_future(
            rank_client(store_name, role=RankRole.PUBLISHER, **selection)
        )
        try:
            await asyncio.wait_for(reached.wait(), 30)
            assert await store_api._shutdown_selected(store_name)
            release.set()
            message = (
                f"routing store '{store_name}' on {other.address} was shut down or replaced "
                "while this client was being built. Call ts.client again, or restart the "
                "store if it refuses the volume as already registered"
            )
            with pytest.raises(RuntimeError, match=f"^{re.escape(message)}$"):
                await asyncio.wait_for(building, 30)
        finally:
            release.set()
            await store_api._shutdown_selected(store_name)
        assert stop_publisher_volumes(store_name) == []
        # No connection to the volume stays, nor to the store but a caller's own handle.
        assert _volume_peers() == []
        assert len(dials) == (0 if via == "handle" else 1)
        assert all(dial._closed for dial in dials)
        warned = [
            r.getMessage() for r in caplog.records if r.name == store_api.__name__
        ]
        failed = ["closing a discarded client's volume handles failed"]
        assert warned == (failed if via == "failing_host_close" else [])
        # This process's shutdown left the store it does not own serving.
        assert _servers(store_name) == ["coordinator"]
        assert store_name not in store_api._control_plane_map
        assert store_name not in store_api._local_clent_map
        answering = await other.connect()
        stack.push_async_callback(answering.close)
        await answering.strategy()


@pytest.mark.parametrize("host", ["none", "another"])
@pytest.mark.asyncio
async def test_a_plane_sharing_records_must_resolve_volumes_through_a_carrier_host(
    monkeypatch: pytest.MonkeyPatch, rank_client: _RankClient, host: str
) -> None:
    """Refuse a publisher whose plane shares records but resolves volumes otherwise. Serve nothing.

    Its strategy holds no host, or a volume host of another kind, which the error names.
    """
    store_name = _store_name()

    async def strategy_without_a_carrier_host() -> ts.TorchStoreStrategy:
        strategy = ts.TorchStoreStrategy()
        if host == "another":
            strategy.set_volume_host(_RecordHost({}))
        return strategy

    async with _other_service(store_name, ts.TorchStoreStrategy()) as service:
        handle = await service.address.connect()
        try:
            monkeypatch.setattr(handle, "strategy", strategy_without_a_carrier_host)
            held = "NoneType" if host == "none" else "_RecordHost"
            with pytest.raises(TypeError, match=f"a CarrierVolumeHost, not {held}$"):
                await rank_client(
                    store_name, role=RankRole.PUBLISHER, control_plane=handle
                )
            assert stop_publisher_volumes(store_name) == []
            assert _servers(store_name) == ["coordinator"]
            assert store_name not in store_api._control_plane_map
        finally:
            await handle.close()


def _race(
    rank_client: _RankClient,
    store_name: str,
    ready: threading.Barrier,
    **kwargs: Any,
) -> Any:
    """Build the store's publisher client on this thread's own loop. Return it or its error."""

    async def build() -> LocalClient:
        return await rank_client(store_name, role=RankRole.PUBLISHER, **kwargs)

    try:
        ready.wait()
        return asyncio.run(build())
    except BaseException as error:
        return error


class _GatedLock:
    """The publisher lock, except that no release returns before ``count`` releases have happened."""

    def __init__(self, lock: threading.Lock, count: int) -> None:
        self._lock = lock
        self._count = count
        self._released = 0
        self._changed = threading.Condition()

    def __enter__(self) -> None:
        self._lock.acquire()

    def __exit__(self, *exc: object) -> None:
        self._lock.release()
        with self._changed:
            self._released += 1
            self._changed.notify_all()
            released = self._changed.wait_for(lambda: self._released >= self._count, 30)
        assert released, "the other releases did not come within 30 s"


@pytest.mark.asyncio
async def test_threads_racing_a_first_publisher_client_start_one_server(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    rank_client: _RankClient,
    start_waits: mock.AsyncMock,
) -> None:
    """Serve one volume for 4 threads, each in its own loop, racing the first publisher client.

    Each looks for a start before any goes on, so one starts it and the other three wait on it.
    """
    caplog.set_level(logging.INFO, logger="torchstore.carrier")
    store_name = _store_name()
    await _initialize_service(store_name)
    gate = _GatedLock(carrier_volumes._publisher_lock, 4)
    monkeypatch.setattr(carrier_volumes, "_publisher_lock", gate)
    try:
        ready = threading.Barrier(4, timeout=30)
        clients = await asyncio.gather(
            *(
                asyncio.to_thread(_race, rank_client, store_name, ready)
                for _ in range(4)
            )
        )
        assert all(isinstance(c, RoutingClient) for c in clients), clients
        assert _publisher_servers(store_name) == ["publisher/0"]
        assert start_waits.await_count == 3
    finally:
        await ts.shutdown(store_name)
    assert len(_serve_lines(caplog, "publisher/0")) == 1


@pytest.mark.asyncio
async def test_first_requester_clients_racing_through_one_address_all_read_bitwise(
    children: _Children, monkeypatch: pytest.MonkeyPatch, rank_client: _RankClient
) -> None:
    """Read bitwise through every first requester client racing through a copy of one address.

    They share one connection and client, and the other racer closes the volume handles of the strategy it read.
    A later call returns that client and shutdown closes its handles.
    """
    store_name = _store_name()
    read = ServiceControlPlane.strategy
    strategies: list[ts.TorchStoreStrategy] = []

    async def recorded(self: ServiceControlPlane) -> ts.TorchStoreStrategy:
        strategies.append(await read(self))
        return strategies[-1]

    async with contextlib.AsyncExitStack() as stack:
        # A store that this process neither initialized nor records yet.
        strategy = ts.TorchStoreStrategy(TransportType.MonarchRPC)
        service = await stack.enter_async_context(_other_service(store_name, strategy))
        stack.push_async_callback(store_api._shutdown_selected, store_name)
        address = service.address
        publisher = _publisher(children, address, 0, 1)
        assert "pid" in await asyncio.to_thread(_reply, publisher, 90)
        await asyncio.to_thread(_push, publisher, 0)
        monkeypatch.setattr(ServiceControlPlane, "strategy", recorded)
        clients = await asyncio.gather(
            *(
                rank_client(
                    store_name,
                    role=RankRole.REQUESTER,
                    group=0,
                    address=copy.copy(address),
                )
                for _ in range(2)
            )
        )
        [client] = set(clients)
        handle = store_api._control_plane_map[store_name].control_plane
        assert isinstance(client, RoutingClient) and client._control_plane is handle
        [discarded] = [other for other in strategies if other is not client.strategy]
        host = _installed_volume_host(discarded)
        assert isinstance(host, CarrierVolumeHost) and host._closed
        again = await rank_client(
            store_name, role=RankRole.REQUESTER, group=0, address=address
        )
        assert again is client
        reader = _requester_of(client)
        reader.check(0, await reader.get())
        # Shutdown closes the shared client's handles, and no connection to the volume stays.
        await store_api._shutdown_selected(store_name)
        with pytest.raises(RuntimeError, match="its carrier volume host is closed"):
            client.strategy.get_storage_volume("publisher/0")
        assert _volume_peers() == []


@pytest.mark.parametrize("outcome", ["cancelled", "refused"])
@pytest.mark.asyncio
async def test_threads_waiting_on_a_failed_start_raise_at_once_and_a_later_call_serves(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    rank_client: _RankClient,
    start_waits: mock.AsyncMock,
    outcome: str,
) -> None:
    """Fail the 3 threads waiting on a start that is cancelled or refused while it registers.

    They raise RuntimeError, never the starter's own error. Either way the start's server closes
    and a later call starts one.
    """
    caplog.set_level(logging.INFO, logger="torchstore.carrier")
    registering, release = threading.Event(), threading.Event()
    register = ServiceControlPlane.register_volume
    refused = KeyError("'publisher/0' is not a publisher rank")

    async def held_register(self: ServiceControlPlane, *args: Any) -> None:
        if not registering.is_set():
            registering.set()
            await asyncio.to_thread(release.wait, 30)
            if outcome == "refused":
                raise refused
        await register(self, *args)

    monkeypatch.setattr(ServiceControlPlane, "register_volume", held_register)
    store_name = _store_name()
    await _initialize_service(store_name)
    loop = asyncio.get_running_loop()
    starting: asyncio.Future[asyncio.Task[Any]] = loop.create_future()

    def first() -> Any:
        async def build() -> Any:
            task = asyncio.ensure_future(
                rank_client(store_name, role=RankRole.PUBLISHER)
            )
            loop.call_soon_threadsafe(starting.set_result, task)
            return await task

        try:
            return asyncio.run(build())
        except BaseException as error:
            return error

    starter = asyncio.ensure_future(asyncio.to_thread(first))
    try:
        task = await asyncio.wait_for(starting, 30)
        assert await asyncio.to_thread(registering.wait, 30)
        ready = threading.Barrier(3, timeout=30)
        waiters = asyncio.gather(
            *(
                asyncio.to_thread(_race, rank_client, store_name, ready)
                for _ in range(3)
            )
        )
        await _until(lambda: start_waits.await_count == 3)
        if outcome == "cancelled":
            task.get_loop().call_soon_threadsafe(task.cancel)
        else:
            release.set()
        start = time.monotonic()
        errors = await asyncio.wait_for(waiters, 30)
        assert time.monotonic() - start < 10
        assert _publisher_servers(store_name) == []
        release.set()
        first_error = await asyncio.wait_for(starter, 30)
        if outcome == "cancelled":
            reason = "was cancelled in another task"
            assert isinstance(first_error, asyncio.CancelledError)
        else:
            reason = f"failed in another task: {refused!r}"
            assert first_error is refused
        assert all(type(error) is RuntimeError for error in errors), errors
        message = f"starting volume 'publisher/0' {reason}"
        assert [str(error) for error in errors] == [message] * 3
        client = await rank_client(store_name, role=RankRole.PUBLISHER)
        assert isinstance(client, RoutingClient)
        assert _publisher_servers(store_name) == ["publisher/0"]
        assert _volume(store_name).current()
    finally:
        release.set()
        await ts.shutdown(store_name)
    assert len(_serve_lines(caplog, "publisher/0")) == 2


def _free(*locks: Any) -> bool:
    """Whether each lock frees within 30 s, as none does that a deadlocked collector holds."""
    for lock in locks:
        if not lock.acquire(timeout=30):
            return False
        lock.release()
    return True


@pytest.mark.parametrize(
    ("collector", "via"),
    [
        ("none", "record"),
        ("publisher_lock", "record"),
        ("publisher_lock", "address"),
        ("selection_lock", "record"),
        # A build through an address also marks the store. The GC's close leaves it marked as attempted.
        ("selection_lock", "address"),
        # A build without rank= gives back the rank it took from Monarch without waiting too.
        ("selection_lock", "monarch-rank"),
    ],
)
@pytest.mark.asyncio
async def test_a_build_abandoned_as_its_volume_starts_leaves_the_next_build_serving(
    monkeypatch: pytest.MonkeyPatch,
    rank_client: _RankClient,
    collector: str,
    via: str,
) -> None:
    """Stop the volume of a build abandoned with its event loop as it registers, leaving one server.

    Collected in a locked step, it stops without waiting and the next build serves before its volume's close runs.
    Left uncollected, it holds the next build ``_START_WAIT_S``, which then stops it and raises.
    """
    monkeypatch.setattr(carrier_volumes, "_START_WAIT_S", 0.5)
    entered = threading.Event()
    register = ServiceControlPlane.register_volume
    # The abandoned volumes' closes, held until the case runs them.
    closes: list[PublisherVolume] = []
    close_on_thread = carrier_volumes._close_on_thread
    monkeypatch.setattr(carrier_volumes, "_close_on_thread", closes.append)

    async def hold(*args: Any, **kwargs: Any) -> Any:
        entered.set()
        await asyncio.Event().wait()

    store_name = _store_name()
    async with contextlib.AsyncExitStack() as stack:
        kwargs: dict[str, Any] = {}
        if via == "address":
            other = await stack.enter_async_context(
                _other_service(store_name, ts.TorchStoreStrategy())
            )
            kwargs = {"address": other.address}
        else:
            await _initialize_service(store_name)
        abandoned = partial(rank_client, store_name, role=RankRole.PUBLISHER, **kwargs)
        if via == "monarch-rank":
            monkeypatch.setattr(store_api, "_actor_rank", lambda: 0)
            abandoned = partial(ts.client, store_name, role=RankRole.PUBLISHER)
        building: list[Any] = []
        try:
            monkeypatch.setattr(ServiceControlPlane, "register_volume", hold)
            loop = asyncio.new_event_loop()
            runner = threading.Thread(target=loop.run_forever, daemon=True)
            runner.start()
            try:
                # Kept: only the locked collection below can free the abandoned build.
                building.append(asyncio.run_coroutine_threadsafe(abandoned(), loop))
                assert await asyncio.to_thread(entered.wait, 30)
            finally:
                # Stopped and closed on every outcome. A loop thread blocked in a call keeps
                # running with its loop open so that the case's own failure is the one reported.
                loop.call_soon_threadsafe(loop.stop)
                await asyncio.to_thread(runner.join, 30)
                if not runner.is_alive():
                    loop.close()
            monkeypatch.setattr(ServiceControlPlane, "register_volume", register)
            volume = _volume(store_name)
            if collector != "none":
                lock = (
                    carrier_volumes._publisher_lock
                    if collector == "publisher_lock"
                    else store_api._selection_lock
                )
                collected = threading.Event()

                def collect() -> None:
                    with lock:
                        building.clear()
                        gc.collect()
                    collected.set()

                threading.Thread(target=collect, daemon=True).start()
                assert await asyncio.to_thread(
                    collected.wait, 30
                ), "the collector deadlocked"
                # It gives its rank back once that lock frees, if not at once.
                await _until(lambda: store_name not in store_api._building_ranks)
                # Its close is still held: nothing but the abandon stopped it.
                assert volume.started.done() and not volume.current()
                assert closes == [volume] and volume._server is not None
                # The abandon takes it out of the map at once unless the publisher lock was held.
                key = (store_name, "publisher/0")
                kept = carrier_volumes._publishers.get(key) is volume
                assert kept == (collector == "publisher_lock")
            else:
                with pytest.raises(TimeoutError, match="did not finish within 0.5 s"):
                    await asyncio.wait_for(
                        rank_client(store_name, role=RankRole.PUBLISHER, **kwargs), 30
                    )
                assert not volume.current() and _publisher_servers(store_name) == []
            client = await asyncio.wait_for(
                rank_client(store_name, role=RankRole.PUBLISHER, **kwargs), 30
            )
            assert isinstance(client, RoutingClient)
            assert _volume(store_name) is not volume and not volume.current()
            for held in closes:
                close_on_thread(held)
            # Only the abandoned volume's server stops once its held close runs.
            await _until(lambda: _publisher_servers(store_name) == ["publisher/0"])
            assert volume._server is None and _volume(store_name).current()
        finally:
            building.clear()
            gc.collect()
            # A collector that deadlocked holds its lock for good, so the case fails without it.
            if _free(carrier_volumes._publisher_lock, store_api._selection_lock):
                await store_api._shutdown_selected(store_name)


@pytest.mark.asyncio
async def test_a_caller_waiting_on_a_start_the_gc_abandons_raises_why(
    monkeypatch: pytest.MonkeyPatch,
    rank_client: _RankClient,
    start_waits: mock.AsyncMock,
) -> None:
    """Fail a caller waiting on a start whose build the GC closes with its event loop.

    It raises a RuntimeError saying the start was abandoned in another task.
    """
    entered = threading.Event()

    async def hold(*args: Any, **kwargs: Any) -> Any:
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(ServiceControlPlane, "register_volume", hold)
    store_name = _store_name()
    await _initialize_service(store_name)
    building: list[Any] = []
    try:
        loop = asyncio.new_event_loop()
        runner = threading.Thread(target=loop.run_forever, daemon=True)
        runner.start()
        try:
            building.append(
                asyncio.run_coroutine_threadsafe(
                    rank_client(store_name, role=RankRole.PUBLISHER), loop
                )
            )
            assert await asyncio.to_thread(entered.wait, 30)
            ready = threading.Barrier(1, timeout=30)
            waiter = asyncio.ensure_future(
                asyncio.to_thread(_race, rank_client, store_name, ready)
            )
            await _until(lambda: start_waits.await_count == 1)
        finally:
            # Stopped and closed on every outcome. A loop thread blocked in a call keeps running
            # with its loop open so that the case's own failure is the one reported.
            loop.call_soon_threadsafe(loop.stop)
            await asyncio.to_thread(runner.join, 30)
            if not runner.is_alive():
                loop.close()
        building.clear()
        gc.collect()
        error = await asyncio.wait_for(waiter, 30)
        assert type(error) is RuntimeError, error
        assert (
            str(error) == "starting volume 'publisher/0' was abandoned in another task"
        )
    finally:
        building.clear()
        gc.collect()
        await store_api._shutdown_selected(store_name)


@pytest.mark.parametrize("finalizing", [False, True])
def test_an_abandoned_volumes_close_runs_on_a_thread_that_logs_a_failure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, finalizing: bool
) -> None:
    """Close an abandoned start's volume on a thread of its own, which logs a failing close.

    A finalizing interpreter starts no thread: the exit hook closed every served volume.
    """
    caplog.set_level(logging.WARNING, logger=carrier_volumes.__name__)
    volume = PublisherVolume("publisher/0", _store_name(), (object(), None))
    started: list[threading.Thread] = []

    def thread(**kwargs: Any) -> threading.Thread:
        started.append(threading.Thread(**kwargs))
        return started[-1]

    async def failing_close() -> None:
        raise OSError("volume close failed")

    monkeypatch.setattr(volume, "close", failing_close)
    # carrier_volumes starts the thread through the shared helper in torchstore.utils.
    monkeypatch.setattr(store_utils, "threading", SimpleNamespace(Thread=thread))
    finalizing_sys = SimpleNamespace(is_finalizing=lambda: finalizing)
    monkeypatch.setattr(store_utils, "sys", finalizing_sys)
    carrier_volumes._close_on_thread(volume)
    for closer in started:
        closer.join(30)
        assert not closer.is_alive()
    logged = [
        (record.getMessage(), record.exc_info and record.exc_info[0])
        for record in caplog.records
        if record.name == carrier_volumes.__name__
    ]
    if finalizing:
        assert started == [] and logged == []
    else:
        assert len(started) == 1
        assert logged == [("stopping routing volume=publisher/0 failed", OSError)]


class _RefusedThread:
    """A thread that the interpreter refuses to start as it does once finalization begins after the check."""

    def __init__(self, **kwargs: Any) -> None:
        pass

    def start(self) -> None:
        raise RuntimeError("can't create new thread at interpreter shutdown")


def test_an_abandoned_volumes_close_whose_thread_is_refused_returns_quietly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return without raising when the close's thread cannot start since the exit hook closes every served volume."""
    volume = PublisherVolume("publisher/0", _store_name(), (object(), None))
    monkeypatch.setattr(
        store_utils, "threading", SimpleNamespace(Thread=_RefusedThread)
    )
    monkeypatch.setattr(
        store_utils, "sys", SimpleNamespace(is_finalizing=lambda: False)
    )
    carrier_volumes._close_on_thread(volume)


@pytest.mark.asyncio
async def test_a_caller_that_gives_up_on_a_live_start_leaves_it_serving(
    monkeypatch: pytest.MonkeyPatch,
    rank_client: _RankClient,
    start_waits: mock.AsyncMock,
) -> None:
    """Raise TimeoutError for a caller that waits past ``_START_WAIT_S`` on a start whose loop runs.

    That start goes on, registers and serves: a waiter stops only a start whose event loop closed. The wait is
    the 150s that ts.client documents unless a test shortens it.
    """
    assert carrier_volumes._START_WAIT_S == 150
    monkeypatch.setattr(carrier_volumes, "_START_WAIT_S", 0.2)
    registering, release = threading.Event(), threading.Event()
    register = ServiceControlPlane.register_volume

    async def held_register(self: ServiceControlPlane, *args: Any) -> None:
        registering.set()
        await asyncio.to_thread(release.wait, 30)
        await register(self, *args)

    monkeypatch.setattr(ServiceControlPlane, "register_volume", held_register)
    store_name = _store_name()
    await _initialize_service(store_name)
    ready = threading.Barrier(1, timeout=30)
    try:
        starter = asyncio.ensure_future(
            asyncio.to_thread(_race, rank_client, store_name, ready)
        )
        assert await asyncio.to_thread(registering.wait, 30)
        volume = _volume(store_name)
        error = await asyncio.to_thread(_race, rank_client, store_name, ready)
        assert type(error) is TimeoutError, error
        assert str(error) == (
            "starting volume 'publisher/0' in another task did not finish within 0.2 s"
        )
        assert start_waits.await_count == 1
        assert volume.current() and volume._server is not None
        release.set()
        client = await asyncio.wait_for(starter, 30)
        assert isinstance(client, RoutingClient), client
        assert _volume(store_name) is volume and volume.current()
        assert _publisher_servers(store_name) == ["publisher/0"]
    finally:
        release.set()
        await store_api._shutdown_selected(store_name)


@pytest.mark.asyncio
async def test_a_start_cancelled_as_it_stops_another_stores_volume_leaves_no_entry(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    rank_client: _RankClient,
    start_waits: mock.AsyncMock,
) -> None:
    """Fail the 3 threads waiting on a start cancelled while it stops another store's volume.

    It never served, so it closes: its entry leaves the map, no server stays, and a later call
    serves and registers a volume of its own.
    """
    caplog.set_level(logging.INFO, logger="torchstore.carrier")
    stopping, release = threading.Event(), threading.Event()
    close = PublisherVolume.close
    old: list[PublisherVolume] = []

    async def held_close(self: PublisherVolume) -> None:
        await close(self)
        if old and self is old[0] and not stopping.is_set():
            stopping.set()
            await asyncio.to_thread(release.wait, 30)

    store_name = _store_name()
    loop = asyncio.get_running_loop()
    starting: asyncio.Future[asyncio.Task[Any]] = loop.create_future()
    async with contextlib.AsyncExitStack() as stack:
        addresses = []
        for _ in range(2):
            other = _other_service(store_name, ts.TorchStoreStrategy())
            addresses.append((await stack.enter_async_context(other)).address)
        stack.push_async_callback(store_api._shutdown_selected, store_name)
        await rank_client(store_name, role=RankRole.PUBLISHER, address=addresses[0])
        old.append(_volume(store_name))
        ts.reset_client(store_name)
        monkeypatch.setattr(PublisherVolume, "close", held_close)
        new = {"address": addresses[1]}

        def first() -> Any:
            async def build() -> Any:
                task = asyncio.ensure_future(
                    rank_client(store_name, role=RankRole.PUBLISHER, **new)
                )
                loop.call_soon_threadsafe(starting.set_result, task)
                return await task

            try:
                return asyncio.run(build())
            except BaseException as error:
                return error

        starter = asyncio.ensure_future(asyncio.to_thread(first))
        try:
            task = await asyncio.wait_for(starting, 30)
            assert await asyncio.to_thread(stopping.wait, 30)
            ready = threading.Barrier(3, timeout=30)
            waiters = asyncio.gather(
                *(
                    asyncio.to_thread(_race, rank_client, store_name, ready, **new)
                    for _ in range(3)
                )
            )
            await _until(lambda: start_waits.await_count == 3, timeout_s=30)
            task.get_loop().call_soon_threadsafe(task.cancel)
            errors = await asyncio.wait_for(waiters, 30)
            assert all(type(error) is RuntimeError for error in errors), errors
            message = "starting volume 'publisher/0' was cancelled in another task"
            assert [str(error) for error in errors] == [message] * 3
            assert carrier_volumes._publishers == {}
            assert _publisher_servers(store_name) == []
            release.set()
            first_error = await asyncio.wait_for(starter, 30)
            assert isinstance(first_error, asyncio.CancelledError)
            await rank_client(store_name, role=RankRole.PUBLISHER, **new)
            volume = _volume(store_name)
            assert volume is not old[0] and volume.current()
            assert _publisher_servers(store_name) == ["publisher/0"]
        finally:
            release.set()
    assert len(_serve_lines(caplog, "publisher/0")) == 2


class _PausingLock:
    """The publisher lock, except that one thread pauses after its first release until resumed."""

    def __init__(self, lock: threading.Lock) -> None:
        self._lock = lock
        self.ident: int | None = None
        self.paused, self.resume = threading.Event(), threading.Event()

    def __enter__(self) -> None:
        self._lock.acquire()

    def __exit__(self, *exc: object) -> None:
        self._lock.release()
        if threading.get_ident() == self.ident and not self.paused.is_set():
            self.paused.set()
            assert self.resume.wait(30), "not resumed within 30 s"


@pytest.mark.parametrize("outcome", ["cancelled", "refused", "stopped"])
@pytest.mark.asyncio
async def test_a_caller_that_finds_a_start_under_way_waits_on_it_though_it_ends_first(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    rank_client: _RankClient,
    outcome: str,
) -> None:
    """Make a caller that found a start under way wait on it, though it ends before the caller looks.

    It raises why the start failed, or once a shutdown stopped it the build's lost-store error,
    and never registers that start's record. A later call starts a volume unless that shutdown ran.
    """
    caplog.set_level(logging.INFO, logger="torchstore.carrier")
    held, release = threading.Event(), threading.Event()
    registered_by: list[int] = []
    register, serve = ServiceControlPlane.register_volume, carrier_volumes.serve_volume
    refused = KeyError("'publisher/0' is not a publisher rank")

    async def held_register(self: ServiceControlPlane, *args: Any) -> None:
        registered_by.append(threading.get_ident())
        if outcome != "stopped" and not held.is_set():
            held.set()
            await asyncio.to_thread(release.wait, 30)
            if outcome == "refused":
                raise refused
        await register(self, *args)

    async def held_serve(*args: Any, **kwargs: Any) -> Any:
        served = await serve(*args, **kwargs)
        held.set()
        await asyncio.to_thread(release.wait, 30)
        return served

    monkeypatch.setattr(ServiceControlPlane, "register_volume", held_register)
    if outcome == "stopped":
        monkeypatch.setattr(carrier_volumes, "serve_volume", held_serve)
    store_name = _store_name()
    address = await _initialize_service(store_name)
    loop = asyncio.get_running_loop()
    starting: asyncio.Future[asyncio.Task[Any]] = loop.create_future()
    lock = _PausingLock(carrier_volumes._publisher_lock)
    idents: dict[str, int] = {}

    def first() -> Any:
        idents["first"] = threading.get_ident()

        async def build() -> Any:
            task = asyncio.ensure_future(
                rank_client(store_name, role=RankRole.PUBLISHER)
            )
            loop.call_soon_threadsafe(starting.set_result, task)
            return await task

        try:
            return asyncio.run(build())
        except BaseException as error:
            return error

    def second() -> Any:
        lock.ident = threading.get_ident()
        try:
            return asyncio.run(rank_client(store_name, role=RankRole.PUBLISHER))
        except BaseException as error:
            return error

    starter = asyncio.ensure_future(asyncio.to_thread(first))
    try:
        task = await asyncio.wait_for(starting, 30)
        assert await asyncio.to_thread(held.wait, 30)
        monkeypatch.setattr(carrier_volumes, "_publisher_lock", lock)
        waiter = asyncio.ensure_future(asyncio.to_thread(second))
        assert await asyncio.to_thread(lock.paused.wait, 30)
        if outcome == "cancelled":
            task.get_loop().call_soon_threadsafe(task.cancel)
            await _until(task.done)
        elif outcome == "stopped":
            assert await store_api._shutdown_selected(store_name)
        release.set()
        first_error = await asyncio.wait_for(starter, 30)
        lock.resume.set()
        start = time.monotonic()
        second_error = await asyncio.wait_for(waiter, 30)
        assert time.monotonic() - start < 10
        if outcome == "stopped":
            # Both built through the record, whose error says to initialize again.
            message = (
                f"routing store '{store_name}' on {address} was shut down or replaced while"
                " this client was being built. Call ts.initialize again or pass address=,"
                " and restart the store if it refuses the volume as already registered"
            )
            for error in (first_error, second_error):
                assert type(error) is RuntimeError and str(error) == message
            assert registered_by == []
            assert stop_publisher_volumes(store_name) == []
            assert _publisher_servers(store_name) == []
            return
        reason = (
            "was cancelled in another task"
            if outcome == "cancelled"
            else f"failed in another task: {refused!r}"
        )
        assert type(second_error) is RuntimeError, second_error
        assert str(second_error) == f"starting volume 'publisher/0' {reason}"
        assert registered_by == [idents["first"]]
        await rank_client(store_name, role=RankRole.PUBLISHER)
        assert _volume(store_name).current()
        assert _publisher_servers(store_name) == ["publisher/0"]
    finally:
        release.set()
        lock.resume.set()
        await store_api._shutdown_selected(store_name)
    assert len(_serve_lines(caplog, "publisher/0")) == 2


@pytest.mark.parametrize("made", ["once", "again"])
@pytest.mark.asyncio
async def test_a_requesters_shutdown_fails_its_reads_in_flight_at_once(
    children: _Children, rank_client: _RankClient, made: str
) -> None:
    """Fail a get held on the volume side within 1 s of a shutdown on another thread.

    A client that ``reset_client`` dropped, still held, closes its handles too, so the process
    keeps no connection to the volume.
    """
    store_name = _store_name()
    address = await _initialize_service(store_name)
    publisher = _publisher(children, address, 0, 1)
    try:
        assert "pid" in await asyncio.to_thread(_reply, publisher, 90)
        await asyncio.to_thread(_push, publisher, 0)
        clients = [await rank_client(store_name, role=RankRole.REQUESTER, group=0)]
        if made == "again":
            reader = _requester_of(clients[0])
            reader.check(0, await reader.get())
            ts.reset_client(store_name)
            clients.append(
                await rank_client(store_name, role=RankRole.REQUESTER, group=0)
            )
        reader = _requester_of(clients[-1])
        reader.check(0, await reader.get())
        _command(publisher, "hold-gets")
        assert await asyncio.to_thread(_reply, publisher) == {"holding": True}
        reading = asyncio.ensure_future(
            asyncio.to_thread(lambda: asyncio.run(reader.get()))
        )
        _command(publisher, "wait-held")
        assert await asyncio.to_thread(_reply, publisher) == {"held": True}
        start = time.monotonic()
        # Timed to the read's failure: the shutdown then goes on to stop the service here.
        stopping = asyncio.ensure_future(ts.shutdown(store_name))
        lost = f"carrier publisher/0 at {_LOCALHOST}:"
        with pytest.raises(PeerLostError, match=lost):
            await asyncio.wait_for(reading, 10)
        assert time.monotonic() - start < 1
        await stopping
        for client in clients:
            with pytest.raises(RuntimeError, match="its carrier volume host is closed"):
                client.strategy.get_storage_volume("publisher/0")
        assert _volume_peers() == []
    finally:
        _command(publisher, "release")
        await store_api._shutdown_selected(store_name)


@pytest.mark.parametrize("dropped", ["collected", "held", "held_failing", "remade"])
@pytest.mark.asyncio
async def test_shutdown_closes_every_dropped_clients_handles_and_gloo_pairs(
    children: _Children,
    monkeypatch: pytest.MonkeyPatch,
    rank_client: _RankClient,
    dropped: str,
) -> None:
    """Close the handles of a client that ``reset_client`` dropped, though nothing holds it.

    Shutdown forgets the pair such a client makes if it is held and reads again, even when closing
    its handles fails, and a newer client's pair, which only that client's own close forgets.
    """
    store_name = _store_name()
    address = await _initialize_service(
        store_name, ts.TorchStoreStrategy(TransportType.Gloo)
    )
    publisher = _publisher(children, address, 0, 1)
    try:
        assert "pid" in await asyncio.to_thread(_reply, publisher, 90)
        await asyncio.to_thread(_push, publisher, 0)
        client = await rank_client(store_name, role=RankRole.REQUESTER, group=0)
        host = _host_of(client)
        reader = _requester_of(client)
        reader.check(0, await reader.get())
        groups = client.strategy.transport_context.get(gloo.GlooProcessGroupCache)
        ts.reset_client(store_name)
        assert "publisher/0" not in gloo._store_addrs
        store_key: str | None = None
        if dropped == "collected":
            collected = weakref.ref(client)
            del client, reader
            gc.collect()
            assert collected() is None
        elif dropped.startswith("held"):
            reader.check(0, await reader.get())
            assert "publisher/0" in gloo._store_addrs and groups._process_groups
        else:
            again = await rank_client(store_name, role=RankRole.REQUESTER, group=0)
            reader = _requester_of(again)
            reader.check(0, await reader.get())
            groups = again.strategy.transport_context.get(gloo.GlooProcessGroupCache)
            [store_key] = groups._process_groups
        if dropped == "held_failing":
            close = host.close

            async def close_then_fail() -> None:
                await close()
                raise OSError("host close failed")

            monkeypatch.setattr(host, "close", close_then_fail)
            with pytest.raises(OSError, match="host close failed"):
                await ts.shutdown(store_name)
        else:
            await ts.shutdown(store_name)
        assert _volume_peers() == []
        assert "publisher/0" not in gloo._store_addrs and groups._process_groups == {}
        if dropped == "remade":
            # Retired once nothing holds its group, for this process's next call to the volume.
            gc.collect()
            [server] = [
                where for where, keys in _waiting().items() if store_key in keys
            ]
            assert (
                isinstance(server, CarrierAddress) and server.service == "publisher/0"
            )
    finally:
        await store_api._shutdown_selected(store_name)


@pytest.mark.asyncio
async def test_a_dropped_client_frees_its_shared_memory_mappings_once_collected(
    rank_client: _RankClient,
) -> None:
    """Free the segments a dropped client mapped once nothing holds it, before any shutdown.

    The store's shutdown still closes its handles, so no connection to the volume stays.
    """
    store_name = _store_name()
    await _initialize_service(
        store_name, ts.TorchStoreStrategy(TransportType.SharedMemory)
    )
    try:
        client = await rank_client(store_name, role=RankRole.PUBLISHER)
        await put_state_dict(client, _state_dict(0), data.KEY)
        mapped = client.strategy.transport_context.get(SharedMemoryCache)
        assert mapped._storages
        freed = weakref.ref(mapped)
        ts.reset_client(store_name)
        del client, mapped
        gc.collect()
        assert freed() is None
        await ts.shutdown(store_name)
        assert _volume_peers() == []
    finally:
        await store_api._shutdown_selected(store_name)


def _host_of(client: LocalClient) -> CarrierVolumeHost:
    host = _installed_volume_host(client.strategy)
    assert isinstance(host, CarrierVolumeHost)
    return host


@pytest.mark.asyncio
async def test_a_new_client_closes_the_volume_hosts_of_collected_dropped_clients(
    rank_client: _RankClient,
) -> None:
    """Close every collected dropped client's volume host but the newest as each new client builds.

    The newest keeps its connection to the volume for the new client to share, so 20 rebuilds after
    ``reset_client`` hold one dropped host and one connection, and shutdown closes the rest.
    """
    store_name = _store_name()
    await _initialize_service(store_name)
    hosts: list[CarrierVolumeHost] = []
    try:
        publisher = await rank_client(store_name, role=RankRole.PUBLISHER)
        await put_state_dict(publisher, _state_dict(0), data.KEY)
        hosts.append(_host_of(publisher))
        del publisher
        [connection] = _volume_peers()
        assert runtime._runtime is not None
        peer = runtime._runtime.peers[connection]
        for _ in range(20):
            ts.reset_client(store_name)
            gc.collect()
            client = await rank_client(store_name, role=RankRole.REQUESTER, group=0)
            hosts.append(_host_of(client))
            reader = _requester_of(client)
            reader.check(0, await reader.get())
            del client, reader
            closed = [host._closed for host in hosts]
            assert closed == [True] * (len(hosts) - 2) + [False, False]
            assert len(store_api._dropped_carriers[store_name]) == 1
            assert runtime._runtime.peers[connection] is peer
    finally:
        await ts.shutdown(store_name)
    assert all(host._closed for host in hosts) and _volume_peers() == []


class _Gone:
    """Freed once its last reference goes, as a collected client's strategy is."""


async def _no_records(volume_ids: Collection[str]) -> dict[str, bytes]:
    raise AssertionError("a host with no volumes read records")


def _closing_host(
    monkeypatch: pytest.MonkeyPatch, entered: asyncio.Event, *, block: bool, fail: bool
) -> CarrierVolumeHost:
    """A real carrier volume host whose close sets ``entered`` and then waits until cancelled or fails."""
    host = CarrierVolumeHost(_no_records)
    close = host.close

    async def held_close() -> None:
        entered.set()
        if block:
            await asyncio.Event().wait()
        await close()
        if fail:
            raise RuntimeError("host close failed")

    monkeypatch.setattr(host, "close", held_close)
    return host


def _noting_closes(
    monkeypatch: pytest.MonkeyPatch,
    host: CarrierVolumeHost,
    name: str,
    closes: list[str],
) -> CarrierVolumeHost:
    """Note in ``closes`` when ``host``'s close starts and when it ends (by a raise or cancel too)."""
    close = host.close

    async def noted_close() -> None:
        closes.append(f"{name} starts")
        try:
            await close()
        finally:
            closes.append(f"{name} ends")

    monkeypatch.setattr(host, "close", noted_close)
    return host


@pytest.mark.parametrize("held_close", ["cancelled", "failing"])
@pytest.mark.asyncio
async def test_closing_collected_hosts_runs_every_close_though_one_is_cancelled_or_fails(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, held_close: str
) -> None:
    """Close every collected dropped client's host but the newest even when one close is cancelled or fails.

    Closes run newest first and each runs after the one before it ends. The cancelled close sits
    between two that must close. A failing close only logs. A host whose client lives stays open.
    """
    caplog.set_level(logging.WARNING, logger=store_api.__name__)
    store_name = _store_name()
    entered = asyncio.Event()
    middle = _closing_host(
        monkeypatch,
        entered,
        block=held_close == "cancelled",
        fail=held_close == "failing",
    )
    closes: list[str] = []
    hosts = [
        _noting_closes(monkeypatch, host, name, closes)
        for host, name in zip(
            [CarrierVolumeHost(_no_records), middle, CarrierVolumeHost(_no_records)],
            ["oldest", "middle", "newer"],
        )
    ]
    newest = CarrierVolumeHost(_no_records)
    held, live = _Gone(), CarrierVolumeHost(_no_records)
    entries: list[Any] = [(weakref.ref(_Gone()), host) for host in [*hosts, newest]]
    entries.insert(1, (weakref.ref(held), live))
    monkeypatch.setitem(store_api._dropped_carriers, store_name, list(entries))
    closing = asyncio.ensure_future(store_api._close_collected_carriers(store_name))
    await asyncio.wait_for(entered.wait(), 30)
    if held_close == "cancelled":
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closing
    else:
        await asyncio.wait_for(closing, 30)
    assert [host._closed for host in hosts] == [True, held_close == "failing", True]
    assert closes == [
        f"{name} {step}"
        for name in ("newer", "middle", "oldest")
        for step in ("starts", "ends")
    ]
    assert not newest._closed and not live._closed
    assert store_api._dropped_carriers[store_name] == [entries[1], entries[-1]]
    warned = [
        record.getMessage()
        for record in caplog.records
        if record.name == store_api.__name__ and record.levelno >= logging.WARNING
    ]
    expected = ["closing a collected client's volume handles failed"]
    assert warned == (expected if held_close == "failing" else [])


@pytest.mark.asyncio
async def test_closing_a_replaced_dial_and_a_service_runs_both_though_the_first_is_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stop the service after a cancel interrupts the dial's earlier close and then raise the cancel."""
    store_name = _store_name()
    async with contextlib.AsyncExitStack() as stack:
        other = await stack.enter_async_context(
            _other_service(store_name, ts.TorchStoreStrategy())
        )
        service = await CoordinatorService.start(_store_name(), host=_LOCALHOST)
        stack.push_async_callback(service.close)
        handle = await other.address.connect()
        stack.push_async_callback(handle.close)
        entered, release = asyncio.Event(), asyncio.Event()

        async def held_close() -> None:
            entered.set()
            await release.wait()

        # A close that the cancel leaves running: the API starts each close as a task.
        monkeypatch.setattr(handle, "close", held_close)
        stack.callback(release.set)
        dialed = store_api._SelectedControlPlane(handle, other.address)
        closing = asyncio.ensure_future(store_api._close_quietly(dialed, None, service))
        await asyncio.wait_for(entered.wait(), 30)
        assert service._server is not None
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closing
        assert (service._server, service._rendezvous) == (None, None)


@pytest.mark.asyncio
async def test_a_requester_built_before_any_publisher_waits_for_nothing(
    children: _Children,
    caplog: pytest.LogCaptureFixture,
    rank_client: _RankClient,
) -> None:
    """Build a requester before any publisher serves, at once and reading no record.

    Records load after its first layouts, once a publisher has pushed, and it reads them bitwise.
    """
    caplog.set_level(logging.INFO)
    store_name = _store_name()
    address = await _initialize_service(store_name)
    try:
        start = time.monotonic()
        requester = await asyncio.wait_for(
            rank_client(store_name, role=RankRole.REQUESTER, group=0), 10
        )
        assert time.monotonic() - start < 5
        assert "volume_records" not in _timed(caplog)
        publisher = _publisher(children, address, 0, 1)
        assert "pid" in await asyncio.to_thread(_reply, publisher, 90)
        await asyncio.to_thread(_push, publisher, 0)
        reader = _requester_of(requester)
        reader.check(0, await reader.get())
        assert _timed(caplog).count("volume_records") == 1
    finally:
        await ts.shutdown(store_name)


@pytest.mark.asyncio
async def test_each_publisher_registers_once_and_each_requester_reads_records_once(
    caplog: pytest.LogCaptureFixture,
    rank_client: _RankClient,
) -> None:
    """Log one ``register_volume`` line per publisher, and one ``volume_records`` per requester."""
    caplog.set_level(logging.INFO)
    store_name = _store_name()
    address = await _initialize_service(store_name)
    try:
        publisher = await rank_client(store_name, role=RankRole.PUBLISHER)
        async with _reader(address) as first, _reader(address, 1, 1) as second:
            for round_ in range(2):
                await put_state_dict(publisher, _state_dict(round_), data.KEY)
                for reader in (first, second):
                    reader.check(round_, await reader.get())
    finally:
        await ts.shutdown(store_name)
    timed = _timed(caplog)
    assert (timed.count("register_volume"), timed.count("volume_records")) == (1, 2)


@pytest.mark.asyncio
async def test_volume_errors_keep_their_types_through_the_mapping_read(
    rank_client: _RankClient,
) -> None:
    """Raise a volume's KeyError with its text. Wrap a closed volume's error at the mapping read."""
    store_name = _store_name()
    address = await _initialize_service(store_name)
    try:
        publisher = await rank_client(store_name, role=RankRole.PUBLISHER)
        await put_state_dict(publisher, _state_dict(0), data.KEY)
        async with _reader(address) as reader:
            reader.check(0, await reader.get())
            with pytest.raises(KeyError, match="Key 'other/MAPPING' not found"):
                await reader.client.get("other/MAPPING")
            await _volume(store_name).close()
            with pytest.raises(
                RuntimeError, match="^Mapping is missing from the store"
            ) as raised:
                await reader.get()
            assert type(raised.value.__cause__) is PeerLostError
    finally:
        await ts.shutdown(store_name)


@pytest.mark.asyncio
async def test_a_killed_publisher_fails_the_next_read_within_bounds(
    children: _Children, rank_client: _RankClient
) -> None:
    """Wrap a lost publisher's error at the next read's mapping, within its bound.

    A requester reads every mapping in volume order, so publisher 0 answers first.
    """
    store_name = _store_name()
    address = await _initialize_service(store_name, publishers=2)
    try:
        publishers = [_publisher(children, address, rank, 2) for rank in range(2)]
        for child in publishers:
            assert "pid" in await asyncio.to_thread(_reply, child, 90)
            await asyncio.to_thread(_push, child, 0)
        requester = await rank_client(store_name, role=RankRole.REQUESTER, group=1)
        reader = _requester_of(requester, 1, 0)
        reader.check(0, await reader.get())
        os.kill(publishers[1].pid, signal.SIGKILL)
        await asyncio.to_thread(publishers[1].wait, 30)
        start = time.monotonic()
        with pytest.raises(RuntimeError, match="^Mapping is missing") as raised:
            await reader.get()
        assert type(raised.value.__cause__) is PeerLostError
        assert time.monotonic() - start < 10
    finally:
        await ts.shutdown(store_name)


@pytest.mark.asyncio
async def test_a_sync_after_the_driver_dies_still_completes_bitwise(
    children: _Children, rank_client: _RankClient
) -> None:
    """Sync again bitwise after the driver dies: later syncs touch no TCPStore or coordinator."""
    store_name = _store_name()
    driver, address = await asyncio.to_thread(
        _driver, children, store_name, "MonarchRPC", 2
    )
    try:
        publishers = [_publisher(children, address, rank, 2) for rank in range(2)]
        for child in publishers:
            assert "pid" in await asyncio.to_thread(_reply, child, 90)
        requester = await rank_client(
            store_name, role=RankRole.REQUESTER, group=1, rank=1, address=address
        )
        reader = _requester_of(requester, 1, 1)
        for round_ in range(2):
            for child in publishers:
                await asyncio.to_thread(_push, child, round_)
            reader.check(round_, await reader.get())
            if round_ == 0:
                driver.kill()
                await asyncio.to_thread(driver.wait, 30)
    finally:
        await store_api._shutdown_selected(store_name)


@pytest.mark.parametrize("how", ["closed", "stopped"])
@pytest.mark.asyncio
async def test_a_tcpstore_lost_before_the_records_load_fails_the_first_get_unwrapped(
    children: _Children,
    monkeypatch: pytest.MonkeyPatch,
    rank_client: _RankClient,
    how: str,
) -> None:
    """Raise a record read's error naming host:port from the first get, after its layouts arrived.

    A closed TCPStore fails it at EOF, and one that stops answering at the TCPStore deadline.
    """
    monkeypatch.setattr(tcpstore, "_DEFAULT_TIMEOUT_S", 2)
    store_name = _store_name()
    driver, address = await asyncio.to_thread(
        _driver, children, store_name, "MonarchRPC", 1
    )
    try:
        publisher = _publisher(children, address, 0, 1)
        assert "pid" in await asyncio.to_thread(_reply, publisher, 90)
        await asyncio.to_thread(_push, publisher, 0)
        requester = await rank_client(
            store_name, role=RankRole.REQUESTER, group=0, address=address
        )
        assert isinstance(requester, RoutingClient)
        handle = requester._control_plane
        get_layouts = handle.get_layouts

        async def layouts_then_lose_the_tcpstore(key: str, **kwargs: Any) -> Any:
            layouts = await get_layouts(key, **kwargs)
            if how == "closed":
                _command(driver, "stop-tcpstore")
                assert await asyncio.to_thread(_reply, driver) == {"stopped": True}
            else:
                driver.send_signal(signal.SIGSTOP)
                await _until(lambda: _stopped(driver))
            return layouts

        monkeypatch.setattr(handle, "get_layouts", layouts_then_lose_the_tcpstore)
        start = time.monotonic()
        lost = ConnectionError if how == "closed" else TimeoutError
        with pytest.raises(lost, match=re.escape(str(address))):
            await _Requester(0, 0, requester).get()
        elapsed = time.monotonic() - start
        if how == "closed":
            assert elapsed < 5
        else:
            assert 1.9 < elapsed < 7
    finally:
        driver.send_signal(signal.SIGCONT)
        await store_api._shutdown_selected(store_name)


@pytest.mark.asyncio
async def test_ranks_sync_with_no_actor_cloudpickle_or_coordinator_module(
    children: _Children,
) -> None:
    """Sync twice, bitwise, across processes that block cloudpickle and volume actors.

    None loads Monarch's runtime, its backend, or a coordinator module, nor does the requester's
    shutdown.
    """
    store_name = _store_name()
    blocked = "cloudpickle,torchstore.monarch.storage_volume"
    driver = _rank(children, "driver", store_name, "MonarchRPC", 1, blocked=blocked)
    address = ServiceAddress(*(await asyncio.to_thread(_reply, driver, 90))["address"])
    publisher = _rank(children, "publisher", store_name, 0, 1, blocked=blocked)
    requester = _rank(children, "requester", store_name, 0, 0, blocked=blocked)
    for child in (publisher, requester):
        _command(child, _fields(address))
        assert "pid" in await asyncio.to_thread(_reply, child, 90)
    for round_ in range(2):
        await asyncio.to_thread(_push, publisher, round_)
        await asyncio.to_thread(_pull, requester, round_)
    reports = []
    for child in (driver, publisher, requester):
        _command(child, "loaded")
        reports.append(await asyncio.to_thread(_reply, child))
    assert reports == [
        {"loaded": [], "servers": ["coordinator"]},
        {"loaded": [], "servers": ["publisher/0"]},
        {"loaded": [], "servers": []},
    ]
    # A requester's shutdown takes the service path, loading none of them either.
    _command(requester, "shutdown")
    assert await asyncio.to_thread(_reply, requester) == {"loaded": [], "servers": []}


@pytest.mark.asyncio
async def test_the_service_plane_shares_records_only_while_open_and_logs_each_call(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Register and read records through a handle, one timing line each. A closed handle refuses."""
    caplog.set_level(logging.INFO)
    store_name = _store_name()
    service = await CoordinatorService.start(store_name, host=_LOCALHOST)
    try:
        driver = await service.connect()
        try:
            await driver.init({"publisher/0"}, ts.TorchStoreStrategy())
            await driver.register_volume("publisher/0", b"record")
            assert await driver.volume_records(["publisher/0"]) == {
                "publisher/0": b"record"
            }
        finally:
            await driver.close()
        for call in (
            lambda: driver.register_volume("publisher/0", b"record"),
            lambda: driver.volume_records(["publisher/0"]),
        ):
            with pytest.raises(RuntimeError, match="this control plane is closed$"):
                await call()
    finally:
        await service.close()
    timed = [
        (match.group(1), record.getMessage())
        for record in caplog.records
        if (match := _TIMING_LINE.search(record.getMessage()))
    ]
    assert [operation for operation, _ in timed] == [
        "init",
        "register_volume",
        "volume_records",
        "register_volume",
        "volume_records",
    ]
    assert all(re.search(r" took \d+\.\d+s$", line) for _, line in timed)


@pytest.mark.asyncio
async def test_a_service_refuses_a_strategy_volume_servers_cannot_honor_before_its_init() -> (
    None
):
    """Refuse a subclass or a refused transport at init, writing nothing, so a later init succeeds."""
    store_name = _store_name()
    service = await CoordinatorService.start(store_name, host=_LOCALHOST)
    try:
        driver = await service.connect()
        try:
            for strategy, error, message in (
                (ts.LocalRankStrategy(), TypeError, "not LocalRankStrategy$"),
                (ts.TorchStoreStrategy(TransportType.NIXL), ValueError, "not NIXL$"),
            ):
                with pytest.raises(error, match=message):
                    await driver.init({"publisher/0"}, strategy)
            with pytest.raises(RuntimeError, match="init has not run"):
                await driver.strategy()
            await driver.init(
                {"publisher/0"}, ts.TorchStoreStrategy(TransportType.Gloo)
            )
            strategy = await driver.strategy()
            assert strategy.default_transport_type is TransportType.Gloo
        finally:
            await driver.close()
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_a_strategy_read_refuses_a_config_naming_a_transport_volume_servers_lack() -> (
    None
):
    """Fail a handle's strategy read naming the store once another writer's config names NIXL.

    The rendezvous decodes any transport and the service plane checks it. The read fails rather
    than a requester's first volume lookup.
    """
    store_name = _store_name()
    async with _other_service(store_name, ts.TorchStoreStrategy()) as service:
        rendezvous = service._rendezvous
        assert rendezvous is not None
        nixl = json.dumps({"transport": "NIXL", "v": 1})
        prefixed = rendezvous._prefixed
        assert prefixed is not None
        prefixed.set(tcpstore._CONFIG, nixl)
        handle = await service.address.connect()
        try:
            with pytest.raises(RuntimeError) as raised:
                await handle.strategy()
        finally:
            await handle.close()
    endpoint = tcpstore._endpoint(rendezvous.address)
    assert str(raised.value) == (
        f"routing store {store_name!r} on TCPStore {endpoint}: volume servers carry"
        " SharedMemory, Gloo and MonarchRPC, or Unset to choose among them, not NIXL"
    )
    assert isinstance(raised.value.__cause__, ValueError)


@pytest.mark.asyncio
async def test_a_store_built_after_another_paired_over_gloo_is_refused_and_leaves_it_syncing(
    rank_client: _RankClient,
) -> None:
    """Refuse the Gloo pushes of stores built after another one here paired with a volume ID they share.

    The first store keeps syncing over its one pair, however often the later store is made again.
    """
    first = _store_name()
    await _initialize_service(first, ts.TorchStoreStrategy(TransportType.Gloo))
    try:
        early = await rank_client(first, role=RankRole.PUBLISHER)
        await put_state_dict(early, _state_dict(0), data.KEY)
        groups = early.strategy.transport_context.get(gloo.GlooProcessGroupCache)
        for round_ in range(1, 4):
            second = _store_name()
            await _initialize_service(second, ts.TorchStoreStrategy(TransportType.Gloo))
            try:
                late = await rank_client(second, role=RankRole.PUBLISHER)
                with pytest.raises(
                    gloo.GlooPairInUseError, match="another live client in this process"
                ):
                    await put_state_dict(late, _state_dict(round_), data.KEY)
                await put_state_dict(early, _state_dict(round_), data.KEY)
                halves = [_gloo_groups(_volume(name)) for name in (first, second)]
                assert (halves, len(groups._process_groups)) == ([1, 0], 1)
            finally:
                await store_api._shutdown_selected(second)
    finally:
        await store_api._shutdown_selected(first)


@pytest.mark.asyncio
async def test_a_dialed_build_cancelled_while_it_closes_its_host_still_closes_its_dial(
    monkeypatch: pytest.MonkeyPatch, rank_client: _RankClient
) -> None:
    """Close a lost dialed build's connection though it is cancelled while its volume handles close.

    A shutdown here stops its volume as it serves, so the build ends without a client and discards.
    """
    reached, release, closing = asyncio.Event(), asyncio.Event(), asyncio.Event()
    serve, connect = carrier_volumes.serve_volume, ServiceAddress.connect
    read, close = ServiceControlPlane.strategy, CarrierVolumeHost.close
    dials: list[ServiceControlPlane] = []
    strategies: list[ts.TorchStoreStrategy] = []

    async def held_after_serve(*args: Any, **kwargs: Any) -> Any:
        served = await serve(*args, **kwargs)
        reached.set()
        await release.wait()
        return served

    async def recorded_connect(self: ServiceAddress) -> ServiceControlPlane:
        dials.append(await connect(self))
        return dials[-1]

    async def recorded(self: ServiceControlPlane) -> ts.TorchStoreStrategy:
        strategies.append(await read(self))
        return strategies[-1]

    async def held_close(self: CarrierVolumeHost) -> None:
        if not strategies or self is not _installed_volume_host(strategies[-1]):
            await close(self)
            return
        closing.set()
        try:
            await asyncio.Event().wait()
        finally:
            await close(self)

    store_name = _store_name()
    async with _other_service(store_name, ts.TorchStoreStrategy()) as other:
        monkeypatch.setattr(carrier_volumes, "serve_volume", held_after_serve)
        monkeypatch.setattr(ServiceAddress, "connect", recorded_connect)
        monkeypatch.setattr(ServiceControlPlane, "strategy", recorded)
        monkeypatch.setattr(CarrierVolumeHost, "close", held_close)
        building = asyncio.ensure_future(
            rank_client(store_name, role=RankRole.PUBLISHER, address=other.address)
        )
        try:
            await asyncio.wait_for(reached.wait(), 30)
            await store_api._shutdown_selected(store_name)
            release.set()
            await asyncio.wait_for(closing.wait(), 30)
            building.cancel()
            with pytest.raises(asyncio.CancelledError):
                await building
            [dial] = dials
            assert dial._closed
        finally:
            release.set()
            await store_api._shutdown_selected(store_name)


class _PauseAfterStartsFirstHold:
    """The publisher lock, whose first release in ``PublisherVolume._start`` waits for ``resume``."""

    def __init__(self, lock: threading.Lock) -> None:
        self._lock = lock
        # Reading the start's code object here makes a rename fail at once rather than as a timeout.
        self._start = PublisherVolume._start.__code__
        self.paused, self.resume = threading.Event(), threading.Event()

    def __enter__(self) -> None:
        self._lock.acquire()

    def __exit__(self, *exc: object) -> None:
        self._lock.release()
        if sys._getframe(1).f_code is self._start and not self.paused.is_set():
            self.paused.set()
            assert self.resume.wait(30), "not resumed within 30 s"


@pytest.mark.asyncio
async def test_a_stop_right_after_a_starts_install_leaves_no_server(
    monkeypatch: pytest.MonkeyPatch, rank_client: _RankClient
) -> None:
    """Stop and close a volume from this thread right after its start's first locked step. No server stays.

    The start checks for a stop and installs its server in that one step, so the close finds the server.
    """
    store_name = _store_name()
    await _initialize_service(store_name)
    gate = _PauseAfterStartsFirstHold(carrier_volumes._publisher_lock)
    monkeypatch.setattr(carrier_volumes, "_publisher_lock", gate)

    def build() -> Any:
        async def run() -> Any:
            return await rank_client(store_name, role=RankRole.PUBLISHER)

        try:
            return asyncio.run(run())
        except Exception as error:
            return error

    building = asyncio.ensure_future(asyncio.to_thread(build))
    try:
        assert await asyncio.to_thread(gate.paused.wait, 30)
        [volume] = stop_publisher_volumes(store_name)
        await volume.close()
        gate.resume.set()
        await asyncio.wait_for(building, 60)
        assert _publisher_servers(store_name) == []
    finally:
        gate.resume.set()
        await store_api._shutdown_selected(store_name)


@pytest.mark.asyncio
async def test_a_racing_client_displaced_by_another_selection_still_closes_at_shutdown(
    monkeypatch: pytest.MonkeyPatch, rank_client: _RankClient
) -> None:
    """Keep a first client that a racing call on another selection displaced, for shutdown to close.

    ``client`` forbids that overlap, but the displaced client stays with its caller. Its volume
    handles close at the store's shutdown as a dropped client's do.
    """
    held, release = asyncio.Event(), asyncio.Event()
    read, connect = ServiceControlPlane.strategy, ServiceAddress.connect
    dials: list[ServiceControlPlane] = []

    async def recorded_connect(self: ServiceAddress) -> ServiceControlPlane:
        dials.append(await connect(self))
        return dials[-1]

    async def held_strategy(self: ServiceControlPlane) -> ts.TorchStoreStrategy:
        if self in dials:
            held.set()
            await release.wait()
        return await read(self)

    store_name = _store_name()
    async with contextlib.AsyncExitStack() as stack:
        other = await stack.enter_async_context(
            _other_service(store_name, ts.TorchStoreStrategy())
        )
        handle = await other.address.connect()
        stack.push_async_callback(handle.close)
        stack.push_async_callback(store_api._shutdown_selected, store_name)
        monkeypatch.setattr(ServiceAddress, "connect", recorded_connect)
        monkeypatch.setattr(ServiceControlPlane, "strategy", held_strategy)
        role = {"role": RankRole.REQUESTER, "group": 0}
        dialing = asyncio.ensure_future(
            rank_client(store_name, address=other.address, **role)
        )
        await asyncio.wait_for(held.wait(), 30)
        displaced = await rank_client(store_name, control_plane=handle, **role)
        release.set()
        latest = await asyncio.wait_for(dialing, 30)
        assert latest is not displaced
        assert store_api._local_clent_map[store_name] is latest
        host = _installed_volume_host(displaced.strategy)
        assert isinstance(host, CarrierVolumeHost) and not host._closed
        await ts.shutdown(store_name)
        assert host._closed


# A caller's plane that shares volume records takes a publisher client in a process that never
# loaded the service backend. argv: the tree's root.
_RECORDS_PLANE = """
import asyncio, json, os, sys
from types import SimpleNamespace

sys.modules["monarch"] = sys.modules["cloudpickle"] = None
sys.path.insert(0, sys.argv[1])
os.environ["TORCHSTORE_CARRIER_HOST"] = "127.0.0.1"
import torchstore as ts
import torchstore.backend as backend_module
from torchstore.carrier_volumes import CarrierVolumeHost
from torchstore.routing.volume_records import VolumeRecords

backend_module._backends["monarch"] = SimpleNamespace(actor_rank=lambda: 0)


class Plane(ts.ControlPlane, VolumeRecords):
    def __init__(self):
        self.records = {}

    async def init(self, publishers, strategy):
        pass

    async def register_layout(self, rank, key, registrations):
        pass

    async def get_layouts(self, key, *, timeout_s=None):
        return {}

    async def strategy(self):
        strategy = ts.TorchStoreStrategy()
        strategy.set_volume_host(CarrierVolumeHost(self.volume_records))
        return strategy

    async def register_volume(self, volume_id, record):
        self.records[volume_id] = record

    async def volume_records(self, volume_ids):
        return {volume_id: self.records[volume_id] for volume_id in volume_ids}


async def main():
    loaded = backend_module.loaded_backend("service") is not None
    plane = Plane()
    await ts.client("records-plane", role=ts.RankRole.PUBLISHER, control_plane=plane)
    registered = sorted(plane.records)
    await ts.shutdown("records-plane")
    print("reply", json.dumps({"loaded": loaded, "registered": registered}), flush=True)


asyncio.run(main())
"""


@pytest.mark.asyncio
async def test_a_publisher_on_a_callers_records_plane_serves_where_no_service_backend_loaded(
    children: _Children,
) -> None:
    """Serve and register a volume for a publisher on a caller's plane that shares records.

    The plane's process never loaded the service backend, which the build loads itself.
    """
    child = children.spawn(_RECORDS_PLANE)
    reply = await asyncio.to_thread(_reply, child, 90)
    assert reply == {"loaded": False, "registered": ["publisher/0"]}
