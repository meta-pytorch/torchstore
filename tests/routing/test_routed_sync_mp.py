# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Routed sync across plain processes that take their identity as arguments, without Monarch.

Each rank runs ``tests/routing/_routed_sync_worker.py`` in a process of its own. The unit
cases check ``num_publishers`` and ``rank`` in this process.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import json
import os
import queue
import re
import socket
import subprocess
import sys
import threading
import uuid
from collections.abc import AsyncIterator, Callable, Collection, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast, NoReturn

import pytest
import torchstore as ts
import torchstore.api as store_api
import torchstore.backend as backend_module
import torchstore.utils as store_utils
from torchstore import carrier_volumes
from torchstore.backend import Backend
from torchstore.carrier import close_all
from torchstore.routing import tcpstore
from torchstore.routing.client import RoutingClient
from torchstore.routing.control_plane import ControlPlane
from torchstore.routing.service import (
    CoordinatorService,
    ServiceAddress,
    ServiceControlPlane,
)
from torchstore.routing.volume_records import VolumeRecords

from ..test_carrier import _in_thread, _kill_group
from . import routed_sync_data as data, tcpstore_fixtures
from ._routed_sync_worker import REPLY
from .file_store_control_plane import FileStoreControlPlane
from .tcpstore_fixtures import _LOCALHOST, _Mesh, _store_name, _strategy, _until
from .test_service_volumes import _servers

pytestmark = [
    pytest.mark.timeout(300),
    pytest.mark.filterwarnings("error::pytest.PytestUnhandledThreadExceptionWarning"),
]

# The shared TCPStore fixture, which pytest finds here under this name.
_joins_its_native_calls = tcpstore_fixtures._joins_its_native_calls

# The tree this file is in, which each rank imports from.
_ROOT = str(Path(__file__).resolve().parents[2])
_WORKER = "tests.routing._routed_sync_worker"
# A process's first reply waits for its imports, which a busy host slows.
_START_TIMEOUT_S = 120.0
_STEP_TIMEOUT_S = 60.0
_SELECTION_LINE = re.compile(
    r"^routing control_plane=service address=(\S+) store=\S+ rank=(\S+) pid=(\d+)$"
)
_SERVE = "carrier serve service=publisher/"
_COORDINATOR = "carrier serve service=coordinator "
# The volume endpoints a routed sync calls on each transport, as one put and one get do.
_TRANSPORT_CALLS = {
    "SharedMemory": ["get", "handshake", "put"],
    "Gloo": ["get", "get_meta", "handshake", "put"],
    "MonarchRPC": ["get", "put"],
}


class _Rank:
    """One rank's process. Threads drain its output, so a full pipe never stalls it."""

    def __init__(
        self, name: str, args: Collection[str], env: Mapping[str, str]
    ) -> None:
        self.name = name
        self.process = subprocess.Popen(
            [sys.executable, "-m", _WORKER, *args],
            cwd=_ROOT,
            env=dict(env),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        self._replies: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._stderr: collections.deque[str] = collections.deque(maxlen=200)
        self._threads = [
            threading.Thread(target=self._read_replies, daemon=True),
            threading.Thread(target=self._read_stderr, daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    def _read_replies(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            if line.startswith(REPLY):
                self._replies.put(json.loads(line[len(REPLY) :]))
        self._replies.put(None)

    def _read_stderr(self) -> None:
        assert self.process.stderr is not None
        self._stderr.extend(self.process.stderr)

    def send(self, line: str) -> None:
        stdin = self.process.stdin
        assert stdin is not None
        stdin.write(f"{line}\n")
        stdin.flush()

    def reply(self, timeout_s: float = _STEP_TIMEOUT_S) -> dict[str, Any]:
        """The rank's next reply, failing with its stderr on an error, an exit or a timeout."""
        try:
            got = self._replies.get(timeout=timeout_s)
        except queue.Empty:
            got = {"error": f"no reply within {timeout_s:g} s"}
        if got is None:
            got = {"error": f"exited with status {self.process.wait(timeout=30)}"}
        if "error" in got:
            try:
                self.kill()
            finally:
                # Also when the kill raises: the rank's own error fails the case.
                pytest.fail(f"{self.name}: {got['error']}\n{''.join(self._stderr)}")
        return got

    def ask(self, command: str, timeout_s: float = _STEP_TIMEOUT_S) -> dict[str, Any]:
        self.send(command)
        return self.reply(timeout_s)

    def kill(self) -> None:
        _kill_group(self.process)
        self.process.wait(timeout=30)
        for thread in self._threads:
            thread.join(timeout=30)


_StartRank = Callable[..., _Rank]


@pytest.fixture
def start_rank() -> Iterator[_StartRank]:
    """Start ranks as ``start_rank(name, *args, env=...)``, each killed after the case though another kill raises."""
    with contextlib.ExitStack() as kills:

        def start(name: str, *args: str, env: Mapping[str, str]) -> _Rank:
            rank = _Rank(name, args, env)
            kills.callback(rank.kill)
            return rank

        yield start


def _env(**changes: str) -> dict[str, str]:
    """This process's environment for a rank: RDMA off as in CI, SharedMemory and Gloo on.

    The TCPStore is here, and volume servers advertise their host's name, as in a job across hosts.
    """
    path = os.environ.get("PYTHONPATH")
    inherited = {k: v for k, v in os.environ.items() if k != "TORCHSTORE_CARRIER_HOST"}
    return {
        **inherited,
        "PYTHONPATH": _ROOT if not path else f"{_ROOT}{os.pathsep}{path}",
        "TORCHSTORE_RDMA_ENABLED": "0",
        "USE_TORCHCOMMS_RDMA": "0",
        "TORCHSTORE_TCPSTORE_HOST": _LOCALHOST,
        # Both on, whatever the caller's environment says. The Gloo case then turns SharedMemory off.
        "TORCHSTORE_SHARED_MEMORY_ENABLED": "1",
        "TORCHSTORE_GLOO_ENABLED": "1",
        # Bounds a stuck Gloo transfer's wait, and so a rank's exit, as other tests do.
        "TORCHSTORE_GLOO_INIT_TIMEOUT": "10",
        **changes,
    }


def _ask_all(ranks: Collection[_Rank], command: str) -> list[dict[str, Any]]:
    """Send ``command`` to every rank before reading any reply, so they run it at once."""
    for rank in ranks:
        rank.send(command)
    return [rank.reply() for rank in ranks]


def _faults(publishers: dict[int, _Rank], requesters: Collection[_Rank]) -> None:
    """Stop, then kill, publisher 1: each requester's next read fails within its bound.

    A requester reads every mapping in volume order, so publisher 0 answers first.
    """
    assert _ask_all(requesters, "deadline 5") == [{"deadline": 5.0}] * len(requesters)
    stopped = publishers[1]
    tcpstore_fixtures._stop(stopped.process)
    for reply in _ask_all(requesters, "pull-fault"):
        fault = reply["fault"]
        assert fault["error"] == "RuntimeError", fault
        assert fault["message"].startswith("Mapping is missing"), fault
        assert fault["cause"] == "CallTimeoutError", fault
        assert "within its 5 s deadline" in fault["cause_message"], fault
        assert 5 <= fault["elapsed_s"] < 20, fault
    # Reaped first, so the next read finds the connection lost.
    stopped.kill()
    del publishers[1]
    for reply in _ask_all(requesters, "pull-fault"):
        fault = reply["fault"]
        assert (fault["error"], fault["cause"]) == ("RuntimeError", "PeerLostError")
        assert fault["elapsed_s"] < 10, fault


def _check_report(
    name: str,
    report: dict[str, Any],
    transport: str,
    address: str,
    selections: list[str],
) -> None:
    """Check what one rank loaded and logged, from its report at shutdown."""
    assert report["torchstore"].startswith(_ROOT), report["torchstore"]
    # No Monarch runtime, RDMA or job module, and no actor-based TorchStore module.
    assert report["loaded"] == []
    lines = report["lines"]
    picked = [m.groups() for line in lines if (m := _SELECTION_LINE.match(line))]
    assert picked == [(address, rank, str(report["pid"])) for rank in selections]
    publishing = [rank for rank in selections if rank.startswith("publisher/")]
    for volume in publishing:
        pid = f"pid={report['pid']}"
        serves = [
            line
            for line in lines
            if line.startswith(f"carrier serve service={volume} ")
        ]
        assert len(serves) == 1 and serves[0].endswith(pid), serves
        volumes = [
            line for line in lines if line.startswith(f"routing volume={volume} ")
        ]
        assert len(volumes) == 1 and volumes[0].endswith(pid), volumes
        stats = [
            line
            for line in lines
            if line.startswith(f"carrier stats service={volume} ")
        ]
        assert len(stats) == 1 and not stats[0].startswith(
            f"carrier stats service={volume} calls=0 "
        ), stats
        called = sorted(endpoint for endpoint, n in report["calls"].items() if n)
        assert called == _TRANSPORT_CALLS[transport], report["calls"]
    if name.startswith("requester/"):
        connects = sorted(
            line.split()[2].removeprefix("service=")
            for line in lines
            if line.startswith("carrier connect ")
        )
        assert connects == ["coordinator", "publisher/0", "publisher/1"], connects
    moved = publishing or name.startswith("requester/")
    # Every transport line names the case's transport, and only ranks that move data log any.
    assert sorted(report["transports"]) == ([transport] if moved else []), report


@pytest.mark.parametrize("transport", ["SharedMemory", "Gloo", "MonarchRPC"])
def test_routed_sync_across_plain_processes_with_ranks_as_arguments(
    start_rank: _StartRank, transport: str
) -> None:
    """Sync 2 publishers into 3 requesters on 2 meshes, bitwise for 3 rounds, without Monarch.

    Each process gets its identity as arguments and the driver's address on stdin. Each case adds
    one scenario so its 5 or 6 processes start only 3 times: SharedMemory's driver is also
    publisher 0, Gloo blocks Monarch and cloudpickle, and MonarchRPC stops and then kills a publisher.
    """
    store = f"routed-sync-mp-{uuid.uuid4()}"
    gloo = transport == "Gloo"
    shared = transport == "SharedMemory"
    flags = ["--store", store, *(["--block-monarch"] if gloo else [])]
    env = _env(TORCHSTORE_SHARED_MEMORY_ENABLED="0") if gloo else _env()
    driver = start_rank(
        "driver",
        *("--role", "driver", "--num-publishers", str(data.PUBLISHER_RANKS)),
        *("--transport", "MonarchRPC" if transport == "MonarchRPC" else "Unset"),
        *(["--rank", "0"] if shared else []),
        *flags,
        env=env,
    )
    address = driver.reply(_START_TIMEOUT_S)
    address_line = REPLY + json.dumps(address)
    printed = f"{address['address']['host']}:{address['address']['port']}"

    # Requesters first: bootstrap never waits on a publisher.
    requesters: dict[str, _Rank] = {}
    for group, (num_ranks, _) in enumerate(data.REQUESTERS):
        for rank in range(num_ranks):
            name = f"requester/{group}/{rank}"
            requesters[name] = start_rank(
                name,
                *("--role", "requester", "--group", str(group), "--rank", str(rank)),
                *flags,
                env=env,
            )
    for child in requesters.values():
        child.send(address_line)
    for child in requesters.values():
        assert child.reply(_START_TIMEOUT_S) == {"ready": child.process.pid}
    # Group 1 starts its layout exchange before any publisher exists, and the coordinator holds
    # both lookups until publishers register. Group 0 runs the exchange inside its first read,
    # as TitanRL's generators do.
    group_1 = [
        requesters[f"requester/1/{rank}"] for rank in range(data.REQUESTERS[1][0])
    ]
    # Each rank only acknowledges the start: a wait of 0 sees any exchange still in flight.
    _ask_all(group_1, "layouts")
    # Asked past the poll's own 60 s: a coordinator that holds nothing then fails with the poll's message.
    lookups = driver.ask(f"lookups {len(group_1)}", timeout_s=2 * _STEP_TIMEOUT_S)
    assert lookups == {"lookups": len(group_1)}

    publishers: dict[int, _Rank] = {}
    for rank in range(data.PUBLISHER_RANKS):
        if shared and rank == 0:
            publishers[0] = driver
            driver.send("publish")
            continue
        # Only locality reads the host name, which each volume's record carries.
        publisher_env = (
            {**env, "TORCHSTORE_HOSTNAME": f"fake-publisher-{rank}"} if gloo else env
        )
        publishers[rank] = start_rank(
            f"publisher/{rank}",
            *("--role", "publisher", "--rank", str(rank)),
            *flags,
            env=publisher_env,
        )
        publishers[rank].send(address_line)
    for child in publishers.values():
        assert child.reply(_START_TIMEOUT_S) == {"ready": child.process.pid}

    for round_ in range(data.ROUNDS):
        pushed = _ask_all(publishers.values(), f"push {round_}")
        assert pushed == [{"pushed": round_}] * len(publishers)
        pulled = _ask_all(requesters.values(), f"pull {round_}")
        assert pulled == [{"pulled": round_}] * len(requesters)

    if transport == "MonarchRPC":
        _faults(publishers, requesters.values())

    own = {f"publisher/{r}": c for r, c in publishers.items() if c is not driver}
    for name, child in {**requesters, **own, "driver": driver}.items():
        report = child.ask("shutdown")["report"]
        assert child.process.wait(timeout=60) == 0, name
        selections = [name]
        if name == "driver":
            selections = ["-", "publisher/0"] if shared else ["-"]
            served = [line for line in report["lines"] if line.startswith(_COORDINATOR)]
            assert len(served) == 1, report["lines"]
        _check_report(name, report, transport, printed, selections)
        if gloo and name.startswith("publisher/"):
            # The host name changes no address: ranks still dial this host's name.
            served = [line for line in report["lines"] if line.startswith(_SERVE)]
            assert f" address={socket.gethostname()}:" in served[0], served
            # Its volume keeps its half of its own pair and of each requester's, though those exited.
            assert report["pairs"] == 1 + len(requesters), report["pairs"]
        if gloo and name.startswith("requester/"):
            # Host names come only from the volumes' records.
            assert report["hostnames"] == {
                f"publisher/{r}": f"fake-publisher-{r}"
                for r in range(data.PUBLISHER_RANKS)
            }


# In-process cases need no Monarch. Those that build clients use a real coordinator service.


def _refuse(*args: Any, **kwargs: Any) -> NoReturn:
    raise AssertionError("a refused call took a later step")


def _refuse_monarch(
    monkeypatch: pytest.MonkeyPatch, rank: Callable[[], int] = _refuse
) -> None:
    """Stand in for the Monarch backend: no actor context is current, and ``rank`` names ranks.

    Every actor call refuses.
    """
    backend = SimpleNamespace(
        actor_rank=lambda: None,
        rank=rank,
        spawn_volumes=_refuse,
        controller_volume=_refuse,
        controller=_refuse,
        coordinator=_refuse,
    )
    monkeypatch.setitem(backend_module._backends, "monarch", cast(Backend, backend))


def _rank_of(client: Any) -> str:
    assert isinstance(client, RoutingClient)
    return client.rank_id


async def _service_store(store_name: str, num_publishers: int = 2) -> ServiceAddress:
    address = await ts.initialize(
        num_publishers=num_publishers,
        strategy=_strategy(),
        client_type=ts.ClientType.ROUTING,
        store_name=store_name,
        control_plane="service",
    )
    assert isinstance(address, ServiceAddress)
    return address


async def _other_service(
    cleanups: contextlib.AsyncExitStack, store_name: str
) -> ServiceAddress:
    """Serve a coordinator for 2 publishers outside this process's records until the case ends."""
    publishers = ("publisher/0", "publisher/1")
    served = tcpstore_fixtures._other_service(store_name, _strategy(), publishers)
    return (await cleanups.enter_async_context(served)).address


@contextlib.asynccontextmanager
async def _cleanups() -> AsyncIterator[contextlib.AsyncExitStack]:
    """Run each cleanup pushed here in reverse order though another raises. The case's own error wins.

    A bare ``AsyncExitStack`` would report a failing cleanup's error in place of the case's.
    """
    stack = contextlib.AsyncExitStack()
    try:
        yield stack
    except BaseException:
        with contextlib.suppress(Exception):
            await stack.aclose()
        raise
    await stack.aclose()


def _counted_closes(monkeypatch: pytest.MonkeyPatch) -> list[ServiceControlPlane]:
    """Record each service handle that closes from now on, in order."""
    closed: list[ServiceControlPlane] = []
    close = ServiceControlPlane.close

    async def counted_close(self: ServiceControlPlane) -> None:
        closed.append(self)
        await close(self)

    monkeypatch.setattr(ServiceControlPlane, "close", counted_close)
    return closed


def _held_strategy(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[asyncio.Event, asyncio.Event]:
    """Hold strategy reads until ``release`` is set. ``entered`` marks the first."""
    entered, release = asyncio.Event(), asyncio.Event()
    strategy = ServiceControlPlane.strategy

    async def held(self: ServiceControlPlane) -> ts.TorchStoreStrategy:
        entered.set()
        await release.wait()
        return await strategy(self)

    monkeypatch.setattr(ServiceControlPlane, "strategy", held)
    return entered, release


@pytest.fixture
def _in_process(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Serve on localhost with empty selection state that is restored after. Close the servers a case leaves."""
    monkeypatch.setenv("TORCHSTORE_TCPSTORE_HOST", _LOCALHOST)
    monkeypatch.setenv("TORCHSTORE_CARRIER_HOST", _LOCALHOST)
    for name in (
        "_local_clent_map",
        "_control_plane_map",
        "_owned_service_map",
        "_forked_stores",
        "_building_ranks",
        "_dropped_carriers",
        "_spmd_state_map",
    ):
        monkeypatch.setattr(store_api, name, {})
    for name in ("_building", "_attempted", "_released_stores", "_standard_stores"):
        monkeypatch.setattr(store_api, name, set())
    monkeypatch.setattr(carrier_volumes, "_publishers", {})
    yield
    _in_thread(close_all(grace_s=0))


@dataclass(frozen=True)
class _NeverDialed(ts.ControlPlaneAddress):
    """Fails the case if any call dials it."""

    async def connect(self) -> ControlPlane:
        raise AssertionError("dialed an address")

    def __str__(self) -> str:
        return "never-dialed"


class _NoRecords(ControlPlane):
    """A control plane without volume records, so its volumes would be Monarch actors."""

    def __init__(self) -> None:
        self.closes = 0

    async def init(self, publishers: Collection[str], strategy: Any) -> None:
        _refuse()

    async def strategy(self) -> ts.TorchStoreStrategy:
        _refuse()

    async def register_layout(self, rank: str, key: str, registrations: Any) -> None:
        _refuse()

    async def get_layouts(
        self, key: str, *, timeout_s: float | None = None
    ) -> ts.Registrations:
        _refuse()

    async def close(self) -> None:
        self.closes += 1


class _Records(ControlPlane, VolumeRecords):
    """A caller's plane that shares volume records, forwarding every call to a service handle."""

    def __init__(self, handle: ServiceControlPlane) -> None:
        self.handle = handle

    async def init(self, publishers: Collection[str], strategy: Any) -> None:
        await self.handle.init(publishers, strategy)

    async def strategy(self) -> ts.TorchStoreStrategy:
        return await self.handle.strategy()

    async def register_layout(self, rank: str, key: str, registrations: Any) -> None:
        await self.handle.register_layout(rank, key, registrations)

    async def get_layouts(
        self, key: str, *, timeout_s: float | None = None
    ) -> ts.Registrations:
        return await self.handle.get_layouts(key, timeout_s=timeout_s)

    async def register_volume(self, volume_id: str, record: bytes) -> None:
        await self.handle.register_volume(volume_id, record)

    async def volume_records(self, volume_ids: Collection[str]) -> dict[str, bytes]:
        return await self.handle.volume_records(volume_ids)


@dataclass(frozen=True)
class _Dials(ts.ControlPlaneAddress):
    """Dials one given control plane."""

    plane: ControlPlane

    async def connect(self) -> ControlPlane:
        return self.plane

    def __str__(self) -> str:
        return "dials"


@pytest.mark.usefixtures("_in_process")
@pytest.mark.asyncio
async def test_num_publishers_names_the_ranks_that_publishers_pass() -> None:
    """Serve publisher 1 of 2 from its argument. Refuse rank 2, closing the volume it served."""
    first, second = _store_name(), _store_name()
    async with _cleanups() as cleanups:
        cleanups.push_async_callback(store_api._shutdown_selected, first)
        cleanups.push_async_callback(store_api._shutdown_selected, second)
        await _service_store(first)
        client = await ts.client(first, role=ts.RankRole.PUBLISHER, rank=1)
        assert _rank_of(client) == "publisher/1"
        assert list(carrier_volumes._publishers) == [(first, "publisher/1")]
        await _service_store(second)
        with pytest.raises(KeyError, match="'publisher/2' is not a publisher rank"):
            await ts.client(second, role=ts.RankRole.PUBLISHER, rank=2)
        assert list(carrier_volumes._publishers) == [(first, "publisher/1")]
        assert _servers(second) == ["coordinator"]


@pytest.mark.parametrize("wrapped", [False, True], ids=["handle", "callers-plane"])
@pytest.mark.usefixtures("_in_process")
@pytest.mark.asyncio
async def test_any_plane_with_volume_records_takes_a_count_and_ranks(
    wrapped: bool,
) -> None:
    """Take both arguments on a plane that shares volume records, whatever its type or name."""
    store_name = _store_name()
    async with _cleanups() as cleanups:
        service = await CoordinatorService.start(store_name, host=_LOCALHOST)
        cleanups.push_async_callback(service.close)
        handle = await service.connect()
        cleanups.push_async_callback(handle.close)
        cleanups.push_async_callback(store_api._shutdown_selected, store_name)
        plane: ControlPlane = _Records(handle) if wrapped else handle
        address = await ts.initialize(
            num_publishers=2,
            strategy=_strategy(),
            client_type=ts.ClientType.ROUTING,
            store_name=store_name,
            control_plane=plane,
        )
        assert address is None
        client = await ts.client(store_name, role=ts.RankRole.PUBLISHER, rank=1)
        assert _rank_of(client) == "publisher/1"
        assert list(carrier_volumes._publishers) == [(store_name, "publisher/1")]


# Upstream's text with the publisher count added.
_INCOMPLETE = (
    "^routing mode requires both mesh and strategy, or num_publishers in place of"
)


# A routing initialize on 'service', which a case changes. A count's own checks need none of it.
_SERVICE_INIT: dict[str, Any] = {
    "num_publishers": 2,
    "strategy": _strategy(),
    "client_type": ts.ClientType.ROUTING,
    "control_plane": "service",
}


@pytest.mark.parametrize(
    ("changes", "error", "match"),
    [
        (
            {
                "num_publishers": 2,
                "mesh": _Mesh(2),
                "client_type": ts.ClientType.ROUTING,
                "control_plane": "service",
            },
            ValueError,
            "^pass mesh or num_publishers, not both$",
        ),
        (
            {"strategy": _strategy(), "client_type": ts.ClientType.ROUTING},
            RuntimeError,
            _INCOMPLETE,
        ),
        ({**_SERVICE_INIT, "strategy": None}, RuntimeError, _INCOMPLETE),
        (
            {"num_publishers": 0},
            ValueError,
            "^num_publishers must be at least 1, got 0$",
        ),
        ({"num_publishers": -1}, ValueError, "at least 1, got -1$"),
        (
            {"num_publishers": True},
            TypeError,
            "^num_publishers must be an int, got True$",
        ),
        ({"num_publishers": 2.0}, TypeError, "must be an int, got 2.0$"),
        ({"num_publishers": "2"}, TypeError, "must be an int, got '2'$"),
        (
            {**_SERVICE_INIT, "num_publishers": 16_385},
            ValueError,
            "^'service' takes at most 16384 publishers, got 16385: four times",
        ),
        (
            {**_SERVICE_INIT, "num_publishers": None, "mesh": _Mesh(16_385)},
            ValueError,
            "^'service' takes at most 16384 publishers, got 16385: four times",
        ),
    ],
    ids=[
        "mesh-and-count",
        "neither",
        "no-strategy",
        "zero",
        "negative",
        "bool",
        "float",
        "str",
        "over-the-cap",
        "mesh-over-the-cap",
    ],
)
@pytest.mark.usefixtures("_in_process")
@pytest.mark.asyncio
async def test_initialize_takes_one_publisher_source_before_serving_anything(
    monkeypatch: pytest.MonkeyPatch, changes: dict[str, Any], error: type, match: str
) -> None:
    """Refuse a mesh with a count, neither, no strategy, or a count that is not a positive int.

    A count over the service's cap, four times the design's largest scale, is refused too.
    """
    _refuse_monarch(monkeypatch)
    monkeypatch.setattr(CoordinatorService, "start", _refuse)
    arguments: dict[str, Any] = {"store_name": _store_name(), **changes}
    with pytest.raises(error, match=match):
        await ts.initialize(**arguments)


@pytest.mark.timeout(120)
@pytest.mark.usefixtures("_in_process")
@pytest.mark.asyncio
async def test_the_service_starts_for_the_most_publishers_it_takes() -> None:
    """Start the service for the most publishers, publishing each one's key, then shut it down.

    Their list, as the rendezvous encodes it, also fits one TCPStore value.
    """
    cap = store_api._MAX_SERVICE_PUBLISHERS
    names = [f"publisher/{rank}" for rank in range(cap)]
    assert len(tcpstore._encode_publishers(names)) <= tcpstore._MAX_VALUE_BYTES
    store_name = _store_name()
    try:
        assert isinstance(await _service_store(store_name, cap), ServiceAddress)
    finally:
        await store_api._shutdown_selected(store_name)
    assert _servers(store_name) == []


@pytest.mark.parametrize("kind", ["monarch", "filestore", "standard"])
@pytest.mark.usefixtures("_in_process")
@pytest.mark.asyncio
async def test_num_publishers_is_refused_where_volumes_are_monarch_actors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, kind: str
) -> None:
    """Refuse a count where Monarch spawns the volumes before anything spawns."""
    _refuse_monarch(monkeypatch)
    monkeypatch.setattr(CoordinatorService, "start", _refuse)
    match = "^num_publishers is only for control planes whose publishers serve"
    if kind == "standard":
        call = ts.initialize(num_publishers=2, store_name=_store_name())
        match = "^standard stores take no num_publishers$"
    else:
        plane: str | ControlPlane = "monarch"
        if kind == "filestore":
            plane = FileStoreControlPlane(str(tmp_path / "plane"))
        call = ts.initialize(
            num_publishers=2,
            strategy=_strategy(),
            client_type=ts.ClientType.ROUTING,
            store_name=_store_name(),
            control_plane=plane,
        )
    with pytest.raises(ValueError, match=match):
        await call


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"rank": 0}, ValueError, "^rank is for routing clients: pass role$"),
        ({"rank": -1}, ValueError, "^rank must be at least 0, got -1$"),
        ({"rank": True}, TypeError, "^rank must be an int, got True$"),
        ({"rank": "0"}, TypeError, "must be an int, got '0'$"),
        ({"rank": 1.0}, TypeError, "must be an int, got 1.0$"),
    ],
    ids=["no-role", "negative", "bool", "str", "float"],
)
@pytest.mark.usefixtures("_in_process")
@pytest.mark.asyncio
async def test_rank_is_checked_before_anything_dials(
    monkeypatch: pytest.MonkeyPatch, kwargs: dict[str, Any], error: type, match: str
) -> None:
    """Refuse a rank without a role or that is not a non-negative int, dialing nothing."""
    _refuse_monarch(monkeypatch)
    monkeypatch.setattr(CoordinatorService, "start", _refuse)
    store_name = _store_name()
    if kwargs != {"rank": 0}:
        role = ts.RankRole.PUBLISHER
        kwargs = {**kwargs, "role": role, "address": _NeverDialed(store_name)}
    with pytest.raises(error, match=match):
        await ts.client(store_name, **kwargs)


@pytest.mark.parametrize("via", ["monarch", "instance", "address"])
@pytest.mark.usefixtures("_in_process")
@pytest.mark.asyncio
async def test_rank_is_refused_where_ranks_come_from_monarch(
    monkeypatch: pytest.MonkeyPatch, via: str
) -> None:
    """Refuse a rank where volumes are Monarch actors, asking Monarch nothing.

    A connection the call dialed closes. A caller's own plane stays open.
    """
    _refuse_monarch(monkeypatch)
    monkeypatch.setattr(CoordinatorService, "start", _refuse)
    store_name = _store_name()
    plane = _NoRecords()
    kwargs: dict[str, Any] = {}
    match = "^rank is only for control planes whose pub"
    if via == "monarch":
        match = r"^rank needs the store's address \(address=\) or control plane"
    elif via == "instance":
        kwargs = {"control_plane": plane}
    else:
        kwargs = {"address": _Dials(store_name, plane)}
    with pytest.raises(ValueError, match=match):
        await ts.client(store_name, role=ts.RankRole.PUBLISHER, rank=0, **kwargs)
    if via != "monarch":
        # The monarch arm passes no plane and so has none to close.
        assert plane.closes == (1 if via == "address" else 0)
    assert store_api._local_clent_map == store_api._control_plane_map == {}
    # Refused before the build began unless the call dialed first.
    assert (store_name in store_api._attempted) is (via == "address")


@pytest.mark.parametrize("rank", [None, 0], ids=["no-rank", "rank"])
@pytest.mark.parametrize("via", ["address", "instance"])
@pytest.mark.parametrize(
    ("role", "group", "match"),
    [
        (ts.RankRole.PUBLISHER, 0, "^publishers are a single mesh and take no group$"),
        (
            ts.RankRole.REQUESTER,
            None,
            "^requesters must pass the index of their requester mesh$",
        ),
    ],
    ids=["publisher-group", "requester-no-group"],
)
@pytest.mark.usefixtures("_in_process")
@pytest.mark.asyncio
async def test_a_client_refused_for_its_role_or_group_marks_nothing(
    monkeypatch: pytest.MonkeyPatch,
    role: ts.RankRole,
    group: int | None,
    match: str,
    via: str,
    rank: int | None,
) -> None:
    """Refuse a role and group that name no mesh before anything starts, with or without a rank.

    So the store keeps no mark, and its shutdown stays Monarch's.
    """
    _refuse_monarch(monkeypatch)
    monkeypatch.setattr(CoordinatorService, "start", _refuse)
    store_name = _store_name()
    plane = _NoRecords()
    selection: dict[str, Any] = (
        {"address": _NeverDialed(store_name)}
        if via == "address"
        else {"control_plane": plane}
    )
    ranked: dict[str, Any] = {} if rank is None else {"rank": rank}
    with pytest.raises(ValueError, match=match):
        await ts.client(store_name, role=role, group=group, **ranked, **selection)
    assert store_name not in store_api._attempted and store_api._building == set()
    if via == "instance":
        # The address arm never dials. Only a caller's own plane could be closed.
        assert plane.closes == 0


class _FailingClose(_NoRecords):
    """A plane without records whose close fails, as a broken dialed connection's can."""

    async def close(self) -> None:
        self.closes += 1
        raise ConnectionResetError("the dialed connection broke while closing")


@pytest.mark.usefixtures("_in_process")
@pytest.mark.asyncio
async def test_a_refused_rank_keeps_its_error_when_closing_the_dial_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raise the refusal's ValueError though closing the connection it dialed fails."""
    _refuse_monarch(monkeypatch)
    monkeypatch.setattr(CoordinatorService, "start", _refuse)
    store_name = _store_name()
    plane = _FailingClose()
    address = _Dials(store_name, plane)
    with pytest.raises(ValueError, match="^rank is only for control planes whose pub"):
        await ts.client(store_name, role=ts.RankRole.PUBLISHER, rank=0, address=address)
    assert plane.closes == 1
    assert store_api._local_clent_map == store_api._control_plane_map == {}


@pytest.mark.usefixtures("_in_process")
@pytest.mark.asyncio
async def test_a_failed_monarch_rank_lookup_closes_the_dialed_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Close the dial when Monarch cannot name the rank on a plane without records."""

    def no_monarch() -> NoReturn:
        raise ModuleNotFoundError("No module named 'monarch'")

    _refuse_monarch(monkeypatch, rank=no_monarch)
    monkeypatch.setattr(CoordinatorService, "start", _refuse)
    store_name = _store_name()
    plane = _NoRecords()
    with pytest.raises(ModuleNotFoundError, match="^No module named 'monarch'$"):
        await ts.client(
            store_name, role=ts.RankRole.PUBLISHER, address=_Dials(store_name, plane)
        )
    assert plane.closes == 1
    assert store_api._local_clent_map == store_api._control_plane_map == {}


@pytest.mark.parametrize("role", [ts.RankRole.PUBLISHER, ts.RankRole.REQUESTER])
@pytest.mark.usefixtures("_in_process")
@pytest.mark.asyncio
async def test_a_plain_process_without_rank_raises_and_closes_its_connection(
    monkeypatch: pytest.MonkeyPatch, role: ts.RankRole
) -> None:
    """Refuse a client outside any Monarch actor that passes no rank, asking Monarch only for its context.

    The connection it dialed closes once, and it records, caches and serves nothing.
    """
    _refuse_monarch(monkeypatch)
    store_name = _store_name()
    async with _cleanups() as cleanups:
        address = await _other_service(cleanups, store_name)
        closed = _counted_closes(monkeypatch)
        group = 0 if role == ts.RankRole.REQUESTER else None
        with pytest.raises(
            ValueError, match="^pass rank=: this call has no rank of its own"
        ):
            await ts.client(store_name, role=role, group=group, address=address)
        assert len(closed) == 1
        assert store_api._local_clent_map == store_api._control_plane_map == {}
        assert carrier_volumes._publishers == {}
        assert _servers(store_name) == ["coordinator"]


@pytest.mark.parametrize("via", ["record", "address"])
@pytest.mark.usefixtures("_in_process")
@pytest.mark.asyncio
async def test_a_refused_client_keeps_the_connection_this_process_recorded(
    monkeypatch: pytest.MonkeyPatch, via: str
) -> None:
    """Refuse a client without ``rank`` on the store this process serves, closing nothing.

    It reuses the record, also through the record's own address, which then serves the next
    client.
    """
    _refuse_monarch(monkeypatch)
    store_name = _store_name()
    try:
        address = await _service_store(store_name)
        monkeypatch.setattr(CoordinatorService, "start", _refuse)
        closed = _counted_closes(monkeypatch)
        kwargs: dict[str, Any] = {"address": address} if via == "address" else {}
        call = ts.client(store_name, role=ts.RankRole.REQUESTER, group=0, **kwargs)
        with pytest.raises(
            ValueError, match="^pass rank=: this call has no rank of its own"
        ):
            await call
        assert closed == []
        client = await ts.client(
            store_name, role=ts.RankRole.PUBLISHER, rank=0, **kwargs
        )
        assert _rank_of(client) == "publisher/0"
    finally:
        await store_api._shutdown_selected(store_name)


@pytest.mark.parametrize("phase", ["building", "cancelled", "refused", "refused_plane"])
@pytest.mark.usefixtures("_in_process")
@pytest.mark.asyncio
async def test_shutdown_of_a_store_this_process_only_connects_to_asks_monarch_nothing(
    monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    """Release nothing and ask Monarch nothing while this process's only client of a store builds.

    So does a shutdown after a client of it was cancelled or refused, through its address or a plane.
    A build the shutdown left running then completes and records the store, for the next shutdown.
    """
    _refuse_monarch(monkeypatch)
    store_name = _store_name()
    async with _cleanups() as cleanups:
        address = await _other_service(cleanups, store_name)
        cleanups.push_async_callback(store_api._shutdown_selected, store_name)
        if phase in ("building", "cancelled"):
            entered, release = _held_strategy(monkeypatch)
            building = asyncio.ensure_future(
                ts.client(
                    store_name, role=ts.RankRole.PUBLISHER, rank=0, address=address
                )
            )
            await asyncio.wait_for(entered.wait(), 30)
            if phase == "cancelled":
                # As a caller's timeout cancels it: the build still counts as begun.
                building.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await building
                await ts.shutdown(store_name)
            else:
                await ts.shutdown(store_name)
                release.set()
                client = await asyncio.wait_for(building, 30)
                assert _rank_of(client) == "publisher/0"
                assert store_api._control_plane_map[store_name].address == address
            await ts.shutdown(store_name)
        else:
            kwargs: dict[str, Any] = {"address": address}
            if phase == "refused_plane":
                plane = await address.connect()
                cleanups.push_async_callback(plane.close)
                kwargs = {"control_plane": plane}
            with pytest.raises(KeyError, match="'publisher/2' is not a publisher rank"):
                await ts.client(
                    store_name, role=ts.RankRole.PUBLISHER, rank=2, **kwargs
                )
            await ts.shutdown(store_name)
        assert store_api._control_plane_map == store_api._local_clent_map == {}
        assert carrier_volumes._publishers == {}
        assert _servers(store_name) == ["coordinator"]


@pytest.mark.parametrize(
    ("via", "held_at"),
    [
        ("service", "before_its_read"),
        ("service", "after_its_read"),
        # The shutdown leaves a caller's plane open, so a read before it fails alike.
        ("callers-plane", "after_its_read"),
    ],
)
@pytest.mark.usefixtures("_in_process")
@pytest.mark.asyncio
async def test_a_shutdown_overlapping_a_client_through_this_processs_record_fails_that_client(
    monkeypatch: pytest.MonkeyPatch, via: str, held_at: str
) -> None:
    """Fail a client connecting through the record this process holds once an overlapping shutdown releases it.

    The client raises RuntimeError and caches nothing whether its strategy read came before the shutdown or after.
    A publisher on a caller's plane also stops the volume it started after that shutdown.
    """
    store_name = _store_name()
    entered, release = asyncio.Event(), asyncio.Event()
    strategy = ServiceControlPlane.strategy

    async def held(self: ServiceControlPlane) -> ts.TorchStoreStrategy:
        if held_at == "before_its_read":
            entered.set()
            await release.wait()
            return await strategy(self)
        read = await strategy(self)
        entered.set()
        await release.wait()
        return read

    async with _cleanups() as cleanups:
        kwargs: dict[str, Any] = {"role": ts.RankRole.REQUESTER, "group": 0, "rank": 0}
        if via == "service":
            await _service_store(store_name)
        else:
            service = await CoordinatorService.start(store_name, host=_LOCALHOST)
            cleanups.push_async_callback(service.close)
            handle = await service.connect()
            cleanups.push_async_callback(handle.close)
            await ts.initialize(
                num_publishers=2,
                strategy=_strategy(),
                client_type=ts.ClientType.ROUTING,
                store_name=store_name,
                control_plane=handle,
            )
            kwargs = {"role": ts.RankRole.PUBLISHER, "rank": 1}
        cleanups.push_async_callback(store_api._shutdown_selected, store_name)
        cleanups.callback(release.set)
        monkeypatch.setattr(ServiceControlPlane, "strategy", held)
        building = asyncio.ensure_future(ts.client(store_name, **kwargs))
        await asyncio.wait_for(entered.wait(), 30)
        await ts.shutdown(store_name)
        release.set()
        failed = (
            "this control plane is closed"
            if via == "service" and held_at == "before_its_read"
            else "was shut down or replaced while this client was being built"
        )
        with pytest.raises(RuntimeError, match=failed):
            await asyncio.wait_for(building, 30)
        assert store_name not in store_api._local_clent_map
        assert store_name not in store_api._control_plane_map
        if via == "callers-plane":
            assert carrier_volumes._publishers == {}
            assert _servers(store_name) == ["coordinator"]


class _PausedSecondStep:
    """The selection lock, pausing ``thread`` before its second locked step until ``resume`` is set."""

    def __init__(self) -> None:
        self.lock = store_api._selection_lock
        self.paused, self.resume = threading.Event(), threading.Event()
        self.thread: threading.Thread | None = None
        self.steps = 0

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        if threading.current_thread() is self.thread:
            self.steps += 1
            if self.steps == 2:
                self.paused.set()
                assert self.resume.wait(30)
        return self.lock.acquire(blocking, timeout)

    def release(self) -> None:
        self.lock.release()

    def locked(self) -> bool:
        return self.lock.locked()

    def __enter__(self) -> bool:
        return self.acquire()

    def __exit__(self, *exc_info: object) -> None:
        self.release()


@pytest.mark.parametrize("via", ["record", "address"])
@pytest.mark.usefixtures("_in_process")
@pytest.mark.asyncio
async def test_a_shutdown_on_another_thread_before_a_build_reads_the_record_fails_that_client(
    monkeypatch: pytest.MonkeyPatch, via: str
) -> None:
    """Fail a client through this process's record that another thread's shutdown releases before the build reads it.

    The driver's call and a rank's call through its record's address both raise RuntimeError and cache nothing.
    """
    store_name = _store_name()
    lock = _PausedSecondStep()
    requester: dict[str, Any] = {"role": ts.RankRole.REQUESTER, "group": 0, "rank": 0}
    outcome: list[object] = []

    def call() -> None:
        try:
            outcome.append(asyncio.run(ts.client(store_name, **requester)))
        except BaseException as error:
            outcome.append(error)

    async with _cleanups() as cleanups:
        if via == "record":
            await _service_store(store_name)
        else:
            requester["address"] = await _other_service(cleanups, store_name)
            await ts.client(store_name, **requester)
            ts.reset_client(store_name)
        cleanups.push_async_callback(store_api._shutdown_selected, store_name)
        cleanups.callback(lock.resume.set)
        monkeypatch.setattr(store_api, "_selection_lock", lock)
        lock.thread = threading.Thread(target=call, daemon=True)
        lock.thread.start()
        assert await asyncio.to_thread(lock.paused.wait, 30)
        await ts.shutdown(store_name)
        lock.resume.set()
        await asyncio.to_thread(lock.thread.join, 30)
        [error] = outcome
        assert isinstance(error, RuntimeError), error
        assert "shut down or replaced while this client was being built" in str(error)
        assert store_name not in store_api._local_clent_map
        assert store_name not in store_api._control_plane_map


@pytest.mark.usefixtures("_in_process")
@pytest.mark.asyncio
async def test_an_actors_rank_names_the_client_unless_rank_is_passed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Take a Monarch actor's rank for both roles without ``rank``, and the argument over it."""
    store_name = _store_name()
    try:
        await _service_store(store_name)
        monkeypatch.setattr(store_api, "_actor_rank", lambda: 1)
        client = await ts.client(store_name, role=ts.RankRole.PUBLISHER)
        assert _rank_of(client) == "publisher/1"
        ts.reset_client(store_name)
        monkeypatch.setattr(store_api, "_actor_rank", lambda: 0)
        again = await ts.client(store_name, role=ts.RankRole.PUBLISHER, rank=1)
        assert _rank_of(again) == "publisher/1" and again is not client
        # The client made again adopts the volume the first one started.
        assert list(carrier_volumes._publishers) == [(store_name, "publisher/1")]
        ts.reset_client(store_name)
        monkeypatch.setattr(store_api, "_actor_rank", lambda: 1)
        requester = await ts.client(store_name, role=ts.RankRole.REQUESTER, group=0)
        assert _rank_of(requester) == "requester/0/1"
    finally:
        await store_api._shutdown_selected(store_name)


@pytest.mark.usefixtures("_in_process")
@pytest.mark.asyncio
async def test_a_cached_client_refuses_another_rank() -> None:
    """Return the cached client for its own rank or none, and refuse any other rank ID.

    Only another publisher rank, where this process serves a publisher volume of the store, needs
    ``ts.shutdown``. Another store's volume here leaves the remedy at ``ts.reset_client``.
    """
    store_name, other_store = _store_name(), _store_name()
    shutdown, reset = (
        "Call ts.shutdown to release it",
        "Call ts.reset_client to replace it",
    )
    publisher_0 = {"role": ts.RankRole.PUBLISHER, "rank": 0}
    publisher_1 = {"role": ts.RankRole.PUBLISHER, "rank": 1}
    requester_0 = {"role": ts.RankRole.REQUESTER, "group": 0, "rank": 0}
    async with _cleanups() as cleanups:
        cleanups.push_async_callback(store_api._shutdown_selected, other_store)
        cleanups.push_async_callback(store_api._shutdown_selected, store_name)
        await _service_store(store_name)
        await _service_store(other_store)
        await ts.client(other_store, role=ts.RankRole.PUBLISHER, rank=1)
        client = await ts.client(store_name, **publisher_0)
        assert await ts.client(store_name) is client
        assert await ts.client(store_name, **publisher_0) is client
        # The rank's role and group are checked before the cached client's rank ID.
        with pytest.raises(ValueError, match="^publishers are a single mesh and take"):
            await ts.client(store_name, **publisher_0, group=0)
        others: list[tuple[dict[str, Any], str, str]] = [
            (publisher_1, "publisher/1", shutdown),
            (requester_0, "requester/0/0", reset),
        ]
        for kwargs, wanted, remedy in others:
            refused = f"is publisher/0, not {wanted}. {remedy}$"
            with pytest.raises(ValueError, match=refused):
                await ts.client(store_name, **kwargs)
        assert await ts.client(store_name) is client
        # Its volume outlives reset_client, so a requester client here takes it into account.
        ts.reset_client(store_name)
        requester = await ts.client(store_name, **requester_0)
        others = [
            (
                {"role": ts.RankRole.REQUESTER, "group": 1, "rank": 0},
                "requester/1/0",
                reset,
            ),
            (
                {"role": ts.RankRole.REQUESTER, "group": 0, "rank": 1},
                "requester/0/1",
                reset,
            ),
            (publisher_1, "publisher/1", shutdown),
            (publisher_0, "publisher/0", reset),
        ]
        for kwargs, wanted, remedy in others:
            refused = f"is requester/0/0, not {wanted}. {remedy}$"
            with pytest.raises(ValueError, match=refused):
                await ts.client(store_name, **kwargs)
        assert await ts.client(store_name, **requester_0) is requester
        assert sorted(carrier_volumes._publishers) == sorted(
            [(store_name, "publisher/0"), (other_store, "publisher/1")]
        )


@pytest.mark.usefixtures("_in_process")
@pytest.mark.asyncio
async def test_a_build_refuses_another_rank_while_one_is_in_flight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refuse a racing build for another rank before it starts anything, so one volume serves at most."""
    store_name = _store_name()
    release = asyncio.Event()
    try:
        await _service_store(store_name)
        entered, release = _held_strategy(monkeypatch)
        building = asyncio.ensure_future(
            ts.client(store_name, role=ts.RankRole.PUBLISHER, rank=0)
        )
        await asyncio.wait_for(entered.wait(), 30)
        racing = "builds store .* as publisher/0, not publisher/1$"
        with pytest.raises(ValueError, match=racing):
            # Bounded, as a call let through would wait on the held read.
            await asyncio.wait_for(
                ts.client(store_name, role=ts.RankRole.PUBLISHER, rank=1), 10
            )
        release.set()
        assert _rank_of(await asyncio.wait_for(building, 30)) == "publisher/0"
        assert list(carrier_volumes._publishers) == [(store_name, "publisher/0")]
        assert store_api._building_ranks == {}
    finally:
        release.set()
        await store_api._shutdown_selected(store_name)


@pytest.mark.parametrize("via", ["record", "address"])
@pytest.mark.parametrize("held", ["rank", "actor"])
@pytest.mark.usefixtures("_in_process")
@pytest.mark.asyncio
async def test_racing_first_clients_for_different_ranks_never_share_a_client(
    monkeypatch: pytest.MonkeyPatch, held: str, via: str
) -> None:
    """Refuse the later of two racing first clients for different rank IDs, one taking the actor's rank.

    A build without ``rank`` claims the actor's rank once its plane is known and before it reads the strategy.
    One refused then closes the connection it dialed, where a call given ``rank`` is refused before it dials.
    """
    store_name = _store_name()
    release = asyncio.Event()
    async with _cleanups() as cleanups:
        requester: dict[str, Any] = {"role": ts.RankRole.REQUESTER, "group": 0}
        if via == "record":
            await _service_store(store_name)
        else:
            requester["address"] = await _other_service(cleanups, store_name)
        cleanups.push_async_callback(store_api._shutdown_selected, store_name)
        cleanups.callback(lambda: release.set())
        monkeypatch.setattr(store_api, "_actor_rank", lambda: 0)
        closed = _counted_closes(monkeypatch)
        entered, release = _held_strategy(monkeypatch)
        given: dict[str, Any] = {"rank": 1} if held == "rank" else {}
        taken: dict[str, Any] = {} if held == "rank" else {"rank": 1}
        building = asyncio.ensure_future(ts.client(store_name, **requester, **given))
        await asyncio.wait_for(entered.wait(), 30)
        claimed = "requester/0/1" if held == "rank" else "requester/0/0"
        refused = "requester/0/0" if held == "rank" else "requester/0/1"
        with pytest.raises(
            ValueError, match=f"builds store .* as {claimed}, not {refused}$"
        ):
            # Bounded since a call let through would wait on the held read.
            await asyncio.wait_for(ts.client(store_name, **requester, **taken), 10)
        assert len(closed) == (1 if via == "address" and held == "rank" else 0)
        release.set()
        client = await asyncio.wait_for(building, 30)
        assert _rank_of(client) == claimed
        assert await ts.client(store_name) is client
        assert store_api._building_ranks == {}


@pytest.mark.usefixtures("_in_process")
@pytest.mark.asyncio
async def test_racing_first_clients_of_the_actors_rank_through_one_address_share_a_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Share the client of a racing first call through one address that the other cached for the same rank ID.

    The later call takes the actor's rank once the earlier one's client is cached and returns that client.
    """
    store_name = _store_name()
    connect = ServiceAddress.connect
    released = asyncio.Event()
    dials = 0

    async def second_held(self: ServiceAddress) -> ControlPlane:
        nonlocal dials
        dials += 1
        if dials == 2:
            await released.wait()
        return await connect(self)

    async with _cleanups() as cleanups:
        address = await _other_service(cleanups, store_name)
        cleanups.push_async_callback(store_api._shutdown_selected, store_name)
        cleanups.callback(released.set)
        monkeypatch.setattr(store_api, "_actor_rank", lambda: 0)
        monkeypatch.setattr(ServiceAddress, "connect", second_held)
        requester = {"role": ts.RankRole.REQUESTER, "group": 0, "address": address}
        building = asyncio.ensure_future(ts.client(store_name, **requester))
        racing = asyncio.ensure_future(ts.client(store_name, **requester))
        first = await asyncio.wait_for(building, 30)
        assert _rank_of(first) == "requester/0/0" and dials == 2
        released.set()
        assert await asyncio.wait_for(racing, 30) is first
        assert store_api._building_ranks == {}


@pytest.mark.usefixtures("_in_process")
@pytest.mark.asyncio
async def test_a_client_taking_the_actors_rank_never_leaves_a_racing_publisher_serving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refuse a requester taking the actor's rank during a publisher's build before the requester caches a client.

    So the publisher's build caches its own client and no refused build leaves a volume serving.
    """
    store_name = _store_name()
    release = asyncio.Event()
    try:
        await _service_store(store_name)
        monkeypatch.setattr(store_api, "_actor_rank", lambda: 0)
        entered, release = _held_strategy(monkeypatch)
        publishing = asyncio.ensure_future(
            ts.client(store_name, role=ts.RankRole.PUBLISHER, rank=0)
        )
        await asyncio.wait_for(entered.wait(), 30)
        racing = "builds store .* as publisher/0, not requester/0/0$"
        with pytest.raises(ValueError, match=racing):
            await asyncio.wait_for(
                ts.client(store_name, role=ts.RankRole.REQUESTER, group=0), 10
            )
        release.set()
        publisher = await asyncio.wait_for(publishing, 30)
        assert _rank_of(publisher) == "publisher/0"
        assert store_api._local_clent_map[store_name] is publisher
        assert list(carrier_volumes._publishers) == [(store_name, "publisher/0")]
        assert store_api._building_ranks == {}
    finally:
        release.set()
        await store_api._shutdown_selected(store_name)


@pytest.mark.usefixtures("_in_process")
@pytest.mark.asyncio
async def test_a_build_taking_the_actors_rank_refuses_a_client_of_another_rank_cached_meanwhile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refuse a build that takes the actor's rank once a client of another rank ID was cached as it resolved that rank.

    It refuses as it claims the rank and so never returns the other rank's client.
    """
    store_name = _store_name()
    resolving, go = threading.Event(), threading.Event()

    def held_rank() -> int:
        resolving.set()
        assert go.wait(30)
        return 0

    try:
        await _service_store(store_name)
        monkeypatch.setattr(store_api, "_actor_rank", held_rank)
        call = ts.client(store_name, role=ts.RankRole.REQUESTER, group=0)
        requesting = asyncio.ensure_future(asyncio.to_thread(asyncio.run, call))
        assert await asyncio.to_thread(resolving.wait, 30)
        publisher = await ts.client(store_name, role=ts.RankRole.PUBLISHER, rank=0)
        go.set()
        racing = "built store .* as publisher/0, not requester/0/0$"
        with pytest.raises(ValueError, match=racing):
            await asyncio.wait_for(requesting, 30)
        assert await ts.client(store_name) is publisher
        assert store_api._building_ranks == {}
    finally:
        go.set()
        await store_api._shutdown_selected(store_name)


@pytest.mark.usefixtures("_in_process")
@pytest.mark.asyncio
async def test_a_cached_client_refuses_another_rank_through_its_address_or_plane() -> None:
    """Refuse another rank ID also when the call passes the store's address or plane."""
    store_name = _store_name()
    try:
        address = await _service_store(store_name)
        client = await ts.client(store_name, role=ts.RankRole.PUBLISHER, rank=0)
        plane = store_api._control_plane_map[store_name].control_plane
        selections: list[dict[str, Any]] = [
            {"address": address},
            {"control_plane": plane},
        ]
        for selection in selections:
            with pytest.raises(
                ValueError, match="is publisher/0, not publisher/1. Call ts.shutdown"
            ):
                await ts.client(
                    store_name, role=ts.RankRole.PUBLISHER, rank=1, **selection
                )
            same = await ts.client(
                store_name, role=ts.RankRole.PUBLISHER, rank=0, **selection
            )
            assert same is client
    finally:
        await store_api._shutdown_selected(store_name)


def test_an_actors_rank_comes_from_the_monarch_backend_or_rank_is_needed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Take the rank the Monarch backend reads from the actor's context. Anywhere else, ask for ``rank``.

    Without torchmonarch the call asks too, and any other missing module still raises.
    """
    ranks: list[int | None] = [3]
    backend = SimpleNamespace(actor_rank=lambda: ranks[0])
    monkeypatch.setitem(backend_module._backends, "monarch", cast(Backend, backend))
    assert store_api._actor_rank() == 3
    ranks[0] = 0
    assert store_api._actor_rank() == 0
    ranks[0] = None
    with pytest.raises(
        ValueError, match="^pass rank=: this call has no rank of its own"
    ):
        store_api._actor_rank()

    def missing(name: str | None) -> Callable[[str], NoReturn]:
        def get_backend(_: str) -> NoReturn:
            raise ModuleNotFoundError(f"no module {name}", name=name)

        return get_backend

    for name in ("monarch", "monarch.actor"):
        monkeypatch.setattr(store_api, "get_backend", missing(name))
        with pytest.raises(
            ValueError, match="^pass rank=: this call has no rank of its own"
        ):
            store_api._actor_rank()
    for name in ("cloudpickle", "monarchy", None):
        monkeypatch.setattr(store_api, "get_backend", missing(name))
        with pytest.raises(ModuleNotFoundError, match=f"^no module {name}$"):
            store_api._actor_rank()


@pytest.mark.usefixtures("_in_process")
@pytest.mark.asyncio
async def test_a_refused_build_gives_its_rank_back() -> None:
    """Free a refused build's rank, so the process may build the store as another rank."""
    store_name = _store_name()
    try:
        await _service_store(store_name)
        with pytest.raises(KeyError, match="'publisher/2' is not a publisher rank"):
            await ts.client(store_name, role=ts.RankRole.PUBLISHER, rank=2)
        client = await ts.client(store_name, role=ts.RankRole.PUBLISHER, rank=1)
        assert _rank_of(client) == "publisher/1"
        assert store_api._building_ranks == {}
    finally:
        await store_api._shutdown_selected(store_name)


@pytest.mark.usefixtures("_in_process")
@pytest.mark.asyncio
async def test_a_cancelled_build_gives_back_only_its_own_claim_on_a_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep refusing another rank while a second build for the cancelled one's rank runs.

    Once both are cancelled, the process may build the store as another rank.
    """
    store_name = _store_name()
    release = asyncio.Event()
    strategy = ServiceControlPlane.strategy
    try:
        await _service_store(store_name)
        entered, release = _held_strategy(monkeypatch)
        kwargs: dict[str, Any] = {"role": ts.RankRole.PUBLISHER, "rank": 0}
        first = asyncio.ensure_future(ts.client(store_name, **kwargs))
        await asyncio.wait_for(entered.wait(), 30)
        second = asyncio.ensure_future(ts.client(store_name, **kwargs))
        await _until(
            lambda: store_api._building_ranks.get(store_name) == ["publisher/0"] * 2
        )
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        racing = "builds store .* as publisher/0, not publisher/1$"
        with pytest.raises(ValueError, match=racing):
            # Bounded, as a call let through would wait on the held read.
            await asyncio.wait_for(
                ts.client(store_name, role=ts.RankRole.PUBLISHER, rank=1), 10
            )
        second.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second
        assert store_api._building_ranks == {}
        monkeypatch.setattr(ServiceControlPlane, "strategy", strategy)
        client = await ts.client(store_name, role=ts.RankRole.PUBLISHER, rank=1)
        assert _rank_of(client) == "publisher/1"
    finally:
        release.set()
        await store_api._shutdown_selected(store_name)


@pytest.mark.usefixtures("_in_process")
@pytest.mark.asyncio
async def test_a_callers_plane_with_records_takes_more_publishers_than_the_service() -> None:
    """Leave the publisher count of a caller's plane uncapped: the cap is the service's."""
    counts: list[int] = []

    class Counting(_Records):
        async def init(self, publishers: Collection[str], strategy: Any) -> None:
            counts.append(len(publishers))

    store_name = _store_name()
    over = store_api._MAX_SERVICE_PUBLISHERS + 1
    try:
        address = await ts.initialize(
            num_publishers=over,
            strategy=_strategy(),
            client_type=ts.ClientType.ROUTING,
            store_name=store_name,
            control_plane=Counting(cast(ServiceControlPlane, None)),
        )
        assert address is None and counts == [over]
    finally:
        await store_api._shutdown_selected(store_name)


@pytest.mark.parametrize("restarted", [False, True], ids=["same-store", "restarted"])
@pytest.mark.usefixtures("_in_process")
@pytest.mark.asyncio
async def test_a_publisher_takes_another_rank_of_its_store_only_after_shutdown(
    restarted: bool,
) -> None:
    """Refuse a publisher rank other than the one whose volume this process serves until ts.shutdown.

    That volume outlives ts.reset_client and counts by the store's name. A store started over at
    another address is refused too, and after ts.shutdown the process takes the other rank.
    """
    store_name = _store_name()
    async with _cleanups() as cleanups:
        first = await _other_service(cleanups, store_name)
        second = await _other_service(cleanups, store_name) if restarted else first
        cleanups.push_async_callback(store_api._shutdown_selected, store_name)
        publisher_0 = {"role": ts.RankRole.PUBLISHER, "rank": 0, "address": first}
        publisher_1 = {"role": ts.RankRole.PUBLISHER, "rank": 1, "address": second}
        client = await ts.client(store_name, **publisher_0)
        if not restarted:
            with pytest.raises(ValueError, match="Call ts.shutdown to release it$"):
                await ts.client(store_name, **publisher_1)
        ts.reset_client(store_name)
        serving = (
            "serves volume publisher/0 of routing store .* cannot serve publisher/1 too."
            " Call ts.shutdown first$"
        )
        with pytest.raises(ValueError, match=serving):
            await ts.client(store_name, **publisher_1)
        assert list(carrier_volumes._publishers) == [(store_name, "publisher/0")]
        if not restarted:
            again = await ts.client(store_name, **publisher_0)
            assert _rank_of(again) == "publisher/0" and again is not client
        await ts.shutdown(store_name)
        assert _rank_of(await ts.client(store_name, **publisher_1)) == "publisher/1"
        assert list(carrier_volumes._publishers) == [(store_name, "publisher/1")]


@pytest.mark.usefixtures("_in_process")
def test_another_publisher_volume_counts_by_store_name_unless_a_stop_took_it() -> None:
    """Count a live volume of the store under any selection but not one that a stop took."""
    store_name = _store_name()
    volume = carrier_volumes.PublisherVolume(
        "publisher/0", store_name, (object(), None)
    )
    carrier_volumes._publishers[(store_name, "publisher/0")] = volume
    serves = carrier_volumes.serves_another_publisher
    assert serves(store_name, "publisher/1")
    assert not serves(store_name, "publisher/0") and not serves(_store_name(), "")
    # A start the GC closed in a locked step leaves it so until its close thread takes it out.
    volume._stopped = True
    assert not serves(store_name, "publisher/1")


@pytest.mark.usefixtures("_in_process")
@pytest.mark.asyncio
async def test_a_build_with_rank_racing_one_without_still_serves_one_volume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refuse a build that takes the actor's rank while one with another ``rank`` is in flight.

    It claims the actor's rank before it starts anything. So the volume of the build in flight serves alone.
    """
    store_name = _store_name()
    release = asyncio.Event()
    try:
        await _service_store(store_name)
        monkeypatch.setattr(store_api, "_actor_rank", lambda: 1)
        entered, release = _held_strategy(monkeypatch)
        held = asyncio.ensure_future(
            ts.client(store_name, role=ts.RankRole.PUBLISHER, rank=0)
        )
        await asyncio.wait_for(entered.wait(), 30)
        racing = "builds store .* as publisher/0, not publisher/1$"
        with pytest.raises(ValueError, match=racing):
            await asyncio.wait_for(
                ts.client(store_name, role=ts.RankRole.PUBLISHER), 10
            )
        assert carrier_volumes._publishers == {}
        release.set()
        assert _rank_of(await asyncio.wait_for(held, 30)) == "publisher/0"
        assert list(carrier_volumes._publishers) == [(store_name, "publisher/0")]
        assert store_api._building_ranks == {}
    finally:
        release.set()
        await store_api._shutdown_selected(store_name)


@pytest.mark.usefixtures("_in_process")
@pytest.mark.asyncio
async def test_a_volume_start_refuses_another_rank_while_the_first_registers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refuse a build with Monarch's rank 1 before it starts anything while publisher 0's volume still registers."""
    store_name = _store_name()
    entered, release = asyncio.Event(), asyncio.Event()
    register = ServiceControlPlane.register_volume

    async def held(self: ServiceControlPlane, volume_id: str, record: bytes) -> None:
        if volume_id == "publisher/0":
            entered.set()
            await release.wait()
        await register(self, volume_id, record)

    try:
        await _service_store(store_name)
        monkeypatch.setattr(ServiceControlPlane, "register_volume", held)
        monkeypatch.setattr(store_api, "_actor_rank", lambda: 1)
        first = asyncio.ensure_future(
            ts.client(store_name, role=ts.RankRole.PUBLISHER, rank=0)
        )
        await asyncio.wait_for(entered.wait(), 30)
        racing = "builds store .* as publisher/0, not publisher/1$"
        with pytest.raises(ValueError, match=racing):
            await asyncio.wait_for(
                ts.client(store_name, role=ts.RankRole.PUBLISHER), 30
            )
        assert list(carrier_volumes._publishers) == [(store_name, "publisher/0")]
        release.set()
        assert _rank_of(await asyncio.wait_for(first, 30)) == "publisher/0"
    finally:
        release.set()
        await store_api._shutdown_selected(store_name)


class _RefusedThread:
    """A thread whose start is refused, as at the OS's thread limit or at interpreter shutdown."""

    def __init__(self, **kwargs: Any) -> None:
        pass

    def start(self) -> None:
        raise RuntimeError("can't start new thread")


@pytest.mark.parametrize("lock", ["free", "held", "finalizing", "refused"])
def test_a_closed_builds_cleanup_never_waits_for_the_selection_lock(
    monkeypatch: pytest.MonkeyPatch, lock: str
) -> None:
    """Run a GC-closed build's update here if the selection lock is free, else on a thread once it frees.

    A finalizing interpreter starts no such thread, and a refused start skips the update as well.
    """
    ran: list[tuple[bool, int]] = []

    def update() -> None:
        ran.append((store_api._selection_lock.locked(), threading.get_ident()))

    if lock == "free":
        store_api._locked_cleanup(update, True)
        assert ran == [(True, threading.get_ident())]
        assert not store_api._selection_lock.locked()
        return
    if lock == "finalizing":
        finalizing = SimpleNamespace(is_finalizing=lambda: True)
        monkeypatch.setattr(store_utils, "sys", finalizing)
    elif lock == "refused":
        monkeypatch.setattr(
            store_utils, "threading", SimpleNamespace(Thread=_RefusedThread)
        )
    with store_api._selection_lock:
        store_api._locked_cleanup(update, True)
        started = [
            thread
            for thread in threading.enumerate()
            if thread.name == "torchstore-selection-update"
        ]
        assert ran == [] and len(started) == (1 if lock == "held" else 0)
    for thread in started:
        thread.join(30)
        assert not thread.is_alive()
    assert ran == [(True, thread.ident) for thread in started]
    assert threading.get_ident() not in [ident for _, ident in ran]
