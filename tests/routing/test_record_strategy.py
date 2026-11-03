# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Routed syncs through strategies built from volume records, without Monarch."""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from collections.abc import Awaitable, Callable, Collection, Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest
import torch
from torch.distributed.tensor import DTensor
from torchstore.carrier_volumes import CarrierVolumeHost
from torchstore.routing._model import RankRole
from torchstore.routing.client import RoutingClient
from torchstore.routing.tcpstore import TCPStoreRendezvous
from torchstore.state_dict_utils import get_state_dict, put_state_dict
from torchstore.storage import StorageVolumeCore
from torchstore.strategy import TorchStoreStrategy
from torchstore.transport import gloo, TransportType

from ..test_carrier_volume_host import _CARRIED, _finally, _Records
from ..test_carrier_volumes import _Volumes
from . import routed_sync_data as data
from .file_store_control_plane import FileStoreControlPlane

pytestmark = [
    pytest.mark.timeout(120),
    pytest.mark.filterwarnings("error::pytest.PytestUnhandledThreadExceptionWarning"),
]

_PUBLISHERS = [f"publisher/{rank}" for rank in range(data.PUBLISHER_RANKS)]
# Every rank of the requester meshes, as (group, rank).
_REQUESTER_RANKS = [
    (group, rank)
    for group, (num_ranks, _) in enumerate(data.REQUESTERS)
    for rank in range(num_ranks)
]
_Resolver = Callable[[Collection[str]], Awaitable[Mapping[str, bytes]]]


@pytest.fixture
def volumes() -> Iterator[_Volumes]:
    volumes = _Volumes()
    yield volumes
    with contextlib.ExitStack() as steps:
        # Gloo caches each client's pair to a volume ID for the whole process.
        steps.callback(gloo._forget_pairs, _PUBLISHERS)
        steps.callback(volumes.close)


async def _serve_publishers(volumes: _Volumes) -> dict[str, bytes]:
    """Serve each publisher's volume here (as its trainer process would) and return their records."""
    records = {}
    for volume_id in _PUBLISHERS:
        _, _, records[volume_id] = await volumes.serve(StorageVolumeCore(volume_id))
    return records


async def _push(publisher: RoutingClient, rank: int, round_: int) -> None:
    state_dict = data.place(
        data.weights(round_), data.PUBLISHER_DIMS, rank, data.PUBLISHER_RANKS
    )
    # A worker thread with a fresh event loop per push, as in TitanRL's trainer.
    await asyncio.to_thread(
        lambda: asyncio.run(put_state_dict(publisher, state_dict, data.KEY))
    )


class _Requester:
    """A requester rank's client and the buffers it reads into every round.

    A strict requester's buffers hold exactly the published keys, as a strict get needs.
    """

    def __init__(
        self, group: int, rank: int, client: RoutingClient, *, strict: bool = False
    ) -> None:
        self.group, self.rank, self.client, self.strict = group, rank, client, strict
        num_ranks, dims = data.REQUESTERS[group]
        zeros = {name: torch.zeros_like(t) for name, t in data.weights(0).items()}
        self.destination = data.place(zeros, dims, rank, num_ranks)
        if not strict:
            self.destination[data.DESTINATION_ONLY] = data.sentinel()

    async def get(self) -> dict[str, Any]:
        return await get_state_dict(
            self.client, data.KEY, self.destination, strict=self.strict
        )

    def check(self, round_: int, result: dict[str, Any]) -> None:
        """Check, bitwise, that the round's values landed in this rank's own buffers."""
        num_ranks, dims = data.REQUESTERS[self.group]
        expected = data.weights(round_)
        label = f"round {round_}, requester/{self.group}/{self.rank}"
        assert sorted(result) == sorted(expected), label
        for name, tensor in expected.items():
            assert result[name] is self.destination[name], label
            held = result[name]
            held = held.to_local() if isinstance(held, DTensor) else held
            data.assert_bitwise_equal(
                held,
                data.local(tensor, dims[name], self.rank, num_ranks),
                f"{label}, {name}",
            )
        if not self.strict:
            sentinel = self.destination[data.DESTINATION_ONLY]
            assert torch.equal(sentinel, data.sentinel()), label


class _Sync:
    """Routing clients whose strategies hold carrier hosts, as ranks build them from records.

    Every client opens the FileStore control plane by its path, as a rank in its own
    process would. ``close`` closes the hosts, the planes and the transport caches.
    """

    def __init__(self, path: Path, transport_type: TransportType) -> None:
        self._path = str(path)
        self._transport_type = transport_type
        self._planes: list[FileStoreControlPlane] = []
        self._hosts: list[CarrierVolumeHost] = []
        self._strategies: list[TorchStoreStrategy] = []
        self.publishers: list[RoutingClient] = []
        self.requesters: list[_Requester] = []

    def _plane(self) -> FileStoreControlPlane:
        self._planes.append(FileStoreControlPlane(self._path))
        return self._planes[-1]

    async def init(self) -> None:
        strategy = TorchStoreStrategy(self._transport_type)
        await self._plane().init(_PUBLISHERS, strategy)

    def host(self, resolver: _Resolver) -> CarrierVolumeHost:
        """A carrier host over ``resolver`` that ``close`` closes."""
        self._hosts.append(CarrierVolumeHost(resolver))
        return self._hosts[-1]

    def strategy(
        self, resolver: _Resolver, transport_type: TransportType | None = None
    ) -> tuple[TorchStoreStrategy, CarrierVolumeHost]:
        """A bare strategy, as the store holds it, with a carrier host over ``resolver``."""
        strategy = TorchStoreStrategy(
            self._transport_type if transport_type is None else transport_type
        )
        host = self.host(resolver)
        strategy.set_volume_host(host)
        self._strategies.append(strategy)
        return strategy, host

    def add_publisher(self, volume_id: str, strategy: TorchStoreStrategy) -> None:
        self.publishers.append(
            RoutingClient(volume_id, RankRole.PUBLISHER, self._plane(), strategy)
        )

    def add_publishers(self, records: Mapping[str, bytes]) -> list[_Records]:
        """Add each publisher with its own strategy and record, then return their resolvers."""
        resolvers = []
        for volume_id in _PUBLISHERS:
            resolvers.append(_Records(records))
            strategy, host = self.strategy(resolvers[-1])
            host.adopt(volume_id, records[volume_id])
            self.add_publisher(volume_id, strategy)
        return resolvers

    def add_requester(
        self,
        group: int,
        rank: int,
        strategy: TorchStoreStrategy,
        *,
        strict: bool = False,
    ) -> None:
        name = f"requester/{group}/{rank}"
        client = RoutingClient(name, RankRole.REQUESTER, self._plane(), strategy)
        self.requesters.append(_Requester(group, rank, client, strict=strict))

    async def push(self, round_: int) -> None:
        await asyncio.gather(
            *(_push(p, rank, round_) for rank, p in enumerate(self.publishers))
        )

    async def pull(self, round_: int) -> None:
        gets = [requester.get() for requester in self.requesters]
        for requester, result in zip(self.requesters, await asyncio.gather(*gets)):
            requester.check(round_, result)

    async def close(self) -> None:
        """Close the hosts, then the planes, then the transport caches even if a step raises."""
        async with contextlib.AsyncExitStack() as steps:
            for strategy in self._strategies:
                steps.callback(strategy.transport_context.clear)
            for plane in self._planes:
                steps.push_async_callback(plane.close)
            for host in self._hosts:
                steps.push_async_callback(host.close)


@pytest.mark.parametrize("transport_type", _CARRIED)
@pytest.mark.asyncio
async def test_strategies_built_from_records_sync_state_dicts_bitwise(
    volumes: _Volumes, tmp_path: Path, transport_type: TransportType
) -> None:
    """Sync two served publishers into two differently sharded requester meshes for 3 rounds.

    Outside the Gloo arm each requester reads both records once as its first layouts return. Gloo clients share
    one strategy whose host holds both records because Gloo refuses a second live client of a volume's pair in one
    process. Each round gathers its pushes and then its pulls.
    """
    records = await _serve_publishers(volumes)
    shared = transport_type is TransportType.Gloo
    sync = _Sync(tmp_path / "control_plane", transport_type)
    async with _finally(sync.close):
        await sync.init()
        if shared:
            reads = _Records(records)
            strategy, host = sync.strategy(reads)
            for volume_id in _PUBLISHERS:
                host.adopt(volume_id, records[volume_id])
                sync.add_publisher(volume_id, strategy)
            for group, rank in _REQUESTER_RANKS:
                sync.add_requester(group, rank, strategy)
            publisher_reads, requester_reads = [], [reads]
        else:
            publisher_reads = sync.add_publishers(records)
            requester_reads = [_Records(records) for _ in _REQUESTER_RANKS]
            for (group, rank), reads in zip(_REQUESTER_RANKS, requester_reads):
                sync.add_requester(group, rank, sync.strategy(reads)[0])

        for round_ in range(data.ROUNDS):
            await sync.push(round_)
            await sync.pull(round_)
            # The process's one Gloo host holds both publishers' records so reads none.
            expected = [[]] if shared else [[_PUBLISHERS]] * len(_REQUESTER_RANKS)
            assert [reads.calls for reads in requester_reads] == expected
            assert [reads.calls for reads in publisher_reads] == [[]] * len(
                publisher_reads
            )
        for requester in sync.requesters:
            for volume_id in _PUBLISHERS:
                ref = requester.client.strategy.get_storage_volume(volume_id)
                assert ref.default_transport_type is transport_type


@pytest.mark.parametrize(
    ("error_type", "message"),
    [
        (
            ConnectionError,
            "volume_records on TCPStore 127.0.0.1:29500 failed: Connection reset by peer",
        ),
        (KeyError, "no volume record for ['publisher/1']"),
    ],
    ids=["connection", "missing"],
)
@pytest.mark.parametrize("strict", [False, True], ids=["not_strict", "strict"])
@pytest.mark.asyncio
async def test_a_failed_record_read_fails_the_get_unwrapped_and_the_next_get_reads_again(
    volumes: _Volumes,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[Exception],
    message: str,
    strict: bool,
) -> None:
    """Raise a failed record read from the first get itself, not the mapping's wrapper.

    The get caches no publisher layouts, installs no plan, and registers no layouts. The next one then fetches the
    layouts, reads again, and completes. Strict gets, whose buffers hold exactly the published keys, do the same.
    """
    records = await _serve_publishers(volumes)
    reads = _Records(records)
    failure = error_type(message)
    failed: list[list[str]] = []

    async def fail_once(volume_ids: Collection[str]) -> Mapping[str, bytes]:
        if not failed:
            failed.append(list(volume_ids))
            raise failure
        return await reads(volume_ids)

    sync = _Sync(tmp_path / "control_plane", TransportType.MonarchRPC)
    async with _finally(sync.close):
        await sync.init()
        sync.add_publishers(records)
        sync.add_requester(0, 0, sync.strategy(fail_once)[0], strict=strict)
        [requester] = sync.requesters
        plane = sync._planes[-1]
        fetch = plane.get_layouts
        fetched: list[str] = []

        async def counted_fetch(key: str) -> Any:
            fetched.append(key)
            return await fetch(key)

        monkeypatch.setattr(plane, "get_layouts", counted_fetch)
        await sync.push(0)
        with pytest.raises(error_type) as raised:
            await requester.get()
        assert raised.value is failure
        assert data.KEY not in requester.client._registered_layouts
        with pytest.raises(RuntimeError, match="has no routes"):
            _ = requester.client._routing_directory.routes
        requester.check(0, await requester.get())
        assert (failed, reads.calls) == ([_PUBLISHERS], [_PUBLISHERS])
        assert fetched == [data.KEY, data.KEY]


@pytest.mark.parametrize("failure", ["refused_transport", "closed_host"])
@pytest.mark.asyncio
async def test_a_refused_transport_or_a_closed_host_fails_a_keys_first_get_unwrapped(
    tmp_path: Path, failure: str
) -> None:
    """Fail a key's first get with an unwrapped error before it reads any record or mapping.

    A strategy naming a transport outside its host's raises ValueError, and a closed host RuntimeError. The publishers
    register empty layouts and serve nothing since both come before any volume is reached.
    """
    refused = failure == "refused_transport"
    sync = _Sync(tmp_path / "control_plane", TransportType.MonarchRPC)
    async with _finally(sync.close):
        await sync.init()
        for volume_id in _PUBLISHERS:
            await sync._plane().register_layout(volume_id, data.KEY, {})
        reads = _Records({})
        strategy, host = sync.strategy(
            reads, TransportType.NIXL if refused else TransportType.MonarchRPC
        )
        sync.add_requester(0, 0, strategy)
        [requester] = sync.requesters
        if refused:
            error_type, message = (
                ValueError,
                "^CarrierVolumeHost reaches its volumes only through Gloo or MonarchRPC"
                " or SharedMemory, not NIXL$",
            )
        else:
            await host.close()
            error_type, message = RuntimeError, "^this carrier volume host is closed$"
        with pytest.raises(error_type, match=message) as raised:
            await requester.get()
        assert raised.value.__cause__ is None
        assert reads.calls == []
        assert data.KEY not in requester.client._registered_layouts


@pytest.mark.parametrize("next_get", ["unplanned_key", "planned_keys", "no_state_dict"])
@pytest.mark.asyncio
async def test_a_host_closed_after_a_keys_first_get_fails_the_next_with_its_own_error(
    volumes: _Volumes, tmp_path: Path, next_get: str
) -> None:
    """Fail the next get at once with the closed host's own error, unwrapped, before its mapping read.

    ``get_state_dict`` would report an error of that read as a missing push. The next get passes a key no
    publisher holds, only keys planned already (a strict get), or no state dict.
    """
    records = await _serve_publishers(volumes)
    sync = _Sync(tmp_path / "control_plane", TransportType.MonarchRPC)
    async with _finally(sync.close):
        await sync.init()
        sync.add_publishers(records)
        reads = _Records(records)
        strategy, host = sync.strategy(reads)
        sync.add_requester(0, 0, strategy, strict=next_get == "planned_keys")
        [requester] = sync.requesters
        await sync.push(0)
        requester.check(0, await requester.get())
        await host.close()
        with pytest.raises(
            RuntimeError, match="^this carrier volume host is closed$"
        ) as raised:
            if next_get == "no_state_dict":
                await get_state_dict(requester.client, data.KEY)
            else:
                await requester.get()
        assert raised.value.__cause__ is None
        assert reads.calls == [_PUBLISHERS]


@pytest.mark.asyncio
async def test_a_host_installed_since_a_keys_first_get_loads_the_records_on_the_next(
    volumes: _Volumes, tmp_path: Path
) -> None:
    """Load every publisher's record into a host the strategy took since the key's first get, on the next get."""
    records = await _serve_publishers(volumes)
    sync = _Sync(tmp_path / "control_plane", TransportType.MonarchRPC)
    async with _finally(sync.close):
        await sync.init()
        sync.add_publishers(records)
        strategy, _ = sync.strategy(_Records(records))
        sync.add_requester(0, 0, strategy)
        [requester] = sync.requesters
        await sync.push(0)
        requester.check(0, await requester.get())
        reads = _Records(records)
        strategy.set_volume_host(sync.host(reads))
        await sync.push(1)
        requester.check(1, await requester.get())
        assert reads.calls == [_PUBLISHERS]


@pytest.mark.asyncio
async def test_records_registered_on_the_tcpstore_rendezvous_sync_bitwise(
    volumes: _Volumes, tmp_path: Path
) -> None:
    """Sync through records that publishers register on the TCPStore rendezvous.

    A get before a publisher registers fails with the store's KeyError naming it, unwrapped,
    and each requester's connection then reads every record once more.
    """
    records = await _serve_publishers(volumes)
    reads: dict[int, list[list[str]]] = {}

    def counted(rendezvous: TCPStoreRendezvous) -> _Resolver:
        async def volume_records(volume_ids: Collection[str]) -> Mapping[str, bytes]:
            reads.setdefault(id(rendezvous), []).append(list(volume_ids))
            return await rendezvous.volume_records(volume_ids)

        return volume_records

    sync = _Sync(tmp_path / "control_plane", TransportType.MonarchRPC)
    stack = contextlib.AsyncExitStack()
    async with _finally(stack.aclose):
        driver = await TCPStoreRendezvous.serve(
            f"record-strategy-{uuid.uuid4()}", host="127.0.0.1"
        )
        stack.push_async_callback(driver.close)

        async def connect() -> TCPStoreRendezvous:
            rendezvous = await TCPStoreRendezvous.connect(driver.address)
            stack.push_async_callback(rendezvous.close)
            return rendezvous

        stack.push_async_callback(sync.close)
        await driver.publish(_PUBLISHERS, TorchStoreStrategy(TransportType.MonarchRPC))
        await sync.init()
        stores = [await connect() for _ in _PUBLISHERS]
        await stores[0].register_volume(_PUBLISHERS[0], records[_PUBLISHERS[0]])
        for volume_id, store in zip(_PUBLISHERS, stores):
            strategy, host = sync.strategy(counted(store))
            host.adopt(volume_id, records[volume_id])
            sync.add_publisher(volume_id, strategy)
        requester_stores = [await connect() for _ in _REQUESTER_RANKS]
        for (group, rank), store in zip(_REQUESTER_RANKS, requester_stores):
            sync.add_requester(group, rank, sync.strategy(counted(store))[0])

        await sync.push(0)
        for error in await asyncio.gather(
            *(requester.get() for requester in sync.requesters), return_exceptions=True
        ):
            assert isinstance(error, KeyError), repr(error)
            assert f"on TCPStore {driver.address.host}:{driver.address.port}" in str(
                error
            )
            assert "no volume record for ['publisher/1']" in str(error)
        assert all(
            data.KEY not in requester.client._registered_layouts
            for requester in sync.requesters
        )
        await stores[1].register_volume(_PUBLISHERS[1], records[_PUBLISHERS[1]])
        await sync.pull(0)
        for round_ in range(1, data.ROUNDS):
            await sync.push(round_)
            await sync.pull(round_)
        assert [reads.get(id(store)) for store in stores] == [None, None]
        assert [reads[id(store)] for store in requester_stores] == [
            [_PUBLISHERS, _PUBLISHERS]
        ] * len(_REQUESTER_RANKS)
