# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import functools
import importlib
import os
import pickle
import random
import subprocess
import sys
from collections.abc import Callable, Iterator
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torchstore.client import LocalClient
from torchstore.controller import ObjectType, StorageInfo
from torchstore.directory import VolumeDirectory
from torchstore.storage_volume import InMemoryStore
from torchstore.strategy import StorageVolumeRef
from torchstore.transport import TransportType
from torchstore.transport.buffers import TransportContext
from torchstore.transport.monarch_rpc import MonarchRPCTransportBuffer
from torchstore.transport.shared_memory import SharedMemoryTransportBuffer
from torchstore.transport.types import Request, TensorSlice

from .routing.utils import tensor_slice
from .test_shared_memory_overwrite import _Endpoint

# The package's ``client`` function hides this module as ``torchstore.client``.
client_module = importlib.import_module("torchstore.client")

_TRANSPORTS = {
    TransportType.SharedMemory: SharedMemoryTransportBuffer,
    TransportType.MonarchRPC: MonarchRPCTransportBuffer,
}


def _offsets(region: TensorSlice | None) -> tuple | None:
    return None if region is None else tuple(region.offsets)


async def _answer_after(
    waits: dict[Any, Any],
    finished: set[Any],
    volume_id: Any,
    handler: Callable,
    *args: Any,
) -> Any:
    """Answers once the volume ``waits`` names for ``volume_id``, if any, is in ``finished``.

    ``finished`` holds the volumes whose fetch has returned. It answers after 1,000 loop turns anyway,
    so a key lock that this volume's fetch holds and the other fetch waits on cannot deadlock the get.
    """
    awaited = waits.get(volume_id)
    for _ in range(1000):
        if awaited is None or awaited in finished:
            break
        await asyncio.sleep(0)
    return await handler(*args)


@dataclasses.dataclass(frozen=True)
class _VolumeName:
    """A hashable volume id without an order."""

    name: str


class _Forward:
    """Forwards attribute reads to ``target`` except those given as keyword arguments.

    The target, not this wrapper, crosses the RPC boundary, so the overrides need not pickle.
    """

    def __init__(self, target: Any, **overrides: Any) -> None:
        self.__dict__.update(overrides, _target=target)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)


class _Store(VolumeDirectory):
    """In-process volumes that log the volume of each fetch as it starts.

    A volume filled by ``add`` answers in one step and copies into the caller's tensor when asked,
    one filled by ``serve`` answers through a real transport buffer, and one in ``failures`` raises.
    """

    def __init__(
        self, monkeypatch: pytest.MonkeyPatch, cleanup: contextlib.ExitStack
    ) -> None:
        self.locations: dict[str, dict[Any, StorageInfo]] = {}
        self.stored: dict[tuple[Any, str, tuple | None], torch.Tensor] = {}
        self.failures: dict[Any, Exception] = {}
        self.waits: dict[Any, Any] = {}
        self.finished: set[Any] = set()
        self.calls: list[Any] = []
        self._refs: dict[Any, StorageVolumeRef] = {}
        self._context = TransportContext()
        self._cleanup = cleanup
        cleanup.callback(self._context.clear)
        monkeypatch.setattr(client_module, "create_transport_buffer", self._buffer)

    def locate(
        self, volume_id: Any, key: str, region: TensorSlice | None = None
    ) -> None:
        """Records that ``volume_id`` holds ``key``, whole or as ``region``."""
        object_type = ObjectType.TENSOR if region is None else ObjectType.TENSOR_SLICE
        info = self.locations.setdefault(key, {}).setdefault(
            volume_id, StorageInfo(object_type)
        )
        if region is not None:
            info.tensor_slices.add(region)

    def add(
        self,
        volume_id: Any,
        key: str,
        value: torch.Tensor,
        region: TensorSlice | None = None,
    ) -> None:
        self.locate(volume_id, key, region)
        self.stored[volume_id, key, _offsets(region)] = value

    async def serve(
        self,
        transport_type: TransportType,
        volume_id: str,
        key: str,
        value: torch.Tensor,
        region: TensorSlice,
    ) -> None:
        """Puts ``value`` as ``region`` of ``key`` on an in-memory volume behind ``transport_type``.

        Its gets cross a pickling RPC boundary, and ``waits`` can hold their answers for another fetch.
        """
        if volume_id not in self._refs:
            volume = InMemoryStore()
            self._cleanup.callback(volume.reset)
            get = functools.partial(
                _answer_after, self.waits, self.finished, volume_id, volume.get
            )
            endpoints = SimpleNamespace(
                handshake=_Endpoint(volume.handshake),
                put=_Endpoint(volume.put),
                get=_Endpoint(get),
            )
            self._refs[volume_id] = StorageVolumeRef(
                endpoints, volume_id, self._context, transport_type
            )
        ref = self._refs[volume_id]
        assert ref.default_transport_type == transport_type, volume_id
        await _TRANSPORTS[transport_type](ref).put_to_storage_volume(
            [Request.from_any(key, value, region)]
        )
        self.locate(volume_id, key, region)

    async def locate_volumes(
        self,
        keys: list[str],
        missing_ok: bool = False,
        require_fully_committed: bool = True,
    ) -> dict[str, dict[Any, StorageInfo]]:
        return {key: self.locations[key] for key in keys}

    def client(self, offset: int | None = None) -> LocalClient:
        """A new client, with ``offset`` in place of its own when one is given."""
        strategy = SimpleNamespace(get_storage_volume=lambda volume_id: volume_id)
        client = LocalClient(self, strategy)
        if offset is not None:
            client._fetch_offset = offset
        return client

    def _buffer(self, volume_id: Any) -> Any:
        """A served volume's real transport buffer, or a stand-in, whose fetches log as they start.

        A fetch also raises the volume's failure if it has one and adds the volume to ``finished``.
        """
        ref = self._refs.get(volume_id)
        real = None if ref is None else _TRANSPORTS[ref.default_transport_type](ref)

        async def get_from_storage_volume(requests: list[Request]) -> list[Any]:
            self.calls.append(volume_id)
            if volume_id in self.failures:
                raise self.failures[volume_id]
            if real is not None:
                results = await real.get_from_storage_volume(requests)
            else:
                results = []
                for r in requests:
                    value = self.stored[volume_id, r.key, _offsets(r.tensor_slice)]
                    results.append(
                        value if r.tensor_val is None else r.tensor_val.copy_(value)
                    )
            self.finished.add(volume_id)
            return results

        if real is None:
            return SimpleNamespace(
                supports_inplace_resharding=True,
                get_from_storage_volume=get_from_storage_volume,
            )
        return _Forward(real, get_from_storage_volume=get_from_storage_volume)


def _one_key_per_volume(store: _Store, count: int) -> None:
    for index in range(count):
        store.add(
            f"volume{index:02d}", f"key{index:02d}", torch.full((2,), float(index))
        )


def _rng_states(np: Any) -> tuple:
    return (
        random.getstate(),
        pickle.dumps(np.random.get_state()),
        torch.get_rng_state().tolist(),
    )


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Store]:
    """A store whose served volumes and transport caches are reset after the test.

    Every reset runs even when another fails, and a failed reset shows as an error at teardown.
    """
    with contextlib.ExitStack() as cleanup:
        yield _Store(monkeypatch, cleanup)


@pytest.fixture
def launch(monkeypatch: pytest.MonkeyPatch) -> Callable[..., None]:
    """Sets the env a launcher gives this process and unsets the identity variables it omits."""

    def launch(**env: str) -> None:
        for name in (
            "TORCHSTORE_HOSTNAME",
            "RANK",
            "LOCAL_RANK",
            "CUDA_VISIBLE_DEVICES",
        ):
            monkeypatch.delenv(name, raising=False)
        for name, value in env.items():
            monkeypatch.setenv(name, value)

    launch()
    return launch


@pytest.fixture
def seed_alike() -> Iterator[Callable[[int], None]]:
    """Seeds random, numpy and torch's CPU generator with one value per call, as vLLM does.

    Each state is restored afterwards, even when restoring another one fails.
    """
    np = pytest.importorskip("numpy")
    with contextlib.ExitStack() as restore:
        restore.callback(random.setstate, random.getstate())
        restore.callback(np.random.set_state, np.random.get_state())
        restore.enter_context(torch.random.fork_rng(devices=[]))

        def seed(value: int) -> None:
            random.seed(value)
            np.random.seed(value)
            torch.default_generator.manual_seed(value)

        yield seed


@pytest.mark.asyncio
async def test_a_client_starts_every_get_at_its_offset_into_the_sorted_volumes(
    store: _Store,
) -> None:
    """Fetches go in sorted volume order from the offset, wrapping, whatever the key order.

    Each client gets twice, and both gets start at its offset.
    """
    _one_key_per_volume(store, 8)
    volumes = [f"volume{index:02d}" for index in range(8)]
    keys = sorted(store.locations)
    shuffle = random.Random(0).shuffle
    for offset in range(16):
        client = store.client(offset)
        for _ in range(2):
            shuffle(keys)
            store.calls.clear()

            results = await client.get_batch(keys)

            assert store.calls == volumes[offset % 8 :] + volumes[: offset % 8], offset
            for index, volume_id in enumerate(volumes):
                key = f"key{index:02d}"
                assert torch.equal(results[key], store.stored[volume_id, key, None])


@pytest.mark.parametrize(
    ("volume_ids", "order"),
    [
        pytest.param([10, 9], [9, 10], id="ints"),
        pytest.param([10, "1", 9, 1], ["1", 1, 10, 9], id="ints-and-strs"),
        pytest.param(
            [_VolumeName("b"), _VolumeName("a")],
            [_VolumeName("a"), _VolumeName("b")],
            id="no-order",
        ),
    ],
)
@pytest.mark.asyncio
async def test_volume_ids_of_any_type_get_one_order(
    store: _Store, volume_ids: list[Any], order: list[Any]
) -> None:
    """Ids that order keep sorted order, and others, such as a mix of int and str, sort by repr.

    Each client starts at its offset into that one order whichever way it lists the keys.
    """
    for index, volume_id in enumerate(volume_ids):
        store.add(volume_id, f"key{index}", torch.full((2,), float(index)))
    keys = sorted(store.locations)
    for offset in range(len(order)):
        for listed in (keys, keys[::-1]):
            store.calls.clear()

            results = await store.client(offset).get_batch(listed)

            assert store.calls == order[offset:] + order[:offset], (offset, listed)
            for index in range(len(volume_ids)):
                expected = torch.full((2,), float(index))
                assert torch.equal(results[f"key{index}"], expected)


@pytest.mark.parametrize(
    "envs",
    [
        pytest.param(
            [{"TORCHSTORE_HOSTNAME": "host", "RANK": str(r)} for r in range(256)],
            id="ranks-on-one-host",
        ),
        pytest.param(
            [{"TORCHSTORE_HOSTNAME": f"host{h}"} for h in range(256)],
            id="hosts-without-ranks",
        ),
    ],
)
@pytest.mark.asyncio
async def test_ranks_and_hosts_spread_their_starts(
    store: _Store, launch: Callable[..., None], envs: list[dict[str, str]]
) -> None:
    """256 ranks on one host, or 256 hosts without ranks, reach all 17 starts over 17 volumes.

    Offsets cut to 16 or fewer values cannot, and a uniform hash misses one with probability 3.1e-6.
    The ranks run under one fixed host name, so their outcome is the same on every machine.
    """
    _one_key_per_volume(store, 17)
    keys = sorted(store.locations)
    starts = set()
    for env in envs:
        launch(**env)
        client = store.client()
        store.calls.clear()

        await client.get_batch(keys)

        starts.add(store.calls[0])
    assert len(starts) == 17


@pytest.mark.parametrize(
    "envs",
    [
        pytest.param(
            [{"LOCAL_RANK": str(r)} for r in range(8)], id="local_rank-without-rank"
        ),
        pytest.param(
            [{"RANK": "0", "LOCAL_RANK": str(r)} for r in range(8)],
            id="local_rank-under-one-rank",
        ),
        pytest.param(
            [{"RANK": str(r), "LOCAL_RANK": "0"} for r in range(8)],
            id="rank-under-one-local_rank",
        ),
        pytest.param(
            [{"CUDA_VISIBLE_DEVICES": str(g)} for g in range(8)],
            id="cuda_visible_devices",
        ),
        pytest.param(
            [
                {"TORCHSTORE_HOSTNAME": "host1", "RANK": "0"},
                {"TORCHSTORE_HOSTNAME": "host", "RANK": "10"},
            ],
            id="host-name-ending-in-a-digit",
        ),
    ],
)
def test_processes_whose_identities_differ_get_their_own_offsets(
    launch: Callable[..., None], envs: list[dict[str, str]]
) -> None:
    """Processes that differ only in RANK, LOCAL_RANK or CUDA_VISIBLE_DEVICES get their own offsets.

    LOCAL_RANK counts with RANK unset or fixed and RANK with LOCAL_RANK fixed, and a host name that
    ends in a digit stays apart from the rank after it.
    """
    offsets = set()
    for env in envs:
        launch(**env)
        offsets.add(LocalClient(None, None)._fetch_offset)
    assert len(offsets) == len(envs)


@pytest.mark.parametrize(
    "name", ["TORCHSTORE_HOSTNAME", "RANK", "LOCAL_RANK", "CUDA_VISIBLE_DEVICES"]
)
def test_identity_bytes_that_are_not_utf8_still_give_their_own_offsets(
    launch: Callable[..., None], name: str
) -> None:
    """A variable holding bytes that are not UTF-8 still gives a client an offset.

    Python reads such bytes as lone surrogates, and two that differ in one byte get two offsets.
    """
    offsets = set()
    for value in ("0\udcfe", "0\udcff"):
        launch(**{name: value})
        offsets.add(LocalClient(None, None)._fetch_offset)
    assert len(offsets) == 2


def test_offsets_ignore_seeding(
    launch: Callable[..., None], seed_alike: Callable[[int], None]
) -> None:
    """Seeding random, numpy and torch alike, as vLLM does in every worker, moves no offset.

    Each rank gets one offset under two seeds and the ranks keep their own.
    """
    offsets = []
    for rank in range(16):
        launch(RANK=str(rank))
        seed_alike(1)
        under_seed_1 = LocalClient(None, None)._fetch_offset
        seed_alike(0)
        offsets.append(LocalClient(None, None)._fetch_offset)
        assert offsets[-1] == under_seed_1, rank
    assert len(set(offsets)) == 16


@pytest.mark.usefixtures("seed_alike")
def test_making_a_client_leaves_the_random_states_as_they_were() -> None:
    """Making a client leaves the states of random, numpy and torch's CPU generator as they were."""
    np = pytest.importorskip("numpy")
    before = _rng_states(np)

    LocalClient(None, None)

    assert _rng_states(np) == before


def test_an_offset_is_the_first_4_bytes_of_a_sha256_of_its_identity(
    launch: Callable[..., None],
) -> None:
    """Host "host" with RANK 3, LOCAL_RANK 1 and CUDA_VISIBLE_DEVICES "2,3" gets 0x8238A8CE.

    That is the first 4 bytes, big-endian, of the SHA-256 of the four values joined by NUL bytes.
    """
    launch(
        TORCHSTORE_HOSTNAME="host", RANK="3", LOCAL_RANK="1", CUDA_VISIBLE_DEVICES="2,3"
    )
    assert LocalClient(None, None)._fetch_offset == 0x8238A8CE


@pytest.mark.timeout(300)
def test_one_launch_gets_one_offset_in_every_process(
    launch: Callable[..., None],
) -> None:
    """Child processes under hash seeds 1 and 2 get the parent's offset for the same four variables.

    Their hash() values differ, so a salted hash would move it.
    """
    launch(
        TORCHSTORE_HOSTNAME="host", RANK="3", LOCAL_RANK="1", CUDA_VISIBLE_DEVICES="2,3"
    )
    # The children put the directory that holds this test's torchstore ahead of their working
    # directory, which python -c searches first, so they import it from any working directory.
    root = os.path.dirname(os.path.dirname(client_module.__file__))
    script = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from torchstore.client import LocalClient; "
        "print(LocalClient(None, None)._fetch_offset, hash('torchstore'))"
    )
    outputs = [
        subprocess.run(
            [sys.executable, "-c", script, root],
            env={**os.environ, "PYTHONHASHSEED": seed},
            stdout=subprocess.PIPE,
            text=True,
            timeout=120,
            check=True,
        ).stdout
        for seed in ("1", "2")
    ]
    offsets, hashes = zip(*(output.split()[-2:] for output in outputs))
    assert [int(offset) for offset in offsets] == [
        LocalClient(None, None)._fetch_offset
    ] * 2
    assert hashes[0] != hashes[1]


@pytest.mark.asyncio
async def test_a_partial_read_fetches_and_spreads_over_only_the_volumes_it_reads(
    store: _Store,
) -> None:
    """Rows 0 to 7 of 16, one row per volume, come from volumes 00 to 07 from the offset mod 8."""
    whole = torch.arange(64.0).reshape(16, 4)
    for row in range(16):
        store.add(
            f"volume{row:02d}",
            "rows",
            whole[row : row + 1].clone(),
            tensor_slice((row, 0), (1, 4), global_shape=(16, 4)),
        )
    read = [f"volume{row:02d}" for row in range(8)]
    for offset in range(16):
        client = store.client(offset)
        store.calls.clear()

        rows = await client.get(
            "rows", tensor_slice_spec=tensor_slice((0, 0), (8, 4), global_shape=(16, 4))
        )

        assert torch.equal(rows, whole[:8]), offset
        assert store.calls == read[offset % 8 :] + read[: offset % 8], offset


@pytest.mark.asyncio
async def test_a_slice_no_stored_slice_meets_raises_without_a_fetch(
    store: _Store,
) -> None:
    """Rows 2 and 3 of a key stored as row 0 raise as before, with no volume fetched."""
    store.add(
        "volume00",
        "split",
        torch.zeros(1, 3),
        tensor_slice((0, 0), (1, 3), global_shape=(4, 3)),
    )
    client = store.client()

    with pytest.raises(RuntimeError, match="No results found for key 'split'"):
        await client.get(
            "split", tensor_slice_spec=tensor_slice((2, 0), (2, 3), global_shape=(4, 3))
        )
    assert store.calls == []


@pytest.mark.asyncio
async def test_a_get_raises_the_error_of_the_failing_volume_it_starts_at(
    store: _Store,
) -> None:
    """When two volumes fail as their fetches start, a get raises the error of its first volume."""
    _one_key_per_volume(store, 2)
    for volume_id in ("volume00", "volume01"):
        store.failures[volume_id] = RuntimeError(f"{volume_id} failed")
    for offset in range(2):
        with pytest.raises(RuntimeError, match=f"^volume{offset:02d} failed$"):
            await store.client(offset).get_batch(["key00", "key01"])


@pytest.mark.parametrize("batch", [False, True], ids=["alone", "in-a-batch"])
@pytest.mark.parametrize(
    "listed",
    [["volume00", "volume01"], ["volume01", "volume00"]],
    ids=["listed-sorted", "listed-reversed"],
)
@pytest.mark.parametrize("offset", [0, 1])
@pytest.mark.asyncio
async def test_plain_gets_keep_mains_copy_from_every_start(
    store: _Store, offset: int, listed: list[str], batch: bool
) -> None:
    """A plain get of a slice two volumes hold keeps the copy main keeps, from every start.

    Alone that is the copy of the volume listed last for the key, and in a batch whose first key
    lists the volumes the other way, the copy of the volume listed first.
    """
    value = {"volume00": 0.0, "volume01": 1.0}
    for volume_id in listed:
        store.add(
            volume_id,
            "copies",
            torch.full((2, 2), value[volume_id]),
            tensor_slice((0, 0), (2, 2), global_shape=(2, 2)),
        )
    if batch:
        for row, volume_id in enumerate(reversed(listed)):
            store.add(
                volume_id,
                "rows",
                torch.zeros(1, 2),
                tensor_slice((row, 0), (1, 2), global_shape=(2, 2)),
            )
    client = store.client(offset)

    if batch:
        kept = (await client.get_batch(["rows", "copies"]))["copies"]
    else:
        kept = await client.get("copies")

    expected = listed[0] if batch else listed[-1]
    assert torch.equal(kept, torch.full((2, 2), value[expected]))


@pytest.mark.asyncio
async def test_in_place_gets_into_views_that_are_not_contiguous_keep_mains_copy(
    store: _Store,
) -> None:
    """A 2x2 destination's columns get no in-place copy, so the get assembles them like a plain get.

    Of two copies of column 0 it keeps main's from every start, also when the other one starts last.
    """
    column0 = tensor_slice((0, 0), (2, 1), global_shape=(2, 2))
    store.add(
        "volume00",
        "w",
        torch.full((2, 1), 9.0),
        tensor_slice((0, 1), (2, 1), global_shape=(2, 2)),
    )
    store.add("volume01", "w", torch.full((2, 1), 1.0), column0)
    store.add("volume02", "w", torch.full((2, 1), 2.0), column0)
    expected = torch.tensor([[2.0, 9.0], [2.0, 9.0]])
    for offset in range(3):
        destination = torch.full((2, 2), -1.0)

        in_place = await store.client(offset).get("w", inplace_tensor=destination)

        assert in_place is destination, offset
        assert torch.equal(in_place, expected), offset


async def _serve_two_copies(store: _Store, transport_type: TransportType) -> None:
    """Puts one 2x2 key, ``copies``, as zeros on volume00 and as ones on volume01."""
    whole = tensor_slice((0, 0), (2, 2), global_shape=(2, 2))
    for index in range(2):
        await store.serve(
            transport_type,
            f"volume{index:02d}",
            "copies",
            torch.full((2, 2), float(index)),
            whole,
        )


@pytest.mark.parametrize("offset", [0, 1])
@pytest.mark.asyncio
async def test_a_shared_memory_in_place_get_of_one_key_keeps_the_copy_whose_fetch_starts_last(
    store: _Store, offset: int
) -> None:
    """In a get of one key, SharedMemory copies into contiguous destination views in start order.

    So the in-place get keeps the copy of the volume whose fetch starts last, even when the volume
    fetched first answers slower.
    """
    await _serve_two_copies(store, TransportType.SharedMemory)
    first, last = f"volume{offset:02d}", f"volume{1 - offset:02d}"
    # The volume fetched first holds its answer for the other fetch, so only the key lock keeps
    # start order.
    store.waits[first] = last

    in_place = await store.client(offset).get(
        "copies", inplace_tensor=torch.full((2, 2), -1.0)
    )

    assert torch.equal(in_place, torch.full((2, 2), float(1 - offset)))


@pytest.mark.parametrize(
    "first_answers_slower",
    [False, True],
    ids=["answers-in-start-order", "first-answers-slower"],
)
@pytest.mark.parametrize("offset", [0, 1])
@pytest.mark.asyncio
async def test_a_monarch_rpc_in_place_get_keeps_the_copy_that_arrives_last(
    store: _Store, offset: int, first_answers_slower: bool
) -> None:
    """MonarchRPC copies each answer into the destination as it arrives, without a lock.

    So the in-place get keeps the copy that arrives last, which follows the start when the volumes
    answer in start order.
    """
    await _serve_two_copies(store, TransportType.MonarchRPC)
    value = {"volume00": 0.0, "volume01": 1.0}
    first, last = f"volume{offset:02d}", f"volume{1 - offset:02d}"
    if first_answers_slower:
        store.waits[first] = last

    in_place = await store.client(offset).get(
        "copies", inplace_tensor=torch.full((2, 2), -1.0)
    )

    arrives_last = first if first_answers_slower else last
    assert torch.equal(in_place, torch.full((2, 2), value[arrives_last]))


@pytest.mark.asyncio
async def test_a_shared_memory_in_place_batch_can_keep_a_copy_other_than_the_last_started(
    store: _Store,
) -> None:
    """In a batch a fetch can wait on one key's lock while a fetch started later copies another.

    Volume01's fetch waits on k1, whose lock volume00's fetch holds until volume02 has copied its
    k2, so the in-place get keeps volume01's copy of k2.
    """
    whole = tensor_slice((0, 0), (2, 2), global_shape=(2, 2))
    for volume_id, key, value, region in (
        ("volume00", "k1", 10.0, tensor_slice((0, 0), (1, 2), global_shape=(2, 2))),
        ("volume01", "k1", 11.0, tensor_slice((1, 0), (1, 2), global_shape=(2, 2))),
        ("volume01", "k2", 1.0, whole),
        ("volume02", "k2", 2.0, whole),
    ):
        tensor = torch.full(tuple(region.local_shape), value)
        await store.serve(TransportType.SharedMemory, volume_id, key, tensor, region)
    store.waits["volume00"] = "volume02"

    in_place = await store.client(0).get_batch(
        {"k1": torch.full((2, 2), -1.0), "k2": torch.full((2, 2), -1.0)}
    )

    assert store.calls == ["volume00", "volume01", "volume02"]
    assert torch.equal(in_place["k1"], torch.tensor([[10.0, 10.0], [11.0, 11.0]]))
    assert torch.equal(in_place["k2"], torch.full((2, 2), 1.0))
