# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import importlib
import random
from collections.abc import Callable
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torchstore.client import LocalClient
from torchstore.controller import ObjectType, StorageInfo
from torchstore.directory import VolumeDirectory
from torchstore.transport.types import Request, TensorSlice

# The package's ``client`` function hides this module as ``torchstore.client``.
client_module = importlib.import_module("torchstore.client")


def _offsets(tensor_slice: TensorSlice | None) -> tuple | None:
    return None if tensor_slice is None else tuple(tensor_slice.offsets)


def _rows(first: int, count: int, shape: tuple[int, int]) -> TensorSlice:
    """Rows ``first`` to ``first + count - 1`` of a 2-D tensor of ``shape``."""
    return TensorSlice(
        offsets=(first, 0),
        coordinates=(),
        global_shape=shape,
        local_shape=(count, shape[1]),
        mesh_shape=(),
    )


class _Store(VolumeDirectory):
    """In-process volumes that log the volume of each fetch as it starts.

    Like SharedMemory, a fetch writes into the caller's tensor when the request carries one.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.locations: dict[str, dict[str, StorageInfo]] = {}
        self.stored: dict[tuple[str, str, tuple | None], torch.Tensor] = {}
        self.calls: list[str] = []
        monkeypatch.setattr(client_module, "create_transport_buffer", self._buffer)

    def add(
        self,
        volume_id: str,
        key: str,
        value: torch.Tensor,
        tensor_slice: TensorSlice | None = None,
    ) -> None:
        object_type = (
            ObjectType.TENSOR if tensor_slice is None else ObjectType.TENSOR_SLICE
        )
        info = self.locations.setdefault(key, {}).setdefault(
            volume_id, StorageInfo(object_type)
        )
        if tensor_slice is not None:
            info.tensor_slices.add(tensor_slice)
        self.stored[volume_id, key, _offsets(tensor_slice)] = value

    async def locate_volumes(
        self,
        keys: list[str],
        missing_ok: bool = False,
        require_fully_committed: bool = True,
    ) -> dict[str, dict[str, StorageInfo]]:
        return {key: self.locations[key] for key in keys}

    def client(self, offset: int | None = None) -> LocalClient:
        """A new client, with ``offset`` in place of its own draw when one is given."""
        client = LocalClient(
            self, SimpleNamespace(get_storage_volume=lambda volume_id: volume_id)
        )
        if offset is not None:
            client._fetch_offset = offset
        return client

    def _buffer(self, volume_id: str) -> SimpleNamespace:
        async def get_from_storage_volume(
            requests: list[Request],
        ) -> list[torch.Tensor]:
            self.calls.append(volume_id)
            results = []
            for r in requests:
                value = self.stored[volume_id, r.key, _offsets(r.tensor_slice)]
                results.append(
                    value if r.tensor_val is None else r.tensor_val.copy_(value)
                )
            return results

        return SimpleNamespace(
            supports_inplace_resharding=True,
            get_from_storage_volume=get_from_storage_volume,
        )


def _one_key_per_volume(monkeypatch: pytest.MonkeyPatch, count: int) -> _Store:
    store = _Store(monkeypatch)
    for index in range(count):
        store.add(
            f"volume{index:02d}", f"key{index:02d}", torch.full((2,), float(index))
        )
    return store


@pytest.fixture
def seed_alike():
    """Seeds random, numpy and torch alike at each call, as vLLM does in every worker.

    CUDA's generators are seeded too once CUDA is up. Every state is restored afterwards.
    """
    cuda = torch.cuda.is_initialized()
    states = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
        torch.cuda.get_rng_state_all() if cuda else None,
    )

    def seed() -> None:
        random.seed(0)
        np.random.seed(0)
        torch.default_generator.manual_seed(0)
        if cuda:
            torch.cuda.manual_seed_all(0)

    yield seed
    random.setstate(states[0])
    np.random.set_state(states[1])
    torch.set_rng_state(states[2])
    if cuda:
        torch.cuda.set_rng_state_all(states[3])


@pytest.mark.asyncio
async def test_a_client_starts_at_its_offset_into_the_sorted_volumes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fetches go in sorted volume order from the offset, wrapping, whatever the key order."""
    store = _one_key_per_volume(monkeypatch, 8)
    volumes = [f"volume{index:02d}" for index in range(8)]
    keys = sorted(store.locations)
    shuffle = random.Random(0).shuffle
    for offset in range(16):
        client = store.client(offset)
        shuffle(keys)
        store.calls.clear()

        results = await client.get_batch(keys)

        assert store.calls == volumes[offset % 8 :] + volumes[: offset % 8]
        for index, volume_id in enumerate(volumes):
            key = f"key{index:02d}"
            assert torch.equal(results[key], store.stored[volume_id, key, None])


@pytest.mark.asyncio
async def test_clients_seeded_alike_spread_their_starts_and_keep_them(
    monkeypatch: pytest.MonkeyPatch, seed_alike: Callable[[], None]
) -> None:
    """64 clients made after seeding alike keep their starts and use 9 or more of 16 volumes.

    Uniform draws fall short with probability 7.0e-16. A draw of 8 or fewer values cannot pass.
    """
    store = _one_key_per_volume(monkeypatch, 16)
    keys = sorted(store.locations)
    starts = set()
    for _ in range(64):
        seed_alike()
        client = store.client()
        firsts = []
        for _ in range(2):
            store.calls.clear()
            await client.get_batch(keys)
            firsts.append(store.calls[0])
        assert firsts[0] == firsts[1]
        starts.add(firsts[0])
    assert len(starts) >= 9


@pytest.mark.asyncio
async def test_a_new_client_keeps_its_whole_draw_from_the_os(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A client keeps the one draw it makes from random.SystemRandom over 2**32 or more values.

    Draws at the bottom, middle and top of that range stay whole and pick its first of 17 volumes.
    """
    store = _one_key_per_volume(monkeypatch, 17)
    draws: list[range] = []

    class _FromOs:
        value = 0

        def randrange(self, *args: int) -> int:
            draws.append(range(*args))
            return draws[-1][self.value]

    monkeypatch.setattr(client_module, "random", SimpleNamespace(SystemRandom=_FromOs))
    for value in (0, 1 << 31, (1 << 32) - 1):
        _FromOs.value = value
        draws.clear()
        store.calls.clear()
        client = store.client()

        await client.get_batch(sorted(store.locations))

        assert len(draws) == 1 and len(draws[0]) >= 1 << 32
        assert client._fetch_offset == value
        assert store.calls[0] == f"volume{value % 17:02d}"


@pytest.mark.asyncio
async def test_a_partial_read_fetches_and_spreads_over_only_the_volumes_it_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rows 0 to 7 of 16, one row per volume, come from volumes 00 to 07 from the offset mod 8."""
    store = _Store(monkeypatch)
    whole = torch.arange(64.0).reshape(16, 4)
    for row in range(16):
        store.add(
            f"volume{row:02d}",
            "rows",
            whole[row : row + 1].clone(),
            _rows(row, 1, (16, 4)),
        )
    read = [f"volume{row:02d}" for row in range(8)]
    for offset in range(16):
        client = store.client(offset)
        store.calls.clear()

        rows = await client.get("rows", tensor_slice_spec=_rows(0, 8, (16, 4)))

        assert torch.equal(rows, whole[:8])
        assert store.calls == read[offset % 8 :] + read[: offset % 8]


@pytest.mark.asyncio
async def test_a_slice_no_stored_slice_meets_raises_without_a_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rows 2 and 3 of a key stored as rows 0 and 1 raise as before, with no volume fetched."""
    store = _Store(monkeypatch)
    whole = torch.arange(12.0).reshape(4, 3)
    for row in range(2):
        store.add(
            f"volume{row:02d}",
            "split",
            whole[row : row + 1].clone(),
            _rows(row, 1, (4, 3)),
        )
    client = store.client()

    with pytest.raises(RuntimeError, match="No results found for key 'split'"):
        await client.get("split", tensor_slice_spec=_rows(2, 2, (4, 3)))
    assert store.calls == []


@pytest.mark.asyncio
async def test_plain_gets_keep_the_copy_of_the_volume_they_reach_last(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pairs come back grouped by volume in the order the get's requests first reach each one.

    So from every start a plain get keeps the copy of the volume it reaches last, in a batch too.
    """
    value = {"volume00": 0.0, "volume01": 1.0}
    whole = _rows(0, 2, (2, 2))
    for listed in (["volume00", "volume01"], ["volume01", "volume00"]):
        store = _Store(monkeypatch)
        for volume_id in listed:
            store.add(volume_id, "copies", torch.full((2, 2), value[volume_id]), whole)
        # A batch that reads "rows" first reaches the volumes in the opposite order.
        for row, volume_id in enumerate(reversed(listed)):
            store.add(volume_id, "rows", torch.zeros(1, 2), _rows(row, 1, (2, 2)))
        for offset in range(2):
            client = store.client(offset)

            alone = await client.get("copies")
            batch = await client.get_batch(["rows", "copies"])

            assert torch.equal(alone, torch.full((2, 2), value[listed[1]]))
            assert torch.equal(batch["copies"], torch.full((2, 2), value[listed[0]]))


@pytest.mark.asyncio
async def test_an_in_place_get_keeps_the_copy_written_last(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In-place copies land in completion order and the last write wins for client-side copies.

    The stand-in copies on the client and finishes each fetch before the next one starts.
    """
    store = _Store(monkeypatch)
    store.add("volume01", "copies", torch.full((2, 2), 1.0), _rows(0, 2, (2, 2)))
    store.add("volume00", "copies", torch.full((2, 2), 0.0), _rows(0, 2, (2, 2)))
    for offset in range(2):
        client = store.client(offset)
        store.calls.clear()

        in_place = await client.get("copies", inplace_tensor=torch.full((2, 2), -1.0))

        assert store.calls == [f"volume{offset:02d}", f"volume{1 - offset:02d}"]
        assert torch.equal(in_place, torch.full((2, 2), float(1 - offset)))
