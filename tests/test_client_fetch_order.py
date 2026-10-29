# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import importlib
import random
from types import SimpleNamespace

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


class _Store(VolumeDirectory):
    """In-process volumes that log the volume of each fetch as it starts."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.locations: dict[str, dict[str, StorageInfo]] = {}
        self.stored: dict[tuple[str, tuple | None], torch.Tensor] = {}
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
        self.stored[key, _offsets(tensor_slice)] = value

    async def locate_volumes(
        self,
        keys: list[str],
        missing_ok: bool = False,
        require_fully_committed: bool = True,
    ) -> dict[str, dict[str, StorageInfo]]:
        return {key: self.locations[key] for key in keys}

    def client(self) -> LocalClient:
        return LocalClient(
            self, SimpleNamespace(get_storage_volume=lambda volume_id: volume_id)
        )

    def _buffer(self, volume_id: str) -> SimpleNamespace:
        async def get_from_storage_volume(
            requests: list[Request],
        ) -> list[torch.Tensor]:
            self.calls.append(volume_id)
            return [self.stored[r.key, _offsets(r.tensor_slice)] for r in requests]

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


def _join_world(monkeypatch: pytest.MonkeyPatch, rank: int, world_size: int) -> None:
    """Make clients created next see this process as one rank of a torch.distributed world."""
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: world_size)


@pytest.mark.asyncio
async def test_ranks_of_one_world_start_at_different_volumes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rank r starts at the r-th volume in sorted order, whatever order it lists keys in."""
    store = _one_key_per_volume(monkeypatch, 8)
    volumes = [f"volume{index:02d}" for index in range(8)]
    keys = sorted(store.locations)
    shuffle = random.Random(0).shuffle
    for rank in range(8):
        _join_world(monkeypatch, rank, world_size=8)
        client = store.client()
        shuffle(keys)
        store.calls.clear()

        results = await client.get_batch(keys)

        assert store.calls[0] == volumes[rank]
        assert sorted(store.calls) == volumes
        for key in keys:
            assert torch.equal(results[key], store.stored[key, None])


@pytest.mark.asyncio
async def test_worlds_seeded_alike_still_spread_their_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Eight two-rank worlds, each seeding ``random`` the same way, spread their starts."""
    store = _one_key_per_volume(monkeypatch, 16)
    keys = sorted(store.locations)
    starts = []
    state = random.getstate()
    try:
        for _ in range(8):
            world = []
            for rank in range(2):
                _join_world(monkeypatch, rank, world_size=2)
                random.seed(0)
                client = store.client()
                store.calls.clear()
                await client.get_batch(keys)
                world.append(store.calls[0])
            assert world[0] != world[1]
            starts += world
    finally:
        random.setstate(state)
    assert len(set(starts)) > 2


@pytest.mark.asyncio
async def test_a_wrapped_fetch_order_still_assembles_and_reports_missing_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rows fetched starting at the last volume still assemble, and a key on no volume raises."""
    store = _Store(monkeypatch)
    whole = torch.arange(12.0).reshape(4, 3)
    for row in range(4):
        tensor_slice = TensorSlice(
            offsets=(row, 0),
            coordinates=(),
            global_shape=(4, 3),
            local_shape=(1, 3),
            mesh_shape=(),
        )
        store.add(
            f"volume{row:02d}", "split", whole[row : row + 1].clone(), tensor_slice
        )
    store.locations["missing"] = {}
    _join_world(monkeypatch, rank=3, world_size=4)
    client = store.client()

    assert torch.equal(await client.get("split"), whole)
    assert store.calls == ["volume03", "volume00", "volume01", "volume02"]
    with pytest.raises(RuntimeError, match="No results found for key 'missing'"):
        await client.get("missing")
