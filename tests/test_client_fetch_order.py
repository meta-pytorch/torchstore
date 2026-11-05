# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import dataclasses
import importlib
import random
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torchstore.client import LocalClient
from torchstore.controller import ObjectType, StorageInfo
from torchstore.directory import VolumeDirectory
from torchstore.transport.types import Request, TensorSlice

from .routing.utils import tensor_slice

# The package's ``client`` function hides this module as ``torchstore.client``.
client_module = importlib.import_module("torchstore.client")


def _offsets(region: TensorSlice | None) -> tuple | None:
    return None if region is None else tuple(region.offsets)


@dataclasses.dataclass(frozen=True)
class _VolumeName:
    """A hashable volume id without an order."""

    name: str


class _Store(VolumeDirectory):
    """In-process volumes that log the volume of each fetch as it starts."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.locations: dict[str, dict[Any, StorageInfo]] = {}
        self.stored: dict[tuple[Any, str, tuple | None], torch.Tensor] = {}
        self.calls: list[Any] = []
        monkeypatch.setattr(client_module, "create_transport_buffer", self._buffer)

    def add(
        self,
        volume_id: Any,
        key: str,
        value: torch.Tensor,
        region: TensorSlice | None = None,
    ) -> None:
        object_type = ObjectType.TENSOR if region is None else ObjectType.TENSOR_SLICE
        info = self.locations.setdefault(key, {}).setdefault(
            volume_id, StorageInfo(object_type)
        )
        if region is not None:
            info.tensor_slices.add(region)
        self.stored[volume_id, key, _offsets(region)] = value

    async def locate_volumes(
        self,
        keys: list[str],
        missing_ok: bool = False,
        require_fully_committed: bool = True,
    ) -> dict[str, dict[Any, StorageInfo]]:
        return {key: self.locations[key] for key in keys}

    def client(self, offset: int | None = None) -> LocalClient:
        """A new client that uses ``offset`` in place of its own when one is given."""
        strategy = SimpleNamespace(get_storage_volume=lambda volume_id: volume_id)
        client = LocalClient(self, strategy)
        if offset is not None:
            client._fetch_offset = offset
        return client

    def _buffer(self, volume_id: Any) -> SimpleNamespace:
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


def _one_key_per_volume(store: _Store, count: int) -> None:
    for index in range(count):
        store.add(
            f"volume{index:02d}", f"key{index:02d}", torch.full((2,), float(index))
        )


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> _Store:
    return _Store(monkeypatch)


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


@pytest.mark.asyncio
async def test_a_client_starts_every_get_at_its_offset_into_the_sorted_volumes(
    store: _Store,
) -> None:
    """Fetches wrap around the sorted volumes from the offset in any key order."""
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
    """Every offset starts into one order of the ids: sorted where they order and by repr otherwise."""
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
    """Either 256 ranks on one host or 256 hosts without ranks reach all 17 starts over 17 volumes."""
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
    """Processes whose host name, RANK, LOCAL_RANK or CUDA_VISIBLE_DEVICES differ get their own offsets."""
    offsets = set()
    for env in envs:
        launch(**env)
        offsets.add(LocalClient(None, None)._fetch_offset)
    assert len(offsets) == len(envs)


def test_an_offset_is_the_first_4_bytes_of_a_sha256_of_its_identity(
    launch: Callable[..., None],
) -> None:
    """0x8238A8CE is the big-endian first 4 bytes of the SHA-256 of the four values joined by NUL bytes."""
    launch(
        TORCHSTORE_HOSTNAME="host", RANK="3", LOCAL_RANK="1", CUDA_VISIBLE_DEVICES="2,3"
    )
    assert LocalClient(None, None)._fetch_offset == 0x8238A8CE


@pytest.mark.asyncio
async def test_a_partial_read_fetches_and_spreads_over_only_the_volumes_it_reads(
    store: _Store,
) -> None:
    """Rows 0 to 7 of 16 rows stored one per volume come from volumes 00 to 07 starting at the offset mod 8."""
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
    """Rows 2 and 3 of a key stored as row 0 raise as before and fetch no volume."""
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
    """Alone or in a batch, a plain get of a slice two volumes hold keeps main's copy from every start."""
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
    """Columns of a 2x2 destination take no in-place copy and keep main's copy from every start."""
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
