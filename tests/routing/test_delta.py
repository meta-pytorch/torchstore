# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from collections.abc import Sequence

import pytest
import torch
import torchstore.routing.delta as delta_module
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Partial, Placement, Replicate, Shard
from torch.distributed.tensor._utils import _compute_local_shape_and_global_offset
from torch.distributed.tensor.placement_types import _StridedShard
from torchstore.routing.delta import (
    get_state_dict_delta,
    put_state_dict_delta,
    StateDictDelta,
)

from .utils import routing_clients

pytestmark = pytest.mark.timeout(120)

_SHAPE = (8, 6)


@pytest.fixture
def full_reads(monkeypatch: pytest.MonkeyPatch) -> dict[object, set[str]]:
    """Flat keys each client read in full; pop a client's entry after a get."""
    reads: dict[object, set[str]] = {}
    full_get_state_dict = delta_module.get_state_dict

    async def get_state_dict(store, key, user_state_dict, strict):
        reads.setdefault(store, set()).update(user_state_dict)
        return await full_get_state_dict(store, key, user_state_dict, strict=strict)

    monkeypatch.setattr(delta_module, "get_state_dict", get_state_dict)
    return reads


def _distribute(
    tensor: torch.Tensor,
    mesh_shape: tuple[int, ...],
    placements: Sequence[Placement],
    rank: int,
) -> torch.Tensor:
    """One rank's DTensor of ``tensor`` on a fake mesh, without a process group."""
    mesh = DeviceMesh(
        "cpu",
        torch.arange(math.prod(mesh_shape)).view(mesh_shape),
        _init_backend=False,
        _rank=rank,
    )
    shape, offsets = _compute_local_shape_and_global_offset(
        tensor.shape, mesh_shape, mesh.get_coordinate(), placements
    )
    local = tensor[tuple(slice(o, o + n) for o, n in zip(offsets, shape))]
    return DTensor.from_local(
        local.clone(),
        mesh,
        placements,
        run_check=False,
        shape=tensor.shape,
        stride=tensor.stride(),
    )


def _versions() -> list[torch.Tensor]:
    """Successive versions: first, sparse, dense (overflows), sparse again."""
    first = torch.randn(_SHAPE, generator=torch.Generator().manual_seed(0))
    sparse = first.clone()
    sparse[0, 0] += 1
    sparse[3, 5] += 1
    sparse[4, 0] += 1
    dense = sparse + 1
    # Changes on both sides of the row- and column-shard boundaries.
    sparse_again = dense.clone()
    sparse_again[3, 2] -= 1
    sparse_again[4, 3] -= 1
    return [first, sparse, dense, sparse_again]


# (mesh shape, placements) per side; each publisher/requester rank gets one shard.
_LAYOUTS = {
    "row_to_column": (((2,), [Shard(0)]), ((2,), [Shard(1)])),
    "row_to_full": (((2,), [Shard(0)]), ((1,), [Replicate()])),
    "full_to_column": (((1,), [Replicate()]), ((2,), [Shard(1)])),
    "column_to_full": (((2,), [Shard(1)]), ((1,), [Replicate()])),
    "replicated_columns_to_2d": (
        ((2, 2), [Replicate(), Shard(1)]),
        ((2, 2), [Shard(0), Shard(1)]),
    ),
    # FSDP over TP rows.
    "strided_rows_to_column": (
        ((2, 2), [_StridedShard(0, split_factor=2), Shard(0)]),
        ((2,), [Shard(1)]),
    ),
}


@pytest.mark.parametrize("layout", list(_LAYOUTS))
@pytest.mark.asyncio
async def test_delta_sync_matches_latest_version(
    layout: str, full_reads: dict[object, set[str]]
) -> None:
    """Delta pulls track every version bitwise across resharding and overflow.

    Replica 1 of each requester skips the dense version, so it then needs a
    full read.
    """
    reads_by_requester: dict[str, list[set[str] | None]] = {}
    (publisher_mesh, publisher_placements), (
        requester_mesh,
        requester_placements,
    ) = _LAYOUTS[layout]
    num_publishers = torch.Size(publisher_mesh).numel()
    num_requesters = torch.Size(requester_mesh).numel()
    versions = _versions()
    bias = torch.arange(4, dtype=torch.float32)

    def publisher_state_dict(version: int, rank: int) -> dict[str, torch.Tensor]:
        return {
            "w": _distribute(
                versions[version], publisher_mesh, publisher_placements, rank
            ),
            # Plain tensor on every publisher, so its replicas collapse to one row.
            "bias": bias + version,
            "step": torch.tensor(version),
        }

    destinations = {
        f"requester/{rank}/{replica}": {
            "w": _distribute(
                torch.zeros(_SHAPE, dtype=torch.bfloat16),
                requester_mesh,
                requester_placements,
                rank,
            ),
            "bias": torch.zeros(4, dtype=torch.bfloat16),
            "step": torch.tensor(-1),
            # No publisher has it, so it is left unchanged.
            "local_only": torch.full((2,), -1.0),
        }
        for rank in range(num_requesters)
        for replica in range(2)
    }
    clients = await routing_clients(
        publishers={f"publisher/{rank}": {} for rank in range(num_publishers)},
        requesters=destinations,
    )
    deltas = {
        rank: StateDictDelta(
            publisher_state_dict(0, rank),
            max_fraction=0.25,
            transfer_dtype=torch.bfloat16,
            preserve_dtype_keys=frozenset({"step"}),
        )
        for rank in range(num_publishers)
    }

    for version in range(len(versions)):
        for rank, delta in deltas.items():
            state_dict = publisher_state_dict(version, rank)
            await put_state_dict_delta(
                clients[f"publisher/{rank}"], delta, state_dict, "model"
            )
        for requester, destination in destinations.items():
            rank, replica = map(int, requester.split("/")[1:])
            client = clients[requester]
            reads = reads_by_requester.setdefault(requester, [])
            if replica == 1 and version == 2:
                reads.append(None)
                continue
            result = await get_state_dict_delta(client, "model", destination)
            reads.append(full_reads.pop(client, set()))
            assert result is destination
            if replica == 1:
                # Already at the latest version: nothing to read or change.
                await get_state_dict_delta(client, "model", destination)
                assert client not in full_reads
            expected = _distribute(
                versions[version].bfloat16(),
                requester_mesh,
                requester_placements,
                rank,
            )
            assert torch.equal(
                destination["w"].to_local().view(torch.int16),
                expected.to_local().view(torch.int16),
            )
            assert torch.equal(
                destination["bias"].view(torch.int16),
                (bias + version).bfloat16().view(torch.int16),
            )
            assert destination["step"].item() == version
            assert torch.equal(destination["local_only"], torch.full((2,), -1.0))

    everything = {"w", "bias", "step"}
    for requester, reads in reads_by_requester.items():
        if requester.endswith("/0"):
            assert reads == [everything, set(), {"w"}, set()]
        else:
            assert reads == [everything, set(), None, everything]


def test_stage_encodes_changed_elements_and_overflow() -> None:
    """Rows hold the changed values and indices, and -1 once they do not fit."""
    weight = torch.zeros(16)
    delta = StateDictDelta({"w": weight}, max_fraction=0.5)
    [entry] = delta._entries.values()
    assert entry.capacity == 8
    count, version, epoch = entry.packed[:24].view(torch.int64)

    delta._stage({"w": weight})
    assert count.item() == -1, "the first version is always sent in full"
    assert version.item() == 1
    assert epoch.item() == delta._epoch

    weight[[2, 9]] = torch.tensor([1.0, -2.0])
    delta._stage({"w": weight})
    assert count.item() == 2
    values = entry.packed[24 : 24 + 8 * 4].view(torch.float32)[:2]
    indices = entry.packed[24 + 8 * 4 :].view(torch.int32)[:2]
    assert indices.tolist() == [2, 9]
    assert values.tolist() == [1.0, -2.0]

    # Negative zero differs bitwise from zero, so it is sent.
    weight[0] = -0.0
    delta._stage({"w": weight})
    assert count.item() == 1

    weight += 1
    delta._stage({"w": weight})
    assert count.item() == -1

    delta._stage({"w": weight})
    assert count.item() == 0
    assert version.item() == 5


def test_stage_failure_starts_a_new_epoch(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed stage may leave rows partly encoded, so requesters must read in full."""
    tensors = {"a": torch.zeros(16), "b": torch.zeros(16)}
    delta = StateDictDelta(tensors, max_fraction=0.5)
    delta._stage(tensors)
    epoch = delta._epoch
    encode = delta._encode
    calls = []

    def fail_second(entry, source):
        calls.append(entry)
        if len(calls) == 2:
            raise RuntimeError("encode failed")
        encode(entry, source)

    monkeypatch.setattr(delta, "_encode", fail_second)
    with pytest.raises(RuntimeError, match="encode failed"):
        delta._stage(tensors)
    assert delta._version == 0
    assert delta._epoch != epoch


@pytest.mark.asyncio
async def test_delta_sync_falls_back_for_stale_destinations(
    full_reads: dict[object, set[str]],
) -> None:
    """Another destination, in-place writes and a restarted publisher read in full."""
    clients = await routing_clients(
        publishers={"publisher/0": {}}, requesters={"requester/0": {}}
    )
    publisher, requester = clients["publisher/0"], clients["requester/0"]
    first, other = torch.zeros(_SHAPE), torch.zeros(_SHAPE)

    async def pull(destination: torch.Tensor, expected: torch.Tensor) -> set[str]:
        await get_state_dict_delta(requester, "model", {"w": destination})
        assert torch.equal(destination, expected)
        return full_reads.pop(requester, set())

    weight = torch.randn(_SHAPE)
    delta = StateDictDelta({"w": weight}, max_fraction=0.5)
    await put_state_dict_delta(publisher, delta, {"w": weight}, "model")
    assert await pull(first, weight) == {"w"}
    weight[0, 0] += 1
    await put_state_dict_delta(publisher, delta, {"w": weight}, "model")
    assert await pull(first, weight) == set()
    assert await pull(other, weight) == {"w"}, "another destination"
    assert await pull(first, weight) == {"w"}, "the route now tracks the other one"

    first[1, 1] = 100.0
    weight[2, 2] += 1
    await put_state_dict_delta(publisher, delta, {"w": weight}, "model")
    assert await pull(first, weight) == {"w"}, "an in-place write"

    # A new publisher instance starts at version 1 again; its version 3 delta
    # is against a base the requester never had.
    restarted = weight + 1
    restarted_delta = StateDictDelta({"w": restarted}, max_fraction=0.5)
    for _ in range(3):
        await put_state_dict_delta(
            publisher, restarted_delta, {"w": restarted}, "model"
        )
    restarted[3, 3] += 1
    await put_state_dict_delta(publisher, restarted_delta, {"w": restarted}, "model")
    assert restarted_delta._version == delta._version + 1
    assert await pull(first, restarted) == {"w"}


@pytest.mark.asyncio
async def test_delta_sync_one_restarted_publisher(
    full_reads: dict[object, set[str]],
) -> None:
    """Rows at different versions apply once the destination holds each of them."""
    clients = await routing_clients(
        publishers={f"publisher/{rank}": {} for rank in range(2)},
        requesters={"requester/0": {}},
    )
    requester = clients["requester/0"]
    weight = torch.randn(_SHAPE)
    destination = torch.zeros(_SHAPE)

    def state_dict(rank: int) -> dict[str, torch.Tensor]:
        return {"w": _distribute(weight, (2,), [Shard(0)], rank)}

    async def publish() -> None:
        for rank, delta in deltas.items():
            await put_state_dict_delta(
                clients[f"publisher/{rank}"], delta, state_dict(rank), "model"
            )

    async def pull() -> set[str]:
        await get_state_dict_delta(requester, "model", {"w": destination})
        assert torch.equal(destination, weight)
        return full_reads.pop(requester, set())

    deltas = {
        rank: StateDictDelta(state_dict(rank), max_fraction=0.5) for rank in range(2)
    }
    await publish()
    assert await pull() == {"w"}
    weight[0, 0] += 1
    await publish()
    assert await pull() == set()

    deltas[1] = StateDictDelta(state_dict(1), max_fraction=0.5)
    weight[1, 1] += 1
    await publish()
    assert await pull() == {"w"}, "the restarted publisher's first version"
    weight[[2, 6], [2, 3]] += 1
    await publish()
    assert await pull() == set()


@pytest.mark.asyncio
async def test_delta_sync_views_of_one_buffer(
    full_reads: dict[object, set[str]],
) -> None:
    """Writing one view of a buffer does not count as modifying the others."""
    clients = await routing_clients(
        publishers={"publisher/0": {}}, requesters={"requester/0": {}}
    )
    publisher, requester = clients["publisher/0"], clients["requester/0"]
    weights = {"a": torch.randn(16), "b": torch.randn(16)}
    buffer = torch.zeros(32)
    destination = {"a": buffer[:16], "b": buffer[16:]}
    delta = StateDictDelta(weights, max_fraction=0.5)
    reads = []
    for version in range(4):
        weights["a" if version % 2 else "b"][version] += 1
        await put_state_dict_delta(publisher, delta, weights, "model")
        await get_state_dict_delta(requester, "model", destination)
        reads.append(full_reads.pop(requester, set()))
        assert torch.equal(buffer, torch.cat([weights["a"], weights["b"]]))
    assert reads == [{"a", "b"}, set(), set(), set()]


@pytest.mark.asyncio
async def test_delta_sync_retries_failed_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clients = await routing_clients(
        publishers={"publisher/0": {}}, requesters={"requester/0": {}}
    )
    publisher, requester = clients["publisher/0"], clients["requester/0"]
    weight = torch.randn(_SHAPE)
    delta = StateDictDelta({"w": weight}, max_fraction=0.5)
    await put_state_dict_delta(publisher, delta, {"w": weight}, "model")

    install = requester._install_requester_plan

    def fail_once(key, registrations):
        monkeypatch.setattr(requester, "_install_requester_plan", install)
        raise RuntimeError("injected")

    monkeypatch.setattr(requester, "_install_requester_plan", fail_once)
    destination = torch.zeros(_SHAPE)
    with pytest.raises(RuntimeError, match="injected"):
        await get_state_dict_delta(requester, "model", {"w": destination})
    await get_state_dict_delta(requester, "model", {"w": destination})
    assert torch.equal(destination, weight)


@pytest.mark.asyncio
async def test_delta_sync_rejects_inference_destinations() -> None:
    clients = await routing_clients(
        publishers={"publisher/0": {}}, requesters={"requester/0": {}}
    )
    delta = StateDictDelta({"w": torch.zeros(_SHAPE)}, max_fraction=0.5)
    await put_state_dict_delta(
        clients["publisher/0"], delta, {"w": torch.randn(_SHAPE)}, "model"
    )
    with torch.inference_mode():
        inference = torch.zeros(_SHAPE)
    with pytest.raises(ValueError, match="inference tensor"):
        await get_state_dict_delta(clients["requester/0"], "model", {"w": inference})


def test_stage_rejects_unsupported_placements() -> None:
    mesh = DeviceMesh("cpu", torch.arange(2), _init_backend=False, _rank=0)
    partial = DTensor.from_local(
        torch.zeros(_SHAPE), mesh, [Partial()], run_check=False
    )
    with pytest.raises(ValueError, match="unsupported placement"):
        StateDictDelta({"w": partial}, max_fraction=0.5)
