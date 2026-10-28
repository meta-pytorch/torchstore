# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Correctness tests for direct_weight_sync.py.

Uses a MockRDMABuffer that copies source bytes into the dest byte view,
simulating what real RDMA does. No GPU or RDMA infrastructure needed.
"""

import asyncio

import pytest
import torch
import torch.distributed as dist

from torchstore import state_dict_utils
from torchstore.client import LocalClient
from torchstore.direct_weight_sync import (
    DirectWeightSyncDest,
    DirectWeightSyncSource,
    RDMAWeightHandle,
)
from torchstore.transport.types import TensorSlice
from torchstore.utils import to_byte_view

pytestmark = pytest.mark.asyncio


class MockRDMABuffer:
    """Simulates Monarch RDMABuffer by copying source bytes into dest."""

    def __init__(self, source_bytes: torch.Tensor):
        self._source = source_bytes

    async def read_into(self, dest_byte_view: torch.Tensor):
        dest_byte_view.copy_(self._source)

    async def drop(self):
        pass


def _make_sharded_handles(
    original: torch.Tensor,
    num_shards: int,
    shard_dim: int,
) -> list[RDMAWeightHandle]:
    """Create mock handles simulating a tensor sharded across num_shards ranks."""
    handles = []
    shard_size = original.shape[shard_dim] // num_shards

    for rank in range(num_shards):
        idx = [slice(None)] * original.ndim
        idx[shard_dim] = slice(rank * shard_size, (rank + 1) * shard_size)
        shard_data = original[tuple(idx)].contiguous()

        offsets = [0] * original.ndim
        offsets[shard_dim] = rank * shard_size
        local_shape = list(original.shape)
        local_shape[shard_dim] = shard_size

        tensor_slice = TensorSlice(
            offsets=tuple(offsets),
            coordinates=(rank,),
            global_shape=tuple(original.shape),
            local_shape=tuple(local_shape),
            mesh_shape=(num_shards,),
        )
        buf = MockRDMABuffer(to_byte_view(shard_data))
        handles.append(
            RDMAWeightHandle(
                rdma_buffer=buf,
                tensor_slice=tensor_slice,
                source_rank=rank,
            )
        )
    return handles


def _make_replicated_handles(
    original: torch.Tensor,
    num_ranks: int,
) -> list[RDMAWeightHandle]:
    """Create mock handles simulating a replicated tensor across num_ranks."""
    handles = []
    for rank in range(num_ranks):
        data = original.contiguous()
        tensor_slice = TensorSlice(
            offsets=tuple(0 for _ in range(original.ndim)),
            coordinates=(rank,),
            global_shape=tuple(original.shape),
            local_shape=tuple(original.shape),
            mesh_shape=(num_ranks,),
        )
        buf = MockRDMABuffer(to_byte_view(data))
        handles.append(
            RDMAWeightHandle(
                rdma_buffer=buf,
                tensor_slice=tensor_slice,
                source_rank=rank,
            )
        )
    return handles


async def test_exact_match():
    """Source is full tensor, dest is full tensor → Case 1 (zero-copy)."""
    original = torch.arange(512 * 512, dtype=torch.float32).reshape(512, 512)
    handles = _make_sharded_handles(original, num_shards=1, shard_dim=0)

    dest = torch.zeros_like(original)
    sync = DirectWeightSyncDest()
    await sync.pull({"weight": handles}, {"weight": dest})

    assert torch.equal(dest, original)
    # Case 1: zero-copy, no recv_buffer or dest_tensor
    assert len(sync._plan) == 1
    assert sync._plan[0].dest_tensor is None
    assert sync._plan[0].recv_buffer is None


@pytest.mark.parametrize(
    "num_shards,shard_dim",
    [
        (2, 0),  # row sharding
        (4, 0),  # finer row sharding
        (2, 1),  # column sharding
    ],
)
async def test_resharding(num_shards, shard_dim):
    """Source is sharded, dest is full tensor → Case 3 (resharding)."""
    original = torch.arange(512 * 512, dtype=torch.float32).reshape(512, 512)
    handles = _make_sharded_handles(
        original, num_shards=num_shards, shard_dim=shard_dim
    )

    dest = torch.zeros_like(original)
    sync = DirectWeightSyncDest()
    await sync.pull({"weight": handles}, {"weight": dest})

    assert torch.equal(dest, original)
    assert len(sync._plan) == num_shards


async def test_resharding_into_parameter():
    """Resharded copies write into parameters that require grad."""
    original = torch.arange(64 * 64, dtype=torch.float32).reshape(64, 64)
    handles = _make_sharded_handles(original, num_shards=2, shard_dim=0)

    dest = torch.nn.Parameter(torch.zeros_like(original))
    await DirectWeightSyncDest().pull({"weight": handles}, {"weight": dest})

    assert torch.equal(dest, original)


async def test_partial_coverage_raises():
    """Handles that leave part of a destination unwritten are rejected."""
    original = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    handles = _make_sharded_handles(original, num_shards=2, shard_dim=0)[:1]

    with pytest.raises(ValueError, match="do not cover 'weight' exactly: 32 of 64"):
        await DirectWeightSyncDest().pull(
            {"weight": handles}, {"weight": torch.zeros_like(original)}
        )


async def test_replicated_dedup():
    """Replicated source (2 ranks, same data) → should only read once."""
    original = torch.arange(512 * 512, dtype=torch.float32).reshape(512, 512)
    handles = _make_replicated_handles(original, num_ranks=2)

    dest = torch.zeros_like(original)
    sync = DirectWeightSyncDest()
    await sync.pull({"weight": handles}, {"weight": dest})

    assert torch.equal(dest, original)
    # Dedup: only 1 op despite 2 source ranks
    assert len(sync._plan) == 1


@pytest.mark.parametrize(
    "shape",
    [pytest.param((0, 8), id="empty"), pytest.param((4, 8), id="nonempty")],
)
async def test_missing_rdma_buffer_only_allowed_for_empty_tensor(shape):
    tensor = torch.empty(shape, dtype=torch.bfloat16)
    tensor_slice = TensorSlice(
        offsets=(0, 0),
        coordinates=(0,),
        global_shape=shape,
        local_shape=shape,
        mesh_shape=(1,),
    )
    handles = [
        RDMAWeightHandle(
            rdma_buffer=None,
            tensor_slice=tensor_slice,
            source_rank=0,
        )
    ]

    sync = DirectWeightSyncDest()
    if tensor.numel() == 0:
        await sync.pull({"weight": handles}, {"weight": tensor})
        assert sync._plan == []
    else:
        with pytest.raises(
            RuntimeError, match="Missing RDMA buffer for nonempty tensor slice weight"
        ):
            await sync.pull({"weight": handles}, {"weight": tensor})


async def test_multiple_params():
    """State dict with multiple params, each handled independently."""
    w1 = torch.arange(100, dtype=torch.float32).reshape(10, 10)
    w2 = torch.arange(100, 200, dtype=torch.float32).reshape(10, 10)

    all_handles = {
        "layer.weight": _make_sharded_handles(w1, num_shards=2, shard_dim=0),
        "layer.bias": _make_sharded_handles(w2, num_shards=1, shard_dim=0),
    }
    dest_sd = {
        "layer.weight": torch.zeros_like(w1),
        "layer.bias": torch.zeros_like(w2),
    }

    sync = DirectWeightSyncDest()
    await sync.pull(all_handles, dest_sd)

    assert torch.equal(dest_sd["layer.weight"], w1)
    assert torch.equal(dest_sd["layer.bias"], w2)


async def test_refresh():
    """After modifying source and calling refresh(), dest sees updated values."""
    source = DirectWeightSyncSource()

    # Create a non-contiguous source tensor (simulates column sharding)
    backing = torch.arange(100, dtype=torch.float32).reshape(10, 10)
    source_tensor = backing[:, :5]  # non-contiguous view, shape (10, 5)
    assert not source_tensor.is_contiguous()

    # Manually set up staging (simulating what register() does)
    staging_buf = source_tensor.contiguous()
    source._staging["weight"] = (staging_buf, source_tensor)

    # Create mock handle pointing at the staging buffer
    mock_buf = MockRDMABuffer(to_byte_view(staging_buf))
    handle = RDMAWeightHandle(
        rdma_buffer=mock_buf,
        tensor_slice=TensorSlice(
            offsets=(0, 0),
            coordinates=(0,),
            global_shape=(10, 5),
            local_shape=(10, 5),
            mesh_shape=(1,),
        ),
        source_rank=0,
    )
    all_handles = {"weight": [handle]}

    # First pull: should get original values
    dest = torch.zeros(10, 5, dtype=torch.float32)
    sync = DirectWeightSyncDest()
    await sync.pull(all_handles, {"weight": dest})
    assert torch.equal(dest, source_tensor.contiguous())

    # Modify source (simulates optimizer.step())
    backing.fill_(99.0)
    assert not torch.equal(staging_buf, source_tensor.contiguous())

    # Refresh copies updated source into staging buffer
    source.refresh()
    assert torch.equal(staging_buf, source_tensor.contiguous())

    # Second pull: should get updated values
    dest2 = torch.zeros(10, 5, dtype=torch.float32)
    sync2 = DirectWeightSyncDest()
    await sync2.pull(all_handles, {"weight": dest2})
    assert torch.equal(dest2, torch.full((10, 5), 99.0))


async def test_transfer_dtype():
    """Source is float32, transfer_dtype=bfloat16 → staging casts, dest gets bfloat16."""
    source = DirectWeightSyncSource()

    # Float32 source tensor (simulates model param)
    original = torch.arange(100, dtype=torch.float32).reshape(10, 10)

    # Set up staging with bfloat16 cast (simulates register(transfer_dtype=bfloat16))
    # Mock RDMABuffer by patching — register() calls RDMABuffer internally,
    # so we need to test via the staging + mock handle approach instead.
    staging_buf = original.to(torch.bfloat16).contiguous()
    source._staging["weight"] = (staging_buf, original)

    # Create mock handle pointing at the bfloat16 staging buffer
    mock_buf = MockRDMABuffer(to_byte_view(staging_buf))
    handle = RDMAWeightHandle(
        rdma_buffer=mock_buf,
        tensor_slice=TensorSlice(
            offsets=(0, 0),
            coordinates=(0,),
            global_shape=(10, 10),
            local_shape=(10, 10),
            mesh_shape=(1,),
        ),
        source_rank=0,
    )

    # Pull into bfloat16 dest (matching transfer dtype)
    dest = torch.zeros(10, 10, dtype=torch.bfloat16)
    sync = DirectWeightSyncDest()
    await sync.pull({"weight": [handle]}, {"weight": dest})

    # Values should match (float32 → bfloat16 → bfloat16)
    expected = original.to(torch.bfloat16)
    assert torch.equal(dest, expected)

    # All params should be staged (due to transfer_dtype)
    assert "weight" in source._staging

    # Modify source (simulates optimizer.step() updating float32 param)
    original.fill_(42.0)

    # Refresh re-casts float32 → bfloat16 into staging buffer
    source.refresh()
    assert torch.equal(staging_buf, torch.full((10, 10), 42.0, dtype=torch.bfloat16))

    # Second pull sees updated values
    dest2 = torch.zeros(10, 10, dtype=torch.bfloat16)
    sync2 = DirectWeightSyncDest()
    await sync2.pull({"weight": [handle]}, {"weight": dest2})
    assert torch.equal(dest2, torch.full((10, 10), 42.0, dtype=torch.bfloat16))


async def test_plan_follows_destination_changes():
    """Rebuild the plan for new dest tensors, subsets and reallocated storage."""
    weights = {
        name: torch.arange(64, dtype=torch.float32).reshape(8, 8) + i
        for i, name in enumerate(("a", "b"))
    }
    all_handles = {
        name: _make_sharded_handles(t, num_shards=2, shard_dim=0)
        for name, t in weights.items()
    }
    sync = DirectWeightSyncDest()

    # Pull one key at a time, as when loading one FSDP group at a time.
    dests = {name: torch.zeros_like(t) for name, t in weights.items()}
    for name in dests:
        await sync.pull(all_handles, {name: dests[name]})
    torch.testing.assert_close(dests, weights, rtol=0, atol=0)

    # Free and reallocate a destination's storage in place at a new address.
    await sync.pull(all_handles, {"a": dests["a"]})
    plan = sync._plan
    storage = dests["a"].untyped_storage()
    nbytes = storage.nbytes()
    old_ptr = storage.data_ptr()
    storage.resize_(0)
    occupied = torch.empty(nbytes, dtype=torch.uint8)  # take the freed memory
    storage.resize_(nbytes)
    assert storage.data_ptr() != old_ptr
    dests["a"].zero_()
    await sync.pull(all_handles, {"a": dests["a"]})
    assert sync._plan is not plan
    torch.testing.assert_close(dests["a"], weights["a"], rtol=0, atol=0)
    del occupied

    # Same memory reuses the plan; extra non-tensor entries are ignored.
    plan = sync._plan
    await sync.pull(all_handles, {"a": dests["a"], "extra": None})
    assert sync._plan is plan

    # A different storage or view of the same size rebuilds it.
    moved = torch.zeros_like(dests["a"])
    await sync.pull(all_handles, {"a": moved})
    assert sync._plan is not plan
    torch.testing.assert_close(moved, weights["a"], rtol=0, atol=0)
    plan = sync._plan
    transposed = torch.zeros_like(moved).t()
    await sync.pull(all_handles, {"a": transposed})
    assert sync._plan is not plan

    # New handles for the same memory and key set rebuild it.
    plan = sync._plan
    same_keys = {
        "a": _make_sharded_handles(weights["a"] + 50, num_shards=2, shard_dim=0)
    }
    await sync.pull(same_keys, {"a": transposed})
    assert sync._plan is not plan
    torch.testing.assert_close(transposed, weights["a"] + 50, rtol=0, atol=0)

    # New handles for all keys.
    new_weights = {name: t + 100 for name, t in weights.items()}
    new_handles = {
        name: _make_sharded_handles(t, num_shards=2, shard_dim=0)
        for name, t in new_weights.items()
    }
    await sync.pull(new_handles, dests)
    torch.testing.assert_close(dests, new_weights, rtol=0, atol=0)


async def test_local_client_publishes_and_fetches_every_rank(monkeypatch):
    weight = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    per_rank = _make_sharded_handles(weight, num_shards=2, shard_dim=0)
    store = {}

    async def put(key, value):
        store[key] = value

    async def get(key):
        return store[key]

    client = object.__new__(LocalClient)
    monkeypatch.setattr(client, "put", put, raising=False)
    monkeypatch.setattr(client, "get", get, raising=False)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    # Rank 1 may publish before rank 0.
    for rank in (1, 0):
        monkeypatch.setattr(dist, "get_rank", lambda rank=rank: rank)
        await client.publish_rdma_handles("w", {"weight": per_rank[rank]})

    all_handles = await client.get_rdma_handles("w")

    assert dict(all_handles) == {"weight": per_rank}
    dest = torch.zeros_like(weight)
    await DirectWeightSyncDest().pull(all_handles, {"weight": dest})
    torch.testing.assert_close(dest, weight, rtol=0, atol=0)
    with pytest.raises(KeyError, match="missing"):
        await client.get_rdma_handles("missing")


async def test_failed_publish_releases_registrations(monkeypatch):
    events = []

    def register(self, state_dict, rank, transfer_dtype=None):
        events.append("register")
        return {}

    async def cleanup(self):
        events.append("cleanup")

    class Store:
        fail = True

        async def publish_rdma_handles(self, key, handles):
            if self.fail:
                raise RuntimeError("publish failed")

    monkeypatch.setattr(DirectWeightSyncSource, "register", register)
    monkeypatch.setattr(DirectWeightSyncSource, "cleanup", cleanup)
    store = Store()
    try:
        with pytest.raises(RuntimeError, match="publish failed"):
            await state_dict_utils._put_state_dict_direct_rdma(store, {}, "w")
        assert events == ["register", "cleanup"]

        store.fail = False
        await state_dict_utils._put_state_dict_direct_rdma(store, {}, "w")
        assert events == ["register", "cleanup", "register"]
    finally:
        state_dict_utils._rdma_cache.pop(id(store), None)


async def test_failed_monarch_submission_waits_for_submitted_reads():
    """Wait for reads Monarch already started before raising a submission error."""

    class StartedRead:
        def __init__(self) -> None:
            self.future = asyncio.get_running_loop().create_future()

        def read_into(self, dest_byte_view: torch.Tensor):
            return self.future

    class FailedSubmission:
        def read_into(self, dest_byte_view: torch.Tensor):
            raise RuntimeError("submit failed")

    started = StartedRead()
    handles = {}
    for name, buf in (("a", started), ("b", FailedSubmission())):
        handles[name] = [
            RDMAWeightHandle(
                rdma_buffer=buf,
                tensor_slice=TensorSlice(
                    offsets=(0,),
                    coordinates=(0,),
                    global_shape=(4,),
                    local_shape=(4,),
                    mesh_shape=(1,),
                ),
                source_rank=0,
            )
        ]
    dest = {"a": torch.zeros(4), "b": torch.zeros(4)}

    pull = asyncio.create_task(DirectWeightSyncDest().pull(handles, dest))
    # Let the pull run until it blocks.
    for _ in range(10):
        await asyncio.sleep(0)
    assert not pull.done()
    started.future.set_result(None)
    with pytest.raises(RuntimeError, match="submit failed"):
        await pull


async def test_cancelled_pull_waits_for_started_reads():
    """Wait for started reads before propagating cancellation, since RDMA writes
    into the destination continue after cancel."""

    class StartedRead:
        def __init__(self) -> None:
            self.future = asyncio.get_running_loop().create_future()

        def read_into(self, dest_byte_view: torch.Tensor):
            return self.future

    started = StartedRead()
    handles = {
        "a": [
            RDMAWeightHandle(
                rdma_buffer=started,
                tensor_slice=TensorSlice(
                    offsets=(0,),
                    coordinates=(0,),
                    global_shape=(4,),
                    local_shape=(4,),
                    mesh_shape=(1,),
                ),
                source_rank=0,
            )
        ]
    }

    pull = asyncio.create_task(
        DirectWeightSyncDest().pull(handles, {"a": torch.zeros(4)})
    )
    # Let the pull run until it blocks.
    for _ in range(10):
        await asyncio.sleep(0)
    pull.cancel()
    for _ in range(10):
        await asyncio.sleep(0)
    assert not pull.done()
    assert not started.future.cancelled()
    started.future.set_result(None)
    with pytest.raises(asyncio.CancelledError):
        await pull
