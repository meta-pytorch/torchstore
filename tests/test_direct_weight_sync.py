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
from unittest.mock import AsyncMock, Mock

import pytest
import torch

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


@pytest.fixture
def registered_buffers(monkeypatch):
    buffers = []

    def register(source_bytes):
        buffer = MockRDMABuffer(source_bytes)
        buffer.drop = AsyncMock()
        buffers.append(buffer)
        return buffer

    monkeypatch.setattr("monarch.rdma.RDMABuffer", register)
    return buffers


@pytest.mark.parametrize("transfer_dtype", [None, torch.bfloat16])
async def test_register_cleanup_is_idempotent(registered_buffers, transfer_dtype):
    source = DirectWeightSyncSource()
    weights = {"weight": torch.ones(4), "bias": torch.ones(2)}
    handles = source.register(weights, rank=3, transfer_dtype=transfer_dtype)

    assert set(handles) == set(weights)
    assert all(handle.source_rank == 3 for handle in handles.values())
    for (name, tensor), buffer in zip(weights.items(), registered_buffers, strict=True):
        expected = tensor.to(transfer_dtype) if transfer_dtype else tensor
        assert torch.equal(buffer._source, to_byte_view(expected))
        assert handles[name].rdma_buffer is buffer

    await source.cleanup()
    await source.cleanup()

    for buffer in registered_buffers:
        buffer.drop.assert_awaited_once()
    assert source.refresh() == 0


async def test_cleanup_after_partial_registration(registered_buffers):
    source = DirectWeightSyncSource()
    weights = {"valid": torch.ones(4), "invalid": torch.ones(2, 3).T}

    with pytest.raises(AssertionError, match="Expected contiguous tensor"):
        source.register(weights, rank=0)

    assert len(registered_buffers) == 1
    await source.cleanup()
    registered_buffers[0].drop.assert_awaited_once()


async def test_cleanup_after_native_registration_failure(monkeypatch):
    source = DirectWeightSyncSource()
    buffer = MockRDMABuffer(to_byte_view(torch.ones(4)))
    buffer.drop = AsyncMock()
    monkeypatch.setattr(
        "monarch.rdma.RDMABuffer",
        Mock(side_effect=[buffer, RuntimeError("native registration failed")]),
    )

    with pytest.raises(RuntimeError, match="native registration failed"):
        source.register({"first": torch.ones(4), "second": torch.ones(4)}, rank=0)

    await source.cleanup()
    buffer.drop.assert_awaited_once()


@pytest.mark.parametrize("transfer_dtype", [None, torch.bfloat16])
async def test_cleanup_owns_repeated_parameter_names(
    registered_buffers, transfer_dtype
):
    source = DirectWeightSyncSource()
    first = torch.ones(4)
    second = torch.zeros(4)
    source.register({"weight": first}, rank=0, transfer_dtype=transfer_dtype)
    source.register({"weight": second}, rank=0, transfer_dtype=transfer_dtype)

    first.fill_(3)
    second.fill_(7)
    assert source.refresh() == (2 if transfer_dtype else 0)
    for tensor, buffer in zip([first, second], registered_buffers, strict=True):
        expected = tensor.to(transfer_dtype) if transfer_dtype else tensor
        assert torch.equal(buffer._source, to_byte_view(expected))

    await source.cleanup()

    assert len(registered_buffers) == 2
    for buffer in registered_buffers:
        buffer.drop.assert_awaited_once()


@pytest.mark.parametrize("transfer_dtype", [None, torch.bfloat16])
async def test_cleanup_continues_and_retries_failed_buffers(
    registered_buffers, transfer_dtype
):
    source = DirectWeightSyncSource()
    source.register(
        {"first": torch.ones(4), "second": torch.ones(4)},
        rank=0,
        transfer_dtype=transfer_dtype,
    )
    first, second = registered_buffers
    failure = RuntimeError("native deregistration failed")
    first.drop.side_effect = [failure, None]

    with pytest.raises(RuntimeError):
        await source.cleanup()

    first.drop.assert_awaited_once()
    second.drop.assert_awaited_once()
    if transfer_dtype:
        assert source.refresh() == 1

    await source.cleanup()
    await source.cleanup()

    assert first.drop.await_count == 2
    second.drop.assert_awaited_once()
    assert source.refresh() == 0


async def test_concurrent_cleanup_does_not_drop_twice(registered_buffers):
    source = DirectWeightSyncSource()
    source.register({"weight": torch.ones(4)}, rank=0)
    started = asyncio.Event()
    release = asyncio.Event()

    async def drop():
        started.set()
        await release.wait()

    registered_buffers[0].drop.side_effect = drop
    first = asyncio.create_task(source.cleanup())
    await started.wait()
    second = asyncio.create_task(source.cleanup())
    release.set()
    await asyncio.gather(first, second)

    registered_buffers[0].drop.assert_awaited_once()


async def test_cleanup_skips_empty_tensor_registrations(registered_buffers):
    source = DirectWeightSyncSource()
    handles = source.register({"empty": torch.empty(0)}, rank=0)

    assert handles["empty"].rdma_buffer is None
    assert registered_buffers == []
    await source.cleanup()
    await source.cleanup()


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
    source._staging[0] = (staging_buf, source_tensor)

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
    source._staging[0] = (staging_buf, original)

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
    assert 0 in source._staging

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
