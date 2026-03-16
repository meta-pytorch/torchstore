# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Correctness tests for direct_weight_sync.py.

Uses a MockRDMABuffer that copies source bytes into the dest byte view,
simulating what real RDMA does. No GPU or RDMA infrastructure needed.
"""

import pytest
import torch

from torchstore.direct_weight_sync import (
    _to_byte_view,
    DirectWeightSyncDest,
    DirectWeightSyncSource,
    RDMAWeightHandle,
)
from torchstore.transport.types import TensorSlice

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
        buf = MockRDMABuffer(_to_byte_view(shard_data))
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
        buf = MockRDMABuffer(_to_byte_view(data))
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
    mock_buf = MockRDMABuffer(_to_byte_view(staging_buf))
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
