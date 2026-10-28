# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Direct weight sync over real NIXL into CUDA destinations (single-process loopback).

The fake agent in test_nixl_transport.py copies with stream-ordered torch ops,
so it cannot show RDMA writes racing queued kernels or stale registrations.
"""

import importlib.util
import pickle

import pytest
import torch
from torchstore.direct_weight_sync import (
    DirectWeightSyncDest,
    DirectWeightSyncSource,
    group_handles,
)
from torchstore.transport import nixl as nixl_transport

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("nixl") is None or not torch.cuda.is_available(),
    reason="requires NIXL and CUDA",
)


@pytest.fixture(autouse=True)
def enable_nixl(monkeypatch):
    monkeypatch.setattr(nixl_transport, "ENV_TORCHSTORE_NIXL_ENABLED", True)
    nixl_transport.nixl_available.cache_clear()
    yield
    nixl_transport.nixl_available.cache_clear()


@pytest.mark.asyncio
async def test_pull_into_reallocated_destination():
    """Pull into a destination freed and reallocated between pulls, as FSDP does.

    Expandable segments unmap freed memory, so the reallocation at the same
    address is backed by new pages that a stale registration does not cover.
    """
    prior = torch._C._accelerator_getAllocatorSettings()
    torch._C._accelerator_setAllocatorSettings("expandable_segments:True")
    weight = torch.zeros(1 << 20).pin_memory()
    source = DirectWeightSyncSource()
    handles = group_handles(
        [pickle.loads(pickle.dumps(source.register({"w": weight}, rank=0)))]
    )
    dest = DirectWeightSyncDest()
    dest_weight = torch.empty_like(weight, device="cuda")
    storage = dest_weight.untyped_storage()
    addresses = []
    try:
        for value in range(1, 4):
            weight.fill_(value)
            storage.resize_(dest_weight.nbytes)
            addresses.append(storage.data_ptr())
            await dest.pull(handles, {"w": dest_weight})
            assert torch.all(dest_weight == value).item(), value
            storage.resize_(0)
            torch.cuda.empty_cache()
        assert len(set(addresses)) < len(addresses), "no address was reused"
    finally:
        await source.cleanup()
        expandable = "expandable_segments:True" in prior
        torch._C._accelerator_setAllocatorSettings(f"expandable_segments:{expandable}")


@pytest.mark.asyncio
async def test_pull_waits_for_queued_destination_kernels():
    """Pull after a kernel still queued on the destination, which RDMA must not race."""
    weight = torch.full((1 << 20,), 7.0).pin_memory()
    source = DirectWeightSyncSource()
    handles = group_handles(
        [pickle.loads(pickle.dumps(source.register({"w": weight}, rank=0)))]
    )
    dest = DirectWeightSyncDest()
    dest_weight = torch.empty_like(weight, device="cuda")
    x = torch.randn(4096, 4096, device="cuda")
    try:
        for _ in range(3):
            # Matmuls keep the stream busy so the fill is still queued at pull.
            y = x
            for _ in range(8):
                y = y @ x
            dest_weight.fill_(-1.0).add_(y[0, 0] * 0)
            await dest.pull(handles, {"w": dest_weight})
            assert torch.all(dest_weight == 7.0).item()
    finally:
        await source.cleanup()
