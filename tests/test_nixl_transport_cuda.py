# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""NIXL gets into CUDA destinations over real NIXL (single-process loopback).

The fake agent in test_nixl_transport.py cannot show writes landing in memory a
stale registration no longer covers.
"""

import asyncio
import importlib.util
import os
import pickle
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from torchstore.transport.buffers import TransportContext
from torchstore.transport.nixl import NixlAgentCache, NixlTransportBuffer
from torchstore.transport.types import Request


@pytest.mark.skipif(
    importlib.util.find_spec("nixl") is None or not torch.cuda.is_available(),
    reason="requires NIXL and CUDA",
)
def test_get_into_reallocated_cuda_destination():
    """Get into a destination freed and reallocated between gets, as FSDP does.

    Runs in a fresh process: expandable segments must be enabled before CUDA
    first allocates, and they unmap freed memory, so a reallocation at the same
    address is backed by new pages that an earlier registration does not cover.
    """
    _run_with_expandable_segments("reallocated")


@pytest.mark.skipif(
    importlib.util.find_spec("nixl") is None or not torch.cuda.is_available(),
    reason="requires NIXL and CUDA",
)
def test_get_into_cuda_destination_spanning_segment_chunks():
    """Get into a destination spanning two separately mapped expandable-segment
    chunks, which UCX's whole-allocation registration only partly covers."""
    _run_with_expandable_segments("spanning")


@pytest.mark.skipif(
    importlib.util.find_spec("nixl") is None or not torch.cuda.is_available(),
    reason="requires NIXL and CUDA",
)
def test_freed_cuda_destination_returns_to_device():
    """CUDA memory freed after a get is released to the device, not kept
    registered (and so pinned) by UCX's registration cache."""
    _run_with_expandable_segments("released")


def _run_with_expandable_segments(case: str) -> None:
    env = {
        **os.environ,
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        "TORCHSTORE_NIXL_ENABLED": "1",
        # Import this checkout's torchstore, not an installed one.
        "PYTHONPATH": os.pathsep.join(
            [os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
            + os.environ.get("PYTHONPATH", "").split(os.pathsep)
        ),
    }
    # These would override the agent's own UCX config.
    env.pop("UCX_CUDA_COPY_REG_WHOLE_ALLOC", None)
    env.pop("UCX_RCACHE_ENABLE", None)
    result = subprocess.run(
        [sys.executable, __file__, case],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


async def _get(
    client_context: TransportContext,
    volume_context: TransportContext,
    stored: torch.Tensor,
    destination: torch.Tensor,
) -> None:
    ref = SimpleNamespace(volume_id="volume", transport_context=client_context)
    request = Request.from_tensor("w", destination)
    client_buffer = NixlTransportBuffer(ref)
    try:
        await client_buffer._pre_get_hook([request])
        volume_buffer = pickle.loads(pickle.dumps(client_buffer))
        await volume_buffer.handle_get_request(
            volume_context, [(request.meta_only(), stored)]
        )
    finally:
        await client_buffer.drop()


async def _get_into_reallocated_cuda_destination() -> None:
    client_context = TransportContext()
    volume_context = TransportContext()
    stored = torch.zeros(1 << 20).pin_memory()
    destination = torch.empty_like(stored, device="cuda")
    storage = destination.untyped_storage()
    addresses = []
    for value in range(1, 4):
        stored.fill_(value)
        storage.resize_(destination.nbytes)
        addresses.append(storage.data_ptr())
        await _get(client_context, volume_context, stored, destination)
        assert torch.all(destination == value).item(), value
        cache = client_context.get(NixlAgentCache)
        assert cache._registrations == {} and cache._leases == {}
        storage.resize_(0)
        torch.cuda.empty_cache()
    assert len(set(addresses)) < len(addresses), "no address was reused"


async def _get_into_cuda_destination_spanning_segment_chunks() -> None:
    # Expandable segments map the large pool in 20 MiB chunks; fill most of the
    # first so the destination continues into the next one.
    pad = torch.empty(19 << 20, dtype=torch.uint8, device="cuda")
    destination = torch.zeros(4 << 20, dtype=torch.uint8, device="cuda")
    assert destination.data_ptr() == pad.data_ptr() + pad.nbytes
    stored = torch.ones_like(destination, device="cpu").pin_memory()
    await _get(TransportContext(), TransportContext(), stored, destination)
    assert torch.all(destination == 1).item()


async def _freed_cuda_destination_returns_to_device() -> None:
    client_context = TransportContext()
    volume_context = TransportContext()
    stored = torch.ones(64 << 20, dtype=torch.uint8)
    # Warm up UCX and CUDA (including lazily loaded kernels) so their one-time
    # device allocations are not counted.
    warmup = torch.zeros(1 << 20, dtype=torch.uint8, device="cuda")
    await _get(client_context, volume_context, stored[: warmup.numel()], warmup)
    assert torch.all(warmup == 1).item()
    del warmup
    torch.cuda.empty_cache()
    free_before, _ = torch.cuda.mem_get_info()
    destination = torch.zeros_like(stored, device="cuda")
    await _get(client_context, volume_context, stored, destination)
    assert torch.all(destination == 1).item()
    del destination
    torch.cuda.empty_cache()
    free_after, _ = torch.cuda.mem_get_info()
    assert free_after > free_before - stored.nbytes // 2, (free_before, free_after)


if __name__ == "__main__":
    asyncio.run(
        {
            "reallocated": _get_into_reallocated_cuda_destination,
            "spanning": _get_into_cuda_destination_spanning_segment_chunks,
            "released": _freed_cuda_destination_returns_to_device,
        }[sys.argv[1]]()
    )
    # Skip interpreter teardown; NIXL can crash on exit after a passing run.
    os._exit(0)
