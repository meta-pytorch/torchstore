# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Torch transport transfers of CUDA views over real NIXL with expandable
segments (single-process loopback)."""

import asyncio
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("torch.distributed._transport")
from torch.distributed._transport import available_transports

from torchstore.transport.buffers import TransportContext
from torchstore.transport.types import Request


@pytest.mark.skipif(
    "nixl" not in available_transports() or not torch.cuda.is_available(),
    reason="requires the NIXL torch transport and CUDA",
)
@pytest.mark.parametrize("op", ["get", "put"])
def test_view_in_second_segment_chunk(op, monkeypatch):
    """Transfer a view lying in the second expandable-segment chunk of its
    storage, which UCX's whole-allocation registration does not cover.

    Runs in a fresh process: expandable segments must be enabled before CUDA
    first allocates.
    """
    from torchstore.transport import torch_distributed as td_transport

    monkeypatch.setattr(td_transport, "ENV_TORCHSTORE_TORCH_TRANSPORT_BACKEND", "nixl")
    td_transport.torch_transport_available.cache_clear()
    try:
        if not td_transport.torch_transport_available():
            pytest.skip("nixl transport is unusable")
    finally:
        td_transport.torch_transport_available.cache_clear()
    env = {
        **os.environ,
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        "TORCHSTORE_TORCH_TRANSPORT_BACKEND": "nixl",
        # Import this checkout's torchstore and tests, not installed ones.
        "PYTHONPATH": os.pathsep.join(
            [os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
            + os.environ.get("PYTHONPATH", "").split(os.pathsep)
        ),
    }
    # These would override the transport's own UCX config.
    env.pop("UCX_CUDA_COPY_REG_WHOLE_ALLOC", None)
    env.pop("UCX_RCACHE_ENABLE", None)
    result = subprocess.run(
        [sys.executable, __file__, op],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


async def _transfer_view_in_second_segment_chunk(op: str) -> None:
    from torchstore.transport import torch_distributed as td_transport

    import tests.test_torch_distributed_transport as t

    assert td_transport.torch_transport_available()
    # Real backend setup can be slow on a loaded host.
    t._BOUND_S = 60
    ref = SimpleNamespace(
        volume_id="volume", volume=t.FakeVolume(), transport_context=TransportContext()
    )
    # Expandable segments map the large pool in 20 MiB chunks; fill most of the
    # first so storage continues into the next one, which holds all of view.
    pad = torch.empty(19 << 20, dtype=torch.uint8, device="cuda")
    storage = torch.zeros(8 << 20, dtype=torch.uint8, device="cuda")
    assert storage.data_ptr() == pad.data_ptr() + pad.nbytes
    view = storage[2 << 20 : 6 << 20]
    try:
        if op == "get":
            stored = torch.ones(view.numel(), dtype=torch.uint8)
            await t._put(ref, [Request.from_tensor("w", stored)])
            await t._get(ref, [Request.from_tensor("w", view)])
            assert torch.all(view == 1).item()
            assert storage[: 2 << 20].sum().item() == 0
            assert storage[6 << 20 :].sum().item() == 0
        else:
            view.fill_(7)
            await t._put(ref, [Request.from_tensor("w", view)])
            (result,) = await t._get(ref, [Request(key="w")])
            assert torch.all(result == 7).item()
    finally:
        ref.transport_context.clear()
        ref.volume.store.reset()


if __name__ == "__main__":
    asyncio.run(_transfer_view_in_second_segment_chunk(sys.argv[1]))
    # Skip interpreter teardown; NIXL can crash on exit after a passing run.
    os._exit(0)
