# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""XCCL transport tests on Intel XPU.

XCCL requires client and storage volume in separate OS processes
(oneCCL's per-process KVS collides otherwise), so these tests use
spawn_procs to create proper multi-process actor meshes.

Skipped when XPU hardware is not available.
"""

import os

import pytest
import torch
import torchstore as ts
from monarch.actor import Actor, current_rank, endpoint
from torchstore.transport import TransportType
from torchstore.utils import spawn_actors

requires_xpu = pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available()),
    reason="No XPU device available",
)


@requires_xpu
def test_xccl_disabled_by_env(monkeypatch):
    from torchstore.transport import xccl

    monkeypatch.setattr(xccl, "TORCHSTORE_XCCL_ENABLED", False)
    assert not xccl.xccl_available()


@requires_xpu
@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_xccl_put_get():
    """Basic put/get round-trip over the xccl transport."""

    class Writer(Actor):
        def __init__(self):
            os.environ["LOCAL_RANK"] = str(current_rank().rank)

        @endpoint
        async def put(self, key: str) -> dict:
            t = torch.randn(4, 8, device="xpu")
            await ts.put(key, t)
            return {"checksum": float(t.float().sum().item())}

    class Reader(Actor):
        def __init__(self):
            os.environ["LOCAL_RANK"] = str(current_rank().rank)

        @endpoint
        async def get(self, key: str) -> dict:
            t = await ts.get(key)
            return {
                "checksum": float(t.float().sum().item()),
                "device": str(t.device),
            }

    await ts.initialize(
        strategy=ts.LocalRankStrategy(TransportType.XCCL),
    )

    writer = await spawn_actors(1, Writer, "writer")
    reader = await spawn_actors(1, Reader, "reader")

    try:
        key = "test_xccl_tensor"
        src = next(iter(await writer.put.call(key)))[1]
        got = next(iter(await reader.get.call(key)))[1]

        assert "xpu" in got["device"]
        assert abs(got["checksum"] - src["checksum"]) < 1e-3
    finally:
        await ts.shutdown()


@requires_xpu
@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_xccl_state_dict():
    """Multi-tensor state_dict round-trip over xccl (LoRA-shaped)."""

    class Trainer(Actor):
        def __init__(self):
            os.environ["LOCAL_RANK"] = str(current_rank().rank)

        @endpoint
        async def publish(self) -> dict:
            sd = {
                f"layer.{i}.weight": torch.randn(16, 8, device="xpu")
                for i in range(4)
            }
            await ts.put_state_dict(sd, "weights")
            return {k: float(v.float().sum().item()) for k, v in sd.items()}

    class Generator(Actor):
        def __init__(self):
            os.environ["LOCAL_RANK"] = str(current_rank().rank)

        @endpoint
        async def fetch(self) -> dict:
            sd = {
                f"layer.{i}.weight": torch.zeros(16, 8, device="xpu")
                for i in range(4)
            }
            sd = await ts.get_state_dict("weights", user_state_dict=sd, strict=True)
            return {k: float(v.float().sum().item()) for k, v in sd.items()}

    await ts.initialize(
        strategy=ts.LocalRankStrategy(TransportType.XCCL),
    )

    trainer = await spawn_actors(1, Trainer, "trainer")
    generator = await spawn_actors(1, Generator, "generator")

    try:
        src = next(iter(await trainer.publish.call()))[1]
        got = next(iter(await generator.fetch.call()))[1]

        assert set(src.keys()) == set(got.keys())
        for k in src:
            assert abs(got[k] - src[k]) < 1e-3, f"{k}: {src[k]} != {got[k]}"
    finally:
        await ts.shutdown()
