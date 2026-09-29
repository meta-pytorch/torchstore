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
    """Basic put/get round-trip over the xccl transport.

    The source checksum is taken *before* the put, and the source tensor is
    checked for modification afterwards. Both matter: the bulk transfer moves
    data with a broadcast, so a root pointing at the receiver instead of the
    sender sends the volume's freshly allocated buffer to the client, which
    both stores uninitialized data and overwrites the caller's tensor in
    place. Checksumming the source after the put cannot see that -- the two
    sides then agree on the same garbage.
    """

    class Writer(Actor):
        def __init__(self):
            os.environ["LOCAL_RANK"] = str(current_rank().rank)

        @endpoint
        async def put(self, key: str) -> dict:
            t = torch.randn(4, 8, device="xpu")
            expected = t.clone()
            checksum = float(expected.float().sum().item())
            await ts.put(key, t)
            return {
                "checksum": checksum,
                "source_intact": bool(torch.equal(t, expected)),
                "source_checksum_after_put": float(t.float().sum().item()),
            }

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
        assert src["source_intact"], (
            "put modified the caller's tensor in place: checksum went from "
            f"{src['checksum']} to {src['source_checksum_after_put']}"
        )
        assert (
            abs(got["checksum"] - src["checksum"]) < 1e-3
        ), f"stored value differs from the source: {src['checksum']} != {got['checksum']}"
    finally:
        await ts.shutdown()


@requires_xpu
@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_xccl_mixed_object_and_tensor_batch():
    """A batch holding both an object and tensors must return both.

    Every key stored on one volume goes into a single get batch, so any state
    dict with a non-tensor entry produces a mixed batch. The object payload
    travels back with the response while the tensors travel in the broadcast
    buffer, and the two have to be recombined in request order.
    """

    class Trainer(Actor):
        def __init__(self):
            os.environ["LOCAL_RANK"] = str(current_rank().rank)

        @endpoint
        async def publish(self) -> dict:
            sd = {
                "layer.0.weight": torch.randn(16, 8, device="xpu"),
                "step": 7,
                "run_name": "mixed-batch",
                "layer.1.weight": torch.randn(16, 8, device="xpu"),
            }
            # Checksum before the put; see test_xccl_put_get for why after is
            # not good enough.
            expected = {
                "layer.0.weight": float(sd["layer.0.weight"].float().sum().item()),
                "step": sd["step"],
                "run_name": sd["run_name"],
                "layer.1.weight": float(sd["layer.1.weight"].float().sum().item()),
            }
            await ts.put_state_dict(sd, "mixed")
            return expected

    class Generator(Actor):
        def __init__(self):
            os.environ["LOCAL_RANK"] = str(current_rank().rank)

        @endpoint
        async def fetch(self) -> dict:
            sd = {
                "layer.0.weight": torch.zeros(16, 8, device="xpu"),
                "step": 0,
                "run_name": "",
                "layer.1.weight": torch.zeros(16, 8, device="xpu"),
            }
            sd = await ts.get_state_dict("mixed", user_state_dict=sd)
            return {
                "layer.0.weight": float(sd["layer.0.weight"].float().sum().item()),
                "step": sd["step"],
                "run_name": sd["run_name"],
                "layer.1.weight": float(sd["layer.1.weight"].float().sum().item()),
            }

    await ts.initialize(
        strategy=ts.LocalRankStrategy(TransportType.XCCL),
    )

    trainer = await spawn_actors(1, Trainer, "trainer")
    generator = await spawn_actors(1, Generator, "generator")

    try:
        src = next(iter(await trainer.publish.call()))[1]
        got = next(iter(await generator.fetch.call()))[1]

        assert got["step"] == 7, f"object entry lost: {got['step']}"
        assert got["run_name"] == "mixed-batch", f"object entry lost: {got['run_name']}"
        for k in ("layer.0.weight", "layer.1.weight"):
            assert abs(got[k] - src[k]) < 1e-3, f"{k}: {src[k]} != {got[k]}"
    finally:
        await ts.shutdown()


@requires_xpu
@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_xccl_zero_element_tensor():
    """A tensor with no elements must round-trip without a transfer.

    Nothing is broadcast for an empty tensor, and both sides have to reach
    that conclusion independently: if only the storage volume broadcasts, the
    collective never completes and the get blocks until the transfer timeout.
    """

    class Writer(Actor):
        def __init__(self):
            os.environ["LOCAL_RANK"] = str(current_rank().rank)

        @endpoint
        async def put(self, key: str) -> None:
            await ts.put(key, torch.zeros(0, 8, device="xpu"))

    class Reader(Actor):
        def __init__(self):
            os.environ["LOCAL_RANK"] = str(current_rank().rank)

        @endpoint
        async def get(self, key: str) -> dict:
            t = await ts.get(key)
            return {"shape": tuple(t.shape), "numel": int(t.numel())}

    await ts.initialize(
        strategy=ts.LocalRankStrategy(TransportType.XCCL),
    )

    writer = await spawn_actors(1, Writer, "writer")
    reader = await spawn_actors(1, Reader, "reader")

    try:
        key = "test_xccl_empty"
        await writer.put.call(key)
        got = next(iter(await reader.get.call(key)))[1]

        assert got["numel"] == 0
        assert got["shape"] == (0, 8), f"wrong shape: {got['shape']}"
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
                f"layer.{i}.weight": torch.randn(16, 8, device="xpu") for i in range(4)
            }
            # Checksum before the put; see test_xccl_put_get for why after is
            # not good enough.
            checksums = {k: float(v.float().sum().item()) for k, v in sd.items()}
            await ts.put_state_dict(sd, "weights")
            return checksums

    class Generator(Actor):
        def __init__(self):
            os.environ["LOCAL_RANK"] = str(current_rank().rank)

        @endpoint
        async def fetch(self) -> dict:
            sd = {
                f"layer.{i}.weight": torch.zeros(16, 8, device="xpu") for i in range(4)
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
