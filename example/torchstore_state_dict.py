# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Multi-tensor state_dict round-trip via TorchStore.

Mirrors the RL trainer/generator weight sync pattern:
- A Trainer actor calls ts.put_state_dict
- A Generator actor calls ts.get_state_dict into pre-allocated buffers
- Multiple tensors per state_dict (LoRA adapter shape)

Device autodetect: cuda > xpu > cpu.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections import OrderedDict
from functools import partial

import torch
import torchstore as ts
from monarch.actor import Actor, endpoint, shutdown_context, this_host

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("state_dict_example")


def _accelerator() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


_ACCEL = _accelerator()
KEY = "policy_state_dict"


def _set_visible_devices(devices: str) -> None:
    if _ACCEL == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = devices
    elif _ACCEL == "xpu":
        os.environ["ZE_AFFINITY_MASK"] = devices


def _make_state_dict(seed: int) -> OrderedDict[str, torch.Tensor]:
    """Build a small multi-tensor state_dict (LoRA-shaped)."""
    g = torch.Generator(device=_ACCEL).manual_seed(seed)
    sd: OrderedDict[str, torch.Tensor] = OrderedDict()
    for i in range(4):
        sd[f"layer.{i}.lora_A.weight"] = torch.randn(8, 16, generator=g, device=_ACCEL)
        sd[f"layer.{i}.lora_B.weight"] = torch.randn(16, 8, generator=g, device=_ACCEL)
    return sd


class Trainer(Actor):
    @endpoint
    async def publish(self, seed: int) -> dict:
        sd = _make_state_dict(seed)
        await ts.put_state_dict(sd, KEY)
        return {
            k: {
                "shape": tuple(v.shape),
                "device": str(v.device),
                "checksum": float(v.float().sum().item()),
            }
            for k, v in sd.items()
        }


class Generator(Actor):
    @endpoint
    async def fetch(self, ref_seed_for_shapes: int) -> dict:
        sd = _make_state_dict(ref_seed_for_shapes)
        for v in sd.values():
            v.zero_()

        sd = await ts.get_state_dict(KEY, user_state_dict=sd, strict=True)
        return {
            k: {
                "shape": tuple(v.shape),
                "device": str(v.device),
                "checksum": float(v.float().sum().item()),
            }
            for k, v in sd.items()
        }


async def main() -> int:
    log.info("accelerator: %s", _ACCEL)

    trainer_mesh = this_host().spawn_procs(
        name="trainer",
        per_host={"gpus": 1},
        bootstrap=partial(_set_visible_devices, "0"),
    )
    gen_mesh = this_host().spawn_procs(
        name="generator",
        per_host={"gpus": 1},
        bootstrap=partial(_set_visible_devices, "1"),
    )

    await ts.initialize()

    trainer = trainer_mesh.spawn("trainer", Trainer)
    generator = gen_mesh.spawn("generator", Generator)

    seed = 17
    log.info("trainer.publish(seed=%d)", seed)
    src_info = next(iter(await trainer.publish.call(seed)))[1]
    log.info("trainer wrote %d tensors", len(src_info))

    log.info("generator.fetch")
    got_info = next(iter(await generator.fetch.call(seed)))[1]
    log.info("generator got %d tensors", len(got_info))

    failures: list[str] = []
    for k, src in src_info.items():
        if k not in got_info:
            failures.append(f"missing: {k}")
            continue
        got = got_info[k]
        if got["shape"] != src["shape"]:
            failures.append(f"{k}: shape mismatch src={src['shape']} got={got['shape']}")
        if abs(got["checksum"] - src["checksum"]) > 1e-3:
            failures.append(
                f"{k}: checksum mismatch src={src['checksum']:.6f} "
                f"got={got['checksum']:.6f}"
            )

    if failures:
        for f in failures:
            log.error("  %s", f)
        log.error("STATE-DICT FAIL")
        await ts.shutdown()
        return 1

    log.info("STATE-DICT PASS — %d tensors round-tripped", len(src_info))
    await ts.shutdown()
    return 0


if __name__ == "__main__":
    import sys

    rc = asyncio.run(main())
    shutdown_context().get(timeout=2.0)
    sys.exit(rc)
