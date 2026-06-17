# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Multi-tensor state_dict round-trip via TorchStore.

Mirrors the pattern an RL pipeline uses for trainer/generator weight
sync:

- A ``Trainer``-style actor on its own proc_mesh calls
  ``ts.put_state_dict``.
- A ``Generator``-style actor on a different proc_mesh calls
  ``ts.get_state_dict`` to pull the weights back into pre-allocated
  buffers.
- Many tensors per state_dict (LoRA adapter shape).

This exercises the per-tensor handshake-cache reuse in transports
that build a process group (gloo, xccl) — a single-tensor smoke
wouldn't catch a regression there.

Device autodetect: cuda > xpu > cpu. Run with::

    python example/torchstore_state_dict.py

On Intel XPU (xccl/oneCCL backed) most systems need libfabric on
the TCP provider for oneCCL handshake to succeed::

    unset FI_TCP_IFACE
    unset CCL_KVS_MODE
    export FI_PROVIDER=tcp
    export CCL_ATL_TRANSPORT=ofi
    export CCL_ATL_OFI_PROVIDER=tcp
    export CCL_PROCESS_LAUNCHER=hydra
    export CCL_ZE_DISABLE_PORT_CHECK=1
    export ZE_FLAT_DEVICE_HIERARCHY=FLAT
    python example/torchstore_state_dict.py
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
    """Pin one process to a single accelerator tile.

    Sets the env var the active backend honors —
    ``CUDA_VISIBLE_DEVICES`` for CUDA, ``ZE_AFFINITY_MASK`` for XPU.
    """
    if _ACCEL == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = devices
    elif _ACCEL == "xpu":
        os.environ["ZE_AFFINITY_MASK"] = devices


def _make_state_dict(seed: int) -> OrderedDict[str, torch.Tensor]:
    """Build a small multi-tensor state_dict (LoRA-shaped).

    Several distinct tensors so ``put_batch`` actually loops; a
    single-tensor put would short-circuit handshake reuse.
    """
    g = torch.Generator(device=_ACCEL).manual_seed(seed)
    sd: OrderedDict[str, torch.Tensor] = OrderedDict()
    for i in range(4):
        sd[f"layer.{i}.lora_A.weight"] = torch.randn(8, 16, generator=g, device=_ACCEL)
        sd[f"layer.{i}.lora_B.weight"] = torch.randn(16, 8, generator=g, device=_ACCEL)
    return sd


class Trainer(Actor):
    """Publishes a state_dict via ts.put_state_dict."""

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
    """Pulls the state_dict via ts.get_state_dict into pre-allocated buffers."""

    @endpoint
    async def fetch(self, ref_seed_for_shapes: int) -> dict:
        # Pre-allocate destination buffers with the right shapes;
        # mirrors how an RL Generator pulls into its own
        # model.state_dict().
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
    if _ACCEL == "cpu":
        log.warning(
            "no GPU detected; running on CPU still validates the actor "
            "wiring and gloo path."
        )

    # Actors live on their own meshes → separate OS processes from
    # the controller's storage volume. Required for xccl (oneCCL's
    # internal KVS is per-process); harmless for gloo/SHM.
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

    log.info("ts.initialize ...")
    await ts.initialize()
    log.info("initialize OK")

    trainer = trainer_mesh.spawn("trainer", Trainer)
    generator = gen_mesh.spawn("generator", Generator)

    seed = 17
    log.info("trainer.publish(seed=%d) — multi-tensor put_state_dict", seed)
    src_info = next(iter(await trainer.publish.call(seed)))[1]
    log.info("trainer wrote %d tensors", len(src_info))
    for k, v in src_info.items():
        log.info(
            "  %s: shape=%s device=%s checksum=%.6f",
            k,
            v["shape"],
            v["device"],
            v["checksum"],
        )

    log.info("generator.fetch — multi-tensor get_state_dict")
    got_info = next(iter(await generator.fetch.call(seed)))[1]
    log.info("generator got %d tensors", len(got_info))

    failures: list[str] = []
    for k, src in src_info.items():
        if k not in got_info:
            failures.append(f"missing: {k}")
            continue
        got = got_info[k]
        if got["shape"] != src["shape"]:
            failures.append(
                f"{k}: shape mismatch src={src['shape']} got={got['shape']}"
            )
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
