# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Put and get AWS Neuron device tensors through TorchStore's NIXL transport.

Needs a Neuron device, torch-neuronx, and NIXL built with the LIBFABRIC plugin:

    TORCHSTORE_NIXL_ENABLED=1 TORCHSTORE_NIXL_BACKEND=LIBFABRIC \\
        python examples/neuron_nixl_roundtrip.py

The storage volume runs in its own process. A put READs the client's Neuron HBM into
the volume; a get WRITEs the volume's copy into a pre-allocated Neuron tensor. Each
size is checked byte for byte and timed.
"""

import asyncio
import os
import sys
import time

import torch
import torchstore as ts
from torchstore.strategy import ControllerStorageVolumes
from torchstore.transport import TransportType

SIZES_MIB = [int(s) for s in os.environ.get("SIZES_MIB", "1,64,1024").split(",")]
REPEATS = int(os.environ.get("REPEATS", "3"))


async def main() -> int:
    import torch_neuronx

    device = torch.device("neuron", torch_neuronx.current_device())
    strategy = ControllerStorageVolumes(default_transport_type=TransportType.NIXL)
    await ts.initialize(num_storage_volumes=1, strategy=strategy)
    failures = 0
    try:
        for mib in SIZES_MIB:
            n = (mib << 20) // 4
            src = torch.arange(n, dtype=torch.float32, device=device)
            dst = torch.zeros(n, dtype=torch.float32, device=device)
            torch_neuronx.synchronize()
            put_s, get_s = [], []
            for i in range(REPEATS):
                t0 = time.perf_counter()
                await ts.put(f"neuron-{mib}", src)
                put_s.append(time.perf_counter() - t0)
                t0 = time.perf_counter()
                await ts.get(f"neuron-{mib}", inplace_tensor=dst)
                torch_neuronx.synchronize()
                get_s.append(time.perf_counter() - t0)
            ok = torch.equal(src.cpu(), dst.cpu())
            failures += not ok
            best_put, best_get = min(put_s), min(get_s)
            print(
                f"RESULT {mib:>5} MiB: put {best_put * 1e3:8.1f} ms "
                f"({mib / 1024 / best_put:6.2f} GB/s), get {best_get * 1e3:8.1f} ms "
                f"({mib / 1024 / best_get:6.2f} GB/s), data match {ok}",
                flush=True,
            )
    finally:
        await ts.shutdown()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
