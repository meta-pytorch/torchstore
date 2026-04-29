# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# To run: torchrun --standalone --nnodes=1 --nproc-per-node=2 example/torchstore_spmd.py

import asyncio
import os

import torch
import torch.distributed as dist
import torchstore as ts


async def main() -> None:
    rank = int(os.environ["RANK"])
    dist.init_process_group("gloo")

    await ts.initialize_spmd(strategy=ts.LocalRankStrategy())

    try:
        if rank == 0:
            value = torch.tensor([123], dtype=torch.int64)
            await ts.put("demo", value)
            print(f"[rank=0] wrote {value}", flush=True)

        dist.barrier()  # wait for rank 0's write before reading
        value = await ts.get("demo")
        print(f"[rank={rank}] got {value}", flush=True)
    finally:
        dist.barrier()  # wait for readers before tearing down storage
        await ts.shutdown()
        dist.destroy_process_group()


if __name__ == "__main__":
    asyncio.run(main())
