# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import time
from logging import getLogger

import pytest
import torch

import torchstore as ts
from monarch.actor import Actor, endpoint
from torchstore.logging import init_logging
from torchstore.utils import spawn_actors

from .utils import main

init_logging()
logger = getLogger(__name__)


@pytest.mark.asyncio
async def test_large_tensors():
    """Test basic put/get functionality for large tensors"""

    class LargeTensorActor(Actor):
        step_size: int = 100  # -> 400mb
        max_step: int = 600  # 4mb -> 2gb

        def __init__(self, generate_benchmark=False) -> None:
            self.generate_benchmark = generate_benchmark
            init_logging()

        @endpoint
        async def put(self):
            dps = []
            for n in range(1, self.max_step, self.step_size):
                shape = (1024, 1024 * n)
                size_mbytes = (
                    math.prod(shape) * 4 // (1024 * 1024)
                )  # float32 is 4 bytes, // mb
                tensor = torch.randn(shape, dtype=torch.float32)

                logger.info(f"Put {n=} {size_mbytes=}")
                t = time.perf_counter()
                try:
                    await ts.put(str(n), tensor)
                except Exception as e:
                    logger.exception(f"Test failed with {size_mbytes=}")
                    raise e

                delta = time.perf_counter() - t
                dps.append((size_mbytes, delta))
                logger.info(f"Took {delta} seconds to put")

            if self.generate_benchmark:
                with open("put_benchmark.csv", "w") as fp:
                    fp.write("size_mbytes, delta\n")
                    for size_mbytes, delta in dps:
                        fp.write(f"{size_mbytes}, {delta}, {size_mbytes/delta}\n")

        @endpoint
        async def get(self):
            dps = []
            for n in range(1, self.max_step, self.step_size):
                shape = (1024, 1024 * n)
                size_mbytes = (
                    math.prod(shape) * 4 // (1024 * 1024)
                )  # float32 is 4 bytes, // mb

                logger.info(f"Get {n=} {size_mbytes=}")
                t = time.perf_counter()
                try:
                    await ts.get(str(n))
                except Exception as e:
                    logger.exception(f"Test failed with {size_mbytes=}")
                    raise e

                delta = time.perf_counter() - t
                dps.append((size_mbytes, delta))
                logger.info(f"Took {delta} seconds to fetch")

            if self.generate_benchmark:
                with open("get_benchmark.csv", "w") as fp:
                    fp.write("size_mbytes, delta\n")
                    for size_mbytes, delta in dps:
                        fp.write(f"{size_mbytes}, {delta}, {size_mbytes/delta}\n")

    # controller code
    await ts.initialize()
    actor = await spawn_actors(1, LargeTensorActor, "large_tensor")
    try:
        await actor.put.call_one()
        await actor.get.call_one()
        # TODO: assert equal tensors from put/get
    finally:
        await actor._proc_mesh.stop()
        await ts.shutdown()


if __name__ == "__main__":
    main(__file__)
