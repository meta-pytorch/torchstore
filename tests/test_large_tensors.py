# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import time
import uuid
from collections.abc import Iterator
from logging import getLogger

import pytest
import torch
import torchstore as ts
from monarch.actor import Actor, endpoint
from torchstore.carrier import CarrierClient, CarrierServer, close_all
from torchstore.logging import init_logging
from torchstore.monarch.rdma import monarch_rdma_transport_available
from torchstore.storage import StorageVolumeCore
from torchstore.strategy import HostStrategy
from torchstore.transport import TransportType
from torchstore.utils import spawn_actors

from .test_carrier_volumes import _Volumes
from .utils import main
from .volume_meshes import get_through as _get, put_through as _put

init_logging()
logger = getLogger(__name__)


@pytest.mark.asyncio
async def test_large_tensors():
    """Test basic put/get functionality for large tensors"""

    class LargeTensorActor(Actor):
        step_size: int = 100  # -> 400mb
        max_step: int = 600  # 4mb -> 2gb
        repeat_test: int = 1  # repeating test causes put to write inplace
        get_inplace: bool = False
        device: str = "cpu"

        def __init__(self, generate_benchmark=False) -> None:
            self.generate_benchmark = generate_benchmark
            init_logging()

        @endpoint
        async def put(self):
            dps = []
            for test_itr in range(self.repeat_test):
                print(f"{test_itr=}\n")
                for n in range(1, self.max_step, self.step_size):
                    shape = (1024, 1024 * n)
                    size_mbytes = (
                        math.prod(shape) * 4 // (1024 * 1024)
                    )  # float32 is 4 bytes, // mb
                    tensor = torch.randn(shape, dtype=torch.float32, device=self.device)

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
                        fp.write(f"{size_mbytes}, {delta}, {size_mbytes / delta}\n")

        @endpoint
        async def get(self):
            for test_itr in range(self.repeat_test):
                print(f"{test_itr=}\n")
                dps = []
                for n in range(1, self.max_step, self.step_size):
                    shape = (1024, 1024 * n)
                    size_mbytes = (
                        math.prod(shape) * 4 // (1024 * 1024)
                    )  # float32 is 4 bytes, // mb

                    logger.info(f"Get {n=} {size_mbytes=}")
                    inplace_tensor = (
                        None
                        if not self.get_inplace
                        else torch.randn(shape, dtype=torch.float32, device=self.device)
                    )
                    t = time.perf_counter()
                    try:
                        await ts.get(str(n), inplace_tensor)
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
                        fp.write(f"{size_mbytes}, {delta}, {size_mbytes / delta}\n")

    # controller code
    await ts.initialize(
        num_storage_volumes=1,
        strategy=HostStrategy(
            default_transport_type=(
                TransportType.MonarchRDMA
                if monarch_rdma_transport_available()
                else TransportType.Unset
            )
        ),
    )
    actor = await spawn_actors(1, LargeTensorActor, "large_tensor")
    try:
        await actor.put.call_one()
        await actor.get.call_one()
        # TODO: assert equal tensors from put/get
    finally:
        # TODO: Investigate monarch bug with proc_mesh.stop()
        # await actor._proc_mesh.stop()
        await ts.shutdown()


class _Echo:
    def echo(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor


@pytest.mark.asyncio
@pytest.mark.timeout(900)
async def test_carrier_echoes_a_tensor_over_4_gib() -> None:
    """Echo a tensor over 4 GiB through the carrier, whose buffer lengths are 64-bit."""
    tensor = torch.randint(0, 256, ((4 << 30) + 4097,), dtype=torch.uint8)
    server = await CarrierServer.serve(
        _Echo(), ["echo"], service="large", scope=f"large/{uuid.uuid4().hex}"
    )
    try:
        start = time.perf_counter()
        echoed = await CarrierClient(server.address).call("echo", tensor)
        logger.info(
            f"Carrier echo of {tensor.nbytes} bytes took {time.perf_counter() - start:.2f} s"
        )
        assert echoed.shape == tensor.shape and torch.equal(echoed, tensor)
    finally:
        await close_all()


@pytest.fixture
def volumes() -> Iterator[_Volumes]:
    # A close in teardown reports its failure beside the case's own instead of replacing it.
    volumes = _Volumes()
    yield volumes
    volumes.close()


@pytest.mark.asyncio
@pytest.mark.timeout(900)
async def test_carrier_volume_proxy_moves_a_tensor_over_4_gib(
    volumes: _Volumes,
) -> None:
    """Put and get a tensor over 4 GiB in band through a carrier volume's proxy, over MonarchRPC."""
    tensor = torch.randint(0, 256, ((4 << 30) + 4097,), dtype=torch.uint8)
    core, _, record = await volumes.serve(
        StorageVolumeCore(f"large-{uuid.uuid4().hex}")
    )
    ref = volumes.ref(record, core.volume_id, TransportType.MonarchRPC)
    start = time.perf_counter()
    await _put(ref, "large", tensor)
    fetched = await _get(ref, "large")
    logger.info(
        f"Carrier volume put and get of {tensor.nbytes} bytes took"
        f" {time.perf_counter() - start:.2f} s"
    )
    assert fetched.shape == tensor.shape and torch.equal(fetched, tensor)


if __name__ == "__main__":
    main(__file__)
