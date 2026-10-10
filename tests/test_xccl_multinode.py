# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Cross-node XCCL transfer tests on Intel XPU (2 nodes).

Cross-node broadcasts take oneCCL's scale-out path, which the single-node tests
in test_xccl_xpu.py never reach. Each transfer is checked byte by byte.

Launch one pytest per rank, each test in fresh processes (oneCCL keeps state
from a process's first communicator). Every rank needs RANK (or PALS_RANKID),
TORCHSTORE_TEST_HEAD (a node-0 hostname), TORCHSTORE_TEST_WORLD=8 and
ZE_AFFINITY_MASK: 8 ranks, 4 per node; node 0 sees tiles 0,1,2,3 and each node-1
rank sees one tile.
"""

import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from torchstore.transport.xccl import (
    _wait_with_timeout,
    _warm_up_communicator,
    _xccl_factory,
    _xpu_device,
    xccl_available,
)

NBYTES = 64 << 20  # large enough for the bulk scale-out path
_rank = os.environ.get("RANK", os.environ.get("PALS_RANKID"))
_head = os.environ.get("TORCHSTORE_TEST_HEAD")

requires_8_ranks = pytest.mark.skipif(
    not (
        xccl_available()
        and _rank is not None
        and _head is not None
        and os.environ.get("TORCHSTORE_TEST_WORLD") == "8"
    ),
    reason="needs 2 XPU nodes x 4 ranks (see module docstring)",
)


def _pattern(src: int, rnd: int, device: torch.device) -> torch.Tensor:
    idx = torch.arange(NBYTES, dtype=torch.int64, device=device)
    return ((idx + 7 * src + 31 * rnd) % 251).to(torch.uint8)


def _pair_group(port: int, name: str, is_volume: bool, device: torch.device):
    """A torchstore 2-rank XCCL group: client rank 0, volume rank 1."""
    store = dist.TCPStore(
        _head, port, 2, is_master=is_volume, timeout=timedelta(seconds=300)
    )
    pg = _xccl_factory(
        store, 1 if is_volume else 0, 2, timedelta(seconds=300), device, name
    )
    _warm_up_communicator(pg, device, name)
    return pg


def _pull(pg, src: int, rnd: int, is_volume: bool, device: torch.device) -> bool:
    """Volume broadcasts a known pattern; returns True if the client got it."""
    buf = (
        _pattern(src, rnd, device)
        if is_volume
        else torch.zeros(NBYTES, dtype=torch.uint8, device=device)
    )
    opts = dist.BroadcastOptions()
    opts.rootRank = 1
    opts.rootTensor = 0
    _wait_with_timeout(pg.broadcast([buf], opts), "broadcast", None)
    torch.xpu.synchronize(device)
    return is_volume or torch.equal(buf, _pattern(src, rnd, device))


def _run_pull(port_base: int, *, intra_node_collective: bool) -> None:
    rank = int(_rank)
    local, is_volume = rank % 4, rank < 4
    # Volume i on tile i; a client sees one tile, which is xpu:0.
    os.environ["LOCAL_RANK"] = str(local) if is_volume else "0"
    device = _xpu_device()
    torch.xpu.set_device(device)

    if is_volume and intra_node_collective:
        # What FSDP does in the trainer process that hosts a storage volume.
        dist.init_process_group(
            "xccl",
            init_method=f"tcp://{_head}:{port_base + 99}",
            rank=local,
            world_size=4,
            device_id=device,
        )
        shard = torch.ones(1 << 20, device=device)
        full = torch.empty(4 << 20, device=device)
        dist.all_gather_into_tensor(full, shard)
        dist.all_reduce(shard)

    pgs = {}
    for peer in range(4):
        client, volume = (peer, local) if is_volume else (local, peer)
        pgs[peer] = _pair_group(
            port_base + 4 * client + volume, f"c{client}v{volume}", is_volume, device
        )
    bad = [
        f"volume {peer} round {rnd}"
        for rnd in range(2)
        for peer in range(4)
        if not _pull(pgs[peer], local if is_volume else peer, rnd, is_volume, device)
    ]
    if dist.is_initialized():
        dist.destroy_process_group()
    assert not bad, f"client {local} received corrupt transfers: {bad}"


@requires_8_ranks
def test_xccl_cross_node_pull():
    """Volumes on every tile send to clients on another node."""
    _run_pull(29400, intra_node_collective=False)


@requires_8_ranks
def test_xccl_pull_after_intra_node_collective():
    """The same pull after the volumes ran a collective on an intra-node group.

    Fails on oneCCL 2022.1.x: broadcasts from tiles 1-3 deliver nothing.
    """
    _run_pull(29500, intra_node_collective=True)
