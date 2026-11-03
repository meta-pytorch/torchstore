# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The routed-sync tests' state dicts: two publishers and two requester meshes.

Monarch-free so the tests that run without Monarch can share them.
"""

from __future__ import annotations

import inspect

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import distribute_tensor, DTensor, Shard

KEY = "model_state_dict"
ROUNDS = 3
PUBLISHER_RANKS = 2
# Shard dim of each entry on a mesh. None means every rank holds the whole tensor.
# Column shards of out.weight are strided in every requester's buffer. Reads copy
# them in rather than writing in place.
PUBLISHER_DIMS: dict[str, int | None] = {
    "proj.weight": 0,
    "norm.weight": None,
    "out.weight": 1,
}
# (number of ranks, shard dims) of each requester mesh, indexed by its group.
REQUESTERS: tuple[tuple[int, dict[str, int | None]], ...] = (
    (1, {"proj.weight": None, "norm.weight": None, "out.weight": None}),
    (2, {"proj.weight": 1, "norm.weight": 0, "out.weight": 0}),
)
DESTINATION_ONLY = "requester_only.weight"


def weights(round_: int) -> dict[str, torch.Tensor]:
    """Whole tensors published in one round. Every value changes each round."""
    proj = torch.arange(24, dtype=torch.float32).view(4, 6) / 7 + round_
    return {
        "proj.weight": proj.to(torch.bfloat16),
        "norm.weight": torch.arange(6, dtype=torch.float32) / 3 - round_,
        "out.weight": torch.arange(24, dtype=torch.float32).view(4, 6) / 5 + 2 * round_,
    }


def sentinel() -> torch.Tensor:
    return torch.full((3,), -1.0)


def local(
    tensor: torch.Tensor, dim: int | None, rank: int, num_ranks: int
) -> torch.Tensor:
    """The part of a whole tensor that one rank holds."""
    return tensor if dim is None else tensor.chunk(num_ranks, dim)[rank]


def shard(tensor: torch.Tensor, dim: int, rank: int, num_ranks: int) -> DTensor:
    """Hold one rank's chunk of ``tensor`` along ``dim`` as a DTensor shard with real offsets and no process group.

    The fake mesh needs DeviceMesh's private ``_init_backend`` and ``_rank`` (torch 2.14 to 2.16 take them). A torch
    that renames either fails every caller here with an AssertionError naming it.
    """
    # Private arguments, which no public call replaces without a process group.
    private = {"_init_backend", "_rank"} - set(inspect.signature(DeviceMesh).parameters)
    assert not private, f"this torch's DeviceMesh takes no {sorted(private)}"
    mesh = DeviceMesh("cpu", torch.arange(num_ranks), _init_backend=False, _rank=rank)
    return distribute_tensor(tensor, mesh, [Shard(dim)], src_data_rank=None)


def place(
    tensors: dict[str, torch.Tensor],
    dims: dict[str, int | None],
    rank: int,
    num_ranks: int,
) -> dict[str, torch.Tensor]:
    """Hold each tensor as one rank of a mesh would."""
    placed = {}
    for name, tensor in tensors.items():
        dim = dims[name]
        placed[name] = (
            tensor.clone() if dim is None else shard(tensor, dim, rank, num_ranks)
        )
    return placed


def assert_bitwise_equal(
    actual: torch.Tensor, expected: torch.Tensor, label: str = ""
) -> None:
    """Assert equal dtypes, shapes and bytes. The messages carry the detail since pytest rewrites no assert here."""
    prefix = f"{label}: " if label else ""
    kinds = (
        (actual.dtype, tuple(actual.shape)),
        (expected.dtype, tuple(expected.shape)),
    )
    assert kinds[0] == kinds[1], f"{prefix}{kinds[0]} != {kinds[1]}"
    assert torch.equal(
        actual.contiguous().view(torch.uint8),
        expected.contiguous().view(torch.uint8),
    ), f"{prefix}{actual} != {expected}"
