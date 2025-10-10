# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import uuid
from logging import getLogger
from typing import List, Optional, Tuple, TYPE_CHECKING

import torch

from monarch.actor import ProcMesh, this_host


if TYPE_CHECKING:
    from torch._prims_common import ShapeType

logger = getLogger(__name__)


async def spawn_actors(num_processes, actor_cls, name, mesh=None, **init_args):
    """Actors are essentially processes wrapped in a class."""

    if mesh is None:
        logger.debug("Spawning actors on the local host")
        mesh = this_host().spawn_procs(per_host={"gpus": num_processes})
        await mesh.initialized
        actors = mesh.spawn(f"{name}_{str(uuid.uuid4())[:8]}", actor_cls, **init_args)
        return actors

    assert isinstance(mesh, ProcMesh)
    actors = mesh.spawn(f"{name}_{str(uuid.uuid4())[:8]}", actor_cls, **init_args)

    return actors


def get_local_tensor(
    global_tensor: "torch.Tensor",
    local_shape: "ShapeType",
    global_offset: Tuple[int, ...],
):
    # Calculate the slices for each dimension
    slices = tuple(
        slice(offset, offset + size)
        for offset, size in zip(global_offset, local_shape, strict=True)
    )

    # Slice the global_tensor to obtain the local_tensor
    local_tensor = global_tensor[slices]
    return local_tensor


def assemble_global_tensor(
    local_tensors: List[torch.Tensor],
    global_shape: "ShapeType",
    global_offsets: List["ShapeType"],
):
    """
    Assemble a global tensor from local tensors based on their shapes and offsets.
    """
    if not local_tensors:
        raise ValueError("local_tensors cannot be empty")

    # Detect which dimension is sharded
    shard_dim = _detect_shard_dimension(local_tensors, global_offsets, global_shape)

    if shard_dim is not None:
        # Fast path: all shards are along one dimension, use cat
        # Sort by offset along shard dimension
        sorted_pairs = sorted(
            zip(local_tensors, global_offsets), key=lambda x: x[1][shard_dim]
        )
        sorted_tensors = [t for t, _ in sorted_pairs]
        expected_size_along_shard_dim = sum(t.shape[shard_dim] for t in sorted_tensors)
        if expected_size_along_shard_dim != global_shape[shard_dim]:
            raise RuntimeError(
                f"Shard sizes don't sum to global size along dim {shard_dim}: "
                f"got {expected_size_along_shard_dim}, expected {global_shape[shard_dim]}, "
                f"global_shape={global_shape}, global_offsets={global_offsets}"
            )
        return torch.cat(sorted_tensors, dim=shard_dim)
    else:
        # Fallback: complex sharding pattern, use slower method
        return _assemble_strided(local_tensors, global_shape, global_offsets)


def _detect_shard_dimension(
    local_tensors: List[torch.Tensor],
    global_offsets: List["ShapeType"],
    global_shape: "ShapeType",
) -> Optional[int]:
    """
    Detect if all shards are along a single dimension.
    Returns the dimension index or None if sharding is multi-dimensional.
    """
    if len(local_tensors) == 1:
        return 0  # Single tensor, dimension doesn't matter

    # Check each dimension
    for dim in range(len(global_shape)):
        # All offsets should vary only in this dimension
        varying_dim = True
        for other_dim in range(len(global_shape)):
            if other_dim == dim:
                continue
            # Check if all offsets are 0 in this dimension
            if not all(offset[other_dim] == 0 for offset in global_offsets):
                varying_dim = False
                break

        if varying_dim:
            # Verify shapes match global shape except in shard dimension
            valid = True
            for tensor, offset in zip(local_tensors, global_offsets):
                for d in range(len(global_shape)):
                    if d != dim and tensor.shape[d] != global_shape[d]:
                        valid = False
                        break
            if valid:
                return dim

    return None  # Multi-dimensional sharding


def _assemble_strided(
    local_tensors: List[torch.Tensor],
    global_shape: "ShapeType",
    global_offsets: List["ShapeType"],
) -> torch.Tensor:
    """Fallback for complex sharding patterns."""
    global_tensor = torch.empty(global_shape, dtype=local_tensors[0].dtype)
    for local_tensor, offset in zip(local_tensors, global_offsets, strict=True):
        slices = tuple(
            slice(o, o + s) for o, s in zip(offset, local_tensor.shape, strict=True)
        )
        global_tensor[slices] = local_tensor
    return global_tensor
