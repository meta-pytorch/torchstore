# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import uuid
from logging import getLogger
from typing import List, Tuple, TYPE_CHECKING

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
        actors = await mesh.spawn(
            f"{name}_{str(uuid.uuid4())[:8]}", actor_cls, **init_args
        )
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

    :param local_tensors: List of local tensors
    :param global_shape: Shape of the final global tensor
    :param global_offsets: List of offsets for each local tensor in the global tensor
    :return: The assembled global tensor
    """
    # Create an empty global tensor of the specified shape
    assert local_tensors

    global_tensor = torch.empty(
        global_shape,
        dtype=local_tensors[0].dtype,
    )

    # Iterate over each local tensor and place it in the correct position in the global tensor
    for local_tensor, offset in zip(local_tensors, global_offsets, strict=True):
        slices = tuple(
            slice(o, o + s) for o, s in zip(offset, local_tensor.shape, strict=True)
        )
        global_tensor[slices] = local_tensor

    return global_tensor
