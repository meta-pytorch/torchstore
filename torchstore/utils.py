# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import uuid
from logging import getLogger
from typing import List, Tuple, TYPE_CHECKING

import numpy as np

import torch

from monarch.actor import ProcMesh, this_host
from torchstore.transport import TensorSlice


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
) -> torch.Tensor:
    """
    Assemble a global tensor from local tensors based on their shapes and offsets. The final shape of the returned
    tensor is the union of local tensors.

    :param local_tensors: List of local tensors
    :param global_shape: Shape of the final global tensor
    :param global_offsets: List of offsets for each local tensor in the global tensor
    :return: The assembled global tensor
    """
    # Create an empty global tensor of the specified shape
    assert local_tensors

    target_shape, target_offset = get_target_tensor_shape_and_offset(
        [local_tensor.shape for local_tensor in local_tensors], global_offsets
    )
    tensor = torch.empty(
        target_shape,
        dtype=local_tensors[0].dtype,
    )

    # Iterate over each local tensor and place it in the correct position in the global tensor
    for local_tensor, offset in zip(local_tensors, global_offsets, strict=True):
        slices = tuple(
            slice(o - to, o - to + s)
            for o, to, s in zip(offset, target_offset, local_tensor.shape, strict=True)
        )
        tensor[slices] = local_tensor

    return tensor


def get_target_tensor_shape_and_offset(
    local_tensor_shapes: List["ShapeType"],
    global_offsets: List["ShapeType"],
) -> Tuple["ShapeType", "ShapeType"]:
    """
    Get the target tensor shape and offset based on the local tensor shapes and global offsets.

    :param local_tensor_shapes: List of shapes of local tensors
    :param global_offsets: List of offsets for each local tensor in the global tensor
    :return: Tuple of target tensor shape and offset
    """
    target_offset = min(global_offsets)
    target_ends = tuple(
        max(
            offset[i] + shape[i]
            for offset, shape in zip(global_offsets, local_tensor_shapes)
        )
        for i in range(len(global_offsets[0]))
    )
    target_shape = [max(0, e - o) for o, e in zip(target_offset, target_ends)]

    # Verify that local tensors can fill the target tensor, this verification is only necessary but not
    # sufficient to guarantee that the target tensor can be filled by local tensors.
    local_tensor_total_size = sum([np.prod(shape) for shape in local_tensor_shapes])
    target_tensor_size = np.prod(target_shape)
    assert (
        local_tensor_total_size <= target_tensor_size
    ), f"Local tensors cannot fill the target tensor. Local tensors total size: {local_tensor_total_size}, Target tensor size: {target_tensor_size}"

    return target_shape, target_offset
