from typing import List, Tuple

import torch
from torch._prims_common import ShapeType

from monarch.actor import proc_mesh


async def spawn_actors(num_processes, actor_cls, name, **init_args):
    """Actors are essentially processes wrapped in a class."""
    mesh = await proc_mesh(gpus=num_processes)
    actors = await mesh.spawn(name, actor_cls, **init_args)
    actors.mesh = mesh
    return actors


def get_local_tensor(
    global_tensor: torch.Tensor,
    local_shape: ShapeType,
    global_offset: Tuple[int, ...],
):
    # Calculate the slices for each dimension
    slices = tuple(
        slice(offset, offset + size) for offset, size in zip(global_offset, local_shape)
    )
    for s in slices:
        assert global_tensor[s].is_contiguous()

    # Slice the global_tensor to obtain the local_tensor
    local_tensor = global_tensor[slices]
    return local_tensor


def assemble_global_tensor(
    local_tensors: List[torch.Tensor],
    global_shape: ShapeType,
    global_offsets: List[ShapeType],
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

    global_tensor = torch.zeros(
        global_shape,
        dtype=local_tensors[0].dtype,
    )  # TODO: could be better to initialize to NaN and do an if NaN check here

    # Iterate over each local tensor and place it in the correct position in the global tensor
    for local_tensor, offset in zip(local_tensors, global_offsets):
        slices = tuple(slice(o, o + s) for o, s in zip(offset, local_tensor.shape))
        global_tensor[slices] = local_tensor

    assert global_tensor.is_contiguous()

    return global_tensor
