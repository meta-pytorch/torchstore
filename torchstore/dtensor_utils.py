# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from torch.distributed.tensor import DTensor, Placement
from torch.distributed.tensor._utils import _compute_local_shape_and_global_offset


def create_tensor_slice_from_dtensor(dtensor: DTensor) -> "TensorSlice":
    """
    Create a TensorSlice from a DTensor.

    Args:
        dtensor: The DTensor to extract metadata from

    Returns:
        TensorSlice containing the distributed tensor metadata
    """
    from torchstore.transport.pipe import TensorSlice

    coordinates = dtensor.device_mesh.get_coordinate()
    _, offsets = _compute_local_shape_and_global_offset(
        dtensor.shape,
        mesh_shape=dtensor.device_mesh.shape,
        my_coordinate=coordinates,
        placements=dtensor.placements,
    )

    return TensorSlice(
        offsets=offsets,
        coordinates=coordinates,
        global_shape=dtensor.shape,
        local_shape=dtensor._local_tensor.shape,
        mesh_shape=dtensor.device_mesh.shape,
    )


def reconstruct_dtensor_from_local_tensor(
    local_tensor: torch.Tensor,
    tensor_slice: "TensorSlice",
    device_mesh: torch.distributed.DeviceMesh,
    placements: Tuple[Placement, ...],
) -> DTensor:
    """
    Reconstruct a DTensor from local tensor data and TensorSlice metadata.

    Args:
        local_tensor: The local tensor shard
        tensor_slice: TensorSlice containing distributed metadata
        device_mesh: The device mesh for the DTensor
        placements: The placements for the DTensor

    Returns:
        Reconstructed DTensor
    """
    return DTensor.from_local(
        local_tensor=local_tensor,
        device_mesh=device_mesh,
        placements=placements,
    )
