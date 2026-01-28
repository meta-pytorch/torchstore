# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import torch
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._utils import _compute_local_shape_and_global_offset

if TYPE_CHECKING:
    pass


@dataclass
class TensorSlice:
    """Metadata describing a slice/shard of a distributed tensor.

    Attributes:
        offsets: Global offsets where this slice begins in each dimension.
        coordinates: Device mesh coordinates identifying which shard this is.
        global_shape: The shape of the full (unsharded) tensor.
        local_shape: The shape of this local slice/shard.
        mesh_shape: The shape of the device mesh used for sharding.
    """

    offsets: tuple
    coordinates: tuple
    global_shape: tuple
    local_shape: tuple
    mesh_shape: tuple

    def __post_init__(self):
        if self.coordinates is not None:
            self.coordinates = tuple(self.coordinates)

    def __hash__(self):
        return hash(
            (
                self.offsets,
                self.coordinates,
                self.global_shape,
                (
                    tuple(self.local_shape)
                    if hasattr(self.local_shape, "__iter__")
                    else self.local_shape
                ),
                self.mesh_shape,
            )
        )


@dataclass
class Request:
    """Request object encapsulating data to be stored or retrieved from TorchStore.

    Attributes:
        tensor_val (Optional[torch.Tensor]): The actual tensor data to store/retrieve.
            For DTensors, this contains the local tensor shard.
        tensor_slice (Optional[TensorSlice]): Metadata about distributed tensor sharding,
            including offsets, coordinates, and shape information.
        objects (Optional[Any]): Arbitrary Python objects that must be pickleable.
        is_object (bool): Flag indicating whether this request contains a non-tensor object.
    """

    tensor_val: torch.Tensor | None = None
    tensor_slice: TensorSlice | None = None
    objects: Any | None = None
    is_object: bool = False

    @classmethod
    def from_any(cls, value: torch.Tensor | DTensor | None) -> "Request":
        if isinstance(value, DTensor):
            request = cls.from_dtensor(value)
        elif isinstance(value, torch.Tensor):
            request = cls.from_tensor(value)
        else:
            request = cls.from_objects(value)

        return request

    @classmethod
    def from_dtensor(cls, dtensor: DTensor) -> "Request":
        coordinates = dtensor.device_mesh.get_coordinate()
        _, offsets = _compute_local_shape_and_global_offset(
            dtensor.shape,
            mesh_shape=dtensor.device_mesh.shape,
            my_coordinate=coordinates,
            placements=dtensor.placements,
        )

        tensor_slice = TensorSlice(
            offsets,
            coordinates,
            dtensor.shape,
            dtensor._local_tensor.shape,
            dtensor.device_mesh.shape,
        )
        return cls(
            tensor_val=dtensor._local_tensor,
            tensor_slice=tensor_slice,
        )

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "Request":
        return cls(tensor_val=tensor)

    @classmethod
    def from_objects(cls, objects) -> "Request":
        return cls(objects=objects, is_object=True)

    @classmethod
    def from_tensor_slice(cls, tensor_slice: TensorSlice) -> "Request":
        return cls(tensor_slice=copy.deepcopy(tensor_slice))

    def meta_only(self) -> "Request":
        """Returns a copy of this request with tensor_val set to None."""
        return Request(
            tensor_val=None,
            tensor_slice=self.tensor_slice,
            objects=self.objects,
            is_object=self.is_object,
        )
