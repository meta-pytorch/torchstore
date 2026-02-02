# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from dataclasses import dataclass
from logging import getLogger
from typing import Any, TYPE_CHECKING

import torch
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._utils import _compute_local_shape_and_global_offset
from torch.distributed.tensor.placement_types import Replicate

logger = getLogger(__name__)

if TYPE_CHECKING:
    from torchstore.state_dict_utils import TorchStoreStateDict


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
    
    @classmethod
    def from_dtensor(cls, dtensor: DTensor) -> "TensorSlice":
        coordinates = dtensor.device_mesh.get_coordinate()
        _, offsets = _compute_local_shape_and_global_offset(
            dtensor.shape,
            mesh_shape=dtensor.device_mesh.shape,
            my_coordinate=coordinates,
            placements=dtensor.placements,
        )

        return cls(
            offsets=offsets,
            coordinates=coordinates,
            global_shape=dtensor.shape,
            local_shape=dtensor._local_tensor.shape,
            mesh_shape=dtensor.device_mesh.shape,
        )



def _is_dtensor_fully_local(dtensor: DTensor) -> bool:
    """
    Check if a DTensor is fully local (not actually distributed).

    A DTensor is considered fully local if:
    1. All placements are Replicate(), OR
    2. The device mesh has only 1 device total

    In these cases, the DTensor doesn't need collective operations
    and can be treated as a regular tensor for storage purposes.

    Args:
        dtensor: The DTensor to check

    Returns:
        True if the DTensor is fully local, False otherwise
    """
    # Check if mesh has only 1 device
    mesh_size = dtensor.device_mesh.size()
    if mesh_size == 1:
        return True

    # Check if all placements are Replicate
    all_replicate = all(isinstance(p, Replicate) for p in dtensor.placements)
    if all_replicate:
        return True

    return False


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
        is_tssd (bool): Flag indicating whether this request contains a TorchStoreStateDict object.
    """

    tensor_val: torch.Tensor | None = None
    tensor_slice: TensorSlice | None = None
    objects: Any | None = None
    is_object: bool = False
    is_tssd: bool = False

    @classmethod
    def from_any(cls, value: torch.Tensor | DTensor | None) -> "Request":
        from torchstore.state_dict_utils import TorchStoreStateDict
        if isinstance(value, DTensor):
            # Check if DTensor is fully local (not actually distributed)
            # If so, treat it as a regular tensor to avoid collective requirements
            # Note: this is due to behavior in torchtitan, where we have Replicate()
            # placement which is not actually replicated along device-mesh
            # todo: Revisit this if this is fixed in torchtitan
            if _is_dtensor_fully_local(value):
                logger.debug(
                    f"DTensor with shape {value.shape} is fully local "
                    f"(placements: {value.placements}, mesh size: {value.device_mesh.size()}). "
                    "Treating as regular tensor for storage."
                )
                request = cls.from_tensor(value._local_tensor)
            else:
                request = cls.from_dtensor(value)
        elif isinstance(value, torch.Tensor):
            request = cls.from_tensor(value)
        elif isinstance(value, TorchStoreStateDict):
            request = cls.from_tssd(value)
        else:
            request = cls.from_objects(value)

        return request

    @classmethod
    def from_tssd(cls, tssd: "TorchStoreStateDict") -> "Request":
        return cls(
            tensor_val=tssd.tensor_blob, objects=tssd.metadata_state_dict, is_tssd=True
        )

    @classmethod
    def from_dtensor(cls, dtensor: DTensor) -> "Request":
        tensor_slice = TensorSlice.from_dtensor(dtensor)
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
            is_tssd=self.is_tssd,
        )
