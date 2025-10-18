# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from dataclasses import dataclass
from logging import getLogger
from typing import Any, Optional, Tuple

import torch
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._utils import _compute_local_shape_and_global_offset

from torchstore.transport.buffers import (
    MonarchTransportBuffer,
    rdma_available,
    RDMATransportBuffer,
    TransportBuffer,
)

logger = getLogger(__name__)


@dataclass
class TensorSlice:
    offsets: Tuple
    coordinates: Tuple
    global_shape: Tuple
    local_shape: Tuple  # TODO: fix type hints
    mesh_shape: Tuple

    def __post_init__(self):
        if self.coordinates is not None:
            self.coordinates = tuple(self.coordinates)

    def __hash__(self):
        # Hash all fields as a tuple, converting local_shape to tuple if it's a torch.Size
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

    tensor_val: Optional[torch.Tensor] = None
    tensor_slice: Optional[TensorSlice] = None
    objects: Optional[Any] = None  # Any, but must be pickleable.
    is_object: bool = False

    @classmethod
    def from_any(cls, value: torch.Tensor | DTensor | None) -> "Request":
        if isinstance(value, DTensor):
            request = cls.from_dtensor(value)
        elif isinstance(value, torch.Tensor):
            request = cls.from_tensor(value)
        else:
            # TODO: consolidate this path for the None case
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


class Pipe:
    """
    Transport wrapper for communicating from local clients to storage volumes.
    """

    def __init__(self, storage_volume) -> None:
        self.storage_volume = storage_volume

    def create_transport_buffer(self) -> TransportBuffer:
        # TODO: eventually this should be dependent on the connections available to a storage_volume
        if rdma_available():
            buffer_cls = RDMATransportBuffer
        else:
            buffer_cls = MonarchTransportBuffer
        return buffer_cls()

    async def put_to_storage_volume(self, key, request: Request):
        transport_buffer = self.create_transport_buffer()
        tensor = request.tensor_val

        transport_buffer.allocate(tensor)
        await transport_buffer.write_from(tensor)

        # transporting tensors is handled by the buffer, so we don't want to send it
        # via monarch RPC since that would generate considerable overhead
        try:
            await self.storage_volume.put.call_one(
                key, transport_buffer, request.meta_only()
            )
        finally:
            # Clean up the transport buffer after the put operation completes
            # This is critical for RDMA buffers to deregister memory regions
            await transport_buffer.drop()

    async def get_from_storage_volume(self, key, request: Request):

        transport_buffer = self.create_transport_buffer()

        try:
            # Certain buffers (RDMA) need to know the size of the tensor
            # so we can allocate the right amount of memory locally.
            # This can be avoided if the request contains a tensor slice.
            # Could likely be optimized away in the future.
            if transport_buffer.requires_meta and request.tensor_val is None:
                meta = await self.storage_volume.get_meta.call_one(
                    key, request.meta_only()
                )
                transport_buffer.allocate(meta)
            else:
                transport_buffer.allocate(request.tensor_val)

            # TODO: consider placing the buffer inside the request or vice versa
            transport_buffer.update(
                await self.storage_volume.get.call_one(
                    key, transport_buffer, request.meta_only()
                )
            )

            if transport_buffer.is_object:
                return transport_buffer.objects

            return await transport_buffer.read_into(request.tensor_val)
        finally:
            # Clean up the transport buffer after the get operation completes
            # This is critical for RDMA buffers to deregister memory regions
            await transport_buffer.drop()
