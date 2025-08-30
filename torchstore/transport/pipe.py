from typing import Optional, Tuple, Any
from dataclasses import dataclass
from logging import getLogger

import torch
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._utils import _compute_local_shape_and_global_offset

from torchstore.transport.buffers import (
    TransportBuffer,
    RDMATransportBuffer,
    MonarchTransportBuffer,
    rdma_available
)

logger = getLogger(__name__)

@dataclass
class TensorSlice:
    offsets: Tuple
    coordinates: Tuple
    global_shape: Tuple
    local_shape: Tuple #TODO: fix type hints 
    mesh_shape: Tuple

    def __post_init__(self):
        self.coordinates = tuple(self.coordinates)

@dataclass
class Message:
    tensor_val: Optional[torch.Tensor] = None
    tensor_slice: Optional[TensorSlice] = None
    objects: Optional[Any] = None # Any, but must be pickleable.
    is_object: bool = False 
    
    @classmethod
    def from_any(cls, value: Any):
        if isinstance(value, DTensor):
            message = cls.from_dtensor(value)
        elif isinstance(value, torch.Tensor):
            message = cls.from_tensor(value)
        else:
            message = cls.from_objects(value)

        return message

    @classmethod
    def from_dtensor(cls, dtensor: DTensor) -> "Message":
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

        tensor = dtensor._local_tensor

        return cls(
            tensor_val=tensor,
            tensor_slice=tensor_slice,
        )

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor):
        return cls(tensor_val=tensor)

    @classmethod
    def from_objects(cls, objects):
        return cls(objects=objects, is_object=True)

    @classmethod
    def from_tensor_offsets(
        cls,
        offsets,
        coordinates,
        global_shape,
        mesh_shape
    ):
        raise NotImplementedError()



class Pipe:
    """
    Transport wrapper for communicating from local clients to storage volumes.
    """
    def __init__(self, storage_volume) -> None:
        self.storage_volume = storage_volume

    def create_transport_buffer(self, force_monarch_comms=False) -> TransportBuffer:
        #TODO: eventually this should be dependent on the connections available to a storage_volume
        if rdma_available() and not force_monarch_comms:
            buffer_cls = RDMATransportBuffer
        else:
            buffer_cls = MonarchTransportBuffer
        return buffer_cls()
    
    async def put_to_storage_volume(self, key, message: Message):

        force_monarch_comms = message.tensor_val is not None and message.tensor_val.dim() == 0
        transport_buffer = self.create_transport_buffer(force_monarch_comms=force_monarch_comms)
        tensor = message.tensor_val
        
        tensor_ref = transport_buffer.allocate(tensor) 
        await transport_buffer.write_from(tensor)
        
        # transporting tensors is handled by the buffer, so we don't want to send it 
        # via monarch RPC since that would generate considerable overhead
        message_without_tensor = Message(
            tensor_val=None,
            tensor_slice=message.tensor_slice,
            objects=message.objects,
            is_object=message.is_object
        )

        await self.storage_volume.put.call_one(key, transport_buffer, message_without_tensor)
        if not message.is_object:
            assert tensor_ref is not None, "it's important tensor_ref is not GC'd before 'storage_volume.put'"
    
    async def get_from_storage_volume(self, key, message: Message):

        #TODO: monarch rdma buffers hate scalars, and this barely makes sense for rdma until we're
        # streaming
        force_monarch_comms = message.tensor_val is not None and message.tensor_val.dim() == 0
        transport_buffer = self.create_transport_buffer(force_monarch_comms=force_monarch_comms)

        # workaround until we develop streaming support. Certain buffers (RDMA) 
        # need to know the size of the tensor so we can allocate the right 
        # amount of memory locally. This can be avoided if the message
        # contains a tensor slice.
        if transport_buffer.requires_meta and message.tensor_val is None:
            meta = await self.storage_volume.get_meta.call_one(key)
            # passing a tensor here is only important so we can create the right buffer size        
            transport_buffer.allocate(meta)
        else:
            transport_buffer.allocate(message.tensor_val)

        # TODO: consider placing the buffer inside the message or vice versa
        message_without_tensor = Message(
            tensor_val=None,
            tensor_slice=message.tensor_slice,
            objects=message.objects
        )
        # buffer after being processed remotely
        transport_buffer.update(
            await self.storage_volume.get.call_one(
                key,
                transport_buffer,
                message_without_tensor
            )
        )

        if transport_buffer.is_object:
            return transport_buffer.objects

        # return message.tensor_val
        return await transport_buffer.read_into(message.tensor_val)
