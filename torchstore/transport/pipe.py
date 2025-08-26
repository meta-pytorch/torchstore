from typing import Optional, Tuple, Any
from dataclasses import dataclass

import torch
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._utils import _compute_local_shape_and_global_offset

from torchstore.transport.buffers import (
    TransportBuffer,
    RDMATransportBuffer,
    MonarchTransportBuffer,
    rdma_available
)



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
            objects=None,
        )

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor):
        return cls(tensor_val=tensor)

    @classmethod
    def from_objects(cls, objects):
        return cls(objects=objects)

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
    

    Storage volumes roughl follow:
    class StorageVolume(Actor):
        @endpoint
        def get(self, k, transport_buffer):
            transport_buffer.write_from(self.kv[k])
            return transport_buffer

        @endpoint
        def put(self, k, transport_buffer):
            
            if k not in self.kv:
                fetched_tensor = transport_buffer.read_into() # should allocate on the fly
                self.kv[k] = fetched_tensor
            else: #inplace update
                transport_buffer.read_into(self.kv[v])

            return transport_buffer

    """
    def __init__(self, storage_volume) -> None:
        self.storage_volume = storage_volume

    def create_transport_buffer(self) -> TransportBuffer:
        #TODO: eventually this should be dependent on the connections available to a storage_volume
        if rdma_available():
            buffer_cls = RDMATransportBuffer
        else:
            buffer_cls = MonarchTransportBuffer
        return buffer_cls()
    
    async def put_to_storage_volume(self, key, message: Message):

        transport_buffer = self.create_transport_buffer()
        tensor = message.tensor_val
        
        transport_buffer.allocate(tensor) 
        transport_buffer.write_from(tensor)
        
        # transporting tensors is handled by the buffer, so we don't want to send it 
        # via monarch RPC since that would generate considerable overhead
        message_without_tensor = Message(
            tensor_val=None,
            tensor_slice=message.tensor_slice,
            objects=message.objects
        )

        await self.storage_volume.put.call_one(key, transport_buffer, message_without_tensor)

    async def get_from_storage_volume(self, key, message: Message):

        # passing a tensor here is only important so we can create the right buffer size
        # maybe split out allocation, maybe don't

        send_buffer = self.create_transport_buffer()
        tensor_ref = send_buffer.allocate(message.tensor_val)

        # TODO: consider placing the buffer inside the message
        message_without_tensor = Message(
            tensor_val=None,
            tensor_slice=message.tensor_slice,
            objects=message.objects
        )
        # buffer after being processed remotely
        recv_buffer = await self.storage_volume.get.call_one(key, send_buffer, message_without_tensor)
        assert send_buffer is not None # it's important send_buffer is not GC'd before 'call_one'

        if recv_buffer.is_object:
            return recv_buffer.objects

        # finialize -- only necessary for MonarchCommsBuffer
        # but in the case of rdma, this was already done remotely.
        if recv_buffer.finalize:
            # still a little confusing but close
            tensor_ref = recv_buffer.read_into(tensor_ref) 

        return tensor_ref
