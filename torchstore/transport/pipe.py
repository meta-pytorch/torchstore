from typing import Optional

import torch

from torchstore.transport.buffers import (
    TransportBuffer,
    RDMATransportBuffer,
    MonarchTransportBuffer,
    rdma_available
)



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
    
    async def put_to_storage_volume(self, k, tensor):
        transport_buffer = self.create_transport_buffer()

        transport_buffer.allocate(tensor) 
        transport_buffer.write_from(tensor)

        await self.storage_volume.put.call_one(k, transport_buffer)

    async def get_from_storage_volume(self, k, inplace_tensor: Optional[torch.Tensor]):
        
        # passing a tensor here is only important so we can create the right buffer size
        # maybe split out allocation, maybe don't
        send_buffer = self.create_transport_buffer()
        tensor_ref = send_buffer.allocate(inplace_tensor)

        # buffer after being processed remotely
        recv_buffer = await self.storage_volume.get.call_one(k, send_buffer)
        
        # it's important send_buffer is not GC'd before 'call_one'
        assert send_buffer is not None

        #TODO: ok, another issue here is rdma transfer in the case of inplace_tensor = None

        # finialize -- only necessary for MonarchCommsBuffer
        # but in the case of rdma, this was already done remotely.
        if recv_buffer.finalize:
            # still a little confusing but close
            tensor_ref = recv_buffer.read_into(tensor_ref) 

        return tensor_ref
