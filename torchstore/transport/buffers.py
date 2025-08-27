import torch
from typing import Optional, Any, Tuple

try:
    from monarch.tensor_engine import is_available as monarch_rdma_available, RDMABuffer
except ImportError:
    monarch_rdma_available = lambda: False
    def RDMABuffer(*args, **kwargs) -> "RDMABuffer":
        raise NotImplementedError("RDMABuffer is not available")
        

def rdma_available():
    return monarch_rdma_available()


class TransportBuffer:
    finalize: bool = False
    is_object: bool = False
    objects: Optional[Any] = None
    requires_meta: bool = False

    def allocate(self, tensor) -> torch.Tensor:
        raise NotImplemented()

    async def read_into(self, tensor) -> torch.Tensor:
        raise NotImplemented()
    
    async def write_from(self, tensor) -> None:
        raise NotImplemented()

class RDMATransportBuffer(TransportBuffer):
    requires_meta: bool = True

    def __init__(self) -> None:
        self.rdma_buff: Optional[RDMABuffer] = None
        # self.tensor_buff:torch.Tensor = None 
        #TODO: really keep it for defence reasons and dump it on pickle
        self.shape = None
        self.dtype = None

    def allocate(self, tensor) -> torch.Tensor:
        
        if isinstance(tensor, str):
            return # is an object, ignore for now

        if isinstance(tensor, Tuple):
            tensor = torch.empty(tensor[0], dtype=tensor[1])
        
        assert isinstance(tensor, torch.Tensor), f"{tensor=} is not a tensor"
        
        tensor_ref = torch.empty_like(tensor)
        self.shape = tensor.shape
        self.dtype = tensor.dtype
        
        self.rdma_buff = RDMABuffer(tensor_ref.view(torch.uint8).flatten())
        return tensor_ref


    # send
    async def read_into(self, tensor=None) -> torch.Tensor:
        if tensor is None:
            # allocate a tensor to return
            tensor = torch.empty(self.shape, dtype=self.dtype)

        assert tensor.is_contiguous()
        assert self.rdma_buff is not None
        await self.rdma_buff.read_into(tensor.view(torch.uint8).flatten())

        return tensor

    # recv
    async def write_from(self, tensor) -> None:
        if tensor is None:
            return
        # if I have tensor_buff copy"???
        assert self.rdma_buff is not None
        await self.rdma_buff.write_from(tensor.view(torch.uint8).flatten())





class MonarchTransportBuffer(TransportBuffer):
    """ This interface is mostly a noop, intended to be used with Monarch's regular RPC. 
    Not expected to be super fast, but always works.
    """
    finalize: bool = True

    def __init__(self) -> None:
        self.tensor = None

    def allocate(self, tensor):
        """ In the case of using monarch comms, we don't do any allocation ahead of time
        """
        return tensor

    # send
    async def read_into(self,tensor=None):
        if tensor is not None:
            # if there is a tensor here, likely this is the 'inplace' case,
            # and we should return back a ptr to the original tensor
            # (as opposed to the stored tensor, which we likely don't want to
            # keep around)
            tensor.copy_(self.tensor)
            return tensor
        
        return self.tensor
    
    # recv
    async def write_from(self, tensor):
        self.tensor = tensor
