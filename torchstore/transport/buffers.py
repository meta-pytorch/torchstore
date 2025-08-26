import torch
from typing import Optional, Any

try:
    from monarch.tensor_engine import is_available as monarch_rdma_available, RDMABuffer
except ImportError:
    monarch_rdma_available = lambda: False
    def RDMABuffer():
        raise RuntimeError("RDMABuffer not available")

def rdma_available():
    return monarch_rdma_available()


class TransportBuffer:
    finalize: bool = False
    is_object: bool = False
    objects: Optional[Any] = None

    def allocate(self, tensor) -> torch.Tensor:
        raise NotImplemented()

    def read_into(self, tensor) -> torch.Tensor:
        raise NotImplemented()
    
    def write_from(self, tensor) -> None:
        raise NotImplemented()

class RDMATransportBuffer(TransportBuffer):
    finalize: bool = False

    def allocate(self, tensor) -> torch.Tensor:
        ...

    # send
    def read_into(self, tensor) -> torch.Tensor:
        pass

    # recv
    def write_from(self, tensor) -> None:
        pass

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
    def read_into(self,tensor=None):
        if tensor is not None:
            # if there is a tensor here, likely this is the 'inplace' case,
            # and we should return back a ptr to the original tensor
            # (as opposed to the stored tensor, which we likely don't want to
            # keep around)
            tensor.copy_(self.tensor)
            return tensor
        
        return self.tensor
    
    # recv
    def write_from(self, tensor):                
        self.tensor = tensor
