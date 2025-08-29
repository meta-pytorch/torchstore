import logging
from sys import exc_info
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

import logging
import sys
logger = logging.getLogger(__name__)


logger.root.setLevel(logging.DEBUG)
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
logger.root.addHandler(stdout_handler)


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
        self.tensor_ref: Optional[torch.Tensor] = None
        self.shape = None
        self.dtype = None

    def __getstate__(self) -> object:
        state = self.__dict__.copy()
        state["tensor_ref"] = None
        return state

    def allocate(self, tensor) -> torch.Tensor:
        #TODO: potentially pass Message object here directly.

        if isinstance(tensor, str) or tensor is None:
            return # is an object, ignore for now
        elif isinstance(tensor, Tuple):
            self.tensor_ref = torch.empty(tensor[0], dtype=tensor[1])
        elif isinstance(tensor, torch.Tensor):
            #TODO: avoid copy if tensor.is_contiguous()
            self.tensor_ref = torch.empty_like(tensor)

        assert isinstance(self.tensor_ref, torch.Tensor)
        self.shape = self.tensor_ref.shape
        self.dtype = self.tensor_ref.dtype

        # Handle scalar tensors specially - they cannot be viewed as uint8 directly
        # safe because t.squeeze/unsqueeze is a view.
        t = self.tensor_ref if self.tensor_ref.dim() > 0 else self.tensor_ref.unsqueeze(0)
        self.rdma_buff = RDMABuffer(t.view(torch.uint8).flatten())

        return self.tensor_ref


    # send
    async def read_into(self, tensor=None) -> torch.Tensor:        
        if tensor is None:
            # allocate a tensor to return
            print(f"{self.shape=} {self.dtype=}") 
            tensor = torch.empty(self.shape, dtype=self.dtype)

        assert tensor.is_contiguous()
        assert self.rdma_buff is not None
        
        # Handle scalar tensors specially
        if tensor.dim() == 0:
            # For scalar tensors, create a temporary 1-element tensor for RDMA
            temp_tensor = tensor.unsqueeze(0)
            await self.rdma_buff.read_into(temp_tensor.view(torch.uint8).flatten())
            # The original scalar tensor is already updated since temp_tensor shares storage
        else:
            await self.rdma_buff.read_into(tensor.view(torch.uint8).flatten())

        return tensor

    # recv
    async def write_from(self, tensor) -> None:
        if tensor is None:
            return

        if self.tensor_ref is not None:
            self.tensor_ref.copy_(tensor)
            return

        assert self.shape == tensor.shape, f"{self.shape} != {tensor.shape}"
        assert self.dtype == tensor.dtype, f"{self.dtype} != {tensor.dtype}"

        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        assert self.rdma_buff is not None
        
        try:
            # Handle scalar tensors specially
            if tensor.dim() == 0:
                # For scalar tensors, create a temporary 1-element tensor for RDMA
                temp_tensor = tensor.unsqueeze(0)
                await self.rdma_buff.write_from(temp_tensor.view(torch.uint8).flatten())
            else:
                await self.rdma_buff.write_from(tensor.view(torch.uint8).flatten())
        except Exception as e:
            logging.exception(f"Failed to write, {self.shape=}, {self.dtype=}", exc_info=e)
            raise e


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
