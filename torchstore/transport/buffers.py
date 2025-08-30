import os
import logging
from typing import Optional, Any, Tuple, List

import torch
try:
    from monarch.tensor_engine import is_available as monarch_rdma_available, RDMABuffer
except ImportError:
    monarch_rdma_available = lambda: False
    def RDMABuffer(*args, **kwargs) -> "RDMABuffer":
        raise NotImplementedError("RDMABuffer is not available")
        

def rdma_available():
    return monarch_rdma_available()


RDMDA_CHUNK_SIZE_MB= int(
    os.environ.get("RDMDA_CHUNK_SIZE_MB", "512")
)

class TransportBuffer:
    finalize: bool = False
    is_object: bool = False
    objects: Optional[Any] = None
    requires_meta: bool = False

    def update(self, other_buffer) -> None:
        self.finalize = other_buffer.finalize
        self.is_object = other_buffer.is_object
        self.objects = other_buffer.objects
        self.requires_meta = other_buffer.requires_meta

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

    def _create_byte_views_from_tensor(self, tensor) -> List[torch.Tensor]:
        if tensor.dim()==0:
            tensor = tensor.unsqueeze(0)
        byte_view = tensor.view(torch.uint8).flatten()
        chunk_size = RDMDA_CHUNK_SIZE_MB * 1024 * 1024
        offset = 0
        tensor_chunks = []
        while offset < byte_view.numel():
            tensor_chunks.append(byte_view[offset:offset + chunk_size])
            offset += chunk_size

        return tensor_chunks

    def allocate(self, tensor) -> torch.Tensor:
        logging.debug("Allocating rdma buffer")
        #TODO: potentially pass Message object here directly.

        if isinstance(tensor, str) or tensor is None:
            return # is an object, ignore for now
        elif isinstance(tensor, Tuple): #TODO: fix this shit
            # nothing was passed inplace, so we need to allocate some memory here
            tensor = torch.empty(tensor[0], dtype=tensor[1]) 
        else:
            tensor = torch.empty_like(tensor)

        assert isinstance(tensor, torch.Tensor)    
        assert tensor.is_contiguous()            
        self.shape = tensor.shape
        self.dtype = tensor.dtype
        self.dim = tensor.dim()
        logging.debug(f"{self.shape=} {self.dtype=} {self.dim=}")

        # special handling for scalars
        byte_view_chunks = self._create_byte_views_from_tensor(tensor)
        self.tensor_ref = [torch.empty_like(chunk) for chunk in byte_view_chunks]
        self.rdma_buff = [RDMABuffer(chunk) for chunk in self.tensor_ref]
        logging.debug(f"Allocted {len(self.rdma_buff)} rdma buffers")

    def update(self, other_buffer):
        super().update(other_buffer)

    # send
    async def read_into(self, tensor=None) -> torch.Tensor:     
        if tensor is None:
            # allocate a tensor to return
            tensor = torch.empty(self.shape, dtype=self.dtype)

        #TODO: assert safe
        assert tensor.dtype == self.dtype, f"{tensor.dtype} != {self.dtype}"
        assert tensor.shape == self.shape, f"{tensor.shape} != {self.shape}"
        assert tensor.is_contiguous()
        assert self.rdma_buff is not None
        
        chunked_byte_view = self._create_byte_views_from_tensor(tensor)
        
        # if we have tensor refs locally, we're still in the local case,
        # and we're just copying over our chunks into the tensor from
        # local memory
        if self.tensor_ref is not None:
            for idx, chunk in enumerate(chunked_byte_view):
                chunk.copy_(self.tensor_ref[idx])
            return tensor
        # else: we are in the remote case (in a different process), and must read from 
        # the rdma buffer

        try:
            for idx, chunk in enumerate(chunked_byte_view):
                await self.rdma_buff[idx].read_into(chunk)
            
        except Exception as e:
            logging.exception(f"Failed read_into, {tensor.shape=}, {tensor.dtype=}", exc_info=e)
            raise e

        return tensor

    # recv
    async def write_from(self, tensor) -> None:
        if tensor is None:
            return

        # if we have tensor refs locally, we're still in the local case,
        # and we're just copying over our chunks into the tensor from
        # local memory
        if self.tensor_ref is not None:
            chunked_byte_view = self._create_byte_views_from_tensor(tensor)
            for idx, chunk in enumerate(chunked_byte_view):
                self.tensor_ref[idx].copy_(chunk)            
            return
        # else: we are in the remote case (in a different process), and must read from 
        # the rdma buffer

        assert self.shape == tensor.shape, f"{self.shape} != {tensor.shape}"
        assert self.dtype == tensor.dtype, f"{self.dtype} != {tensor.dtype}"
        assert self.rdma_buff is not None
        
        chunked_byte_view = self._create_byte_views_from_tensor(tensor)

        for idx, chunk in enumerate(chunked_byte_view):
            await self.rdma_buff[idx].write_from(chunk)


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
    async def read_into(self, tensor=None):
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

    def update(self, other_buffer):
        super().update(other_buffer)
        self.tensor = other_buffer.tensor
