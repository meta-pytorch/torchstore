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
        

#TODO: for some reason, RDMABuffer is breaking for certain tensors on the HF models (qwen, llama)
# but setting this chunk size works around the issue until we can fix it
# N.B. from benchmarking, we know the ideal size is any size >=256mb.
RDMDA_CHUNK_SIZE_MB= int(
    os.environ.get("TORCHSTORE_RDMDA_CHUNK_SIZE_MB", "1") 
)
assert RDMDA_CHUNK_SIZE_MB <= 1024, "Monarch does not support 1gb chunks via rdma"

RDMA_ENABLED = os.environ.get("TORCHSTORE_RDMA_ENABLED", "1") == "1"

def rdma_available():
    return RDMA_ENABLED and monarch_rdma_available()

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
        # Any time that we serialize the transport buffer, the idea is 
        # that tensors will be transported via tensor_enginer.RDMABuffer, so it makes
        # no sense to hold this reference when we are serializing
        state = self.__dict__.copy()
        state["tensor_ref"] = None
        return state

    def _create_byte_views_from_tensor(self, tensor) -> List[torch.Tensor]:
        # handle scalar values
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

    def _assert_safe_tensor(self, tensor):
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == self.dtype, f"{tensor.dtype} != {self.dtype}"
        assert tensor.shape == self.shape, f"{tensor.shape} != {self.shape}"
        assert tensor.is_contiguous()

    def allocate(self, tensor) -> torch.Tensor:
        logging.debug("Allocating rdma buffer")

        if isinstance(tensor, str) or tensor is None:
            # tensor is just an object, nothing to allocte
            return 
        elif isinstance(tensor, Tuple):
            # we know the size of the tensor from fetching metadata
            tensor = torch.empty(tensor[0], dtype=tensor[1]) 
        else:
            # we have an inplace tensor, allocate a copy
            assert isinstance(tensor, torch.Tensor)    
            tensor = torch.empty_like(tensor)

        # store tensor meta
        self.shape = tensor.shape
        self.dtype = tensor.dtype
        self.dim = tensor.dim()

        self._assert_safe_tensor(tensor)

        byte_view_chunks = self._create_byte_views_from_tensor(tensor)
        self.tensor_ref = [torch.empty_like(chunk) for chunk in byte_view_chunks]
        self.rdma_buff = [RDMABuffer(chunk) for chunk in self.tensor_ref]

        chunk_sizes = set()
        for chunk in self.tensor_ref:
            chunk_sizes.add(chunk.shape)
        logging.debug(f"Allocted {len(self.rdma_buff)} rdma buffers {chunk_sizes=}")

    def update(self, other_buffer):
        super().update(other_buffer)

    # send
    async def read_into(self, tensor=None) -> torch.Tensor:     
        if tensor is None:
            # allocate a tensor to return
            tensor = torch.empty(self.shape, dtype=self.dtype)

        self._assert_safe_tensor(tensor)
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

        self._assert_safe_tensor(tensor)
        assert self.rdma_buff is not None
        

        chunked_byte_view = self._create_byte_views_from_tensor(tensor)

        # if we have tensor refs locally, we're still in the local case,
        # and we're just copying over from the tensor into local memory
        if self.tensor_ref is not None:
            for idx, chunk in enumerate(chunked_byte_view):
                self.tensor_ref[idx].copy_(chunk)            
            return
        # else: we are in the remote case (in a different process), and must read from 
        # the rdma buffer

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
