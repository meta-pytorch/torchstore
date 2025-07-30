from typing import Any, Optional, Any
import torch
from torch._prims_common import ShapeType
from monarch.tensor_engine import is_available as rdma_available, RDMABuffer


class Message:
    def __init__(self, value) -> None:
        self.value = value

    @classmethod
    async def pack(cls, value: Any) -> "Message":
        return cls(value)

    def unpack(self, inplace_tensor=None) -> Any:
        assert inplace_tensor is None, "unpack does not support inplace"
        return self.value

class RDMAMessage(Message):
    def __init__(
        self,
        value:Any,
        t_shape: Optional[ShapeType]=None,
        t_dtype: Optional[torch.dtype]=None,
        timeout=10
    ):
        self.value = value
        self.t_shape = t_shape
        self.t_dtype = t_dtype
        self.timeout = timeout

    @classmethod
    async def pack(cls, value, timeout=10) -> "RDMAMessage":
        assert rdma_available(), "RDMATransport requires RDMA support"
        if not isinstance(value, torch.Tensor):
            #TODO: consider pickling and sending as raw bytes, which could
            # be helpful for large non-tensor objects
            return cls(value, timeout=timeout)

        # it's important that this tensor is not GC'd before we call
        # `unpack` ont the other side.
        byte_tensor = value.view(torch.uint8).flatten()
        buffer = RDMABuffer(byte_tensor)        
        return RDMAMessage(buffer, value.shape, value.dtype, timeout)

    async def unpack(self, inplace_tensor=None) -> Any:
        if not isinstance(self.value, RDMABuffer):
            return self.value

        if inplace_tensor is None:
            inplace_tensor = torch.empty(self.t_shape, dtype=self.t_dtype)
        else:
            assert inplace_tensor.shape == self.t_shape, "inplace tensor shape mismatch"
            assert inplace_tensor.dtype == self.t_dtype, "inplace tensor dtype mismatch"

        byte_tensor = inplace_tensor.view(torch.uint8).flatten()
        await self.value.read_into(byte_tensor, timeout=self.timeout)
        return inplace_tensor
