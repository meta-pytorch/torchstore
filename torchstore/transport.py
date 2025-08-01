from dataclasses import dataclass
from typing import Any, Optional, Any, Tuple

import torch
from torch._prims_common import ShapeType
from monarch.tensor_engine import is_available as rdma_available, RDMABuffer


@dataclass
class DTensorPack:
    offsets: Tuple
    coordinates: Tuple
    local_tensor: torch.Tensor
    global_shape: Tuple
    mesh_shape: Tuple

    def __post_init__(self):        
        self.coordinates = tuple(self.coordinates)

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
    """Naive implementation of RDMABuffer as a transport which is un-optimized and
    implements multiple unnecessary allocations. Since this is using rdma transport,
    it's essential that the tensor is not GC'd before the other side has unpacked it.
    """

    def __init__(
        self,
        value:Any,
        t_shape: Optional[ShapeType]=None,
        t_dtype: Optional[torch.dtype]=None,
        tensor_ref: Optional[torch.Tensor]=None,
        timeout=10
    ):
        self.value = value
        self.t_shape = t_shape
        self.t_dtype = t_dtype
        self.timeout = timeout
        self.tensor_ref = tensor_ref
    
    def __reduce__(self) -> str | tuple[Any, ...]:
        """This is a hack which relies on the fact that the RDMAMessage is not usually gc'd locally until 
        we actually 'await' the unpack call in the endpoint call. This is useful for the case where the tensor
        would normally be GC'd but the message is still in flight. 

        e.g.
        local_tensor = torch.rand(4,4)
        
        # since local_tensor[0] is a view, RDMAMEssage.pack will copy it, and keep an internal reference
        # in the following line. 
        message = RDMAMEssage.pack(local_tensor[0])         
        await self.client.get.call_one(key, message)        

        TODO: We might want to reconsider this logic when monarch supports views.
        """

        # remove the reference of the tensor since we don't actually want to use that.
        return (self.__class__, (self.value, self.t_shape, self.t_dtype, None, self.timeout))


    @classmethod
    def _create_rdma_buffer(cls, tensor: torch.Tensor) -> RDMABuffer:
        # practically this occurs every time your tensor is actually a view of a global tensor
        # monarch doesn't support views yet, so for now we copy although this is not ideal
        # TODO: see if we an support RDMABufer from views
        # if tensor.storage_offset() != 0:
        #     print(f"Storage offset is not 0!!!! {tensor.storage_offset()=}")

        # tensor = tensor.clone().detach()

        if not tensor.is_contiguous():
            # tensor = tensor.contiguous()
            raise RuntimeError("RDMATransport only supports contiguous tensors")
        byte_tensor = tensor.view(torch.uint8).flatten()
        print(f"Packed byte_tensor, {byte_tensor=} {tensor.storage_offset()=}", flush=True)
        return RDMABuffer(byte_tensor, offset_=tensor.storage_offset()*tensor.element_size()), tensor

    @classmethod
    async def pack(cls, value, timeout=10) -> "RDMAMessage":
        # import fbvscode
        # fbvscode.set_trace()

        assert rdma_available(), "RDMATransport requires RDMA support"
        if not isinstance(value, (torch.Tensor, DTensorPack)):
            #TODO: consider pickling and sending as raw bytes, which could
            # be helpful for large non-tensor objects
            return cls(value, timeout=timeout)

        if isinstance(value, DTensorPack):
            buffer, tensor_ref = cls._create_rdma_buffer(value.local_tensor)
            dtensor_pack = DTensorPack(
                value.offsets,
                value.coordinates,
                buffer,
                value.global_shape,
                value.mesh_shape,
            )

            return cls(dtensor_pack, tensor_ref.shape, tensor_ref.dtype, tensor_ref=tensor_ref, timeout=timeout)

        #TODO (critical): it's important that this tensor is not GC'd before we call
        # `unpack` ont the other side.
        assert isinstance(value, torch.Tensor)
        buffer, tensor_ref = cls._create_rdma_buffer(value)
        return cls(buffer, value.shape, value.dtype, tensor_ref=tensor_ref, timeout=timeout)

    async def unpack(self, inplace_tensor=None) -> Any:
        # import fbvscode
        # fbvscode.set_trace()

        if not isinstance(self.value, (RDMABuffer, DTensorPack)):
            return self.value

        if inplace_tensor is None:
            inplace_tensor = torch.empty(self.t_shape, dtype=self.t_dtype)
        else:
            assert inplace_tensor.shape == self.t_shape, "inplace tensor shape mismatch"
            assert inplace_tensor.dtype == self.t_dtype, "inplace tensor dtype mismatch"

        if isinstance(self.value, DTensorPack):
            byte_tensor = inplace_tensor.view(torch.uint8).flatten()
            await self.value.local_tensor.read_into(byte_tensor, timeout=self.timeout)
            return DTensorPack(
                self.value.offsets,
                self.value.coordinates,
                inplace_tensor,
                self.value.global_shape,
                self.value.mesh_shape,
            )

        byte_tensor = inplace_tensor.view(torch.uint8).flatten()
        print(f"{byte_tensor.storage_offset()=}", flush=True)
        print(f"{self.value._buffer}=")
        import fbvscode
        # fbvscode.set_trace()
        await self.value.read_into(byte_tensor, timeout=self.timeout)
        print(f"unpacked {byte_tensor=}", flush=True)
        import time
        import random
        time.sleep(random.randint(1,3))

        # print(f"unpacked, {inplace_tensor=}")
        return inplace_tensor
