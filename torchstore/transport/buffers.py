# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING, Union

import torch

from torchstore.transport.torchcomms.cache import RdmaTransportCache

if TYPE_CHECKING:
    from torchstore.transport.pipe import Request, StorageVolumeRef


def rdma_available() -> bool:
    rdma_enabled = (
        os.environ.get("TORCHSTORE_RDMA_ENABLED", "1") == "1"
    )  # TODO: enable on this build
    return rdma_enabled and monarch_rdma_available()


class TransportContext:
    RDMA_TRANSPORT_CACHE = "rdma_transport_cache"

    def __init__(self):
        self.transport_context = {}

    def get_transport_context(self) -> Dict[Any, Any]:
        return self.transport_context

    def get_rdma_transport_cache(self) -> RdmaTransportCache:
        if self.RDMA_TRANSPORT_CACHE not in self.transport_context:
            self.transport_context[self.RDMA_TRANSPORT_CACHE] = RdmaTransportCache()
        return self.transport_context[self.RDMA_TRANSPORT_CACHE]


class TransportBuffer:

    requires_handshake: bool = False

    def __init__(self, storage_volume_ref: "StorageVolumeRef"):
        self.storage_volume_ref = storage_volume_ref

    # Client-side interface. Called by the client to send/recv data to the storage volume.
    async def put_to_storage_volume(self, key, request: "Request"):
        try:
            # _give concrete implementaiton a chance to parse the request
            await self._pre_put_hook(request)

            if self.requires_handshake:
                await self.storage_volume_ref.volume.handshake.call(self)

            await self.storage_volume_ref.volume.put.call(key, self, request)
        finally:
            self.drop()

    async def get_from_storage_volume(self, key, request: "Request"):
        try:
            await self._pre_get_hook(key, request)

            if self.requires_handshake:
                self.storage_volume_ref.volume.handshake.call(self)

            # when fetching data, we may need to handle the response from the storage volume
            # TODO: think of a good prefix to differentiate this between remote handlers
            response = await self._handle_storage_volume_response(
                await self.storage_volume_ref.volume.get.call_one(key, self, request)
            )
        finally:
            self.drop()

        return response

    async def _drop(self, response: Any):
        pass

    async def _pre_put_hook(self, request: "Request"):
        pass

    async def _pre_get_hook(self, key:str, request: "Request"):
        pass

    async def _handle_storage_volume_response(self, response: Any) -> Any:
        raise NotImplementedError()

    # StorageVolume handlers -- must be implemented by concrete implementaiton
    # These methods are called by the StorageVolume on the remote side

    async def handle_handshake_request(self):
        # called on the storage volume side
        raise NotImplementedError()

    async def handle_put_request(self, key, request: "Request"):
        # called on the storage volume side
        raise NotImplementedError()

    async def handle_get_request(self, key, request: "Request"):
        # called on the storage volume side
        raise NotImplementedError()

    # Helper methods
    def _assert_valid_tensor(
        self,
        tensor: torch.Tensor,
        dtype: torch.dtype,
        shape: torch.Size,
        must_be_contiguous=False,
    ) -> None:
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == dtype, f"{tensor.dtype} != {dtype}"
        assert tensor.shape == shape, f"{tensor.shape} != {shape}"
        assert not must_be_contiguous or tensor.is_contiguous()


# class TransportBuffer:
#     finalize: bool = False
#     is_object: bool = False
#     objects: Optional[Any] = None
#     requires_meta: bool = False

#     def update(self, other_buffer: "TransportBuffer") -> None:
#         self.finalize = other_buffer.finalize
#         self.is_object = other_buffer.is_object
#         self.objects = other_buffer.objects
#         self.requires_meta = other_buffer.requires_meta

#     def allocate(self, tensor_like: Union[torch.Tensor, Tuple]) -> None:
#         """Allocates internal buffers based on either an existing tensor
#         or a Tuple of (shape, dtype)
#         """
#         raise NotImplementedError()

#     async def read_into(
#         self, tensor: Optional[torch.Tensor], transport_context: TransportContext
#     ) -> torch.Tensor:
#         raise NotImplementedError()

#     async def write_from(
#         self, tensor: Optional[torch.Tensor], transport_context: TransportContext
#     ) -> None:
#         raise NotImplementedError()

#     async def handshake(
#         self, tensor: torch.Tensor, volume_ref: "StorageVolumeRef"
#     ) -> None:
#         """Establish a handshake with the remote volume, such as for RDMA."""
#         pass

#     async def recv_handshake(
#         self, transport_context: TransportContext
#     ) -> Optional[Any]:
#         """Confirm a handshake initiated by the local client, and return some result."""
#         pass

#     async def drop(self) -> None:
#         """Clean up any resources held by this buffer. Override in subclasses if needed."""
#         pass

#     def _assert_valid_tensor(
#         self,
#         tensor: torch.Tensor,
#         dtype: torch.dtype,
#         shape: torch.Size,
#         must_be_contiguous=False,
#     ) -> None:
#         assert isinstance(tensor, torch.Tensor)
#         assert tensor.dtype == dtype, f"{tensor.dtype} != {dtype}"
#         assert tensor.shape == shape, f"{tensor.shape} != {shape}"
#         assert not must_be_contiguous or tensor.is_contiguous()


class MonarchTransportBuffer(TransportBuffer):
    """This interface is mostly a noop, intended to be used with Monarch's regular RPC.
    Not expected to be super fast, but always works.
    """

    finalize: bool = True

    def __init__(self) -> None:
        self.tensor: Optional[torch.Tensor] = None

    def allocate(self, tensor_like: Union[torch.Tensor, Tuple]) -> None:
        """In the case of using monarch comms, we don't do any allocation ahead of time"""
        return None

    # send
    async def read_into(
        self, tensor: Optional[torch.Tensor], transport_context: TransportContext
    ) -> torch.Tensor:
        if tensor is not None:
            # if there is a tensor here, likely this is the 'inplace' case,
            # and we should return back a ptr to the original tensor
            # (as opposed to the stored tensor, which we likely don't want to
            # keep around)
            tensor.copy_(self.tensor)
            return tensor

        return self.tensor

    # recv
    async def write_from(
        self, tensor: Optional[torch.Tensor], transport_context: TransportContext
    ) -> None:
        self.tensor = tensor

    def update(self, other_buffer: "TransportBuffer") -> None:
        super().update(other_buffer)
        self.tensor = other_buffer.tensor
