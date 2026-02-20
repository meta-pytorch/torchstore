# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, TYPE_CHECKING

import torch

from torchstore.transport.buffers import TransportBuffer
from torchstore.transport.types import KeyedRequest

try:
    from torchcomms._transport import RdmaMemory, RdmaRemoteBuffer
except ImportError:
    pass

if TYPE_CHECKING:
    from torchstore.strategy import StorageVolumeRef
    from torchstore.transport.buffers import TransportContext


class TorchCommsRdmaTransportBuffer(TransportBuffer):
    """
    Transport buffer implementation using TorchComms RDMA for efficient tensor transfer.
    """

    def __init__(self, storage_volume_ref: "StorageVolumeRef") -> None:
        super().__init__(storage_volume_ref)

        # local client's rdmatransport address. used by storage volume to retrieve cached peer transport.
        self.address: bytes | None = None

        self.tensor_ref: torch.Tensor | None = (
            None  # reference to local client's destination tensor
        )
        self.rdma_memory: RdmaMemory | None = (
            None  # must be kept alive until transport is done
        )
        self.rdma_remote_buffer: RdmaRemoteBuffer | None = (
            None  # remote reference of rdma memory
        )

        self.shape: torch.Size | None = None
        self.dtype: torch.dtype | None = None

        # Object handling fields (non-tensor data)
        self.is_object: bool = False
        self.objects: Any = None

        # Connection state for handshake
        self._local_transport: Any = None
        self._connection_exists: bool = False

    def _setup_local_transport(self, tensor: torch.Tensor | None) -> None:
        """Get local transport from cache and check if connection exists."""
        device = tensor.device if tensor is not None else 0
        transport_cache = (
            self.storage_volume_ref.transport_context.get_rdma_transport_cache()
        )
        self._connection_exists = transport_cache.contains(
            self.storage_volume_ref.volume_id, device
        )
        self._local_transport, self.address = transport_cache.get(
            self.storage_volume_ref.volume_id, device
        )

    def requires_handshake(self, entries: list[KeyedRequest]) -> bool:
        """Setup transport from request if needed, then check if handshake is required."""
        request = entries[0].request
        if not request.is_object:
            self._setup_local_transport(request.tensor_val)

            return not self._connection_exists
        return False

    async def _post_handshake(
        self,
        handshake_results: list[Any],
        entries: list[KeyedRequest],
    ) -> None:
        """Connect local transport to peer after handshake."""
        self._local_transport.connect(handshake_results[0])

    async def recv_handshake(
        self,
        ctx: "TransportContext",
        entries: list[tuple[KeyedRequest, Any]],
    ) -> list[Any]:
        """Confirm a handshake initiated by the local client (storage volume side)."""
        transport_cache = ctx.get_rdma_transport_cache()
        transport, addr = transport_cache.put(self.address, device=0)
        transport.connect(self.address)
        return [addr]

    def __getstate__(self) -> dict[str, Any]:
        """Serialize the state of the buffer, excluding non-serializable components."""
        state = self.__dict__.copy()
        state["rdma_memory"] = None
        state["tensor_ref"] = None
        state["storage_volume_ref"] = None
        state["_local_transport"] = None
        return state

    def _allocate(self, tensor: torch.Tensor) -> None:
        self.shape = tensor.shape
        self.dtype = tensor.dtype
        self._assert_valid_tensor(tensor, self.dtype, self.shape)
        self.rdma_memory = RdmaMemory(tensor)
        self.rdma_remote_buffer = self.rdma_memory.to_remote_buffer()

    async def _pre_put_hook(self, entries: list[KeyedRequest]) -> None:
        """Allocate RDMA memory for put (transport already set up)."""
        request = entries[0].request
        if request.is_object:
            return
        self._allocate(request.tensor_val)

    async def _pre_get_hook(self, entries: list[KeyedRequest]) -> None:
        """Fetch metadata if needed and allocate RDMA buffers."""
        key, request = entries[0]
        tensor_like = request.tensor_val
        if tensor_like is None:
            meta = await self.storage_volume_ref.volume.get_meta.call_one(
                key, request.meta_only()
            )
            if isinstance(meta, str) or meta is None:
                return  # Objects don't need RDMA setup
            if request.tensor_slice is not None:
                meta = (request.tensor_slice.local_shape, *meta[1:])
            tensor_like = meta

        if isinstance(tensor_like, tuple):
            self.tensor_ref = torch.zeros(
                tensor_like[0], dtype=tensor_like[1], device=torch.device("cpu")
            )
        else:
            assert isinstance(tensor_like, torch.Tensor)
            self.tensor_ref = tensor_like

        self._allocate(self.tensor_ref)

    async def handle_put_request(
        self,
        ctx: "TransportContext",
        entries: list[tuple[KeyedRequest, Any]],
    ) -> dict[str, Any]:
        """Called by storage volume. Read from client's source RdmaMemory (put)."""
        (key, request), maybe_tensor = entries[0]

        if request.is_object:
            return {key: request.objects}

        if maybe_tensor is None:
            maybe_tensor = torch.zeros(
                self.shape, dtype=self.dtype, device=torch.device("cpu")
            )

        assert self.rdma_remote_buffer is not None
        self._assert_valid_tensor(maybe_tensor, self.dtype, self.shape)

        transport_cache = ctx.get_rdma_transport_cache()
        transport = transport_cache.get(self.address, 0)[0]

        receiving_buffer = RdmaMemory(maybe_tensor)
        res = transport.read(
            receiving_buffer.to_mutable_view(), self.rdma_remote_buffer
        )
        assert res == 0, f"RDMA read failed: conn code {res}"

        return {key: maybe_tensor}

    async def handle_get_request(
        self,
        ctx: "TransportContext",
        entries: list[tuple[KeyedRequest, Any]],
    ) -> None:
        """Called by storage volume. Write to client's dest RdmaMemory (get)."""
        _entry, data = entries[0]
        if not isinstance(data, torch.Tensor):
            self.is_object = True
            self.objects = data
            return

        tensor = data

        if not tensor.is_contiguous():
            contiguous_buffer = torch.zeros_like(
                tensor,
                device="cpu",
                memory_format=torch.contiguous_format,
            )
            contiguous_buffer.copy_(tensor)
            tensor = contiguous_buffer

        assert self.rdma_remote_buffer is not None
        self._assert_valid_tensor(tensor, self.dtype, self.shape)
        rdma_memory = RdmaMemory(tensor)

        transport_cache = ctx.get_rdma_transport_cache()
        transport, _ = transport_cache.get(self.address, 0)
        res = transport.write(rdma_memory.to_view(), self.rdma_remote_buffer)
        assert res == 0, f"RDMA write failed: conn code {res}"

    async def _handle_storage_volume_response(
        self, transport_buffer: "TransportBuffer"
    ) -> list[Any]:
        """Extract data from response buffer on client side."""
        if transport_buffer.is_object:
            return [transport_buffer.objects]

        # Data was written directly into self.tensor_ref via RDMA
        return [self.tensor_ref]

    async def drop(self) -> None:
        """Clean up any resources held by this buffer."""
        if self.rdma_remote_buffer is not None:
            del self.rdma_remote_buffer
            self.rdma_remote_buffer = None
        if self.rdma_memory is not None:
            del self.rdma_memory
            self.rdma_memory = None
        self.tensor_ref = None
