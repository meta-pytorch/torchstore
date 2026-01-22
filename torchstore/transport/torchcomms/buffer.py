# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional, TYPE_CHECKING

import torch
from torchstore.transport.buffers import TransportBuffer

try:
    from torchcomms._transport import RdmaMemory, RdmaRemoteBuffer
except ImportError:
    pass

if TYPE_CHECKING:
    from torchstore.strategy import StorageVolumeRef
    from torchstore.transport.buffers import TransportContext
    from torchstore.transport.types import Request


class TorchCommsRdmaTransportBuffer(TransportBuffer):
    """
    Transport buffer implementation using TorchComms RDMA for efficient tensor transfer.
    """

    requires_handshake: bool = True

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

        self.shape: Optional[torch.Size] = None
        self.dtype: Optional[torch.dtype] = None

        # Object handling fields (non-tensor data)
        self.is_object: bool = False
        self.objects: Any = None

        # Connection state for handshake
        self._local_transport: Any = None
        self._connection_exists: bool = False

    def _setup_local_transport(self, tensor: Optional[torch.Tensor]) -> None:
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

    async def _should_handshake(self, request: "Request") -> bool:
        """Only handshake if RDMA was set up and connection is not already cached."""
        if self.address is None:
            return False
        return not self._connection_exists

    async def _post_handshake(self, handshake_result: Any) -> None:
        """Connect local transport to peer after handshake."""
        self._local_transport.connect(handshake_result)

    async def recv_handshake(
        self, transport_context: "TransportContext"
    ) -> Optional[Any]:
        """Confirm a handshake initiated by the local client (storage volume side)."""
        transport_cache = transport_context.get_rdma_transport_cache()
        transport, addr = transport_cache.put(self.address, device=0)
        transport.connect(self.address)
        return addr

    def __getstate__(self) -> Dict[str, Any]:
        """Serialize the state of the buffer, excluding non-serializable components."""
        state = self.__dict__.copy()
        state["rdma_memory"] = None
        state["tensor_ref"] = None
        state["storage_volume_ref"] = None
        state["_local_transport"] = None
        return state

    def _allocate(self, tensor: torch.Tensor) -> None:
        self._assert_valid_tensor(tensor, self.dtype, self.shape)
        self.rdma_memory = RdmaMemory(tensor)
        self.rdma_remote_buffer = self.rdma_memory.to_remote_buffer()

    async def _pre_put_hook(self, request: "Request") -> None:
        """Prepare buffers before sending put request (client-side)."""
        if request.is_object or request.tensor_val is None:
            return

        tensor = request.tensor_val
        self._setup_local_transport(tensor)

        # allocate_source logic
        self.shape = tensor.shape
        self.dtype = tensor.dtype
        self._allocate(tensor)

    async def _pre_get_hook(self, key: str, request: "Request") -> None:
        """Prepare buffers before sending get request (client-side)."""
        # Fetch metadata if no tensor provided
        tensor_like = request.tensor_val
        if tensor_like is None:
            meta = await self.storage_volume_ref.volume.get_meta.call_one(
                key, request.meta_only()
            )
            if isinstance(meta, str) or meta is None:
                return  # Objects don't need RDMA setup
            tensor_like = meta  # (shape, dtype) tuple

        # Setup local transport - use tensor device if available, else use default
        self._setup_local_transport(
            tensor_like if isinstance(tensor_like, torch.Tensor) else None
        )

        # allocate_dest logic
        if isinstance(tensor_like, tuple):
            self.tensor_ref = torch.zeros(
                tensor_like[0], dtype=tensor_like[1], device=torch.device("cpu")
            )
            self.shape, self.dtype = tensor_like
        else:
            assert isinstance(tensor_like, torch.Tensor)
            self.tensor_ref = tensor_like
            self.shape, self.dtype = tensor_like.shape, tensor_like.dtype

        self._allocate(self.tensor_ref)

    async def handle_put_request(
        self,
        request: "Request",
        current_object,
        storage_transport_context: "TransportContext",
    ) -> Any:
        """Called by storage volume. Read from client's source RdmaMemory (put)."""
        if request.is_object:
            return request.objects

        tensor = current_object
        if tensor is None:
            tensor = torch.zeros(
                self.shape, dtype=self.dtype, device=torch.device("cpu")
            )

        assert self.rdma_remote_buffer is not None
        self._assert_valid_tensor(tensor, self.dtype, self.shape)

        transport_cache = storage_transport_context.get_rdma_transport_cache()
        transport = transport_cache.get(self.address, 0)[0]

        receiving_buffer = RdmaMemory(tensor)
        res = transport.read(
            receiving_buffer.to_mutable_view(), self.rdma_remote_buffer
        )
        assert res == 0, f"RDMA read failed: conn code {res}"

        return tensor

    async def handle_get_request(
        self, data, storage_transport_context: "TransportContext"
    ) -> None:
        """Called by storage volume. Write to client's dest RdmaMemory (get)."""
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

        transport_cache = storage_transport_context.get_rdma_transport_cache()
        transport, _ = transport_cache.get(self.address, 0)
        res = transport.write(rdma_memory.to_view(), self.rdma_remote_buffer)
        assert res == 0, f"RDMA write failed: conn code {res}"

    async def _handle_storage_volume_response(
        self, transport_buffer: "TransportBuffer"
    ) -> Any:
        """Extract data from response buffer on client side."""
        if hasattr(transport_buffer, "is_object") and transport_buffer.is_object:
            return transport_buffer.objects

        # Data was written directly into self.tensor_ref via RDMA
        return self.tensor_ref

    async def drop(self) -> None:
        """Clean up any resources held by this buffer."""
        if self.rdma_remote_buffer is not None:
            del self.rdma_remote_buffer
            self.rdma_remote_buffer = None
        if self.rdma_memory is not None:
            del self.rdma_memory
            self.rdma_memory = None
        self.tensor_ref = None
