# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from typing import Any, TYPE_CHECKING

import torch

from torchstore.transport.buffers import TransportBuffer

try:
    from torchcomms._transport import RdmaMemory, RdmaRemoteBuffer
except ImportError:
    pass

if TYPE_CHECKING:
    from torchstore.transport.buffers import TransportContext
    from torchstore.transport.pipe import StorageVolumeRef


# Environment variable to control GPU Direct RDMA
# Set to "1" to enable GPU-to-GPU RDMA transfers (requires GPUDirect RDMA support)
# Set to "0" to force CPU-only transfers (safer, always works)
GPU_DIRECT_RDMA_ENABLED = os.environ.get("TORCHSTORE_GPU_DIRECT_RDMA", "1") == "1"

# Environment variable to enable GPU storage on storage volume side
# WARNING: This can cause OOM if trainer GPU doesn't have enough memory
# Set to "1" to enable (faster ~4s, but risk OOM)
# Set to "0" to disable (safer ~10s, uses CPU for storage)
GPU_STORAGE_ENABLED = os.environ.get("TORCHSTORE_GPU_STORAGE", "0") == "1"

logger = logging.getLogger(__name__)


class TorchCommsRdmaTransportBuffer(TransportBuffer):
    """Transport buffer implementation using TorchComms RDMA for efficient tensor transfer.

    Supports GPU Direct RDMA when TORCHSTORE_GPU_DIRECT_RDMA=1 (default).
    This enables zero-copy GPU-to-GPU transfers, reducing weight update time
    from ~20s to ~2s for 8B models.
    """

    requires_meta: bool = True

    def __init__(self, target_device: torch.device | None = None) -> None:
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

        # Target device for allocation (GPU Direct RDMA support)
        # If None, will be determined based on GPU_DIRECT_RDMA_ENABLED
        self.target_device = target_device

    async def handshake(
        self, tensor: torch.Tensor, volume_ref: "StorageVolumeRef"
    ) -> None:
        """
        Establish an RDMA handshake with the storage volume, and save the local RdmaTransport address.
        """
        device = tensor.device if tensor is not None else 0
        transport_cache = volume_ref.transport_context.get_rdma_transport_cache()
        connection_exists = transport_cache.contains(volume_ref.volume_id, device)
        local_transport, self.address = transport_cache.get(
            volume_ref.volume_id, device
        )

        if connection_exists:
            return

        peer_addr = await volume_ref.volume.handshake.call_one(self)
        local_transport.connect(peer_addr)

    async def recv_handshake(self, transport_context: "TransportContext") -> Any | None:
        """
        Confirm a handshake initiated by the local client.
        """
        transport_cache = transport_context.get_rdma_transport_cache()
        transport, addr = transport_cache.put(self.address, device=0)
        transport.connect(self.address)
        return addr

    def __getstate__(self) -> dict[str, Any]:
        """
        Serialize the state of the buffer, including RdmaRemoteBuffer but excluding the RdmaMemory and local dest tensor ref.
        """
        state = self.__dict__.copy()
        state["rdma_memory"] = None
        state["tensor_ref"] = None
        return state

    def _allocate(self, tensor: torch.Tensor) -> None:
        self._assert_valid_tensor(tensor, self.dtype, self.shape)
        self.rdma_memory = RdmaMemory(tensor)
        self.rdma_remote_buffer = self.rdma_memory.to_remote_buffer()

    def _get_allocation_device(self) -> torch.device:
        """Determine the device to use for tensor allocation.

        Returns GPU device if:
        1. GPU Direct RDMA is enabled (TORCHSTORE_GPU_DIRECT_RDMA=1)
        2. CUDA is available
        3. No explicit target_device was set to CPU

        Otherwise returns CPU.
        """
        if self.target_device is not None:
            return self.target_device

        if GPU_DIRECT_RDMA_ENABLED and torch.cuda.is_available():
            # Use the current CUDA device for GPU Direct RDMA
            device = torch.device("cuda", torch.cuda.current_device())
            logger.debug(f"GPU Direct RDMA enabled, allocating on {device}")
            return device
        else:
            return torch.device("cpu")

    def allocate_dest(self, tensor_like: torch.Tensor | tuple) -> None:
        """Called by the local client. Allocate RdmaMemory for the destination tensor (get).

        When GPU Direct RDMA is enabled and CUDA is available, tensors are allocated
        directly on GPU memory, enabling zero-copy RDMA transfers.
        """
        if isinstance(tensor_like, str) or tensor_like is None:
            return
        elif isinstance(tensor_like, tuple):
            device = self._get_allocation_device()
            self.tensor_ref = torch.zeros(
                tensor_like[0], dtype=tensor_like[1], device=device
            )
            self.shape, self.dtype = tensor_like
            if device.type == "cuda":
                logger.info(
                    f"[GPU Direct RDMA] Allocated destination tensor on {device}, "
                    f"shape={tensor_like[0]}, dtype={tensor_like[1]}"
                )
        else:
            # Use the inplace tensor directly (preserves original device)
            assert isinstance(tensor_like, torch.Tensor)
            self.tensor_ref = tensor_like
            self.shape, self.dtype = tensor_like.shape, tensor_like.dtype

        self._allocate(self.tensor_ref)

    # TODO @amirafzali: add test case and support for non-contiguous input
    def allocate_source(self, tensor: torch.Tensor | None) -> None:
        """Called by the local client. Allocate RdmaMemory for the source tensor (put)."""
        if tensor is None:
            return

        self.shape = tensor.shape
        self.dtype = tensor.dtype

        self._allocate(tensor)

    async def read_into(
        self, tensor: torch.Tensor | None, transport_context: "TransportContext"
    ) -> torch.Tensor:
        """Called by the remote storage volume. Read from the local client's source RdmaMemory (put)

        Note: This is called on the STORAGE VOLUME side. By default uses CPU to avoid OOM.
        Set TORCHSTORE_GPU_STORAGE=1 to enable GPU storage for faster transfers (~4s vs ~10s).
        """
        if tensor is None:
            # Use GPU storage only if explicitly enabled (for max performance)
            if GPU_STORAGE_ENABLED and torch.cuda.is_available():
                device = torch.device("cuda", torch.cuda.current_device())
            else:
                device = torch.device("cpu")
            tensor = torch.zeros(
                self.shape, dtype=self.dtype, device=device
            )

        assert self.rdma_remote_buffer is not None
        self._assert_valid_tensor(tensor, self.dtype, self.shape)

        # Get the transport for the tensor's device
        device_index = tensor.device.index if tensor.device.type == "cuda" else 0
        transport_cache = transport_context.get_rdma_transport_cache()
        transport = transport_cache.get(self.address, device_index)[0]

        receiving_buffer = RdmaMemory(tensor)
        res = transport.read(
            receiving_buffer.to_mutable_view(), self.rdma_remote_buffer
        )
        assert res == 0, f"RDMA read failed: conn code {res}"

        return tensor

    async def write_from(
        self, tensor: torch.Tensor | None, transport_context: "TransportContext"
    ) -> None:
        """Called by the remote storage volume. Write to the local client's dest RdmaMemory (get)"""
        if tensor is None:
            return

        if not tensor.is_contiguous():
            # Make contiguous on the same device as the tensor
            contiguous_buffer = torch.zeros_like(
                tensor,
                device=tensor.device,  # Preserve device (GPU or CPU)
                memory_format=torch.contiguous_format,
            )
            contiguous_buffer.copy_(tensor)
            tensor = contiguous_buffer

        assert self.rdma_remote_buffer is not None
        self._assert_valid_tensor(tensor, self.dtype, self.shape)
        rdma_memory = RdmaMemory(tensor)

        # Get the transport for the tensor's device
        device_index = tensor.device.index if tensor.device.type == "cuda" else 0
        transport_cache = transport_context.get_rdma_transport_cache()
        transport, _ = transport_cache.get(self.address, device_index)
        res = transport.write(rdma_memory.to_view(), self.rdma_remote_buffer)
        assert res == 0, f"RDMA write failed: conn code {res}"

    async def drop(self) -> None:
        """Clean up any resources held by this buffer."""
        del self.rdma_remote_buffer
        del self.rdma_memory
        self.tensor_ref = None

    def _assert_valid_tensor(
        self,
        tensor: torch.Tensor,
        dtype: torch.dtype,
        shape: torch.Size,
        must_be_contiguous: bool = False,
    ) -> None:
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == dtype, f"{tensor.dtype} != {dtype}"
        assert tensor.shape == shape, f"{tensor.shape} != {shape}"
        assert not must_be_contiguous or tensor.is_contiguous()
