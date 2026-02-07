# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CUDA IPC transport buffer for intra-node GPU-to-GPU transfers.

This transport uses CUDA IPC (Inter-Process Communication) to enable direct
GPU-to-GPU memory access between processes on the same node, bypassing CPU
staging entirely.

Requirements:
- Single-node deployment (CUDA IPC doesn't work across nodes)
- CUDA-capable GPUs with P2P access (NVLink or PCIe P2P)
- All processes must have access to the same GPUs

Performance characteristics:
- Eliminates GPU->CPU->GPU copies
- Uses NVLink bandwidth when available (~600 GB/s for NVLink 4.0)
- Falls back to PCIe P2P if NVLink not available (~32 GB/s for PCIe 4.0 x16)
"""

import logging
import os
import time
import uuid
import weakref
from typing import Any, TYPE_CHECKING

import torch
from torch.multiprocessing.reductions import rebuild_cuda_tensor

from torchstore.transport.buffers import TransportBuffer

# Enable detailed tracing with TORCHSTORE_TRACE_TRANSFERS=1
TRACE_TRANSFERS = os.environ.get("TORCHSTORE_TRACE_TRANSFERS", "0") == "1"
_trace_logger = logging.getLogger("torchstore.trace")

if TYPE_CHECKING:
    from torchstore.strategy import StorageVolumeRef
    from torchstore.transport.buffers import TransportContext
    from torchstore.transport.types import Request


def cuda_ipc_available() -> bool:
    """Check if CUDA IPC is available on this system."""
    if not torch.cuda.is_available():
        return False
    # CUDA IPC requires at least one GPU
    if torch.cuda.device_count() < 1:
        return False
    # Check if IPC handles can be created
    try:
        t = torch.zeros(1, device="cuda:0")
        handle = t._typed_storage()._share_cuda_()
        return handle is not None
    except Exception:
        return False


class CudaIPCHandle:
    """Serializable CUDA IPC handle for cross-process GPU memory sharing.

    This class wraps the CUDA IPC handle in a format that can be pickled
    and sent via Monarch RPC, then used to reconstruct a tensor on the
    receiving process.

    The handle format matches PyTorch's internal CUDA tensor serialization
    used in torch.multiprocessing.
    """

    def __init__(
        self,
        tensor_size: tuple,
        tensor_stride: tuple,
        tensor_offset: int,
        dtype: torch.dtype,
        storage_device: int,
        storage_handle: bytes,
        storage_size_bytes: int,
        storage_offset_bytes: int,
        requires_grad: bool,
        ref_counter_handle: bytes,
        ref_counter_offset: int,
        event_handle: bytes,
        event_sync_required: bool,
    ):
        self.tensor_size = tensor_size
        self.tensor_stride = tensor_stride
        self.tensor_offset = tensor_offset
        self.dtype = dtype
        self.storage_device = storage_device
        self.storage_handle = storage_handle
        self.storage_size_bytes = storage_size_bytes
        self.storage_offset_bytes = storage_offset_bytes
        self.requires_grad = requires_grad
        self.ref_counter_handle = ref_counter_handle
        self.ref_counter_offset = ref_counter_offset
        self.event_handle = event_handle
        self.event_sync_required = event_sync_required

    def reconstruct_tensor(self) -> torch.Tensor:
        """Reconstruct the tensor on the receiving process using CUDA IPC."""
        try:
            # Use PyTorch's official rebuild function
            return rebuild_cuda_tensor(
                torch.Tensor,  # tensor_cls
                self.tensor_size,
                self.tensor_stride,
                self.tensor_offset,
                torch.storage.TypedStorage,  # storage_cls
                self.dtype,
                self.storage_device,
                self.storage_handle,
                self.storage_size_bytes,
                self.storage_offset_bytes,
                self.requires_grad,
                self.ref_counter_handle,
                self.ref_counter_offset,
                self.event_handle,
                self.event_sync_required,
            )
        except Exception as e:
            # Log the error with context for debugging
            _trace_logger.error(
                f"Failed to reconstruct CUDA IPC tensor: {e}. "
                f"Handle details - device: {self.storage_device}, "
                f"size: {self.tensor_size}, dtype: {self.dtype}"
            )
            # Clean up any partial resources - the underlying CUDA handles
            # will be cleaned up by PyTorch's garbage collection
            raise RuntimeError(f"CUDA IPC tensor reconstruction failed: {e}") from e


def create_ipc_handle(tensor: torch.Tensor) -> CudaIPCHandle:
    """Create a CUDA IPC handle for a GPU tensor.

    The tensor must be on a CUDA device and contiguous.
    """
    if not tensor.is_cuda:
        raise ValueError("Tensor must be on a CUDA device for IPC")
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

    # Get storage and its IPC handle
    storage = tensor._typed_storage()
    (
        device,
        handle,
        storage_size_bytes,
        storage_offset_bytes,
        ref_counter_handle,
        ref_counter_offset,
        event_handle,
        event_sync_required,
    ) = storage._share_cuda_()

    return CudaIPCHandle(
        tensor_size=tuple(tensor.size()),
        tensor_stride=tuple(tensor.stride()),
        tensor_offset=tensor.storage_offset(),
        dtype=tensor.dtype,
        storage_device=device,
        storage_handle=handle,
        storage_size_bytes=storage_size_bytes,
        storage_offset_bytes=storage_offset_bytes,
        requires_grad=tensor.requires_grad,
        ref_counter_handle=ref_counter_handle,
        ref_counter_offset=ref_counter_offset,
        event_handle=event_handle,
        event_sync_required=event_sync_required,
    )


class CudaIPCTransportBuffer(TransportBuffer):
    """Transport buffer using CUDA IPC for GPU-direct transfers.

    This transport keeps tensor data on GPU and uses CUDA IPC handles
    to enable cross-process GPU memory access without CPU copies.

    Architecture:
    - PUT: Source creates IPC handle, sends handle via RPC, dest reconstructs tensor
    - GET: Storage creates IPC handle, sends handle via RPC, client reconstructs tensor
    """

    requires_handshake: bool = False

    # Class-level registry to prevent tensor deallocation during IPC transfers
    _active_tensors: dict[str, torch.Tensor] = {}
    _tensor_refs: dict[str, weakref.ReferenceType] = {}

    def __init__(self, storage_volume_ref: "StorageVolumeRef"):
        super().__init__(storage_volume_ref)

        # For PUT: IPC handle to be sent to storage volume
        self.ipc_handle: CudaIPCHandle | None = None

        # Metadata
        self.shape: torch.Size | None = None
        self.dtype: torch.dtype | None = None

        # Object handling (non-tensor data)
        self.is_object: bool = False
        self.objects: Any = None

        # Keep reference to source tensor to prevent premature deallocation
        self._source_tensor: torch.Tensor | None = None

        # Unique ID for tracking tensor lifetime in the registry
        self._tensor_id: str | None = None

        # Request stored for get operations
        self.request: Any = None

    def _register_tensor(self, tensor: torch.Tensor) -> str:
        """Register tensor in class registry to prevent deallocation during IPC transfer."""
        tensor_id = str(uuid.uuid4())

        # Store strong reference to prevent deallocation
        self._active_tensors[tensor_id] = tensor

        # Also store weak reference for cleanup detection
        def cleanup_callback(ref):
            self._active_tensors.pop(tensor_id, None)
            self._tensor_refs.pop(tensor_id, None)
            if TRACE_TRANSFERS:
                _trace_logger.debug(
                    f"[CUDA_IPC] Cleaned up tensor registry entry {tensor_id}"
                )

        self._tensor_refs[tensor_id] = weakref.ref(tensor, cleanup_callback)

        if TRACE_TRANSFERS:
            _trace_logger.debug(
                f"[CUDA_IPC] Registered tensor {tensor_id} in lifetime registry"
            )

        return tensor_id

    def _unregister_tensor(self, tensor_id: str | None) -> None:
        """Unregister tensor from class registry after IPC transfer complete."""
        if tensor_id is None:
            return

        self._active_tensors.pop(tensor_id, None)
        self._tensor_refs.pop(tensor_id, None)

        if TRACE_TRANSFERS:
            _trace_logger.debug(
                f"[CUDA_IPC] Unregistered tensor {tensor_id} from lifetime registry"
            )

    async def _pre_put_hook(self, request: "Request") -> None:
        """Prepare IPC handle before sending put request (client-side)."""
        if request.is_object:
            self.is_object = True
            self.objects = request.objects
            return

        tensor = request.tensor_val
        if tensor is None:
            raise ValueError("tensor_val must not be None for non-object requests")

        if TRACE_TRANSFERS:
            _trace_logger.info(
                f"[CUDA_IPC _pre_put_hook] tensor.device={tensor.device}, "
                f"shape={tensor.shape}, dtype={tensor.dtype}, "
                f"size_mb={tensor.numel() * tensor.element_size() / 1024 / 1024:.2f}"
            )

        # Ensure tensor is on GPU
        if not tensor.is_cuda:
            tensor = tensor.cuda()
            if TRACE_TRANSFERS:
                _trace_logger.info("[CUDA_IPC _pre_put_hook] Moved tensor to GPU")

        # Make contiguous if needed
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        # Register tensor to prevent deallocation during IPC transfer
        self._tensor_id = self._register_tensor(tensor)
        self._source_tensor = tensor

        # Create IPC handle
        self.ipc_handle = create_ipc_handle(tensor)
        self.shape = tensor.shape
        self.dtype = tensor.dtype

        if TRACE_TRANSFERS:
            _trace_logger.info(
                f"[CUDA_IPC _pre_put_hook] Created IPC handle for device cuda:{self.ipc_handle.storage_device}"
            )

    async def _pre_get_hook(self, key: str, request: "Request") -> None:
        """Prepare for get request (client-side)."""
        # Store request for later use in response handling
        self.request = request

        # Fetch metadata if no tensor provided
        tensor_like = request.tensor_val
        if tensor_like is None:
            meta = await self.storage_volume_ref.volume.get_meta.call_one(
                key, request.meta_only()
            )
            if isinstance(meta, str) or meta is None:
                return  # Objects don't need special handling
            tensor_like = meta  # (shape, dtype) tuple

        if isinstance(tensor_like, tuple):
            self.shape, self.dtype = tensor_like
        else:
            self.shape = tensor_like.shape
            self.dtype = tensor_like.dtype

    async def handle_put_request(
        self,
        ctx: "TransportContext",
        request: "Request",
        maybe_tensor,
    ) -> Any:
        """Called by storage volume. Reconstruct tensor from IPC handle."""
        if self.is_object:
            return self.objects

        if TRACE_TRANSFERS:
            t0 = time.perf_counter()

        # Reconstruct tensor from IPC handle
        tensor = self.ipc_handle.reconstruct_tensor()

        if TRACE_TRANSFERS:
            elapsed = time.perf_counter() - t0
            size_mb = tensor.numel() * tensor.element_size() / 1024 / 1024
            _trace_logger.info(
                f"[CUDA_IPC handle_put_request] IPC reconstruct completed: "
                f"{size_mb:.2f}MB in {elapsed*1000:.2f}ms on {tensor.device}"
            )

        # Clone to ensure we own the data (IPC handle might be invalidated)
        return tensor.clone()

    async def handle_get_request(self, ctx: "TransportContext", data) -> None:
        """Called by storage volume. Create IPC handle for stored data."""
        if not isinstance(data, torch.Tensor):
            self.is_object = True
            self.objects = data
            return

        tensor = data

        if TRACE_TRANSFERS:
            _trace_logger.info(
                f"[CUDA_IPC handle_get_request] Creating IPC handle for "
                f"shape={tensor.shape}, device={tensor.device}"
            )

        # Ensure tensor is on GPU
        if not tensor.is_cuda:
            tensor = tensor.cuda()

        # Make contiguous
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        # Register tensor to prevent deallocation during IPC transfer
        self._tensor_id = self._register_tensor(tensor)
        self._source_tensor = tensor
        self.ipc_handle = create_ipc_handle(tensor)
        self.shape = tensor.shape
        self.dtype = tensor.dtype

    async def _handle_storage_volume_response(
        self, transport_buffer: "TransportBuffer"
    ) -> Any:
        """Extract data from response buffer on client side."""
        if transport_buffer.is_object:
            return transport_buffer.objects

        if TRACE_TRANSFERS:
            t0 = time.perf_counter()

        # Reconstruct tensor from IPC handle
        tensor = transport_buffer.ipc_handle.reconstruct_tensor()

        if TRACE_TRANSFERS:
            elapsed = time.perf_counter() - t0
            size_mb = tensor.numel() * tensor.element_size() / 1024 / 1024
            _trace_logger.info(
                f"[CUDA_IPC response] IPC reconstruct completed: "
                f"{size_mb:.2f}MB in {elapsed*1000:.2f}ms on {tensor.device}"
            )

        # Copy to user's tensor if provided
        if self.request is not None and self.request.tensor_val is not None:
            self.request.tensor_val.copy_(tensor)
            return self.request.tensor_val

        # Clone to own the data
        return tensor.clone()

    async def drop(self) -> None:
        """Clean up resources."""
        # Unregister tensor from lifetime management
        self._unregister_tensor(self._tensor_id)

        self.ipc_handle = None
        self._source_tensor = None
        self._tensor_id = None
        self.objects = None
        self.request = None
