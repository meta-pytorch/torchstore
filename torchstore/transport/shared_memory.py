# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared memory transport buffer for same-host tensor transfers.

When client and storage volume are on the same host, tensors can be stored
in shared memory, allowing:
- Direct writes from client to storage volume's shared memory tensor
- Direct reads from storage volume's shared memory tensor
- Persistence - stored tensor remains in shared memory for O(1) subsequent access
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import torch

from torchstore.logging import LatencyTracker
from torchstore.transport.buffers import TransportBuffer
from torchstore.transport.types import KeyedRequest, Request
from torchstore.utils import get_local_hostname

if TYPE_CHECKING:
    from torchstore.strategy import StorageVolumeRef
    from torchstore.transport.buffers import TransportContext

logger = logging.getLogger(__name__)


def is_local_to_volume(storage_volume_ref: "StorageVolumeRef") -> bool:
    """Check if client is on the same host as the storage volume."""
    return storage_volume_ref.volume_hostname == get_local_hostname()


def allocate_shared_tensor(shape: torch.Size, dtype: torch.dtype) -> torch.Tensor:
    """Allocate a tensor backed by shared memory."""
    size_bytes = shape.numel() * dtype.itemsize
    storage = torch.UntypedStorage._new_using_filename_cpu(size_bytes)
    tensor = torch.empty(0, dtype=dtype).set_(storage).view(shape)
    tensor.fill_(0)  # Prefault memory
    return tensor


SHOULD_PIN_SHM = os.environ.get("TORCHSTORE_PIN_SHM", "1") == "1"
MUTABLE_SHM = os.environ.get("TORCHSTORE_MUTABLE_SHM", "1") == "1"
# Disabling by default on initial release
SHM_ENABLED = os.environ.get("TORCHSTORE_SHARED_MEMORY_ENABLED", "1") == "1"


def pin_memory(tensor: torch.Tensor) -> None:
    """Pin tensor's memory for faster CUDA transfers.

    Uses cudaHostRegister with cudaHostRegisterPortable flag to make the
    memory accessible from all CUDA contexts.
    """
    if not SHOULD_PIN_SHM or not torch.cuda.is_available():
        return

    cudart = torch.cuda.cudart()
    if cudart is None:
        return  # No CUDA runtime available, skip pinning

    data_ptr = tensor.data_ptr()
    size = tensor.numel() * tensor.element_size()
    err = int(cudart.cudaHostRegister(data_ptr, size, 1))  # cudaHostRegisterPortable
    if err == 712:  # cudaErrorHostMemoryAlreadyRegistered
        logger.info("[SHM] Tensor is already pinned.")
        return
    if err != 0:
        raise RuntimeError(
            f"[SHM] cudaHostRegister failed with error {err}. "
            "Consider launching with CUDA_LAUNCH_BLOCKING=1 to debug."
        )


def unpin_memory(tensor: torch.Tensor) -> None:
    """Unpin tensor's memory."""
    if not SHOULD_PIN_SHM or not torch.cuda.is_available():
        return

    cudart = torch.cuda.cudart()
    if cudart is None:
        return

    err = int(cudart.cudaHostUnregister(tensor.data_ptr()))
    if err == 713:  # cudaErrorHostMemoryNotRegistered
        logger.info("[SHM] Tensor is already unpinned.")
        return
    if err != 0:
        logger.warning(f"cudaHostUnregister failed with error {err}")


@dataclass
class SharedMemoryDescriptor:
    """Serializable descriptor for PyTorch shared memory storage.

    This is sent from storage volume to client. The client uses attach()
    to connect to the storage and get a usable entry.
    """

    manager_handle: bytes
    storage_handle: bytes
    size: int
    shape: torch.Size
    dtype: torch.dtype

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "SharedMemoryDescriptor | None":
        """Derive SharedMemoryDescriptor from a tensor backed by shared memory.

        Returns None if the tensor is not shared or is a view/slice of a larger
        shared tensor (which can't be efficiently represented as shared memory).
        """
        if not tensor.is_shared():
            logger.info("Tensor is not in shared memory.")
            return None

        # Check if tensor is a view/slice of a larger storage
        expected_size = tensor.numel() * tensor.element_size()
        storage = tensor.untyped_storage()
        if storage.size() != expected_size:
            # Tensor is a view/slice - can't use shared memory for this
            logger.info("Tensor is a view/slice, cannot use shared memory.")
            return None

        manager_handle, storage_handle, size = storage._share_filename_cpu_()
        return cls(
            manager_handle=manager_handle,
            storage_handle=storage_handle,
            size=size,
            shape=tensor.shape,
            dtype=tensor.dtype,
        )

    def attach(self) -> "SharedMemoryEntry":
        """Client-side: attach to shared storage."""
        storage = torch.UntypedStorage._new_shared_filename_cpu(
            self.manager_handle, self.storage_handle, self.size
        )
        return SharedMemoryEntry(storage=storage, descriptor=self)


@dataclass
class SharedMemoryEntry:
    """Entry wrapping PyTorch shared storage."""

    storage: torch.UntypedStorage
    descriptor: SharedMemoryDescriptor
    _tensor: torch.Tensor | None = field(default=None, repr=False)

    @property
    def name(self) -> str:
        return self.descriptor.storage_handle.decode()

    @property
    def shape(self) -> torch.Size:
        return self.descriptor.shape

    @property
    def dtype(self) -> torch.dtype:
        return self.descriptor.dtype

    def get_tensor(self) -> torch.Tensor:
        """Create tensor view backed by shared storage."""
        if self._tensor is None:
            # Create tensor from storage with proper dtype/shape
            self._tensor = (
                torch.empty(0, dtype=self.dtype).set_(self.storage).view(self.shape)
            )
        return self._tensor


class SharedMemoryCache:
    """Client-side cache for shared memory segments.

    Uses (key, storage_handle) as cache key. Stale entries (after delete/re-PUT)
    are never accessed because the server returns new handles.
    """

    def __init__(self):
        self._entries: dict[tuple[str, bytes], SharedMemoryEntry] = {}

    def allocate(
        self,
        key: str,
        shape: torch.Size,
        dtype: torch.dtype,
    ) -> tuple[SharedMemoryEntry, SharedMemoryDescriptor]:
        """Allocate new shared memory and cache it."""
        new_tensor = allocate_shared_tensor(shape, dtype)
        descriptor = SharedMemoryDescriptor.from_tensor(new_tensor)
        assert descriptor is not None
        entry = self.attach(key, descriptor)
        return entry, descriptor

    def attach(self, key: str, descriptor: SharedMemoryDescriptor) -> SharedMemoryEntry:
        """Attach to shared memory segment, caching the entry."""
        cache_key = (key, descriptor.storage_handle)

        if cache_key in self._entries:
            return self._entries[cache_key]

        entry = descriptor.attach()
        pin_memory(entry.get_tensor())
        self._entries[cache_key] = entry
        return entry

    def clear(self) -> None:
        """Clear all entries."""
        for entry in self._entries.values():
            unpin_memory(entry.get_tensor())
        self._entries.clear()

    def __del__(self):
        self.clear()


class SharedMemoryTransportBuffer(TransportBuffer):
    """Transport using POSIX shared memory for same-host transfers.

    The storage volume owns the shared memory segment. On PUT, data is
    written directly to the storage volume's shared memory. On GET, data
    is read directly from it.


    DATA FLOW

    PUT

    1. Client: requires_handshake checks if any entry has a tensor (needs SHM handshake)
    2. SV: recv_handshake: Return descriptors for existing tensors, None for new
    3. Client: _post_handshake: Allocate/attach SHM segments and copy tensor data
    4. Client: _pre_put_hook: Store objects in _batch_objects for RPC transport
    5. SV: handle_put_request: Route objects from _batch_objects, tensors via SHM attachment

    GET
    1. _pre_get_hook: Save some metadata
    2. handle_get_request: Return the shared memory descriptor if possible.
                           Fallback to RPC if stored tensor is object, not shared, or a view (resharding case)
    3. _handle_storage_volume_response: Parse server response and copy data according to path

    """

    supports_batch_puts = True

    def __init__(self, storage_volume_ref: "StorageVolumeRef"):
        super().__init__(storage_volume_ref)
        self.shm_descriptor: SharedMemoryDescriptor | None = None
        self.is_object: bool = False
        self.objects: Any = None  # Python objects use RPC, not shared memory

        # Request metadata (serialized)
        self._key: str | None = None

        # Client-side only (excluded from serialization)
        self._client_tensor: torch.Tensor | None = None
        # SHM only needs handshake during PUT, not GET
        self._needs_handshake: bool = False

        # Batch state
        self._batch_shm_descriptors: dict[str, SharedMemoryDescriptor | None] = {}
        # Objects travel via RPC serialization (not excluded from __getstate__)
        self._batch_objects: dict[str, Any] = {}

    async def put_to_storage_volume(self, entries: list[KeyedRequest]) -> None:
        self._needs_handshake = True
        await super().put_to_storage_volume(entries)

    def requires_handshake(self, entries: list[KeyedRequest]) -> bool:
        return self._needs_handshake

    async def _post_handshake(
        self,
        handshake_results: list[Any],
        entries: list[KeyedRequest],
    ) -> None:
        """Allocate/attach SHM segments and copy tensor data for entries.

        Iterates entries and handshake_results in lockstep. For each tensor entry, either
        attaches to an existing SHM segment (if the SV returned a descriptor)
        or allocates a new one. Then copies the client tensor into the SHM
        region with non_blocking=True, and synchronizes to ensure all copies
        complete before returning.
        """
        latency_tracker = LatencyTracker("post_handshake")

        shm_cache = self.storage_volume_ref.transport_context.get_shm_cache()

        for (key, request), descriptor in zip(entries, handshake_results):
            if request.is_object:
                continue

            tensor = request.tensor_val
            assert tensor is not None

            if not tensor.is_contiguous():
                tensor = tensor.cpu().contiguous()

            if descriptor is not None:
                client_entry = shm_cache.attach(key, descriptor)
            else:
                client_entry, descriptor = shm_cache.allocate(
                    key, tensor.shape, tensor.dtype
                )

            self._batch_shm_descriptors[key] = descriptor

            shm_tensor = client_entry.get_tensor()
            shm_tensor.copy_(tensor, non_blocking=True)
        latency_tracker.track_step("alloc_and_copy")

        # Wait for all async copies (GPU->CPU DMA) to complete
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency_tracker.track_step("cuda_synchronize")

    async def _pre_put_hook(self, entries: list[KeyedRequest]):
        for key, request in entries:
            if request.is_object:
                self._batch_objects[key] = request.objects

    async def recv_handshake(
        self,
        ctx: "TransportContext",
        entries: list[tuple[KeyedRequest, Any]],
    ) -> list["SharedMemoryDescriptor | None"]:
        """Storage volume: return existing descriptors if available, else None."""
        results = []
        for entry, current_object in entries:
            if not isinstance(current_object, torch.Tensor):
                results.append(None)
            else:
                descriptor = SharedMemoryDescriptor.from_tensor(current_object)
                assert descriptor is not None, "Stored tensor is not in shared memory."
                results.append(descriptor)
        return results

    async def handle_put_request(
        self,
        ctx: "TransportContext",
        entries: list[tuple[KeyedRequest, Any]],
    ) -> dict[str, Any]:
        """SV side: handle batch of put requests for tensors and objects."""
        results = {}
        for entry, current_object in entries:
            if entry.key in self._batch_objects:
                results[entry.key] = self._batch_objects[entry.key]
            else:
                descriptor = self._batch_shm_descriptors.get(entry.key)
                assert descriptor is not None, f"No descriptor for {entry.key}"

                if isinstance(current_object, torch.Tensor):
                    existing = SharedMemoryDescriptor.from_tensor(current_object)
                    assert existing is not None
                    assert existing.storage_handle == descriptor.storage_handle
                    results[entry.key] = current_object
                else:
                    shm_entry = descriptor.attach()
                    results[entry.key] = shm_entry.get_tensor()
        return results

    def __getstate__(self) -> dict[str, Any]:
        """Exclude non-serializable objects when sending buffer to storage volume."""
        state = self.__dict__.copy()
        # Exclude client-side handles
        state["_client_tensor"] = None
        state["storage_volume_ref"] = None
        return state

    async def _pre_get_hook(self, key: str, request: Request) -> None:
        """Prepare for receiving - may fetch metadata if not provided."""
        self._key = key
        self._client_tensor = request.tensor_val

    async def handle_get_request(self, ctx: "TransportContext", data: Any) -> None:
        """Derive descriptor from stored tensor if backed by shared memory."""
        if not isinstance(data, torch.Tensor):
            self.is_object = True
            self.objects = data
            return

        descriptor = SharedMemoryDescriptor.from_tensor(data)
        if descriptor is not None:
            self.shm_descriptor = descriptor
            return

        logger.debug(
            f"Key {self._key} is a view or not in shared memory, using RPC fallback"
        )

        self.shm_descriptor = None
        self.objects = data

    async def _handle_storage_volume_response(
        self, transport_buffer: "TransportBuffer"
    ) -> Any:
        """Client-side: extract tensor from shared memory or handle RPC fallback."""
        if transport_buffer.is_object:
            return transport_buffer.objects

        # RPC fallback path (no shared memory descriptor, stored by another transport)
        if transport_buffer.shm_descriptor is None:
            if self._client_tensor is not None:
                self._client_tensor.copy_(transport_buffer.objects)
                return self._client_tensor
            return transport_buffer.objects

        try:
            shm_cache = self.storage_volume_ref.transport_context.get_shm_cache()
            client_entry = shm_cache.attach(self._key, transport_buffer.shm_descriptor)
        except RuntimeError as e:
            if "No such file" in str(e):
                raise RuntimeError(
                    "Shared memory storage not found. "
                    "This may indicate the storage volume is on a different host."
                ) from e
            raise

        shm_tensor = client_entry.get_tensor()

        # Copy to client's tensor if provided (inplace), otherwise clone
        if self._client_tensor is not None:
            self._client_tensor.copy_(shm_tensor)
            return self._client_tensor
        else:
            return shm_tensor if MUTABLE_SHM else shm_tensor.clone()

    async def drop(self) -> None:
        """
        Drop some references, but even this is not necessary if the transport object as a whole is dropped
        """
        self._client_tensor = None
        self.objects = None
        self._batch_shm_descriptors = {}
        self._batch_objects = {}
