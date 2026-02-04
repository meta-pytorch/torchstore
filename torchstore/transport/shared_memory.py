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
import socket
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

import torch

from torchstore.transport.buffers import TransportBuffer

if TYPE_CHECKING:
    from torchstore.strategy import StorageVolumeRef
    from torchstore.transport.buffers import TransportContext
    from torchstore.transport.types import Request

logger = logging.getLogger(__name__)


def get_local_hostname() -> str:
    """Get the current machine's hostname."""
    return os.environ.get("HOSTNAME", socket.gethostname())


def is_local_to_volume(storage_volume_ref: "StorageVolumeRef") -> bool:
    """Check if client is on the same host as the storage volume."""
    return storage_volume_ref.volume_hostname == get_local_hostname()


@dataclass
class SharedMemoryDescriptor:
    """Serializable descriptor for PyTorch shared memory storage.

    This is sent from storage volume to client. The client uses attach()
    to connect to the storage and get a usable entry.
    """

    manager_handle: bytes  # Path to PyTorch shared memory manager socket
    storage_handle: bytes  # Shared memory segment name
    size: int  # Size in bytes
    shape: torch.Size
    dtype: torch.dtype

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
    _tensor: Optional[torch.Tensor] = field(default=None, repr=False)

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

    def close(self) -> None:
        """Detach from shared storage (client-side cleanup)."""
        self._tensor = None


class SharedMemoryCache:
    """Unified cache for shared memory segments.

    Storage uses allocate() with scope="" (default).
    Client uses attach() with scope=volume_id.
    """

    def __init__(self):
        self._entries: Dict[Tuple[str, str], SharedMemoryEntry] = {}

    # === Common Operations ===

    def get(self, key: str, scope: str = "") -> Optional[SharedMemoryEntry]:
        """Return existing segment or None."""
        return self._entries.get((scope, key))

    def delete(self, key: str, scope: str = "") -> None:
        """Remove segment from cache."""
        cache_key = (scope, key)
        if cache_key in self._entries:
            entry = self._entries.pop(cache_key)
            entry.close()

    def reset(self) -> None:
        """Clean up all segments."""
        for entry in self._entries.values():
            entry.close()
        self._entries.clear()

    # === Storage-Side: Allocation ===

    def allocate(
        self, key: str, shape: torch.Size, dtype: torch.dtype, scope: str = ""
    ) -> SharedMemoryEntry:
        """Allocate a new segment or return existing one.

        Used by storage volume to create shared memory segments.

        Raises:
            AssertionError: If key exists with different shape/dtype. Delete first.
        """
        cache_key = (scope, key)
        if cache_key in self._entries:
            entry = self._entries[cache_key]
            # Shape/dtype must match - segments are immutable once created
            assert entry.shape == shape and entry.dtype == dtype, (
                f"Key '{key}' exists with shape={entry.shape}, dtype={entry.dtype}. "
                f"Cannot overwrite with shape={shape}, dtype={dtype}. Delete first."
            )
            return entry

        # Allocate directly in shared memory
        size_bytes = shape.numel() * dtype.itemsize
        storage = torch.UntypedStorage._new_using_filename_cpu(size_bytes)
        tensor = torch.empty(0, dtype=dtype).set_(storage).view(shape)
        tensor.fill_(0)  # Prefault the memory

        # Get handles for serialization
        manager_handle, storage_handle, size = storage._share_filename_cpu_()

        descriptor = SharedMemoryDescriptor(
            manager_handle=manager_handle,
            storage_handle=storage_handle,
            size=size,
            shape=shape,
            dtype=dtype,
        )
        entry = SharedMemoryEntry(
            storage=storage, descriptor=descriptor, _tensor=tensor
        )
        self._entries[cache_key] = entry
        return entry

    # === Client-Side: Attachment ===

    def attach(
        self,
        key: str,
        descriptor: SharedMemoryDescriptor,
        scope: str = "",
    ) -> SharedMemoryEntry:
        """Attach to an existing shared memory segment.

        Used by client to connect to storage volume's shared memory.
        Handles staleness validation by comparing storage_handle.

        Args:
            key: The key for the tensor
            descriptor: The shared memory descriptor from the storage volume
            scope: Namespace for cache keys (typically volume_id for client)

        Returns:
            Cached or newly attached SharedMemoryEntry
        """
        cache_key = (scope, key)

        cached = self._entries.get(cache_key)
        if cached is not None:
            # Validate: storage_handle must match
            if cached.descriptor.storage_handle == descriptor.storage_handle:
                return cached
            # Stale - invalidate
            cached.close()
            del self._entries[cache_key]

        # Attach and cache
        entry = descriptor.attach()
        self._entries[cache_key] = entry
        return entry


class SharedMemoryTransportBuffer(TransportBuffer):
    """Transport using POSIX shared memory for same-host transfers.

    The storage volume owns the shared memory segment. On PUT, data is
    written directly to the storage volume's shared memory. On GET, data
    is read directly from it.
    """

    def __init__(self, storage_volume_ref: "StorageVolumeRef"):
        super().__init__(storage_volume_ref)
        # Serialized state (metadata only)
        self.shm_descriptor: Optional[SharedMemoryDescriptor] = None
        self.is_object: bool = False
        self.objects: Any = None  # only for python objects, never tensors
        self._key: Optional[str] = None

        # For PUT: metadata before handshake
        self.shape: Optional[torch.Size] = None
        self.dtype: Optional[torch.dtype] = None

        # Client-side only (not serialized)
        self._request: Optional["Request"] = None
        self._source_tensor: Optional[torch.Tensor] = None  # For PUT: tensor to write
        self._client_entry: Optional[SharedMemoryEntry] = None  # Attached entry

    @property
    def requires_handshake(self) -> bool:
        """Handshake needed for tensor PUT to get segment allocation."""
        # Putting this in handshake is a little ugly, ideally we have a _post_put instead
        # Need handshake for PUT (to get segment name from storage)
        # _source_tensor is set in _pre_put_hook for tensor PUTs
        return self._source_tensor is not None

    async def recv_handshake(self, ctx: "TransportContext") -> "SharedMemoryDescriptor":
        """Storage volume: allocate shared memory segment, return descriptor."""
        assert not self.is_object

        shm_cache = ctx.get_shm_cache()
        entry = shm_cache.allocate(self._key, self.shape, self.dtype)
        return entry.descriptor

    async def _post_handshake(self, handshake_result: Any) -> None:
        """Client: attach to segment and write tensor data."""
        descriptor: SharedMemoryDescriptor = handshake_result

        # Use client cache to avoid repeated mmap/munmap overhead
        shm_cache = self.storage_volume_ref.transport_context.get_shm_cache()
        self._client_entry = shm_cache.attach(
            self._key,
            descriptor,
            scope=self.storage_volume_ref.volume_id,
        )

        # Copy tensor data to shared memory
        shm_tensor = self._client_entry.get_tensor()
        shm_tensor.copy_(self._source_tensor)

    def __getstate__(self) -> Dict[str, Any]:
        """Exclude non-serializable objects when sending buffer to storage volume."""
        state = self.__dict__.copy()
        # Exclude client-side handles
        state["_client_entry"] = None
        state["_request"] = None
        state["_source_tensor"] = None
        state["storage_volume_ref"] = None
        return state

    # Client-side methods
    async def put_to_storage_volume(self, key, request: "Request"):
        """Override to capture the key for shared memory segment naming."""
        self._key = key
        await super().put_to_storage_volume(key, request)

    async def _pre_put_hook(self, request: "Request") -> None:
        """Store tensor metadata; actual copy happens in _post_handshake."""
        self._request = request

        if request.is_object:
            self.is_object = True
            self.objects = request.objects  # Objects use RPC
            return

        tensor = request.tensor_val
        assert tensor is not None

        self.shape = tensor.shape
        self.dtype = tensor.dtype

        # Handle non-contiguous tensors
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        self._source_tensor = tensor

    async def handle_put_request(
        self,
        ctx: "TransportContext",
        request: "Request",
        current_object: Any,
    ) -> Any:
        """Read tensor from shared memory segment"""
        if request.is_object or self.is_object:
            return self.objects

        # Data is already in shared memory (written by client in _post_handshake)
        # Just return the tensor backed by the segment
        shm_cache = ctx.get_shm_cache()
        entry = shm_cache.get(self._key)

        assert entry is not None, f"Segment for {self._key} not found after handshake"
        return entry.get_tensor()

    async def _pre_get_hook(self, key: str, request: "Request") -> None:
        """Prepare for receiving - may fetch metadata if not provided."""
        self._request = request
        self._key = key

        if request.tensor_val is not None:
            self.shape = request.tensor_val.shape
            self.dtype = request.tensor_val.dtype
        else:
            meta = await self.storage_volume_ref.volume.get_meta.call_one(
                key, request.meta_only()
            )
            if isinstance(meta, str) or meta is None:
                # object - will use RPC fallback
                return
            self.shape, self.dtype = meta

    async def handle_get_request(self, ctx: "TransportContext", data: Any) -> None:
        """Data is already in shared memory - store descriptor for client."""
        if not isinstance(data, torch.Tensor):
            self.is_object = True
            self.objects = data
            return

        # Check if we have this key's segment in the cache
        shm_cache = ctx.get_shm_cache()
        entry = shm_cache.get(self._key)

        if entry is not None:
            # Tensor is backed by shared memory - send descriptor
            self.shm_descriptor = entry.descriptor
        else:
            # If the tensor was written by a different transport, it is not backed
            # by shared memory so we can't handle it appropriately by this stage.
            # We need to consider a design for this, like maybe a "force_shm" param
            # on other transports to back their memory by SHM
            logger.debug(
                f"Key {self._key} not in shared memory cache, using RPC fallback"
            )
            self.shm_descriptor = None
            self.objects = data

    async def _handle_storage_volume_response(
        self, transport_buffer: "TransportBuffer"
    ) -> Any:
        """Extract tensor from shared memory or return object."""
        if transport_buffer.is_object:
            return transport_buffer.objects

        # If the tensor was written by a different transport, it is not backed
        # by shared memory so we can't handle it appropriately by this stage.
        # We need to consider a design for this, like maybe a "force_shm" param
        # on other transports to back their memory by SHM
        if transport_buffer.shm_descriptor is None:
            if transport_buffer.objects is not None:
                if self._request is not None and self._request.tensor_val is not None:
                    self._request.tensor_val.copy_(transport_buffer.objects)
                    return self._request.tensor_val
                return transport_buffer.objects
            raise RuntimeError("No shm_descriptor or objects in response")

        try:
            # Use client cache to avoid repeated mmap/munmap overhead
            shm_cache = self.storage_volume_ref.transport_context.get_shm_cache()
            self._client_entry = shm_cache.attach(
                self._key,
                transport_buffer.shm_descriptor,
                scope=self.storage_volume_ref.volume_id,
            )
        except RuntimeError as e:
            if "No such file" in str(e):
                raise RuntimeError(
                    "Shared memory storage not found. "
                    "This may indicate the storage volume is on a different host."
                ) from e
            raise

        shm_tensor = self._client_entry.get_tensor()

        # Copy to client's tensor if provided (inplace), otherwise clone
        if self._request is not None and self._request.tensor_val is not None:
            self._request.tensor_val.copy_(shm_tensor)
            return self._request.tensor_val
        else:
            return shm_tensor.clone()

    async def drop(self) -> None:
        """Client-side cleanup.

        Note: We don't close _client_entry here - the client cache owns its
        lifetime. This allows cached entries to be reused across operations.
        """
        self._client_entry = None
        self._source_tensor = None
        self.objects = None
        self._request = None
