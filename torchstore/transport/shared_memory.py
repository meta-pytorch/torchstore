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
from typing import Any, TYPE_CHECKING

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
    """Unified cache for shared memory segments.

    Storage uses allocate() with volume_id=None (default).
    Client uses attach() with volume_id set.
    """

    def __init__(self):
        # _entries: (volume_id, key) -> {coordinates -> entry}
        # coordinates: DTensor shard coordinates (None for regular tensors)
        self._entries: dict[
            tuple[str | None, str], dict[tuple | None, SharedMemoryEntry]
        ] = {}

    def get(
        self,
        key: str,
        volume_id: str | None = None,
        coordinates: tuple | None = None,
    ) -> SharedMemoryEntry | None:
        """Return existing segment or None."""
        tensor_key = (volume_id, key)
        entries = self._entries.get(tensor_key)
        if entries is None:
            return None
        return entries.get(coordinates)

    def delete(
        self,
        key: str,
        volume_id: str | None = None,
        coordinates: tuple | None = None,
    ) -> None:
        """Remove segment(s) from cache."""
        tensor_key = (volume_id, key)
        entries = self._entries.get(tensor_key)
        # should we fail hard?
        if not entries or coordinates not in entries:
            return

        if coordinates is None or len(entries) == 1:
            # delete all entries
            del self._entries[tensor_key]
        else:
            # delete specific shard
            del self._entries[tensor_key][coordinates]

    def reset(self) -> None:
        self._entries.clear()

    def _put(
        self,
        key: str,
        entry: SharedMemoryEntry,
        volume_id: str | None = None,
        coordinates: tuple | None = None,
    ) -> None:
        """Add entry to cache."""
        tensor_key = (volume_id, key)
        if tensor_key not in self._entries:
            self._entries[tensor_key] = {}
        self._entries[tensor_key][coordinates] = entry

    def allocate(
        self,
        key: str,
        shape: torch.Size,
        dtype: torch.dtype,
        volume_id: str | None = None,
        coordinates: tuple | None = None,
    ) -> SharedMemoryEntry:
        """Allocate a new segment or return existing one.

        Used by storage volume to create shared memory segments.

        Raises:
            AssertionError: If key exists with different shape/dtype. Delete first.
        """
        existing = self.get(key, volume_id, coordinates)
        if existing is not None:
            # Shape/dtype must match - segments are immutable once created
            assert existing.shape == shape and existing.dtype == dtype, (
                f"Key '{key}' exists with shape={existing.shape}, dtype={existing.dtype}. "
                f"Cannot overwrite with shape={shape}, dtype={dtype}. Delete first."
            )
            return existing

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
        self._put(key, entry, volume_id, coordinates)
        return entry

    # === Client-Side: Attachment ===

    def attach(
        self,
        key: str,
        descriptor: SharedMemoryDescriptor,
        volume_id: str | None = None,
        coordinates: tuple | None = None,
    ) -> SharedMemoryEntry:
        """Attach to an existing shared memory segment.

        Used by client to connect to storage volume's shared memory.
        Handles staleness validation by comparing storage_handle.

        Args:
            key: The key for the tensor
            descriptor: The shared memory descriptor from the storage volume
            volume_id: Storage volume ID for cache namespacing
            coordinates: DTensor shard coordinates (None for regular tensors)

        Returns:
            Cached or newly attached SharedMemoryEntry
        """
        cached = self.get(key, volume_id, coordinates)
        if cached is not None:
            # Validate: storage_handle must match
            if cached.descriptor.storage_handle == descriptor.storage_handle:
                return cached
            # Stale - invalidate
            self.delete(key, volume_id, coordinates)

        # Attach and cache
        entry = descriptor.attach()
        self._put(key, entry, volume_id, coordinates)
        return entry


class SharedMemoryTransportBuffer(TransportBuffer):
    """Transport using POSIX shared memory for same-host transfers.

    The storage volume owns the shared memory segment. On PUT, data is
    written directly to the storage volume's shared memory. On GET, data
    is read directly from it.
    """

    def __init__(self, storage_volume_ref: "StorageVolumeRef"):
        super().__init__(storage_volume_ref)
        self.shm_descriptor: SharedMemoryDescriptor | None = None
        self.is_object: bool = False
        self.objects: Any = None  # Python objects use RPC, not shared memory

        # Request metadata (serialized)
        self._key: str | None = None
        self._coordinates: tuple | None = None  # DTensor shard coordinates
        self.shape: torch.Size | None = None  # For PUT: tensor shape
        self.dtype: torch.dtype | None = None  # For PUT: tensor dtype

        # Client-side only (excluded from serialization)
        self._client_tensor: torch.Tensor | None = None
        self._needs_handshake: bool = False

    @property
    def requires_handshake(self) -> bool:
        """Handshake needed for tensor PUT to get segment allocation."""
        return self._needs_handshake

    async def recv_handshake(self, ctx: "TransportContext") -> "SharedMemoryDescriptor":
        """Storage volume: allocate shared memory segment, return descriptor."""
        # We can only allocate after talking to the storage volume once, to see if a shared memory
        # descriptor already exists. Open to other ways to go about this so its not in handshake.

        assert not self.is_object

        shm_cache = ctx.get_shm_cache()
        entry = shm_cache.allocate(
            self._key, self.shape, self.dtype, coordinates=self._coordinates
        )
        return entry.descriptor

    async def _post_handshake(self, handshake_result: Any) -> None:
        """Client: attach to segment and write tensor data."""
        descriptor: SharedMemoryDescriptor = handshake_result

        # Use client cache to avoid repeated mmap/munmap overhead
        shm_cache = self.storage_volume_ref.transport_context.get_shm_cache()
        client_entry = shm_cache.attach(
            self._key,
            descriptor,
            volume_id=self.storage_volume_ref.volume_id,
            coordinates=self._coordinates,
        )

        # Copy tensor data to shared memory
        shm_tensor = client_entry.get_tensor()
        shm_tensor.copy_(self._client_tensor)

    def __getstate__(self) -> dict[str, Any]:
        """Exclude non-serializable objects when sending buffer to storage volume."""
        state = self.__dict__.copy()
        # Exclude client-side handles
        state["_client_tensor"] = None
        state["storage_volume_ref"] = None
        return state

    # Client-side methods
    async def put_to_storage_volume(self, key, request: "Request"):
        """Override to capture the key for shared memory segment naming."""
        self._key = key
        await super().put_to_storage_volume(key, request)

    async def _pre_put_hook(self, request: "Request") -> None:
        """Store tensor metadata; actual copy happens in _post_handshake."""
        # Extract coordinates for DTensor shards
        if request.tensor_slice is not None:
            self._coordinates = request.tensor_slice.coordinates

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

        self._client_tensor = tensor
        # Need handshake for PUT (to get segment name from storage)
        self._needs_handshake = True

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
        entry = shm_cache.get(self._key, coordinates=self._coordinates)

        assert entry is not None, f"Segment for {self._key} not found after handshake"
        return entry.get_tensor()

    async def _pre_get_hook(self, key: str, request: "Request") -> None:
        """Prepare for receiving - may fetch metadata if not provided."""
        self._key = key
        self._client_tensor = request.tensor_val

        # Extract coordinates for DTensor shards
        if request.tensor_slice is not None:
            self._coordinates = request.tensor_slice.coordinates

    async def handle_get_request(self, ctx: "TransportContext", data: Any) -> None:
        """Data is already in shared memory - store descriptor for client."""
        if not isinstance(data, torch.Tensor):
            self.is_object = True
            self.objects = data
            return

        # Check if we have this key's segment in the cache
        shm_cache = ctx.get_shm_cache()
        entry = shm_cache.get(self._key, coordinates=self._coordinates)

        if entry is not None:
            if entry.shape == data.shape:
                self.shm_descriptor = entry.descriptor
                return
            else:
                # we run into this for resharded retrieval...
                logger.debug(
                    f"Key {self._key} cache shape mismatch: cached={entry.shape}, "
                    f"requested={data.shape}, using RPC fallback"
                )

        # If the tensor was written by a different transport, it is not backed
        # by shared memory so we can't handle it appropriately at this stage.
        # We need to consider a design for this, like maybe a "force_shm" param
        # on request to force transports to back their memory by SHM
        if entry is None:
            logger.debug(
                f"Key {self._key} not in shared memory cache, using RPC fallback"
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
            client_entry = shm_cache.attach(
                self._key,
                transport_buffer.shm_descriptor,
                volume_id=self.storage_volume_ref.volume_id,
                coordinates=self._coordinates,
            )
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
            return shm_tensor.clone()

    async def drop(self) -> None:
        """
        Drop some references, but even this is not necessary if the transport object as a whole is dropped
        """
        self._client_tensor = None
        self.objects = None
