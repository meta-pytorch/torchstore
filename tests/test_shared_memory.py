# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""unit tests made by claude, have to vet them"""

import pytest
import torch
from torchstore.transport.shared_memory import (
    get_local_hostname,
    is_local_to_volume,
    SharedMemoryCache,
    SharedMemoryDescriptor,
    SharedMemoryTransportBuffer,
)


class MockTransportContext:
    """Mock TransportContext for testing."""

    def __init__(self):
        self._shm_cache = SharedMemoryCache()

    def get_shm_cache(self):
        return self._shm_cache

    def reset(self):
        self._shm_cache.reset()


class MockStorageVolumeRef:
    """Mock StorageVolumeRef for testing."""

    def __init__(self, volume_hostname=None, volume_id="test_volume"):
        self.volume_hostname = volume_hostname
        self.volume_id = volume_id
        self.transport_context = MockTransportContext()


class TestHelperFunctions:
    """Test helper functions."""

    def test_get_local_hostname(self):
        """Test that get_local_hostname returns a string."""
        hostname = get_local_hostname()
        assert isinstance(hostname, str)
        assert len(hostname) > 0

    def test_get_local_hostname_with_env(self, monkeypatch):
        """Test that get_local_hostname respects HOSTNAME env var."""
        monkeypatch.setenv("HOSTNAME", "test-hostname")
        assert get_local_hostname() == "test-hostname"

    def test_is_local_to_volume_same_host(self):
        """Test is_local_to_volume returns True for same host."""
        hostname = get_local_hostname()
        ref = MockStorageVolumeRef(volume_hostname=hostname)
        assert is_local_to_volume(ref) is True

    def test_is_local_to_volume_different_host(self):
        """Test is_local_to_volume returns False for different host."""
        ref = MockStorageVolumeRef(volume_hostname="some-other-host-12345")
        assert is_local_to_volume(ref) is False

    def test_is_local_to_volume_no_hostname(self):
        """Test is_local_to_volume returns False when hostname not set."""
        ref = MockStorageVolumeRef(volume_hostname=None)
        assert is_local_to_volume(ref) is False


class TestSharedMemoryEntry:
    """Test SharedMemoryEntry dataclass."""

    def test_get_tensor(self):
        """Test creating a tensor from shared memory entry."""
        cache = SharedMemoryCache()
        try:
            shape = torch.Size([4, 8])
            dtype = torch.float32

            entry = cache.allocate("test_key", shape, dtype)

            tensor = entry.get_tensor()
            assert tensor.shape == shape
            assert tensor.dtype == dtype
            assert tensor.numel() == 4 * 8

            # Verify we can write to the tensor
            tensor.fill_(42.0)
            tensor2 = entry.get_tensor()
            assert torch.all(tensor2 == 42.0)
        finally:
            cache.reset()

    def test_descriptor(self):
        """Test accessing the descriptor from an entry."""
        cache = SharedMemoryCache()
        try:
            shape = torch.Size([16])
            dtype = torch.float32

            entry = cache.allocate("test_key", shape, dtype)

            # Descriptor should have correct fields
            assert isinstance(entry.descriptor, SharedMemoryDescriptor)
            assert isinstance(entry.descriptor.manager_handle, bytes)
            assert isinstance(entry.descriptor.storage_handle, bytes)
            assert entry.descriptor.size > 0
            assert entry.descriptor.shape == shape
            assert entry.descriptor.dtype == dtype
        finally:
            cache.reset()


class TestSharedMemoryDescriptor:
    """Test SharedMemoryDescriptor dataclass."""

    def test_attach(self):
        """Test attaching to a segment via descriptor."""
        cache = SharedMemoryCache()
        try:
            shape = torch.Size([10, 10])
            dtype = torch.float32

            # Create entry via cache
            entry = cache.allocate("test_key", shape, dtype)

            # Write some data
            original = torch.randn(10, 10)
            shm_tensor = entry.get_tensor()
            shm_tensor.copy_(original)

            # Create new entry via descriptor attach
            descriptor = entry.descriptor
            attached_entry = descriptor.attach()

            # Verify entry is valid
            assert attached_entry.shape == shape
            assert attached_entry.dtype == dtype

            # Verify we can read the data
            tensor = attached_entry.get_tensor()
            assert torch.allclose(tensor, original)
        finally:
            cache.reset()


class TestSharedMemoryCache:
    """Test SharedMemoryCache."""

    def test_allocate(self):
        """Test creating a new segment."""
        cache = SharedMemoryCache()
        try:
            shape = torch.Size([10, 10])
            dtype = torch.float32

            entry = cache.allocate("test_key", shape, dtype)
            assert entry is not None
            assert entry.shape == shape
            assert entry.dtype == dtype

            # Should return the same entry for the same key
            entry2 = cache.allocate("test_key", shape, dtype)
            assert entry2.name == entry.name
        finally:
            cache.reset()

    def test_allocate_raises_on_shape_mismatch(self):
        """Test that AssertionError is raised when shape/dtype changes."""
        cache = SharedMemoryCache()
        try:
            shape1 = torch.Size([10, 10])
            shape2 = torch.Size([20, 20])
            dtype = torch.float32

            cache.allocate("test_key", shape1, dtype)

            # Attempting to use different shape should raise AssertionError
            with pytest.raises(AssertionError) as exc_info:
                cache.allocate("test_key", shape2, dtype)

            assert "Cannot overwrite" in str(exc_info.value)
            assert "Delete first" in str(exc_info.value)
        finally:
            cache.reset()

    def test_get(self):
        """Test getting an existing segment."""
        cache = SharedMemoryCache()
        try:
            # Non-existent key returns None
            assert cache.get("nonexistent") is None

            # Create an entry
            shape = torch.Size([5, 5])
            dtype = torch.float32
            cache.allocate("test_key", shape, dtype)

            # Now it should exist
            entry = cache.get("test_key")
            assert entry is not None
            assert entry.shape == shape
        finally:
            cache.reset()

    def test_delete(self):
        """Test deleting a segment."""
        cache = SharedMemoryCache()
        try:
            shape = torch.Size([5, 5])
            dtype = torch.float32
            cache.allocate("test_key", shape, dtype)

            cache.delete("test_key")

            # Should no longer be in cache
            assert cache.get("test_key") is None
        finally:
            cache.reset()

    def test_reset(self):
        """Test resetting the cache."""
        cache = SharedMemoryCache()

        shape = torch.Size([5, 5])
        dtype = torch.float32
        cache.allocate("key1", shape, dtype)
        cache.allocate("key2", shape, dtype)

        cache.reset()

        # All keys should be gone
        assert cache.get("key1") is None
        assert cache.get("key2") is None


class TestSharedMemoryCacheAttach:
    """Test SharedMemoryCache attachment functionality for client-side handle caching."""

    def test_attach_caches_entry(self):
        """Test that entries are cached after first attach."""
        storage_cache = SharedMemoryCache()
        client_cache = SharedMemoryCache()
        try:
            # Create a segment on "storage"
            shape = torch.Size([10, 10])
            dtype = torch.float32
            entry = storage_cache.allocate("test_key", shape, dtype)
            descriptor = entry.descriptor

            # Client attaches
            client_entry1 = client_cache.attach(
                "test_key", descriptor, volume_id="vol1"
            )
            assert client_entry1 is not None
            assert client_entry1.shape == shape

            # Second call should return cached entry (same object)
            client_entry2 = client_cache.attach(
                "test_key", descriptor, volume_id="vol1"
            )
            assert client_entry2 is client_entry1
        finally:
            client_cache.reset()
            storage_cache.reset()

    def test_attach_different_volumes(self):
        """Test that different volumes have separate cache entries."""
        storage_cache = SharedMemoryCache()
        client_cache = SharedMemoryCache()
        try:
            shape = torch.Size([10, 10])
            dtype = torch.float32
            entry = storage_cache.allocate("test_key", shape, dtype)
            descriptor = entry.descriptor

            # Attach from two different volumes
            client_entry1 = client_cache.attach(
                "test_key", descriptor, volume_id="vol1"
            )
            client_entry2 = client_cache.attach(
                "test_key", descriptor, volume_id="vol2"
            )

            # Should be different cache entries (different volume_ids)
            assert client_entry1 is not client_entry2
        finally:
            client_cache.reset()
            storage_cache.reset()

    def test_attach_invalidates_stale_entry(self):
        """Test that stale entries are invalidated when storage_handle changes."""
        storage_cache = SharedMemoryCache()
        client_cache = SharedMemoryCache()
        try:
            shape = torch.Size([10, 10])
            dtype = torch.float32

            # Create first segment
            entry1 = storage_cache.allocate("test_key", shape, dtype)
            descriptor1 = entry1.descriptor

            # Client attaches
            client_entry1 = client_cache.attach(
                "test_key", descriptor1, volume_id="vol1"
            )
            original_handle = client_entry1.descriptor.storage_handle

            # Simulate storage recreating the segment (e.g., after delete)
            storage_cache.delete("test_key")
            entry2 = storage_cache.allocate("test_key", shape, dtype)
            descriptor2 = entry2.descriptor

            # The storage_handle should be different
            assert descriptor2.storage_handle != original_handle

            # Client re-attaches - should get new entry
            client_entry2 = client_cache.attach(
                "test_key", descriptor2, volume_id="vol1"
            )
            assert client_entry2 is not client_entry1
            assert client_entry2.descriptor.storage_handle == descriptor2.storage_handle
        finally:
            client_cache.reset()
            storage_cache.reset()

    def test_delete_with_volume_id(self):
        """Test deleting a specific cache entry with volume_id."""
        storage_cache = SharedMemoryCache()
        client_cache = SharedMemoryCache()
        try:
            shape = torch.Size([10, 10])
            dtype = torch.float32
            entry = storage_cache.allocate("test_key", shape, dtype)
            descriptor = entry.descriptor

            # Attach and cache
            client_entry1 = client_cache.attach(
                "test_key", descriptor, volume_id="vol1"
            )

            # Delete
            client_cache.delete("test_key", volume_id="vol1")

            # Next attach should create new entry
            client_entry2 = client_cache.attach(
                "test_key", descriptor, volume_id="vol1"
            )
            assert client_entry2 is not client_entry1
        finally:
            client_cache.reset()
            storage_cache.reset()

    def test_reset_clears_all_entries(self):
        """Test that reset clears all cached entries."""
        storage_cache = SharedMemoryCache()
        client_cache = SharedMemoryCache()
        try:
            shape = torch.Size([5, 5])
            dtype = torch.float32

            # Create and cache entries from two volumes
            entry = storage_cache.allocate("key1", shape, dtype)
            client_cache.attach("key1", entry.descriptor, volume_id="vol1")
            client_cache.attach("key1", entry.descriptor, volume_id="vol2")

            assert len(client_cache._entries) == 2

            # Reset
            client_cache.reset()

            assert len(client_cache._entries) == 0
        finally:
            storage_cache.reset()


class TestSharedMemoryTransportBuffer:
    """Test SharedMemoryTransportBuffer."""

    def test_init(self):
        """Test buffer initialization."""
        ref = MockStorageVolumeRef(volume_hostname="localhost")
        buffer = SharedMemoryTransportBuffer(ref)

        assert buffer.storage_volume_ref == ref
        assert buffer.shm_descriptor is None
        assert buffer.shape is None
        assert buffer.dtype is None
        assert buffer.is_object is False
        assert buffer.objects is None
        assert buffer.requires_handshake is False

    def test_getstate(self):
        """Test serialization excludes client-side handles."""
        ref = MockStorageVolumeRef(volume_hostname="localhost")
        buffer = SharedMemoryTransportBuffer(ref)
        buffer._client_tensor = "should_be_excluded"

        state = buffer.__getstate__()

        assert state["_client_tensor"] is None
        assert state["storage_volume_ref"] is None


class TestTensorRoundtrip:
    """Integration-style tests for tensor operations."""

    def test_tensor_copy_to_shared_memory(self):
        """Test copying a tensor to shared memory and back."""
        cache = SharedMemoryCache()
        try:
            # Create a test tensor
            original = torch.randn(100, 100)

            # Create shared memory entry
            entry = cache.allocate("test", original.shape, original.dtype)
            shm_tensor = entry.get_tensor()

            # Copy tensor to shared memory
            shm_tensor.copy_(original)

            # Verify data is correct
            assert torch.allclose(shm_tensor, original)

            # Create another view and verify
            shm_tensor2 = entry.get_tensor()
            assert torch.allclose(shm_tensor2, original)
        finally:
            cache.reset()

    def test_different_dtypes(self):
        """Test with different tensor dtypes."""
        cache = SharedMemoryCache()
        try:
            dtypes = [
                torch.float32,
                torch.float64,
                torch.int32,
                torch.int64,
                torch.float16,
            ]

            for dtype in dtypes:
                if dtype in (torch.float16, torch.bfloat16):
                    original = torch.randn(10, 10).to(dtype)
                elif dtype in (torch.int32, torch.int64):
                    original = torch.randint(0, 100, (10, 10), dtype=dtype)
                else:
                    original = torch.randn(10, 10, dtype=dtype)

                key = f"test_{dtype}"
                entry = cache.allocate(key, original.shape, original.dtype)
                shm_tensor = entry.get_tensor()
                shm_tensor.copy_(original)

                assert torch.equal(shm_tensor, original), f"Failed for {dtype}"
        finally:
            cache.reset()

    def test_persistence(self):
        """Test that tensor persists in shared memory for multiple accesses."""
        cache = SharedMemoryCache()
        try:
            original = torch.randn(50, 50)
            entry = cache.allocate("persistent", original.shape, original.dtype)

            # Write to shared memory
            shm_tensor = entry.get_tensor()
            shm_tensor.copy_(original)

            # Access multiple times - should get same data
            for _ in range(5):
                tensor = entry.get_tensor()
                assert torch.allclose(tensor, original)
        finally:
            cache.reset()


class MockRequest:
    """Mock Request for testing."""

    __slots__ = ("tensor_val", "objects", "is_object", "tensor_slice")

    def __init__(
        self, tensor_val=None, objects=None, is_object=False, tensor_slice=None
    ):
        self.tensor_val = tensor_val
        self.objects = objects
        self.is_object = is_object
        self.tensor_slice = tensor_slice


class TestSharedMemoryTransportBufferIntegration:
    """Integration tests for SharedMemoryTransportBuffer PUT/GET flow."""

    def test_put_does_not_serialize_tensor(self):
        """Verify tensor data is NOT included in serialization."""
        ref = MockStorageVolumeRef(volume_hostname="localhost")
        buffer = SharedMemoryTransportBuffer(ref)

        # Simulate _pre_put_hook with a tensor
        tensor = torch.randn(100, 100)
        buffer._key = "test_key"
        buffer.shape = tensor.shape
        buffer.dtype = tensor.dtype
        buffer._client_tensor = tensor

        # Get serialized state
        state = buffer.__getstate__()

        # Verify tensor data is NOT included in serialization
        assert state["_client_tensor"] is None, "Tensor should not be serialized"
        assert state["objects"] is None, "Objects should be None for tensor PUT"
        assert state["storage_volume_ref"] is None

        # Verify metadata IS serialized
        assert state["shape"] == tensor.shape
        assert state["dtype"] == tensor.dtype
        assert state["_key"] == "test_key"

    def test_requires_handshake_for_tensor_put(self):
        """Verify handshake is required for tensor PUT."""
        ref = MockStorageVolumeRef(volume_hostname="localhost")
        buffer = SharedMemoryTransportBuffer(ref)

        # Initially no handshake required
        assert buffer.requires_handshake is False

        # Simulate _pre_put_hook with tensor (sets both _client_tensor and _needs_handshake)
        buffer._client_tensor = torch.randn(10, 10)
        buffer._needs_handshake = True

        # Now handshake should be required
        assert buffer.requires_handshake is True

    def test_requires_handshake_not_for_objects(self):
        """Verify handshake is NOT required for object PUT."""
        ref = MockStorageVolumeRef(volume_hostname="localhost")
        buffer = SharedMemoryTransportBuffer(ref)

        # Simulate object PUT
        buffer.is_object = True
        buffer.objects = {"key": "value"}
        buffer._client_tensor = None

        # No handshake for objects
        assert buffer.requires_handshake is False

    @pytest.mark.asyncio
    async def test_pre_put_hook_tensor(self):
        """Test _pre_put_hook stores tensor locally without serializing."""
        ref = MockStorageVolumeRef(volume_hostname="localhost")
        buffer = SharedMemoryTransportBuffer(ref)

        tensor = torch.randn(50, 50)
        request = MockRequest(tensor_val=tensor)

        await buffer._pre_put_hook(request)

        # Verify metadata is stored
        assert buffer.shape == tensor.shape
        assert buffer.dtype == tensor.dtype

        # Verify tensor is stored locally
        assert buffer._client_tensor is not None
        assert torch.equal(buffer._client_tensor, tensor)

        # Verify objects is NOT set (critical for avoiding RPC of tensor)
        assert buffer.objects is None
        assert buffer.is_object is False

    @pytest.mark.asyncio
    async def test_pre_put_hook_non_contiguous_tensor(self):
        """Test _pre_put_hook handles non-contiguous tensor."""
        ref = MockStorageVolumeRef(volume_hostname="localhost")
        buffer = SharedMemoryTransportBuffer(ref)

        # Create non-contiguous tensor (transpose creates a view)
        tensor = torch.randn(50, 100).t()  # Now 100x50, non-contiguous
        assert not tensor.is_contiguous()

        request = MockRequest(tensor_val=tensor)

        await buffer._pre_put_hook(request)

        # Verify _client_tensor is contiguous
        assert buffer._client_tensor is not None
        assert buffer._client_tensor.is_contiguous()
        assert torch.equal(buffer._client_tensor, tensor)

    @pytest.mark.asyncio
    async def test_pre_put_hook_object(self):
        """Test _pre_put_hook handles objects correctly."""
        ref = MockStorageVolumeRef(volume_hostname="localhost")
        buffer = SharedMemoryTransportBuffer(ref)

        obj = {"key": "value", "list": [1, 2, 3]}
        request = MockRequest(objects=obj, is_object=True)

        await buffer._pre_put_hook(request)

        # Objects should use RPC path
        assert buffer.is_object is True
        assert buffer.objects == obj
        assert buffer._client_tensor is None

    @pytest.mark.asyncio
    async def test_handshake_allocates_segment(self):
        """Test recv_handshake allocates segment on storage and returns descriptor."""
        ref = MockStorageVolumeRef(volume_hostname="localhost")
        buffer = SharedMemoryTransportBuffer(ref)
        ctx = MockTransportContext()

        try:
            # Setup buffer as if it was sent from client
            buffer._key = "test_key"
            buffer.shape = torch.Size([100, 100])
            buffer.dtype = torch.float32
            buffer.is_object = False

            # Storage handles handshake
            descriptor = await buffer.recv_handshake(ctx)

            # Verify descriptor was returned
            assert isinstance(descriptor, SharedMemoryDescriptor)
            assert isinstance(descriptor.manager_handle, bytes)
            assert isinstance(descriptor.storage_handle, bytes)
            assert descriptor.size > 0
            assert descriptor.shape == buffer.shape
            assert descriptor.dtype == buffer.dtype

            # Verify segment exists in cache
            entry = ctx.get_shm_cache().get("test_key")
            assert entry is not None
        finally:
            ctx.reset()

    @pytest.mark.asyncio
    async def test_post_handshake_writes_to_segment(self):
        """Test _post_handshake writes tensor data to storage's segment."""
        cache = SharedMemoryCache()
        try:
            ref = MockStorageVolumeRef(volume_hostname="localhost")
            buffer = SharedMemoryTransportBuffer(ref)

            # Setup buffer with tensor data
            tensor = torch.randn(50, 50)
            buffer._key = "test_key"
            buffer.shape = tensor.shape
            buffer.dtype = tensor.dtype
            buffer._client_tensor = tensor

            # Create segment (simulating what storage does in recv_handshake)
            entry = cache.allocate("test_key", tensor.shape, tensor.dtype)
            descriptor = entry.descriptor

            # Client receives descriptor and writes data
            await buffer._post_handshake(descriptor)

            # Verify data was written to shared memory
            shm_tensor = entry.get_tensor()
            assert torch.allclose(shm_tensor, tensor)
        finally:
            ref.transport_context.reset()
            cache.reset()

    @pytest.mark.asyncio
    async def test_handle_put_reads_from_segment(self):
        """Test handle_put_request reads from segment after handshake."""
        ref = MockStorageVolumeRef(volume_hostname="localhost")
        buffer = SharedMemoryTransportBuffer(ref)
        ctx = MockTransportContext()

        try:
            # Setup buffer as if handshake completed
            tensor = torch.randn(50, 50)
            buffer._key = "test_key"
            buffer.shape = tensor.shape
            buffer.dtype = tensor.dtype

            # Create segment and write data (simulating handshake + _post_handshake)
            entry = ctx.get_shm_cache().allocate("test_key", tensor.shape, tensor.dtype)
            shm_tensor = entry.get_tensor()
            shm_tensor.copy_(tensor)

            # Storage handles put request
            request = MockRequest()
            result = await buffer.handle_put_request(ctx, request, None)

            # Verify result is the tensor backed by shared memory
            assert torch.allclose(result, tensor)
        finally:
            ctx.reset()

    @pytest.mark.asyncio
    async def test_full_put_get_roundtrip(self):
        """Test full PUT/GET roundtrip with shared memory."""
        ctx = MockTransportContext()

        try:
            ref = MockStorageVolumeRef(volume_hostname="localhost")
            original_tensor = torch.randn(100, 100)

            # === PUT FLOW ===

            # 1. Client: Create buffer and run _pre_put_hook
            put_buffer = SharedMemoryTransportBuffer(ref)
            put_buffer._key = "roundtrip_key"
            request = MockRequest(tensor_val=original_tensor)
            await put_buffer._pre_put_hook(request)

            # Verify handshake is required
            assert put_buffer.requires_handshake is True

            # 2. Storage: Handle handshake (allocate segment, return descriptor)
            descriptor = await put_buffer.recv_handshake(ctx)
            assert isinstance(descriptor, SharedMemoryDescriptor)

            # 3. Client: Post-handshake (write data to segment)
            await put_buffer._post_handshake(descriptor)

            # 4. Storage: Handle put request (get tensor from segment)
            stored_tensor = await put_buffer.handle_put_request(ctx, request, None)
            assert torch.allclose(stored_tensor, original_tensor)

            # 5. Client: Cleanup
            await put_buffer.drop()

            # === GET FLOW ===

            # 1. Client: Create buffer for GET
            get_buffer = SharedMemoryTransportBuffer(ref)
            get_buffer._key = "roundtrip_key"
            dest_tensor = torch.zeros(100, 100)
            get_buffer._client_tensor = dest_tensor

            # 2. Storage: Handle get request (data is in shared memory)
            await get_buffer.handle_get_request(ctx, stored_tensor)

            # Verify descriptor is set
            assert get_buffer.shm_descriptor is not None
            assert get_buffer.shm_descriptor.shape == original_tensor.shape
            assert get_buffer.shm_descriptor.dtype == original_tensor.dtype

            # 3. Client: Handle response
            result = await get_buffer._handle_storage_volume_response(get_buffer)

            # Verify data is correct
            assert torch.allclose(result, original_tensor)
            # Verify inplace copy to dest_tensor
            assert torch.allclose(dest_tensor, original_tensor)
            assert result is dest_tensor

            # 4. Client: Cleanup
            await get_buffer.drop()

        finally:
            ref.transport_context.reset()
            ctx.reset()

    @pytest.mark.asyncio
    async def test_drop_clears_client_tensor(self):
        """Test that drop clears _client_tensor."""
        ref = MockStorageVolumeRef(volume_hostname="localhost")
        buffer = SharedMemoryTransportBuffer(ref)

        # Setup buffer with data
        buffer._client_tensor = torch.randn(10, 10)
        buffer.objects = {"key": "value"}

        await buffer.drop()

        # Verify everything is cleared
        assert buffer._client_tensor is None
        assert buffer.objects is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestSharedMemoryTransportBufferGPU:
    """GPU-specific tests for SharedMemoryTransportBuffer."""

    @pytest.mark.asyncio
    async def test_pre_put_hook_gpu_tensor(self):
        """Test _pre_put_hook keeps GPU tensor reference (copy happens in _post_handshake)."""
        ref = MockStorageVolumeRef(volume_hostname="localhost")
        buffer = SharedMemoryTransportBuffer(ref)

        # Create GPU tensor
        tensor = torch.randn(50, 50, device="cuda")

        request = MockRequest(tensor_val=tensor)

        await buffer._pre_put_hook(request)

        # _client_tensor keeps original device - copy to CPU happens in _post_handshake
        assert buffer._client_tensor is not None
        assert buffer._client_tensor.device.type == "cuda"

    @pytest.mark.asyncio
    async def test_gpu_tensor_copied_in_post_handshake(self):
        """Test GPU tensor is copied directly to shared memory in _post_handshake."""
        cache = SharedMemoryCache()
        try:
            ref = MockStorageVolumeRef(volume_hostname="localhost")
            buffer = SharedMemoryTransportBuffer(ref)

            # Create GPU tensor
            tensor = torch.randn(50, 50, device="cuda")
            buffer._key = "test_key"
            buffer.shape = tensor.shape
            buffer.dtype = tensor.dtype
            buffer._client_tensor = tensor

            # Create segment (simulating what storage does in recv_handshake)
            entry = cache.allocate("test_key", tensor.shape, tensor.dtype)
            descriptor = entry.descriptor

            # Client receives descriptor and writes data (GPU -> shm copy)
            await buffer._post_handshake(descriptor)

            # Verify data was copied correctly to shared memory
            shm_tensor = entry.get_tensor()
            assert torch.allclose(shm_tensor, tensor.cpu())
        finally:
            ref.transport_context.reset()
            cache.reset()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
