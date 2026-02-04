# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for shared memory transport."""

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
        self._shm_cache.clear()


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

            # Verify we can write to the tensor
            tensor.fill_(42.0)
            tensor2 = entry.get_tensor()
            assert torch.all(tensor2 == 42.0)
        finally:
            cache.clear()


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
            cache.clear()


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
            cache.clear()

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
            cache.clear()

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
            cache.clear()

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
            cache.clear()

    def test_clear(self):
        """Test clearing the cache."""
        cache = SharedMemoryCache()

        shape = torch.Size([5, 5])
        dtype = torch.float32
        cache.allocate("key1", shape, dtype)
        cache.allocate("key2", shape, dtype)

        cache.clear()

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
            client_cache.clear()
            storage_cache.clear()

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
            client_cache.clear()
            storage_cache.clear()

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
            client_cache.clear()
            storage_cache.clear()

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
            client_cache.clear()
            storage_cache.clear()

    def test_clear_clears_all_entries(self):
        """Test that clear clears all cached entries."""
        storage_cache = SharedMemoryCache()
        client_cache = SharedMemoryCache()
        try:
            shape = torch.Size([5, 5])
            dtype = torch.float32

            # Create and cache entries from two volumes
            entry = storage_cache.allocate("key1", shape, dtype)
            client_cache.attach("key1", entry.descriptor, volume_id="vol1")
            client_cache.attach("key1", entry.descriptor, volume_id="vol2")

            # Verify entries exist via public API
            assert client_cache.get("key1", volume_id="vol1") is not None
            assert client_cache.get("key1", volume_id="vol2") is not None

            # Clear all entries
            client_cache.clear()

            # Verify entries are cleared via public API
            assert client_cache.get("key1", volume_id="vol1") is None
            assert client_cache.get("key1", volume_id="vol2") is None
        finally:
            storage_cache.clear()


class TestSharedMemoryCacheCoordinates:
    """Test SharedMemoryCache with DTensor shard coordinates."""

    def test_allocate_with_coordinates(self):
        """Test allocating different shards of the same key."""
        cache = SharedMemoryCache()
        try:
            shape = torch.Size([10, 10])
            dtype = torch.float32

            # Allocate two shards with different coordinates
            entry1 = cache.allocate("key", shape, dtype, coordinates=(0, 0))
            entry2 = cache.allocate("key", shape, dtype, coordinates=(0, 1))

            # Should be different entries
            assert entry1 is not entry2
            assert entry1.descriptor.storage_handle != entry2.descriptor.storage_handle

            # Each should be retrievable by its coordinates
            assert cache.get("key", coordinates=(0, 0)) is entry1
            assert cache.get("key", coordinates=(0, 1)) is entry2

            # Without coordinates should return None
            assert cache.get("key", coordinates=None) is None
        finally:
            cache.clear()

    def test_get_with_coordinates(self):
        """Test getting specific shard by coordinates."""
        cache = SharedMemoryCache()
        try:
            shape = torch.Size([5, 5])
            dtype = torch.float32

            # Allocate regular tensor (coordinates=None) and shards
            entry_regular = cache.allocate("key", shape, dtype, coordinates=None)
            entry_shard = cache.allocate("key", shape, dtype, coordinates=(1, 2))

            # Get by coordinates
            assert cache.get("key", coordinates=None) is entry_regular
            assert cache.get("key", coordinates=(1, 2)) is entry_shard
            assert cache.get("key", coordinates=(9, 9)) is None
        finally:
            cache.clear()

    @pytest.mark.parametrize(
        "delete_coords,expect_remaining",
        [
            ((0, 1), [(0, 0), (1, 0)]),  # specific shard
            (None, []),  # all shards
        ],
        ids=["specific", "all"],
    )
    def test_delete_shards(self, delete_coords, expect_remaining):
        """Test deleting shards with specific or None coordinates."""
        cache = SharedMemoryCache()
        try:
            shape = torch.Size([5, 5])
            dtype = torch.float32

            # Allocate multiple shards
            cache.allocate("key", shape, dtype, coordinates=(0, 0))
            cache.allocate("key", shape, dtype, coordinates=(0, 1))
            cache.allocate("key", shape, dtype, coordinates=(1, 0))

            # Delete shard(s)
            cache.delete("key", coordinates=delete_coords)

            # Check expected remaining shards
            all_coords = [(0, 0), (0, 1), (1, 0)]
            for coords in all_coords:
                if coords in expect_remaining:
                    assert cache.get("key", coordinates=coords) is not None
                else:
                    assert cache.get("key", coordinates=coords) is None
        finally:
            cache.clear()

    def test_attach_with_coordinates(self):
        """Test client-side attachment with coordinates."""
        storage_cache = SharedMemoryCache()
        client_cache = SharedMemoryCache()
        try:
            shape = torch.Size([10, 10])
            dtype = torch.float32

            # Storage allocates shards
            entry1 = storage_cache.allocate("key", shape, dtype, coordinates=(0, 0))
            entry2 = storage_cache.allocate("key", shape, dtype, coordinates=(0, 1))

            # Client attaches to each shard
            client_entry1 = client_cache.attach(
                "key", entry1.descriptor, volume_id="vol1", coordinates=(0, 0)
            )
            client_entry2 = client_cache.attach(
                "key", entry2.descriptor, volume_id="vol1", coordinates=(0, 1)
            )

            # Should be different entries
            assert client_entry1 is not client_entry2

            # Each should be cached separately
            assert (
                client_cache.get("key", volume_id="vol1", coordinates=(0, 0))
                is client_entry1
            )
            assert (
                client_cache.get("key", volume_id="vol1", coordinates=(0, 1))
                is client_entry2
            )
        finally:
            client_cache.clear()
            storage_cache.clear()

    def test_mixed_regular_and_sharded(self):
        """Test that regular tensors and sharded tensors coexist properly."""
        cache = SharedMemoryCache()
        try:
            shape = torch.Size([5, 5])
            dtype = torch.float32

            # Allocate a regular tensor (no coordinates)
            regular = cache.allocate("regular_key", shape, dtype)

            # Allocate sharded tensor
            shard1 = cache.allocate("sharded_key", shape, dtype, coordinates=(0,))
            shard2 = cache.allocate("sharded_key", shape, dtype, coordinates=(1,))

            # All should be retrievable
            assert cache.get("regular_key") is regular
            assert cache.get("sharded_key", coordinates=(0,)) is shard1
            assert cache.get("sharded_key", coordinates=(1,)) is shard2

            # Regular key without coordinates returns the entry
            assert cache.get("regular_key", coordinates=None) is regular
        finally:
            cache.clear()


class TestSharedMemoryTransportBuffer:
    """Test SharedMemoryTransportBuffer."""

    def test_getstate(self):
        """Test serialization excludes client-side handles."""
        ref = MockStorageVolumeRef(volume_hostname="localhost")
        buffer = SharedMemoryTransportBuffer(ref)
        buffer._client_tensor = "should_be_excluded"

        state = buffer.__getstate__()

        assert state["_client_tensor"] is None
        assert state["storage_volume_ref"] is None


class TestTensorRoundtripIntegration:
    """Integration tests for tensor operations."""

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
            cache.clear()


class MockTensorSlice:
    """Mock TensorSlice for testing DTensor coordinate extraction."""

    __slots__ = ("coordinates",)

    def __init__(self, coordinates: tuple):
        self.coordinates = coordinates


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
    @pytest.mark.parametrize(
        "hook_type,coordinates",
        [
            ("put", (1, 2)),
            ("get", (3, 4)),
        ],
    )
    async def test_hook_extracts_coordinates_from_tensor_slice(
        self, hook_type, coordinates
    ):
        """Test that _pre_put_hook and _pre_get_hook extract coordinates from tensor_slice."""
        ref = MockStorageVolumeRef(volume_hostname="localhost")
        buffer = SharedMemoryTransportBuffer(ref)

        tensor = torch.randn(50, 50) if hook_type == "put" else torch.zeros(50, 50)
        tensor_slice = MockTensorSlice(coordinates=coordinates)
        request = MockRequest(tensor_val=tensor, tensor_slice=tensor_slice)

        if hook_type == "put":
            await buffer._pre_put_hook(request)
            # For PUT, verify additional state
            assert buffer.shape == tensor.shape
            assert buffer.dtype == tensor.dtype
            assert buffer._client_tensor is not None
            assert buffer._needs_handshake is True
        else:
            await buffer._pre_get_hook("test_key", request)
            # For GET, verify key is set
            assert buffer._key == "test_key"
            assert buffer._client_tensor is tensor

        # Both should extract coordinates
        assert buffer._coordinates == coordinates

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "coordinates,shape",
        [
            (None, torch.Size([100, 100])),
            ((0, 1), torch.Size([50, 50])),
        ],
        ids=["no_coords", "with_coords"],
    )
    async def test_handshake_allocates_segment(self, coordinates, shape):
        """Test recv_handshake allocates segment on storage and returns descriptor."""
        ref = MockStorageVolumeRef(volume_hostname="localhost")
        buffer = SharedMemoryTransportBuffer(ref)
        ctx = MockTransportContext()

        try:
            # Setup buffer as if it was sent from client
            buffer._key = "test_key"
            buffer._coordinates = coordinates
            buffer.shape = shape
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

            # Verify segment exists in cache with correct coordinates
            entry = ctx.get_shm_cache().get("test_key", coordinates=coordinates)
            assert entry is not None
            assert entry.shape == shape

            if coordinates is not None:
                # Without coordinates should return None when we allocated with coords
                assert ctx.get_shm_cache().get("test_key", coordinates=None) is None
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
            cache.clear()

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
    @pytest.mark.parametrize(
        "scenario",
        ["no_entry", "shape_mismatch"],
    )
    async def test_handle_get_request_rpc_fallback(self, scenario):
        """Test handle_get_request falls back to RPC when entry missing or shape mismatch."""
        ref = MockStorageVolumeRef(volume_hostname="localhost")
        buffer = SharedMemoryTransportBuffer(ref)
        ctx = MockTransportContext()

        try:
            buffer._key = "test_key"

            if scenario == "shape_mismatch":
                # Allocate a segment with one shape
                ctx.get_shm_cache().allocate(
                    "test_key", torch.Size([10, 10]), torch.float32
                )
                # Request with different shape
                data = torch.randn(20, 20)
            else:
                # No entry in cache
                data = torch.randn(50, 50)

            await buffer.handle_get_request(ctx, data)

            # Should fall back to RPC
            assert buffer.shm_descriptor is None
            assert buffer.objects is data
            if scenario == "no_entry":
                assert buffer.is_object is False
        finally:
            ctx.reset()

    @pytest.mark.asyncio
    async def test_handle_get_request_with_coordinates(self):
        """Test handle_get_request uses coordinates to find correct shard."""
        ref = MockStorageVolumeRef(volume_hostname="localhost")
        buffer = SharedMemoryTransportBuffer(ref)
        ctx = MockTransportContext()

        try:
            buffer._key = "test_key"
            buffer._coordinates = (0, 1)

            # Allocate two shards with different coordinates
            entry1 = ctx.get_shm_cache().allocate(
                "test_key", torch.Size([10, 10]), torch.float32, coordinates=(0, 0)
            )
            entry2 = ctx.get_shm_cache().allocate(
                "test_key", torch.Size([10, 10]), torch.float32, coordinates=(0, 1)
            )

            data = torch.randn(10, 10)
            await buffer.handle_get_request(ctx, data)

            # Should find the entry with matching coordinates
            assert buffer.shm_descriptor is entry2.descriptor
        finally:
            ctx.reset()

    @pytest.mark.asyncio
    async def test_handle_get_request_object(self):
        """Test handle_get_request handles non-tensor data."""
        ref = MockStorageVolumeRef(volume_hostname="localhost")
        buffer = SharedMemoryTransportBuffer(ref)
        ctx = MockTransportContext()

        try:
            buffer._key = "test_key"

            # Pass a non-tensor (object)
            data = {"key": "value", "list": [1, 2, 3]}
            await buffer.handle_get_request(ctx, data)

            # Should set is_object and objects
            assert buffer.is_object is True
            assert buffer.objects == data
            assert buffer.shm_descriptor is None
        finally:
            ctx.reset()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("has_client_tensor", [True, False])
    async def test_handle_response_rpc_fallback(self, has_client_tensor):
        """Test _handle_storage_volume_response RPC fallback with and without client tensor."""
        ref = MockStorageVolumeRef(volume_hostname="localhost")
        buffer = SharedMemoryTransportBuffer(ref)

        # Setup: client may or may not have a destination tensor
        if has_client_tensor:
            dest_tensor = torch.zeros(10, 10)
            buffer._client_tensor = dest_tensor
        else:
            buffer._client_tensor = None

        # Response buffer with no shm_descriptor (RPC fallback)
        response_buffer = SharedMemoryTransportBuffer(ref)
        response_buffer.shm_descriptor = None
        response_buffer.is_object = False
        response_buffer.objects = torch.randn(10, 10)

        result = await buffer._handle_storage_volume_response(response_buffer)

        if has_client_tensor:
            # Should copy to dest_tensor
            assert result is dest_tensor
            assert torch.allclose(dest_tensor, response_buffer.objects)
        else:
            # Should return objects directly
            assert result is response_buffer.objects

    @pytest.mark.asyncio
    async def test_handle_response_clones_when_no_client_tensor(self):
        """Test _handle_storage_volume_response clones when no client tensor provided."""
        cache = SharedMemoryCache()
        try:
            ref = MockStorageVolumeRef(volume_hostname="localhost")
            buffer = SharedMemoryTransportBuffer(ref)
            buffer._key = "test_key"
            buffer._client_tensor = None  # No destination tensor

            # Create a segment and write data
            entry = cache.allocate("test_key", torch.Size([10, 10]), torch.float32)
            original_data = torch.randn(10, 10)
            entry.get_tensor().copy_(original_data)

            # Response buffer with descriptor
            response_buffer = SharedMemoryTransportBuffer(ref)
            response_buffer.shm_descriptor = entry.descriptor
            response_buffer.is_object = False

            result = await buffer._handle_storage_volume_response(response_buffer)

            # Should be a clone (not the same object as shm tensor)
            shm_tensor = entry.get_tensor()
            assert torch.allclose(result, shm_tensor)
            # Verify it's a clone by modifying result
            result.fill_(999)
            assert not torch.allclose(shm_tensor, result)
        finally:
            ref.transport_context.reset()
            cache.clear()

    @pytest.mark.asyncio
    async def test_handle_response_with_coordinates(self):
        """Test _handle_storage_volume_response uses coordinates for client cache."""
        cache = SharedMemoryCache()
        try:
            ref = MockStorageVolumeRef(volume_hostname="localhost")
            buffer = SharedMemoryTransportBuffer(ref)
            buffer._key = "test_key"
            buffer._coordinates = (1, 2)
            buffer._client_tensor = torch.zeros(10, 10)

            # Create a segment
            entry = cache.allocate("test_key", torch.Size([10, 10]), torch.float32)
            original_data = torch.randn(10, 10)
            entry.get_tensor().copy_(original_data)

            # Response buffer with descriptor
            response_buffer = SharedMemoryTransportBuffer(ref)
            response_buffer.shm_descriptor = entry.descriptor
            response_buffer.is_object = False

            await buffer._handle_storage_volume_response(response_buffer)

            # Verify client cache has entry with coordinates
            client_cache = ref.transport_context.get_shm_cache()
            cached_entry = client_cache.get(
                "test_key", volume_id="test_volume", coordinates=(1, 2)
            )
            assert cached_entry is not None
        finally:
            ref.transport_context.reset()
            cache.clear()


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
            cache.clear()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
