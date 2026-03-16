# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for shared memory transport."""

import pytest
import torch
from torchstore.transport.buffers import TransportContext
from torchstore.transport.shared_memory import (
    allocate_shared_tensor,
    is_local_to_volume,
    SharedMemoryCache,
    SharedMemoryDescriptor,
    SharedMemoryTransportBuffer,
    ShmContext,
)
from torchstore.transport.types import Request
from torchstore.utils import get_local_hostname


class MockStorageVolumeRef:
    """Mock StorageVolumeRef for testing."""

    def __init__(self, volume_hostname=None, volume_id="test_volume"):
        self.volume_hostname = volume_hostname
        self.volume_id = volume_id
        # Client-side transport context (mirrors TorchStoreStrategy.transport_context)
        self.transport_context = TransportContext()


@pytest.fixture
def ref():
    """MockStorageVolumeRef with automatic client-side transport context teardown."""
    ref = MockStorageVolumeRef(volume_hostname="localhost")
    yield ref
    ref.transport_context.clear()


@pytest.fixture
def ctx():
    """Server-side TransportContext with automatic teardown."""
    ctx = TransportContext()
    yield ctx
    ctx.clear()


class TestHelperFunctions:
    """Test helper functions."""

    def test_get_local_hostname(self, monkeypatch):
        """Test get_local_hostname returns string and respects env var."""
        # Default behavior
        hostname = get_local_hostname()
        assert isinstance(hostname, str)
        assert len(hostname) > 0

        # With env var override
        monkeypatch.setenv("HOSTNAME", "test-hostname")
        assert get_local_hostname() == "test-hostname"

    def test_is_local_to_volume_same_host(self):
        """Test is_local_to_volume returns True for same host."""
        hostname = get_local_hostname()
        ref = MockStorageVolumeRef(volume_hostname=hostname)
        assert is_local_to_volume(ref) is True

    def test_is_local_to_volume_different_host(self):
        """Test is_local_to_volume returns False for different or None host."""
        # Different host
        ref = MockStorageVolumeRef(volume_hostname="some-other-host-12345")
        assert is_local_to_volume(ref) is False

        # None hostname
        ref = MockStorageVolumeRef(volume_hostname=None)
        assert is_local_to_volume(ref) is False


class TestAllocateSharedTensor:
    """Test allocate_shared_tensor helper function."""

    def test_allocate_shared_tensor(self):
        """Test allocating a shared memory tensor with correct properties."""
        shape = torch.Size([10, 10])
        dtype = torch.float32

        tensor = allocate_shared_tensor(shape, dtype)

        assert tensor.shape == shape
        assert tensor.dtype == dtype
        assert tensor.is_shared()
        # Verify memory is prefaulted (initialized to 0)
        assert torch.all(tensor == 0)


class TestSharedMemoryDescriptor:
    """Test SharedMemoryDescriptor dataclass."""

    def test_from_tensor_shared(self):
        """Test deriving descriptor from a shared memory tensor."""
        tensor = allocate_shared_tensor(torch.Size([10, 10]), torch.float32)

        descriptor = SharedMemoryDescriptor.from_tensor(tensor)

        assert descriptor is not None
        assert isinstance(descriptor, SharedMemoryDescriptor)
        assert descriptor.shape == tensor.shape
        assert descriptor.dtype == tensor.dtype
        assert isinstance(descriptor.manager_handle, bytes)
        assert isinstance(descriptor.storage_handle, bytes)
        assert descriptor.size > 0

    def test_from_tensor_non_shared(self):
        """Test from_tensor returns None for non-shared tensor."""
        tensor = torch.randn(10, 10)

        descriptor = SharedMemoryDescriptor.from_tensor(tensor)

        assert descriptor is None

    def test_from_tensor_view(self):
        """Test from_tensor returns None for view/slice tensor."""
        # Allocate shared memory tensor
        full_tensor = allocate_shared_tensor(torch.Size([100]), torch.float32)

        # Create a view/slice (storage.size() != tensor.numel() * element_size)
        view_tensor = full_tensor[:50]
        assert view_tensor.is_shared()  # Still shared, but is a view

        descriptor = SharedMemoryDescriptor.from_tensor(view_tensor)

        # Should return None because it's a view
        assert descriptor is None

    def test_attach_and_get_tensor(self):
        """Test attaching to a segment via descriptor and getting tensor."""
        shape = torch.Size([10, 10])
        dtype = torch.float32

        # Allocate shared memory tensor and write data
        tensor = allocate_shared_tensor(shape, dtype)
        original = torch.randn(10, 10)
        tensor.copy_(original)

        # Get descriptor and attach
        descriptor = SharedMemoryDescriptor.from_tensor(tensor)
        entry = descriptor.attach()

        # Verify entry properties
        assert entry.shape == shape
        assert entry.dtype == dtype

        # Verify we can read the data
        result = entry.get_tensor()
        assert torch.allclose(result, original)

        # Verify modifications persist
        result.fill_(42.0)
        result2 = entry.get_tensor()
        assert torch.all(result2 == 42.0)


class TestSharedMemoryCache:
    """Test SharedMemoryCache."""

    def test_allocate(self):
        """Test allocate method creates and caches shared memory."""
        cache = SharedMemoryCache()
        shape = torch.Size([10, 10])
        dtype = torch.float32

        entry, descriptor = cache.allocate("test_key", shape, dtype)

        # Verify entry and descriptor are valid
        assert entry is not None
        assert descriptor is not None
        assert entry.shape == shape
        assert entry.dtype == dtype
        assert descriptor.shape == shape
        assert descriptor.dtype == dtype

        # Verify entry is cached
        cache_key = ("test_key", descriptor.storage_handle)
        assert cache_key in cache._entries
        assert cache._entries[cache_key] is entry

        cache.clear()

    def test_attach_caches_entry(self):
        """Test that entries are cached and reused on attach."""
        cache = SharedMemoryCache()
        shape = torch.Size([10, 10])
        dtype = torch.float32

        # Create a shared memory tensor and get descriptor
        tensor = allocate_shared_tensor(shape, dtype)
        descriptor = SharedMemoryDescriptor.from_tensor(tensor)

        # First attach
        entry1 = cache.attach("test_key", descriptor)
        assert entry1 is not None
        assert entry1.shape == shape

        # Second attach with same key and handle returns cached entry
        entry2 = cache.attach("test_key", descriptor)
        assert entry2 is entry1

        # Different handle creates different entry
        tensor2 = allocate_shared_tensor(shape, dtype)
        descriptor2 = SharedMemoryDescriptor.from_tensor(tensor2)
        entry3 = cache.attach("test_key", descriptor2)
        assert entry3 is not entry1
        assert entry3.descriptor.storage_handle != entry1.descriptor.storage_handle

        cache.clear()

    def test_clear(self):
        """Test clearing the cache removes all entries."""
        cache = SharedMemoryCache()
        shape = torch.Size([5, 5])
        dtype = torch.float32

        tensor = allocate_shared_tensor(shape, dtype)
        descriptor = SharedMemoryDescriptor.from_tensor(tensor)

        cache.attach("key1", descriptor)
        cache.attach("key2", descriptor)

        cache.clear()

        assert len(cache._entries) == 0

    @pytest.mark.asyncio
    async def test_cache_reuse_on_same_key_puts(self, ref):
        """Test that putting twice to same key with same-spec tensor reuses cache entry."""
        shm_cache = ref.transport_context.get(SharedMemoryCache)

        # First PUT: allocate new shared memory
        buffer1 = SharedMemoryTransportBuffer(ref)
        tensor1 = torch.randn(50, 50)
        requests1 = [Request(key="test_key", tensor_val=tensor1)]

        await buffer1._post_handshake([None], requests1)  # No existing descriptor

        # Verify cache has 1 entry after first PUT
        assert len(shm_cache._entries) == 1
        first_descriptor = buffer1._contexts[0].descriptor

        # Second PUT: reuse existing shared memory
        buffer2 = SharedMemoryTransportBuffer(ref)
        tensor2 = torch.randn(50, 50)
        requests2 = [Request(key="test_key", tensor_val=tensor2)]

        await buffer2._post_handshake(
            [first_descriptor], requests2
        )  # Existing descriptor from handshake

        # Verify cache still has only 1 entry (same SHM reused)
        assert len(shm_cache._entries) == 1
        assert (
            buffer2._contexts[0].descriptor.storage_handle
            == first_descriptor.storage_handle
        )

        # Verify the data was updated
        entry = shm_cache._entries[("test_key", first_descriptor.storage_handle)]
        assert torch.allclose(entry.get_tensor(), tensor2)


class TestSharedMemoryTransportBuffer:
    """Test SharedMemoryTransportBuffer."""

    def test_getstate(self, ref):
        """Test serialization excludes client-side handles."""
        buffer = SharedMemoryTransportBuffer(ref)

        state = buffer.__getstate__()

        assert state["storage_volume_ref"] is None


class TestSharedMemoryTransportBufferPUT:
    """Tests for SharedMemoryTransportBuffer PUT flow."""

    def test_requires_handshake_reflects_needs_handshake(self, ref):
        """Test requires_handshake mirrors _needs_handshake (True in PUT, False in GET)."""
        buffer = SharedMemoryTransportBuffer(ref)
        requests = [Request(key="key1", tensor_val=torch.randn(5, 5))]

        assert buffer.requires_handshake(requests) is False
        buffer._needs_handshake = True
        assert buffer.requires_handshake(requests) is True

    @pytest.mark.asyncio
    async def test_post_handshake_stores_objects(self, ref):
        """Test _post_handshake stores objects in _contexts."""
        buffer = SharedMemoryTransportBuffer(ref)

        obj = {"key": "value", "list": [1, 2, 3]}
        requests = [Request(key="obj_key", objects=obj, is_object=True)]

        await buffer._post_handshake([None], requests)

        assert buffer._contexts[0].objects == obj

    @pytest.mark.asyncio
    async def test_post_handshake_allocates_and_copies(self, ref):
        """Test _post_handshake allocates new or reuses existing segment and copies data."""
        # Case 1: No descriptor - allocate new
        buffer1 = SharedMemoryTransportBuffer(ref)
        tensor1 = torch.randn(50, 50)
        requests1 = [Request(key="test_key_1", tensor_val=tensor1)]

        await buffer1._post_handshake([None], requests1)

        descriptor1 = buffer1._contexts[0].descriptor
        assert descriptor1 is not None
        assert descriptor1.shape == tensor1.shape
        entry1 = descriptor1.attach()
        assert torch.allclose(entry1.get_tensor(), tensor1)

        # Case 2: With descriptor - reuse existing
        buffer2 = SharedMemoryTransportBuffer(ref)
        tensor2 = torch.randn(50, 50)

        shm_tensor = allocate_shared_tensor(tensor2.shape, tensor2.dtype)
        descriptor = SharedMemoryDescriptor.from_tensor(shm_tensor)
        requests2 = [Request(key="test_key_2", tensor_val=tensor2)]

        await buffer2._post_handshake([descriptor], requests2)

        assert buffer2._contexts[0].descriptor is descriptor
        assert torch.allclose(shm_tensor, tensor2)

    @pytest.mark.asyncio
    async def test_handle_put_attaches_and_returns_tensor(self, ref, ctx):
        """Test handle_put_request attaches to new segment and returns tensor."""
        buffer = SharedMemoryTransportBuffer(ref)

        # Setup: create segment and store descriptor
        tensor = torch.randn(50, 50)
        shm_tensor = allocate_shared_tensor(tensor.shape, tensor.dtype)
        shm_tensor.copy_(tensor)
        descriptor = SharedMemoryDescriptor.from_tensor(shm_tensor)

        buffer._contexts = [ShmContext(descriptor=descriptor)]

        # Handle put with no current_object (new key)
        results = await buffer.handle_put_request(
            ctx, [(Request(key="test_key"), None)]
        )

        assert len(results) == 1
        assert torch.allclose(results[0], tensor)

        # Handle put with matching existing tensor returns existing
        buffer._contexts = [ShmContext(descriptor=descriptor)]
        results2 = await buffer.handle_put_request(
            ctx, [(Request(key="test_key"), shm_tensor)]
        )
        assert results2[0] is shm_tensor

    @pytest.mark.asyncio
    async def test_handle_put_batch_request_mixed(self, ref, ctx):
        """Verify SV handles batch with both tensor and object entries."""
        buffer = SharedMemoryTransportBuffer(ref)

        # Tensor entry via SHM
        t1 = torch.randn(10, 10)
        shm_tensor1 = allocate_shared_tensor(t1.shape, t1.dtype)
        shm_tensor1.copy_(t1)
        descriptor1 = SharedMemoryDescriptor.from_tensor(shm_tensor1)
        buffer._contexts = [
            ShmContext(descriptor=descriptor1),
            ShmContext(objects={"value": 99}, use_rpc=True),
        ]

        results = await buffer.handle_put_request(
            ctx,
            [
                (Request(key="tensor_key"), None),
                (Request(key="obj_key", is_object=True), None),
            ],
        )

        assert torch.allclose(results[0], t1)
        assert results[1] == {"value": 99}


class TestSharedMemoryTransportBufferGET:
    """Tests for SharedMemoryTransportBuffer GET flow."""

    @pytest.mark.asyncio
    async def test_handle_get_shared_tensor(self, ref, ctx):
        """Test handle_get_request populates _contexts for shared tensor."""
        buffer = SharedMemoryTransportBuffer(ref)

        data = allocate_shared_tensor(torch.Size([10, 10]), torch.float32)
        expected_descriptor = SharedMemoryDescriptor.from_tensor(data)

        request = Request(key="test_key")
        await buffer.handle_get_request(ctx, [(request, data)])

        assert len(buffer._contexts) == 1
        assert buffer._contexts[0].descriptor is not None
        assert (
            buffer._contexts[0].descriptor.storage_handle
            == expected_descriptor.storage_handle
        )
        assert buffer._contexts[0].use_rpc is False

    @pytest.mark.asyncio
    async def test_handle_get_non_shared_fallback(self, ref, ctx):
        """Test handle_get_request falls back to RPC for non-shared tensor."""
        buffer = SharedMemoryTransportBuffer(ref)

        data = torch.randn(50, 50)
        assert not data.is_shared()

        request = Request(key="test_key")
        await buffer.handle_get_request(ctx, [(request, data)])

        assert len(buffer._contexts) == 1
        assert buffer._contexts[0].use_rpc is True
        assert buffer._contexts[0].objects is data

    @pytest.mark.asyncio
    async def test_handle_get_view_fallback(self, ref, ctx):
        """Test handle_get_request falls back to RPC for view/slice tensor."""
        buffer = SharedMemoryTransportBuffer(ref)

        # Create a view of a shared tensor
        full_tensor = allocate_shared_tensor(torch.Size([100]), torch.float32)
        view_tensor = full_tensor[:50]
        assert view_tensor.is_shared()

        request = Request(key="test_key")
        await buffer.handle_get_request(ctx, [(request, view_tensor)])

        # Should fall back to RPC because it's a view
        assert len(buffer._contexts) == 1
        assert buffer._contexts[0].use_rpc is True
        assert buffer._contexts[0].objects is view_tensor

    @pytest.mark.asyncio
    async def test_handle_get_object(self, ref, ctx):
        """Test handle_get_request handles non-tensor data."""
        buffer = SharedMemoryTransportBuffer(ref)

        data = {"key": "value", "list": [1, 2, 3]}

        request = Request(key="test_key", is_object=True)
        await buffer.handle_get_request(ctx, [(request, data)])

        assert len(buffer._contexts) == 1
        assert buffer._contexts[0].use_rpc is True
        assert buffer._contexts[0].objects == data

    @pytest.mark.asyncio
    async def test_handle_response_shared_memory(self, ref):
        """Test _handle_storage_volume_response with shared memory path."""
        buffer = SharedMemoryTransportBuffer(ref)
        dest_tensor = torch.zeros(10, 10)
        requests = [Request(key="test_key", tensor_val=dest_tensor)]

        # Create segment and response
        shm_tensor = allocate_shared_tensor(torch.Size([10, 10]), torch.float32)
        original_data = torch.randn(10, 10)
        shm_tensor.copy_(original_data)
        descriptor = SharedMemoryDescriptor.from_tensor(shm_tensor)

        response_buffer = SharedMemoryTransportBuffer(ref)
        response_buffer._contexts = [ShmContext(descriptor=descriptor)]

        results = await buffer._handle_storage_volume_response(
            requests, response_buffer
        )

        # Should copy to dest_tensor
        assert len(results) == 1
        assert results[0] is dest_tensor
        assert torch.allclose(dest_tensor, original_data)

        # Verify entry is cached
        cache_key = ("test_key", descriptor.storage_handle)
        assert cache_key in ref.transport_context.get(SharedMemoryCache)._entries

    @pytest.mark.asyncio
    async def test_handle_response_rpc_fallback(self, ref):
        """Test _handle_storage_volume_response RPC fallback path."""
        # Case 1: With client tensor - copies to it
        buffer1 = SharedMemoryTransportBuffer(ref)
        dest_tensor = torch.zeros(10, 10)
        requests1 = [Request(key="test_key", tensor_val=dest_tensor)]

        response1 = SharedMemoryTransportBuffer(ref)
        rpc_data = torch.randn(10, 10)
        response1._contexts = [ShmContext(objects=rpc_data, use_rpc=True)]

        results1 = await buffer1._handle_storage_volume_response(requests1, response1)
        assert len(results1) == 1
        assert results1[0] is dest_tensor
        assert torch.allclose(dest_tensor, rpc_data)

        # Case 2: No client tensor - returns objects directly
        buffer2 = SharedMemoryTransportBuffer(ref)
        requests2 = [Request(key="test_key", tensor_val=None)]

        response2 = SharedMemoryTransportBuffer(ref)
        rpc_data2 = torch.randn(10, 10)
        response2._contexts = [ShmContext(objects=rpc_data2, use_rpc=True)]

        results2 = await buffer2._handle_storage_volume_response(requests2, response2)
        assert len(results2) == 1
        assert results2[0] is rpc_data2

    @pytest.mark.asyncio
    async def test_handle_response_shm_not_found_error(self, ref):
        """Test _handle_storage_volume_response raises helpful error for missing SHM."""
        buffer = SharedMemoryTransportBuffer(ref)
        requests = [Request(key="missing_key", tensor_val=torch.zeros(10, 10))]

        # Create a descriptor with bogus handles that won't resolve
        bad_descriptor = SharedMemoryDescriptor(
            manager_handle=b"/invalid_shm_manager_999",
            storage_handle=b"/invalid_shm_storage_999",
            size=400,
            shape=torch.Size([10, 10]),
            dtype=torch.float32,
        )

        response = SharedMemoryTransportBuffer(ref)
        response._contexts = [ShmContext(descriptor=bad_descriptor)]

        with pytest.raises(RuntimeError, match="Shared memory storage not found"):
            await buffer._handle_storage_volume_response(requests, response)

    @pytest.mark.asyncio
    async def test_mutable_shm_env_var(self, ref, monkeypatch):
        """Test TORCHSTORE_MUTABLE_SHM env var controls clone behavior."""
        import torchstore.transport.shared_memory as shm_module

        # Create shared memory segment with data
        shm_tensor = allocate_shared_tensor(torch.Size([10, 10]), torch.float32)
        original_data = torch.randn(10, 10)
        shm_tensor.copy_(original_data)
        descriptor = SharedMemoryDescriptor.from_tensor(shm_tensor)

        # Test with MUTABLE_SHM=False (default) - should return cloned tensor
        monkeypatch.setattr(shm_module, "MUTABLE_SHM", False)
        buffer1 = SharedMemoryTransportBuffer(ref)
        requests1 = [Request(key="test_key", tensor_val=None)]

        response1 = SharedMemoryTransportBuffer(ref)
        response1._contexts = [ShmContext(descriptor=descriptor)]

        results1 = await buffer1._handle_storage_volume_response(requests1, response1)

        result1 = results1[0]
        # Should be a clone (different storage)
        assert not result1.is_shared()  # Clone is not in shared memory
        assert torch.allclose(result1, original_data)

        # Modifying the clone should NOT affect original shared memory
        result1.fill_(999.0)
        assert torch.allclose(shm_tensor, original_data)

        # Test with MUTABLE_SHM=True - should return tensor backed by shared memory
        monkeypatch.setattr(shm_module, "MUTABLE_SHM", True)
        ref.transport_context.clear()  # Clear cache to force re-attach
        buffer2 = SharedMemoryTransportBuffer(ref)
        requests2 = [Request(key="test_key", tensor_val=None)]

        response2 = SharedMemoryTransportBuffer(ref)
        response2._contexts = [ShmContext(descriptor=descriptor)]

        results2 = await buffer2._handle_storage_volume_response(requests2, response2)

        result2 = results2[0]
        # Should share storage with the shared memory segment
        assert result2.is_shared()
        assert torch.allclose(result2, original_data)

        # Modifying the result SHOULD affect original shared memory
        result2.fill_(123.0)
        assert torch.all(shm_tensor == 123.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestSharedMemoryTransportBufferGPU:
    """GPU-specific tests for SharedMemoryTransportBuffer."""

    @pytest.mark.asyncio
    async def test_gpu_tensor_copied_in_post_handshake(self, ref):
        """Test GPU tensor is copied to shared memory in _post_handshake."""
        buffer = SharedMemoryTransportBuffer(ref)

        tensor = torch.randn(50, 50, device="cuda")
        entries = [Request(key="test_key", tensor_val=tensor)]

        shm_tensor = allocate_shared_tensor(tensor.shape, tensor.dtype)
        descriptor = SharedMemoryDescriptor.from_tensor(shm_tensor)

        await buffer._post_handshake([descriptor], entries)

        assert torch.allclose(shm_tensor, tensor.cpu())


class TestSharedMemoryTransportBufferBatch:
    """Tests for SharedMemoryTransportBuffer batch operations."""

    @pytest.mark.asyncio
    async def test_post_handshake_allocates_and_copies_batch(self, ref):
        """Verify _post_handshake allocates/attaches correctly for multiple tensors and copies data."""
        buffer = SharedMemoryTransportBuffer(ref)

        t1 = torch.randn(10, 10)
        t2 = torch.randn(20, 5)

        # Simulate post-handshake: no existing descriptors (all None)
        requests = [
            Request(key="k1", tensor_val=t1),
            Request(key="k2", tensor_val=t2),
        ]
        descriptors = [None, None]
        await buffer._post_handshake(descriptors, requests)

        # Verify both descriptors were allocated
        assert buffer._contexts[0].descriptor is not None
        assert buffer._contexts[1].descriptor is not None

        # Verify data was copied
        entry1 = buffer._contexts[0].descriptor.attach()
        assert torch.allclose(entry1.get_tensor(), t1)

        entry2 = buffer._contexts[1].descriptor.attach()
        assert torch.allclose(entry2.get_tensor(), t2)

    @pytest.mark.asyncio
    async def test_recv_handshake(self, ref, ctx):
        """Verify SV returns descriptors for existing tensors, None for new."""
        buffer = SharedMemoryTransportBuffer(ref)

        existing_tensor = allocate_shared_tensor(torch.Size([10, 10]), torch.float32)
        expected_descriptor = SharedMemoryDescriptor.from_tensor(existing_tensor)

        results = await buffer.recv_handshake(
            ctx,
            [
                (Request(key="k1"), existing_tensor),
                (Request(key="k2"), None),
                (Request(key="k3"), "not_a_tensor"),
            ],
        )

        assert len(results) == 3
        assert results[0] is not None
        assert results[0].storage_handle == expected_descriptor.storage_handle
        assert results[1] is None
        assert results[2] is None

    @pytest.mark.asyncio
    async def test_batch_drop_clears_all_state(self, ref):
        """Verify drop clears all batch fields."""
        buffer = SharedMemoryTransportBuffer(ref)

        buffer._contexts = [ShmContext()]
        await buffer.drop()

        assert buffer._contexts == []

    @pytest.mark.asyncio
    async def test_handle_get_request_mixed(self, ref, ctx):
        """Mix of SHM tensors, objects, and RPC fallback tensors."""
        buffer = SharedMemoryTransportBuffer(ref)

        shm_tensor = allocate_shared_tensor(torch.Size([10, 10]), torch.float32)
        obj_data = {"key": "val"}
        rpc_tensor = torch.randn(5, 5)  # Not shared, will fall back to RPC

        entries = [
            (Request(key="k_shm"), shm_tensor),
            (Request(key="k_obj", is_object=True), obj_data),
            (Request(key="k_rpc"), rpc_tensor),
        ]
        await buffer.handle_get_request(ctx, entries)

        assert len(buffer._contexts) == 3

        # SHM path
        assert buffer._contexts[0].descriptor is not None
        assert buffer._contexts[0].descriptor.shape == torch.Size([10, 10])
        assert buffer._contexts[0].use_rpc is False

        # Object path
        assert buffer._contexts[1].use_rpc is True
        assert buffer._contexts[1].objects == {"key": "val"}

        # RPC fallback path (non-shared tensor)
        assert buffer._contexts[2].use_rpc is True
        assert buffer._contexts[2].objects is rpc_tensor

    @pytest.mark.asyncio
    async def test_handle_get_response_mixed(self, ref):
        """Client-side response with SHM + RPC fallback entries in one batch."""
        buffer = SharedMemoryTransportBuffer(ref)

        dest_tensor = torch.zeros(10, 10)
        requests = [
            Request(key="shm_key", tensor_val=dest_tensor),
            Request(key="obj_key", tensor_val=None),
        ]

        shm_tensor = allocate_shared_tensor(torch.Size([10, 10]), torch.float32)
        original = torch.randn(10, 10)
        shm_tensor.copy_(original)
        descriptor = SharedMemoryDescriptor.from_tensor(shm_tensor)

        response = SharedMemoryTransportBuffer(ref)
        response._contexts = [
            ShmContext(descriptor=descriptor),
            ShmContext(objects={"data": 42}, use_rpc=True),
        ]

        results = await buffer._handle_storage_volume_response(requests, response)

        assert len(results) == 2
        assert results[0] is dest_tensor
        assert torch.allclose(dest_tensor, original)
        assert results[1] == {"data": 42}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
