# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for shared memory transport."""

import pytest
import torch
from torchstore.transport.shared_memory import (
    allocate_shared_tensor,
    is_local_to_volume,
    SharedMemoryCache,
    SharedMemoryDescriptor,
    SharedMemoryTransportBuffer,
)
from torchstore.transport.types import KeyedRequest
from torchstore.utils import get_local_hostname


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


class MockRequest:
    """Mock Request for testing."""

    __slots__ = ("tensor_val", "objects", "is_object")

    def __init__(self, tensor_val=None, objects=None, is_object=False):
        self.tensor_val = tensor_val
        self.objects = objects
        self.is_object = is_object


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
    async def test_cache_reuse_on_same_key_puts(self):
        """Test that putting twice to same key with same-spec tensor reuses cache entry."""
        ref = MockStorageVolumeRef(volume_hostname="localhost")
        shm_cache = ref.transport_context.get_shm_cache()

        # First PUT: allocate new shared memory
        buffer1 = SharedMemoryTransportBuffer(ref)
        tensor1 = torch.randn(50, 50)
        entries1 = [KeyedRequest("test_key", MockRequest(tensor_val=tensor1))]

        await buffer1._post_handshake([None], entries1)  # No existing descriptor

        # Verify cache has 1 entry after first PUT
        assert len(shm_cache._entries) == 1
        first_descriptor = buffer1._batch_shm_descriptors["test_key"]

        # Second PUT: reuse existing shared memory
        buffer2 = SharedMemoryTransportBuffer(ref)
        tensor2 = torch.randn(50, 50)
        entries2 = [KeyedRequest("test_key", MockRequest(tensor_val=tensor2))]

        await buffer2._post_handshake(
            [first_descriptor], entries2
        )  # Existing descriptor from handshake

        # Verify cache still has only 1 entry (same SHM reused)
        assert len(shm_cache._entries) == 1
        assert (
            buffer2._batch_shm_descriptors["test_key"].storage_handle
            == first_descriptor.storage_handle
        )

        # Verify the data was updated
        entry = shm_cache._entries[("test_key", first_descriptor.storage_handle)]
        assert torch.allclose(entry.get_tensor(), tensor2)

        ref.transport_context.reset()


class TestSharedMemoryTransportBuffer:
    """Test SharedMemoryTransportBuffer."""

    def test_getstate(self):
        """Test serialization excludes client-side handles."""
        ref = MockStorageVolumeRef(volume_hostname="localhost")
        buffer = SharedMemoryTransportBuffer(ref)
        buffer._batch_client_tensors = [torch.randn(10, 10)]
        buffer._batch_keys = ["test_key"]

        state = buffer.__getstate__()

        assert state["_batch_client_tensors"] == []
        assert state["_batch_keys"] == []
        assert state["storage_volume_ref"] is None


class TestSharedMemoryTransportBufferPUT:
    """Tests for SharedMemoryTransportBuffer PUT flow."""

    @pytest.mark.asyncio
    async def test_requires_handshake_true_in_put_context(self):
        """Test requires_handshake returns True in PUT context."""
        ref = MockStorageVolumeRef(volume_hostname="localhost")
        buffer = SharedMemoryTransportBuffer(ref)
        buffer._needs_handshake = True

        tensor = torch.randn(50, 50)
        entries = [KeyedRequest("key1", MockRequest(tensor_val=tensor))]

        assert buffer.requires_handshake(entries) is True

    @pytest.mark.asyncio
    async def test_requires_handshake_false_outside_put_context(self):
        """Test requires_handshake returns False when not in PUT context (e.g., GET)."""
        ref = MockStorageVolumeRef(volume_hostname="localhost")
        buffer = SharedMemoryTransportBuffer(ref)
        # _needs_handshake defaults to False

        tensor = torch.randn(50, 50)
        entries = [KeyedRequest("key1", MockRequest(tensor_val=tensor))]

        result = buffer.requires_handshake(entries)

        assert result is False

    @pytest.mark.asyncio
    async def test_pre_put_hook_stores_objects(self):
        """Test _pre_put_hook stores objects in _batch_objects."""
        ref = MockStorageVolumeRef(volume_hostname="localhost")
        buffer = SharedMemoryTransportBuffer(ref)

        obj = {"key": "value", "list": [1, 2, 3]}
        entries = [KeyedRequest("obj_key", MockRequest(objects=obj, is_object=True))]

        await buffer._pre_put_hook(entries)

        assert buffer._batch_objects["obj_key"] == obj

    @pytest.mark.asyncio
    async def test_recv_handshake_returns_none_for_new(self):
        """Test recv_handshake returns None for new key (client will allocate)."""
        ref = MockStorageVolumeRef(volume_hostname="localhost")
        buffer = SharedMemoryTransportBuffer(ref)
        ctx = MockTransportContext()

        # No current_object means new allocation on client
        results = await buffer.recv_handshake(
            ctx, [(KeyedRequest("key1", MockRequest()), None)]
        )

        assert len(results) == 1
        assert results[0] is None

    @pytest.mark.asyncio
    async def test_recv_handshake_returns_descriptor_for_existing(self):
        """Test recv_handshake returns descriptor for existing shared tensor."""
        ref = MockStorageVolumeRef(volume_hostname="localhost")
        buffer = SharedMemoryTransportBuffer(ref)
        ctx = MockTransportContext()

        # Create an existing shared tensor
        existing_tensor = allocate_shared_tensor(torch.Size([50, 50]), torch.float32)
        expected_descriptor = SharedMemoryDescriptor.from_tensor(existing_tensor)

        results = await buffer.recv_handshake(
            ctx, [(KeyedRequest("key1", MockRequest()), existing_tensor)]
        )

        assert len(results) == 1
        assert results[0] is not None
        assert results[0].storage_handle == expected_descriptor.storage_handle

    @pytest.mark.asyncio
    async def test_post_handshake_allocates_and_copies(self):
        """Test _post_handshake allocates new or reuses existing segment and copies data."""
        ref = MockStorageVolumeRef(volume_hostname="localhost")

        # Case 1: No descriptor - allocate new
        buffer1 = SharedMemoryTransportBuffer(ref)
        tensor1 = torch.randn(50, 50)
        entries1 = [KeyedRequest("test_key_1", MockRequest(tensor_val=tensor1))]

        await buffer1._post_handshake([None], entries1)

        descriptor1 = buffer1._batch_shm_descriptors["test_key_1"]
        assert descriptor1 is not None
        assert descriptor1.shape == tensor1.shape
        entry1 = descriptor1.attach()
        assert torch.allclose(entry1.get_tensor(), tensor1)

        # Case 2: With descriptor - reuse existing
        buffer2 = SharedMemoryTransportBuffer(ref)
        tensor2 = torch.randn(50, 50)

        shm_tensor = allocate_shared_tensor(tensor2.shape, tensor2.dtype)
        descriptor = SharedMemoryDescriptor.from_tensor(shm_tensor)
        entries2 = [KeyedRequest("test_key_2", MockRequest(tensor_val=tensor2))]

        await buffer2._post_handshake([descriptor], entries2)

        assert buffer2._batch_shm_descriptors["test_key_2"] is descriptor
        assert torch.allclose(shm_tensor, tensor2)

        ref.transport_context.reset()

    @pytest.mark.asyncio
    async def test_handle_put_attaches_and_returns_tensor(self):
        """Test handle_put_request attaches to new segment and returns tensor."""
        ref = MockStorageVolumeRef(volume_hostname="localhost")
        buffer = SharedMemoryTransportBuffer(ref)
        ctx = MockTransportContext()

        # Setup: create segment and store descriptor
        tensor = torch.randn(50, 50)
        shm_tensor = allocate_shared_tensor(tensor.shape, tensor.dtype)
        shm_tensor.copy_(tensor)
        descriptor = SharedMemoryDescriptor.from_tensor(shm_tensor)

        buffer._batch_shm_descriptors = {"test_key": descriptor}

        # Handle put with no current_object (new key)
        request = MockRequest()
        results = await buffer.handle_put_request(
            ctx, [(KeyedRequest("test_key", request), None)]
        )

        assert "test_key" in results
        assert torch.allclose(results["test_key"], tensor)

        # Handle put with matching existing tensor returns existing
        results2 = await buffer.handle_put_request(
            ctx, [(KeyedRequest("test_key", request), shm_tensor)]
        )
        assert results2["test_key"] is shm_tensor

    @pytest.mark.asyncio
    async def test_handle_put_batch_request_tensors(self):
        """Verify SV handles batch of tensor put requests."""
        ref = MockStorageVolumeRef(volume_hostname="localhost")
        buffer = SharedMemoryTransportBuffer(ref)
        ctx = MockTransportContext()

        # Setup two tensor entries
        t1 = torch.randn(10, 10)
        shm_tensor1 = allocate_shared_tensor(t1.shape, t1.dtype)
        shm_tensor1.copy_(t1)
        descriptor1 = SharedMemoryDescriptor.from_tensor(shm_tensor1)

        t2 = torch.randn(5, 5)
        shm_tensor2 = allocate_shared_tensor(t2.shape, t2.dtype)
        shm_tensor2.copy_(t2)
        descriptor2 = SharedMemoryDescriptor.from_tensor(shm_tensor2)

        buffer._batch_shm_descriptors = {"k1": descriptor1, "k2": descriptor2}

        req1 = MockRequest()
        req2 = MockRequest()

        results = await buffer.handle_put_request(
            ctx,
            [
                (KeyedRequest("k1", req1), None),  # new tensor
                (KeyedRequest("k2", req2), None),  # new tensor
            ],
        )

        assert "k1" in results
        assert "k2" in results
        assert torch.allclose(results["k1"], t1)
        assert torch.allclose(results["k2"], t2)

    @pytest.mark.asyncio
    async def test_handle_put_batch_request_mixed(self):
        """Verify SV handles batch with both tensor and object entries."""
        ref = MockStorageVolumeRef(volume_hostname="localhost")
        buffer = SharedMemoryTransportBuffer(ref)
        ctx = MockTransportContext()

        # Tensor entry via SHM
        t1 = torch.randn(10, 10)
        shm_tensor1 = allocate_shared_tensor(t1.shape, t1.dtype)
        shm_tensor1.copy_(t1)
        descriptor1 = SharedMemoryDescriptor.from_tensor(shm_tensor1)
        buffer._batch_shm_descriptors = {"tensor_key": descriptor1}

        # Object entry via _batch_objects
        buffer._batch_objects = {"obj_key": {"value": 99}}

        results = await buffer.handle_put_request(
            ctx,
            [
                (KeyedRequest("tensor_key", MockRequest()), None),
                (KeyedRequest("obj_key", MockRequest(is_object=True)), None),
            ],
        )

        assert torch.allclose(results["tensor_key"], t1)
        assert results["obj_key"] == {"value": 99}


class TestSharedMemoryTransportBufferGET:
    """Tests for SharedMemoryTransportBuffer GET flow."""

    @pytest.mark.asyncio
    async def test_handle_get_shared_tensor(self):
        """Test handle_get_request populates _batch_shm_descriptors for shared tensor."""
        ref = MockStorageVolumeRef(volume_hostname="localhost")
        buffer = SharedMemoryTransportBuffer(ref)
        ctx = MockTransportContext()

        data = allocate_shared_tensor(torch.Size([10, 10]), torch.float32)
        expected_descriptor = SharedMemoryDescriptor.from_tensor(data)

        entry = KeyedRequest("test_key", MockRequest())
        await buffer.handle_get_request(ctx, [(entry, data)])

        assert 0 in buffer._batch_shm_descriptors
        assert (
            buffer._batch_shm_descriptors[0].storage_handle
            == expected_descriptor.storage_handle
        )
        assert 0 not in buffer._batch_objects

    @pytest.mark.asyncio
    async def test_handle_get_non_shared_fallback(self):
        """Test handle_get_request falls back to RPC for non-shared tensor."""
        ref = MockStorageVolumeRef(volume_hostname="localhost")
        buffer = SharedMemoryTransportBuffer(ref)
        ctx = MockTransportContext()

        data = torch.randn(50, 50)
        assert not data.is_shared()

        entry = KeyedRequest("test_key", MockRequest())
        await buffer.handle_get_request(ctx, [(entry, data)])

        assert 0 not in buffer._batch_shm_descriptors
        assert 0 in buffer._batch_objects
        assert buffer._batch_objects[0] is data

    @pytest.mark.asyncio
    async def test_handle_get_view_fallback(self):
        """Test handle_get_request falls back to RPC for view/slice tensor."""
        ref = MockStorageVolumeRef(volume_hostname="localhost")
        buffer = SharedMemoryTransportBuffer(ref)
        ctx = MockTransportContext()

        # Create a view of a shared tensor
        full_tensor = allocate_shared_tensor(torch.Size([100]), torch.float32)
        view_tensor = full_tensor[:50]
        assert view_tensor.is_shared()

        entry = KeyedRequest("test_key", MockRequest())
        await buffer.handle_get_request(ctx, [(entry, view_tensor)])

        # Should fall back to RPC because it's a view
        assert 0 not in buffer._batch_shm_descriptors
        assert 0 in buffer._batch_objects
        assert buffer._batch_objects[0] is view_tensor

    @pytest.mark.asyncio
    async def test_handle_get_object(self):
        """Test handle_get_request handles non-tensor data."""
        ref = MockStorageVolumeRef(volume_hostname="localhost")
        buffer = SharedMemoryTransportBuffer(ref)
        ctx = MockTransportContext()

        data = {"key": "value", "list": [1, 2, 3]}

        entry = KeyedRequest("test_key", MockRequest(is_object=True))
        await buffer.handle_get_request(ctx, [(entry, data)])

        assert 0 in buffer._batch_objects
        assert buffer._batch_objects[0] == data
        assert 0 not in buffer._batch_shm_descriptors

    @pytest.mark.asyncio
    async def test_handle_response_shared_memory(self):
        """Test _handle_storage_volume_response with shared memory path."""
        ref = MockStorageVolumeRef(volume_hostname="localhost")
        buffer = SharedMemoryTransportBuffer(ref)
        buffer._batch_keys = ["test_key"]
        dest_tensor = torch.zeros(10, 10)
        buffer._batch_client_tensors = [dest_tensor]

        # Create segment and response
        shm_tensor = allocate_shared_tensor(torch.Size([10, 10]), torch.float32)
        original_data = torch.randn(10, 10)
        shm_tensor.copy_(original_data)
        descriptor = SharedMemoryDescriptor.from_tensor(shm_tensor)

        response_buffer = SharedMemoryTransportBuffer(ref)
        response_buffer._batch_shm_descriptors = {0: descriptor}

        results = await buffer._handle_storage_volume_response(response_buffer)

        # Should copy to dest_tensor
        assert len(results) == 1
        assert results[0] is dest_tensor
        assert torch.allclose(dest_tensor, original_data)

        # Verify entry is cached
        cache_key = ("test_key", descriptor.storage_handle)
        assert cache_key in ref.transport_context.get_shm_cache()._entries

        ref.transport_context.reset()

    @pytest.mark.asyncio
    async def test_handle_response_rpc_fallback(self):
        """Test _handle_storage_volume_response RPC fallback path."""
        ref = MockStorageVolumeRef(volume_hostname="localhost")

        # Case 1: With client tensor - copies to it
        buffer1 = SharedMemoryTransportBuffer(ref)
        dest_tensor = torch.zeros(10, 10)
        buffer1._batch_client_tensors = [dest_tensor]
        buffer1._batch_keys = ["test_key"]

        response1 = SharedMemoryTransportBuffer(ref)
        rpc_data = torch.randn(10, 10)
        response1._batch_objects = {0: rpc_data}

        results1 = await buffer1._handle_storage_volume_response(response1)
        assert len(results1) == 1
        assert results1[0] is dest_tensor
        assert torch.allclose(dest_tensor, rpc_data)

        # Case 2: No client tensor - returns objects directly
        buffer2 = SharedMemoryTransportBuffer(ref)
        buffer2._batch_client_tensors = [None]
        buffer2._batch_keys = ["test_key"]

        response2 = SharedMemoryTransportBuffer(ref)
        rpc_data2 = torch.randn(10, 10)
        response2._batch_objects = {0: rpc_data2}

        results2 = await buffer2._handle_storage_volume_response(response2)
        assert len(results2) == 1
        assert results2[0] is rpc_data2

    @pytest.mark.asyncio
    async def test_mutable_shm_env_var(self, monkeypatch):
        """Test TORCHSTORE_MUTABLE_SHM env var controls clone behavior."""
        import torchstore.transport.shared_memory as shm_module

        ref = MockStorageVolumeRef(volume_hostname="localhost")

        # Create shared memory segment with data
        shm_tensor = allocate_shared_tensor(torch.Size([10, 10]), torch.float32)
        original_data = torch.randn(10, 10)
        shm_tensor.copy_(original_data)
        descriptor = SharedMemoryDescriptor.from_tensor(shm_tensor)

        # Test with MUTABLE_SHM=False (default) - should return cloned tensor
        monkeypatch.setattr(shm_module, "MUTABLE_SHM", False)
        buffer1 = SharedMemoryTransportBuffer(ref)
        buffer1._batch_keys = ["test_key"]
        buffer1._batch_client_tensors = [None]

        response1 = SharedMemoryTransportBuffer(ref)
        response1._batch_shm_descriptors = {0: descriptor}

        results1 = await buffer1._handle_storage_volume_response(response1)

        result1 = results1[0]
        # Should be a clone (different storage)
        assert not result1.is_shared()  # Clone is not in shared memory
        assert torch.allclose(result1, original_data)

        # Modifying the clone should NOT affect original shared memory
        result1.fill_(999.0)
        assert torch.allclose(shm_tensor, original_data)

        # Test with MUTABLE_SHM=True - should return tensor backed by shared memory
        monkeypatch.setattr(shm_module, "MUTABLE_SHM", True)
        ref.transport_context.reset()  # Clear cache to force re-attach
        buffer2 = SharedMemoryTransportBuffer(ref)
        buffer2._batch_keys = ["test_key"]
        buffer2._batch_client_tensors = [None]

        response2 = SharedMemoryTransportBuffer(ref)
        response2._batch_shm_descriptors = {0: descriptor}

        results2 = await buffer2._handle_storage_volume_response(response2)

        result2 = results2[0]
        # Should share storage with the shared memory segment
        assert result2.is_shared()
        assert torch.allclose(result2, original_data)

        # Modifying the result SHOULD affect original shared memory
        result2.fill_(123.0)
        assert torch.all(shm_tensor == 123.0)

        ref.transport_context.reset()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestSharedMemoryTransportBufferGPU:
    """GPU-specific tests for SharedMemoryTransportBuffer."""

    @pytest.mark.asyncio
    async def test_gpu_tensor_copied_in_post_handshake(self):
        """Test GPU tensor is copied to shared memory in _post_handshake."""
        ref = MockStorageVolumeRef(volume_hostname="localhost")
        buffer = SharedMemoryTransportBuffer(ref)

        tensor = torch.randn(50, 50, device="cuda")
        entries = [KeyedRequest("test_key", MockRequest(tensor_val=tensor))]

        shm_tensor = allocate_shared_tensor(tensor.shape, tensor.dtype)
        descriptor = SharedMemoryDescriptor.from_tensor(shm_tensor)

        await buffer._post_handshake([descriptor], entries)

        assert torch.allclose(shm_tensor, tensor.cpu())

        ref.transport_context.reset()


class TestSharedMemoryTransportBufferBatch:
    """Tests for SharedMemoryTransportBuffer batch operations."""

    @pytest.mark.asyncio
    async def test_post_handshake_allocates_and_copies_batch(self):
        """Verify _post_handshake allocates/attaches correctly for multiple tensors and copies data."""
        ref = MockStorageVolumeRef(volume_hostname="localhost")
        buffer = SharedMemoryTransportBuffer(ref)

        t1 = torch.randn(10, 10)
        t2 = torch.randn(20, 5)

        # Simulate post-handshake: no existing descriptors (all None)
        entries = [
            KeyedRequest("k1", MockRequest(tensor_val=t1)),
            KeyedRequest("k2", MockRequest(tensor_val=t2)),
        ]
        descriptors = [None, None]
        await buffer._post_handshake(descriptors, entries)

        # Verify both descriptors were allocated
        assert buffer._batch_shm_descriptors["k1"] is not None
        assert buffer._batch_shm_descriptors["k2"] is not None

        # Verify data was copied
        entry1 = buffer._batch_shm_descriptors["k1"].attach()
        assert torch.allclose(entry1.get_tensor(), t1)

        entry2 = buffer._batch_shm_descriptors["k2"].attach()
        assert torch.allclose(entry2.get_tensor(), t2)

        ref.transport_context.reset()

    @pytest.mark.asyncio
    async def test_recv_handshake(self):
        """Verify SV returns descriptors for existing tensors, None for new."""
        ref = MockStorageVolumeRef(volume_hostname="localhost")
        buffer = SharedMemoryTransportBuffer(ref)
        ctx = MockTransportContext()

        existing_tensor = allocate_shared_tensor(torch.Size([10, 10]), torch.float32)
        expected_descriptor = SharedMemoryDescriptor.from_tensor(existing_tensor)

        results = await buffer.recv_handshake(
            ctx,
            [
                (KeyedRequest("k1", MockRequest()), existing_tensor),
                (KeyedRequest("k2", MockRequest()), None),
                (KeyedRequest("k3", MockRequest()), "not_a_tensor"),
            ],
        )

        assert len(results) == 3
        assert results[0] is not None
        assert results[0].storage_handle == expected_descriptor.storage_handle
        assert results[1] is None
        assert results[2] is None

    @pytest.mark.asyncio
    async def test_batch_drop_clears_all_state(self):
        """Verify drop clears all batch fields."""
        ref = MockStorageVolumeRef(volume_hostname="localhost")
        buffer = SharedMemoryTransportBuffer(ref)

        buffer._batch_shm_descriptors = {"k1": None}
        buffer._batch_objects = {"k2": "some_object"}
        buffer._batch_client_tensors = [torch.randn(5, 5)]
        buffer._batch_keys = ["k1"]

        await buffer.drop()

        assert buffer._batch_shm_descriptors == {}
        assert buffer._batch_objects == {}
        assert buffer._batch_client_tensors == []
        assert buffer._batch_keys == []

    @pytest.mark.asyncio
    async def test_handle_get_request_shm_tensors(self):
        """SV populates _batch_shm_descriptors by position for shared tensors."""
        ref = MockStorageVolumeRef(volume_hostname="localhost")
        buffer = SharedMemoryTransportBuffer(ref)
        ctx = MockTransportContext()

        t1 = allocate_shared_tensor(torch.Size([10, 10]), torch.float32)
        t2 = allocate_shared_tensor(torch.Size([5, 5]), torch.float32)

        entries = [
            (KeyedRequest("k1", MockRequest()), t1),
            (KeyedRequest("k2", MockRequest()), t2),
        ]
        await buffer.handle_get_request(ctx, entries)

        assert 0 in buffer._batch_shm_descriptors
        assert 1 in buffer._batch_shm_descriptors
        assert buffer._batch_shm_descriptors[0].shape == torch.Size([10, 10])
        assert buffer._batch_shm_descriptors[1].shape == torch.Size([5, 5])
        assert len(buffer._batch_objects) == 0

    @pytest.mark.asyncio
    async def test_handle_get_request_objects(self):
        """SV stores non-tensors in _batch_objects."""
        ref = MockStorageVolumeRef(volume_hostname="localhost")
        buffer = SharedMemoryTransportBuffer(ref)
        ctx = MockTransportContext()

        obj1 = {"data": 1}
        obj2 = [1, 2, 3]

        entries = [
            (KeyedRequest("k1", MockRequest(is_object=True)), obj1),
            (KeyedRequest("k2", MockRequest(is_object=True)), obj2),
        ]
        await buffer.handle_get_request(ctx, entries)

        assert 0 in buffer._batch_objects
        assert 1 in buffer._batch_objects
        assert buffer._batch_objects[0] == {"data": 1}
        assert buffer._batch_objects[1] == [1, 2, 3]
        assert len(buffer._batch_shm_descriptors) == 0

    @pytest.mark.asyncio
    async def test_handle_get_request_mixed(self):
        """Mix of SHM tensors, objects, and RPC fallback tensors."""
        ref = MockStorageVolumeRef(volume_hostname="localhost")
        buffer = SharedMemoryTransportBuffer(ref)
        ctx = MockTransportContext()

        shm_tensor = allocate_shared_tensor(torch.Size([10, 10]), torch.float32)
        obj_data = {"key": "val"}
        rpc_tensor = torch.randn(5, 5)  # Not shared, will fall back to RPC

        entries = [
            (KeyedRequest("k_shm", MockRequest()), shm_tensor),
            (KeyedRequest("k_obj", MockRequest(is_object=True)), obj_data),
            (KeyedRequest("k_rpc", MockRequest()), rpc_tensor),
        ]
        await buffer.handle_get_request(ctx, entries)

        # SHM path
        assert 0 in buffer._batch_shm_descriptors
        assert buffer._batch_shm_descriptors[0].shape == torch.Size([10, 10])

        # Object path
        assert 1 in buffer._batch_objects
        assert buffer._batch_objects[1] == {"key": "val"}

        # RPC fallback path (non-shared tensor)
        assert 2 in buffer._batch_objects
        assert buffer._batch_objects[2] is rpc_tensor

    @pytest.mark.asyncio
    async def test_handle_get_response_shm(self):
        """Client-side SHM attach + copy to inplace."""
        ref = MockStorageVolumeRef(volume_hostname="localhost")
        buffer = SharedMemoryTransportBuffer(ref)

        dest = torch.zeros(10, 10)
        buffer._batch_keys = ["test_key"]
        buffer._batch_client_tensors = [dest]

        shm_tensor = allocate_shared_tensor(torch.Size([10, 10]), torch.float32)
        original = torch.randn(10, 10)
        shm_tensor.copy_(original)
        descriptor = SharedMemoryDescriptor.from_tensor(shm_tensor)

        response = SharedMemoryTransportBuffer(ref)
        response._batch_shm_descriptors = {0: descriptor}

        results = await buffer._handle_storage_volume_response(response)

        assert len(results) == 1
        assert results[0] is dest
        assert torch.allclose(dest, original)

        ref.transport_context.reset()

    @pytest.mark.asyncio
    async def test_handle_get_response_rpc_fallback(self):
        """Client-side fallback path."""
        ref = MockStorageVolumeRef(volume_hostname="localhost")
        buffer = SharedMemoryTransportBuffer(ref)

        dest = torch.zeros(10, 10)
        buffer._batch_keys = ["test_key"]
        buffer._batch_client_tensors = [dest]

        rpc_data = torch.randn(10, 10)
        response = SharedMemoryTransportBuffer(ref)
        response._batch_objects = {0: rpc_data}

        results = await buffer._handle_storage_volume_response(response)

        assert len(results) == 1
        assert results[0] is dest
        assert torch.allclose(dest, rpc_data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
