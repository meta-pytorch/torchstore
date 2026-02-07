# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for CUDA IPC transport."""

import asyncio
import gc
import threading
from unittest.mock import Mock

import pytest
import torch

from torchstore.transport.cuda_ipc import (
    create_ipc_handle,
    cuda_ipc_available,
    CudaIPCHandle,
    CudaIPCTransportBuffer,
)


class MockStorageVolumeRef:
    """Mock StorageVolumeRef for testing."""

    __slots__ = ["volume_id"]

    def __init__(self, volume_id="test_volume"):
        self.volume_id = volume_id


class MockRequest:
    """Mock Request for testing."""

    __slots__ = ("tensor_val", "objects", "is_object")

    def __init__(self, tensor_val=None, objects=None, is_object=False):
        self.tensor_val = tensor_val
        self.objects = objects
        self.is_object = is_object


class TestCudaIPCAvailability:
    """Test CUDA IPC availability detection."""

    def test_cuda_ipc_available(self):
        """Test cuda_ipc_available returns a boolean."""
        available = cuda_ipc_available()
        assert isinstance(available, bool)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_ipc_with_cuda(self):
        """Test CUDA IPC when CUDA is available."""
        # Should return True if CUDA is working properly
        available = cuda_ipc_available()
        # We don't assert True because some CUDA setups may not support IPC
        assert isinstance(available, bool)


class TestCudaIPCHandle:
    """Test CUDA IPC handle creation and reconstruction."""

    @pytest.mark.skipif(not cuda_ipc_available(), reason="CUDA IPC not available")
    def test_create_ipc_handle(self):
        """Test creating IPC handle from a GPU tensor."""
        tensor = torch.randn(10, 10, device="cuda:0")
        handle = create_ipc_handle(tensor)

        assert isinstance(handle, CudaIPCHandle)
        assert handle.tensor_size == tuple(tensor.size())
        assert handle.dtype == tensor.dtype
        assert handle.storage_device >= 0

    @pytest.mark.skipif(not cuda_ipc_available(), reason="CUDA IPC not available")
    def test_create_ipc_handle_non_contiguous(self):
        """Test creating IPC handle from non-contiguous tensor."""
        tensor = torch.randn(20, 20, device="cuda:0")[::2, ::2]
        assert not tensor.is_contiguous()

        # Should succeed (will make contiguous internally)
        handle = create_ipc_handle(tensor)
        assert isinstance(handle, CudaIPCHandle)

    def test_create_ipc_handle_cpu_tensor_fails(self):
        """Test that creating IPC handle from CPU tensor raises error."""
        tensor = torch.randn(10, 10)
        with pytest.raises(ValueError, match="must be on a CUDA device"):
            create_ipc_handle(tensor)

    @pytest.mark.skipif(not cuda_ipc_available(), reason="CUDA IPC not available")
    def test_reconstruct_tensor(self):
        """Test reconstructing tensor from IPC handle."""
        original = torch.randn(10, 10, device="cuda:0")
        handle = create_ipc_handle(original)

        reconstructed = handle.reconstruct_tensor()

        assert reconstructed.shape == original.shape
        assert reconstructed.dtype == original.dtype
        assert reconstructed.device.type == "cuda"
        # Values should match (same underlying storage)
        assert torch.allclose(reconstructed, original)


class TestCudaIPCTransportBuffer:
    """Test CUDA IPC transport buffer."""

    @pytest.mark.skipif(not cuda_ipc_available(), reason="CUDA IPC not available")
    @pytest.mark.asyncio
    async def test_pre_put_hook_gpu_tensor(self):
        """Test _pre_put_hook with GPU tensor."""
        ref = MockStorageVolumeRef()
        buffer = CudaIPCTransportBuffer(ref)

        tensor = torch.randn(10, 10, device="cuda:0")
        request = MockRequest(tensor_val=tensor)

        await buffer._pre_put_hook(request)

        assert buffer.ipc_handle is not None
        assert buffer.shape == tensor.shape
        assert buffer.dtype == tensor.dtype
        assert not buffer.is_object

    @pytest.mark.skipif(not cuda_ipc_available(), reason="CUDA IPC not available")
    @pytest.mark.asyncio
    async def test_pre_put_hook_cpu_tensor(self):
        """Test _pre_put_hook with CPU tensor (should move to GPU)."""
        ref = MockStorageVolumeRef()
        buffer = CudaIPCTransportBuffer(ref)

        tensor = torch.randn(10, 10)  # CPU tensor
        request = MockRequest(tensor_val=tensor)

        await buffer._pre_put_hook(request)

        assert buffer.ipc_handle is not None
        assert buffer._source_tensor.is_cuda
        assert buffer.shape == tensor.shape
        assert buffer.dtype == tensor.dtype

    @pytest.mark.asyncio
    async def test_pre_put_hook_object(self):
        """Test _pre_put_hook with non-tensor object."""
        ref = MockStorageVolumeRef()
        buffer = CudaIPCTransportBuffer(ref)

        obj = {"key": "value"}
        request = MockRequest(objects=obj, is_object=True)

        await buffer._pre_put_hook(request)

        assert buffer.is_object
        assert buffer.objects == obj
        assert buffer.ipc_handle is None

    @pytest.mark.asyncio
    async def test_drop(self):
        """Test drop cleans up resources."""
        ref = MockStorageVolumeRef()
        buffer = CudaIPCTransportBuffer(ref)

        # Set some state
        buffer.ipc_handle = "dummy"
        buffer._source_tensor = torch.randn(5, 5)
        buffer.objects = {"key": "value"}

        await buffer.drop()

        assert buffer.ipc_handle is None
        assert buffer._source_tensor is None
        assert buffer.objects is None


class TestCudaIPCErrorHandling:
    """Test error handling scenarios in CUDA IPC operations."""

    @pytest.mark.skipif(not cuda_ipc_available(), reason="CUDA IPC not available")
    def test_reconstruct_tensor_with_invalid_handle(self):
        """Test error handling when IPC reconstruction fails with invalid handle."""
        # Create a valid handle first
        tensor = torch.randn(5, 5, device="cuda:0")
        handle = create_ipc_handle(tensor)

        # Corrupt the handle by modifying the storage handle
        handle.storage_handle = b"invalid_handle_data"

        # Should raise RuntimeError with proper error message
        with pytest.raises(RuntimeError, match="CUDA IPC tensor reconstruction failed"):
            handle.reconstruct_tensor()

    def test_create_ipc_handle_validation_error(self):
        """Test proper validation error for None tensor in _pre_put_hook."""
        ref = MockStorageVolumeRef()
        buffer = CudaIPCTransportBuffer(ref)

        # Create request with None tensor_val
        request = MockRequest(tensor_val=None, is_object=False)

        # Should raise ValueError with proper message
        with pytest.raises(
            ValueError, match="tensor_val must not be None for non-object requests"
        ):
            asyncio.run(buffer._pre_put_hook(request))

    @pytest.mark.skipif(not cuda_ipc_available(), reason="CUDA IPC not available")
    @pytest.mark.asyncio
    async def test_tensor_lifetime_management(self):
        """Test that tensor lifetime management prevents premature deallocation."""
        ref = MockStorageVolumeRef()
        buffer = CudaIPCTransportBuffer(ref)

        # Create a tensor and request
        tensor = torch.randn(10, 10, device="cuda:0")
        original_data = tensor.clone()
        request = MockRequest(tensor_val=tensor)

        # Pre-put hook should register tensor
        await buffer._pre_put_hook(request)

        # Verify tensor is registered
        assert buffer._tensor_id is not None
        assert buffer._tensor_id in CudaIPCTransportBuffer._active_tensors

        # Delete original tensor reference
        del tensor
        gc.collect()

        # Tensor should still be alive in registry
        registered_tensor = CudaIPCTransportBuffer._active_tensors[buffer._tensor_id]
        assert torch.allclose(registered_tensor, original_data)

        # Drop should clean up registration
        await buffer.drop()
        assert buffer._tensor_id not in CudaIPCTransportBuffer._active_tensors

    @pytest.mark.skipif(not cuda_ipc_available(), reason="CUDA IPC not available")
    def test_create_ipc_handle_contiguous_conversion(self):
        """Test that non-contiguous tensors are properly handled."""
        # Create non-contiguous tensor
        tensor = torch.randn(20, 20, device="cuda:0")[::2, ::2]
        assert not tensor.is_contiguous()
        original_data = tensor.clone()

        # Should succeed and preserve data
        handle = create_ipc_handle(tensor)
        reconstructed = handle.reconstruct_tensor()

        assert torch.allclose(reconstructed, original_data)
        assert reconstructed.is_contiguous()


class TestCudaIPCConcurrency:
    """Test concurrent access patterns and thread safety."""

    @pytest.mark.skipif(not cuda_ipc_available(), reason="CUDA IPC not available")
    @pytest.mark.asyncio
    async def test_concurrent_handle_creation(self):
        """Test creating multiple IPC handles concurrently."""
        num_tensors = 5
        tensors = [torch.randn(100, 100, device="cuda:0") for _ in range(num_tensors)]

        async def create_handle_async(tensor):
            return create_ipc_handle(tensor)

        # Create handles concurrently
        tasks = [create_handle_async(tensor) for tensor in tensors]
        handles = await asyncio.gather(*tasks)

        # Verify all handles are valid
        for i, handle in enumerate(handles):
            reconstructed = handle.reconstruct_tensor()
            assert torch.allclose(reconstructed, tensors[i])

    @pytest.mark.skipif(not cuda_ipc_available(), reason="CUDA IPC not available")
    @pytest.mark.asyncio
    async def test_concurrent_transport_operations(self):
        """Test concurrent transport buffer operations."""
        num_buffers = 3
        refs = [MockStorageVolumeRef(f"volume_{i}") for i in range(num_buffers)]
        buffers = [CudaIPCTransportBuffer(ref) for ref in refs]

        async def pre_put_async(buffer, tensor):
            request = MockRequest(tensor_val=tensor)
            await buffer._pre_put_hook(request)
            return buffer

        tensors = [torch.randn(50, 50, device="cuda:0") for _ in range(num_buffers)]

        # Process all buffers concurrently
        tasks = [
            pre_put_async(buffer, tensor) for buffer, tensor in zip(buffers, tensors)
        ]
        completed_buffers = await asyncio.gather(*tasks)

        # Verify all operations completed successfully
        for i, buffer in enumerate(completed_buffers):
            assert buffer.ipc_handle is not None
            assert buffer.shape == tensors[i].shape

        # Clean up
        for buffer in completed_buffers:
            await buffer.drop()

    @pytest.mark.skipif(not cuda_ipc_available(), reason="CUDA IPC not available")
    def test_thread_safety_tensor_registry(self):
        """Test thread safety of the tensor registration system."""
        num_threads = 4
        tensors_per_thread = 5
        results = []

        def register_tensors():
            thread_results = []
            ref = MockStorageVolumeRef()
            buffer = CudaIPCTransportBuffer(ref)

            for i in range(tensors_per_thread):
                tensor = torch.randn(10, 10, device="cuda:0")
                tensor_id = buffer._register_tensor(tensor)
                thread_results.append((tensor_id, tensor))

            results.append(thread_results)

        # Run concurrent registration
        threads = [
            threading.Thread(target=register_tensors) for _ in range(num_threads)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Verify all tensors are registered uniquely
        all_ids = []
        for thread_results in results:
            for tensor_id, tensor in thread_results:
                all_ids.append(tensor_id)
                assert tensor_id in CudaIPCTransportBuffer._active_tensors

        assert len(set(all_ids)) == len(all_ids)  # All IDs should be unique


class TestCudaIPCLargeTensors:
    """Test handling of large tensors and memory pressure scenarios."""

    @pytest.mark.skipif(not cuda_ipc_available(), reason="CUDA IPC not available")
    @pytest.mark.skipif(
        torch.cuda.get_device_properties(0).total_memory < 2**30,
        reason="Requires at least 1GB GPU memory",
    )
    def test_large_tensor_ipc(self):
        """Test IPC with large tensors (100MB+)."""
        # Create ~100MB tensor
        tensor = torch.randn(1024, 1024, 100, device="cuda:0", dtype=torch.float32)
        size_bytes = tensor.numel() * tensor.element_size()
        assert size_bytes >= 100 * 1024 * 1024  # At least 100MB

        # Should handle large tensor successfully
        handle = create_ipc_handle(tensor)
        reconstructed = handle.reconstruct_tensor()

        # Verify data integrity with random sampling (full comparison too expensive)
        sample_indices = torch.randint(0, tensor.numel(), (1000,))
        flat_original = tensor.view(-1)
        flat_reconstructed = reconstructed.view(-1)

        assert torch.allclose(
            flat_original[sample_indices], flat_reconstructed[sample_indices], rtol=1e-5
        )

    @pytest.mark.skipif(not cuda_ipc_available(), reason="CUDA IPC not available")
    @pytest.mark.asyncio
    async def test_multiple_large_tensors_lifecycle(self):
        """Test lifecycle management with multiple large tensors."""
        ref = MockStorageVolumeRef()
        buffers = []

        # Create multiple buffers with large tensors
        for i in range(3):
            buffer = CudaIPCTransportBuffer(ref)
            tensor = torch.randn(512, 512, 50, device="cuda:0")  # ~50MB each
            request = MockRequest(tensor_val=tensor)

            await buffer._pre_put_hook(request)
            buffers.append(buffer)

        # All tensors should be registered
        assert len(CudaIPCTransportBuffer._active_tensors) >= 3

        # Drop all buffers
        for buffer in buffers:
            await buffer.drop()

        # Registry should be cleaned up
        # (Note: Some tensors might still be referenced elsewhere, so we just check
        # that the specific tensor_ids are removed)
        for buffer in buffers:
            assert buffer._tensor_id not in CudaIPCTransportBuffer._active_tensors


class TestCudaIPCIntegration:
    """Integration tests for complete CUDA IPC workflows."""

    @pytest.mark.skipif(not cuda_ipc_available(), reason="CUDA IPC not available")
    @pytest.mark.asyncio
    async def test_full_put_get_workflow(self):
        """Test complete PUT/GET workflow with transport buffers."""
        # Simulate storage volume
        storage_ref = MockStorageVolumeRef("storage")
        client_ref = MockStorageVolumeRef("client")

        storage_buffer = CudaIPCTransportBuffer(storage_ref)
        client_buffer = CudaIPCTransportBuffer(client_ref)

        # CREATE: Client prepares data
        original_tensor = torch.randn(100, 100, device="cuda:0")
        put_request = MockRequest(tensor_val=original_tensor)
        await client_buffer._pre_put_hook(put_request)

        # SEND: Storage receives and processes
        ctx = Mock()
        stored_data = await storage_buffer.handle_put_request(
            ctx, put_request, client_buffer
        )

        # RETRIEVE: Storage prepares for GET
        await storage_buffer.handle_get_request(ctx, stored_data)

        # RECEIVE: Client reconstructs data
        reconstructed_data = await client_buffer._handle_storage_volume_response(
            storage_buffer
        )

        # Verify data integrity
        assert torch.allclose(original_tensor, reconstructed_data)
        assert reconstructed_data.device.type == "cuda"

        # Clean up
        await client_buffer.drop()
        await storage_buffer.drop()

    @pytest.mark.skipif(not cuda_ipc_available(), reason="CUDA IPC not available")
    @pytest.mark.asyncio
    async def test_mixed_object_tensor_handling(self):
        """Test handling both tensor and non-tensor objects."""
        ref = MockStorageVolumeRef()

        # Test tensor
        tensor_buffer = CudaIPCTransportBuffer(ref)
        tensor = torch.randn(10, 10, device="cuda:0")
        tensor_request = MockRequest(tensor_val=tensor)
        await tensor_buffer._pre_put_hook(tensor_request)
        assert not tensor_buffer.is_object
        assert tensor_buffer.ipc_handle is not None

        # Test object
        object_buffer = CudaIPCTransportBuffer(ref)
        obj = {"key": "value", "data": [1, 2, 3]}
        object_request = MockRequest(objects=obj, is_object=True)
        await object_buffer._pre_put_hook(object_request)
        assert object_buffer.is_object
        assert object_buffer.ipc_handle is None
        assert object_buffer.objects == obj

        # Clean up
        await tensor_buffer.drop()
        await object_buffer.drop()
