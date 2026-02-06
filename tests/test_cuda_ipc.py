# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for CUDA IPC transport."""

import pytest
import torch

from torchstore.transport.cuda_ipc import (
    cuda_ipc_available,
    create_ipc_handle,
    CudaIPCHandle,
    CudaIPCTransportBuffer,
)


class MockStorageVolumeRef:
    """Mock StorageVolumeRef for testing."""

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
