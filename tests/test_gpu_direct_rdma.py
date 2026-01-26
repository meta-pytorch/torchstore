# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for GPU Direct RDMA functionality.

These tests verify that tensors can be transferred directly between GPUs
using RDMA without going through CPU memory.
"""

import os
import pytest
import torch

# Set GPU Direct RDMA environment variable before imports
os.environ["TORCHSTORE_GPU_DIRECT_RDMA"] = "1"

import torchstore as ts
from tests.utils import run_with_store


needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)

needs_multi_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Need at least 2 GPUs for multi-GPU tests",
)


@needs_cuda
@pytest.mark.asyncio
async def test_gpu_direct_rdma_allocation():
    """Test that tensors are allocated on GPU when GPU Direct RDMA is enabled."""
    from torchstore.transport.buffers import GPU_DIRECT_RDMA_ENABLED, RDMATransportBuffer
    from torchstore.transport.torchcomms.buffer import (
        GPU_DIRECT_RDMA_ENABLED as TC_GPU_DIRECT_ENABLED,
        TorchCommsRdmaTransportBuffer,
    )

    # Verify environment variable is set
    assert GPU_DIRECT_RDMA_ENABLED, "GPU_DIRECT_RDMA should be enabled"
    assert TC_GPU_DIRECT_ENABLED, "TorchComms GPU_DIRECT_RDMA should be enabled"

    # Test RDMATransportBuffer device allocation
    rdma_buffer = RDMATransportBuffer()
    device = rdma_buffer._get_allocation_device()
    assert device.type == "cuda", f"Expected CUDA device, got {device}"

    # Test TorchCommsRdmaTransportBuffer device allocation
    tc_buffer = TorchCommsRdmaTransportBuffer()
    device = tc_buffer._get_allocation_device()
    assert device.type == "cuda", f"Expected CUDA device, got {device}"

    print("✓ GPU Direct RDMA allocation test passed")


@needs_cuda
def test_gpu_tensor_put_get():
    """Test putting and getting GPU tensors through torchstore."""

    async def _test():
        # Create a GPU tensor
        original_tensor = torch.randn(100, 100, device="cuda")
        original_data = original_tensor.clone()

        # Put the tensor
        await ts.put("gpu_test_tensor", original_tensor)

        # Get the tensor back
        retrieved_tensor = await ts.get("gpu_test_tensor")

        # Verify the data matches
        assert torch.allclose(
            original_data.cpu(), retrieved_tensor.cpu()
        ), "Retrieved tensor doesn't match original"

        # Check if GPU Direct RDMA was used (tensor should be on GPU)
        if os.environ.get("TORCHSTORE_GPU_DIRECT_RDMA", "1") == "1":
            # With GPU Direct, the retrieved tensor should be on GPU
            print(f"Retrieved tensor device: {retrieved_tensor.device}")

        print("✓ GPU tensor put/get test passed")

    run_with_store(_test)


@needs_cuda
def test_gpu_direct_state_dict():
    """Test putting and getting a state dict with GPU tensors."""

    async def _test():
        # Create a simple model state dict on GPU
        state_dict = {
            "layer1.weight": torch.randn(256, 128, device="cuda"),
            "layer1.bias": torch.randn(256, device="cuda"),
            "layer2.weight": torch.randn(64, 256, device="cuda"),
            "layer2.bias": torch.randn(64, device="cuda"),
        }

        # Clone original data
        original_data = {k: v.clone().cpu() for k, v in state_dict.items()}

        # Put the state dict
        await ts.put_state_dict(state_dict, "gpu_model_checkpoint")

        # Get the state dict back
        retrieved_state_dict = await ts.get_state_dict("gpu_model_checkpoint")

        # Verify the data matches
        for key in original_data:
            assert key in retrieved_state_dict, f"Missing key: {key}"
            assert torch.allclose(
                original_data[key], retrieved_state_dict[key].cpu()
            ), f"Mismatch for key: {key}"

        print("✓ GPU state dict test passed")

    run_with_store(_test)


@needs_cuda
def test_gpu_direct_large_tensor():
    """Test GPU Direct RDMA with a large tensor (simulating model weights)."""

    async def _test():
        # Create a large tensor (64MB, typical for a transformer layer)
        # Shape: (4096, 4096) with float32 = 64MB
        large_tensor = torch.randn(4096, 4096, device="cuda", dtype=torch.float32)
        original_data = large_tensor.clone()

        print(
            f"Testing large tensor: {large_tensor.shape}, "
            f"{large_tensor.numel() * 4 / 1024 / 1024:.1f} MB"
        )

        # Put the tensor
        await ts.put("large_gpu_tensor", large_tensor)

        # Get the tensor back
        import time

        start = time.perf_counter()
        retrieved_tensor = await ts.get("large_gpu_tensor")
        elapsed = time.perf_counter() - start

        print(f"Retrieved in {elapsed*1000:.1f}ms")
        print(f"Retrieved tensor device: {retrieved_tensor.device}")

        # Verify the data matches
        assert torch.allclose(
            original_data.cpu(), retrieved_tensor.cpu()
        ), "Large tensor data mismatch"

        print("✓ Large GPU tensor test passed")

    run_with_store(_test)


@needs_cuda
def test_gpu_direct_disabled():
    """Test that CPU fallback works when GPU Direct is disabled."""

    async def _test():
        # Temporarily disable GPU Direct
        original_setting = os.environ.get("TORCHSTORE_GPU_DIRECT_RDMA", "1")
        os.environ["TORCHSTORE_GPU_DIRECT_RDMA"] = "0"

        try:
            # Reimport to pick up new setting
            import importlib
            from torchstore.transport import buffers

            importlib.reload(buffers)

            # Create and store a tensor
            tensor = torch.randn(100, 100, device="cuda")
            await ts.put("cpu_fallback_test", tensor)
            retrieved = await ts.get("cpu_fallback_test")

            # Data should still match
            assert torch.allclose(
                tensor.cpu(), retrieved.cpu()
            ), "CPU fallback data mismatch"

            print("✓ CPU fallback test passed")

        finally:
            # Restore original setting
            os.environ["TORCHSTORE_GPU_DIRECT_RDMA"] = original_setting

    run_with_store(_test)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
