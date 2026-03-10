#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
Test the new put_slice() and get_slice() APIs for GPU-direct weight sync.

Run with:
    cd ./torchstore
    TORCHSTORE_LOG_LEVEL=DEBUG pytest -vs tests/test_slice_api.py
"""

import pytest
import torch

from torchstore.transport.types import TensorSlice


class TestSliceAPI:
    """Tests for put_slice and get_slice APIs."""

    @pytest.mark.asyncio
    async def test_put_slice_basic(self):
        """Test storing a tensor slice with metadata."""
        import torchstore.api as ts

        await ts.initialize()

        try:
            # For DTensor storage, ALL shards must be stored before it's "complete"
            # Simulate both FSDP ranks storing their shards
            global_shape = (512, 512)

            # Shard 0: rows 0-255
            tensor_0 = torch.randn(256, 512)
            slice_0 = TensorSlice(
                offsets=(0, 0),
                coordinates=(0,),
                global_shape=global_shape,
                local_shape=(256, 512),
                mesh_shape=(2,),
            )

            # Shard 1: rows 256-511
            tensor_1 = torch.randn(256, 512)
            slice_1 = TensorSlice(
                offsets=(256, 0),
                coordinates=(1,),
                global_shape=global_shape,
                local_shape=(256, 512),
                mesh_shape=(2,),
            )

            await ts.put_slice("test_put_slice_basic", tensor_0, slice_0)
            await ts.put_slice("test_put_slice_basic", tensor_1, slice_1)

            # Verify it was stored (only complete after both shards are in)
            assert await ts.exists("test_put_slice_basic")

            # Clean up
            await ts.delete("test_put_slice_basic")
        finally:
            await ts.shutdown()

    @pytest.mark.asyncio
    async def test_put_and_get_slice(self):
        """Test storing multiple slices and fetching them."""
        import torchstore.api as ts

        await ts.initialize()

        try:
            # Store two FSDP shards (row-wise sharding)
            global_shape = (512, 512)

            # Shard 0: rows 0-255
            shard_0 = torch.randn(256, 512)
            slice_0 = TensorSlice(
                offsets=(0, 0),
                coordinates=(0,),
                global_shape=global_shape,
                local_shape=(256, 512),
                mesh_shape=(2,),
            )

            # Shard 1: rows 256-511
            shard_1 = torch.randn(256, 512)
            slice_1 = TensorSlice(
                offsets=(256, 0),
                coordinates=(1,),
                global_shape=global_shape,
                local_shape=(256, 512),
                mesh_shape=(2,),
            )

            await ts.put_slice("test_fsdp_shards", shard_0, slice_0)
            await ts.put_slice("test_fsdp_shards", shard_1, slice_1)

            # Now fetch a TP slice (column-wise, first half of columns)
            tp_slice = TensorSlice(
                offsets=(0, 0),
                coordinates=(0,),
                global_shape=global_shape,
                local_shape=(512, 256),
                mesh_shape=(2,),
            )

            result = await ts.get_slice("test_fsdp_shards", tp_slice)

            # Verify shape
            assert result.shape == (
                512,
                256,
            ), f"Expected (512, 256), got {result.shape}"

            # Verify data integrity
            # Top half should come from shard_0[:, :256]
            expected_top = shard_0[:, :256]
            assert torch.allclose(
                result[:256], expected_top, atol=1e-5
            ), "Top half data mismatch"

            # Bottom half should come from shard_1[:, :256]
            expected_bottom = shard_1[:, :256]
            assert torch.allclose(
                result[256:], expected_bottom, atol=1e-5
            ), "Bottom half data mismatch"

            # Clean up
            await ts.delete("test_fsdp_shards")

        finally:
            await ts.shutdown()

    @pytest.mark.asyncio
    async def test_get_slice_full_tensor(self):
        """Test fetching full tensor when stored as single shard."""
        import torchstore.api as ts

        await ts.initialize()

        try:
            # Store single shard that covers full tensor (mesh_shape=(1,) means single shard)
            tensor = torch.randn(256, 256)
            slice_spec = TensorSlice(
                offsets=(0, 0),
                coordinates=(0,),
                global_shape=(256, 256),
                local_shape=(256, 256),
                mesh_shape=(1,),  # Single shard - no waiting for other shards
            )

            await ts.put_slice("test_full", tensor, slice_spec)

            # Fetch full tensor (same slice spec)
            result = await ts.get_slice("test_full", slice_spec)

            assert result.shape == tensor.shape
            assert torch.allclose(result, tensor, atol=1e-5)

            await ts.delete("test_full")

        finally:
            await ts.shutdown()

    @pytest.mark.asyncio
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    async def test_get_slice_to_gpu(self):
        """Test fetching slice directly to GPU."""
        import torchstore.api as ts

        await ts.initialize()

        try:
            # Store CPU tensor
            tensor = torch.randn(128, 256)
            slice_spec = TensorSlice(
                offsets=(0, 0),
                coordinates=(0,),
                global_shape=(128, 256),
                local_shape=(128, 256),
                mesh_shape=(1,),
            )

            await ts.put_slice("test_gpu", tensor, slice_spec)

            # Fetch to GPU
            result = await ts.get_slice("test_gpu", slice_spec, target_device="cuda:0")

            assert result.device == torch.device(
                "cuda:0"
            ), f"Expected cuda:0, got {result.device}"
            assert result.shape == (128, 256)

            # Verify data (compare on CPU)
            assert torch.allclose(result.cpu(), tensor, atol=1e-5)

            await ts.delete("test_gpu")

        finally:
            await ts.shutdown()


class TestTPSliceComputation:
    """Tests for TP slice computation logic."""

    def test_column_parallel_slice(self):
        """Test column-parallel (QKV) slice computation."""
        from torchstore.transport.types import TensorSlice

        global_shape = (4096, 4096)
        tp_size = 2

        # TP rank 0 should get first half of columns
        slice_0 = TensorSlice(
            offsets=(0, 0),
            coordinates=(0,),
            global_shape=global_shape,
            local_shape=(4096, 2048),
            mesh_shape=(tp_size,),
        )

        assert slice_0.local_shape == (4096, 2048)
        assert slice_0.offsets == (0, 0)

        # TP rank 1 should get second half of columns
        slice_1 = TensorSlice(
            offsets=(0, 2048),
            coordinates=(1,),
            global_shape=global_shape,
            local_shape=(4096, 2048),
            mesh_shape=(tp_size,),
        )

        assert slice_1.local_shape == (4096, 2048)
        assert slice_1.offsets == (0, 2048)

    def test_row_parallel_slice(self):
        """Test row-parallel (O proj, down proj) slice computation."""
        from torchstore.transport.types import TensorSlice

        global_shape = (4096, 4096)
        tp_size = 2

        # TP rank 0 should get first half of rows
        slice_0 = TensorSlice(
            offsets=(0, 0),
            coordinates=(0,),
            global_shape=global_shape,
            local_shape=(2048, 4096),
            mesh_shape=(tp_size,),
        )

        assert slice_0.local_shape == (2048, 4096)
        assert slice_0.offsets == (0, 0)

    def test_fsdp_to_tp_intersection(self):
        """Test intersection of FSDP (row) shard with TP (column) slice."""
        from torchstore.transport.types import TensorSlice
        from torchstore.utils import get_slice_intersection

        global_shape = (1000, 1000)

        # FSDP shard 0: rows 0-499 (all columns)
        fsdp_shard = TensorSlice(
            offsets=(0, 0),
            coordinates=(0,),
            global_shape=global_shape,
            local_shape=(500, 1000),
            mesh_shape=(2,),
        )

        # TP slice 0: all rows, columns 0-499
        tp_slice = TensorSlice(
            offsets=(0, 0),
            coordinates=(0,),
            global_shape=global_shape,
            local_shape=(1000, 500),
            mesh_shape=(2,),
        )

        # Intersection should be rows 0-499, columns 0-499
        intersection = get_slice_intersection(fsdp_shard, tp_slice)

        assert intersection is not None
        assert intersection.local_shape == (
            500,
            500,
        ), f"Expected (500, 500), got {intersection.local_shape}"
        assert intersection.offsets == (0, 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
