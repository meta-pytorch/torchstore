# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchstore.constants import MONARCH_HOSTMESH_V1

if MONARCH_HOSTMESH_V1:
    from monarch._rust_bindings.monarch_hyperactor.channel import ChannelTransport
    from monarch._rust_bindings.monarch_hyperactor.config import configure

    configure(
        default_transport=ChannelTransport.MetaTlsWithHostname,
    )

import os
from logging import getLogger

import pytest
import torch

import torchstore as ts
from monarch.actor import Actor, current_rank, endpoint
from torchstore.utils import spawn_actors

from .utils import main

logger = getLogger(__name__)


class BufferCacheTest(Actor):
    def __init__(self):
        ts.init_logging()
        self.rank = current_rank().rank
        self.cache = ts.TransportBufferCache()

    def rlog(self, msg):
        logger.info(f"rank: {self.rank} {msg}")

    @endpoint
    async def test_basic_cache(self):
        """Test basic buffer caching functionality for both put and get."""
        self.rlog("Testing basic buffer cache")
        
        # Create a simple tensor
        tensor = torch.randn(100, 100)
        
        # First put - should allocate buffer and cache it
        await ts.put("test_tensor", tensor, cache=self.cache)
        
        # Verify cache has the buffer
        assert self.cache.get("test_tensor") is not None, "Cache should have buffer after first put"
        initial_cache_size = len(self.cache._buffers)
        
        # Second put - should reuse cached buffer
        tensor2 = torch.randn(100, 100)
        await ts.put("test_tensor", tensor2, cache=self.cache)
        
        # Verify cache size hasn't changed (buffer reused)
        assert len(self.cache._buffers) == initial_cache_size, "Cache size should not change on second put"
        
        # Clear cache to test get caching
        self.cache.clear()
        
        # First get - should allocate buffer and cache it
        retrieved = await ts.get("test_tensor", cache=self.cache)
        assert torch.allclose(retrieved, tensor2), "Retrieved tensor should match tensor2"
        assert self.cache.get("test_tensor") is not None, "Cache should have buffer after first get"
        
        # Second get - should reuse cached buffer
        retrieved2 = await ts.get("test_tensor", cache=self.cache)
        assert torch.allclose(retrieved2, tensor2), "Retrieved tensor should match tensor2"
        
        self.rlog("Basic cache test passed")
        return True

    @endpoint
    async def test_state_dict_cache(self):
        """Test state_dict caching functionality."""
        self.rlog("Testing state_dict cache")

        # Create a simple model state dict
        state_dict = {
            "layer1.weight": torch.randn(10, 10),
            "layer1.bias": torch.randn(10),
            "layer2.weight": torch.randn(5, 10),
            "layer2.bias": torch.randn(5),
        }

        # First save - should allocate buffers and cache them
        await ts.put_state_dict(state_dict, "checkpoint", cache=self.cache)

        # Verify cache has buffers for all keys
        assert (
            len(self.cache._buffers) > 0
        ), "Cache should have buffers after put_state_dict"
        initial_cache_size = len(self.cache._buffers)
        self.rlog(f"Cache has {initial_cache_size} buffers")

        # Second save with updated state dict - should reuse buffers
        state_dict2 = {
            "layer1.weight": torch.randn(10, 10),
            "layer1.bias": torch.randn(10),
            "layer2.weight": torch.randn(5, 10),
            "layer2.bias": torch.randn(5),
        }
        await ts.put_state_dict(state_dict2, "checkpoint", cache=self.cache)

        # Cache size should remain the same (buffers reused)
        assert (
            len(self.cache._buffers) == initial_cache_size
        ), "Cache size should not change on second put"

        # Retrieve and verify
        retrieved = await ts.get_state_dict("checkpoint")
        assert set(retrieved.keys()) == set(
            state_dict2.keys()
        ), "Retrieved keys should match"
        for key in state_dict2.keys():
            assert torch.allclose(
                retrieved[key], state_dict2[key]
            ), f"Retrieved {key} should match"

        self.rlog("State dict cache test passed")
        return True

    @endpoint
    async def test_cache_clear(self):
        """Test cache clearing functionality."""
        self.rlog("Testing cache clear")

        tensor = torch.randn(50, 50)
        await ts.put("test_tensor", tensor, cache=self.cache)

        assert len(self.cache._buffers) > 0, "Cache should have buffers before clear"

        # Clear cache
        self.cache.clear()
        assert len(self.cache._buffers) == 0, "Cache should be empty after clear"

        self.rlog("Cache clear test passed")
        return True

    @endpoint
    async def test_cache_remove(self):
        """Test removing specific cached buffers."""
        self.rlog("Testing cache remove")

        tensor1 = torch.randn(50, 50)
        tensor2 = torch.randn(50, 50)

        await ts.put("tensor1", tensor1, cache=self.cache)
        await ts.put("tensor2", tensor2, cache=self.cache)

        assert self.cache.get("tensor1") is not None, "tensor1 should be cached"
        assert self.cache.get("tensor2") is not None, "tensor2 should be cached"

        # Remove one
        self.cache.remove("tensor1")
        assert self.cache.get("tensor1") is None, "tensor1 should be removed"
        assert self.cache.get("tensor2") is not None, "tensor2 should still be cached"

        self.rlog("Cache remove test passed")
        return True

    @endpoint
    async def test_without_cache(self):
        """Test that existing functionality works without cache parameter."""
        self.rlog("Testing without cache")

        tensor = torch.randn(100, 100)

        # Should work without cache
        await ts.put("test_tensor_no_cache", tensor)
        retrieved = await ts.get("test_tensor_no_cache")
        assert torch.allclose(
            retrieved, tensor
        ), "Retrieved tensor should match without cache"

        # State dict should also work without cache
        state_dict = {"weight": torch.randn(10, 10)}
        await ts.put_state_dict(state_dict, "checkpoint_no_cache")
        retrieved_sd = await ts.get_state_dict("checkpoint_no_cache")
        assert torch.allclose(
            retrieved_sd["weight"], state_dict["weight"]
        ), "State dict should work without cache"

        self.rlog("Without cache test passed")
        return True


@pytest.mark.asyncio
async def test_buffer_cache_basic():
    """Test basic buffer caching functionality."""
    await _run_cache_test("test_basic_cache")


@pytest.mark.asyncio
async def test_state_dict_cache():
    """Test state_dict caching functionality."""
    await _run_cache_test("test_state_dict_cache")


@pytest.mark.asyncio
async def test_cache_clear():
    """Test cache clearing functionality."""
    await _run_cache_test("test_cache_clear")


@pytest.mark.asyncio
async def test_cache_remove():
    """Test removing specific cached buffers."""
    await _run_cache_test("test_cache_remove")


@pytest.mark.asyncio
async def test_without_cache():
    """Test that existing functionality works without cache parameter."""
    await _run_cache_test("test_without_cache")


async def _run_cache_test(test_method_name):
    """Helper to run a cache test with proper setup and teardown."""
    ts.init_logging()
    logger.info(f"Testing {test_method_name}")

    await ts.initialize()
    try:
        # Spawn actor
        actors = await spawn_actors(
            1,
            BufferCacheTest,
            "cache_test_world",
        )

        # Call the test method
        test_method = getattr(actors, test_method_name)
        result = await test_method.call_one()
        assert result, f"Test {test_method_name} failed"

        logger.info(f"Test {test_method_name} passed")
    finally:
        await ts.shutdown()


if __name__ == "__main__":
    main([__file__])
