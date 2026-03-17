# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import pytest
import torch


class MockRdmaMemory:
    """Lightweight mock that records the tensor it was created with."""

    def __init__(self, tensor: torch.Tensor) -> None:
        self.data_ptr = tensor.data_ptr()
        self.nbytes = tensor.nbytes


@pytest.fixture(autouse=True)
def _patch_rdma_memory(monkeypatch):
    import torchstore.transport.torchcomms.cache as cache_mod

    monkeypatch.setattr(cache_mod, "RdmaMemory", MockRdmaMemory)


from torchstore.transport.torchcomms.cache import RdmaMemoryCache  # noqa: E402


class TestRdmaMemoryCacheHitMiss:
    def test_cache_hit_same_tensor(self):
        cache = RdmaMemoryCache()
        tensor = torch.randn(4, 4)

        mem1 = cache.get_or_register(tensor)
        mem2 = cache.get_or_register(tensor)

        assert mem1 is mem2

    def test_cache_miss_different_tensors(self):
        cache = RdmaMemoryCache()
        t1 = torch.randn(4, 4)
        t2 = torch.randn(4, 4)

        mem1 = cache.get_or_register(t1)
        mem2 = cache.get_or_register(t2)

        assert mem1 is not mem2
        assert len(cache._cache) == 2


class TestRdmaMemoryCacheClear:
    def test_clear(self):
        cache = RdmaMemoryCache()
        t1 = torch.randn(2, 2)
        t2 = torch.randn(3, 3)
        cache.get_or_register(t1)
        cache.get_or_register(t2)

        assert len(cache._cache) == 2
        cache.clear()
        assert len(cache._cache) == 0
        assert len(cache._storage_refs) == 0


class TestRdmaMemoryCacheWeakrefEviction:
    def test_evicts_when_tensor_is_deleted(self):
        cache = RdmaMemoryCache()
        t = torch.randn(4, 4)
        key = (t.data_ptr(), t.nbytes)

        cache.get_or_register(t)
        assert key in cache._cache

        del t
        # refcount GC fires the weakref callback synchronously
        assert key not in cache._cache
        assert key not in cache._storage_refs

    def test_survives_when_view_still_alive(self):
        cache = RdmaMemoryCache()
        t = torch.randn(4, 4)
        view = t[0:2]  # shares the same storage
        key = (t.data_ptr(), t.nbytes)

        cache.get_or_register(t)
        del t
        # storage is still alive via 'view', so entry should persist
        assert key in cache._cache

        del view
        # storage is now freed
        assert key not in cache._cache

    def test_new_tensor_after_eviction(self):
        cache = RdmaMemoryCache()
        t1 = torch.randn(4, 4)
        mem1 = cache.get_or_register(t1)

        del t1  # evicts

        t2 = torch.randn(4, 4)
        mem2 = cache.get_or_register(t2)

        # should be a fresh registration, not the old one
        assert mem2 is not mem1
        assert len(cache._cache) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
