# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Real CPU SHM and storage with a serialized, in-process RPC boundary."""

import asyncio
import pickle
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torchstore.storage_volume import InMemoryStore
from torchstore.transport.buffers import TransportContext
from torchstore.transport.shared_memory import (
    SharedMemoryCache,
    SharedMemoryDescriptor,
    SharedMemoryTransportBuffer,
    unpin_memory,
)
from torchstore.transport.types import Request, TensorSlice


class _Endpoint:
    def __init__(self, handler):
        self.handler = handler

    async def call(self, buffer, requests):
        remote_buffer = pickle.loads(pickle.dumps(buffer))
        result = await self.handler(remote_buffer, requests)
        return pickle.loads(pickle.dumps(result))

    call_one = call


@pytest.fixture
def store_refs():
    store = InMemoryStore()
    volume = SimpleNamespace(
        handshake=_Endpoint(store.handshake),
        put=_Endpoint(store.put),
        get=_Endpoint(store.get),
    )
    refs = [
        SimpleNamespace(volume=volume, transport_context=TransportContext())
        for _ in range(2)
    ]
    try:
        yield store, *refs
    finally:
        for ref in refs:
            ref.transport_context.clear()
        store.reset()


async def _put(ref, tensor, key="logits/0"):
    await SharedMemoryTransportBuffer(ref).put_to_storage_volume(
        [Request.from_tensor(key, tensor)]
    )


async def _get(ref, key="logits/0"):
    return (
        await SharedMemoryTransportBuffer(ref).get_from_storage_volume(
            [Request(key=key)]
        )
    )[0]


def _handle(store, key="logits/0"):
    descriptor = SharedMemoryDescriptor.from_tensor(store.kv[key])
    assert descriptor is not None
    return descriptor.storage_handle


def test_warm_attach_does_not_scan_or_unpin_other_allocations():
    class NoIterationDict(dict):
        def __iter__(self):
            raise AssertionError("warm attachment traversed the cache")

    cache = SharedMemoryCache()
    try:
        entry, descriptor = cache.allocate("weights", torch.Size([4]), torch.float32)
        cache.allocate("other", torch.Size([8]), torch.float32)
        with (
            patch.object(cache, "_storages", NoIterationDict(cache._storages)),
            patch("torchstore.transport.shared_memory.unpin_memory") as unpin,
        ):
            assert cache.attach("weights", descriptor).storage is entry.storage
            unpin.assert_not_called()
    finally:
        cache.clear()


def test_replacement_unpins_only_obsolete_mapping():
    writer = SharedMemoryCache()
    reader = SharedMemoryCache()
    try:
        _, old = writer.allocate("logits/0", torch.Size([4]), torch.float32)
        reader.attach("logits/0", old)
        _, other = writer.allocate("other", torch.Size([2]), torch.float32)
        reader.attach("other", other)
        old_mapping = reader._storages[("logits/0", old.storage_handle)]
        _, new = writer.allocate("logits/0", torch.Size([8]), torch.float32)
        with patch(
            "torchstore.transport.shared_memory.unpin_memory", wraps=unpin_memory
        ) as unpin:
            reader.attach("logits/0", new)
            unpin.assert_called_once_with(old_mapping)
        assert set(reader._storages) == {
            ("logits/0", new.storage_handle),
            ("other", other.storage_handle),
        }
    finally:
        reader.clear()
        writer.clear()


@pytest.mark.asyncio
async def test_resize_and_compatible_overwrite(store_refs):
    store, writer, reader = store_refs
    previous = None
    for tensor in [
        torch.arange(4),
        torch.arange(8),
        torch.arange(2),
        torch.arange(2).float(),
        torch.arange(2).float().reshape(1, 2),
        torch.arange(2).float().reshape(1, 2) + 10,
        torch.empty(0),
        torch.tensor(3.0),
        torch.arange(20).reshape(4, 5)[:, :3],
    ]:
        await _put(writer, tensor)
        result = await _get(reader)
        torch.testing.assert_close(result, tensor)
        handle = _handle(store)
        if previous is not None:
            old_shape, old_dtype, old_handle = previous
            assert (handle == old_handle) == (
                tensor.shape == old_shape and tensor.dtype == old_dtype
            )
        previous = tensor.shape, tensor.dtype, handle
        for ref in (writer, reader):
            cache = ref.transport_context.get(SharedMemoryCache)
            assert len(cache._storages) == 1
            assert not cache._key_locks


@pytest.mark.asyncio
async def test_delete_and_reput_bounds_mappings_and_reclaims_shm(store_refs):
    store, writer, reader = store_refs
    retired_handles = []
    for version in range(96):
        key = f"logits/{version % 3}"
        if key in store.kv:
            retired_handles.append(_handle(store, key))
            # Deletion by another client leaves writer/reader caches untouched.
            await store.delete(key)
        tensor = torch.full((1, 1 + version % 7, 8), float(version))
        await _put(writer, tensor, key)
        torch.testing.assert_close(await _get(reader, key), tensor)
        for ref in (writer, reader):
            cache = ref.transport_context.get(SharedMemoryCache)
            assert len(cache._storages) <= 3
            assert sum(s.nbytes() for s in cache._storages.values()) <= 3 * 7 * 8 * 4
    assert all(
        not (Path("/dev/shm") / handle.decode().lstrip("/")).exists()
        for handle in retired_handles
    )


@pytest.mark.asyncio
async def test_failed_put_allows_same_key_retry(store_refs):
    store, writer, reader = store_refs
    await _put(writer, torch.arange(4))
    old_handle = _handle(store)
    with patch.object(writer.volume.put, "handler", side_effect=RuntimeError):
        with pytest.raises(RuntimeError):
            await _put(writer, torch.arange(8))
    assert _handle(store) == old_handle
    await _put(writer, torch.arange(12))
    torch.testing.assert_close(await _get(reader), torch.arange(12))
    cache = writer.transport_context.get(SharedMemoryCache)
    assert len(cache._storages) == 1
    assert not cache._key_locks


@pytest.mark.asyncio
async def test_cancelled_dispatched_put_does_not_damage_retry(store_refs):
    store, writer, reader = store_refs
    dispatched = asyncio.Event()
    deliver = asyncio.Event()
    remote = None

    async def delayed_put(buffer, requests):
        nonlocal remote
        remote_buffer = pickle.loads(pickle.dumps(buffer))

        async def delivery():
            await deliver.wait()
            await store.put(remote_buffer, requests)

        remote = asyncio.create_task(delivery())
        dispatched.set()
        await asyncio.shield(remote)

    task = None
    try:
        with patch.object(writer.volume.put, "call", delayed_put):
            task = asyncio.create_task(_put(writer, torch.arange(8)))
            await asyncio.wait_for(dispatched.wait(), 5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        await _put(writer, torch.arange(12))
        deliver.set()
        assert remote is not None
        # The cancelled operation may fail to attach its retired allocation.
        try:
            await asyncio.wait_for(remote, 5)
        except RuntimeError as error:
            assert "No such file" in str(error)
        torch.testing.assert_close(await _get(reader), torch.arange(12))
        cache = writer.transport_context.get(SharedMemoryCache)
        assert len(cache._storages) == 1
        assert not cache._key_locks
    finally:
        deliver.set()
        for pending in (task, remote):
            if pending is not None:
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)


def test_dtensor_shards_keep_existing_allocations():
    store = InMemoryStore()
    tensor = torch.arange(4)
    tensor_slice = TensorSlice((0,), (0,), (8,), (4,), (2,))
    request = Request.from_any("weight", tensor, tensor_slice=tensor_slice)
    store._store(request, tensor)
    try:
        assert store._extract_existing(request.meta_only()) is tensor
        torch.testing.assert_close(store._get_data(request), tensor)
        # This feature must not opt sharded tensors into allocation replacement.
        resized = Request.from_tensor("weight", torch.arange(2)).meta_only()
        resized.tensor_slice = TensorSlice((0,), (0,), (4,), (2,), (2,))
        assert store._extract_existing(resized) is tensor
    finally:
        store.reset()
