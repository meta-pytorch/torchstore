# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the storage volume's request handling as a plain class.

A stand-in volume pickles every argument and successful reply, as calls to the Monarch actor
do, while a core's error reaches the caller as is, where Monarch wraps it in ``ActorError``.
"""

import ast
import contextlib
import inspect
import pickle
import uuid
from typing import Any

import pytest
import torch
import torchstore.storage
import torchstore.transport.gloo as gloo
from torchstore.storage import InMemoryStore, StorageImpl, StorageVolumeCore
from torchstore.strategy import StorageVolumeRef
from torchstore.transport import create_transport_buffer, TransportType
from torchstore.transport.buffers import (
    TransportBuffer,
    TransportCache,
    TransportContext,
)
from torchstore.transport.types import Request, TensorSlice
from torchstore.utils import get_local_hostname

pytestmark = pytest.mark.timeout(120)

# The volume calls that one put and one get make on each transport.
_TRANSPORT_CALLS = {
    TransportType.SharedMemory: ["handshake", "put", "get"],
    TransportType.Gloo: ["handshake", "put", "get_meta", "get"],
    TransportType.MonarchRPC: ["put", "get"],
}


def _wire(value: Any) -> Any:
    return pickle.loads(pickle.dumps(value))


class _Endpoint:
    def __init__(self, calls: list[str], name: str, method: Any) -> None:
        self._calls = calls
        self._name = name
        self._method = method

    async def call_one(self, *args: Any) -> Any:
        self._calls.append(self._name)
        return _wire(await self._method(*_wire(args)))

    call = call_one


class _StandInVolume:
    """Serves a core with every argument and successful reply pickled, as the actor does.

    Unlike the actor, its ``call`` answers once rather than per volume, and calls neither queue nor finish once cancelled.
    Its refs share one client transport context.
    """

    def __init__(self, core: StorageVolumeCore) -> None:
        self.core = core
        self.calls: list[str] = []
        self.client_context = TransportContext()
        self.get_id = _Endpoint(self.calls, "get_id", core.get_id)
        self.handshake = _Endpoint(self.calls, "handshake", core.handshake)
        self.put = _Endpoint(self.calls, "put", core.put)
        self.get = _Endpoint(self.calls, "get", core.get)
        self.get_meta = _Endpoint(self.calls, "get_meta", core.get_meta)
        self.delete = _Endpoint(self.calls, "delete", core.delete)
        self.delete_batch = _Endpoint(self.calls, "delete_batch", core.delete_batch)
        self.reset = _Endpoint(self.calls, "reset", core.reset)

    def __reduce__(self) -> Any:
        raise TypeError("a request must not carry its volume handle")


class _KeyedCache(TransportCache):
    """A transport cache with entries per key, like the shared-memory cache."""

    def __init__(self) -> None:
        self.keys: set[str] = set()

    def delete(self, keys: set[str]) -> None:
        self.keys -= keys

    def clear(self) -> None:
        self.keys.clear()


class _HandshakeProbe(TransportBuffer):
    """A transport whose handshake reports which requests the store would update in place."""

    async def recv_handshake(
        self, ctx: TransportContext, entries: list[tuple[Request, Any]]
    ) -> list[bool]:
        return [existing is not None for _, existing in entries]


def _ref(volume: Any, transport: TransportType) -> StorageVolumeRef:
    return StorageVolumeRef(
        volume,
        volume.core.volume_id,
        volume.client_context,
        transport,
        get_local_hostname(),
    )


async def _put(ref: StorageVolumeRef, key: str, value: torch.Tensor) -> None:
    buffer = create_transport_buffer(ref)
    await buffer.put_to_storage_volume([Request.from_tensor(key, value)])


async def _get(ref: StorageVolumeRef, key: str) -> torch.Tensor:
    buffer = create_transport_buffer(ref)
    [value] = await buffer.get_from_storage_volume([Request(key=key)])
    return value


@pytest.fixture
def volume():
    """A stand-in volume whose store, client context and Gloo address go when the test ends."""
    core = StorageVolumeCore(f"volume-{uuid.uuid4()}")
    stand_in = _StandInVolume(core)
    with contextlib.ExitStack() as cleanup:
        cleanup.callback(core.store.reset)
        cleanup.callback(lambda: gloo._store_addrs.pop(core.volume_id, None))
        cleanup.callback(stand_in.client_context.clear)
        yield stand_in


@pytest.mark.parametrize("transport", list(_TRANSPORT_CALLS))
@pytest.mark.asyncio
async def test_transport_round_trip_makes_only_its_calls(volume, transport) -> None:
    """Put and get a tensor through each transport, which calls only what it needs."""
    ref = _ref(volume, transport)
    source = torch.arange(12, dtype=torch.float32).view(3, 4)
    await _put(ref, "weights", source)
    result = await _get(ref, "weights")
    assert torch.equal(result, source)
    assert volume.calls == _TRANSPORT_CALLS[transport]
    # MonarchRPC never calls get_meta, so every case asks the core directly.
    meta = await volume.core.get_meta([Request(key="weights")])
    assert meta == [(source.shape, source.dtype)]


async def _put_batch(volume: _StandInVolume) -> tuple[torch.Tensor, torch.Tensor]:
    """Put two tensors and an object at ``a``, ``b`` and ``c`` in one shared-memory call."""
    first = torch.arange(3, dtype=torch.float32)
    second = torch.ones(2, 2, dtype=torch.int64)
    puts = [
        Request.from_tensor("a", first),
        Request.from_tensor("b", second),
        Request.from_objects("c", {"step": 1}),
    ]
    ref = _ref(volume, TransportType.SharedMemory)
    await create_transport_buffer(ref).put_to_storage_volume(puts)
    return first, second


@pytest.mark.asyncio
async def test_batch_calls_answer_every_request_in_order(volume) -> None:
    """Answer each request of a batched put, get and metadata call, in request order."""
    first, second = await _put_batch(volume)
    gets = [Request(key=key) for key in ("c", "b", "a")]
    ref = _ref(volume, TransportType.SharedMemory)
    results = await create_transport_buffer(ref).get_from_storage_volume(gets)
    assert volume.calls == ["handshake", "put", "get"]
    assert results[0] == {"step": 1}
    assert torch.equal(results[1], second) and torch.equal(results[2], first)
    meta = await volume.core.get_meta(gets)
    assert meta == ["obj", (second.shape, second.dtype), (first.shape, first.dtype)]


@pytest.mark.asyncio
async def test_handshake_answers_every_request_in_order(volume) -> None:
    """Report for each request, in request order, whether the store holds its key."""
    await _put_batch(volume)
    probe = _HandshakeProbe(_ref(volume, TransportType.SharedMemory))
    existing = await volume.core.handshake(probe, [Request(key="b"), Request(key="d")])
    assert existing == [True, False]


@pytest.mark.parametrize(
    ("offsets", "local_shape"),
    [((3,), (3,)), ((-1,), (2,)), ((0, 0), (2, 1))],
    ids=["past_its_end", "before_its_start", "in_a_dim_it_lacks"],
)
@pytest.mark.asyncio
async def test_a_get_of_a_slice_outside_its_stored_tensor_raises_value_error(
    volume, offsets, local_shape
) -> None:
    """Refuse a slice outside the stored tensor, which indexing would clip, wrap or fail on with IndexError."""
    ref = _ref(volume, TransportType.MonarchRPC)
    outside = TensorSlice(
        offsets=offsets,
        coordinates=(),
        global_shape=(4,) * len(offsets),
        local_shape=local_shape,
        mesh_shape=(),
    )
    await _put(ref, "s", torch.arange(4.0))
    buffer = create_transport_buffer(ref)
    with pytest.raises(
        ValueError, match=r"falls outside the stored tensor of shape \(4,\)"
    ):
        await buffer.get_from_storage_volume([Request(key="s", tensor_slice=outside)])


@pytest.mark.asyncio
async def test_a_get_of_a_slice_that_ends_at_its_stored_tensors_end_returns_that_part(
    volume,
) -> None:
    """Return a slice that ends at the stored tensor's own end since the bound check takes that end."""
    ref = _ref(volume, TransportType.MonarchRPC)
    last = TensorSlice(
        offsets=(2,), coordinates=(), global_shape=(4,), local_shape=(2,), mesh_shape=()
    )
    await _put(ref, "s", torch.arange(4.0))
    buffer = create_transport_buffer(ref)
    [part] = await buffer.get_from_storage_volume([Request(key="s", tensor_slice=last)])
    assert torch.equal(part, torch.tensor([2.0, 3.0]))


@pytest.mark.asyncio
async def test_get_id_reads_the_hostname_on_every_call(monkeypatch) -> None:
    """Return the volume id with ``$TORCHSTORE_HOSTNAME``, read on every call."""
    volume_id = f"volume-{uuid.uuid4()}"
    core = StorageVolumeCore(volume_id)
    monkeypatch.setenv("TORCHSTORE_HOSTNAME", "first-host")
    assert await core.get_id() == (volume_id, "first-host")
    monkeypatch.setenv("TORCHSTORE_HOSTNAME", "second-host")
    assert await core.get_id() == (volume_id, "second-host")


def test_each_core_has_its_own_store() -> None:
    """Serve a new in-memory store per core."""
    first, second = (StorageVolumeCore(f"volume-{uuid.uuid4()}") for _ in range(2))
    assert isinstance(first.store, InMemoryStore)
    assert first.store is not second.store


async def _stored_and_cached(volume: _StandInVolume) -> _KeyedCache:
    """Store ``a``, ``b``, ``c``, ``e`` and ``f``, and cache entries for those and ``d``."""
    ref = _ref(volume, TransportType.MonarchRPC)
    for key in ("a", "b", "c", "e", "f"):
        await _put(ref, key, torch.ones(2))
    cache = volume.core.store.transport_context.get(_KeyedCache)
    cache.keys.update({"a", "b", "c", "d", "e", "f"})
    return cache


@pytest.mark.asyncio
async def test_delete_drops_the_key_and_its_cache_entries(volume) -> None:
    """Drop the deleted key and its cache entries, and leave other keys alone."""
    cache = await _stored_and_cached(volume)
    await volume.core.delete("a")
    assert sorted(volume.core.store.kv) == ["b", "c", "e", "f"]
    assert cache.keys == {"b", "c", "d", "e", "f"}
    with pytest.raises(KeyError, match="Key 'a' not found"):
        await _get(_ref(volume, TransportType.MonarchRPC), "a")


@pytest.mark.asyncio
async def test_deleting_a_missing_key_keeps_every_cache_entry(volume) -> None:
    """Raise for a missing key before any cache entry is dropped."""
    cache = await _stored_and_cached(volume)
    with pytest.raises(KeyError, match="Key 'd' not found"):
        await volume.core.delete("d")
    assert cache.keys == {"a", "b", "c", "d", "e", "f"}


@pytest.mark.asyncio
async def test_delete_batch_drops_every_key_and_its_cache_entries(volume) -> None:
    """Drop each key of a batch, stored or not, and its cache entries."""
    cache = await _stored_and_cached(volume)
    # Stored keys come first, last and twice, so a batch that skips one fails.
    await volume.core.delete_batch(["e", "b", "b", "d", "f"])
    assert (sorted(volume.core.store.kv), cache.keys) == (["a", "c"], {"a", "c"})


class _FailingBatchStore(InMemoryStore):
    """A store whose batched delete raises as ``StorageImpl``'s own does."""

    async def delete_batch(self, keys: list[str]) -> None:
        raise RuntimeError("batch delete failed")


@pytest.mark.asyncio
async def test_a_failed_batch_delete_keeps_the_cache_entries() -> None:
    """Drop a batch's cache entries only once the store has dropped its keys."""
    core = StorageVolumeCore(f"volume-{uuid.uuid4()}")
    core.store = _FailingBatchStore()
    cache = core.store.transport_context.get(_KeyedCache)
    cache.keys.update({"a", "b"})
    with pytest.raises(RuntimeError, match="batch delete failed"):
        await core.delete_batch(["a", "b"])
    assert cache.keys == {"a", "b"}


class _OtherKeyedCache(_KeyedCache):
    """A second cache type, so a reset that clears only one cache shows."""


@pytest.mark.asyncio
async def test_reset_drops_every_key_and_cache(volume) -> None:
    """Drop every key and clear every cache, so the next cache lookup builds a new one."""
    ref = _ref(volume, TransportType.MonarchRPC)
    for key in ("a", "b"):
        await _put(ref, key, torch.ones(2))
    store = volume.core.store
    caches = [store.transport_context.get(t) for t in (_KeyedCache, _OtherKeyedCache)]
    for cache in caches:
        cache.keys.update({"a", "b"})

    await volume.core.reset()

    assert (store.kv, [cache.keys for cache in caches]) == ({}, [set(), set()])
    for cache in caches:
        assert store.transport_context.get(type(cache)) is not cache


class _ForwardingStore(StorageImpl):
    """A store that defines no ``reset`` and forwards unknown names to an inner store."""

    def __init__(self) -> None:
        super().__init__()
        self.inner = InMemoryStore()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


@pytest.mark.asyncio
async def test_reset_reaches_a_forwarding_store() -> None:
    """Reset through a store's ``__getattr__``."""
    core = StorageVolumeCore(f"volume-{uuid.uuid4()}")
    core.store = _ForwardingStore()
    core.store.inner.kv["a"] = torch.ones(2)
    await core.reset()
    assert core.store.inner.kv == {}


@pytest.mark.asyncio
async def test_reset_needs_a_store_that_defines_reset() -> None:
    """Raise ``AttributeError`` for a store that neither defines nor forwards ``reset``."""
    core = StorageVolumeCore(f"volume-{uuid.uuid4()}")
    core.store = StorageImpl()
    with pytest.raises(AttributeError, match="reset"):
        await core.reset()


def test_store_reset_is_declared_for_type_checkers_only() -> None:
    """Declare ``StorageImpl.reset`` for type checkers only so a store's ``__getattr__`` can still forward it."""
    tree = ast.parse(inspect.getsource(torchstore.storage))
    [impl] = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "StorageImpl"
    ]
    declared = [
        statement.name
        for guard in impl.body
        if isinstance(guard, ast.If) and ast.unparse(guard.test) == "TYPE_CHECKING"
        for statement in guard.body
        if isinstance(statement, ast.FunctionDef)
    ]
    assert declared == ["reset"]
    assert not hasattr(StorageImpl, "reset")


def test_storage_module_exports_its_classes() -> None:
    """List the store classes and the core in ``__all__``, which a star import gives."""
    assert torchstore.storage.__all__ == [
        "InMemoryStore",
        "StorageImpl",
        "StorageVolumeCore",
    ]
