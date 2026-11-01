# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the storage volume's request handling as a plain class.

A stand-in volume pickles every argument and successful reply, as calls to the Monarch actor
do, while a core's error reaches the caller as is, where Monarch wraps it in ``ActorError``.
"""

import pickle
import uuid
from typing import Any

import pytest
import torch
import torchstore.storage
from torchstore.storage import InMemoryStore, StorageImpl, StorageVolumeCore
from torchstore.strategy import StorageVolumeRef
from torchstore.transport import create_transport_buffer, TransportType
from torchstore.transport.buffers import (
    TransportBuffer,
    TransportCache,
    TransportContext,
)
from torchstore.transport.types import Request
from torchstore.utils import get_local_hostname

pytestmark = pytest.mark.timeout(120)

_ENDPOINTS = (
    "get_id",
    "handshake",
    "put",
    "get",
    "get_meta",
    "delete",
    "delete_batch",
    "reset",
)
# The volume calls that one put and one get make on each transport.
_TRANSPORT_CALLS = {
    TransportType.SharedMemory: ["handshake", "put", "get"],
    TransportType.Gloo: ["handshake", "put", "get_meta", "get"],
    TransportType.MonarchRPC: ["put", "get"],
}


def _wire(value: Any) -> Any:
    return pickle.loads(pickle.dumps(value))


class _Endpoint:
    def __init__(self, volume: "_StandInVolume", name: str) -> None:
        self._volume = volume
        self._name = name

    async def call_one(self, *args: Any) -> Any:
        self._volume.calls.append(self._name)
        method = getattr(self._volume.core, self._name)
        return _wire(await method(*_wire(args)))

    call = call_one


class _StandInVolume:
    """Serves a core with every argument and successful reply pickled, as the actor does.

    Unlike the actor, its ``call`` answers once rather than per volume, and calls neither queue nor finish once cancelled.
    """

    def __init__(self, core: StorageVolumeCore) -> None:
        self.core = core
        self.calls: list[str] = []
        for name in _ENDPOINTS:
            setattr(self, name, _Endpoint(self, name))

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


def _ref(volume: Any, volume_id: str, transport: TransportType) -> StorageVolumeRef:
    return StorageVolumeRef(
        volume, volume_id, TransportContext(), transport, get_local_hostname()
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
    core = StorageVolumeCore(f"volume-{uuid.uuid4()}")
    yield _StandInVolume(core)
    core.store.reset()


@pytest.mark.parametrize("transport", list(_TRANSPORT_CALLS))
@pytest.mark.asyncio
async def test_transport_round_trip_makes_only_its_calls(volume, transport) -> None:
    """Put and get a tensor through each transport, which calls only what it needs."""
    ref = _ref(volume, volume.core.volume_id, transport)
    source = torch.arange(12, dtype=torch.float32).view(3, 4)
    try:
        await _put(ref, "weights", source)
        result = await _get(ref, "weights")
    finally:
        ref.transport_context.clear()
    assert torch.equal(result, source)
    assert volume.calls == _TRANSPORT_CALLS[transport]
    # MonarchRPC never calls get_meta, so every case asks the core directly.
    meta = await volume.core.get_meta([Request(key="weights")])
    assert meta == [(source.shape, source.dtype)]


@pytest.mark.asyncio
async def test_batch_calls_answer_every_request_in_order(volume) -> None:
    """Answer each request of a batched put, get, metadata and handshake call, in request order."""
    ref = _ref(volume, volume.core.volume_id, TransportType.SharedMemory)
    first = torch.arange(3, dtype=torch.float32)
    second = torch.ones(2, 2, dtype=torch.int64)
    puts = [
        Request.from_tensor("a", first),
        Request.from_tensor("b", second),
        Request.from_objects("c", {"step": 1}),
    ]
    gets = [Request(key=key) for key in ("c", "b", "a")]
    try:
        await create_transport_buffer(ref).put_to_storage_volume(puts)
        results = await create_transport_buffer(ref).get_from_storage_volume(gets)
    finally:
        ref.transport_context.clear()
    assert volume.calls == ["handshake", "put", "get"]
    assert results[0] == {"step": 1}
    assert torch.equal(results[1], second) and torch.equal(results[2], first)
    meta = await volume.core.get_meta(gets)
    assert meta == ["obj", (second.shape, second.dtype), (first.shape, first.dtype)]
    probe = _HandshakeProbe(ref)
    existing = await volume.core.handshake(probe, [Request(key="b"), Request(key="d")])
    assert existing == [True, False]


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


@pytest.mark.asyncio
async def test_delete_drops_the_key_and_its_cache_entries(volume) -> None:
    """Drop each deleted key and its cache entries, and leave other keys alone."""
    ref = _ref(volume, volume.core.volume_id, TransportType.MonarchRPC)
    for key in ("a", "b", "c", "e", "f"):
        await _put(ref, key, torch.ones(2))
    store = volume.core.store
    cache = store.transport_context.get(_KeyedCache)
    cache.keys.update({"a", "b", "c", "d", "e", "f"})

    await volume.core.delete("a")
    assert sorted(store.kv) == ["b", "c", "e", "f"]
    assert cache.keys == {"b", "c", "d", "e", "f"}
    with pytest.raises(KeyError, match="Key 'a' not found"):
        await _get(ref, "a")
    # A missing key raises before any cache entry is dropped.
    with pytest.raises(KeyError, match="Key 'd' not found"):
        await volume.core.delete("d")
    assert cache.keys == {"b", "c", "d", "e", "f"}
    # Stored keys come first, last and twice, so a batch that skips one fails.
    await volume.core.delete_batch(["e", "b", "b", "d", "f"])
    assert (sorted(store.kv), cache.keys) == (["c"], {"c"})


class _OtherKeyedCache(_KeyedCache):
    """A second cache type, so a reset that clears only one cache shows."""


@pytest.mark.asyncio
async def test_reset_drops_every_key_and_cache(volume) -> None:
    """Drop every key and clear every cache, so the next cache lookup builds a new one."""
    ref = _ref(volume, volume.core.volume_id, TransportType.MonarchRPC)
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
async def test_reset_reaches_a_forwarding_store_and_needs_a_reset_otherwise() -> None:
    """Reset through a store's ``__getattr__``, and raise ``AttributeError`` without either."""
    core = StorageVolumeCore(f"volume-{uuid.uuid4()}")
    core.store = _ForwardingStore()
    core.store.inner.kv["a"] = torch.ones(2)
    await core.reset()
    assert core.store.inner.kv == {}

    core.store = StorageImpl()
    with pytest.raises(AttributeError, match="reset"):
        await core.reset()


def test_storage_module_exports_its_classes() -> None:
    """List the store classes and the core in ``__all__``, which a star import gives."""
    assert torchstore.storage.__all__ == [
        "InMemoryStore",
        "StorageImpl",
        "StorageVolumeCore",
    ]
