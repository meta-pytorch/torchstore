# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the storage volume's request handling as a plain class.

A stand-in volume pickles every argument and reply, as calls to the Monarch actor do.
"""

import asyncio
import inspect
import pickle
import uuid
from functools import partial
from types import SimpleNamespace
from typing import Any
from unittest.mock import create_autospec

import pytest
import torch
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

from .volume_meshes import get_through, put_through

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
    """Serves a core as the actor does, with every argument and reply pickled."""

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
    """A transport whose handshake reports which keys already hold data."""

    async def recv_handshake(
        self, ctx: TransportContext, entries: list[tuple[Request, Any]]
    ) -> list[bool]:
        return [existing is not None for _, existing in entries]


def _ref(volume: Any, volume_id: str, transport: TransportType) -> StorageVolumeRef:
    return StorageVolumeRef(
        volume, volume_id, TransportContext(), transport, get_local_hostname()
    )


async def _endpoint_shape(self) -> None:
    pass


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
        await put_through(ref, "weights", source)
        result = await get_through(ref, "weights")
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
    core = StorageVolumeCore(f"volume-{uuid.uuid4()}")
    monkeypatch.setenv("TORCHSTORE_HOSTNAME", "first-host")
    assert await core.get_id() == (core.volume_id, "first-host")
    monkeypatch.setenv("TORCHSTORE_HOSTNAME", "second-host")
    assert await core.get_id() == (core.volume_id, "second-host")


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
        await put_through(ref, key, torch.ones(2))
    store = volume.core.store
    cache = store.transport_context.get(_KeyedCache)
    cache.keys.update({"a", "b", "c", "d", "e", "f"})

    await volume.core.delete("a")
    assert sorted(store.kv) == ["b", "c", "e", "f"]
    assert cache.keys == {"b", "c", "d", "e", "f"}
    with pytest.raises(KeyError, match="Key 'a' not found"):
        await get_through(ref, "a")
    # A missing key raises before any cache entry is dropped.
    with pytest.raises(KeyError, match="Key 'd' not found"):
        await volume.core.delete("d")
    assert cache.keys == {"b", "c", "d", "e", "f"}
    # Stored keys come first, last and twice, so a batch that skips one fails.
    await volume.core.delete_batch(["e", "b", "b", "d", "missing", "f"])
    assert (sorted(store.kv), cache.keys) == (["c"], {"c"})


@pytest.mark.asyncio
async def test_reset_drops_every_key_and_cache(volume) -> None:
    """Drop every key and clear every cache, so the next cache lookup builds a new one."""
    ref = _ref(volume, volume.core.volume_id, TransportType.MonarchRPC)
    await put_through(ref, "a", torch.ones(2))
    store = volume.core.store
    cache = store.transport_context.get(_KeyedCache)
    cache.keys.add("a")

    await volume.core.reset()

    assert (store.kv, cache.keys) == ({}, set())
    assert store.transport_context.get(_KeyedCache) is not cache


def test_storage_volume_is_an_actor_over_a_plain_core() -> None:
    """Keep the volume an actor whose endpoints match the core's public methods."""
    actor = pytest.importorskip("monarch.actor")
    from torchstore import storage_volume

    assert issubclass(storage_volume.StorageVolume, actor.Actor)
    assert not issubclass(StorageVolumeCore, actor.Actor)
    methods = {name for name in vars(StorageVolumeCore) if not name.startswith("_")}
    assert methods == set(_ENDPOINTS)
    endpoint_type = type(actor.endpoint(_endpoint_shape))
    endpoints = {
        name
        for name, value in vars(storage_volume.StorageVolume).items()
        if isinstance(value, endpoint_type)
    }
    assert endpoints == set(_ENDPOINTS)
    # The volume module re-exports the store classes.
    assert storage_volume.StorageImpl is StorageImpl
    assert storage_volume.InMemoryStore is InMemoryStore


def test_storage_volume_keeps_its_store_and_volume_id() -> None:
    """Read and set the core's store and volume ID, as on upstream's actor."""
    pytest.importorskip("monarch.actor")
    from torchstore.storage_volume import StorageVolume

    stand_in = SimpleNamespace(core=StorageVolumeCore("volume"))
    store, properties = InMemoryStore(), vars(StorageVolume)
    properties["store"].fset(stand_in, store)
    assert stand_in.core.store is store
    assert properties["store"].fget(stand_in) is store
    assert properties["volume_id"].fget(stand_in) == "volume"
    properties["volume_id"].fset(stand_in, "renamed")
    assert stand_in.core.volume_id == "renamed"
    assert properties["volume_id"].fget(stand_in) == "renamed"


def test_storage_volume_endpoints_stay_serial() -> None:
    """Keep each endpoint serial, so a volume runs one call at a time."""
    actor = pytest.importorskip("monarch.actor")
    from torchstore.storage_volume import StorageVolume

    serial = type(actor.endpoint(_endpoint_shape))
    assert serial is not type(actor.concurrent_endpoint(_endpoint_shape))
    for name in _ENDPOINTS:
        assert type(StorageVolume.__dict__[name]) is serial, name


@pytest.mark.asyncio
async def test_storage_volume_delegates_each_endpoint_to_its_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward each endpoint's arguments, by the core's parameter names, and its result and error."""
    pytest.importorskip("monarch.actor")
    from monarch.actor import ActorError, get_or_spawn_controller
    from torchstore import storage_volume

    core_class = create_autospec(StorageVolumeCore)
    core = core_class.return_value
    # The volume actor runs in this process and builds its core by this name.
    monkeypatch.setattr(storage_volume, "StorageVolumeCore", core_class)
    volume = await get_or_spawn_controller(
        f"volume-{uuid.uuid4()}", storage_volume.StorageVolume, partial(str, "volume")
    )
    requests = [Request(key="weights"), Request(key="bias")]
    calls = {
        "get_id": (),
        "handshake": ("buffer", requests),
        "put": ("buffer", requests),
        "get": ("buffer", requests),
        "get_meta": (requests,),
        "delete": ("weights",),
        "delete_batch": (["weights", "bias"],),
        "reset": (),
    }
    assert set(calls) == set(_ENDPOINTS)
    # Only the core produces each reply, so a dropped one cannot pass.
    replies = {
        "get_id": ("volume", "host"),
        "handshake": [True, True],
        "put": None,
        "get": "reply-buffer",
        "get_meta": [(torch.Size([2]), torch.float32), "object"],
        "delete": None,
        "delete_batch": None,
        "reset": None,
    }
    for name, reply in replies.items():
        getattr(core, name).return_value = reply

    for name, args in calls.items():
        method = getattr(core, name)
        names = list(inspect.signature(getattr(StorageVolumeCore, name)).parameters)
        endpoint = getattr(volume, name)
        # Arguments and results cross the actor as copies, so they match by value.
        result = await asyncio.wait_for(
            endpoint.call_one(**dict(zip(names[1:], args))), timeout=30
        )
        method.assert_awaited_once_with(*args)
        assert result == replies[name], name
        error = RuntimeError(name)
        method.side_effect = error
        with pytest.raises(ActorError) as raised:
            await asyncio.wait_for(endpoint.call_one(*args), timeout=30)
        # The core's own error crosses as a copy, not wrapped in another.
        assert repr(raised.value.exception) == repr(error), name
    core_class.assert_called_once_with("volume")
