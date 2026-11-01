# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The Monarch actor that serves a storage volume, a subclass of its plain core."""

import asyncio
import inspect
import uuid
from functools import partial
from unittest.mock import ANY, AsyncMock

import pytest
import torchstore.storage
from monarch.actor import (
    Actor,
    ActorError,
    concurrent_endpoint,
    endpoint,
    get_or_spawn_controller,
)
from torchstore.monarch import storage_volume
from torchstore.monarch.storage_volume import StorageVolume
from torchstore.storage import InMemoryStore, StorageImpl, StorageVolumeCore
from torchstore.transport.types import Request

from .test_volume_core import _ENDPOINTS

pytestmark = pytest.mark.timeout(120)


async def _endpoint_shape(self) -> None:
    pass


def _actor_endpoints() -> set[str]:
    """The names of the volume actor's endpoints."""
    endpoint_type = type(endpoint(_endpoint_shape))
    return {
        name
        for name, value in vars(StorageVolume).items()
        if isinstance(value, endpoint_type)
    }


def test_storage_volume_is_an_actor_over_its_plain_core() -> None:
    """Keep the volume an actor that subclasses the core, with an endpoint per core method."""
    assert issubclass(StorageVolume, Actor)
    assert issubclass(StorageVolume, StorageVolumeCore)
    assert not issubclass(StorageVolumeCore, Actor)
    methods = {name for name in vars(StorageVolumeCore) if not name.startswith("_")}
    assert methods == set(_ENDPOINTS)
    assert _actor_endpoints() == set(_ENDPOINTS)
    # The volume module re-exports the store classes, and lists them in its ``__all__``.
    assert storage_volume.StorageImpl is StorageImpl
    assert storage_volume.InMemoryStore is InMemoryStore
    assert storage_volume.__all__ == [
        "FULL_TENSOR",
        "InMemoryStore",
        "StorageImpl",
        "StorageVolume",
    ]


def test_storage_volume_endpoints_stay_serial() -> None:
    """Keep each endpoint serial, so a volume runs one call at a time."""
    serial = type(endpoint(_endpoint_shape))
    assert serial is not type(concurrent_endpoint(_endpoint_shape))
    for name in _ENDPOINTS:
        assert type(StorageVolume.__dict__[name]) is serial, name


def test_storage_volume_keeps_plain_store_and_volume_id() -> None:
    """Keep ``store`` and ``volume_id`` plain instance attributes, as main's actor did."""
    volume = StorageVolume(partial(str, "volume"))
    assert sorted(vars(volume)) == ["store", "volume_id"]
    assert isinstance(volume.store, InMemoryStore)
    assert volume.volume_id == "volume"
    # The class defines neither, so a client's handle offers neither.
    assert not hasattr(StorageVolume, "store")
    assert not hasattr(StorageVolume, "volume_id")


def test_storage_volume_builds_its_store_before_it_asks_for_its_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Build the store through ``torchstore.storage`` and then call ``id_func``, as main did."""
    calls: list[str] = []

    class _RecordingStore(InMemoryStore):
        def __init__(self) -> None:
            calls.append("store")
            super().__init__()

    def id_func() -> str:
        calls.append("id_func")
        return "volume"

    monkeypatch.setattr(torchstore.storage, "InMemoryStore", _RecordingStore)
    volume = StorageVolume(id_func)
    assert calls == ["store", "id_func"]
    assert list(vars(volume)) == ["store", "volume_id"]
    assert type(volume.store) is _RecordingStore


@pytest.mark.asyncio
async def test_storage_volume_delegates_each_endpoint_to_its_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward each endpoint's arguments, by the core's parameter names, its reply and its error."""
    signatures = {
        name: list(inspect.signature(vars(StorageVolumeCore)[name]).parameters)
        for name in _ENDPOINTS
    }
    # The volume actor runs in this process, so it calls these stand-ins.
    core = {name: AsyncMock(return_value=f"{name}-reply") for name in _ENDPOINTS}
    for name, method in core.items():
        monkeypatch.setattr(StorageVolumeCore, name, method)
    volume = await get_or_spawn_controller(
        f"volume-{uuid.uuid4()}", StorageVolume, partial(str, "volume")
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
    assert set(calls) == _actor_endpoints()

    for name, args in calls.items():
        method, endpoint_ = core[name], getattr(volume, name)
        # Arguments and replies cross the actor as copies, so they match by value.
        keywords = dict(zip(signatures[name][1:], args))
        result = await asyncio.wait_for(endpoint_.call_one(**keywords), timeout=30)
        method.assert_awaited_once_with(ANY, *args)
        assert result == f"{name}-reply", name
        error = RuntimeError(name)
        method.side_effect = error
        with pytest.raises(ActorError) as raised:
            await asyncio.wait_for(endpoint_.call_one(*args), timeout=30)
        # The core's own error crosses as a copy, not wrapped in another.
        assert repr(raised.value.exception) == repr(error), name
