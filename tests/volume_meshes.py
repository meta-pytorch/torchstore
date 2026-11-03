# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Monarch-free stand-ins for a storage volume and its mesh, put and get through a volume, and bounded waits."""

from __future__ import annotations

import asyncio
import pickle
from types import SimpleNamespace

import torch
from torchstore.storage import InMemoryStore
from torchstore.strategy import StorageVolumeRef
from torchstore.transport import create_transport_buffer
from torchstore.transport.types import Request

# The storage volume's endpoints, which its plain core defines as methods.
VOLUME_ENDPOINTS = (
    "get_id",
    "handshake",
    "put",
    "get",
    "get_meta",
    "delete",
    "delete_batch",
    "reset",
)

# How long bounded() and until() wait before cancelling, far above how long the in-process
# transport tests' steps take. A cancelled call's own cleanup can run past it.
BOUND_S = 5


def rpc_copy(value):
    """Copy ``value`` through pickle, as an RPC does."""
    return pickle.loads(pickle.dumps(value))


async def bounded(awaitable):
    return await asyncio.wait_for(awaitable, timeout=BOUND_S)


async def until(predicate):
    """Wait until ``predicate`` holds, failing after ``BOUND_S``."""

    async def poll():
        while not predicate():
            await asyncio.sleep(0.001)

    await bounded(poll())


class FakeVolume:
    """Runs StorageVolume endpoints in-process, pickling buffers like an RPC."""

    def __init__(self) -> None:
        self.store = InMemoryStore()
        self.handshake = SimpleNamespace(call_one=self._handshake)
        self.put = SimpleNamespace(call=self._put)
        self.get = SimpleNamespace(call_one=self._get)
        self.get_meta = SimpleNamespace(call_one=self.store.get_meta)
        self.handshakes = 0
        # Like an RPC, cancelling a caller does not cancel the volume's task.
        self.tasks: list[asyncio.Task] = []

    def _spawn(self, coro):
        task = asyncio.ensure_future(coro)
        self.tasks.append(task)
        return asyncio.shield(task)

    async def _handshake(self, buffer, requests):
        buffer = rpc_copy(buffer)
        self.handshakes += 1
        return rpc_copy(await self._spawn(self.store.handshake(buffer, requests)))

    async def _put(self, buffer, requests):
        await self._spawn(self.store.put(rpc_copy(buffer), requests))

    async def _get(self, buffer, requests):
        return rpc_copy(await self._spawn(self.store.get(rpc_copy(buffer), requests)))


async def put_through(ref: StorageVolumeRef, key: str, value: torch.Tensor) -> None:
    """Put ``value`` under ``key`` in the volume ``ref`` names, over its transport."""
    buffer = create_transport_buffer(ref)
    await buffer.put_to_storage_volume([Request.from_tensor(key, value)])


async def get_through(ref: StorageVolumeRef, key: str) -> torch.Tensor:
    """Get ``key`` from the volume ``ref`` names, over its transport."""
    buffer = create_transport_buffer(ref)
    [value] = await buffer.get_from_storage_volume([Request(key=key)])
    return value


class _GetId:
    """A mesh's ``get_id``, answering one (coordinates, (ID, host name)) pair per volume.

    Coordinates come as plain dicts, where Monarch's ``call`` gives its ``Point``s.
    """

    def __init__(self, volumes: list[tuple[str, str]]) -> None:
        self._volumes = volumes

    async def call(self) -> list[tuple[dict[str, int], tuple[str, str]]]:
        return [({"procs": rank}, volume) for rank, volume in enumerate(self._volumes)]


class VolumeMesh:
    """A stand-in volume mesh that pickles with its strategy. Slices name their coordinates."""

    def __init__(self, volumes: list[tuple[str, str]]) -> None:
        self.get_id = _GetId(volumes)

    def slice(self, **coord: int) -> tuple[str, dict[str, int]]:
        return ("slice", coord)
