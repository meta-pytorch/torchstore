# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from logging import getLogger
from typing import Any

import torch
from monarch.actor import Actor, endpoint

from torchstore.storage import InMemoryStore, StorageImpl, StorageVolumeCore
from torchstore.transport.buffers import TransportBuffer
from torchstore.transport.types import Request
from torchstore.utils import spawn_actors

__all__ = ["FULL_TENSOR", "InMemoryStore", "StorageImpl", "StorageVolume"]

logger = getLogger(__name__)


FULL_TENSOR = "full_tensor"


class StorageVolume(Actor):
    """The remote logic for storage. Recieves remote put/get requests and handles them via the storage abstraction"""

    actor_name: str = "StorageVolumes"

    def __init__(
        self,
        id_func,
    ) -> None:
        self.core = StorageVolumeCore(id_func())

    @classmethod
    async def spawn(
        cls,
        num_volumes: int,
        mesh,
        *init_args: Any,
        **init_kwargs: Any,
    ) -> "StorageVolume":
        actors = await spawn_actors(
            num_volumes, cls, cls.actor_name, mesh, *init_args, **init_kwargs
        )

        return actors

    @endpoint
    async def get_id(self) -> tuple[str, str]:
        return await self.core.get_id()

    @endpoint
    async def handshake(
        self,
        transport_buffer: TransportBuffer,
        requests: list[Request],
    ) -> list[Any]:
        return await self.core.handshake(transport_buffer, requests)

    @endpoint
    async def put(
        self,
        transport_buffer: TransportBuffer,
        requests: list[Request],
    ) -> None:
        await self.core.put(transport_buffer, requests)

    @endpoint
    async def get(
        self,
        transport_buffer: TransportBuffer,
        requests: list[Request],
    ) -> TransportBuffer:
        return await self.core.get(transport_buffer, requests)

    @endpoint
    async def get_meta(
        self,
        requests: list[Request],
    ) -> list[tuple[torch.Size, torch.dtype] | str]:
        return await self.core.get_meta(requests)

    @endpoint
    async def delete(self, key: str) -> None:
        await self.core.delete(key)

    @endpoint
    async def delete_batch(self, keys: list[str]) -> None:
        await self.core.delete_batch(keys)

    @endpoint
    async def reset(self) -> None:
        await self.core.reset()
