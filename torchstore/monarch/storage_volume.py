# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import uuid
from logging import getLogger
from typing import Any

import torch
from monarch.actor import Actor, endpoint, this_host

# Registers Monarch RDMA wherever a volume handle loads, such as in a worker handed a
# controller, so transport selection offers it there.
import torchstore.monarch.rdma  # noqa: F401
import torchstore.storage
from torchstore.storage import InMemoryStore, StorageImpl, StorageVolumeCore
from torchstore.transport.buffers import TransportBuffer
from torchstore.transport.types import Request

__all__ = ["FULL_TENSOR", "InMemoryStore", "StorageImpl", "StorageVolume"]

# The old module's logger, which ``torchstore.storage_volume`` still re-exports, and the
# logger main's ``spawn_actors`` used in ``torchstore.utils``.
logger = getLogger("torchstore.storage_volume")
_spawn_logger = getLogger("torchstore.utils")


FULL_TENSOR = "full_tensor"


async def spawn_actors(num_processes, actor_cls, name, mesh=None, **init_args):
    """Actors are essentially processes wrapped in a class."""

    if mesh is None:
        _spawn_logger.debug("Spawning actors on the local host")
        mesh = this_host().spawn_procs(per_host={"gpus": num_processes})

    assert hasattr(mesh, "spawn")
    await mesh.initialized
    actors = mesh.spawn(f"{name}_{str(uuid.uuid4())[:8]}", actor_cls, **init_args)

    return actors


class StorageVolume(Actor, StorageVolumeCore):
    """The remote logic for storage. Recieves remote put/get requests and handles them via the storage abstraction"""

    actor_name: str = "StorageVolumes"

    def __init__(
        self,
        id_func,
    ) -> None:
        # Builds the store before calling id_func, from the module where patches of it apply.
        self.store: StorageImpl = torchstore.storage.InMemoryStore()
        self.volume_id: str = id_func()

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
        return await StorageVolumeCore.get_id(self)

    @endpoint
    async def handshake(
        self,
        transport_buffer: TransportBuffer,
        requests: list[Request],
    ) -> list[Any]:
        return await StorageVolumeCore.handshake(self, transport_buffer, requests)

    @endpoint
    async def put(
        self,
        transport_buffer: TransportBuffer,
        requests: list[Request],
    ) -> None:
        return await StorageVolumeCore.put(self, transport_buffer, requests)

    @endpoint
    async def get(
        self,
        transport_buffer: TransportBuffer,
        requests: list[Request],
    ) -> TransportBuffer:
        return await StorageVolumeCore.get(self, transport_buffer, requests)

    @endpoint
    async def get_meta(
        self,
        requests: list[Request],
    ) -> list[tuple[torch.Size, torch.dtype] | str]:
        return await StorageVolumeCore.get_meta(self, requests)

    @endpoint
    async def delete(self, key: str) -> None:
        return await StorageVolumeCore.delete(self, key)

    @endpoint
    async def delete_batch(self, keys: list[str]) -> None:
        return await StorageVolumeCore.delete_batch(self, keys)

    @endpoint
    async def reset(self) -> None:
        return await StorageVolumeCore.reset(self)
