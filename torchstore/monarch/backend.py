# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The Monarch backend, which ``torchstore.backend.get_backend("monarch")`` returns."""

from collections.abc import Callable
from datetime import timedelta
from typing import Any

import torch
from monarch.actor import current_rank, get_or_spawn_controller

# The old module paths, so code that reads one as an attribute once this backend loads, such
# as ``ts.transport.monarch_rdma``, finds it.
import torchstore.controller  # noqa: F401
import torchstore.routing.coordinator  # noqa: F401
import torchstore.storage_volume  # noqa: F401
import torchstore.transport.monarch_rdma  # noqa: F401

from torchstore.backend import Backend
from torchstore.directory import VolumeDirectory
from torchstore.monarch.controller import Controller, ControllerDirectory
from torchstore.monarch.coordinator import MonarchControlPlane, RoutingCoordinator
from torchstore.monarch.rdma import RDMABuffer
from torchstore.monarch.spmd import _SPMDSession, initialize_spmd
from torchstore.monarch.storage_volume import spawn_actors, StorageVolume
from torchstore.routing.control_plane import ControlPlane
from torchstore.spmd import SPMDEnv
from torchstore.strategy import TorchStoreStrategy

__all__ = ["MonarchBackend"]


class MonarchBackend(Backend):
    """Runs volumes, controllers and routing coordinators as Monarch actors.

    Monarch's ``get_or_spawn_controller`` finds controllers, the controller volume and coordinators
    by name in the caller's context: one job's actors and the driver's threads reach the same one,
    while another root process or a plain thread in an actor's process finds or spawns its own.
    """

    def rank(self) -> int:
        """Monarch's ``current_rank()``, 0 outside an actor's context.

        Inside one, the position in the mesh or slice that the current message went to.
        """
        return current_rank().rank

    async def spawn_volumes(
        self, mesh: Any, num_volumes: int, id_func: Callable[[], str], /
    ) -> StorageVolume:
        return await StorageVolume.spawn(num_volumes, mesh, id_func=id_func)

    async def controller_volume(self, id_func: Callable[[], str], /) -> StorageVolume:
        return await get_or_spawn_controller(
            "storage_volume_controller", StorageVolume, id_func=id_func
        )

    async def controller(self, store_name: str, /) -> Controller:
        return await get_or_spawn_controller(store_name, Controller)

    def directory(self, controller: Any, /) -> VolumeDirectory:
        return ControllerDirectory(controller)

    async def coordinator(self, store_name: str, /) -> ControlPlane:
        coordinator = await get_or_spawn_controller(store_name, RoutingCoordinator)
        return MonarchControlPlane(coordinator)

    async def spawn_actors(
        self,
        num_processes: int,
        actor_cls: type,
        name: str,
        mesh: Any = None,
        **init_args: Any,
    ) -> Any:
        return await spawn_actors(num_processes, actor_cls, name, mesh, **init_args)

    async def initialize_spmd(
        self,
        strategy: TorchStoreStrategy,
        store_name: str,
        /,
        *,
        env: SPMDEnv | None,
        rendezvous_timeout: timedelta,
        transport: str,
        monarch_port: int,
        configure_job: Callable[[Any], None] | None,
    ) -> _SPMDSession:
        return await initialize_spmd(
            strategy,
            store_name,
            env=env,
            rendezvous_timeout=rendezvous_timeout,
            transport=transport,
            monarch_port=monarch_port,
            configure_job=configure_job,
        )

    def rdma_buffer(self, data: torch.Tensor, /) -> Any:
        """Monarch's ``RDMABuffer``, which falls back to TCP or raises without RDMA support.

        Where ``monarch.rdma`` fails to import, the stand-in raises ``NotImplementedError``.
        """
        return RDMABuffer(data)
