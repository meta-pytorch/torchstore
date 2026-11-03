# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The Monarch backend, which ``torchstore.backend.get_backend("monarch")`` returns."""

import os
from collections.abc import Callable
from datetime import timedelta
from typing import Any

import torch
from monarch._src.actor import actor_mesh
from monarch.actor import current_rank, get_or_spawn_controller
from monarch.config import configure, get_global_config

import torchstore

# The old module paths, so code that reads one as an attribute once this backend loads, such
# as ``ts.transport.monarch_rdma``, finds it.
import torchstore.controller  # noqa: F401
import torchstore.routing.coordinator  # noqa: F401
import torchstore.storage_volume  # noqa: F401
import torchstore.transport.monarch_rdma  # noqa: F401

from torchstore.backend import Backend, register_backend
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

# Monarch's own frame limit, in torchmonarch 0.6.0 and 0.7.0.
_MONARCH_FRAME_LENGTH = 10737418240


def _apply_frame_length_default() -> None:
    """Give Monarch the default this process's import exported, and log it once in effect.

    Acts only while the variable still holds the export. A Monarch that loaded first gets it
    only over its own limit, which its config cannot tell from a limit configured to that
    value, so other limits stay. The log follows Monarch's config read after ``configure``.
    """
    exported = str(torchstore._FRAME_LENGTH)
    if not torchstore._exported_frame_length:
        return
    if os.environ.get("HYPERACTOR_CODEC_MAX_FRAME_LENGTH") != exported:
        return
    if get_global_config()["codec_max_frame_length"] == _MONARCH_FRAME_LENGTH:
        configure(codec_max_frame_length=torchstore._FRAME_LENGTH)
    if get_global_config()["codec_max_frame_length"] == torchstore._FRAME_LENGTH:
        torchstore.logger.warning(
            "Warning: setting HYPERACTOR_CODEC_MAX_FRAME_LENGTH since this needs to be set"
            " to enable large RPC calls via Monarch"
        )


# A reload of this module keeps the flag and so skips the step. An import after the module
# left sys.modules starts without the flag and runs it again.
if not globals().get("_frame_length_default_applied", False):
    _frame_length_default_applied = True
    _apply_frame_length_default()


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

    def actor_rank(self) -> int | None:
        # current_rank() would start Monarch's root client where no context is current, and a
        # driver task that used Monarch holds that root client's context, which is no actor's.
        context = actor_mesh._context.get()
        if context is None or context is actor_mesh._client_context.try_get():
            return None
        return context.message_rank.rank

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
        return MonarchControlPlane(coordinator, store_name=store_name)

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


register_backend("monarch", MonarchBackend())
