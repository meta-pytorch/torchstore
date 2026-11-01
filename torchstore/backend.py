# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The runtime that hosts a store's actors, and how core modules reach it.

Internal to TorchStore and subject to change.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import timedelta
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from torchstore.directory import VolumeDirectory
    from torchstore.routing.control_plane import ControlPlane
    from torchstore.spmd import SPMDEnv
    from torchstore.strategy import TorchStoreStrategy

__all__ = ["Backend", "get_backend"]


class Backend(ABC):
    """Spawns and finds the actors that serve TorchStore's stores on one runtime.

    Core modules call the controller and volume handles returned here as Monarch actor
    handles, through ``call``, ``call_one`` and a mesh's ``slice`` on the endpoints those
    actors define, and never import the runtime themselves.
    """

    @abstractmethod
    def rank(self) -> int:
        """Return the caller's rank, which routing volume IDs use.

        That is its position in the mesh, or the slice of it, that its current message went to.
        """

    @abstractmethod
    async def spawn_volumes(
        self, mesh: Any, num_volumes: int, id_func: Callable[[], str], /
    ) -> Any:
        """Spawn storage volumes on ``mesh``, or ``num_volumes`` on a new local mesh if None.

        Each volume calls ``id_func`` once to name itself. Returns the volume mesh that the
        strategies resolve, through its ``get_id`` endpoint and ``slice``.
        """

    @abstractmethod
    async def controller_volume(self, id_func: Callable[[], str], /) -> Any:
        """Return the storage volume ``ControllerStorageVolumes`` uses, spawning it if none.

        A volume this call spawns calls ``id_func`` once to name itself.
        """

    @abstractmethod
    async def controller(self, store_name: str, /) -> Any:
        """Return store ``store_name``'s controller actor, spawning it if the lookup finds none.

        Core calls endpoints such as ``init`` and ``teardown``, and ``LocalClient`` calls
        others, such as ``locate_volumes``.
        """

    @abstractmethod
    def directory(self, controller: Any, /) -> VolumeDirectory:
        """Return the directory that locates keys through ``controller``."""

    @abstractmethod
    async def coordinator(self, store_name: str, /) -> ControlPlane:
        """Return routing store ``store_name``'s coordinator as a control plane.

        The lookup spawns the coordinator if it finds none.
        """

    @abstractmethod
    async def spawn_actors(
        self,
        num_processes: int,
        actor_cls: type,
        name: str,
        mesh: Any = None,
        **init_args: Any,
    ) -> Any:
        """Spawn ``actor_cls`` on ``mesh``, or on ``num_processes`` new local processes if None.

        Returns the spawned actors' mesh.
        """

    @abstractmethod
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
    ) -> Any:
        """Join the runtime on every rank and start the store on rank 0, for ``torchstore.spmd.initialize``.

        Returns the session, whose ``controller`` ``ts.client`` uses. Its async ``shutdown()``
        tears the store down on rank 0 and, on every other rank, waits for rank 0's outcome.
        """

    @abstractmethod
    def rdma_buffer(self, data: torch.Tensor, /) -> Any:
        """Register ``data``, a byte view, for one-sided reads, as direct weight sync does.

        Returns the registered buffer, whose ``drop()`` the caller must await to unpin the
        memory once no reader needs it. Without RDMA support the runtime chooses what happens.
        """


# The module and class of each backend, loaded on first use. No other core code names them.
_BACKENDS = {"monarch": ("torchstore.monarch.backend", "MonarchBackend")}

# Each backend loaded so far, by name. Tests install stand-ins here.
_backends: dict[str, Backend] = {}


def get_backend(name: str, /) -> Backend:
    """Return backend ``name``, importing its module on the first call.

    Raises ``ValueError`` for an unknown name, and ``ModuleNotFoundError`` naming the missing
    module, such as ``monarch`` from the torchmonarch package, if its runtime is not installed.
    """
    backend = _backends.get(name)
    if backend is not None:
        return backend
    if name not in _BACKENDS:
        raise ValueError(
            f"unknown backend {name!r}, expected one of {sorted(_BACKENDS)}"
        )
    module_name, class_name = _BACKENDS[name]
    backend = getattr(importlib.import_module(module_name), class_name)()
    _backends[name] = backend
    return backend
