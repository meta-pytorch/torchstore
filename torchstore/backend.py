# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The runtimes that host a store's actors and services, and how core modules reach them.

Internal to TorchStore and subject to change.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Iterable
from datetime import timedelta
from typing import Any, Literal, overload, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from torchstore.carrier_volumes import CarrierVolumeHost, PublisherVolume
    from torchstore.directory import VolumeDirectory
    from torchstore.routing.control_plane import ControlPlane, ControlPlaneAddress
    from torchstore.routing.service import CoordinatorService
    from torchstore.spmd import SPMDEnv
    from torchstore.strategy import TorchStoreStrategy
    from torchstore.transport.buffers import TransportContext

__all__ = [
    "Backend",
    "ServiceBackend",
    "get_backend",
    "loaded_backend",
    "register_backend",
]


class Backend(ABC):
    """Spawns and finds the actors that serve TorchStore's stores on one runtime.

    Core modules use the returned handles as Monarch actor handles without importing the
    runtime: ``call`` (iterated as point and value pairs), ``call_one`` and ``slice(**point)``
    on the actors' endpoints, and ``len()`` of a volume mesh.
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


class ServiceBackend(ABC):
    """Serves routing stores' coordinators and publishers' volumes over the carrier, beside a TCPStore.

    Its modules register exit and fork hooks when they load, so it loads with the first call
    that needs it, or when unpickling one of its addresses imports it.
    """

    @abstractmethod
    async def start_coordinator(self, store_name: str) -> CoordinatorService:
        """Start a coordinator service for routing store ``store_name`` in this process."""

    @abstractmethod
    def service_address(self, control_plane: object) -> ControlPlaneAddress | None:
        """Return the address of the service ``control_plane`` is a handle to, else None."""

    @abstractmethod
    def check_strategy(self, strategy: TorchStoreStrategy) -> None:
        """Refuse a strategy that the volumes publishers serve cannot honor."""

    @abstractmethod
    def volume_host(self, strategy: object) -> CarrierVolumeHost | None:
        """Return ``strategy``'s carrier volume host, if it resolves volumes through one."""

    @abstractmethod
    def forget_context_pairs(
        self, volume_ids: Iterable[str], transport_context: TransportContext
    ) -> None:
        """Forget the Gloo pairs to these carrier volumes whose groups ``transport_context`` holds."""

    @abstractmethod
    async def start_publisher_volume(
        self,
        volume_id: str,
        *,
        store_name: str,
        selection: tuple[Any, Any],
        register: Callable[[str, bytes], Awaitable[None]],
    ) -> PublisherVolume:
        """Serve this process's publisher volume of ``store_name``, registered through ``register``.

        ``selection`` is the (control plane, address) pair that names the store.
        """

    @abstractmethod
    def stop_publisher_volumes(self, store_name: str) -> list[PublisherVolume]:
        """Withdraw this process's publisher volumes of ``store_name``. The caller closes each."""


# The module of each backend, which registers it as it loads. No other core code names them.
_BACKENDS = {
    "monarch": "torchstore.monarch.backend",
    "service": "torchstore.routing.service",
}

# By name, what each module registered as it loaded or a stand-in a test installed. A name whose stand-in
# came before its module loaded drops out once that stand-in goes.
_backends: dict[str, Backend | ServiceBackend] = {}
# What each module registered, which stands again once a stand-in installed before it goes.
_registered: dict[str, Backend | ServiceBackend] = {}


@overload
def get_backend(name: Literal["monarch"], /) -> Backend:
    ...


@overload
def get_backend(name: Literal["service"], /) -> ServiceBackend:
    ...


@overload
def get_backend(name: str, /) -> Backend | ServiceBackend:
    ...


def get_backend(name: str, /) -> Backend | ServiceBackend:
    """Return backend ``name``, importing its module on the first call.

    Concurrent first calls return the one backend its module registers. Raises ``ValueError``
    for an unknown name, and ``ModuleNotFoundError`` naming the missing module, such as
    ``monarch`` from the torchmonarch package, if its runtime is not installed.
    """
    backend = loaded_backend(name)
    if backend is not None:
        return backend
    if name not in _BACKENDS:
        raise ValueError(
            f"unknown backend {name!r}, expected one of {sorted(_BACKENDS)}"
        )
    importlib.import_module(_BACKENDS[name])
    return _backends[name]


@overload
def loaded_backend(name: Literal["monarch"]) -> Backend | None:
    ...


@overload
def loaded_backend(name: Literal["service"]) -> ServiceBackend | None:
    ...


@overload
def loaded_backend(name: str) -> Backend | ServiceBackend | None:
    ...


def loaded_backend(name: str) -> Backend | ServiceBackend | None:
    """Return a test's stand-in for ``name``, else what its module registered, else None, importing nothing.

    A module still loading, even in another thread, has not registered yet.
    """
    backend = _backends.get(name)
    return _registered.get(name) if backend is None else backend


def register_backend(name: str, backend: Backend | ServiceBackend) -> None:
    """Record ``backend`` as ``name`` unless an earlier call did. Its module calls this as it loads, however that happens.

    A stand-in installed first stays, and the first registration stands again once the stand-in goes, also after a reload.
    """
    first = _registered.setdefault(name, backend)
    _backends.setdefault(name, first)
