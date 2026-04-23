# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import logging
import os
import socket
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import cloudpickle
from monarch._src.actor import actor_mesh
from monarch._src.spmd.host_mesh import host_mesh_from_store
from monarch.actor import get_or_spawn_controller
from torch.distributed import TCPStore

import torchstore.api as _api
from torchstore.controller import Controller
from torchstore.strategy import HostStrategy, LocalRankStrategy, TorchStoreStrategy

logger: logging.Logger = logging.getLogger(__name__)


def _spmd_key(store_name: str, suffix: str) -> str:
    return f"torchstore/spmd/{store_name}/{suffix}"


@dataclass(frozen=True)
class SPMDEnv:
    rank: int
    local_rank: int
    world_size: int
    local_world_size: int
    master_addr: str
    master_port: int

    @property
    def num_hosts(self) -> int:
        return self.world_size // self.local_world_size

    @property
    def group_rank(self) -> int:
        return self.rank // self.local_world_size

    @staticmethod
    def _parse(name: str, default: str | None = None) -> str:
        value = os.environ.get(name, default)
        if value is None:
            raise RuntimeError(
                f"SPMD TorchStore initialization requires the {name} env var"
            )
        return value

    @classmethod
    def from_env(
        cls,
        *,
        master_addr: str | None = None,
        master_port: int | None = None,
    ) -> "SPMDEnv":
        rank = int(cls._parse("RANK"))
        local_rank = int(cls._parse("LOCAL_RANK"))
        world_size = int(cls._parse("WORLD_SIZE"))
        local_world_size = int(cls._parse("LOCAL_WORLD_SIZE"))

        if world_size % local_world_size != 0:
            raise ValueError(
                f"world_size ({world_size}) must be divisible by "
                f"local_world_size ({local_world_size})"
            )

        return cls(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            local_world_size=local_world_size,
            master_addr=cls._parse("MASTER_ADDR", master_addr),
            master_port=int(cls._parse("MASTER_PORT", master_port)),
        )


# TODO: Move this to ExceptionGroup once we drop python 3.10
@contextlib.contextmanager
def _capture(errors: list[Exception]):
    try:
        yield
    except Exception as e:
        errors.append(e)


class _SPMDSession:
    """Resources created by ``torchstore.spmd.initialize()``.

    The ``rendezvous`` attribute is a ``torch.distributed.TCPStore`` that
    callers can reuse for their own cross-rank coordination.
    """

    def __init__(
        self,
        *,
        rendezvous: TCPStore,
        controller: Controller,
        store_name: str,
        host_mesh: Any | None,
        is_primary: bool,
    ) -> None:
        self.rendezvous = rendezvous
        self.controller = controller
        self.is_primary = is_primary
        self._store_name = store_name
        self._host_mesh = host_mesh

    async def shutdown(self) -> None:
        """Tear down TorchStore SPMD state and any Monarch resources created here.

        Safe to call multiple times.
        """

        errors: list[Exception] = []

        # first clean up the controller and broadcast the outcome to peer ranks
        if _api._spmd_state_map.pop(self._store_name, None) is not None:
            try:
                with _capture(errors):
                    await _shutdown_spmd_state(self, self._store_name)
            finally:
                _api.reset_client(self._store_name)

        # then clean up any other resources like the hostmesh
        if self.is_primary and self._host_mesh is not None:
            with _capture(errors):
                await self._host_mesh.shutdown()
            self._host_mesh = None

        # TODO: Move this to ExceptionGroup once we drop python 3.10
        if errors:
            raise errors[0]


async def _shutdown_primary(
    session: _SPMDSession,
    store_name: str,
) -> None:
    """Tear down the controller and broadcast the outcome to peer ranks."""
    shutdown_key = _spmd_key(store_name, "shutdown")

    error: Exception | None = None
    shutdown_status = "ok"
    try:
        await session.controller.teardown.call()
    except Exception as e:
        error = e
        shutdown_status = repr(e)
    try:
        session.rendezvous.set(shutdown_key, shutdown_status)
    except Exception as e:
        if error is None:
            error = e
    if error is not None:
        raise error


async def _shutdown(
    session: _SPMDSession,
    store_name: str,
) -> None:
    """Wait for the primary rank's shutdown outcome; re-raise on failure."""
    shutdown_key = _spmd_key(store_name, "shutdown")

    try:
        shutdown_status = session.rendezvous.get(shutdown_key).decode()
    except Exception as e:
        raise RuntimeError("Timed out waiting for TorchStore shutdown") from e

    if shutdown_status != "ok":
        raise RuntimeError(
            f"TorchStore SPMD shutdown failure - '{store_name}': {shutdown_status}"
        )


async def _shutdown_spmd_state(
    session: _SPMDSession,
    store_name: str,
) -> None:
    if session.is_primary:
        await _shutdown_primary(session, store_name)
    else:
        await _shutdown(session, store_name)


async def _best_effort_cleanup(host_mesh: Any | None) -> None:
    """Best-effort cleanup of partial init state. Logs instead of raising."""
    if host_mesh is not None:
        try:
            await host_mesh.shutdown()
        except Exception:
            logger.warning(
                "torchstore.spmd: host mesh cleanup after init failure raised",
                exc_info=True,
            )


def _storage_mesh(
    strategy: TorchStoreStrategy,
    host_mesh: Any,
    *,
    local_world_size: int,
    store_name: str,
) -> Any:
    if isinstance(strategy, HostStrategy):
        # each host gets a volume
        return host_mesh.spawn_procs(name=f"{store_name}_spmd")

    # each rank gets a volume
    return host_mesh.spawn_procs(
        per_host={"gpus": local_world_size},
        name=f"{store_name}_spmd",
    )


def _validate_strategy(
    strategy: TorchStoreStrategy | None,
) -> HostStrategy | LocalRankStrategy:
    if isinstance(strategy, (HostStrategy, LocalRankStrategy)):
        return strategy
    raise RuntimeError(
        "SPMD mode requires an explicit HostStrategy or LocalRankStrategy"
    )


async def initialize(
    strategy: TorchStoreStrategy | None = None,
    store_name: str = _api.DEFAULT_TORCHSTORE_NAME,
    *,
    env: SPMDEnv | None = None,
    rendezvous_timeout: timedelta = timedelta(seconds=120),
    transport: str = "ipc",
    monarch_port: int = 26600,
) -> None:
    """Initialize TorchStore from a torchrun-style SPMD environment.

    SPMD derives its storage topology from the strategy: one volume per host
    for ``HostStrategy`` and one volume per rank for ``LocalRankStrategy``.

    When ``env`` is omitted, topology is read from the standard ``RANK`` /
    ``LOCAL_RANK`` / ``WORLD_SIZE`` / ``LOCAL_WORLD_SIZE`` / ``MASTER_ADDR``
    / ``MASTER_PORT`` env vars via :meth:`SPMDEnv.from_env`. Callers that
    need to drive init from an explicit config can build an ``SPMDEnv``
    directly and pass it via ``env=``. All ranks then call
    :func:`monarch._src.spmd.host_mesh.host_mesh_from_store` collectively —
    global rank 0 gets a ``HostMesh`` back, and non-primary ranks get ``None``.
    Global rank 0 then spawns TorchStore volumes on that mesh and broadcasts the
    controller handle through the same ``TCPStore``.

    ``transport`` selects the worker listen scheme. ``"ipc"`` (default)
    uses a per-worker Unix socket and is limited to single-host
    deployments; ``"tcp"``, ``"metatls"`` and ``"metatls-hostname"`` bind
    on ``socket.getfqdn():monarch_port`` and work across hosts. IPC is
    the default so that importing the helper doesn't silently open TCP
    ports on user machines — multi-host callers must opt in explicitly.

    This API bootstraps a fresh Monarch context for torchrun/torchx style
    callers. If the process is already running inside Monarch, use
    ``torchstore.initialize`` with an explicit ``mesh`` instead.

    Args:
        strategy: ``HostStrategy`` or ``LocalRankStrategy`` describing the
            desired volume fan-out.
        store_name: Unique name for this store instance.
        env: Optional explicit SPMD environment; defaults to
            :meth:`SPMDEnv.from_env`.
        rendezvous_timeout: Timeout for the bootstrap ``TCPStore``.
        transport: ``"ipc"`` (default), ``"tcp"``, ``"metatls"`` or
            ``"metatls-hostname"``.
        monarch_port: Port the worker binds on for TCP/metatls transports;
            ignored for ``"ipc"``.
    """

    strategy = _validate_strategy(strategy)
    if actor_mesh._context.get() is not None:
        raise RuntimeError(
            "torchstore.spmd.initialize() requires a fresh Monarch context; "
            "use torchstore.initialize(mesh=...) from an existing Monarch runtime"
        )

    if env is None:
        env = SPMDEnv.from_env()

    num_storage_volumes = (
        env.num_hosts if isinstance(strategy, HostStrategy) else env.world_size
    )

    os.environ.setdefault("HOSTNAME", socket.gethostname())

    rendezvous = TCPStore(
        env.master_addr,
        env.master_port,
        env.world_size,
        is_master=(env.rank == 0),
        timeout=rendezvous_timeout,
    )
    host_mesh = host_mesh_from_store(
        rendezvous,
        monarch_port=monarch_port,
        name=f"torchstore_{store_name}",
        transport=transport,
        rank=env.rank,
        local_rank=env.local_rank,
        world_size=env.world_size,
        local_world_size=env.local_world_size,
    )

    try:
        controller_key = _spmd_key(store_name, "controller")
        if host_mesh is not None:
            storage_mesh = _storage_mesh(
                strategy,
                host_mesh,
                local_world_size=env.local_world_size,
                store_name=store_name,
            )

            await _api.initialize(
                num_storage_volumes=num_storage_volumes,
                strategy=strategy,
                store_name=store_name,
                mesh=storage_mesh,
            )
            controller = await get_or_spawn_controller(store_name, Controller)

            # @lint-ignore PYTHONPICKLEISBAD Handle broadcast is required for SPMD init.
            rendezvous.set(controller_key, cloudpickle.dumps(controller))
        else:
            # @lint-ignore PYTHONPICKLEISBAD Handle broadcast is required for SPMD init.
            controller = cloudpickle.loads(rendezvous.get(controller_key))

        session = _SPMDSession(
            rendezvous=rendezvous,
            controller=controller,
            store_name=store_name,
            host_mesh=host_mesh,
            is_primary=(env.rank == 0),
        )
        _api._spmd_state_map[store_name] = session
    except Exception:
        await _best_effort_cleanup(host_mesh)
        raise


__all__ = ["SPMDEnv", "initialize"]
