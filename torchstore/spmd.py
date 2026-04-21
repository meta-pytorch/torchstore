# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import logging
import os
import socket
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import cloudpickle
from monarch._src.actor import actor_mesh
from monarch._src.job.job import SSHJob
from monarch.actor import enable_transport, get_or_spawn_controller
from torch.distributed import TCPStore

import torchstore.api as _api
from torchstore.controller import Controller
from torchstore.strategy import HostStrategy, LocalRankStrategy, TorchStoreStrategy


logger: logging.Logger = logging.getLogger(__name__)


def _spmd_key(store_name: str, suffix: str) -> str:
    return f"torchstore/spmd/{store_name}/{suffix}"


def _host_key(group_rank: int) -> str:
    return f"torchstore/spmd/hosts/{group_rank}"


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


def _create_ssh_job(
    *,
    python_exe: str,
    ssh_args: Sequence[str],
    monarch_port: int,
    transport: str,
) -> SSHJob:
    try:
        # only nightly monarch takes a transport arg for now
        return SSHJob(
            python_exe=python_exe,
            ssh_args=ssh_args,
            monarch_port=monarch_port,
            transport=transport,
        )
    except TypeError as e:
        if transport != "tcp":
            raise RuntimeError(
                "This Monarch runtime does not support custom SSHJob transports"
            ) from e
        return SSHJob(
            python_exe=python_exe,
            ssh_args=ssh_args,
            monarch_port=monarch_port,
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
    callers can reuse for their own cross-rank coordination
    """

    def __init__(
        self,
        *,
        rendezvous: TCPStore,
        controller: Controller,
        store_name: str,
        host_mesh: Any | None,
        job: Any | None,
        is_primary: bool,
    ) -> None:
        self.rendezvous = rendezvous
        self.controller = controller
        self.is_primary = is_primary
        self._store_name = store_name
        self._host_mesh = host_mesh
        self._job = job

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
        if self.is_primary:
            with _capture(errors):
                await self._shutdown_primary_resources()

        # TODO: Move this to ExceptionGroup once we drop python 3.10
        if errors:
            raise errors[0]

    async def _shutdown_primary_resources(self) -> None:
        if self._host_mesh is not None:
            await self._host_mesh.shutdown()
        elif self._job is not None:
            self._job.kill()

        self._host_mesh = None
        self._job = None


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


async def _best_effort_cleanup(host_mesh: Any | None, job: Any | None) -> None:
    """Best-effort cleanup of partial init state. Logs instead of raising."""
    try:
        if host_mesh is not None:
            await host_mesh.shutdown()
        elif job is not None:
            job.kill()
    except Exception:
        logger.warning("Cleanup after SPMD init failure raised", exc_info=True)


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
    transport: str = "tcp",
    python_exe: str | None = None,
    monarch_port: int = 26600,
    ssh_args: Sequence[str] = (),
) -> None:
    """Initialize TorchStore from a torchrun-style SPMD environment.

    SPMD derives its storage topology from the strategy: one volume per host for
    ``HostStrategy`` and one volume per rank for ``LocalRankStrategy``.

    When ``env`` is omitted, it is populated from the
    standard ``RANK`` / ``WORLD_SIZE`` / ``MASTER_ADDR`` / ``MASTER_PORT`` env
    vars via :meth:`SPMDEnv.from_env`.

    This API bootstraps a fresh Monarch context for torchrun/torchx style callers. If
    the process is already running inside Monarch, use ``torchstore.initialize``
    with an explicit ``mesh`` instead.
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
    host_mesh: Any | None = None
    job: Any | None = None

    try:
        if env.local_rank == 0:
            # only one process per host needs to set the hostname
            rendezvous.set(_host_key(env.group_rank), socket.getfqdn().encode())

        # gather all hostnames for the job
        hostnames = [
            rendezvous.get(_host_key(host_idx)).decode()
            for host_idx in range(env.num_hosts)
        ]
        enable_transport(transport)

        controller_key = _spmd_key(store_name, "controller")

        if env.rank == 0:
            job = _create_ssh_job(
                python_exe=sys.executable if python_exe is None else python_exe,
                ssh_args=ssh_args,
                monarch_port=monarch_port,
                transport=transport,
            )
            job.add_mesh("hosts", hostnames)
            job.apply()

            host_mesh = job.state(cached_path=None).hosts
            await host_mesh.initialized

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
            job=job,
            is_primary=(env.rank == 0),
        )
        _api._spmd_state_map[store_name] = session
    except Exception:
        if env.rank == 0:
            await _best_effort_cleanup(host_mesh, job)
        raise


__all__ = ["SPMDEnv", "initialize"]
