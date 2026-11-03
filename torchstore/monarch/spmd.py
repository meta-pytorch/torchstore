# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TorchStore's SPMD bootstrap on Monarch, which ``torchstore.spmd.initialize`` reaches."""

import contextlib
import logging
import os
import socket
from collections.abc import Callable
from datetime import timedelta
from typing import Any

import cloudpickle
from monarch._src.actor import actor_mesh
from monarch.actor import get_or_spawn_controller
from torch.distributed import TCPStore

try:
    from monarch.job.spmd import StoreJob
except ImportError as e:
    _store_job_import_error: ImportError = e

    class StoreJob:
        """Fallback that reports a missing store-backed Monarch Job API."""

        @classmethod
        def from_store(cls, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError(
                "StoreJob.from_store() is not available. This environment was not built with a version of monarch "
                "that includes this API. Try installing nightly monarch."
            ) from _store_job_import_error


import torchstore.api as _api
from torchstore.monarch.controller import Controller
from torchstore.spmd import _spmd_key, SPMDEnv
from torchstore.strategy import HostStrategy, TorchStoreStrategy
from torchstore.utils import TORCHSTORE_HOSTNAME_ENV

# The name these records had before the bootstrap moved here from ``torchstore.spmd``.
logger: logging.Logger = logging.getLogger("torchstore.spmd")


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
        job: StoreJob | None,
        host_mesh: Any | None,
        is_primary: bool,
    ) -> None:
        self.rendezvous = rendezvous
        self.controller = controller
        self.is_primary = is_primary
        self._store_name = store_name
        self._job = job
        self._host_mesh = host_mesh
        self._store_open = True

    async def shutdown(self) -> None:
        """Tear down TorchStore SPMD state and any Monarch resources created here.

        Safe to call multiple times, and the store is torn down at most once. ``ts.shutdown``
        drops the session before calling this and the client once this returns or raises.
        """

        errors: list[Exception] = []

        # first clean up the controller and broadcast the outcome to peer ranks
        if self._store_open:
            self._store_open = False
            with _capture(errors):
                await _shutdown_spmd_state(self, self._store_name)

        if self.is_primary and self._job is not None:
            with _capture(errors):
                self._job.kill()
            self._job = None

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


async def _best_effort_cleanup(
    job: StoreJob | None,
    host_mesh: Any | None,
) -> None:
    """Best-effort cleanup of partial init state. Logs instead of raising."""
    if job is not None:
        try:
            job.kill()
        except Exception:
            logger.warning(
                "torchstore.spmd: job cleanup after init failure raised",
                exc_info=True,
            )
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


async def initialize_spmd(
    strategy: TorchStoreStrategy,
    store_name: str,
    *,
    env: SPMDEnv | None,
    rendezvous_timeout: timedelta,
    transport: str,
    monarch_port: int,
    configure_job: Callable[[StoreJob], None] | None,
) -> _SPMDSession:
    """Join the SPMD job from a fresh Monarch context, start the store on rank 0 and return the session."""
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

    os.environ.setdefault(TORCHSTORE_HOSTNAME_ENV, socket.gethostname())

    rendezvous = TCPStore(
        env.master_addr,
        env.master_port,
        env.world_size,
        is_master=(env.rank == 0),
        timeout=rendezvous_timeout,
    )
    mesh_name = f"torchstore_{store_name}"
    job = StoreJob.from_store(
        rendezvous,
        monarch_port=monarch_port,
        name=mesh_name,
        transport=transport,
        rank=env.rank,
        local_rank=env.local_rank,
        world_size=env.world_size,
        local_world_size=env.local_world_size,
    )
    host_mesh = None

    try:
        controller_key = _spmd_key(store_name, "controller")
        if job is not None:
            if configure_job is not None:
                configure_job(job)
            host_mesh = getattr(job.state(), mesh_name)
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

        return _SPMDSession(
            rendezvous=rendezvous,
            controller=controller,
            store_name=store_name,
            job=job,
            host_mesh=host_mesh,
            is_primary=(env.rank == 0),
        )
    except Exception:
        await _best_effort_cleanup(job, host_mesh)
        raise
