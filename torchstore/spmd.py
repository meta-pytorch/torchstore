# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING

import torchstore.api as _api
from torchstore.backend import get_backend
from torchstore.strategy import HostStrategy, LocalRankStrategy, TorchStoreStrategy

if TYPE_CHECKING:
    from torchstore.monarch.spmd import StoreJob


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
    configure_job: "Callable[[StoreJob], None] | None" = None,
) -> None:
    """Initialize TorchStore from a torchrun-style SPMD environment.

    SPMD derives its storage topology from the strategy: one volume per host
    for ``HostStrategy`` and one volume per rank for ``LocalRankStrategy``.

    When ``env`` is omitted, topology is read from the standard ``RANK`` /
    ``LOCAL_RANK`` / ``WORLD_SIZE`` / ``LOCAL_WORLD_SIZE`` / ``MASTER_ADDR``
    / ``MASTER_PORT`` env vars via :meth:`SPMDEnv.from_env`. Callers that
    need to drive init from an explicit config can build an ``SPMDEnv``
    directly and pass it via ``env=``. All ranks then call
    :meth:`monarch.job.spmd.StoreJob.from_store` collectively. Global rank 0
    configures and connects the job, spawns TorchStore volumes on its host mesh,
    and broadcasts the controller handle through the same ``TCPStore``.

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
        configure_job: Optional callback that configures the rank 0
            :class:`monarch.job.spmd.StoreJob` before it connects. Use this to
            enable Job API components such as telemetry or remote mounts.
    """
    strategy = _validate_strategy(strategy)
    session = await get_backend("monarch").initialize_spmd(
        strategy,
        store_name,
        env=env,
        rendezvous_timeout=rendezvous_timeout,
        transport=transport,
        monarch_port=monarch_port,
        configure_job=configure_job,
    )
    _api._start_spmd_session(store_name, session)


__all__ = ["SPMDEnv", "initialize"]
