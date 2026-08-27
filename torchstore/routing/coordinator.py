# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Collects every participant's state-dict layout so routes can be planned."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field

from monarch.actor import (  # type: ignore[import-untyped]
    Actor,
    concurrent_endpoint,
    endpoint,
)

from torchstore.strategy import TorchStoreStrategy

from ._model import KeyRegistration, RankRole, Registrations
from .plan import RoutingPlan
from .service import RoutingService, RoutingServiceGroup

__all__ = ["RoutingCoordinator"]


@dataclass
class _Barrier:
    """One state-dict namespace's registrations and the plan they produce."""

    publishers: Registrations = field(default_factory=dict)
    requesters: Registrations = field(default_factory=dict)
    complete: bool = False
    # Built once, by whichever rank asks first, then shared
    plan: RoutingPlan | None = None
    error: Exception | None = None
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)

    @property
    def ranks(self) -> set[str]:
        return self.publishers.keys() | self.requesters.keys()


class RoutingCoordinator(Actor):
    """Turns per-rank layouts into per-rank route tables, one namespace at a time.

    Every namespace gets its own barrier, so a rank registers each state dict it
    routes when it is ready to and never has to declare that it is finished.
    """

    def __init__(self) -> None:
        self._services: RoutingServiceGroup | None = None
        self._strategy: TorchStoreStrategy | None = None
        self._barriers: dict[str, _Barrier] = {}

    @endpoint
    async def init(
        self,
        services: RoutingServiceGroup,
        strategy: TorchStoreStrategy,
    ) -> None:
        self._services = services
        self._strategy = strategy
        self._barriers = {}

    @endpoint
    async def strategy(self) -> TorchStoreStrategy:
        """The strategy resolving every participating volume mesh."""
        if self._strategy is None:
            raise RuntimeError("RoutingCoordinator.init has not run")
        return self._strategy

    async def _gathered(
        self,
        rank: str,
        role: RankRole,
        key: str,
        registrations: Mapping[str, KeyRegistration],
    ) -> _Barrier:
        """Record one rank's layout for ``key`` and wait for every other rank."""
        if self._services is None:
            raise RuntimeError("RoutingCoordinator.init has not run")
        if rank not in self._services.ranks:
            raise KeyError(f"{rank!r} is not a routing participant")
        barrier = self._barriers.setdefault(key, _Barrier())
        by_role = (
            barrier.publishers if role == RankRole.PUBLISHER else barrier.requesters
        )
        async with barrier.condition:
            if rank in barrier.ranks:
                raise RuntimeError(f"rank {rank!r} registered {key!r} twice")
            by_role[rank] = dict(registrations)
            if barrier.ranks == self._services.ranks:
                barrier.complete = True
                barrier.condition.notify_all()
            else:
                await barrier.condition.wait_for(lambda: barrier.complete)
        return barrier

    @concurrent_endpoint
    async def register(
        self,
        rank: str,
        role: RankRole,
        key: str,
        registrations: Mapping[str, KeyRegistration],
    ) -> tuple[RoutingPlan, dict[str, RoutingService]]:
        """Report one rank's layout for ``key`` and get its plan back built here."""
        barrier = await self._gathered(rank, role, key, registrations)
        async with barrier.condition:
            if barrier.plan is None and barrier.error is None:
                # Remembered, or the ranks behind this one each rebuild a plan
                # that is never going to succeed.
                try:
                    barrier.plan = RoutingPlan.build(
                        barrier.publishers, barrier.requesters
                    )
                except Exception as error:  # noqa: BLE001 - re-raised below
                    barrier.error = error
        if barrier.error is not None:
            raise barrier.error
        assert barrier.plan is not None
        assert self._services is not None  # _gathered raises when it is not

        peers = {
            peer
            for entry in barrier.plan._local(rank).keys.values()
            for route in entry.routes
            for peer in route.notify_peers
        }
        return (
            barrier.plan.for_rank(rank),
            {peer: self._services.get_service(peer) for peer in peers | {rank}},
        )

    @concurrent_endpoint
    async def register_layouts(
        self,
        rank: str,
        role: RankRole,
        key: str,
        registrations: Mapping[str, KeyRegistration],
    ) -> tuple[Registrations, Registrations, dict[str, RoutingService]]:
        """Report one rank's layout for ``key`` and get every rank's back."""
        barrier = await self._gathered(rank, role, key, registrations)
        assert self._services is not None  # _gathered raises when it is not
        return (
            barrier.publishers,
            barrier.requesters,
            {peer: self._services.get_service(peer) for peer in self._services.ranks},
        )
