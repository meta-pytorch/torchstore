# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Collects every participant's state-dict layout so routes can be planned."""

from __future__ import annotations

import asyncio
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field

from monarch.actor import (  # type: ignore[import-untyped]
    Actor,
    concurrent_endpoint,
    endpoint,
)

from torchstore.strategy import TorchStoreStrategy

from ._model import KeyRegistration, RankRole, Registrations

__all__ = ["RoutingCoordinator"]


@dataclass
class _Barrier:
    """One state-dict namespace's registrations to exchange."""

    publishers: Registrations = field(default_factory=dict)
    requesters: Registrations = field(default_factory=dict)
    complete: bool = False
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)

    @property
    def ranks(self) -> set[str]:
        return self.publishers.keys() | self.requesters.keys()


class RoutingCoordinator(Actor):
    """Exchanges per-rank layouts, one state-dict namespace at a time.

    Every namespace gets its own barrier, so a rank registers each state dict it
    routes when it is ready to and never has to declare that it is finished.
    """

    def __init__(self) -> None:
        self._ranks: frozenset[str] | None = None
        self._strategy: TorchStoreStrategy | None = None
        self._barriers: dict[str, _Barrier] = {}

    @endpoint
    async def init(
        self,
        ranks: Collection[str],
        strategy: TorchStoreStrategy,
    ) -> None:
        self._ranks = frozenset(ranks)
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
        if self._ranks is None:
            raise RuntimeError("RoutingCoordinator.init has not run")
        if rank not in self._ranks:
            raise KeyError(f"{rank!r} is not a routing participant")
        barrier = self._barriers.setdefault(key, _Barrier())
        by_role = (
            barrier.publishers if role == RankRole.PUBLISHER else barrier.requesters
        )
        async with barrier.condition:
            if rank in barrier.ranks:
                raise RuntimeError(f"rank {rank!r} registered {key!r} twice")
            by_role[rank] = dict(registrations)
            if barrier.ranks == self._ranks:
                barrier.complete = True
                barrier.condition.notify_all()
            else:
                await barrier.condition.wait_for(lambda: barrier.complete)
        return barrier

    @concurrent_endpoint
    async def register_layouts(
        self,
        rank: str,
        role: RankRole,
        key: str,
        registrations: Mapping[str, KeyRegistration],
    ) -> tuple[Registrations, Registrations]:
        """Report one rank's layout for ``key`` and get every rank's back."""
        barrier = await self._gathered(rank, role, key, registrations)
        return barrier.publishers, barrier.requesters
