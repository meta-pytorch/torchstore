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

_LAYOUT_REGISTRATION_TIMEOUT_S = 5 * 60


@dataclass
class _Layouts:
    """Stored registrations for one state-dict namespace."""

    publishers: Registrations = field(default_factory=dict)
    requesters: Registrations = field(default_factory=dict)
    # Coordinate idempotent layout registration:
    # - The first call from a rank stores its layout.
    # - Later calls from that rank leave the stored layout unchanged.
    # - Once all ranks have registered, notify every waiter; subsequent callers
    #   satisfy the condition immediately and receive the same stored layouts.
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)

    @property
    def ranks(self) -> set[str]:
        return self.publishers.keys() | self.requesters.keys()


class RoutingCoordinator(Actor):
    """Exchanges per-rank layouts, one state-dict namespace at a time.

    Every namespace stores its own layouts, so a rank registers each state dict
    it routes when ready and receives the complete set once all ranks arrive.
    """

    def __init__(self) -> None:
        self._ranks: frozenset[str] | None = None
        self._strategy: TorchStoreStrategy | None = None
        self._layouts: dict[str, _Layouts] = {}

    @endpoint
    async def init(
        self,
        ranks: Collection[str],
        strategy: TorchStoreStrategy,
    ) -> None:
        self._ranks = frozenset(ranks)
        self._strategy = strategy
        self._layouts = {}

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
    ) -> _Layouts:
        """Record one rank's layout for ``key`` and wait for every other rank."""
        if self._ranks is None:
            raise RuntimeError("RoutingCoordinator.init has not run")
        if rank not in self._ranks:
            raise KeyError(f"{rank!r} is not a routing participant")
        layouts = self._layouts.setdefault(key, _Layouts())
        by_role = (
            layouts.publishers if role == RankRole.PUBLISHER else layouts.requesters
        )
        async with layouts.condition:
            if rank not in layouts.ranks:
                by_role[rank] = dict(registrations)
                if layouts.ranks == self._ranks:
                    layouts.condition.notify_all()
            try:
                await asyncio.wait_for(
                    layouts.condition.wait_for(lambda: layouts.ranks == self._ranks),
                    timeout=_LAYOUT_REGISTRATION_TIMEOUT_S,
                )
            except asyncio.TimeoutError as error:
                missing = self._ranks - layouts.ranks
                raise TimeoutError(
                    f"timed out after {_LAYOUT_REGISTRATION_TIMEOUT_S} seconds "
                    f"waiting for layouts for {key!r}; "
                    f"missing ranks: {sorted(missing)}"
                ) from error
        return layouts

    @concurrent_endpoint
    async def register_layouts(
        self,
        rank: str,
        role: RankRole,
        key: str,
        registrations: Mapping[str, KeyRegistration],
    ) -> tuple[Registrations, Registrations]:
        """Report one rank's layout for ``key`` and get every rank's back."""
        layouts = await self._gathered(rank, role, key, registrations)
        return layouts.publishers, layouts.requesters
