# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Collects publisher layouts so requester routes can be planned."""

from __future__ import annotations

import asyncio
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from typing import Any

from monarch.actor import (  # type: ignore[import-untyped]
    Actor,
    concurrent_endpoint,
    endpoint,
)

from torchstore.strategy import TorchStoreStrategy
from ._model import KeyRegistration, Registrations

__all__ = ["RoutingCoordinator"]

_LAYOUT_REGISTRATION_TIMEOUT_S = 5 * 60


@dataclass
class _PublisherTable:
    """One value per publisher for one state-dict namespace."""

    publishers: dict[str, Any] = field(default_factory=dict)
    # Coordinate idempotent publisher registration with requester reads:
    # - The first call from a publisher stores its value.
    # - Later calls from that publisher leave the stored value unchanged.
    # - Once all publishers have registered, notify every requester; subsequent
    #   requesters satisfy the condition immediately and receive the same values.
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)


class RoutingCoordinator(Actor):
    """Publishes source layouts to requesters, one namespace at a time."""

    def __init__(self) -> None:
        self._publishers: frozenset[str] | None = None
        self._strategy: TorchStoreStrategy | None = None
        self._layouts: dict[str, _PublisherTable] = {}
        self._rdma_handles: dict[str, _PublisherTable] = {}

    @endpoint
    async def init(
        self,
        publishers: Collection[str],
        strategy: TorchStoreStrategy,
    ) -> None:
        self._publishers = frozenset(publishers)
        self._strategy = strategy
        self._layouts = {}
        self._rdma_handles = {}

    @endpoint
    async def strategy(self) -> TorchStoreStrategy:
        """The strategy resolving every participating volume mesh."""
        if self._strategy is None:
            raise RuntimeError("RoutingCoordinator.init has not run")
        return self._strategy

    @concurrent_endpoint
    async def register_layout(
        self,
        rank: str,
        key: str,
        registrations: Mapping[str, KeyRegistration],
    ) -> None:
        """Record one publisher's layout without waiting for other ranks."""
        await self._register(self._layouts, rank, key, dict(registrations))

    @concurrent_endpoint
    async def get_layouts(
        self,
        key: str,
    ) -> Registrations:
        """Wait for and return every publisher layout for ``key``."""
        return await self._wait_for_publishers(self._layouts, key, "layouts")

    @concurrent_endpoint
    async def register_rdma_handles(
        self,
        rank: str,
        key: str,
        handles: Mapping[str, Any],
    ) -> None:
        """Record one publisher's direct-RDMA handles without waiting."""
        await self._register(self._rdma_handles, rank, key, dict(handles))

    @concurrent_endpoint
    async def get_rdma_handles(self, key: str) -> dict[str, Mapping[str, Any]]:
        """Wait for and return every publisher's direct-RDMA handles for ``key``."""
        return await self._wait_for_publishers(self._rdma_handles, key, "RDMA handles")

    async def _register(
        self,
        tables: dict[str, _PublisherTable],
        rank: str,
        key: str,
        value: Any,
    ) -> None:
        if self._publishers is None:
            raise RuntimeError("RoutingCoordinator.init has not run")
        if rank not in self._publishers:
            raise KeyError(f"{rank!r} is not a publisher rank")
        table = tables.setdefault(key, _PublisherTable())
        async with table.condition:
            if rank in table.publishers:
                return
            table.publishers[rank] = value
            if table.publishers.keys() == self._publishers:
                table.condition.notify_all()

    async def _wait_for_publishers(
        self,
        tables: dict[str, _PublisherTable],
        key: str,
        what: str,
    ) -> dict[str, Any]:
        if self._publishers is None:
            raise RuntimeError("RoutingCoordinator.init has not run")
        table = tables.setdefault(key, _PublisherTable())
        async with table.condition:
            try:
                await asyncio.wait_for(
                    table.condition.wait_for(
                        lambda: table.publishers.keys() == self._publishers
                    ),
                    timeout=_LAYOUT_REGISTRATION_TIMEOUT_S,
                )
            except asyncio.TimeoutError as error:
                missing = self._publishers - table.publishers.keys()
                # A registration can land while the timed-out wait reacquires the lock.
                if missing:
                    raise TimeoutError(
                        f"timed out after {_LAYOUT_REGISTRATION_TIMEOUT_S} seconds "
                        f"waiting for publisher {what} for {key!r}; "
                        f"missing publishers: {sorted(missing)}"
                    ) from error
        return table.publishers
