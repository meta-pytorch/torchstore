# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Collects publisher layouts so requester routes can be planned."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Collection, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field

from monarch.actor import (  # type: ignore[import-untyped]
    Actor,
    concurrent_endpoint,
    endpoint,
)

from torchstore.strategy import TorchStoreStrategy
from ._model import KeyRegistration, Registrations
from .control_plane import ControlPlane

__all__ = ["MonarchControlPlane", "RoutingCoordinator"]

logger = logging.getLogger(__name__)

_LAYOUT_REGISTRATION_TIMEOUT_S = 5 * 60


@dataclass
class _Layouts:
    """Publisher registrations for one state-dict namespace."""

    publishers: Registrations = field(default_factory=dict)
    # Coordinate idempotent publisher registration with requester reads:
    # - The first call from a publisher stores its layout.
    # - Later calls from that publisher leave the stored layout unchanged.
    # - Once all publishers have registered, notify every requester; subsequent
    #   requesters satisfy the condition immediately and receive the same layouts.
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)


class RoutingCoordinator(Actor):
    """Publishes source layouts to requesters, one namespace at a time."""

    def __init__(self) -> None:
        self._publishers: frozenset[str] | None = None
        self._strategy: TorchStoreStrategy | None = None
        self._layouts: dict[str, _Layouts] = {}

    @endpoint
    async def init(
        self,
        publishers: Collection[str],
        strategy: TorchStoreStrategy,
    ) -> None:
        self._publishers = frozenset(publishers)
        self._strategy = strategy
        self._layouts = {}

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
        if self._publishers is None:
            raise RuntimeError("RoutingCoordinator.init has not run")
        if rank not in self._publishers:
            raise KeyError(f"{rank!r} is not a publisher rank")
        layouts = self._layouts.setdefault(key, _Layouts())
        async with layouts.condition:
            if rank not in layouts.publishers:
                layouts.publishers[rank] = dict(registrations)
                if layouts.publishers.keys() == self._publishers:
                    layouts.condition.notify_all()

    @concurrent_endpoint
    async def get_layouts(
        self,
        key: str,
        *,
        timeout_s: float | None = None,
    ) -> Registrations:
        """Wait for and return every publisher layout for ``key``.

        ``timeout_s=None`` waits ``_LAYOUT_REGISTRATION_TIMEOUT_S`` and
        ``timeout_s <= 0`` checks once.
        """
        if self._publishers is None:
            raise RuntimeError("RoutingCoordinator.init has not run")
        if timeout_s is None:
            timeout_s = _LAYOUT_REGISTRATION_TIMEOUT_S
        timeout_s = timeout_s if timeout_s > 0 else 0
        layouts = self._layouts.setdefault(key, _Layouts())
        async with layouts.condition:
            try:
                await asyncio.wait_for(
                    layouts.condition.wait_for(
                        lambda: layouts.publishers.keys() == self._publishers
                    ),
                    timeout=timeout_s,
                )
            except asyncio.TimeoutError as error:
                missing = self._publishers - layouts.publishers.keys()
                # A registration can land while the timed-out wait reacquires the lock.
                if missing:
                    raise TimeoutError(
                        f"timed out after {timeout_s} seconds "
                        f"waiting for publisher layouts for {key!r}; "
                        f"missing publishers: {sorted(missing)}"
                    ) from error
        return layouts.publishers


@contextmanager
def _timed(operation: str) -> Iterator[None]:
    """Log one INFO line naming the call, how it ended and how long it took."""
    outcome = ""
    start = time.perf_counter()
    try:
        yield
    except asyncio.CancelledError:
        outcome = " cancelled"
        raise
    except BaseException as error:
        outcome = f" failed with {type(error).__name__}"
        raise
    finally:
        elapsed = time.perf_counter() - start
        logger.info(
            "routing control_plane=monarch %s%s took %.4fs", operation, outcome, elapsed
        )


class MonarchControlPlane(ControlPlane):
    """Adapts a ``RoutingCoordinator`` actor to the control-plane interface.

    Actor errors pass through unchanged as ``ActorError``. Calls to a dead or unreachable
    actor fail with ``SupervisionError`` at Monarch's watchdog, but nothing ends one a live
    actor never answers.
    """

    def __init__(self, coordinator: RoutingCoordinator) -> None:
        self._coordinator = coordinator

    async def init(
        self,
        publishers: Collection[str],
        strategy: TorchStoreStrategy,
    ) -> None:
        with _timed("init"):
            await self._coordinator.init.call_one(
                publishers=publishers, strategy=strategy
            )

    async def strategy(self) -> TorchStoreStrategy:
        with _timed("strategy"):
            return await self._coordinator.strategy.call_one()

    async def register_layout(
        self,
        rank: str,
        key: str,
        registrations: Mapping[str, KeyRegistration],
    ) -> None:
        with _timed(f"register_layout key={key}"):
            await self._coordinator.register_layout.call_one(
                rank=rank,
                key=key,
                registrations=registrations,
            )

    async def get_layouts(
        self,
        key: str,
        *,
        timeout_s: float | None = None,
    ) -> Registrations:
        """Wait for every publisher layout for ``key``, bounded by ``timeout_s``.

        None waits the default 300 s and ``<= 0`` checks once. A cancelled lookup keeps
        waiting in the actor until its timeout.
        """
        with _timed(f"get_layouts key={key}"):
            return await self._coordinator.get_layouts.call_one(
                key=key, timeout_s=timeout_s
            )
