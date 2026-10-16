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

from monarch.actor import (  # type: ignore[import-untyped]
    Actor,
    concurrent_endpoint,
    endpoint,
)

from torchstore.strategy import TorchStoreStrategy
from ._model import KeyRegistration, Registrations
from .control_plane import ControlPlane
from .layout_registry import LayoutRegistry

__all__ = ["MonarchControlPlane", "RoutingCoordinator"]

logger = logging.getLogger(__name__)


class RoutingCoordinator(Actor):
    """Publishes source layouts to requesters, one namespace at a time."""

    def __init__(self) -> None:
        self._registry = LayoutRegistry()

    @endpoint
    async def init(
        self,
        publishers: Collection[str],
        strategy: TorchStoreStrategy,
    ) -> None:
        await self._registry.init(publishers, strategy)

    @endpoint
    async def strategy(self) -> TorchStoreStrategy:
        """The strategy resolving every participating volume mesh."""
        return await self._registry.strategy()

    @concurrent_endpoint
    async def register_layout(
        self,
        rank: str,
        key: str,
        registrations: Mapping[str, KeyRegistration],
    ) -> None:
        """Record one publisher's layout without waiting for other ranks."""
        await self._registry.register_layout(rank, key, registrations)

    @concurrent_endpoint
    async def get_layouts(
        self,
        key: str,
        *,
        timeout_s: float | None = None,
    ) -> Registrations:
        """Wait for and return every publisher layout for ``key``.

        ``timeout_s=None`` waits ``layout_registry._LAYOUT_REGISTRATION_TIMEOUT_S``,
        ``timeout_s <= 0`` checks once and NaN raises ``ValueError``.
        """
        return await self._registry.get_layouts(key, timeout_s=timeout_s)


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

    Actor errors pass through as ``ActorError``. Calls to a dead or unreachable actor fail
    with ``SupervisionError`` at Monarch's watchdog, but nothing ends one a live actor never
    answers. A second ``init`` from any instance resets it, as ``LayoutRegistry.init`` says.
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

        None waits the default 300 s, ``<= 0`` checks once and NaN fails at once with
        ``ActorError``. A cancelled lookup keeps waiting in the actor until its timeout.
        """
        with _timed(f"get_layouts key={key}"):
            if timeout_s is None:
                return await self._coordinator.get_layouts.call_one(key=key)
            return await self._coordinator.get_layouts.call_one(
                key=key, timeout_s=timeout_s
            )
