# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The routing coordinator's rules, as a plain class that any host can run."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

# Annotations only: this module's own imports stay in the standard library.
if TYPE_CHECKING:
    from torchstore.strategy import TorchStoreStrategy
    from ._model import KeyRegistration, Registrations

__all__ = ["LayoutRegistry"]

logger = logging.getLogger(__name__)

_LAYOUT_REGISTRATION_TIMEOUT_S = 5 * 60


@dataclass
class _Layouts:
    """Publisher registrations for one state-dict namespace."""

    publishers: Registrations = field(default_factory=dict)
    # Set once every publisher registers. No lock: no call awaits between its checks
    # and writes, so a cancelled call holds nothing.
    complete: asyncio.Event = field(default_factory=asyncio.Event)
    # Lookups in progress. The last to leave drops the entry if no layout arrived.
    lookups: int = 0


class LayoutRegistry:
    """Holds one routing store's publishers, strategy and publisher layouts.

    A host runs every call on one event loop and lets calls interleave, as the
    ``RoutingCoordinator`` actor does. Results are the registry's own objects, so a
    host that does not serialize replies hands each caller a copy of the layouts.
    """

    def __init__(self) -> None:
        self._publishers: frozenset[str] | None = None
        self._strategy: TorchStoreStrategy | None = None
        self._layouts: dict[str, _Layouts] = {}

    async def init(
        self,
        publishers: Collection[str],
        strategy: TorchStoreStrategy,
    ) -> None:
        """Record the publishers and strategy, at most once.

        A second call raises, since a store is never initialized again.
        """
        if self._publishers is not None:
            raise RuntimeError("RoutingCoordinator.init already ran")
        if not publishers:
            raise ValueError("a routing store needs at least one publisher rank")
        self._publishers = frozenset(publishers)
        self._strategy = strategy

    async def strategy(self) -> TorchStoreStrategy:
        """Return a strategy that resolves the registered publishers' volumes."""
        if self._strategy is None:
            raise RuntimeError("RoutingCoordinator.init has not run")
        return self._strategy

    async def register_layout(
        self,
        rank: str,
        key: str,
        registrations: Mapping[str, KeyRegistration],
    ) -> None:
        """Record one publisher's layout without waiting for other ranks.

        A rank's first layout per key wins, and only publisher ranks register.
        """
        if self._publishers is None:
            raise RuntimeError("RoutingCoordinator.init has not run")
        if rank not in self._publishers:
            raise KeyError(f"{rank!r} is not a publisher rank")
        layouts = self._layouts.setdefault(key, _Layouts())
        if rank not in layouts.publishers:
            layouts.publishers[rank] = dict(registrations)
            if layouts.publishers.keys() == self._publishers:
                layouts.complete.set()

    async def get_layouts(
        self,
        key: str,
        *,
        timeout_s: float | None = None,
    ) -> Registrations:
        """Wait for and return every publisher layout for ``key``.

        ``timeout_s=None`` waits ``_LAYOUT_REGISTRATION_TIMEOUT_S`` and
        ``timeout_s <= 0`` checks once. A timeout names the missing publishers. Layouts
        that complete only as a positive wait times out are returned with a WARNING.
        """
        if self._publishers is None:
            raise RuntimeError("RoutingCoordinator.init has not run")
        if timeout_s is None:
            timeout_s = _LAYOUT_REGISTRATION_TIMEOUT_S
        timeout_s = timeout_s if timeout_s > 0 else 0
        layouts = self._layouts.setdefault(key, _Layouts())
        layouts.lookups += 1
        try:
            await asyncio.wait_for(layouts.complete.wait(), timeout=timeout_s)
        except asyncio.TimeoutError as error:
            # The last registration can land as the deadline fires.
            if not layouts.complete.is_set():
                missing = self._publishers - layouts.publishers.keys()
                raise TimeoutError(
                    f"timed out after {timeout_s} seconds "
                    f"waiting for publisher layouts for {key!r}; "
                    f"missing publishers: {sorted(missing)}"
                ) from error
            # A check of zero or less always lands here, so only a real wait warns.
            if timeout_s > 0:
                logger.warning(
                    "publisher layouts for %r completed only as their %s s wait "
                    "timed out",
                    key,
                    timeout_s,
                )
        finally:
            layouts.lookups -= 1
            # The last lookup to leave drops an entry that no layout reached.
            if not (layouts.lookups or layouts.publishers):
                del self._layouts[key]
        return layouts.publishers
