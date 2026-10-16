# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The routing coordinator's rules, as a plain class that any host can run."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

# Annotations only: this module's own imports stay in the standard library.
if TYPE_CHECKING:
    from torchstore.strategy import TorchStoreStrategy
    from ._model import KeyRegistration, Registrations

__all__ = ["LayoutRegistry"]

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
        """Record the publishers and strategy.

        A later call replaces both and drops every layout. A lookup already waiting
        stays on the dropped layouts: it returns them if they completed before this
        call, and otherwise fails at its timeout.
        """
        self._publishers = frozenset(publishers)
        self._strategy = strategy
        self._layouts = {}

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
        layouts = self._layouts.get(key)
        if layouts is not None and rank in layouts.publishers:
            return
        # Copy before creating the entry, so a layout that fails to copy leaves none.
        registration = dict(registrations)
        layouts = self._layouts.setdefault(key, _Layouts())
        layouts.publishers[rank] = registration
        if layouts.publishers.keys() == self._publishers:
            layouts.complete.set()

    async def get_layouts(
        self,
        key: str,
        *,
        timeout_s: float | None = None,
    ) -> Registrations:
        """Wait for and return every publisher layout for ``key``.

        ``timeout_s=None`` waits ``_LAYOUT_REGISTRATION_TIMEOUT_S``, ``timeout_s <= 0``
        checks once and NaN raises ``ValueError``. A timeout names the missing
        publishers, or says that a later init dropped the layouts.
        """
        if self._publishers is None:
            raise RuntimeError("RoutingCoordinator.init has not run")
        if timeout_s is None:
            timeout_s = _LAYOUT_REGISTRATION_TIMEOUT_S
        # Refuse NaN outright, since the clamp below would make it a 0 s check.
        if math.isnan(timeout_s):
            raise ValueError("get_layouts timeout_s must not be NaN")
        timeout_s = timeout_s if timeout_s > 0 else 0
        layouts = self._layouts.setdefault(key, _Layouts())
        layouts.lookups += 1
        try:
            # Complete layouts return without starting a wait.
            if layouts.publishers.keys() == self._publishers:
                return layouts.publishers
            try:
                await asyncio.wait_for(layouts.complete.wait(), timeout=timeout_s)
            except asyncio.TimeoutError as error:
                # The last registration can land as the deadline fires. Layouts that
                # completed return even if a later init dropped them. Others never do.
                if layouts.complete.is_set():
                    return layouts.publishers
                if self._layouts.get(key) is not layouts:
                    raise TimeoutError(
                        f"timed out after {timeout_s} seconds waiting for publisher "
                        f"layouts for {key!r}, which a later init dropped"
                    ) from error
                missing = self._publishers - layouts.publishers.keys()
                raise TimeoutError(
                    f"timed out after {timeout_s} seconds "
                    f"waiting for publisher layouts for {key!r}; "
                    f"missing publishers: {sorted(missing)}"
                ) from error
            return layouts.publishers
        finally:
            layouts.lookups -= 1
            # The last lookup to leave drops an entry that no layout reached, unless a
            # later init already replaced it.
            current = self._layouts.get(key) is layouts
            if current and not (layouts.lookups or layouts.publishers):
                del self._layouts[key]
