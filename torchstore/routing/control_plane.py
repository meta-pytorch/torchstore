# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Control-plane interface that routing uses to share publishers and layouts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection, Mapping
from typing import TYPE_CHECKING

from ._model import KeyRegistration, Registrations

if TYPE_CHECKING:
    from torchstore.strategy import TorchStoreStrategy

__all__ = ["ControlPlane"]


class ControlPlane(ABC):
    """Rank coordination that happens before any tensor moves.

    Errors are not normalized (Monarch raises its own, such as ``ActorError``), so
    callers match them by the message each method names. Calls may come from any thread
    or event loop, so hold no loop-bound state.
    """

    @abstractmethod
    async def init(
        self,
        publishers: Collection[str],
        strategy: TorchStoreStrategy,
    ) -> None:
        """Fix the publisher set and the volume-resolution strategy.

        Call at most once per routing store, on one instance. Until that call has run,
        the other methods raise on every instance of the store, with ``init has not run``
        in the message.
        """
        raise NotImplementedError

    @abstractmethod
    async def register_layout(
        self,
        rank: str,
        key: str,
        registrations: Mapping[str, KeyRegistration],
    ) -> None:
        """Record one publisher's layout. Idempotent per rank, never blocks.

        A rank's first layout per key wins. Ranks outside the publisher set raise, with
        ``not a publisher rank`` in the message.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_layouts(self, key: str) -> Registrations:
        """Every publisher layout for ``key``. Blocks until the set is complete.

        Routed reads pass no timeout, so the plane's own deadline (300 s on TorchStore's
        planes) ends a wait, raising with ``missing publishers: [...]`` in the message.
        """
        raise NotImplementedError

    @abstractmethod
    async def strategy(self) -> TorchStoreStrategy:
        """Return a strategy that resolves the registered publishers' volumes."""
        raise NotImplementedError
