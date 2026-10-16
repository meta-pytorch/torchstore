# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Control-plane interface that routing uses to share publishers and layouts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ._model import KeyRegistration, Registrations

if TYPE_CHECKING:
    from torchstore.strategy import TorchStoreStrategy

__all__ = ["ControlPlane", "ControlPlaneAddress"]


class ControlPlane(ABC):
    """Rank coordination that happens before any tensor moves.

    Errors are not normalized (Monarch raises its own, such as ``ActorError``), so
    callers match them by message. Implementations that own resources add an async
    ``close``. Calls may come from any thread or event loop: hold no loop-bound state.
    """

    @abstractmethod
    async def init(
        self,
        publishers: Collection[str],
        strategy: TorchStoreStrategy,
    ) -> None:
        """Fix the publisher set and the volume-resolution strategy.

        Call at most once per routing store, on one instance. The other methods
        raise on every instance of the store until that call has run.
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

        A rank's first layout per key wins. Only publisher ranks register.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_layouts(self, key: str) -> Registrations:
        """Every publisher layout for ``key``. Blocks until the set is complete.

        Routed reads pass no timeout, so the plane's own deadline (300 s on TorchStore's
        planes) ends a wait, naming the missing publishers.
        """
        raise NotImplementedError

    @abstractmethod
    async def strategy(self) -> TorchStoreStrategy:
        """Return a strategy that resolves the registered publishers' volumes."""
        raise NotImplementedError


@dataclass(frozen=True)
class ControlPlaneAddress(ABC):
    """Where every rank connects to one routing store's control plane.

    The driver hands it to each rank, so subclasses are frozen dataclasses: they
    pickle and compare by value.
    """

    store_name: str

    @abstractmethod
    async def connect(self) -> ControlPlane:
        """Return a new control plane connected to this address on each call.

        TorchStore closes it, if it has ``close``, once a shutdown or a new selection drops
        its record, or the ``ts.client`` call that dialed it fails or finds the address
        recorded. The close is a task on the caller's event loop, cancelled if that loop ends.
        """
        raise NotImplementedError

    @abstractmethod
    def __str__(self) -> str:
        """Name the address for log lines without whitespace, such as ``host:port``."""
        raise NotImplementedError
