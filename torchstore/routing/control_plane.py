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
from typing import ClassVar, TYPE_CHECKING

from ._model import KeyRegistration, Registrations

if TYPE_CHECKING:
    from torchstore.strategy import TorchStoreStrategy

__all__ = ["ControlPlane", "ControlPlaneAddress"]


class ControlPlane(ABC):
    """Rank coordination that happens before any tensor moves.

    Errors are not normalized (Monarch raises its own, such as ``ActorError``), so
    callers match them by the message each method names. Calls may come from any thread
    or event loop, so hold no loop-bound state.
    """

    # Names the plane in TorchStore's log lines, without whitespace. None means its class.
    kind: ClassVar[str | None] = None

    @abstractmethod
    async def init(
        self,
        publishers: Collection[str],
        strategy: TorchStoreStrategy,
    ) -> None:
        """Fix the publisher set and the volume-resolution strategy.

        ``ts.initialize`` calls it on the driver's instance on every run. A second call
        starts the store over, as Monarch's coordinator does, or raises. Until it has run,
        the other methods raise on every instance of the store, with ``init has not run``.
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

    async def close(self) -> None:
        """Release what the plane holds, by default nothing.

        TorchStore calls it only on a plane that a ``ControlPlaneAddress`` connected, maybe
        while clients from other threads or loops still call it. Fail those calls, not hang.
        """


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

        TorchStore calls its ``close`` once a shutdown or a new selection drops its record,
        or the ``ts.client`` call that dialed it fails or finds the address recorded. The
        close is a task on the caller's event loop, cancelled if that loop ends.
        """
        raise NotImplementedError

    @abstractmethod
    def __str__(self) -> str:
        """Name the address for log lines without whitespace, such as ``host:port``."""
        raise NotImplementedError
