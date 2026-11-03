# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Volume lookup: from a storage volume's ID to its endpoints and host name.

Internal to TorchStore and subject to change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection, Iterable
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from torchstore.strategy import StorageVolumeRef
    from torchstore.transport import TransportType

__all__ = ["VolumeHost"]


class VolumeHost(ABC):
    """Resolves storage volume IDs for a strategy, in place of its Monarch volume mesh.

    ``volume_ids`` lists exactly the IDs ``endpoints`` and ``hostname`` resolve. Lookups are synchronous since clients
    look volumes up inline. A strategy pickles only if its host does, and a carrier volume host never does. As
    transports cache connections by ID, each ID must name one volume in every host and mesh of a process.
    """

    @abstractmethod
    def volume_ids(self) -> Collection[str]:
        """Return the ID of every volume this host resolves (none once it is closed)."""
        raise NotImplementedError

    @abstractmethod
    def endpoints(self, volume_id: str) -> Any:
        """Return a handle to volume ``volume_id``, used as one volume's slice of a Monarch mesh.

        Clients call its ``put``, ``delete`` and ``delete_batch`` with ``call`` and its ``get``,
        ``handshake`` and ``get_meta`` with ``call_one``.

        Raises:
            KeyError: If this host has no volume ``volume_id``.
            RuntimeError: If this host is closed.
        """
        raise NotImplementedError

    @abstractmethod
    def hostname(self, volume_id: str) -> str:
        """Return the host name volume ``volume_id`` reports, for locality checks.

        Raises:
            KeyError: If this host has no volume ``volume_id``.
            RuntimeError: If this host is closed.
        """
        raise NotImplementedError

    @property
    def transports(self) -> Collection[TransportType] | None:
        """The transports that reach this host's volumes, or None where any may. By default, None.

        A strategy's references name them so automatic selection and the transport log run no other probe (selection
        still ends at MonarchRPC when none passes). Each routed get refuses a strategy naming another transport.
        """
        return None

    async def prepare(self, volume_ids: Iterable[str], /) -> None:
        """Load what lookups of ``volume_ids`` need, before the first. By default, nothing.

        Routing calls it on every get of a key from any thread or loop and concurrently once ``transports`` admits the
        strategy's transport. It must be idempotent, thread-safe, and cheap once loaded. Errors reach the get unwrapped.
        """

    def resolve_transport(self, ref: StorageVolumeRef, /) -> TransportType:
        """Return the transport that reaches ``ref``'s volume. By default, the strategy's.

        It runs on every lookup so it must stay cheap, and ``ref`` names this host's ``transports``. ``Unset`` defers
        to automatic selection, and a strategy's transport outside ``transports`` must raise ValueError.
        """
        return ref.default_transport_type

    async def close(self) -> None:
        """Release what this host holds. By default it does nothing and the host keeps resolving.

        A host that closes lists no volumes and raises RuntimeError from ``endpoints``, ``hostname``, and ``prepare``.
        """
