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
from collections.abc import Collection
from typing import Any

__all__ = ["VolumeHost"]


class VolumeHost(ABC):
    """Resolves storage volume IDs for a strategy, in place of its Monarch volume mesh.

    ``volume_ids`` lists exactly the IDs ``endpoints`` and ``hostname`` resolve. Clients look
    volumes up inline, so every lookup is synchronous, and a host installed on a strategy
    travels with it, so it pickles.
    """

    @abstractmethod
    def volume_ids(self) -> Collection[str]:
        """Return the ID of every volume this host resolves."""
        raise NotImplementedError

    @abstractmethod
    def endpoints(self, volume_id: str) -> Any:
        """Return a handle to volume ``volume_id``, used as one volume's slice of a Monarch mesh.

        Clients call its ``put``, ``delete`` and ``delete_batch`` with ``call`` and its ``get``,
        ``handshake`` and ``get_meta`` with ``call_one``.

        Raises:
            KeyError: If this host has no volume ``volume_id``.
        """
        raise NotImplementedError

    @abstractmethod
    def hostname(self, volume_id: str) -> str:
        """Return the host name volume ``volume_id`` reports, for locality checks.

        Raises:
            KeyError: If this host has no volume ``volume_id``.
        """
        raise NotImplementedError
