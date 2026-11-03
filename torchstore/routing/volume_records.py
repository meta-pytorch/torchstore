# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Interface through which routing ranks share their storage volumes' records."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection

__all__ = ["VolumeRecords"]


class VolumeRecords(ABC):
    """Shares one routing store's volume records: opaque non-empty bytes with at most one per publisher rank.

    On an open handle in the process that made it, a non-bytes record raises ``TypeError`` and an empty record, an id
    UTF-8 cannot encode or an oversized id or record ``ValueError`` before any store call. Calls that reach the store
    raise ``RuntimeError`` naming "init has not run" until the implementation's own init. Any thread or loop may call.
    """

    @abstractmethod
    async def register_volume(self, volume_id: str, record: bytes, /) -> None:
        """Record a publisher's volume. The first record stored wins.

        Another id raises ``KeyError`` naming "not a publisher rank" and a record that is not bytes ``TypeError``.
        The same record again returns, and another raises ``RuntimeError`` naming "already registered".
        """
        raise NotImplementedError

    @abstractmethod
    async def volume_records(self, volume_ids: Collection[str], /) -> dict[str, bytes]:
        """Return the record of each id in ``volume_ids``, without waiting for any.

        A KeyError names the ids that have no record, 16 at most.
        """
        raise NotImplementedError
