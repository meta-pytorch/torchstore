# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Storage-location interface shared by TorchStore clients."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torchstore.controller import StorageInfo


class VolumeDirectory(ABC):
    """Resolves storage keys to the volumes that hold them."""

    @abstractmethod
    async def locate_volumes(
        self,
        keys: list[str],
        missing_ok: bool = False,
        require_fully_committed: bool = True,
    ) -> dict[str, dict[str, StorageInfo]]:
        """Return storage metadata grouped by key and volume."""
        raise NotImplementedError
