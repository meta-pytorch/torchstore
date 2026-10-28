# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Storage-location interface and metadata shared by TorchStore clients."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import auto, Enum

from torchstore.transport.types import Request, TensorSlice


# TODO: move this into request as a field
class ObjectType(Enum):
    OBJECT = auto()
    TENSOR = auto()
    TENSOR_SLICE = auto()

    @classmethod
    def from_request(cls, request: Request) -> ObjectType:
        if request.is_object:
            return cls.OBJECT
        elif request.tensor_slice is not None:
            return cls.TENSOR_SLICE
        else:
            return cls.TENSOR


@dataclass
class StorageInfo:
    object_type: ObjectType
    tensor_slices: set[TensorSlice | None] = field(default_factory=set)

    def update(self, other_storage_info: StorageInfo):
        assert (
            self.object_type == other_storage_info.object_type
        ), "Particularly dangerous to change storage type of an existing key, are you sure? Raise an issue if so."

        self.tensor_slices.update(other_storage_info.tensor_slices)


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
