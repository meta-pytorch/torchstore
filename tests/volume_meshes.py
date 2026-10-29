# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A Monarch-free stand-in for a storage volume mesh, and put and get through a volume."""

from __future__ import annotations

import torch
from torchstore.strategy import StorageVolumeRef
from torchstore.transport import create_transport_buffer
from torchstore.transport.types import Request


async def put_through(ref: StorageVolumeRef, key: str, value: torch.Tensor) -> None:
    """Put ``value`` under ``key`` in the volume ``ref`` names, over its transport."""
    buffer = create_transport_buffer(ref)
    await buffer.put_to_storage_volume([Request.from_tensor(key, value)])


async def get_through(ref: StorageVolumeRef, key: str) -> torch.Tensor:
    """Get ``key`` from the volume ``ref`` names, over its transport."""
    buffer = create_transport_buffer(ref)
    [value] = await buffer.get_from_storage_volume([Request(key=key)])
    return value


class _GetId:
    """A mesh's ``get_id``, answering for one volume per rank as Monarch's ``call`` does."""

    def __init__(
        self, volumes: list[tuple[str, str]], coords: list[dict[str, int]]
    ) -> None:
        self._volumes = volumes
        self._coords = coords

    async def call(self) -> list[tuple[dict[str, int], tuple[str, str]]]:
        return [
            (dict(coord), volume)
            for coord, volume in zip(self._coords, self._volumes, strict=True)
        ]


class VolumeMesh:
    """A stand-in volume mesh that pickles with its strategy. Slices name their coordinates.

    Volume ``i`` sits at ``coords[i]``, by default ``{"procs": i}``.
    """

    def __init__(
        self,
        volumes: list[tuple[str, str]],
        coords: list[dict[str, int]] | None = None,
    ) -> None:
        if coords is None:
            coords = [{"procs": rank} for rank in range(len(volumes))]
        self.get_id = _GetId(volumes, coords)

    def slice(self, **coord: int) -> tuple[str, dict[str, int]]:
        return ("slice", coord)
