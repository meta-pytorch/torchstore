# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A Monarch-free stand-in for a storage volume mesh."""

from __future__ import annotations


class _GetId:
    """A mesh's ``get_id``, answering for one volume per rank as Monarch's ``call`` does."""

    def __init__(
        self, volumes: list[tuple[str, str]], coords: list[dict[str, int]]
    ) -> None:
        self._volumes = volumes
        self._coords = coords

    async def call(self) -> list[tuple[dict[str, int], tuple[str, str]]]:
        return [
            (dict(coord), volume) for coord, volume in zip(self._coords, self._volumes)
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
