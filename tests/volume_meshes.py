# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A Monarch-free stand-in for a storage volume mesh."""

from __future__ import annotations


class _GetId:
    """A mesh's ``get_id``, answering for one volume per rank as Monarch's ``call`` does."""

    def __init__(self, volumes: list[tuple[str, str]]) -> None:
        self._volumes = volumes

    async def call(self) -> list[tuple[dict[str, int], tuple[str, str]]]:
        return [({"procs": rank}, volume) for rank, volume in enumerate(self._volumes)]


class VolumeMesh:
    """A stand-in volume mesh that pickles with its strategy. Slices name their coordinates."""

    def __init__(self, volumes: list[tuple[str, str]]) -> None:
        self.get_id = _GetId(volumes)

    def slice(self, **coord: int) -> tuple[str, dict[str, int]]:
        return ("slice", coord)
