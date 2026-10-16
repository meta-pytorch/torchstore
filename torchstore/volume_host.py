# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Volume lookup: from a storage volume's ID to its endpoints and host name."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection, Mapping
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from torchstore.storage_volume import StorageVolume

__all__ = ["MonarchVolumeHost", "VolumeHost"]


class VolumeHost(ABC):
    """Resolves storage volume IDs to handles shaped like Monarch mesh slices.

    Each endpoint offers ``call`` and ``call_one``, as on a one-volume slice. Clients
    resolve volumes inline, so every lookup is synchronous and cheap. A host carried
    inside a pickled strategy, as the Monarch host is, must pickle too.
    """

    @abstractmethod
    def volume_ids(self) -> Collection[str]:
        """Return the ID of every volume this host resolves."""
        raise NotImplementedError

    @abstractmethod
    def endpoints(self, volume_id: str) -> Any:
        """Return a handle whose put, get, handshake, get_meta, delete and delete_batch reach ``volume_id``.

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


class MonarchVolumeHost(VolumeHost):
    """Resolves the volumes of one Monarch actor mesh to slices of that mesh.

    Its pickle holds the mesh, which pickles by reference, and two dicts, never a slice. Each
    copy keeps one slice per volume, since Monarch holds native state for every slice called.
    """

    def __init__(
        self,
        mesh: StorageVolume,
        coords: Mapping[str, Mapping[str, int]],
        hostnames: Mapping[str, str],
    ) -> None:
        if coords.keys() != hostnames.keys():
            raise ValueError(
                f"coords and hostnames name different volumes: {sorted(coords)} and "
                f"{sorted(hostnames)}"
            )
        self._mesh = mesh
        self._coords = {volume_id: dict(coord) for volume_id, coord in coords.items()}
        self._hostnames = dict(hostnames)
        self._slices: dict[str, StorageVolume] = {}

    def __getstate__(self) -> dict[str, Any]:
        return {**self.__dict__, "_slices": {}}

    @classmethod
    async def from_mesh(cls, mesh: StorageVolume) -> MonarchVolumeHost:
        """Ask every volume in ``mesh`` for its ID and host name."""
        coords: dict[str, Mapping[str, int]] = {}
        hostnames: dict[str, str] = {}
        for coord, (volume_id, hostname) in await mesh.get_id.call():
            coords[volume_id] = coord
            hostnames[volume_id] = hostname
        return cls(mesh, coords, hostnames)

    def volume_ids(self) -> Collection[str]:
        return self._coords.keys()

    def endpoints(self, volume_id: str) -> StorageVolume:
        # Reused, so once that volume's procs stop, a call through it can wait until
        # Monarch's supervision watchdog fires (2 minutes by default) instead of failing at once.
        endpoints = self._slices.get(volume_id)
        if endpoints is None:
            endpoints = self._mesh.slice(**self._coords[volume_id])
            endpoints = self._slices.setdefault(volume_id, endpoints)
        return endpoints

    def hostname(self, volume_id: str) -> str:
        return self._hostnames[volume_id]
