# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TorchStore sharding strategies for distributing tensors across storage volumes.

This module defines strategies for determining how storage is distributed across
multiple storage volumes. Strategies map client processes to storage volumes.
"""

import logging
import os
from typing import TYPE_CHECKING

from torchstore.backend import get_backend
from torchstore.transport import TransportType
from torchstore.transport.buffers import TransportContext
from torchstore.utils import get_local_hostname

if TYPE_CHECKING:
    from torchstore.monarch.storage_volume import StorageVolume
    from torchstore.volume_host import VolumeHost

logger = logging.getLogger(__name__)

# The instance-dict key of an installed volume host, mangled so that no subclass attribute shares it.
_VOLUME_HOST = "_TorchStoreStrategy__volume_host"
# The methods that read or drop an installed host, so a subclass overriding one would ignore it.
_HOST_METHODS = ("set_storage_volumes", "select_storage_volume", "get_storage_volume")


class StorageVolumeRef:
    __slots__ = (
        "volume",
        "volume_id",
        "transport_context",
        "default_transport_type",
        "volume_hostname",
    )

    def __init__(
        self,
        volume: "StorageVolume",
        volume_id: str,
        transport_context: TransportContext,
        default_transport_type: TransportType,
        volume_hostname: str | None = None,
    ):
        self.volume = volume
        self.volume_id = volume_id
        # useful for caching elements that should survive the lifetime of the client/volume
        self.transport_context = transport_context
        self.default_transport_type = default_transport_type
        self.volume_hostname = volume_hostname


class TorchStoreStrategy:
    """Base class for TorchStore distribution strategies.

    A strategy defines how tensors are distributed across storage volumes by:
    1. Assigning unique volume IDs to storage volumes
    2. Mapping client processes to storage volumes
    3. Providing access to the appropriate storage volume for operations

    Subclasses must implement get_volume_id() and get_client_id() methods.
    """

    def __init__(self, default_transport_type: TransportType = TransportType.Unset):
        self.default_transport_type = default_transport_type
        logger.info(f"Initializing TorchStoreStrategy with {default_transport_type=}")

        self.storage_volumes = None
        self.volume_id_to_coord = {}
        self.volume_id_to_hostname = {}
        self.transport_context = TransportContext()

    def __str__(self) -> str:
        storage_vol_len = (
            len(self.storage_volumes) if self.storage_volumes is not None else 0
        )
        return f"{self.__class__.__name__}(storage_volume_len={storage_vol_len})"

    @classmethod
    def get_volume_id(cls):
        """Get the unique ID for this process's storage volume. Called by volume on init.

        Returns:
            str: Unique identifier for the storage volume this process should use.
        """
        raise NotImplementedError(f"{cls.__name__} must implement 'get_volume_id'")

    @classmethod
    def get_client_id(cls):
        """Get the unique ID for this client process. Called by the client on each put.

        Returns:
            str: Unique identifier for this client process.
        """
        raise NotImplementedError(f"{cls.__name__} must implement 'get_client_id'")

    async def set_storage_volumes(self, storage_volumes):
        """Configure the storage volumes and build ID-to-coordinate mapping.

        Lookups then resolve these volumes, not a host installed before.

        Args:
            storage_volumes: Actor mesh of storage volume actors.
        """
        vars(self).pop(_VOLUME_HOST, None)
        self.storage_volumes = storage_volumes
        self.volume_id_to_coord = {}
        self.volume_id_to_hostname = {}
        for coord, (volume_id, hostname) in await self.storage_volumes.get_id.call():
            self.volume_id_to_coord[volume_id] = coord
            self.volume_id_to_hostname[volume_id] = hostname

    def set_volume_host(self, host: "VolumeHost") -> None:
        """Resolve every volume ID through ``host`` until ``set_storage_volumes`` runs again.

        Internal and subject to change, for TorchStore's own volume hosts.

        Raises:
            TypeError: If this strategy's class overrides a method that reads or drops the host.
        """
        overridden = [
            name
            for name in _HOST_METHODS
            if getattr(type(self), name) is not getattr(TorchStoreStrategy, name)
        ]
        if overridden:
            raise TypeError(
                f"{type(self).__name__} overrides {', '.join(overridden)}, which would"
                " ignore a volume host"
            )
        vars(self)[_VOLUME_HOST] = host

    def select_storage_volume(self) -> StorageVolumeRef:
        """Select the storage volume for the current client process.

        Returns:
            StorageVolumeRef: Reference to the storage volume for this client.
        """
        # client_id == volume_id for this strategy
        client_id = self.get_client_id()
        host = vars(self).get(_VOLUME_HOST)
        if host is None:
            if client_id not in self.volume_id_to_coord:
                raise KeyError(
                    f"No corresponding storage volume found for {client_id} {self.volume_id_to_coord=}"
                )
        elif client_id not in host.volume_ids():
            raise KeyError(
                f"No corresponding storage volume found for {client_id} in the volume host"
            )

        return self.get_storage_volume(client_id)

    def get_storage_volume(self, volume_id: str) -> StorageVolumeRef:
        """Return the reference to the storage volume with a given ID.

        A volume host, once installed, resolves the ID instead of the attributes.

        Args:
            volume_id (str): The volume ID to look up.

        Returns:
            StorageVolumeRef: Reference whose ``volume`` is the volume's slice of the mesh, or
            the installed host's handle for it.
        """
        host = vars(self).get(_VOLUME_HOST)
        if host is not None:
            return StorageVolumeRef(
                host.endpoints(volume_id),
                volume_id,
                self.transport_context,
                self.default_transport_type,
                volume_hostname=host.hostname(volume_id),
            )
        volume_coord = self.volume_id_to_coord[volume_id]

        return StorageVolumeRef(
            self.storage_volumes.slice(**volume_coord),
            volume_id,
            self.transport_context,
            self.default_transport_type,
            volume_hostname=self.volume_id_to_hostname.get(volume_id),
        )


class HostStrategy(TorchStoreStrategy):
    """Assumes one storage volume per host.

    Each process uses its hostname to determine which storage volume to connect to.
    The hostname is read from the TORCHSTORE_HOSTNAME environment variable if set,
    otherwise from ``socket.gethostname()``.
    """

    @classmethod
    def get_volume_id(cls):
        # Note: this should only called at spawn, which makes this safe.
        return get_local_hostname()

    @classmethod
    def get_client_id(cls):
        return get_local_hostname()


class LocalRankStrategy(TorchStoreStrategy):
    """Strategy that maps storage volumes based on rank environment variables.

    Storage volumes are identified by Monarch proc rank. Clients prefer the global
    ``RANK`` environment variable when it is available, so that every rank gets a
    distinct storage volume, and fall back to ``LOCAL_RANK`` for existing
    actor-based callers that do not set ``RANK``.

    When using this strategy, data is moved to a storage volume that is spawned on the
    mesh passed into ``ts.initialize(mesh=...)``. The storage volumes are distributed
    across the mesh, and each client process connects to its corresponding volume based
    on its resolved rank ID.
    """

    @classmethod
    def get_volume_id(cls):
        # Note: this should only called at spawn, which makes this safe.
        return str(get_backend("monarch").rank())

    @classmethod
    def get_client_id(cls):
        rank = os.environ.get("RANK")
        if rank is not None:
            return rank
        return os.environ["LOCAL_RANK"]


class ControllerStorageVolumes(TorchStoreStrategy):
    """Strategy that creates a singleton controller as the storage volume. This is a workaround
    for lack of support for actor -> actor comms in monarch when using remote allocations.
    """

    def __str__(self) -> str:
        storage_vol_len = (
            len(self.storage_volumes) if self.storage_volumes is not None else 0
        )
        return f"{self.__class__.__name__}(storage_volume_len={storage_vol_len})"

    @classmethod
    def get_volume_id(cls):
        return "0"

    @classmethod
    def get_client_id(cls):
        return "0"

    async def set_storage_volumes(self, storage_volumes):
        """Configure the storage volumes and build ID-to-coordinate mapping.

        Args:
            storage_volumes: Actor mesh of storage volume actors.
        """
        self.storage_volumes = storage_volumes
        self.volume_id_to_coord = {"0"}
        # For controller storage volumes, get hostname from the single volume
        self.volume_id_to_hostname = {}
        volume_id, hostname = await self.storage_volumes.get_id.call_one()
        self.volume_id_to_hostname[volume_id] = hostname

    def select_storage_volume(self) -> StorageVolumeRef:
        """Select the storage volume for the current client process.

        Returns:
            StorageVolumeRef: Reference to the storage volume for this client.
        """
        # client_id is hardcoded to a controller only volume
        client_id = self.get_client_id()
        if client_id not in self.volume_id_to_coord:
            raise KeyError(
                f"No corresponding storage volume found for {client_id} {self.volume_id_to_coord=}"
            )

        return self.get_storage_volume(client_id)

    def get_storage_volume(self, volume_id: str) -> StorageVolumeRef:
        return StorageVolumeRef(
            self.storage_volumes,
            volume_id,
            self.transport_context,
            self.default_transport_type,
            volume_hostname=self.volume_id_to_hostname.get(volume_id),
        )
