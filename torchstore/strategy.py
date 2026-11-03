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
from collections.abc import Collection, Iterable
from typing import TYPE_CHECKING

from torchstore.backend import get_backend
from torchstore.transport import TransportType
from torchstore.transport.buffers import TransportContext
from torchstore.utils import get_local_hostname
from torchstore.volume_host import VolumeHost

if TYPE_CHECKING:
    from torchstore.monarch.storage_volume import StorageVolume

logger = logging.getLogger(__name__)

# The instance-dict key of an installed volume host. No subclass attribute shares it since no
# attribute syntax spells it.
_VOLUME_HOST = "torchstore.volume_host"
# Methods that read or drop an installed host, the attribute lookup that can stand in for any of
# them and hooks that decide what a copy keeps. A class or instance overriding one or a patch of
# the base class's own may skip the host, which TorchStore cannot check.
_HOST_METHODS = (
    "set_storage_volumes",
    "select_storage_volume",
    "get_storage_volume",
    "_prepare_volumes",
    "__getattribute__",
    "__getstate__",
    "__setstate__",
    "__reduce__",
    "__reduce_ex__",
    "__copy__",
    "__deepcopy__",
)


def _installed_volume_host(strategy: object) -> "VolumeHost | None":
    """Return the volume host ``set_volume_host`` installed on ``strategy``, else None."""
    state = getattr(strategy, "__dict__", None)
    return None if state is None else state.get(_VOLUME_HOST)


class StorageVolumeRef:
    __slots__ = (
        "volume",
        "volume_id",
        "transport_context",
        "default_transport_type",
        "volume_hostname",
        "transports",
    )

    def __init__(
        self,
        volume: "StorageVolume",
        volume_id: str,
        transport_context: TransportContext,
        default_transport_type: TransportType,
        volume_hostname: str | None = None,
        *,
        transports: Collection[TransportType] | None = None,
    ):
        self.volume = volume
        self.volume_id = volume_id
        # useful for caching elements that should survive the lifetime of the client/volume
        self.transport_context = transport_context
        self.default_transport_type = default_transport_type
        self.volume_hostname = volume_hostname
        # The transports that reach the volume, or None where any may. Automatic selection
        # and the transport log run only their probes.
        self.transports = transports


def _host_ref(
    strategy: "TorchStoreStrategy", host: VolumeHost, volume_id: str
) -> StorageVolumeRef:
    """Resolve ``volume_id`` through ``host`` with the strategy's transport settings and the host's transport."""
    ref = StorageVolumeRef(
        host.endpoints(volume_id),
        volume_id,
        strategy.transport_context,
        strategy.default_transport_type,
        volume_hostname=host.hostname(volume_id),
        transports=host.transports,
    )
    ref.default_transport_type = host.resolve_transport(ref)
    return ref


def _mesh_ref(strategy: "TorchStoreStrategy", volume_id: str) -> StorageVolumeRef:
    """Resolve ``volume_id`` through the strategy's mesh tables."""
    volume_coord = strategy.volume_id_to_coord[volume_id]

    return StorageVolumeRef(
        strategy.storage_volumes.slice(**volume_coord),
        volume_id,
        strategy.transport_context,
        strategy.default_transport_type,
        volume_hostname=strategy.volume_id_to_hostname.get(volume_id),
    )


def _overrides(strategy: "TorchStoreStrategy", name: str) -> bool:
    """Whether the strategy's instance or class has a ``name`` other than the base class's at import."""
    return (
        name in vars(strategy)
        or getattr(type(strategy), name, None) is not _BASE_METHODS[name]
    )


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

        Lookups then resolve these volumes, not a host installed before or while this runs. That host
        is left unclosed.

        Args:
            storage_volumes: Actor mesh of storage volume actors.
        """
        vars(self).pop(_VOLUME_HOST, None)
        self.storage_volumes = storage_volumes
        self.volume_id_to_coord = {}
        self.volume_id_to_hostname = {}
        try:
            volumes = await self.storage_volumes.get_id.call()
            for coord, (volume_id, hostname) in volumes:
                self.volume_id_to_coord[volume_id] = coord
                self.volume_id_to_hostname[volume_id] = hostname
        finally:
            vars(self).pop(_VOLUME_HOST, None)

    def set_volume_host(self, host: VolumeHost) -> None:
        """Resolve every volume ID through ``host`` until a ``set_storage_volumes`` call starts or ends.

        Internal and subject to change, for TorchStore's own volume hosts. Because transports cache
        their connection to a volume by its ID (Gloo for the whole process), ``host`` must resolve
        each ID to the volume that ID names everywhere else in this process. A host this replaces
        is left unclosed. Close one that holds connections first.

        Raises:
            TypeError: If ``host`` is no ``VolumeHost``, or this strategy's instance, its class or a
                patch of the base class replaces ``__getattribute__``, a method that reads or drops
                the host or a hook that decides what a copy keeps.
        """
        if not isinstance(host, VolumeHost):
            raise TypeError(f"expected a VolumeHost, got {type(host).__name__}")
        overridden = [name for name in _HOST_METHODS if _overrides(self, name)]
        if overridden:
            raise TypeError(
                f"this {type(self).__name__} overrides {', '.join(overridden)}, and TorchStore"
                " cannot tell whether that honors a volume host"
            )
        vars(self)[_VOLUME_HOST] = host

    async def _prepare_volumes(self, volume_ids: Iterable[str], /) -> None:
        """Let the host load what looking up ``volume_ids`` needs. Without one, return.

        Internal and subject to change, for TorchStore's own volume hosts.

        Raises:
            TypeError: If ``volume_ids`` is one str rather than an iterable of IDs.
            ValueError: If this strategy names a transport outside the host's ``transports``.
        """
        if isinstance(volume_ids, str):
            raise TypeError(
                f"need an iterable of volume IDs, not the str {volume_ids!r}"
            )
        host = vars(self).get(_VOLUME_HOST)
        if host is None:
            return
        transports = host.transports
        requested = self.default_transport_type
        if not (
            requested is TransportType.Unset
            or transports is None
            or requested in transports
        ):
            names = " or ".join(sorted(transport.name for transport in transports))
            raise ValueError(
                f"{type(host).__name__} reaches its volumes only through {names},"
                f" not {requested.name}"
            )
        await host.prepare(volume_ids)

    def select_storage_volume(self) -> StorageVolumeRef:
        """Select the storage volume for the current client process.

        One read of the installed host decides the path. A host installed meanwhile never resolves
        the client's ID.

        Returns:
            StorageVolumeRef: Reference to the storage volume for this client.
        """
        # client_id == volume_id for this strategy
        client_id = self.get_client_id()
        host = vars(self).get(_VOLUME_HOST)
        if host is not None:
            if client_id not in host.volume_ids():
                raise KeyError(
                    f"No corresponding storage volume found for {client_id} in the volume host"
                )
            return _host_ref(self, host, client_id)
        if client_id not in self.volume_id_to_coord:
            raise KeyError(
                f"No corresponding storage volume found for {client_id} {self.volume_id_to_coord=}"
            )
        lookup = self.get_storage_volume
        if getattr(lookup, "__func__", None) is not _BASE_METHODS["get_storage_volume"]:
            # As at main, a lookup that a subclass, a patch or __getattribute__ supplies resolves the ID.
            return lookup(client_id)
        return _mesh_ref(self, client_id)

    def get_storage_volume(self, volume_id: str) -> StorageVolumeRef:
        """Return the reference to the storage volume with a given ID.

        A volume host, once installed, resolves the ID instead of the attributes.

        Args:
            volume_id (str): The volume ID to look up.

        Returns:
            StorageVolumeRef: Reference whose ``volume`` is the volume's slice of the mesh, or
            the installed host's handle for it with that host's transport.

        Raises:
            ValueError: If the volume's host refuses this strategy's transport.
            RuntimeError: If the volume's host is a closed carrier volume host.
        """
        host = vars(self).get(_VOLUME_HOST)
        if host is not None:
            return _host_ref(self, host, volume_id)
        return _mesh_ref(self, volume_id)


# The base class's own host methods as imported so that a later patch of one counts as an override.
_BASE_METHODS = {
    name: getattr(TorchStoreStrategy, name, None) for name in _HOST_METHODS
}


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
