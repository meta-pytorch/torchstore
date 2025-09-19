# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TorchStore sharding strategies for distributing tensors across storage volumes.

This module defines strategies for determining how storage is distributed across
multiple storage volumes. Strategies map client processes to storage volumes.
"""

import os

from monarch.actor import current_rank

from torchstore.storage_volume import StorageVolume


class TorchStoreStrategy:
    """Base class for TorchStore distribution strategies.

    A strategy defines how tensors are distributed across storage volumes by:
    1. Assigning unique volume IDs to storage volumes
    2. Mapping client processes to storage volumes
    3. Providing access to the appropriate storage volume for operations

    Subclasses must implement get_volume_id() and get_client_id() methods.
    """

    def __init__(self):
        self.storage_volumes = None
        self.volume_id_to_coord = {}

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

        Args:
            storage_volumes: Actor mesh of storage volume actors.
        """
        self.storage_volumes = storage_volumes
        self.volume_id_to_coord = {
            val: coord for coord, val in await self.storage_volumes.get_id.call()
        }

    def select_storage_volume(self):
        """Select the storage volume for the current client process.

        Returns:
            tuple: (StorageVolume actor, volume_id) for this client.
        """
        client_id = self.get_client_id()
        if client_id not in self.volume_id_to_coord:
            raise KeyError(
                f"No corresponding storage volume found for {client_id} {self.volume_id_to_coord=}"
            )

        return (
            self.get_storage_volume(client_id),
            client_id,
        )  # client_id == volume_id for this strategy

    def get_storage_volume(self, volume_id: str) -> StorageVolume:
        """Retrieves storage volume actor for a given volume ID.

        Args:
            volume_id (str): The volume ID to look up.

        Returns:
            StorageVolume: The storage volume actor for the given ID.
        """
        volume_coord = self.volume_id_to_coord[volume_id]
        return self.storage_volumes.slice(**volume_coord)


class SingletonStrategy(TorchStoreStrategy):
    """There can be only one. Likely to OOM if used unwisely.

    Used when only one storage volume is needed. All operations are routed
    to the single volume. This is the default strategy for simple setups.
    """

    strategy_id: str = "Singleton"

    @classmethod
    def get_volume_id(cls):
        """Return the singleton volume ID.

        Returns:
            str: Always returns "Singleton" for the single volume.
        """
        return cls.strategy_id

    @classmethod
    def get_client_id(cls):
        return cls.strategy_id

    async def set_storage_volumes(self, storage_volumes):
        assert (
            len(storage_volumes) == 1
        ), f"{self.__class__.__name__} support only one storage volume"
        await super().set_storage_volumes(storage_volumes)


class HostStrategy(TorchStoreStrategy):
    """Assumes one storage volume per host.

    Each process uses 'HOSTNAME' to determine which storage volume to connect to.
    """

    @classmethod
    def get_volume_id(cls):
        # Note: this should only called at spawn, which makes this safe.
        return os.environ["HOSTNAME"]

    @classmethod
    def get_client_id(cls):
        return os.environ["HOSTNAME"]


class LocalRankStrategy(TorchStoreStrategy):
    """Strategy that maps storage volumes based on LOCAL_RANK environment variable.

    Each process uses its LOCAL_RANK to determine which storage volume to connect to.
    This strategy requires the LOCAL_RANK environment variable to be set and assumes
    one storage volume per local rank.
    """

    @classmethod
    def get_volume_id(cls):
        # Note: this should only called at spawn, which makes this safe.
        return str(current_rank().rank)

    @classmethod
    def get_client_id(cls):
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

    def select_storage_volume(self):
        """Select the storage volume for the current client process.

        Returns:
            tuple: (StorageVolume actor, volume_id) for this client.
        """
        client_id = self.get_client_id()
        if client_id not in self.volume_id_to_coord:
            raise KeyError(
                f"No corresponding storage volume found for {client_id} {self.volume_id_to_coord=}"
            )

        return (
            self.get_storage_volume(client_id),
            client_id,
        )  # client_id == volume_id for this strategy

    def get_storage_volume(self, volume_id: str) -> StorageVolume:
        return self.storage_volumes
