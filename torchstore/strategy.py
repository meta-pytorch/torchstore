import os

from monarch.actor import current_rank

from torchstore.storage_volume import StorageVolume


class TorchStoreStrategy:

    def __init__(self):
        self.storage_volumes = None
        self.volume_id_to_coord = {}

    @classmethod
    def get_volume_id(cls):
        raise NotImplementedError(f"{cls.__name__} must implement 'get_volume_id'")

    @classmethod
    def get_client_id(cls):
        raise NotImplementedError(f"{cls.__name__} must implement 'get_client_id'")

    async def set_storage_volumes(self, storage_volumes):
        self.storage_volumes = storage_volumes
        self.volume_id_to_coord = {
            val: coord for coord, val in await self.storage_volumes.get_id.call()
        }

    async def set_storage_volumes(self, storage_volumes):
        self.storage_volumes = storage_volumes
        self.volume_id_to_coord = {
            val: coord for coord, val in await self.storage_volumes.get_id.call()
        }

    def select_storage_volume(self):
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
        volume_coord = self.volume_id_to_coord[volume_id]
        return self.storage_volumes.slice(**volume_coord)


class SingletonStrategy(TorchStoreStrategy):
    """There can be only one!"""

    strategy_id: str = "Singleton"

    @classmethod
    def get_volume_id(cls):
        return cls.strategy_id

    @classmethod
    def get_client_id(cls):
        return cls.strategy_id

    async def set_storage_volumes(self, storage_volumes):
        assert (
            len(storage_volumes) == 1
        ), f"{self.__class__.__name__} support only one storage volume"
        await super().set_storage_volumes(storage_volumes)


class LocalRankStrategy(TorchStoreStrategy):
    """Relies on 'LOCAL_RANK' set from env."""

    def __init__(
        self,
    ):
        self.storage_volumes = None
        self.volume_id_to_coord = {}

    @classmethod
    def get_volume_id(cls):
        return str(current_rank().rank)

    @classmethod
    def get_client_id(cls):
        return os.environ["LOCAL_RANK"]

    def select_storage_volume(self):
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
        volume_coord = self.volume_id_to_coord[volume_id]
        return self.storage_volumes.slice(**volume_coord)
