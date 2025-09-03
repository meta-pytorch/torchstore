import os
from typing import TYPE_CHECKING
import asyncio

from monarch.actor import Actor, endpoint, current_rank

from torchstore.storage_volume import StorageVolume
from torchstore.transport.pipe import Request

if TYPE_CHECKING:
    from torchstore import LocalClient

class TorchStoreStrategy:
    async def set_storage_volumes(self, storage_volumes):
        raise NotImplementedError()

    @classmethod
    def get_volume_id(cls):
        raise NotImplementedError()

    @classmethod
    def get_client_id(cls):
        raise NotImplementedError()

    def select_storage_volume(self):
        raise NotImplementedError()

#TODO:
# class SingletonStrategy(TorchStoreStrategy):
#     pass
# class LocalHostStrategy(TorchStoreStrategy):
#     pass

class LocalRankStrategy(TorchStoreStrategy):
    """Relies on 'LOCAL_RANK' set from env.
    """

    def __init__(
        self,
    ):
        self.storage_volumes = None
        self.volume_id_to_coord = {}

    async def set_storage_volumes(self, storage_volumes):
        self.storage_volumes = storage_volumes
        self.volume_id_to_coord = {
            val : coord for coord, val in await self.storage_volumes.get_id.call()
        }        

    @classmethod
    def get_volume_id(cls):
        return str(current_rank().rank)

    @classmethod
    def get_client_id(cls):
        return os.environ["LOCAL_RANK"]

    def select_storage_volume(self):
        client_id = self.get_client_id()
        if client_id not in self.volume_id_to_coord:
            raise KeyError(f"No corresponding storage volume found for {client_id} {self.volume_id_to_coord=}")
            
        volume_coord = self.volume_id_to_coord[client_id]
        return self.storage_volumes.slice(**volume_coord)

    def _hash_coord(self, coord):
        return str(coord)
class Controller(Actor):
    def __init__(
        self,
        strategy: TorchStoreStrategy,
        num_storage_volumes: int
    ):
        self.keys_to_storage_volumes = {}
        self.strategy = strategy
        self.num_storage_volumes = num_storage_volumes
        self.is_initialized = False

    def assert_initialized(self):
        assert self.is_initialized, (
            "Please call torchstore.initialize_store before attempting to use store."
        ) 

    @endpoint
    async def init(self):
        if self.is_initialized:
            raise RuntimeError("TorchStore is already initialized")

        self.storage_volumes = await StorageVolume.spawn(
            num_volumes=self.num_storage_volumes,
            id_func=self.strategy.get_volume_id
        )
        
        await self.strategy.set_storage_volumes(self.storage_volumes)
        self.is_initialized = True

    @endpoint
    def get_controller_strategy(self):
        self.assert_initialized()
        return self.strategy
    
    @endpoint
    def locate_request_volumes(
        self,
        key: str,
        request: Request,
        local_client: "LocalClient"
    ):
        # to start with, something like storage_volume_map = {
        # storage_volume_map = {
        #     "<dtensor_fqn>": {
        #         "<storage_volume_id>": set([
        #             "<tensor_slice>",
        #             "<tensor_slice>",
        #             "<tensor_slice>",
        #         ]),
        #         ...
        #     }
        #     ,
        #     ...
        # }

        # dtensor fqn goes here
        storage_volume_map = self.keys_to_storage_volumes[key]        
        storage_volumes_that_have_the_data = {}
        # filter for storage volumes that have the data
        for storage_volume_id, tensor_slice_set in storage_volume_map.items():
            tensors_overlap = False # TODO: request.tensor_slice
            if tensors_overlap:
                storage_volumes_that_have_the_data[storage_volume_id] = tensor_slice_set            

        # we return something that looks like
        # storage_volumes_that_have_the_data = {
        #     "<storage_volume_id>": set([
        #         "<tensor_slice>",
        #         "<tensor_slice>",
        #         "<tensor_slice>",
        #     ]),
        #     ...
        # }
        return storage_volumes_that_have_the_data 

    @endpoint
    def notify_put(self,
        key: str,
        request: Request,
        storage_volume_id: str
    ):

        if key not in self.keys_to_storage_volumes[key]:
            self.keys_to_storage_volumes[key] = {}

        if storage_volume_id not in self.keys_to_storage_volumes[key]:
            self.keys_to_storage_volumes[key][storage_volume_id] = set()

        self.keys_to_storage_volumes[key][storage_volume_id].add(request.tensor_slice)
