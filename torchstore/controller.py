from typing import TYPE_CHECKING

from monarch.actor import Actor, endpoint

from torchstore.transport.pipe import Request

if TYPE_CHECKING:
    from torchstore import LocalClient

class Controller(Actor):
    def __init__(self, placement_strategy) -> None:
        self.storage_volumes = None
        self.keys_to_storage_volumes = {}
        self.host_id_to_storage_volume = {}

    @endpoint
    def set_storage_volumes(self, storage_volumes):
        self.storage_volumes = storage_volumes

    @endpoint
    def select_storage_volume_for_put(
        self,
        key: str,
        request: Request,
        client_id: str
    ) -> str:
        return self.client_id_to_storage_volume[client_id]

    def select_storage_volume_for_get(
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
