import os
from enum import Enum, auto
from typing import Optional, Set, Dict
from dataclasses import dataclass, field

from monarch.actor import Actor, endpoint

from torchstore.strategy import TorchStoreStrategy
from torchstore.storage_volume import StorageVolume
from torchstore.transport.pipe import Request, TensorSlice

#TODO: actually just move this into request as a field
class ObjectType(Enum):
    OBJECT = auto()
    TENSOR = auto()
    TENSOR_SLICE = auto()

    @classmethod
    def from_request(cls, request: Request):
        if request.is_object:
            return cls.OBJECT
        elif request.tensor_slice is not None:
            return cls.TENSOR_SLICE
        else:
            return cls.TENSOR


@dataclass
class StorageInfo:
    object_type: ObjectType 
    tensor_slices: Set[Optional[TensorSlice]] = field(default_factory=set)    

    def update(self, other_storage_info: "StorageInfo"):
        assert self.object_type == other_storage_info.object_type, (
            "Particularly dangerous to change storage type of an existing key, are you sure? Raise an issue if so."
        ) 

        self.tensor_slices.update(other_storage_info.tensor_slices)


class Controller(Actor):
    def __init__(
        self,
    ):
        self.keys_to_storage_volumes = {}        
        self.is_initialized = False

    def assert_initialized(self):
        assert self.is_initialized, (
            "Please call torchstore.initialize before attempting to use store."
        ) 

    @endpoint
    async def init(
        self,
        strategy: TorchStoreStrategy,
        num_storage_volumes: int,
        storage_volumes: StorageVolume
    ):
        if self.is_initialized:
            raise RuntimeError("TorchStore is already initialized")

        self.strategy = strategy
        self.storage_volumes = storage_volumes
        self.num_storage_volumes = num_storage_volumes
        
        await self.strategy.set_storage_volumes(self.storage_volumes)
        self.is_initialized = True

    @endpoint
    def get_controller_strategy(self):
        self.assert_initialized()
        return self.strategy
    
    @endpoint
    def locate_volumes(
        self,
        key: str,
        request: Request,
    ) -> Dict[str, StorageInfo]:
        self.assert_initialized()

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

        if key not in self.keys_to_storage_volumes:
            raise KeyError(f"Unable to locate {key} in any storage volumes.")
        return self.keys_to_storage_volumes[key]        

    @endpoint
    def notify_put(self,
        key: str,
        request: Request,
        storage_volume_id: str
    ):
        self.assert_initialized()

        if key not in self.keys_to_storage_volumes:
            self.keys_to_storage_volumes[key] = {}

        storage_info = StorageInfo(
            object_type =ObjectType.from_request(request),
            tensor_slices=set([request.tensor_slice])
        )

        if storage_volume_id not in self.keys_to_storage_volumes[key]:
            self.keys_to_storage_volumes[key][storage_volume_id] = storage_info
        else:
            self.keys_to_storage_volumes[key][storage_volume_id].update(storage_info)

    @endpoint
    def teardown(self):
        self.is_initialized = False
        self.keys_to_storage_volumes = {}
        self.strategy = None
        self.storage_volumes = None
        self.num_storage_volumes = None
