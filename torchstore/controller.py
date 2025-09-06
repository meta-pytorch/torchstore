import os
from dataclasses import dataclass, field
from enum import auto, Enum
from typing import Any, Dict, Optional, Set

from monarch.actor import Actor, endpoint

from torchstore.storage_volume import StorageVolume

from torchstore.strategy import TorchStoreStrategy
from torchstore.transport.pipe import Request, TensorSlice


#TODO: move this into request as a field
class ObjectType(Enum):
    OBJECT = auto()
    TENSOR = auto()
    TENSOR_SLICE = auto()

    @classmethod
    def from_request(cls, request: Request) -> "ObjectType":
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
        assert (
            self.object_type == other_storage_info.object_type
        ), "Particularly dangerous to change storage type of an existing key, are you sure? Raise an issue if so."

        self.tensor_slices.update(other_storage_info.tensor_slices)


class Controller(Actor):
    def __init__(
        self,
    ) -> None:
        self.keys_to_storage_volumes: Dict[str, Dict[str, StorageInfo]] = {}
        self.is_initialized: bool = False
        self.strategy: Optional[TorchStoreStrategy] = None
        self.storage_volumes: Optional[StorageVolume] = None
        self.num_storage_volumes: Optional[int] = None
        self.strategy: Optional[TorchStoreStrategy] = None

    def assert_initialized(self) -> None:
        assert (
            self.is_initialized
        ), "Please call torchstore.initialize before attempting to use store."

    @endpoint
    async def init(
        self,
        strategy: TorchStoreStrategy,
        num_storage_volumes: int,
        storage_volumes: StorageVolume,
    ) -> None:
        if self.is_initialized:
            raise RuntimeError("TorchStore is already initialized")

        self.strategy = strategy
        self.storage_volumes = storage_volumes
        self.num_storage_volumes = num_storage_volumes

        await self.strategy.set_storage_volumes(self.storage_volumes)
        self.is_initialized = True

    @endpoint
    def get_controller_strategy(self) -> TorchStoreStrategy:
        self.assert_initialized()
        assert self.strategy is not None, "Strategy is not set"
        return self.strategy

    @endpoint
    def locate_volumes(
        self,
        key: str,
    ) -> Dict[str, StorageInfo]:
        """Locate storage volumes containing shards of the specified key.

        Returns {<storage_volume_id> -> StorageInfo} where <storage_volume_id> 
        are IDs of storage volumes holding shards of the data.

        For example, if the data is a DTensor with 3 shards, the returned map will look like:
        storage_volume_map = {
            "<dtensor_fqn>": {
                "<storage_volume_id>": StorageInfo.tensor_slice=set([
                    "<tensor_slice>",
                    "<tensor_slice>",
                    "<tensor_slice>",
                ]),
                ...
            }
            ,
            ...
        }

        Args:
            key (str): The key to locate in storage volumes.
            
        Returns:
            Dict[str, StorageInfo]: Mapping from storage volume IDs to StorageInfo 
                objects containing metadata about the stored data shards.
                
        Raises:
            KeyError: If the key is not found in any storage volumes.
        """
        self.assert_initialized()

        if key not in self.keys_to_storage_volumes:
            raise KeyError(f"Unable to locate {key} in any storage volumes.")
        return self.keys_to_storage_volumes[key]

    @endpoint
    def notify_put(self, key: str, request: Request, storage_volume_id: str) -> None:
        """Notify the controller that data has been stored in a storage volume.
        
        This should called after a successful put operation to
        maintain the distributed storage index.
        
        Args:
            key (str): The unique identifier for the stored data.
            request (Request): The storage request containing metadata about the stored data.
            storage_volume_id (str): ID of the storage volume where the data was stored.
        """
        self.assert_initialized()

        if key not in self.keys_to_storage_volumes:
            self.keys_to_storage_volumes[key] = {}

        storage_info = StorageInfo(
            object_type=ObjectType.from_request(request),
            tensor_slices=set([request.tensor_slice]),
        )

        if storage_volume_id not in self.keys_to_storage_volumes[key]:
            self.keys_to_storage_volumes[key][storage_volume_id] = storage_info
        else:
            self.keys_to_storage_volumes[key][storage_volume_id].update(storage_info)

    @endpoint
    def teardown(self) -> None:
        self.is_initialized = False
        self.keys_to_storage_volumes = {}
        self.strategy = None
        self.storage_volumes = None
        self.num_storage_volumes = None
