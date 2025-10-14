# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from enum import auto, Enum
from itertools import product
from typing import Dict, List, Mapping, Optional, Set

from monarch.actor import Actor, endpoint

from torchstore.storage_utils.trie import Trie
from torchstore.storage_volume import StorageVolume
from torchstore.strategy import TorchStoreStrategy
from torchstore.transport.pipe import Request, TensorSlice


# TODO: move this into request as a field
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
        self.keys_to_storage_volumes = Trie()
        self.is_initialized: bool = False
        self.strategy: Optional[TorchStoreStrategy] = None
        self.storage_volumes: Optional[StorageVolume] = None
        self.num_storage_volumes: Optional[int] = None
        self.strategy: Optional[TorchStoreStrategy] = None

    def assert_initialized(self) -> None:
        assert (
            self.is_initialized
        ), "Please call torchstore.initialize before attempting to use store."

    def _is_dtensor_fully_committed(
        self, key: str, volume_map: Dict[str, StorageInfo]
    ) -> bool:
        """
        Check if all shards of a DTensor have been committed.

        For a DTensor to be fully committed, we need all coordinates in the mesh
        to have been stored. The mesh_shape tells us the total number of shards,
        and coordinates tell us which shards we have.

        Args:
            key (str): The key to check.
            volume_map (Dict[str, StorageInfo]): Mapping from storage volume IDs to StorageInfo.

        Returns:
            bool: True if fully committed, False if partial.
        """
        # Collect all tensor slices across all storage volumes
        all_slices = set()
        mesh_shape = None

        for storage_info in volume_map.values():
            if storage_info.object_type != ObjectType.TENSOR_SLICE:
                return True  # Not a DTensor, so it's "fully committed"

            for tensor_slice in storage_info.tensor_slices:
                all_slices.add(tensor_slice.coordinates)
                if mesh_shape is None:
                    mesh_shape = tensor_slice.mesh_shape
                else:
                    assert (
                        mesh_shape == tensor_slice.mesh_shape
                    ), "Inconsistent mesh shapes in stored slices"

        # Generate all expected coordinates for the mesh
        expected_coords = set(product(*(range(s) for s in mesh_shape)))

        # Check if we have all coordinates
        return all_slices == expected_coords

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
            KeyError: If the key is not found in any storage volumes, or if the key
                is a DTensor that is only partially committed.
        """
        self.assert_initialized()

        if key not in self.keys_to_storage_volumes:
            raise KeyError(f"Unable to locate {key} in any storage volumes.")

        volume_map = self.keys_to_storage_volumes[key]

        # Check if this is a DTensor and if it's fully committed
        if not self._is_dtensor_fully_committed(key, volume_map):
            raise KeyError(
                f"DTensor '{key}' is only partially committed. "
                f"Not all shards have been stored yet. "
                f"Please ensure all ranks complete their put() operations."
            )

        return volume_map

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
        assert (
            request.tensor_val is None
        ), "request should not contain tensor data, as this will significantly increase e2e latency"

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
    async def teardown(self) -> None:
        self.is_initialized = False
        self.keys_to_storage_volumes = Trie()
        self.strategy = None
        # StorageVolume in ControllerStrategy can be reused because it was spawned with get_or_spawn_controller.
        # So we have to reset it, otherwise new TensorSlice values for the same key will get piled up in the set.
        if self.storage_volumes is not None:
            await self.storage_volumes.reset.call()
        self.storage_volumes = None
        self.num_storage_volumes = None

    @endpoint
    def keys(self, prefix=None) -> List[str]:
        if prefix is None:
            return list(self.keys_to_storage_volumes.keys())
        return self.keys_to_storage_volumes.keys().filter_by_prefix(prefix)

    @endpoint
    def notify_delete(self, key: str, storage_volume_id: str) -> None:
        """
        Notify the controller that deletion of data is initiated in a storage volume.

        This should called after a successful delete operation to
        maintain the distributed storage index.
        """
        self.assert_initialized()
        if key not in self.keys_to_storage_volumes:
            raise KeyError(f"Unable to locate {key} in any storage volumes.")
        if storage_volume_id not in self.keys_to_storage_volumes[key]:
            raise KeyError(
                f"Unable to locate {key} in storage volume {storage_volume_id}."
            )
        del self.keys_to_storage_volumes[key][storage_volume_id]
        if len(self.keys_to_storage_volumes[key]) == 0:
            del self.keys_to_storage_volumes[key]

    def get_keys_to_storage_volumes(self) -> Mapping[str, Dict[str, StorageInfo]]:
        return self.keys_to_storage_volumes
