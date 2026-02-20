# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import auto, Enum
from itertools import product

from monarch.actor import Actor, endpoint

from torchstore.storage_utils.trie import Trie
from torchstore.storage_volume import StorageVolume
from torchstore.strategy import ControllerStorageVolumes, TorchStoreStrategy
from torchstore.transport.types import KeyedRequest, Request, TensorSlice


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
    tensor_slices: set[TensorSlice | None] = field(default_factory=set)

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
        self.strategy: TorchStoreStrategy | None = None
        self.storage_volumes: StorageVolume | None = None
        self.num_storage_volumes: int | None = None
        self.strategy: TorchStoreStrategy | None = None

    def assert_initialized(self) -> None:
        assert (
            self.is_initialized
        ), "Please call torchstore.initialize before attempting to use store."

    def _is_dtensor_fully_committed(
        self, key: str, volume_map: dict[str, StorageInfo]
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

        if isinstance(strategy, ControllerStorageVolumes):
            warnings.warn(
                "ControllerStorageVolumes is deprecated and will be removed in a future "
                "release. It spawns a singleton storage volume on the controller, which "
                "may become a bottleneck. Use LocalRankStrategy for better scalability.",
                DeprecationWarning,
                stacklevel=2,
            )

        self.strategy = strategy
        self.storage_volumes = storage_volumes
        self.num_storage_volumes = num_storage_volumes

        await self.strategy.set_storage_volumes(self.storage_volumes)
        self.is_initialized = True

    @endpoint
    async def get_controller_strategy(self) -> TorchStoreStrategy:
        self.assert_initialized()
        assert self.strategy is not None, "Strategy is not set"
        return self.strategy

    @endpoint
    async def locate_volumes(
        self,
        keys: list[str],
    ) -> dict[str, dict[str, StorageInfo]]:
        """Locate storage volumes containing shards of the specified keys.

        Returns {<key> -> {<storage_volume_id> -> StorageInfo}} where each key maps to
        the storage volumes holding shards of its data.

        For example, if a key holds a DTensor with 3 shards, the returned map will look like:
        {
            "<key>": {
                "<storage_volume_id>": StorageInfo.tensor_slice=set([
                    "<tensor_slice>",
                    "<tensor_slice>",
                    "<tensor_slice>",
                ]),
                ...
            },
            ...
        }

        Args:
            keys (list[str]): The keys to locate in storage volumes.

        Returns:
            Dict[str, Dict[str, StorageInfo]]: Mapping from each key to a mapping from
                storage volume IDs to StorageInfo objects containing metadata about
                the stored data shards.

        Raises:
            KeyError: If any key is not found in any storage volumes, or if a key
                is a DTensor that is only partially committed.
        """
        self.assert_initialized()
        result = {}
        for key in keys:
            if key not in self.keys_to_storage_volumes:
                raise KeyError(f"Unable to locate {key} in any storage volumes.")
            volume_map = self.keys_to_storage_volumes[key]
            if not self._is_dtensor_fully_committed(key, volume_map):
                raise KeyError(f"DTensor '{key}' is only partially committed.")
            result[key] = volume_map
        return result

    @endpoint
    async def notify_put_batch(
        self,
        entries: list[KeyedRequest],
        storage_volume_id: str,
    ) -> None:
        """Notify the controller that one or more keys have been stored.

        Args:
            entries: List of KeyedRequests
            storage_volume_id: ID of the storage volume where the data was stored.
        """
        self.assert_initialized()

        for key, request in entries:
            assert (
                request.tensor_val is None
            ), "request should not contain tensor data, as this will significantly increase e2e latency"

            if key not in self.keys_to_storage_volumes:
                self.keys_to_storage_volumes[key] = {}

            storage_info = StorageInfo(
                object_type=ObjectType.from_request(request),
                tensor_slices={request.tensor_slice},
            )

            if storage_volume_id not in self.keys_to_storage_volumes[key]:
                self.keys_to_storage_volumes[key][storage_volume_id] = storage_info
            else:
                self.keys_to_storage_volumes[key][storage_volume_id].update(
                    storage_info
                )

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
    async def keys(self, prefix=None) -> list[str]:
        if prefix is None:
            return list(self.keys_to_storage_volumes.keys())
        return self.keys_to_storage_volumes.keys().filter_by_prefix(prefix)

    @endpoint
    async def notify_delete(self, key: str, storage_volume_id: str) -> None:
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

    def get_keys_to_storage_volumes(self) -> Mapping[str, dict[str, StorageInfo]]:
        return self.keys_to_storage_volumes
