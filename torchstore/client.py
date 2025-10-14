# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
from logging import getLogger
from typing import Any, Union

import torch
from torch.distributed.tensor import DTensor

from torchstore.controller import ObjectType
from torchstore.logging import LatencyTracker
from torchstore.transport import Pipe, Request, TensorSlice
from torchstore.utils import assemble_tensor, get_local_tensor, get_slice_intersection

logger = getLogger(__name__)


class LocalClient:
    """This class represents the local store, which exists on every process. Remote storage
    is handled by the client.
    """

    def __init__(
        self,
        controller,
        strategy,
    ):
        self._controller = controller
        self.strategy = strategy

    async def _locate_volumes(self, key: str):
        """Helper method to call locate_volumes and convert any error to KeyError for missing keys."""
        try:
            return await self._controller.locate_volumes.call_one(key)
        except Exception as e:
            raise KeyError(str(e)) from e

    @torch.no_grad
    async def put(self, key: str, value: Union[torch.Tensor, Any]):
        latency_tracker = LatencyTracker(f"put:{key}")
        request = Request.from_any(value)
        # for now, we only write to one storage volume.
        # we probably don't need a remote call for this case since
        # it will never be dynamic. e.g. it's always based on the
        # TorchstoreStrategy defined during intiailization
        storage_volume, volume_id = self.strategy.select_storage_volume()

        pipe = Pipe(storage_volume)

        await pipe.put_to_storage_volume(key, request)
        latency_tracker.track_step("put_to_storage_volume")

        await self._controller.notify_put.call(key, request.meta_only(), volume_id)
        latency_tracker.track_step("notify_put")
        latency_tracker.track_e2e()

    @torch.no_grad
    async def get(
        self,
        key: str,
        inplace_tensor: torch.Tensor | DTensor | None = None,
        tensor_slice_spec: TensorSlice | None = None,
    ):
        logger.debug(f"Fetching {key}")
        latency_tracker = LatencyTracker(f"get:{key}")
        stored_object_type = await self._get_stored_object_type(key)

        self._verify_get_args(inplace_tensor, tensor_slice_spec, stored_object_type)

        # Get a spec of the shape and offset of the tensor slice to fetch. Fetch the whole tensor otherwise

        if stored_object_type is ObjectType.OBJECT:
            return await self._get_object(key)

        if stored_object_type is ObjectType.TENSOR:
            # TODO: we should get the part of interest in this branch.
            fetched_tensor = await self._get_tensor(key)
            if tensor_slice_spec is not None:
                fetched_tensor = get_local_tensor(
                    fetched_tensor,
                    tensor_slice_spec.local_shape,
                    tensor_slice_spec.offsets,
                )
        else:
            # Strored object is a DTensor. Return full tensor if
            # inplace_tensor is None, or return DTensor if inplace_tensor
            # is DTensor.
            # Here we abused request a bit to get tensor_slice from inplace DTensor. Otherwise
            # Request.from_any(inplace_tensor) will return None, and we use the tensor_slice_spec.
            assert stored_object_type is ObjectType.TENSOR_SLICE
            tensor_slice = (
                Request.from_any(inplace_tensor).tensor_slice or tensor_slice_spec
            )
            # Here full tensor should be the part of interest.
            fetched_tensor = await self._get_and_assemble_tensor(key, tensor_slice)

        # Pipe does not have support for inplace copies of fetched tensors yet,
        # so we just copy
        if inplace_tensor is not None:
            if hasattr(inplace_tensor, "_local_tensor"):
                # DTensor case - copy to the local tensor to avoid type mismatch
                inplace_tensor._local_tensor.copy_(fetched_tensor)
            else:
                # Regular tensor case
                inplace_tensor.copy_(fetched_tensor)

            return inplace_tensor

        latency_tracker.track_e2e()
        return fetched_tensor

    async def keys(self, prefix: str | None = None) -> list[str]:
        """
        Get all keys that match the given prefix.

        This method retrieves all keys from the storage that start with the specified prefix.

        Args:
            prefix (str): The prefix to match against stored keys.

        Returns:
            List[str]: A list of keys that match the given prefix.
        """
        # Keys are synced across all storage volumes, so we just call one.
        return await self._controller.keys.call_one(prefix)

    async def delete(self, key: str) -> None:
        """
        Delete a key from the distributed store.

        Args:
            key (str): The key to delete.

        Returns:
            None

        Raises:
            KeyError: If the key does not exist in the store.
        """
        latency_tracker = LatencyTracker(f"delete:{key}")
        volume_map = await self._controller.locate_volumes.call_one(key)

        async def delete_from_volume(volume_id: str):
            volume = self.strategy.get_storage_volume(volume_id)
            # Notify should come before the actual delete, so that the controller
            # doesn't think the key is still in the store when delete is happening.
            await self._controller.notify_delete.call_one(key, volume_id)
            await volume.delete.call(key)

        await asyncio.gather(
            *[delete_from_volume(volume_id) for volume_id in volume_map]
        )

        latency_tracker.track_e2e()

    async def exists(self, key: str) -> bool:
        """Check if a key exists in the distributed store.

        This is an efficient operation that only checks metadata at the controller level
        without retrieving the actual data.

        Args:
            key (str): The key to check for existence.

        Returns:
            bool: True if the key exists, False otherwise.
        """
        logger.debug(f"Checking existence of {key}")
        try:
            # Use the controller to check if key exists
            # This is efficient as it only checks metadata
            await self._controller.locate_volumes.call_one(key)
            return True
        except Exception as e:
            # Controller raises KeyError if key doesn't exist, but it comes wrapped
            # in an ActorError from the Monarch framework
            if "KeyError" in str(e) or "Unable to locate" in str(e):
                return False
            # Re-raise if it's a different kind of error
            raise e

    def _verify_get_args(
        self,
        inplace_tensor: torch.Tensor | DTensor | None,
        tensor_slice_spec: TensorSlice | None,
        stored_object_type: ObjectType | None,
    ):
        """
        Verify that the provided arguments are valid for the get() method.
        """
        # Error if request a Tensor or DTensor but the stored_object_type is OBJECT
        if stored_object_type == ObjectType.OBJECT and (
            inplace_tensor is not None or tensor_slice_spec is not None
        ):
            raise ValueError(
                "inplace_tensor or tensor_slice_spec is specified but the value stored is an object"
            )

        # inplace_tensor can only be None, Tensor, or DTensor
        if inplace_tensor is not None and not isinstance(
            inplace_tensor, (torch.Tensor, DTensor)
        ):
            raise ValueError(
                f"Invalid type for inplace_tensor: {type(inplace_tensor)}. Must be None, torch.Tensor, or DTensor."
            )

        if isinstance(inplace_tensor, torch.Tensor):
            if (
                tensor_slice_spec
                and tensor_slice_spec.local_shape != inplace_tensor.shape
            ):
                raise ValueError(
                    f"Requested tensor slice shape {tensor_slice_spec.local_shape} "
                    f"does not match in-place tensor shape {inplace_tensor.shape}"
                )

        if isinstance(inplace_tensor, DTensor):
            if tensor_slice_spec:
                raise ValueError(
                    "Cannot specify a tensor slice when fetching a DTensor"
                )

    async def _get_stored_object_type(self, key: str) -> ObjectType | None:
        """Peek into storage info for the given key and return the stored object type."""
        volume_map = await self._locate_volumes(key)
        for storage_info in volume_map.values():
            return storage_info.object_type
        raise ValueError(f"Unable to get stored object type for key `{key}`")

    async def _get_object(self, key: str):
        volume_map = await self._locate_volumes(key)
        volume_id, _ = volume_map.popitem()
        storage_volume = self.strategy.get_storage_volume(volume_id)
        pipe = Pipe(storage_volume)
        request = Request.from_any(None)
        return await pipe.get_from_storage_volume(key, request)

    async def _get_tensor(self, key: str) -> torch.Tensor:
        """Fetches the tensor which is stored in one volume storage"""
        volume_map = await self._locate_volumes(key)

        # if the storage is a Tensor instead of DTensor, just fetch and return it.
        for volume_id, _ in volume_map.items():
            storage_volume = self.strategy.get_storage_volume(volume_id)
            pipe = Pipe(storage_volume)
            # TODO: consolidate the logic here - None indicates it is an object request,
            # which is sematically inappropriate here.
            request = Request.from_any(None)
            return await pipe.get_from_storage_volume(key, request)

    async def _get_and_assemble_tensor(
        self, key: str, tensor_slice_spec: TensorSlice | None = None
    ) -> torch.Tensor:
        """Fetches slices from all volume storages and stitch together to return the whole tensor.

        Args:
            key: The key to fetch from storage
            tensor_slice_spec: Optional tensor slice to optimize fetching by only retrieving
                          intersecting portions from storage volumes. If None, fetches the whole tensor.

        Returns:
            The assembled tensor from all storage volumes
        """
        volume_map = await self._locate_volumes(key)
        # Handle the tensor case
        partial_results = []
        for volume_id, storage_info in volume_map.items():
            storage_volume = self.strategy.get_storage_volume(volume_id)
            pipe = Pipe(storage_volume)

            # fetch from all storage volumes, something like this
            # TODO: fix so we can request all tensor slices from a storage volume
            # at once, this is silly
            for tensor_slice in storage_info.tensor_slices:
                # Intersect the tensor slice with the DTensor slice to optimize fetching
                if tensor_slice_spec is not None:
                    # Check if stored tensor_slice overlaps with requested dtensor_slice
                    tensor_slice = get_slice_intersection(
                        tensor_slice, tensor_slice_spec
                    )

                    if tensor_slice is None:
                        # No overlap, skip fetching this slice
                        continue

                tensor_slice_request = Request.from_tensor_slice(tensor_slice)

                local_tensor = await pipe.get_from_storage_volume(
                    key, tensor_slice_request
                )
                partial_results.append((local_tensor, tensor_slice))
        if not partial_results:
            raise RuntimeError(
                f"No tensor slices found for key '{key}' that intersect with the requested slice"
            )

        # build the entire tensor.
        # TODO: again, we should have better control over
        # rebuilding only the portion I need, but this is a good start

        local_tensors = []
        global_offsets = []
        global_shape = None
        device_mesh_shape = None
        for local_tensor, tensor_slice in partial_results:
            local_tensors.append(local_tensor)

            global_offsets.append(tensor_slice.offsets)
            if global_shape is None:
                global_shape = tensor_slice.global_shape
            else:
                assert global_shape == tensor_slice.global_shape

            if device_mesh_shape is None:
                device_mesh_shape = tensor_slice.mesh_shape
            else:
                assert device_mesh_shape == tensor_slice.mesh_shape

        assembled_tensor = assemble_tensor(
            local_tensors,
            global_shape,
            global_offsets,
        )

        return assembled_tensor
