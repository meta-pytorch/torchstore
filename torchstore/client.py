# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
from logging import getLogger
from typing import Any

import torch
from torch.distributed.tensor import DTensor

from torchstore.controller import ObjectType
from torchstore.logging import LatencyTracker
from torchstore.strategy import TorchStoreStrategy
from torchstore.transport import create_transport_buffer, Request, TensorSlice
from torchstore.transport.buffers import TransportContext
from torchstore.utils import assemble_tensor, get_slice_intersection

logger = getLogger(__name__)


class LocalClient:
    """Client-side interface for TorchStore operations.

    LocalClient runs in the user's process and coordinates with remote StorageVolumes
    via TransportBuffers. It handles put/get operations by selecting appropriate storage
    volumes through the configured strategy and managing data transport.
    """

    def __init__(
        self,
        controller,
        strategy,
    ):
        self._controller = controller
        self.strategy: TorchStoreStrategy = strategy
        self.transport_context = TransportContext()

    async def _locate_volumes(self, key: str):
        """Helper method to call locate_volumes and convert any error to KeyError for missing keys."""
        try:
            return await self._controller.locate_volumes.call_one(key)
        except Exception as e:
            raise KeyError(str(e)) from e

    @torch.no_grad
    async def put(self, key: str, value: torch.Tensor | Any):
        latency_tracker = LatencyTracker(f"put:{key}")

        # Create request based on value type
        if isinstance(value, (torch.Tensor, DTensor)):
            request = Request.from_any(value)
        else:
            request = Request.from_objects(value)

        storage_volume_ref = self.strategy.select_storage_volume()
        transport_buffer = create_transport_buffer(storage_volume_ref)
        latency_tracker.track_step("create transport buffer")

        await transport_buffer.put_to_storage_volume(key, request)
        latency_tracker.track_step("put_to_storage_volume")

        await self._controller.notify_put.call(
            key, request.meta_only(), storage_volume_ref.volume_id
        )
        latency_tracker.track_step("notify_put")
        latency_tracker.track_e2e()

    @torch.no_grad
    async def _put_batch(self, items: dict[str, torch.Tensor | Any]) -> None:
        """Internal method to store multiple tensors or objects in a single batched call.

        This is significantly more efficient than calling put() multiple times as it
        parallelizes storage operations and batches controller notifications.

        Args:
            items: Dictionary mapping keys to values to store.
        """
        if not items:
            return

        latency_tracker = LatencyTracker(f"put_batch:{len(items)}_keys")

        # Select storage volume (all items go to same volume)
        storage_volume_ref = self.strategy.select_storage_volume()
        latency_tracker.track_step("select storage volume")

        # Put all items in parallel using asyncio.gather
        # NOTE: Each operation needs its own transport buffer to avoid race conditions.
        # Transport buffers maintain internal state (ipc_handle, tensor_ref, shape, dtype, etc.)
        # that gets corrupted when multiple concurrent operations share the same instance.
        async def put_single(key: str, value: torch.Tensor | Any):
            transport_buffer = create_transport_buffer(storage_volume_ref)
            # Create request based on value type (same logic as put method)
            if isinstance(value, (torch.Tensor, DTensor)):
                request = Request.from_any(value)
            else:
                request = Request.from_objects(value)
            await transport_buffer.put_to_storage_volume(key, request)
            return key, request

        put_results = await asyncio.gather(
            *[put_single(key, value) for key, value in items.items()]
        )
        latency_tracker.track_step("put_to_storage_volume_batch")

        # Notify controller with a single batched RPC call (not individual calls)
        notifications = [
            (key, request.meta_only(), storage_volume_ref.volume_id)
            for key, request in put_results
        ]
        await self._controller.notify_put_batch.call(notifications)
        latency_tracker.track_step("notify_put_batch")
        latency_tracker.track_e2e()

    @torch.no_grad
    async def get(
        self,
        key: str,
        inplace_tensor: torch.Tensor | DTensor | None = None,
        tensor_slice_spec: TensorSlice | None = None,
    ):
        """Fetch data from TorchStore.

        Args:
            key: The key to fetch.
            inplace_tensor: Optional pre-allocated tensor for in-place retrieval.
                If a DTensor is provided, its sharding info is used to fetch the
                appropriate slice. The transport buffer will attempt to write
                directly into this tensor to avoid extra allocations.
            tensor_slice_spec: Optional explicit tensor slice to fetch. If provided
                with a regular tensor inplace_tensor, fetches just that slice.
                Cannot be used with DTensor inplace_tensor (use the DTensor's
                sharding info instead).

        Returns:
            The fetched data. If inplace_tensor was provided, returns it after
            populating with the fetched data.
        """
        logger.debug(f"Fetching {key}")
        latency_tracker = LatencyTracker(f"get:{key}")

        request = Request.from_any(inplace_tensor, tensor_slice_spec)

        # Fetch the data
        fetched = await self._fetch(key, request)
        latency_tracker.track_step("fetch")

        # TODO: remove this copy and instead assert.
        # unfortunately, during resharding cases, we don't yet support writing inplace
        # from multiple regions into the inplace tensor, which leads to _fetch returning
        # a new tensor.
        if (
            inplace_tensor is not None
            and fetched.data_ptr() != request.tensor_val.data_ptr()
        ):
            # request tensor_val is a ref to _local_tensor if inplace is dtensor.
            request.tensor_val.copy_(fetched)
            latency_tracker.track_e2e()
            return inplace_tensor

        latency_tracker.track_e2e()

        # returning inplace_tensor since fetched will point to _local_tensor in
        # the case of DTensor.
        return inplace_tensor if inplace_tensor is not None else fetched

    async def _fetch(
        self,
        key: str,
        request: Request,
    ) -> torch.Tensor | Any:
        """Unified fetch that handles tensors, objects, and tensor slices.

        Args:
            key: Storage key to fetch.
            request: Request containing tensor_slice and optional inplace tensor.

        Returns:
            The fetched data (tensor, assembled tensor, or object).
        """
        volume_map = await self._locate_volumes(key)
        partial_results = []

        for volume_id, storage_info in volume_map.items():
            volume_ref = self.strategy.get_storage_volume(volume_id)
            transport_buffer = create_transport_buffer(volume_ref)

            # no sharding for objects or regular tensors.
            if storage_info.object_type == ObjectType.OBJECT:
                request.is_object = True
                return await transport_buffer.get_from_storage_volume(key, request)
            if storage_info.object_type == ObjectType.TENSOR:
                return await transport_buffer.get_from_storage_volume(key, request)

            # Has tensor slices - fetch each relevant slice
            for stored_slice in storage_info.tensor_slices:
                fetch_slice = stored_slice
                if request.tensor_slice is not None:
                    # TODO: we should also continue if we have already fetched this region in a previous call
                    # and also return completely if we've already fetched all regions. This is extra inneficient
                    # in the case of DP, where we fetch all Replicate shards unnecessarily
                    fetch_slice = get_slice_intersection(
                        stored_slice, request.tensor_slice
                    )
                    if fetch_slice is None:
                        continue

                # TODO: We should optimize this.
                # this unfortunately creates a new allocation on every fetch. (and fetches each slice separately)
                slice_request = Request.from_tensor_slice(fetch_slice)
                local_tensor = await transport_buffer.get_from_storage_volume(
                    key, slice_request
                )
                partial_results.append((local_tensor, fetch_slice))

        # If we get here, we need to assemble from partial results
        if not partial_results:
            raise RuntimeError(
                f"No tensor slices found for key '{key}' that intersect with the requested slice"
            )

        local_tensors = []
        global_offsets = []
        for local_tensor, slice_info in partial_results:
            local_tensors.append(local_tensor)
            global_offsets.append(slice_info.offsets)

        # TODO: this is yet another new allocation on every fetch.
        assembled_tensor = assemble_tensor(local_tensors, global_offsets)
        if request.tensor_slice is not None:
            assert assembled_tensor.shape == request.tensor_slice.local_shape
        return assembled_tensor

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
            volume_ref = self.strategy.get_storage_volume(volume_id)
            # Notify should come before the actual delete, so that the controller
            # doesn't think the key is still in the store when delete is happening.
            await self._controller.notify_delete.call_one(key, volume_id)
            await volume_ref.volume.delete.call(key)

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
