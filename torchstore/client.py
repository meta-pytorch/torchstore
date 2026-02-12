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


def _get_destination_view(
    dest_tensor: torch.Tensor,
    dest_slice: TensorSlice,
    fetch_slice: TensorSlice,
) -> torch.Tensor | None:
    """Get a view of the destination tensor where the fetched slice should be written.

    Args:
        dest_tensor: The destination tensor (must be contiguous)
        dest_slice: TensorSlice describing the destination tensor's position in global space
        fetch_slice: TensorSlice describing the slice being fetched

    Returns:
        A view of dest_tensor for the region where fetch_slice should be written,
        or None if the fetch_slice doesn't map to a contiguous region in dest_tensor.
    """
    if not dest_tensor.is_contiguous():
        return None

    # Compute the local indices within dest_tensor where fetch_slice should be written
    slices = []

    for dim in range(len(fetch_slice.global_shape)):
        # fetch_slice offset in global coordinates
        fetch_start = fetch_slice.offsets[dim]
        fetch_end = fetch_start + fetch_slice.local_shape[dim]

        # dest_slice offset in global coordinates
        dest_start = dest_slice.offsets[dim]

        # Convert to local coordinates within dest_tensor
        local_start = fetch_start - dest_start
        local_end = fetch_end - dest_start

        # Validate bounds
        if local_start < 0 or local_end > dest_slice.local_shape[dim]:
            return None

        slices.append(slice(local_start, local_end))

    # Create the view
    view = dest_tensor[tuple(slices)]

    # Check if the view is contiguous - required for RDMA transports
    if not view.is_contiguous():
        return None

    return view


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

        # Eagerly create transport buffers for all volumes
        transport_buffer_map = {
            volume_id: create_transport_buffer(
                self.strategy.get_storage_volume(volume_id)
            )
            for volume_id in volume_map.keys()
        }

        # only attempt inplace if buffer has support, tensor is contiguous, and tensor_slice is provided
        use_inplace_views = (
            all(tb.supports_inplace_resharding for tb in transport_buffer_map.values())
            and request.tensor_val is not None
            and request.tensor_val.is_contiguous()
            and request.tensor_slice is not None
        )

        for volume_id, storage_info in volume_map.items():
            transport_buffer = transport_buffer_map[volume_id]

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

                # Try to get a view of the destination tensor for inplace writes
                dest_view = (
                    _get_destination_view(
                        request.tensor_val, request.tensor_slice, fetch_slice
                    )
                    if use_inplace_views
                    else None
                )

                if dest_view is not None:
                    # Pass the view as tensor_val - transport writes directly inplace
                    slice_request = Request.from_tensor_slice(fetch_slice)
                    slice_request.tensor_val = dest_view
                    await transport_buffer.get_from_storage_volume(key, slice_request)
                    partial_results.append((dest_view, fetch_slice))
                else:
                    # TODO: ensure this is not in the common path, or that we have a solution for
                    # this pattern since it creates a new allocation on every fetch, and should be
                    # is generally avoidable.
                    slice_request = Request.from_tensor_slice(fetch_slice)
                    local_tensor = await transport_buffer.get_from_storage_volume(
                        key, slice_request
                    )
                    partial_results.append((local_tensor, fetch_slice))

        # Check if all results share memory with the destination tensor (inplace)
        if (
            use_inplace_views
            and partial_results
            and all(
                t.data_ptr() >= request.tensor_val.data_ptr()
                and t.data_ptr()
                < request.tensor_val.data_ptr() + request.tensor_val.nbytes
                for t, _ in partial_results
            )
        ):
            return request.tensor_val

        # Tensor was not resharded inplace, requires us to rebuild the slice 'manually'
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
