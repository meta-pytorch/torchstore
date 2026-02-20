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
from torchstore.transport import (
    create_transport_buffer,
    KeyedRequest,
    Request,
    TensorSlice,
)
from torchstore.transport.buffers import TransportContext
from torchstore.utils import (
    assemble_tensor,
    get_destination_view,
    get_slice_intersection,
    tensors_overlap_in_memory,
)

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

    async def _locate_volumes(self, keys: list[str]):
        """Helper method to call locate_volumes and convert any error to KeyError for missing keys."""
        try:
            return await self._controller.locate_volumes.call_one(keys)
        except Exception as e:
            raise KeyError(str(e)) from e

    @torch.no_grad
    async def put(self, key: str, value: torch.Tensor | Any):
        latency_tracker = LatencyTracker(f"put:{key}")
        await self.put_batch([(key, value)])
        latency_tracker.track_e2e()

    @torch.no_grad
    async def put_batch(self, entries: list[tuple[str, torch.Tensor | Any]]):
        """Batch put multiple key-value pairs in a single operation.

        Args:
            entries: List of (key, value) tuples to store.
        """
        latency_tracker = LatencyTracker("put_batch")

        requests = []
        for key, value in entries:
            if isinstance(value, (torch.Tensor, DTensor)):
                request = Request.from_any(value)
            else:
                request = Request.from_objects(value)
            requests.append(KeyedRequest(key, request))

        storage_volume_ref = self.strategy.select_storage_volume()
        transport_buffer = create_transport_buffer(storage_volume_ref)
        latency_tracker.track_step("create transport buffer")

        await transport_buffer.put_to_storage_volume(requests)
        latency_tracker.track_step("put_to_storage_volume")

        await self._controller.notify_put_batch.call(
            [r.meta_only() for r in requests],
            storage_volume_ref.volume_id,
        )
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
        results = await self._fetch({key: request})
        latency_tracker.track_step("fetch")

        result = self._apply_inplace(results[key], inplace_tensor, request)
        latency_tracker.track_e2e()
        return result

    @torch.no_grad
    async def get_batch(
        self,
        keys: list[str],
        inplace_tensors: dict[str, torch.Tensor | DTensor] | None = None,
    ) -> dict[str, Any]:
        """Batch get multiple keys in a single operation.

        Args:
            keys: List of keys to fetch.
            inplace_tensors: Optional mapping of key -> pre-allocated tensor for
                in-place retrieval.

        Returns:
            dict mapping each key to its fetched data.
        """
        latency_tracker = LatencyTracker("get_batch")
        inplace_map = inplace_tensors or {}

        per_key_requests: dict[str, Request] = {}
        for key in keys:
            inplace = inplace_map.get(key)
            per_key_requests[key] = (
                Request.from_any(inplace) if inplace is not None else Request()
            )

        results = await self._fetch(per_key_requests)
        latency_tracker.track_step("fetch")

        final_results: dict[str, Any] = {}
        for key in keys:
            final_results[key] = self._apply_inplace(
                results[key], inplace_map.get(key), per_key_requests[key]
            )

        latency_tracker.track_e2e()
        return final_results

    def _apply_inplace(
        self,
        fetched: Any,
        inplace_tensor: torch.Tensor | DTensor | None,
        request: Request,
    ) -> Any:
        """Handle inplace copy-back for DTensor resharding cases.

        Always returns inplace if provided. Copies fetched data into
        request.tensor_val when data_ptr differs (resharding produced a new tensor).
        """
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
            return inplace_tensor
        # returning inplace_tensor since fetched will point to _local_tensor in
        # the case of DTensor.
        return inplace_tensor if inplace_tensor is not None else fetched

    async def _fetch(
        self,
        requests: dict[str, Request],
    ) -> dict[str, Any]:
        """Locate volumes, expand entries, fetch per-volume, assemble results.

        Args:
            requests: Pre-built Request per key (may include tensor_slice).

        Returns:
            dict mapping each key to its raw fetched data (before inplace copy-back).
        """
        keys = list(requests)
        volume_maps = await self._locate_volumes(keys)
        all_volume_ids: set[str] = {vid for vm in volume_maps.values() for vid in vm}

        # Eagerly create transport buffers for all volumes
        transport_buffer_map = {
            volume_id: create_transport_buffer(
                self.strategy.get_storage_volume(volume_id)
            )
            for volume_id in all_volume_ids
        }

        # Expand entries per volume
        volume_requests: dict[str, list[KeyedRequest]] = {}

        for key, request in requests.items():
            volume_map = volume_maps[key]

            # only attempt inplace if buffer has support, tensor is contiguous, and tensor_slice is provided
            use_inplace = (
                all(
                    transport_buffer_map[vid].supports_inplace_resharding
                    for vid in volume_map
                )
                and request.tensor_val is not None
                and request.tensor_val.is_contiguous()
                and request.tensor_slice is not None
            )

            for volume_id, storage_info in volume_map.items():
                volume_requests.setdefault(volume_id, [])

                # no sharding for objects or regular tensors.
                if storage_info.object_type == ObjectType.OBJECT:
                    obj_request = Request(is_object=True)
                    volume_requests[volume_id].append(KeyedRequest(key, obj_request))
                    break
                elif storage_info.object_type == ObjectType.TENSOR:
                    volume_requests[volume_id].append(KeyedRequest(key, request))
                    break
                else:
                    # TENSOR_SLICE
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

                        slice_request = Request.from_tensor_slice(fetch_slice)

                        if use_inplace:
                            # Try to get a view of the destination tensor for inplace writes
                            dest_view = get_destination_view(
                                request.tensor_val,
                                request.tensor_slice,
                                fetch_slice,
                            )
                            if dest_view is not None:
                                # Pass the view as tensor_val - transport writes directly inplace
                                slice_request.tensor_val = dest_view

                        volume_requests[volume_id].append(
                            KeyedRequest(key, slice_request)
                        )

        # Fetch per volume. direct results go straight to final_results
        final_results: dict[str, Any] = {}
        partial_results: dict[str, list[tuple[Any, TensorSlice]]] = {}

        for volume_id, entries in volume_requests.items():
            results = await transport_buffer_map[volume_id].get_from_storage_volume(
                entries
            )
            for result, entry in zip(results, entries):
                object_type = volume_maps[entry.key][volume_id].object_type
                if object_type in (ObjectType.OBJECT, ObjectType.TENSOR):
                    final_results[entry.key] = result
                else:
                    partial_results.setdefault(entry.key, []).append(
                        (result, entry.request.tensor_slice)
                    )

        # Assemble sliced results
        for key, parts in partial_results.items():
            request = requests[key]
            if (
                request.tensor_val is not None
                and request.tensor_slice is not None
                and tensors_overlap_in_memory(parts, request.tensor_val)
            ):
                final_results[key] = request.tensor_val
            else:
                local_tensors = [t for t, _ in parts]
                global_offsets = [s.offsets for _, s in parts]
                final_results[key] = assemble_tensor(local_tensors, global_offsets)
                if request.tensor_slice is not None:
                    assert final_results[key].shape == request.tensor_slice.local_shape

        for key in keys:
            if key not in final_results:
                raise RuntimeError(f"No results found for key '{key}'")

        return final_results

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
        volume_map = (await self._controller.locate_volumes.call_one([key]))[key]

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
            await self._controller.locate_volumes.call_one([key])
            return True
        except Exception as e:
            # Controller raises KeyError if key doesn't exist, but it comes wrapped
            # in an ActorError from the Monarch framework
            if "KeyError" in str(e) or "Unable to locate" in str(e):
                return False
            # Re-raise if it's a different kind of error
            raise e
