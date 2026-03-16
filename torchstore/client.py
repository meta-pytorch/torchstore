# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
from collections import defaultdict
from logging import getLogger
from typing import Any

import torch
from torch.distributed.tensor import DTensor

from torchstore.controller import ObjectType
from torchstore.logging import LatencyTracker
from torchstore.strategy import TorchStoreStrategy
from torchstore.transport import create_transport_buffer, Request, TensorSlice
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

    async def _locate_volumes(self, keys: list[str]):
        """Helper method to call locate_volumes and convert any error to KeyError for missing keys."""
        try:
            return await self._controller.locate_volumes.call_one(keys)
        except Exception as e:
            raise KeyError(str(e)) from e

    @torch.no_grad
    async def put(self, key: str, value: torch.Tensor | Any):
        latency_tracker = LatencyTracker(f"put:{key}")
        await self.put_batch({key: value})
        latency_tracker.track_e2e()

    @torch.no_grad
    async def put_batch(self, entries: dict[str, torch.Tensor | Any]):
        """Batch put multiple key-value pairs in a single operation.

        Args:
            entries: Dict mapping keys to values to store.
        """
        assert (
            isinstance(entries, dict) and entries
        ), "put_batch requires a non-empty dict"

        latency_tracker = LatencyTracker("put_batch")

        requests = []
        for key, value in entries.items():
            if isinstance(value, (torch.Tensor, DTensor)):
                request = Request.from_any(key, value)
            else:
                request = Request.from_objects(key, value)
            requests.append(request)

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

        Raises:
            KeyError: If the key does not exist.
        """
        logger.debug(f"Fetching {key}")
        latency_tracker = LatencyTracker(f"get:{key}")

        request = Request.from_any(key, inplace_tensor, tensor_slice_spec)
        results = await self._fetch([request])
        latency_tracker.track_step("fetch")

        result = self._apply_inplace(results[key], inplace_tensor, request)
        latency_tracker.track_e2e()
        return result

    @torch.no_grad
    async def get_batch(
        self,
        keys: list[str] | dict[str, torch.Tensor | DTensor | None],
    ) -> dict[str, Any]:
        """Batch get multiple keys in a single operation.

        All-or-nothing: if any key is missing, the entire batch raises
        and no partial results are returned.

        Args:
            keys: Either a list of keys to fetch, or a dict mapping keys to
                optional pre-allocated tensors for in-place retrieval.

        Returns:
            dict mapping each key to its fetched data.

        Raises:
            KeyError: If any key does not exist.
        """
        latency_tracker = LatencyTracker("get_batch")

        if not keys:
            raise ValueError("get_batch requires a non-empty dict or list")

        inplace_dict = {}
        if isinstance(keys, dict):
            inplace_dict = keys
        elif isinstance(keys, list):
            if len(keys) != len(set(keys)):
                raise ValueError("get_batch keys must be unique")
        else:
            raise TypeError(f"get_batch expects list[str] or dict, got {type(keys)}")

        requests = [Request.from_any(key, inplace_dict.get(key)) for key in keys]
        results = await self._fetch(requests)
        latency_tracker.track_step("fetch")

        final_results: dict[str, Any] = {}
        for req in requests:
            final_results[req.key] = self._apply_inplace(
                results[req.key], inplace_dict.get(req.key), req
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
        requests: list[Request],
    ) -> dict[str, Any]:
        """Locate volumes, expand entries, fetch per-volume, assemble results.

        Args:
            requests: Pre-built Request per key (may include tensor_slice).

        Returns:
            dict mapping each key to its raw fetched data (before inplace copy-back).
        """
        keys = [r.key for r in requests]
        volume_maps = await self._locate_volumes(keys)
        all_volume_ids: set[str] = {vid for vm in volume_maps.values() for vid in vm}

        # eagerly make transport buffers for all volumes
        transport_buffer_map = {
            volume_id: create_transport_buffer(
                self.strategy.get_storage_volume(volume_id)
            )
            for volume_id in all_volume_ids
        }

        # collect the requests for each volume
        volume_requests, whole_keys = self._build_volume_requests(
            requests, volume_maps, transport_buffer_map
        )

        # fetch (request, volume_result) pairs from all volumes in parallel
        fetch_pairs = await self._fetch_results(volume_requests, transport_buffer_map)

        # assemble final results
        return self._assemble_results(requests, fetch_pairs, whole_keys)

    def _build_volume_requests(
        self,
        requests: list[Request],
        volume_maps: dict[str, dict],
        transport_buffer_map: dict,
    ) -> tuple[dict[str, list[Request]], set[str]]:
        """Expand per-key requests into per-volume request lists.

        Returns:
            (volume_requests, whole_keys) where whole_keys holds keys stored
            as OBJECT or TENSOR (complete results, not assembled from slices).
        """
        volume_requests: dict[str, list[Request]] = defaultdict(list)
        whole_keys: set[str] = set()

        for request in requests:
            volume_map = volume_maps[request.key]

            use_inplace = (
                all(
                    transport_buffer_map[vid].supports_inplace_resharding
                    for vid in volume_map
                )
                and request.tensor_val is not None
                and request.tensor_val.is_contiguous()
            )

            for volume_id, storage_info in volume_map.items():
                if storage_info.object_type == ObjectType.OBJECT:
                    volume_requests[volume_id].append(
                        Request(key=request.key, is_object=True)
                    )
                    whole_keys.add(request.key)
                    break
                elif storage_info.object_type == ObjectType.TENSOR:
                    volume_requests[volume_id].append(request)
                    whole_keys.add(request.key)
                    break
                else:
                    volume_requests[volume_id].extend(
                        self._expand_tensor_slices(request, storage_info, use_inplace)
                    )

        return dict(volume_requests), whole_keys

    def _expand_tensor_slices(
        self,
        request: Request,
        storage_info,
        use_inplace: bool,
    ) -> list[Request]:
        """Expand a single key's tensor slices into sub-requests."""
        sub_requests = []
        for stored_slice in storage_info.tensor_slices:
            fetch_slice = stored_slice
            if request.tensor_slice is not None:
                # TODO: we should also continue if we have already fetched this region in a previous call
                # and also return completely if we've already fetched all regions. This is extra inneficient
                # in the case of DP, where we fetch all Replicate shards unnecessarily
                fetch_slice = get_slice_intersection(stored_slice, request.tensor_slice)
                if fetch_slice is None:
                    continue

            slice_request = Request.from_tensor_slice(request.key, fetch_slice)

            if use_inplace:
                dest_view = get_destination_view(
                    request.tensor_val,
                    request.tensor_slice,
                    fetch_slice,
                )
                if dest_view is not None:
                    slice_request.tensor_val = dest_view

            sub_requests.append(slice_request)
        return sub_requests

    async def _fetch_results(
        self,
        volume_requests: dict[str, list[Request]],
        transport_buffer_map: dict,
    ) -> list[tuple[Request, Any]]:
        """Fetch from all volumes in parallel. Returns (sub_request, result) pairs."""

        async def _fetch_one(volume_id: str, sub_requests: list[Request]):
            results = await transport_buffer_map[volume_id].get_from_storage_volume(
                sub_requests
            )
            return list(zip(sub_requests, results, strict=True))

        per_volume = await asyncio.gather(
            *[_fetch_one(vid, reqs) for vid, reqs in volume_requests.items()]
        )
        # Flatten list-of-lists into a single list of (sub_request, result) pairs
        return [pair for pairs in per_volume for pair in pairs]

    def _assemble_results(
        self,
        requests: list[Request],
        fetch_pairs: list[tuple[Request, Any]],
        whole_keys: set[str],
    ) -> dict[str, Any]:
        """Classify fetch results and assemble tensor slices into final values."""
        final: dict[str, Any] = {}
        slice_parts: dict[str, list[tuple[Any, TensorSlice]]] = defaultdict(list)

        for sub_request, result in fetch_pairs:
            if sub_request.key in whole_keys:
                final[sub_request.key] = result
            else:
                slice_parts[sub_request.key].append((result, sub_request.tensor_slice))

        request_by_key = {r.key: r for r in requests}
        for key, parts in slice_parts.items():
            request = request_by_key[key]
            if request.tensor_val is not None and tensors_overlap_in_memory(
                parts, request.tensor_val
            ):
                final[key] = request.tensor_val
            else:
                local_tensors = [t for t, _ in parts]
                global_offsets = [s.offsets for _, s in parts]
                # TODO: this is yet another new allocation on every fetch.
                final[key] = assemble_tensor(local_tensors, global_offsets)
                if request.tensor_slice is not None:
                    assert final[key].shape == request.tensor_slice.local_shape

        for r in requests:
            if r.key not in final:
                raise RuntimeError(
                    f"No results found for key '{r.key}'. If this key contains "
                    "tensor slices, no stored slices intersect with the requested slice."
                )

        return final

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
