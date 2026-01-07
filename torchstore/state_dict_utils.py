# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from dataclasses import dataclass
from logging import getLogger
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.distributed.checkpoint._nested_dict import (
    flatten_state_dict,
    unflatten_state_dict,
)
from torch.distributed.tensor import DTensor, Placement
from torchstore.logging import init_logging, LatencyTracker

from torchstore.transport.pipe import TensorSlice

DELIM = "/"
MAPPING = "MAPPING"
TORCHSTORE_STATE_DICT = "TORCHSTORE_STATE_DICT"

logger = getLogger(__name__)


def tssd_enabled() -> bool:
    """
    Check if TorchStoreStateDict is enabled for put and get. If enabled, we will use the
    TorchStoreStateDict to batch tensors in the state dict into one blob and transfer
    them more efficiently.
    """

    return os.environ.get("TORCHSTORE_EXPERIMENTAL_BATCH_STATE_DICT", "1") == "1"


async def put_state_dict(store, state_dict, key):
    """
    We have an option here. Either we can "flatten state dict", by turning state dict names into a single key,
    or I can actually just maintain the dictionary representation of the state dict, and we can allow some
    recursive behavior in the store.

    Overall, this might not even be something we want to solve for in the TorchStore, but I'm adding this
    utility so we can test sharding models.

    """
    flattened_state_dict, mapping = flatten_state_dict(state_dict)
    for flattened_key, value in flattened_state_dict.items():
        await store.put(f"{key}{DELIM}{flattened_key}", value)

    await store.put(f"{key}{DELIM}{MAPPING}", mapping)


async def put_state_dict_batch(store, state_dict, key):
    """
    Turning state dict names into a single key.
    """
    init_logging()
    latency_tracker = LatencyTracker(f"put_state_dict_batch:{key}")
    torchstore_state_dict: TorchStoreStateDict = TorchStoreStateDict.from_state_dict(
        state_dict
    )
    latency_tracker.track_step("from_state_dict")
    # Store the TorchStoreStateDict object
    await store.put(f"{key}{DELIM}{TORCHSTORE_STATE_DICT}", torchstore_state_dict)
    latency_tracker.track_step("store_put_tssd")

    await store.put(f"{key}{DELIM}{MAPPING}", torchstore_state_dict.mapping)
    latency_tracker.track_step("store_put_mapping")
    latency_tracker.track_e2e()


async def get_state_dict(
    store, key, user_state_dict: Optional[dict] = None, strict=True
):
    """Unflatten the state dict from the store"""

    try:
        # Since the mapping is the last thing we write out, it also guarantees the state dict is not pending
        fetched_mapping = await store.get(f"{key}{DELIM}{MAPPING}")
    except Exception as e:
        raise RuntimeError(
            f"Mapping is missing from the store. This most likely means there is no matching 'push' call for this key: {key=}"
        ) from e

    user_flattened_state_dict, user_mapping = (
        flatten_state_dict(user_state_dict)
        if user_state_dict is not None
        else ({}, None)
    )
    if strict and user_mapping is not None:
        assert user_mapping == fetched_mapping

    fetched_state_dict = {}
    for flattened_key in fetched_mapping.keys():
        inplace_tensor = user_flattened_state_dict.get(flattened_key, None)
        fetched_state_dict[flattened_key] = await store.get(
            f"{key}{DELIM}{flattened_key}",
            inplace_tensor if isinstance(inplace_tensor, torch.Tensor) else None,
        )

    # # Prepare all the coroutines first
    # coros = []
    # keys = []
    # for flattened_key in fetched_mapping.keys():
    #     inplace_tensor = user_flattened_state_dict.get(flattened_key, None)
    #     keys.append(flattened_key)
    #     coros.append(
    #         store.get(
    #             f"{key}{DELIM}{flattened_key}",
    #             inplace_tensor if isinstance(inplace_tensor, torch.Tensor) else None,
    #         )
    #     )
    # # Run all requests concurrently
    # results = await asyncio.gather(*coros)
    # # Build the result dictionary
    # fetched_state_dict = dict(zip(keys, results))

    return unflatten_state_dict(fetched_state_dict, fetched_mapping)


async def get_state_dict_batch(
    store, key, user_state_dict: Optional[dict] = None, strict=True
):
    # TODO: add support for user_state_dict and strict
    try:
        # Since the mapping is the last thing we write out, it also guarantees the state dict is not pending
        fetched_mapping = await store.get(f"{key}{DELIM}{MAPPING}")
    except Exception as e:
        raise RuntimeError(
            f"Mapping is missing from the store. This most likely means there is no matching 'push' call for this key: {key=}"
        ) from e

    flattened_keys = list(fetched_mapping.keys())
    flattened_state_dict = await store.get_batch(f"{key}{DELIM}", flattened_keys)

    return unflatten_state_dict(flattened_state_dict, fetched_mapping)


@dataclass
class TensorMetadata:
    """Metadata for a tensor in a tensor blob"""

    shape: Tuple[int, ...]
    dtype: torch.dtype
    offset: int  # Byte offset in the blob
    size: int  # Size in bytes
    tensor_slice: TensorSlice | None = None  # TensorSlice for DTensor reconstruction
    device_mesh: Any | None = None  # DeviceMesh for DTensor reconstruction
    placements: Tuple[Any, ...] | None = None  # Placements for DTensor reconstruction


def _state_dict_size(state_dict):
    """Returns the size of the state dict in MBs"""
    size = 0
    sd, _ = flatten_state_dict(state_dict)
    for tensor in sd.values():
        if not isinstance(tensor, torch.Tensor):
            continue

        size += tensor.numel() * tensor.element_size()
    return size // (1024 * 1024)


class TorchStoreStateDict:
    """
    A torchstore representation of a state dict. It contains a flattened state dict and a tensor tensor_blob.
    All of the tensors in the flattened state dict are replaced with TensorMetadata objects.
    """

    def __init__(
        self,
        tensor_blob: torch.Tensor,
        metadata_state_dict: Dict[str, Any],
        mapping: Dict[str, Any],
    ):
        self.tensor_blob = tensor_blob
        self.metadata_state_dict = metadata_state_dict
        self.mapping = mapping

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, Any]) -> "TorchStoreStateDict":
        """Create a TorchStoreStateDict from a state_dict."""
        # store the tensors in one blob
        # 1. flatten the state dict
        flattened_state_dict, mapping = flatten_state_dict(state_dict)

        # 2. iterate through the flattened state dict, collect all tensors and replace them with TensorMetadata objects
        tensor_list: List[Tuple[torch.Tensor, TensorMetadata]] = []
        metadata_state_dict = {}
        current_offset = 0

        for key, value in flattened_state_dict.items():
            if isinstance(value, DTensor):
                # Handle DTensor: store local tensor and add TensorSlice metadata
                local_tensor = value._local_tensor
                tensor_size = local_tensor.numel() * local_tensor.element_size()
                tensor_slice = TensorSlice.from_dtensor(value)

                tensor_metadata = TensorMetadata(
                    shape=tuple(local_tensor.shape),
                    dtype=local_tensor.dtype,
                    offset=current_offset,
                    size=tensor_size,
                    tensor_slice=tensor_slice,
                    device_mesh=value.device_mesh,
                    placements=value.placements,
                )
                tensor_list.append((local_tensor, tensor_metadata))
                metadata_state_dict[key] = tensor_metadata

                current_offset += tensor_size
            elif isinstance(value, torch.Tensor):
                # Handle regular tensor
                tensor_size = value.numel() * value.element_size()
                tensor_metadata = TensorMetadata(
                    shape=tuple(value.shape),
                    dtype=value.dtype,
                    offset=current_offset,
                    size=tensor_size,
                )
                tensor_list.append((value, tensor_metadata))
                metadata_state_dict[key] = tensor_metadata
                current_offset += tensor_size
            else:
                metadata_state_dict[key] = value

        # 3. create the tensor tensor_blob by concatenating all tensors
        if not tensor_list:
            tensor_blob = torch.empty(0, dtype=torch.uint8)
        else:
            tensor_blob = torch.empty(current_offset, dtype=torch.uint8)

            # Copy tensor data
            for tensor, tensor_metadata in tensor_list:
                tensor_cpu = tensor.detach().cpu().contiguous()
                # Convert scalar tensors from 0D to 1D
                if tensor_cpu.dim() == 0:
                    tensor_cpu = tensor_cpu.unsqueeze(0)

                byte_view = tensor_cpu.view(torch.uint8).flatten()

                # Copy to tensor_blob
                tensor_blob[
                    tensor_metadata.offset : tensor_metadata.offset
                    + tensor_metadata.size
                ] = byte_view

        # 4. return the TorchStoreStateDict object
        return cls(tensor_blob, metadata_state_dict, mapping)

    def to_state_dict(self) -> Dict[str, Any]:
        """
        Convert the TorchStoreStateDict back to a state_dict. All TensorMetadata objects are replaced with
        the corresponding tensors from the tensor blob. DTensors are reconstructed using stored metadata.
        """
        state_dict = unflatten_state_dict(
            unpack_metadata_state_dict(self.metadata_state_dict, self.tensor_blob),
            self.mapping,
        )

        return state_dict


def reconstruct_dtensor_from_local_tensor(
    local_tensor: torch.Tensor,
    tensor_slice: "TensorSlice",
    device_mesh: torch.distributed.DeviceMesh,
    placements: Tuple[Placement, ...],
) -> DTensor:
    """
    Reconstruct a DTensor from local tensor data and TensorSlice metadata.

    Args:
        local_tensor: The local tensor shard
        tensor_slice: TensorSlice containing distributed metadata
        device_mesh: The device mesh for the DTensor
        placements: The placements for the DTensor

    Returns:
        Reconstructed DTensor
    """
    return DTensor.from_local(
        local_tensor=local_tensor,
        device_mesh=device_mesh,
        placements=placements,
    )


def unpack_metadata_state_dict(
    metadata_state_dict: Dict[str, Any],
    tensor_blob: torch.Tensor,
) -> Dict[str, Any]:
    """
    Takes a metadata_state_dict and replaces all TensorMetadata objects with
    the corresponding tensors from the tensor blob.
    """
    unpacked_flattened_state_dict = {}

    for key, value in metadata_state_dict.items():
        if isinstance(value, TensorMetadata):
            # Pre-allocate tensor with correct shape and dtype (TorchStore approach)
            tensor = torch.empty(value.shape, dtype=value.dtype)

            # Get byte view of the allocated tensor
            if tensor.dim() == 0:
                tensor_unsqueezed = tensor.unsqueeze(0)
                byte_view = tensor_unsqueezed.view(torch.uint8).flatten()
            else:
                byte_view = tensor.view(torch.uint8).flatten()

            # Copy bytes from blob into tensor's byte view
            # will also modify the data in the actual tensor
            tensor_bytes = tensor_blob[value.offset : value.offset + value.size]
            byte_view.copy_(tensor_bytes)

            # Check if this should be reconstructed as a DTensor
            if (
                value.tensor_slice is not None
                and value.device_mesh is not None
                and value.placements is not None
            ):
                tensor = reconstruct_dtensor_from_local_tensor(
                    local_tensor=tensor,
                    tensor_slice=value.tensor_slice,
                    device_mesh=value.device_mesh,
                    placements=value.placements,
                )

            unpacked_flattened_state_dict[key] = tensor
        else:
            unpacked_flattened_state_dict[key] = value
    return unpacked_flattened_state_dict
