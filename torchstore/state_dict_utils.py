# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from dataclasses import dataclass
from logging import getLogger
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from torch.distributed.checkpoint._nested_dict import (
    flatten_state_dict,
    unflatten_state_dict,
)
from torch.distributed.tensor import DTensor

from torchstore.dtensor_utils import create_tensor_slice_from_dtensor
from torchstore.transport.pipe import TensorSlice

DELIM = "/"

# Resereved key segments for torchstore internal handling of state dict
MAPPING = "__TORCHSTORE_STATE_DICT_MAPPING"
TENSOR_BLOB = "__TORCHSTORE_STATE_DICT_TENSOR_BLOB"
FLATTENED_STATE_DICT = "__TORCHSTORE_STATE_DICT_FLATTENED_STATE_DICT"

TORCHSTORE_TSSD_ENABLED_FLAG = "TORCHSTORE_TSSD_ENABLED"

logger = getLogger(__name__)


def tssd_enabled() -> bool:
    """
    Check if TorchStoreStateDict is enabled for put and get. If enabled, we will use the
    TSSD to batch tensors in the state dict into one blob and transfer it more efficiently.
    """

    return os.environ.get(TORCHSTORE_TSSD_ENABLED_FLAG, "0") == "1"


def is_tssd_key(key: str) -> bool:
    """
    Check if a key is a TorchStoreStateDict key. This is used to determine if we should use
    the TSSD method for put and get.
    """
    return (
        key.endswith(DELIM + MAPPING)
        or key.endswith(DELIM + TENSOR_BLOB)
        or key.endswith(DELIM + FLATTENED_STATE_DICT)
    )


def tssd_keys(state_dict_key: str) -> Set[str]:
    """
    Get all TorchStoreStateDict keys for a given key. This is used to determine if we should use
    the TSSD method for put and get.
    Args:
        state_dict_key: The key of the whole state dict without any internal segments.
    """
    return {
        state_dict_key + DELIM + MAPPING,
        state_dict_key + DELIM + TENSOR_BLOB,
        state_dict_key + DELIM + FLATTENED_STATE_DICT,
    }


def get_state_dict_key(key: str) -> str:
    """
    Get the key of the whole state dict from a TorchStoreStateDict key. This is used to determine if we should use
    the TSSD method for put and get.
    Args:
        key: The key of the whole state dict without any internal segments.
    """
    return key.split(DELIM)[0]


async def put_state_dict(store, state_dict, key):
    """
    Store a state dict using either the original method or TorchStoreStateDict.

    Args:
        store: The torchstore instance to store data in
        state_dict: The state dictionary to store
        key: The key prefix to store under
    """
    if tssd_enabled():
        # Use TorchStoreStateDict method for efficient tensor serialization
        torchstore_state_dict = TorchStoreStateDict.from_state_dict(state_dict)

        # Store the tensor blob
        await store.put(f"{key}{DELIM}{TENSOR_BLOB}", torchstore_state_dict.tensor_blob)

        # Store the flattened state dict (contains TensorReferences and non-tensor data)
        await store.put(
            f"{key}{DELIM}{FLATTENED_STATE_DICT}",
            torchstore_state_dict.flattened_state_dict,
        )

        # Store the mapping (this serves as the completion indicator)
        await store.put(f"{key}{DELIM}{MAPPING}", torchstore_state_dict.mapping)
    else:
        # Original method: flatten and store each tensor individually
        flattened_state_dict, mapping = flatten_state_dict(state_dict)
        for flattened_key, value in flattened_state_dict.items():
            await store.put(f"{key}{DELIM}{flattened_key}", value)

        await store.put(f"{key}{DELIM}{MAPPING}", mapping)


async def get_state_dict(
    store,
    key,
    user_state_dict: Optional[dict] = None,
    strict=True,
):
    """
    Get a state dict from the store using either the original method or TorchStoreStateDict.

    Args:
        store: The torchstore instance to get data from
        key: The key prefix to retrieve from
        user_state_dict: Optional user state dict for validation/inplace tensors
        strict: Whether to strictly validate mappings
    """
    try:
        # Since the mapping is the last thing we write out, it also guarantees the state dict is not pending
        fetched_mapping = await store.get(f"{key}{DELIM}{MAPPING}")
    except Exception as e:
        raise RuntimeError(
            f"Mapping is missing from the store. This most likely means there is no matching 'push' call for this key: {key=}"
        ) from e

    if False:
        # Use TorchStoreStateDict method for efficient retrieval
        try:
            # Get the tensor blob and flattened state dict
            tensor_blob = await store.get(f"{key}{DELIM}{TENSOR_BLOB}")
            flattened_state_dict = await store.get(
                f"{key}{DELIM}{FLATTENED_STATE_DICT}"
            )

            # Reconstruct TorchStoreStateDict and convert back to state dict
            torchstore_state_dict = TorchStoreStateDict(
                tensor_blob=tensor_blob,
                flattened_state_dict=flattened_state_dict,
                mapping=fetched_mapping,
            )

            return torchstore_state_dict.to_state_dict()

        except Exception as e:
            raise RuntimeError(
                f"Failed to retrieve TorchStoreStateDict data for key: {key=}"
            ) from e
    else:
        # Original method: get each tensor individually
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

        return unflatten_state_dict(fetched_state_dict, fetched_mapping)


def _state_dict_size(state_dict):
    """Returns the size of the state dict in MBs"""
    size = 0
    sd, _ = flatten_state_dict(state_dict)
    for tensor in sd.values():
        if not isinstance(tensor, torch.Tensor):
            continue

        size += tensor.numel() * tensor.element_size()
    return size // (1024 * 1024)


@dataclass
class TensorReference:
    """Metadata for a tensor in a tensor blob"""

    shape: Tuple[int, ...]
    dtype: torch.dtype
    offset: int  # Byte offset in the blob
    size: int  # Size in bytes
    tensor_slice: TensorSlice | None = None  # TensorSlice for DTensor reconstruction
    device_mesh: Any | None = None  # DeviceMesh for DTensor reconstruction
    placements: Tuple[Any, ...] | None = None  # Placements for DTensor reconstruction


class TorchStoreStateDict:
    """
    A torchstore representation of a state dict. It contains a flattened state dict and a tensor blob.
    All of the tensors in the flattened state dict are replaced with TensorReference objects.
    """

    def __init__(
        self,
        tensor_blob: torch.Tensor,
        flattened_state_dict: Dict[str, Any],
        mapping: Dict[str, Any],
    ):
        self.tensor_blob = tensor_blob
        self.flattened_state_dict = flattened_state_dict
        self.mapping = mapping

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, Any]) -> "TorchStoreStateDict":
        """
        Create a TorchStoreStateDict from a state_dict. All tensors in the state_dict are replaced with
        TensorReference objects. The tensor blob is created by concatenating all tensors in the state_dict.
        """
        # 1. flatten the state dict
        flattened_state_dict, mapping = flatten_state_dict(state_dict)

        # 2. iterate through the flattened state dict, collect all tensors and replace them with TensorReference objects
        tensor_list: List[Tuple[torch.Tensor, TensorReference]] = []
        modified_flattened_state_dict = {}
        current_offset = 0

        for key, value in flattened_state_dict.items():
            if isinstance(value, DTensor):
                # Handle DTensor: store local tensor and add TensorSlice metadata
                local_tensor = value._local_tensor
                tensor_size = local_tensor.numel() * local_tensor.element_size()
                tensor_slice = create_tensor_slice_from_dtensor(value)

                ref = TensorReference(
                    shape=tuple(local_tensor.shape),
                    dtype=local_tensor.dtype,
                    offset=current_offset,
                    size=tensor_size,
                    tensor_slice=tensor_slice,
                    device_mesh=value.device_mesh,
                    placements=value.placements,
                )
                tensor_list.append((local_tensor, ref))
                modified_flattened_state_dict[key] = ref
                current_offset += tensor_size
            elif isinstance(value, torch.Tensor):
                # Handle regular tensor
                tensor_size = value.numel() * value.element_size()
                ref = TensorReference(
                    shape=tuple(value.shape),
                    dtype=value.dtype,
                    offset=current_offset,
                    size=tensor_size,
                )
                tensor_list.append((value, ref))
                modified_flattened_state_dict[key] = ref
                current_offset += tensor_size
            else:
                modified_flattened_state_dict[key] = value

        # 3. create the tensor blob by concatenating all tensors
        if not tensor_list:
            blob = torch.empty(0, dtype=torch.uint8)
        else:
            blob = torch.empty(current_offset, dtype=torch.uint8)

            # Copy tensor data
            for tensor, ref in tensor_list:
                # Handle scalar tensors
                tensor_cpu = tensor.detach().cpu()
                if tensor_cpu.dim() == 0:
                    tensor_cpu = tensor_cpu.unsqueeze(0)

                byte_view = tensor_cpu.view(torch.uint8).flatten()

                # Copy to blob
                blob[ref.offset : ref.offset + ref.size] = byte_view

        # 4. return the TorchStoreStateDict object
        return cls(blob, modified_flattened_state_dict, mapping)

    def to_state_dict(self) -> Dict[str, Any]:
        """
        Convert the TorchStoreStateDict back to a state_dict. All TensorReference objects are replaced with
        the corresponding tensors from the tensor blob. DTensors are reconstructed using stored metadata.
        """
        state_dict = unflatten_state_dict(
            deref_flattened_state_dict(self.flattened_state_dict, self.tensor_blob),
            self.mapping,
        )

        # 3. return the state dict
        return state_dict


def deref_flattened_state_dict(
    flattened_state_dict: Dict[str, Any],
    tensor_blob: torch.Tensor,
) -> Dict[str, Any]:
    from torchstore.dtensor_utils import reconstruct_dtensor_from_local_tensor

    """
    Dereference a flattened state dict. All TensorReference objects are replaced with
    the corresponding tensors from the tensor blob.
    """
    derefed_flattened_state_dict = {}

    for key, value in flattened_state_dict.items():
        if isinstance(value, TensorReference):
            # Pre-allocate tensor with correct shape and dtype (TorchStore approach)
            tensor = torch.empty(value.shape, dtype=value.dtype)

            # Get byte view of the allocated tensor
            if tensor.dim() == 0:
                tensor_unsqueezed = tensor.unsqueeze(0)
                byte_view = tensor_unsqueezed.view(torch.uint8).flatten()
            else:
                byte_view = tensor.view(torch.uint8).flatten()

            # Copy bytes from blob into tensor's byte view
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

            derefed_flattened_state_dict[key] = tensor
        else:
            derefed_flattened_state_dict[key] = value
    return derefed_flattened_state_dict
