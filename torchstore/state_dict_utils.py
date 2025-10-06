# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from logging import getLogger
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.distributed.checkpoint._nested_dict import (
    flatten_state_dict,
    unflatten_state_dict,
)

DELIM = "/"
MAPPING = "MAPPING"

logger = getLogger(__name__)


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


async def get_state_dict(
    store, key, user_state_dict: Optional[dict] = None, strict=True
):
    """Unflatten the state dict from the store"""

    try:
        # Since the mapping is the last thing we write out, it also gaurantees the state dict is not pending
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


def generate_tensor_blob(state_dict: Dict[str, Any]):
    """
    Extract all tensors from state_dict and create a blob. Replace the tensors
    with corresponding references and returns a state_dict with only tensor references,
    and the tensor blob.

    Args:
      state_dict: Dictionary that may contain tensors at any level

    Returns:
      - Modified dictionary with tensors replaced by TensorReference objects
      - 1D uint8 tensor blob containing all serialized tensor data
    """

    def _extract_recursive(
        obj: Dict[str, Any],
        tensor_list: List[Tuple[torch.Tensor, TensorReference]],
        path: str = "",
    ):
        """Recursively extract tensors and replace with TensorReference objects"""
        if isinstance(obj, torch.Tensor):
            # Create placeholder reference (offset will be filled later)
            ref = TensorReference(
                shape=tuple(obj.shape),
                dtype=obj.dtype,
                offset=-1,  # Will be updated when building blob
                size=obj.numel() * obj.element_size(),
            )
            tensor_list.append((obj, ref))
            return ref  # Replace tensor with TensorReference
        elif isinstance(obj, dict):
            return {
                k: _extract_recursive(v, tensor_list, f"{path}.{k}")
                for k, v in obj.items()
            }
        elif isinstance(obj, (list, tuple)):
            return type(obj)(
                _extract_recursive(item, tensor_list, f"{path}[{i}]")
                for i, item in enumerate(obj)
            )
        else:
            return obj  # Non-tensor data stays as-is

    tensor_list: List[Tuple[torch.Tensor, TensorReference]] = []

    modified_state_dict = _extract_recursive(state_dict, tensor_list)

    if not tensor_list:
        return modified_state_dict, torch.empty(0, dtype=torch.uint8)

    total_bytes = sum([ref.size for _, ref in tensor_list])

    blob = torch.empty(total_bytes, dtype=torch.uint8)

    # Copy tensor data using your efficient approach
    for tensor, ref in tensor_list:
        # Handle scalar tensors
        tensor_cpu = tensor.detach().cpu()
        if tensor_cpu.dim() == 0:
            tensor_cpu = tensor_cpu.unsqueeze(0)

        byte_view = tensor_cpu.view(torch.uint8).flatten()

        # Copy to blob
        blob[ref.offset : ref.offset + ref.size] = byte_view

    return modified_state_dict, blob


def reconstruct_state_dict_from_tensor_blob(
    state_dict_with_tensor_refs: Dict[str, Any], blob: torch.Tensor
) -> Dict[str, Any]:
    """
    Reconstruct a state_dict which only contains tensor references by
    reconstructing the tensors using the tensor blob and the tensor references.
    Returns the reconstructed state dict.
    """

    def _reconstruct_recursive(obj):
        if isinstance(obj, TensorReference):
            # Pre-allocate tensor with correct shape and dtype (TorchStore approach)
            tensor = torch.empty(obj.shape, dtype=obj.dtype)

            # Get byte view of the allocated tensor
            if tensor.dim() == 0:
                tensor_unsqueezed = tensor.unsqueeze(0)
                byte_view = tensor_unsqueezed.view(torch.uint8).flatten()
            else:
                byte_view = tensor.view(torch.uint8).flatten()

            # Copy bytes from blob into tensor's byte view
            tensor_bytes = blob[obj.offset : obj.offset + obj.size]
            byte_view.copy_(tensor_bytes)

            return tensor
        elif isinstance(obj, dict):
            return {k: _reconstruct_recursive(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return type(obj)(_reconstruct_recursive(item) for item in obj)
        else:
            return obj

    return _reconstruct_recursive(state_dict_with_tensor_refs)
