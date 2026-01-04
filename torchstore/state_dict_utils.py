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
from torch.distributed.tensor import DTensor
from torchstore.transport.pipe import TensorSlice

DELIM = "/"
MAPPING = "MAPPING"

# Reserved keys for torchstore internal handling of state dict
TENSOR_BLOB = "TORCHSTORE_STATE_DICT_TENSOR_BLOB"
FLATTENED_STATE_DICT = "TORCHSTORE_STATE_DICT_FLATTENED_STATE_DICT"

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


async def put_state_dict_batch(store, state_dict, key):
    """
    Turning state dict names into a single key.
    """
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

    # see if state_dict was stored as one single key with TorchStoreStateDict
    try:
        # Get the tensor blob and flattened state dict
        tensor_blob = await store.get(f"{key}{DELIM}{TENSOR_BLOB}")
        flattened_state_dict = await store.get(f"{key}{DELIM}{FLATTENED_STATE_DICT}")

        # Reconstruct TorchStoreStateDict and convert back to state dict
        torchstore_state_dict = TorchStoreStateDict(
            tensor_blob=tensor_blob,
            flattened_state_dict=flattened_state_dict,
            mapping=fetched_mapping,
        )

        import fbvscode

        fbvscode.set_trace()

        return torchstore_state_dict.to_state_dict()
    except Exception as e:
        logger.info(
            f"Failed to retrieve TorchStoreStateDict data for key: {key=}. Falling back to getting each tenso individually."
        )
        # TODO: Deprecate this code path eventually once we have fully moved to TorchStoreStateDict
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
        print(f"tenosr blob: {self.tensor_blob}")
        print(f"flattened state dict: {self.flattened_state_dict}")

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, Any]) -> "TorchStoreStateDict":
        """Create a TorchStoreStateDict from a state_dict."""
        # store the tensors in one blob
        # 1. flatten the state dict
        flattened_state_dict, mapping = flatten_state_dict(state_dict)
        print(flattened_state_dict)
        print(mapping)

        # 2. iterate through the flattened state dict, collect all tensors and replace them with TensorReference objects
        tensor_list: List[Tuple[torch.Tensor, TensorReference]] = []
        modified_flattened_state_dict = {}
        current_offset = 0

        for key, value in flattened_state_dict.items():
            if isinstance(value, DTensor):
                raise NotImplementedError("DTensor is not supported yet")
                # # Handle DTensor: store local tensor and add TensorSlice metadata
                # local_tensor = value._local_tensor
                # tensor_size = local_tensor.numel() * local_tensor.element_size()
                # tensor_slice = create_tensor_slice_from_dtensor(value)

                # ref = TensorReference(
                #     shape=tuple(local_tensor.shape),
                #     dtype=local_tensor.dtype,
                #     offset=current_offset,
                #     size=tensor_size,
                #     tensor_slice=tensor_slice,
                #     device_mesh=value.device_mesh,
                #     placements=value.placements,
                # )
                # tensor_list.append((local_tensor, ref))
                # modified_flattened_state_dict[key] = ref
                # current_offset += tensor_size
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
                # Convert scalar tensors from 0D to 1D
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

        return state_dict


def deref_flattened_state_dict(
    flattened_state_dict: Dict[str, Any],
    tensor_blob: torch.Tensor,
) -> Dict[str, Any]:
    # from torchstore.dtensor_utils import reconstruct_dtensor_from_local_tensor

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
            # will also modify the data in the actual tensor
            tensor_bytes = tensor_blob[value.offset : value.offset + value.size]
            byte_view.copy_(tensor_bytes)

            # Check if this should be reconstructed as a DTensor
            # todo: add DTensor support
            # if (
            #     value.tensor_slice is not None
            #     and value.device_mesh is not None
            #     and value.placements is not None
            # ):
            #     tensor = reconstruct_dtensor_from_local_tensor(
            #         local_tensor=tensor,
            #         tensor_slice=value.tensor_slice,
            #         device_mesh=value.device_mesh,
            #         placements=value.placements,
            #     )

            derefed_flattened_state_dict[key] = tensor
        else:
            derefed_flattened_state_dict[key] = value
    return derefed_flattened_state_dict
