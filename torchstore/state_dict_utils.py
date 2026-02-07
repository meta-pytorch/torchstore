# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
from logging import getLogger

import torch
from torch.distributed.checkpoint._nested_dict import (
    flatten_state_dict,
    unflatten_state_dict,
)

DELIM = "/"
MAPPING = "MAPPING"

logger = getLogger(__name__)


async def put_state_dict(store, state_dict, key):
    """Store a PyTorch state_dict using automatic batching for efficiency.

    This function automatically batches all tensor storage operations into a single
    RPC call to minimize overhead, making it significantly faster than individual
    put() calls for each parameter.

    Args:
        store: TorchStore client instance
        state_dict: PyTorch model state_dict to store
        key: Unique identifier for this state_dict
    """
    flattened_state_dict, mapping = flatten_state_dict(state_dict)

    # Automatically batch all tensor/parameter puts for efficiency
    # This parallelizes storage and uses a single RPC to notify the controller
    items = {f"{key}{DELIM}{flattened_key}": value
             for flattened_key, value in flattened_state_dict.items()}
    await store._put_batch(items)

    # Store mapping last to indicate completion
    await store.put(f"{key}{DELIM}{MAPPING}", mapping)


async def get_state_dict(store, key, user_state_dict: dict | None = None, strict=True):
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

    # Fetch all tensors in parallel for efficiency
    coros = []
    keys = []
    for flattened_key in fetched_mapping.keys():
        inplace_tensor = user_flattened_state_dict.get(flattened_key, None)
        keys.append(flattened_key)
        coros.append(
            store.get(
                f"{key}{DELIM}{flattened_key}",
                inplace_tensor if isinstance(inplace_tensor, torch.Tensor) else None,
            )
        )
    # Run all requests concurrently
    results = await asyncio.gather(*coros)
    # Build the result dictionary
    fetched_state_dict = dict(zip(keys, results))

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
