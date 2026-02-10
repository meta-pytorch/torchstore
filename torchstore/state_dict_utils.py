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

from torchstore.client import _DEFAULT_MAX_CONCURRENT

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

    fetched_state_dict = {}
    for flattened_key in fetched_mapping.keys():
        inplace_tensor = user_flattened_state_dict.get(flattened_key, None)
        fetched_state_dict[flattened_key] = await store.get(
            f"{key}{DELIM}{flattened_key}",
            inplace_tensor if isinstance(inplace_tensor, torch.Tensor) else None,
        )

    return unflatten_state_dict(fetched_state_dict, fetched_mapping)


async def put_state_dict_batch(
    store, state_dict, key, max_concurrent=_DEFAULT_MAX_CONCURRENT
):
    """Store a PyTorch state_dict using automatic batching for efficiency.

    This function automatically batches all tensor storage operations into a single
    RPC call to minimize overhead, making it significantly faster than individual
    put() calls for each parameter.

    Args:
        store: TorchStore client instance
        state_dict: PyTorch model state_dict to store
        key: Unique identifier for this state_dict
        max_concurrent: Maximum number of concurrent storage operations.
    """
    flattened_state_dict, mapping = flatten_state_dict(state_dict)

    # Automatically batch all tensor/parameter puts for efficiency
    # This parallelizes storage and uses a single RPC to notify the controller
    items = {
        f"{key}{DELIM}{flattened_key}": value
        for flattened_key, value in flattened_state_dict.items()
    }
    await store._put_batch(items, max_concurrent=max_concurrent)

    # Store mapping last to indicate completion
    await store.put(f"{key}{DELIM}{MAPPING}", mapping)


async def get_state_dict_batch(
    store,
    key,
    user_state_dict: dict | None = None,
    strict=True,
    max_concurrent: int = _DEFAULT_MAX_CONCURRENT,
):
    """Retrieve a state_dict using parallel fetches for efficiency.

    Args:
        store: TorchStore client instance.
        key: Unique identifier for the state_dict.
        user_state_dict: Optional pre-allocated state_dict for in-place retrieval.
        strict: If True, assert that the user mapping matches the stored mapping.
        max_concurrent: Maximum number of concurrent fetch operations.
    """

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
        if user_mapping != fetched_mapping:
            raise ValueError(
                "User state_dict mapping does not match the stored mapping. "
                "Set strict=False to skip this check."
            )

    # Fetch all tensors in parallel with a semaphore to limit concurrency.
    # Without throttling, large models (hundreds of parameters) would create
    # an unbounded number of concurrent transport buffers and connections.
    semaphore = asyncio.Semaphore(max_concurrent)

    async def fetch_one(flattened_key):
        async with semaphore:
            inplace_tensor = user_flattened_state_dict.get(flattened_key, None)
            return await store.get(
                f"{key}{DELIM}{flattened_key}",
                inplace_tensor if isinstance(inplace_tensor, torch.Tensor) else None,
            )

    keys = list(fetched_mapping.keys())
    results = await asyncio.gather(
        *[fetch_one(k) for k in keys], return_exceptions=True
    )

    # Check for fetch failures
    failures = [r for r in results if isinstance(r, BaseException)]
    if failures:
        raise RuntimeError(
            f"{len(failures)}/{len(results)} get operations failed in batch. "
            f"First error: {failures[0]}"
        ) from failures[0]

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
