# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from logging import getLogger

import torch
import torch.distributed as dist
from torch.distributed.checkpoint._nested_dict import (
    flatten_state_dict,
    unflatten_state_dict,
)

DELIM = "/"
MAPPING = "MAPPING"

logger = getLogger(__name__)


async def put_state_dict(store, state_dict, key, direct_rdma=False):
    """Store a state dict in TorchStore.

    When ``direct_rdma=False`` (default), each parameter is stored in a
    StorageVolume via the normal two-hop transport path.

    When ``direct_rdma=True``, RDMA handles are registered against the
    caller's GPU memory and only the handles (not tensor data) are stored
    in TorchStore.  On the first call the handles are registered; on
    subsequent calls only non-contiguous staging buffers are refreshed.
    ``torch.distributed`` must be initialised before using this mode.
    """
    if direct_rdma:
        await _put_state_dict_direct_rdma(store, state_dict, key)
        return

    flattened_state_dict, mapping = flatten_state_dict(state_dict)
    for flattened_key, value in flattened_state_dict.items():
        await store.put(f"{key}{DELIM}{flattened_key}", value)

    await store.put(f"{key}{DELIM}{MAPPING}", mapping)


async def get_state_dict(
    store, key, user_state_dict: dict | None = None, strict=True, direct_rdma=False
):
    """Retrieve a state dict from TorchStore.

    When ``direct_rdma=False`` (default), each parameter is fetched from
    a StorageVolume via the normal two-hop transport path.

    When ``direct_rdma=True``, RDMA handles are fetched from TorchStore
    (first call only, cached afterwards) and weights are pulled directly
    from the source's GPU memory via one-sided RDMA reads.
    ``user_state_dict`` must be provided in this mode so the destination
    tensors are available for in-place writes.
    """
    if direct_rdma:
        assert user_state_dict is not None, (
            "user_state_dict is required for direct_rdma mode"
        )
        await _get_state_dict_direct_rdma(store, key, user_state_dict)
        return user_state_dict

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


# ---------------------------------------------------------------------------
# Direct RDMA helpers (used when direct_rdma=True)
# ---------------------------------------------------------------------------


async def _put_state_dict_direct_rdma(store, state_dict, key):
    """Register or refresh RDMA handles and publish via TorchStore.

    First call for a given key: registers RDMA handles for each param,
    stores handles in TorchStore as objects.
    Subsequent calls: refreshes staging buffers for non-contiguous params.
    """
    from torchstore.direct_weight_sync import DirectWeightSyncSource

    if store._direct_rdma_source is None:
        store._direct_rdma_source = DirectWeightSyncSource()

    if key not in store._direct_rdma_registered:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        handles = store._direct_rdma_source.register(state_dict, rank=rank)
        await store.put(f"{key}/rank_{rank}", handles)
        if rank == 0:
            await store.put(f"{key}/num_ranks", world_size)
        store._direct_rdma_registered.add(key)
    else:
        store._direct_rdma_source.refresh()


async def _get_state_dict_direct_rdma(store, key, user_state_dict):
    """Fetch RDMA handles and pull weights via direct RDMA reads.

    First call for a given key: fetches handles from TorchStore, caches them.
    Transfer plan is built on first pull and cached by DirectWeightSyncDest.
    All RDMA reads are issued concurrently for maximum throughput.
    """
    from torchstore.direct_weight_sync import DirectWeightSyncDest

    if store._direct_rdma_dest is None:
        store._direct_rdma_dest = DirectWeightSyncDest()

    if key not in store._direct_rdma_handles:
        num_ranks = await store.get(f"{key}/num_ranks")
        all_handles = defaultdict(list)
        for r in range(num_ranks):
            rank_handles = await store.get(f"{key}/rank_{r}")
            for name, handle in rank_handles.items():
                all_handles[name].append(handle)
        store._direct_rdma_handles[key] = all_handles

    await store._direct_rdma_dest.pull(
        store._direct_rdma_handles[key], user_state_dict
    )
