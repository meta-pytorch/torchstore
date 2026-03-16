# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from dataclasses import dataclass, field
from logging import getLogger
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.checkpoint._nested_dict import (
    flatten_state_dict,
    unflatten_state_dict,
)

DELIM = "/"
MAPPING = "MAPPING"

logger = getLogger(__name__)


@dataclass
class _DirectRDMACache:
    """Per-client cache for direct RDMA weight sync state."""

    source: Any = None  # DirectWeightSyncSource, lazily created
    dest: Any = None  # DirectWeightSyncDest, lazily created
    registered: set = field(default_factory=set)
    handles: dict = field(default_factory=dict)


_rdma_cache: dict[int, _DirectRDMACache] = {}


def _get_rdma_cache(store) -> _DirectRDMACache:
    """Get or create the RDMA cache for a given client."""
    key = id(store)
    if key not in _rdma_cache:
        _rdma_cache[key] = _DirectRDMACache()
    return _rdma_cache[key]


async def put_state_dict(
    store, state_dict, key, direct_rdma=False, transfer_dtype=None
):
    """Store a state dict in TorchStore.

    When ``direct_rdma=False`` (default), each parameter is stored in a
    StorageVolume via the normal two-hop transport path.

    When ``direct_rdma=True``, RDMA handles are registered against the
    caller's GPU memory and only the handles (not tensor data) are stored
    in TorchStore.  On the first call the handles are registered; on
    subsequent calls only non-contiguous staging buffers are refreshed.
    ``state_dict`` may be ``None`` on subsequent calls to skip the
    (potentially expensive) ``model.state_dict()`` construction.
    ``torch.distributed`` must be initialised before using this mode.


    Args:
        transfer_dtype: If set, cast weights to this dtype for transfer.
            Only used with ``direct_rdma=True``. Allows the source to
            keep higher-precision master weights while transferring in a
            lower precision (e.g. bfloat16).
    """
    if direct_rdma:
        await _put_state_dict_direct_rdma(store, state_dict, key, transfer_dtype)
        return

    flattened_state_dict, mapping = flatten_state_dict(state_dict)

    # Batch all tensor entries, then put the mapping separately.
    # The mapping is stored last so it acts as a commit marker for the state dict.
    entries = {f"{key}{DELIM}{k}": v for k, v in flattened_state_dict.items()}
    await store.put_batch(entries)
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
        assert (
            user_state_dict is not None
        ), "user_state_dict is required for direct_rdma mode"
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

    get_id = lambda fk: f"{key}{DELIM}{fk}"
    flattened_keys = list(fetched_mapping.keys())

    get_batch_dict = {}
    for fk in flattened_keys:
        t = user_flattened_state_dict.get(fk, None)
        # inplace can only be a tensor, so skip non-tensor values
        if t is not None and not isinstance(t, torch.Tensor):
            t = None
            logger.warning(f"non-tensor value found for in-place: {fk}")
        get_batch_dict[get_id(fk)] = t

    results = await store.get_batch(get_batch_dict)
    fetched_state_dict = {fk: results[get_id(fk)] for fk in flattened_keys}

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


async def _put_state_dict_direct_rdma(store, state_dict, key, transfer_dtype=None):
    """Register or refresh RDMA handles and publish via TorchStore.

    First call for a given key: registers RDMA handles for each param,
    stores handles in TorchStore as objects.  ``state_dict`` must be
    provided on this first call.

    Subsequent calls: refreshes staging buffers for non-contiguous params.
    ``state_dict`` may be ``None`` to skip building it when only a refresh
    is needed.
    """
    from torchstore.direct_weight_sync import DirectWeightSyncSource

    cache = _get_rdma_cache(store)

    if cache.source is None:
        cache.source = DirectWeightSyncSource()

    if key not in cache.registered:
        assert (
            state_dict is not None
        ), "state_dict is required on first put_state_dict call with direct_rdma=True"
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        handles = cache.source.register(
            state_dict, rank=rank, transfer_dtype=transfer_dtype
        )
        await store.put(f"{key}/rank_{rank}", handles)
        if rank == 0:
            await store.put(f"{key}/num_ranks", world_size)
        cache.registered.add(key)
    else:
        cache.source.refresh()


async def _get_state_dict_direct_rdma(store, key, user_state_dict):
    """Fetch RDMA handles and pull weights via direct RDMA reads.

    First call for a given key: fetches handles from TorchStore, caches them.
    Transfer plan is built on first pull and cached by DirectWeightSyncDest.
    All RDMA reads are issued concurrently for maximum throughput.
    """
    from torchstore.direct_weight_sync import DirectWeightSyncDest

    cache = _get_rdma_cache(store)

    if cache.dest is None:
        cache.dest = DirectWeightSyncDest()

    if key not in cache.handles:
        num_ranks = await store.get(f"{key}/num_ranks")
        all_handles = defaultdict(list)
        for r in range(num_ranks):
            rank_handles = await store.get(f"{key}/rank_{r}")
            for name, handle in rank_handles.items():
                all_handles[name].append(handle)
        cache.handles[key] = all_handles

    await cache.dest.pull(cache.handles[key], user_state_dict)
