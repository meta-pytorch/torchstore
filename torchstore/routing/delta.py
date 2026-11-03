# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Delta state-dict sync over precomputed routes.

Every version, a publisher puts the full state dict, then one fixed-size packed
row per tensor shard holding only the elements whose wire bytes changed since
the previous version:

    [header: int64 x 3][values: wire dtype x cap][indices: int32 x cap]

``indices`` are flat offsets into the publisher's local shard. The header holds
the number of changed elements (-1 when they do not fit in ``cap``, and on the
first version), the number of put versions, and a random epoch per publisher
instance. Rows live under ``{key}/delta/{flat_key}`` as a
uint8 tensor of global shape ``(*sharded mesh axis sizes, nbytes)``, so each
shard's row sits at its mesh coordinates and replicas collapse to one row.

A requester holding the previous version fetches the rows overlapping its own
slice, scatters the changed values, and fetches the full slice of any tensor
whose row overflowed or that it did not get at the previous version of the
same publishers. Values are absolute, so applying a delta is bitwise exact.

Puts and gets of one key must not run concurrently: rows are overwritten in
place, so a concurrent get can read a mix of two versions. Once a key has
deltas, put every version of it with :func:`put_state_dict_delta`: a version
put with ``put_state_dict`` alone leaves the rows at the previous version, and
requesters keep the older data.
"""

from __future__ import annotations

import logging
import math
import os
import weakref
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch.distributed.checkpoint._nested_dict import flatten_state_dict
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.placement_types import _StridedShard

from torchstore.state_dict_utils import (
    _state_dict_storage_key,
    _wire_dtype,
    DELIM,
    get_state_dict,
    put_state_dict,
)
from torchstore.transport.types import _is_dtensor_fully_local, Request, TensorSlice
from torchstore.utils import get_slice_intersection
from ._model import KeyRegistration
from .client import RoutingClient

__all__ = ["StateDictDelta", "get_state_dict_delta", "put_state_dict_delta"]

logger = logging.getLogger(__name__)

DELTA = "delta"
# Row header: [count, version, epoch].
_HEADER_DTYPE = torch.int64
_HEADER_BYTES = 3 * _HEADER_DTYPE.itemsize
_INDEX_DTYPE = torch.int32
_OVERFLOW = -1
_WIRE_DTYPES = (
    torch.float32,
    torch.bfloat16,
    torch.float16,
    torch.float64,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.int64,
    torch.int32,
    torch.int16,
    torch.int8,
    torch.uint8,
    torch.bool,
)
# Bitwise comparison dtype per element size.
_BITS_DTYPES = {
    1: torch.uint8,
    2: torch.int16,
    4: torch.int32,
    8: torch.int64,
}


def _delta_key(key: str) -> str:
    return f"{key}{DELIM}{DELTA}"


def _row_bytes(capacity: int, dtype: torch.dtype) -> int:
    return capacity * (dtype.itemsize + _INDEX_DTYPE.itemsize) + _HEADER_BYTES


def _row_capacity(nbytes: int, dtype: torch.dtype) -> int:
    return (nbytes - _HEADER_BYTES) // (dtype.itemsize + _INDEX_DTYPE.itemsize)


def _shard_grid(value: torch.Tensor) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """This shard's coordinates and the mesh sizes over its sharded mesh axes.

    Replicated mesh axes are dropped, so replicas share one coordinate.
    """
    if not isinstance(value, DTensor) or _is_dtensor_fully_local(value):
        return (), ()
    mesh = value.device_mesh
    coordinate = mesh.get_coordinate()
    assert coordinate is not None, "rank is not part of the DTensor mesh"
    for placement in value.placements:
        # The row only needs a unique coordinate. Elements are mapped through
        # the box the full put registers, which is exact for _StridedShard when
        # each rank holds one contiguous box, as FSDP over TP does.
        if not isinstance(placement, (Replicate, Shard, _StridedShard)):
            raise ValueError(f"unsupported placement {placement} for a delta")
    axes = [
        axis for axis, p in enumerate(value.placements) if not isinstance(p, Replicate)
    ]
    return (
        tuple(coordinate[axis] for axis in axes),
        tuple(mesh.size(axis) for axis in axes),
    )


@dataclass
class _Entry:
    """Publisher state for one flat key."""

    wire_dtype: torch.dtype
    capacity: int
    # Previous version in the wire dtype, in host memory.
    previous: torch.Tensor
    packed: torch.Tensor
    # Host copy of ``packed`` that is put; ``packed`` itself on CPU sources.
    host_packed: torch.Tensor
    row_slice: TensorSlice


class StateDictDelta:
    """Publisher-side encoder of per-version state-dict deltas.

    Keeps a copy of the previous version in the wire dtype in host memory, and
    fixed-size packed buffers. Accelerator tensors are compared on their device,
    copying the previous version there one tensor at a time. Pass it to
    :func:`put_state_dict_delta` with each new version.

    Args:
        state_dict: Template with the same keys, shapes, placements and devices
            as every put version.
        max_fraction: Fraction of a shard's elements that fit in its row. A
            version changing more elements sends the full tensor instead.
            Rows are fixed size, so a get reads about
            ``max_fraction * (itemsize + 4) / itemsize`` of the bytes of the
            publisher shards it overlaps, and at least 8 elements plus a
            24-byte header per shard.
        transfer_dtype: Wire dtype of floating-point tensors, as in
            ``put_state_dict``.
        preserve_dtype_keys: Flat keys that keep their own dtype, as in
            ``put_state_dict``.
    """

    def __init__(
        self,
        state_dict: Mapping[str, Any],
        *,
        max_fraction: float,
        transfer_dtype: torch.dtype | None = None,
        preserve_dtype_keys: frozenset[str] = frozenset(),
    ) -> None:
        if not 0 < max_fraction <= 1:
            raise ValueError(f"max_fraction must be in (0, 1], got {max_fraction}")
        flattened, _mapping = flatten_state_dict(dict(state_dict))
        self._transfer_dtype = transfer_dtype
        self._preserve_dtype_keys = preserve_dtype_keys
        self._entries: dict[str, _Entry] = {}
        self._version = 0
        # Distinguishes publisher instances that reuse version numbers.
        self._epoch = int.from_bytes(os.urandom(8), "little") >> 1
        for flat_key, value in flattened.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"state dict delta only supports tensors; {flat_key!r} has "
                    f"type {type(value)}"
                )
            local = value._local_tensor if isinstance(value, DTensor) else value
            wire_dtype = _wire_dtype(
                flat_key, local, transfer_dtype, preserve_dtype_keys
            )
            if wire_dtype not in _WIRE_DTYPES:
                raise TypeError(f"unsupported dtype {wire_dtype} for {flat_key!r}")
            if local.numel() > torch.iinfo(_INDEX_DTYPE).max:
                raise ValueError(f"{flat_key!r} has too many elements per shard")
            coordinate, grid = _shard_grid(value)
            # A pure function of the global tensor, so every shard's row has
            # the same size. Multiple of 8 keeps each region of a row aligned.
            per_shard = math.ceil(value.numel() / math.prod(grid))
            capacity = 8 * max(1, math.ceil(max_fraction * per_shard / 8))
            nbytes = _row_bytes(capacity, wire_dtype)
            packed = torch.empty(nbytes, dtype=torch.uint8, device=local.device)
            host_packed = (
                packed
                if local.device.type == "cpu"
                else torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
            )
            self._entries[flat_key] = _Entry(
                wire_dtype=wire_dtype,
                capacity=capacity,
                previous=torch.empty(
                    local.shape,
                    dtype=wire_dtype,
                    pin_memory=local.device.type != "cpu",
                ),
                packed=packed,
                host_packed=host_packed,
                row_slice=TensorSlice(
                    offsets=coordinate + (0,),
                    coordinates=(),
                    global_shape=grid + (nbytes,),
                    local_shape=(1,) * len(grid) + (nbytes,),
                    mesh_shape=(),
                ),
            )

    @torch.no_grad()
    def _stage(self, state_dict: Mapping[str, Any]) -> None:
        """Encode ``state_dict`` against the previous version and copy it to host."""
        flattened, _mapping = flatten_state_dict(dict(state_dict))
        if flattened.keys() != self._entries.keys():
            raise ValueError("state dict keys do not match the delta template")
        devices = set()
        for flat_key, entry in self._entries.items():
            value = flattened[flat_key]
            local = value._local_tensor if isinstance(value, DTensor) else value
            wire = local.to(entry.wire_dtype).contiguous()
            capacity = entry.capacity
            values_end = _HEADER_BYTES + capacity * entry.wire_dtype.itemsize
            indices_end = values_end + capacity * _INDEX_DTYPE.itemsize
            header = entry.packed[:_HEADER_BYTES].view(_HEADER_DTYPE)
            count = header[0]
            header[1].fill_(self._version + 1)
            header[2].fill_(self._epoch)
            if self._version == 0:
                count.fill_(_OVERFLOW)
            elif wire.numel() == 0:
                count.zero_()
            else:
                bits_dtype = _BITS_DTYPES[entry.wire_dtype.itemsize]
                previous = entry.previous.to(local.device, non_blocking=True)
                changed = (wire.view(bits_dtype) != previous.view(bits_dtype)).flatten()
                num_changed = torch.count_nonzero(changed)
                indices = torch.nonzero_static(changed, size=capacity, fill_value=0)
                indices = indices.squeeze(1)
                entry.packed[_HEADER_BYTES:values_end].view(entry.wire_dtype).copy_(
                    wire.flatten()[indices]
                )
                entry.packed[values_end:indices_end].view(_INDEX_DTYPE).copy_(indices)
                count.copy_(torch.where(num_changed > capacity, _OVERFLOW, num_changed))
            # Stream-ordered after the read of the previous version above.
            entry.previous.copy_(wire, non_blocking=True)
            if entry.host_packed is not entry.packed:
                entry.host_packed.copy_(entry.packed, non_blocking=True)
                devices.add(entry.packed.device)
        # Not the whole device: callers may overlap work on other streams.
        for device in devices:
            torch.accelerator.current_stream(device).synchronize()
        self._version += 1


async def put_state_dict_delta(
    store: RoutingClient,
    delta: StateDictDelta,
    state_dict: Mapping[str, Any],
    key: str,
) -> None:
    """Put a new version ``state_dict`` of ``key``: in full, then its delta.

    Requesters read the full tensors when a delta does not apply. Encoding
    runs synchronously on the calling thread.
    """
    if not isinstance(store, RoutingClient):
        raise TypeError("state dict deltas require a routing client")
    delta._stage(state_dict)
    # Full first: a requester that sees a row's version reads the full tensors
    # of at least that version.
    await put_state_dict(
        store,
        state_dict,
        key,
        transfer_dtype=delta._transfer_dtype,
        preserve_dtype_keys=delta._preserve_dtype_keys,
    )
    delta_key = _delta_key(key)
    if delta_key not in store._registered_layouts:
        await store._register_publisher(
            delta_key,
            {
                _state_dict_storage_key(delta_key, flat_key): KeyRegistration(
                    entry.row_slice, dtype=torch.uint8
                )
                for flat_key, entry in delta._entries.items()
            },
        )
    await store.put_batch(
        {
            _state_dict_storage_key(delta_key, flat_key): entry.host_packed.view(
                entry.row_slice.local_shape
            )
            for flat_key, entry in delta._entries.items()
        }
    )


@dataclass
class _Route:
    """Requester plan for one flat key."""

    destination_slice: TensorSlice
    wire_dtype: torch.dtype
    capacity: int
    # Fetched block of rows: offsets of its first row over the leading dims.
    first_row: tuple[int, ...]
    # Row coordinates -> that shard's slice of the full tensor, for rows that
    # overlap the own slice.
    shards: dict[tuple[int, ...], TensorSlice]
    # Reused receive buffer for the block, so transports register it once.
    block: torch.Tensor | None = None
    # (epoch, version) of each row the destination holds, None before the
    # first get.
    held: dict[tuple[int, ...], tuple[int, int]] | None = None
    # Destination identity after the last get, to detect another destination
    # or in-place writes to it.
    destination_id: tuple[int, int] | None = None


# Per requester client: state dict key -> flat key -> route.
_routes: weakref.WeakKeyDictionary[
    RoutingClient, dict[str, dict[str, _Route]]
] = weakref.WeakKeyDictionary()


async def get_state_dict_delta(
    store: RoutingClient, key: str, user_state_dict: dict[str, Any]
) -> dict[str, Any]:
    """Update ``user_state_dict`` from the previous to the latest version in place.

    Tensors are fetched in full unless they hold what the previous call on
    this client wrote to them: pass the same destination tensors every call,
    and write them only through this function. Writes that bypass autograd's
    version counter (``.data``, numpy, DLPack) are not detected. Keys that no
    publisher has are left unchanged. Call after the first
    :func:`put_state_dict_delta` of ``key`` completed.
    """
    if not isinstance(store, RoutingClient):
        raise TypeError("state dict deltas require a routing client")
    flattened, _mapping = flatten_state_dict(user_state_dict)
    for flat_key, value in flattened.items():
        if isinstance(value, torch.Tensor) and value.is_inference():
            raise ValueError(
                f"{flat_key!r} is an inference tensor, whose writes cannot be "
                "detected; allocate destinations outside inference mode"
            )
    routes = await _plan(store, key, flattened)
    if not routes:
        return user_state_dict

    delta_key = _delta_key(key)
    blocks = await store.get_batch(
        {
            _state_dict_storage_key(delta_key, flat_key): route.block
            for flat_key, route in routes.items()
        }
    )
    # Before any write: views of one buffer share an in-place version counter.
    unmodified = {
        flat_key: route.destination_id == _identity(_local(flattened[flat_key]))
        for flat_key, route in routes.items()
    }
    full = {}
    latest = {}
    for flat_key, route in routes.items():
        block = blocks[_state_dict_storage_key(delta_key, flat_key)]
        rows = _apply(route, _local(flattened[flat_key]), block, unmodified[flat_key])
        if rows is not None:
            full[flat_key] = flattened[flat_key]
            latest[flat_key] = rows
    if full:
        logger.info(
            "Reading %d of %d tensors of %r in full", len(full), len(routes), key
        )
        await get_state_dict(store, key, full, strict=False)
    for flat_key, rows in latest.items():
        routes[flat_key].held = rows
    for flat_key, route in routes.items():
        route.destination_id = _identity(_local(flattened[flat_key]))
    return user_state_dict


def _identity(tensor: torch.Tensor) -> tuple[int, int]:
    # The version counter increments on every in-place write.
    return tensor.data_ptr(), tensor._version


def _local(value: torch.Tensor) -> torch.Tensor:
    return value._local_tensor if isinstance(value, DTensor) else value


async def _plan(
    store: RoutingClient, key: str, flattened: Mapping[str, Any]
) -> dict[str, _Route]:
    """Routes for ``flattened``, planning keys seen for the first time."""
    routes = _routes.setdefault(store, {}).setdefault(key, {})
    new_keys = [flat_key for flat_key in flattened if flat_key not in routes]
    if new_keys:
        delta_key = _delta_key(key)
        publishers = await store._publisher_registrations(key)
        delta_publishers = await store._publisher_registrations(delta_key)
        new_routes = {}
        registrations = {}
        for flat_key in new_keys:
            route = _route(
                key, flat_key, flattened[flat_key], publishers, delta_publishers
            )
            if route is None:
                continue
            new_routes[flat_key] = route
            storage_key = _state_dict_storage_key(delta_key, flat_key)
            row_global = _row_global_shape(storage_key, delta_publishers)
            last_row = tuple(
                max(coordinate[axis] for coordinate in route.shards) + 1
                for axis in range(len(route.first_row))
            )
            block_shape = tuple(
                end - start for start, end in zip(route.first_row, last_row)
            ) + (row_global[-1],)
            route.block = torch.empty(block_shape, dtype=torch.uint8)
            registrations[storage_key] = KeyRegistration(
                TensorSlice(
                    offsets=route.first_row + (0,),
                    coordinates=(),
                    global_shape=row_global,
                    local_shape=block_shape,
                    mesh_shape=(),
                ),
                dtype=torch.uint8,
            )
        if registrations:
            store._install_requester_plan(delta_key, registrations)
        # Only after the plan is installed, so a failed plan is retried.
        routes.update(new_routes)
    return {flat_key: routes[flat_key] for flat_key in flattened if flat_key in routes}


def _row_global_shape(
    storage_key: str, delta_publishers: Mapping[str, Mapping[str, KeyRegistration]]
) -> tuple[int, ...]:
    for registrations in delta_publishers.values():
        if storage_key in registrations:
            return tuple(registrations[storage_key].tensor_slice.global_shape)
    raise KeyError(f"no publisher has a delta row for {storage_key!r}")


def _route(
    key: str,
    flat_key: str,
    value: Any,
    publishers: Mapping[str, Mapping[str, KeyRegistration]],
    delta_publishers: Mapping[str, Mapping[str, KeyRegistration]],
) -> _Route | None:
    """Plan one flat key, or None if no publisher has it or it is empty."""
    from torchstore.direct_weight_sync import _request_to_slice

    storage_key = _state_dict_storage_key(key, flat_key)
    delta_storage_key = _state_dict_storage_key(_delta_key(key), flat_key)
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            f"state dict delta only supports tensors; {flat_key!r} has type "
            f"{type(value)}"
        )
    request = Request.from_any(storage_key, value)
    assert request.tensor_val is not None
    destination_slice = _request_to_slice(request, request.tensor_val)

    wire_dtype = None
    shards: dict[tuple[int, ...], TensorSlice] = {}
    for rank, registrations in publishers.items():
        full = registrations.get(storage_key)
        if full is None:
            continue
        wire_dtype = full.dtype
        if get_slice_intersection(full.tensor_slice, destination_slice) is None:
            continue
        row = delta_publishers.get(rank, {}).get(delta_storage_key)
        if row is None:
            raise KeyError(
                f"publisher {rank} put {storage_key!r} without a delta row; put "
                "every version with put_state_dict_delta"
            )
        shards[tuple(row.tensor_slice.offsets[:-1])] = full.tensor_slice
    if wire_dtype is None or not shards:
        return None
    num_axes = len(next(iter(shards)))
    first_row = tuple(
        min(coordinate[axis] for coordinate in shards) for axis in range(num_axes)
    )
    row_global = _row_global_shape(delta_storage_key, delta_publishers)
    return _Route(
        destination_slice=destination_slice,
        wire_dtype=wire_dtype,
        capacity=_row_capacity(row_global[-1], wire_dtype),
        first_row=first_row,
        shards=shards,
    )


def _apply(
    route: _Route,
    destination: torch.Tensor,
    block: torch.Tensor,
    unmodified: bool,
) -> dict[tuple[int, ...], tuple[int, int]] | None:
    """Scatter one fetched block of rows into ``destination``.

    Returns the rows' (epoch, version) if the tensor must be fetched in full
    instead: a row overflowed, or ``destination`` is not ``unmodified`` since
    the last get or does not hold the previous version of the same publishers.
    """
    if tuple(destination.shape) != tuple(route.destination_slice.local_shape):
        raise ValueError(
            f"destination has shape {tuple(destination.shape)}, expected "
            f"{route.destination_slice.local_shape}"
        )
    if not destination.is_contiguous():
        raise ValueError("destination must be contiguous")
    capacity = route.capacity
    values_end = _HEADER_BYTES + capacity * route.wire_dtype.itemsize
    indices_end = values_end + capacity * _INDEX_DTYPE.itemsize
    rows = {
        coordinate: block[
            tuple(c - f for c, f in zip(coordinate, route.first_row))
        ].contiguous()
        for coordinate in route.shards
    }
    headers = {
        coordinate: row[:_HEADER_BYTES].view(_HEADER_DTYPE).tolist()
        for coordinate, row in rows.items()
    }
    # Rows may be at different versions, e.g. after one publisher restarted;
    # each is checked against the version the destination holds from it.
    latest = {
        coordinate: (header[2], header[1]) for coordinate, header in headers.items()
    }
    if unmodified and route.held == latest:
        return None
    previous = {
        coordinate: (epoch, version - 1)
        for coordinate, (epoch, version) in latest.items()
    }
    if (
        not unmodified
        or route.held != previous
        or any(header[0] == _OVERFLOW for header in headers.values())
    ):
        return latest

    device = destination.device
    flat_destination = destination.view(-1)
    # Keep int32 indices unless destination offsets can exceed them.
    long_indices = flat_destination.numel() > torch.iinfo(_INDEX_DTYPE).max
    for coordinate, shard in route.shards.items():
        row = rows[coordinate]
        count = headers[coordinate][0]
        values = row[_HEADER_BYTES:values_end].view(route.wire_dtype)[:count]
        indices = row[values_end:indices_end].view(_INDEX_DTYPE)[:count]
        if long_indices:
            indices = indices.long()
        indices, values = _shard_to_destination(
            indices, values, shard, route.destination_slice
        )
        flat_destination[indices.to(device)] = values.to(
            device=device, dtype=destination.dtype
        )
    route.held = latest
    return None


def _shard_to_destination(
    indices: torch.Tensor,
    values: torch.Tensor,
    shard: TensorSlice,
    destination_slice: TensorSlice,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map ascending shard-local flat ``indices`` to destination-local ones.

    Drops elements outside the destination. Uses as few tensor ops as
    possible: each op has a fixed cost that dominates small deltas.
    """
    shard_shape = tuple(shard.local_shape)
    own_shape = tuple(destination_slice.local_shape)
    shift = [s - o for s, o in zip(shard.offsets, destination_slice.offsets)]
    differing = [
        dim
        for dim in range(len(own_shape))
        if shift[dim] or shard_shape[dim] != own_shape[dim]
    ]
    if not differing:
        return indices, values
    if len(differing) > 1:
        # Shard-local flat offsets -> global coordinates -> own-local offsets.
        position = torch.stack(
            torch.unravel_index(indices.long(), shard_shape), dim=1
        ) + torch.tensor(shard.offsets)
        position = position - torch.tensor(destination_slice.offsets)
        inside = ((position >= 0) & (position < torch.tensor(own_shape))).all(dim=1)
        strides = torch.tensor(_contiguous_strides(own_shape))
        return (position[inside] * strides).sum(dim=1), values[inside]

    [dim] = differing
    inner = math.prod(shard_shape[dim + 1 :])
    # Overlap along ``dim`` in shard-local coordinates.
    start = max(0, -shift[dim])
    end = min(shard_shape[dim], own_shape[dim] - shift[dim])
    if math.prod(shard_shape[:dim]) == 1:
        # The overlap is one contiguous range of the ascending indices.
        first, last = torch.searchsorted(
            indices, torch.tensor([start * inner, end * inner], dtype=indices.dtype)
        ).tolist()
        return indices[first:last] + shift[dim] * inner, values[first:last]

    outer = indices // (shard_shape[dim] * inner)
    # Each outer step is ``(own - shard) * inner`` longer in the destination.
    remapped = indices + outer * ((own_shape[dim] - shard_shape[dim]) * inner)
    remapped += shift[dim] * inner
    if start == 0 and end == shard_shape[dim]:
        return remapped, values
    along = (indices // inner if inner > 1 else indices) - outer * shard_shape[dim]
    if start > 0 and end < shard_shape[dim]:
        inside = (along >= start) & (along < end)
    elif start > 0:
        inside = along >= start
    else:
        inside = along < end
    keep = inside.nonzero().squeeze(1)
    return remapped[keep], values[keep]


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides = []
    stride = 1
    for size in reversed(shape):
        strides.append(stride)
        stride *= size
    return tuple(reversed(strides))
