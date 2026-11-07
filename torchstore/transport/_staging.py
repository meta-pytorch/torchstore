# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Helpers for staging non-contiguous tensors through contiguous buffers."""

import asyncio
import os
from collections.abc import Awaitable, Callable, Sequence
from typing import TypeVar

import torch

# Maximum size of each of the two buffers that stage non-contiguous tensors.
STAGING_BUFFER_BYTES = int(
    os.environ.get("TORCHSTORE_STAGING_BUFFER_BYTES", str(32 * 1024 * 1024))
)
# Staged pieces start at this byte alignment so any dtype can view the buffer.
STAGING_ALIGNMENT = 64

T = TypeVar("T")


def aligned(nbytes: int) -> int:
    return -(-nbytes // STAGING_ALIGNMENT) * STAGING_ALIGNMENT


def staging_pieces(tensor: torch.Tensor, max_bytes: int):
    """Split a tensor into row-major pieces of at most max_bytes.

    Each piece is a view covering a contiguous byte range of the tensor's
    contiguous layout, so it maps to one destination descriptor.
    """
    if tensor.dim() == 0 or tensor.nbytes <= max_bytes:
        yield tensor
        return
    row_bytes = tensor.nbytes // tensor.shape[0]
    if row_bytes > max_bytes:
        for row in tensor:
            yield from staging_pieces(row, max_bytes)
        return
    rows = max_bytes // row_bytes
    for start in range(0, tensor.shape[0], rows):
        yield tensor[start : start + rows]


async def stage(
    sources: Sequence[tuple[torch.Tensor, T]],
    allocate: Callable[[int, int], torch.Tensor],
    flush: Callable[[int, list[tuple[torch.Tensor, T, int]]], None],
    wait: Callable[[int], Awaitable[None]],
) -> None:
    """Copy (tensor, target) sources through up to two reused buffers of at
    most ``STAGING_BUFFER_BYTES``, packing one while the other's transfers
    run.

    ``allocate(index, nbytes)`` returns a contiguous 1-D uint8 buffer of at
    least ``nbytes``, starting at a ``STAGING_ALIGNMENT`` aligned address.
    When the next piece does not fit in buffer ``index``, ``flush(index,
    pieces)`` starts transferring each ``(local, target, offset)`` piece: the
    bytes of ``local``, a slice of the buffer, go to byte ``offset`` of
    ``target``. It may submit them from tasks it schedules. Before refilling a
    flushed buffer, ``await wait(index)`` must return once its transfers
    finished. The last buffer is flushed too. On return or exception, transfers
    of either buffer may still run; the caller must wait for all of them.
    """
    if not sources:
        return
    if STAGING_BUFFER_BYTES < STAGING_ALIGNMENT:
        raise ValueError(
            "TORCHSTORE_STAGING_BUFFER_BYTES must be at least "
            f"{STAGING_ALIGNMENT}, got {STAGING_BUFFER_BYTES}"
        )
    size = STAGING_BUFFER_BYTES
    pieces: list[tuple[torch.Tensor, T, int]] = []
    for source, target in sources:
        target_offset = 0
        for piece in staging_pieces(source, size):
            pieces.append((piece, target, target_offset))
            target_offset += piece.nbytes
    # Pack in a dry run first so each buffer is no larger than what it holds.
    extents = [0, 0]
    index = offset = 0
    for piece, _, _ in pieces:
        start = aligned(offset)
        if start + piece.nbytes > size:
            index ^= 1
            start = 0
        offset = start + piece.nbytes
        extents[index] = max(extents[index], offset)
    buffers: list[torch.Tensor | None] = [None, None]
    flushed = [False, False]
    pending: list[tuple[torch.Tensor, T, int]] = []
    index = 0
    offset = 0
    for piece, target, target_offset in pieces:
        start = aligned(offset)
        # Same check as the dry run; buffers may be larger than their extent.
        if buffers[index] is None or start + piece.nbytes > size:
            if pending:
                flush(index, pending)
                pending = []
                flushed[index] = True
                # Let tasks scheduled by flush (e.g. NIXL's) submit before
                # filling the other buffer.
                await asyncio.sleep(0)
                index ^= 1
            if flushed[index]:
                await wait(index)
                flushed[index] = False
            if buffers[index] is None:
                buffers[index] = allocate(index, extents[index])
            start = 0
        buffer = buffers[index]
        assert buffer is not None
        local = buffer[start : start + piece.nbytes]
        local.view(piece.dtype).view(piece.shape).copy_(piece)
        pending.append((local, target, target_offset))
        offset = start + piece.nbytes
    flush(index, pending)
