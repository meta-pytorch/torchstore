# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Helpers for staging non-contiguous tensors through contiguous buffers."""

import torch

# Staged pieces start at this byte alignment so any dtype can view the buffer.
STAGING_ALIGNMENT = 64


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
