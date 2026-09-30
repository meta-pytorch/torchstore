# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from torchstore.transport.types import TensorSlice


def tensor_slice(
    offsets: tuple[int, ...],
    local_shape: tuple[int, ...],
    *,
    global_shape: tuple[int, ...] = (8, 6),
    coordinates: tuple[int, ...] = (),
    mesh_shape: tuple[int, ...] = (),
) -> TensorSlice:
    """Build tensor-slice metadata with empty placement by default."""
    return TensorSlice(
        offsets=offsets,
        coordinates=coordinates,
        global_shape=global_shape,
        local_shape=local_shape,
        mesh_shape=mesh_shape,
    )
