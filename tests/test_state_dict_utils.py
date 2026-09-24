# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio

import pytest
import torch

from torchstore.state_dict_utils import _state_dict_storage_metadata, get_state_dict


def test_state_dict_metadata() -> None:
    """Describe nested tensors with their storage keys and wire element sizes."""
    slices, element_sizes, mapping = _state_dict_storage_metadata(
        {
            "layer": {"weight": torch.empty(3, 5)},
            "running": torch.empty(2, dtype=torch.float32),
        },
        "model",
        transfer_dtype=torch.bfloat16,
        preserve_dtype_keys=frozenset({"running"}),
    )

    weight = slices["model/layer.weight"]
    assert weight.offsets == (0, 0)
    assert weight.global_shape == (3, 5)
    assert weight.local_shape == (3, 5)
    assert element_sizes == {"model/layer.weight": 2, "model/running": 4}
    assert mapping == {"layer.weight": ("layer", "weight"), "running": ("running",)}


def test_state_dict_metadata_rejects_objects() -> None:
    """Reject values that cannot participate in tensor routing."""
    with pytest.raises(TypeError, match="only supports tensor values.*'epoch'.*int"):
        _state_dict_storage_metadata({"epoch": 1}, "model")


def test_strict_state_dict_mismatch() -> None:
    """Explain that strict reads cannot use a rank-local routed layout."""

    class Client:
        async def get(self, key):
            return {"weight": ("weight",)}

    with pytest.raises(AssertionError, match="routing does not support strict=True"):
        asyncio.run(
            get_state_dict(
                Client(),
                "model",
                user_state_dict={"other": torch.empty(1)},
            )
        )
