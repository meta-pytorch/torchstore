# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
import torch
from torchstore.client import LocalClient
from torchstore.directory import VolumeDirectory
from torchstore.state_dict_utils import (
    _state_dict_storage_metadata,
    get_state_dict,
    put_state_dict,
)


def test_state_dict_metadata() -> None:
    """Describe nested tensors with their storage keys and wire dtypes."""
    slices, dtypes, mapping = _state_dict_storage_metadata(
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
    assert dtypes == {
        "model/layer.weight": torch.bfloat16,
        "model/running": torch.float32,
    }
    assert mapping == {"layer.weight": ("layer", "weight"), "running": ("running",)}


def test_state_dict_metadata_rejects_objects() -> None:
    """Reject values that cannot participate in tensor routing."""
    with pytest.raises(TypeError, match="only supports tensor values.*'epoch'.*int"):
        _state_dict_storage_metadata({"epoch": 1}, "model")


def test_strict_state_dict_mismatch() -> None:
    """Explain that strict reads cannot use a rank-local routed layout."""

    class Client:
        async def get_layouts(self, state_dict, key, *, strict=True):
            pass

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


@pytest.mark.parametrize("subsets", [False, True], ids=["standard", "subset"])
def test_non_strict_state_dict_reads_follow_client_capability(
    monkeypatch: pytest.MonkeyPatch, subsets: bool
) -> None:
    """Non-strict staged reads honor the client capability."""
    client = LocalClient(directory=Mock(spec=VolumeDirectory), strategy=None)
    if subsets:
        monkeypatch.setattr(client, "fetches_state_dict_subsets", True)
    stored = {
        "model/first": torch.tensor([2.0, 3.0], dtype=torch.float32),
        "model/second": torch.tensor([5.0, 7.0], dtype=torch.float32),
    }
    monkeypatch.setattr(
        client,
        "get",
        AsyncMock(return_value={"first": ("first",), "second": ("second",)}),
    )
    monkeypatch.setattr(
        client,
        "get_batch",
        AsyncMock(side_effect=lambda entries: {key: stored[key] for key in entries}),
    )
    result = asyncio.run(
        get_state_dict(client, "model", {"first": torch.zeros(2)}, strict=False)
    )
    expected_keys = {"first"} if subsets else {"first", "second"}
    assert set(result) == expected_keys
    for name in expected_keys:
        torch.testing.assert_close(result[name], stored[f"model/{name}"])


def test_put_state_dict_rejects_unknown_preserve_dtype_keys() -> None:
    """A mistyped preserved key would otherwise be cast silently."""
    with pytest.raises(ValueError, match=r"not in the state dict: \['bufer'\]"):
        asyncio.run(
            put_state_dict(
                object(),
                {"weight": torch.empty(1), "buffer": torch.empty(1)},
                "model",
                transfer_dtype=torch.bfloat16,
                preserve_dtype_keys=frozenset({"bufer"}),
            )
        )


def test_put_state_dict_ignores_preserve_dtype_keys_without_transfer_dtype() -> (None):
    """Callers may pass buffer names that are absent from the state dict."""
    calls = []

    class Store:
        async def register_layout(self, state_dict, key, **kwargs):
            calls.append(("register_layout", kwargs))

        async def put_batch(self, entries, **kwargs):
            calls.append(("put_batch", kwargs))

        async def put(self, key, value):
            calls.append(("put", key))

    asyncio.run(
        put_state_dict(
            Store(),
            {"weight": torch.empty(1)},
            "model",
            preserve_dtype_keys=frozenset({"cache"}),
        )
    )
    assert [name for name, _ in calls] == ["register_layout", "put_batch", "put"]
    assert calls[1][1] == {}


def test_put_state_dict_direct_rdma_rejects_preserve_dtype_keys() -> None:
    with pytest.raises(NotImplementedError, match="direct_rdma=True"):
        asyncio.run(
            put_state_dict(
                object(),
                {"weight": torch.empty(1), "buffer": torch.empty(1)},
                "model",
                direct_rdma=True,
                transfer_dtype=torch.bfloat16,
                preserve_dtype_keys=frozenset({"buffer"}),
            )
        )
