# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import sys
from types import ModuleType
from typing import cast

import pytest
import torchstore.transport.torch_distributed as td_transport
from torchstore.storage_volume import StorageVolume
from torchstore.strategy import StorageVolumeRef
from torchstore.transport import create_transport_buffer, TransportType
from torchstore.transport.buffers import TransportContext


def test_missing_torch_transport_retains_unavailable_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "torch.distributed._transport"
    monkeypatch.setattr(td_transport, "ENV_TORCHSTORE_TORCH_TRANSPORT_BACKEND", "nixl")
    modules = cast(dict[str, ModuleType | None], sys.modules)
    monkeypatch.setitem(modules, module_name, None)
    td_transport.torch_transport_available.cache_clear()
    ref = StorageVolumeRef(
        StorageVolume(lambda: "missing-module"),
        "missing-module",
        TransportContext(),
        TransportType.TorchDistributed,
    )
    try:
        with pytest.raises(ModuleNotFoundError) as error:
            td_transport._load_transport_api()
        assert error.value.name == module_name
        availability: bool | ImportError
        try:
            availability = td_transport.torch_transport_available()
        except ImportError as failure:
            availability = failure
        assert availability is False
        with pytest.raises(
            RuntimeError, match="torch.distributed transport is unavailable"
        ):
            create_transport_buffer(ref)
    finally:
        td_transport.torch_transport_available.cache_clear()
