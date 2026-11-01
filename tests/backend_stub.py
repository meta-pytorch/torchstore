# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A Monarch-free stand-in for the Monarch backend, installed in the factory's cache."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torchstore.backend as backend_module
from torchstore.backend import Backend


def install_backend_stub(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Make ``get_backend("monarch")`` return a mock ``Backend`` at rank 0 for one test."""
    stub = MagicMock(spec=Backend)
    stub.rank.return_value = 0
    monkeypatch.setitem(backend_module._backends, "monarch", stub)
    return stub
