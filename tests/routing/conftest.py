# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fixtures that more than one routing test module uses."""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any

import pytest
import torchstore as ts
from torchstore.client import LocalClient


@pytest.fixture
def rank_client() -> Callable[..., Coroutine[Any, Any, LocalClient]]:
    """Get ``ts.client(store_name, rank=rank, **kwargs)`` in this process with ``rank`` 0 by default."""

    async def client(store_name: str, *, rank: int = 0, **kwargs: Any) -> LocalClient:
        return await ts.client(store_name, rank=rank, **kwargs)

    return client
