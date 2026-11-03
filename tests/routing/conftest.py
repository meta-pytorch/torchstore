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
import torchstore.api as store_api
from torchstore.client import LocalClient


@pytest.fixture
def rank_client(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., Coroutine[Any, Any, LocalClient]]:
    """Get ``ts.client(store_name, **kwargs)`` in this process as rank ``rank`` of its mesh.

    Service cases that build clients here take their ranks from this one place.
    """

    async def client(store_name: str, *, rank: int = 0, **kwargs: Any) -> LocalClient:
        # A rank ID's last part is Monarch's rank, set here without loading the Monarch backend.
        monkeypatch.setattr(
            store_api, "_routing_volume_id", lambda namespace: f"{namespace}/{rank}"
        )
        return await ts.client(store_name, **kwargs)

    return client
