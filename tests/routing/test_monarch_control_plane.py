# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import logging
import re
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, call

import pytest
import torchstore as ts
import torchstore.routing.coordinator as coordinator_module
from torchstore.routing.coordinator import MonarchControlPlane
from torchstore.transport import TransportType

from .test_coordinator import _registration

pytestmark = pytest.mark.timeout(60)

_TIMING_LINE = re.compile(r"routing control_plane=monarch (.+) took (\d+\.\d+)s$")


def _timing_lines(caplog: pytest.LogCaptureFixture) -> list[tuple[str, str]]:
    """Each adapter timing line's call and seconds. All must be INFO on its logger."""
    lines = []
    for record in caplog.records:
        if match := _TIMING_LINE.search(record.getMessage()):
            assert (record.name, record.levelno) == (
                coordinator_module.__name__,
                logging.INFO,
            )
            lines.append(match.groups())
    return lines


@pytest.mark.asyncio
async def test_monarch_adapter_forwards_each_call_and_logs_it(caplog) -> None:
    """Forward each call and its result unchanged, logging one INFO line per call.

    The line names the call and how it ended, a failure or a cancel included.
    """
    strategy = ts.TorchStoreStrategy(TransportType.MonarchRPC)
    publishers, registrations = {"publisher"}, _registration()
    layouts = {"publisher": registrations}
    endpoints = {
        "init": AsyncMock(),
        "strategy": AsyncMock(return_value=strategy),
        "register_layout": AsyncMock(),
        "get_layouts": AsyncMock(return_value=layouts),
    }
    control_plane = MonarchControlPlane(
        SimpleNamespace(
            **{name: SimpleNamespace(call_one=mock) for name, mock in endpoints.items()}
        )
    )
    caplog.set_level(logging.INFO, logger=coordinator_module.__name__)

    await control_plane.init(publishers, strategy)
    assert await control_plane.strategy() is strategy
    await control_plane.register_layout("publisher", "model", registrations)
    assert await control_plane.get_layouts("model") is layouts
    assert await control_plane.get_layouts("model", timeout_s=7) is layouts
    endpoints["init"].assert_awaited_once_with(publishers=publishers, strategy=strategy)
    endpoints["strategy"].assert_awaited_once_with()
    endpoints["register_layout"].assert_awaited_once_with(
        rank="publisher", key="model", registrations=registrations
    )
    # Without a timeout, the coordinator applies its own default.
    assert endpoints["get_layouts"].await_args_list == [
        call(key="model", timeout_s=None),
        call(key="model", timeout_s=7),
    ]

    error = RuntimeError("register_layout")
    endpoints["register_layout"].side_effect = error
    with pytest.raises(RuntimeError) as raised:
        await control_plane.register_layout("publisher", "model", registrations)
    assert raised.value is error
    entered = asyncio.Event()

    async def wait_forever(**_: Any) -> None:
        entered.set()
        await asyncio.Event().wait()

    endpoints["get_layouts"].side_effect = wait_forever
    pending = asyncio.ensure_future(control_plane.get_layouts("model"))
    await asyncio.wait_for(entered.wait(), timeout=5)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending

    assert [operation for operation, _ in _timing_lines(caplog)] == [
        "init",
        "strategy",
        "register_layout key=model",
        "get_layouts key=model",
        "get_layouts key=model",
        "register_layout key=model failed with RuntimeError",
        "get_layouts key=model cancelled",
    ]
