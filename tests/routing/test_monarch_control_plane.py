# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import inspect
import logging
import re
import uuid
from collections.abc import Collection
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, call

import pytest
import torchstore as ts
import torchstore.routing.coordinator as coordinator_module
from monarch.actor import ActorError, get_or_spawn_controller
from torchstore.routing.control_plane import ControlPlane
from torchstore.routing.coordinator import MonarchControlPlane, RoutingCoordinator
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


def _plus_timeout(signature: inspect.Signature) -> inspect.Signature:
    """``signature`` plus the keyword-only ``timeout_s=None`` that lookups add."""
    timeout = inspect.Parameter(
        "timeout_s",
        inspect.Parameter.KEYWORD_ONLY,
        default=None,
        annotation="float | None",
    )
    return signature.replace(parameters=[*signature.parameters.values(), timeout])


async def _monarch_control_plane(publishers: Collection[str]) -> MonarchControlPlane:
    """Wrap a fresh coordinator that knows ``publishers``."""
    coordinator = await get_or_spawn_controller(
        f"routing-coordinator-test-{uuid.uuid4()}", RoutingCoordinator
    )
    control_plane = MonarchControlPlane(coordinator)
    await control_plane.init(
        publishers, ts.TorchStoreStrategy(TransportType.MonarchRPC)
    )
    return control_plane


@pytest.mark.asyncio
async def test_monarch_adapter_forwards_each_call_and_logs_it(caplog) -> None:
    """Forward each call and its result unchanged, logging one INFO line per call.

    The line names the call and how it ended, a failure or a cancel included. The calls
    take the interface's parameters, the lookup adding only its timeout.
    """
    for name in ControlPlane.__abstractmethods__:
        interface = inspect.signature(getattr(ControlPlane, name))
        if name == "get_layouts":
            interface = _plus_timeout(interface)
        assert inspect.signature(getattr(MonarchControlPlane, name)) == interface, name

    strategy = ts.TorchStoreStrategy(TransportType.MonarchRPC)
    publishers, registrations = {"publisher"}, _registration()
    layouts = {"publisher": registrations}

    async def slow_init(**_: Any) -> None:
        await asyncio.sleep(0.05)

    endpoints = {
        "init": AsyncMock(side_effect=slow_init),
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
    # An ActorError's line names the coordinator's own error, which it carries.
    wrapped = RuntimeError("actor failed")
    wrapped.exception = KeyError("'requester' is not a publisher rank")
    endpoints["strategy"].side_effect = wrapped
    with pytest.raises(RuntimeError):
        await control_plane.strategy()
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

    lines = _timing_lines(caplog)
    # Each line reports its call's own duration, the slow init's included.
    assert float(lines[0][1]) >= 0.05
    assert [operation for operation, _ in lines] == [
        "init",
        "strategy",
        "register_layout key=model",
        "get_layouts key=model",
        "get_layouts key=model",
        "register_layout key=model failed with RuntimeError",
        "strategy failed with KeyError",
        "get_layouts key=model cancelled",
    ]


@pytest.mark.asyncio
async def test_monarch_timeout_reports_the_clamped_wait() -> None:
    """Report the caller's timeout, 0 if zero or less, and the missing publishers.

    A positive timeout is waited out and zero or less checks once.
    """
    control_plane = await _monarch_control_plane({"publisher/0", "publisher/1"})
    await control_plane.register_layout("publisher/0", "model", _registration())
    loop = asyncio.get_running_loop()
    for timeout_s, reported in ((0.2, 0.2), (0, 0), (-1, 0)):
        message = (
            f"timed out after {reported} seconds waiting for publisher layouts for "
            "'model'; missing publishers: ['publisher/1']"
        )
        started = loop.time()
        with pytest.raises(ActorError, match=re.escape(message)):
            await asyncio.wait_for(
                control_plane.get_layouts("model", timeout_s=timeout_s), timeout=5
            )
        assert loop.time() - started >= timeout_s - 0.005, "it must wait its timeout"
