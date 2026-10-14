# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import inspect
import logging
import math
import re
import uuid
from collections.abc import Collection
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, call

import pytest
import torchstore as ts
import torchstore.api as store_api
import torchstore.routing.coordinator as coordinator_module
from monarch.actor import ActorError, get_or_spawn_controller
from torchstore import RankRole
from torchstore.client import ClientType
from torchstore.routing.control_plane import ControlPlane
from torchstore.routing.coordinator import MonarchControlPlane, RoutingCoordinator
from torchstore.storage_volume import StorageVolume
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


def test_api_entry_points_call_the_coordinator_through_the_adapter(
    monkeypatch, caplog
) -> None:
    """Route initialize's init and client's strategy lookup through the adapter."""
    strategy = SimpleNamespace(set_storage_volumes=AsyncMock())
    coordinator = SimpleNamespace(
        init=SimpleNamespace(call_one=AsyncMock()),
        strategy=SimpleNamespace(call_one=AsyncMock(return_value=strategy)),
    )
    monkeypatch.setattr(StorageVolume, "spawn", AsyncMock(return_value=object()))
    monkeypatch.setattr(
        store_api, "get_or_spawn_controller", AsyncMock(return_value=coordinator)
    )
    monkeypatch.setattr(store_api, "current_rank", lambda: SimpleNamespace(rank=0))
    store_name = f"routing-test-{uuid.uuid4()}"
    caplog.set_level(logging.INFO, logger=coordinator_module.__name__)

    async def run() -> None:
        await store_api.initialize(
            mesh=SimpleNamespace(size=lambda: 2),
            strategy=strategy,
            client_type=ClientType.ROUTING,
            store_name=store_name,
        )
        await store_api.client(store_name, role=RankRole.PUBLISHER)

    try:
        asyncio.run(run())
    finally:
        store_api.reset_client(store_name)
    assert [name for name, _ in _timing_lines(caplog)] == ["init", "strategy"]


@pytest.mark.asyncio
async def test_monarch_adapter_forwards_each_call_and_logs_it(
    caplog, monkeypatch
) -> None:
    """Forward each call, its result and its error unchanged, logging one INFO line.

    The line names the call, how it ended and how long it took, cancelled calls too.
    The calls take the interface's parameters, the lookup adding only its timeout.
    """
    for name in ControlPlane.__abstractmethods__:
        interface = inspect.signature(getattr(ControlPlane, name))
        if name == "get_layouts":
            interface = _plus_timeout(interface)
        assert inspect.signature(getattr(MonarchControlPlane, name)) == interface, name

    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(
        coordinator_module, "time", SimpleNamespace(perf_counter=lambda: clock.now)
    )
    strategy = ts.TorchStoreStrategy(TransportType.MonarchRPC)
    publishers, registrations = {"publisher"}, _registration()
    layouts = {"publisher": registrations}
    endpoints = {
        "init": AsyncMock(),
        "strategy": AsyncMock(return_value=strategy),
        "register_layout": AsyncMock(),
        "get_layouts": AsyncMock(return_value=layouts),
    }

    def takes_1_5_s(mock: AsyncMock) -> SimpleNamespace:
        async def call_one(**kwargs: Any) -> Any:
            clock.now += 1.5
            return await mock(**kwargs)

        return SimpleNamespace(call_one=call_one)

    control_plane = MonarchControlPlane(
        SimpleNamespace(**{name: takes_1_5_s(mock) for name, mock in endpoints.items()})
    )
    calls = [
        ("init", lambda: control_plane.init(publishers, strategy)),
        ("strategy", control_plane.strategy),
        (
            "register_layout",
            lambda: control_plane.register_layout("publisher", "model", registrations),
        ),
        ("get_layouts", lambda: control_plane.get_layouts("model")),
        ("get_layouts", lambda: control_plane.get_layouts("model", timeout_s=7)),
    ]
    caplog.set_level(logging.INFO, logger=coordinator_module.__name__)

    results = [await make_call() for _, make_call in calls]
    assert results[1] is strategy and results[3] is results[4] is layouts
    assert endpoints["init"].await_args_list == [
        call(publishers=publishers, strategy=strategy)
    ]
    assert endpoints["init"].await_args.kwargs["publishers"] is publishers
    assert endpoints["strategy"].await_args_list == [call()]
    assert endpoints["register_layout"].await_args_list == [
        call(rank="publisher", key="model", registrations=registrations)
    ]
    # Without a timeout, the coordinator applies its own default.
    assert endpoints["get_layouts"].await_args_list == [
        call(key="model"),
        call(key="model", timeout_s=7),
    ]

    for name, mock in endpoints.items():
        mock.side_effect = RuntimeError(name)
    for name, make_call in calls:
        with pytest.raises(RuntimeError) as raised:
            await make_call()
        assert raised.value is endpoints[name].side_effect

    entered = asyncio.Event()

    async def wait_forever(**_: Any) -> None:
        entered.set()
        await asyncio.Event().wait()

    for mock in endpoints.values():
        mock.side_effect = wait_forever
    for _, make_call in calls:
        entered.clear()
        pending = asyncio.ensure_future(make_call())
        await asyncio.wait_for(entered.wait(), timeout=5)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending

    operations = ["init", "strategy", "register_layout key=model"]
    operations += 2 * ["get_layouts key=model"]
    assert _timing_lines(caplog) == [
        (operation + outcome, "1.5000")
        for outcome in ("", " failed with RuntimeError", " cancelled")
        for operation in operations
    ]


@pytest.mark.asyncio
async def test_monarch_timeout_reports_the_clamped_wait() -> None:
    """Report the caller's timeout, 0 if zero or less, and the missing publishers."""
    control_plane = await _monarch_control_plane({"publisher/0", "publisher/1"})
    await control_plane.register_layout("publisher/0", "model", _registration())
    for timeout_s, reported in ((0.2, 0.2), (0, 0), (0.0, 0), (-0.0, 0), (-1, 0)):
        message = (
            f"timed out after {reported} seconds waiting for publisher layouts for "
            "'model'; missing publishers: ['publisher/1']"
        )
        with pytest.raises(ActorError, match=re.escape(message)):
            await asyncio.wait_for(
                control_plane.get_layouts("model", timeout_s=timeout_s), timeout=5
            )


@pytest.mark.asyncio
async def test_nan_timeout_is_refused() -> None:
    """Refuse a NaN timeout at once."""
    control_plane = await _monarch_control_plane({"publisher"})
    with pytest.raises(
        ActorError, match="ValueError: get_layouts timeout_s must not be NaN"
    ):
        await asyncio.wait_for(
            control_plane.get_layouts("nan", timeout_s=math.nan), timeout=10
        )
