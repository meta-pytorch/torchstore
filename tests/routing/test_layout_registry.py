# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import inspect
import logging
import sys
from collections.abc import Awaitable, Callable
from types import SimpleNamespace

import pytest
import torch
import torchstore.routing.layout_registry as layout_registry_module
from torchstore.routing._model import KeyRegistration
from torchstore.routing.control_plane import ControlPlane
from torchstore.routing.layout_registry import LayoutRegistry
from torchstore.transport.types import TensorSlice

pytestmark = pytest.mark.timeout(60)

_NOT_INITIALIZED = "RoutingCoordinator.init has not run"


def _registration(key: str = "model/w") -> dict[str, KeyRegistration]:
    shape = (4,)
    return {
        key: KeyRegistration(
            TensorSlice(
                offsets=(0,),
                coordinates=(),
                global_shape=shape,
                local_shape=shape,
                mesh_shape=(),
            ),
            dtype=torch.float32,
        )
    }


class _ManualClockLoop(asyncio.SelectorEventLoop):
    """Fires timers only once a test moves ``now`` past them."""

    now = 0.0

    def time(self) -> float:
        return self.now


async def _spin(steps: int) -> None:
    """Let ``steps`` loop iterations run."""
    for _ in range(steps):
        await asyncio.sleep(0)


def _expire_each_wait(
    monkeypatch: pytest.MonkeyPatch,
    at_deadline: Callable[[], Awaitable[None]] | None = None,
) -> list[float]:
    """Fire each registry wait's deadline before the wait starts, after ``at_deadline``.

    Only the registry module sees the fake. Returns the timeout each wait was given.
    """
    deadlines: list[float] = []

    async def wait_for(wait, timeout):
        deadlines.append(timeout)
        wait.close()  # the deadline fires before the wait starts
        if at_deadline is not None:
            await at_deadline()
        raise asyncio.TimeoutError

    monkeypatch.setattr(
        layout_registry_module,
        "asyncio",
        SimpleNamespace(wait_for=wait_for, TimeoutError=asyncio.TimeoutError),
    )
    return deadlines


def _plus_timeout(signature: inspect.Signature) -> inspect.Signature:
    """``signature`` plus the keyword-only ``timeout_s=None`` that lookups add."""
    timeout = inspect.Parameter(
        "timeout_s",
        inspect.Parameter.KEYWORD_ONLY,
        default=None,
        annotation="float | None",
    )
    return signature.replace(parameters=[*signature.parameters.values(), timeout])


def test_registry_offers_the_interface_calls() -> None:
    """Offer exactly the interface's coroutines. The lookup adds only its timeout."""
    public = {name for name in dir(LayoutRegistry) if not name.startswith("_")}
    assert public == ControlPlane.__abstractmethods__
    for name in public:
        method = getattr(LayoutRegistry, name)
        assert inspect.iscoroutinefunction(method), name
        interface = inspect.signature(getattr(ControlPlane, name))
        if name == "get_layouts":
            interface = _plus_timeout(interface)
        assert inspect.signature(method) == interface, name


def test_default_timeout_is_300_seconds() -> None:
    """Keep the coordinator's five-minute wait for lookups without a timeout."""
    assert layout_registry_module._LAYOUT_REGISTRATION_TIMEOUT_S == 300


@pytest.mark.asyncio
async def test_errors_keep_the_coordinator_types_and_texts() -> None:
    """Raise the coordinator's exact error types and messages."""
    registry = LayoutRegistry()
    for call in (
        registry.strategy,
        lambda: registry.register_layout("publisher/0", "model", _registration()),
        lambda: registry.get_layouts("model"),
    ):
        with pytest.raises(RuntimeError) as raised:
            await asyncio.wait_for(call(), timeout=1)
        assert type(raised.value) is RuntimeError
        assert raised.value.args == (_NOT_INITIALIZED,)

    await registry.init({"publisher/0", "publisher/1"}, object())
    with pytest.raises(KeyError) as rejected:
        await registry.register_layout("requester", "model", _registration())
    assert type(rejected.value) is KeyError
    assert rejected.value.args == ("'requester' is not a publisher rank",)
    await registry.register_layout("publisher/0", "model", _registration())
    # Zero or less checks once and reports a 0-second wait.
    for timeout_s in (0, -0.0, -1):
        with pytest.raises(TimeoutError) as timed_out:
            await asyncio.wait_for(
                registry.get_layouts("model", timeout_s=timeout_s), timeout=1
            )
        assert type(timed_out.value) is TimeoutError
        assert str(timed_out.value) == (
            "timed out after 0 seconds waiting for publisher layouts for 'model'; "
            "missing publishers: ['publisher/1']"
        )


@pytest.mark.asyncio
async def test_init_takes_publishers_once() -> None:
    """Refuse an empty publisher set and a second init: re-init is unsupported."""
    registry = LayoutRegistry()
    with pytest.raises(ValueError, match="needs at least one publisher rank"):
        await registry.init([], object())
    await registry.init({"publisher/0"}, object())
    with pytest.raises(RuntimeError, match="RoutingCoordinator.init already ran"):
        await registry.init({"publisher/1"}, object())
    with pytest.raises(KeyError, match="'publisher/1' is not a publisher rank"):
        await registry.register_layout("publisher/1", "model", _registration())


@pytest.mark.asyncio
async def test_timeout_names_missing_publishers_sorted() -> None:
    """Name missing publishers in sorted order, whatever the set's iteration order."""
    registry = LayoutRegistry()
    publishers = {f"publisher/{rank}" for rank in range(12)}
    await registry.init(publishers, object())
    await registry.register_layout("publisher/5", "model", _registration())
    with pytest.raises(TimeoutError) as timed_out:
        await asyncio.wait_for(registry.get_layouts("model", timeout_s=0), timeout=1)
    missing = sorted(publishers - {"publisher/5"})
    assert str(timed_out.value).endswith(f"missing publishers: {missing}")


@pytest.mark.asyncio
async def test_first_layout_per_rank_and_key_wins() -> None:
    """Keep each rank's first layout for a key, before and after the set completes."""
    registry = LayoutRegistry()
    firsts = {"publisher/0": "model/first", "publisher/1": "model/second"}
    await registry.init(set(firsts), object())
    # publisher/0 repeats while the set is incomplete, publisher/1 once it is complete.
    for rank, first in firsts.items():
        await registry.register_layout(rank, "model", _registration(first))
        await registry.register_layout(rank, "model", _registration("model/late"))

    layouts = await asyncio.wait_for(
        registry.get_layouts("model", timeout_s=0), timeout=1
    )
    assert layouts == {rank: _registration(first) for rank, first in firsts.items()}


@pytest.mark.asyncio
async def test_registry_copies_its_publishers_and_registrations() -> None:
    """Take any publisher collection, and keep copies so callers can reuse theirs."""
    registry = LayoutRegistry()
    publishers = ["publisher/0", "publisher/1"]
    await registry.init(publishers, object())
    publishers.clear()
    registrations = _registration()
    for rank in ("publisher/0", "publisher/1"):
        await registry.register_layout(rank, "model", registrations)
    registrations.clear()

    layouts = await asyncio.wait_for(
        registry.get_layouts("model", timeout_s=30), timeout=1
    )
    assert layouts == {"publisher/0": _registration(), "publisher/1": _registration()}


@pytest.mark.asyncio
async def test_lookups_that_miss_leave_no_entry() -> None:
    """Drop a key's entry once its last lookup times out before any layout arrives.

    A registered key keeps its layouts, and lookups on other keys wait on.
    """
    registry = LayoutRegistry()
    await registry.init({"publisher/0", "publisher/1"}, object())
    await registry.register_layout("publisher/0", "kept", _registration())
    pending = asyncio.ensure_future(registry.get_layouts("pending", timeout_s=30))
    await _spin(10)
    for key in ("kept", "missed"):
        for timeout_s in (0, 0.01):
            with pytest.raises(TimeoutError, match="missing publishers"):
                await asyncio.wait_for(
                    registry.get_layouts(key, timeout_s=timeout_s), timeout=5
                )
            assert set(registry._layouts) == {"kept", "pending"}, (key, timeout_s)
    pending.cancel()
    await asyncio.gather(pending, return_exceptions=True)
    await registry.register_layout("publisher/1", "kept", _registration("model/late"))
    layouts = await asyncio.wait_for(
        registry.get_layouts("kept", timeout_s=0), timeout=1
    )
    assert layouts == {
        "publisher/0": _registration(),
        "publisher/1": _registration("model/late"),
    }


@pytest.mark.asyncio
async def test_a_cancelled_or_refused_call_leaves_no_entry() -> None:
    """Leave no entry for a cancelled lookup or a registration from another rank."""
    registry = LayoutRegistry()
    await registry.init({"publisher/0"}, object())
    with pytest.raises(KeyError, match="not a publisher rank"):
        await registry.register_layout("requester", "refused", _registration())
    cancelled = asyncio.ensure_future(registry.get_layouts("cancelled", timeout_s=30))
    await _spin(10)
    assert not cancelled.done() and set(registry._layouts) == {"cancelled"}
    cancelled.cancel()
    await asyncio.gather(cancelled, return_exceptions=True)
    assert cancelled.cancelled()
    assert registry._layouts == {}


@pytest.mark.skipif(
    sys.version_info >= (3, 12), reason="Python 3.12's wait_for has no such window"
)
def test_cancelled_lookups_leave_the_key_usable() -> None:
    """Leave a key usable when its lookups are cancelled at their shared deadline.

    A host cancels every running lookup, as Monarch's cleanup does.
    """

    async def cancel_at_the_deadline() -> None:
        loop = asyncio.get_running_loop()
        assert isinstance(loop, _ManualClockLoop)
        registry = LayoutRegistry()
        publishers = {"publisher/0", "publisher/1"}
        await registry.init(publishers, object())
        lookups = [
            asyncio.ensure_future(registry.get_layouts("model", timeout_s=1))
            for _ in range(4)
        ]
        await _spin(10)
        loop.now = 2.0  # every lookup's deadline passes at once
        # Four loop passes in, Python 3.10's timed-out waits are still unwinding.
        await _spin(4)
        running = [lookup for lookup in lookups if not lookup.done()]
        assert running, "every lookup ended before the cancel, so nothing was checked"
        for lookup in lookups:
            lookup.cancel()
        await _spin(30)
        for lookup in lookups:
            assert lookup.done(), "a cancelled lookup kept running"
            if not lookup.cancelled():
                assert isinstance(lookup.exception(), TimeoutError)
        # A lookup still running at the cancel ends cancelled, never timed out.
        assert all(lookup.cancelled() for lookup in running)
        # Later calls on the key neither block nor miss a layout.
        later = [
            asyncio.ensure_future(
                registry.register_layout(rank, "model", _registration(rank))
            )
            for rank in sorted(publishers)
        ]
        lookup = asyncio.ensure_future(registry.get_layouts("model", timeout_s=0))
        await _spin(10)
        calls = [*later, lookup]
        assert all(call.done() for call in calls), "the key stayed locked"
        assert set(lookup.result()) == publishers

    loop = _ManualClockLoop()
    try:
        # Closed without draining, so a wedged call fails instead of hanging.
        loop.run_until_complete(cancel_at_the_deadline())
    finally:
        loop.close()


@pytest.mark.asyncio
async def test_lookup_waits_exactly_its_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wait out a lookup's whole timeout, 300 s by default. Zero or less never waits."""
    registry = LayoutRegistry()
    await registry.init({"publisher"}, object())
    deadlines = _expire_each_wait(monkeypatch)
    for timeout_s, waits_s in ((None, 300), (120, 120), (0, 0), (-1, 0)):
        with pytest.raises(TimeoutError, match=f"timed out after {waits_s} seconds"):
            await registry.get_layouts("model", timeout_s=timeout_s)
    assert deadlines == [300, 120, 0, 0]


@pytest.mark.asyncio
async def test_timeout_racing_a_registration(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Return the layouts when the last registration lands as the wait times out.

    That read logs a WARNING, and a zero-timeout check of complete layouts logs none.
    """
    registry = LayoutRegistry()
    await registry.init({"publisher/0", "publisher/1"}, object())
    await registry.register_layout("publisher/0", "model", _registration("model/0"))

    async def register_the_last() -> None:
        await registry.register_layout("publisher/1", "model", _registration("model/1"))

    deadlines = _expire_each_wait(monkeypatch, register_the_last)
    caplog.set_level(logging.WARNING, logger=layout_registry_module.__name__)
    expected = {
        "publisher/0": _registration("model/0"),
        "publisher/1": _registration("model/1"),
    }
    assert await registry.get_layouts("model") == expected
    assert await registry.get_layouts("model", timeout_s=0) == expected
    assert deadlines == [300, 0]
    assert [
        record.getMessage()
        for record in caplog.records
        if record.name == layout_registry_module.__name__
    ] == ["publisher layouts for 'model' completed only as their 300 s wait timed out"]
