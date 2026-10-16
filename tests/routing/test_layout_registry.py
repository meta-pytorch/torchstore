# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import inspect
import math
import re
from collections.abc import Awaitable, Callable
from types import SimpleNamespace

import pytest
import torch
import torchstore.routing.layout_registry as layout_registry_module
from torchstore.routing._model import KeyRegistration, Registrations
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


def _expire_next_wait(
    monkeypatch: pytest.MonkeyPatch, while_parked: Callable[[], Awaitable[None]]
) -> list[float]:
    """Park the registry's next wait, run ``while_parked`` and then fire its deadline.

    Only the registry module sees the fake. Returns the timeout each wait was given.
    """
    deadlines: list[float] = []

    async def wait_for(wait, timeout):
        wait.send(None)  # the wait parks on its key's event
        await while_parked()
        deadlines.append(timeout)
        try:
            wait.throw(asyncio.CancelledError())  # the deadline fires before it resumes
        except asyncio.CancelledError:
            raise asyncio.TimeoutError from None
        raise AssertionError("the wait outlived its deadline")

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
    with pytest.raises(ValueError) as refused:
        await asyncio.wait_for(
            registry.get_layouts("model", timeout_s=math.nan), timeout=1
        )
    assert type(refused.value) is ValueError
    assert refused.value.args == ("get_layouts timeout_s must not be NaN",)


@pytest.mark.asyncio
async def test_second_init_replaces_publishers_strategy_and_layouts() -> None:
    """Replace the publishers and strategy, and drop every layout, on a second init.

    A lookup that the last layout woke before that init still returns those layouts,
    even once a new publisher registers the key.
    """
    registry = LayoutRegistry()
    first, second = object(), object()
    await registry.init({"publisher/0", "publisher/1"}, first)
    await registry.register_layout("publisher/0", "model", _registration())
    woken = asyncio.ensure_future(registry.get_layouts("model", timeout_s=30))
    await _spin(10)
    assert not woken.done()
    # This wakes the lookup, and no call yields until it is awaited after the init.
    await registry.register_layout("publisher/1", "model", _registration())
    assert await registry.strategy() is first

    await registry.init({"publisher/1", "publisher/2"}, second)
    assert await registry.strategy() is second
    await registry.register_layout("publisher/2", "model", _registration("model/new"))
    assert not woken.done()
    assert await asyncio.wait_for(woken, timeout=1) == {
        "publisher/0": _registration(),
        "publisher/1": _registration(),
    }
    with pytest.raises(
        TimeoutError, match=re.escape("missing publishers: ['publisher/1']")
    ):
        await asyncio.wait_for(registry.get_layouts("model", timeout_s=0), timeout=1)
    with pytest.raises(KeyError, match="'publisher/0' is not a publisher rank"):
        await registry.register_layout("publisher/0", "model", _registration())


@pytest.mark.asyncio
async def test_second_init_with_the_same_publishers_starts_over() -> None:
    """Replace the strategy and drop every layout even when the publishers repeat."""
    registry = LayoutRegistry()
    publishers, first, second = {"publisher/0", "publisher/1"}, object(), object()
    await registry.init(publishers, first)
    for rank in sorted(publishers):
        await registry.register_layout(rank, "model", _registration("model/old"))

    await registry.init(publishers, second)
    assert await registry.strategy() is second
    with pytest.raises(
        TimeoutError,
        match=re.escape("missing publishers: ['publisher/0', 'publisher/1']"),
    ):
        await asyncio.wait_for(registry.get_layouts("model", timeout_s=0), timeout=1)
    for rank in sorted(publishers):
        await registry.register_layout(rank, "model", _registration("model/new"))
    layouts = await asyncio.wait_for(
        registry.get_layouts("model", timeout_s=0), timeout=1
    )
    assert layouts == {rank: _registration("model/new") for rank in publishers}


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
async def test_lookups_that_miss_leave_no_state() -> None:
    """Drop a key's entry once its last lookup leaves before any layout arrives.

    A key that a publisher registered keeps its layouts for later lookups, and a
    lookup left from before a second init leaves the new entry alone.
    """
    registry = LayoutRegistry()
    await registry.init({"publisher/0", "publisher/1"}, object())
    await registry.register_layout("publisher/0", "kept", _registration())
    with pytest.raises(KeyError, match="not a publisher rank"):
        await registry.register_layout("requester", "rejected", _registration())
    with pytest.raises(ValueError, match="must not be NaN"):
        await registry.get_layouts("refused", timeout_s=math.nan)
    with pytest.raises(TypeError, match="not iterable"):
        await registry.register_layout("publisher/0", "malformed", None)
    # A lookup waiting on another key must not hold back a missed key's cleanup.
    pending = asyncio.ensure_future(registry.get_layouts("pending", timeout_s=30))
    await _spin(10)
    for key in ("kept", "missed"):
        for timeout_s in (0, -1, 0.01):
            with pytest.raises(TimeoutError, match="missing publishers"):
                await asyncio.wait_for(
                    registry.get_layouts(key, timeout_s=timeout_s), timeout=5
                )
            assert set(registry._layouts) == {"kept", "pending"}, (key, timeout_s)
    assert not registry._layouts["kept"].complete._waiters
    pending.cancel()
    await asyncio.gather(pending, return_exceptions=True)
    assert pending.cancelled()
    cancelled = asyncio.ensure_future(registry.get_layouts("cancelled", timeout_s=30))
    await _spin(10)
    assert not cancelled.done() and "cancelled" in registry._layouts
    cancelled.cancel()
    await asyncio.gather(cancelled, return_exceptions=True)
    assert cancelled.cancelled()

    assert set(registry._layouts) == {"kept"}
    await registry.register_layout("publisher/1", "kept", _registration("model/late"))
    layouts = await asyncio.wait_for(
        registry.get_layouts("kept", timeout_s=0), timeout=1
    )
    assert layouts == {
        "publisher/0": _registration(),
        "publisher/1": _registration("model/late"),
    }

    left = asyncio.ensure_future(registry.get_layouts("model", timeout_s=30))
    await _spin(10)
    await registry.init({"publisher/0", "publisher/1"}, object())
    await registry.register_layout("publisher/0", "model", _registration())
    assert not left.done()
    left.cancel()
    await asyncio.gather(left, return_exceptions=True)
    assert left.cancelled()
    assert set(registry._layouts) == {"model"}


def test_cancelled_lookups_leave_the_key_usable() -> None:
    """Leave a key usable however cancels land around its lookups' shared deadline.

    A host cancels every running lookup, as Monarch's cleanup does, and a second
    cancel can land while the first unwinds.
    """

    async def cancel_at(step: int, again: int | None) -> None:
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
        await _spin(step)
        running = [lookup for lookup in lookups if not lookup.done()]
        for lookup in lookups:
            lookup.cancel()
        if again is not None:
            await _spin(again)
            lookups[0].cancel()
        await _spin(30)
        for lookup in lookups:
            assert lookup.done(), f"a cancelled lookup kept running: {step=} {again=}"
            if not lookup.cancelled():
                assert isinstance(lookup.exception(), TimeoutError), (step, again)
        # A lookup still running at the cancel ends cancelled, never timed out.
        assert all(lookup.cancelled() for lookup in running), (step, again)
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
        assert all(call.done() for call in calls), f"the key stayed locked: {step=}"
        assert set(lookup.result()) == publishers

    for step in range(8):
        for again in (None, 0, 1, 2):
            loop = _ManualClockLoop()
            try:
                # Closed without draining, so a wedged call fails instead of hanging.
                loop.run_until_complete(cancel_at(step, again))
            finally:
                loop.close()


def test_lookup_waits_exactly_its_timeout() -> None:
    """Wait out a lookup's whole timeout, 300 s by default. Zero or less never waits."""

    def expire(timeout_s: float | None, waits_s: float) -> None:
        """Move the lookup's clock to just before ``waits_s``, then just past it."""
        registry = LayoutRegistry()
        loop = _ManualClockLoop()

        async def wait() -> None:
            await registry.init({"publisher"}, object())
            # The wait sets its deadline at time 0, where the clock starts.
            waiter = asyncio.ensure_future(
                registry.get_layouts("model", timeout_s=timeout_s)
            )
            try:
                await _spin(20)
                if waits_s:
                    loop.now = waits_s - 1
                    await _spin(20)
                    assert not waiter.done(), f"gave up before {waits_s} s"
                    loop.now = waits_s + 1
                    await _spin(20)
                assert waiter.done(), f"still waiting after {waits_s} s"
            finally:
                waiter.cancel()
                await asyncio.gather(waiter, return_exceptions=True)
            message = f"timed out after {waits_s} seconds"
            with pytest.raises(TimeoutError, match=message):
                waiter.result()

        try:
            loop.run_until_complete(wait())
        finally:
            loop.close()

    for timeout_s, waits_s in ((None, 300), (120, 120), (0, 0), (-1, 0)):
        expire(timeout_s, waits_s)


@pytest.mark.parametrize(
    ("registered", "racing", "missing"),
    [
        (("publisher/0",), ("publisher/1",), []),
        ((), ("publisher/0",), ["publisher/1"]),
    ],
    ids=("completes-layouts", "leaves-one-missing"),
)
@pytest.mark.asyncio
async def test_timeout_racing_a_registration(
    monkeypatch: pytest.MonkeyPatch,
    registered: tuple[str, ...],
    racing: tuple[str, ...],
    missing: list[str],
) -> None:
    """Return layouts that a registration racing the timeout completes, else name who is missing."""
    registry = LayoutRegistry()
    publishers = {"publisher/0", "publisher/1"}
    await registry.init(publishers, object())
    for rank in registered:
        await registry.register_layout(rank, "model", _registration(rank))

    async def register_racing() -> None:
        for rank in racing:
            await registry.register_layout(rank, "model", _registration(rank))

    deadlines = _expire_next_wait(monkeypatch, register_racing)
    if missing:
        with pytest.raises(
            TimeoutError, match=re.escape(f"missing publishers: {missing}")
        ):
            await registry.get_layouts("model")
    else:
        layouts = await registry.get_layouts("model")
        assert layouts == {rank: _registration(rank) for rank in publishers}
        # Complete layouts return without starting a wait.
        assert await registry.get_layouts("model") == layouts
    assert deadlines == [layout_registry_module._LAYOUT_REGISTRATION_TIMEOUT_S]


@pytest.mark.parametrize(
    "reinit",
    [{"publisher/0"}, {"publisher/0", "publisher/1"}],
    ids=("one-new-publisher", "two-new-publishers"),
)
@pytest.mark.asyncio
async def test_a_read_across_a_second_init_times_out_on_the_dropped_layouts(
    monkeypatch: pytest.MonkeyPatch, reinit: set[str]
) -> None:
    """Time out a read on layouts that a second init dropped before they completed.

    The error names that init, even when the dropped layouts hold every new publisher.
    The new layouts complete and serve later reads.
    """
    registry = LayoutRegistry()
    await registry.init({"publisher/0", "publisher/1", "publisher/2"}, object())
    for rank in ("publisher/0", "publisher/1"):
        await registry.register_layout(rank, "model", _registration(rank))

    async def reinit_and_register() -> None:
        await registry.init(reinit, object())
        for rank in sorted(reinit):
            await registry.register_layout(rank, "model", _registration(f"{rank}/new"))

    deadlines = _expire_next_wait(monkeypatch, reinit_and_register)
    message = (
        "timed out after 300 seconds waiting for publisher layouts for 'model', "
        "which a later init dropped"
    )
    with pytest.raises(TimeoutError, match=f"^{re.escape(message)}$"):
        await registry.get_layouts("model")
    assert deadlines == [300]
    layouts = await asyncio.wait_for(
        registry.get_layouts("model", timeout_s=0), timeout=1
    )
    assert layouts == {rank: _registration(f"{rank}/new") for rank in reinit}


def test_layouts_completed_before_a_second_init_return_at_the_deadline() -> None:
    """Return layouts that completed before a second init, even as the deadline fires.

    The last registration, the init and a new layout for the key land in the loop pass
    that expires the wait.
    """

    async def race(reinit: set[str]) -> Registrations:
        loop = asyncio.get_running_loop()
        assert isinstance(loop, _ManualClockLoop)
        registry = LayoutRegistry()
        await registry.init({"publisher/0", "publisher/1"}, object())
        await registry.register_layout("publisher/0", "model", _registration("model/0"))
        lookup = asyncio.ensure_future(registry.get_layouts("model", timeout_s=1))
        await _spin(10)
        assert not lookup.done()

        async def complete_then_reinit() -> None:
            await registry.register_layout(
                "publisher/1", "model", _registration("model/1")
            )
            await registry.init(reinit, object())
            await registry.register_layout(
                "publisher/1", "model", _registration("model/new")
            )

        loop.call_at(0.5, lambda: asyncio.ensure_future(complete_then_reinit()))
        loop.now = 2.0  # the registration's timer runs just before the deadline's
        await _spin(50)
        return lookup.result()

    for reinit in ({"publisher/0", "publisher/1"}, {"publisher/1", "publisher/2"}):
        loop = _ManualClockLoop()
        try:
            assert loop.run_until_complete(race(reinit)) == {
                "publisher/0": _registration("model/0"),
                "publisher/1": _registration("model/1"),
            }, reinit
        finally:
            loop.close()
