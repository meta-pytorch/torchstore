# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import inspect
import re
import uuid
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torchstore.routing.coordinator as coordinator_module
from monarch.actor import ActorError, get_or_spawn_controller
from torchstore.routing._model import KeyRegistration, Registrations
from torchstore.routing.coordinator import RoutingCoordinator

from .utils import tensor_slice


async def _coordinator(publishers=None, strategy=None):
    coordinator = await get_or_spawn_controller(
        f"routing-coordinator-test-{uuid.uuid4()}", RoutingCoordinator
    )
    if publishers is not None:
        await coordinator.init.call_one(publishers, strategy)
    return coordinator


def _registration(key: str = "model/w") -> dict[str, KeyRegistration]:
    return {
        key: KeyRegistration(
            tensor_slice((0,), (4,), global_shape=(4,)),
            dtype=torch.float32,
        )
    }


class _ManualClockLoop(asyncio.SelectorEventLoop):
    """Fires timers only once a test moves ``now`` past them."""

    now = 0.0

    def time(self) -> float:
        return self.now


def _endpoint_body(name: str) -> Callable[..., Any]:
    """The undecorated ``RoutingCoordinator`` method, callable without an actor."""
    return inspect.unwrap(RoutingCoordinator.__dict__[name]._method)


@pytest.mark.asyncio
async def test_strategy_requires_initialization() -> None:
    """Reject strategy lookup before the coordinator is initialized."""
    coordinator = await _coordinator()

    with pytest.raises(ActorError, match="init has not run"):
        await coordinator.strategy.call_one()


@pytest.mark.asyncio
async def test_publishers_do_not_wait_and_requesters_wait_for_publishers() -> None:
    """Return publisher registration immediately but gate requester lookup."""
    coordinator = await _coordinator({"publisher/0", "publisher/1"}, strategy=object())

    await asyncio.wait_for(
        coordinator.register_layout.call_one(
            "publisher/0", "model", _registration("model/first")
        ),
        timeout=1,
    )
    # Two waiting lookups, so waking only one of them fails too.
    lookups = [
        asyncio.ensure_future(coordinator.get_layouts.call_one("model"))
        for _ in range(2)
    ]
    done, _ = await asyncio.wait(lookups, timeout=0.5)
    assert not done, "get_layouts returned before every publisher registered"
    await asyncio.wait_for(
        coordinator.register_layout.call_one(
            "publisher/1", "model", _registration("model/second")
        ),
        timeout=1,
    )
    # Far below the coordinator's timeout, so a lost wake-up fails here.
    results = await asyncio.wait_for(asyncio.gather(*lookups), timeout=5)

    expected = {
        "publisher/0": _registration("model/first"),
        "publisher/1": _registration("model/second"),
    }
    assert results == [expected, expected]


@pytest.mark.asyncio
async def test_rejects_registration_from_a_nonpublisher() -> None:
    """Restrict registration to publisher ranks."""
    coordinator = await _coordinator({"publisher"})

    with pytest.raises(ActorError, match="not a publisher rank"):
        await coordinator.register_layout.call_one(
            "requester", "model", _registration()
        )


@pytest.mark.asyncio
async def test_reuses_completed_layouts() -> None:
    """Return stored layouts without replacing a repeated registration."""
    coordinator = await _coordinator({"publisher"})
    await coordinator.register_layout.call_one("publisher", "model", _registration())
    await coordinator.register_layout.call_one(
        "publisher", "model", _registration("replacement/w")
    )
    layouts = await asyncio.wait_for(
        coordinator.get_layouts.call_one("model"), timeout=1
    )

    assert set(layouts["publisher"]) == {"model/w"}


@pytest.mark.asyncio
async def test_timeout_preserves_layout_for_retry(monkeypatch) -> None:
    """Name only the missing publishers on timeout and allow lookup to retry."""
    timeout = coordinator_module._LAYOUT_REGISTRATION_TIMEOUT_S
    monkeypatch.setattr(coordinator_module, "_LAYOUT_REGISTRATION_TIMEOUT_S", 0)
    coordinator = await _coordinator({"publisher/0", "publisher/1"})
    first, second = _registration("model/first"), _registration("model/second")

    with pytest.raises(
        ActorError, match=r"missing publishers: \['publisher/0', 'publisher/1'\]"
    ):
        await asyncio.wait_for(
            coordinator.get_layouts.call_one("model"),
            timeout=1,
        )
    await coordinator.register_layout.call_one("publisher/0", "model", first)
    with pytest.raises(ActorError, match=r"missing publishers: \['publisher/1'\]"):
        await asyncio.wait_for(
            coordinator.get_layouts.call_one("model"),
            timeout=1,
        )

    monkeypatch.setattr(coordinator_module, "_LAYOUT_REGISTRATION_TIMEOUT_S", timeout)
    await coordinator.register_layout.call_one("publisher/1", "model", second)
    layouts = await asyncio.wait_for(
        coordinator.get_layouts.call_one("model"), timeout=1
    )
    assert layouts == {"publisher/0": first, "publisher/1": second}


@pytest.mark.asyncio
async def test_state_dict_namespaces_have_independent_layouts() -> None:
    """Gather model and optimizer publisher layouts independently."""
    coordinator = await _coordinator({"publisher"})
    for namespace in ("model", "optimizer"):
        await coordinator.register_layout.call_one(
            "publisher", namespace, _registration(f"{namespace}/w")
        )

    results = await asyncio.wait_for(
        asyncio.gather(
            *(
                coordinator.get_layouts.call_one(namespace)
                for namespace in ("model", "optimizer")
            )
        ),
        timeout=1,
    )
    assert all(set(layouts) == {"publisher"} for layouts in results)


@pytest.mark.timeout(60)
def test_timeout_racing_the_last_registration_returns_layouts(monkeypatch) -> None:
    """Return the layouts only when the last one lands as the wait times out."""
    monkeypatch.setattr(coordinator_module, "_LAYOUT_REGISTRATION_TIMEOUT_S", 1)
    init, get_layouts, register_layout = map(
        _endpoint_body, ("init", "get_layouts", "register_layout")
    )
    timeouts: list[tuple[asyncio.AbstractEventLoop, float]] = []
    wait_for = asyncio.wait_for

    async def recording_wait_for(fut, timeout):
        try:
            return await wait_for(fut, timeout)
        except asyncio.TimeoutError:
            timeouts.append((asyncio.get_running_loop(), timeout))
            raise

    # Without this record, a registration that wakes the read first also passes.
    monkeypatch.setattr(asyncio, "wait_for", recording_wait_for)

    def time_out(
        publishers: set[str],
        *,
        registered: tuple[str, ...] = (),
        racing: tuple[str, ...] = (),
    ) -> Registrations:
        """Park a read after ``registered`` ranks, then expire it as ``racing`` register."""
        actor = SimpleNamespace(_publishers=None, _strategy=None, _layouts={})
        loop = _ManualClockLoop()
        registrations: list[asyncio.Future[None]] = []

        def register() -> None:
            for rank in racing:
                registrations.append(
                    asyncio.ensure_future(
                        register_layout(actor, rank, "model", _registration(rank))
                    )
                )

        async def race() -> Registrations:
            await init(actor, publishers, None)
            for rank in registered:
                await register_layout(actor, rank, "model", _registration(rank))
            waiter = asyncio.ensure_future(get_layouts(actor, "model"))
            for _ in range(10):
                await asyncio.sleep(0)
                if waiter.done():
                    waiter.result()  # raises the read's own error first
                    raise AssertionError("get_layouts returned without waiting")
                layouts = actor._layouts.get("model")
                if layouts is not None and not layouts.condition.locked():
                    break  # the read is parked on the condition
            else:
                waiter.cancel()
                raise AssertionError("get_layouts never waited")
            # Both timers come due at once, and the earlier one (the registration)
            # runs first, so the last layout lands as the read's deadline expires.
            loop.call_at(0.5, register)
            loop.now = 2.0
            try:
                await asyncio.wait({waiter})
            finally:
                await asyncio.gather(*registrations)
            expired = [timeout for seen, timeout in timeouts if seen is loop]
            assert expired == [1], "the read's deadline must expire"
            return waiter.result()

        try:
            return loop.run_until_complete(race())
        finally:
            loop.close()

    publishers = {"publisher/0", "publisher/1"}
    # The last registration lands as the read's deadline expires.
    assert time_out(
        publishers, registered=("publisher/0",), racing=("publisher/1",)
    ) == {rank: _registration(rank) for rank in publishers}
    # A registration that leaves another publisher missing still names it.
    with pytest.raises(
        TimeoutError, match=re.escape("missing publishers: ['publisher/1']")
    ):
        time_out(publishers, racing=("publisher/0",))
