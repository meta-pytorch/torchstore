# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import re
import uuid
from types import SimpleNamespace

import pytest
import torch
import torchstore.routing.coordinator as coordinator_module
from monarch.actor import ActorError, get_or_spawn_controller
from torchstore.routing._model import KeyRegistration
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
    monkeypatch,
    registered: tuple[str, ...],
    racing: tuple[str, ...],
    missing: list[str],
) -> None:
    """Return layouts that a registration racing the timeout completes, else name who is missing."""
    publishers = {"publisher/0", "publisher/1"}
    coordinator = await _coordinator(publishers)
    for rank in registered:
        await coordinator.register_layout.call_one(rank, "model", _registration(rank))
    deadlines: list[float] = []

    async def wait_for(wait, timeout):
        # Step the real wait by hand so it stays parked while the racing ranks register.
        wait.send(None)
        for rank in racing:
            await coordinator.register_layout.call_one(
                rank, "model", _registration(rank)
            )
        deadlines.append(timeout)
        try:
            wait.throw(asyncio.CancelledError())  # the deadline fires before it resumes
        except asyncio.CancelledError:
            raise asyncio.TimeoutError from None
        raise AssertionError("the wait outlived its deadline")

    # Only the coordinator module sees the fake, not the rest of the process.
    monkeypatch.setattr(
        coordinator_module,
        "asyncio",
        SimpleNamespace(wait_for=wait_for, TimeoutError=asyncio.TimeoutError),
    )
    lookup = asyncio.wait_for(coordinator.get_layouts.call_one("model"), timeout=5)
    if missing:
        with pytest.raises(
            ActorError, match=re.escape(f"missing publishers: {missing}")
        ):
            await lookup
    else:
        assert await lookup == {rank: _registration(rank) for rank in publishers}
    assert deadlines == [coordinator_module._LAYOUT_REGISTRATION_TIMEOUT_S]
