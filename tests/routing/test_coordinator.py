# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio

import pytest
import torchstore.routing.coordinator as coordinator_module
from torchstore.routing._model import KeyRegistration, RankRole
from torchstore.routing.coordinator import RoutingCoordinator

from .utils import _endpoint_method, tensor_slice


async def _initialized_coordinator(ranks, strategy=None) -> RoutingCoordinator:
    coordinator = RoutingCoordinator()
    await _endpoint_method(coordinator, "init")(ranks, strategy)
    return coordinator


def _registration(key: str = "model/w") -> dict[str, KeyRegistration]:
    return {
        key: KeyRegistration(
            tensor_slice((0,), (4,), global_shape=(4,)),
            element_size=4,
        )
    }


def test_strategy_requires_initialization() -> None:
    """Reject strategy lookup before the coordinator is initialized."""
    coordinator = RoutingCoordinator()

    with pytest.raises(RuntimeError, match="init has not run"):
        asyncio.run(_endpoint_method(coordinator, "strategy")())


def test_register_layouts_waits_for_all_participants() -> None:
    """Release registrations only after every configured rank arrives."""

    async def run():
        # 1. Start one participant and show that it remains blocked.
        coordinator = await _initialized_coordinator(
            {"publisher", "requester"}, strategy=object()
        )
        register = _endpoint_method(coordinator, "register_layouts")
        publisher = asyncio.create_task(
            register("publisher", RankRole.PUBLISHER, "model", _registration())
        )
        await asyncio.sleep(0)
        assert not publisher.done()
        # 2. Register the final participant, releasing both calls.
        requester = asyncio.create_task(
            register("requester", RankRole.REQUESTER, "model", _registration())
        )
        return await asyncio.gather(publisher, requester)

    publisher_result, requester_result = asyncio.run(run())

    # 3. Both ranks receive the same complete publisher/requester layouts.
    assert publisher_result == requester_result
    publishers, requesters = publisher_result
    assert set(publishers) == {"publisher"}
    assert set(requesters) == {"requester"}


def test_rejects_an_unexpected_rank() -> None:
    """Reject registrations from ranks outside the initialized roster."""

    async def run():
        coordinator = await _initialized_coordinator({"publisher"})
        register = _endpoint_method(coordinator, "register_layouts")
        return await register("other", RankRole.PUBLISHER, "model", _registration())

    with pytest.raises(KeyError, match="not a routing participant"):
        asyncio.run(run())


def test_reuses_completed_layouts() -> None:
    """Return stored layouts without replacing them on a repeated request."""

    async def run():
        coordinator = await _initialized_coordinator({"publisher"})
        register = _endpoint_method(coordinator, "register_layouts")
        first = await register(
            "publisher", RankRole.PUBLISHER, "model", _registration()
        )
        repeated = await register(
            "publisher",
            RankRole.PUBLISHER,
            "model",
            _registration("replacement/w"),
        )
        return first, repeated

    first, repeated = asyncio.run(run())
    assert repeated == first
    assert set(repeated[0]["publisher"]) == {"model/w"}


def test_timeout_preserves_layout_for_retry(monkeypatch) -> None:
    """Report missing ranks without discarding an already registered layout."""
    registration_timeout = coordinator_module._LAYOUT_REGISTRATION_TIMEOUT_S
    monkeypatch.setattr(coordinator_module, "_LAYOUT_REGISTRATION_TIMEOUT_S", 0)

    async def run():
        coordinator = await _initialized_coordinator({"publisher", "requester"})
        register = _endpoint_method(coordinator, "register_layouts")
        with pytest.raises(
            TimeoutError,
            match=r"layouts for 'model'.*missing ranks: \['requester'\]",
        ):
            await register("publisher", RankRole.PUBLISHER, "model", _registration())

        monkeypatch.setattr(
            coordinator_module,
            "_LAYOUT_REGISTRATION_TIMEOUT_S",
            registration_timeout,
        )
        requester_result = await register(
            "requester", RankRole.REQUESTER, "model", _registration()
        )
        publisher_retry = await register(
            "publisher", RankRole.PUBLISHER, "model", _registration()
        )
        return requester_result, publisher_retry

    requester_result, publisher_retry = asyncio.run(run())
    assert publisher_retry == requester_result


def test_state_dict_namespaces_have_independent_layouts() -> None:
    """Gather model and optimizer layouts independently."""

    async def run():
        coordinator = await _initialized_coordinator({"publisher", "requester"})
        register = _endpoint_method(coordinator, "register_layouts")
        tasks = []
        for namespace in ("model", "optimizer"):
            for rank, role in (
                ("publisher", RankRole.PUBLISHER),
                ("requester", RankRole.REQUESTER),
            ):
                tasks.append(
                    register(rank, role, namespace, _registration(f"{namespace}/w"))
                )
        return await asyncio.gather(*tasks)

    results = asyncio.run(run())
    assert len(results) == 4
    assert all(set(publishers) == {"publisher"} for publishers, _ in results)
    assert all(set(requesters) == {"requester"} for _, requesters in results)
