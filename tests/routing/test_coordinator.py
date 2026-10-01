# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import uuid

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
    layouts, _ = await asyncio.gather(
        coordinator.get_layouts.call_one("model"),
        coordinator.register_layout.call_one(
            "publisher/1", "model", _registration("model/second")
        ),
    )

    assert set(layouts) == {"publisher/0", "publisher/1"}


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
    layouts = await coordinator.get_layouts.call_one("model")

    assert set(layouts["publisher"]) == {"model/w"}


@pytest.mark.asyncio
async def test_timeout_preserves_layout_for_retry(monkeypatch) -> None:
    """Report missing publishers and allow requester lookup to retry."""
    timeout = coordinator_module._LAYOUT_REGISTRATION_TIMEOUT_S
    monkeypatch.setattr(coordinator_module, "_LAYOUT_REGISTRATION_TIMEOUT_S", 0)
    coordinator = await _coordinator({"publisher"})

    with pytest.raises(ActorError, match=r"missing publishers: \['publisher'\]"):
        await asyncio.wait_for(
            coordinator.get_layouts.call_one("model"),
            timeout=1,
        )

    monkeypatch.setattr(coordinator_module, "_LAYOUT_REGISTRATION_TIMEOUT_S", timeout)
    await coordinator.register_layout.call_one("publisher", "model", _registration())
    layouts = await coordinator.get_layouts.call_one("model")
    assert set(layouts) == {"publisher"}


@pytest.mark.asyncio
async def test_state_dict_namespaces_have_independent_layouts() -> None:
    """Gather model and optimizer publisher layouts independently."""
    coordinator = await _coordinator({"publisher"})
    for namespace in ("model", "optimizer"):
        await coordinator.register_layout.call_one(
            "publisher", namespace, _registration(f"{namespace}/w")
        )

    results = await asyncio.gather(
        *(
            coordinator.get_layouts.call_one(namespace)
            for namespace in ("model", "optimizer")
        )
    )
    assert all(set(layouts) == {"publisher"} for layouts in results)
