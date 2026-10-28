# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import gc
from collections import Counter
from collections.abc import Collection, Mapping
from typing import Any

import pytest
import torch
import torchstore as ts
from torchstore.routing._model import KeyRegistration, RankRole, Registrations
from torchstore.routing.client import RoutingClient
from torchstore.routing.control_plane import ControlPlane
from torchstore.routing.plan import RoutingPlan
from torchstore.state_dict_utils import _state_dict_storage_metadata
from torchstore.transport import TransportType

from .utils import registered_clients


async def _publisher_and_requester_clients():
    return await registered_clients(
        publishers={"publisher/0": {"w": torch.empty(4, dtype=torch.int64)}},
        requesters={"requester/0/0": {"w": torch.empty(4, dtype=torch.int64)}},
    )


class _InMemoryControlPlane(ControlPlane):
    """Serves the layouts registered so far, counting the calls it gets.

    Its lookup takes only the key, the interface's minimal form.
    """

    def __init__(self) -> None:
        self.calls: Counter[str] = Counter()
        self._layouts: dict[str, Registrations] = {}

    async def init(
        self, publishers: Collection[str], strategy: ts.TorchStoreStrategy
    ) -> None:
        raise NotImplementedError

    async def strategy(self) -> ts.TorchStoreStrategy:
        raise NotImplementedError

    async def register_layout(
        self, rank: str, key: str, registrations: Mapping[str, KeyRegistration]
    ) -> None:
        self.calls["register_layout"] += 1
        self._layouts.setdefault(key, {}).setdefault(rank, registrations)

    async def get_layouts(self, key: str) -> Registrations:
        self.calls["get_layouts"] += 1
        return self._layouts[key]


def _in_memory_clients() -> tuple[_InMemoryControlPlane, RoutingClient, RoutingClient]:
    """A publisher and a requester client sharing one in-memory control plane."""
    control_plane = _InMemoryControlPlane()
    strategy = ts.TorchStoreStrategy(TransportType.MonarchRPC)
    return (
        control_plane,
        RoutingClient("publisher/0", RankRole.PUBLISHER, control_plane, strategy),
        RoutingClient("requester/0/0", RankRole.REQUESTER, control_plane, strategy),
    )


@pytest.mark.asyncio
async def test_publisher_rejects_unknown_keys_and_wrong_shapes() -> None:
    """Reject tensors that do not match the publisher's installed plan."""
    publisher = (await _publisher_and_requester_clients())["publisher/0"]

    with pytest.raises(KeyError, match="does not store"):
        await publisher.put_batch({"other": torch.empty(4)})
    with pytest.raises(ValueError, match="has shape .* expected"):
        await publisher.put_batch({"model/w": torch.empty(3)})


@pytest.mark.asyncio
async def test_requester_rejects_a_wrong_destination_shape() -> None:
    """Reject an in-place buffer that cannot hold the planned requester slice."""
    requester = (await _publisher_and_requester_clients())["requester/0/0"]

    with pytest.raises(ValueError, match="destination .* has shape .* expected"):
        await requester.get("model/w", torch.empty(3))


@pytest.mark.asyncio
async def test_requester_rejects_an_unplanned_tensor_key() -> None:
    """Do not misclassify a tensor missing from the route table as an object."""
    requester = (await _publisher_and_requester_clients())["requester/0/0"]

    with pytest.raises(KeyError, match="has no tensor route"):
        await requester.get("model/other")


@pytest.mark.parametrize("operation", ["keys", "delete", "delete_batch", "exists"])
@pytest.mark.asyncio
async def test_unsupported_operations_fail_explicitly(operation: str) -> None:
    """Expose unsupported mutable-store operations consistently."""
    client = (await _publisher_and_requester_clients())["requester/0/0"]
    arguments = {
        "keys": (),
        "delete": ("key",),
        "delete_batch": (["key"],),
        "exists": ("key",),
    }

    with pytest.raises(NotImplementedError, match=f"{operation} is unsupported"):
        await getattr(client, operation)(*arguments[operation])


@pytest.mark.asyncio
async def test_registration_methods_install_local_plans() -> None:
    """Send local metadata to the coordinator and install the returned plan."""
    requester = (await _publisher_and_requester_clients())["requester/0/0"]

    assert requester._routing_directory.routes.lookup("model/w")


@pytest.mark.asyncio
async def test_routing_client_runs_on_any_control_plane() -> None:
    """Exchange layouts through a non-Monarch control plane, once per state dict."""
    control_plane, publisher, requester = _in_memory_clients()
    state_dict = {"w": torch.empty(4, dtype=torch.int64)}
    for _ in range(2):
        for key in ("model", "optimizer"):
            await publisher.register_layout(state_dict, key)
            await requester.get_layouts(state_dict, key)

    assert control_plane.calls == {"register_layout": 2, "get_layouts": 2}
    for key in ("model", "optimizer"):
        assert publisher._routing_directory.routes.is_planned_tensor(f"{key}/w")
        route = requester._routing_directory.routes.lookup(f"{key}/w")
        assert route is not None
        [transfer] = route.transfers
        assert transfer.source == "publisher/0"


@pytest.mark.asyncio
async def test_a_failed_layout_exchange_is_retried_on_the_next_sync() -> None:
    """Call the control plane again after a rank's first exchange fails."""
    control_plane, publisher, requester = _in_memory_clients()
    state_dict = {"w": torch.empty(4, dtype=torch.int64)}
    register_layout = control_plane.register_layout

    async def fail_once(*args: Any, **kwargs: Any) -> None:
        control_plane.register_layout = register_layout
        raise RuntimeError("control plane unavailable")

    control_plane.register_layout = fail_once
    with pytest.raises(RuntimeError, match="unavailable"):
        await publisher.register_layout(state_dict, "model")
    # Nothing is registered yet, so the requester's first lookup fails too.
    with pytest.raises(KeyError, match="'model'"):
        await requester.get_layouts(state_dict, "model")
    await publisher.register_layout(state_dict, "model")
    await requester.get_layouts(state_dict, "model")

    assert control_plane.calls == {"register_layout": 1, "get_layouts": 2}
    route = requester._routing_directory.routes.lookup("model/w")
    assert route is not None
    [transfer] = route.transfers
    assert transfer.source == "publisher/0"


@pytest.mark.parametrize(
    "requested, enabled",
    [("w", True), ("other", True), ("other", False)],
    ids=["plans-enabled", "planning_raises-enabled", "planning_raises-disabled"],
)
@pytest.mark.asyncio
async def test_a_requester_pauses_collection_only_while_it_plans(
    monkeypatch: pytest.MonkeyPatch, requested: str, enabled: bool
) -> None:
    """Pause collection only while a requester plans its routes, then leave it as it was, also on a raise.

    The control plane's lookup, which can wait for minutes, and every other step run
    with collection as the caller left it.
    """
    assert gc.isenabled()
    control_plane, publisher, requester = _in_memory_clients()
    await publisher.register_layout({"w": torch.empty(4, dtype=torch.int64)}, "model")
    seen: list[tuple[str, bool]] = []

    def recorded(step: str, call: Any) -> Any:
        def record(*args: Any, **kwargs: Any) -> Any:
            seen.append((step, gc.isenabled()))
            return call(*args, **kwargs)

        return record

    get_layouts = control_plane.get_layouts

    async def lookup(key: str) -> Registrations:
        # Recorded as the lookup runs, not when the requester creates its coroutine.
        seen.append(("lookup", gc.isenabled()))
        return await get_layouts(key)

    directory = requester._routing_directory
    monkeypatch.setattr(control_plane, "get_layouts", lookup)
    monkeypatch.setattr(
        "torchstore.routing.client._state_dict_storage_metadata",
        recorded("metadata", _state_dict_storage_metadata),
    )
    monkeypatch.setattr(
        RoutingPlan, "for_requester", recorded("plan", RoutingPlan.for_requester)
    )
    monkeypatch.setattr(directory, "install", recorded("install", directory.install))
    state_dict = {requested: torch.empty(4, dtype=torch.int64)}
    if not enabled:
        gc.disable()
    try:
        if requested == "w":
            await requester.get_layouts(state_dict, "model")
            after = gc.isenabled()
        else:
            with pytest.raises(
                KeyError, match="requester keys have no publisher"
            ) as raised:
                await requester.get_layouts(state_dict, "model")
            # Read before dropping the error, so only the requester's own close passes.
            after = gc.isenabled()
            del raised
    finally:
        gc.enable()
    steps = [("lookup", enabled), ("metadata", enabled), ("plan", False)]
    if requested == "w":
        steps.append(("install", enabled))
        assert directory.routes.lookup("model/w")
    assert seen == steps
    assert after is enabled
