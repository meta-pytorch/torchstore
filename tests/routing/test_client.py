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
import torchstore.routing.client as client_module
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


def _record_steps(
    monkeypatch: pytest.MonkeyPatch,
    control_plane: ControlPlane,
    requester: RoutingClient,
) -> list[tuple[str, bool]]:
    """Record each planning step of ``requester`` with whether collection was on."""
    seen: list[tuple[str, bool]] = []

    def recorded(step: str, call: Any) -> Any:
        def record(*args: Any, **kwargs: Any) -> Any:
            seen.append((step, gc.isenabled()))
            return call(*args, **kwargs)

        return record

    class RecordedPublishers(dict):
        """The layouts the lookup returns, recording the filter's read."""

        def items(self) -> Any:
            seen.append(("filter", gc.isenabled()))
            return super().items()

    get_layouts = control_plane.get_layouts

    async def lookup(key: str) -> Registrations:
        # Recorded as the lookup runs, not when the requester creates its coroutine.
        seen.append(("lookup", gc.isenabled()))
        return RecordedPublishers(await get_layouts(key))

    class RecordedRequested(set):
        """The requester's requested keys, recording the new-entry check and each update."""

        def __contains__(self, key: object) -> bool:
            seen.append(("check", gc.isenabled()))
            return super().__contains__(key)

        def update(self, *keys: Any) -> None:
            seen.append(("requested", gc.isenabled()))
            super().update(*keys)

    class RecordedLayouts(set):
        """The requester's planned keys, recording each addition."""

        def add(self, key: str) -> None:
            seen.append(("bookkeeping", gc.isenabled()))
            super().add(key)

    class RecordedPlan(RoutingPlan):
        """The client module's plan class, recording each requester plan."""

        @classmethod
        def for_requester(cls, *args: Any) -> RoutingPlan:
            seen.append(("plan", gc.isenabled()))
            return RoutingPlan.for_requester(*args)

    paused = client_module.call_gc_paused

    def recorded_pause(fn: Any, *args: Any) -> Any:
        seen.append(("pause", gc.isenabled()))
        try:
            return paused(fn, *args)
        finally:
            seen.append(("resume", gc.isenabled()))

    monkeypatch.setattr(client_module, "call_gc_paused", recorded_pause)
    directory = requester._routing_directory
    monkeypatch.setattr(requester, "_requested_keys", {"model": RecordedRequested()})
    monkeypatch.setattr(requester, "_registered_layouts", RecordedLayouts())
    monkeypatch.setattr(
        client_module,
        "flatten_state_dict",
        recorded("flatten", client_module.flatten_state_dict),
    )
    monkeypatch.setattr(
        client_module, "KeyRegistration", recorded("registration", KeyRegistration)
    )
    monkeypatch.setattr(control_plane, "get_layouts", lookup)
    monkeypatch.setattr(
        client_module,
        "_state_dict_storage_metadata",
        recorded("metadata", _state_dict_storage_metadata),
    )
    monkeypatch.setattr(client_module, "RoutingPlan", RecordedPlan)
    monkeypatch.setattr(directory, "install", recorded("install", directory.install))
    return seen


def _steps_to_the_plan(enabled: bool) -> list[tuple[str, bool]]:
    """The steps a first strict sync of one entry takes up to its plan's resume."""
    return [
        ("flatten", enabled),
        ("check", enabled),
        ("lookup", enabled),
        ("metadata", enabled),
        ("filter", enabled),
        ("registration", enabled),
        ("pause", enabled),
        ("plan", False),
        ("resume", enabled),
    ]


@pytest.mark.parametrize("enabled", [True, False], ids=["enabled", "disabled"])
@pytest.mark.asyncio
async def test_a_requester_pauses_collection_only_while_it_plans(
    monkeypatch: pytest.MonkeyPatch, enabled: bool
) -> None:
    """Pause collection only while a requester plans its routes, then leave it as it was.

    The pause opens just before planning and closes just after it, so every other step runs
    with collection as the caller left it.
    """
    assert gc.isenabled()
    control_plane, publisher, requester = _in_memory_clients()
    await publisher.register_layout({"w": torch.empty(4, dtype=torch.int64)}, "model")
    seen = _record_steps(monkeypatch, control_plane, requester)
    if not enabled:
        gc.disable()
    try:
        await requester.get_layouts({"w": torch.empty(4, dtype=torch.int64)}, "model")
        after = gc.isenabled()
    finally:
        gc.enable()
    assert seen == _steps_to_the_plan(enabled) + [
        ("install", enabled),
        ("requested", enabled),
        ("bookkeeping", enabled),
    ]
    assert requester._routing_directory.routes.lookup("model/w")
    assert after is enabled


@pytest.mark.parametrize("enabled", [True, False], ids=["enabled", "disabled"])
@pytest.mark.asyncio
async def test_a_requester_whose_plan_raises_leaves_collection_as_it_was(
    monkeypatch: pytest.MonkeyPatch, enabled: bool
) -> None:
    """Resume collection, if the caller left it on, after a strict plan raises."""
    control_plane, publisher, requester = _in_memory_clients()
    await publisher.register_layout({"w": torch.empty(4, dtype=torch.int64)}, "model")
    seen = _record_steps(monkeypatch, control_plane, requester)
    if not enabled:
        gc.disable()
    try:
        with pytest.raises(KeyError, match="requester keys have no publisher"):
            await requester.get_layouts({"other": torch.empty(1)}, "model")
        after = gc.isenabled()
    finally:
        gc.enable()
    assert seen == _steps_to_the_plan(enabled)
    assert after is enabled


@pytest.mark.asyncio
async def test_a_repeat_sync_of_planned_entries_pauses_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return after the new-entry check when a sync passes only planned entries."""
    control_plane, publisher, requester = _in_memory_clients()
    await publisher.register_layout({"w": torch.empty(4, dtype=torch.int64)}, "model")
    seen = _record_steps(monkeypatch, control_plane, requester)
    state_dict = {"w": torch.empty(4, dtype=torch.int64)}
    await requester.get_layouts(state_dict, "model")
    seen.clear()
    await requester.get_layouts(state_dict, "model")
    assert seen == [("flatten", True), ("check", True)]


@pytest.mark.asyncio
async def test_a_non_strict_sync_pauses_only_for_published_new_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pause only to plan a published new entry, never for a requester-only one, which stays unplanned."""
    control_plane, publisher, requester = _in_memory_clients()
    await publisher.register_layout({"w": torch.empty(4, dtype=torch.int64)}, "model")
    paused = client_module.call_gc_paused
    pauses: list[list[str]] = []

    def recorded_pause(fn: Any, rank: str, registrations: Any, publishers: Any) -> Any:
        pauses.append(sorted(registrations))
        return paused(fn, rank, registrations, publishers)

    monkeypatch.setattr(client_module, "call_gc_paused", recorded_pause)
    state_dict = {
        "w": torch.empty(4, dtype=torch.int64),
        "local": torch.empty(2, dtype=torch.int64),
    }
    for _ in range(2):
        await requester.get_layouts(state_dict, "model", strict=False)
    assert pauses == [["model/w"]]
    assert requester._routing_directory.routes.lookup("model/w")


@pytest.mark.asyncio
async def test_a_first_non_strict_sync_of_only_unpublished_entries_installs_an_empty_plan() -> (
    None
):
    """Install an empty plan without touching collection. Later syncs of the key need no state dict."""
    _, publisher, requester = _in_memory_clients()
    await publisher.register_layout({"w": torch.empty(4, dtype=torch.int64)}, "model")
    state_dict = {"local": torch.empty(2, dtype=torch.int64)}
    gc.disable()
    try:
        for _ in range(2):
            await requester.get_layouts(state_dict, "model", strict=False)
            assert not gc.isenabled()
    finally:
        gc.enable()
    await requester.get_layouts(None, "model")
    with pytest.raises(KeyError, match="no metadata for key 'model/local'"):
        requester._routing_directory.routes.lookup("model/local")
