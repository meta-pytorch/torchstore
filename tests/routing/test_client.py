# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import contextlib
import gc
from collections import Counter
from collections.abc import Collection, Mapping
from typing import Any

import monarch.actor as monarch_actor
import pytest
import torch
import torchstore as ts
from torchstore.routing import _gc
from torchstore.routing._model import KeyRegistration, RankRole, Registrations
from torchstore.routing.client import RoutingClient
from torchstore.routing.control_plane import ControlPlane
from torchstore.routing.plan import RoutingPlan
from torchstore.state_dict_utils import (
    _state_dict_storage_metadata,
    get_state_dict,
    put_state_dict,
)
from torchstore.transport import TransportType

from .utils import registered_clients, routing_clients, tensor_slice

# One deadline covers the whole sync and leaves the marker's other 60 s for cleanup.
_DEADLINE_S = 60.0


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


def _fsdp_layouts(world: int) -> Registrations:
    """Each publisher's slices under 1-D FSDP, unevenly chunked, plus a buffer every rank holds whole."""
    sharded = {
        "model/embed": ((15, 4), 0),
        "model/wqkv": ((24, 6), 0),
        "model/wo": ((6, 15), 1),
    }
    layouts = {}
    for rank in range(world):
        placement = {"coordinates": (rank,), "mesh_shape": (world,)}
        layout = {}
        for key, (shape, dim) in sharded.items():
            chunk = -(-shape[dim] // world)
            start = min(rank * chunk, shape[dim])
            offsets = tuple(start if d == dim else 0 for d in range(len(shape)))
            local = list(shape)
            local[dim] = min(chunk, shape[dim] - start)
            layout[key] = KeyRegistration(
                tensor_slice(offsets, tuple(local), global_shape=shape, **placement),
                dtype=torch.bfloat16,
            )
        layout["model/freqs"] = KeyRegistration(
            tensor_slice((0, 0), (4, 2), global_shape=(4, 2), **placement),
            dtype=torch.float32,
        )
        layouts[f"publisher/{rank}"] = layout
    return layouts


class _CountingControlPlane(ControlPlane):
    """Forwards to another control plane, counting the layout calls."""

    def __init__(self, inner: ControlPlane, calls: Counter[str]) -> None:
        self._inner = inner
        self._calls = calls

    async def init(
        self, publishers: Collection[str], strategy: ts.TorchStoreStrategy
    ) -> None:
        await self._inner.init(publishers, strategy)

    async def strategy(self) -> ts.TorchStoreStrategy:
        return await self._inner.strategy()

    async def register_layout(
        self, rank: str, key: str, registrations: Mapping[str, KeyRegistration]
    ) -> None:
        self._calls["register_layout"] += 1
        await self._inner.register_layout(rank, key, registrations)

    async def get_layouts(self, key: str) -> Registrations:
        self._calls["get_layouts"] += 1
        return await self._inner.get_layouts(key)


class _ManualClockLoop(asyncio.SelectorEventLoop):
    """Fires timers only once a test moves ``now`` past them."""

    now = 0.0

    def time(self) -> float:
        return self.now


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


@pytest.mark.timeout(60)
def test_a_requester_waits_as_long_as_its_control_plane() -> None:
    """Set no deadline of its own: still await a lookup the plane holds for a day."""
    control_plane, publisher, requester = _in_memory_clients()
    state_dict = {"w": torch.empty(4, dtype=torch.int64)}
    lookup = control_plane.get_layouts
    loop = _ManualClockLoop()

    async def spin() -> None:
        for _ in range(20):
            await asyncio.sleep(0)

    async def run() -> None:
        released = asyncio.Event()

        async def held_lookup(key: str) -> Registrations:
            await released.wait()
            return await lookup(key)

        control_plane.get_layouts = held_lookup
        await publisher.register_layout(state_dict, "model")
        waiting = asyncio.ensure_future(requester.get_layouts(state_dict, "model"))
        await spin()
        loop.now = 86_400.0
        await spin()
        assert not waiting.done(), f"the client gave up first: {waiting!r}"
        released.set()
        await waiting

    try:
        loop.run_until_complete(run())
    finally:
        loop.close()
    assert requester._routing_directory.routes.lookup("model/w") is not None


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


@pytest.mark.parametrize("enabled", [True, False], ids=["enabled", "disabled"])
@pytest.mark.parametrize("requested", ["w", "other"], ids=["plans", "planning_raises"])
@pytest.mark.asyncio
async def test_a_requester_pauses_collection_only_while_it_plans(
    monkeypatch: pytest.MonkeyPatch, requested: str, enabled: bool
) -> None:
    """Pause collection only while a requester plans its routes, then leave it as it was, also on a raise.

    The control plane's lookup, which can wait for minutes, and every other step run
    with collection as the caller left it.
    """
    assert gc.isenabled() and _gc._windows == 0
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
            after, windows = gc.isenabled(), _gc._windows
        else:
            with pytest.raises(
                KeyError, match="requester keys have no publisher"
            ) as raised:
                await requester.get_layouts(state_dict, "model")
            # Read before dropping the error, so only the requester's own close passes.
            after, windows = gc.isenabled(), _gc._windows
            del raised
    finally:
        gc.enable()
    steps = [("lookup", enabled), ("metadata", enabled), ("plan", False)]
    if requested == "w":
        steps.append(("install", enabled))
        assert directory.routes.lookup("model/w")
    assert seen == steps
    assert after is enabled and windows == 0


@pytest.mark.asyncio
async def test_a_paused_plan_matches_one_built_with_collection_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Install, from 8 synthetic FSDP layouts, the routes that planning with frequent collections builds."""
    assert gc.isenabled() and _gc._windows == 0
    enabled: list[bool] = []
    for_requester = RoutingPlan.for_requester

    def planned(*args: Any, **kwargs: Any) -> RoutingPlan:
        enabled.append(gc.isenabled())
        return for_requester(*args, **kwargs)

    monkeypatch.setattr(RoutingPlan, "for_requester", planned)
    layouts = _fsdp_layouts(8)
    control_plane = _InMemoryControlPlane()
    control_plane._layouts["model"] = layouts
    strategy = ts.TorchStoreStrategy(TransportType.MonarchRPC)
    requester = RoutingClient(
        "requester/0/0", RankRole.REQUESTER, control_plane, strategy
    )
    state_dict = {
        "embed": torch.empty(15, 4, dtype=torch.bfloat16),
        "wqkv": torch.empty(24, 6, dtype=torch.bfloat16),
        "wo": torch.empty(6, 15, dtype=torch.bfloat16),
        "freqs": torch.empty(4, 2),
    }
    await requester.get_layouts(state_dict, "model")

    slices, dtypes, _mapping = _state_dict_storage_metadata(state_dict, "model")
    registrations = {
        key: KeyRegistration(local_slice, dtype=dtypes[key])
        for key, local_slice in slices.items()
    }
    assert gc.isenabled()
    # Collect after every allocation, so a plan that depended on collection would differ.
    threshold = gc.get_threshold()
    collections = gc.get_stats()[0]["collections"]
    gc.set_threshold(1)
    try:
        expected = for_requester("requester/0/0", registrations, layouts).routes
    finally:
        gc.set_threshold(*threshold)
    assert gc.get_stats()[0]["collections"] > collections
    assert enabled == [False]
    assert requester._routing_directory.routes == expected
    transfers: list[int] = []
    for plan in expected.keys.values():
        assert plan.route is not None
        transfers.append(len(plan.route.transfers))
    # Sorted keys: embed, freqs, wo and wqkv. The buffer is read from one of its 8 holders.
    assert transfers == [8, 1, 8, 8]


@pytest.mark.timeout(120)
@pytest.mark.asyncio
async def test_control_plane_is_called_only_on_the_first_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exchange layouts once per rank on the first sync. Later syncs skip it."""
    # Monarch's default fault hook interrupts the whole session, so record this
    # test's mesh faults instead. monkeypatch restores the hook after any outcome.
    faults: list[monarch_actor.MeshFailure] = []
    monkeypatch.setattr(monarch_actor, "unhandled_fault_hook", faults.append)
    calls: Counter[str] = Counter()

    def published(round_: int) -> dict[str, dict[str, torch.Tensor]]:
        return {
            "publisher/0": {"first": torch.arange(2) + round_},
            "publisher/1": {"second": torch.arange(2, 4) - round_},
        }

    destinations = {
        rank: {
            "first": torch.zeros(2, dtype=torch.int64),
            "second": torch.zeros(2, dtype=torch.int64),
        }
        for rank in ("requester/0/0", "requester/1/0")
    }

    async def sync(stack: contextlib.AsyncExitStack) -> None:
        clients = await routing_clients(
            published(0),
            destinations,
            wrap_control_plane=lambda inner: _CountingControlPlane(inner, calls),
            exit_stack=stack,
        )

        for round_ in range(3):
            state_dicts = published(round_)
            for rank, state_dict in state_dicts.items():
                await put_state_dict(clients[rank], state_dict, "model")
            expected = {**state_dicts["publisher/0"], **state_dicts["publisher/1"]}
            for rank, destination in destinations.items():
                result = await get_state_dict(clients[rank], "model", destination)
                torch.testing.assert_close(result, expected)
            assert calls == {"register_layout": 2, "get_layouts": 2}, f"round {round_}"
        # A publisher that died or stalled after its last call fails here, while this
        # test's recorder still takes the fault Monarch reports after the mesh stops.
        strategy = clients["publisher/0"].strategy
        volumes = [strategy.get_storage_volume(rank).volume for rank in published(0)]
        await asyncio.gather(*(volume.get_id.call_one() for volume in volumes))

    async with contextlib.AsyncExitStack() as stack:
        await asyncio.wait_for(sync(stack), _DEADLINE_S)
    assert not faults, [fault.report() for fault in faults]
