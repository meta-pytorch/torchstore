# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from __future__ import annotations

import asyncio
import importlib
import inspect
from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from monarch._src.actor.endpoint import EndpointProperty
from monarch.actor import Actor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate

import torchstore as ts
import torchstore.api as store_api
from torchstore.client import LocalClient
from torchstore.routing._model import RankRole
from torchstore.routing.client import RoutingClient
from torchstore.routing.plan import KeyRegistration, RoutingPlan
from torchstore.routing.service import RoutingService
from torchstore.state_dict_utils import (
    _state_dict_storage_metadata,
    get_state_dict,
    put_state_dict,
)
from torchstore.transport.types import TensorSlice
from torchstore.utils import slice_covers

client_module = importlib.import_module("torchstore.client")
routing_client_module = importlib.import_module("torchstore.routing.client")


def _slice(
    offsets,
    shape,
    *,
    global_shape=(8, 6),
    coordinates=(),
    mesh_shape=(),
) -> TensorSlice:
    return TensorSlice(
        offsets=offsets,
        coordinates=coordinates,
        global_shape=global_shape,
        local_shape=shape,
        mesh_shape=mesh_shape,
    )


def _ranks(entries, element_sizes):
    """Registrations per rank and key, as its client would report them."""
    return {
        rank: {
            name: KeyRegistration(item, element_sizes[name])
            for name, item in slices.items()
        }
        for rank, slices in entries.items()
    }


def test_routing_exposes_the_production_api() -> None:
    from torchstore import routing

    assert routing.__all__ == []
    assert not hasattr(ts, "describe_state_dict_routing")
    assert not hasattr(ts, "initialize_state_dict_routing")
    assert issubclass(RoutingClient, LocalClient)
    assert issubclass(RoutingService, Actor)
    for name in ("wait_ready", "notify_ready"):
        assert isinstance(getattr(RoutingService, name), EndpointProperty)
    wait = inspect.signature(_endpoint_method(RoutingService(id_func=lambda: "test"), "wait_ready"))
    notify = inspect.signature(
        _endpoint_method(RoutingService(id_func=lambda: "test"), "notify_ready")
    )
    assert tuple(wait.parameters) == ("generation", "relay_id")
    assert tuple(notify.parameters) == ("generation", "relay_id")


def test_routing_client_preserves_existing_client_signatures() -> None:
    for name in ("put", "put_batch", "get", "get_batch"):
        local = list(inspect.signature(getattr(LocalClient, name)).parameters.values())
        routed = list(
            inspect.signature(getattr(RoutingClient, name)).parameters.values()
        )
        assert [(item.name, item.default) for item in routed] == [
            (item.name, item.default) for item in local
        ]


def test_client_constructs_routing_client_in_calling_process(monkeypatch) -> None:
    calls = []

    class Client(LocalClient):
        def __init__(self, name: str) -> None:
            self.name = name

        async def put(self, key: str, value: object) -> None:
            calls.append(("put", self.name, key, value))

        async def put_batch(self, entries: dict[str, object]) -> None:
            calls.append(("put_batch", self.name, entries))

        async def get(self, key, inplace_tensor=None, tensor_slice_spec=None):
            return (self.name, key)

        async def get_batch(self, keys):
            return {key: (self.name, key) for key in keys}

    override = Client("override")
    created = []
    strategy = object()

    def create_routing_client(rank, role, coordinator, client_strategy):
        created.append((rank, role, client_strategy))
        return override

    class StrategyEndpoint:
        async def call_one(self):
            return strategy

    async def fake_coordinator(_name, _cls, **_kwargs):
        return SimpleNamespace(strategy=StrategyEndpoint())

    monkeypatch.setattr(store_api, "get_or_spawn_controller", fake_coordinator)
    monkeypatch.setattr(store_api, "current_rank", lambda: SimpleNamespace(rank=3))

    class Endpoint:
        async def call_one(self):
            return object()

        async def call(self):
            return None

    class Controller:
        get_controller_strategy = Endpoint()
        teardown = Endpoint()

    async def fake_controller(_store_name):
        return Controller()

    async def fake_put_state_dict(*, store, state_dict, key, **_kwargs):
        calls.append(("put_state_dict", store.name, key, state_dict))

    async def fake_get_state_dict(store, key, *_args, **_kwargs):
        calls.append(("get_state_dict", store.name, key))
        return {"store": store.name}

    monkeypatch.setattr(store_api, "_controller", fake_controller)
    monkeypatch.setattr(
        store_api,
        "RoutingClient",
        create_routing_client,
    )
    monkeypatch.setattr(
        store_api.torchstore.state_dict_utils,
        "put_state_dict",
        fake_put_state_dict,
    )
    monkeypatch.setattr(
        store_api.torchstore.state_dict_utils,
        "get_state_dict",
        fake_get_state_dict,
    )

    async def run():
        routed_client = await store_api.client("routing-test", role="publisher")
        await store_api.put_state_dict({}, "routed", store_name="routing-test")
        await store_api.put("ordinary", torch.ones(1), store_name="routing-test")
        routed = await store_api.get_state_dict("routed", store_name="routing-test")
        store_api.reset_client("routing-test")
        fallback_after = await store_api.client("routing-test")
        return routed_client, routed, fallback_after

    try:
        routed_client, routed, fallback_after = asyncio.run(run())
    finally:
        store_api.reset_client("routing-test")

    assert routed_client is override
    assert created == [("publisher/3", RankRole.PUBLISHER, strategy)]
    assert routed == {"store": "override"}
    assert isinstance(fallback_after, LocalClient)
    assert ("put_state_dict", "override", "routed", {}) in calls
    assert any(call[:3] == ("put", "override", "ordinary") for call in calls)


def test_cross_axis_reshard_and_replication_are_inferred_from_geometry() -> None:
    left0 = _slice((0, 0), (8, 3), coordinates=(0, 0), mesh_shape=(2, 2))
    left1 = _slice((0, 0), (8, 3), coordinates=(0, 1), mesh_shape=(2, 2))
    right0 = _slice((0, 3), (8, 3), coordinates=(1, 0), mesh_shape=(2, 2))
    right1 = _slice((0, 3), (8, 3), coordinates=(1, 1), mesh_shape=(2, 2))
    plan = RoutingPlan.build(
        _ranks(
            {
                "trainer-top": {"w": _slice((0, 0), (4, 6))},
                "trainer-bottom": {"w": _slice((4, 0), (4, 6))},
            },
            {"w": 2},
        ),
        _ranks(
            {
                "left-dp0": {"w": left0},
                "left-dp1": {"w": left1},
                "right-dp0": {"w": right0},
                "right-dp1": {"w": right1},
            },
            {"w": 2},
        ),
    )
    routes = [
        route for rank in plan.ranks for route in plan.lookup(rank, "w")
    ]
    publisher = [
        transfer
        for route in routes
        if route.wait_for_relay_id is None
        for transfer in route.transfers
    ]
    relay = [
        transfer
        for route in routes
        if route.wait_for_relay_id is not None
        for transfer in route.transfers
    ]

    assert sum(x.nbytes for x in publisher) == 8 * 6 * 2
    assert sum(x.nbytes for x in relay) == 8 * 6 * 2
    assert len(publisher) == 4
    assert len(relay) == 2
    relay_ids = {
        route.wait_for_relay_id
        for route in routes
        if route.wait_for_relay_id is not None
    }
    signal_ids = {
        route.notify_relay_id for route in routes if route.notify_relay_id is not None
    }
    assert relay_ids == signal_ids
    assert {x.source for x in publisher} == {"trainer-top", "trainer-bottom"}
    assert {x.source_volume_id for x in publisher} == {"trainer-top", "trainer-bottom"}


def test_missing_multidimensional_coverage_is_rejected_at_setup() -> None:
    with pytest.raises(ValueError, match="does not cover"):
        RoutingPlan.build(
            _ranks({"producer": {"w": _slice((0, 0), (7, 6))}}, {"w": 2}),
            _ranks({"consumer": {"w": _slice((0, 0), (8, 6))}}, {"w": 2}),
        )


def test_rank_local_plan_keeps_volume_ids() -> None:
    full = _slice((0, 0), (8, 6))
    sizes = {"model/w": 2}
    plan = RoutingPlan.build(
        _ranks({"trainer": {"model/w": full}}, sizes),
        _ranks({"generator": {"model/w": full}}, sizes),
    ).for_rank("generator")

    assert plan.ranks == ("generator",)
    transfer = plan.lookup("generator", "model/w")[0].transfers[0]
    assert transfer.source == "trainer"
    assert transfer.source_volume_id == "trainer"
    # The mapping is not planned: it comes from whichever volume holds the
    # tensors this rank reads.
    assert {
        key: entry.tensor_slice
        for key, entry in plan._local("generator").keys.items()
    } == {"model/w": full}


def test_plan_pairs_relay_notifications_with_waiters() -> None:
    full = _slice((0, 0), (8, 6))
    replica = _slice((0, 0), (8, 6), coordinates=(1,), mesh_shape=(2,))
    plan = RoutingPlan.build(
        _ranks({"trainer": {"w": full}}, {"w": 2}),
        _ranks(
            {"generator/0": {"w": full}, "generator/1": {"w": replica}},
            {"w": 2},
        ),
    )

    notifications = [
        route
        for rank in plan.ranks
        for route in plan.lookup(rank, "w")
        if route.notify_relay_id is not None
    ]
    relays = [
        route
        for rank in plan.ranks
        for route in plan.lookup(rank, "w")
        if route.wait_for_relay_id is not None
    ]
    assert len(notifications) == len(relays) == 1
    assert notifications[0].notify_relay_id == relays[0].wait_for_relay_id


class _MemoryTransport:
    supports_inplace_resharding = True

    def __init__(self, volume_id, stores, events) -> None:
        self.volume_id = volume_id
        self.stores = stores
        self.events = events

    async def put_to_storage_volume(self, requests) -> None:
        self.events.append(("put", self.volume_id, tuple(r.key for r in requests)))
        volume = self.stores[self.volume_id]
        for request in requests:
            if request.is_object:
                volume[request.key] = request.objects
                continue
            assert request.tensor_slice is not None
            assert request.tensor_val is not None
            volume[request.key] = (
                request.tensor_slice,
                request.tensor_val.detach().clone(),
            )

    async def get_from_storage_volume(self, requests):
        self.events.append(("get", self.volume_id, tuple(r.key for r in requests)))
        volume = self.stores[self.volume_id]
        results = []
        for request in requests:
            if request.is_object:
                results.append(volume[request.key])
                continue
            assert request.tensor_slice is not None
            requested = request.tensor_slice
            stored, tensor = volume[request.key]
            assert slice_covers(stored, requested)
            indices = tuple(
                slice(offset - start, offset - start + size)
                for offset, start, size in zip(
                    requested.offsets, stored.offsets, requested.local_shape
                )
            )
            result = tensor[indices].clone()
            if request.tensor_val is not None:
                request.tensor_val.copy_(result)
                result = request.tensor_val
            results.append(result)
        return results


class _TransportFactory:
    def __init__(self) -> None:
        self.stores = defaultdict(dict)
        self.events = []

    def __call__(self, volume_id):
        return _MemoryTransport(volume_id, self.stores, self.events)


class _Strategy:
    def __init__(self, factory: _TransportFactory) -> None:
        self.factory = factory

    def get_storage_volume(self, volume_id):
        return type("Volume", (), {"volume_id": volume_id})()


def _endpoint_method(service: RoutingService, name: str):
    declared = inspect.getattr_static(type(service), name)
    method = getattr(declared._method, "__wrapped__", declared._method)
    return method.__get__(service, type(service))


class _LocalEndpoint:
    def __init__(self, method) -> None:
        self._method = method

    async def call_one(self, *args, **kwargs):
        return await self._method(*args, **kwargs)


class _LocalHandle:
    def __init__(self, service: RoutingService) -> None:
        for name in ("wait_ready", "notify_ready"):
            setattr(self, name, _LocalEndpoint(_endpoint_method(service, name)))


def _role(rank: str) -> RankRole:
    return RankRole.PUBLISHER if "trainer" in rank else RankRole.REQUESTER


def _installed(client, plan, handles):
    """A client with one namespace's routes already installed."""
    client._controller.install(plan, handles)
    return client


def _clients(plan: RoutingPlan, factory: _TransportFactory):
    def create_transport(volume):
        return factory(volume.volume_id)

    client_module.create_transport_buffer = create_transport
    routing_client_module.create_transport_buffer = create_transport
    handles = {
        rank: _LocalHandle(RoutingService(id_func=lambda: "test"))
        for rank in plan.ranks
    }
    return {
        rank: _installed(
            RoutingClient(rank, _role(rank), None, _Strategy(factory)),
            plan,
            handles,
        )
        for rank in plan.ranks
    }


def test_state_dict_utils_remain_compatible_with_existing_clients() -> None:
    class LegacyClient:
        def __init__(self) -> None:
            self.values = {}

        async def put(self, key, value):
            self.values[key] = value

        async def put_batch(self, entries):
            self.values.update(entries)

        async def get(self, key):
            return self.values[key]

        async def get_batch(self, keys):
            return {key: self.values[key] for key in keys}

    source = {"weight": torch.arange(4)}
    store = LegacyClient()

    async def run():
        await put_state_dict(store, source, "model")
        return await get_state_dict(store, "model")

    result = asyncio.run(run())
    torch.testing.assert_close(result["weight"], source["weight"])


def test_real_destination_contents_and_multi_source_assembly() -> None:
    top = _slice((0, 0), (2, 4), global_shape=(4, 4))
    bottom = _slice((2, 0), (2, 4), global_shape=(4, 4))
    full = _slice((0, 0), (4, 4), global_shape=(4, 4))
    plan = RoutingPlan.build(
        _ranks(
            {"trainer/0": {"w": top}, "trainer/1": {"w": bottom}}, {"w": 4}
        ),
        _ranks({"generator/0": {"w": full}}, {"w": 4}),
    )
    factory = _TransportFactory()
    clients = _clients(plan, factory)

    async def run():
        await clients["trainer/0"].put_batch({"w": torch.arange(8).view(2, 4)})
        await clients["trainer/1"].put_batch(
            {"w": torch.arange(8, 16).view(2, 4)}
        )
        destination = torch.full((4, 4), -1)
        result = await clients["generator/0"].get("w", destination)
        allocated = await clients["generator/0"].get("w")
        return result, destination, allocated

    result, destination, allocated = asyncio.run(run())
    expected = torch.arange(16).view(4, 4)
    assert result is destination
    torch.testing.assert_close(destination, expected)
    torch.testing.assert_close(allocated, expected)
    get_volumes = {volume for kind, volume, _keys in factory.events if kind == "get"}
    assert get_volumes == {"trainer/0", "trainer/1"}


def test_state_dict_api_parity_mapping_is_local_and_controller_is_bypassed() -> None:
    source = {"weight": torch.arange(6).view(2, 3), "nested": {"bias": torch.ones(2)}}
    slices, sizes, _dtypes, _mapping = _state_dict_storage_metadata(source, "model")
    plan = RoutingPlan.build(
        _ranks({"trainer": slices}, sizes),
        _ranks({"generator": slices}, sizes),
    )
    factory = _TransportFactory()
    clients = _clients(plan, factory)

    async def run():
        await put_state_dict(clients["trainer"], source, "model")
        destination = {
            "weight": torch.zeros_like(source["weight"]),
            "nested": {"bias": torch.zeros_like(source["nested"]["bias"])},
        }
        result = await get_state_dict(clients["generator"], "model", destination)
        updated = {
            "weight": source["weight"] + 10,
            "nested": {"bias": source["nested"]["bias"] + 10},
        }
        await put_state_dict(clients["trainer"], updated, "model")
        second = await get_state_dict(clients["generator"], "model", destination)
        return result, destination, second, updated

    result, destination, second, updated = asyncio.run(run())
    torch.testing.assert_close(destination["weight"], updated["weight"])
    torch.testing.assert_close(destination["nested"]["bias"], updated["nested"]["bias"])
    assert result["weight"] is destination["weight"]
    assert second["weight"] is destination["weight"]
    mapping_events = [
        (kind, volume)
        for kind, volume, keys in factory.events
        if keys == ("model/MAPPING",)
    ]
    assert mapping_events == [
        ("put", "trainer"),
        ("get", "trainer"),
        ("put", "trainer"),
        ("get", "trainer"),
    ]


def test_ingress_publishes_mapping_during_registration() -> None:
    source = {"weight": torch.arange(4)}
    slices, sizes, _dtypes, mapping = _state_dict_storage_metadata(source, "model")
    plan = RoutingPlan.build(
        _ranks({"trainer": slices}, sizes),
        _ranks({"generator/a": slices, "generator/b": slices}, sizes),
    )
    factory = _TransportFactory()
    clients = _clients(plan, factory)

    async def register(**_kwargs):
        return plan, {}

    coordinator = SimpleNamespace(register=_LocalEndpoint(register))
    clients["generator/a"]._coordinator = coordinator
    clients["generator/b"]._coordinator = coordinator

    async def run():
        await asyncio.gather(
            clients["generator/a"].register_state_dict(source, "model"),
            clients["generator/b"].register_state_dict(source, "model"),
        )
        await put_state_dict(clients["trainer"], source, "model")
        destinations = [
            {"weight": torch.zeros_like(source["weight"])} for _ in range(2)
        ]
        return await asyncio.gather(
            get_state_dict(clients["generator/a"], "model", destinations[0]),
            get_state_dict(clients["generator/b"], "model", destinations[1]),
        )

    results = asyncio.run(run())

    ingress = next(
        rank
        for rank in ("generator/a", "generator/b")
        if "model/MAPPING" in factory.stores[rank]
    )
    assert factory.stores[ingress]["model/MAPPING"] == mapping
    for result in results:
        torch.testing.assert_close(result["weight"], source["weight"])


def test_routing_casts_each_key_to_its_registered_wire_dtype() -> None:
    source = {
        "weight": torch.arange(4, dtype=torch.float32),
        "running": torch.arange(4, dtype=torch.float32),
    }
    destination = {
        "weight": torch.zeros(4, dtype=torch.bfloat16),
        "running": torch.zeros(4, dtype=torch.float32),
    }
    preserved = frozenset({"running"})
    source_slices, source_sizes, _source_dtypes, _ = _state_dict_storage_metadata(
        source,
        "model",
        transfer_dtype=torch.bfloat16,
        preserve_dtype_keys=preserved,
    )
    dest_slices, dest_sizes, _dest_dtypes, _ = _state_dict_storage_metadata(
        destination, "model"
    )
    plan = RoutingPlan.build(
        _ranks({"trainer": source_slices}, source_sizes),
        _ranks({"generator": dest_slices}, dest_sizes),
    )
    factory = _TransportFactory()
    clients = _clients(plan, factory)

    async def register(**_kwargs):
        return plan, {}

    coordinator = SimpleNamespace(register=_LocalEndpoint(register))
    clients["trainer"]._coordinator = coordinator
    clients["generator"]._coordinator = coordinator

    async def run():
        await clients["trainer"].register_state_dict(
            source,
            "model",
            transfer_dtype=torch.bfloat16,
            preserve_dtype_keys=preserved,
        )
        await clients["generator"].register_state_dict(destination, "model")
        # The route owns the wire-dtype policy; callers publish their native tensors.
        await put_state_dict(clients["trainer"], source, "model")
        return await get_state_dict(clients["generator"], "model", destination)

    result = asyncio.run(run())
    stored = factory.stores["trainer"]
    assert stored["model/weight"][1].dtype == torch.bfloat16
    assert stored["model/running"][1].dtype == torch.float32
    assert result["weight"] is destination["weight"]
    assert result["running"] is destination["running"]
    torch.testing.assert_close(destination["weight"], source["weight"].bfloat16())
    torch.testing.assert_close(destination["running"], source["running"])


def test_fully_local_dtensor_state_dict_uses_local_tensor(tmp_path) -> None:
    if dist.is_initialized():
        pytest.skip("test requires ownership of the default process group")
    init_path = tmp_path / "routing-dtensor-init"
    dist.init_process_group(
        "gloo", init_method=f"file://{init_path}", rank=0, world_size=1
    )
    try:
        mesh = DeviceMesh("cpu", [0])
        source_tensor = torch.arange(4)
        destination_tensor = torch.zeros(4, dtype=torch.int64)
        source = DTensor.from_local(
            source_tensor, mesh, [Replicate()], run_check=False
        )
        destination = DTensor.from_local(
            destination_tensor, mesh, [Replicate()], run_check=False
        )
        slices, sizes, _dtypes, _mapping = _state_dict_storage_metadata(
            {"weight": source}, "model"
        )
        plan = RoutingPlan.build(
            _ranks({"trainer": slices}, sizes),
            _ranks({"generator": slices}, sizes),
        )
        factory = _TransportFactory()
        clients = _clients(plan, factory)

        async def run():
            await put_state_dict(clients["trainer"], {"weight": source}, "model")
            return await get_state_dict(
                clients["generator"], "model", {"weight": destination}
            )

        result = asyncio.run(run())
        assert result["weight"] is destination
        torch.testing.assert_close(destination.to_local(), source_tensor)
    finally:
        dist.destroy_process_group()


def test_relay_readiness_is_consumed_between_updates() -> None:
    full = _slice((0,), (4,), global_shape=(4,))
    replica = _slice(
        (0,), (4,), global_shape=(4,), coordinates=(1,), mesh_shape=(2,)
    )
    plan = RoutingPlan.build(
        _ranks({"trainer": {"w": full}}, {"w": 4}),
        _ranks(
            {"generator/a": {"w": full}, "generator/b": {"w": replica}},
            {"w": 4},
        ),
    )
    factory = _TransportFactory()
    clients = _clients(plan, factory)

    async def run():
        await clients["trainer"].put_batch({"w": torch.arange(4)})
        await clients["generator/a"].get(
            "w", torch.empty(4, dtype=torch.int64)
        )
        first = await clients["generator/b"].get(
            "w", torch.empty(4, dtype=torch.int64)
        )

        await clients["trainer"].put_batch({"w": torch.arange(4) + 10})
        second_task = asyncio.create_task(
            clients["generator/b"].get(
                "w", torch.empty(4, dtype=torch.int64)
            )
        )
        done, _pending = await asyncio.wait({second_task}, timeout=0.02)
        assert not done, "stale readiness incorrectly released the next update"
        await clients["generator/a"].get(
            "w", torch.empty(4, dtype=torch.int64)
        )
        second = await asyncio.wait_for(second_task, timeout=1)
        return first, second

    first, second = asyncio.run(run())
    torch.testing.assert_close(first, torch.arange(4))
    torch.testing.assert_close(second, torch.arange(4) + 10)


def test_batched_relay_fetches_ingress_keys_before_waiting_for_peers() -> None:
    full = _slice((0,), (4,), global_shape=(4,))
    replica = _slice(
        (0,), (4,), global_shape=(4,), coordinates=(1,), mesh_shape=(2,)
    )
    plan = RoutingPlan.build(
        _ranks({"trainer": {"a": full, "b": full}}, {"a": 8, "b": 8}),
        _ranks(
            {
                "generator/a": {"a": full, "b": full},
                "generator/b": {"a": replica, "b": replica},
            },
            {"a": 8, "b": 8},
        ),
    )
    factory = _TransportFactory()
    clients = _clients(plan, factory)

    async def run():
        expected = {"a": torch.arange(4), "b": torch.arange(4) + 10}
        await clients["trainer"].put_batch(expected)
        first, second = await asyncio.wait_for(
            asyncio.gather(
                clients["generator/a"].get_batch(["a", "b"]),
                clients["generator/b"].get_batch(["a", "b"]),
            ),
            timeout=1,
        )
        return expected, first, second

    expected, first, second = asyncio.run(run())
    for result in (first, second):
        torch.testing.assert_close(result["a"], expected["a"])
        torch.testing.assert_close(result["b"], expected["b"])
