# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import importlib
import inspect
from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

import torchstore as ts
import torchstore.api as store_api
from torchstore.routing._model import RankRole
from torchstore.routing.client import RoutingClient
from torchstore.routing._model import KeyRegistration
from torchstore.routing.coordinator import RoutingCoordinator
from torchstore.routing.directory import RoutingDirectory
from torchstore.routing.service import RoutingService, RoutingServiceGroup
from torchstore.state_dict_utils import _state_dict_storage_metadata
from torchstore.storage_volume import StorageVolume
from torchstore.strategy import MultiMeshStrategy

client_module = importlib.import_module("torchstore.client")
routing_client_module = importlib.import_module("torchstore.routing.client")

STATE_DICT_KEY = "model_state_dict"


class _MemoryTransport:
    supports_inplace_resharding = True

    def __init__(self, volume_id, stores) -> None:
        self._volume_id = volume_id
        self._stores = stores

    async def put_to_storage_volume(self, requests) -> None:
        for request in requests:
            self._stores[self._volume_id][request.key] = (
                request.objects
                if request.is_object
                else request.tensor_val.clone()
            )

    async def get_from_storage_volume(self, requests):
        results = []
        for request in requests:
            stored = self._stores[self._volume_id][request.key]
            result = stored if request.is_object else stored.clone()
            if request.tensor_val is not None:
                request.tensor_val.copy_(result)
                result = request.tensor_val
            results.append(result)
        return results


class _TransportFactory:
    def __init__(self) -> None:
        self.stores = defaultdict(dict)

    def __call__(self, volume_id):
        return _MemoryTransport(volume_id, self.stores)


class _Strategy:
    def get_storage_volume(self, volume_id):
        return SimpleNamespace(volume_id=volume_id)


def _endpoint_method(actor, name: str):
    declared = inspect.getattr_static(type(actor), name)
    method = getattr(declared._method, "__wrapped__", declared._method)
    return method.__get__(actor, type(actor))


class _Services(RoutingServiceGroup):
    """A service group whose ranks are known without spawning anything."""

    def __init__(self, ranks) -> None:
        super().__init__()
        self.rank_to_mesh = {rank: None for rank in ranks}

    def get_service(self, rank):
        return None


def _coordinator(ranks) -> RoutingCoordinator:
    """A coordinator whose roster is already installed, as initialize does."""
    coordinator = RoutingCoordinator()
    asyncio.run(
        _endpoint_method(coordinator, "init")(
            services=_Services(ranks),
            strategy=_Strategy(),
        )
    )
    return coordinator


def _registrations(state_dict, key):
    """What one rank reports for a state dict: one registration per key."""
    slices, element_sizes, _dtypes, _mapping = _state_dict_storage_metadata(
        state_dict, key
    )
    return {
        name: KeyRegistration(tensor_slice, element_sizes[name])
        for name, tensor_slice in slices.items()
    }


def _register(coordinator, ranks, key=STATE_DICT_KEY):
    """Register every rank concurrently and return each rank's result."""

    async def run():
        register = _endpoint_method(coordinator, "register")
        results = await asyncio.gather(
            *(
                register(
                    rank=rank,
                    role=role,
                    key=key,
                    registrations=_registrations(state_dict, key),
                )
                for rank, (role, state_dict) in ranks.items()
            )
        )
        return dict(zip(ranks, results))

    return asyncio.run(run())


def test_state_dict_metadata_uses_torchstore_keys_and_tensor_dtypes() -> None:
    slices, element_sizes, dtypes, _mapping = _state_dict_storage_metadata(
        {
            "layer": {"weight": torch.empty(3, 5)},
            "running": torch.empty(2, dtype=torch.float32),
        },
        STATE_DICT_KEY,
        transfer_dtype=torch.bfloat16,
        preserve_dtype_keys=frozenset({"running"}),
    )

    weight_key = "model_state_dict/layer.weight"
    weight_slice = slices[weight_key]
    assert weight_slice.offsets == (0, 0)
    assert weight_slice.global_shape == (3, 5)
    assert weight_slice.local_shape == (3, 5)
    assert element_sizes[weight_key] == 2
    assert element_sizes["model_state_dict/running"] == 4
    assert dtypes[weight_key] == torch.bfloat16
    assert dtypes["model_state_dict/running"] == torch.float32


def test_plan_separates_logical_ranks_and_physical_volumes() -> None:
    weight = {"weight": torch.empty(4, dtype=torch.bfloat16)}
    coordinator = _coordinator(["trainer/0", "generator/2/0"])

    results = _register(
        coordinator,
        {
            "trainer/0": (RankRole.PUBLISHER, weight),
            "generator/2/0": (RankRole.REQUESTER, weight),
        },
    )

    plan = results["generator/2/0"][0]
    route = plan.lookup("generator/2/0", "model_state_dict/weight")[0]
    assert len(route.transfers) == 1
    assert route.transfers[0].source == "trainer/0"
    assert route.transfers[0].source_volume_id == "trainer/0"


def test_plan_allows_rank_local_state_dict_subsets() -> None:
    first = {"first": torch.empty(2)}
    second = {"second": torch.empty(2)}
    coordinator = _coordinator(
        ["trainer/0", "trainer/1", "generator/0/0", "generator/0/1"]
    )

    results = _register(
        coordinator,
        {
            "trainer/0": (RankRole.PUBLISHER, first),
            "trainer/1": (RankRole.PUBLISHER, second),
            "generator/0/0": (RankRole.REQUESTER, first),
            "generator/0/1": (RankRole.REQUESTER, second),
        },
    )

    plan = results["generator/0/0"][0]
    assert plan.lookup("generator/0/0", "model_state_dict/first")
    assert set(plan._local("generator/0/0").keys) == {"model_state_dict/first"}


def test_registration_from_an_unexpected_rank_is_rejected() -> None:
    coordinator = _coordinator(["trainer/0", "generator/0/0"])

    with pytest.raises(KeyError, match="not a routing participant"):
        _register(
            coordinator,
            {"trainer/9": (RankRole.PUBLISHER, {"weight": torch.empty(2)})},
        )


def test_requester_keys_without_publishers_are_not_routed() -> None:
    coordinator = _coordinator(["trainer/0", "generator/0/0"])

    results = _register(
        coordinator,
        {
            "trainer/0": (RankRole.PUBLISHER, {"first": torch.empty(2)}),
            "generator/0/0": (RankRole.REQUESTER, {"second": torch.empty(2)}),
        },
    )

    assert not results["generator/0/0"][0]._local("generator/0/0").keys


def test_metadata_drives_routing_state_dict_api() -> None:
    source = {"layer": {"weight": torch.arange(6).view(2, 3)}}
    destination = {"layer": {"weight": torch.zeros(2, 3, dtype=torch.long)}}
    coordinator = _coordinator(["trainer/0", "generator/0/0"])
    results = _register(
        coordinator,
        {
            "trainer/0": (RankRole.PUBLISHER, source),
            "generator/0/0": (RankRole.REQUESTER, destination),
        },
    )
    factory = _TransportFactory()

    def create_transport(volume):
        return factory(volume.volume_id)

    client_module.create_transport_buffer = create_transport
    routing_client_module.create_transport_buffer = create_transport
    strategy = _Strategy()
    def client(rank, role):
        item = RoutingClient(rank, role, None, strategy)
        item._controller.install(results[rank][0], {})
        return item

    publisher = client("trainer/0", RankRole.PUBLISHER)
    requester = client("generator/0/0", RankRole.REQUESTER)

    async def run():
        store_api._local_clent_map["routing-test"] = publisher
        await ts.put_state_dict(
            source, STATE_DICT_KEY, store_name="routing-test"
        )
        store_api._local_clent_map["routing-test"] = requester
        return await ts.get_state_dict(
            STATE_DICT_KEY,
            destination,
            strict=False,
            store_name="routing-test",
        )

    try:
        result = asyncio.run(run())
    finally:
        ts.reset_client("routing-test")
    torch.testing.assert_close(result["layer"]["weight"], source["layer"]["weight"])


def test_routing_initialization_spawns_volumes_services_and_coordinator(
    monkeypatch,
) -> None:
    events: list[str] = []

    class ResultEndpoint:
        def __init__(self, value):
            self.value = value

        async def call(self):
            return self.value

        async def call_one(self, *args, **kwargs):
            return (args, kwargs)

    class Resources:
        def __init__(self, name, entry):
            self.name = name
            self.get_id = ResultEndpoint([({"gpus": 0}, entry)])

        def slice(self, **coord):
            return f"{self.name}/{sorted(coord.items())}"

    async def spawn_volume(num_volumes=1, mesh=None, id_func=None):
        del num_volumes
        events.append(f"spawn-volume:{mesh}")
        return Resources(f"volume:{mesh}", (id_func(), "host"))

    async def spawn_service(mesh, id_func=None):
        events.append(f"spawn-service:{mesh}")
        return Resources(f"service:{mesh}", id_func())

    class InitEndpoint:
        def __init__(self):
            self.kwargs = None

        async def call_one(self, **kwargs):
            self.kwargs = kwargs

    class Coordinator:
        def __init__(self):
            self.init = InitEndpoint()

    coordinator = Coordinator()

    async def spawn_coordinator(name, cls, **_kwargs):
        events.append(f"spawn-coordinator:{name}")
        assert cls is RoutingCoordinator
        return coordinator

    monkeypatch.setattr(StorageVolume, "spawn", spawn_volume)
    monkeypatch.setattr(RoutingService, "spawn", spawn_service)
    monkeypatch.setattr(store_api, "get_or_spawn_controller", spawn_coordinator)
    monkeypatch.setattr(store_api, "current_rank", lambda: SimpleNamespace(rank=0))

    async def reject_controller(_store_name):
        raise AssertionError("routing initialization must not create a controller")

    monkeypatch.setattr(store_api, "_controller", reject_controller)

    asyncio.run(
        ts.initialize(
            mesh="publisher-mesh",
            relay_meshes=["mesh-0", "mesh-1"],
            strategy=MultiMeshStrategy(),
            store_name="routing-init",
        )
    )

    assert events == [
        "spawn-volume:publisher-mesh",
        "spawn-volume:mesh-0",
        "spawn-volume:mesh-1",
        "spawn-service:publisher-mesh",
        "spawn-service:mesh-0",
        "spawn-service:mesh-1",
        "spawn-coordinator:routing-init",
    ]
    # A rank name is its volume ID; the namespace keeps meshes that both number
    # their ranks from zero apart.
    assert coordinator.init.kwargs["services"].ranks == frozenset(
        {"publisher/0", "requester/0/0", "requester/1/0"}
    )


def test_participants_name_themselves_the_way_volumes_are_named(monkeypatch) -> None:
    monkeypatch.setattr(store_api, "current_rank", lambda: SimpleNamespace(rank=3))

    def rank(role, group):
        return store_api._routing_volume_id(
            store_api._routing_namespace(role, group)
        )

    assert rank(RankRole.PUBLISHER, None) == "publisher/3"
    assert rank(RankRole.REQUESTER, 1) == "requester/1/3"

    with pytest.raises(ValueError, match="take no group"):
        store_api._routing_namespace(RankRole.PUBLISHER, 0)
    with pytest.raises(ValueError, match="must pass the index"):
        store_api._routing_namespace(RankRole.REQUESTER, None)


def test_multi_mesh_strategy_does_not_name_or_select_volumes() -> None:
    with pytest.raises(NotImplementedError, match="does not name volumes"):
        MultiMeshStrategy.get_volume_id()
    with pytest.raises(NotImplementedError, match="does not map a process"):
        MultiMeshStrategy.get_client_id()


def test_each_state_dict_has_its_own_barrier_and_merges_into_one_table() -> None:
    ranks = {
        "trainer/0": (RankRole.PUBLISHER, {"w": torch.empty(2)}),
        "generator/0/0": (RankRole.REQUESTER, {"w": torch.empty(2)}),
    }
    coordinator = _coordinator(["trainer/0", "generator/0/0"])

    model = _register(coordinator, ranks, key="model")
    optim = _register(coordinator, ranks, key="optim")

    directory = RoutingDirectory("generator/0/0")
    directory.install(model["generator/0/0"][0], {})
    directory.install(optim["generator/0/0"][0], {})

    # Both namespaces route, and their storage keys keep them apart.
    assert set(directory.routes.keys) == {"model/w", "optim/w"}
    assert directory.routes.lookup("model/w")
    assert directory.routes.lookup("optim/w")


def test_relay_ids_do_not_collide_across_state_dicts() -> None:
    replicas = {
        "trainer/0": (RankRole.PUBLISHER, {"w": torch.empty(2)}),
        "generator/0/0": (RankRole.REQUESTER, {"w": torch.empty(2)}),
        "generator/0/1": (RankRole.REQUESTER, {"w": torch.empty(2)}),
    }
    coordinator = _coordinator(list(replicas))

    def relay_ids(key):
        plan = _register(coordinator, replicas, key=key)["trainer/0"][0]
        return {
            route.notify_relay_id or route.wait_for_relay_id
            for rank in plan.ranks
            for entry in plan._local(rank).keys.values()
            for route in entry.routes
        } - {None}

    assert not relay_ids("model") & relay_ids("optim")


def test_pipeline_parallel_publishers_split_the_mapping() -> None:
    """Each stage describes its own layers; a reader merges their objects."""
    coordinator = _coordinator(["trainer/0", "trainer/1", "generator/0/0"])
    results = _register(
        coordinator,
        {
            "trainer/0": (RankRole.PUBLISHER, {"first": torch.empty(2)}),
            "trainer/1": (RankRole.PUBLISHER, {"second": torch.empty(2)}),
            "generator/0/0": (
                RankRole.REQUESTER,
                {"first": torch.empty(2), "second": torch.empty(2)},
            ),
        },
    )

    # Both stages' volumes are read, so both mapping objects are merged.
    plan = results["generator/0/0"][0]
    sources = {
        transfer.source_volume_id
        for entry in plan._local("generator/0/0").keys.values()
        for route in entry.routes
        for transfer in route.transfers
    }
    assert sources == {"trainer/0", "trainer/1"}
