# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for resolving storage volume IDs through a volume host."""

import os
import pickle
import uuid
from collections.abc import Collection
from functools import partial
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
import torch
from torchstore.strategy import (
    ControllerStorageVolumes,
    HostStrategy,
    LocalRankStrategy,
    TorchStoreStrategy,
)
from torchstore.transport import TransportType
from torchstore.volume_host import MonarchVolumeHost, VolumeHost

from .volume_meshes import get_through, put_through, VolumeMesh as _FakeMesh

pytestmark = pytest.mark.timeout(120)


class _SingleVolume:
    """The controller's one-volume handle, which answers ``get_id`` itself."""

    def __init__(self) -> None:
        self.get_id = SimpleNamespace(call_one=AsyncMock(return_value=("0", "host-0")))

    def __len__(self) -> int:
        return 1


class _RecordHost(VolumeHost):
    """A host over published volume records, with no mesh behind it."""

    def __init__(self, records: dict[str, tuple[Any, str]]) -> None:
        self._records = records

    def volume_ids(self) -> Collection[str]:
        return self._records.keys()

    def endpoints(self, volume_id: str) -> Any:
        return self._records[volume_id][0]

    def hostname(self, volume_id: str) -> str:
        return self._records[volume_id][1]


class _UnpicklableSlice:
    """A slice that refuses to pickle, since each process slices the mesh itself."""

    def __init__(self, coord: dict[str, int]) -> None:
        self.coord = coord

    def __reduce__(self) -> Any:
        raise pickle.PicklingError("a volume slice was pickled")


class _UnpicklableSliceMesh:
    """A stand-in volume mesh whose slices must never travel inside a pickled strategy."""

    def __init__(self, volumes: list[tuple[str, str]]) -> None:
        self.get_id = _FakeMesh(volumes).get_id
        self.sliced: list[dict[str, int]] = []

    def slice(self, **coord: int) -> _UnpicklableSlice:
        self.sliced.append(coord)
        return _UnpicklableSlice(coord)


def _name_volume_and_host(prefix: str) -> str:
    """Name a volume, and the host name it reports, after the process that builds it."""
    os.environ["TORCHSTORE_HOSTNAME"] = f"Host-{os.getpid()}.Region.example.com"
    return f"{prefix}/{os.getpid()}"


@pytest.mark.asyncio
async def test_strategy_resolves_each_volume_to_its_mesh_slice() -> None:
    """Resolve each ID to its slice and verbatim host name, also from a pickled copy."""
    strategy = TorchStoreStrategy(TransportType.Gloo)
    names = [("a", "Host-a.Region.example.com"), ("b", "Host-b.Region.example.com")]
    await strategy.set_storage_volumes(_FakeMesh(names))

    assert isinstance(strategy.volume_host, MonarchVolumeHost)
    assert list(strategy.volume_host.volume_ids()) == ["a", "b"]
    # Upstream's attributes stay filled for code that reads them.
    assert strategy.volume_id_to_coord == {"a": {"procs": 0}, "b": {"procs": 1}}
    assert strategy.volume_id_to_hostname == dict(names)
    for resolver in (strategy, pickle.loads(pickle.dumps(strategy))):
        for rank, volume_id in enumerate(["a", "b"]):
            ref = resolver.get_storage_volume(volume_id)
            assert ref.volume == ("slice", {"procs": rank})
            assert (ref.volume_id, ref.volume_hostname) == (
                volume_id,
                f"Host-{volume_id}.Region.example.com",
            )
            assert ref.transport_context is resolver.transport_context
            assert ref.default_transport_type is TransportType.Gloo


@pytest.mark.asyncio
async def test_strategy_slices_every_mesh_dimension() -> None:
    """Slice a 2-D (hosts x gpus) mesh at every coordinate of each volume."""
    coords = [{"hosts": h, "gpus": g} for h in range(2) for g in range(2)]
    volumes = [(f"v{i}", f"host-{i}") for i in range(len(coords))]
    strategy = TorchStoreStrategy()
    await strategy.set_storage_volumes(_FakeMesh(volumes, coords))
    for (volume_id, _), coord in zip(volumes, coords):
        assert strategy.get_storage_volume(volume_id).volume == ("slice", coord)


@pytest.mark.asyncio
async def test_setting_volumes_again_replaces_the_host() -> None:
    """Resolve only the volumes of the latest ``set_storage_volumes`` call."""
    strategy = HostStrategy()
    await strategy.set_storage_volumes(_FakeMesh([("a", "host-a")]))
    await strategy.set_storage_volumes(_FakeMesh([("b", "host-b"), ("c", "host-c")]))
    host = strategy.volume_host
    assert host is not None and sorted(host.volume_ids()) == ["b", "c"]
    assert strategy.get_storage_volume("c").volume_hostname == "host-c"
    with pytest.raises(KeyError) as raised:
        strategy.get_storage_volume("a")
    assert raised.value.args == ("a",)


@pytest.mark.asyncio
async def test_monarch_host_pickles_no_slice() -> None:
    """Slice the mesh on each lookup, and pickle no slice before lookups or after."""
    strategy = HostStrategy()
    mesh = _UnpicklableSliceMesh([("a", "host-a"), ("b", "host-b")])
    await strategy.set_storage_volumes(mesh)
    pickle.dumps(strategy)
    for volume_id in ("a", "b", "a"):
        strategy.get_storage_volume(volume_id)
    assert mesh.sliced == [{"procs": 0}, {"procs": 1}, {"procs": 0}]
    copy = pickle.loads(pickle.dumps(strategy))
    assert copy.get_storage_volume("b").volume.coord == {"procs": 1}


@pytest.mark.asyncio
async def test_unknown_volume_ids_raise_key_error() -> None:
    """Raise ``KeyError`` with the unknown ID, before and after volumes are set."""
    strategy = TorchStoreStrategy()
    with pytest.raises(KeyError) as before:
        strategy.get_storage_volume("c")
    assert before.value.args == ("c",)

    await strategy.set_storage_volumes(_FakeMesh([("a", "host-a")]))
    host = strategy.volume_host
    assert host is not None
    for lookup in (strategy.get_storage_volume, host.endpoints, host.hostname):
        with pytest.raises(KeyError) as raised:
            lookup("c")
        assert raised.value.args == ("c",)


@pytest.mark.asyncio
async def test_select_storage_volume_resolves_the_clients_volume(monkeypatch) -> None:
    """Select the volume named by the client's rank, or raise ``KeyError`` listing the IDs."""
    monkeypatch.setenv("RANK", "1")
    strategy = LocalRankStrategy()
    assert strategy.volume_host is None
    with pytest.raises(KeyError, match=r"found for 1 volume_ids=\[\]"):
        strategy.select_storage_volume()

    await strategy.set_storage_volumes(_FakeMesh([("0", "host-0"), ("1", "host-1")]))
    ref = strategy.select_storage_volume()
    assert (ref.volume, ref.volume_id) == (("slice", {"procs": 1}), "1")
    assert ref.volume_hostname == "host-1"
    monkeypatch.setenv("RANK", "2")
    with pytest.raises(KeyError, match=r"found for 2 volume_ids=\['0', '1'\]"):
        strategy.select_storage_volume()


@pytest.mark.asyncio
async def test_str_counts_the_volumes_the_host_resolves() -> None:
    """Count no volumes before they are set, then each volume ID."""
    strategy = HostStrategy()
    assert str(strategy) == "HostStrategy(storage_volume_len=0)"
    await strategy.set_storage_volumes(_FakeMesh([("a", "host-a"), ("b", "host-b")]))
    assert str(strategy) == "HostStrategy(storage_volume_len=2)"


@pytest.mark.asyncio
async def test_a_repeated_volume_id_resolves_to_its_last_volume() -> None:
    """Resolve an ID that two volumes report to the later one, which ``str`` counts once."""
    strategy = HostStrategy()
    await strategy.set_storage_volumes(_FakeMesh([("a", "first"), ("a", "second")]))

    ref = strategy.get_storage_volume("a")
    assert (ref.volume, ref.volume_hostname) == (("slice", {"procs": 1}), "second")
    assert str(strategy) == "HostStrategy(storage_volume_len=1)"


@pytest.mark.asyncio
async def test_strategy_resolves_through_any_volume_host(monkeypatch) -> None:
    """Resolve, select and count volumes through a host that is not a Monarch mesh."""
    monkeypatch.setenv("RANK", "b")
    endpoints_a, endpoints_b = object(), object()
    strategy = LocalRankStrategy(TransportType.MonarchRPC)
    strategy.set_volume_host(
        _RecordHost({"a": (endpoints_a, "host-a"), "b": (endpoints_b, "host-b")})
    )

    ref = strategy.get_storage_volume("a")
    assert (ref.volume, ref.volume_id, ref.volume_hostname) == (
        endpoints_a,
        "a",
        "host-a",
    )
    assert ref.default_transport_type is TransportType.MonarchRPC
    assert strategy.select_storage_volume().volume is endpoints_b
    assert str(strategy) == "LocalRankStrategy(storage_volume_len=2)"
    # Setting volumes, as ts.initialize does, replaces the installed host.
    await strategy.set_storage_volumes(_FakeMesh([("b", "host-b")]))
    assert isinstance(strategy.volume_host, MonarchVolumeHost)
    assert strategy.select_storage_volume().volume == ("slice", {"procs": 0})


@pytest.mark.asyncio
async def test_lookups_read_volumes_a_subclass_sets_itself(monkeypatch) -> None:
    """Resolve volumes that a subclass puts on upstream's attributes without a host."""

    class SelfSetStrategy(LocalRankStrategy):
        async def set_storage_volumes(self, storage_volumes) -> None:
            self.storage_volumes = storage_volumes
            self.volume_id_to_coord = {"1": {"procs": 1}}
            self.volume_id_to_hostname = {"1": "host-1"}

    monkeypatch.setenv("RANK", "1")
    strategy = SelfSetStrategy()
    await strategy.set_storage_volumes(_FakeMesh([("0", "host-0"), ("1", "host-1")]))
    ref = strategy.select_storage_volume()
    assert (ref.volume, ref.volume_id, ref.volume_hostname) == (
        ("slice", {"procs": 1}),
        "1",
        "host-1",
    )
    with pytest.raises(KeyError) as raised:
        strategy.get_storage_volume("0")
    assert raised.value.args == ("0",)


def test_controller_storage_volumes_refuses_a_volume_host() -> None:
    """Refuse a volume host, which the controller volume's lookups would never consult."""
    strategy = ControllerStorageVolumes()
    with pytest.raises(
        TypeError, match="^ControllerStorageVolumes reaches its one controller volume"
    ):
        strategy.set_volume_host(_RecordHost({}))
    assert strategy.volume_host is None


@pytest.mark.asyncio
async def test_controller_storage_volumes_keeps_its_single_volume() -> None:
    """Resolve to the whole controller volume with the given transport, and build no host."""
    assert ControllerStorageVolumes().default_transport_type is TransportType.Unset
    strategy = ControllerStorageVolumes(TransportType.Gloo)
    assert str(strategy) == "ControllerStorageVolumes(storage_volume_len=0)"
    with pytest.raises(KeyError, match="No corresponding storage volume found for 0"):
        strategy.select_storage_volume()

    volume = _SingleVolume()
    await strategy.set_storage_volumes(volume)

    ref = strategy.select_storage_volume()
    assert ref.volume is volume
    assert (ref.volume_id, ref.volume_hostname) == ("0", "host-0")
    assert ref.default_transport_type is TransportType.Gloo
    assert str(strategy) == "ControllerStorageVolumes(storage_volume_len=1)"
    assert strategy.volume_host is None


@pytest.mark.asyncio
async def test_monarch_host_resolves_each_volume_of_a_real_mesh() -> None:
    """Resolve each volume of a 2-process mesh to its own slice, also from a pickled copy."""
    actor = pytest.importorskip("monarch.actor")
    cloudpickle = pytest.importorskip("cloudpickle")
    from torchstore.storage_volume import StorageVolume

    prefix = f"volume-{uuid.uuid4()}"
    mesh = actor.this_host().spawn_procs(per_host={"procs": 2})
    try:
        id_func = partial(_name_volume_and_host, prefix)
        volumes = await StorageVolume.spawn(2, mesh, id_func=id_func)
        strategy = TorchStoreStrategy(TransportType.MonarchRPC)
        await strategy.set_storage_volumes(volumes)
        host = strategy.volume_host
        assert isinstance(host, MonarchVolumeHost)
        volume_ids = sorted(host.volume_ids())
        replies = [await host.endpoints(v).get_id.call_one() for v in volume_ids]
        # Put through the strategy and get through its copy, a different tensor per volume.
        copy = cloudpickle.loads(cloudpickle.dumps(strategy))
        for index, volume_id in enumerate(volume_ids):
            value = torch.full((4,), float(index))
            await put_through(strategy.get_storage_volume(volume_id), "weights", value)
        copied_refs = [copy.get_storage_volume(v) for v in volume_ids]
        fetched = [await get_through(ref, "weights") for ref in copied_refs]
        with pytest.raises(KeyError) as missing:
            strategy.get_storage_volume(f"{prefix}/missing")
        assert missing.value.args == (f"{prefix}/missing",)
        text, size = str(strategy), len(volumes)
    finally:
        await mesh.stop()

    pids = [volume_id.removeprefix(f"{prefix}/") for volume_id in volume_ids]
    assert len(set(pids)) == 2 and str(os.getpid()) not in pids
    hostnames = [f"Host-{pid}.Region.example.com" for pid in pids]
    assert [host.hostname(v) for v in volume_ids] == hostnames
    assert [ref.volume_hostname for ref in copied_refs] == hostnames
    assert replies == list(zip(volume_ids, hostnames))
    for index, value in enumerate(fetched):
        assert torch.equal(value, torch.full((4,), float(index)))
    assert (text, size) == ("TorchStoreStrategy(storage_volume_len=2)", 2)
