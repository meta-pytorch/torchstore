# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for resolving storage volume IDs through a volume host."""

import importlib.util
import json
import os
import pickle
import subprocess
import sys
import uuid
from collections.abc import Collection
from functools import partial
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
import torch
from torchstore.strategy import (
    ControllerStorageVolumes,
    HostStrategy,
    LocalRankStrategy,
    StorageVolumeRef,
    TorchStoreStrategy,
)
from torchstore.transport import create_transport_buffer, TransportType
from torchstore.transport.types import Request
from torchstore.volume_host import MonarchVolumeHost, VolumeHost

from .volume_meshes import VolumeMesh as _FakeMesh

pytestmark = pytest.mark.timeout(120)

# Child processes import the tree that pytest imported.
_ROOT = str(Path(importlib.util.find_spec("torchstore").origin).parents[1])

# Puts once through each volume of a 2-process mesh, stops the mesh, then times the next
# put through each and writes [outcome, seconds] per volume.
_STOPPED_VOLUMES_CHILD = """
import asyncio, json, os, sys, time, uuid
from functools import partial

root, out = sys.argv[1:]
sys.path.insert(0, root)
import torch
from monarch.actor import this_host
from monarch.config import configure
from torchstore.storage_volume import StorageVolume
from torchstore.strategy import TorchStoreStrategy
from torchstore.transport import create_transport_buffer, TransportType
from torchstore.transport.types import Request


def volume_id(prefix):
    return f"{prefix}/{os.getpid()}"


async def put(ref, key):
    buffer = create_transport_buffer(ref)
    await buffer.put_to_storage_volume([Request.from_tensor(key, torch.ones(4))])


async def main():
    mesh = this_host().spawn_procs(per_host={"procs": 2})
    stopped = False
    try:
        volumes = await StorageVolume.spawn(
            2, mesh, id_func=partial(volume_id, f"volume-{uuid.uuid4()}")
        )
        strategy = TorchStoreStrategy(TransportType.MonarchRPC)
        await strategy.set_storage_volumes(volumes)
        volume_ids = sorted(strategy.volume_host.volume_ids())
        # Shortened only after setup: on a loaded host a 5 s watchdog can fire during it.
        configure(supervision_watchdog_timeout="5s")
        for volume in volume_ids:
            await put(strategy.get_storage_volume(volume), "before")
        await mesh.stop()
        stopped = True
        calls = []
        for volume in volume_ids:
            start = time.monotonic()
            try:
                await asyncio.wait_for(put(strategy.get_storage_volume(volume), "after"), 60)
                outcome = "returned"
            except asyncio.TimeoutError:
                outcome = "timed out"
            except RuntimeError as error:
                outcome = f"raised {type(error).__name__}"
            calls.append([outcome, time.monotonic() - start])
        return calls
    finally:
        if not stopped:
            await mesh.stop()


calls = asyncio.run(main())
with open(out, "w") as f:
    json.dump(calls, f)
"""


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


class _RecordStrategy(LocalRankStrategy):
    """Resolves volumes through a record host by overriding ``volume_host``."""

    _records: _RecordHost | None = None

    async def set_storage_volumes(self, storage_volumes: _RecordHost) -> None:
        self._records = storage_volumes

    @property
    def volume_host(self) -> VolumeHost | None:
        return self._records


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


async def _put(ref: StorageVolumeRef, key: str, value: torch.Tensor) -> None:
    buffer = create_transport_buffer(ref)
    await buffer.put_to_storage_volume([Request.from_tensor(key, value)])


async def _get(ref: StorageVolumeRef, key: str) -> torch.Tensor:
    buffer = create_transport_buffer(ref)
    [value] = await buffer.get_from_storage_volume([Request(key=key)])
    return value


def test_volume_host_declares_three_lookups() -> None:
    """Make a host list its volume IDs and look up each one's endpoints and host name."""
    assert VolumeHost.__abstractmethods__ == {"volume_ids", "endpoints", "hostname"}


@pytest.mark.asyncio
async def test_strategy_resolves_each_volume_to_its_mesh_slice() -> None:
    """Resolve each ID to its slice and verbatim host name, also from a pickled copy.

    The strategy sets no ``storage_volumes``, ``volume_id_to_coord`` or ``volume_id_to_hostname``.
    """
    strategy = TorchStoreStrategy(TransportType.Gloo)
    names = [("a", "Host-a.Region.example.com"), ("b", "Host-b.Region.example.com")]
    await strategy.set_storage_volumes(_FakeMesh(names))
    copies = {"storage_volumes", "volume_id_to_coord", "volume_id_to_hostname"}
    assert not copies & set(vars(strategy))

    assert isinstance(strategy.volume_host, MonarchVolumeHost)
    assert list(strategy.volume_host.volume_ids()) == ["a", "b"]
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
async def test_monarch_host_slices_each_volume_once_and_pickles_no_slice() -> None:
    """Reuse one slice per volume, and pickle the strategy without slices before and after lookups."""
    strategy = HostStrategy()
    mesh = _UnpicklableSliceMesh([("a", "host-a"), ("b", "host-b")])
    await strategy.set_storage_volumes(mesh)
    pickle.dumps(strategy)
    host = strategy.volume_host
    assert host is not None
    handles = {}
    for rank, volume_id in enumerate(["a", "b"]):
        handles[volume_id] = host.endpoints(volume_id)
        assert handles[volume_id] is host.endpoints(volume_id)
        assert handles[volume_id].coord == {"procs": rank}
    # Interleaved lookups through the strategy reuse each volume's slice.
    for volume_id in ("a", "b", "a"):
        assert strategy.get_storage_volume(volume_id).volume is handles[volume_id]
    assert mesh.sliced == [{"procs": 0}, {"procs": 1}]
    copy = pickle.loads(pickle.dumps(strategy))
    assert copy.get_storage_volume("b").volume.coord == {"procs": 1}
    # Pickling leaves the source's slices in place.
    assert all(host.endpoints(v) is handles[v] for v in ("a", "b"))
    assert mesh.sliced == [{"procs": 0}, {"procs": 1}]


def test_monarch_host_copies_the_mappings_it_is_given() -> None:
    """Copy the given mappings, so later edits by the caller miss the host and read-only ones pickle.

    Both must name the same volumes.
    """
    coords, hostnames = {"a": {"procs": 0}}, {"a": "host-a"}
    hosts = [
        MonarchVolumeHost(_FakeMesh([]), coords, hostnames),
        MonarchVolumeHost(
            _FakeMesh([]), MappingProxyType(coords), MappingProxyType(hostnames)
        ),
        MonarchVolumeHost(
            _FakeMesh([]), {"a": MappingProxyType({"procs": 0})}, hostnames.copy()
        ),
    ]
    coords["a"]["procs"] = 1
    coords["b"] = {"procs": 1}
    hostnames["a"] = "host-b"
    for host in hosts:
        for resolver in (host, pickle.loads(pickle.dumps(host))):
            assert list(resolver.volume_ids()) == ["a"]
            assert resolver.endpoints("a") == ("slice", {"procs": 0})
            assert resolver.hostname("a") == "host-a"
    with pytest.raises(ValueError, match="name different volumes"):
        MonarchVolumeHost(_FakeMesh([]), {"a": {"procs": 0}}, {"b": "host-b"})


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
    strategy = _RecordStrategy(TransportType.MonarchRPC)
    await strategy.set_storage_volumes(
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
    assert str(strategy) == "_RecordStrategy(storage_volume_len=2)"


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
        assert all(host.endpoints(v) is host.endpoints(v) for v in volume_ids)
        replies = [await host.endpoints(v).get_id.call_one() for v in volume_ids]
        # Put through the strategy and get through its copy, a different tensor per volume.
        copy = cloudpickle.loads(cloudpickle.dumps(strategy))
        for index, volume_id in enumerate(volume_ids):
            value = torch.full((4,), float(index))
            await _put(strategy.get_storage_volume(volume_id), "weights", value)
        copied_refs = [copy.get_storage_volume(v) for v in volume_ids]
        fetched = [await _get(ref, "weights") for ref in copied_refs]
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


@pytest.mark.timeout(240)
def test_a_used_volume_fails_once_its_procs_stop_within_the_watchdog(
    tmp_path: Path,
) -> None:
    """Fail each used volume's next call once its procs stop, within Monarch's watchdog.

    A host copy reuses each slice, and Monarch may report the stop to a slice called
    before it only when its supervision watchdog fires, which the child shortens to 5 s.
    """
    pytest.importorskip("monarch")
    out = tmp_path / "calls.json"
    # run() kills the child on its timeout and on any other error, pytest's own timeout too.
    child = subprocess.run(
        [sys.executable, "-c", _STOPPED_VOLUMES_CHILD, _ROOT, str(out)],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    calls = json.loads(out.read_text())
    assert len(calls) == 2, calls
    for outcome, seconds in calls:
        assert outcome.startswith("raised ") and seconds < 30, calls
