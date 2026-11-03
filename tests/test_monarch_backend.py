# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The Monarch backend makes the Monarch calls that core modules once made themselves."""

import asyncio
import logging
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch
import torchstore.backend as backend_module
import torchstore.monarch.backend as monarch_backend
import torchstore.monarch.storage_volume as storage_volume_module
from torchstore.monarch.controller import Controller, ControllerDirectory
from torchstore.monarch.coordinator import MonarchControlPlane, RoutingCoordinator
from torchstore.spmd import SPMDEnv
from torchstore.strategy import HostStrategy


def _id_func() -> str:
    return "volume"


def test_get_backend_loads_the_monarch_backend_once(monkeypatch) -> None:
    """Return one ``MonarchBackend`` for "monarch"."""
    monkeypatch.setattr(backend_module, "_backends", {})
    backend = backend_module.get_backend("monarch")
    assert isinstance(backend, monarch_backend.MonarchBackend)
    assert backend_module.get_backend("monarch") is backend


def test_get_backend_refuses_an_unknown_name(monkeypatch) -> None:
    monkeypatch.setattr(backend_module, "_backends", {})
    with pytest.raises(ValueError, match="unknown backend 'ray'"):
        backend_module.get_backend("ray")
    assert backend_module._backends == {}


@pytest.fixture
def lookup(monkeypatch) -> AsyncMock:
    """Monarch's ``get_or_spawn_controller`` as the backend sees it, echoing its arguments."""
    lookup = AsyncMock(side_effect=lambda name, cls, **kwargs: (name, cls, kwargs))
    monkeypatch.setattr(monarch_backend, "get_or_spawn_controller", lookup)
    return lookup


def test_monarch_backend_reads_monarchs_rank(monkeypatch) -> None:
    """Give the rank of Monarch's ``current_rank()``."""
    monkeypatch.setattr(
        monarch_backend, "current_rank", lambda: SimpleNamespace(rank=5)
    )
    assert monarch_backend.MonarchBackend().rank() == 5


def test_monarch_backend_spawns_volumes_on_the_mesh(monkeypatch) -> None:
    """Hand ``num_volumes``, the mesh and ``id_func`` to ``StorageVolume.spawn``."""
    spawn = AsyncMock(return_value="volumes")
    monkeypatch.setattr(monarch_backend, "StorageVolume", SimpleNamespace(spawn=spawn))
    backend = monarch_backend.MonarchBackend()
    assert asyncio.run(backend.spawn_volumes("mesh", 3, _id_func)) == "volumes"
    spawn.assert_awaited_once_with(3, "mesh", id_func=_id_func)


def test_monarch_backend_finds_the_controller_volume_by_name(lookup) -> None:
    """Find or spawn the one controller volume under its fixed name."""
    volume = asyncio.run(monarch_backend.MonarchBackend().controller_volume(_id_func))
    assert volume == (
        "storage_volume_controller",
        storage_volume_module.StorageVolume,
        {"id_func": _id_func},
    )


def test_monarch_backend_finds_controllers_and_coordinators_by_store(lookup) -> None:
    """Find a store's controller and coordinator under the store's name."""
    backend = monarch_backend.MonarchBackend()
    assert asyncio.run(backend.controller("store")) == ("store", Controller, {})
    coordinator = asyncio.run(backend.coordinator("store"))
    assert isinstance(coordinator, MonarchControlPlane)
    assert coordinator._coordinator == ("store", RoutingCoordinator, {})
    assert coordinator._store_name == "store"


def test_monarch_backend_locates_keys_through_the_controller() -> None:
    """Wrap the controller in the directory that asks it where keys live."""
    directory = monarch_backend.MonarchBackend().directory("controller")
    assert isinstance(directory, ControllerDirectory)
    assert directory._controller == "controller"


def test_monarch_backend_forwards_spawns(monkeypatch) -> None:
    """Pass every argument of ``spawn_actors`` on."""
    spawn_actors = AsyncMock(return_value="actors")
    monkeypatch.setattr(monarch_backend, "spawn_actors", spawn_actors)
    backend = monarch_backend.MonarchBackend()
    actors = asyncio.run(backend.spawn_actors(2, object, "name", "mesh", flag=True))
    assert actors == "actors"
    spawn_actors.assert_awaited_once_with(2, object, "name", "mesh", flag=True)


def test_monarch_backend_forwards_rdma_buffers(monkeypatch) -> None:
    """Build each RDMA buffer from the byte view it is handed."""
    rdma_buffer = MagicMock(return_value="buffer")
    monkeypatch.setattr(monarch_backend, "RDMABuffer", rdma_buffer)
    view = torch.zeros(4, dtype=torch.uint8)
    assert monarch_backend.MonarchBackend().rdma_buffer(view) == "buffer"
    rdma_buffer.assert_called_once_with(view)


def test_monarch_backend_forwards_spmd_initialization(monkeypatch) -> None:
    """Pass every argument of SPMD initialization on."""
    initialize_spmd = AsyncMock(return_value="session")
    monkeypatch.setattr(monarch_backend, "initialize_spmd", initialize_spmd)
    backend = monarch_backend.MonarchBackend()
    strategy = HostStrategy()
    env = SPMDEnv(
        rank=0,
        local_rank=0,
        world_size=1,
        local_world_size=1,
        master_addr="127.0.0.1",
        master_port=0,
    )
    timeout = timedelta(seconds=7)

    session = asyncio.run(
        backend.initialize_spmd(
            strategy,
            "store",
            env=env,
            rendezvous_timeout=timeout,
            transport="tcp",
            monarch_port=1234,
            configure_job=print,
        )
    )
    assert session == "session"
    initialize_spmd.assert_awaited_once_with(
        strategy,
        "store",
        env=env,
        rendezvous_timeout=timeout,
        transport="tcp",
        monarch_port=1234,
        configure_job=print,
    )


class _Initialized:
    """A mesh's ``initialized``, counting how often it was awaited."""

    def __init__(self) -> None:
        self.awaits = 0

    def __await__(self):
        self.awaits += 1
        return iter(())


def _counting_mesh() -> SimpleNamespace:
    """A mesh whose every spawn returns how often the mesh was awaited by then."""
    initialized = _Initialized()
    return SimpleNamespace(
        initialized=initialized,
        spawn=MagicMock(side_effect=lambda *args, **kwargs: initialized.awaits),
    )


def test_spawn_actors_waits_for_the_mesh_and_names_each_spawn_anew() -> None:
    """Await the mesh before each spawn on it, and suffix every spawn's name with its own id."""
    first, second = _counting_mesh(), _counting_mesh()

    async def run() -> list[int]:
        return [
            await storage_volume_module.spawn_actors(
                1, object, "probe", mesh, flag=True
            )
            for mesh in (first, first, second)
        ]

    assert asyncio.run(run()) == [1, 2, 1]
    spawns = first.spawn.call_args_list + second.spawn.call_args_list
    names = [spawn.args[0] for spawn in spawns]
    assert len(set(names)) == 3
    assert all(name.startswith("probe_") and len(name) == 14 for name in names)
    assert all(
        spawn.args[1:] == (object,) and spawn.kwargs == {"flag": True}
        for spawn in spawns
    )


def test_spawn_actors_refuses_a_mesh_that_cannot_spawn() -> None:
    """Refuse a mesh without ``spawn`` before waiting for it."""
    mesh = SimpleNamespace(initialized=_Initialized())
    with pytest.raises(AssertionError):
        asyncio.run(storage_volume_module.spawn_actors(1, object, "probe", mesh))
    assert mesh.initialized.awaits == 0


def test_spawn_actors_without_a_mesh_uses_this_host_and_logs_as_before(
    monkeypatch, caplog
) -> None:
    """Spawn on this host's new processes, logging under ``torchstore.utils`` as before the move."""
    mesh = SimpleNamespace(initialized=_Initialized(), spawn=MagicMock())
    host = SimpleNamespace(spawn_procs=MagicMock(return_value=mesh))
    monkeypatch.setattr(storage_volume_module, "this_host", lambda: host)
    caplog.set_level(logging.DEBUG, logger="torchstore.utils")

    asyncio.run(storage_volume_module.spawn_actors(2, object, "probe"))

    host.spawn_procs.assert_called_once_with(per_host={"gpus": 2})
    assert mesh.spawn.call_count == 1
    assert [
        (record.name, record.getMessage())
        for record in caplog.records
        if "Spawning actors" in record.getMessage()
    ] == [("torchstore.utils", "Spawning actors on the local host")]
