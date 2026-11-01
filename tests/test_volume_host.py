# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for resolving storage volume IDs through a volume host a strategy installs."""

import pickle
import typing
from collections.abc import Collection
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from torchstore.strategy import (
    ControllerStorageVolumes,
    HostStrategy,
    LocalRankStrategy,
    StorageVolumeRef,
    TorchStoreStrategy,
)
from torchstore.transport import TransportType
from torchstore.volume_host import VolumeHost

from .volume_meshes import VolumeMesh

pytestmark = pytest.mark.timeout(60)


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


@pytest.mark.asyncio
async def test_an_installed_host_resolves_ids_in_place_of_the_attributes() -> None:
    """Resolve each ID through the installed host only, also one the attributes hold too."""
    strategy = TorchStoreStrategy(TransportType.Gloo)
    await strategy.set_storage_volumes(VolumeMesh([("a", "host-a"), ("b", "mesh-b")]))
    strategy.set_volume_host(_RecordHost({"b": ("handle-b", "host-b")}))

    ref = strategy.get_storage_volume("b")
    assert (ref.volume, ref.volume_id, ref.volume_hostname) == (
        "handle-b",
        "b",
        "host-b",
    )
    assert ref.transport_context is strategy.transport_context
    assert ref.default_transport_type is TransportType.Gloo
    with pytest.raises(KeyError):
        strategy.get_storage_volume("a")


def test_a_second_host_replaces_the_first() -> None:
    """Resolve through the host installed last."""
    strategy = TorchStoreStrategy()
    strategy.set_volume_host(_RecordHost({"a": ("first", "first-host")}))
    strategy.set_volume_host(_RecordHost({"a": ("second", "second-host")}))

    ref = strategy.get_storage_volume("a")
    assert (ref.volume, ref.volume_hostname) == ("second", "second-host")


def test_an_installed_host_travels_with_a_pickled_strategy() -> None:
    """Resolve through the host of a strategy's pickled copy, as the strategy's readers do."""
    strategy = TorchStoreStrategy()
    strategy.set_volume_host(_RecordHost({"a": ("handle-a", "host-a")}))

    ref = pickle.loads(pickle.dumps(strategy)).get_storage_volume("a")
    assert (ref.volume, ref.volume_hostname) == ("handle-a", "host-a")


def test_a_host_must_implement_every_lookup() -> None:
    """Refuse to build a host that leaves a lookup abstract."""

    class NoEndpoints(VolumeHost):
        def volume_ids(self) -> Collection[str]:
            return ()

        def hostname(self, volume_id: str) -> str:
            return "host"

    with pytest.raises(TypeError, match="abstract"):
        NoEndpoints()


@pytest.mark.asyncio
async def test_selection_checks_the_installed_hosts_ids(monkeypatch) -> None:
    """Select the client's volume among the host's IDs, refusing one only the attributes know."""
    strategy = LocalRankStrategy()
    await strategy.set_storage_volumes(VolumeMesh([("0", "host-0")]))
    strategy.set_volume_host(_RecordHost({"1": ("handle-1", "host-1")}))

    monkeypatch.setenv("RANK", "1")
    assert strategy.select_storage_volume().volume == "handle-1"
    monkeypatch.setenv("RANK", "0")
    with pytest.raises(KeyError, match="found for 0 in the volume host"):
        strategy.select_storage_volume()


@pytest.mark.asyncio
async def test_selection_without_a_host_names_a_missing_volume(monkeypatch) -> None:
    """Refuse a client ID the attributes lack with main's message, with no host installed."""
    strategy = LocalRankStrategy()
    await strategy.set_storage_volumes(VolumeMesh([("0", "host-0")]))

    monkeypatch.setenv("RANK", "1")
    with pytest.raises(KeyError, match="found for 1 self.volume_id_to_coord="):
        strategy.select_storage_volume()


@pytest.mark.asyncio
async def test_setting_volumes_drops_an_installed_host() -> None:
    """Resolve the volumes ``set_storage_volumes`` sets, not a host installed before."""
    strategy = TorchStoreStrategy()
    strategy.set_volume_host(_RecordHost({"a": ("handle-a", "host-a")}))
    await strategy.set_storage_volumes(VolumeMesh([("a", "mesh-host-a")]))

    ref = strategy.get_storage_volume("a")
    assert (ref.volume, ref.volume_hostname) == (("slice", {"procs": 0}), "mesh-host-a")


@pytest.mark.asyncio
async def test_a_failed_set_drops_the_host_and_resolves_nothing() -> None:
    """Leave the new mesh with no volumes, not even the last mesh's, and no host, as main did."""
    strategy = TorchStoreStrategy()
    await strategy.set_storage_volumes(VolumeMesh([("a", "host-a")]))
    strategy.set_volume_host(_RecordHost({"a": ("handle-a", "host-a")}))
    mesh = Mock(get_id=Mock(call=AsyncMock(side_effect=RuntimeError("a volume died"))))
    with pytest.raises(RuntimeError, match="a volume died"):
        await strategy.set_storage_volumes(mesh)

    assert strategy.storage_volumes is mesh
    with pytest.raises(KeyError, match="^'a'$"):
        strategy.get_storage_volume("a")


def test_a_subclass_that_skips_the_base_constructor_resolves_its_attributes() -> None:
    """Resolve through the attributes a subclass fills itself, with no host installed."""

    class OwnTables(TorchStoreStrategy):
        def __init__(self) -> None:
            self.storage_volumes = VolumeMesh([])
            self.volume_id_to_coord = {"a": {"procs": 3}}
            self.volume_id_to_hostname = {"a": "host-a"}
            self.transport_context = None
            self.default_transport_type = TransportType.Unset

    ref = OwnTables().get_storage_volume("a")
    assert (ref.volume, ref.volume_hostname) == (("slice", {"procs": 3}), "host-a")


# Each overrides one method that reads or drops a host, even when it defers to the base.
class _OwnSet(TorchStoreStrategy):
    async def set_storage_volumes(self, storage_volumes):
        await super().set_storage_volumes(storage_volumes)


class _OwnSelect(TorchStoreStrategy):
    def select_storage_volume(self) -> StorageVolumeRef:
        return super().select_storage_volume()


class _OwnLookup(TorchStoreStrategy):
    def get_storage_volume(self, volume_id: str) -> StorageVolumeRef:
        return super().get_storage_volume(volume_id)


@pytest.mark.parametrize(
    "strategy_cls, overridden",
    [
        (_OwnSet, "set_storage_volumes"),
        (_OwnSelect, "select_storage_volume"),
        (_OwnLookup, "get_storage_volume"),
        (
            ControllerStorageVolumes,
            "set_storage_volumes, select_storage_volume, get_storage_volume",
        ),
    ],
)
def test_a_strategy_that_overrides_a_host_method_refuses_a_host(
    strategy_cls: type[TorchStoreStrategy], overridden: str
) -> None:
    """Refuse a host where the class overrides a method that reads or drops it, and install none."""
    strategy = strategy_cls()
    with pytest.raises(TypeError, match=f"overrides {overridden}, which would ignore"):
        strategy.set_volume_host(_RecordHost({}))
    assert vars(strategy).keys() == vars(strategy_cls()).keys()


def test_type_hints_of_every_strategy_class_resolve() -> None:
    """Resolve each strategy class's hints at run time, as on main."""
    for strategy_cls in (
        TorchStoreStrategy,
        LocalRankStrategy,
        HostStrategy,
        ControllerStorageVolumes,
    ):
        assert typing.get_type_hints(strategy_cls) == {}


@pytest.mark.asyncio
async def test_setting_volumes_adds_no_host_attribute() -> None:
    """Leave a strategy's instance dict with main's attributes after it sets its volumes."""
    strategy = LocalRankStrategy()
    await strategy.set_storage_volumes(VolumeMesh([("0", "host-0")]))
    assert sorted(vars(strategy)) == [
        "default_transport_type",
        "storage_volumes",
        "transport_context",
        "volume_id_to_coord",
        "volume_id_to_hostname",
    ]


@pytest.mark.asyncio
async def test_a_subclass_attribute_named_volume_host_is_its_own() -> None:
    """Keep a subclass's own ``_volume_host`` through set and lookups, as on main."""

    class OwnName(TorchStoreStrategy):
        def __init__(self) -> None:
            super().__init__()
            self._volume_host = "rack-7"

    strategy = OwnName()
    await strategy.set_storage_volumes(VolumeMesh([("a", "host-a")]))
    ref = strategy.get_storage_volume("a")
    assert strategy._volume_host == "rack-7"
    assert (ref.volume, ref.volume_hostname) == (("slice", {"procs": 0}), "host-a")


def test_a_lookup_without_a_host_reads_the_attributes_in_mains_order() -> None:
    """Name ``transport_context`` first when a subclass that skips the constructor lacks it."""

    class Partial(TorchStoreStrategy):
        def __init__(self) -> None:
            self.storage_volumes = VolumeMesh([])
            self.volume_id_to_coord = {"a": {"procs": 0}}

    with pytest.raises(AttributeError, match="transport_context"):
        Partial().get_storage_volume("a")
