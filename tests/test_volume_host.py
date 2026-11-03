# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for resolving storage volume IDs through a volume host a strategy installs."""

import asyncio
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
    strategy.set_volume_host(
        _RecordHost(
            {"b": (("slice", {"procs": 1}), "host-b"), "c": ("handle-c", "host-c")}
        )
    )

    ref = strategy.get_storage_volume("c")
    assert (ref.volume, ref.volume_id, ref.volume_hostname) == (
        "handle-c",
        "c",
        "host-c",
    )
    assert ref.transport_context is strategy.transport_context
    assert ref.default_transport_type is TransportType.Gloo
    assert strategy.get_storage_volume("b").volume_hostname == "host-b"
    with pytest.raises(KeyError):
        strategy.get_storage_volume("a")


def test_a_second_host_replaces_the_first() -> None:
    """Resolve through the host installed last."""
    strategy = TorchStoreStrategy()
    strategy.set_volume_host(
        _RecordHost({"a": ("handle-a", "first-host"), "b": ("handle-b", "host-b")})
    )
    strategy.set_volume_host(_RecordHost({"a": ("handle-a", "second-host")}))

    ref = strategy.get_storage_volume("a")
    assert (ref.volume, ref.volume_hostname) == ("handle-a", "second-host")
    with pytest.raises(KeyError):
        strategy.get_storage_volume("b")


class _SliceMesh(VolumeMesh):
    """A stand-in volume mesh whose every slice is a new handle that equals only itself as Monarch's is."""

    def slice(self, **coord: int) -> object:
        return object()


@pytest.mark.asyncio
async def test_a_host_that_hands_out_new_handles_for_mesh_ids_installs() -> None:
    """Install a host whose handles for the mesh's own IDs are new objects and resolve through it."""
    strategy = TorchStoreStrategy()
    await strategy.set_storage_volumes(_SliceMesh([("a", "mesh-host-a")]))
    handle = strategy.storage_volumes.slice(procs=0)
    strategy.set_volume_host(_RecordHost({"a": (handle, "host-a")}))

    ref = strategy.get_storage_volume("a")
    assert (ref.volume, ref.volume_hostname) == (handle, "host-a")


def test_only_a_volume_host_installs() -> None:
    """Refuse anything but a ``VolumeHost`` and install nothing."""
    strategy = TorchStoreStrategy()
    before = set(vars(strategy))
    with pytest.raises(TypeError, match="expected a VolumeHost, got dict"):
        strategy.set_volume_host({})
    assert set(vars(strategy)) == before


def test_an_installed_host_travels_with_a_pickled_strategy() -> None:
    """Resolve through the host of a strategy's pickled copy as readers that receive a copy do."""
    strategy = TorchStoreStrategy()
    strategy.set_volume_host(_RecordHost({"a": ("handle-a", "host-a")}))

    ref = pickle.loads(pickle.dumps(strategy)).get_storage_volume("a")
    assert (ref.volume, ref.volume_hostname) == ("handle-a", "host-a")


_LOOKUPS = {
    "volume_ids": lambda self: (),
    "endpoints": lambda self, volume_id: "handle",
    "hostname": lambda self, volume_id: "host",
}


@pytest.mark.parametrize("missing", sorted(_LOOKUPS))
def test_a_host_must_implement_every_lookup(missing: str) -> None:
    """Refuse to build a host that leaves any one lookup abstract."""
    lookups = {name: lookup for name, lookup in _LOOKUPS.items() if name != missing}
    with pytest.raises(TypeError, match=f"abstract.*{missing}"):
        type("Partial", (VolumeHost,), lookups)()


@pytest.mark.asyncio
async def test_selection_checks_the_installed_hosts_ids(monkeypatch) -> None:
    """Select the client's volume among the host's IDs."""
    strategy = LocalRankStrategy()
    await strategy.set_storage_volumes(VolumeMesh([("0", "host-0")]))
    strategy.set_volume_host(_RecordHost({"1": ("handle-1", "host-1")}))

    monkeypatch.setenv("RANK", "1")
    assert strategy.select_storage_volume().volume == "handle-1"


@pytest.mark.asyncio
async def test_selection_refuses_an_id_only_the_attributes_know(monkeypatch) -> None:
    """Refuse a client ID the attributes hold but the installed host lacks."""
    strategy = LocalRankStrategy()
    await strategy.set_storage_volumes(VolumeMesh([("0", "host-0")]))
    strategy.set_volume_host(_RecordHost({"1": ("handle-1", "host-1")}))

    monkeypatch.setenv("RANK", "0")
    with pytest.raises(KeyError, match="found for 0 in the volume host"):
        strategy.select_storage_volume()


class _SizedHost(_RecordHost):
    """A host whose length makes an empty one read as false."""

    def __len__(self) -> int:
        return len(self._records)


@pytest.mark.asyncio
async def test_an_empty_host_that_reads_as_false_still_resolves(monkeypatch) -> None:
    """Resolve through an installed host that reads as false, never through the attributes."""
    strategy = LocalRankStrategy()
    await strategy.set_storage_volumes(VolumeMesh([("0", "host-0")]))
    host = _SizedHost({})
    assert not host
    strategy.set_volume_host(host)

    with pytest.raises(KeyError, match="^'0'$"):
        strategy.get_storage_volume("0")
    monkeypatch.setenv("RANK", "0")
    with pytest.raises(KeyError, match="found for 0 in the volume host"):
        strategy.select_storage_volume()


def test_selection_resolves_through_the_host_it_checked(monkeypatch) -> None:
    """Resolve the client's ID through the host that listed it even if another replaced it since."""
    monkeypatch.setenv("RANK", "1")
    strategy = LocalRankStrategy()

    class Replaced(_RecordHost):
        def volume_ids(self) -> Collection[str]:
            # Another thread may install a host between the check and the lookup.
            strategy.set_volume_host(_RecordHost({}))
            return super().volume_ids()

    strategy.set_volume_host(Replaced({"1": ("handle-1", "host-1")}))
    assert strategy.select_storage_volume().volume == "handle-1"


@pytest.mark.asyncio
async def test_selection_without_a_host_ignores_a_first_host_installed_meanwhile(
    monkeypatch,
) -> None:
    """Resolve the client's ID through the mesh tables, not through a first host installed after the check."""
    monkeypatch.setenv("RANK", "1")
    strategy = LocalRankStrategy()
    await strategy.set_storage_volumes(VolumeMesh([("1", "host-1")]))

    class Installs(dict):
        def __contains__(self, volume_id: object) -> bool:
            # Another thread may install a host between the check and the lookup.
            strategy.set_volume_host(_RecordHost({}))
            return super().__contains__(volume_id)

    strategy.volume_id_to_coord = Installs(strategy.volume_id_to_coord)
    assert strategy.select_storage_volume().volume == ("slice", {"procs": 0})


def _own_ref(strategy: TorchStoreStrategy, volume_id: str) -> StorageVolumeRef:
    """The reference that each strategy below resolves every ID to."""
    return StorageVolumeRef(
        "own-handle",
        volume_id,
        strategy.transport_context,
        strategy.default_transport_type,
    )


class _SelectsOwnLookup(LocalRankStrategy):
    """Resolves every ID through a lookup of its own."""

    def get_storage_volume(self, volume_id: str) -> StorageVolumeRef:
        return _own_ref(self, volume_id)


class _InheritsOwnLookup(_SelectsOwnLookup):
    """Inherits its parent's lookup."""


class _ServesLookup(LocalRankStrategy):
    """Serves a lookup of its own through ``__getattribute__``."""

    def __getattribute__(self, name: str) -> Any:
        if name == "get_storage_volume":
            return lambda volume_id: _own_ref(self, volume_id)
        return super().__getattribute__(name)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "strategy_cls", [_SelectsOwnLookup, _InheritsOwnLookup, _ServesLookup]
)
async def test_selection_resolves_through_a_subclass_lookup(
    monkeypatch, strategy_cls: type[LocalRankStrategy]
) -> None:
    """Resolve the client's ID through a lookup a subclass defines, inherits or serves as main did."""
    monkeypatch.setenv("RANK", "0")
    strategy = strategy_cls()
    await strategy.set_storage_volumes(VolumeMesh([("0", "host-0")]))
    assert strategy.select_storage_volume().volume == "own-handle"


@pytest.mark.asyncio
@pytest.mark.parametrize("patch_base", [False, True], ids=["instance", "base"])
async def test_selection_resolves_through_a_patched_lookup(
    monkeypatch, patch_base: bool
) -> None:
    """Resolve the client's ID through a patch of the instance's or the base class's lookup as main did."""
    monkeypatch.setenv("RANK", "0")
    strategy = LocalRankStrategy()
    await strategy.set_storage_volumes(VolumeMesh([("0", "host-0")]))
    looked_up: list[str] = []

    def lookup(*args: Any) -> str:
        # Python binds a patch of the class's lookup to the strategy and not one of the instance's.
        looked_up.append(args[-1])
        return "patched-ref"

    target = TorchStoreStrategy if patch_base else strategy
    monkeypatch.setattr(target, "get_storage_volume", lookup)
    assert strategy.select_storage_volume() == "patched-ref"
    assert looked_up == ["0"]


@pytest.mark.asyncio
async def test_selection_names_a_missing_volume_before_a_patched_lookup(
    monkeypatch,
) -> None:
    """Refuse a client ID the attributes lack with main's message and never reach a patched lookup."""
    monkeypatch.setenv("RANK", "1")
    strategy = LocalRankStrategy()
    await strategy.set_storage_volumes(VolumeMesh([("0", "host-0")]))
    lookup = Mock(return_value="patched-ref")
    monkeypatch.setattr(strategy, "get_storage_volume", lookup)
    with pytest.raises(KeyError, match="found for 1 self.volume_id_to_coord="):
        strategy.select_storage_volume()
    lookup.assert_not_called()


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
    strategy.set_volume_host(_RecordHost({"b": ("handle-b", "host-b")}))
    mesh = Mock(get_id=Mock(call=AsyncMock(side_effect=RuntimeError("a volume died"))))
    with pytest.raises(RuntimeError, match="a volume died"):
        await strategy.set_storage_volumes(mesh)

    assert strategy.storage_volumes is mesh
    for volume_id in ("a", "b"):
        with pytest.raises(KeyError, match=f"^'{volume_id}'$"):
            strategy.get_storage_volume(volume_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("fails", [False, True])
async def test_a_host_installed_while_volumes_are_set_is_dropped(fails: bool) -> None:
    """Drop the host a set finds and one installed while it waits for the mesh even if it fails."""
    strategy = TorchStoreStrategy()
    strategy.set_volume_host(_RecordHost({"b": ("handle-b", "host-b")}))
    asked, answer = asyncio.Event(), asyncio.Event()

    async def get_id() -> list:
        asked.set()
        await answer.wait()
        if fails:
            raise RuntimeError("a volume died")
        return [({"procs": 0}, ("a", "mesh-host-a"))]

    mesh = VolumeMesh([])
    mesh.get_id = Mock(call=get_id)
    setting = asyncio.ensure_future(strategy.set_storage_volumes(mesh))
    await asked.wait()
    with pytest.raises(KeyError, match="^'b'$"):
        strategy.get_storage_volume("b")
    strategy.set_volume_host(_RecordHost({"c": ("handle-c", "host-c")}))
    answer.set()
    if fails:
        with pytest.raises(RuntimeError, match="a volume died"):
            await setting
    else:
        await setting
        assert strategy.get_storage_volume("a").volume == ("slice", {"procs": 0})

    with pytest.raises(KeyError, match="^'c'$"):
        strategy.get_storage_volume("c")


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


# Each inherits its overrides from a parent class.
class _InheritsLookup(_OwnLookup):
    pass


class _InheritsController(ControllerStorageVolumes):
    pass


@pytest.mark.parametrize(
    "strategy_cls, overridden",
    [
        (_OwnSet, "set_storage_volumes"),
        (_OwnSelect, "select_storage_volume"),
        (_OwnLookup, "get_storage_volume"),
        (_InheritsLookup, "get_storage_volume"),
        (
            ControllerStorageVolumes,
            "set_storage_volumes, select_storage_volume, get_storage_volume",
        ),
        (
            _InheritsController,
            "set_storage_volumes, select_storage_volume, get_storage_volume",
        ),
    ],
)
def test_a_strategy_that_overrides_a_host_method_refuses_a_host(
    strategy_cls: type[TorchStoreStrategy], overridden: str
) -> None:
    """Refuse a host where the class overrides a method that reads or drops it, and install none."""
    strategy = strategy_cls()
    with pytest.raises(
        TypeError, match=f"overrides {overridden}, and TorchStore cannot tell"
    ):
        strategy.set_volume_host(_RecordHost({}))
    assert vars(strategy).keys() == vars(strategy_cls()).keys()


_COPY_HOOKS = (
    "__getstate__",
    "__setstate__",
    "__reduce__",
    "__reduce_ex__",
    "__copy__",
    "__deepcopy__",
)


@pytest.mark.parametrize("hook", _COPY_HOOKS)
def test_a_strategy_class_with_its_own_copy_hook_refuses_a_host(hook: str) -> None:
    """Refuse a host where the class decides what its copies keep and install none."""
    strategy = type(
        "OwnHook", (TorchStoreStrategy,), {hook: lambda self, *args: None}
    )()
    before = set(vars(strategy))
    with pytest.raises(
        TypeError, match=f"overrides {hook}, and TorchStore cannot tell"
    ):
        strategy.set_volume_host(_RecordHost({}))
    assert set(vars(strategy)) == before


@pytest.mark.parametrize(
    "name",
    [
        "set_storage_volumes",
        "select_storage_volume",
        "get_storage_volume",
        "__reduce_ex__",
    ],
)
def test_a_strategy_instance_with_its_own_host_method_refuses_a_host(name: str) -> None:
    """Refuse a host where a patch gave the instance its own set, lookup or copy hook."""
    strategy = TorchStoreStrategy()
    vars(strategy)[name] = lambda *args: None
    before = set(vars(strategy))
    with pytest.raises(
        TypeError, match=f"overrides {name}, and TorchStore cannot tell"
    ):
        strategy.set_volume_host(_RecordHost({}))
    assert set(vars(strategy)) == before


@pytest.mark.parametrize(
    "name",
    [
        "set_storage_volumes",
        "select_storage_volume",
        "get_storage_volume",
        "__reduce_ex__",
    ],
)
def test_a_patch_of_the_base_class_host_method_refuses_a_host(
    monkeypatch, name: str
) -> None:
    """Refuse a host where a patch replaced the base class's own set, lookup or copy hook that a strategy inherits."""
    monkeypatch.setattr(TorchStoreStrategy, name, lambda *args: None)
    strategy = LocalRankStrategy()
    before = set(vars(strategy))
    with pytest.raises(
        TypeError, match=f"overrides {name}, and TorchStore cannot tell"
    ):
        strategy.set_volume_host(_RecordHost({}))
    assert set(vars(strategy)) == before


def test_a_strategy_class_with_its_own_attribute_lookup_refuses_a_host() -> None:
    """Refuse a host where the class's ``__getattribute__`` could stand in for any host method."""
    strategy = _ServesLookup()
    before = set(vars(strategy))
    with pytest.raises(
        TypeError, match="overrides __getattribute__, and TorchStore cannot tell"
    ):
        strategy.set_volume_host(_RecordHost({}))
    assert set(vars(strategy)) == before


@pytest.mark.parametrize(
    "strategy_cls",
    [TorchStoreStrategy, LocalRankStrategy, HostStrategy, ControllerStorageVolumes],
)
def test_type_hints_of_every_strategy_class_resolve(
    strategy_cls: type[TorchStoreStrategy],
) -> None:
    """Resolve each strategy class's hints and those of its host method at run time."""
    assert typing.get_type_hints(strategy_cls) == {}
    assert typing.get_type_hints(strategy_cls.set_volume_host) == {
        "host": VolumeHost,
        "return": type(None),
    }


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


class _OwnName(TorchStoreStrategy):
    """Keeps an attribute of its own named like the host."""

    def __init__(self) -> None:
        super().__init__()
        self._volume_host = "rack-7"

    def own(self) -> str:
        return self._volume_host


class _TorchStoreStrategy(TorchStoreStrategy):
    """Keeps a private ``__volume_host`` that Python mangles by this class's name."""

    def __init__(self) -> None:
        super().__init__()
        self.__volume_host = "rack-7"

    def own(self) -> str:
        return self.__volume_host


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy_cls", [_OwnName, _TorchStoreStrategy])
async def test_a_subclass_attribute_named_volume_host_is_its_own(
    strategy_cls: type[_OwnName | _TorchStoreStrategy],
) -> None:
    """Keep a subclass's own ``_volume_host`` or ``__volume_host`` through set and lookups."""
    strategy = strategy_cls()
    with pytest.raises(KeyError, match="^'a'$"):
        strategy.get_storage_volume("a")
    await strategy.set_storage_volumes(VolumeMesh([("a", "host-a")]))
    ref = strategy.get_storage_volume("a")
    assert strategy.own() == "rack-7"
    assert (ref.volume, ref.volume_hostname) == (("slice", {"procs": 0}), "host-a")


def test_a_lookup_without_a_host_reads_the_attributes_in_mains_order() -> None:
    """Name ``transport_context`` first when a subclass that skips the constructor lacks it."""

    class Partial(TorchStoreStrategy):
        def __init__(self) -> None:
            self.storage_volumes = VolumeMesh([])
            self.volume_id_to_coord = {"a": {"procs": 0}}

    with pytest.raises(AttributeError, match="transport_context"):
        Partial().get_storage_volume("a")
