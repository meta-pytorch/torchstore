# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import ast
import dataclasses
import inspect
import pickle
from abc import ABC
from collections.abc import Iterable

import pytest
import torchstore as ts
import torchstore.routing.control_plane as control_plane_module
from torchstore.routing._model import KeyRegistration, RankRole, Registrations
from torchstore.routing.control_plane import ControlPlane, ControlPlaneAddress
from torchstore.strategy import TorchStoreStrategy

from . import file_store_control_plane

# The design doc's ControlPlane sketch.
_DESIGN = {
    "init": "(self, publishers: 'Collection[str]', strategy: 'TorchStoreStrategy') -> 'None'",
    "register_layout": "(self, rank: 'str', key: 'str', registrations: 'Mapping[str, KeyRegistration]') -> 'None'",
    "get_layouts": "(self, key: 'str') -> 'Registrations'",
    "strategy": "(self) -> 'TorchStoreStrategy'",
}


@dataclasses.dataclass(frozen=True)
class _Address(ControlPlaneAddress):
    port: int

    async def connect(self) -> ControlPlane:
        raise NotImplementedError

    def __str__(self) -> str:
        return f"host:{self.port}"


def _imported(nodes: Iterable[ast.AST]) -> set[str]:
    """The modules that the import statements among ``nodes`` name."""
    modules = set()
    for node in nodes:
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.add("." * node.level + (node.module or ""))
    return modules


def test_interface_signatures_match_the_design() -> None:
    """Keep the design doc's four async methods, with exactly its signatures."""
    assert ControlPlane.__abstractmethods__ == set(_DESIGN)
    public = {name for name in dir(ControlPlane) if not name.startswith("_")}
    assert public == set(_DESIGN)
    for name, signature in _DESIGN.items():
        method = getattr(ControlPlane, name)
        assert inspect.iscoroutinefunction(method), name
        assert str(inspect.signature(method)) == signature, name


def test_control_plane_imports_no_backend() -> None:
    """Keep the interface a plain ABC of coroutines that imports no backend itself."""
    # Exact, because any torchstore import can load Monarch, as ``_model`` already does.
    tree = ast.parse(inspect.getsource(control_plane_module))
    type_checking = [
        statement
        for node in tree.body
        if isinstance(node, ast.If) and ast.unparse(node.test) == "TYPE_CHECKING"
        for statement in node.body
    ]
    on_load = {
        "__future__",
        "abc",
        "collections.abc",
        "dataclasses",
        "typing",
        "._model",
    }
    assert _imported(tree.body) == on_load
    assert _imported(type_checking) == {"torchstore.strategy"}
    assert _imported(ast.walk(tree)) == on_load | {"torchstore.strategy"}
    assert ControlPlane.__mro__[1:] == (ABC, object)
    assert all(
        inspect.iscoroutinefunction(getattr(ControlPlane, name))
        for name in ControlPlane.__abstractmethods__
    )


def test_address_is_an_abstract_frozen_dataclass() -> None:
    """Name a store, connect asynchronously, and pickle and compare by value."""
    assert ts.ControlPlaneAddress is ControlPlaneAddress
    assert ts.ControlPlane is ControlPlane
    assert ControlPlaneAddress.__abstractmethods__ == {"connect", "__str__"}
    assert inspect.iscoroutinefunction(ControlPlaneAddress.connect)
    assert [field.name for field in dataclasses.fields(ControlPlaneAddress)] == [
        "store_name"
    ]
    with pytest.raises(TypeError, match="abstract"):
        ControlPlaneAddress("store")

    # A dataclass's str() is its repr, whose spaces would split the log line's field.
    @dataclasses.dataclass(frozen=True)
    class Unnamed(ControlPlaneAddress):
        async def connect(self) -> ControlPlane:
            raise NotImplementedError

    with pytest.raises(TypeError, match="abstract"):
        Unnamed("store")

    address = _Address("store", 1)
    assert pickle.loads(pickle.dumps(address)) == address
    assert hash(address) == hash(_Address("store", 1))
    assert address not in (_Address("store", 2), _Address("other", 1))
    with pytest.raises(dataclasses.FrozenInstanceError):
        address.port = 2


def test_control_plane_types_are_public() -> None:
    """Export the interface, its address, the TorchStore types it names and the role enum."""
    public = {
        "ControlPlane": ControlPlane,
        "ControlPlaneAddress": ControlPlaneAddress,
        "KeyRegistration": KeyRegistration,
        "Registrations": Registrations,
        "TorchStoreStrategy": TorchStoreStrategy,
        "RankRole": RankRole,
    }
    for name, value in public.items():
        assert name in ts.__all__ and getattr(ts, name) is value, name


def test_file_store_control_plane_uses_only_public_names() -> None:
    """Build the test's own control plane from ``ts.__all__`` and the standard library.

    Private names appear only on the plane itself, never on what TorchStore hands it.
    """
    tree = ast.parse(inspect.getsource(file_store_control_plane))
    # Only what the helper needs, so no import machinery or introspection module.
    allowed = {"__future__", "asyncio", "collections", "json", "math", "os"}
    allowed |= {"pickle", "tempfile", "torchstore", "urllib"}
    public = set(ts.__all__)
    aliases, imported = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] in allowed, alias.name
                if alias.name.split(".")[0] == "torchstore":
                    assert alias.name == "torchstore", alias.name
                    aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom):
            package = (node.module or "").split(".")[0]
            assert node.level == 0 and package in allowed, ast.unparse(node)
            if package == "torchstore":
                names = {alias.name for alias in node.names}
                assert node.module == "torchstore", ast.unparse(node)
                assert names <= public, ast.unparse(node)
                imported |= names
    # The package appears only as ``ts.<public name>``, so ``getattr(ts, ...)`` fails too.
    attributes = [node for node in ast.walk(tree) if isinstance(node, ast.Attribute)]
    used = {
        node.attr
        for node in attributes
        if isinstance(node.value, ast.Name) and node.value.id in aliases
    }
    bases = {id(node.value) for node in attributes}
    # Dunders such as __builtins__, and these builtins, reach names no import shows.
    banned = {"compile", "eval", "exec", "getattr", "globals", "locals", "vars"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Name, ast.Attribute)):
            name = node.id if isinstance(node, ast.Name) else node.attr
            assert not name.startswith("__"), f"line {node.lineno}"
            assert name not in banned, f"line {node.lineno}"
        if isinstance(node, ast.Name):
            assert node.id not in aliases or id(node) in bases, f"line {node.lineno}"
    for node in attributes:
        if node.attr.startswith("_"):
            on_self = isinstance(node.value, ast.Name) and node.value.id == "self"
            assert on_self, f"line {node.lineno}"
    assert (used or imported) and used <= public, used
    assert issubclass(file_store_control_plane.FileStoreControlPlane, ts.ControlPlane)
