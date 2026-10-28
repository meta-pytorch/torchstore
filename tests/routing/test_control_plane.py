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
from torchstore.routing._model import KeyRegistration, Registrations
from torchstore.routing.control_plane import ControlPlane, ControlPlaneAddress

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
    """Keep the design doc's four async methods, with exactly its signatures.

    Beside them, only an optional log name and an async ``close`` that does nothing.
    """
    assert ControlPlane.__abstractmethods__ == set(_DESIGN)
    public = {name for name in dir(ControlPlane) if not name.startswith("_")}
    assert public == set(_DESIGN) | {"kind", "close"}
    for name, signature in _DESIGN.items():
        method = getattr(ControlPlane, name)
        assert inspect.iscoroutinefunction(method), name
        assert str(inspect.signature(method)) == signature, name
    assert ControlPlane.kind is None
    assert inspect.iscoroutinefunction(ControlPlane.close)
    assert str(inspect.signature(ControlPlane.close)) == "(self) -> 'None'"


def test_control_plane_imports_no_backend() -> None:
    """Keep the interface a plain ABC that imports no backend itself."""
    # Exact for TorchStore modules: any of them can load Monarch, as ``_model`` does.
    tree = ast.parse(inspect.getsource(control_plane_module))
    type_checking = [
        statement
        for node in tree.body
        if isinstance(node, ast.If) and ast.unparse(node.test) == "TYPE_CHECKING"
        for statement in node.body
    ]

    def torchstore_imports(nodes: Iterable[ast.AST]) -> set[str]:
        return {
            module
            for module in _imported(nodes)
            if module.startswith(".") or module.split(".")[0] == "torchstore"
        }

    assert torchstore_imports(tree.body) == {"._model"}
    assert torchstore_imports(type_checking) == {"torchstore.strategy"}
    assert torchstore_imports(ast.walk(tree)) == {"._model", "torchstore.strategy"}
    assert ControlPlane.__mro__[1:] == (ABC, object)


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
    """Export the two TorchStore types that the interface's signatures name."""
    public = {"KeyRegistration": KeyRegistration, "Registrations": Registrations}
    for name, value in public.items():
        assert name in ts.__all__ and getattr(ts, name) is value, name
