# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import ast
import inspect
from abc import ABC
from collections.abc import Iterable

import torchstore.routing.control_plane as control_plane_module
from torchstore.routing.control_plane import ControlPlane

# The design doc's ControlPlane sketch.
_DESIGN = {
    "init": "(self, publishers: 'Collection[str]', strategy: 'TorchStoreStrategy') -> 'None'",
    "register_layout": "(self, rank: 'str', key: 'str', registrations: 'Mapping[str, KeyRegistration]') -> 'None'",
    "get_layouts": "(self, key: 'str') -> 'Registrations'",
    "strategy": "(self) -> 'TorchStoreStrategy'",
}


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
    on_load = {"__future__", "abc", "collections.abc", "typing", "._model"}
    assert _imported(tree.body) == on_load
    assert _imported(type_checking) == {"torchstore.strategy"}
    assert _imported(ast.walk(tree)) == on_load | {"torchstore.strategy"}
    assert ControlPlane.__mro__[1:] == (ABC, object)
    assert all(
        inspect.iscoroutinefunction(getattr(ControlPlane, name))
        for name in ControlPlane.__abstractmethods__
    )
