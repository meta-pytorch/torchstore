# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import ast
import importlib
import importlib.util
from collections.abc import Iterator
from pathlib import Path

import pytest
import torchstore

from .test_control_plane import _imported

_PACKAGE = Path(torchstore.__file__).parent
# Main's check whether nixl is installed, which predates the rule against such checks.
_NIXL_PROBE = "torchstore.transport.nixl"

# Each name in an old module path's ``__all__``, and the TorchStore module that defines it or
# imports it from Monarch.
_OLD_PATHS = {
    "torchstore.controller": {
        "Controller": "torchstore.monarch.controller",
        "ControllerDirectory": "torchstore.monarch.controller",
        "ObjectType": "torchstore.directory",
        "StorageInfo": "torchstore.directory",
    },
    "torchstore.routing.coordinator": {
        "MonarchControlPlane": "torchstore.monarch.coordinator",
        "RoutingCoordinator": "torchstore.monarch.coordinator",
    },
    "torchstore.storage_volume": {
        name: "torchstore.monarch.storage_volume"
        for name in ("FULL_TENSOR", "InMemoryStore", "StorageImpl", "StorageVolume")
    },
    "torchstore.transport.monarch_rdma": {
        name: "torchstore.monarch.rdma"
        for name in (
            "MonarchRDMATransportBuffer",
            "RDMAAction",
            "RDMABuffer",
            "RdmaContext",
            "monarch_rdma_available",
            "monarch_rdma_transport_available",
        )
    },
}
# Names the old modules defined outside ``__all__``, which their old paths still bind.
_OLD_PRIVATE = {
    "torchstore.routing.coordinator": {
        "_timed": "torchstore.monarch.coordinator",
        "logger": "torchstore.monarch.coordinator",
    },
    "torchstore.storage_volume": {"logger": "torchstore.monarch.storage_volume"},
}
# The old paths that load Monarch. Monarch RDMA's loads without it, as its new path does.
_OLD_PATHS_NEEDING_MONARCH = {
    "torchstore.controller",
    "torchstore.routing.coordinator",
    "torchstore.storage_volume",
}


def _modules() -> dict[str, tuple[ast.Module, bool]]:
    """Each library module by dotted name, parsed, and whether it is a package."""
    found = {}
    for path in sorted(_PACKAGE.rglob("*.py")):
        parts = path.relative_to(_PACKAGE.parent).with_suffix("").parts
        is_package = parts[-1] == "__init__"
        name = ".".join(parts[:-1] if is_package else parts)
        found[name] = (ast.parse(path.read_text(encoding="utf-8")), is_package)
    return found


def _runtime_nodes(tree: ast.Module) -> Iterator[ast.AST]:
    """Every node at any scope, except those under ``if TYPE_CHECKING:``."""
    exempt = {
        id(node)
        for guard in ast.walk(tree)
        if isinstance(guard, ast.If) and ast.unparse(guard.test) == "TYPE_CHECKING"
        for statement in guard.body
        for node in ast.walk(statement)
    }
    return (node for node in ast.walk(tree) if id(node) not in exempt)


def _imports(module: str, is_package: bool, node: ast.AST) -> list[str]:
    """The modules an import statement names, relative ones resolved."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom):
        package = module if is_package else module.rpartition(".")[0]
        relative = "." * node.level + (node.module or "")
        name = importlib.util.resolve_name(relative, package)
        return [name, *(f"{name}.{alias.name}" for alias in node.names)]
    return []


def _importers(prefix: str) -> set[str]:
    """Modules that import ``prefix`` or a submodule of it outside ``TYPE_CHECKING``."""
    return {
        module
        for module, (tree, is_package) in _modules().items()
        for node in _runtime_nodes(tree)
        for name in _imports(module, is_package, node)
        if name == prefix or name.startswith(f"{prefix}.")
    }


def _in_monarch_package(module: str) -> bool:
    return module == "torchstore.monarch" or module.startswith("torchstore.monarch.")


def test_only_the_monarch_package_imports_monarch() -> None:
    """Name Monarch in import statements only in ``torchstore.monarch``, at any scope.

    ``TYPE_CHECKING`` blocks aside. The by-name test covers imports without a statement.
    """
    assert _importers("monarch")
    assert {m for m in _importers("monarch") if not _in_monarch_package(m)} == set()


def test_core_modules_reach_the_monarch_package_only_through_the_backend() -> None:
    """Import ``torchstore.monarch`` in statements only within it and from the old paths.

    ``torchstore.backend``, the only module that imports by name, names it only in its table.
    """
    outside = {
        m for m in _importers("torchstore.monarch") if not _in_monarch_package(m)
    }
    assert outside == set(_OLD_PATHS)
    tree = _modules()["torchstore.backend"][0]
    table = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "_BACKENDS"
    )
    in_table = {id(node) for node in ast.walk(table)}
    naming = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and "torchstore.monarch" in node.value
    ]
    assert naming and all(id(node) in in_table for node in naming)


def test_no_module_imports_an_old_path() -> None:
    """Import the moved modules from ``torchstore.monarch``, never from their old paths.

    Only the Monarch backend imports each old path, so that the old paths are attributes again
    once it loads, as they were after ``import torchstore`` on main.
    """
    for old in _OLD_PATHS:
        assert _importers(old) - set(_OLD_PATHS) == {"torchstore.monarch.backend"}, old


def _names(tree: ast.Module) -> Iterator[tuple[str, ast.AST]]:
    """Each name a module reads, as a bare name or an attribute, with its node."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            yield node.id, node
        elif isinstance(node, ast.Attribute):
            yield node.attr, node


def _bindings(node: ast.AST) -> list[tuple[str, ast.AST]]:
    """Each name an assignment statement or expression binds, with the value bound."""
    if isinstance(node, ast.Assign):
        return [(t.id, node.value) for t in node.targets if isinstance(t, ast.Name)]
    if isinstance(node, (ast.AnnAssign, ast.NamedExpr)) and node.value is not None:
        return (
            [(node.target.id, node.value)] if isinstance(node.target, ast.Name) else []
        )
    return []


def _is_sys(node: ast.AST, sys_names: set[str]) -> bool:
    """Whether ``node`` reads a name bound to ``sys`` or an attribute named ``sys``."""
    return (isinstance(node, ast.Name) and node.id in sys_names) or (
        isinstance(node, ast.Attribute) and node.attr == "sys"
    )


def _sys_names(tree: ast.Module) -> set[str]:
    """``sys``, each name an import binds to it and each name assigned one of those."""
    names = {"sys"} | {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
        if alias.name == "sys"
    }
    bindings = [binding for node in ast.walk(tree) for binding in _bindings(node)]
    while more := {n for n, value in bindings if _is_sys(value, names)} - names:
        names |= more
    return names


def test_no_core_module_checks_sys_modules_or_find_spec() -> None:
    """Read ``sys.modules`` in no core module, and ``find_spec`` only in main's NIXL probe.

    ``sys`` is any name an import or assignment binds to it, or an attribute so named, and
    ``.modules``, ``.__dict__``, ``getattr``, ``vars`` and ``from sys import modules`` count.
    """
    for module, (tree, is_package) in _modules().items():
        if _in_monarch_package(module):
            continue
        sys_names = _sys_names(tree)
        for node in ast.walk(tree):
            assert "sys.modules" not in _imports(module, is_package, node), module
            if isinstance(node, ast.Call) and ast.unparse(node.func) in (
                "getattr",
                "vars",
            ):
                assert not node.args or not _is_sys(node.args[0], sys_names), module
        for name, node in _names(tree):
            if name in ("modules", "__dict__") and isinstance(node, ast.Attribute):
                assert not _is_sys(node.value, sys_names), module
            if name == "find_spec":
                assert module == _NIXL_PROBE, (module, ast.unparse(node))


def test_only_the_backend_imports_by_name() -> None:
    """Import ``importlib``, ``pkgutil`` or ``builtins`` only in ``torchstore.backend``.

    No other core module names ``__import__`` or ``import_module``, as a name or a string.
    Main's NIXL probe imports ``importlib.util.find_spec`` alone.
    """
    found = set()
    for module, (tree, is_package) in _modules().items():
        if _in_monarch_package(module) or module == "torchstore.backend":
            continue
        for node in ast.walk(tree):
            imported = _imports(module, is_package, node)
            if any(
                name.split(".")[0] in ("builtins", "importlib", "pkgutil")
                for name in imported
            ):
                found.add((module, ast.unparse(node)))
            if isinstance(node, ast.Constant) and node.value in (
                "__import__",
                "import_module",
            ):
                found.add((module, ast.unparse(node)))
        for name, node in _names(tree):
            if name in ("__import__", "import_module"):
                found.add((module, ast.unparse(node)))
    assert found == {(_NIXL_PROBE, "from importlib.util import find_spec")}


def test_layout_registry_loads_only_the_standard_library() -> None:
    """Import only the standard library when the layout registry loads or runs."""
    path = Path(torchstore.__file__).parent / "routing" / "layout_registry.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    type_checking = [
        statement
        for node in tree.body
        if isinstance(node, ast.If) and ast.unparse(node.test) == "TYPE_CHECKING"
        for statement in node.body
    ]
    on_load = {
        "__future__",
        "asyncio",
        "collections.abc",
        "dataclasses",
        "logging",
        "typing",
    }
    annotations = {"._model", "torchstore.strategy"}
    assert _imported(tree.body) == on_load
    assert _imported(type_checking) == annotations
    guarded = {id(node) for statement in type_checking for node in ast.walk(statement)}
    unguarded = (node for node in ast.walk(tree) if id(node) not in guarded)
    assert _imported(unguarded) == on_load


def _assert_reexports(old: str) -> None:
    """Bind each name an old module defined at its old path, as the same object."""
    module = importlib.import_module(old)
    assert sorted(module.__all__) == sorted(_OLD_PATHS[old])
    for name, home in {**_OLD_PATHS[old], **_OLD_PRIVATE.get(old, {})}.items():
        assert getattr(module, name) is getattr(importlib.import_module(home), name)


@pytest.mark.parametrize("old", sorted(set(_OLD_PATHS) - _OLD_PATHS_NEEDING_MONARCH))
def test_old_module_paths_reexport_the_moved_names(old: str) -> None:
    """Re-export the moved names at the old paths that load without Monarch.

    ``test_old_module_paths.py`` checks the paths that need Monarch.
    """
    _assert_reexports(old)
