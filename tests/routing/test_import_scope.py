# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import ast
import importlib
import importlib.util
import re
from collections.abc import Iterator
from pathlib import Path

import pytest
import torchstore

from .test_control_plane import _imported

_PACKAGE = Path(torchstore.__file__).parent
# Main's check whether nixl is installed, which predates the rule against such checks.
_NIXL_PROBE = "torchstore.transport.nixl"
# The coordinator service's modules, which register exit and fork hooks when they load.
_SERVICE_STACK = (
    "torchstore.carrier",
    "torchstore.routing.service",
    "torchstore.routing.tcpstore",
)

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
        "FULL_TENSOR": "torchstore.monarch.storage_volume",
        "InMemoryStore": "torchstore.storage",
        "StorageImpl": "torchstore.storage",
        "StorageVolume": "torchstore.monarch.storage_volume",
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


def _in_service_stack(module: str) -> bool:
    return any(module == m or module.startswith(f"{m}.") for m in _SERVICE_STACK)


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


def _docstrings(modules: dict[str, tuple[ast.Module, bool]]) -> set[int]:
    """The ids of every module, class and function docstring node."""
    return {
        id(node.body[0].value)
        for tree, _ in modules.values()
        for node in ast.walk(tree)
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        )
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
    }


def test_core_modules_reach_the_service_stack_only_through_the_backend() -> None:
    """Import the coordinator service's modules only within them, at any scope but under ``TYPE_CHECKING``.

    Outside them, no string but a docstring names one (under ``TYPE_CHECKING`` too), apart from ``torchstore.backend``'s
    table of backends, whose entries alone that module imports by name.
    """
    for stack_module in _SERVICE_STACK:
        outside = {m for m in _importers(stack_module) if not _in_service_stack(m)}
        assert outside == set(), (stack_module, outside)
    modules = _modules()
    docstrings = _docstrings(modules)
    naming = [
        (module, node)
        for module, (tree, _) in modules.items()
        if not _in_service_stack(module)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
        and any(
            re.search(rf"(?<![\w.]){re.escape(m)}(?!\w)", node.value)
            for m in _SERVICE_STACK
        )
    ]
    table = next(
        node
        for node in modules["torchstore.backend"][0].body
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "_BACKENDS"
    )
    assert [(m, node.value) for m, node in naming] == [
        ("torchstore.backend", "torchstore.routing.service")
    ]
    assert naming[0][1] in list(ast.walk(table))
    # Each read of import_module calls it on a table entry and nothing renames or spells it. So no
    # name built from pieces reaches one.
    backend = modules["torchstore.backend"][0]
    by_name = ("__import__", "import_module")
    calls = {
        id(node.func): node for node in ast.walk(backend) if isinstance(node, ast.Call)
    }
    reads = [node for name, node in _names(backend) if name in by_name]
    assert reads and all(
        id(node) in calls
        and [ast.unparse(arg) for arg in calls[id(node)].args] == ["_BACKENDS[name]"]
        and not calls[id(node)].keywords
        for node in reads
    ), [ast.unparse(node) for node in reads]
    assert {name for name, _ in _names(backend)} & set(by_name) == {"import_module"}
    refused = (*by_name, "__builtins__")
    spelled = [
        ast.unparse(node)
        for node in ast.walk(backend)
        if (
            isinstance(node, ast.alias)
            and node.asname
            and node.name.rpartition(".")[2] in by_name
        )
        or (isinstance(node, ast.Constant) and node.value in refused)
        or (
            isinstance(node, ast.Name)
            and node.id in ("__builtins__", "compile", "eval", "exec")
        )
    ]
    assert spelled == []


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


def _target_names(target: ast.AST) -> list[str]:
    """Each name a binding target binds, inside tuple, list and starred targets too."""
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, ast.Starred):
        return _target_names(target.value)
    if isinstance(target, (ast.Tuple, ast.List)):
        return [name for element in target.elts for name in _target_names(element)]
    return []


def _bindings(node: ast.AST) -> list[tuple[str, ast.AST]]:
    """Each name an assignment, a ``for`` or ``with`` target or a comprehension binds.

    Each comes with the expression it is bound from.
    """
    if isinstance(node, ast.Assign):
        pairs = [(target, node.value) for target in node.targets]
    elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)) and node.value is not None:
        pairs = [(node.target, node.value)]
    elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
        pairs = [(node.target, node.iter)]
    elif isinstance(node, (ast.With, ast.AsyncWith)):
        pairs = [
            (item.optional_vars, item.context_expr)
            for item in node.items
            if item.optional_vars is not None
        ]
    else:
        return []
    return [(name, value) for target, value in pairs for name in _target_names(target)]


def _direct_bindings(node: ast.AST) -> list[tuple[str, ast.AST]]:
    """Each parameter with its default and each attribute an assignment binds (as ``.name``) with its value."""
    if isinstance(node, ast.arguments):
        positional = [*node.posonlyargs, *node.args]
        defaulted = zip(
            positional[len(positional) - len(node.defaults) :], node.defaults
        )
        keywords = zip(node.kwonlyargs, node.kw_defaults)
        return [
            (arg.arg, default)
            for arg, default in (*defaulted, *keywords)
            if default is not None
        ]
    if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        return [
            (f".{target.attr}", node.value)
            for target in targets
            if isinstance(target, ast.Attribute)
        ]
    return []


def _is_sys(node: ast.AST, sys_names: set[str]) -> bool:
    """Whether ``node`` reads a name bound to ``sys``, an attribute so named or assigned it, or imports ``sys`` by name."""
    if isinstance(node, ast.Name):
        return node.id in sys_names
    if isinstance(node, ast.Attribute):
        return node.attr == "sys" or f".{node.attr}" in sys_names
    return (
        isinstance(node, ast.Call)
        and ast.unparse(node.func).rpartition(".")[2] in ("__import__", "import_module")
        and bool(node.args)
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "sys"
    )


def _string_value(node: ast.AST) -> str | None:
    """The value of a string constant, a ``+`` of such or an f-string of constants alone. None for anything else."""
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _string_value(node.left), _string_value(node.right)
        return None if left is None or right is None else left + right
    if isinstance(node, ast.JoinedStr):
        joined = ""
        for value in node.values:
            part = _string_value(value)
            if part is None:
                return None
            joined += part
        return joined
    return None


def _sys_names(tree: ast.Module) -> set[str]:
    """``sys``, each name an import binds to it and each name bound from an expression reading one.

    Also each name, parameter default or attribute (as ``.name``) bound to ``sys`` itself or to such an alias of it.
    """
    imported = {"sys"} | {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
        if alias.name == "sys"
    }
    names = set(imported)
    bindings = [binding for node in ast.walk(tree) for binding in _bindings(node)]
    while (
        more := {
            n
            for n, value in bindings
            if any(_is_sys(node, names) for node in ast.walk(value))
        }
        - names
    ):
        names |= more
    # These aliases are bound to sys itself only. So they spread to no name that merely reads one.
    aliases = set(imported)
    direct = bindings + [b for node in ast.walk(tree) for b in _direct_bindings(node)]
    while more := {n for n, value in direct if _is_sys(value, aliases)} - aliases:
        aliases |= more
    return names | aliases


def test_no_core_module_checks_sys_modules_or_find_spec() -> None:
    """Read ``sys.modules`` in no core module, and ``find_spec`` only in main's NIXL probe.

    ``sys`` is any name an import binds to it or a binding binds from an expression reading it,
    or an attribute so named. ``.modules``, ``.__dict__``, ``getattr``, ``vars`` and
    ``from sys import modules`` count.
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
    """Import ``importlib``, ``pkgutil``, ``builtins``, ``runpy``, ``imp`` or ``zipimport`` only in the backend.

    No other module (the Monarch package's included) names ``__import__``, ``import_module`` or ``__builtins__`` (as a
    name or a string) or calls ``exec``, ``eval`` or ``compile``. Main's NIXL probe imports ``importlib.util.find_spec`` alone.
    """
    loaders = ("builtins", "imp", "importlib", "pkgutil", "runpy", "zipimport")
    by_name = ("__builtins__", "__import__", "import_module")
    found = set()
    for module, (tree, is_package) in _modules().items():
        if module == "torchstore.backend":
            continue
        for node in ast.walk(tree):
            imported = _imports(module, is_package, node)
            if any(name.split(".")[0] in loaders for name in imported):
                found.add((module, ast.unparse(node)))
            if isinstance(node, ast.Constant) and node.value in by_name:
                found.add((module, ast.unparse(node)))
            if isinstance(node, ast.Name) and node.id in ("compile", "eval", "exec"):
                found.add((module, ast.unparse(node)))
        for name, node in _names(tree):
            if name in by_name:
                found.add((module, ast.unparse(node)))
    assert found == {(_NIXL_PROBE, "from importlib.util import find_spec")}


def test_no_module_looks_modules_up_by_name() -> None:
    """Read ``sys.modules`` or call ``find_spec`` in no TorchStore module but main's nixl probe, which predates the rule.

    ``sys`` is a name or attribute that an import, assignment, loop, ``with``, comprehension or parameter default binds to
    it, or a call importing it by name. ``getattr``, ``vars``, ``from sys import modules`` and a star import from ``sys``
    count. So do an imported name and a string (joined with ``+`` too) other than a docstring naming ``find_spec``.
    """
    modules = _modules()
    docstrings = _docstrings(modules)
    found = set()
    for module, (tree, is_package) in modules.items():
        sys_names = _sys_names(tree)
        for node in ast.walk(tree):
            if "sys.modules" in _imports(module, is_package, node):
                found.add((module, ast.unparse(node)))
            elif isinstance(node, ast.Call) and ast.unparse(node.func) in (
                "getattr",
                "vars",
            ):
                if node.args and _is_sys(node.args[0], sys_names):
                    found.add((module, ast.unparse(node)))
            elif isinstance(node, ast.alias) and "find_spec" in (
                node.name,
                node.asname,
            ):
                found.add((module, node.name))
            elif isinstance(node, ast.ImportFrom) and node.module == "sys":
                if any(alias.name == "*" for alias in node.names):
                    found.add((module, ast.unparse(node)))
            elif (
                isinstance(node, (ast.Constant, ast.BinOp, ast.JoinedStr))
                and "find_spec" in (_string_value(node) or "")
                and id(node) not in docstrings
            ):
                found.add((module, ast.unparse(node)))
        for name, node in _names(tree):
            if name in ("modules", "__dict__") and isinstance(node, ast.Attribute):
                if _is_sys(node.value, sys_names):
                    found.add((module, ast.unparse(node)))
            elif name == "find_spec":
                found.add((module, ast.unparse(node)))
    assert {module for module, _ in found} <= {_NIXL_PROBE}, found


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
    assert _imported(_runtime_nodes(tree)) == on_load


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
