# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

import torchstore

from .test_control_plane import _imported

# The Actor modules: the only ones with unguarded module-level Monarch imports.
_IMPORT_MONARCH_ON_LOAD = {
    "controller",
    "routing.coordinator",
    "storage_volume",
}
# spmd keeps its try-guarded StoreJob import at module scope because tests patch it.
_GUARDED_ON_LOAD = {"spmd"}
_IMPORT_CALLS = {"__import__", "import_module", "importlib.import_module"}


def _imports_monarch(statement: ast.stmt) -> bool:
    if isinstance(statement, ast.Import):
        return any(alias.name.split(".")[0] == "monarch" for alias in statement.names)
    if isinstance(statement, ast.ImportFrom):
        return (
            not statement.level and (statement.module or "").split(".")[0] == "monarch"
        )
    call = getattr(statement, "value", None)
    return (
        isinstance(call, ast.Call)
        and ast.unparse(call.func) in _IMPORT_CALLS
        and bool(call.args)
        and isinstance(call.args[0], ast.Constant)
        and str(call.args[0].value).split(".")[0] == "monarch"
    )


def _load_time_statements(body: list[ast.stmt]) -> Iterator[ast.stmt]:
    """Statements that run on load, including under try, with, if, for, while, match and class bodies, but not TYPE_CHECKING."""
    for statement in body:
        yield statement
        if isinstance(statement, ast.Try):
            yield from _load_time_statements(statement.body)
            for handler in statement.handlers:
                yield from _load_time_statements(handler.body)
            yield from _load_time_statements(statement.orelse)
            yield from _load_time_statements(statement.finalbody)
        elif isinstance(statement, (ast.With, ast.For, ast.While)):
            yield from _load_time_statements(statement.body)
            yield from _load_time_statements(getattr(statement, "orelse", []))
        elif isinstance(statement, ast.If):
            if ast.unparse(statement.test) != "TYPE_CHECKING":
                yield from _load_time_statements(statement.body)
            yield from _load_time_statements(statement.orelse)
        elif isinstance(statement, ast.ClassDef):
            yield from _load_time_statements(statement.body)
        elif isinstance(statement, ast.Match):
            for case in statement.cases:
                yield from _load_time_statements(case.body)


def _monarch_importers(*, guarded: bool) -> set[str]:
    """Modules importing Monarch at module level. With ``guarded``, under try, with, if, for, while, match and class bodies too."""
    package = Path(torchstore.__file__).parent
    found = set()
    for path in package.rglob("*.py"):
        body = ast.parse(path.read_text(encoding="utf-8")).body
        statements = _load_time_statements(body) if guarded else body
        if any(_imports_monarch(statement) for statement in statements):
            found.add(".".join(path.relative_to(package).with_suffix("").parts))
    return found


def test_only_known_modules_import_monarch_on_load() -> None:
    """Keep unguarded module-level Monarch imports to the modules that have them."""
    assert _monarch_importers(guarded=False) == _IMPORT_MONARCH_ON_LOAD


def test_only_known_modules_guard_a_load_time_monarch_import() -> None:
    """Find load-time Monarch imports, statements or import calls, under try, with, if, for, while, match or class bodies too."""
    assert (
        _monarch_importers(guarded=True) == _IMPORT_MONARCH_ON_LOAD | _GUARDED_ON_LOAD
    )


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


def test_type_checkers_see_the_lazy_names_but_not_module_getattr() -> None:
    """Hide ``__getattr__`` from type checkers, which import the lazy names, so they flag unknown ones."""
    tree = ast.parse(Path(torchstore.__file__).read_text(encoding="utf-8"))
    guards = [
        node
        for node in tree.body
        if isinstance(node, ast.If) and ast.unparse(node.test) == "TYPE_CHECKING"
    ]
    typed = {
        alias.asname or alias.name
        for guard in guards
        for node in guard.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    getattrs = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "__getattr__"
    ]
    assert {"spmd", "initialize_spmd"} <= typed
    assert len(getattrs) == 1
    assert any(getattrs[0] in guard.orelse for guard in guards)
