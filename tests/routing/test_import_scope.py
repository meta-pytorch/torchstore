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

# New routing modules, starting with the control-plane interface, must not join this list.
_IMPORT_MONARCH_ON_LOAD = {
    "api",
    "controller",
    "routing.coordinator",
    "spmd",
    "storage_volume",
    "strategy",
    "utils",
}
# A bare handler, or one catching any of these, guards the imports in its try block.
_IMPORT_GUARDS = {"BaseException", "Exception", "ImportError", "ModuleNotFoundError"}


def _imports_monarch(statement: ast.stmt) -> bool:
    if isinstance(statement, ast.Import):
        return any(alias.name.split(".")[0] == "monarch" for alias in statement.names)
    return (
        isinstance(statement, ast.ImportFrom)
        and not statement.level
        and (statement.module or "").split(".")[0] == "monarch"
    )


def _guards_imports(statement: ast.Try) -> bool:
    for handler in statement.handlers:
        caught = handler.type
        names = caught.elts if isinstance(caught, ast.Tuple) else [caught]
        if caught is None or any(
            isinstance(name, ast.Name) and name.id in _IMPORT_GUARDS for name in names
        ):
            return True
    return False


def _unguarded_monarch_imports(body: list[ast.stmt]) -> Iterator[ast.stmt]:
    """Monarch imports that run on load, in module-level blocks and class bodies too.

    Those under ``TYPE_CHECKING`` or in a ``try`` catching ``ImportError`` are guarded.
    """
    for statement in body:
        if _imports_monarch(statement):
            yield statement
        elif isinstance(statement, ast.If):
            if ast.unparse(statement.test) != "TYPE_CHECKING":
                yield from _unguarded_monarch_imports(statement.body)
            yield from _unguarded_monarch_imports(statement.orelse)
        elif isinstance(statement, ast.Try):
            if not _guards_imports(statement):
                yield from _unguarded_monarch_imports(statement.body)
            for handler in statement.handlers:
                yield from _unguarded_monarch_imports(handler.body)
            yield from _unguarded_monarch_imports(statement.orelse)
            yield from _unguarded_monarch_imports(statement.finalbody)
        elif isinstance(statement, (ast.With, ast.For, ast.While)):
            yield from _unguarded_monarch_imports(statement.body)
            yield from _unguarded_monarch_imports(getattr(statement, "orelse", []))
        elif isinstance(statement, ast.ClassDef):
            yield from _unguarded_monarch_imports(statement.body)
        elif isinstance(statement, ast.Match):
            for case in statement.cases:
                yield from _unguarded_monarch_imports(case.body)


def test_only_known_modules_import_monarch_on_load() -> None:
    """Keep unguarded module-level Monarch imports to the modules that have them."""
    package = Path(torchstore.__file__).parent
    found = {
        ".".join(path.relative_to(package).with_suffix("").parts)
        for path in package.rglob("*.py")
        if any(
            _unguarded_monarch_imports(ast.parse(path.read_text(encoding="utf-8")).body)
        )
    }

    assert found == _IMPORT_MONARCH_ON_LOAD


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
        "math",
        "typing",
    }
    annotations = {"._model", "torchstore.strategy"}
    assert _imported(tree.body) == on_load
    assert _imported(type_checking) == annotations
    guarded = {id(node) for statement in type_checking for node in ast.walk(statement)}
    unguarded = (node for node in ast.walk(tree) if id(node) not in guarded)
    assert _imported(unguarded) == on_load
