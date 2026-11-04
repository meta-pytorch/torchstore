# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import ast
from pathlib import Path

import torchstore

from .test_control_plane import _imported


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
