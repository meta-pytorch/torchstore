# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import ast
import importlib
import logging
from pathlib import Path

import pytest
import torchstore

from .test_control_plane import _imported

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


@pytest.mark.parametrize("old", sorted(_OLD_PATHS))
def test_old_module_paths_reexport_the_moved_names(old: str) -> None:
    """Bind each name an old module defined at its old path, as the same object."""
    module = importlib.import_module(old)
    assert sorted(module.__all__) == sorted(_OLD_PATHS[old])
    for name, home in {**_OLD_PATHS[old], **_OLD_PRIVATE.get(old, {})}.items():
        assert getattr(module, name) is getattr(importlib.import_module(home), name)


@pytest.mark.parametrize(
    ("home", "old"),
    [
        ("torchstore.monarch.coordinator", "torchstore.routing.coordinator"),
        ("torchstore.monarch.storage_volume", "torchstore.storage_volume"),
    ],
)
def test_moved_modules_log_under_their_old_names(home: str, old: str) -> None:
    """Keep the moved coordinator's and volume's loggers under their old module names."""
    assert importlib.import_module(home).logger is logging.getLogger(old)
