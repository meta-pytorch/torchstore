# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The old module paths that load Monarch re-export the moved names and keep their loggers."""

import importlib
import logging

import pytest

from .test_import_scope import _assert_reexports, _OLD_PATHS_NEEDING_MONARCH

pytest.importorskip("monarch")


@pytest.mark.parametrize("old", sorted(_OLD_PATHS_NEEDING_MONARCH))
def test_old_module_paths_reexport_the_moved_names(old: str) -> None:
    """Bind each name an old module defined at its old path, as the same object."""
    _assert_reexports(old)


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
