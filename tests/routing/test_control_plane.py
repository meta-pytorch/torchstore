# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import inspect

from torchstore.routing.control_plane import ControlPlane

# The public interface's signatures, frozen from the design's ControlPlane sketch.
_DESIGN = {
    "init": "(self, publishers: 'Collection[str]', strategy: 'TorchStoreStrategy') -> 'None'",
    "register_layout": "(self, rank: 'str', key: 'str', registrations: 'Mapping[str, KeyRegistration]') -> 'None'",
    "get_layouts": "(self, key: 'str') -> 'Registrations'",
    "strategy": "(self) -> 'TorchStoreStrategy'",
}


def test_interface_signatures_match_the_design() -> None:
    """Keep the design doc's four async methods, with exactly its signatures."""
    assert ControlPlane.__abstractmethods__ == set(_DESIGN)
    public = {name for name in dir(ControlPlane) if not name.startswith("_")}
    assert public == set(_DESIGN)
    for name, signature in _DESIGN.items():
        method = getattr(ControlPlane, name)
        assert inspect.iscoroutinefunction(method), name
        assert str(inspect.signature(method)) == signature, name
