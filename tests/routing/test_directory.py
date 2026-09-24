# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from torchstore.routing._model import KeyRegistration
from torchstore.routing.directory import RoutingDirectory
from torchstore.routing.plan import RoutingPlan
from torchstore.transport.types import Request

from .utils import tensor_slice


def _routing_plan(namespace: str = "model") -> RoutingPlan:
    whole = tensor_slice((0,), (4,), global_shape=(4,))
    key = f"{namespace}/w"
    publishers = {"publisher": {key: KeyRegistration(whole, element_size=4)}}
    requesters = {"requester": {key: KeyRegistration(whole, element_size=4)}}
    return RoutingPlan(
        {
            rank: RoutingPlan.build_for(rank, publishers, requesters)._local(rank)
            for rank in ("publisher", "requester")
        }
    )


def test_routes_are_unavailable_until_a_plan_is_installed() -> None:
    """Require registration before exposing rank-local routing metadata."""
    directory = RoutingDirectory("requester")

    with pytest.raises(RuntimeError, match="register a state dict first"):
        _ = directory.routes


def test_install_merges_state_dict_namespaces() -> None:
    """Merge independently registered namespaces into one local table."""
    directory = RoutingDirectory("requester")
    directory.install(_routing_plan("model"))
    directory.install(_routing_plan("optimizer"))

    assert set(directory.routes.keys) == {"model/w", "optimizer/w"}


def test_install_rejects_a_changed_volume() -> None:
    """Keep a rank's storage volume stable across installations."""
    directory = RoutingDirectory("requester")
    original = _routing_plan()
    directory.install(original)
    changed_table = replace(
        original._local("requester"),
        volume_id="other-volume",
    )
    changed = RoutingPlan({"requester": changed_table})

    with pytest.raises(ValueError, match="cannot change volume or role"):
        directory.install(changed)


def test_resolve_get_batch_validates_request_metadata() -> None:
    """Reject mismatched slices and requests that still carry tensor data."""
    directory = RoutingDirectory("requester")
    directory.install(_routing_plan())
    wrong = tensor_slice((1,), (3,), global_shape=(4,))

    with pytest.raises(ValueError, match="does not match the plan"):
        directory.resolve_get_batch([Request(key="model/w", tensor_slice=wrong)])
    with pytest.raises(ValueError, match="metadata-only"):
        directory.resolve_get_batch([Request(key="model/w", tensor_val=torch.empty(4))])


def test_only_requesters_can_resolve_gets() -> None:
    """Prevent publisher-local tables from executing requester reads."""
    directory = RoutingDirectory("publisher")
    directory.install(_routing_plan())

    with pytest.raises(RuntimeError, match="only requester clients"):
        directory.resolve_get_batch([Request(key="model/w")])
