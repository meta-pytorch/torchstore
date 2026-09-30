# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pytest
import torch

from torchstore.routing._model import KeyRegistration
from torchstore.routing.plan import RoutingPlan

from .utils import tensor_slice


def test_collapses_replicated_publishers_to_one_deterministic_source() -> None:
    """Choose one stable source when multiple publishers hold identical data."""
    # Build the same plan twice to verify that source selection is reproducible.
    whole = tensor_slice((0,), (8,), global_shape=(8,))
    entry = {"weight": KeyRegistration(whole, dtype=torch.float32)}
    publishers = {"publisher/0": entry, "publisher/1": entry}
    requesters = {"requester": entry}

    transfers = []
    for _ in range(2):
        plan = RoutingPlan.build_for("requester", publishers, requesters)
        route = plan.lookup("requester", "weight")
        assert route is not None
        [transfer] = route.transfers
        transfers.append(transfer)

    assert transfers[0] == transfers[1]
    assert transfers[0].source in publishers
    assert transfers[0].nbytes == 32


def test_rejects_requester_keys_that_no_publisher_has() -> None:
    """Require every requested key to have at least one publisher."""
    whole = tensor_slice((0,), (8,), global_shape=(8,))
    with pytest.raises(KeyError, match="requester keys have no publisher.*local-only"):
        RoutingPlan.build_for(
            "requester",
            {"publisher": {"published": KeyRegistration(whole, dtype=torch.float32)}},
            {"requester": {"local-only": KeyRegistration(whole, dtype=torch.float32)}},
        )


def test_publisher_plan_skips_requester_transfer_planning() -> None:
    """A publisher only needs metadata for the slice it serves."""
    publisher_slice = tensor_slice((0,), (2,), global_shape=(4,))
    requester_slice = tensor_slice((0,), (4,), global_shape=(4,))
    plan = RoutingPlan.build_for(
        "publisher",
        {"publisher": {"w": KeyRegistration(publisher_slice, dtype=torch.float32)}},
        {"requester": {"w": KeyRegistration(requester_slice, dtype=torch.float32)}},
    )

    assert plan._local("publisher").keys["w"].tensor_slice is publisher_slice
    assert plan.lookup("publisher", "w") is None


@pytest.mark.parametrize(
    ("publishers", "requesters", "message"),
    [
        ({}, {"requester": {}}, "at least one publisher"),
        ({"publisher": {}}, {}, "at least one requester"),
    ],
)
def test_requires_both_roles(publishers, requesters, message: str) -> None:
    """Require at least one publisher and requester before planning."""
    with pytest.raises(ValueError, match=message):
        RoutingPlan.build_for("requester", publishers, requesters)


def test_rejects_an_unregistered_target_rank() -> None:
    """Build local routes only for a rank that supplied a layout."""
    with pytest.raises(KeyError, match="rank did not register a layout"):
        RoutingPlan.build_for("missing", {"publisher": {}}, {"requester": {}})


def test_rejects_inconsistent_global_shapes() -> None:
    """Reject publisher and requester views of different global tensors."""
    publisher = tensor_slice((0,), (8,), global_shape=(8,))
    requester = tensor_slice((0,), (8,), global_shape=(9,))

    with pytest.raises(ValueError, match="inconsistent global shape"):
        RoutingPlan.build_for(
            "requester",
            {"publisher": {"w": KeyRegistration(publisher, dtype=torch.float32)}},
            {"requester": {"w": KeyRegistration(requester, dtype=torch.float32)}},
        )


def test_rejects_inconsistent_dtypes_with_the_same_element_size() -> None:
    """Do not accept equal-width wire dtypes as interchangeable."""
    whole = tensor_slice((0,), (8,), global_shape=(8,))

    with pytest.raises(ValueError, match="inconsistent dtype"):
        RoutingPlan.build_for(
            "requester",
            {"publisher": {"w": KeyRegistration(whole, dtype=torch.float16)}},
            {"requester": {"w": KeyRegistration(whole, dtype=torch.bfloat16)}},
        )


@pytest.mark.parametrize(
    "publisher_slices",
    [
        [((0,), (3,))],
        [((0,), (3,)), ((2,), (2,))],
    ],
    ids=("gap", "overlap"),
)
def test_rejects_inexact_publisher_coverage(publisher_slices) -> None:
    """Reject gaps and overlaps that could cause concurrent destination writes."""
    whole = tensor_slice((0,), (4,), global_shape=(4,))
    publishers = {
        f"publisher/{index}": {
            "w": KeyRegistration(
                tensor_slice(offset, shape, global_shape=(4,)),
                dtype=torch.float32,
            )
        }
        for index, (offset, shape) in enumerate(publisher_slices)
    }

    with pytest.raises(ValueError, match="does not cover .* exactly"):
        RoutingPlan.build_for(
            "requester",
            publishers,
            {"requester": {"w": KeyRegistration(whole, dtype=torch.float32)}},
        )


def test_accepts_a_zero_sized_requester_shard() -> None:
    """Plan an empty destination without manufacturing a data transfer."""
    published = tensor_slice((0,), (2,), global_shape=(2,))
    empty = tensor_slice((2,), (0,), global_shape=(2,))
    plan = RoutingPlan.build_for(
        "requester",
        {"publisher": {"w": KeyRegistration(published, dtype=torch.float32)}},
        {"requester": {"w": KeyRegistration(empty, dtype=torch.float32)}},
    )

    route = plan.lookup("requester", "w")
    assert route is not None
    assert route.destination_slice is empty
    assert route.transfers == ()
