# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pytest

from torchstore.routing._model import KeyRegistration
from torchstore.routing.plan import RoutingPlan

from .utils import tensor_slice


def test_collapses_replicated_publishers_to_one_deterministic_source() -> None:
    """Choose one stable source when multiple publishers hold identical data."""
    # Build the same plan twice to verify that source selection is reproducible.
    whole = tensor_slice((0,), (8,), global_shape=(8,))
    entry = {"weight": KeyRegistration(whole, element_size=4)}
    publishers = {"publisher/0": entry, "publisher/1": entry}
    requesters = {"requester": entry}

    transfers = []
    for _ in range(2):
        plan = RoutingPlan.build_for("requester", publishers, requesters)
        [transfer] = plan.lookup("requester", "weight")[0].transfers
        transfers.append(transfer)

    assert transfers[0] == transfers[1]
    assert transfers[0].source in publishers
    assert transfers[0].nbytes == 32


def test_ignores_requester_keys_that_no_publisher_has() -> None:
    """Leave requester-only keys out of the routed local table."""
    whole = tensor_slice((0,), (8,), global_shape=(8,))
    plan = RoutingPlan.build_for(
        "requester",
        {"publisher": {"published": KeyRegistration(whole, element_size=4)}},
        {"requester": {"local-only": KeyRegistration(whole, element_size=4)}},
    )

    assert plan._local("requester").keys == {}


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
    with pytest.raises(KeyError, match="ranks did not register a layout"):
        RoutingPlan.build_for("missing", {"publisher": {}}, {"requester": {}})


def test_rejects_inconsistent_global_shapes() -> None:
    """Reject publisher and requester views of different global tensors."""
    publisher = tensor_slice((0,), (8,), global_shape=(8,))
    requester = tensor_slice((0,), (8,), global_shape=(9,))

    with pytest.raises(ValueError, match="inconsistent global shape"):
        RoutingPlan.build_for(
            "requester",
            {"publisher": {"w": KeyRegistration(publisher, element_size=4)}},
            {"requester": {"w": KeyRegistration(requester, element_size=4)}},
        )


def test_rejects_inconsistent_or_invalid_element_sizes() -> None:
    """Require one positive wire element size for each storage key."""
    whole = tensor_slice((0,), (8,), global_shape=(8,))
    with pytest.raises(ValueError, match="inconsistent element size"):
        RoutingPlan.build_for(
            "requester",
            {"publisher": {"w": KeyRegistration(whole, element_size=2)}},
            {"requester": {"w": KeyRegistration(whole, element_size=4)}},
        )

    with pytest.raises(ValueError, match="must be positive"):
        RoutingPlan.build_for(
            "requester",
            {"publisher": {"w": KeyRegistration(whole, element_size=0)}},
            {"requester": {"w": KeyRegistration(whole, element_size=0)}},
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
                tensor_slice(offset, shape, global_shape=(4,)), element_size=4
            )
        }
        for index, (offset, shape) in enumerate(publisher_slices)
    }

    with pytest.raises(ValueError, match="does not cover .* exactly"):
        RoutingPlan.build_for(
            "requester",
            publishers,
            {"requester": {"w": KeyRegistration(whole, element_size=4)}},
        )


def test_accepts_a_zero_sized_requester_shard() -> None:
    """Plan an empty destination without manufacturing a data transfer."""
    published = tensor_slice((0,), (2,), global_shape=(2,))
    empty = tensor_slice((2,), (0,), global_shape=(2,))
    plan = RoutingPlan.build_for(
        "requester",
        {"publisher": {"w": KeyRegistration(published, element_size=4)}},
        {"requester": {"w": KeyRegistration(empty, element_size=4)}},
    )

    [route] = plan.lookup("requester", "w")
    assert route.destination_slice is empty
    assert route.transfers == ()
