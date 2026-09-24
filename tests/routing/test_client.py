# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio

import pytest
import torch

from .utils import registered_clients


def _publisher_and_requester_clients():
    return registered_clients(
        publishers={"publisher": {"w": torch.empty(4, dtype=torch.int64)}},
        requesters={"requester": {"w": torch.empty(4, dtype=torch.int64)}},
    )


def test_publisher_rejects_unknown_keys_and_wrong_shapes() -> None:
    """Reject tensors that do not match the publisher's installed plan."""
    publisher = _publisher_and_requester_clients()["publisher"]

    with pytest.raises(KeyError, match="does not store"):
        asyncio.run(publisher.put_batch({"other": torch.empty(4)}))
    with pytest.raises(ValueError, match="has shape .* expected"):
        asyncio.run(publisher.put_batch({"model/w": torch.empty(3)}))


def test_requester_rejects_a_wrong_destination_shape() -> None:
    """Reject an in-place buffer that cannot hold the planned requester slice."""
    requester = _publisher_and_requester_clients()["requester"]

    with pytest.raises(ValueError, match="destination .* has shape .* expected"):
        asyncio.run(requester.get("model/w", torch.empty(3)))


@pytest.mark.parametrize("operation", ["keys", "delete", "delete_batch", "exists"])
def test_unsupported_operations_fail_explicitly(operation: str) -> None:
    """Expose unsupported mutable-store operations consistently."""
    client = _publisher_and_requester_clients()["requester"]
    arguments = {
        "keys": (),
        "delete": ("key",),
        "delete_batch": (["key"],),
        "exists": ("key",),
    }

    with pytest.raises(NotImplementedError, match=f"{operation} is unsupported"):
        asyncio.run(getattr(client, operation)(*arguments[operation]))


def test_register_state_dict_exchanges_metadata_and_installs_local_plan() -> None:
    """Send local metadata to the coordinator and install the returned plan."""
    # Register matching publisher and requester state dicts through the public
    # client API, then verify the requester received its planned route.
    requester = _publisher_and_requester_clients()["requester"]

    assert requester._controller.routes.lookup("model/w")
