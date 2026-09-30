# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pytest
import torch

from .utils import registered_clients


async def _publisher_and_requester_clients():
    return await registered_clients(
        publishers={"publisher/0": {"w": torch.empty(4, dtype=torch.int64)}},
        requesters={"requester/0/0": {"w": torch.empty(4, dtype=torch.int64)}},
    )


@pytest.mark.asyncio
async def test_publisher_rejects_unknown_keys_and_wrong_shapes() -> None:
    """Reject tensors that do not match the publisher's installed plan."""
    publisher = (await _publisher_and_requester_clients())["publisher/0"]

    with pytest.raises(KeyError, match="does not store"):
        await publisher.put_batch({"other": torch.empty(4)})
    with pytest.raises(ValueError, match="has shape .* expected"):
        await publisher.put_batch({"model/w": torch.empty(3)})


@pytest.mark.asyncio
async def test_requester_rejects_a_wrong_destination_shape() -> None:
    """Reject an in-place buffer that cannot hold the planned requester slice."""
    requester = (await _publisher_and_requester_clients())["requester/0/0"]

    with pytest.raises(ValueError, match="destination .* has shape .* expected"):
        await requester.get("model/w", torch.empty(3, dtype=torch.int64))


@pytest.mark.asyncio
async def test_requester_rejects_an_unplanned_tensor_key() -> None:
    """Do not misclassify a tensor missing from the route table as an object."""
    requester = (await _publisher_and_requester_clients())["requester/0/0"]

    with pytest.raises(KeyError, match="has no tensor route"):
        await requester.get("model/other")


@pytest.mark.parametrize("operation", ["keys", "delete", "delete_batch", "exists"])
@pytest.mark.asyncio
async def test_unsupported_operations_fail_explicitly(operation: str) -> None:
    """Expose unsupported mutable-store operations consistently."""
    client = (await _publisher_and_requester_clients())["requester/0/0"]
    arguments = {
        "keys": (),
        "delete": ("key",),
        "delete_batch": (["key"],),
        "exists": ("key",),
    }

    with pytest.raises(NotImplementedError, match=f"{operation} is unsupported"):
        await getattr(client, operation)(*arguments[operation])


@pytest.mark.asyncio
async def test_registration_methods_install_local_plans() -> None:
    """Send local metadata to the coordinator and install the returned plan."""
    requester = (await _publisher_and_requester_clients())["requester/0/0"]

    assert requester._routing_directory.routes.lookup("model/w")
