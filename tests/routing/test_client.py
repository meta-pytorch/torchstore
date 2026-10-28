# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pytest
import torch
from torchstore.direct_weight_sync import RDMAWeightHandle

from .utils import registered_clients, routing_clients, tensor_slice


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
        await requester.get("model/w", torch.empty(3))


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


@pytest.mark.asyncio
async def test_rdma_handles_are_grouped_by_publisher() -> None:
    """Return every publisher's handles per key, whatever order they publish in."""
    publishers = ("publisher/0", "publisher/1")
    clients = await routing_clients(
        publishers={rank: {} for rank in publishers},
        requesters={"requester/0/0": {}},
    )
    handles = {
        rank: {
            "w": RDMAWeightHandle(
                rdma_buffer=None,
                tensor_slice=tensor_slice((2 * index,), (2,), global_shape=(4,)),
                source_rank=index,
            )
        }
        for index, rank in enumerate(publishers)
    }
    for rank in reversed(publishers):
        await clients[rank].publish_rdma_handles("model", handles[rank])

    grouped = await clients["requester/0/0"].get_rdma_handles("model")

    assert dict(grouped) == {"w": [handles[rank]["w"] for rank in publishers]}
    with pytest.raises(RuntimeError, match="only publisher"):
        await clients["requester/0/0"].publish_rdma_handles("model", {})
    with pytest.raises(RuntimeError, match="only requester"):
        await clients["publisher/0"].get_rdma_handles("model")
