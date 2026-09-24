# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard

from torchstore.state_dict_utils import get_state_dict, put_state_dict

from .utils import registered_clients, transport_factory


def test_multi_source_fetch_assembles_into_allocated_and_inplace_tensors(
    monkeypatch,
) -> None:
    """Assemble two publisher shards for allocated and caller-owned outputs."""
    factory = transport_factory(monkeypatch)

    # 1. Represent each publisher's local rows as a DTensor shard. Fake meshes
    # avoid process-group setup while retaining real global-offset metadata.
    local_tensors = [torch.arange(8).view(2, 4), torch.arange(8, 16).view(2, 4)]
    publisher_tensors = [
        DTensor.from_local(
            local_tensor,
            DeviceMesh("cpu", torch.arange(2), _init_backend=False, _rank=rank),
            [Shard(0)],
            run_check=False,
            shape=torch.Size((4, 4)),
            stride=(4, 1),
        )
        for rank, local_tensor in enumerate(local_tensors)
    ]
    destination = torch.full((4, 4), -1)
    clients = registered_clients(
        publishers={
            "publisher/0": {"w": publisher_tensors[0]},
            "publisher/1": {"w": publisher_tensors[1]},
        },
        requesters={"requester": {"w": destination}},
    )

    # 2. Register the inferred layouts, publish both shards, then exercise
    # allocating and in-place reads.
    async def run():
        await clients["publisher/0"].put_batch({"model/w": publisher_tensors[0]})
        await clients["publisher/1"].put_batch({"model/w": publisher_tensors[1]})
        allocated = await clients["requester"].get("model/w")
        inplace = await clients["requester"].get("model/w", destination)
        return allocated, destination, inplace

    allocated, destination, inplace = asyncio.run(run())

    # 3. Verify assembly, identity preservation, and both source-volume reads.
    expected = torch.arange(16).view(4, 4)
    torch.testing.assert_close(allocated, expected)
    torch.testing.assert_close(destination, expected)
    assert inplace is destination
    assert {
        volume
        for volume, transport in factory.transports.items()
        if transport.get_from_storage_volume.await_count
    } == {"publisher/0", "publisher/1"}


def test_split_publisher_state_dict(monkeypatch) -> None:
    """Rebuild one nested state dict whose tensors live on two publishers."""
    factory = transport_factory(monkeypatch)

    # 1. Build routes where each publisher owns one nested tensor and the
    # requester expects both tensors.
    first_tensor = torch.arange(4)
    second_tensor = torch.arange(10, 14)
    destination = {
        "stage": {
            "first": torch.zeros_like(first_tensor),
            "second": torch.zeros_like(second_tensor),
        }
    }
    clients = registered_clients(
        publishers={
            "publisher/0": {"stage": {"first": first_tensor}},
            "publisher/1": {"stage": {"second": second_tensor}},
        },
        requesters={"requester": destination},
    )

    # 2. Publish each state-dict fragment, then fetch their merged state dict
    # into the requester's preallocated tensors.
    async def run():
        await put_state_dict(
            clients["publisher/0"], {"stage": {"first": first_tensor}}, "model"
        )
        await put_state_dict(
            clients["publisher/1"], {"stage": {"second": second_tensor}}, "model"
        )
        return await get_state_dict(clients["requester"], "model", destination)

    result = asyncio.run(run())

    # 3. Verify the tensors were updated in place with values from both
    # publishers and that both mapping fragments were fetched.
    assert result["stage"]["first"] is destination["stage"]["first"]
    assert result["stage"]["second"] is destination["stage"]["second"]
    torch.testing.assert_close(destination["stage"]["first"], first_tensor)
    torch.testing.assert_close(destination["stage"]["second"], second_tensor)
    mapping_reads = {
        volume
        for volume, transport in factory.transports.items()
        if any(
            tuple(request.key for request in call.args[0]) == ("model/MAPPING",)
            for call in transport.get_from_storage_volume.await_args_list
        )
    }
    assert mapping_reads == {"publisher/0", "publisher/1"}
