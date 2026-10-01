# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pytest
import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard
from torchstore.state_dict_utils import get_state_dict, put_state_dict

from .utils import registered_clients, routing_clients


@pytest.mark.asyncio
async def test_multi_source_fetch_assembles_into_allocated_and_inplace_tensors() -> (
    None
):
    """Assemble two publisher shards for allocated and caller-owned outputs."""
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
    clients = await registered_clients(
        publishers={
            "publisher/0": {"w": publisher_tensors[0]},
            "publisher/1": {"w": publisher_tensors[1]},
        },
        requesters={"requester/0/0": {"w": destination}},
    )

    # 2. Register the inferred layouts, publish both shards, then exercise
    # allocating and in-place reads.
    await clients["publisher/0"].put_batch({"model/w": publisher_tensors[0]})
    await clients["publisher/1"].put_batch({"model/w": publisher_tensors[1]})
    allocated = await clients["requester/0/0"].get("model/w")
    inplace = await clients["requester/0/0"].get("model/w", destination)

    # 3. Verify assembly, identity preservation, and both source-volume reads.
    expected = torch.arange(16).view(4, 4)
    torch.testing.assert_close(allocated, expected)
    torch.testing.assert_close(destination, expected)
    assert inplace is destination


@pytest.mark.asyncio
async def test_split_publisher_state_dict() -> None:
    """Rebuild one nested state dict whose tensors live on two publishers."""
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
    clients = await routing_clients(
        publishers={
            "publisher/0": {"stage": {"first": first_tensor}},
            "publisher/1": {"stage": {"second": second_tensor}},
        },
        requesters={"requester/0/0": destination},
    )

    # 2. Lazily register layouts while publishing each fragment, then lazily
    # register the requester and fetch into its preallocated tensors.
    await put_state_dict(
        clients["publisher/0"], {"stage": {"first": first_tensor}}, "model"
    )
    await put_state_dict(
        clients["publisher/1"], {"stage": {"second": second_tensor}}, "model"
    )
    result = await get_state_dict(clients["requester/0/0"], "model", destination)

    # 3. Verify the tensors were updated in place with values from both
    # publishers and that both mapping fragments were fetched.
    assert result["stage"]["first"] is destination["stage"]["first"]
    assert result["stage"]["second"] is destination["stage"]["second"]
    torch.testing.assert_close(destination["stage"]["first"], first_tensor)
    torch.testing.assert_close(destination["stage"]["second"], second_tensor)


@pytest.mark.asyncio
async def test_identical_mapping_entries_from_dtensor_shards_are_merged() -> None:
    """Accept the same FQN-to-path entry from every shard of one DTensor."""
    local_tensors = [torch.arange(2), torch.arange(2, 4)]
    publisher_tensors = [
        DTensor.from_local(
            local_tensor,
            DeviceMesh("cpu", torch.arange(2), _init_backend=False, _rank=rank),
            [Shard(0)],
            run_check=False,
            shape=torch.Size((4,)),
            stride=(1,),
        )
        for rank, local_tensor in enumerate(local_tensors)
    ]
    destination = {"w": torch.full((4,), -1)}
    clients = await routing_clients(
        publishers={
            "publisher/0": {"w": publisher_tensors[0]},
            "publisher/1": {"w": publisher_tensors[1]},
        },
        requesters={"requester/0/0": destination},
    )

    await put_state_dict(clients["publisher/0"], {"w": publisher_tensors[0]}, "model")
    await put_state_dict(clients["publisher/1"], {"w": publisher_tensors[1]}, "model")
    result = await get_state_dict(clients["requester/0/0"], "model", destination)

    assert result["w"] is destination["w"]
    torch.testing.assert_close(result["w"], torch.arange(4))


@pytest.mark.asyncio
async def test_conflicting_mapping_entries_are_rejected() -> None:
    """Reject one flattened FQN having different nested paths across publishers."""
    first = torch.arange(2)
    second = torch.arange(2, 4)
    clients = await routing_clients(
        publishers={
            "publisher/0": {"first": first},
            "publisher/1": {"second": second},
        },
        requesters={
            "requester/0/0": {
                "first": torch.zeros_like(first),
                "second": torch.zeros_like(second),
            }
        },
    )

    await put_state_dict(clients["publisher/0"], {"first": first}, "model")
    await put_state_dict(clients["publisher/1"], {"second": second}, "model")
    await clients["publisher/1"].put(
        "model/MAPPING",
        {"first": ("different",), "second": ("second",)},
    )
    with pytest.raises(RuntimeError, match="Mapping is missing") as error:
        await get_state_dict(
            clients["requester/0/0"],
            "model",
            {"first": torch.zeros_like(first), "second": torch.zeros_like(second)},
            strict=False,
        )
    assert isinstance(error.value.__cause__, ValueError)
    assert "conflicting state-dict mapping" in str(error.value.__cause__)


@pytest.mark.asyncio
async def test_source_only_mapping_entry_is_not_requested() -> None:
    """A non-strict routed read fetches only the registered key intersection."""
    first = torch.arange(2)
    extra = torch.arange(2, 4)
    clients = await routing_clients(
        publishers={"publisher/0": {"first": first, "extra": extra}},
        requesters={"requester/0/0": {"first": torch.zeros_like(first)}},
    )

    await put_state_dict(
        clients["publisher/0"], {"first": first, "extra": extra}, "model"
    )
    result = await get_state_dict(
        clients["requester/0/0"],
        "model",
        {"first": torch.zeros_like(first)},
        strict=False,
    )

    assert set(result) == {"first"}
    torch.testing.assert_close(result["first"], first)
