# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard
from torchstore.state_dict_utils import get_state_dict, put_state_dict

from .utils import registered_clients, routing_clients


def shard(tensor: torch.Tensor, dim: int, rank: int, num_ranks: int) -> DTensor:
    """Hold one rank's chunk of ``tensor`` along ``dim`` as a DTensor shard.

    Fake meshes give real global offsets without a process group.
    """
    mesh = DeviceMesh("cpu", torch.arange(num_ranks), _init_backend=False, _rank=rank)
    return DTensor.from_local(
        tensor.chunk(num_ranks, dim)[rank].clone(),
        mesh,
        [Shard(dim)],
        run_check=False,
        shape=tensor.shape,
        stride=tensor.stride(),
    )


@pytest.mark.asyncio
async def test_multi_source_fetch_assembles_into_allocated_and_inplace_tensors() -> (
    None
):
    """Assemble two publisher shards for allocated and caller-owned outputs."""
    # 1. Represent each publisher's local rows as a DTensor shard.
    publisher_tensors = [
        shard(torch.arange(16).view(4, 4), 0, rank, 2) for rank in range(2)
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
    publisher_tensors = [shard(torch.arange(4), 0, rank, 2) for rank in range(2)]
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


@pytest.mark.asyncio
async def test_destination_only_entry_is_ignored_by_non_strict_read() -> None:
    """Keep requester-local tensors that have no corresponding publisher key."""
    source = torch.arange(2)
    destination = {
        "weight": torch.zeros_like(source),
        "local_only": torch.full((2,), -1),
    }
    clients = await routing_clients(
        publishers={"publisher/0": {"weight": source}},
        requesters={"requester/0/0": destination},
    )

    await put_state_dict(clients["publisher/0"], {"weight": source}, "model")
    result = await get_state_dict(
        clients["requester/0/0"],
        "model",
        destination,
        strict=False,
    )

    assert set(result) == {"weight"}
    torch.testing.assert_close(result["weight"], source)
    torch.testing.assert_close(destination["local_only"], torch.full((2,), -1))


@pytest.mark.asyncio
async def test_transfer_dtype_preserves_keys() -> None:
    """Publish weights in transfer_dtype while preserved keys keep their dtype."""
    source = {"weight": torch.full((4,), 1.5), "buffer": torch.full((4,), 1 / 3)}
    destination = {
        "weight": torch.zeros(4, dtype=torch.bfloat16),
        "buffer": torch.zeros(4),
    }
    clients = await routing_clients(
        publishers={"publisher/0": source},
        requesters={"requester/0/0": destination},
    )

    await put_state_dict(
        clients["publisher/0"],
        source,
        "model",
        transfer_dtype=torch.bfloat16,
        preserve_dtype_keys=frozenset({"buffer"}),
    )
    result = await get_state_dict(clients["requester/0/0"], "model", destination)

    torch.testing.assert_close(result["weight"], source["weight"].bfloat16())
    torch.testing.assert_close(result["buffer"], source["buffer"])


@pytest.mark.asyncio
async def test_strict_read_rejects_key_skipped_by_non_strict_read() -> None:
    source = torch.arange(2)
    destination = {"weight": torch.zeros_like(source), "local_only": torch.zeros(2)}
    clients = await routing_clients(
        publishers={"publisher/0": {"weight": source}},
        requesters={"requester/0/0": destination},
    )
    await put_state_dict(clients["publisher/0"], {"weight": source}, "model")
    requester = clients["requester/0/0"]
    await get_state_dict(requester, "model", destination, strict=False)

    with pytest.raises(KeyError, match="no publisher"):
        await get_state_dict(
            requester, "model", {"local_only": destination["local_only"]}
        )


@pytest.mark.asyncio
async def test_non_strict_reads_plan_and_fetch_one_subset_per_call() -> None:
    """Read one subset of the state dict per call, as a per-layer load does."""
    sources = {"first": torch.arange(2), "second": torch.arange(2, 4)}
    sources["third"] = torch.arange(4, 6)
    clients = await routing_clients(
        publishers={
            "publisher/0": {"first": sources["first"]},
            "publisher/1": {"second": sources["second"], "third": sources["third"]},
        },
        requesters={"requester/0/0": {}},
    )
    await put_state_dict(clients["publisher/0"], {"first": sources["first"]}, "model")
    await put_state_dict(
        clients["publisher/1"],
        {"second": sources["second"], "third": sources["third"]},
        "model",
    )
    requester = clients["requester/0/0"]
    requester._control_plane = Mock(wraps=requester._control_plane)
    get_batch = Mock(wraps=requester.get_batch)
    requester.get_batch = get_batch

    # A failed plan leaves its keys to be planned by the next get.
    with pytest.raises(ValueError, match="inconsistent global shape"):
        await get_state_dict(
            requester, "model", {"first": torch.zeros(3)}, strict=False
        )
    # Subsets: new, new, already planned, mixed, then planned after a get with
    # no published keys.
    for names, local_only in (
        (["first"], False),
        (["second"], False),
        (["first"], False),
        (["second", "third"], False),
        ([], True),
        (["first", "third"], False),
    ):
        get_batch.reset_mock()
        destination = {name: torch.zeros_like(sources[name]) for name in names}
        if local_only:
            destination["local_only"] = torch.zeros(2)
        result = await get_state_dict(requester, "model", destination, strict=False)

        assert set(result) == set(names)
        for name in names:
            assert result[name] is destination[name]
            torch.testing.assert_close(result[name], sources[name])
        if names:
            ((batch,), _kwargs) = get_batch.call_args
            assert set(batch) == {f"model/{name}" for name in names}
        else:
            get_batch.assert_not_called()
    requester._control_plane.get_layouts.assert_called_once_with("model")


# Shard dim of each tensor on two publishers, or None for a whole copy on each.
# Column shards of out.weight are strided in every requester's buffer, so reads
# copy them in rather than writing in place.
_PUBLISHER_DIMS = {"proj.weight": 0, "norm.weight": None, "out.weight": 1}


def _weights(round_: int) -> dict[str, torch.Tensor]:
    """Whole tensors published in one round, with every value new each round."""
    proj = torch.arange(24, dtype=torch.float32).view(4, 6) / 7 + round_
    return {
        "proj.weight": proj.to(torch.bfloat16),
        "norm.weight": torch.arange(6, dtype=torch.float32) / 3 - round_,
        "out.weight": torch.arange(24, dtype=torch.float32).view(4, 6) / 5 + 2 * round_,
    }


def _place(
    tensors: dict[str, torch.Tensor],
    dims: dict[str, int | None],
    rank: int,
    num_ranks: int,
) -> dict[str, torch.Tensor]:
    """Hold each tensor as one rank of a mesh would."""
    return {
        name: (
            tensor.clone()
            if dims[name] is None
            else shard(tensor, dims[name], rank, num_ranks)
        )
        for name, tensor in tensors.items()
    }


def _locals(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The rank-local part of each tensor."""
    return {
        name: tensor.to_local() if isinstance(tensor, DTensor) else tensor
        for name, tensor in state_dict.items()
    }


@pytest.mark.parametrize(
    ("num_ranks", "dims"),
    [
        (1, {"proj.weight": None, "norm.weight": None, "out.weight": None}),
        (2, {"proj.weight": 1, "norm.weight": 0, "out.weight": 0}),
    ],
    ids=["whole", "resharded"],
)
@pytest.mark.asyncio
async def test_reads_refill_requester_buffers_every_round(
    num_ranks: int, dims: dict[str, int | None]
) -> None:
    """Refill caller-owned buffers each round, sharded unlike the publishers."""
    published = [
        {
            f"publisher/{rank}": _place(_weights(round_), _PUBLISHER_DIMS, rank, 2)
            for rank in range(2)
        }
        for round_ in range(2)
    ]
    zeros = {name: torch.zeros_like(tensor) for name, tensor in _weights(0).items()}
    destinations = {
        f"requester/0/{rank}": _place(zeros, dims, rank, num_ranks)
        for rank in range(num_ranks)
    }
    clients = await routing_clients(publishers=published[0], requesters=destinations)

    for round_, state_dicts in enumerate(published):
        for publisher, state_dict in state_dicts.items():
            await put_state_dict(clients[publisher], state_dict, "model")
        for requester, destination in destinations.items():
            await get_state_dict(clients[requester], "model", destination)
        torch.testing.assert_close(
            {name: _locals(destination) for name, destination in destinations.items()},
            {
                f"requester/0/{rank}": _locals(
                    _place(_weights(round_), dims, rank, num_ranks)
                )
                for rank in range(num_ranks)
            },
            rtol=0,
            atol=0,
        )
