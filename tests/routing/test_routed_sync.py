# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import uuid

import monarch.actor as monarch_actor
import pytest
import torch
import torchstore as ts
from monarch.actor import Actor, concurrent_endpoint, current_rank, this_host
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard
from torchstore.transport import TransportType

_KEY = "model_state_dict"
_ROUNDS = 3
_PUBLISHER_RANKS = 2
# Shard dim of each entry on a mesh; None means every rank holds the whole tensor.
# Column shards of out.weight are strided in every requester's buffer, so reads
# copy them in rather than writing in place.
_PUBLISHER_DIMS = {"proj.weight": 0, "norm.weight": None, "out.weight": 1}
# (number of ranks, shard dims) of each requester mesh, indexed by its group.
_REQUESTERS = (
    (1, {"proj.weight": None, "norm.weight": None, "out.weight": None}),
    (2, {"proj.weight": 1, "norm.weight": 0, "out.weight": 0}),
)
_DESTINATION_ONLY = "requester_only.weight"
_STILL_WAITING_S = 1.0
# A hung step fails at its own line, before the timeout marker fires.
_STEP_TIMEOUT_S = 60.0


def _weights(round_: int) -> dict[str, torch.Tensor]:
    """Whole tensors published in one round; every value changes each round."""
    proj = torch.arange(24, dtype=torch.float32).view(4, 6) / 7 + round_
    return {
        "proj.weight": proj.to(torch.bfloat16),
        "norm.weight": torch.arange(6, dtype=torch.float32) / 3 - round_,
        "out.weight": torch.arange(24, dtype=torch.float32).view(4, 6) / 5 + 2 * round_,
    }


def _sentinel() -> torch.Tensor:
    return torch.full((3,), -1.0)


def _local(
    tensor: torch.Tensor, dim: int | None, rank: int, num_ranks: int
) -> torch.Tensor:
    """The part of a whole tensor that one rank holds."""
    return tensor if dim is None else tensor.chunk(num_ranks, dim)[rank]


def _place(
    tensors: dict[str, torch.Tensor],
    dims: dict[str, int | None],
    rank: int,
    num_ranks: int,
) -> dict[str, torch.Tensor]:
    """Hold each tensor as one rank of a mesh would."""
    # Fake meshes give real global offsets without a process group.
    mesh = DeviceMesh("cpu", torch.arange(num_ranks), _init_backend=False, _rank=rank)
    placed = {}
    for name, tensor in tensors.items():
        local = _local(tensor, dims[name], rank, num_ranks).clone()
        if dims[name] is not None:
            local = DTensor.from_local(
                local,
                mesh,
                [Shard(dims[name])],
                run_check=False,
                shape=tensor.shape,
                stride=tensor.stride(),
            )
        placed[name] = local
    return placed


def _assert_bitwise_equal(
    actual: torch.Tensor, expected: torch.Tensor, label: str
) -> None:
    assert (actual.dtype, actual.shape) == (expected.dtype, expected.shape), label
    assert torch.equal(
        actual.contiguous().view(torch.uint8),
        expected.contiguous().view(torch.uint8),
    ), f"{label}: {actual} != {expected}"


class _Publisher(Actor):
    """A trainer rank that publishes the way TitanRL's trainer does."""

    def __init__(self, store_name: str) -> None:
        self._store_name = store_name
        self._rank = current_rank().rank

    @concurrent_endpoint
    async def initialize_client(self) -> None:
        await ts.client(self._store_name, role=ts.RankRole.PUBLISHER)

    @concurrent_endpoint
    async def push(self, round_: int) -> None:
        state_dict = _place(
            _weights(round_), _PUBLISHER_DIMS, self._rank, _PUBLISHER_RANKS
        )
        # A worker thread with a fresh event loop per push, as in TitanRL's trainer.
        await asyncio.to_thread(
            lambda: asyncio.run(
                ts.put_state_dict(state_dict, _KEY, store_name=self._store_name)
            )
        )


class _Requester(Actor):
    """A generator rank that reads into the same buffers every round."""

    def __init__(self, store_name: str, group: int) -> None:
        num_ranks, dims = _REQUESTERS[group]
        zeros = {name: torch.zeros_like(t) for name, t in _weights(0).items()}
        self._store_name = store_name
        self._group = group
        self._destination = {
            **_place(zeros, dims, current_rank().rank, num_ranks),
            _DESTINATION_ONLY: _sentinel(),
        }
        self._exchange_started = asyncio.Event()

    @concurrent_endpoint
    async def initialize_client(self) -> None:
        await ts.client(self._store_name, role=ts.RankRole.REQUESTER, group=self._group)

    @concurrent_endpoint
    async def wait_for_layouts(self) -> None:
        # Signal even when the lookup fails, so the driver reports its error at once.
        try:
            client = await ts.client(self._store_name)
        finally:
            self._exchange_started.set()
        await client.get_layouts(self._destination, _KEY, strict=False)

    @concurrent_endpoint
    async def exchange_started(self) -> None:
        await self._exchange_started.wait()

    @concurrent_endpoint
    async def pull(self) -> tuple[list[str], dict[str, torch.Tensor]]:
        result = await ts.get_state_dict(
            _KEY, self._destination, strict=False, store_name=self._store_name
        )
        local = {
            name: tensor.to_local() if isinstance(tensor, DTensor) else tensor
            for name, tensor in self._destination.items()
        }
        return sorted(result), local


@pytest.mark.timeout(120)
@pytest.mark.asyncio
async def test_routed_weight_sync_across_actor_meshes() -> None:
    """Sync threaded publishers into two differently sharded requester meshes."""
    # Monarch's default fault hook interrupts the whole session; recording faults
    # lets a dead or stalled peer fail its step instead. A failed run keeps this
    # hook, since its stopped meshes can still report faults after it ends.
    faults: list[monarch_actor.MeshFailure] = []
    default_fault_hook = monarch_actor.unhandled_fault_hook
    monarch_actor.unhandled_fault_hook = faults.append
    store_name = f"routed-sync-test-{uuid.uuid4()}"
    publisher_mesh = this_host().spawn_procs(per_host={"procs": _PUBLISHER_RANKS})
    requester_meshes = [
        this_host().spawn_procs(per_host={"procs": num_ranks})
        for num_ranks, _ in _REQUESTERS
    ]
    layouts = None
    try:
        await asyncio.wait_for(
            ts.initialize(
                mesh=publisher_mesh,
                strategy=ts.TorchStoreStrategy(TransportType.MonarchRPC),
                client_type=ts.ClientType.ROUTING,
                store_name=store_name,
            ),
            _STEP_TIMEOUT_S,
        )
        publishers = publisher_mesh.spawn("publisher", _Publisher, store_name)
        requesters = [
            mesh.spawn(f"requester_{group}", _Requester, store_name, group)
            for group, mesh in enumerate(requester_meshes)
        ]
        # Let every call settle first: stopping a mesh mid-lookup can break Monarch's
        # controller lookups for the rest of the session.
        results = await asyncio.wait_for(
            asyncio.gather(
                publishers.initialize_client.call(),
                *(requester.initialize_client.call() for requester in requesters),
                return_exceptions=True,
            ),
            _STEP_TIMEOUT_S,
        )
        for result in results:
            if isinstance(result, BaseException):
                raise result

        # Group 1 starts the layout exchange, a routed read's first step, before any
        # publisher pushes; group 0 runs it inside its first read, as TitanRL does.
        # Data reads follow the push: a layout registers before its tensors land, so
        # an earlier read can miss the mapping.
        layouts = asyncio.ensure_future(requesters[1].wait_for_layouts.call())
        await asyncio.wait_for(requesters[1].exchange_started.call(), _STEP_TIMEOUT_S)
        done, _ = await asyncio.wait({layouts}, timeout=_STILL_WAITING_S)
        assert not done, f"requesters did not wait: {layouts.exception()!r}"

        for round_ in range(_ROUNDS):
            await asyncio.wait_for(publishers.push.call(round_), _STEP_TIMEOUT_S)
            if round_ == 0:
                await asyncio.wait_for(layouts, _STEP_TIMEOUT_S)
            pulled = await asyncio.wait_for(
                asyncio.gather(*(r.pull.call() for r in requesters)), _STEP_TIMEOUT_S
            )
            expected = _weights(round_)
            for group, values in enumerate(pulled):
                num_ranks, dims = _REQUESTERS[group]
                for point, (keys, local) in values.items():
                    label = f"round {round_}, requester/{group}/{point.rank}"
                    assert keys == sorted(expected), label
                    for name, tensor in expected.items():
                        _assert_bitwise_equal(
                            local[name],
                            _local(tensor, dims[name], point.rank, num_ranks),
                            f"{label}, {name}",
                        )
                    assert torch.equal(local[_DESTINATION_ONLY], _sentinel()), label
    finally:
        if layouts is not None:
            # Cancel and await the exchange, or its later failure is logged at exit.
            layouts.cancel()
            await asyncio.gather(layouts, return_exceptions=True)
        await asyncio.gather(
            *(mesh.stop() for mesh in (*requester_meshes, publisher_mesh))
        )
    assert not faults, [fault.report() for fault in faults]
    monarch_actor.unhandled_fault_hook = default_fault_hook
