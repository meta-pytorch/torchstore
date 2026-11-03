# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import socket
import uuid
from pathlib import Path

import monarch.actor as monarch_actor
import pytest
import torch
import torchstore as ts
import torchstore.api as store_api
from monarch.actor import Actor, concurrent_endpoint, current_rank, this_host
from torchstore.transport import TransportType

from .file_store_control_plane import FileStoreControlPlane

_KEY = "model_state_dict"
_PUBLISHER_RANKS = 2
_REQUESTER_RANKS = 2
# Each push gets a fresh loop, so client state tied to an earlier loop fails one.
_ROUNDS = 2
# One deadline covers the whole sync and leaves the marker's other 60 s for cleanup.
_DEADLINE_S = 60.0


def _layer(rank: int, round_: int) -> torch.Tensor:
    """The tensor that publisher ``rank`` pushes in one round."""
    return torch.arange(6, dtype=torch.float32) + 10 * rank + round_


def _weights(round_: int) -> dict[str, torch.Tensor]:
    """Every publisher's tensor in one round, with each value new every round."""
    return {
        f"layers.{rank}.weight": _layer(rank, round_)
        for rank in range(_PUBLISHER_RANKS)
    }


def _raise_first(results: list[object]) -> None:
    """Raise the first exception that a gather with ``return_exceptions`` returned."""
    for result in results:
        if isinstance(result, BaseException):
            raise result


def _control_plane(path: str | None) -> FileStoreControlPlane | None:
    """The test's own control plane on ``path``, or None for Monarch's coordinator."""
    return None if path is None else FileStoreControlPlane(path)


class _Participant(Actor):
    """A rank that reaches the store through Monarch's coordinator, its own plane or an address."""

    def __init__(self, store_name: str, plane_path: str | None) -> None:
        self._store_name = store_name
        self._plane_path = plane_path

    async def _client(
        self,
        role: ts.RankRole,
        group: int | None,
        address: ts.ControlPlaneAddress | None,
    ) -> tuple[ts.ControlPlaneAddress, str] | None:
        """Make this process's client and return the address it got, as itself and printed."""
        await ts.client(
            self._store_name,
            role=role,
            group=group,
            address=address,
            control_plane=_control_plane(self._plane_path),
        )
        return None if address is None else (address, str(address))


class _Publisher(_Participant):
    """A trainer rank that publishes the way TitanRL's trainer does."""

    def __init__(self, store_name: str, plane_path: str | None) -> None:
        super().__init__(store_name, plane_path)
        self._rank = current_rank().rank

    @concurrent_endpoint
    async def initialize_client(
        self, address: ts.ControlPlaneAddress | None
    ) -> tuple[ts.ControlPlaneAddress, str] | None:
        return await self._client(ts.RankRole.PUBLISHER, None, address)

    @concurrent_endpoint
    async def push(self, round_: int) -> None:
        state_dict = {f"layers.{self._rank}.weight": _layer(self._rank, round_)}
        # A worker thread with a fresh event loop per push, as in TitanRL's trainer.
        await asyncio.to_thread(
            lambda: asyncio.run(
                ts.put_state_dict(state_dict, _KEY, store_name=self._store_name)
            )
        )


class _Requester(_Participant):
    """A generator rank that reads into the same buffers every round."""

    def __init__(self, store_name: str, plane_path: str | None) -> None:
        super().__init__(store_name, plane_path)
        self._destination = {
            name: torch.zeros_like(tensor) for name, tensor in _weights(0).items()
        }

    @concurrent_endpoint
    async def initialize_client(
        self, address: ts.ControlPlaneAddress | None
    ) -> tuple[ts.ControlPlaneAddress, str] | None:
        return await self._client(ts.RankRole.REQUESTER, 0, address)

    @concurrent_endpoint
    async def pull(self) -> dict[str, torch.Tensor]:
        await ts.get_state_dict(
            _KEY, self._destination, strict=False, store_name=self._store_name
        )
        return self._destination


@pytest.mark.timeout(120)
@pytest.mark.asyncio
@pytest.mark.parametrize("control_plane", ["monarch", "bring_your_own", "service"])
async def test_routed_weight_sync_across_actor_meshes(
    control_plane: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sync publishers that push from worker threads into a requester mesh.

    The bring-your-own arm initializes only a control plane the test wrote, which every
    process opens on its own, and service processes get the driver's address as TitanRL
    forwards it, so a rank that fell back to Monarch's would fail.
    """
    # Monarch's default fault hook interrupts the whole session, so record this
    # test's mesh faults instead. monkeypatch restores the hook after any outcome.
    faults: list[monarch_actor.MeshFailure] = []
    monkeypatch.setattr(monarch_actor, "unhandled_fault_hook", faults.append)
    store_name = f"routed-sync-test-{uuid.uuid4()}"
    bring_your_own = control_plane == "bring_your_own"
    plane_path = str(tmp_path / "control_plane") if bring_your_own else None
    plane = _control_plane(plane_path)
    # The coordinator service advertises this host, which every actor here reaches.
    monkeypatch.setenv("TORCHSTORE_TCPSTORE_HOST", "127.0.0.1")
    # Imported here, so actor processes never load it through this module.
    from torchstore.routing.service import ServiceAddress

    publisher_mesh = this_host().spawn_procs(per_host={"procs": _PUBLISHER_RANKS})
    requester_mesh = this_host().spawn_procs(per_host={"procs": _REQUESTER_RANKS})
    address: ts.ControlPlaneAddress | None = None

    async def sync() -> None:
        nonlocal address
        address = await ts.initialize(
            mesh=publisher_mesh,
            strategy=ts.TorchStoreStrategy(TransportType.MonarchRPC),
            client_type=ts.ClientType.ROUTING,
            store_name=store_name,
            control_plane=control_plane if plane is None else plane,
        )
        if control_plane == "service":
            assert isinstance(address, ServiceAddress)
            assert str(address) == f"127.0.0.1:{address.port}"
        else:
            assert address is None
        publishers = publisher_mesh.spawn(
            "publisher", _Publisher, store_name, plane_path
        )
        requesters = requester_mesh.spawn(
            "requester", _Requester, store_name, plane_path
        )
        # Let every lookup settle before raising a failed one, since stopping a mesh
        # mid-lookup can break Monarch's controller lookups for the session. A hang
        # still reaches the deadline with lookups in flight.
        results = await asyncio.gather(
            publishers.initialize_client.call(address),
            requesters.initialize_client.call(address),
            return_exceptions=True,
        )
        _raise_first(results)
        forwarded = None if address is None else (address, str(address))
        for result in results:
            # Each rank got its own copy of the address, equal to the driver's.
            for point, received in result.items():
                assert received == forwarded, point

        async def sync_round(round_: int) -> None:
            # Each read follows the push, as one that starts first can miss the mapping.
            await publishers.push.call(round_)
            pulled = await requesters.pull.call()
            torch.testing.assert_close(
                {(round_, point.rank): values for point, values in pulled.items()},
                {(round_, rank): _weights(round_) for rank in range(_REQUESTER_RANKS)},
                rtol=0,
                atol=0,
            )

        for round_ in range(_ROUNDS):
            await sync_round(round_)
        if isinstance(address, ServiceAddress):
            server = store_api._owned_service_map[store_name]._server
            assert server is not None
            # With the driver's coordinator and TCPStore stopped, the planned key syncs
            # from the volumes alone.
            await ts.shutdown(store_name)
            for port in (address.port, server.address.port):
                with pytest.raises(ConnectionRefusedError):
                    socket.create_connection(("127.0.0.1", port)).close()
            await sync_round(_ROUNDS)

    try:
        await asyncio.wait_for(sync(), _DEADLINE_S)
    finally:
        try:
            if address is not None:
                await ts.shutdown(store_name)
        finally:
            stopped = await asyncio.gather(
                requester_mesh.stop(), publisher_mesh.stop(), return_exceptions=True
            )
    assert not faults, [fault.report() for fault in faults]
    _raise_first(stopped)
