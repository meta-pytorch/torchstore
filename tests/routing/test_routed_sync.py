# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import logging
import re
import sys
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

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
_SELECTION_LINE = re.compile(
    r"routing control_plane=(\S+) address=\S+ store=\S+ rank=\S+ pid=\d+"
)


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


def _record_lookups(lookups: list[str]) -> Callable[..., Any]:
    """Wrap Monarch's controller lookup to record each class it looks up.

    TorchStore imports it on each call, so its own lookups are recorded too.
    """
    lookup = monarch_actor.get_or_spawn_controller

    def recording_lookup(name: str, cls: type, *args: Any, **kwargs: Any) -> Any:
        lookups.append(cls.__name__)
        return lookup(name, cls, *args, **kwargs)

    return recording_lookup


def _record_constructions(cls: type, constructed: list[str]) -> Callable[..., None]:
    """Wrap ``cls.__init__`` to record each instance this process constructs."""
    init = cls.__init__

    def recording_init(self: Any, *args: Any, **kwargs: Any) -> None:
        constructed.append(type(self).__name__)
        init(self, *args, **kwargs)

    return recording_init


class _SelectionKinds(logging.Handler):
    """Collects the control-plane kind that each selection line names."""

    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.kinds: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        if match := _SELECTION_LINE.search(record.getMessage()):
            self.kinds.append(match.group(1))


class _Participant(Actor):
    """A rank that records the control planes its process selects, looks up or hosts."""

    def __init__(self, store_name: str, plane_path: str | None) -> None:
        self._store_name = store_name
        self._plane_path = plane_path
        # Each participant has a process of its own, so it patches module state freely.
        self._selections = _SelectionKinds()
        logger = logging.getLogger(store_api.__name__)
        logger.setLevel(logging.INFO)
        logger.addHandler(self._selections)
        self._lookups: list[str] = []
        monarch_actor.get_or_spawn_controller = _record_lookups(self._lookups)

    async def _client(self, role: ts.RankRole, group: int | None = None) -> None:
        await ts.client(
            self._store_name,
            role=role,
            group=group,
            control_plane=_control_plane(self._plane_path),
        )

    @concurrent_endpoint
    async def recorded(self) -> tuple[list[str], list[str], bool]:
        # Hosting a coordinator needs its module, which only Monarch's path loads here.
        loaded = "torchstore.routing.coordinator" in sys.modules
        return self._selections.kinds, self._lookups, loaded


class _Publisher(_Participant):
    """A trainer rank that publishes the way TitanRL's trainer does."""

    def __init__(self, store_name: str, plane_path: str | None) -> None:
        super().__init__(store_name, plane_path)
        self._rank = current_rank().rank

    @concurrent_endpoint
    async def initialize_client(self) -> None:
        await self._client(ts.RankRole.PUBLISHER)

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
    async def initialize_client(self) -> None:
        await self._client(ts.RankRole.REQUESTER, 0)

    @concurrent_endpoint
    async def pull(self) -> dict[str, torch.Tensor]:
        await ts.get_state_dict(
            _KEY, self._destination, strict=False, store_name=self._store_name
        )
        return self._destination


@pytest.mark.timeout(120)
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bring_your_own", [False, True], ids=["monarch", "bring_your_own"]
)
async def test_routed_weight_sync_across_actor_meshes(
    bring_your_own: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Sync publishers that push from worker threads into a requester mesh.

    The bring-your-own arm passes every process a control plane the test wrote, and
    no process looks up or constructs Monarch's coordinator.
    """
    # Monarch's default fault hook interrupts the whole session, so record this
    # test's mesh faults instead. monkeypatch restores the hook after any outcome.
    faults: list[monarch_actor.MeshFailure] = []
    monkeypatch.setattr(monarch_actor, "unhandled_fault_hook", faults.append)
    store_name = f"routed-sync-test-{uuid.uuid4()}"
    plane_path = str(tmp_path / "control_plane") if bring_your_own else None
    plane = _control_plane(plane_path)
    driver_lookups: list[str] = []
    monkeypatch.setattr(
        monarch_actor, "get_or_spawn_controller", _record_lookups(driver_lookups)
    )
    # Imported here, so actor processes never load it through this module.
    from torchstore.routing.coordinator import RoutingCoordinator

    constructed: list[str] = []
    monkeypatch.setattr(
        RoutingCoordinator,
        "__init__",
        _record_constructions(RoutingCoordinator, constructed),
    )
    caplog.set_level(logging.INFO, logger=store_api.__name__)
    publisher_mesh = this_host().spawn_procs(per_host={"procs": _PUBLISHER_RANKS})
    requester_mesh = this_host().spawn_procs(per_host={"procs": _REQUESTER_RANKS})

    async def sync() -> None:
        publishers = publisher_mesh.spawn(
            "publisher", _Publisher, store_name, plane_path
        )
        # The publishers install their recorders before initialize runs volumes there.
        await publishers.recorded.call()
        initialized = await ts.initialize(
            mesh=publisher_mesh,
            strategy=ts.TorchStoreStrategy(TransportType.MonarchRPC),
            client_type=ts.ClientType.ROUTING,
            store_name=store_name,
            control_plane="monarch" if plane is None else plane,
        )
        assert initialized is None
        requesters = requester_mesh.spawn(
            "requester", _Requester, store_name, plane_path
        )
        # Let every lookup settle before raising a failed one, since stopping a mesh
        # mid-lookup can break Monarch's controller lookups for the session. A hang
        # still reaches the deadline with lookups in flight.
        _raise_first(
            await asyncio.gather(
                publishers.initialize_client.call(),
                requesters.initialize_client.call(),
                return_exceptions=True,
            )
        )
        for round_ in range(_ROUNDS):
            # Each read follows the push, as one that starts first can miss the mapping.
            await publishers.push.call(round_)
            pulled = await requesters.pull.call()
            torch.testing.assert_close(
                {(round_, point.rank): values for point, values in pulled.items()},
                {(round_, rank): _weights(round_) for rank in range(_REQUESTER_RANKS)},
                rtol=0,
                atol=0,
            )

        if plane is not None:
            # Shutdown drops this process's record and client, not the caller's plane.
            await ts.shutdown(store_name)
            served = await plane.get_layouts(_KEY, timeout_s=0)
            assert sorted(served) == ["publisher/0", "publisher/1"]
            # A closed plane refuses this read, so the one above shows it stayed open.
            await plane.close()
            with pytest.raises(RuntimeError, match="is closed"):
                await plane.get_layouts(_KEY, timeout_s=0)
        # One selection line per process. Only the Monarch arm looks up a coordinator,
        # which Monarch hosts in this process, and loads its module in the actors.
        kind, lookups = (
            ("filestore", []) if bring_your_own else ("monarch", ["RoutingCoordinator"])
        )
        for actors in (publishers, requesters):
            records = await actors.recorded.call()
            for point, recorded in records.items():
                assert recorded == ([kind], lookups, not bring_your_own), point
        driver_kinds = [
            match.group(1)
            for record in caplog.records
            if (match := _SELECTION_LINE.search(record.getMessage()))
        ]
        driver_records = (driver_kinds, driver_lookups, constructed)
        assert driver_records == ([kind], lookups, lookups)

    try:
        await asyncio.wait_for(sync(), _DEADLINE_S)
    finally:
        if plane is not None:
            await plane.close()
        stopped = await asyncio.gather(
            requester_mesh.stop(), publisher_mesh.stop(), return_exceptions=True
        )
    assert not faults, [fault.report() for fault in faults]
    _raise_first(stopped)
