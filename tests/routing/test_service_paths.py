# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Service-backend refusals, repeat shutdowns, failed starts and close failures, checked without Monarch."""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import re
import sys
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any, cast

import pytest
import pytest_asyncio
import torch
import torchstore as ts
import torchstore.api as store_api
import torchstore.backend as backend_module
from torchstore.backend import Backend
from torchstore.carrier_volumes import CarrierVolumeHost
from torchstore.routing.client import RoutingClient
from torchstore.routing.service import (
    CoordinatorService,
    ServiceAddress,
    ServiceControlPlane,
)
from torchstore.routing.tcpstore import TCPStoreRendezvous
from torchstore.strategy import _installed_volume_host

from . import routed_sync_data as data, tcpstore_fixtures, test_routed_sync_mp
from .test_service import _close_each, _served

pytestmark = [
    pytest.mark.timeout(120),
    pytest.mark.filterwarnings("error::pytest.PytestUnhandledThreadExceptionWarning"),
]

# The shared fixtures, which pytest finds here under these names.
_joins_its_native_calls = tcpstore_fixtures._joins_its_native_calls
_in_process = test_routed_sync_mp._in_process

_PUBLISHER = ("publisher/0",)


class _Asked:
    """The stand-in for Monarch that records each name a case asks it for apart from ``actor_rank``."""

    def __init__(self, stand_in: object, asked: list[str]) -> None:
        self._stand_in, self._asked = stand_in, asked

    def __getattr__(self, name: str) -> Any:
        if name != "actor_rank":
            self._asked.append(name)
        return getattr(self._stand_in, name)


@pytest_asyncio.fixture
async def store_name(
    _in_process: None, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[str]:
    """A fresh store with this process's selection state and pending closes empty and a stand-in for Monarch.

    The stand-in reports no actor context and refuses every other call. Teardown releases what the case left recorded
    without asking Monarch, then fails on any call the stand-in got because the library may only log its refusal.
    """
    # _in_process empties the rest of the selection state.
    monkeypatch.setattr(store_api, "_closing", set())
    test_routed_sync_mp._refuse_monarch(monkeypatch)
    asked: list[str] = []
    stand_in = _Asked(backend_module._backends["monarch"], asked)
    monkeypatch.setitem(backend_module._backends, "monarch", cast(Backend, stand_in))
    name = f"service-paths-{uuid.uuid4()}"
    yield name
    await store_api._shutdown_selected(name)
    assert asked == []


async def _initialize(store_name: str) -> ServiceAddress:
    return await tcpstore_fixtures._initialize_service(
        store_name, ts.TorchStoreStrategy()
    )


@pytest.mark.asyncio
async def test_a_second_shutdown_asks_monarch_nothing(store_name: str) -> None:
    """Shut a store down twice, then refuse a role-less client: the released record keeps Monarch out."""
    await _initialize(store_name)
    await ts.shutdown(store_name)
    await ts.shutdown(store_name)
    with pytest.raises(ValueError, match="needs role on its first ts.client call$"):
        await ts.client(store_name)


@pytest.mark.asyncio
async def test_a_role_less_first_client_of_a_released_store_is_refused(
    store_name: str,
) -> None:
    """Refuse a role-less first client of a store shut down here. Monarch is asked nothing."""
    await _initialize(store_name)
    await ts.shutdown(store_name)
    with pytest.raises(ValueError, match="needs role on its first ts.client call$"):
        await ts.client(store_name)


@pytest.mark.parametrize(
    ("arguments", "match"),
    [
        ({}, "needs role on its first ts.client call$"),
        (
            {"role": ts.RankRole.REQUESTER},
            "^requesters must pass the index of their requester mesh$",
        ),
        (
            {"role": ts.RankRole.PUBLISHER, "group": 0},
            "^publishers are a single mesh and take no group$",
        ),
    ],
    ids=["no-role", "requester-no-group", "publisher-group"],
)
@pytest.mark.asyncio
async def test_a_first_client_of_a_store_initialized_here_refuses_its_role_or_group(
    store_name: str, arguments: dict[str, Any], match: str
) -> None:
    """Refuse a first client of a store this process initialized for its role or group. Monarch is asked nothing."""
    await _initialize(store_name)
    with pytest.raises(ValueError, match=match):
        await ts.client(store_name, **arguments)


@pytest.mark.parametrize(
    ("arguments", "error", "match", "recorded"),
    [
        (
            lambda address: {"address": "host:1"},
            TypeError,
            "^address must be a ControlPlaneAddress, got 'host:1'$",
            False,
        ),
        (
            lambda address: {"address": address},
            ValueError,
            "^address and control_plane are for routing clients: pass role$",
            False,
        ),
        (
            lambda address: {
                "role": ts.RankRole.PUBLISHER,
                "address": dataclasses.replace(address, store_name="other-store"),
            },
            ValueError,
            "^address is for store 'other-store', not ",
            False,
        ),
        # A store initialized here has a record, whose check would refuse this call for its role if it came first.
        (
            lambda address: {"address": address},
            ValueError,
            "^address and control_plane are for routing clients: pass role$",
            True,
        ),
    ],
    ids=[
        "address-not-an-address",
        "address-no-role",
        "address-of-another-store",
        "address-no-role-recorded",
    ],
)
@pytest.mark.asyncio
async def test_a_first_client_refuses_its_address_before_dialing_it(
    store_name: str,
    arguments: Callable[[ServiceAddress], dict[str, Any]],
    error: type[Exception],
    match: str,
    recorded: bool,
) -> None:
    """Refuse a first client for its address before it dials, marks the store, checks its record or asks Monarch.

    The address is built in place. With ``recorded`` this process initializes the store first.
    """
    if recorded:
        await _initialize(store_name)
    address = ServiceAddress(
        store_name, tcpstore_fixtures._LOCALHOST, 1, uuid.uuid4().hex
    )
    with pytest.raises(error, match=match):
        await ts.client(store_name, **arguments(address))
    assert store_name not in store_api._attempted and store_api._building == set()


@pytest.mark.asyncio
async def test_a_cached_client_refuses_an_address_it_did_not_connect_through(
    store_name: str,
) -> None:
    """Refuse an address with another nonce once this process caches a client of the store."""
    address = await _initialize(store_name)
    await ts.client(store_name, role=ts.RankRole.PUBLISHER, rank=0)
    other = dataclasses.replace(address, nonce=uuid.uuid4().hex)
    refusal = f"did not connect through {re.escape(str(other))}\\. Call ts.reset_client"
    with pytest.raises(ValueError, match=refusal):
        await ts.client(store_name, role=ts.RankRole.PUBLISHER, address=other)


def _rows(rank: int) -> torch.Tensor:
    """Rank ``rank``'s rows of an 8-row tensor sharded over 2 ranks."""
    return data.place({"w": torch.arange(8)}, {"w": 0}, rank, 2)["w"]


async def _publisher(store_name: str) -> RoutingClient:
    """The store's publisher client with a plan that holds rank 0's rows of ``model/w``."""
    publisher = await ts.client(store_name, role=ts.RankRole.PUBLISHER, rank=0)
    assert isinstance(publisher, RoutingClient)
    await publisher.register_layout({"w": _rows(0)}, "model")
    return publisher


@pytest.mark.parametrize(
    ("batch", "error", "match"),
    [
        (lambda: {"other": torch.empty(4)}, KeyError, "does not store"),
        (lambda: {"model/w": torch.empty(3)}, ValueError, "has shape .* expected"),
        # The other rank's rows have the plan's shapes but not its offsets.
        (lambda: {"model/w": _rows(1)}, ValueError, "does not match the routing plan"),
    ],
    ids=["unknown-key", "wrong-shape", "other-ranks-rows"],
)
@pytest.mark.asyncio
async def test_a_publisher_refuses_a_tensor_its_plan_does_not_hold(
    store_name: str,
    batch: Callable[[], dict[str, torch.Tensor]],
    error: type[Exception],
    match: str,
) -> None:
    """Refuse a publisher's unknown key, wrong shape or another rank's rows."""
    await _initialize(store_name)
    publisher = await _publisher(store_name)
    with pytest.raises(error, match=match):
        await publisher.put_batch(batch())


@pytest.mark.parametrize(
    ("key", "destination", "error", "match"),
    [
        ("model/w", torch.empty(3), ValueError, "destination .* has shape .* expected"),
        ("model/missing", None, KeyError, "has no tensor route"),
    ],
    ids=["wrong-destination", "unplanned-key"],
)
@pytest.mark.asyncio
async def test_a_requester_refuses_a_destination_or_key_its_plan_does_not_hold(
    store_name: str,
    key: str,
    destination: torch.Tensor | None,
    error: type[Exception],
    match: str,
) -> None:
    """Refuse a requester's wrong destination shape or a key its plan lacks."""
    address = await _initialize(store_name)
    await _publisher(store_name)
    handle = await address.connect()
    host: CarrierVolumeHost | None = None
    try:
        strategy = await handle.strategy()
        installed = _installed_volume_host(strategy)
        assert isinstance(installed, CarrierVolumeHost)
        host = installed
        reader = RoutingClient("requester/0/0", ts.RankRole.REQUESTER, handle, strategy)
        await reader.get_layouts({"w": _rows(0)}, "model")
        with pytest.raises(error, match=match):
            await reader.get(key, destination)
    finally:
        closes = [handle.close] if host is None else [host.close, handle.close]
        await _close_each(closes, sys.exc_info()[1] is not None)


@pytest.mark.parametrize(
    ("control_plane", "match"),
    [
        (
            "other",
            "^control_plane must be 'monarch', 'service' or a ControlPlane instance,"
            " got 'other'$",
        ),
        ("service", "^standard stores take only control_plane='monarch'$"),
    ],
)
@pytest.mark.asyncio
async def test_initialize_refuses_a_plane_it_cannot_take(
    store_name: str, control_plane: str, match: str
) -> None:
    """Refuse a plane name it does not know, and ``"service"`` for a standard store, asking Monarch nothing."""
    with pytest.raises(ValueError, match=match):
        await ts.initialize(store_name=store_name, control_plane=control_plane)


@pytest.mark.asyncio
async def test_a_failed_service_init_keeps_its_error_and_leaves_nothing_behind(
    store_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep a failed init's error, close its handle, stop its coordinator and TCPStore, and record nothing."""
    started: list[tuple[CoordinatorService, TCPStoreRendezvous | None]] = []
    handles: list[ServiceControlPlane] = []
    start = CoordinatorService.start

    async def recording_start(*args: Any, **kwargs: Any) -> CoordinatorService:
        service = await start(*args, **kwargs)
        started.append((service, service._rendezvous))
        return service

    async def refuse(handle: ServiceControlPlane, *args: Any, **kwargs: Any) -> None:
        handles.append(handle)
        raise RuntimeError("init refused")

    monkeypatch.setattr(CoordinatorService, "start", recording_start)
    monkeypatch.setattr(ServiceControlPlane, "init", refuse)
    with pytest.raises(RuntimeError, match="^init refused$"):
        await _initialize(store_name)
    [(service, rendezvous)], [handle] = started, handles
    assert handle._closed
    assert test_routed_sync_mp._servers(store_name) == []
    assert rendezvous is not None and rendezvous._store is None
    assert store_name not in store_api._control_plane_map
    assert store_name not in store_api._owned_service_map


@pytest.mark.parametrize("stop", ["timeout", "cancel"])
@pytest.mark.asyncio
async def test_a_dialed_close_that_shutdown_stops_waiting_for_runs_on_and_logs_its_failure(
    store_name: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    stop: str,
) -> None:
    """Drop the record, leave a dialed close running when shutdown stops waiting for it, and log its failure.

    Shutdown raises TimeoutError once its wait runs out, or CancelledError when it is cancelled.
    """
    caplog.set_level(logging.WARNING, logger=store_api.logger.name)
    if stop == "timeout":
        monkeypatch.setattr(store_api, "_CLOSE_WAIT_S", 0.2)
    entered, release = asyncio.Event(), asyncio.Event()
    async with _served(_PUBLISHER, store_name=store_name) as served:
        address = served.service.address
        client = await ts.client(
            store_name, role=ts.RankRole.REQUESTER, group=0, rank=0, address=address
        )
        assert isinstance(client, RoutingClient)
        dialed = client.control_plane
        assert isinstance(dialed, ServiceControlPlane)
        close = dialed.close

        async def failing() -> None:
            entered.set()
            await release.wait()
            await close()
            raise ConnectionResetError("peer gone")

        monkeypatch.setattr(dialed, "close", failing)
        try:
            shutting = asyncio.ensure_future(ts.shutdown(store_name))
            await asyncio.wait_for(entered.wait(), 30)
            if stop == "cancel":
                shutting.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await shutting
            else:
                with pytest.raises(TimeoutError, match="took over 0.2s$"):
                    await shutting
            assert store_name not in store_api._control_plane_map
            (running,) = store_api._closing
            assert not running.done()
        finally:
            release.set()
            await asyncio.gather(*store_api._closing, return_exceptions=True)
        # Inside the block so that a failing cleanup of the served service cannot skip these.
        [logged] = [r for r in caplog.records if r.name == store_api.logger.name]
        failed = f"closing the connection dialed through {address} failed"
        assert (logged.levelno, logged.getMessage()) == (logging.WARNING, failed)
        assert logged.exc_info is not None
        assert isinstance(logged.exc_info[1], ConnectionResetError)
