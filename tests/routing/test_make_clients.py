# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The routing tests' ``make_clients`` machinery: its fault log and its teardown checks."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from collections.abc import Callable, Iterator
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import monarch.actor as monarch_actor
import pytest
import torchstore as ts
import torchstore.api as store_api
from torchstore.routing.service import ServiceAddress, ServiceControlPlane
from torchstore.transport import TransportType

from . import utils

pytestmark = pytest.mark.timeout(60)

# The shared fixture, which pytest finds here under this name.
make_clients = utils.make_clients
# The session's set where ts.shutdown records released stores, read before any case copies it.
_RELEASED_STORES = store_api._released_stores
# A name the session's set holds while one case runs.
_SEED = f"routing-test-released-{uuid.uuid4()}"


def _failure() -> monarch_actor.MeshFailure:
    """A stand-in for Monarch's MeshFailure, which has no Python constructor, with a report of its own."""
    report = f"it died {uuid.uuid4()}"
    return cast(monarch_actor.MeshFailure, SimpleNamespace(report=lambda: report))


def _levels(
    caplog: pytest.LogCaptureFixture, failure: monarch_actor.MeshFailure
) -> list[int]:
    """The levels of the fault log's records that name this failure's report."""
    return [
        record.levelno
        for record in caplog.records
        if record.name == utils.logger.name and failure.report() in record.getMessage()
    ]


def _config(cleanups: list[Callable[[], object]]) -> pytest.Config:
    return cast(pytest.Config, SimpleNamespace(add_cleanup=cleanups.append))


def _stub_mesh(*answers: AsyncMock) -> Any:
    """A volume mesh whose volume at each index answers ``get_id`` through that mock."""
    mesh = Mock()
    mesh.get_id.call = AsyncMock(
        return_value=[
            ({"procs": index}, (f"publisher/{index}", "localhost"))
            for index in range(len(answers))
        ]
    )
    mesh.slice.side_effect = lambda procs: SimpleNamespace(
        get_id=SimpleNamespace(call_one=answers[procs])
    )
    return mesh


class _Mesh:
    """A stand-in for a publisher mesh, which records when it starts and stops."""

    def __init__(self, steps: list[str]) -> None:
        self.steps = steps

    async def __aenter__(self) -> _Mesh:
        self.steps.append("mesh started")
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.steps.append("mesh stopped")


class _Rendezvous:
    """A stand-in for a handle's TCPStore connection."""

    async def read_strategy(self) -> ts.TorchStoreStrategy:
        return ts.TorchStoreStrategy(TransportType.MonarchRPC)

    async def close(self) -> None:
        pass


@pytest.mark.parametrize(
    ("second", "raised", "deadline_s"),
    [
        ("answers", None, 30.0),
        ("errs", RuntimeError, 30.0),
        ("stalls", pytest.fail.Exception, 0.1),
        ("is cancelled", asyncio.CancelledError, 30.0),
    ],
    ids=["answers", "errs", "stalls", "cancelled"],
)
@pytest.mark.asyncio
async def test_the_volume_check_asks_every_volume_and_logs_later_faults_once_one_fails(
    monkeypatch: pytest.MonkeyPatch,
    second: str,
    raised: type[BaseException] | None,
    deadline_s: float,
) -> None:
    """Ask each volume once. Pass when both answer, and fail when the second errs, stalls or is cancelled.

    Once one fails the fault log logs faults from then on.
    """
    monkeypatch.setattr(utils, "_VOLUME_ANSWER_S", deadline_s)
    log = utils._FaultLog()
    monkeypatch.setattr(utils, "_fault_log", log)
    answers = {
        "answers": AsyncMock(return_value=None),
        "errs": AsyncMock(side_effect=RuntimeError("the publisher's process died")),
        "stalls": AsyncMock(side_effect=asyncio.Event().wait),
        "is cancelled": AsyncMock(side_effect=asyncio.CancelledError),
    }
    first = AsyncMock(return_value=None)
    strategy = ts.TorchStoreStrategy(TransportType.MonarchRPC)
    await strategy.set_storage_volumes(_stub_mesh(first, answers[second]))

    with pytest.raises(raised) if raised else contextlib.nullcontext():
        await utils._assert_volumes_answer(strategy)
    assert log.late is (raised is not None)
    first.assert_awaited_once_with()
    answers[second].assert_awaited_once_with()


@pytest.mark.asyncio
async def test_the_volume_check_fails_a_store_with_no_volumes() -> None:
    """Fail a store whose strategy maps no volume so a check that asks nothing never passes."""
    strategy = ts.TorchStoreStrategy(TransportType.MonarchRPC)
    await strategy.set_storage_volumes(_stub_mesh())
    with pytest.raises(AssertionError, match="the store has no volumes"):
        await utils._assert_volumes_answer(strategy)


@pytest.mark.asyncio
async def test_the_fault_log_installs_once_a_session_and_its_cleanup_starts_over(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Install the hook in a session's first case, and put the replaced one back with no failed check at its end."""
    # Made before the hook it replaces is set, as utils makes its log when it is imported.
    log = utils._FaultLog()
    handed: list[object] = []
    monkeypatch.setattr(monarch_actor, "unhandled_fault_hook", handed.append)
    cleanups: list[Callable[[], object]] = []
    for _ in range(2):
        async with log.case(_config(cleanups)):
            pass
    assert (monarch_actor.unhandled_fault_hook, len(cleanups)) == (log, 1)
    log.late = True
    cleanups[0]()
    assert monarch_actor.unhandled_fault_hook == handed.append and not log.late
    # A later session in this interpreter installs it again.
    async with log.case(_config(cleanups)):
        assert monarch_actor.unhandled_fault_hook is log
    assert len(cleanups) == 2


@pytest.mark.asyncio
async def test_the_fault_log_logs_faults_while_a_case_runs_and_hands_them_on_between_cases(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Log a fault at WARNING while a case runs, and hand faults on after it ends, also after a case that raised."""
    caplog.set_level(logging.INFO, logger=utils.logger.name)
    log = utils._FaultLog()
    handed: list[object] = []
    monkeypatch.setattr(monarch_actor, "unhandled_fault_hook", handed.append)
    failure = _failure()
    async with log.case(_config([])):
        log(failure)
    log(failure)
    with pytest.raises(KeyError):
        async with log.case(_config([])):
            raise KeyError("a case that raises")
    log(failure)
    assert handed == [failure, failure]
    assert _levels(caplog, failure) == [logging.WARNING]


@pytest.mark.asyncio
async def test_the_fault_log_logs_faults_at_info_once_a_volume_check_failed(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Log a fault after its case at INFO once a volume check failed, and one during a case at WARNING."""
    caplog.set_level(logging.INFO, logger=utils.logger.name)
    log = utils._FaultLog()
    handed: list[object] = []
    monkeypatch.setattr(monarch_actor, "unhandled_fault_hook", handed.append)
    async with log.case(_config([])):
        pass
    log.late = True
    failure = _failure()
    log(failure)
    async with log.case(_config([])):
        log(failure)
    assert handed == []
    assert _levels(caplog, failure) == [logging.INFO, logging.WARNING]


@pytest.fixture
def seeded_released_stores() -> Iterator[None]:
    """Seed the session's released stores before ``make_clients`` copies them, and check the copy goes with the case."""
    _RELEASED_STORES.add(_SEED)
    try:
        yield
        assert store_api._released_stores is _RELEASED_STORES, "the case's copy stayed"
    finally:
        _RELEASED_STORES.discard(_SEED)


@pytest.mark.parametrize("make_clients", ["monarch"], indirect=True)
@pytest.mark.usefixtures("seeded_released_stores", "make_clients")
@pytest.mark.asyncio
async def test_make_clients_runs_each_case_under_the_fault_log_with_its_own_released_stores(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Log a fault Monarch reports during a case, and give the case a copy of the session's released stores."""
    caplog.set_level(logging.WARNING, logger=utils.logger.name)
    assert monarch_actor.unhandled_fault_hook is utils._fault_log
    handed: list[object] = []
    monkeypatch.setattr(utils._fault_log, "_previous", handed.append)
    failure = _failure()
    monarch_actor.unhandled_fault_hook(failure)
    assert (handed, _levels(caplog, failure)) == ([], [logging.WARNING])
    assert store_api._released_stores is not _RELEASED_STORES
    assert _SEED in store_api._released_stores


@pytest.mark.asyncio
async def test_teardown_asks_each_volume_to_answer_then_stops_the_mesh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check the volumes while the publisher mesh runs, then stop the mesh."""
    steps: list[str] = []

    async def initialize(**kwargs: Any) -> None:
        steps.append("initialized")

    async def check(strategy: ts.TorchStoreStrategy) -> None:
        steps.append("volumes checked")

    monkeypatch.setattr(ts, "initialize", initialize)
    monkeypatch.setattr(utils, "_assert_volumes_answer", check)
    async with contextlib.AsyncExitStack() as exit_stack:
        await utils._initialize(cast(Any, _Mesh(steps)), "monarch", exit_stack)
        assert steps == ["mesh started", "initialized"]
    assert steps == ["mesh started", "initialized", "volumes checked", "mesh stopped"]


@pytest.mark.parametrize(
    "failure",
    ["shutdown", "handle close", None],
    ids=["shutdown-fails", "handle-close-fails", "releases"],
)
@pytest.mark.asyncio
async def test_a_service_store_teardown_reports_its_failing_step_and_keeps_no_release(
    monkeypatch: pytest.MonkeyPatch, failure: str | None
) -> None:
    """Report a failing teardown step rather than the release check after it.

    A shutdown that releases the store leaves its name out of the process's released stores once the stack closes.
    """
    monkeypatch.setattr(store_api, "_control_plane_map", {})
    monkeypatch.setattr(store_api, "_owned_service_map", {})
    released = store_api._released_stores
    steps: list[str] = []
    stores: list[str] = []
    driver, service = Mock(), Mock()

    async def failing_close() -> None:
        raise RuntimeError("the handle close failed")

    async def initialize(*, strategy: Any, store_name: str, **kwargs: Any) -> Any:
        await strategy.set_storage_volumes(_stub_mesh(AsyncMock(return_value=None)))
        stores.append(store_name)
        store_api._control_plane_map[store_name] = store_api._SelectedControlPlane(
            driver, None
        )
        store_api._owned_service_map[store_name] = service
        address = ServiceAddress(store_name, "127.0.0.1", 1, "nonce")
        handle = ServiceControlPlane(address, cast(Any, _Rendezvous()))
        if failure == "handle close":
            monkeypatch.setattr(handle, "close", failing_close)
        return SimpleNamespace(connect=AsyncMock(return_value=handle))

    async def shutdown(store_name: str) -> None:
        # As ts.shutdown releases a store it records and owns before anything it closes can fail.
        del store_api._control_plane_map[store_name]
        del store_api._owned_service_map[store_name]
        store_api._released_stores.add(store_name)
        if failure == "shutdown":
            raise RuntimeError("the shutdown failed")

    host = SimpleNamespace(spawn_procs=lambda **kwargs: _Mesh(steps))
    monkeypatch.setattr(utils, "this_host", lambda: host)
    monkeypatch.setattr(ts, "initialize", initialize)
    monkeypatch.setattr(ts, "shutdown", shutdown)
    reported = pytest.raises(RuntimeError, match=f"^the {failure} failed$")
    with reported if failure else contextlib.nullcontext():
        async with contextlib.AsyncExitStack() as exit_stack:
            await utils.routing_clients({"publisher/0": {}}, {}, "service", exit_stack)
    assert steps == ["mesh started", "mesh stopped"]
    assert store_api._control_plane_map == store_api._owned_service_map == {}
    assert store_api._released_stores is released and stores[0] not in released


class _StubMesh:
    """A stand-in for a recorded publisher mesh, which starts again unless it stopped."""

    def __init__(self, stopped: bool) -> None:
        self.stopped = stopped

    async def __aenter__(self) -> _StubMesh:
        if self.stopped:
            raise RuntimeError("this mesh has stopped")
        return self

    async def __aexit__(self, *exc: object) -> None:
        pass


@pytest.mark.parametrize(
    ("control_plane", "left", "meshes", "looked_up", "raised", "message"),
    [
        ("service", [], [True], [], None, ""),
        ("monarch", [], [True], ["RoutingCoordinator"], None, ""),
        ("service", ["routing-test-left"], [True], [], AssertionError, "left recorded"),
        ("service", [], [False], [], pytest.fail.Exception, "DID NOT RAISE"),
        ("service", [], [], [], AssertionError, "but 0 meshes recorded"),
        (
            "service",
            [],
            [True],
            ["RoutingCoordinator"],
            AssertionError,
            "reached Monarch",
        ),
    ],
    ids=[
        "stopped",
        "monarch-lookup",
        "store-left",
        "mesh-running",
        "mesh-unrecorded",
        "service-lookup",
    ],
)
@pytest.mark.asyncio
async def test_the_fixtures_last_checks_fail_on_what_a_case_left(
    control_plane: str,
    left: list[str],
    meshes: list[bool],
    looked_up: list[str],
    raised: type[BaseException] | None,
    message: str,
) -> None:
    """Fail on a store left recorded, a mesh left running or never recorded, or a Monarch lookup on the service arm."""
    spawned = [_StubMesh(stopped) for stopped in meshes]
    checked = utils._assert_torn_down(control_plane, left, spawned, 1, looked_up, [])
    with pytest.raises(raised, match=message) if raised else contextlib.nullcontext():
        await checked


@pytest.mark.asyncio
async def test_routing_clients_take_only_the_values_ts_initialize_takes() -> None:
    """Refuse a backend other than "monarch" and "service" before anything starts."""
    async with contextlib.AsyncExitStack() as exit_stack:
        with pytest.raises(ValueError, match="no routing test backend 'Service'"):
            await utils.routing_clients({"publisher/0": {}}, {}, "Service", exit_stack)
