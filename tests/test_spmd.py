# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import logging
import multiprocessing
import os
import queue
import socket
import traceback
from datetime import timedelta
from typing import Any

import pytest
import torch
import torchstore as ts
import torchstore.api as _api

logger: logging.Logger = logging.getLogger(__name__)

_WORLD_SIZE = 2
_LOCAL_WORLD_SIZE = 2
_PROCESS_TIMEOUT_SECS = 150.0
_RENDEZVOUS_TIMEOUT = timedelta(seconds=20)
_PORT_RETRY_TOKENS = ("EADDRINUSE", "address already in use")
_STRATEGIES: dict[str, type[ts.TorchStoreStrategy]] = {
    "host": ts.HostStrategy,
    "local_rank": ts.LocalRankStrategy,
}


def _barrier(
    rendezvous: Any,
    *,
    phase: str,
    store_name: str,
    rank: int,
) -> None:
    prefix = f"test/{store_name}/{phase}"
    rendezvous.set(f"{prefix}/{rank}", b"1")
    for peer in range(_WORLD_SIZE):
        rendezvous.get(f"{prefix}/{peer}")


async def _run_case(
    *,
    rank: int,
    strategy_name: str,
    store_name: str,
) -> None:
    await ts.initialize_spmd(
        strategy=_STRATEGIES[strategy_name](),
        store_name=store_name,
        transport="ipc",
        rendezvous_timeout=_RENDEZVOUS_TIMEOUT,
    )
    # Reuse the session's internal TCPStore for test-side barriers.
    rendezvous = _api._spmd_state_map[store_name].rendezvous
    try:
        await ts.put(
            f"rank_{rank}",
            torch.tensor([rank], dtype=torch.int64),
            store_name=store_name,
        )

        _barrier(rendezvous, phase="put_done", store_name=store_name, rank=rank)

        for peer in range(_WORLD_SIZE):
            value = await ts.get(
                f"rank_{peer}",
                store_name=store_name,
            )
            assert torch.equal(
                value,
                torch.tensor([peer], dtype=torch.int64),
            )

        _barrier(rendezvous, phase="get_done", store_name=store_name, rank=rank)
    finally:
        await ts.shutdown(store_name)


def _reserve_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _child_main(
    rank: int,
    strategy_name: str,
    store_name: str,
    master_port: int,
    results: Any,
) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(_WORLD_SIZE)
    os.environ["LOCAL_WORLD_SIZE"] = str(_LOCAL_WORLD_SIZE)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)

    try:
        asyncio.run(
            _run_case(
                rank=rank,
                strategy_name=strategy_name,
                store_name=store_name,
            )
        )
        results.put({"rank": rank, "ok": True})
    except Exception:
        results.put(
            {
                "rank": rank,
                "ok": False,
                "error": traceback.format_exc(),
            }
        )


def _run_spmd_case(strategy_name: str) -> None:
    # _reserve_local_port() briefly closes its socket before handing the port
    # to child processes, so a concurrent process on the test host can race us.
    # The retry hides CI flakiness for this known TOCTOU; log each retry so
    # chronic contention doesn't silently slow the suite.
    last_errors: list[str] = []
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        store_name = f"torchstore_spmd_{strategy_name}"
        master_port = _reserve_local_port()
        errors = _run_spmd_case_once(
            strategy_name=strategy_name,
            store_name=store_name,
            master_port=master_port,
        )
        if not errors:
            return

        last_errors = errors
        port_conflict = any(
            token in error for error in errors for token in _PORT_RETRY_TOKENS
        )
        if not port_conflict:
            break
        if attempt < max_attempts:
            logger.warning(
                "SPMD test %s hit a port conflict on attempt %d/%d; retrying",
                strategy_name,
                attempt,
                max_attempts,
            )

    assert not last_errors, "\n\n".join(last_errors)


def _run_spmd_case_once(
    *,
    strategy_name: str,
    store_name: str,
    master_port: int,
) -> list[str]:
    ctx = multiprocessing.get_context("spawn")
    results = ctx.Queue()
    processes: list[multiprocessing.Process] = []

    for rank in range(_WORLD_SIZE):
        process = ctx.Process(
            target=_child_main,
            args=(
                rank,
                strategy_name,
                store_name,
                master_port,
                results,
            ),
        )
        process.start()
        processes.append(process)

    for process in processes:
        process.join(timeout=_PROCESS_TIMEOUT_SECS)

    errors = []
    timed_out_pids: set[int | None] = set()
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5.0)
            timed_out_pids.add(process.pid)
            errors.append(
                f"rank process pid={process.pid} timed out in {strategy_name}"
            )

    outcomes = []
    while len(outcomes) < _WORLD_SIZE:
        try:
            outcomes.append(results.get(timeout=1.0))
        except queue.Empty:
            break

    for outcome in outcomes:
        if not outcome["ok"]:
            errors.append(
                f"rank {outcome['rank']} failed in {strategy_name}:\n{outcome['error']}"
            )

    # Skip processes already reported above as timed out (their -15 exitcode
    # would otherwise produce a duplicate error entry for the same PID).
    for process in processes:
        if process.pid in timed_out_pids:
            continue
        if process.exitcode not in (0, None):
            errors.append(
                f"rank process pid={process.pid} exited with code {process.exitcode}"
            )

    return errors


class _UnsupportedStrategy(ts.TorchStoreStrategy):
    @classmethod
    def get_volume_id(cls) -> str:
        return "0"

    @classmethod
    def get_client_id(cls) -> str:
        return "0"


class _FakeTeardown:
    def __init__(self) -> None:
        self.calls = 0

    async def call(self) -> None:
        self.calls += 1


class _FakeController:
    def __init__(self) -> None:
        self.teardown = _FakeTeardown()


class _FakeRendezvous:
    """In-memory stand-in for torch.distributed.TCPStore. A missing key on
    .get() falls through to ``get_value`` (tests that need a synthetic primary
    outcome pass it explicitly); ``get_error``, if set, wins and raises.
    """

    def __init__(
        self,
        *,
        get_value: bytes | None = None,
        get_error: Exception | None = None,
    ) -> None:
        self.values: dict[str, bytes] = {}
        self._get_value = get_value
        self._get_error = get_error

    def set(self, key: str, value: str | bytes) -> None:
        self.values[key] = value.encode() if isinstance(value, str) else value

    def get(self, key: str) -> bytes:
        if self._get_error is not None:
            raise self._get_error
        if key in self.values:
            return self.values[key]
        if self._get_value is None:
            raise KeyError(key)
        return self._get_value


@pytest.fixture
def primary_session():
    """Factory fixture: builds a fake primary SPMDSession, registers it in
    the state map, and auto-cleans up on teardown regardless of test outcome.
    """
    names: list[str] = []

    def _build(
        store_name: str,
    ) -> tuple[_FakeController, _FakeRendezvous, "ts.spmd._SPMDSession"]:
        controller = _FakeController()
        rendezvous = _FakeRendezvous()
        session = ts.spmd._SPMDSession(
            rendezvous=rendezvous,
            controller=controller,
            store_name=store_name,
            host_mesh=None,
            is_primary=True,
        )
        _api._spmd_state_map[store_name] = session
        names.append(store_name)
        return controller, rendezvous, session

    yield _build

    for name in names:
        _api._spmd_state_map.pop(name, None)
        _api.reset_client(name)


@pytest.fixture
def non_primary_session():
    """Factory fixture: builds a fake non-primary SPMDSession against a
    caller-supplied rendezvous and auto-cleans up on teardown.
    """
    names: list[str] = []

    def _build(
        store_name: str,
        rendezvous: _FakeRendezvous,
    ) -> tuple[_FakeController, "ts.spmd._SPMDSession"]:
        controller = _FakeController()
        session = ts.spmd._SPMDSession(
            rendezvous=rendezvous,
            controller=controller,
            store_name=store_name,
            host_mesh=None,
            is_primary=False,
        )
        _api._spmd_state_map[store_name] = session
        names.append(store_name)
        return controller, session

    yield _build

    for name in names:
        _api._spmd_state_map.pop(name, None)
        _api.reset_client(name)


def test_local_rank_strategy_prefers_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.delenv("RANK", raising=False)
    assert ts.LocalRankStrategy.get_client_id() == "1"

    monkeypatch.setenv("RANK", "7")
    assert ts.LocalRankStrategy.get_client_id() == "7"


@pytest.mark.asyncio
async def test_spmd_initialize_requires_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RANK", raising=False)
    with pytest.raises(RuntimeError, match="RANK"):
        await ts.spmd.initialize(strategy=ts.LocalRankStrategy())


@pytest.mark.asyncio
async def test_spmd_initialize_rejects_unsupported_strategy() -> None:
    with pytest.raises(
        RuntimeError,
        match="HostStrategy or LocalRankStrategy",
    ):
        await ts.spmd.initialize(strategy=_UnsupportedStrategy())


@pytest.mark.asyncio
async def test_spmd_shutdown_delegates_to_session(primary_session) -> None:
    store_name = "shutdown_delegates_to_session"
    controller, rendezvous, _ = primary_session(store_name)

    await ts.shutdown(store_name)

    assert controller.teardown.calls == 1
    assert rendezvous.values[ts.spmd._spmd_key(store_name, "shutdown")] == b"ok"
    assert store_name not in _api._spmd_state_map


@pytest.mark.asyncio
async def test_spmd_session_shutdown_is_idempotent(primary_session) -> None:
    store_name = "session_shutdown_is_idempotent"
    controller, _, session = primary_session(store_name)

    await session.shutdown()
    await session.shutdown()

    assert controller.teardown.calls == 1


@pytest.mark.asyncio
async def test_spmd_shutdown_timeout_has_clear_error(non_primary_session) -> None:
    store_name = "shutdown_timeout_has_clear_error"
    rendezvous = _FakeRendezvous(get_error=RuntimeError("timed out"))
    non_primary_session(store_name, rendezvous)

    with pytest.raises(
        RuntimeError,
        match="Timed out waiting for TorchStore shutdown",
    ):
        await ts.shutdown(store_name)

    assert store_name not in _api._spmd_state_map


@pytest.mark.asyncio
async def test_spmd_shutdown_surfaces_primary_failure_to_non_primary(
    non_primary_session,
) -> None:
    store_name = "shutdown_surfaces_primary_failure"
    rendezvous = _FakeRendezvous(get_value=b"RuntimeError('teardown failed')")
    non_primary_session(store_name, rendezvous)

    with pytest.raises(
        RuntimeError,
        match="TorchStore SPMD shutdown failure",
    ):
        await ts.shutdown(store_name)

    assert store_name not in _api._spmd_state_map


@pytest.mark.asyncio
async def test_spmd_initialize_validates_world_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "3")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29500")

    with pytest.raises(ValueError, match="divisible"):
        await ts.spmd.initialize(strategy=ts.LocalRankStrategy())


@pytest.mark.parametrize("strategy_name", ["host", "local_rank"])
def test_spmd_initialize_end_to_end(strategy_name: str) -> None:
    _run_spmd_case(strategy_name)
