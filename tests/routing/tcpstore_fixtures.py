# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TCPStore helpers and fixtures that the rendezvous and service tests share, free of Monarch."""

from __future__ import annotations

import asyncio
import contextlib
import os
import select
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Collection, Iterator
from typing import Any, cast

import pytest
import torchstore as ts
from torchstore.routing import tcpstore
from torchstore.routing.service import (
    CoordinatorService,
    ServiceAddress,
    ServiceControlPlane,
)
from torchstore.routing.tcpstore import TCPStoreAddress, TCPStoreRendezvous
from torchstore.transport import TransportType

_LOCALHOST = "127.0.0.1"
_PUBLISHER_RANKS = frozenset({"publisher/0", "publisher/1"})
# A TCPStore server the tests can stop or kill. It exits when its stdin closes.
_SERVER = (
    "import sys\n"
    "from torch.distributed import TCPStore\n"
    f"store = TCPStore({_LOCALHOST!r}, 0, is_master=True, wait_for_workers=False)\n"
    "print(store.port, flush=True)\n"
    "sys.stdin.read()\n"
)


class _Mesh:
    """A stand-in mesh whose size is the publisher count, which is all a driver may read."""

    def __init__(self, size: int) -> None:
        self._size = size

    def size(self) -> int:
        return self._size


def _store_name() -> str:
    return f"tcpstore-test-{uuid.uuid4()}"


def _strategy() -> ts.TorchStoreStrategy:
    return ts.TorchStoreStrategy(TransportType.MonarchRPC)


def _deadline() -> float:
    return time.monotonic() + 60


def _native_calls(name: str = "") -> list[threading.Thread]:
    prefix = f"torchstore-tcpstore {name}"
    return [t for t in threading.enumerate() if t.name.startswith(prefix)]


_earlier_calls: set[threading.Thread] = set()


def _join_native_calls() -> None:
    """Let this test's native calls finish, so none outlives the test."""
    mine = set(_native_calls()) - _earlier_calls
    for thread in mine:
        thread.join(timeout=60)
    alive = [thread for thread in mine if thread.is_alive()]
    assert not alive, "native TCPStore calls outlived the test"


@pytest.fixture(autouse=True)
def _joins_its_native_calls() -> Iterator[None]:
    """Fail the test that leaves a native call running, not a later one."""
    _earlier_calls.clear()
    _earlier_calls.update(_native_calls())
    yield
    _join_native_calls()
    assert not tcpstore._live_calls - _earlier_calls, "finished calls stayed registered"


def _stop(process: subprocess.Popen[bytes]) -> None:
    """Stop the server and wait until the kernel reports it stopped."""
    process.send_signal(signal.SIGSTOP)
    os.waitpid(process.pid, os.WUNTRACED)


async def _until(condition: Callable[[], object], timeout_s: float = 30) -> None:
    deadline = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline, "condition not met before the deadline"
        await asyncio.sleep(0.01)


@contextlib.asynccontextmanager
async def _served(
    host: str | None = _LOCALHOST, store_name: str | None = None
) -> AsyncIterator[TCPStoreRendezvous]:
    rendezvous = await TCPStoreRendezvous.serve(store_name or _store_name(), host=host)
    try:
        yield rendezvous
    finally:
        await rendezvous.close()


@contextlib.asynccontextmanager
async def _connected(address: TCPStoreAddress) -> AsyncIterator[TCPStoreRendezvous]:
    rendezvous = await TCPStoreRendezvous.connect(address)
    try:
        yield rendezvous
    finally:
        await rendezvous.close()


@pytest.fixture
def server() -> Iterator[tuple[subprocess.Popen[bytes], TCPStoreAddress]]:
    """Serve a TCPStore from a subprocess so a test can stop or kill it."""
    with subprocess.Popen(
        [sys.executable, "-c", _SERVER], stdin=subprocess.PIPE, stdout=subprocess.PIPE
    ) as process:
        try:
            assert process.stdout is not None
            readable, _, _ = select.select([process.stdout], [], [], 60)
            line = process.stdout.readline() if readable else b""
            if not line:
                process.kill()  # a no-op once it has exited
                pytest.fail(
                    "the TCPStore server subprocess did not report its port "
                    f"(exit status {process.wait(timeout=60)})"
                )
            nonce = uuid.uuid4().hex
            yield process, TCPStoreAddress(_LOCALHOST, int(line), _store_name(), nonce)
        finally:
            process.kill()
            process.wait(timeout=60)


async def _initialize_service(
    store_name: str,
    strategy: ts.TorchStoreStrategy | None = None,
    *,
    publishers: int = 1,
) -> ServiceAddress:
    """Initialize a routing store on the coordinator service in this process, as its driver, over MonarchRPC by default."""
    address = await ts.initialize(
        mesh=cast(Any, _Mesh(publishers)),
        strategy=_strategy() if strategy is None else strategy,
        client_type=ts.ClientType.ROUTING,
        store_name=store_name,
        control_plane="service",
    )
    assert isinstance(address, ServiceAddress)
    return address


async def _close_each(
    closes: Collection[Callable[[], Awaitable[object]]], failing: bool
) -> None:
    """Run every close in turn, then raise the first one's error unless the test's own is on its way."""
    first: BaseException | None = None
    for close in closes:
        try:
            await close()
        except BaseException as error:
            first = first or error
    if first is not None and not failing:
        raise first


@contextlib.asynccontextmanager
async def _other_service(
    store_name: str, strategy: ts.TorchStoreStrategy
) -> AsyncIterator[CoordinatorService]:
    """Serve another initialized coordinator for a store, outside this process's records."""
    service = await CoordinatorService.start(store_name, host=_LOCALHOST)
    closes: list[Callable[[], Awaitable[None]]] = [service.close]
    try:
        driver = await service.connect()
        closes.insert(0, driver.close)
        await driver.init(publishers={"publisher/0"}, strategy=strategy)
        yield service
    finally:
        # Every close runs, and a failing case's own error stays the one reported.
        await _close_each(closes, sys.exc_info()[1] is not None)


@contextlib.asynccontextmanager
async def service_store() -> AsyncIterator[
    Callable[[], Awaitable[ServiceControlPlane]]
]:
    """Serve a coordinator service in this process. Handles reach it over TCP.

    The first handle is the driver's, which may init, and later ones dial the address as ranks do.
    Handles keep the lookup's keyword-only timeout, as the contract suite's calls need.
    """
    service = await CoordinatorService.start(
        f"service-store-{uuid.uuid4()}", host=_LOCALHOST
    )
    handles: list[ServiceControlPlane] = []

    async def connect() -> ServiceControlPlane:
        dial = service.address.connect if handles else service.connect
        handles.append(await dial())
        return handles[-1]

    try:
        yield connect
    finally:
        # Handles first, so no call left waiting holds the service's close grace. Every close
        # runs, and a failing test's own error stays the one reported.
        closes = [*(handle.close for handle in handles), service.close]
        await _close_each(closes, sys.exc_info()[1] is not None)
