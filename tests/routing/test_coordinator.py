# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import contextlib
import inspect
import math
import os
import re
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Collection, Mapping
from pathlib import Path
from typing import Any, Protocol
from unittest.mock import ANY, create_autospec

import pytest
import pytest_asyncio
import torch
import torchstore as ts
import torchstore.routing.coordinator as coordinator_module
import torchstore.routing.layout_registry as layout_registry_module
from monarch.actor import (
    ActorError,
    concurrent_endpoint,
    endpoint,
    get_or_spawn_controller,
)
from torchstore.routing._model import KeyRegistration, Registrations
from torchstore.routing.control_plane import ControlPlane
from torchstore.routing.coordinator import MonarchControlPlane, RoutingCoordinator
from torchstore.routing.layout_registry import LayoutRegistry
from torchstore.transport import TransportType

from ..volume_meshes import VolumeMesh
from . import file_store_control_plane
from .file_store_control_plane import FileStoreControlPlane
from .utils import registry_store, tensor_slice

pytestmark = pytest.mark.timeout(60)

# A lookup that does not wait returns well within this. A correct one cannot.
_STILL_PENDING_S = 0.5
# A timeout fires at most a timer's slack early, and on a loaded host within _LATE_S
# of its deadline.
_EARLY_S, _LATE_S = 0.01, 1.5


class TimedControlPlane(Protocol):
    """The control-plane interface, its lookup also taking a keyword-only timeout."""

    async def init(
        self, publishers: Collection[str], strategy: ts.TorchStoreStrategy
    ) -> None:
        ...

    async def register_layout(
        self, rank: str, key: str, registrations: Mapping[str, KeyRegistration]
    ) -> None:
        ...

    async def get_layouts(
        self, key: str, *, timeout_s: float | None = None
    ) -> Registrations:
        ...

    async def strategy(self) -> ts.TorchStoreStrategy:
        ...


@contextlib.asynccontextmanager
async def _monarch_store() -> AsyncIterator[Callable[[], Awaitable[TimedControlPlane]]]:
    name = f"routing-coordinator-test-{uuid.uuid4()}"

    async def connect() -> TimedControlPlane:
        return MonarchControlPlane(
            await get_or_spawn_controller(name, RoutingCoordinator)
        )

    yield connect


async def _coordinator(publishers=None, strategy=None):
    coordinator = await get_or_spawn_controller(
        f"routing-coordinator-test-{uuid.uuid4()}", RoutingCoordinator
    )
    if publishers is not None:
        await coordinator.init.call_one(publishers, strategy)
    return coordinator


@contextlib.asynccontextmanager
async def _file_store() -> AsyncIterator[Callable[[], Awaitable[TimedControlPlane]]]:
    """Open handles on one ``FileStoreControlPlane`` store, and close each at exit."""
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "control_plane")
        handles: list[FileStoreControlPlane] = []

        async def connect() -> TimedControlPlane:
            handles.append(FileStoreControlPlane(path))
            return handles[-1]

        try:
            yield connect
        finally:
            await asyncio.gather(*(handle.close() for handle in handles))


# The coordinator actor runs in this process, so the Monarch and registry stores
# both read this patched default.
_LOOKUP_DEFAULT = (layout_registry_module, "_LAYOUT_REGISTRATION_TIMEOUT_S")


@pytest_asyncio.fixture(
    params=[
        pytest.param((_monarch_store, *_LOOKUP_DEFAULT), id="monarch"),
        pytest.param((registry_store, *_LOOKUP_DEFAULT), id="registry"),
        pytest.param(
            (_file_store, file_store_control_plane, "_LAYOUT_REGISTRATION_TIMEOUT_S"),
            id="filestore",
        ),
    ]
)
async def make_control_plane(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[Callable[..., Awaitable[Any]]]:
    """Open a fresh store per call and return a client handle, or ``handles`` of them.

    Handles connect on their own, as ranks do. Store contexts exit after the test,
    though Monarch's coordinator outlives its context. Error types differ, so match
    messages. ``default_timeout_s`` patches the param's lookup default.
    """
    new_store, default_timeout_owner, default_timeout_name = request.param
    async with contextlib.AsyncExitStack() as stack:

        async def make(
            publishers: Collection[str] | None = None,
            strategy: ts.TorchStoreStrategy | None = None,
            *,
            handles: int = 1,
            default_timeout_s: float | None = None,
        ) -> Any:
            if default_timeout_s is not None:
                monkeypatch.setattr(
                    default_timeout_owner, default_timeout_name, default_timeout_s
                )
            connect = await stack.enter_async_context(new_store())
            driver = await connect()
            if publishers is not None:
                await driver.init(
                    publishers,
                    strategy or ts.TorchStoreStrategy(TransportType.MonarchRPC),
                )
            clients = [await connect() for _ in range(handles)]
            return clients[0] if handles == 1 else clients

        yield make


def _registration(key: str = "model/w") -> dict[str, KeyRegistration]:
    # Two keys of distinct, non-default fields: an encoding that drops a key, drops
    # or reorders dimensions, or swaps fields changes what comes back.
    return {
        key: KeyRegistration(
            tensor_slice(
                (8, 2),
                (4, 3),
                global_shape=(12, 5),
                coordinates=(2, 1),
                mesh_shape=(3, 2),
            ),
            dtype=torch.bfloat16,
        ),
        f"{key}/bias": KeyRegistration(
            tensor_slice(
                (6,), (3,), global_shape=(9,), coordinates=(2,), mesh_shape=(4,)
            ),
            dtype=torch.float16,
        ),
    }


def _plus_timeout(signature: inspect.Signature) -> inspect.Signature:
    """``signature`` plus the keyword-only ``timeout_s=None`` that lookups add."""
    timeout = inspect.Parameter(
        "timeout_s",
        inspect.Parameter.KEYWORD_ONLY,
        default=None,
        annotation="float | None",
    )
    return signature.replace(parameters=[*signature.parameters.values(), timeout])


async def _assert_still_pending(*lookups: asyncio.Future[Any]) -> None:
    done, _ = await asyncio.wait(lookups, timeout=_STILL_PENDING_S)
    returned = [
        "cancelled" if lookup.cancelled() else lookup.exception() or lookup.result()
        for lookup in done
    ]
    assert not done, f"get_layouts returned early: {returned!r}"


@pytest.mark.asyncio
async def test_strategy_requires_initialization(make_control_plane) -> None:
    """Reject strategy lookup before initialization instead of waiting for it."""
    control_plane = await make_control_plane()

    with pytest.raises(Exception, match="init has not run"):
        await asyncio.wait_for(control_plane.strategy(), timeout=1)


@pytest.mark.asyncio
async def test_publishers_do_not_wait_and_requesters_wait_for_publishers(
    make_control_plane,
) -> None:
    """Return publisher registration immediately but gate requester lookup."""
    control_plane = await make_control_plane({"publisher/0", "publisher/1"})

    await asyncio.wait_for(
        control_plane.register_layout(
            "publisher/0", "model", _registration("model/first")
        ),
        timeout=1,
    )
    lookup = asyncio.ensure_future(control_plane.get_layouts("model"))
    await _assert_still_pending(lookup)
    await asyncio.wait_for(
        control_plane.register_layout(
            "publisher/1", "model", _registration("model/second")
        ),
        timeout=1,
    )
    layouts = await asyncio.wait_for(lookup, timeout=5)

    assert layouts == {
        "publisher/0": _registration("model/first"),
        "publisher/1": _registration("model/second"),
    }


@pytest.mark.asyncio
async def test_rejects_registration_from_a_nonpublisher(make_control_plane) -> None:
    """Restrict registration to publisher ranks, discarding a rejected layout.

    A key that every publisher has registered still refuses other ranks.
    """
    control_plane = await make_control_plane({"publisher"})

    with pytest.raises(Exception, match="not a publisher rank"):
        await control_plane.register_layout("requester", "model", _registration())

    await control_plane.register_layout("publisher", "model", _registration())
    with pytest.raises(Exception, match="not a publisher rank"):
        await asyncio.wait_for(
            control_plane.register_layout(
                "requester", "model", _registration("requester/w")
            ),
            timeout=1,
        )
    layouts = await asyncio.wait_for(
        control_plane.get_layouts("model", timeout_s=0), timeout=1
    )
    assert layouts == {"publisher": _registration()}


@pytest.mark.asyncio
async def test_reuses_completed_layouts(make_control_plane) -> None:
    """Return stored layouts, ignoring a repeated registration from any handle."""
    first, second, reader = await make_control_plane({"publisher"}, handles=3)
    await first.register_layout("publisher", "model", _registration())
    for handle in (second, first):
        await handle.register_layout(
            "publisher", "model", _registration("replacement/w")
        )

    for handle in (reader, first, second):
        layouts = await asyncio.wait_for(handle.get_layouts("model"), timeout=1)
        assert layouts == {"publisher": _registration()}


@pytest.mark.asyncio
async def test_a_key_completes_once_every_publisher_registers_it(
    make_control_plane,
) -> None:
    """Complete a key once every publisher registers it, in any order.

    Other keys, repeats and rejected ranks do not count, and a repeat keeps the first.
    """
    ranks = ("publisher/0", "publisher/1", "publisher/2")
    *publishers, requester = await make_control_plane(set(ranks), handles=4)
    layouts = {rank: _registration(f"model/{rank}") for rank in ranks}
    lookup = asyncio.ensure_future(requester.get_layouts("model"))
    # The missing publisher sorts between the registered ones and registers another key.
    for index, key in ((2, "model"), (1, "optimizer"), (0, "model")):
        rank = ranks[index]
        await asyncio.wait_for(
            publishers[index].register_layout(rank, key, layouts[rank]), timeout=1
        )
    # publisher/2 repeats from another handle before the key completes.
    await asyncio.wait_for(
        publishers[0].register_layout("publisher/2", "model", _registration("late/w")),
        timeout=1,
    )
    with pytest.raises(Exception, match="not a publisher rank"):
        await asyncio.wait_for(
            requester.register_layout("requester/0/0", "model", _registration()),
            timeout=1,
        )
    await _assert_still_pending(lookup)
    with pytest.raises(Exception, match=r"missing publishers: \['publisher/1'\]"):
        await asyncio.wait_for(requester.get_layouts("model", timeout_s=0), timeout=1)
    await asyncio.wait_for(
        publishers[1].register_layout("publisher/1", "model", layouts["publisher/1"]),
        timeout=1,
    )
    assert await asyncio.wait_for(lookup, timeout=5) == layouts


@pytest.mark.asyncio
async def test_timeout_preserves_layout_for_retry(make_control_plane) -> None:
    """Name only the missing publishers on timeout and allow lookup to retry."""
    control_plane = await make_control_plane({"publisher/0", "publisher/1"})
    with pytest.raises(
        Exception, match=r"missing publishers: \['publisher/0', 'publisher/1'\]"
    ):
        await asyncio.wait_for(
            control_plane.get_layouts("model", timeout_s=0), timeout=1
        )
    first, second = _registration("model/first"), _registration("model/second")
    await control_plane.register_layout("publisher/0", "model", first)

    loop = asyncio.get_running_loop()
    # The error reports the timeout the lookup waited, 0 when zero or less.
    for timeout_s, guard_s, reported in ((0, 1, 0), (-1, 1, 0), (0.1, 5, 0.1)):
        message = (
            f"timed out after {reported} seconds waiting for publisher layouts "
            "for 'model'; missing publishers: ['publisher/1']"
        )
        started = loop.time()
        with pytest.raises(Exception, match=re.escape(message)):
            await asyncio.wait_for(
                control_plane.get_layouts("model", timeout_s=timeout_s),
                timeout=guard_s,
            )
        elapsed = loop.time() - started
        assert elapsed >= timeout_s - _EARLY_S, "a positive timeout must wait"
        assert elapsed < max(timeout_s, 0) + _LATE_S, "a timeout must fire on time"

    retry = asyncio.ensure_future(control_plane.get_layouts("model", timeout_s=30))
    await _assert_still_pending(retry)
    await asyncio.wait_for(
        control_plane.register_layout("publisher/1", "model", second),
        timeout=1,
    )
    layouts = await asyncio.wait_for(retry, timeout=5)
    assert layouts == {"publisher/0": first, "publisher/1": second}


@pytest.mark.asyncio
async def test_default_timeout_names_missing_publishers(make_control_plane) -> None:
    """Fail a lookup without a timeout after the default, naming missing publishers."""
    control_plane = await make_control_plane(
        {"publisher/0", "publisher/1"}, default_timeout_s=0.2
    )
    await control_plane.register_layout("publisher/0", "model", _registration())

    message = (
        "timed out after 0.2 seconds waiting for publisher layouts for 'model'; "
        "missing publishers: ['publisher/1']"
    )
    loop = asyncio.get_running_loop()
    started = loop.time()
    with pytest.raises(Exception, match=re.escape(message)):
        await asyncio.wait_for(control_plane.get_layouts("model"), timeout=5)
    elapsed = loop.time() - started
    assert elapsed >= 0.2 - _EARLY_S, "the default timeout must wait"
    assert elapsed < 0.2 + _LATE_S, "the default timeout must fire on time"


@pytest.mark.asyncio
async def test_a_timeout_ends_only_its_own_lookup(make_control_plane) -> None:
    """Fail a lookup at its deadline, naming publishers missing then. Others wait on.

    A registration during the wait does not move the deadline.
    """
    publisher, patient_handle, impatient_handle = await make_control_plane(
        {"publisher/0", "publisher/1"}, handles=3
    )
    first, second = _registration("model/first"), _registration("model/second")
    patient = asyncio.ensure_future(patient_handle.get_layouts("model", timeout_s=30))
    expiring = asyncio.ensure_future(impatient_handle.get_layouts("model", timeout_s=4))
    await _assert_still_pending(patient, expiring)
    # A check and a short wait of the still-empty key time out while both lookups wait.
    for timeout_s, guard_s in ((0, 1), (0.1, 5)):
        with pytest.raises(
            Exception, match=r"missing publishers: \['publisher/0', 'publisher/1'\]"
        ):
            await asyncio.wait_for(
                impatient_handle.get_layouts("model", timeout_s=timeout_s),
                timeout=guard_s,
            )
    await _assert_still_pending(patient, expiring)
    loop = asyncio.get_running_loop()
    registering = loop.time()
    await asyncio.wait_for(
        publisher.register_layout("publisher/0", "model", first), timeout=1
    )

    message = (
        "timed out after 4 seconds waiting for publisher layouts for 'model'; "
        "missing publishers: ['publisher/1']"
    )
    with pytest.raises(Exception, match=re.escape(message)):
        await asyncio.wait_for(expiring, timeout=10)
    # About 3 s of the wait remained. A deadline the registration restarted needs 4 s.
    assert loop.time() - registering < 3.75, "the registration moved the deadline"
    await _assert_still_pending(patient)
    await asyncio.wait_for(
        publisher.register_layout("publisher/1", "model", second), timeout=1
    )
    layouts = await asyncio.wait_for(patient, timeout=5)
    assert layouts == {"publisher/0": first, "publisher/1": second}


@pytest.mark.asyncio
async def test_state_dict_namespaces_have_independent_layouts(
    make_control_plane,
) -> None:
    """Gather model and optimizer publisher layouts independently."""
    control_plane = await make_control_plane({"publisher"})
    model = asyncio.ensure_future(control_plane.get_layouts("model"))
    await _assert_still_pending(model)
    optimizer = asyncio.ensure_future(control_plane.get_layouts("optimizer"))
    await _assert_still_pending(model, optimizer)

    # The optimizer lookup must not queue behind the still-waiting model lookup.
    for namespace, lookup in (("optimizer", optimizer), ("model", model)):
        await asyncio.wait_for(
            control_plane.register_layout(
                "publisher", namespace, _registration(f"{namespace}/w")
            ),
            timeout=1,
        )
        layouts = await asyncio.wait_for(lookup, timeout=2)
        assert layouts == {"publisher": _registration(f"{namespace}/w")}

    # Completing one state dict leaves the other readable with its own layout.
    results = await asyncio.wait_for(
        asyncio.gather(
            *(
                control_plane.get_layouts(namespace, timeout_s=0)
                for namespace in ("model", "optimizer")
            )
        ),
        timeout=1,
    )
    assert results == [
        {"publisher": _registration("model/w")},
        {"publisher": _registration("optimizer/w")},
    ]


@pytest.mark.asyncio
async def test_every_waiting_requester_gets_layouts_from_other_handles(
    make_control_plane,
) -> None:
    """Release every requester's lookup once publishers register on other handles.

    More requesters wait than publishers register, and each owns the layouts it gets.
    """
    first, second, *requesters = await make_control_plane(
        {"publisher/0", "publisher/1"}, handles=5
    )

    await asyncio.wait_for(
        first.register_layout("publisher/0", "model", _registration("model/first")),
        timeout=1,
    )
    lookups = [asyncio.ensure_future(r.get_layouts("model")) for r in requesters]
    await _assert_still_pending(*lookups)
    await asyncio.wait_for(
        second.register_layout("publisher/1", "model", _registration("model/second")),
        timeout=1,
    )
    results = list(await asyncio.wait_for(asyncio.gather(*lookups), timeout=5))

    expected = {
        "publisher/0": _registration("model/first"),
        "publisher/1": _registration("model/second"),
    }
    assert results == [expected] * len(requesters)
    # A requester's edits reach neither another requester's layouts nor later lookups.
    assert results[0]["publisher/0"] is not results[2]["publisher/0"]
    results[0].pop("publisher/1")
    results[1]["publisher/0"] = {}
    assert results[2] == expected
    # Serving the waiting lookups leaves the layouts for later ones.
    for requester in requesters:
        layouts = await asyncio.wait_for(
            requester.get_layouts("model", timeout_s=0), timeout=1
        )
        assert layouts == expected


@pytest.mark.asyncio
async def test_a_cancelled_lookup_leaves_other_lookups_working(
    make_control_plane,
) -> None:
    """Cancel one waiting lookup: it ends cancelled at once.

    The last registration and the other lookups still work.
    """
    publisher, cancelled_handle, survivor_handle = await make_control_plane(
        {"publisher/0", "publisher/1"}, handles=3
    )
    first, second = _registration("model/first"), _registration("model/second")
    await asyncio.wait_for(
        publisher.register_layout("publisher/0", "model", first), timeout=1
    )
    cancelled = asyncio.ensure_future(
        cancelled_handle.get_layouts("model", timeout_s=30)
    )
    survivor = asyncio.ensure_future(survivor_handle.get_layouts("model", timeout_s=30))
    await _assert_still_pending(cancelled, survivor)
    cancelled.cancel()
    done, _ = await asyncio.wait({cancelled}, timeout=1)
    assert done and cancelled.cancelled(), f"the cancelled lookup ended {done!r}"

    await asyncio.wait_for(
        publisher.register_layout("publisher/1", "model", second), timeout=1
    )
    expected = {"publisher/0": first, "publisher/1": second}
    assert await asyncio.wait_for(survivor, timeout=5) == expected
    layouts = await asyncio.wait_for(
        cancelled_handle.get_layouts("model", timeout_s=0), timeout=1
    )
    assert layouts == expected


@pytest.mark.asyncio
async def test_a_handle_serves_calls_from_another_event_loop(
    make_control_plane,
) -> None:
    """Serve one handle from another thread's loop, waking lookups on both loops."""
    control_plane = await make_control_plane({"publisher/0", "publisher/1"})
    expected = {
        "publisher/0": _registration("model/first"),
        "publisher/1": _registration("model/second"),
    }
    # A real wait first, so any loop-bound state binds to this loop.
    with pytest.raises(Exception, match="missing publishers"):
        await asyncio.wait_for(
            control_plane.get_layouts("model", timeout_s=0.1), timeout=5
        )
    # Registrations on the other loop must also wake a lookup waiting on this one.
    waiting_here = asyncio.ensure_future(
        control_plane.get_layouts("model", timeout_s=30)
    )
    registering: list[float] = []
    outcome: list[Any] = []

    async def on_another_loop() -> Registrations:
        lookup = asyncio.ensure_future(control_plane.get_layouts("model", timeout_s=30))
        await _assert_still_pending(lookup)
        for rank, registration in expected.items():
            # The last stamp precedes the registration that completes the key.
            registering.append(time.monotonic())
            await asyncio.wait_for(
                control_plane.register_layout(rank, "model", registration), timeout=1
            )
        return await asyncio.wait_for(lookup, timeout=5)

    def run() -> None:
        try:
            outcome.append(asyncio.run(on_another_loop()))
        except BaseException as error:
            outcome.append(error)

    # TitanRL publishes the same way: asyncio.run on a worker thread. This one is a
    # daemon outside this loop's executor, so only the plane can wake this loop, and a
    # thread stuck in the plane holds up neither this loop's teardown nor exit.
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        layouts = await asyncio.wait_for(waiting_here, timeout=10)
    except asyncio.TimeoutError:
        raise AssertionError(f"this loop's lookup never woke: {outcome!r}") from None
    assert layouts == expected
    # Only the guard's timer could wake this loop otherwise, 10 s after it slept.
    assert time.monotonic() - registering[-1] < 5, "the plane left this loop asleep"
    deadline = time.monotonic() + 15
    while thread.is_alive() and time.monotonic() < deadline:
        await asyncio.sleep(0.01)
    assert outcome == [expected], f"the other loop did not finish: {outcome!r}"
    layouts = await asyncio.wait_for(
        control_plane.get_layouts("model", timeout_s=0), timeout=1
    )
    assert layouts == expected


@pytest.mark.asyncio
async def test_zero_timeout_returns_completed_layouts(make_control_plane) -> None:
    """Check once with a zero or negative timeout and return every publisher's layout."""
    registrations = {
        "publisher/0": _registration("model/first"),
        "publisher/1": _registration("model/second"),
    }
    requester, *publishers = await make_control_plane(set(registrations), handles=3)
    # Publishers register at once, each on its own handle, as ranks do.
    await asyncio.wait_for(
        asyncio.gather(
            *(
                handle.register_layout(rank, "model", registration)
                for handle, (rank, registration) in zip(
                    publishers, registrations.items()
                )
            )
        ),
        timeout=1,
    )

    for timeout_s in (0, -1):
        layouts = await asyncio.wait_for(
            requester.get_layouts("model", timeout_s=timeout_s),
            timeout=1,
        )
        assert layouts == registrations


@pytest.mark.asyncio
async def test_nan_timeout_is_refused(make_control_plane) -> None:
    """Refuse a NaN timeout at once, also for complete layouts."""
    control_plane = await make_control_plane({"publisher"})
    with pytest.raises(Exception, match="must not be NaN"):
        await asyncio.wait_for(
            control_plane.get_layouts("nan", timeout_s=math.nan), timeout=1
        )
    await control_plane.register_layout("publisher", "first", _registration())
    # Complete layouts do not exempt a NaN lookup from the refusal.
    with pytest.raises(Exception, match="must not be NaN"):
        await asyncio.wait_for(
            control_plane.get_layouts("first", timeout_s=math.nan), timeout=1
        )


@pytest.mark.asyncio
async def test_layout_registration_requires_initialization(make_control_plane) -> None:
    """Reject layout registration before initialization instead of waiting for it."""
    control_plane = await make_control_plane()

    with pytest.raises(Exception, match="init has not run"):
        await asyncio.wait_for(
            control_plane.register_layout("publisher", "model", _registration()),
            timeout=1,
        )


@pytest.mark.asyncio
async def test_layout_lookup_requires_initialization(make_control_plane) -> None:
    """Reject layout lookup before initialization instead of waiting for it.

    Each call takes the interface's parameters, which callers pass by keyword, and
    the lookup adds only a keyword-only timeout.
    """
    control_plane = await make_control_plane()
    for name in sorted(ControlPlane.__abstractmethods__):
        expected = inspect.signature(getattr(ControlPlane, name))
        if name == "get_layouts":
            expected = _plus_timeout(expected)
        actual = inspect.signature(getattr(type(control_plane), name))
        assert [(p.name, p.kind, p.default) for p in actual.parameters.values()] == [
            (p.name, p.kind, p.default) for p in expected.parameters.values()
        ], name

    with pytest.raises(Exception, match="init has not run"):
        await asyncio.wait_for(control_plane.get_layouts("model"), timeout=1)


@pytest.mark.asyncio
async def test_strategy_returns_the_registered_strategy(make_control_plane) -> None:
    """Return a strategy of the registered type and transport, resolving its volumes."""
    volumes = [("0", "host-0"), ("1", "host-1")]
    for transport in (TransportType.MonarchRPC, TransportType.Gloo):
        registered = ts.TorchStoreStrategy(transport)
        await registered.set_storage_volumes(VolumeMesh(volumes))
        control_plane = await make_control_plane({"publisher"}, registered)
        strategy = await control_plane.strategy()
        assert type(strategy) is ts.TorchStoreStrategy
        assert strategy.default_transport_type == transport
        for rank, (volume_id, hostname) in enumerate(volumes):
            volume = strategy.get_storage_volume(volume_id)
            assert volume.volume == ("slice", {"procs": rank}), volume_id
            assert volume.volume_hostname == hostname, volume_id


@pytest.mark.asyncio
async def test_open_stores_keep_separate_state(make_control_plane) -> None:
    """Keep each open store's publishers, strategy and layouts apart."""
    first = await make_control_plane(
        {"publisher/0"}, ts.TorchStoreStrategy(TransportType.MonarchRPC)
    )
    second = await make_control_plane(
        {"publisher/0", "publisher/1"}, ts.TorchStoreStrategy(TransportType.Gloo)
    )
    await first.register_layout("publisher/0", "model", _registration("model/first"))

    with pytest.raises(
        Exception, match=r"missing publishers: \['publisher/0', 'publisher/1'\]"
    ):
        await asyncio.wait_for(second.get_layouts("model", timeout_s=0), timeout=1)
    strategy = await asyncio.wait_for(first.strategy(), timeout=1)
    assert strategy.default_transport_type == TransportType.MonarchRPC
    layouts = await asyncio.wait_for(first.get_layouts("model", timeout_s=0), timeout=1)
    assert layouts == {"publisher/0": _registration("model/first")}


@pytest.mark.asyncio
async def test_control_plane_mirrors_the_coordinator_endpoints() -> None:
    """Take the coordinator's endpoints as the interface, less the lookup's timeout.

    Monarch checks a call's arguments against its endpoint's signature before sending.
    """
    calls = {
        "init": ({"publisher"}, ts.TorchStoreStrategy(TransportType.MonarchRPC)),
        "register_layout": ("publisher", "model", _registration()),
        "get_layouts": ("model",),
        "strategy": (),
    }
    assert ControlPlane.__abstractmethods__ == set(calls)
    coordinator = await _coordinator()
    for name, args in calls.items():
        names = list(inspect.signature(getattr(ControlPlane, name)).parameters)[1:]
        endpoint = getattr(coordinator, name)
        # Each takes the interface's parameters by name or in order, and no others.
        await asyncio.wait_for(endpoint.call_one(**dict(zip(names, args))), timeout=5)
        await asyncio.wait_for(endpoint.call_one(*args), timeout=5)
        for given, missing in enumerate(names):
            with pytest.raises(TypeError, match=f"required argument: '{missing}'"):
                endpoint.call_one(*args[:given])
        with pytest.raises(TypeError, match="too many positional arguments"):
            endpoint.call_one(*args, None)
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            endpoint.call_one(*args, unknown=None)
    # The lookup's timeout is keyword-only, as the positional call above shows.
    layouts = await asyncio.wait_for(
        coordinator.get_layouts.call_one("model", timeout_s=None), timeout=5
    )
    assert layouts == {"publisher": _registration()}


def test_coordinator_keeps_its_endpoint_kinds() -> None:
    """Keep ``init`` and ``strategy`` as endpoints, and the layout calls concurrent."""

    async def serial_body(self) -> None:
        pass

    async def concurrent_body(self) -> None:
        pass

    serial = type(endpoint(serial_body))
    concurrent = type(concurrent_endpoint(concurrent_body))
    assert serial is not concurrent
    kinds = {
        name: type(RoutingCoordinator.__dict__[name])
        for name in ControlPlane.__abstractmethods__
    }
    assert kinds == {
        "init": serial,
        "strategy": serial,
        "register_layout": concurrent,
        "get_layouts": concurrent,
    }


@pytest.mark.asyncio
async def test_coordinator_delegates_each_endpoint_to_its_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward each endpoint's arguments, result and error through one registry."""
    registry_class = create_autospec(LayoutRegistry)
    registry = registry_class.return_value
    # The coordinator actor runs in this process and builds its registry by this name.
    monkeypatch.setattr(coordinator_module, "LayoutRegistry", registry_class)
    coordinator = await _coordinator()
    strategy = ts.TorchStoreStrategy(TransportType.MonarchRPC)
    layouts = {"publisher": _registration()}
    registry.strategy.return_value = strategy
    registry.get_layouts.return_value = layouts
    calls = {
        "init": (({"publisher"}, strategy), {}),
        "strategy": ((), {}),
        "register_layout": (("publisher", "model", _registration()), {}),
        "get_layouts": (("model",), {"timeout_s": 7}),
    }

    results = {
        name: await asyncio.wait_for(
            getattr(coordinator, name).call_one(*args, **kwargs), timeout=5
        )
        for name, (args, kwargs) in calls.items()
    }
    # Arguments and results cross the actor as copies, so strategies match by value.
    registry.init.assert_awaited_once_with({"publisher"}, ANY)
    for copy in (registry.init.await_args.args[1], results["strategy"]):
        assert type(copy) is ts.TorchStoreStrategy
        assert copy.default_transport_type == TransportType.MonarchRPC
    registry.strategy.assert_awaited_once_with()
    registry.register_layout.assert_awaited_once_with(
        "publisher", "model", _registration()
    )
    registry.get_layouts.assert_awaited_once_with("model", timeout_s=7)
    assert (results["init"], results["register_layout"]) == (None, None)
    assert results["get_layouts"] == layouts
    # Without a timeout, the registry applies its own default.
    await asyncio.wait_for(coordinator.get_layouts.call_one("model"), timeout=5)
    registry.get_layouts.assert_awaited_with("model", timeout_s=None)

    for name, (args, kwargs) in calls.items():
        error = RuntimeError(f"{name} failed")
        getattr(registry, name).side_effect = error
        with pytest.raises(ActorError) as raised:
            await asyncio.wait_for(
                getattr(coordinator, name).call_one(*args, **kwargs), timeout=5
            )
        # The registry's own error crosses as a copy, not wrapped in another.
        assert repr(raised.value.exception) == repr(error)
    # Both init calls reset the coordinator's one registry in place.
    registry_class.assert_called_once_with()


@pytest.mark.asyncio
async def test_registry_store_exit_fails_when_its_loop_does_not_stop() -> None:
    """Fail the registry store's exit at its time limit when the host loop is stuck."""
    release = threading.Event()
    # Without the limit, exit waits for this timer to free the loop, then passes.
    unblock = threading.Timer(5, release.set)
    unblock.start()
    try:
        with pytest.raises(AssertionError, match="host loop did not stop"):
            async with registry_store(stop_timeout_s=0.2) as connect:
                handle = await connect()
                handle._loop.call_soon_threadsafe(release.wait)
    finally:
        release.set()
        unblock.cancel()


@pytest.mark.asyncio
async def test_monarch_adapter_passes_actor_errors_through() -> None:
    """Surface each coordinator error unchanged, as ``ActorError``."""
    async with _monarch_store() as connect:
        control_plane = await connect()
        with pytest.raises(ActorError, match="init has not run"):
            await asyncio.wait_for(control_plane.strategy(), timeout=1)
        await control_plane.init(
            {"publisher"}, ts.TorchStoreStrategy(TransportType.MonarchRPC)
        )
        with pytest.raises(ActorError, match="not a publisher rank"):
            await control_plane.register_layout("requester", "model", _registration())
        with pytest.raises(ActorError, match="missing publishers"):
            await asyncio.wait_for(
                control_plane.get_layouts("model", timeout_s=0), timeout=1
            )
        with pytest.raises(
            ActorError, match="ValueError: get_layouts timeout_s must not be NaN"
        ):
            await asyncio.wait_for(
                control_plane.get_layouts("model", timeout_s=math.nan), timeout=1
            )


@pytest.mark.asyncio
async def test_monarch_second_init_replaces_publishers_and_clears_layouts() -> None:
    """Pin the coordinator's second init: new publishers and strategy, every layout dropped.

    A second init breaks the interface's run-once rule, so this pins existing behavior. A
    repeated publisher set starts over too, and lookups already waiting keep the old layouts.
    """
    async with _monarch_store() as connect:
        first, second = await connect(), await connect()
        publishers = {"publisher/0", "publisher/1"}
        await first.init(publishers, ts.TorchStoreStrategy(TransportType.MonarchRPC))
        for rank in sorted(publishers):
            await first.register_layout(rank, "model", _registration())
        await second.init(publishers, ts.TorchStoreStrategy(TransportType.SharedMemory))
        with pytest.raises(
            ActorError, match=r"missing publishers: \['publisher/0', 'publisher/1'\]"
        ):
            await asyncio.wait_for(first.get_layouts("model", timeout_s=0), timeout=1)
        strategy = await asyncio.wait_for(first.strategy(), timeout=1)
        assert strategy.default_transport_type == TransportType.SharedMemory
        for rank in sorted(publishers):
            await first.register_layout(rank, "model", _registration())
        await second.init(
            {"publisher/1", "publisher/2"}, ts.TorchStoreStrategy(TransportType.Gloo)
        )
        with pytest.raises(
            ActorError, match=r"missing publishers: \['publisher/1', 'publisher/2'\]"
        ):
            await asyncio.wait_for(first.get_layouts("model", timeout_s=0), timeout=1)
        strategy = await asyncio.wait_for(first.strategy(), timeout=1)
        assert strategy.default_transport_type == TransportType.Gloo


def test_file_store_default_timeout_is_300_seconds() -> None:
    """Keep the interface's five-minute wait for lookups without a timeout."""
    assert file_store_control_plane._LAYOUT_REGISTRATION_TIMEOUT_S == 300


@pytest.mark.asyncio
async def test_file_store_second_init_refuses_the_directory() -> None:
    """Refuse a second init from any handle, keeping the first publishers and strategy.

    The first init makes the directory with mode 0o700.
    """
    async with _file_store() as connect:
        first, second = await connect(), await connect()
        await first.init(
            {"publisher/0", "publisher/1"},
            ts.TorchStoreStrategy(TransportType.MonarchRPC),
        )
        assert stat.S_IMODE(os.stat(first._path).st_mode) == 0o700
        await first.register_layout("publisher/0", "model", _registration())
        with pytest.raises(FileExistsError):
            await second.init(
                {"publisher/1", "publisher/2"},
                ts.TorchStoreStrategy(TransportType.Gloo),
            )
        with pytest.raises(
            TimeoutError, match=r"missing publishers: \['publisher/1'\]"
        ):
            await asyncio.wait_for(second.get_layouts("model", timeout_s=0), timeout=1)
        strategy = await asyncio.wait_for(second.strategy(), timeout=1)
        assert strategy.default_transport_type == TransportType.MonarchRPC


@pytest.mark.asyncio
async def test_file_store_close_refuses_later_calls() -> None:
    """Refuse every call on a closed handle, leaving other handles on it serving."""
    async with _file_store() as connect:
        closed, other = await connect(), await connect()
        strategy = ts.TorchStoreStrategy(TransportType.MonarchRPC)
        await closed.init({"publisher"}, strategy)
        # A handle that already served a call refuses it once closed.
        await asyncio.wait_for(closed.strategy(), timeout=1)
        await closed.close()
        calls = (
            lambda: closed.init({"publisher"}, strategy),
            closed.strategy,
            lambda: closed.register_layout("publisher", "model", _registration()),
            lambda: closed.get_layouts("model", timeout_s=0),
        )
        for make_call in calls:
            with pytest.raises(RuntimeError, match="is closed"):
                await asyncio.wait_for(make_call(), timeout=1)
        served = await asyncio.wait_for(other.strategy(), timeout=1)
        assert served.default_transport_type == TransportType.MonarchRPC


@pytest.mark.asyncio
async def test_file_store_init_failing_before_ready_leaves_it_uninitialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail fast with "init has not run" on every handle until init writes ready."""
    write_once = FileStoreControlPlane._write_once

    def failing_write_once(self: FileStoreControlPlane, key: str, value: bytes) -> None:
        if key == file_store_control_plane._STRATEGY:
            raise OSError("disk full")
        write_once(self, key, value)

    monkeypatch.setattr(FileStoreControlPlane, "_write_once", failing_write_once)
    async with _file_store() as connect:
        writer, reader = await connect(), await connect()
        with pytest.raises(OSError, match="disk full"):
            await writer.init(
                {"publisher"}, ts.TorchStoreStrategy(TransportType.MonarchRPC)
            )
        calls = (
            reader.strategy,
            lambda: reader.register_layout("publisher", "model", _registration()),
            lambda: reader.get_layouts("model", timeout_s=0),
        )
        for make_call in calls:
            with pytest.raises(Exception, match="init has not run"):
                await asyncio.wait_for(make_call(), timeout=1)


@pytest.mark.asyncio
async def test_file_store_calls_run_off_the_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run every file call on a worker thread, so slow files never block the loop."""
    loop_thread = threading.get_ident()
    on_loop: list[str] = []
    directory = FileStoreControlPlane._directory

    def recording_directory(self: FileStoreControlPlane) -> str:
        if threading.get_ident() == loop_thread:
            on_loop.append(inspect.stack(context=0)[1].function)
        return directory(self)

    monkeypatch.setattr(FileStoreControlPlane, "_directory", recording_directory)
    async with _file_store() as connect:
        publisher, requester = await connect(), await connect()
        await publisher.init(
            {"publisher"}, ts.TorchStoreStrategy(TransportType.MonarchRPC)
        )
        await requester.strategy()
        lookup = asyncio.ensure_future(requester.get_layouts("model"))
        await _assert_still_pending(lookup)
        await publisher.register_layout("publisher", "model", _registration())
        layouts = await asyncio.wait_for(lookup, timeout=5)
        assert layouts == {"publisher": _registration()}
    assert on_loop == []


@pytest.mark.asyncio
async def test_file_store_fails_fast_without_its_parent_directory(
    tmp_path: Path,
) -> None:
    """Fail at once on a path whose parent directory is missing."""
    plane = FileStoreControlPlane(str(tmp_path / "missing" / "control_plane"))
    with pytest.raises(FileNotFoundError):
        await asyncio.wait_for(
            plane.init({"publisher"}, ts.TorchStoreStrategy(TransportType.MonarchRPC)),
            timeout=1,
        )
    with pytest.raises(RuntimeError, match="init has not run"):
        await asyncio.wait_for(plane.strategy(), timeout=1)
    assert not (tmp_path / "missing").exists()


# The repository root, so a child process imports this directory's helper.
_ROOT = str(Path(__file__).resolve().parents[2])
_FILE_STORE_CHILD = """
import asyncio, os, resource, signal, sys

sys.path.insert(0, sys.argv[1])
from tests.routing.file_store_control_plane import FileStoreControlPlane

path, mode = sys.argv[2:]


async def main():
    plane = FileStoreControlPlane(path)
    # The handle serves a call first, as a rank's has when it dies or fills the disk.
    await plane.strategy()
    if mode == "killed":
        # Die after writing the value, before linking it into place.
        os.link = lambda *args: os.kill(os.getpid(), signal.SIGKILL)
    else:
        # A full disk: a file holds one byte, so a write lands partly, then fails.
        _, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
        resource.setrlimit(resource.RLIMIT_FSIZE, (1, hard))
    try:
        await plane.register_layout("publisher/0", "model", {})
    except OSError as error:
        print("write failed:", error.strerror)
    await plane.close()
    print("closed")


asyncio.run(main())
"""


def _run_file_store_child(path: str, mode: str) -> subprocess.CompletedProcess[str]:
    """Register a layout on ``path`` from a child process that ``mode`` breaks."""
    return subprocess.run(
        [sys.executable, "-c", _FILE_STORE_CHILD, _ROOT, path, mode],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


@pytest.mark.timeout(240)
@pytest.mark.asyncio
async def test_file_store_ignores_a_writer_killed_mid_write() -> None:
    """Leave no partial value when a writer dies before linking its value into place."""
    async with _file_store() as connect:
        driver, requester = await connect(), await connect()
        await driver.init(
            {"publisher/0"}, ts.TorchStoreStrategy(TransportType.MonarchRPC)
        )
        child = await asyncio.to_thread(_run_file_store_child, driver._path, "killed")
        assert child.returncode == -signal.SIGKILL, child.stdout + child.stderr
        # Its value stays in a temporary file that no call reads.
        assert len(os.listdir(driver._path)) == 4
        with pytest.raises(
            TimeoutError, match=r"missing publishers: \['publisher/0'\]"
        ):
            await asyncio.wait_for(
                requester.get_layouts("model", timeout_s=0.1), timeout=5
            )
        publisher = await connect()
        await publisher.register_layout("publisher/0", "model", _registration())
        layouts = await asyncio.wait_for(requester.get_layouts("model"), timeout=5)
        assert layouts == {"publisher/0": _registration()}


@pytest.mark.timeout(240)
@pytest.mark.asyncio
async def test_file_store_write_errors_raise_and_leave_nothing() -> None:
    """Raise a full disk's error from the write, leave no file, and close cleanly."""
    async with _file_store() as connect:
        driver = await connect()
        await driver.init(
            {"publisher/0"}, ts.TorchStoreStrategy(TransportType.MonarchRPC)
        )
        written = sorted(os.listdir(driver._path))
        child = await asyncio.to_thread(_run_file_store_child, driver._path, "full")
        assert child.returncode == 0, child.stdout + child.stderr
        assert child.stdout.splitlines() == ["write failed: File too large", "closed"]
        assert sorted(os.listdir(driver._path)) == written
