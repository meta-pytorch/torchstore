# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import inspect
import re
import threading
import time
import uuid
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Collection,
    Coroutine,
    Mapping,
)
from typing import Any, Protocol
from unittest.mock import ANY, create_autospec

import pytest
import pytest_asyncio
import torch
import torchstore as ts
import torchstore.routing.coordinator as coordinator_module
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
from .utils import tensor_slice

pytestmark = pytest.mark.timeout(60)

# A lookup that does not wait returns well within this. A correct one cannot.
_STILL_PENDING_S = 0.5


class TimedControlPlane(Protocol):
    """The control-plane interface, its lookup also taking a keyword-only timeout.

    ``timeout_s=None`` waits the plane's default and zero or less checks once. A lookup
    that times out raises ``timed out after <wait> seconds waiting for publisher layouts
    for <key>; missing publishers: <sorted ranks>``, the wait 0 for zero or less.
    """

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


class _HostedRegistry(ControlPlane):
    """Runs each call as a task on the registry's loop, like a concurrent endpoint."""

    def __init__(
        self, registry: LayoutRegistry, loop: asyncio.AbstractEventLoop
    ) -> None:
        self._registry = registry
        self._loop = loop

    async def _on_host(self, coroutine: Coroutine[Any, Any, Any]) -> Any:
        return await asyncio.wrap_future(
            asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        )

    async def init(
        self, publishers: Collection[str], strategy: ts.TorchStoreStrategy
    ) -> None:
        await self._on_host(self._registry.init(publishers, strategy))

    async def strategy(self) -> ts.TorchStoreStrategy:
        return await self._on_host(self._registry.strategy())

    async def register_layout(
        self, rank: str, key: str, registrations: Mapping[str, KeyRegistration]
    ) -> None:
        await self._on_host(self._registry.register_layout(rank, key, registrations))

    async def get_layouts(
        self, key: str, *, timeout_s: float | None = None
    ) -> Registrations:
        layouts = await self._on_host(
            self._registry.get_layouts(key, timeout_s=timeout_s)
        )
        # Callers own their layouts, as with Monarch's pickled replies.
        return {rank: dict(registrations) for rank, registrations in layouts.items()}


@contextlib.asynccontextmanager
async def _registry_store() -> AsyncIterator[Callable[[], Awaitable[_HostedRegistry]]]:
    """Host one ``LayoutRegistry`` on a thread's own event loop, without Monarch.

    Exit fails if the host loop has not stopped within 10 s.
    """
    hosting: concurrent.futures.Future = concurrent.futures.Future()

    async def host() -> None:
        stop = asyncio.Event()
        hosting.set_result((asyncio.get_running_loop(), stop))
        await stop.wait()

    thread = threading.Thread(target=asyncio.run, args=(host(),), daemon=True)
    thread.start()
    loop, stop = await asyncio.wrap_future(hosting)
    registry = LayoutRegistry()

    async def connect() -> _HostedRegistry:
        return _HostedRegistry(registry, loop)

    try:
        yield connect
    finally:
        # ``asyncio.run`` cancels any call still running on the host loop.
        loop.call_soon_threadsafe(stop.set)
        await asyncio.to_thread(thread.join, 10)
        assert not thread.is_alive(), "the registry's host loop did not stop"


@pytest_asyncio.fixture(
    params=[
        pytest.param(_monarch_store, id="monarch"),
        pytest.param(_registry_store, id="registry"),
    ]
)
async def make_control_plane(
    request: pytest.FixtureRequest,
) -> AsyncIterator[Callable[..., Awaitable[Any]]]:
    """Open a fresh store per call and return a client handle, or ``handles`` of them.

    Handles connect on their own, as ranks do. Store contexts exit after the test,
    though Monarch's coordinator outlives its context. Error types differ, so match
    messages.
    """
    new_store = request.param
    async with contextlib.AsyncExitStack() as stack:

        async def make(
            publishers: Collection[str] | None = None,
            strategy: ts.TorchStoreStrategy | None = None,
            *,
            handles: int = 1,
        ) -> Any:
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
    retry = asyncio.ensure_future(control_plane.get_layouts("model", timeout_s=30))
    await _assert_still_pending(retry)
    await asyncio.wait_for(
        control_plane.register_layout("publisher/1", "model", second),
        timeout=1,
    )
    layouts = await asyncio.wait_for(retry, timeout=5)
    assert layouts == {"publisher/0": first, "publisher/1": second}


@pytest.mark.asyncio
async def test_a_timeout_reports_the_wait_it_took(make_control_plane) -> None:
    """Wait out a positive timeout, and report it on expiry, or 0 for zero or less."""
    control_plane = await make_control_plane({"publisher/0", "publisher/1"})
    await control_plane.register_layout("publisher/0", "model", _registration())
    loop = asyncio.get_running_loop()
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
        assert loop.time() - started >= timeout_s / 2, "a positive timeout must wait"


@pytest.mark.asyncio
async def test_a_timeout_ends_only_its_own_lookup(make_control_plane) -> None:
    """Fail a lookup at its deadline, naming publishers missing then. Others wait on."""
    publisher, patient_handle, impatient_handle = await make_control_plane(
        {"publisher/0", "publisher/1"}, handles=3
    )
    first, second = _registration("model/first"), _registration("model/second")
    patient = asyncio.ensure_future(patient_handle.get_layouts("model", timeout_s=30))
    expiring = asyncio.ensure_future(
        impatient_handle.get_layouts("model", timeout_s=1.5)
    )
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
    await asyncio.wait_for(
        publisher.register_layout("publisher/0", "model", first), timeout=1
    )

    message = (
        "timed out after 1.5 seconds waiting for publisher layouts for 'model'; "
        "missing publishers: ['publisher/1']"
    )
    with pytest.raises(Exception, match=re.escape(message)):
        await asyncio.wait_for(expiring, timeout=10)
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

    More requesters wait than publishers register.
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
    results = await asyncio.wait_for(asyncio.gather(*lookups), timeout=5)

    expected = {
        "publisher/0": _registration("model/first"),
        "publisher/1": _registration("model/second"),
    }
    assert results == [expected] * len(requesters)
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
    """Reject layout lookup before initialization instead of waiting for it."""
    control_plane = await make_control_plane()

    with pytest.raises(Exception, match="init has not run"):
        await asyncio.wait_for(control_plane.get_layouts("model"), timeout=1)


@pytest.mark.asyncio
async def test_each_call_takes_the_interface_parameters(make_control_plane) -> None:
    """Take each interface call's parameters, which callers pass by keyword.

    The lookup adds only a keyword-only timeout.
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
    # The coordinator built its one registry when it started.
    registry_class.assert_called_once_with()


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
