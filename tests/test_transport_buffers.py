# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the base transport buffer's cleanup after its volume call, its refusal hooks and context caches."""

import asyncio
import logging
import threading
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torchstore.strategy import StorageVolumeRef
from torchstore.transport import TransportType
from torchstore.transport.buffers import (
    TransportBuffer,
    TransportCache,
    TransportContext,
)
from torchstore.transport.monarch_rpc import MonarchRPCTransportBuffer
from torchstore.transport.types import Request

from .volume_meshes import BOUND_S, bounded, FakeVolume

pytestmark = pytest.mark.timeout(60)


class _Cleanup(MonarchRPCTransportBuffer):
    """Counts its drops, each failing with ``fail`` if given or waiting for a cancel if ``block``."""

    def __init__(self, ref, *, fail=None, block=False) -> None:
        super().__init__(ref)
        self.fail, self.block = fail, block
        self.drops = 0
        self.in_drop = asyncio.Event()

    async def drop(self) -> None:
        await super().drop()
        self.drops += 1
        self.in_drop.set()
        if self.block:
            await asyncio.Event().wait()
        if self.fail is not None:
            raise self.fail("drop failed")


class _Minimal(TransportBuffer):
    """Defines only what a client's calls need and so has no ``drop``."""

    async def _handle_storage_volume_response(self, requests, transport_buffer):
        return [None for _ in requests]


class _DropMixin:
    """Counts the drops of a buffer that lists it after its transport class."""

    dropped = 0

    async def drop(self) -> None:
        self.dropped += 1


class _MixedIn(_Minimal, _DropMixin):
    pass


class _Delegating(_Minimal):
    """Reaches what it lacks (``drop`` included) through ``__getattr__`` on ``inner``."""

    def __init__(self, ref, inner) -> None:
        super().__init__(ref)
        self.inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(vars(self)["inner"], name)


def _volume(outcome: str) -> Any:
    """A volume that refuses, serves or holds every call, setting ``arrived`` once one reaches it."""
    arrived = asyncio.Event()

    async def call(buffer, requests):
        arrived.set()
        if outcome == "refused":
            raise KeyError("refused")
        if outcome == "held":
            await asyncio.Event().wait()
        return buffer

    endpoint = SimpleNamespace(call=call, call_one=call)
    return SimpleNamespace(put=endpoint, get=endpoint, arrived=arrived)


def _ref(volume: Any) -> StorageVolumeRef:
    return StorageVolumeRef(
        volume, "volume", TransportContext(), TransportType.MonarchRPC
    )


def _call(buffer, direction):
    if direction == "put":
        return buffer.put_to_storage_volume([Request.from_tensor("k", torch.ones(2))])
    return buffer.get_from_storage_volume([Request(key="k")])


def _timed_steps(caplog) -> list[str]:
    """The steps each call's latency tracker logged, such as ``put:drop`` and ``put``."""
    return [
        record.getMessage().split(" took ")[0]
        for record in caplog.records
        if record.name == "root" and " took " in record.getMessage()
    ]


@pytest.fixture(autouse=True)
def _capture_logs(caplog):
    # The log asserts hold whatever level TORCHSTORE_LOG_LEVEL gives the root logger.
    caplog.set_level(logging.DEBUG)


def _assert_drop_failure_logged(caplog) -> None:
    [record] = [r for r in caplog.records if r.name == "torchstore.transport.buffers"]
    assert record.levelno == logging.WARNING
    assert record.getMessage() == "_Cleanup.drop() failed after its call raised"
    assert record.exc_info is not None and record.exc_info[0] is ValueError


@pytest.mark.parametrize("direction", ["put", "get"])
@pytest.mark.asyncio
async def test_a_refused_call_raises_its_own_error_over_its_failing_drop(
    caplog, direction
):
    """A refused call raises the refusal, and logs its drop's error at WARNING with no drop timing."""
    refused = _Cleanup(_ref(_volume("refused")), fail=ValueError)
    with pytest.raises(KeyError, match="refused"):
        await bounded(_call(refused, direction))
    _assert_drop_failure_logged(caplog)
    assert refused.drops == 1
    assert not {f"{direction}:drop", direction} & set(_timed_steps(caplog))


@pytest.mark.parametrize("direction", ["put", "get"])
@pytest.mark.asyncio
async def test_a_refused_call_whose_drop_returns_logs_its_drop_timing(
    caplog, direction
):
    """A refused call whose drop returns raises the refusal, drops once and logs its drop timing."""
    refused = _Cleanup(_ref(_volume("refused")))
    with pytest.raises(KeyError, match="refused"):
        await bounded(_call(refused, direction))
    assert refused.drops == 1
    assert _timed_steps(caplog)[-2:] == [f"{direction}:drop", direction]


@pytest.mark.parametrize("direction", ["put", "get"])
@pytest.mark.asyncio
async def test_a_served_call_raises_its_failing_drops_error(direction):
    """A call its volume served raises its drop's error."""
    served = _Cleanup(_ref(_volume("served")), fail=ValueError)
    with pytest.raises(ValueError, match="drop failed"):
        await bounded(_call(served, direction))
    assert served.drops == 1


@pytest.mark.parametrize("direction", ["put", "get"])
@pytest.mark.asyncio
async def test_a_call_cancelled_at_its_volume_ends_cancelled_over_its_failing_drop(
    caplog, direction
):
    """A call cancelled while its volume holds it ends cancelled, though its drop runs and fails."""
    volume = _volume("held")
    cancelled = _Cleanup(_ref(volume), fail=ValueError)
    calling = asyncio.ensure_future(_call(cancelled, direction))
    await bounded(volume.arrived.wait())
    calling.cancel()
    with pytest.raises(asyncio.CancelledError):
        await bounded(calling)
    _assert_drop_failure_logged(caplog)
    assert cancelled.drops == 1


@pytest.mark.parametrize("direction", ["put", "get"])
@pytest.mark.asyncio
async def test_a_cancel_during_a_refused_calls_drop_propagates(direction):
    """A cancel that lands while a refused call drops propagates in place of the refusal."""
    blocked = _Cleanup(_ref(_volume("refused")), block=True)
    calling = asyncio.ensure_future(_call(blocked, direction))
    await bounded(blocked.in_drop.wait())
    calling.cancel()
    with pytest.raises(asyncio.CancelledError):
        await bounded(calling)


@pytest.mark.parametrize("direction", ["put", "get"])
@pytest.mark.asyncio
async def test_a_call_through_a_buffer_without_drop_returns_and_logs_its_drop_step(
    caplog, direction
):
    """A call through a buffer that has no ``drop`` returns and logs its drop step's timing."""
    await bounded(_call(_Minimal(_ref(_volume("served"))), direction))
    assert _timed_steps(caplog)[-2:] == [f"{direction}:drop", direction]


@pytest.mark.parametrize("source", ["mixin", "delegation"])
@pytest.mark.parametrize("direction", ["put", "get"])
@pytest.mark.asyncio
async def test_a_call_runs_a_drop_its_buffer_gets_from_a_later_mixin_or_through_getattr(
    direction, source
):
    """Run a ``drop`` from a mixin listed after ``TransportBuffer`` or from ``__getattr__`` as main did."""
    ref = _ref(_volume("served"))
    cleanup = _DropMixin()
    buffer = _MixedIn(ref) if source == "mixin" else _Delegating(ref, cleanup)
    await bounded(_call(buffer, direction))
    assert (buffer if source == "mixin" else cleanup).dropped == 1


@pytest.mark.parametrize(
    "direction, stored, error, match",
    [
        ("put", {"obj": "an object"}, AssertionError, "Existing data is an object"),
        ("get", None, KeyError, "'k' not found"),
    ],
    ids=["put", "get"],
)
@pytest.mark.asyncio
async def test_a_call_the_store_refuses_raises_the_refusal_through_the_base_hooks(
    caplog, direction, stored, error, match
):
    """The store raises its refusal, and the base refusal hooks it runs first return without a log.

    The store would log a WARNING for a hook that raised.
    """
    volume: Any = FakeVolume()
    if stored is not None:
        volume.store.kv["k"] = stored
    with pytest.raises(error, match=match):
        await bounded(_call(MonarchRPCTransportBuffer(_ref(volume)), direction))
    assert [
        r
        for r in caplog.records
        if r.levelno >= logging.WARNING and r.name.startswith("torchstore")
    ] == []


def test_threads_making_a_contexts_cache_at_once_all_get_the_one_stored_first():
    """Give every thread that makes a transport context's cache at once the cache stored first.

    The first thread's cache is made only after another thread's whole lookup returned.
    """
    making, other_returned = threading.Event(), threading.Event()

    class Cache(TransportCache):
        made = 0

        def __init__(self) -> None:
            Cache.made += 1
            if Cache.made == 1:
                making.set()
                assert other_returned.wait(BOUND_S)

        def clear(self) -> None:
            pass

    context = TransportContext()
    got: dict[str, Any] = {}
    first = threading.Thread(target=lambda: got.update(first=context.get(Cache)))
    first.start()
    assert making.wait(BOUND_S)
    got["other"] = context.get(Cache)
    other_returned.set()
    first.join(BOUND_S)
    assert not first.is_alive() and got["first"] is got["other"]
