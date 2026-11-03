# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for Gloo transfers sharing one cached pair.

A fake pair stands in for Gloo's process groups with both ranks in this process. As on a Gloo pair,
each tag's sends and receives match in posting order, and a send fills the start of its receive or,
if larger, fails both halves where Gloo aborts the receiving process. One test runs a real pair.
"""

import asyncio
import concurrent.futures
import contextlib
import gc
import itertools
import json
import logging
import socket
import subprocess
import sys
import threading
import time
import uuid
import weakref
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import pytest_asyncio
import torch
import torchstore.transport.gloo as gloo
from torch.distributed import DistNetworkError, TCPStore
from torchstore.client import LocalClient
from torchstore.directory import ObjectType, StorageInfo
from torchstore.storage import InMemoryStore
from torchstore.strategy import StorageVolumeRef
from torchstore.transport import TransportType
from torchstore.transport.buffers import TransportContext
from torchstore.transport.gloo import GlooProcessGroupCache, GlooTransportBuffer
from torchstore.transport.types import Request, TensorSlice

from .volume_meshes import BOUND_S, bounded, FakeVolume, rpc_copy, until

pytestmark = pytest.mark.timeout(60)

# Far past every bounded wait, so only expire() fails a half that a passing test leaves unmatched.
_BACKSTOP_S = 30
_CLOSED = "Connection closed by peer"
# The store reads only that a key holds DTensor shards, not what they hold.
_SHARDS = {(0,): {}}
_RECEIVE_FAILED = (gloo.__name__, "A Gloo receive its get did not await failed")


class _Work(concurrent.futures.Future):
    """A posted send or receive, which the pair completes when it matches or expires it."""

    def wait(self) -> None:
        self.result(timeout=_BACKSTOP_S)


class _Group(SimpleNamespace):
    """One rank of a fake pair, which can be referenced weakly as a process group can."""


class _Pair:
    """Both ranks of one Gloo pair. Each post is logged as (rank, op, tag, bytes)."""

    def __init__(self, ref: StorageVolumeRef) -> None:
        self.ref = ref
        self.lock = threading.Lock()
        self.closed = False
        self.posted: list[tuple[int, str, int, int]] = []
        self.unmatched: dict[
            tuple[str, int, int], list[tuple[torch.Tensor, _Work]]
        ] = {}

    def group(self, rank: int) -> Any:
        return _Group(
            rank=lambda: rank,
            send=lambda tensors, dstRank, tag: self.post(rank, "send", tensors[0], tag),
            recv=lambda tensors, srcRank, tag: self.post(rank, "recv", tensors[0], tag),
        )

    def post(self, rank: int, op: str, tensor: torch.Tensor, tag: int) -> _Work:
        # torch's own checks on a tag.
        if tag < 0:
            raise RuntimeError("Tag must be nonnegative")
        if tag > 0x7FFFFFFF:
            raise TypeError(f"{op}(): incompatible function arguments")
        sender = rank if op == "send" else 1 - rank
        work = _Work()
        with self.lock:
            if self.closed:
                raise RuntimeError(_CLOSED)
            self.posted.append((rank, op, tag, tensor.numel() * tensor.element_size()))
            match = ("recv" if op == "send" else "send", sender, tag)
            if match not in self.unmatched:
                self.unmatched.setdefault((op, sender, tag), []).append((tensor, work))
                return work
            peer, peer_work = self.unmatched[match].pop(0)
            if not self.unmatched[match]:
                del self.unmatched[match]
        source, destination = (tensor, peer) if op == "send" else (peer, tensor)
        size = source.numel() * source.element_size()
        if size <= destination.numel() * destination.element_size():
            # Gloo copies the send's bytes into the start of the receive.
            destination.view(-1).view(torch.uint8)[:size].copy_(
                source.reshape(-1).view(torch.uint8)
            )
            peer_work.set_result(None)
            work.set_result(None)
        else:
            # Gloo aborts the receiving process, so here both halves fail.
            for failed in (peer_work, work):
                failed.set_exception(
                    RuntimeError("Received data size doesn't match expected size")
                )
        return work

    def waiting(self) -> list[torch.Tensor]:
        """The tensors of the halves still waiting for a match."""
        with self.lock:
            return [tensor for queue in self.unmatched.values() for tensor, _ in queue]

    def sends(self, rank: int) -> list[tuple[int, int]]:
        """The (tag, bytes) of each send ``rank`` posted."""
        return [
            (tag, size) for r, op, tag, size in self.posted if (r, op) == (rank, "send")
        ]

    def receives(self, rank: int) -> list[tuple[int, int]]:
        """The (tag, bytes) of each receive ``rank`` posted."""
        return [
            (tag, size) for r, op, tag, size in self.posted if (r, op) == (rank, "recv")
        ]

    def expire(self) -> None:
        """Fail every unmatched send and receive as Gloo's timeout does, and every later post on the pair it closes."""
        with self.lock:
            self.closed = True
            expired = [
                (op, work)
                for (op, _, _), queue in self.unmatched.items()
                for _, work in queue
            ]
            self.unmatched.clear()
        for op, work in expired:
            work.set_exception(
                RuntimeError(f"Timed out waiting for {op} operation to complete")
            )


class _FirstReceiveLate:
    """Rank 0 of ``pair``, whose first receive waits in its thread for ``release``, as on a busy host."""

    def __init__(self, pair: _Pair) -> None:
        self.pair = pair
        self.holding = threading.Event()
        self.release = threading.Event()

    def rank(self) -> int:
        return 0

    def send(self, tensors, dstRank, tag) -> _Work:
        return self.pair.post(0, "send", tensors[0], tag)

    @property
    def recv(self):
        # Python looks up a call's function before it evaluates the arguments, such as the tag.
        if not self.holding.is_set():
            self.holding.set()
            self.release.wait(_BACKSTOP_S)
        return lambda tensors, srcRank, tag: self.pair.post(0, "recv", tensors[0], tag)


def _ref() -> StorageVolumeRef:
    volume: Any = FakeVolume()
    volume_id = f"volume-{uuid.uuid4().hex[:8]}"
    return StorageVolumeRef(volume, volume_id, TransportContext(), TransportType.Gloo)


def _cached_pair(monkeypatch, takes_tags: bool) -> _Pair:
    ref, key = _ref(), "torchstore_gloo_fake"
    pair = _Pair(ref)
    monkeypatch.setattr(
        gloo, "_store_addrs", {ref.volume_id: ("localhost", 0, key, takes_tags)}
    )
    # A fresh count, so the first tag drawn is 1.
    monkeypatch.setattr(gloo, "_tags", itertools.count())
    ref.transport_context.get(GlooProcessGroupCache).put(key, pair.group(0))
    ref.volume.store.transport_context.get(GlooProcessGroupCache).put(
        key, pair.group(1)
    )
    return pair


def _record_receives(monkeypatch) -> list[asyncio.Task]:
    """Record the task of each receive a Gloo buffer posts, on the client or the volume."""
    receives: list[asyncio.Task] = []
    receive_tensor = GlooTransportBuffer._receive_tensor

    async def recorded(self, tensor, transport_context, tag):
        # No local holds the task, so a traceback through this frame lets it be collected.
        receives.append(cast(asyncio.Task, asyncio.current_task()))
        return await receive_tensor(self, tensor, transport_context, tag)

    monkeypatch.setattr(GlooTransportBuffer, "_receive_tensor", recorded)
    return receives


def _change_on_get(monkeypatch, pair, changes, before=None) -> None:
    """Apply ``changes`` (a key's new value, or None to delete it) as that key's volume get starts.

    As a put or delete landing between the get's metadata fetch and its volume call. ``before``, if
    given, is awaited first.
    """
    store, get = pair.ref.volume.store, pair.ref.volume.get.call_one

    async def get_after_a_change(buffer, requests):
        key = requests[0].key
        if key in changes:
            if before is not None:
                await before()
            if changes[key] is None:
                del store.kv[key]
            else:
                store.kv[key] = changes[key]
        return await get(buffer, requests)

    monkeypatch.setattr(pair.ref.volume.get, "call_one", get_after_a_change)


def _warnings(caplog) -> list[tuple[str, str]]:
    """The records torchstore logged at WARNING or above, as (logger, message)."""
    return [
        (r.name, r.getMessage())
        for r in caplog.records
        if r.levelno >= logging.WARNING and r.name.startswith("torchstore")
    ]


async def _warnings_once_ended(receives, caplog) -> list[tuple[str, str]]:
    """The torchstore WARNINGs once each of ``receives`` ended and the callbacks drop() added to it ran."""
    if receives:
        # A task runs its done callbacks in the order they were added, so drop()'s run first.
        await bounded(asyncio.wait(receives))
    return _warnings(caplog)


def _local_client(ref: StorageVolumeRef) -> LocalClient:
    """A client whose directory places every key as a whole tensor on ``ref``'s volume."""

    async def locate_volumes(keys):
        return {key: {ref.volume_id: StorageInfo(ObjectType.TENSOR)} for key in keys}

    directory: Any = SimpleNamespace(locate_volumes=locate_volumes)
    return LocalClient(directory, SimpleNamespace(get_storage_volume=lambda _: ref))


@pytest.fixture(autouse=True)
def _capture_warnings(caplog):
    # The log asserts hold whatever level TORCHSTORE_LOG_LEVEL gives the root logger.
    caplog.set_level(logging.WARNING)


@pytest_asyncio.fixture
async def asyncio_reports():
    """What asyncio reports on this test's loop, which runs only the test's own tasks."""
    loop, reports = asyncio.get_running_loop(), []
    previous = loop.get_exception_handler()
    loop.set_exception_handler(lambda loop, context: reports.append(context))
    yield reports
    loop.set_exception_handler(previous)


@pytest_asyncio.fixture
async def pair(monkeypatch):
    """A client and an in-process volume whose cached Gloo pair is a fake one that takes tags."""
    pair = _cached_pair(monkeypatch, takes_tags=True)
    yield pair
    pair.expire()


@pytest_asyncio.fixture
async def untagged_pair(monkeypatch):
    """The same with the pair cached as taking no tags, so the client moves everything on tag 0.

    The volume then sees what a client before tags sends it.
    """
    pair = _cached_pair(monkeypatch, takes_tags=False)
    yield pair
    pair.expire()


@pytest.fixture
def late_client(pair):
    """Rank 0 of ``pair`` for the client, its first receive held in its thread until released, at the latest when this ends."""
    client = _FirstReceiveLate(pair)
    store_key = gloo._store_addrs[pair.ref.volume_id][2]
    pair.ref.transport_context.get(GlooProcessGroupCache).put(store_key, client)
    yield client
    client.release.set()


@pytest.fixture
def handshaking_pair(monkeypatch):
    """A fake pair that no client has cached, so the first transfer pairs through the handshake."""
    pair = _Pair(_ref())
    monkeypatch.setattr(gloo, "_store_addrs", {})
    monkeypatch.setattr(gloo, "_tags", itertools.count())
    monkeypatch.setattr(gloo, "TCPStore", lambda **kwargs: None)
    monkeypatch.setattr(
        gloo, "_gloo_factory", lambda store, rank, **kwargs: pair.group(rank)
    )
    yield pair
    pair.expire()


@pytest.mark.parametrize("direction", ["put", "get"])
@pytest.mark.asyncio
async def test_two_gloo_transfers_in_flight_on_one_pair_keep_their_own_bytes(
    monkeypatch, pair, direction
):
    """Two transfers keep their own bytes when the client posts its halves in the other order."""
    values = {"a": torch.full((4,), 1.0), "b": torch.full((4,), 2.0)}
    store = pair.ref.volume.store
    if direction == "get":
        store.kv.update({key: value.clone() for key, value in values.items()})
    first, second = GlooTransportBuffer(pair.ref), GlooTransportBuffer(pair.ref)
    # The first transfer's client half waits until the second transfer has posted both halves.
    release = asyncio.Event()
    hook = "_receive_tensor" if direction == "get" else "_send_tensor"
    original = getattr(GlooTransportBuffer, hook)
    filled = {}

    async def held(self, tensor, transport_context, tag):
        if self is first:
            await release.wait()
        result = await original(self, tensor, transport_context, tag)
        filled[self] = tensor
        return result

    monkeypatch.setattr(GlooTransportBuffer, hook, held)

    def call(buffer, key):
        if direction == "put":
            return buffer.put_to_storage_volume([Request.from_tensor(key, values[key])])
        return buffer.get_from_storage_volume([Request(key=key)])

    calls = [asyncio.ensure_future(call(first, "a"))]
    # The first transfer's volume half posts first, so one shared tag would hand it to the second.
    await until(lambda: len(pair.posted) == 1)
    calls.append(asyncio.ensure_future(call(second, "b")))
    await until(lambda: len(pair.posted) == 3)
    release.set()
    results = await bounded(asyncio.gather(*calls))
    if direction == "put":
        got = store.kv
    else:
        got = {"a": results[0][0], "b": results[1][0]}
        # Each get returns the tensor its receive filled, so the volume's reply carried no copy.
        assert got["a"] is filled[first] and got["b"] is filled[second]
    assert torch.equal(got["a"], values["a"]) and torch.equal(got["b"], values["b"])


@pytest.mark.parametrize(
    "stored, request_, drains",
    [
        # A dtype other than the default, so the volume's drain must take the client's dtype.
        (
            {"obj": "an object"},
            Request.from_tensor("k", torch.ones(4, dtype=torch.float64)),
            [(2, 32)],
        ),
        (_SHARDS, Request.from_objects("k", "an object"), []),
    ],
    ids=["tensor_over_an_object", "object_over_shards"],
)
@pytest.mark.asyncio
async def test_a_gloo_put_its_volume_refuses_raises_the_refusal_without_waiting(
    caplog, pair, stored, request_, drains
):
    """A refused put raises the refusal at once, as the volume first takes any tensor the client sent.

    Its buffer first put another key, whose tag an object put must not keep. Neither side logs.
    """
    volume = pair.ref.volume
    volume.store.kv["k"] = stored
    warm = Request.from_tensor("warm", torch.arange(4.0))
    putting = asyncio.ensure_future(
        GlooTransportBuffer(pair.ref).put_to_storage_volume([warm, request_])
    )

    def drained_or_answered():
        return len(volume.tasks) == 2 and (
            volume.tasks[1].done() or len(pair.receives(1)) == 2
        )

    await until(drained_or_answered)
    # The warm key's 4 floats on tag 1, then any drain of the refused put's 4 doubles on its own tag.
    assert pair.receives(1) == [(1, 16)] + drains
    with pytest.raises(AssertionError, match="Existing data is"):
        await bounded(volume.tasks[1])
    # The volume has answered, so any drain has met the client's send and nothing waits.
    assert pair.waiting() == []
    with pytest.raises(AssertionError, match="Existing data is"):
        await bounded(putting)
    assert _warnings(caplog) == []


@pytest.mark.parametrize(
    "stored, after, error, match, ending_sends",
    [
        (torch.arange(4.0), None, KeyError, "not found", [(2, 0)]),
        ({"obj": "an object"}, None, KeyError, "not found", []),
        (torch.arange(4.0), _SHARDS, RuntimeError, "sharded tensor", [(2, 0)]),
    ],
    ids=["deleted_tensor", "deleted_object", "tensor_to_shards"],
)
@pytest.mark.asyncio
async def test_a_gloo_get_its_volume_fails_leaves_no_half_waiting(
    monkeypatch, caplog, pair, stored, after, error, match, ending_sends
):
    """A get whose key is deleted or turns into shards after its metadata fetch raises.

    Its buffer first got another key. The volume ends any receive the get posted with an empty send
    on the get's tag, and neither side logs.
    """
    store = pair.ref.volume.store
    store.kv.update(warm=torch.arange(4.0), k=stored)
    _change_on_get(monkeypatch, pair, {"k": after})
    receives = _record_receives(monkeypatch)
    with pytest.raises(error, match=match):
        await bounded(
            GlooTransportBuffer(pair.ref).get_from_storage_volume(
                [Request(key="warm"), Request(key="k")]
            )
        )
    assert pair.waiting() == []
    # The warm key's 4 floats on tag 1, then whatever ended k's receive.
    assert pair.sends(1) == [(1, 16)] + ending_sends
    assert await _warnings_once_ended(receives, caplog) == []


async def _unreached(pair, buffer, requests):
    # An RPC fails after it yields, so the get's receive has begun, as a call that fails at once skips it.
    await asyncio.sleep(0)
    raise ConnectionError("never reached the volume")


async def _served_without_a_send(pair, buffer, requests):
    # The receive times out before the call returns, and the get then awaits it.
    await until(pair.waiting)
    pair.expire()
    return rpc_copy(buffer)


@pytest.mark.parametrize(
    "call, error, match, ending, logged",
    [
        (_unreached, ConnectionError, "never reached", "timed_out", True),
        (_unreached, ConnectionError, "never reached", "cancelled", False),
        (_served_without_a_send, RuntimeError, "Timed out waiting", None, False),
    ],
    ids=["timed_out", "cancelled", "awaited"],
)
@pytest.mark.asyncio
async def test_a_gloo_get_logs_the_failure_of_a_receive_it_did_not_await(
    monkeypatch, caplog, asyncio_reports, pair, call, error, match, ending, logged
):
    """A get whose call never reaches its volume raises at once, and its receive's timeout logs a WARNING.

    A cancelled receive logs nothing, nor does a get that raised the timeout of the receive it awaited.
    """
    receives = _record_receives(monkeypatch)
    monkeypatch.setattr(pair.ref.volume.get, "call_one", lambda *a: call(pair, *a))
    pair.ref.volume.store.kv["k"] = torch.arange(4.0)
    with pytest.raises(error, match=match):
        await bounded(
            GlooTransportBuffer(pair.ref).get_from_storage_volume([Request(key="k")])
        )
    if ending is not None:
        await until(pair.waiting)
    [receive] = receives
    if ending == "timed_out":
        pair.expire()
    elif ending == "cancelled":
        receive.cancel()
    await bounded(asyncio.wait([receive]))
    del receive, receives[:]
    # Asyncio reports an unretrieved error when it collects the task, and an error a callback raises.
    gc.collect()
    found = [
        (r.name, r.levelno, r.exc_info and str(r.exc_info[1]))
        for r in caplog.records
        if r.levelno >= logging.WARNING and r.name == gloo.__name__
    ]
    timeout = "Timed out waiting for recv operation to complete"
    assert found == ([(gloo.__name__, logging.WARNING, timeout)] if logged else [])
    assert asyncio_reports == []


@pytest.mark.parametrize(
    "before, after",
    [
        (torch.arange(4.0), {"obj": "an object"}),
        # The same number of elements, so only the shape tells the tensors apart.
        (torch.arange(4.0), torch.arange(4.0).reshape(2, 2)),
        (torch.arange(4.0), torch.arange(4, dtype=torch.int32)),
        # More bytes than the receive holds, which would fail both halves if the volume sent them.
        (torch.arange(4.0), torch.arange(8.0)),
        ({"obj": "an object"}, torch.arange(4.0)),
    ],
    ids=[
        "tensor_to_object",
        "other_shape",
        "other_dtype",
        "larger",
        "object_to_tensor",
    ],
)
@pytest.mark.asyncio
async def test_a_gloo_get_whose_key_changed_after_its_metadata_fetch_returns_its_new_value(
    monkeypatch, caplog, pair, before, after
):
    """The volume returns a changed key's value in band and ends any receive the get posted.

    It ends that receive with an empty send on the get's tag, and neither side logs.
    """
    store = pair.ref.volume.store
    store.kv["k"] = before
    _change_on_get(monkeypatch, pair, {"k": after})
    receives = _record_receives(monkeypatch)
    [got] = await bounded(
        GlooTransportBuffer(pair.ref).get_from_storage_volume([Request(key="k")])
    )
    if isinstance(after, dict):
        assert got == after["obj"]
    else:
        assert torch.equal(got, after)
    assert pair.waiting() == []
    # A get that took the key for an object posted no receive, so nothing went to it.
    assert pair.sends(1) == ([] if isinstance(before, dict) else [(1, 0)])
    assert await _warnings_once_ended(receives, caplog) == []


@pytest.mark.asyncio
async def test_a_gloo_buffer_that_got_an_object_gets_the_next_keys_tensor_over_the_pair(
    pair,
):
    """A buffer moves a tensor over the pair after it got an object key, as ``get_batch`` reuses one buffer per volume."""
    store = pair.ref.volume.store
    store.kv.update(o={"obj": "an object"}, t=torch.arange(4.0))
    got = await bounded(
        GlooTransportBuffer(pair.ref).get_from_storage_volume(
            [Request(key="o"), Request(key="t")]
        )
    )
    assert got[0] == "an object" and torch.equal(got[1], store.kv["t"])
    # The tensor's 4 floats met the get's receive on tag 1 rather than coming back in band.
    assert pair.sends(1) == [(1, 16)]
    assert pair.waiting() == []


@pytest.mark.parametrize(
    "destination", [None, torch.full((3,), -1.0)], ids=["plain", "in_place"]
)
@pytest.mark.asyncio
async def test_a_gloo_get_of_a_slice_past_its_tensors_end_is_refused(
    caplog, pair, destination
):
    """The volume refuses a slice that runs past its tensor's end, and an in-place get leaves its destination alone.

    The receive of the requested 3 elements ends with an empty send, and neither side logs.
    """
    pair.ref.volume.store.kv["s"] = torch.arange(4.0)
    past_end = TensorSlice(
        offsets=(3,), coordinates=(), global_shape=(4,), local_shape=(3,), mesh_shape=()
    )
    with pytest.raises(
        ValueError, match=r"falls outside the stored tensor of shape \(4,\)"
    ):
        await bounded(
            _local_client(pair.ref).get(
                "s", inplace_tensor=destination, tensor_slice_spec=past_end
            )
        )
    if destination is not None:
        assert torch.equal(destination, torch.full((3,), -1.0))
    assert pair.waiting() == []
    assert pair.sends(1) == [(1, 0)]
    assert _warnings(caplog) == []


def _refused_put(pair, monkeypatch):
    pair.ref.volume.store.kv["k"] = {"obj": "an object"}
    return GlooTransportBuffer(pair.ref).put_to_storage_volume(
        [Request.from_tensor("k", torch.ones(4))]
    )


def _refused_get(pair, monkeypatch):
    pair.ref.volume.store.kv["k"] = torch.arange(4.0)
    _change_on_get(monkeypatch, pair, {"k": None})
    return GlooTransportBuffer(pair.ref).get_from_storage_volume([Request(key="k")])


def _served_put(pair, monkeypatch):
    return GlooTransportBuffer(pair.ref).put_to_storage_volume(
        [Request.from_tensor("k", torch.ones(4))]
    )


def _served_get(pair, monkeypatch):
    pair.ref.volume.store.kv["k"] = torch.arange(4.0)
    return GlooTransportBuffer(pair.ref).get_from_storage_volume([Request(key="k")])


_DROP_FAILED = (
    "torchstore.transport.buffers",
    "GlooTransportBuffer.drop() failed after its call raised",
)


@pytest.mark.parametrize(
    "call, error, match, volume_warnings, client_warning",
    [
        (
            _refused_put,
            AssertionError,
            "is an object",
            [
                (
                    "torchstore.storage",
                    "GlooTransportBuffer.handle_refused_put() failed after the volume refused its request",
                )
            ],
            _DROP_FAILED,
        ),
        (
            _refused_get,
            KeyError,
            "not found",
            [
                (
                    "torchstore.storage",
                    "GlooTransportBuffer.handle_refused_get() failed after the volume refused its request",
                )
            ],
            _RECEIVE_FAILED,
        ),
        # Calls the store accepts fail in the transport's own handler, and run no refusal hook.
        (_served_put, RuntimeError, _CLOSED, [], _DROP_FAILED),
        (_served_get, RuntimeError, _CLOSED, [], _RECEIVE_FAILED),
    ],
    ids=["refused_put", "refused_get", "served_put", "served_get"],
)
@pytest.mark.asyncio
async def test_on_a_closed_gloo_pair_a_call_raises_its_own_error(
    monkeypatch, caplog, pair, call, error, match, volume_warnings, client_warning
):
    """Once a timeout has closed the pair, a call raises the volume's own error and the client logs its half's.

    The volume logs its refusal hook's failure for a refusal, and nothing for a call it accepted.
    """
    receives = _record_receives(monkeypatch)
    pair.expire()
    with pytest.raises(error, match=match):
        await bounded(call(pair, monkeypatch))
    warnings = await _warnings_once_ended(receives, caplog)
    assert warnings == volume_warnings + [client_warning]
    for record in caplog.records:
        if record.levelno >= logging.WARNING and record.name.startswith("torchstore"):
            assert record.exc_info is not None and str(record.exc_info[1]) == _CLOSED


@pytest.mark.asyncio
async def test_on_a_closed_gloo_pair_a_get_whose_key_changed_returns_its_new_value(
    monkeypatch, caplog, pair
):
    """Once a timeout has closed the pair, the volume cannot end a changed key's receive.

    The volume logs that failure and returns the key's value in band. The client logs its receive's
    failure once, though the get's next key, an object, posts no receive.
    """
    store = pair.ref.volume.store
    store.kv.update(k=torch.arange(4.0), o={"obj": "o's object"})
    _change_on_get(monkeypatch, pair, {"k": {"obj": "k's object"}})
    receives = _record_receives(monkeypatch)
    pair.expire()
    got = await bounded(
        GlooTransportBuffer(pair.ref).get_from_storage_volume(
            [Request(key="k"), Request(key="o")]
        )
    )
    assert got == ["k's object", "o's object"]
    assert await _warnings_once_ended(receives, caplog) == [
        (
            gloo.__name__,
            "Ending a Gloo get's receive failed, and the get's value still goes back in band",
        ),
        _RECEIVE_FAILED,
    ]
    for record in caplog.records:
        if record.levelno >= logging.WARNING and record.name.startswith("torchstore"):
            assert record.exc_info is not None and _CLOSED in str(record.exc_info[1])


@pytest.mark.asyncio
async def test_a_gloo_receive_whose_thread_starts_late_moves_on_its_own_tag(
    monkeypatch, pair, late_client
):
    """A receive whose thread posts after its buffer drew the next key's tag moves on its own tag.

    Its key changed after its metadata fetch, and ending its receive failed at the volume.
    """
    store = pair.ref.volume.store
    store.kv.update(k=torch.arange(4.0), b=torch.full((4,), 2.0))
    # Key b starts only once the thread of k's receive waits.
    _change_on_get(
        monkeypatch,
        pair,
        {"k": {"obj": "k's object"}},
        before=lambda: until(late_client.holding.is_set),
    )

    async def ending_fails(self, ctx):
        raise RuntimeError("could not end the receive")

    monkeypatch.setattr(GlooTransportBuffer, "_end_client_receive", ending_fails)
    got = await bounded(
        GlooTransportBuffer(pair.ref).get_from_storage_volume(
            [Request(key="k"), Request(key="b")]
        )
    )
    late_client.release.set()

    def receive_tags():
        return [tag for tag, _ in pair.receives(0)]

    await until(lambda: len(receive_tags()) == 2)
    assert got[0] == "k's object" and torch.equal(got[1], store.kv["b"])
    # Key b drew tag 2 and its receive posted first.
    assert receive_tags() == [2, 1]


@pytest.mark.parametrize(
    "after", [None, {"obj": "an object"}], ids=["refused", "changed_key"]
)
@pytest.mark.asyncio
async def test_a_gloo_volume_cancelled_while_it_ends_a_receive_ends_cancelled(
    monkeypatch, pair, late_client, after
):
    """A cancel that lands while the volume's empty send waits for the get's receive propagates.

    It replaces the volume's refusal of a deleted key or its in-band answer for a changed one.
    """
    pair.ref.volume.store.kv["k"] = torch.arange(4.0)
    _change_on_get(monkeypatch, pair, {"k": after})
    getting = asyncio.ensure_future(
        GlooTransportBuffer(pair.ref).get_from_storage_volume([Request(key="k")])
    )
    # The client's receive has not posted, so the volume's empty send waits for it.
    await until(lambda: pair.sends(1) == [(1, 0)])
    [volume_get] = pair.ref.volume.tasks
    volume_get.cancel()
    with pytest.raises(asyncio.CancelledError):
        await bounded(volume_get)
    with pytest.raises(asyncio.CancelledError):
        await bounded(getting)


@pytest.mark.asyncio
async def test_gloo_tags_wrap_to_1_after_the_largest_tag_torch_accepts(
    monkeypatch, pair
):
    """Puts and gets move on to tag 1 after tag 0x7FFFFFFF, as torch refuses larger tags and 0 is kept."""
    monkeypatch.setattr(gloo, "_tags", itertools.count(0x7FFFFFFE))
    tags, got = [], []
    for key, value in (("a", 1.0), ("b", 2.0)):
        buffer = GlooTransportBuffer(pair.ref)
        await bounded(
            buffer.put_to_storage_volume(
                [Request.from_tensor(key, torch.full((4,), value))]
            )
        )
        tags.append(buffer.tag)
    for key in ("a", "b"):
        buffer = GlooTransportBuffer(pair.ref)
        got += await bounded(buffer.get_from_storage_volume([Request(key=key)]))
        tags.append(buffer.tag)
    assert tags == [0x7FFFFFFF, 1, 2, 3]
    assert torch.equal(got[0], torch.full((4,), 1.0))
    assert torch.equal(got[1], torch.full((4,), 2.0))


# Fields of this revision's Gloo buffers that buffers pickled by clients before tags lack.
_SINCE_TAGS = ("tag", "_pg", "_takes_tags", "_half_started")


def _pickled_before_tags(buffer: GlooTransportBuffer) -> GlooTransportBuffer:
    """``buffer`` as a volume unpickles it from a client before tags, whose buffers lack the fields tags added.

    Built from the pickled state directly, as this revision's own ``__getstate__`` would add a field back.
    """
    state = rpc_copy(buffer).__dict__
    unpickled = GlooTransportBuffer.__new__(GlooTransportBuffer)
    unpickled.__dict__.update(
        {name: value for name, value in state.items() if name not in _SINCE_TAGS}
    )
    return unpickled


@pytest.mark.asyncio
async def test_a_gloo_volume_moves_a_client_before_tags_on_tag_0(pair):
    """A volume moves tensors on tag 0 for a buffer a client before tags pickled, which lacks every field tags added."""
    client, store = pair.group(0), pair.ref.volume.store
    sent, received = torch.arange(4.0), torch.empty(4)
    put, get = Request.from_tensor("k", sent), Request(key="k")
    putting, getting = GlooTransportBuffer(pair.ref), GlooTransportBuffer(pair.ref)
    # The cached pair's address and the tensor's metadata, as such a client's own hooks set them.
    for buffer in (putting, getting):
        buffer.requires_handshake([put])
        buffer.shape, buffer.dtype = sent.shape, sent.dtype
    sending = asyncio.to_thread(lambda: client.send([sent], dstRank=1, tag=0).wait())
    await bounded(
        asyncio.gather(
            sending, store.put(_pickled_before_tags(putting), [put.meta_only()])
        )
    )
    receiving = asyncio.to_thread(
        lambda: client.recv([received], srcRank=1, tag=0).wait()
    )
    await bounded(
        asyncio.gather(receiving, store.get(_pickled_before_tags(getting), [get]))
    )
    assert torch.equal(store.kv["k"], sent)
    assert torch.equal(received, sent)


@pytest.mark.asyncio
async def test_a_gloo_volume_leaves_a_refused_put_of_a_client_before_tags_waiting(
    caplog, untagged_pair
):
    """On tag 0 the volume receives nothing for a put it refuses, so the client's send waits.

    It waits for Gloo's timeout, as with earlier volumes, and its put raises the refusal.
    """
    pair = untagged_pair
    pair.ref.volume.store.kv["k"] = {"obj": "an object"}
    sent = torch.ones(4)
    putting = asyncio.ensure_future(
        GlooTransportBuffer(pair.ref).put_to_storage_volume(
            [Request.from_tensor("k", sent)]
        )
    )
    await until(lambda: pair.ref.volume.tasks)
    [volume_put] = pair.ref.volume.tasks
    with pytest.raises(AssertionError, match="is an object"):
        await bounded(volume_put)
    [waiting] = pair.waiting()
    assert waiting is sent
    pair.expire()
    with pytest.raises(AssertionError, match="is an object"):
        await bounded(putting)
    assert pair.sends(0) == [(0, 16)] and pair.sends(1) == []
    assert _warnings(caplog) == [_DROP_FAILED]


@pytest.mark.parametrize(
    "after, ending_sends",
    [
        (None, []),
        ({"obj": "an object"}, []),
        # More bytes than the receive holds, which earlier volumes sent and so aborted the client.
        (torch.arange(8.0), [(0, 0)]),
    ],
    ids=["deleted", "turned_into_an_object", "turned_into_a_larger_tensor"],
)
@pytest.mark.asyncio
async def test_a_gloo_volume_sends_a_changed_get_of_a_client_before_tags_only_where_earlier_volumes_did(
    monkeypatch, caplog, untagged_pair, after, ending_sends
):
    """On tag 0 the volume sends a changed key's get only where earlier volumes sent the key's tensor.

    There it sends an empty tensor, and elsewhere the receive waits for Gloo's timeout as before.
    """
    pair = untagged_pair
    pair.ref.volume.store.kv["k"] = torch.arange(4.0)
    _change_on_get(monkeypatch, pair, {"k": after})
    receives = _record_receives(monkeypatch)
    getting = GlooTransportBuffer(pair.ref).get_from_storage_volume([Request(key="k")])
    if after is None:
        with pytest.raises(KeyError, match="not found"):
            await bounded(getting)
    else:
        [got] = await bounded(getting)
        assert (
            got == after["obj"] if isinstance(after, dict) else torch.equal(got, after)
        )
    assert pair.sends(1) == ending_sends
    assert len(pair.waiting()) == (0 if ending_sends else 1)
    pair.expire()
    expected = [] if ending_sends else [_RECEIVE_FAILED]
    assert await _warnings_once_ended(receives, caplog) == expected


@pytest.mark.parametrize("volume_takes_tags", [True, False])
@pytest.mark.asyncio
async def test_a_gloo_client_draws_tags_only_for_a_volume_whose_handshake_offers_them(
    monkeypatch, handshaking_pair, volume_takes_tags
):
    """A client pairs through the handshake and draws tags only if the volume's reply offers them."""
    pair = handshaking_pair
    if not volume_takes_tags:
        handshake = GlooTransportBuffer.recv_handshake

        async def answer_as_before_tags(self, ctx, entries):
            await handshake(self, ctx, entries)
            return [None]

        monkeypatch.setattr(
            GlooTransportBuffer, "recv_handshake", answer_as_before_tags
        )
    value = torch.arange(4.0)
    await bounded(
        GlooTransportBuffer(pair.ref).put_to_storage_volume(
            [Request.from_tensor("k", value)]
        )
    )
    # A second buffer finds the pair in the cache.
    [got] = await bounded(
        GlooTransportBuffer(pair.ref).get_from_storage_volume([Request(key="k")])
    )
    assert torch.equal(got, value)
    tags = sorted(tag for _, _, tag, _ in pair.posted)
    assert tags == ([1, 1, 2, 2] if volume_takes_tags else [0, 0, 0, 0])


# A client before tags in its own process: it posts receives of the listed sizes in order on tag 0, as
# such a client's gets do, waits for the listed ones and prints the first 16 floats of each.
_CLIENT_BEFORE_TAGS = """
import datetime, json, os, sys
import torch
from torch.distributed import ProcessGroupGloo, TCPStore

port, sizes, awaited = int(sys.argv[1]), json.loads(sys.argv[2]), json.loads(sys.argv[3])
store = TCPStore("127.0.0.1", port, 2, False, timeout=datetime.timedelta(seconds=30))
group = ProcessGroupGloo(store, 0, 2, datetime.timedelta(seconds=30))
receives = [torch.full((size,), -1.0) for size in sizes]
works = [group.recv([receive], 1, 0) for receive in receives]
store.set("posted", "1")
for i in awaited:
    works[i].wait()
print(json.dumps([receive[:16].tolist() for receive in receives]), flush=True)
# Skip interpreter shutdown, which only adds time.
os._exit(0)
"""


@pytest.mark.parametrize(
    "a_after, awaited, expected",
    [
        (None, [0], [list(range(16)), [-1] * 16]),
        (torch.arange(64.0), [0, 1], [[-1] * 16, list(range(16))]),
        ({"obj": "an object"}, [0], [list(range(16)), [-1] * 16]),
    ],
    ids=["deleted", "turned_into_a_larger_tensor", "turned_into_an_object"],
)
@pytest.mark.asyncio
async def test_a_gloo_volume_sends_a_changed_key_at_most_an_empty_tensor_with_two_gets_in_flight(
    a_after, awaited, expected
):
    """Over a real pair, a client before tags posts receives of 16 floats for key b, then 4096 for a.

    Key a changed after its metadata fetch, and the volume serves it first. It sends a at most an empty
    tensor, which no receive is too small for, and only where earlier volumes sent a's tensor.
    """
    store = TCPStore(
        "127.0.0.1", 0, 2, True, timeout=timedelta(seconds=30), wait_for_workers=False
    )
    args = [str(store.port), json.dumps([16, 4096]), json.dumps(awaited)]
    client = subprocess.Popen(
        [sys.executable, "-c", _CLIENT_BEFORE_TAGS, *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    volume = InMemoryStore()
    volume.kv["b"] = torch.arange(16.0)
    if a_after is not None:
        volume.kv["a"] = a_after

    async def serve():
        group = await asyncio.to_thread(
            gloo._gloo_factory, store, 1, 2, timedelta(seconds=30)
        )
        volume.transport_context.get(GlooProcessGroupCache).put("pair", group)
        await asyncio.to_thread(store.wait, ["posted"])
        for key, size in (("a", 4096), ("b", 16)):
            buffer = GlooTransportBuffer(_ref())
            buffer.store_key, buffer.shape = "pair", torch.Size([size])
            buffer.dtype = torch.float32
            with contextlib.suppress(KeyError):
                await bounded(
                    volume.get(_pickled_before_tags(buffer), [Request(key=key)])
                )

    serving = asyncio.ensure_future(serve())
    try:
        out, err = await asyncio.to_thread(client.communicate, timeout=_BACKSTOP_S)
        assert client.returncode == 0, err
        await bounded(serving)
    finally:
        serving.cancel()
        if client.poll() is None:
            client.kill()
            client.communicate()
    assert json.loads(out) == expected


# What a refused client's tensor transfers raise.
_IN_USE = "another live client in this process holds the Gloo pair"


class _Store(SimpleNamespace):
    """A fake TCPStore, which a test can reference weakly as a real one."""


class _Pairings:
    """Fake TCPStores and Gloo groups, so pairings run Gloo's own code with both ranks in this process.

    A client's store and its volume's at one port share one fake pair. Each retired pair is
    logged as (volume ID, store key), and each master store is kept weakly by port.
    """

    def __init__(self) -> None:
        self.volume = FakeVolume()
        self.pairs: dict[int, _Pair] = {}
        self.stores: dict[int, weakref.ref[_Store]] = {}
        # Each pairing's client half by port, so in the order the pairings started.
        self.client_halves: dict[int, weakref.ref[Any]] = {}
        self.retired: list[tuple[str, str]] = []
        self.retiring = threading.Condition()
        self.ports = itertools.count(1)

    def store(self, **kwargs: Any) -> _Store:
        store = _Store(port=kwargs["port"])
        if kwargs["is_master"]:
            self.stores[store.port] = weakref.ref(store)
        return store

    def group(self, store: _Store, rank: int, **kwargs: Any) -> Any:
        group = self.pairs.setdefault(store.port, _Pair(cast(Any, None))).group(rank)
        if rank == 0:
            self.client_halves[store.port] = weakref.ref(group)
        return group

    def retire(self, volume_id: str, store_key: str) -> None:
        with self.retiring:
            self.retired.append((volume_id, store_key))
            self.retiring.notify_all()

    async def retirements(self, count: int) -> list[tuple[str, str]]:
        """Wait until ``count`` pairs retired and return them. A transfer's worker thread can let go of a group late."""

        def wait() -> bool:
            with self.retiring:
                return self.retiring.wait_for(
                    lambda: len(self.retired) >= count, BOUND_S
                )

        assert await asyncio.to_thread(wait), self.retired
        return list(self.retired)

    def halves(self) -> dict[str, Any]:
        """The volume's halves by store key."""
        return self.volume.store.transport_context.get(
            GlooProcessGroupCache
        )._process_groups


@pytest_asyncio.fixture
async def pairing(monkeypatch):
    """Pairings over fake TCPStores and fake pairs, with no pair cached yet."""
    pairings = _Pairings()
    for name, value in (
        ("_store_addrs", {}),
        ("_pair_owners", weakref.WeakValueDictionary()),
        ("_pair_groups", weakref.WeakValueDictionary()),
        ("_retired_listeners", [pairings.retire]),
        ("_tags", itertools.count()),
        ("TCPStore", pairings.store),
        ("_gloo_factory", pairings.group),
        ("_get_hostname", lambda: "localhost"),
        ("portpicker", SimpleNamespace(pick_unused_port=lambda: next(pairings.ports))),
    ):
        monkeypatch.setattr(gloo, name, value)
    yield pairings
    for pair in pairings.pairs.values():
        pair.expire()


def _freed(weak: weakref.ref[Any]) -> threading.Event:
    """An event set once ``weak``'s referent is collected, on whatever thread lets go of it last."""
    freed, referent = threading.Event(), weak()
    if referent is None:
        freed.set()
    else:
        weakref.finalize(referent, freed.set)
    return freed


def _client(volume: Any) -> StorageVolumeRef:
    """A client's reference to ``volume``, with a transport context of its own."""
    return StorageVolumeRef(volume, "volume-0", TransportContext(), TransportType.Gloo)


def _groups(ref: StorageVolumeRef) -> dict[str, Any]:
    return ref.transport_context.get(GlooProcessGroupCache)._process_groups


async def _put(ref: StorageVolumeRef, key: str, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        request = Request.from_tensor(key, value)
    else:
        request = Request.from_objects(key, value)
    await bounded(GlooTransportBuffer(ref).put_to_storage_volume([request]))


async def _get(ref: StorageVolumeRef, request: Request) -> Any:
    [value] = await bounded(GlooTransportBuffer(ref).get_from_storage_volume([request]))
    return value


@pytest.mark.parametrize("old", ["gone", "lost"])
@pytest.mark.asyncio
async def test_a_gloo_transfer_pairs_anew_once_the_cached_pairs_client_is_gone_or_lost_it(
    pairing, old
):
    """Pair anew once the cached pair's client is collected, as after ``ts.reset_client()``, or no longer holds it.

    The old pair retires, so its volume half can go too.
    """
    first = _client(pairing.volume)
    await _put(first, "weights", torch.ones(4))
    [old_key] = _groups(first)
    if old == "gone":
        del first
        client = _client(pairing.volume)
    else:
        _groups(first).clear()
        client = first
    await _put(client, "weights", torch.arange(4.0))
    assert torch.equal(await _get(client, Request(key="weights")), torch.arange(4.0))
    assert pairing.volume.handshakes == 2
    assert pairing.retired == [("volume-0", old_key)]


@pytest.mark.parametrize("when", ["after", "alongside"])
@pytest.mark.asyncio
async def test_a_second_live_gloo_client_of_a_volume_id_is_refused_tensors_but_moves_objects(
    monkeypatch, pairing, when
):
    """Refuse a second live client's tensor puts and gets to a volume ID with a ``RuntimeError``, while its objects move.

    One that paired alongside the first but cached second is refused too, leaving the first's pair cached
    and retiring its own. Once the first client is gone the refused one pairs, though it keeps its error.
    """
    volume = pairing.volume
    first, second = _client(volume), _client(volume)
    if when == "after":
        await _put(first, "a", torch.ones(4))
        with pytest.raises(gloo.GlooPairInUseError, match=_IN_USE) as refused:
            await _put(second, "b", torch.zeros(4))
        with pytest.raises(gloo.GlooPairInUseError, match=_IN_USE):
            await _get(second, Request(key="a"))
        assert volume.handshakes == 1
    else:
        # Each pairing waits at the volume until released so both start before either caches. A handshake
        # carries no requests so each is told apart by its client.
        arrived, releases = [], {"a": asyncio.Event(), "b": asyncio.Event()}
        keys = {id(first): "a", id(second): "b"}
        handshake = volume.handshake.call_one

        async def held(buffer, requests):
            key = keys[id(buffer.storage_volume_ref)]
            arrived.append(key)
            await bounded(releases[key].wait())
            return await handshake(buffer, requests)

        monkeypatch.setattr(volume.handshake, "call_one", held)
        puts = [
            asyncio.ensure_future(_put(ref, key, torch.ones(4)))
            for ref, key in ((first, "a"), (second, "b"))
        ]
        await until(lambda: len(arrived) == 2)
        releases["a"].set()
        await bounded(puts[0])
        releases["b"].set()
        [first_key] = _groups(first)
        with pytest.raises(gloo.GlooPairInUseError, match=_IN_USE) as refused:
            await bounded(puts[1])
        [refused_key] = set(pairing.halves()) - {first_key}
        assert gloo.cached_store_key("volume-0") == first_key and _groups(second) == {}
        assert pairing.retired == [("volume-0", refused_key)]
        # The refused pairing's client half goes at once, though its error is kept.
        live = [port for port, half in pairing.client_halves.items() if half()]
        assert live == [1]
        # The first client keeps its pair.
        assert torch.equal(await _get(first, Request(key="a")), torch.ones(4))
        assert volume.handshakes == 2
    await _put(second, "config", {"lr": 0.5})
    assert await _get(second, Request(key="config", is_object=True)) == {"lr": 0.5}
    # The kept error holds nothing of the first client, so once that is gone the second pairs.
    del first
    await _put(second, "b", torch.full((4,), 2.0))
    assert torch.equal(await _get(second, Request(key="b")), torch.full((4,), 2.0))
    assert isinstance(refused.value, RuntimeError)
    assert not isinstance(refused.value, KeyError)


@pytest.mark.asyncio
async def test_a_gloo_get_marked_as_an_object_rides_in_its_call_and_takes_a_tensor(
    pairing,
):
    """Move a get marked as an object in its call alone, so it takes even a tensor its key now holds.

    Another client may delete and put that key between the get's lookup and its call.
    """
    client = _client(pairing.volume)
    await _put(client, "config", {"lr": 0.5})
    pairing.volume.store.kv["config"] = torch.arange(4.0)
    config = await _get(client, Request(key="config", is_object=True))
    assert torch.equal(config, torch.arange(4.0)) and pairing.volume.handshakes == 0


@pytest.mark.asyncio
async def test_a_first_gloo_tensor_put_the_store_refuses_still_pairs_and_ends_its_send(
    pairing,
):
    """Pair for a client's first tensor put though the store refuses it so the volume takes its send into scratch.

    The client's earlier object put to that key paired nothing. Its next tensor put reuses the pair.
    """
    client = _client(pairing.volume)
    await _put(client, "k", "an object")
    with pytest.raises(AssertionError, match="is an object"):
        await _put(client, "k", torch.ones(4))
    [pair] = pairing.pairs.values()
    assert pair.receives(1) == [(1, 16)] and pair.waiting() == []
    await _put(client, "t", torch.arange(4.0))
    assert torch.equal(pairing.volume.store.kv["t"], torch.arange(4.0))
    assert pairing.volume.handshakes == 1


def _first_handshake_held(
    monkeypatch, volume: Any, fails: bool = False
) -> SimpleNamespace:
    """Hold handshakes at ``volume`` until released and record by key each buffer's checks for a pair.

    With ``fails`` the first handshake to arrive raises once released. A handshake carries no requests so
    each one is named by the key its buffer checked.
    """
    held = SimpleNamespace(arrived=[], checks=[], keys={}, released=asyncio.Event())
    handshake = volume.handshake.call_one
    requires_handshake = GlooTransportBuffer.requires_handshake

    async def hold(buffer, requests):
        key = held.keys[id(buffer)]
        held.arrived.append(key)
        await bounded(held.released.wait())
        if fails and held.arrived[0] == key:
            raise RuntimeError("the first handshake failed")
        return await handshake(buffer, requests)

    def checked(self, requests):
        held.keys[id(self)] = requests[0].key
        held.checks.append(requests[0].key)
        return requires_handshake(self, requests)

    monkeypatch.setattr(volume.handshake, "call_one", hold)
    monkeypatch.setattr(GlooTransportBuffer, "requires_handshake", checked)
    return held


@pytest.mark.parametrize("first", ["pairs", "fails"])
@pytest.mark.asyncio
async def test_racing_first_puts_of_one_client_pair_one_at_a_time(
    monkeypatch, pairing, first
):
    """Pair for only one of two racing first puts of one client while the other waits for that pairing.

    The other reuses the pair that pairing cached or pairs itself once that pairing failed.
    """
    client = _client(pairing.volume)
    held = _first_handshake_held(monkeypatch, pairing.volume, fails=first == "fails")
    values = {"a": torch.full((4,), 1.0), "b": torch.full((4,), 2.0)}
    puts = {
        key: asyncio.ensure_future(_put(client, key, value))
        for key, value in values.items()
    }
    # The pairing put checked for a pair before its handshake and again under the lock as it took its place.
    # The other checked once before it waits.
    await until(lambda: len(held.arrived) == 1 and len(held.checks) == 3)
    [pairing_key] = held.arrived
    [waiting_key] = set(values) - {pairing_key}
    held.released.set()
    outcomes = await bounded(asyncio.gather(*puts.values(), return_exceptions=True))
    stored = [key for key, outcome in zip(puts, outcomes) if outcome is None]
    if first == "pairs":
        assert stored == ["a", "b"] and held.arrived == [pairing_key]
    else:
        assert stored == [waiting_key] and held.arrived == [pairing_key, waiting_key]
        assert (
            str(outcomes[list(puts).index(pairing_key)]) == "the first handshake failed"
        )
    for key in stored:
        assert torch.equal(pairing.volume.store.kv[key], values[key])
    assert (
        list(pairing.halves())
        == list(_groups(client))
        == [gloo.cached_store_key("volume-0")]
    )
    assert pairing.retired == []


@pytest.mark.asyncio
async def test_a_put_cancelled_while_it_waits_for_its_clients_pairing_leaves_that_pairing_alone(
    monkeypatch, pairing
):
    """Let a pairing finish and a later put reuse its pair after another put waiting for that pairing is cancelled."""
    client = _client(pairing.volume)
    held = _first_handshake_held(monkeypatch, pairing.volume)
    pairing_put = asyncio.ensure_future(_put(client, "a", torch.ones(4)))
    await until(lambda: held.arrived == ["a"])
    waiting = asyncio.ensure_future(_put(client, "b", torch.zeros(4)))
    await until(lambda: held.checks.count("b") == 1)
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    held.released.set()
    await bounded(pairing_put)
    await _put(client, "c", torch.full((4,), 2.0))
    assert torch.equal(pairing.volume.store.kv["c"], torch.full((4,), 2.0))
    assert held.arrived == ["a"] and "b" not in pairing.volume.store.kv


@pytest.mark.parametrize("client", ["frozen", "frozen_after_connect", "unreachable"])
def test_a_volumes_pairing_gives_up_only_past_torchs_own_bounds(monkeypatch, client):
    """Raise ``TimeoutError`` at 4 Gloo timeouts and the slack when the client's TCPStore accepts and never replies.

    Torch's client waits for that reply with no deadline, as making the group can once a client froze
    after the connect. One that never listens fails in torch first, with torch's own error, even at a 1s
    timeout. Either way the pairing's thread ends with the store.
    """
    monkeypatch.setattr(gloo, "TORCHSTORE_GLOO_INIT_TIMEOUT", 1)
    if client != "unreachable":
        # Short, so the frozen cases end soon and the message shows both terms of the deadline.
        monkeypatch.setattr(gloo, "_PAIRING_SLACK_S", 0.5)
    unblock = threading.Event()
    if client == "frozen_after_connect":

        def group_waits(store: Any, rank: int, **kwargs: Any) -> Any:
            # As making the group waits on a client that froze once connected.
            if unblock.wait(BOUND_S * 2):
                raise RuntimeError("the client store went away")
            raise RuntimeError("the group was made past the deadline")

        monkeypatch.setattr(gloo, "TCPStore", lambda **kwargs: _Store())
        monkeypatch.setattr(gloo, "_gloo_factory", group_waits)
    try:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            if client == "frozen":
                listener.listen(8)
            buffer = GlooTransportBuffer(cast(Any, None))
            buffer.master_addr, buffer.master_port = listener.getsockname()
            buffer.store_key = "torchstore_gloo_unanswered"
            name = (
                "torchstore-gloo pairing through the TCPStore at"
                f" {buffer.master_addr}:{buffer.master_port}"
            )

            def release() -> None:
                listener.close()
                unblock.set()

            # A pairing without its deadline would hang asyncio.run's executor shutdown. Closing the
            # listener resets its connect and the case then fails instead.
            releaser = threading.Timer(_BACKSTOP_S, release)
            releaser.start()
            start = time.monotonic()
            try:
                with pytest.raises(Exception) as raised:
                    asyncio.run(buffer.recv_handshake(TransportContext(), []))
            finally:
                releaser.cancel()
            elapsed = time.monotonic() - start
            pairings = [t for t in threading.enumerate() if t.name == name]
    finally:
        unblock.set()
    if client == "unreachable":
        assert raised.type is DistNetworkError, raised.value
    else:
        assert raised.type is TimeoutError, raised.value
        assert str(raised.value) == f"{name[16:]} did not finish within 4.5 s"
        assert elapsed >= 4.5
    for thread in pairings:
        assert thread.daemon
        thread.join(30)
        assert not thread.is_alive()


def test_gloo_exit_waits_for_pairings_within_one_bound_then_refuses_new_ones(
    monkeypatch, caplog
):
    """Give the pairings still running at exit one shared wait, warn naming those that outlive it, then start none."""
    now = [0.0]
    joins: list[tuple[str, float]] = []

    class Pairing:
        def __init__(self, name: str, takes_s: float, ends: bool) -> None:
            self.name, self.takes_s, self.ends, self.alive = name, takes_s, ends, True

        def join(self, timeout: float) -> None:
            joins.append((self.name, timeout))
            now[0] += min(self.takes_s, timeout)
            self.alive = not self.ends

        def is_alive(self) -> bool:
            return self.alive

    running = [Pairing("a", 3.0, True), Pairing("b", 9.0, False)]
    monkeypatch.setattr(gloo, "_pairings", running)
    monkeypatch.setattr(gloo, "_exit_deadline", None)
    monkeypatch.setattr(gloo, "time", SimpleNamespace(monotonic=lambda: now[0]))
    gloo._drain_pairings()
    monkeypatch.setattr(gloo, "time", time)
    assert joins == [("a", 5.0), ("b", 2.0)]
    assert _warnings(caplog) == [
        (
            gloo.__name__,
            "Exiting with 1 Gloo pairings still running after 5s, which may abort the"
            " process: b",
        )
    ]
    with pytest.raises(RuntimeError, match="^late pairing was not started: exiting$"):
        asyncio.run(
            gloo._pair_with_deadline(lambda: cast(Any, "group"), "late pairing")
        )


def test_a_gloo_pairing_counts_among_what_exit_waits_for_until_it_returns(monkeypatch):
    """Count a pairing's thread among those exit waits for while the pairing runs, and drop it once it returns."""
    monkeypatch.setattr(gloo, "_pairings", set())
    monkeypatch.setattr(gloo, "_exit_deadline", None)
    counted = []

    def pair() -> Any:
        counted.append(threading.current_thread() in gloo._pairings)
        return "group"

    assert asyncio.run(gloo._pair_with_deadline(pair, "quick pairing")) == "group"
    assert counted == [True] and gloo._pairings == set()


# Forks once exit began while a thread holds Gloo's pair-cache and pairing locks and a pairing is in flight. The
# child takes both locks, which waits on the copies it inherited unless the fork hook replaced them. It checks that
# the hook also dropped the inherited pairs and the pairing in flight that its transfers would wait for forever, and
# that it still pairs. Prints its exit code.
_FORK_WITH_LOCKS_HELD = """
import os, sys, threading
from concurrent import futures

sys.path.insert(0, sys.argv[1])
from torchstore.transport import gloo

assert gloo.__file__.startswith(sys.argv[1]), gloo.__file__
held, release = threading.Event(), threading.Event()
groups = gloo.GlooProcessGroupCache()
gloo._pairing_now[groups] = {"volume-0": futures.Future()}


def hold():
    with gloo._cache_lock, gloo._pairings_lock:
        held.set()
        release.wait(30)


# As the parent's exit hooks do before a fork from one of them.
gloo.begin_exit()
threading.Thread(target=hold, daemon=True).start()
assert held.wait(30)


# Stands in for a pair's group in Gloo's weak caches.
class Group:
    pass


# A pair in each of Gloo's caches in a process that never loads the carrier volume module.
owner, group = gloo.GlooProcessGroupCache(), Group()
gloo._store_addrs["publisher/0"] = ("parent", 1, "parent-pair", True)
gloo._pair_owners["parent-pair"] = owner
gloo._pair_groups["parent-pair"] = group
pid = os.fork()
if pid == 0:
    fresh = gloo._cache_lock.acquire(timeout=5) and gloo._pairings_lock.acquire(timeout=5)
    forgot = not (gloo._store_addrs or gloo._pair_owners or gloo._pair_groups)
    if fresh:
        gloo._pairings_lock.release()
    try:
        paired = fresh and gloo._counted(lambda: "group", "child pairing") == "group"
    except RuntimeError:
        paired = False
    os._exit(0 if fresh and forgot and groups not in gloo._pairing_now and paired else 1)
release.set()
_, status = os.waitpid(pid, 0)
print(os.waitstatus_to_exitcode(status), flush=True)
"""


def test_a_child_forked_while_a_thread_holds_gloos_locks_gets_fresh_ones() -> None:
    """Replace Gloo's locks and its record of pairings in flight in a forked child where their holders never run.

    The child also forgets the pairs it inherited from its parent and pairs though forked once its parent's exit began.
    """
    root = str(Path(__file__).resolve().parents[1])
    child = subprocess.run(
        [sys.executable, "-c", _FORK_WITH_LOCKS_HELD, root],
        capture_output=True,
        text=True,
        timeout=50,
    )
    assert child.stdout.splitlines()[-1:] == ["0"], child.stderr[-2000:]


def test_a_failed_volume_pairing_frees_its_buffer_without_a_cyclic_gc(monkeypatch):
    """Free a volume's buffer once its failed pairing's error is dropped, with the cyclic GC off.

    The loop stays open, as a volume's does, and the pairing's deadline stays far off.
    """
    monkeypatch.setattr(gloo, "TORCHSTORE_GLOO_INIT_TIMEOUT", 60)

    def refuse(**kwargs: Any) -> Any:
        raise DistNetworkError("no TCPStore listens there")

    monkeypatch.setattr(gloo, "TCPStore", refuse)

    async def fail() -> weakref.ref[GlooTransportBuffer]:
        buffer = GlooTransportBuffer(cast(Any, None))
        buffer.master_addr, buffer.master_port = "localhost", 1
        buffer.store_key = "torchstore_gloo_refused"
        freed = weakref.ref(buffer)
        with pytest.raises(DistNetworkError, match="no TCPStore listens there"):
            await buffer.recv_handshake(TransportContext(), [])
        return freed

    loop = asyncio.new_event_loop()
    gc.disable()
    try:
        freed = loop.run_until_complete(fail())
        # The pairing's thread holds its error until it returns, a moment after it settles the call.
        name = "torchstore-gloo pairing through the TCPStore at localhost:1"
        for thread in threading.enumerate():
            if thread.name == name:
                thread.join(BOUND_S)
        assert freed() is None
    finally:
        gc.enable()
        loop.close()


class _WaitsForTheOtherPairing(dict):
    """A Gloo pair cache whose first lookup, inside the pair-cache lock, waits until another thread waits for that lock."""

    def __init__(self, waiting: threading.Event) -> None:
        super().__init__()
        self.waiting = waiting
        self.inside = threading.Event()

    def get(self, key: Any, default: Any = None) -> Any:
        if threading.current_thread().name == "first pairing":
            self.inside.set()
            assert self.waiting.wait(5), "the second pairing never waited for the lock"
        return super().get(key, default)


class _SeenLock:
    """Gloo's pair-cache lock, which tells when a thread has to wait for it."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.waiting = threading.Event()

    def __enter__(self) -> "_SeenLock":
        if not self.lock.acquire(blocking=False):
            self.waiting.set()
            self.lock.acquire()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.lock.release()


def test_two_threads_pairing_two_clients_cache_one_pair_and_refuse_the_other(
    monkeypatch, pairing
):
    """Cache only the first of two clients' pairings that end on two threads at once, and refuse the other.

    Each pairing checks the cached pair and caches its own under Gloo's pair-cache lock.
    """
    lock = _SeenLock()
    cache = _WaitsForTheOtherPairing(lock.waiting)
    monkeypatch.setattr(gloo, "_cache_lock", lock)
    monkeypatch.setattr(gloo, "_store_addrs", cache)
    clients = {name: _client(pairing.volume) for name in ("first", "second")}
    outcomes: dict[str, BaseException | None] = {}

    def post(name: str) -> None:
        buffer = GlooTransportBuffer(clients[name])
        buffer.master_addr, buffer.master_port = "localhost", 1
        buffer.store_key = f"torchstore_gloo_{name}"

        async def run() -> None:
            half = asyncio.get_running_loop().create_future()
            half.set_result(_Group())
            buffer._pg_task = cast(Any, half)
            await buffer._post_handshake([gloo._TAGS], [])

        try:
            asyncio.run(run())
            outcomes[name] = None
        except BaseException as e:
            outcomes[name] = e

    threads = {
        name: threading.Thread(target=post, args=(name,), name=f"{name} pairing")
        for name in clients
    }
    threads["first"].start()
    assert cache.inside.wait(BOUND_S)
    threads["second"].start()
    for thread in threads.values():
        thread.join(BOUND_S * 2)
        assert not thread.is_alive()
    assert outcomes["first"] is None
    assert isinstance(outcomes["second"], gloo.GlooPairInUseError)
    assert gloo.cached_store_key("volume-0") == "torchstore_gloo_first"


@pytest.mark.asyncio
async def test_forgetting_an_unheld_gloo_pair_spares_one_a_live_client_holds(pairing):
    """Keep a pair a live client holds, and forget one it no longer holds, retiring it once no transfer holds its group."""
    client = _client(pairing.volume)
    await _put(client, "weights", torch.ones(4))
    [(key, group)] = _groups(client).items()
    gloo.forget_unheld_pair("volume-0")
    assert gloo.cached_store_key("volume-0") == key and pairing.retired == []
    # A transfer under way still holds the group, so the pair retires only once it lets go.
    _groups(client).clear()
    gloo.forget_unheld_pair("volume-0")
    assert gloo.cached_store_key("volume-0") is None and pairing.retired == []
    del group
    assert await pairing.retirements(1) == [("volume-0", key)]


class _ForgottenDuringLookup(dict):
    """A Gloo pair cache whose lookup forgets the pair, as a GC closing an abandoned call on this thread may."""

    def get(self, key: Any, default: Any = None) -> Any:
        return self.pop(key, default)


def _cache_pairs(*volume_ids: str) -> tuple[TransportContext, dict[str, Any]]:
    """Cache a pair for each of ``volume_ids`` in a new client context, as a pairing does.

    Returns the context and each pair's group by volume ID, held as a transfer under way holds it.
    """
    context = TransportContext()
    groups = context.get(GlooProcessGroupCache)
    held = {}
    for port, volume_id in enumerate(volume_ids):
        key = f"key-{volume_id}"
        held[volume_id] = _Group()
        groups.put(key, held[volume_id])
        gloo._pair_owners[key] = groups
        gloo._pair_groups[key] = held[volume_id]
        gloo._store_addrs[volume_id] = ("host", port, key, True)
    return context, held


def test_forgetting_gloo_pairs_drops_the_named_ones_and_retires_each_once_unheld(
    pairing,
):
    """Forget the named volume IDs' pairs and drop their groups from the client context given.

    Each retires once no transfer holds its group, and the other pairs stay cached.
    """
    context, held = _cache_pairs("a", "b", "c")
    gloo._forget_pairs(iter(["missing", "a", "b"]), context)
    assert set(gloo._store_addrs) == {"c"}
    assert list(context.get(GlooProcessGroupCache)._process_groups) == ["key-c"]
    # Transfers under way still hold the dropped groups.
    assert pairing.retired == []
    del held["a"], held["b"]
    assert pairing.retired == [("a", "key-a"), ("b", "key-b")]


def test_forgetting_a_gloo_pair_without_its_context_retires_it_once_its_client_lets_go(
    pairing,
):
    """Forget a pair without its client context, which keeps the group, and retire it once that client drops it."""
    context, held = _cache_pairs("d")
    groups = context.get(GlooProcessGroupCache)
    gloo._forget_pairs(["d"])
    assert gloo._store_addrs == {} and groups.lookup("key-d") is held["d"]
    del held["d"]
    assert pairing.retired == []
    groups.clear()
    assert pairing.retired == [("d", "key-d")]


def test_forgetting_a_gloo_pair_under_a_key_no_longer_cached_keeps_the_cached_one(
    pairing,
):
    """Keep the cached pair when forgetting an older store key, and drop only that key's group from the context.

    This forget reports nothing: whatever uncached the older pair reported it.
    """
    context, _ = _cache_pairs("c")
    groups = context.get(GlooProcessGroupCache)
    groups.put("key-old", _Group())
    gloo.forget_pair("c", "key-old", context)
    assert gloo.cached_store_key("c") == "key-c"
    assert list(groups._process_groups) == ["key-c"] and pairing.retired == []


def test_a_gloo_pair_a_gc_forgets_during_its_lookup_is_reported_by_that_forget_alone(
    monkeypatch, pairing
):
    """Report nothing from a forget whose lookup finds the pair gone: the GC forget that took it reports it."""
    context, held = _cache_pairs("a")
    monkeypatch.setattr(gloo, "_store_addrs", _ForgottenDuringLookup(gloo._store_addrs))
    gloo.forget_pair("a", "key-a", context)
    groups = context.get(GlooProcessGroupCache)
    assert (gloo._store_addrs, groups.lookup("key-a")) == ({}, None)
    del held["a"]
    assert pairing.retired == []


def test_forgetting_gloo_pairs_takes_a_bare_id_as_one_volume(pairing):
    """Take a bare volume ID as one volume's, not as its characters."""
    _cache_pairs("volume-e", "v")
    gloo._forget_pairs("volume-e")
    assert set(gloo._store_addrs) == {"v"}


@pytest.mark.asyncio
async def test_a_gloo_pair_forgotten_through_another_context_retires_once_its_own_client_lets_go(
    monkeypatch, pairing
):
    """Retire a pair forgotten through a context that does not hold it only once its own client lets go.

    A put that found the pair before the forget finishes on it, and the volume keeps its half meanwhile.
    """
    client, other = _client(pairing.volume), _client(pairing.volume)
    await _put(client, "warm", torch.zeros(4))
    [key] = _groups(client)
    pre_put_hook = GlooTransportBuffer._pre_put_hook

    async def forget_through_the_other_context(self, requests):
        gloo._forget_pairs("volume-0", other.transport_context)
        await pre_put_hook(self, requests)

    monkeypatch.setattr(
        GlooTransportBuffer, "_pre_put_hook", forget_through_the_other_context
    )
    await _put(client, "a", torch.full((4,), 1.0))
    monkeypatch.setattr(GlooTransportBuffer, "_pre_put_hook", pre_put_hook)
    assert torch.equal(pairing.volume.store.kv["a"], torch.full((4,), 1.0))
    assert gloo.cached_store_key("volume-0") is None and list(_groups(client)) == [key]
    assert pairing.retired == []
    del client
    assert await pairing.retirements(1) == [("volume-0", key)]


@pytest.mark.parametrize("found", ["cached", "freshly_paired"])
@pytest.mark.asyncio
async def test_a_gloo_put_finishes_on_the_group_it_holds_after_its_pair_is_forgotten(
    monkeypatch, pairing, found
):
    """Finish a put, on its own tag, whose pair another call forgot right after the put found it."""
    client = _client(pairing.volume)
    if found == "cached":
        await _put(client, "warm", torch.zeros(4))
    pre_put_hook = GlooTransportBuffer._pre_put_hook

    async def forget_first(self, requests):
        # As another call's failure drops the pair.
        gloo._forget_pairs("volume-0", client.transport_context)
        await pre_put_hook(self, requests)

    monkeypatch.setattr(GlooTransportBuffer, "_pre_put_hook", forget_first)
    await _put(client, "a", torch.full((4,), 1.0))
    assert torch.equal(pairing.volume.store.kv["a"], torch.full((4,), 1.0))
    assert gloo.cached_store_key("volume-0") is None
    [pair] = pairing.pairs.values()
    assert all(tag != 0 for tag, _ in pair.sends(0))


@pytest.mark.asyncio
async def test_a_finished_gloo_buffer_holds_no_pair(pairing):
    """Let go of a buffer's group and TCPStore master once its put ends, so a buffer kept alive holds neither.

    A kept error keeps its buffer so.
    """
    client = _client(pairing.volume)
    buffer = GlooTransportBuffer(client)
    await bounded(
        buffer.put_to_storage_volume([Request.from_tensor("a", torch.ones(4))])
    )
    [key] = _groups(client)
    [store] = pairing.stores.values()
    store_freed = _freed(store)
    gloo._forget_pairs("volume-0", client.transport_context)
    # The pair retires once its group is collected.
    assert await pairing.retirements(1) == [("volume-0", key)]
    assert await asyncio.to_thread(store_freed.wait, BOUND_S)
    assert buffer.storage_volume_ref is client


@pytest.mark.parametrize("direction", ["put", "get"])
@pytest.mark.asyncio
async def test_a_gloo_call_that_fails_before_its_half_starts_never_posts_it(
    monkeypatch, caplog, pair, direction
):
    """Skip a transfer's half when its call fails before the loop runs the half, after the buffer moved another key.

    The call raises its own error at once, the half ends cancelled, and nothing waits for Gloo's timeout or logs.
    """
    store = pair.ref.volume.store
    store.kv["first"] = torch.arange(4.0)
    store.kv["second"] = torch.arange(4.0)
    endpoint = pair.ref.volume.put if direction == "put" else pair.ref.volume.get
    name = "call" if direction == "put" else "call_one"
    call = getattr(endpoint, name)

    async def refuse_the_second(buffer, requests):
        if requests[0].key == "second":
            raise ConnectionError("refused before it was sent")
        return await call(buffer, requests)

    monkeypatch.setattr(endpoint, name, refuse_the_second)
    hook_name = "_pre_put_hook" if direction == "put" else "_pre_get_hook"
    hook = getattr(GlooTransportBuffer, hook_name)
    halves: list[asyncio.Task] = []

    async def keep_the_half(self, requests):
        await hook(self, requests)
        halves.append(self._send_task if direction == "put" else self._recv_task)

    monkeypatch.setattr(GlooTransportBuffer, hook_name, keep_the_half)
    buffer = GlooTransportBuffer(pair.ref)
    with pytest.raises(ConnectionError, match="refused before it was sent"):
        if direction == "put":
            await bounded(
                buffer.put_to_storage_volume(
                    [Request.from_tensor(key, torch.ones(4)) for key in store.kv]
                )
            )
        else:
            await bounded(
                buffer.get_from_storage_volume([Request(key=key) for key in store.kv])
            )
    # The second key's half ended cancelled once the loop ran it, and only the first key's halves posted.
    await bounded(asyncio.wait([halves[1]]))
    assert halves[1].cancelled()
    assert len(pair.posted) == 2 and pair.waiting() == []
    assert _warnings(caplog) == []


@pytest.mark.asyncio
async def test_a_refused_gloo_handshake_logs_its_waiting_half_once_that_fails(
    monkeypatch, caplog, pairing
):
    """Raise a handshake the volume refused at once, then log its client half's failure once that half gives up.

    The buffer lets go of the half's TCPStore master, which only the half holds until then.
    """
    may_give_up = threading.Event()

    def client_half_alone(store: Any, rank: int, **kwargs: Any) -> Any:
        if rank == 0:
            assert may_give_up.wait(BOUND_S)
            raise RuntimeError("the volume never joined")
        return pairing.group(store, rank)

    monkeypatch.setattr(gloo, "_gloo_factory", client_half_alone)
    # As in a volume's process whose exit began, which starts no pairing.
    monkeypatch.setattr(gloo, "_exit_deadline", time.monotonic())
    buffer = GlooTransportBuffer(_client(pairing.volume))
    with pytest.raises(RuntimeError, match="was not started: exiting$"):
        await bounded(
            buffer.put_to_storage_volume([Request.from_tensor("k", torch.ones(4))])
        )
    [store] = pairing.stores.values()
    assert store() is not None and _warnings(caplog) == []
    may_give_up.set()
    message = "The client half of a Gloo pairing whose handshake failed then failed"
    await until(lambda: _warnings(caplog) == [(gloo.__name__, message)])
    [record] = [r for r in caplog.records if r.name == gloo.__name__]
    assert record.exc_info is not None
    assert str(record.exc_info[1]) == "the volume never joined"
    # The error's traceback holds the half's frames, and so the store, while the record keeps it.
    for kept in caplog.records:
        kept.exc_info = None
    del record
    await until(lambda: store() is None)
    assert [r for r in caplog.records if r.name == "asyncio"] == []


@pytest.mark.parametrize("ending", ["failed", "cancelled"])
@pytest.mark.asyncio
async def test_a_gloo_pairing_whose_client_half_fails_or_is_cancelled_retires_and_caches_nothing(
    monkeypatch, caplog, pairing, ending
):
    """Raise a pairing whose client half failed, or that was cancelled while it waited for that half.

    The volume made its half as the handshake returned, so that pair retires at once. Nothing is cached,
    and the half's own error is not logged again.
    """
    gave_up, may_end = RuntimeError("the client half failed"), threading.Event()
    waiting = threading.Event()

    def client_half_late(store: Any, rank: int, **kwargs: Any) -> Any:
        if rank == 0:
            waiting.set()
            assert may_end.wait(BOUND_S)
            raise gave_up
        return pairing.group(store, rank)

    monkeypatch.setattr(gloo, "_gloo_factory", client_half_late)
    post_handshake, posting = GlooTransportBuffer._post_handshake, asyncio.Event()

    async def entered(self, *args):
        posting.set()
        await post_handshake(self, *args)

    monkeypatch.setattr(GlooTransportBuffer, "_post_handshake", entered)
    client = _client(pairing.volume)
    putting = asyncio.ensure_future(_put(client, "k", torch.ones(4)))
    # The volume made its half, and the put now waits for the client's.
    await bounded(posting.wait())
    assert waiting.wait(BOUND_S)
    [key] = pairing.halves()
    if ending == "failed":
        may_end.set()
        with pytest.raises(RuntimeError, match="the client half failed"):
            await bounded(putting)
    else:
        putting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await putting
        may_end.set()
    assert pairing.retired == [("volume-0", key)]
    assert gloo._store_addrs == {} and _groups(client) == {}
    assert _warnings(caplog) == []


@pytest.mark.parametrize(
    "failure", ["unawaited_receive", "awaited_receive", "send", "awaited_half"]
)
def test_a_failed_gloo_transfer_frees_its_buffer_without_a_cyclic_gc(
    monkeypatch, caplog, failure
):
    """Free a buffer once its failed transfer ended and its error was dropped, with the cyclic GC off.

    A get's receive that its failed call left fails later, a receive it awaited fails as it waits, a put's
    send fails on a closed pair and a pairing's client half fails as its put waits for it.
    """
    # Kept log records hold their errors, so none is kept.
    caplog.set_level(logging.CRITICAL)

    async def scenario() -> weakref.ref[GlooTransportBuffer]:
        pair = _cached_pair(monkeypatch, takes_tags=True)
        volume = pair.ref.volume
        volume.store.kv["k"] = torch.arange(4.0)
        may_fail, failed = asyncio.Event(), asyncio.Event()

        async def receive_fails(self, tensor, transport_context, tag):
            # A fresh error, as each failed Gloo wait raises, so no fake work keeps one.
            await may_fail.wait()
            failed.set()
            raise RuntimeError("the receive failed")

        async def served_once_the_receive_failed(buffer, requests):
            await failed.wait()
            return rpc_copy(buffer)

        def client_half_fails(store, rank, **kwargs):
            if rank == 0:
                raise RuntimeError("the client half failed")
            return pair.group(rank)

        if failure == "send":
            pair.expire()
        elif failure == "awaited_half":
            # No pair cached, so the put pairs and awaits its failing client half.
            monkeypatch.setattr(gloo, "_store_addrs", {})
            monkeypatch.setattr(gloo, "TCPStore", lambda **kwargs: None)
            monkeypatch.setattr(gloo, "_gloo_factory", client_half_fails)
        else:
            monkeypatch.setattr(GlooTransportBuffer, "_receive_tensor", receive_fails)
        if failure == "unawaited_receive":
            monkeypatch.setattr(volume.get, "call_one", lambda *a: _unreached(pair, *a))
        elif failure == "awaited_receive":
            may_fail.set()
            monkeypatch.setattr(volume.get, "call_one", served_once_the_receive_failed)
        buffer = GlooTransportBuffer(pair.ref)
        freed = weakref.ref(buffer)
        error, match = {
            "unawaited_receive": (ConnectionError, "never reached"),
            "awaited_receive": (RuntimeError, "the receive failed"),
            "send": (RuntimeError, _CLOSED),
            "awaited_half": (RuntimeError, "the client half failed"),
        }[failure]
        # Not through bounded(): on Python 3.10 a failed wait_for holds its task in a cycle.
        with pytest.raises(error, match=match):
            if failure in ("send", "awaited_half"):
                await buffer.put_to_storage_volume(
                    [Request.from_tensor("k", torch.ones(4))]
                )
            else:
                await buffer.get_from_storage_volume([Request(key="k")])
        del buffer
        if failure == "unawaited_receive":
            may_fail.set()
            await bounded(failed.wait())
            # The receive's done callbacks run in the next turns.
            await asyncio.sleep(0)
            await asyncio.sleep(0)
        # The fake volume keeps its failed tasks, whose errors' tracebacks reach the buffer.
        volume.tasks.clear()
        return freed

    gc.disable()
    try:
        assert asyncio.run(scenario())() is None
    finally:
        gc.enable()


def _post_handshake_with(ref: StorageVolumeRef, store_key: str, group: Any) -> None:
    """Cache a pairing of ``ref``'s client under ``store_key`` whose group is ``group``, on this thread."""
    buffer = GlooTransportBuffer(ref)
    buffer.master_addr, buffer.master_port = "host", 2
    buffer.store_key = store_key

    async def post() -> None:
        made: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        made.set_result(group)
        buffer._pg_task = cast(Any, made)
        await buffer._post_handshake([gloo._TAGS], [])

    asyncio.run(post())


class _ReadMidCaching(dict):
    """A Gloo pair cache that lets another thread read it as a pairing writes the address.

    The pairing goes on once ``go`` is set: by the reader after its read, or as it waits for the cache lock.
    """

    def __init__(self, written: threading.Event, go: threading.Event) -> None:
        super().__init__()
        self.written, self.go = written, go

    def __setitem__(self, key: Any, value: Any) -> None:
        super().__setitem__(key, value)
        if threading.current_thread().name == "pairing":
            self.written.set()
            assert self.go.wait(BOUND_S)


class _NotedLock:
    """Gloo's pair-cache lock, setting ``waiting`` when a thread must wait for it."""

    def __init__(self, waiting: threading.Event) -> None:
        self.lock, self.waiting = threading.RLock(), waiting

    def __enter__(self) -> "_NotedLock":
        if not self.lock.acquire(blocking=False):
            self.waiting.set()
            self.lock.acquire()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.lock.release()


def test_a_transfer_reading_a_pair_its_client_is_caching_reuses_it(
    monkeypatch, pairing
):
    """Reuse a pair whose address another thread of the client just wrote: its group and owner came first.

    The read lands while the pairing waits mid-caching however late the reader runs.
    """
    written, go = threading.Event(), threading.Event()
    cache = _ReadMidCaching(written, go)
    monkeypatch.setattr(gloo, "_store_addrs", cache)
    monkeypatch.setattr(gloo, "_cache_lock", _NotedLock(go))
    ref = _client(pairing.volume)
    group = _Group()
    reused: list[bool] = []

    def read_mid_caching() -> None:
        assert written.wait(BOUND_S)
        request = Request.from_tensor("weights", torch.ones(1))
        reused.append(not GlooTransportBuffer(ref).requires_handshake([request]))
        go.set()

    threads = [
        threading.Thread(
            target=_post_handshake_with,
            args=(ref, "torchstore_gloo_mid", group),
            name="pairing",
        ),
        threading.Thread(target=read_mid_caching),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(BOUND_S * 2)
        assert not thread.is_alive()
    assert reused == [True]
    assert dict(cache) == {"volume-0": ("host", 2, "torchstore_gloo_mid", True)}


def test_a_pairing_retires_an_unheld_pair_it_displaces_once_no_transfer_holds_its_group(
    monkeypatch, pairing
):
    """Retire the unheld pair a pairing displaces only once the transfer still holding its group lets go.

    The pairing registers its own group, so its pair waits the same way once displaced.
    """
    old = ("host", 1, "torchstore_gloo_old", True)
    held = _Group()
    monkeypatch.setattr(
        gloo, "_pair_groups", weakref.WeakValueDictionary({old[2]: held})
    )
    monkeypatch.setattr(gloo, "_store_addrs", {"volume-0": old})
    new = _Group()
    _post_handshake_with(_client(pairing.volume), "torchstore_gloo_new", new)
    assert (pairing.retired, gloo._pair_groups.get("torchstore_gloo_new")) == ([], new)
    del held
    assert pairing.retired == [("volume-0", old[2])]


class _BoundedLock:
    """Gloo's pair-cache lock, waited for at most ``BOUND_S``: one that is not reentrant fails the test, not the run."""

    def __init__(self, lock: Any) -> None:
        self.lock = lock

    def __enter__(self) -> "_BoundedLock":
        if not self.lock.acquire(timeout=BOUND_S):
            raise TimeoutError("Gloo's pair-cache lock was not taken again")
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.lock.release()


def test_a_pair_is_forgotten_on_a_thread_that_holds_the_pair_cache_lock(
    monkeypatch, pairing
):
    """Forget a pair on a thread holding Gloo's pair-cache lock, as a GC closing an abandoned call there does."""
    monkeypatch.setattr(gloo, "_cache_lock", _BoundedLock(gloo._cache_lock))
    gloo._store_addrs["volume-0"] = ("host", 1, "torchstore_gloo_held", True)
    errors: list[BaseException] = []

    def forget_holding_the_lock() -> None:
        try:
            with gloo._cache_lock:
                gloo._forget_pairs("volume-0")
        except BaseException as e:
            errors.append(e)

    thread = threading.Thread(target=forget_holding_the_lock, daemon=True)
    thread.start()
    thread.join(BOUND_S * 2)
    assert not thread.is_alive() and errors == [] and gloo._store_addrs == {}


class _ReadTask(asyncio.Task):
    """A task that notes when its outcome is read."""

    read = False

    def exception(self) -> BaseException | None:
        self.read = True
        return super().exception()


def test_logging_an_unawaited_cancelled_gloo_task_reads_its_cancel():
    """Read an unawaited task's cancel, which from Python 3.11 the task holds with its traceback's frames until read.

    The test checks the read itself, which makes a Python 3.10 run fail on a regression too.
    """

    async def cancelled() -> _ReadTask:
        task = _ReadTask(asyncio.Event().wait())
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.wait([task])
        gloo._log_unawaited_failure(task, "unused")
        return task

    assert asyncio.run(cancelled()).read


class _Interrupted(BaseException):
    """An error that is no ``Exception`` and that a pairing's native call could raise."""


def test_a_volume_pairing_raises_its_own_error_that_is_no_exception(monkeypatch):
    """Raise a pairing's own error though it is no ``Exception``, not a ``TimeoutError`` at the pairing's deadline."""
    monkeypatch.setattr(gloo, "_pairings", set())
    monkeypatch.setattr(gloo, "_exit_deadline", None)
    # A deadline 5s away: a pairing that never reports its error fails soon.
    monkeypatch.setattr(gloo, "TORCHSTORE_GLOO_INIT_TIMEOUT", 1)
    monkeypatch.setattr(gloo, "_PAIRING_SLACK_S", 1.0)

    def interrupted() -> Any:
        raise _Interrupted("the pairing was interrupted")

    with pytest.raises(_Interrupted, match="the pairing was interrupted"):
        asyncio.run(gloo._pair_with_deadline(interrupted, "interrupted pairing"))


def test_a_volume_pairing_that_raises_stop_iteration_raises_a_runtime_error_from_it(
    monkeypatch,
):
    """Raise at once a ``RuntimeError`` caused by a pairing's ``StopIteration`` since a future refuses that error."""
    monkeypatch.setattr(gloo, "_pairings", set())
    monkeypatch.setattr(gloo, "_exit_deadline", None)
    # A deadline 5s away: a pairing that never reports its error fails soon.
    monkeypatch.setattr(gloo, "TORCHSTORE_GLOO_INIT_TIMEOUT", 1)
    monkeypatch.setattr(gloo, "_PAIRING_SLACK_S", 1.0)

    def stops() -> Any:
        raise StopIteration("the pairing stopped")

    with pytest.raises(
        RuntimeError, match="^the native call raised StopIteration$"
    ) as raised:
        asyncio.run(gloo._pair_with_deadline(stops, "stopped pairing"))
    assert type(raised.value.__cause__) is StopIteration
    assert str(raised.value.__cause__) == "the pairing stopped"
