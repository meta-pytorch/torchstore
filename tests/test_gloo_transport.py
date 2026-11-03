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
import subprocess
import sys
import threading
import uuid
from datetime import timedelta
from types import SimpleNamespace
from typing import Any, cast

import pytest
import pytest_asyncio
import torch
import torchstore.transport.gloo as gloo
from torch.distributed import TCPStore
from torchstore.client import LocalClient
from torchstore.controller import ObjectType, StorageInfo
from torchstore.storage_volume import InMemoryStore
from torchstore.strategy import StorageVolumeRef
from torchstore.transport import TransportType
from torchstore.transport.buffers import TransportContext
from torchstore.transport.gloo import GlooProcessGroupCache, GlooTransportBuffer
from torchstore.transport.types import Request, TensorSlice

from .utils import bounded, FakeVolume, rpc_copy, until

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
        return SimpleNamespace(
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
    monkeypatch, caplog, pair, call, error, match, ending, logged
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
    name = receive.get_name()
    del receive, receives[:]
    found = [
        (r.name, r.levelno, r.exc_info and str(r.exc_info[1]))
        for r in caplog.records
        if r.levelno >= logging.WARNING and r.name == gloo.__name__
    ]
    # Asyncio reports an unretrieved error when it collects the task, which a logged traceback keeps
    # alive through the callback's frame, so the kept records drop their tracebacks first.
    for record in caplog.records:
        record.exc_info = None
    gc.collect()

    def about_this_receive(record):
        # Its unretrieved error names the task, and an error the get's callback raises names that.
        message = record.getMessage()
        callback = f"{gloo._log_receive_failure.__name__}("
        return f"name='{name}'" in message or callback in message

    found += [
        (r.name, r.levelno, None)
        for r in caplog.records
        if r.levelno >= logging.WARNING
        and r.name == "asyncio"
        and about_this_receive(r)
    ]
    timeout = "Timed out waiting for recv operation to complete"
    assert found == ([(gloo.__name__, logging.WARNING, timeout)] if logged else [])


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
    "destination, expected",
    [(None, torch.tensor([3.0])), (torch.full((3,), -1.0), torch.full((3,), 3.0))],
    ids=["plain", "in_place"],
)
@pytest.mark.asyncio
async def test_a_gloo_get_of_a_slice_past_its_tensors_end_returns_the_clipped_slice(
    caplog, pair, destination, expected
):
    """The volume clips such a slice and returns it in band, so an in-place get broadcasts it.

    The receive of the requested 3 elements ends with an empty send, and neither side logs.
    """
    pair.ref.volume.store.kv["s"] = torch.arange(4.0)
    past_end = TensorSlice(
        offsets=(3,), coordinates=(), global_shape=(4,), local_shape=(3,), mesh_shape=()
    )
    got = await bounded(
        _local_client(pair.ref).get(
            "s", inplace_tensor=destination, tensor_slice_spec=past_end
        )
    )
    assert torch.equal(got, expected)
    if destination is not None:
        assert got is destination
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
                    "torchstore.storage_volume",
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
                    "torchstore.storage_volume",
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


@pytest.mark.asyncio
async def test_a_gloo_volume_moves_a_client_before_tags_on_tag_0(pair):
    """A volume moves tensors on tag 0 for a buffer that carries no tag, as clients before tags send none."""
    client = pair.group(0)
    sent, received = torch.arange(4.0), torch.empty(4)
    put, get = Request.from_tensor("k", sent), Request(key="k")
    putting, getting = GlooTransportBuffer(pair.ref), GlooTransportBuffer(pair.ref)
    # The cached pair's address and the tensor's metadata, as such a client's own hooks set them.
    for buffer in (putting, getting):
        buffer.requires_handshake([put])
        buffer.shape, buffer.dtype = sent.shape, sent.dtype
        # Buffers that clients before tags pickle carry no tag.
        vars(buffer).pop("tag", None)
    sending = asyncio.to_thread(lambda: client.send([sent], dstRank=1, tag=0).wait())
    await bounded(
        asyncio.gather(sending, pair.ref.volume.put.call(putting, [put.meta_only()]))
    )
    receiving = asyncio.to_thread(
        lambda: client.recv([received], srcRank=1, tag=0).wait()
    )
    await bounded(
        asyncio.gather(receiving, pair.ref.volume.get.call_one(getting, [get]))
    )
    assert torch.equal(pair.ref.volume.store.kv["k"], sent)
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
            # The volume's copy of the client's buffer, which carries no tag.
            buffer = GlooTransportBuffer(_ref())
            buffer.store_key, buffer.shape = "pair", torch.Size([size])
            buffer.dtype = torch.float32
            with contextlib.suppress(KeyError):
                await bounded(volume.get(rpc_copy(buffer), [Request(key=key)]))

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
