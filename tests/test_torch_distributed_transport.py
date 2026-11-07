# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the torch.distributed._transport StorageVolume transport."""

import asyncio
import gc
import logging
import pickle
import sys
import threading
import uuid
import weakref
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import pytest_asyncio
import torch

pytest.importorskip("torch.distributed._transport")

import torchstore.transport.torch_distributed as td_transport
from torch.distributed import Work
from torch.distributed._transport import (
    available_transports,
    register_transport,
    Transport,
)
from torchstore.storage_volume import InMemoryStore
from torchstore.transport.buffers import TransportContext
from torchstore.transport.torch_distributed import (
    _overlaps,
    TorchDistributedTransportBuffer,
    TorchTransportCache,
)
from torchstore.transport.types import Request, TensorSlice


class FakeWork(Work):
    def __init__(self, done: bool = True, error: Exception | None = None) -> None:
        super().__init__()
        self._future = torch.futures.Future()
        if error is not None:
            self._future.set_exception(error)
        elif done:
            self._future.set_result([])

    def get_future(self):
        return self._future

    def is_completed(self) -> bool:
        return self._future.done()


@dataclass(frozen=True)
class FakeRemoteBuffer:
    owner: str
    address: int
    length: int
    # Registration metadata, like a NIXL rkey.
    metadata: int = 0
    # (device, address, length) of the registration, which the peer loads.
    region: tuple[str, int, int] = ("", 0, 0)

    def serialize(self) -> bytes:
        return pickle.dumps(self)


class FakeMemory:
    def __init__(
        self, transport: "FakeTransport", tensor: torch.Tensor, state: SimpleNamespace
    ) -> None:
        self.transport = transport
        self.tensor = tensor
        # Shared by aliases of the same byte range.
        self.state = state

    def check(self) -> "FakeMemory":
        if not self.state.active:
            raise RuntimeError("memory is not registered")
        return self

    def to_view(self, offset=None, length=None):
        self.check()
        offset = offset or 0
        if length is None:
            length = self.tensor.nbytes - offset
        assert 0 <= offset and offset + length <= self.tensor.nbytes
        return FakeView(self, offset, length)

    def to_mutable_view(self, offset=None, length=None):
        return self.to_view(offset, length)

    def to_remote_buffer(self, *, timeout=None):
        self.check()
        region = (str(self.tensor.device), self.tensor.data_ptr(), self.tensor.nbytes)
        return FakeRemoteBuffer(
            self.transport.name, region[1], region[2], self.state.metadata, region
        )

    def bytes(self) -> torch.Tensor:
        return self.tensor.reshape(-1).view(torch.uint8)


class FakeView:
    def __init__(self, memory: FakeMemory, offset: int, length: int) -> None:
        self.memory = memory
        self.offset = offset
        self.length = length

    def check(self) -> "FakeView":
        self.memory.check()
        return self

    def bytes(self) -> torch.Tensor:
        return self.memory.bytes()[self.offset : self.offset + self.length]


class FakeTransport(Transport):
    """In-process one-sided transport resolving peers through a class registry."""

    endpoints: dict[str, "FakeTransport"] = {}
    hang = False
    registrations = 0
    # Failed unregisters (busy, invalid memory, closed transport); must stay empty.
    failed_unregisters: list["FakeMemory"] = []

    def __init__(self, device=None) -> None:
        super().__init__(device)
        self.name = uuid.uuid4().hex
        self.peers: set[str] = set()
        self.memories: dict[int, FakeMemory] = {}
        # Like NIXL, registering a byte range again aliases its registration.
        self.ranges: dict[tuple[int, int, str], SimpleNamespace] = {}
        self.closed = False
        # (work, local memory, remote buffer, copy) of hung transfers.
        self.pending: list[tuple[FakeWork, FakeView, FakeRemoteBuffer, object]] = []
        FakeTransport.endpoints[self.name] = self

    @staticmethod
    def supported() -> bool:
        return True

    def _check_open(self) -> None:
        if self.closed:
            raise RuntimeError("transport is closed")

    def bind(self, *, timeout=None) -> bytes:
        self._check_open()
        return self.name.encode()

    def connect(self, peer_url: bytes, *, timeout=None) -> int:
        self._check_open()
        if peer_url.decode() not in FakeTransport.endpoints:
            raise RuntimeError("unknown peer")
        self.peers.add(peer_url.decode())
        return 0

    def connected(self) -> bool:
        return bool(self.peers) and not self.closed

    def register_memory(self, tensor, *, timeout=None):
        self._check_open()
        # Like NIXL, which cannot register 0-dim tensors.
        assert tensor.is_contiguous() and tensor.dim() > 0 and tensor.numel() > 0
        key = (tensor.data_ptr(), tensor.nbytes, str(tensor.device))
        state = self.ranges.get(key)
        if state is None:
            # New metadata per registration, like a NIXL rkey.
            FakeTransport.registrations += 1
            metadata = FakeTransport.registrations
            state = self.ranges[key] = SimpleNamespace(active=True, metadata=metadata)
        memory = FakeMemory(self, tensor, state)
        self.memories[id(memory)] = memory
        return memory

    def _busy(self, memory: FakeMemory) -> bool:
        """Whether an unfinished local or remote transfer uses memory."""
        for transport in FakeTransport.endpoints.values():
            for work, local, remote, _ in transport.pending:
                if work.get_future().done():
                    continue
                start = memory.tensor.data_ptr()
                if local.memory is memory or (
                    remote.owner == self.name
                    and start <= remote.address < start + memory.tensor.nbytes
                ):
                    return True
        return False

    def unregister_memory(self, memory, *, timeout=None) -> None:
        try:
            self._check_open()
            memory.check()
            if self._busy(memory):
                raise RuntimeError("memory is in use by a pending transfer")
        except Exception:
            FakeTransport.failed_unregisters.append(memory)
            raise
        # Unregistering any alias invalidates all of them.
        memory.state.active = False
        self.ranges = {k: s for k, s in self.ranges.items() if s.active}
        self.memories = {k: m for k, m in self.memories.items() if m.state.active}

    def _remote(self, remote_buffer: FakeRemoteBuffer) -> torch.Tensor:
        # Like NIXL, transfers need a connected peer.
        assert remote_buffer.owner in self.peers
        peer = FakeTransport.endpoints[remote_buffer.owner]
        assert not peer.closed
        device, start, length = remote_buffer.region
        state = peer.ranges.get((start, length, device))
        if state is None or state.metadata != remote_buffer.metadata:
            # Like RDMA with a revoked rkey.
            raise RuntimeError("remote access error: stale descriptor")
        offset = remote_buffer.address - start
        assert 0 <= offset and offset + remote_buffer.length <= length
        for memory in peer.memories.values():
            if memory.state is state:
                return memory.bytes()[offset : offset + remote_buffer.length]
        raise AssertionError("active registration without memory")

    def _transfer(self, local_buffer, remote_buffer, copy) -> FakeWork:
        self._check_open()
        local_buffer.check()
        # Like NIXL, which transfers the local view's size.
        assert local_buffer.length <= remote_buffer.length
        if FakeTransport.hang:
            work = FakeWork(done=False)
            self.pending.append((work, local_buffer, remote_buffer, copy))
            return work
        copy()
        return FakeWork()

    def write(self, local_buffer, remote_buffer, *, async_op=False, timeout=None):
        def copy():
            local = local_buffer.bytes()
            self._remote(remote_buffer)[: local.numel()].copy_(local)

        return self._transfer(local_buffer, remote_buffer, copy)

    def read(self, local_buffer, remote_buffer, *, async_op=False, timeout=None):
        def copy():
            local = local_buffer.bytes()
            local.copy_(self._remote(remote_buffer)[: local.numel()])

        return self._transfer(local_buffer, remote_buffer, copy)

    def close(self, *, timeout=None) -> None:
        if any(self._busy(memory) for memory in self.memories.values()):
            raise RuntimeError("transport has pending transfers")
        self.closed = True
        for state in self.ranges.values():
            state.active = False
        self.ranges.clear()
        self.memories.clear()


class UnsupportedTransport(FakeTransport):
    @staticmethod
    def supported() -> bool:
        return False


if "torchstore-fake" not in available_transports():
    register_transport("torchstore-fake", FakeTransport)
    register_transport("torchstore-unsupported", UnsupportedTransport)


def _copy(value):
    return pickle.loads(pickle.dumps(value))


class FakeVolume:
    """Runs StorageVolume endpoints in-process, pickling buffers like an RPC."""

    def __init__(self) -> None:
        self.store = InMemoryStore()
        self.put = SimpleNamespace(call=self._put)
        self.get = SimpleNamespace(call_one=self._get)
        self.get_meta = SimpleNamespace(call_one=self._get_meta)
        # Like an RPC, cancelling a caller does not cancel the volume's task.
        self.tasks: list[asyncio.Task] = []
        # Like an actor, handle one request at a time, in order.
        self.lock = asyncio.Lock()

    def _spawn(self, coro):
        async def serial():
            async with self.lock:
                return await coro

        task = asyncio.ensure_future(serial())
        self.tasks.append(task)
        return asyncio.shield(task)

    async def _put(self, buffer, requests):
        await self._spawn(self.store.put(_copy(buffer), requests))

    async def _get(self, buffer, requests):
        return _copy(await self._spawn(self.store.get(_copy(buffer), requests)))

    async def _get_meta(self, requests):
        return _copy(await self._spawn(self.store.get_meta(requests)))


@pytest.fixture
def probe(monkeypatch):
    td_transport.torch_transport_available.cache_clear()
    yield monkeypatch
    td_transport.torch_transport_available.cache_clear()


@pytest.fixture
def backend(probe):
    probe.setattr(
        td_transport, "ENV_TORCHSTORE_TORCH_TRANSPORT_BACKEND", "torchstore-fake"
    )
    assert td_transport.torch_transport_available()
    FakeTransport.endpoints = {}
    FakeTransport.hang = False
    FakeTransport.registrations = 0
    FakeTransport.failed_unregisters = []
    yield
    assert not FakeTransport.failed_unregisters


@pytest_asyncio.fixture
async def ref(backend):
    transport_context = TransportContext()
    volume = FakeVolume()
    yield SimpleNamespace(
        volume_id="test-volume",
        volume=volume,
        transport_context=transport_context,
    )
    # Fail transfers a failed test left hung, so its requests drain before the
    # event loop closes.
    for transport in FakeTransport.endpoints.values():
        for work, *_ in transport.pending:
            if not work.get_future().done():
                work.get_future().set_exception(RuntimeError("test ended"))
    transport_context.clear()
    volume.store.reset()


def _client_cache(ref):
    return ref.transport_context.get(TorchTransportCache)


def _client_endpoint(ref):
    return _client_cache(ref).endpoint


def _other_volume(ref):
    """A second storage volume sharing ref's client."""
    return SimpleNamespace(
        volume_id="other-volume",
        volume=FakeVolume(),
        transport_context=ref.transport_context,
    )


def _volume_cache(ref):
    return ref.volume.store.transport_context.get(TorchTransportCache)


def _volume_endpoint(ref):
    return _volume_cache(ref).volume_endpoint


_BOUND_S = 5


async def _bounded(awaitable):
    return await asyncio.wait_for(awaitable, timeout=_BOUND_S)


async def _until(predicate):
    async def poll():
        while not predicate():
            await asyncio.sleep(0)

    await _bounded(poll())


async def _put(ref, requests):
    await _bounded(TorchDistributedTransportBuffer(ref).put_to_storage_volume(requests))


async def _get(ref, requests):
    return await _bounded(
        TorchDistributedTransportBuffer(ref).get_from_storage_volume(requests)
    )


def _finish(transport, count=None):
    """Complete the first count (default all) hung transfers in submission
    order; both sides' memory must still be registered."""
    for work, local, _, copy in transport.pending:
        if work.get_future().done():
            continue
        if count is not None:
            if count == 0:
                return
            count -= 1
        assert id(local.memory) in transport.memories
        copy()
        work.get_future().set_result([])


@pytest.mark.parametrize("name", ["", "does-not-exist", "torchstore-unsupported"])
def test_unavailable_backend(probe, name):
    probe.setattr(td_transport, "ENV_TORCHSTORE_TORCH_TRANSPORT_BACKEND", name)
    assert not td_transport.torch_transport_available()


def test_probe_closes_its_transport(probe):
    probe.setattr(
        td_transport, "ENV_TORCHSTORE_TORCH_TRANSPORT_BACKEND", "torchstore-fake"
    )
    FakeTransport.endpoints = {}
    assert td_transport.torch_transport_available()
    (transport,) = FakeTransport.endpoints.values()
    assert transport.closed


@pytest.mark.asyncio
async def test_put_get_roundtrip_with_objects_and_empty_tensors(ref):
    weight = torch.arange(8, dtype=torch.float32)
    empty = torch.empty(0, 3)
    non_contiguous = torch.arange(12, dtype=torch.int64).reshape(3, 4).t()
    scalar = torch.tensor(5.0)
    await _put(
        ref,
        [
            Request.from_tensor("weight", weight),
            Request.from_objects("config", {"lr": 0.1}),
            Request.from_tensor("empty", empty),
            Request.from_tensor("transposed", non_contiguous),
            Request.from_tensor("scalar", scalar),
        ],
    )

    destination = torch.zeros(8)
    results = await _get(
        ref,
        [
            Request.from_tensor("weight", destination),
            Request(key="config"),
            Request(key="empty"),
            Request(key="transposed"),
            Request(key="scalar"),
        ],
    )

    assert results[0] is destination
    assert torch.equal(destination, weight)
    assert results[1] == {"lr": 0.1}
    assert results[2].shape == empty.shape
    assert torch.equal(results[3], non_contiguous)
    assert torch.equal(results[4], scalar)
    assert _volume_cache(ref).peers == {_client_cache(ref).url}


@pytest.mark.asyncio
async def test_concurrent_requests_sharing_a_tensor(ref):
    tensor = torch.arange(4, dtype=torch.float32)
    await _put(ref, [Request.from_tensor("w", torch.zeros(4))])
    endpoint = _client_endpoint(ref)
    entered, release = [], asyncio.Event()
    put = ref.volume.put.call

    async def held_put(buffer, requests):
        entered.append(buffer)
        await release.wait()
        await put(buffer, requests)

    ref.volume.put.call = held_put
    registered = []
    register_memory = endpoint.transport.register_memory
    endpoint.transport.register_memory = lambda t, **kw: (
        registered.append(t) or register_memory(t, **kw)
    )
    tasks = asyncio.gather(
        *(_put(ref, [Request.from_tensor(f"w{i}", tensor)]) for i in range(4))
    )
    try:
        await _until(lambda: len(entered) == 4)
        ((_, count),) = endpoint._registrations.values()
        assert count == 4 and len(registered) == 1
    finally:
        release.set()
    await _bounded(tasks)

    assert not endpoint.transport.memories
    results = await _get(ref, [Request(key=f"w{i}") for i in range(4)])
    for result in results:
        assert torch.equal(result, tensor)


@pytest.mark.asyncio
async def test_first_get_with_tensor_slice_of_full_tensor(ref):
    source = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    await _put(ref, [Request.from_tensor("w", source)])
    # A new client's first request carries a slice of a regular tensor.
    other = SimpleNamespace(
        volume_id="test-volume",
        volume=ref.volume,
        transport_context=TransportContext(),
    )
    tensor_slice = TensorSlice(
        offsets=(1, 0),
        coordinates=(1,),
        global_shape=(2, 4),
        local_shape=(1, 4),
        mesh_shape=(2,),
    )
    try:
        (result,) = await _get(other, [Request(key="w", tensor_slice=tensor_slice)])
    finally:
        other.transport_context.clear()
    assert torch.equal(result, source[1:])


@pytest.mark.asyncio
async def test_non_contiguous_get_destination_is_staged(ref):
    source = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    await _put(ref, [Request.from_tensor("w", source)])
    destination = torch.empty(3, 2).t()

    # The result is returned in a contiguous staging tensor.
    (result,) = await _get(ref, [Request.from_tensor("w", destination)])
    assert result.is_contiguous() and torch.equal(result, source)
    assert result.data_ptr() != destination.data_ptr()


def test_request_context_state_strips_client_tensors():
    context = td_transport.TorchTransportRequestContext(tensor=torch.zeros(2))
    state = pickle.loads(pickle.dumps(context))
    assert state.tensor is None and context.tensor is not None


def _real_backend_params():
    params = []
    if "nixl" in available_transports():
        params.append(("nixl", "cpu"))
        if torch.cuda.is_available():
            params.append(("nixl", "cuda"))
    return params


@pytest.mark.parametrize("backend_name, device", _real_backend_params())
@pytest.mark.asyncio
async def test_real_backend_roundtrip(backend_name, device, probe, caplog):
    probe.setattr(td_transport, "ENV_TORCHSTORE_TORCH_TRANSPORT_BACKEND", backend_name)
    # Real backend setup can be slow on a loaded host.
    probe.setitem(globals(), "_BOUND_S", 60)
    if not td_transport.torch_transport_available():
        pytest.skip(f"{backend_name} transport is unusable")
    transport_context = TransportContext()
    volume = FakeVolume()
    ref = SimpleNamespace(
        volume_id="test-volume", volume=volume, transport_context=transport_context
    )
    weights = [
        torch.randn(1024, device=device),
        torch.arange(17, dtype=torch.int32, device=device),
        torch.zeros(8, dtype=torch.int64),
        torch.tensor(3.0, device=device),
    ]
    destinations = [torch.zeros_like(w) for w in weights]
    # Row views of one storage.
    rows = torch.randn(64, 16, device=device)
    destination_rows = torch.zeros_like(rows)
    weights += [rows[8:24], rows[40:56]]
    destinations += [destination_rows[48:], destination_rows[:16]]
    try:
        # Weight sync updates the same buffers in place every step.
        for step in range(6):
            for weight in weights:
                weight.add_(step)
            await _put(
                ref,
                [Request.from_tensor(f"w{i}", w) for i, w in enumerate(weights)],
            )
            await _get(
                ref,
                [Request.from_tensor(f"w{i}", d) for i, d in enumerate(destinations)],
            )
            for destination, weight in zip(destinations, weights, strict=True):
                assert torch.equal(destination, weight)
        assert not _client_endpoint(ref)._registrations
    finally:
        transport_context.clear()
        volume.store.reset()
    # Unregister and close failures are only logged.
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


@pytest.mark.parametrize("backend_name, device", _real_backend_params())
@pytest.mark.asyncio
async def test_real_backend_two_clients_two_volumes(
    backend_name, device, probe, caplog
):
    probe.setattr(td_transport, "ENV_TORCHSTORE_TORCH_TRANSPORT_BACKEND", backend_name)
    probe.setitem(globals(), "_BOUND_S", 60)
    if not td_transport.torch_transport_available():
        pytest.skip(f"{backend_name} transport is unusable")
    contexts = [TransportContext(), TransportContext()]
    volumes = [FakeVolume(), FakeVolume()]
    refs = [
        [
            SimpleNamespace(volume_id=f"v{v}", volume=volume, transport_context=ctx)
            for v, volume in enumerate(volumes)
        ]
        for ctx in contexts
    ]
    try:
        # Each client endpoint serves both volumes; each volume endpoint
        # connects to both clients.
        for c, client_refs in enumerate(refs):
            for v, ref in enumerate(client_refs):
                weight = torch.full((256,), 10.0 * c + v, device=device)
                await _put(ref, [Request.from_tensor(f"w{c}", weight)])
        for c, client_refs in enumerate(refs):
            for v, ref in enumerate(client_refs):
                other = 1 - c
                destination = torch.zeros(256, device=device)
                await _get(ref, [Request.from_tensor(f"w{other}", destination)])
                assert torch.equal(
                    destination.cpu(), torch.full((256,), 10.0 * other + v)
                )
        urls = {_client_cache(client_refs[0]).url for client_refs in refs}
        for ref in refs[0]:
            assert _volume_cache(ref).peers == urls
    finally:
        for ctx in contexts:
            ctx.clear()
        for volume in volumes:
            volume.store.reset()
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


@pytest.mark.asyncio
async def test_volumes_share_one_client_endpoint(ref, monkeypatch):
    binds = []
    bind = FakeTransport.bind
    monkeypatch.setattr(
        FakeTransport, "bind", lambda self, **kw: binds.append(self) or bind(self)
    )
    other = _other_volume(ref)
    try:
        weight = torch.arange(4, dtype=torch.float32)
        await _bounded(
            asyncio.gather(
                _put(ref, [Request.from_tensor("w", weight)]),
                _put(other, [Request.from_tensor("w", weight)]),
            )
        )
        cache = _client_cache(ref)
        client = _client_endpoint(ref)
        assert binds == [client.transport]
        assert not client.transport.memories
        for volume in (ref, other):
            assert _volume_cache(volume).peers == {cache.url}
        (result,) = await _get(other, [Request(key="w")])
        assert torch.equal(result, weight)
    finally:
        other.volume.store.reset()


@pytest.mark.asyncio
async def test_one_volume_endpoint_serves_two_clients(ref, monkeypatch):
    connects = []
    connect = FakeTransport.connect

    def connect_spy(self, peer_url, **kwargs):
        connects.append((self, peer_url))
        return connect(self, peer_url, **kwargs)

    monkeypatch.setattr(FakeTransport, "connect", connect_spy)
    other = SimpleNamespace(
        volume_id=ref.volume_id, volume=ref.volume, transport_context=TransportContext()
    )
    try:
        for _ in range(2):
            await _put(ref, [Request.from_tensor("a", torch.arange(4.0))])
            await _put(other, [Request.from_tensor("b", torch.arange(4.0) + 4)])
        volume = _volume_endpoint(ref).transport
        assert connects == [(volume, _client_cache(c).url) for c in (ref, other)]
        a, b = await _get(other, [Request(key="a"), Request(key="b")])
        assert torch.equal(a, torch.arange(4.0))
        assert torch.equal(b, torch.arange(4.0) + 4)
    finally:
        other.transport_context.clear()


def _block_volume_creation(monkeypatch, count=None):
    """Block FakeTransport creation off the main thread until released.

    Returns (created, started, release); blocks only the first count (default
    all) creations.
    """
    created, started, release = [], threading.Event(), threading.Event()
    init = FakeTransport.__init__

    def blocking_init(self, *args, **kwargs):
        if threading.current_thread() is not threading.main_thread():
            created.append(self)
            if count is None or len(created) <= count:
                started.set()
                release.wait(_BOUND_S)
        init(self, *args, **kwargs)

    monkeypatch.setattr(FakeTransport, "__init__", blocking_init)
    return created, started, release


@pytest.mark.asyncio
async def test_volume_endpoint_created_once_under_concurrent_requests(
    backend, monkeypatch
):
    client = FakeTransport()
    url = client.bind()
    created, started, release = _block_volume_creation(monkeypatch)
    cache = TorchTransportCache()
    tasks = asyncio.gather(*(cache.connect("torchstore-fake", url) for _ in range(2)))
    try:
        await _bounded(asyncio.to_thread(started.wait, _BOUND_S))
    finally:
        release.set()
    first, second = await _bounded(tasks)
    assert first is second is cache.volume_endpoint
    assert created == [first.transport]
    assert cache.peers == {url}
    cache.clear()
    assert first.transport.closed
    client.close()


@pytest.mark.asyncio
async def test_failed_volume_endpoint_creation_is_retried(backend, monkeypatch):
    client = FakeTransport()
    url = client.bind()
    init = FakeTransport.__init__
    attempts = []

    def flaky_init(self, *args, **kwargs):
        attempts.append(None)
        if len(attempts) == 1:
            raise RuntimeError("agent failed")
        init(self, *args, **kwargs)

    monkeypatch.setattr(FakeTransport, "__init__", flaky_init)
    cache = TorchTransportCache()
    with pytest.raises(RuntimeError, match="failed to create"):
        await _bounded(cache.connect("torchstore-fake", url))
    endpoint = await _bounded(cache.connect("torchstore-fake", url))
    assert endpoint.transport.connected()
    cache.clear()
    client.close()


@pytest.mark.asyncio
async def test_volume_clear_during_creation_closes_endpoint(backend, monkeypatch):
    client = FakeTransport()
    url = client.bind()
    created, started, release = _block_volume_creation(monkeypatch, count=1)
    cache = TorchTransportCache()
    task = asyncio.ensure_future(cache.connect("torchstore-fake", url))
    try:
        await _bounded(asyncio.to_thread(started.wait, _BOUND_S))
        cache.clear()
    finally:
        release.set()
    with pytest.raises(RuntimeError, match="cleared while connecting"):
        await _bounded(task)
    (old,) = created
    await _until(lambda: old.closed)
    assert cache.volume_endpoint is None

    endpoint = await _bounded(cache.connect("torchstore-fake", url))
    assert endpoint.transport is not old and endpoint.transport.connected()
    cache.clear()
    client.close()


@pytest.mark.asyncio
async def test_stale_creation_failure_keeps_new_creation(backend, monkeypatch):
    client = FakeTransport()
    url = client.bind()
    gates = [threading.Event(), threading.Event()]
    entered = []
    init = FakeTransport.__init__

    def gated_init(self, *args, **kwargs):
        if threading.current_thread() is not threading.main_thread():
            i = len(entered)
            entered.append(self)
            if i < 2:
                gates[i].wait(_BOUND_S)
                if i == 0:
                    raise RuntimeError("agent failed")
        init(self, *args, **kwargs)

    monkeypatch.setattr(FakeTransport, "__init__", gated_init)
    cache = TorchTransportCache()
    stale = asyncio.ensure_future(cache.connect("torchstore-fake", url))
    await _until(lambda: len(entered) == 1)
    cache.clear()
    first = asyncio.ensure_future(cache.connect("torchstore-fake", url))
    await _until(lambda: len(entered) == 2)
    gates[0].set()
    with pytest.raises(RuntimeError, match="failed to create"):
        await _bounded(stale)
    second = asyncio.ensure_future(cache.connect("torchstore-fake", url))
    gates[1].set()
    a, b = await _bounded(asyncio.gather(first, second))
    assert a is b and len(entered) == 2
    cache.clear()
    client.close()


@pytest.mark.asyncio
async def test_cancelled_connect_does_not_recreate_endpoint(backend, monkeypatch):
    client = FakeTransport()
    url = client.bind()
    created, started, release = _block_volume_creation(monkeypatch)
    cache = TorchTransportCache()
    task = asyncio.ensure_future(cache.connect("torchstore-fake", url))
    try:
        await _bounded(asyncio.to_thread(started.wait, _BOUND_S))
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release.set()
    endpoint = await _bounded(cache.connect("torchstore-fake", url))
    assert created == [endpoint.transport]
    cache.clear()
    client.close()


@pytest.mark.asyncio
async def test_creation_waiter_replaces_endpoint_broken_meanwhile(backend, monkeypatch):
    a, b = FakeTransport(), FakeTransport()
    url_a, url_b = a.bind(), b.bind()
    created, started, release = _block_volume_creation(monkeypatch, count=1)
    cache = TorchTransportCache()

    async def connect_and_fail():
        endpoint = await cache.connect("torchstore-fake", url_a)
        # A failed descriptor load closes the transport.
        endpoint.transport.closed = True

    first = asyncio.ensure_future(connect_and_fail())
    second = asyncio.ensure_future(cache.connect("torchstore-fake", url_b))
    try:
        await _bounded(asyncio.to_thread(started.wait, _BOUND_S))
    finally:
        release.set()
    await _bounded(first)
    endpoint = await _bounded(second)
    assert endpoint.transport is not created[0]
    assert endpoint.transport.connected()
    cache.clear()
    a.close()
    b.close()


@pytest.mark.asyncio
async def test_creation_waiters_share_endpoint_replaced_meanwhile(backend, monkeypatch):
    transports = [FakeTransport() for _ in range(3)]
    urls = [t.bind() for t in transports]
    created, started, release = _block_volume_creation(monkeypatch, count=1)
    cache = TorchTransportCache()

    async def connect_and_fail():
        endpoint = await cache.connect("torchstore-fake", urls[0])
        # A failed descriptor load closes the transport.
        endpoint.transport.closed = True

    first = asyncio.ensure_future(connect_and_fail())
    waiters = [
        asyncio.ensure_future(cache.connect("torchstore-fake", url)) for url in urls[1:]
    ]
    try:
        await _bounded(asyncio.to_thread(started.wait, _BOUND_S))
    finally:
        release.set()
    await _bounded(first)
    second, third = await _bounded(asyncio.gather(*waiters))
    assert second is third
    assert second.transport is not created[0]
    cache.clear()
    for t in transports:
        t.close()


@pytest.mark.asyncio
async def test_request_during_replacement_waits_for_new_endpoint(backend, monkeypatch):
    a, b = FakeTransport(), FakeTransport()
    url_a, url_b = a.bind(), b.bind()
    cache = TorchTransportCache()
    old = await _bounded(cache.connect("torchstore-fake", url_a))
    # A failed descriptor load closes the transport.
    old.transport.closed = True
    created, started, release = _block_volume_creation(monkeypatch, count=1)
    first = asyncio.ensure_future(cache.connect("torchstore-fake", url_a))
    try:
        await _bounded(asyncio.to_thread(started.wait, _BOUND_S))
        second = asyncio.ensure_future(cache.connect("torchstore-fake", url_b))
        await _until(lambda: second.get_coro().cr_await is not None)
    finally:
        release.set()
    new, other = await _bounded(asyncio.gather(first, second))
    assert new is other is not old
    assert created == [new.transport]
    cache.clear()
    a.close()
    b.close()


@pytest.mark.asyncio
async def test_replaced_endpoint_closes_after_its_requests(backend):
    client = FakeTransport()
    url = client.bind()
    cache = TorchTransportCache()
    old = await _bounded(cache.connect("torchstore-fake", url))
    old.active += 1
    # Broken but not yet closed by the backend.
    old.transport.peers.clear()
    new = await _bounded(cache.connect("torchstore-fake", url))
    assert new is not old
    assert not old.transport.closed
    old.release()
    assert old.transport.closed
    cache.clear()
    client.close()


@pytest.mark.asyncio
async def test_failed_volume_connect_is_retried(ref, monkeypatch):
    connect = FakeTransport.connect
    calls = []

    def flaky(self, url, **kwargs):
        calls.append(None)
        if len(calls) == 1:
            raise RuntimeError("connect failed")
        return connect(self, url, **kwargs)

    monkeypatch.setattr(FakeTransport, "connect", flaky)
    with pytest.raises(RuntimeError, match="connect failed"):
        await _put(ref, [Request.from_tensor("w", torch.zeros(4))])
    endpoint = _volume_endpoint(ref)
    await _put(ref, [Request.from_tensor("w", torch.ones(4))])
    assert len(calls) == 2
    # A connect failure does not rebuild the endpoint.
    assert _volume_endpoint(ref) is endpoint and not endpoint.transport.closed


@pytest.mark.asyncio
async def test_broken_volume_endpoint_is_replaced(ref, monkeypatch):
    await _put(ref, [Request.from_tensor("w", torch.zeros(4))])
    old = _volume_endpoint(ref)
    monkeypatch.setattr(old.transport, "connected", lambda: False)
    await _put(ref, [Request.from_tensor("w", torch.ones(4))])
    assert old.transport.closed
    assert _volume_endpoint(ref) is not old
    (result,) = await _get(ref, [Request(key="w")])
    assert torch.equal(result, torch.ones(4))


@pytest.mark.asyncio
async def test_bind_failure_closes_transport(ref, monkeypatch):
    bind = FakeTransport.bind
    transports = []

    def flaky(self, **kwargs):
        transports.append(self)
        if len(transports) == 1:
            raise RuntimeError("bind failed")
        return bind(self, **kwargs)

    monkeypatch.setattr(FakeTransport, "bind", flaky)
    with pytest.raises(RuntimeError, match="bind failed"):
        await _put(ref, [Request.from_tensor("w", torch.zeros(4))])
    assert transports[0].closed
    await _put(ref, [Request.from_tensor("w", torch.zeros(4))])


@pytest.mark.asyncio
async def test_failed_transfer_raises_and_releases_memory(ref, monkeypatch):
    await _put(ref, [Request.from_tensor("w", torch.zeros(4))])
    volume = _volume_endpoint(ref)
    monkeypatch.setattr(
        volume.transport,
        "read",
        lambda *args, **kwargs: FakeWork(error=RuntimeError("transfer failed")),
    )
    with pytest.raises(RuntimeError, match="transfer failed"):
        await _put(ref, [Request.from_tensor("w", torch.ones(4))])
    assert not _client_endpoint(ref).transport.memories
    assert not volume.transport.memories


def _spy_stream_waits(monkeypatch, record):
    """Call record(kind, device) when accelerator work is marked or waited for."""
    current_stream = torch.accelerator.current_stream

    class PendingEvent:
        def __init__(self, event, device):
            self.event = event
            self.device = device

        def query(self):
            return False

        def synchronize(self):
            record("wait", self.device)
            self.event.synchronize()

    def record_event(stream, event):
        record("record", stream.device)
        return PendingEvent(stream.record_event(event), stream.device)

    def spy(device=None):
        stream = current_stream(device)
        return SimpleNamespace(record_event=lambda event: record_event(stream, event))

    monkeypatch.setattr(torch.accelerator, "current_stream", spy)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.asyncio
async def test_client_waits_for_queued_kernels(ref, monkeypatch):
    events = []

    def spy(kind, device):
        if kind == "wait":
            off_loop = threading.current_thread() is not threading.main_thread()
            events.append(("wait", off_loop))

    _spy_stream_waits(monkeypatch, spy)
    register_memory = FakeTransport.register_memory

    def register_spy(self, tensor, **kwargs):
        if tensor.is_cuda:
            events.append(("register", None))
        return register_memory(self, tensor, **kwargs)

    monkeypatch.setattr(FakeTransport, "register_memory", register_spy)
    read = FakeTransport.read

    def read_spy(self, *args, **kwargs):
        events.append(("read", None))
        return read(self, *args, **kwargs)

    monkeypatch.setattr(FakeTransport, "read", read_spy)
    await _put(ref, [Request.from_tensor("w", torch.ones(4, device="cuda"))])
    # Registration overlaps queued kernels; the wait leaves the loop free.
    assert events == [("register", None), ("wait", True), ("read", None)]


@pytest.mark.asyncio
async def test_object_only_requests_skip_endpoints(ref):
    await _put(
        ref,
        [
            Request.from_objects("config", {"lr": 0.1}),
            Request.from_tensor("empty", torch.empty(0)),
        ],
    )
    results = await _get(ref, [Request(key="config"), Request(key="empty")])
    assert results[0] == {"lr": 0.1}
    assert results[1].numel() == 0
    assert _client_endpoint(ref) is None
    assert _volume_endpoint(ref) is None

    await _put(
        ref,
        [
            Request.from_objects("config", {"lr": 0.2}),
            Request.from_tensor("w", torch.ones(4)),
        ],
    )
    config, w = await _get(ref, [Request(key="config"), Request(key="w")])
    assert config == {"lr": 0.2}
    assert torch.equal(w, torch.ones(4))


@pytest.mark.asyncio
async def test_failed_submission_drains_submitted_transfers(ref, monkeypatch):
    await _put(ref, [Request.from_tensor("w", torch.zeros(4))])
    client = _client_endpoint(ref)
    volume = _volume_endpoint(ref)
    reads = []
    read = volume.transport.read

    def flaky_read(*args, **kwargs):
        reads.append(None)
        if len(reads) == 2:
            raise RuntimeError("submission failed")
        return read(*args, **kwargs)

    monkeypatch.setattr(volume.transport, "read", flaky_read)
    FakeTransport.hang = True
    task = asyncio.create_task(
        _put(
            ref,
            [
                Request.from_tensor("a", torch.ones(4)),
                Request.from_tensor("b", torch.ones(4)),
            ],
        )
    )
    await _until(lambda: volume.transport.pending)
    assert not task.done()

    # Both sides keep memory registered until the submitted read finishes.
    _finish(volume.transport)
    with pytest.raises(RuntimeError, match="submission failed"):
        await _bounded(task)
    assert volume.active == 0
    assert not client.transport.closed and not volume.transport.closed
    assert not client.transport.memories

    FakeTransport.hang = False
    await _put(ref, [Request.from_tensor("a", torch.ones(4))])
    assert _volume_endpoint(ref) is volume


@pytest.mark.asyncio
async def test_cancelled_volume_transfer_waits_for_drain(ref, monkeypatch):
    await _put(ref, [Request.from_tensor("w", torch.zeros(4))])
    volume = _volume_endpoint(ref)

    waits = asyncio.Queue()
    wait = asyncio.wait

    async def spy(*args, **kwargs):
        # Count only the drain waits in _Endpoint.transfer, not wait_all's.
        if sys._getframe(1).f_code is td_transport._Endpoint.transfer.__code__:
            waits.put_nowait(None)
        return await wait(*args, **kwargs)

    monkeypatch.setattr(td_transport.asyncio, "wait", spy)
    monkeypatch.setattr(td_transport, "_leaked", [])
    FakeTransport.hang = True
    task = asyncio.create_task(_get(ref, [Request(key="w")]))
    await _until(lambda: volume.transport.pending)
    volume_task = ref.volume.tasks[-1]
    # Each cancellation must return the volume RPC to waiting for the drain.
    for _ in range(3):
        volume_task.cancel()
        await _bounded(waits.get())
    # The volume RPC stays open, so the client keeps its memory.
    assert not volume_task.done()
    assert not task.done()

    _finish(volume.transport)
    with pytest.raises(asyncio.CancelledError):
        await task
    assert volume.active == 0
    # The client sees a cancelled call, so it cannot know the volume finished.
    assert td_transport._leaked


def test_overlaps():
    assert _overlaps((4, 8, "cpu"), (8, 2, "cpu"))
    assert _overlaps((8, 2, "cpu"), (4, 8, "cpu"))
    assert not _overlaps((0, 4, "cpu"), (4, 4, "cpu"))
    assert not _overlaps((4, 4, "cpu"), (0, 4, "cpu"))
    assert not _overlaps((0, 4, "cpu"), (0, 4, "cuda:0"))


@pytest.mark.asyncio
async def test_cancelled_get_keeps_destination_until_volume_answers(ref, caplog):
    await _put(ref, [Request.from_tensor("w", torch.arange(4.0))])
    client = _client_endpoint(ref)
    volume = _volume_endpoint(ref)
    FakeTransport.hang = True
    cancelled = asyncio.create_task(_get(ref, [Request(key="w")]))
    await _until(lambda: len(volume.transport.pending) == 1)
    ((memory, _),) = client._registrations.values()
    result = weakref.ref(memory.tensor.untyped_storage())
    del memory

    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    del cancelled
    gc.collect()
    # The volume still writes into the result.
    assert result() is not None and client.transport.memories
    assert "keeping 1 torch transport destinations registered" in caplog.text

    _finish(volume.transport)
    await _until(lambda: not client.transport.memories)
    gc.collect()
    assert result() is None
    assert not client.transport.closed


@pytest.mark.asyncio
async def test_cancelled_get_destination_is_refused_until_volume_answers(ref):
    source = torch.arange(4.0)
    await _put(ref, [Request.from_tensor("w", source)])
    client = _client_endpoint(ref)
    volume = _volume_endpoint(ref)
    memory = torch.zeros(8)
    destination = memory[:4]
    FakeTransport.hang = True
    cancelled = asyncio.create_task(_get(ref, [Request.from_tensor("w", destination)]))
    await _until(lambda: len(volume.transport.pending) == 1)
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled

    # The volume may still write into memory overlapping these, including
    # other storages aliasing it.
    for overlapping in (destination, memory[2:6], torch.from_dlpack(memory)[1:5]):
        with pytest.raises(RuntimeError, match="cancelled get"):
            await _get(ref, [Request.from_tensor("w", overlapping)])
    key, _ = client.register(memory[4:])
    client.unregister([key])
    _finish(volume.transport)
    await _until(lambda: not client.transport.memories)

    FakeTransport.hang = False
    await _get(ref, [Request.from_tensor("w", destination)])
    assert torch.equal(destination, source)
    assert not client._abandoned


@pytest.mark.asyncio
async def test_cancelled_get_releases_when_volume_fails(ref):
    await _put(ref, [Request.from_tensor("w", torch.arange(4.0))])
    client = _client_endpoint(ref)
    volume = _volume_endpoint(ref)
    FakeTransport.hang = True
    cancelled = asyncio.create_task(
        _get(ref, [Request.from_tensor("w", torch.zeros(4))])
    )
    await _until(lambda: len(volume.transport.pending) == 1)
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    assert client._abandoned and client.active

    ((work, *_),) = volume.transport.pending
    work.get_future().set_exception(RuntimeError("transfer failed"))
    await _until(lambda: not client.active)
    assert not client._abandoned and not client.transport.memories


@pytest.mark.parametrize("request_cancelled", [True, False])
@pytest.mark.asyncio
async def test_cancelled_volume_call_leaks_destination(
    ref, monkeypatch, request_cancelled
):
    await _put(ref, [Request.from_tensor("w", torch.arange(4.0))])
    client = _client_endpoint(ref)
    volume = _volume_endpoint(ref)
    dest = torch.zeros(4)
    exposed = weakref.ref(dest.untyped_storage())
    calls = []
    get = ref.volume.get.call_one

    def call_one(*args):
        calls.append(asyncio.ensure_future(get(*args)))
        return calls[-1]

    monkeypatch.setattr(ref.volume.get, "call_one", call_one)
    monkeypatch.setattr(td_transport, "_leaked", [])
    FakeTransport.hang = True
    cancelled = asyncio.create_task(_get(ref, [Request.from_tensor("w", dest)]))
    await _until(lambda: len(volume.transport.pending) == 1)
    if request_cancelled:
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
    # E.g. event loop shutdown cancels every remaining task.
    calls[0].cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    await _until(lambda: td_transport._leaked)
    assert client._abandoned and client.active
    # The volume may still write into the destination; only _leaked keeps it.
    volume.transport.pending = []
    ref.transport_context.clear()
    FakeTransport.endpoints.pop(client.transport.name)
    del cancelled, dest, client, calls
    gc.collect()
    assert exposed() is not None


@pytest.mark.asyncio
async def test_two_cancelled_gets_hold_destination_until_both_answer(ref):
    await _put(ref, [Request.from_tensor("w", torch.arange(4.0))])
    client = _client_endpoint(ref)
    volume = _volume_endpoint(ref)
    dest = torch.zeros(4)
    FakeTransport.hang = True
    first = asyncio.create_task(_get(ref, [Request.from_tensor("w", dest)]))
    await _until(lambda: len(volume.transport.pending) == 1)
    second = asyncio.create_task(_get(ref, [Request.from_tensor("w", dest)]))
    await _until(lambda: len(ref.volume.tasks) == 3)
    for task in (first, second):
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    _finish(volume.transport)
    await _until(lambda: sum(client._abandoned.values()) == 1)
    # The second get's volume call may still write into dest.
    with pytest.raises(RuntimeError, match="cancelled get"):
        client.register(dest)
    await _until(lambda: len(volume.transport.pending) == 2)
    _finish(volume.transport)
    await _until(lambda: not client._abandoned)


@pytest.mark.asyncio
async def test_clear_closes_endpoint_once_cancelled_get_is_answered(ref):
    await _put(ref, [Request.from_tensor("w", torch.arange(4.0))])
    client = _client_endpoint(ref)
    volume = _volume_endpoint(ref)
    FakeTransport.hang = True
    cancelled = asyncio.create_task(_get(ref, [Request(key="w")]))
    await _until(lambda: len(volume.transport.pending) == 1)
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    ref.transport_context.clear()
    assert not client.transport.closed

    _finish(volume.transport)
    await _until(lambda: client.transport.closed)


@pytest.mark.asyncio
async def test_volume_reset_reconnects_without_failure(ref):
    await _put(ref, [Request.from_tensor("w", torch.zeros(4))])
    old = _volume_endpoint(ref)
    ref.volume.store.reset()
    assert old.transport.closed

    await _put(ref, [Request.from_tensor("w", torch.arange(4.0))])
    new = _volume_endpoint(ref)
    assert new is not old
    assert new.transport.peers == {_client_endpoint(ref).transport.name}
    (result,) = await _get(ref, [Request(key="w")])
    assert torch.equal(result, torch.arange(4.0))


@pytest.mark.asyncio
async def test_client_clear_creates_new_endpoint(ref):
    await _put(ref, [Request.from_tensor("w", torch.zeros(4))])
    old = _client_endpoint(ref)
    ref.transport_context.clear()
    assert old.transport.closed

    await _put(ref, [Request.from_tensor("w", torch.arange(4.0))])
    new = _client_endpoint(ref)
    assert new is not old
    volume = _volume_endpoint(ref)
    assert len(_volume_cache(ref).peers) == 2
    assert volume.transport.peers == {old.transport.name, new.transport.name}
    (result,) = await _get(ref, [Request(key="w")])
    assert torch.equal(result, torch.arange(4.0))


@pytest.mark.asyncio
async def test_clear_closes_endpoint_once_requests_finish(ref):
    await _put(ref, [Request.from_tensor("w", torch.zeros(4))])
    client = _client_endpoint(ref)
    volume = _volume_endpoint(ref)
    FakeTransport.hang = True
    task = asyncio.create_task(_put(ref, [Request.from_tensor("w", torch.ones(4))]))
    await _until(lambda: volume.transport.pending)
    ref.transport_context.clear()
    assert not client.transport.closed

    _finish(volume.transport)
    await _bounded(task)
    assert client.transport.closed


@pytest.mark.asyncio
async def test_failed_request_across_client_clear(ref, monkeypatch):
    put = ref.volume.put.call

    async def failing(buffer, requests):
        ref.transport_context.clear()
        raise RuntimeError("put failed")

    monkeypatch.setattr(ref.volume.put, "call", failing)
    with pytest.raises(RuntimeError, match="put failed"):
        await _put(ref, [Request.from_tensor("f", torch.arange(4.0))])
    monkeypatch.setattr(ref.volume.put, "call", put)

    await _put(ref, [Request.from_tensor("g", torch.ones(4))])
    (result,) = await _get(ref, [Request(key="g")])
    assert torch.equal(result, torch.ones(4))
