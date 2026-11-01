# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the torch.distributed._transport StorageVolume transport."""

import asyncio
import logging
import pickle
import sys
import uuid
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
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
        return self.check()

    def to_mutable_view(self, offset=None, length=None):
        return self.check()

    def to_remote_buffer(self, *, timeout=None):
        self.check()
        return FakeRemoteBuffer(
            self.transport.name,
            self.tensor.data_ptr(),
            self.tensor.nbytes,
            self.state.metadata,
        )

    def bytes(self) -> torch.Tensor:
        return self.tensor.reshape(-1).view(torch.uint8)


class FakeTransport(Transport):
    """In-process one-sided transport resolving peers through a class registry."""

    endpoints: dict[str, "FakeTransport"] = {}
    hang = False
    # Give each registration new metadata, as when memory is reallocated.
    rekey = False
    registrations = 0
    # Failed unregisters (busy, invalid memory, closed transport); must stay empty.
    failed_unregisters: list["FakeMemory"] = []

    def __init__(self, device=None) -> None:
        super().__init__(device)
        self.name = uuid.uuid4().hex
        self.peer: str | None = None
        self.memories: dict[int, FakeMemory] = {}
        # Like NIXL, registering a byte range again aliases its registration.
        self.ranges: dict[tuple[int, int, str], SimpleNamespace] = {}
        self.closed = False
        # (work, local memory, remote buffer, copy) of hung transfers.
        self.pending: list[tuple[FakeWork, FakeMemory, FakeRemoteBuffer, object]] = []
        # Like NIXL, reject remote descriptors whose metadata changed.
        self.loaded: dict[tuple[int, int], int] = {}
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
        assert self.peer is None
        if peer_url.decode() not in FakeTransport.endpoints:
            raise RuntimeError("unknown peer")
        self.peer = peer_url.decode()
        return 0

    def connected(self) -> bool:
        return self.peer is not None and not self.closed

    def register_memory(self, tensor, *, timeout=None):
        self._check_open()
        # Like NIXL, which cannot register 0-dim tensors.
        assert tensor.is_contiguous() and tensor.dim() > 0 and tensor.numel() > 0
        key = (tensor.data_ptr(), tensor.nbytes, str(tensor.device))
        state = self.ranges.get(key)
        if state is None:
            FakeTransport.registrations += 1
            metadata = FakeTransport.registrations if FakeTransport.rekey else 0
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
                if local is memory or (
                    remote.owner == self.name
                    and remote.address == memory.tensor.data_ptr()
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
        assert remote_buffer.owner == self.peer
        descriptor = (remote_buffer.address, remote_buffer.length)
        loaded = self.loaded.setdefault(descriptor, remote_buffer.metadata)
        if loaded != remote_buffer.metadata:
            raise RuntimeError("NOT_ALLOWED: descriptor metadata changed")
        peer = FakeTransport.endpoints[self.peer]
        assert not peer.closed
        for memory in peer.memories.values():
            if memory.tensor.data_ptr() == remote_buffer.address:
                return memory.bytes()
        raise RuntimeError("remote buffer is not registered")

    def _transfer(self, local_buffer, remote_buffer, copy) -> FakeWork:
        self._check_open()
        local_buffer.check()
        if FakeTransport.hang:
            work = FakeWork(done=False)
            self.pending.append((work, local_buffer, remote_buffer, copy))
            return work
        copy()
        return FakeWork()

    def write(self, local_buffer, remote_buffer, *, async_op=False, timeout=None):
        def copy():
            self._remote(remote_buffer).copy_(local_buffer.bytes())

        return self._transfer(local_buffer, remote_buffer, copy)

    def read(self, local_buffer, remote_buffer, *, async_op=False, timeout=None):
        def copy():
            local_buffer.bytes().copy_(self._remote(remote_buffer))

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
        self.handshake = SimpleNamespace(call_one=self._handshake)
        self.put = SimpleNamespace(call=self._put)
        self.get = SimpleNamespace(call_one=self._get)
        self.get_meta = SimpleNamespace(call_one=self.store.get_meta)
        self.handshakes = 0
        # Like an RPC, cancelling a caller does not cancel the volume's task.
        self.tasks: list[asyncio.Task] = []

    def _spawn(self, coro):
        task = asyncio.ensure_future(coro)
        self.tasks.append(task)
        return asyncio.shield(task)

    async def _handshake(self, buffer, requests):
        buffer = _copy(buffer)
        self.handshakes += 1
        return _copy(await self._spawn(self.store.handshake(buffer, requests)))

    async def _put(self, buffer, requests):
        await self._spawn(self.store.put(_copy(buffer), requests))

    async def _get(self, buffer, requests):
        return _copy(await self._spawn(self.store.get(_copy(buffer), requests)))


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
    FakeTransport.rekey = False
    FakeTransport.failed_unregisters = []
    yield
    assert not FakeTransport.failed_unregisters


@pytest.fixture
def ref(backend):
    transport_context = TransportContext()
    volume = FakeVolume()
    yield SimpleNamespace(
        volume_id="test-volume",
        volume=volume,
        transport_context=transport_context,
    )
    transport_context.clear()
    volume.store.reset()


def _client_cache(ref):
    return ref.transport_context.get(TorchTransportCache)


def _client_endpoint(ref):
    return _client_cache(ref).endpoints["test-volume"]


def _volume_endpoints(ref):
    return ref.volume.store.transport_context.get(TorchTransportCache).peers


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


def _finish(transport):
    """Complete hung transfers; both sides' memory must still be registered."""
    for work, local, _, copy in transport.pending:
        if work.get_future().done():
            continue
        assert id(local) in transport.memories
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
    assert ref.volume.handshakes == 1
    # Registrations are released on both sides once requests complete.
    assert not _client_endpoint(ref).transport.memories
    (volume_endpoint,) = _volume_endpoints(ref).values()
    assert not volume_endpoint.transport.memories


@pytest.mark.asyncio
async def test_concurrent_requests_share_one_handshake(ref):
    tensors = [torch.full((4,), float(i)) for i in range(8)]
    await _bounded(
        asyncio.gather(
            *(
                _put(ref, [Request.from_tensor(f"w{i}", tensor)])
                for i, tensor in enumerate(tensors)
            )
        )
    )

    assert ref.volume.handshakes == 1
    results = await _get(ref, [Request(key=f"w{i}") for i in range(8)])
    for result, tensor in zip(results, tensors, strict=True):
        assert torch.equal(result, tensor)


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
    tasks = asyncio.gather(
        *(_put(ref, [Request.from_tensor(f"w{i}", tensor)]) for i in range(4))
    )
    try:
        await _until(lambda: len(entered) == 4)
        ((_, count),) = endpoint._registrations.values()
        assert count == 4
    finally:
        release.set()
    await _bounded(tasks)

    assert not endpoint.transport.memories
    results = await _get(ref, [Request(key=f"w{i}") for i in range(4)])
    for result in results:
        assert torch.equal(result, tensor)


@pytest.mark.asyncio
async def test_handshake_failure_closes_pending_transport(ref, monkeypatch):
    async def fail(*args):
        raise RuntimeError("handshake failed")

    monkeypatch.setattr(ref.volume.handshake, "call_one", fail)
    with pytest.raises(RuntimeError, match="handshake failed"):
        await _put(ref, [Request.from_tensor("w", torch.zeros(4))])

    (transport,) = FakeTransport.endpoints.values()
    assert transport.closed
    assert not _client_cache(ref).endpoints


@pytest.mark.asyncio
async def test_closed_endpoints_are_reported_to_their_volume(ref):
    other = SimpleNamespace(
        volume_id="other-volume",
        volume=FakeVolume(),
        transport_context=ref.transport_context,
    )
    try:
        await _put(ref, [Request.from_tensor("w", torch.zeros(4))])
        await _put(other, [Request.from_tensor("w", torch.zeros(4))])
        (old_volume,) = _volume_endpoints(ref).values()
        cache = _client_cache(ref)
        cache.retire("test-volume", _client_endpoint(ref))

        # Another volume's request must not consume the report.
        await _put(other, [Request.from_tensor("w", torch.ones(4))])
        assert cache.retired
        await _put(ref, [Request.from_tensor("w", torch.ones(4))])
        assert old_volume.transport.closed
        assert not cache.retired
    finally:
        other.volume.store.reset()


@pytest.mark.asyncio
async def test_persistent_descriptor_conflicts_fail_the_request(ref, monkeypatch):
    monkeypatch.setattr(td_transport._Endpoint, "expose", lambda *args: None)
    with pytest.raises(RuntimeError, match="endpoint attempts"):
        await _put(ref, [Request.from_tensor("w", torch.zeros(4))])
    assert ref.volume.handshakes == td_transport._MAX_ENDPOINT_ATTEMPTS + 1


@pytest.mark.asyncio
async def test_reregistered_memory_with_new_metadata_replaces_endpoint(ref):
    FakeTransport.rekey = True
    tensor = torch.zeros(4)
    await _put(ref, [Request.from_tensor("w", tensor)])
    old_client = _client_endpoint(ref)
    (old_volume,) = _volume_endpoints(ref).values()

    # Same byte range, new registration metadata.
    tensor.copy_(torch.arange(4))
    await _put(ref, [Request.from_tensor("w", tensor)])

    assert ref.volume.handshakes == 2
    assert old_client.transport.closed
    (result,) = await _get(ref, [Request(key="w")])
    assert torch.equal(result, tensor)
    assert old_volume.transport.closed


@pytest.mark.asyncio
async def test_in_flight_request_survives_endpoint_replacement(ref):
    FakeTransport.rekey = True
    tensor = torch.zeros(4)
    await _put(ref, [Request.from_tensor("w", tensor)])
    old_client = _client_endpoint(ref)
    (old_volume,) = _volume_endpoints(ref).values()

    # Hold the next put RPC after the client registered its memory.
    entered, release = asyncio.Event(), asyncio.Event()
    put = ref.volume.put.call

    async def blocked_put(buffer, requests):
        if not entered.is_set():
            entered.set()
            await release.wait()
        await put(buffer, requests)

    ref.volume.put.call = blocked_put
    in_flight_tensor = torch.full((4,), 7.0)
    in_flight = asyncio.create_task(
        _put(ref, [Request.from_tensor("in-flight", in_flight_tensor)])
    )
    await _bounded(entered.wait())

    # A descriptor conflict reconnects without waiting.
    tensor.fill_(1)
    await _put(ref, [Request.from_tensor("w", tensor)])
    assert ref.volume.handshakes == 2
    assert not old_client.transport.closed

    release.set()
    await _bounded(in_flight)
    assert old_client.transport.closed
    (result,) = await _get(ref, [Request(key="in-flight")])
    assert torch.equal(result, in_flight_tensor)
    assert old_volume.transport.closed


@pytest.mark.asyncio
async def test_too_many_exposed_descriptors_replaces_endpoint_once(ref, monkeypatch):
    monkeypatch.setattr(td_transport, "_MAX_EXPOSED", 2)
    await _put(
        ref,
        [
            Request.from_tensor("a", torch.zeros(4)),
            Request.from_tensor("b", torch.zeros(4)),
        ],
    )
    replace = TorchDistributedTransportBuffer._replace_endpoint
    replacing = []

    async def spy(self, endpoint):
        replacing.append(endpoint)
        await replace(self, endpoint)

    monkeypatch.setattr(TorchDistributedTransportBuffer, "_replace_endpoint", spy)
    lock = _client_cache(ref).lock("test-volume")
    await lock.acquire()
    try:
        tasks = [
            asyncio.create_task(_put(ref, [Request.from_tensor(key, torch.ones(4))]))
            for key in "ab"
        ]
        await _until(lambda: len(replacing) == 2)
    finally:
        lock.release()

    await _bounded(asyncio.gather(*tasks))
    # The second request found the endpoint already replaced.
    assert ref.volume.handshakes == 2


@pytest.mark.asyncio
async def test_failed_transfer_reconnects(ref, monkeypatch):
    await _put(ref, [Request.from_tensor("w", torch.zeros(4))])
    old_client = _client_endpoint(ref)
    (old_volume,) = _volume_endpoints(ref).values()
    monkeypatch.setattr(
        old_volume.transport,
        "read",
        lambda *args, **kwargs: FakeWork(error=RuntimeError("transfer failed")),
    )

    with pytest.raises(RuntimeError, match="transfer failed"):
        await _put(ref, [Request.from_tensor("w", torch.ones(4))])
    assert old_client.transport.closed

    await _put(ref, [Request.from_tensor("w", torch.ones(4))])
    assert ref.volume.handshakes == 2
    assert old_volume.transport.closed
    (result,) = await _get(ref, [Request(key="w")])
    assert torch.equal(result, torch.ones(4))


@pytest.mark.asyncio
async def test_volume_reset_reconnects_after_one_failed_request(ref):
    await _put(ref, [Request.from_tensor("w", torch.zeros(4))])
    ref.volume.store.reset()

    with pytest.raises(RuntimeError, match="No torch transport endpoint"):
        await _put(ref, [Request.from_tensor("w", torch.ones(4))])
    await _put(ref, [Request.from_tensor("w", torch.ones(4))])
    assert ref.volume.handshakes == 2
    (result,) = await _get(ref, [Request(key="w")])
    assert torch.equal(result, torch.ones(4))


@pytest.mark.parametrize("failing_connect", [1, 2], ids=["volume", "client"])
@pytest.mark.asyncio
async def test_connect_failure_closes_both_sides(ref, monkeypatch, failing_connect):
    connects = []
    connect = FakeTransport.connect

    def flaky_connect(self, peer_url, **kwargs):
        # The volume connects first, then the client.
        connects.append(self)
        if len(connects) == failing_connect:
            raise RuntimeError("connect failed")
        return connect(self, peer_url, **kwargs)

    monkeypatch.setattr(FakeTransport, "connect", flaky_connect)
    with pytest.raises(RuntimeError, match="connect failed"):
        await _put(ref, [Request.from_tensor("w", torch.zeros(4))])
    first_volume = connects[0]
    assert connects[-1].closed

    await _put(ref, [Request.from_tensor("w", torch.ones(4))])
    # The client reported its failed endpoint and the volume closed its side.
    assert first_volume.closed
    assert len(_volume_endpoints(ref)) == 1
    assert not _client_cache(ref).retired


@pytest.mark.asyncio
async def test_failed_submission_drains_submitted_transfers(ref, monkeypatch):
    await _put(ref, [Request.from_tensor("w", torch.zeros(4))])
    old_client = _client_endpoint(ref)
    (old_volume,) = _volume_endpoints(ref).values()
    reads = []
    read = old_volume.transport.read

    def flaky_read(*args, **kwargs):
        reads.append(None)
        if len(reads) == 2:
            raise RuntimeError("submission failed")
        return read(*args, **kwargs)

    monkeypatch.setattr(old_volume.transport, "read", flaky_read)
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
    await _until(lambda: old_volume.transport.pending)
    assert not task.done()

    # Both sides keep memory registered until the submitted read finishes.
    _finish(old_volume.transport)
    with pytest.raises(RuntimeError, match="submission failed"):
        await _bounded(task)
    assert old_client.transport.closed

    FakeTransport.hang = False
    await _put(ref, [Request.from_tensor("a", torch.ones(4))])
    assert old_volume.transport.closed
    assert ref.volume.handshakes == 2


@pytest.mark.asyncio
async def test_closed_endpoint_report_survives_failed_request(ref, monkeypatch):
    await _put(ref, [Request.from_tensor("w", torch.zeros(4))])
    (old_volume,) = _volume_endpoints(ref).values()
    cache = _client_cache(ref)
    cache.retire("test-volume", _client_endpoint(ref))
    handshake = ref.volume.handshake.call_one

    async def fail(*args):
        raise RuntimeError("handshake failed")

    monkeypatch.setattr(ref.volume.handshake, "call_one", fail)
    with pytest.raises(RuntimeError, match="handshake failed"):
        await _put(ref, [Request.from_tensor("w", torch.ones(4))])
    assert not old_volume.transport.closed

    monkeypatch.setattr(ref.volume.handshake, "call_one", handshake)
    await _put(ref, [Request.from_tensor("w", torch.ones(4))])
    assert old_volume.transport.closed
    assert not cache.retired


@pytest.mark.asyncio
async def test_cancelled_volume_transfer_waits_for_drain(ref, monkeypatch):
    await _put(ref, [Request.from_tensor("w", torch.zeros(4))])
    old_client = _client_endpoint(ref)
    (old_volume,) = _volume_endpoints(ref).values()

    waits = asyncio.Queue()
    wait = asyncio.wait

    async def spy(*args, **kwargs):
        # Count only the drain waits in _Endpoint.transfer, not wait_all's.
        if sys._getframe(1).f_code is td_transport._Endpoint.transfer.__code__:
            waits.put_nowait(None)
        return await wait(*args, **kwargs)

    monkeypatch.setattr(td_transport.asyncio, "wait", spy)
    FakeTransport.hang = True
    task = asyncio.create_task(_get(ref, [Request(key="w")]))
    await _until(lambda: old_volume.transport.pending)
    volume_task = ref.volume.tasks[-1]
    # Each cancellation must return the volume RPC to waiting for the drain.
    for _ in range(3):
        volume_task.cancel()
        await _bounded(waits.get())
    # The volume RPC stays open, so the client keeps its memory.
    assert not volume_task.done()
    assert not task.done()

    _finish(old_volume.transport)
    with pytest.raises(asyncio.CancelledError):
        await task
    assert old_volume.active == 0
    assert old_client.transport.closed


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

    # The client copies the staged result into the destination.
    (result,) = await _get(ref, [Request.from_tensor("w", destination)])
    assert result.is_contiguous() and torch.equal(result, source)
    assert result.data_ptr() != destination.data_ptr()
    assert not _client_cache(ref).retired
    assert ref.volume.handshakes == 1


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
        # NIXL CUDA re-registrations may carry new metadata and replace the
        # endpoint, but not on every request.
        assert volume.handshakes <= (1 if device == "cpu" else 3)
    finally:
        transport_context.clear()
        volume.store.reset()
    # Unregister and close failures are only logged.
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
