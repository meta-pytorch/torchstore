# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the torch.distributed._transport StorageVolume transport."""

import asyncio
import pickle
import threading
import uuid
from collections import OrderedDict
from concurrent.futures import Future
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("torch.distributed._transport")

import torch.distributed._transport as transport_api
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
        assert tensor.is_contiguous() and tensor.numel() > 0
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
        self.aborts = 0
        self.aborting = asyncio.Event()
        self.handshake_buffers = []
        # Like an RPC, cancelling a caller does not cancel the volume's task.
        self.tasks: list[asyncio.Task] = []

    def _spawn(self, coro):
        task = asyncio.ensure_future(coro)
        self.tasks.append(task)
        return asyncio.shield(task)

    async def _handshake(self, buffer, requests):
        buffer = _copy(buffer)
        if buffer._abort:
            self.aborts += 1
            self.aborting.set()
        else:
            self.handshakes += 1
            self.handshake_buffers.append(buffer)
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


def _reset_globals(monkeypatch) -> None:
    """Isolate state that outlives transport context resets."""
    monkeypatch.setattr(td_transport, "_background", set())
    monkeypatch.setattr(td_transport, "_orphans", [])
    monkeypatch.setattr(td_transport, "_leaked", [])
    monkeypatch.setattr(td_transport, "_leaked_requests", OrderedDict())
    monkeypatch.setattr(td_transport, "_closed_peers", OrderedDict())


@pytest.fixture
def backend(probe):
    probe.setattr(
        td_transport, "ENV_TORCHSTORE_TORCH_TRANSPORT_BACKEND", "torchstore-fake"
    )
    # Tests needing real timeouts set their own.
    probe.setenv("TORCHSTORE_TORCH_TRANSPORT_TIMEOUT_S", "30")
    _reset_globals(probe)
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


def _set_timeout(monkeypatch, timeout: str) -> None:
    monkeypatch.setenv("TORCHSTORE_TORCH_TRANSPORT_TIMEOUT_S", timeout)


def _time_out_first_wait(monkeypatch) -> None:
    """Time out the next wait_all at once, leaving its transfers pending."""
    wait_all = transport_api.wait_all
    calls = 0

    async def patched(works, timeout=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("timed out")
        return await wait_all(works, timeout=timeout)

    monkeypatch.setattr(transport_api, "wait_all", patched)


def _spy_aborts(monkeypatch) -> list[str]:
    results = []
    abort = TorchTransportCache.abort

    async def spy(self, peer, request_id):
        result = await abort(self, peer, request_id)
        results.append(result)
        return result

    monkeypatch.setattr(TorchTransportCache, "abort", spy)
    return results


def _client_cache(ref):
    return ref.transport_context.get(TorchTransportCache)


def _client_endpoint(ref):
    return _client_cache(ref).endpoints["test-volume"]


def _volume_cache(ref):
    return ref.volume.store.transport_context.get(TorchTransportCache)


def _volume_endpoints(ref):
    return _volume_cache(ref).peers


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


def test_unavailable_without_backend(probe):
    probe.setattr(td_transport, "ENV_TORCHSTORE_TORCH_TRANSPORT_BACKEND", "")
    assert not td_transport.torch_transport_available()


def test_unavailable_with_unknown_backend(probe):
    probe.setattr(
        td_transport, "ENV_TORCHSTORE_TORCH_TRANSPORT_BACKEND", "does-not-exist"
    )
    assert not td_transport.torch_transport_available()


def test_unavailable_with_unsupported_backend(probe):
    probe.setattr(
        td_transport, "ENV_TORCHSTORE_TORCH_TRANSPORT_BACKEND", "torchstore-unsupported"
    )
    assert not td_transport.torch_transport_available()


def test_probe_closes_its_transport(probe):
    probe.setattr(
        td_transport, "ENV_TORCHSTORE_TORCH_TRANSPORT_BACKEND", "torchstore-fake"
    )
    FakeTransport.endpoints = {}
    assert td_transport.torch_transport_available()
    (transport,) = FakeTransport.endpoints.values()
    assert transport.closed


@pytest.mark.parametrize("timeout", ["0", "-1", "inf", "nan", "abc"])
def test_invalid_timeout_is_rejected(ref, monkeypatch, timeout):
    _set_timeout(monkeypatch, timeout)
    with pytest.raises(ValueError, match="TIMEOUT_S"):
        TorchDistributedTransportBuffer(ref)


def test_invalid_timeout_disables_transport(probe):
    probe.setattr(
        td_transport, "ENV_TORCHSTORE_TORCH_TRANSPORT_BACKEND", "torchstore-fake"
    )
    _set_timeout(probe, "0")
    assert not td_transport.torch_transport_available()


@pytest.mark.asyncio
async def test_put_get_roundtrip_with_objects_and_empty_tensors(ref):
    weight = torch.arange(8, dtype=torch.float32)
    empty = torch.empty(0, 3)
    non_contiguous = torch.arange(12, dtype=torch.int64).reshape(3, 4).t()
    await _put(
        ref,
        [
            Request.from_tensor("weight", weight),
            Request.from_objects("config", {"lr": 0.1}),
            Request.from_tensor("empty", empty),
            Request.from_tensor("transposed", non_contiguous),
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
        ],
    )

    assert results[0] is destination
    assert torch.equal(destination, weight)
    assert results[1] == {"lr": 0.1}
    assert results[2].shape == empty.shape
    assert torch.equal(results[3], non_contiguous)
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
async def test_shared_tensor_registration_is_refcounted(ref):
    await _put(ref, [Request.from_tensor("w", torch.zeros(4))])
    endpoint = _client_endpoint(ref)
    tensor = torch.ones(4)

    first, memory = endpoint.register(tensor)
    second, same_memory = endpoint.register(tensor)
    assert first == second and memory is same_memory
    endpoint.unregister(first)
    assert endpoint.transport.memories
    endpoint.unregister(second)
    assert not endpoint.transport.memories


@pytest.mark.asyncio
async def test_transfer_timeout_releases_memory_after_drain(ref, monkeypatch):
    _set_timeout(monkeypatch, "2")
    source = torch.arange(4, dtype=torch.float32)
    await _put(ref, [Request.from_tensor("w", source)])
    old_client = _client_endpoint(ref)
    (old_volume,) = _volume_endpoints(ref).values()

    FakeTransport.hang = True
    _time_out_first_wait(monkeypatch)
    task = asyncio.create_task(_get(ref, [Request(key="w")]))
    await _bounded(ref.volume.aborting.wait())
    # The volume dropped the peer; both sides keep memory registered while
    # DMA may continue.
    assert old_volume not in _volume_endpoints(ref).values()
    assert old_volume.transport.memories and old_client.transport.memories

    _finish(old_volume.transport)
    with pytest.raises(TimeoutError, match="timed out"):
        await task
    assert old_volume.transport.closed and old_client.transport.closed
    assert ref.volume.aborts == 1

    FakeTransport.hang = False
    (result,) = await _get(ref, [Request(key="w")])
    assert torch.equal(result, source)
    assert ref.volume.handshakes == 2
    assert not _client_cache(ref).retired


@pytest.mark.asyncio
async def test_unfinished_transfer_leaks_memory(ref, monkeypatch):
    _set_timeout(monkeypatch, "0.5")
    await _put(ref, [Request.from_tensor("w", torch.zeros(4))])
    old_client = _client_endpoint(ref)
    (old_volume,) = _volume_endpoints(ref).values()
    aborts = _spy_aborts(monkeypatch)

    FakeTransport.hang = True
    with pytest.raises(TimeoutError, match="timed out"):
        await _get(ref, [Request(key="w")])

    # Unregistering memory a transfer may still use is unsafe.
    assert aborts == ["leaked"]
    assert old_volume._registrations and old_volume.transport.memories
    assert old_client._registrations and old_client.transport.memories
    assert not old_volume.closed and not old_client.closed
    assert old_client.retired
    assert td_transport._leaked
    # The leaked endpoint is no longer rescanned for closed-endpoint reports.
    assert not _client_cache(ref).retired
    assert old_client in td_transport._orphans


@pytest.mark.asyncio
async def test_cancelled_request_keeps_endpoint_once_transfer_finishes(
    ref, monkeypatch
):
    await _put(ref, [Request.from_tensor("w", torch.zeros(4))])
    old_client = _client_endpoint(ref)
    (old_volume,) = _volume_endpoints(ref).values()
    aborts = _spy_aborts(monkeypatch)

    FakeTransport.hang = True
    task = asyncio.create_task(_get(ref, [Request(key="w")]))
    await _until(lambda: old_volume.transport.pending)
    task.cancel()
    await _bounded(ref.volume.aborting.wait())
    # The volume's transfer keeps running.
    assert old_client.transport.memories

    _finish(old_volume.transport)
    with pytest.raises(asyncio.CancelledError):
        await task
    assert aborts == ["kept"]
    assert _client_endpoint(ref) is old_client
    assert not old_client.transport.memories
    assert not old_volume.transport.memories


@pytest.mark.asyncio
async def test_cancelled_volume_transfer_releases_memory_after_drain(ref):
    await _put(ref, [Request.from_tensor("w", torch.zeros(4))])
    old_client = _client_endpoint(ref)
    (old_volume,) = _volume_endpoints(ref).values()

    FakeTransport.hang = True
    task = asyncio.create_task(_get(ref, [Request(key="w")]))
    await _until(lambda: old_volume.transport.pending)
    ref.volume.tasks[-1].cancel()
    await _bounded(ref.volume.aborting.wait())
    assert old_volume not in _volume_endpoints(ref).values()
    assert old_volume.transport.memories and old_client.transport.memories

    _finish(old_volume.transport)
    with pytest.raises(asyncio.CancelledError):
        await task
    assert old_volume.transport.closed and old_client.transport.closed


@pytest.mark.asyncio
async def test_failed_submission_waits_for_submitted_transfers(ref, monkeypatch):
    await _put(ref, [Request.from_tensor("a", torch.zeros(4))])
    old_client = _client_endpoint(ref)
    (volume,) = _volume_endpoints(ref).values()
    read = volume.transport.read
    unregister_memory = volume.transport.unregister_memory
    works = []
    # Whether the submitted transfer finished at each unregister.
    finished_at_unregister = []

    def spy_unregister(memory, **kwargs):
        finished_at_unregister.append(works[0].get_future().done())
        unregister_memory(memory, **kwargs)

    def fail_second(*args, **kwargs):
        if works:
            raise RuntimeError("submit failed")
        FakeTransport.hang = True
        work = read(*args, **kwargs)
        FakeTransport.hang = False
        works.append(work)
        asyncio.get_running_loop().call_soon(_finish, volume.transport)
        return work

    monkeypatch.setattr(volume.transport, "read", fail_second)
    monkeypatch.setattr(volume.transport, "unregister_memory", spy_unregister)
    with pytest.raises(RuntimeError, match="submit failed"):
        await _put(
            ref,
            [
                Request.from_tensor("a", torch.ones(4)),
                Request.from_tensor("b", torch.ones(4)),
            ],
        )

    # The submitted transfer finished before the volume released its memory
    # and dropped the peer.
    assert finished_at_unregister and all(finished_at_unregister)
    assert volume.transport.closed and old_client.transport.closed
    await _put(ref, [Request.from_tensor("a", torch.ones(4))])
    assert ref.volume.handshakes == 2


@pytest.mark.asyncio
async def test_unawaitable_transfer_leaks_memory(ref, monkeypatch):
    await _put(ref, [Request.from_tensor("w", torch.zeros(4))])
    old_client = _client_endpoint(ref)
    (old_volume,) = _volume_endpoints(ref).values()
    aborts = _spy_aborts(monkeypatch)
    write = old_volume.transport.write

    class NoFutureWork(FakeWork):
        def get_future(self):
            raise RuntimeError("no future")

    def unawaitable_write(*args, **kwargs):
        FakeTransport.hang = True
        try:
            write(*args, **kwargs)
        finally:
            FakeTransport.hang = False
        return NoFutureWork(done=False)

    monkeypatch.setattr(old_volume.transport, "write", unawaitable_write)
    with pytest.raises(RuntimeError, match="no future"):
        await _get(ref, [Request(key="w")])

    # wait_all couldn't wait for the transfer, so its memory is leaked.
    assert aborts == ["leaked"]
    assert old_volume._registrations and old_client._registrations


@pytest.mark.asyncio
async def test_fatal_error_quarantines_endpoint(ref, monkeypatch):
    await _put(ref, [Request.from_tensor("w", torch.zeros(4))])
    (old_volume,) = _volume_endpoints(ref).values()

    class Fatal(BaseException):
        pass

    class FatalWork(FakeWork):
        def get_future(self):
            raise Fatal()

    monkeypatch.setattr(
        old_volume.transport, "write", lambda *args, **kwargs: FatalWork(done=False)
    )
    with pytest.raises(Fatal):
        await _get(ref, [Request(key="w")])

    # The stored tensor is no longer reused in place.
    assert old_volume in _volume_cache(ref)._quarantined
    assert old_volume._registrations


@pytest.mark.asyncio
async def test_validation_failure_keeps_endpoint(ref):
    await _put(ref, [Request.from_tensor("w", torch.zeros(4))])
    endpoint = _client_endpoint(ref)

    with pytest.raises(KeyError, match="missing"):
        await _get(ref, [Request.from_tensor("missing", torch.zeros(4))])

    assert ref.volume.aborts == 1
    assert _client_endpoint(ref) is endpoint
    assert not endpoint.transport.memories
    await _get(ref, [Request(key="w")])
    assert ref.volume.handshakes == 1


@pytest.mark.asyncio
async def test_aborted_request_is_refused(ref):
    await _put(ref, [Request.from_tensor("w", torch.zeros(4))])
    endpoint = _client_endpoint(ref)
    entered, held = asyncio.Event(), []

    async def held_put(buffer, requests):
        # The RPC reaches the volume after the client gave up.
        held.append((_copy(buffer), requests))
        entered.set()
        await asyncio.Event().wait()

    ref.volume.put.call = held_put
    task = asyncio.create_task(_put(ref, [Request.from_tensor("w", torch.ones(4))]))
    await _bounded(entered.wait())
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert _client_endpoint(ref) is endpoint
    assert not endpoint.transport.memories

    ((buffer, requests),) = held
    with pytest.raises(RuntimeError, match="aborted"):
        await ref.volume.store.put(buffer, requests)
    (result,) = await _get(ref, [Request(key="w")])
    assert torch.equal(result, torch.zeros(4))


@pytest.mark.asyncio
async def test_failed_abort_leaks_memory(ref, monkeypatch):
    await _put(ref, [Request.from_tensor("w", torch.zeros(4))])
    endpoint = _client_endpoint(ref)

    async def fail(*args):
        raise RuntimeError("volume unreachable")

    monkeypatch.setattr(ref.volume.get, "call_one", fail)
    monkeypatch.setattr(ref.volume.handshake, "call_one", fail)
    with pytest.raises(RuntimeError, match="volume unreachable"):
        await _get(ref, [Request.from_tensor("w", torch.zeros(4))])

    assert endpoint.retired and endpoint.transport.memories
    assert not endpoint.closed


@pytest.mark.asyncio
async def test_client_reconnects_after_volume_reset(ref):
    await _put(ref, [Request.from_tensor("w", torch.zeros(4))])
    ref.volume.store.transport_context.clear()

    with pytest.raises(RuntimeError, match="No torch transport endpoint"):
        await _put(ref, [Request.from_tensor("w", torch.ones(4))])

    await _put(ref, [Request.from_tensor("w", torch.ones(4))])
    (result,) = await _get(ref, [Request(key="w")])
    assert torch.equal(result, torch.ones(4))
    assert ref.volume.handshakes == 2


@pytest.mark.asyncio
async def test_reconnect_closes_volume_endpoint_once_client_side_closes(ref):
    await _put(ref, [Request.from_tensor("w", torch.zeros(4))])
    (old_volume,) = _volume_endpoints(ref).values()
    cache = _client_cache(ref)
    cache.retire("test-volume", _client_endpoint(ref))

    await _put(ref, [Request.from_tensor("w", torch.ones(4))])

    assert old_volume.transport.closed
    assert len(_volume_endpoints(ref)) == 1
    assert not cache.retired


@pytest.mark.asyncio
async def test_failed_close_is_retried(ref, monkeypatch):
    await _put(ref, [Request.from_tensor("w", torch.zeros(4))])
    old_client = _client_endpoint(ref)
    (old_volume,) = _volume_endpoints(ref).values()
    close = old_client.transport.close
    calls = 0

    def fail_once(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("close failed")
        close(**kwargs)

    monkeypatch.setattr(old_client.transport, "close", fail_once)
    cache = _client_cache(ref)
    cache.retire("test-volume", old_client)
    assert not old_client.closed

    await _put(ref, [Request.from_tensor("w", torch.ones(4))])
    assert old_client.closed and calls == 2
    await _put(ref, [Request.from_tensor("w", torch.ones(4))])
    assert old_volume.transport.closed
    assert not cache.retired


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
async def test_cancelled_bind_closes_transport(ref, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    bind = FakeTransport.bind

    def blocking_bind(self, **kwargs):
        entered.set()
        release.wait(5)
        return bind(self, **kwargs)

    monkeypatch.setattr(FakeTransport, "bind", blocking_bind)
    task = asyncio.create_task(_put(ref, [Request.from_tensor("w", torch.zeros(4))]))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release.set()
    (transport,) = FakeTransport.endpoints.values()
    await _until(lambda: transport.closed)


@pytest.mark.asyncio
async def test_failed_client_connect_closes_volume_peer(ref, monkeypatch):
    connect = td_transport._connect
    monkeypatch.setattr(
        td_transport, "_connect", lambda transport, _: connect(transport, b"missing")
    )
    with pytest.raises(RuntimeError, match="unknown peer"):
        await _put(ref, [Request.from_tensor("w", torch.zeros(4))])
    (abandoned,) = _volume_endpoints(ref).values()
    cache = _client_cache(ref)
    assert cache.abandoned == [("test-volume", abandoned.id)]
    assert not cache.endpoints
    (client_transport,) = (
        t for t in FakeTransport.endpoints.values() if t is not abandoned.transport
    )
    assert client_transport.closed

    monkeypatch.setattr(td_transport, "_connect", connect)
    await _put(ref, [Request.from_tensor("w", torch.ones(4))])

    assert abandoned.transport.closed
    assert abandoned not in _volume_endpoints(ref).values()
    assert not cache.abandoned


@pytest.mark.asyncio
async def test_handshake_finishing_after_abandonment_closes_peer(ref, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    bind_and_connect = td_transport._bind_and_connect

    def blocking_bind_and_connect(*args):
        entered.set()
        release.wait()
        return bind_and_connect(*args)

    monkeypatch.setattr(td_transport, "_bind_and_connect", blocking_bind_and_connect)
    task = asyncio.create_task(_put(ref, [Request.from_tensor("w", torch.zeros(4))]))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        # A later request reports the handshake abandoned before it finishes.
        (buffer,) = ref.volume.handshake_buffers
        _volume_cache(ref).close_peers(buffer._client_id, [buffer._endpoint_id])
    finally:
        release.set()

    with pytest.raises(RuntimeError, match="abandoned"):
        await task
    assert not _volume_endpoints(ref)
    assert all(transport.closed for transport in FakeTransport.endpoints.values())


@pytest.mark.asyncio
async def test_volume_reset_during_transfer_leaks_memory(ref, monkeypatch):
    await _put(ref, [Request.from_tensor("w", torch.zeros(4))])
    old_client = _client_endpoint(ref)
    (old_volume,) = _volume_endpoints(ref).values()
    aborts = _spy_aborts(monkeypatch)

    FakeTransport.hang = True
    task = asyncio.create_task(_get(ref, [Request(key="w")]))
    await _until(lambda: old_volume.transport.pending)
    volume_task = ref.volume.tasks[-1]
    ref.volume.store.transport_context.clear()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # The reset lost the request's state, but its transfer may still run.
    assert aborts == ["leaked"]
    assert old_client.retired and old_client.transport.memories
    _finish(old_volume.transport)
    await _bounded(volume_task)
    assert old_volume.transport.closed


@pytest.mark.asyncio
async def test_volume_reset_keeps_draining_transfers(ref, monkeypatch):
    _set_timeout(monkeypatch, "2")
    await _put(ref, [Request.from_tensor("w", torch.zeros(4))])
    (old_volume,) = _volume_endpoints(ref).values()

    FakeTransport.hang = True
    _time_out_first_wait(monkeypatch)
    task = asyncio.create_task(_get(ref, [Request(key="w")]))
    await _bounded(ref.volume.aborting.wait())
    ref.volume.store.transport_context.clear()

    _finish(old_volume.transport)
    with pytest.raises(TimeoutError, match="timed out"):
        await task
    await _until(lambda: old_volume.transport.closed)


@pytest.mark.asyncio
async def test_volume_reset_during_handshake_closes_transports(ref, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    bind_and_connect = td_transport._bind_and_connect

    def blocking_bind_and_connect(*args):
        entered.set()
        release.wait()
        return bind_and_connect(*args)

    monkeypatch.setattr(td_transport, "_bind_and_connect", blocking_bind_and_connect)
    task = asyncio.create_task(_put(ref, [Request.from_tensor("w", torch.zeros(4))]))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        ref.volume.store.transport_context.clear()
    finally:
        release.set()

    with pytest.raises(RuntimeError, match="reset during the handshake"):
        await task
    assert not _volume_endpoints(ref)
    assert all(transport.closed for transport in FakeTransport.endpoints.values())


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
    monkeypatch.setattr(td_transport._Endpoint, "expose", lambda *args: False)
    with pytest.raises(RuntimeError, match="endpoint attempts"):
        await _put(ref, [Request.from_tensor("w", torch.zeros(4))])
    assert ref.volume.handshakes == td_transport._MAX_ENDPOINT_ATTEMPTS + 1
    assert not any(t.memories for t in FakeTransport.endpoints.values())


@pytest.mark.asyncio
async def test_reregistered_memory_with_new_metadata_replaces_endpoint(ref):
    FakeTransport.rekey = True
    tensor = torch.arange(4, dtype=torch.float32)
    await _put(ref, [Request.from_tensor("w", tensor)])
    old_client = _client_endpoint(ref)
    (old_volume,) = _volume_endpoints(ref).values()

    # Same address, new registration metadata.
    tensor += 1
    await _put(ref, [Request.from_tensor("w", tensor)])

    assert ref.volume.handshakes == 2
    assert old_client.transport.closed and old_volume.transport.closed
    (result,) = await _get(ref, [Request(key="w")])
    assert torch.equal(result, tensor)


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

    # Re-registering with new metadata reconnects without waiting.
    await _put(ref, [Request.from_tensor("w", tensor)])
    assert ref.volume.handshakes == 2
    assert not old_client.transport.closed
    assert not old_volume.transport.closed

    release.set()
    await _bounded(in_flight)
    assert old_client.transport.closed
    (result,) = await _get(ref, [Request(key="in-flight")])
    assert torch.equal(result, in_flight_tensor)
    assert old_volume.transport.closed


@pytest.mark.asyncio
async def test_failed_transfer_drops_peer(ref, monkeypatch):
    await _put(ref, [Request.from_tensor("w", torch.zeros(4))])
    old_client = _client_endpoint(ref)
    (volume,) = _volume_endpoints(ref).values()
    aborts = _spy_aborts(monkeypatch)
    monkeypatch.setattr(
        volume.transport,
        "read",
        lambda *args, **kwargs: FakeWork(error=RuntimeError("transfer failed")),
    )

    with pytest.raises(RuntimeError, match="transfer failed"):
        await _put(ref, [Request.from_tensor("w", torch.ones(4))])
    assert aborts == ["dropped"]
    assert volume.transport.closed and old_client.transport.closed

    await _put(ref, [Request.from_tensor("w", torch.ones(4))])
    assert ref.volume.handshakes == 2


@pytest.mark.asyncio
async def test_failed_drain_releases_memory(ref, monkeypatch):
    await _put(ref, [Request.from_tensor("w", torch.zeros(4))])
    (volume,) = _volume_endpoints(ref).values()
    aborts = _spy_aborts(monkeypatch)

    FakeTransport.hang = True
    _time_out_first_wait(monkeypatch)
    task = asyncio.create_task(_get(ref, [Request(key="w")]))
    await _bounded(ref.volume.aborting.wait())
    for work, *_ in volume.transport.pending:
        work.get_future().set_exception(RuntimeError("transfer failed"))

    with pytest.raises(TimeoutError, match="timed out"):
        await task
    assert aborts == ["dropped"]
    assert volume.transport.closed and not td_transport._leaked


@pytest.mark.asyncio
async def test_failed_request_without_registrations_skips_abort(ref):
    with pytest.raises(KeyError, match="missing"):
        await _get(ref, [Request.from_tensor("missing", torch.empty(0))])
    assert ref.volume.aborts == 0


@pytest.mark.asyncio
async def test_failed_bind_closes_transport(ref, monkeypatch):
    bind = FakeTransport.bind

    def fail(self, **kwargs):
        raise RuntimeError("bind failed")

    monkeypatch.setattr(FakeTransport, "bind", fail)
    with pytest.raises(RuntimeError, match="bind failed"):
        await _put(ref, [Request.from_tensor("w", torch.zeros(4))])
    (transport,) = FakeTransport.endpoints.values()
    assert transport.closed

    monkeypatch.setattr(FakeTransport, "bind", bind)
    await _put(ref, [Request.from_tensor("w", torch.zeros(4))])


@pytest.mark.asyncio
async def test_cancelled_volume_handshake_closes_transports(ref, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    bind_and_connect = td_transport._bind_and_connect

    def blocking_bind_and_connect(*args):
        entered.set()
        release.wait()
        return bind_and_connect(*args)

    monkeypatch.setattr(td_transport, "_bind_and_connect", blocking_bind_and_connect)
    task = asyncio.create_task(_put(ref, [Request.from_tensor("w", torch.zeros(4))]))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        ref.volume.tasks[-1].cancel()
    finally:
        release.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert not _volume_endpoints(ref)
    transports = list(FakeTransport.endpoints.values())
    await _until(lambda: all(transport.closed for transport in transports))


def test_remember_evicts_oldest():
    entries = OrderedDict()
    for key in "abc":
        td_transport._remember(entries, key, 3)
    # Remembering a key again refreshes it.
    td_transport._remember(entries, "a", 3)
    td_transport._remember(entries, "d", 3)
    assert list(entries) == ["c", "a", "d"]


def test_close_setup_result_closes_returned_transport():
    transport = FakeTransport()
    returned = Future()
    returned.set_result((transport, b"url"))
    td_transport._close_setup_result(returned)
    assert transport.closed

    abandoned, failed, cancelled = Future(), Future(), Future()
    abandoned.set_result(None)
    failed.set_exception(RuntimeError("bind failed"))
    cancelled.cancel()
    for future in (abandoned, failed, cancelled):
        td_transport._close_setup_result(future)


@pytest.mark.asyncio
async def test_close_peers_prunes_finished_outcomes(backend):
    cache = TorchTransportCache()
    cache.start(("client", "closed"), "finished").set_result("kept")
    cache.start(("client", "closed"), "running")
    cache.start(("client", "open"), "finished").set_result("kept")

    cache.close_peers("client", ["closed"])
    assert set(cache._outcomes) == {
        ("client", "closed", "running"),
        ("client", "open", "finished"),
    }
    assert cache.is_closed(("client", "closed"))
    assert not cache.is_closed(("client", "open"))


@pytest.mark.asyncio
async def test_client_reset_closes_endpoint_once_request_finishes(ref):
    await _put(ref, [Request.from_tensor("w", torch.zeros(4))])
    endpoint = _client_endpoint(ref)
    entered, release = asyncio.Event(), asyncio.Event()
    put = ref.volume.put.call

    async def held_put(buffer, requests):
        entered.set()
        await release.wait()
        await put(buffer, requests)

    ref.volume.put.call = held_put
    task = asyncio.create_task(_put(ref, [Request.from_tensor("w", torch.ones(4))]))
    await _bounded(entered.wait())
    ref.transport_context.clear()
    assert endpoint in td_transport._orphans and not endpoint.closed

    release.set()
    await _bounded(task)
    assert endpoint.closed
    ref.volume.put.call = put
    await _put(ref, [Request.from_tensor("w", torch.ones(4))])
    assert endpoint not in td_transport._orphans


@pytest.mark.asyncio
async def test_endpoint_retired_after_handshake_reconnects(ref, monkeypatch):
    await _put(ref, [Request.from_tensor("w", torch.zeros(4))])
    synchronize = TorchDistributedTransportBuffer._synchronize_streams
    retired = []

    def retire_once(self):
        if not retired:
            retired.append(_client_endpoint(ref))
            _client_cache(ref).retire("test-volume", retired[0])
        synchronize(self)

    monkeypatch.setattr(
        TorchDistributedTransportBuffer, "_synchronize_streams", retire_once
    )
    await _put(ref, [Request.from_tensor("w", torch.ones(4))])
    assert ref.volume.handshakes == 2
    (result,) = await _get(ref, [Request(key="w")])
    assert torch.equal(result, torch.ones(4))


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
async def test_new_put_avoids_storage_of_leaked_put(ref, monkeypatch):
    _set_timeout(monkeypatch, "0.5")
    await _put(ref, [Request.from_tensor("w", torch.zeros(4))])
    stored = ref.volume.store.kv["w"]
    # Puts normally update stored tensors in place.
    await _put(ref, [Request.from_tensor("w", torch.zeros(4))])
    assert ref.volume.store.kv["w"] is stored
    (old_volume,) = _volume_endpoints(ref).values()

    FakeTransport.hang = True
    with pytest.raises(TimeoutError, match="timed out"):
        await _put(ref, [Request.from_tensor("w", torch.ones(4))])
    FakeTransport.hang = False
    # Validation still runs before the put stores into a new tensor.
    with pytest.raises(AssertionError, match="!="):
        await _put(ref, [Request.from_tensor("w", torch.zeros(8))])
    assert ref.volume.store.kv["w"] is stored
    await _put(ref, [Request.from_tensor("w", torch.full((4,), 2.0))])
    assert ref.volume.store.kv["w"] is not stored

    # The stale DMA lands in the old storage only.
    _finish(old_volume.transport)
    assert torch.equal(stored, torch.ones(4))
    (result,) = await _get(ref, [Request(key="w")])
    assert torch.equal(result, torch.full((4,), 2.0))


@pytest.mark.asyncio
async def test_new_put_avoids_storage_of_draining_put(ref, monkeypatch):
    await _put(ref, [Request.from_tensor("w", torch.zeros(4))])
    stored = ref.volume.store.kv["w"]
    (old_volume,) = _volume_endpoints(ref).values()

    FakeTransport.hang = True
    _time_out_first_wait(monkeypatch)
    task = asyncio.create_task(_put(ref, [Request.from_tensor("w", torch.ones(4))]))
    await _bounded(ref.volume.aborting.wait())
    FakeTransport.hang = False
    # Another client puts while the volume drains the timed-out put.
    other = SimpleNamespace(
        volume_id="test-volume",
        volume=ref.volume,
        transport_context=TransportContext(),
    )
    try:
        await _put(other, [Request.from_tensor("w", torch.full((4,), 2.0))])
        assert ref.volume.store.kv["w"] is not stored

        _finish(old_volume.transport)
        with pytest.raises(TimeoutError, match="timed out"):
            await task
        assert torch.equal(stored, torch.ones(4))
        (result,) = await _get(other, [Request(key="w")])
        assert torch.equal(result, torch.full((4,), 2.0))
    finally:
        other.transport_context.clear()


@pytest.mark.asyncio
async def test_new_put_avoids_storage_of_orphaned_endpoint(ref, monkeypatch):
    await _put(ref, [Request.from_tensor("w", torch.zeros(4))])
    stored = ref.volume.store.kv["w"]
    (old_volume,) = _volume_endpoints(ref).values()
    # A volume reset orphans an endpoint whose transfer still uses the storage.
    key, _ = old_volume.register(stored)
    ref.volume.store.transport_context.clear()
    assert old_volume in td_transport._orphans
    with pytest.raises(RuntimeError, match="No torch transport endpoint"):
        await _put(ref, [Request.from_tensor("w", torch.ones(4))])

    await _put(ref, [Request.from_tensor("w", torch.ones(4))])
    assert ref.volume.store.kv["w"] is not stored
    old_volume.unregister(key)
    assert old_volume.closed


@pytest.mark.asyncio
async def test_endpoint_overlap_checks_byte_ranges(ref):
    await _put(ref, [Request.from_tensor("w", torch.zeros(4))])
    endpoint = _client_endpoint(ref)
    base = torch.zeros(8)
    key, _ = endpoint.register(base[2:4])
    try:
        assert endpoint.overlaps(base[3:6])
        assert endpoint.overlaps(base)
        assert not endpoint.overlaps(base[4:6])
        assert not endpoint.overlaps(base[:2])
        assert not endpoint.overlaps(torch.zeros(8))
    finally:
        endpoint.unregister(key)


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
async def test_client_reset_during_handshake_closes_transport(ref):
    handshake = ref.volume.handshake.call_one

    async def reset_after_handshake(buffer, requests):
        result = await handshake(buffer, requests)
        ref.transport_context.clear()
        return result

    ref.volume.handshake.call_one = reset_after_handshake
    with pytest.raises(RuntimeError, match="reset during the handshake"):
        await _put(ref, [Request.from_tensor("w", torch.zeros(4))])
    client_url = ref.volume.handshake_buffers[0]._client_url.decode()
    assert FakeTransport.endpoints[client_url].closed
    assert not _client_cache(ref).endpoints

    ref.volume.handshake.call_one = handshake
    await _put(ref, [Request.from_tensor("w", torch.ones(4))])
    (result,) = await _get(ref, [Request(key="w")])
    assert torch.equal(result, torch.ones(4))


@pytest.mark.asyncio
async def test_non_contiguous_get_destination_is_staged(ref):
    source = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    await _put(ref, [Request.from_tensor("w", source)])
    destination = torch.empty(3, 2).t()

    # The client copies the staged result into the destination.
    (result,) = await _get(ref, [Request.from_tensor("w", destination)])
    assert result.is_contiguous() and torch.equal(result, source)
    assert result.data_ptr() != destination.data_ptr()
    assert not _client_endpoint(ref)._registrations
    assert not _client_cache(ref).retired
    await _get(ref, [Request(key="w")])
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
async def test_real_backend_roundtrip(backend_name, device, probe):
    probe.setattr(td_transport, "ENV_TORCHSTORE_TORCH_TRANSPORT_BACKEND", backend_name)
    _reset_globals(probe)
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
    ]
    destinations = [torch.zeros_like(w) for w in weights]
    try:
        # Weight sync updates the same buffers in place every step.
        for step in range(3):
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

        # A request failing before its transfers start keeps the endpoint.
        with pytest.raises(KeyError, match="missing"):
            await _get(ref, [Request.from_tensor("missing", destinations[0])])
        assert volume.aborts == 1
        # NIXL CUDA re-registrations may carry new metadata and replace the
        # endpoint.
        if device == "cpu":
            assert volume.handshakes == 1
        client = transport_context.get(TorchTransportCache).endpoints["test-volume"]
        (peer,) = volume.store.transport_context.get(TorchTransportCache).peers.values()
        assert not client._registrations and not peer._registrations
    finally:
        transport_context.clear()
        volume.store.reset()
