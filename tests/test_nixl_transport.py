# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Focused tests for the NIXL StorageVolume transport lifecycle."""

import asyncio
import gc
import pickle
import time
import weakref
from types import SimpleNamespace
from typing import NamedTuple
from unittest.mock import AsyncMock, call, Mock

import pytest
import torch

import torchstore.transport.nixl as nixl_transport
from torchstore.transport.buffers import TransportContext
from torchstore.transport.nixl import (
    _NixlTransferOperation,
    _NixlTransferStatus,
    NixlAgentCache,
    NixlTransportBuffer,
)
from torchstore.transport.types import Request


class FakeNixlConfig:
    __slots__ = ("kwargs",)

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class FakeXferDList:
    def __init__(self, descriptors, memory_type="DRAM") -> None:
        self.descriptors = list(descriptors)
        self.memory_type = memory_type

    def getType(self):
        # Real NIXL returns an enum value, not the memory type name.
        return FakeNixlAgent.nixl_mems[self.memory_type]

    def descCount(self):
        return len(self.descriptors)

    def append(self, descriptor):
        self.descriptors.append(descriptor)

    def __getitem__(self, index):
        return self.descriptors[index]


class FakeXferHandle(NamedTuple):
    handle_id: int
    operation: _NixlTransferOperation
    local_descriptors: list[tuple[int, int, int]]
    remote_descriptors: list[tuple[int, int, int]]
    remote_agent: str
    remote_memory_type: int


class FakeNixlAgent:
    # Shared fake backend state.
    # Descriptors are (address, length, device) like NIXL's; overrides keyed by
    # tensor id simulate mixed DRAM/VRAM descriptors - tests can run purely on CPU.
    memory_types: dict[int, str] = {}
    devices: dict[int, int] = {}
    nixl_mems = {"DRAM": 0, "VRAM": 1}

    # Instance tracking for inspecting the client and server agents independently.
    instances: list["FakeNixlAgent"] = []

    # Per-test behavior controls.
    # Status sequences are keyed by the stable ID assigned to each transfer handle.
    state_sequences: dict[int, list[str]] = {}
    # Report every handle as processing until the test clears this flag.
    force_processing = False
    # Raise while dispatching the transfer with this handle ID.
    transfer_error_handle_id: int | None = None
    # Raise on the first status check of the transfer with this handle ID.
    check_error_once_handle_id: int | None = None

    def __init__(self, name, config) -> None:
        self.name = name
        self.config = config
        self.initialized = []
        self._check_errors_raised: set[int] = set()
        self._in_flight: dict[int, FakeXferHandle] = {}
        # Registered address ranges by descriptor; like NIXL, a transfer
        # descriptor must fall inside one of them.
        self.regions: dict[int, tuple[int, int, weakref.ReferenceType]] = {}
        self.calls = Mock()
        for method_name in ("transfer", "check_xfer_state", "release_xfer_handle"):
            method = Mock(wraps=getattr(self, method_name))
            setattr(self, method_name, method)
            self.calls.attach_mock(method, method_name)
        for method_name in (
            "register_memory",
            "deregister_memory",
            "add_remote_agent",
            "remove_remote_agent",
        ):
            setattr(self, method_name, Mock(wraps=getattr(self, method_name)))
        FakeNixlAgent.instances.append(self)

    def create_backend(self, backend, initParams) -> None:
        self.backend_params = {backend: initParams}

    def register_memory(self, tensor, *, backends):
        assert tensor.numel() > 0
        descriptor = id(tensor)
        self.regions[descriptor] = (
            tensor.data_ptr(),
            tensor.data_ptr() + tensor.nbytes,
            # Weak, so registration does not keep the storage alive.
            weakref.ref(tensor.untyped_storage()),
        )
        return [descriptor]

    def deregister_memory(self, descriptors):
        for descriptor in descriptors:
            self.regions.pop(descriptor, None)

    def get_xfer_descs(self, descs, mem_type=None):
        if not isinstance(descs, torch.Tensor):
            return FakeXferDList(descs, mem_type)
        assert descs.numel() > 0
        descriptor = (descs.data_ptr(), descs.nbytes, self.devices.get(id(descs), 0))
        assert self._registered(
            descriptor
        ), "transfer descriptor outside registered memory"
        return FakeXferDList([descriptor], self.memory_types.get(id(descs), "DRAM"))

    def _registered(self, descriptor) -> bool:
        start, length, _ = descriptor
        return any(
            lo <= start and start + length <= hi for lo, hi, _ in self.regions.values()
        )

    def _bytes(self, descriptor) -> torch.Tensor:
        # Byte view of the registered memory a descriptor points to.
        start, length, _ = descriptor
        for lo, hi, storage_ref in self.regions.values():
            if lo <= start and start + length <= hi:
                storage = storage_ref()
                assert storage is not None, "registered storage was freed"
                offset = start - storage.data_ptr()
                flat = torch.empty(0, dtype=torch.uint8).set_(storage)
                return flat[offset : offset + length]
        raise AssertionError("descriptor outside registered memory")

    def get_serialized_descs(self, descriptors):
        return pickle.dumps(descriptors)

    def deserialize_descs(self, descriptors):
        return pickle.loads(descriptors)

    def get_agent_metadata(self):
        return self.name.encode()

    def add_remote_agent(self, metadata):
        return metadata.split(b":", 1)[0]

    def initialize_xfer(
        self,
        operation,
        local_descs,
        remote_descs,
        remote_agent,
        *,
        backends,
    ):
        handle = FakeXferHandle(
            handle_id=len(self.initialized),
            operation=_NixlTransferOperation(operation),
            local_descriptors=local_descs.descriptors,
            remote_descriptors=remote_descs.descriptors,
            remote_agent=remote_agent,
            remote_memory_type=remote_descs.getType(),
        )
        self.initialized.append(handle)
        return handle

    def transfer(self, handle):
        if self.transfer_error_handle_id == handle.handle_id:
            raise RuntimeError("transfer failed")
        # Like DMA, the copy is not visible until the transfer reports DONE.
        self._in_flight[handle.handle_id] = handle
        return _NixlTransferStatus.PROCESSING

    def _complete(self, handle):
        (remote,) = (i for i in self.instances if i.name == handle.remote_agent)
        for local_descriptor, remote_descriptor in zip(
            handle.local_descriptors, handle.remote_descriptors, strict=True
        ):
            local = self._bytes(local_descriptor)
            remote_bytes = remote._bytes(remote_descriptor)
            assert local.numel() == remote_bytes.numel()
            if handle.operation == _NixlTransferOperation.READ:
                local.copy_(remote_bytes)
            elif handle.operation == _NixlTransferOperation.WRITE:
                remote_bytes.copy_(local)
            else:
                raise AssertionError(f"unexpected operation: {handle.operation}")

    def check_xfer_state(self, handle):
        if self.force_processing:
            return _NixlTransferStatus.PROCESSING
        handle_id = handle.handle_id
        if (
            self.check_error_once_handle_id == handle_id
            and handle_id not in self._check_errors_raised
        ):
            self._check_errors_raised.add(handle_id)
            raise RuntimeError("check failed")
        # Consume the status for this poll; default to DONE once none remain.
        sequence = self.state_sequences.get(handle_id)
        if sequence:
            return sequence.pop(0)
        if handle_id in self._in_flight:
            self._complete(self._in_flight.pop(handle_id))
        return _NixlTransferStatus.DONE

    def release_xfer_handle(self, handle):
        pass

    def remove_remote_agent(self, remote_name):
        pass


@pytest.fixture
def ref():
    transport_context = TransportContext()
    volume_ref = SimpleNamespace(
        volume_id="test-volume",
        volume=SimpleNamespace(
            get_meta=SimpleNamespace(call_one=AsyncMock(return_value=[]))
        ),
        transport_context=transport_context,
    )
    yield volume_ref
    transport_context.clear()


@pytest.fixture
def ctx():
    context = TransportContext()
    yield context
    context.clear()


def _storage_volume_copy(buffer: NixlTransportBuffer) -> NixlTransportBuffer:
    return pickle.loads(pickle.dumps(buffer))


def _put_entries(requests: list[Request]) -> list[tuple[Request, None]]:
    return [(request.meta_only(), None) for request in requests]


def _expected_handle_calls(handles: list[FakeXferHandle]):
    """Build expected unittest.mock calls with each handle as the sole argument."""
    return [call(handle) for handle in handles]


async def _prepare_mixed_memory_put(ref):
    sources = [torch.arange(4), torch.arange(4, 8)]
    FakeNixlAgent.memory_types[id(sources[1])] = "VRAM"
    requests = [
        Request.from_tensor(f"weight-{index}", source)
        for index, source in enumerate(sources)
    ]
    client_buffer = NixlTransportBuffer(ref)
    await client_buffer._pre_put_hook(requests)
    return sources, requests, _storage_volume_copy(client_buffer)


@pytest.fixture(autouse=True)
def fake_nixl(monkeypatch):
    # Each test creates new client/server agent instances. Give those instances
    # fresh class-level backend state that is shared only for this test.
    FakeNixlAgent.memory_types = {}
    FakeNixlAgent.devices = {}
    FakeNixlAgent.instances = []
    FakeNixlAgent.state_sequences = {}
    FakeNixlAgent.force_processing = False
    FakeNixlAgent.transfer_error_handle_id = None
    FakeNixlAgent.check_error_once_handle_id = None
    monkeypatch.setattr(nixl_transport, "ENV_TORCHSTORE_NIXL_ENABLED", True)
    monkeypatch.setattr(nixl_transport, "find_spec", lambda _name: object())
    monkeypatch.setattr(
        nixl_transport,
        "_load_nixl_api",
        lambda: (FakeNixlAgent, FakeNixlConfig),
    )
    nixl_transport.nixl_available.cache_clear()
    yield
    nixl_transport.nixl_available.cache_clear()


@pytest.fixture
def agent_cache(fake_nixl):
    cache = NixlAgentCache()
    yield cache
    cache.clear()


def test_availability_check_does_not_load_nixl_api(monkeypatch):
    load_nixl_api = Mock()
    monkeypatch.setattr(nixl_transport, "_load_nixl_api", load_nixl_api)

    assert nixl_transport.nixl_available()
    load_nixl_api.assert_not_called()


def test_transport_instantiation_loads_nixl_api(ref, monkeypatch):
    load_nixl_api = Mock(return_value=(FakeNixlAgent, FakeNixlConfig))
    monkeypatch.setattr(nixl_transport, "_load_nixl_api", load_nixl_api)

    NixlTransportBuffer(ref)

    load_nixl_api.assert_called_once_with()


@pytest.mark.asyncio
async def test_put_reads_client_tensor_into_storage_volume(ref, ctx):
    client_buffer = NixlTransportBuffer(ref)
    source = torch.arange(8, dtype=torch.float32)
    request = Request.from_tensor("weight", source)

    assert not client_buffer.requires_handshake([request])
    await client_buffer._pre_put_hook([request])
    assert client_buffer._client_metadata is not None
    storage_volume_buffer = _storage_volume_copy(client_buffer)
    results = await storage_volume_buffer.handle_put_request(
        ctx, _put_entries([request])
    )

    assert torch.equal(results[0], source)
    assert results[0].device.type == "cpu"
    assert FakeNixlAgent.instances[-1].release_xfer_handle.call_count == 1


@pytest.mark.asyncio
async def test_get_writes_storage_tensor_into_client_destination(ref, ctx):
    client_buffer = NixlTransportBuffer(ref)
    destination = torch.zeros(8)
    request = Request.from_tensor("weight", destination)

    await client_buffer._pre_get_hook([request])
    storage_volume_buffer = _storage_volume_copy(client_buffer)
    stored = torch.arange(8, dtype=torch.float32)
    await storage_volume_buffer.handle_get_request(ctx, [(request.meta_only(), stored)])
    results = await client_buffer._handle_storage_volume_response(
        [request], storage_volume_buffer
    )

    assert results[0] is destination
    assert torch.equal(destination, stored)
    agent = FakeNixlAgent.instances[-1]
    assert agent.initialized[0].operation == _NixlTransferOperation.WRITE
    assert agent.release_xfer_handle.call_args_list == _expected_handle_calls(
        agent.initialized
    )


@pytest.mark.asyncio
async def test_put_batches_multiple_tensors_in_one_transfer(ref, ctx):
    client_buffer = NixlTransportBuffer(ref)
    sources = [torch.arange(4), torch.arange(4, 8)]
    requests = [
        Request.from_tensor(f"weight-{index}", source)
        for index, source in enumerate(sources)
    ]

    await client_buffer._pre_put_hook(requests)
    storage_volume_buffer = _storage_volume_copy(client_buffer)
    results = await storage_volume_buffer.handle_put_request(
        ctx, _put_entries(requests)
    )

    assert all(
        torch.equal(result, source)
        for result, source in zip(results, sources, strict=True)
    )
    agent = FakeNixlAgent.instances[-1]
    assert len(agent.initialized) == 1
    expected_calls = _expected_handle_calls(agent.initialized)
    assert agent.transfer.call_args_list == expected_calls
    assert agent.check_xfer_state.call_args_list == expected_calls
    assert agent.release_xfer_handle.call_args_list == expected_calls


@pytest.mark.asyncio
async def test_put_splits_mixed_memory_types_into_separate_transfers(ref, ctx):
    sources, requests, storage_volume_buffer = await _prepare_mixed_memory_put(ref)
    results = await storage_volume_buffer.handle_put_request(
        ctx, _put_entries(requests)
    )

    agent = FakeNixlAgent.instances[-1]
    assert all(
        torch.equal(result, source)
        for result, source in zip(results, sources, strict=True)
    )
    assert len(agent.initialized) == 2
    expected_calls = _expected_handle_calls(agent.initialized)
    assert agent.transfer.call_args_list == expected_calls
    assert agent.check_xfer_state.call_args_list == expected_calls
    assert agent.release_xfer_handle.call_args_list == expected_calls
    assert agent.calls.mock_calls == [
        call.transfer(agent.initialized[0]),
        call.transfer(agent.initialized[1]),
        call.check_xfer_state(agent.initialized[0]),
        call.check_xfer_state(agent.initialized[1]),
        call.release_xfer_handle(agent.initialized[0]),
        call.release_xfer_handle(agent.initialized[1]),
    ]


@pytest.mark.asyncio
async def test_transfer_drains_all_handles_before_reporting_failure(ref, ctx):
    FakeNixlAgent.state_sequences = {
        0: ["ERR"],
        1: [_NixlTransferStatus.PROCESSING, _NixlTransferStatus.DONE],
    }
    _, requests, storage_volume_buffer = await _prepare_mixed_memory_put(ref)

    with pytest.raises(RuntimeError, match="NIXL READ failed"):
        await storage_volume_buffer.handle_put_request(
            ctx,
            _put_entries(requests),
        )

    agent = FakeNixlAgent.instances[-1]
    assert agent.check_xfer_state.call_args_list == [
        call(agent.initialized[0]),
        call(agent.initialized[1]),
        call(agent.initialized[1]),
    ]
    assert agent.release_xfer_handle.call_args_list == _expected_handle_calls(
        agent.initialized
    )


@pytest.mark.asyncio
async def test_dispatch_error_drains_started_handles_before_releasing(ref, ctx):
    FakeNixlAgent.transfer_error_handle_id = 1
    _, requests, storage_volume_buffer = await _prepare_mixed_memory_put(ref)

    with pytest.raises(RuntimeError, match="NIXL READ failed"):
        await storage_volume_buffer.handle_put_request(
            ctx,
            _put_entries(requests),
        )

    agent = FakeNixlAgent.instances[-1]
    expected_calls = _expected_handle_calls(agent.initialized)
    assert agent.check_xfer_state.call_args_list == expected_calls
    assert agent.release_xfer_handle.call_args_list == expected_calls


@pytest.mark.asyncio
async def test_status_check_error_retries_and_drains_all_handles_before_releasing(
    ref, ctx
):
    FakeNixlAgent.check_error_once_handle_id = 0
    _, requests, storage_volume_buffer = await _prepare_mixed_memory_put(ref)

    with pytest.raises(RuntimeError, match="NIXL READ failed") as exc_info:
        await storage_volume_buffer.handle_put_request(
            ctx,
            _put_entries(requests),
        )

    agent = FakeNixlAgent.instances[-1]
    assert str(exc_info.value.__cause__) == "check failed"
    assert agent.check_xfer_state.call_args_list == [
        call(agent.initialized[0]),
        call(agent.initialized[0]),
        call(agent.initialized[1]),
    ]
    assert agent.release_xfer_handle.call_args_list == _expected_handle_calls(
        agent.initialized
    )


@pytest.mark.asyncio
async def test_transfer_timeout_drains_handle_before_releasing(monkeypatch, ref, ctx):
    monkeypatch.setattr(nixl_transport, "ENV_TORCHSTORE_NIXL_TIMEOUT_S", 0)
    FakeNixlAgent.state_sequences = {
        0: [_NixlTransferStatus.PROCESSING, _NixlTransferStatus.DONE]
    }
    client_buffer = NixlTransportBuffer(ref)
    source = torch.arange(4)
    request = Request.from_tensor("weight", source)

    await client_buffer._pre_put_hook([request])
    storage_volume_buffer = _storage_volume_copy(client_buffer)

    with pytest.raises(RuntimeError, match="NIXL READ failed") as exc_info:
        await storage_volume_buffer.handle_put_request(ctx, _put_entries([request]))

    agent = FakeNixlAgent.instances[-1]
    assert isinstance(exc_info.value.__cause__, TimeoutError)
    assert agent.check_xfer_state.call_args_list == [
        call(agent.initialized[0]),
        call(agent.initialized[0]),
    ]
    assert agent.release_xfer_handle.call_args_list == _expected_handle_calls(
        agent.initialized
    )


@pytest.mark.asyncio
async def test_transfer_cancellation_waits_for_handle_before_releasing(ref, ctx):
    FakeNixlAgent.force_processing = True
    client_buffer = NixlTransportBuffer(ref)
    source = torch.arange(4)
    request = Request.from_tensor("weight", source)

    await client_buffer._pre_put_hook([request])
    storage_volume_buffer = _storage_volume_copy(client_buffer)
    transfer_task = asyncio.create_task(
        storage_volume_buffer.handle_put_request(ctx, _put_entries([request]))
    )
    while (
        len(FakeNixlAgent.instances) < 2
        or not FakeNixlAgent.instances[-1].check_xfer_state.called
    ):
        await asyncio.sleep(0)
    agent = FakeNixlAgent.instances[-1]

    transfer_task.cancel()
    await asyncio.sleep(0)

    assert not transfer_task.done()
    agent.release_xfer_handle.assert_not_called()

    FakeNixlAgent.force_processing = False
    with pytest.raises(asyncio.CancelledError):
        await transfer_task

    assert agent.release_xfer_handle.call_args_list == _expected_handle_calls(
        agent.initialized
    )


@pytest.mark.parametrize("shape", [(0,), (0, 3)])
@pytest.mark.asyncio
async def test_put_empty_tensor_skips_nixl_transfer(shape, ref, ctx):
    client_buffer = NixlTransportBuffer(ref)
    source = torch.ones(shape)
    request = Request.from_tensor("empty", source)

    await client_buffer._pre_put_hook([request])

    assert client_buffer._contexts[0].remote_descriptors is None
    assert client_buffer._client_metadata is None
    assert FakeNixlAgent.instances == []

    storage_volume_buffer = _storage_volume_copy(client_buffer)
    results = await storage_volume_buffer.handle_put_request(
        ctx, _put_entries([request])
    )

    assert results[0].shape == source.shape
    assert results[0].dtype == source.dtype
    assert results[0].numel() == 0
    assert FakeNixlAgent.instances == []


@pytest.mark.parametrize("shape", [(0,), (0, 3)])
@pytest.mark.asyncio
async def test_get_empty_tensor_skips_nixl_transfer(shape, ref, ctx):
    ref.volume.get_meta.call_one.return_value = [(torch.Size(shape), torch.float32)]
    client_buffer = NixlTransportBuffer(ref)
    request = Request.from_any("empty", None)

    await client_buffer._pre_get_hook([request])

    destination = client_buffer._contexts[0].tensor
    assert destination is not None
    assert client_buffer._contexts[0].remote_descriptors is None
    assert client_buffer._client_metadata is None
    assert FakeNixlAgent.instances == []

    storage_volume_buffer = _storage_volume_copy(client_buffer)
    await storage_volume_buffer.handle_get_request(
        ctx, [(request.meta_only(), torch.ones(shape))]
    )
    results = await client_buffer._handle_storage_volume_response(
        [request], storage_volume_buffer
    )

    assert results == [destination]
    assert destination.shape == torch.Size(shape)
    assert destination.numel() == 0
    assert FakeNixlAgent.instances == []


@pytest.mark.asyncio
async def test_put_mixed_batch_preserves_empty_and_object_entries(ref, ctx):
    client_buffer = NixlTransportBuffer(ref)
    sources = [torch.arange(4), torch.empty(0, 3), {"step": 3}, torch.arange(4, 8)]
    requests = [
        Request.from_tensor("first", sources[0]),
        Request.from_tensor("empty", sources[1]),
        Request.from_objects("metadata", sources[2]),
        Request.from_tensor("last", sources[3]),
    ]

    await client_buffer._pre_put_hook(requests)

    assert client_buffer._contexts[1].remote_descriptors is None
    assert client_buffer._contexts[2].is_object
    assert client_buffer._client_metadata is not None

    storage_volume_buffer = _storage_volume_copy(client_buffer)
    results = await storage_volume_buffer.handle_put_request(
        ctx, _put_entries(requests)
    )

    assert torch.equal(results[0], sources[0])
    assert results[1].shape == sources[1].shape
    assert results[1].numel() == 0
    assert results[2] == sources[2]
    assert torch.equal(results[3], sources[3])
    agent = FakeNixlAgent.instances[-1]
    assert len(agent.initialized) == 1
    assert len(agent.initialized[0].local_descriptors) == 2
    assert agent.release_xfer_handle.call_args_list == _expected_handle_calls(
        agent.initialized
    )


@pytest.mark.asyncio
async def test_get_mixed_batch_preserves_empty_and_object_entries(ref, ctx):
    metas = [
        (torch.Size([4]), torch.int64),
        (torch.Size([0, 3]), torch.float32),
        "obj",
        (torch.Size([4]), torch.int64),
    ]
    ref.volume.get_meta.call_one.return_value = metas
    client_buffer = NixlTransportBuffer(ref)
    requests = [
        Request.from_any("first", None),
        Request.from_any("empty", None),
        Request.from_any("metadata", None),
        Request.from_any("last", None),
    ]

    await client_buffer._pre_get_hook(requests)

    destinations = [context.tensor for context in client_buffer._contexts]
    assert destinations[0] is not None
    assert destinations[0].device.type == "cpu"
    assert destinations[1] is not None
    assert destinations[1].numel() == 0
    assert client_buffer._contexts[1].remote_descriptors is None
    assert client_buffer._contexts[2].is_object
    assert client_buffer._client_metadata is not None

    stored = [torch.arange(4), torch.empty(0, 3), {"step": 3}, torch.arange(4, 8)]
    storage_volume_buffer = _storage_volume_copy(client_buffer)
    await storage_volume_buffer.handle_get_request(
        ctx,
        [
            (request.meta_only(), value)
            for request, value in zip(requests, stored, strict=True)
        ],
    )
    results = await client_buffer._handle_storage_volume_response(
        requests, storage_volume_buffer
    )

    assert results[0] is destinations[0]
    assert torch.equal(results[0], stored[0])
    assert results[1] is destinations[1]
    assert results[1].shape == stored[1].shape
    assert results[1].numel() == 0
    assert results[2] == stored[2]
    assert results[3] is destinations[3]
    assert torch.equal(results[3], stored[3])
    agent = FakeNixlAgent.instances[-1]
    assert len(agent.initialized) == 1
    assert len(agent.initialized[0].local_descriptors) == 2
    assert agent.release_xfer_handle.call_args_list == _expected_handle_calls(
        agent.initialized
    )


@pytest.mark.asyncio
async def test_objects_stay_on_rpc_path_without_creating_nixl_agent(ref, ctx):
    client_buffer = NixlTransportBuffer(ref)
    request = Request.from_objects("metadata", {"step": 3})

    await client_buffer._pre_put_hook([request])
    storage_volume_buffer = _storage_volume_copy(client_buffer)
    results = await storage_volume_buffer.handle_put_request(
        ctx, _put_entries([request])
    )

    assert results == [{"step": 3}]
    assert FakeNixlAgent.instances == []


@pytest.mark.asyncio
async def test_uncached_registration_lasts_until_last_drop(ref):
    """Buffers of one get share a storage's registration until both drop."""
    first = NixlTransportBuffer(ref)
    second = NixlTransportBuffer(ref)
    destination = torch.zeros(4, 16)

    first._register_client_tensor(destination[0], cached=False)
    first._register_client_tensor(destination[1], cached=False)
    second._register_client_tensor(destination[2], cached=False)
    cache = first._client_cache()
    agent = cache.agent
    ((descriptors, holders),) = cache._leases.values()

    assert holders == 2
    agent.register_memory.assert_called_once()
    assert cache._registrations == {}
    assert _storage_volume_copy(first)._leases == set()
    await first.drop()
    agent.deregister_memory.assert_not_called()
    await second.drop()
    agent.deregister_memory.assert_called_once_with(descriptors)
    assert cache._leases == {}


def test_agent_disables_ucx_registration_reuse(agent_cache):
    assert agent_cache.agent.config.kwargs["backends"] == []
    assert agent_cache.agent.backend_params == {
        "UCX": {"engine_config": "CUDA_COPY_REG_WHOLE_ALLOC=off,RCACHE_ENABLE=n"}
    }


def test_memory_registration_is_reused_for_same_tensor(agent_cache):
    tensor = torch.zeros(16)

    first = agent_cache.register(tensor)
    second = agent_cache.register(tensor)

    assert first is second
    assert agent_cache.agent.register_memory.call_count == 1


def test_memory_registration_covers_all_views_of_a_storage(agent_cache):
    tensor = torch.zeros(4, 16)

    first = agent_cache.register(tensor[0])
    second = agent_cache.register(tensor[1:3])

    assert first is second
    assert agent_cache.agent.register_memory.call_count == 1
    (region,) = agent_cache.agent.register_memory.call_args.args
    assert region.data_ptr() == tensor.data_ptr()
    assert region.nbytes == tensor.nbytes


def test_memory_registration_is_evicted_with_tensor_storage(agent_cache):
    tensor = torch.zeros(16)
    key = (tensor.data_ptr(), tensor.nbytes)
    agent = agent_cache.agent

    (descriptor,) = agent_cache.register(tensor)
    assert key in agent_cache._registrations

    # Mock call history retains arguments; clear it so it does not own the tensor.
    agent.register_memory.reset_mock()
    del tensor
    gc.collect()

    assert key not in agent_cache._registrations
    agent.deregister_memory.assert_called_once_with([descriptor])


def test_memory_deregistration_failure_propagates(agent_cache):
    tensor = torch.zeros(16)
    key = (tensor.data_ptr(), tensor.nbytes)
    agent_cache.register(tensor)
    agent_cache.agent.deregister_memory.side_effect = RuntimeError("deregister failed")

    with pytest.raises(RuntimeError, match="deregister failed"):
        agent_cache.clear()

    assert key in agent_cache._registrations
    agent_cache.agent.deregister_memory.side_effect = None


def test_remote_agent_metadata_is_loaded_once(agent_cache):
    first = agent_cache.add_remote_agent("remote-agent", b"remote-agent:v1")
    second = agent_cache.add_remote_agent("remote-agent", b"remote-agent:v1")

    assert first == second == "remote-agent"
    assert agent_cache.agent.add_remote_agent.call_count == 1

    refreshed = agent_cache.add_remote_agent("remote-agent", b"remote-agent:v2")

    assert refreshed == "remote-agent"
    assert agent_cache.agent.add_remote_agent.call_count == 2
    agent_cache.agent.remove_remote_agent.assert_called_once_with("remote-agent")


def test_remote_agent_removal_failure_propagates(agent_cache):
    agent_cache.add_remote_agent("remote-agent", b"remote-agent:v1")
    agent_cache.agent.remove_remote_agent.side_effect = RuntimeError("remove failed")

    with pytest.raises(RuntimeError, match="remove failed"):
        agent_cache.clear()

    assert "remote-agent" in agent_cache._remote_agents
    agent_cache.agent.remove_remote_agent.side_effect = None


async def _spin_until(condition, timeout_s=30.0):
    # Yield to the event loop until condition holds; fail instead of hanging.
    deadline = time.monotonic() + timeout_s
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0)


async def _get_into_client(
    ref, ctx, stored: list[torch.Tensor], memory_type="DRAM", device=0
):
    client_buffer = NixlTransportBuffer(ref)
    destinations = [
        torch.zeros_like(tensor, memory_format=torch.contiguous_format)
        for tensor in stored
    ]
    for destination in destinations:
        # Simulate destinations on another memory type and device.
        FakeNixlAgent.memory_types[id(destination)] = memory_type
        FakeNixlAgent.devices[id(destination)] = device
    requests = [
        Request.from_tensor(f"weight-{index}", destination)
        for index, destination in enumerate(destinations)
    ]
    await client_buffer._pre_get_hook(requests)
    storage_volume_buffer = _storage_volume_copy(client_buffer)
    await storage_volume_buffer.handle_get_request(
        ctx,
        [
            (request.meta_only(), tensor)
            for request, tensor in zip(requests, stored, strict=True)
        ],
    )
    return destinations


@pytest.mark.asyncio
async def test_get_stages_non_contiguous_tensors_through_two_buffers(
    monkeypatch, ref, ctx
):
    monkeypatch.setattr(nixl_transport, "ENV_TORCHSTORE_NIXL_STAGING_BUFFER_BYTES", 64)
    # Keep transfers in flight so reusing a buffer early corrupts the data.
    FakeNixlAgent.state_sequences = {
        handle_id: [_NixlTransferStatus.PROCESSING] * 3 for handle_id in range(13)
    }
    stored = [
        torch.arange(64).reshape(8, 8)[:, 2:6],  # 32-byte rows, 2 per buffer
        torch.arange(100, 104),  # contiguous, sent directly
        torch.arange(256).reshape(4, 64)[:, ::4],  # 128-byte rows split in halves
    ]

    destinations = await _get_into_client(ref, ctx, stored)

    for destination, tensor in zip(destinations, stored, strict=True):
        assert torch.equal(destination, tensor)
    agent = FakeNixlAgent.instances[-1]
    # 1 direct + 4 chunks of the first tensor + 8 chunks of the third.
    assert len(agent.initialized) == 13
    staging_buffers = [
        region
        for (region,), _ in agent.register_memory.call_args_list
        if region.nbytes == 64
    ]
    assert len(staging_buffers) == 2
    assert agent.release_xfer_handle.call_args_list == _expected_handle_calls(
        agent.initialized
    )


@pytest.mark.asyncio
async def test_get_staging_buffer_is_capped_by_batch_size(ref, ctx):
    stored = [torch.arange(64).reshape(8, 8)[:, 2:6], torch.arange(8).reshape(2, 4).t()]

    destinations = await _get_into_client(ref, ctx, stored)

    for destination, tensor in zip(destinations, stored, strict=True):
        assert torch.equal(destination, tensor)
    agent = FakeNixlAgent.instances[-1]
    assert len(agent.initialized) == 1
    ((region,), _) = agent.register_memory.call_args
    assert region.nbytes == sum(tensor.nbytes for tensor in stored)


@pytest.mark.asyncio
async def test_get_staging_failure_drains_in_flight_chunks(monkeypatch, ref, ctx):
    monkeypatch.setattr(nixl_transport, "ENV_TORCHSTORE_NIXL_STAGING_BUFFER_BYTES", 64)
    FakeNixlAgent.state_sequences = {
        0: ["ERR"],
        1: [_NixlTransferStatus.PROCESSING, _NixlTransferStatus.DONE],
    }

    with pytest.raises(RuntimeError, match="NIXL WRITE failed"):
        await _get_into_client(ref, ctx, [torch.arange(64).reshape(8, 8)[:, 2:6]])

    agent = FakeNixlAgent.instances[-1]
    # Reusing the first buffer waits on its failed transfer; no chunk starts after.
    assert len(agent.initialized) == 2
    assert agent.release_xfer_handle.call_args_list == _expected_handle_calls(
        agent.initialized
    )


@pytest.mark.asyncio
async def test_get_staging_packs_pieces_at_aligned_offsets(ref, ctx):
    stored = [
        torch.arange(12, dtype=torch.uint8).reshape(3, 4)[:, 1],
        torch.arange(12, dtype=torch.uint8).reshape(4, 3)[:, 0],
        torch.arange(8).reshape(2, 4).t(),
    ]

    destinations = await _get_into_client(ref, ctx, stored)

    for destination, tensor in zip(destinations, stored, strict=True):
        assert torch.equal(destination, tensor)
    agent = FakeNixlAgent.instances[-1]
    (handle,) = agent.initialized
    addresses = [address for address, _, _ in handle.local_descriptors]
    assert [address - addresses[0] for address in addresses] == [0, 64, 128]
    ((region,), _) = agent.register_memory.call_args
    assert region.nbytes == 192


@pytest.mark.asyncio
async def test_get_staging_rejects_tiny_buffer(monkeypatch, ref, ctx):
    monkeypatch.setattr(nixl_transport, "ENV_TORCHSTORE_NIXL_STAGING_BUFFER_BYTES", 0)

    with pytest.raises(ValueError, match="TORCHSTORE_NIXL_STAGING_BUFFER_BYTES"):
        await _get_into_client(ref, ctx, [torch.arange(8).reshape(2, 4).t()])


@pytest.mark.asyncio
async def test_get_staging_cancel_waits_for_in_flight_transfers(monkeypatch, ref, ctx):
    monkeypatch.setattr(nixl_transport, "ENV_TORCHSTORE_NIXL_STAGING_BUFFER_BYTES", 64)
    FakeNixlAgent.force_processing = True
    stored = [torch.arange(64).reshape(8, 8)[:, 2:6], torch.arange(4)]
    get = asyncio.create_task(_get_into_client(ref, ctx, stored))
    try:
        # Direct transfer (handle 0) plus both buffers (handles 1, 2) in flight.
        await _spin_until(
            lambda: FakeNixlAgent.instances
            and len(FakeNixlAgent.instances[-1].initialized) >= 3
        )
        agent = FakeNixlAgent.instances[-1]

        # The first cancel interrupts the wait on buffer 0's transfer; the second
        # lands in the final wait while the other transfers are in flight.
        FakeNixlAgent.state_sequences = {
            0: [_NixlTransferStatus.PROCESSING] * 1000,
            2: [_NixlTransferStatus.PROCESSING] * 1000,
        }
        FakeNixlAgent.force_processing = False

        async def wait_for_direct_polls(count):
            # Each direct transfer poll gives the get task time to make progress.
            target = len(FakeNixlAgent.state_sequences[0]) - count
            await _spin_until(lambda: len(FakeNixlAgent.state_sequences[0]) <= target)

        get.cancel()
        await _spin_until(lambda: agent.release_xfer_handle.called)
        await wait_for_direct_polls(5)
        get.cancel()
        await wait_for_direct_polls(5)
        assert not get.done()
        assert agent.release_xfer_handle.call_count == 1

        FakeNixlAgent.state_sequences = {}
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(get, timeout=30)
        released = [args[0] for args, _ in agent.release_xfer_handle.call_args_list]
        assert sorted(released) == agent.initialized
    finally:
        # Let a stuck get drain so a failure does not hang loop teardown.
        FakeNixlAgent.force_processing = False
        FakeNixlAgent.state_sequences = {}


@pytest.mark.asyncio
async def test_get_direct_failure_while_staging_raises(monkeypatch, ref, ctx):
    monkeypatch.setattr(nixl_transport, "ENV_TORCHSTORE_NIXL_STAGING_BUFFER_BYTES", 64)
    FakeNixlAgent.state_sequences = {0: ["ERR"]}

    with pytest.raises(RuntimeError, match="NIXL WRITE failed"):
        await _get_into_client(
            ref, ctx, [torch.arange(4), torch.arange(64).reshape(8, 8)[:, 2:6]]
        )

    agent = FakeNixlAgent.instances[-1]
    # Direct transfer plus four 64-byte staged chunks, all drained.
    assert len(agent.initialized) == 5
    released = [args[0] for args, _ in agent.release_xfer_handle.call_args_list]
    assert sorted(released) == agent.initialized


@pytest.mark.asyncio
async def test_get_staging_keeps_remote_memory_type_and_device(monkeypatch, ref, ctx):
    monkeypatch.setattr(nixl_transport, "ENV_TORCHSTORE_NIXL_STAGING_BUFFER_BYTES", 64)
    stored = [torch.arange(64).reshape(8, 8)[:, 2:6]]

    destinations = await _get_into_client(
        ref, ctx, stored, memory_type="VRAM", device=3
    )

    torch.testing.assert_close(destinations[0], stored[0])
    agent = FakeNixlAgent.instances[-1]
    assert len(agent.initialized) == 4
    for handle in agent.initialized:
        assert handle.remote_memory_type == FakeNixlAgent.nixl_mems["VRAM"]
        assert {device for _, _, device in handle.remote_descriptors} == {3}
