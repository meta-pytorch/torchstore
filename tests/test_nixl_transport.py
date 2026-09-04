# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Focused tests for the NIXL StorageVolume transport lifecycle."""

import asyncio
import gc
import pickle
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
        return self.memory_type

    def descCount(self):
        return len(self.descriptors)

    def append(self, descriptor):
        self.descriptors.append(descriptor)

    def __getitem__(self, index):
        return self.descriptors[index]


class FakeXferHandle(NamedTuple):
    handle_id: int
    operation: _NixlTransferOperation
    local_descriptors: list[int]
    remote_descriptors: list[int]
    remote_agent: str


class FakeNixlAgent:
    # Shared fake backend state.
    # The registry lets client and server agents resolve the same descriptors;
    # overrides simulate mixed DRAM/VRAM descriptors - tests can run purely on CPU.
    tensors: dict[int, weakref.ReferenceType[torch.Tensor]] = {}
    memory_types: dict[int, str] = {}

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

    def register_memory(self, tensor, *, backends):
        assert tensor.numel() > 0
        descriptor = id(tensor)
        self.tensors[descriptor] = weakref.ref(tensor)
        return [descriptor]

    def deregister_memory(self, descriptors):
        for descriptor in descriptors:
            self.tensors.pop(descriptor, None)

    def get_xfer_descs(self, tensor):
        assert tensor.numel() > 0
        return FakeXferDList([id(tensor)], self.memory_types.get(id(tensor), "DRAM"))

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
        )
        self.initialized.append(handle)
        return handle

    def transfer(self, handle):
        if self.transfer_error_handle_id == handle.handle_id:
            raise RuntimeError("transfer failed")
        for local_descriptor, remote_descriptor in zip(
            handle.local_descriptors, handle.remote_descriptors, strict=True
        ):
            storage_volume_tensor = self.tensors[local_descriptor]()
            client_tensor = self.tensors[remote_descriptor]()
            assert storage_volume_tensor is not None
            assert client_tensor is not None
            if handle.operation == _NixlTransferOperation.READ:
                storage_volume_tensor.copy_(client_tensor)
            elif handle.operation == _NixlTransferOperation.WRITE:
                client_tensor.copy_(storage_volume_tensor)
            else:
                raise AssertionError(f"unexpected operation: {handle.operation}")
        return _NixlTransferStatus.PROCESSING

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
    FakeNixlAgent.tensors = {}
    FakeNixlAgent.memory_types = {}
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


def test_memory_registration_is_reused_for_same_tensor(agent_cache):
    tensor = torch.zeros(16)

    first = agent_cache.register(tensor)
    second = agent_cache.register(tensor)

    assert first is second
    assert agent_cache.agent.register_memory.call_count == 1


def test_memory_registration_is_evicted_with_tensor_storage(agent_cache):
    tensor = torch.zeros(16)
    key = (tensor.data_ptr(), tensor.nbytes)
    descriptor = id(tensor)
    agent = agent_cache.agent

    agent_cache.register(tensor)
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
