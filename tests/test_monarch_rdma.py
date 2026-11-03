# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the batched Monarch RDMA transport.

The Monarch RDMA primitives need real hardware, so these tests patch
``RDMABuffer`` and ``RDMAAction`` with in-process fakes that move bytes
between the registered byte views. That keeps the batching orchestration
-- context alignment, object routing, and inplace copy-back -- under test
without an ibverbs backend.
"""

import logging
import pickle
import re
import threading
from types import SimpleNamespace

import pytest
import torch
import torchstore.monarch.rdma as monarch_rdma
from torchstore import transport
from torchstore.monarch.rdma import MonarchRDMATransportBuffer, RdmaContext
from torchstore.transport import TransportType
from torchstore.transport.buffers import TransportContext
from torchstore.transport.types import Request


class FakeRDMABuffer:
    """Stand-in for ``monarch.rdma.RDMABuffer`` wrapping a byte view.

    A pickled copy is the same buffer, as a real one still names the same memory.
    """

    # The test's buffers by key, so that unpickling finds the original.
    registry: dict[int, "FakeRDMABuffer"] = {}

    def __init__(self, byte_view: torch.Tensor) -> None:
        # The fake action reads/writes this view to emulate a transfer.
        self.mem = byte_view
        self.dropped = False
        self.key = len(FakeRDMABuffer.registry)
        FakeRDMABuffer.registry[self.key] = self

    def __reduce__(self) -> tuple:
        return _registered_buffer, (self.key,)

    async def drop(self) -> None:
        self.dropped = True


def _registered_buffer(key: int) -> FakeRDMABuffer:
    return FakeRDMABuffer.registry[key]


class FakeRDMAAction:
    """Stand-in for ``monarch.rdma.RDMAAction`` that copies on submit."""

    instances: list["FakeRDMAAction"] = []

    def __init__(self) -> None:
        self.ops: list[tuple[str, object, object]] = []
        self.submitted = False
        FakeRDMAAction.instances.append(self)

    def read_remote(self, dst: torch.Tensor, src: FakeRDMABuffer) -> "FakeRDMAAction":
        self.ops.append(("read", dst, src))
        return self

    def write_remote(self, dst: FakeRDMABuffer, src: torch.Tensor) -> "FakeRDMAAction":
        self.ops.append(("write", dst, src))
        return self

    async def submit(self, *, timeout: int = 60) -> None:
        self.submitted = True
        for kind, a, b in self.ops:
            if kind == "read":
                # SV reads remote (client) buffer into its local byte view.
                a.copy_(b.mem)
            else:
                # SV writes its local byte view into the remote (client) buffer.
                a.mem.copy_(b)


class MockEndpoint:
    def __init__(self, result) -> None:
        self._result = result

    async def call_one(self, *args, **kwargs):
        return self._result


class MockVolume:
    def __init__(self, metas) -> None:
        self.get_meta = MockEndpoint(metas)


class MockStorageVolumeRef:
    def __init__(self, metas=None) -> None:
        self.volume_id = "test_volume"
        self.volume = MockVolume(metas or [])
        self.transport_context = TransportContext()
        # Pickle refuses it, so a buffer copy that kept this handle could not be sent.
        self.lock = threading.Lock()


@pytest.fixture(autouse=True)
def fake_rdma(monkeypatch):
    """Patch the module's RDMA primitives with in-process fakes."""
    monkeypatch.setattr(FakeRDMABuffer, "registry", {})
    monkeypatch.setattr(FakeRDMAAction, "instances", [])
    monkeypatch.setattr(monarch_rdma, "RDMABuffer", FakeRDMABuffer)
    monkeypatch.setattr(monarch_rdma, "RDMAAction", FakeRDMAAction)


@pytest.fixture
def ref():
    return MockStorageVolumeRef()


@pytest.fixture
def ctx():
    ctx = TransportContext()
    yield ctx
    ctx.clear()


def _sv_entries_for_put(requests):
    """Mimic InMemoryStore.put: pair each request with no existing object."""
    return [(r.meta_only(), None) for r in requests]


def _volume_copy(buffer):
    """The pickled copy of ``buffer`` that a call hands its storage volume."""
    return pickle.loads(pickle.dumps(buffer))


async def _serve_get(buffer, entries, requests):
    """Run the volume's get on a pickled copy and hand the client a pickled reply."""
    volume_copy = _volume_copy(buffer)
    await volume_copy.handle_get_request(TransportContext(), entries)
    return await buffer._handle_storage_volume_response(
        requests, _volume_copy(volume_copy)
    )


class TestMonarchRDMABatchPut:
    """PUT batch flow."""

    @pytest.mark.asyncio
    async def test_pre_put_hook_builds_aligned_contexts(self, ref):
        buffer = MonarchRDMATransportBuffer(ref)
        t1, t2 = torch.randn(4, 4), torch.randn(8)
        requests = [
            Request.from_tensor("k1", t1),
            Request.from_objects("k2", {"x": 1}),
            Request.from_tensor("k3", t2),
        ]

        await buffer._pre_put_hook(requests)

        assert len(buffer._contexts) == 3
        assert not buffer._contexts[0].is_object
        assert buffer._contexts[1].is_object
        assert buffer._contexts[1].objects == {"x": 1}
        assert buffer._contexts[0].shape == t1.shape

    @pytest.mark.asyncio
    async def test_handle_put_request_reads_all_tensors_in_one_submit(self, ref):
        buffer = MonarchRDMATransportBuffer(ref)
        t1, t2 = torch.randn(4, 4), torch.randn(8)
        requests = [Request.from_tensor("k1", t1), Request.from_tensor("k2", t2)]
        await buffer._pre_put_hook(requests)

        results = await _volume_copy(buffer).handle_put_request(
            TransportContext(), _sv_entries_for_put(requests)
        )

        # One action batched both reads, submitted once.
        assert len(FakeRDMAAction.instances) == 1
        assert FakeRDMAAction.instances[0].submitted
        assert len(FakeRDMAAction.instances[0].ops) == 2
        # SV tensors now hold the client tensors' bytes.
        assert torch.equal(results[0], t1)
        assert torch.equal(results[1], t2)

    @pytest.mark.asyncio
    async def test_handle_put_request_mixed_objects_and_tensors(self, ref):
        buffer = MonarchRDMATransportBuffer(ref)
        t1 = torch.randn(4, 4)
        obj = {"hello": "world"}
        requests = [
            Request.from_objects("k0", obj),
            Request.from_tensor("k1", t1),
        ]
        await buffer._pre_put_hook(requests)

        results = await _volume_copy(buffer).handle_put_request(
            TransportContext(), _sv_entries_for_put(requests)
        )

        assert results[0] == obj
        assert torch.equal(results[1], t1)
        # Only the tensor produced an RDMA op.
        assert len(FakeRDMAAction.instances[0].ops) == 1

    @pytest.mark.asyncio
    async def test_handle_put_request_no_submit_for_pure_object_batch(self, ref):
        buffer = MonarchRDMATransportBuffer(ref)
        requests = [Request.from_objects("k0", 1), Request.from_objects("k1", 2)]
        await buffer._pre_put_hook(requests)

        results = await _volume_copy(buffer).handle_put_request(
            TransportContext(), _sv_entries_for_put(requests)
        )

        assert results == [1, 2]
        assert not FakeRDMAAction.instances[0].submitted

    @pytest.mark.asyncio
    async def test_empty_tensor_skips_put_and_get_rdma_operations(self, ref):
        tensor = torch.empty(0, 8, dtype=torch.bfloat16)
        requests = [Request.from_tensor("empty", tensor)]

        put_buffer = MonarchRDMATransportBuffer(ref)
        await put_buffer._pre_put_hook(requests)
        put_results = await _volume_copy(put_buffer).handle_put_request(
            TransportContext(), _sv_entries_for_put(requests)
        )

        get_buffer = MonarchRDMATransportBuffer(
            MockStorageVolumeRef(metas=[(tensor.shape, tensor.dtype)])
        )
        get_requests = [Request.from_any("empty", None)]
        await get_buffer._pre_get_hook(get_requests)
        get_results = await _serve_get(
            get_buffer, [(get_requests[0], tensor)], get_requests
        )

        assert put_results[0].shape == get_results[0].shape == tensor.shape
        assert put_results[0].dtype == get_results[0].dtype == tensor.dtype
        assert put_buffer._contexts[0].rdma_buffer is None
        assert get_buffer._contexts[0].rdma_buffer is None
        assert all(not action.submitted for action in FakeRDMAAction.instances)


class TestMonarchRDMABatchGet:
    """GET batch flow."""

    @pytest.mark.asyncio
    async def test_inplace_get_writes_into_user_tensor(self, ref):
        buffer = MonarchRDMATransportBuffer(ref)
        dst1, dst2 = torch.zeros(4, 4), torch.zeros(8)
        requests = [
            Request.from_tensor("k1", dst1),
            Request.from_tensor("k2", dst2),
        ]
        await buffer._pre_get_hook(requests)

        stored1, stored2 = torch.randn(4, 4), torch.randn(8)
        results = await _serve_get(
            buffer, [(requests[0], stored1), (requests[1], stored2)], requests
        )

        assert len(FakeRDMAAction.instances) == 1
        assert len(FakeRDMAAction.instances[0].ops) == 2
        assert torch.equal(results[0], stored1)
        assert torch.equal(results[1], stored2)
        # Contiguous inplace destinations are written directly.
        assert results[0] is dst1

    @pytest.mark.asyncio
    async def test_get_with_allocation_uses_fetched_meta(self):
        metas = [(torch.Size([5]), torch.float32)]
        ref = MockStorageVolumeRef(metas=metas)
        buffer = MonarchRDMATransportBuffer(ref)
        requests = [Request.from_any("k1", None)]

        await buffer._pre_get_hook(requests)

        assert len(buffer._contexts) == 1
        assert buffer._contexts[0].shape == torch.Size([5])

        stored = torch.randn(5)
        results = await _serve_get(buffer, [(requests[0], stored)], requests)

        assert torch.equal(results[0], stored)

    @pytest.mark.parametrize("operation", ["put", "get"])
    @pytest.mark.asyncio
    async def test_missing_rdma_buffer_rejected_for_nonempty_tensor(
        self, ref, operation
    ):
        buffer = MonarchRDMATransportBuffer(ref)
        tensor = torch.ones(4)
        request = Request.from_tensor("nonempty", tensor)
        buffer._contexts = [RdmaContext(shape=tensor.shape, dtype=tensor.dtype)]

        with pytest.raises(
            RuntimeError, match="Missing remote RDMA buffer for nonempty tensor"
        ):
            if operation == "put":
                await _volume_copy(buffer).handle_put_request(
                    TransportContext(), [(request.meta_only(), None)]
                )
            else:
                await _volume_copy(buffer).handle_get_request(
                    TransportContext(), [(request.meta_only(), tensor)]
                )

    @pytest.mark.asyncio
    async def test_get_object_routes_through_context(self):
        # k1 stores a tensor (meta = shape/dtype); k2 stores an object, for
        # which get_meta returns the "obj" sentinel.
        ref = MockStorageVolumeRef(metas=[(torch.Size([4]), torch.float32), "obj"])
        buffer = MonarchRDMATransportBuffer(ref)
        requests = [Request.from_any("k1", None), Request.from_any("k2", None)]
        await buffer._pre_get_hook(requests)

        # k2 was classified as an object up front and got no RDMA buffer.
        assert buffer._contexts[1].is_object
        assert buffer._contexts[1].rdma_buffer is None

        stored = torch.randn(4)
        obj = {"payload": [1, 2, 3]}
        results = await _serve_get(
            buffer, [(requests[0], stored), (requests[1], obj)], requests
        )

        assert torch.equal(results[0], stored)
        assert results[1] == obj
        # Only the tensor entry produced an RDMA op.
        assert len(FakeRDMAAction.instances[0].ops) == 1

    @pytest.mark.asyncio
    async def test_non_contiguous_destination_rejected(self, ref):
        buffer = MonarchRDMATransportBuffer(ref)
        # This transport registers tensors directly without staging a
        # contiguous copy, so a non-contiguous destination is rejected.
        base = torch.zeros(4, 4)
        dst = base[:, 1]
        assert not dst.is_contiguous()
        requests = [Request.from_tensor("k1", dst)]

        with pytest.raises(AssertionError):
            await buffer._pre_get_hook(requests)


class TestMonarchRDMADrop:
    """Resource cleanup."""

    @pytest.mark.asyncio
    async def test_drop_releases_all_buffers(self, ref):
        buffer = MonarchRDMATransportBuffer(ref)
        requests = [
            Request.from_tensor("k1", torch.randn(4)),
            Request.from_objects("k2", "obj"),
            Request.from_tensor("k3", torch.randn(4)),
        ]
        await buffer._pre_put_hook(requests)
        rdma_buffers = [
            c.rdma_buffer for c in buffer._contexts if c.rdma_buffer is not None
        ]

        await buffer.drop()

        assert len(rdma_buffers) == 2
        assert all(b.dropped for b in rdma_buffers)
        assert buffer._contexts == []


class TestRdmaContext:
    """Serialization of per-entry context."""

    def test_getstate_strips_local_tensor(self):
        t = torch.randn(4)
        ctx = RdmaContext(
            rdma_buffer=object(),
            tensor=t,
            shape=t.shape,
            dtype=t.dtype,
        )

        state = ctx.__getstate__()

        assert state["tensor"] is None
        # Remote handle and metadata survive serialization.
        assert state["rdma_buffer"] is not None
        assert state["shape"] == t.shape
        assert state["dtype"] == t.dtype


@pytest.fixture
def selection(monkeypatch):
    """Transport selection with an empty registry, Gloo as the only built-in left."""
    monkeypatch.setattr(transport, "_backend_transports", {})
    monkeypatch.setattr(transport, "SHM_ENABLED", False)
    for name in ("torch_transport", "nixl", "torchcomms_uniflow", "torchcomms_rdma"):
        monkeypatch.setattr(transport, f"{name}_available", lambda: False)
    monkeypatch.setattr(transport, "gloo_available", lambda: True)


# The entry ``torchstore.monarch.rdma`` registered when this file imported it.
_REGISTERED_ON_LOAD = dict(transport._backend_transports)


class _PatchedRDMABuffer(MonarchRDMATransportBuffer):
    """A buffer class patched over the RDMA module's own."""


def test_selection_reads_the_rdma_modules_probe_and_buffer_when_it_runs(
    selection, monkeypatch
):
    """Steer selection and buffer creation by patching the RDMA module's names."""
    monkeypatch.setitem(
        transport._backend_transports,
        TransportType.MonarchRDMA,
        _REGISTERED_ON_LOAD[TransportType.MonarchRDMA],
    )
    unset = SimpleNamespace(default_transport_type=TransportType.Unset)
    forced = SimpleNamespace(default_transport_type=TransportType.MonarchRDMA)
    for usable, chosen in (
        (True, TransportType.MonarchRDMA),
        (False, TransportType.Gloo),
    ):
        monkeypatch.setattr(
            monarch_rdma, "monarch_rdma_transport_available", lambda: usable
        )
        assert transport.get_available_transport(unset) is chosen
    monkeypatch.setattr(monarch_rdma, "MonarchRDMATransportBuffer", _PatchedRDMABuffer)
    assert type(transport.create_transport_buffer(forced)) is _PatchedRDMABuffer


def test_selection_takes_monarch_rdma_only_once_registered_and_usable(
    selection, caplog
):
    """Pick Monarch RDMA for an unset transport only while it is registered and usable."""
    caplog.set_level(logging.INFO, logger="torchstore.transport")
    unset = SimpleNamespace(default_transport_type=TransportType.Unset)
    forced = SimpleNamespace(default_transport_type=TransportType.MonarchRDMA)
    with pytest.raises(RuntimeError, match="Monarch RDMA transport is not registered"):
        transport.create_transport_buffer(forced)
    assert transport.get_available_transport(unset) is TransportType.Gloo

    def register(usable: bool) -> None:
        transport.register_backend_transport(
            TransportType.MonarchRDMA, MonarchRDMATransportBuffer, lambda: usable
        )

    register(True)
    assert type(transport.create_transport_buffer(unset)) is MonarchRDMATransportBuffer
    register(False)
    assert transport.get_available_transport(unset) is TransportType.Gloo
    assert type(transport.create_transport_buffer(forced)) is MonarchRDMATransportBuffer
    # The resolution log reports the registered probe's answer each time.
    assert [
        re.search(r"monarch_rdma=(\w+)", record.getMessage()).group(1)
        for record in caplog.records
        if "[ts-transport]" in record.getMessage()
    ] == ["False", "True", "False"]


@pytest.mark.parametrize("torchcomms", ["torchcomms_uniflow", "torchcomms_rdma"])
def test_selection_prefers_torchcomms_to_a_usable_monarch_rdma(
    selection, monkeypatch, torchcomms
):
    """Pick TorchComms ahead of a registered, usable Monarch RDMA, as the README's order says."""
    monkeypatch.setattr(transport, f"{torchcomms}_available", lambda: True)
    transport.register_backend_transport(
        TransportType.MonarchRDMA, MonarchRDMATransportBuffer, lambda: True
    )
    unset = SimpleNamespace(default_transport_type=TransportType.Unset)
    assert transport.get_available_transport(unset) is TransportType.TorchComms


def test_registry_takes_only_transports_selection_has_a_place_for(selection):
    """Refuse a transport type that selection would never ask the registry for."""
    with pytest.raises(ValueError, match="no place for TransportType.Gloo"):
        transport.register_backend_transport(
            TransportType.Gloo, MonarchRDMATransportBuffer, lambda: True
        )
    assert transport._backend_transports == {}
