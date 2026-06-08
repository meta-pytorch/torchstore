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

import pytest
import torch
import torchstore.transport.monarch_rdma as monarch_rdma
from torchstore.transport.buffers import TransportContext
from torchstore.transport.monarch_rdma import MonarchRDMATransportBuffer, RdmaContext
from torchstore.transport.types import Request


class FakeRDMABuffer:
    """Stand-in for ``monarch.rdma.RDMABuffer`` wrapping a byte view."""

    def __init__(self, byte_view: torch.Tensor) -> None:
        # The fake action reads/writes this view to emulate a transfer.
        self.mem = byte_view
        self.dropped = False

    async def drop(self) -> None:
        self.dropped = True


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


@pytest.fixture(autouse=True)
def fake_rdma(monkeypatch):
    """Patch the module's RDMA primitives with in-process fakes."""
    FakeRDMAAction.instances = []
    monkeypatch.setattr(monarch_rdma, "RDMABuffer", FakeRDMABuffer)
    monkeypatch.setattr(monarch_rdma, "RDMAAction", FakeRDMAAction)
    yield


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

        results = await buffer.handle_put_request(
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

        results = await buffer.handle_put_request(
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

        results = await buffer.handle_put_request(
            TransportContext(), _sv_entries_for_put(requests)
        )

        assert results == [1, 2]
        assert not FakeRDMAAction.instances[0].submitted


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
        await buffer.handle_get_request(
            TransportContext(), [(requests[0], stored1), (requests[1], stored2)]
        )
        results = await buffer._handle_storage_volume_response(requests, buffer)

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
        await buffer.handle_get_request(TransportContext(), [(requests[0], stored)])
        results = await buffer._handle_storage_volume_response(requests, buffer)

        assert torch.equal(results[0], stored)

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
        await buffer.handle_get_request(
            TransportContext(), [(requests[0], stored), (requests[1], obj)]
        )
        results = await buffer._handle_storage_volume_response(requests, buffer)

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
