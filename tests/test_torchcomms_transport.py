# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import weakref
from types import SimpleNamespace

import pytest
import torch

import torchstore.transport as transport_module
import torchstore.transport.torchcomms.cache as cache_mod
from torchstore.transport import TransportType
from torchstore.transport.buffers import TransportContext
from torchstore.transport.torchcomms.uniflow_buffer import TorchCommsTransportBuffer
from torchstore.transport.types import Request

# Unit tests authored by Codex: they cover the main TorchComms workflows and
# strategy choices: availability gates, handshakes, cache lifecycle, and buffer
# transfer cleanup.

requires_uniflow = pytest.mark.skipif(
    cache_mod._uniflow_transport is None,
    reason="uniflow is not available",
)


class _FakeResult:
    def __init__(self, value=None, error: str | None = None) -> None:
        self._value = value
        self._error = error

    def has_value(self) -> bool:
        return self._error is None

    def has_error(self) -> bool:
        return self._error is not None

    def value(self):
        return self._value

    def error(self):
        return self._error

    def unwrap(self):
        if self._error is not None:
            raise RuntimeError(self._error)
        return self._value


class _FakeSegment:
    def __init__(self, tensor: torch.Tensor) -> None:
        self.length = tensor.nbytes
        self.tensor_ref = tensor

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "_FakeSegment":
        return cls(tensor)


class _FakeSpan:
    # __weakref__ kept so _WeakTransferRequest can hold weak references to spans.
    __slots__ = ("offset", "length", "__weakref__")

    def __init__(self, offset: int, length: int) -> None:
        self.offset = offset
        self.length = length


class _FakeRemoteSegment:
    def span(self, offset: int, length: int) -> _FakeSpan:
        return _FakeSpan(offset, length)


class _FakeRegisteredSegment:
    def __init__(self, segment: _FakeSegment) -> None:
        self.length = segment.length

    def export_id(self) -> _FakeResult:
        return _FakeResult(b"export-id")

    def span(self, offset: int, length: int) -> _FakeSpan:
        return _FakeSpan(offset, length)


class _FakeFactory:
    def __init__(self) -> None:
        self.register_calls = 0

    def register_segment(self, segment: _FakeSegment) -> _FakeResult:
        self.register_calls += 1
        return _FakeResult(_FakeRegisteredSegment(segment))


class _FakeFuture:
    def __init__(self, request=None) -> None:
        self._request = request

    def get(self) -> _FakeResult:
        assert_alive = getattr(self._request, "assert_alive", None)
        if assert_alive is not None:
            assert_alive()
        return _FakeResult(None)


class _FakeTransferRequest:
    __slots__ = ("local", "remote")

    def __init__(self, local, remote) -> None:
        self.local = local
        self.remote = remote


class _WeakTransferRequest:
    """Fake native request that fails if span lifetimes are not preserved."""

    def __init__(self, local, remote) -> None:
        self.local = weakref.ref(local)
        self.remote = weakref.ref(remote)

    def assert_alive(self) -> None:
        assert self.local() is not None
        assert self.remote() is not None


class _FakeTransport:
    def __init__(self) -> None:
        self.shutdown_calls = 0
        self.connect_calls: list[bytes] = []

    def bind(self) -> _FakeResult:
        return _FakeResult(b"bind-info")

    def connect(self, info: bytes) -> _FakeResult:
        self.connect_calls.append(info)
        return _FakeResult(None)

    def get(self, requests) -> _FakeFuture:
        return _FakeFuture(requests[0] if requests else None)

    def put(self, requests) -> _FakeFuture:
        return _FakeFuture(requests[0] if requests else None)

    def shutdown(self) -> None:
        self.shutdown_calls += 1


class _FakeMultiTransportFactory:
    created_transports: list[_FakeTransport] = []

    def __init__(self, device_id: int) -> None:
        self.device_id = device_id

    @staticmethod
    def supported(_transport_type) -> _FakeResult:
        return _FakeResult(None)

    def register_segment(self, segment: _FakeSegment) -> _FakeResult:
        return _FakeResult(_FakeRegisteredSegment(segment))

    def import_segment(self, export_id: bytes) -> _FakeResult:
        del export_id
        return _FakeResult(_FakeRemoteSegment())

    def create_transport(self, peer_topology: bytes) -> _FakeResult:
        del peer_topology
        transport = _FakeTransport()
        type(self).created_transports.append(transport)
        return _FakeResult(transport)

    def get_topology(self) -> bytes:
        return f"topology-{self.device_id}".encode()


class _FailingRegisterMultiTransportFactory(_FakeMultiTransportFactory):
    def register_segment(self, segment: _FakeSegment) -> _FakeResult:
        del segment
        return _FakeResult(error="register failed")


class _UnsupportedMultiTransportFactory(_FakeMultiTransportFactory):
    @staticmethod
    def supported(_transport_type) -> _FakeResult:
        return _FakeResult(error="not supported")


class _FailingConnectHandshake:
    def __init__(self) -> None:
        self.server_ctx = TransportContext()
        self.calls: list[str] = []
        self._failed_connect = False

    async def call_one(self, transport_buffer, meta_requests):
        del meta_requests
        self.calls.append(transport_buffer._handshake_phase)
        if transport_buffer._handshake_phase == "connect" and not self._failed_connect:
            self._failed_connect = True
            await transport_buffer.recv_handshake(self.server_ctx, [])
            raise RuntimeError("connect failed")
        return await transport_buffer.recv_handshake(self.server_ctx, [])


class _SuccessfulPutVolume:
    def __init__(self) -> None:
        self.server_ctx = TransportContext()
        self.handshake = SimpleNamespace(call_one=self._handshake_call_one)
        self.put = SimpleNamespace(call=self._put_call)

    async def _handshake_call_one(self, transport_buffer, meta_requests):
        del meta_requests
        return await transport_buffer.recv_handshake(self.server_ctx, [])

    async def _put_call(self, transport_buffer, meta_requests):
        entries = [(request, None) for request in meta_requests]
        await transport_buffer.handle_put_request(self.server_ctx, entries)


def _clear_torchcomms_caches() -> None:
    cache_mod.torchcomms_uniflow_available.cache_clear()
    cache_mod.torchcomms_rdma_available.cache_clear()


def _set_uniflow_module(monkeypatch, module) -> None:
    monkeypatch.setattr(cache_mod, "_uniflow_transport", module, raising=False)
    _clear_torchcomms_caches()


def _set_torchcomms_transport_module(monkeypatch, module) -> None:
    monkeypatch.setattr(cache_mod, "_torchcomms_transport", module, raising=False)
    _clear_torchcomms_caches()


def _fake_uniflow_module(
    factory_cls=_FakeMultiTransportFactory,
    transfer_request_cls=_FakeTransferRequest,
):
    return SimpleNamespace(
        MultiTransportFactory=factory_cls,
        Segment=_FakeSegment,
        TransferRequest=transfer_request_cls,
        TransportType=SimpleNamespace(RDMA="rdma"),
    )


def _storage_volume_ref(
    default_transport_type: TransportType = TransportType.Unset,
    volume_hostname: str = "remote-host",
):
    return SimpleNamespace(
        volume=None,
        volume_id="volume-0",
        transport_context=TransportContext(),
        default_transport_type=default_transport_type,
        volume_hostname=volume_hostname,
    )


@pytest.fixture(autouse=True)
def _reset_torchcomms_state():
    _clear_torchcomms_caches()
    yield
    _clear_torchcomms_caches()


class TestTorchCommsAvailability:
    def test_uniflow_device_id_keeps_cpu_distinct_from_cuda_zero(self) -> None:
        assert cache_mod.UniflowCache.device_to_id(torch.device("cpu")) == -1
        assert cache_mod.UniflowCache.device_to_id(torch.device("cuda:0")) == 0
        assert cache_mod.RdmaTransportCache.device_to_index(torch.device("cpu")) == 0

    @requires_uniflow
    def test_torchcomms_rdma_available_for_legacy_bindings(self, monkeypatch) -> None:
        monkeypatch.setenv("USE_TORCHCOMMS", "1")
        rdma_transport = type(
            "FakeRdmaTransport",
            (),
            {"supported": staticmethod(lambda: True)},
        )
        _set_uniflow_module(monkeypatch, None)
        _set_torchcomms_transport_module(
            monkeypatch,
            SimpleNamespace(RdmaTransport=rdma_transport),
        )

        assert not cache_mod.torchcomms_uniflow_available()
        assert cache_mod.torchcomms_rdma_available()

    @requires_uniflow
    def test_torchcomms_uniflow_available_for_new_bindings(self, monkeypatch) -> None:
        monkeypatch.setenv("USE_TORCHCOMMS", "1")
        _set_uniflow_module(monkeypatch, _fake_uniflow_module())
        _set_torchcomms_transport_module(monkeypatch, None)

        assert cache_mod.torchcomms_uniflow_available()
        assert not cache_mod.torchcomms_rdma_available()

    def test_torchcomms_uniflow_unavailable_when_rdma_not_supported(
        self, monkeypatch
    ) -> None:
        monkeypatch.setenv("USE_TORCHCOMMS", "1")
        _set_uniflow_module(
            monkeypatch,
            _fake_uniflow_module(factory_cls=_UnsupportedMultiTransportFactory),
        )
        _set_torchcomms_transport_module(monkeypatch, None)

        assert not cache_mod.torchcomms_uniflow_available()

    def test_use_torchcomms_disables_legacy_fallback(self, monkeypatch) -> None:
        monkeypatch.setenv("USE_TORCHCOMMS", "0")
        rdma_transport = type(
            "FakeRdmaTransport",
            (),
            {"supported": staticmethod(lambda: True)},
        )
        _set_uniflow_module(monkeypatch, SimpleNamespace(MultiTransportFactory=object))
        _set_torchcomms_transport_module(
            monkeypatch,
            SimpleNamespace(RdmaTransport=rdma_transport),
        )

        assert not cache_mod.torchcomms_uniflow_available()
        assert not cache_mod.torchcomms_rdma_available()

    @requires_uniflow
    def test_uniflow_cache_reuses_registration(self, monkeypatch) -> None:
        _set_uniflow_module(
            monkeypatch,
            _fake_uniflow_module(),
        )

        cache = cache_mod.UniflowCache()
        factory = _FakeFactory()
        tensor = torch.randn(4, 4)

        reg1 = cache.get_or_register(tensor, factory)
        reg2 = cache.get_or_register(tensor, factory)

        assert reg1 is reg2
        assert factory.register_calls == 1

    @requires_uniflow
    def test_uniflow_cache_evicts_registration_with_tensor(self, monkeypatch) -> None:
        _set_uniflow_module(
            monkeypatch,
            _fake_uniflow_module(),
        )

        cache = cache_mod.UniflowCache()
        factory = _FakeFactory()
        tensor = torch.randn(4, 4)
        key = (tensor.data_ptr(), tensor.nbytes)

        cache.get_or_register(tensor, factory)
        assert key in cache._registrations

        del tensor

        assert key not in cache._registrations
        assert key not in cache._storage_refs


@requires_uniflow
class TestTorchCommsCaches:
    def test_uniflow_cache_transport_replacement_shuts_down_previous_transport(
        self, monkeypatch
    ) -> None:
        _set_uniflow_module(monkeypatch, _fake_uniflow_module())
        cache = cache_mod.UniflowCache()
        first = _FakeTransport()
        second = _FakeTransport()

        cache.put_transport("peer", 0, first)
        cache.put_transport("peer", 0, second)

        assert first.shutdown_calls == 1
        assert cache.get_transport("peer", 0) is second

    def test_uniflow_cache_clear_shuts_down_transports(self, monkeypatch) -> None:
        _set_uniflow_module(monkeypatch, _fake_uniflow_module())
        cache = cache_mod.UniflowCache()
        first = _FakeTransport()
        second = _FakeTransport()

        cache.put_transport("peer-a", 0, first)
        cache.put_handshake_transport("peer-b", 0, second)
        cache.clear()

        assert first.shutdown_calls == 1
        assert second.shutdown_calls == 1
        assert cache._transports == {}
        assert cache._handshake_transports == {}

    def test_uniflow_cache_discard_handshake_shuts_down_handshake_transports(
        self, monkeypatch
    ) -> None:
        _set_uniflow_module(monkeypatch, _fake_uniflow_module())
        cache = cache_mod.UniflowCache()
        stale_a = _FakeTransport()
        stale_b = _FakeTransport()
        fresh = _FakeTransport()

        cache.put_handshake_transport("stale", 0, stale_a)
        cache.put_handshake_transport("stale", 1, stale_b)
        cache.put_handshake_transport("fresh", 0, fresh)
        cache.discard_handshake("stale")

        assert stale_a.shutdown_calls == 1
        assert stale_b.shutdown_calls == 1
        assert fresh.shutdown_calls == 0
        assert "stale" not in cache._handshake_transports
        assert cache.get_handshake_transport("fresh", 0) is fresh


class TestTorchCommsSelection:
    def test_torchcomms_rdma_alias_preserves_existing_callers(self) -> None:
        assert TransportType["TorchCommsRDMA"] is TransportType.TorchComms

    def test_get_available_transport_prefers_shared_memory(self, monkeypatch) -> None:
        monkeypatch.setattr(transport_module, "SHM_ENABLED", True)
        monkeypatch.setattr(transport_module, "is_local_to_volume", lambda _ref: True)
        monkeypatch.setattr(
            transport_module,
            "torchcomms_uniflow_available",
            lambda: True,
        )

        assert (
            transport_module.get_available_transport(_storage_volume_ref())
            == TransportType.SharedMemory
        )

    @requires_uniflow
    def test_create_transport_buffer_prefers_uniflow(self, monkeypatch) -> None:
        _set_uniflow_module(monkeypatch, _fake_uniflow_module())
        monkeypatch.setattr(
            transport_module,
            "torchcomms_uniflow_available",
            lambda: True,
        )
        monkeypatch.setattr(
            transport_module,
            "torchcomms_rdma_available",
            lambda: True,
        )

        buffer = transport_module.create_transport_buffer(
            _storage_volume_ref(default_transport_type=TransportType.TorchComms)
        )

        assert isinstance(buffer, TorchCommsTransportBuffer)

    def test_create_transport_buffer_falls_back_to_legacy_rdma(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            transport_module,
            "torchcomms_uniflow_available",
            lambda: False,
        )
        monkeypatch.setattr(
            transport_module,
            "torchcomms_rdma_available",
            lambda: True,
        )

        buffer = transport_module.create_transport_buffer(
            _storage_volume_ref(default_transport_type=TransportType.TorchComms)
        )

        assert buffer.__class__.__name__ == "TorchCommsRdmaTransportBuffer"

    def test_get_available_transport_returns_torchcomms_for_legacy_only(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            transport_module,
            "torchcomms_uniflow_available",
            lambda: False,
        )
        monkeypatch.setattr(
            transport_module,
            "torchcomms_rdma_available",
            lambda: True,
        )

        assert (
            transport_module.get_available_transport(_storage_volume_ref())
            == TransportType.TorchComms
        )

    def test_create_transport_buffer_rejects_missing_torchcomms(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            transport_module,
            "torchcomms_uniflow_available",
            lambda: False,
        )
        monkeypatch.setattr(
            transport_module,
            "torchcomms_rdma_available",
            lambda: False,
        )

        with pytest.raises(RuntimeError, match="TorchComms transport is not available"):
            transport_module.create_transport_buffer(
                _storage_volume_ref(default_transport_type=TransportType.TorchComms)
            )


@requires_uniflow
class TestTorchCommsHandshake:
    def test_get_sv_transport_raises_clear_error_for_missing_cached_transport(
        self,
        monkeypatch,
    ) -> None:
        _set_uniflow_module(monkeypatch, _fake_uniflow_module())
        buffer = TorchCommsTransportBuffer(
            _storage_volume_ref(default_transport_type=TransportType.TorchComms)
        )

        with pytest.raises(
            RuntimeError,
            match="Missing cached TorchComms transport on storage volume",
        ):
            buffer._get_sv_transport(TransportContext(), -1)

    def test_perform_handshake_keeps_uniflow_transports_private_until_request_success(
        self, monkeypatch
    ) -> None:
        _FakeMultiTransportFactory.created_transports = []
        _set_uniflow_module(
            monkeypatch,
            _fake_uniflow_module(),
        )
        volume = _SuccessfulPutVolume()
        storage_volume_ref = _storage_volume_ref(
            default_transport_type=TransportType.TorchComms
        )
        storage_volume_ref.volume = volume
        buffer = TorchCommsTransportBuffer(storage_volume_ref)
        request = Request.from_tensor("key", torch.zeros(4))

        assert buffer.requires_handshake([request])
        asyncio.run(buffer.perform_handshake([request], [request.meta_only()]))

        client_cache = storage_volume_ref.transport_context.get(cache_mod.UniflowCache)
        server_cache = volume.server_ctx.get(cache_mod.UniflowCache)

        assert not client_cache.contains_transport(buffer.peer_key, -1)
        assert not server_cache.contains_transport(buffer.peer_key, -1)
        assert server_cache.contains_handshake_transport(buffer.handshake_id, -1)

    def test_perform_handshake_aborts_and_cleans_up_on_connect_failure(
        self, monkeypatch
    ) -> None:
        _FakeMultiTransportFactory.created_transports = []
        _set_uniflow_module(
            monkeypatch,
            _fake_uniflow_module(transfer_request_cls=object),
        )
        handshake = _FailingConnectHandshake()
        storage_volume_ref = _storage_volume_ref(
            default_transport_type=TransportType.TorchComms
        )
        storage_volume_ref.volume = SimpleNamespace(
            handshake=SimpleNamespace(call_one=handshake.call_one)
        )
        buffer = TorchCommsTransportBuffer(storage_volume_ref)
        request = Request.from_tensor("key", torch.zeros(4))

        assert buffer.requires_handshake([request])

        with pytest.raises(RuntimeError, match="connect failed"):
            asyncio.run(buffer.perform_handshake([request], [request.meta_only()]))

        client_cache = storage_volume_ref.transport_context.get(cache_mod.UniflowCache)
        server_cache = handshake.server_ctx.get(cache_mod.UniflowCache)

        assert handshake.calls == ["topology", "connect", "abort"]
        assert not client_cache.contains_transport(buffer.peer_key, -1)
        assert server_cache._transports == {}
        assert server_cache._handshake_transports == {}
        assert _FakeMultiTransportFactory.created_transports[0].shutdown_calls == 1

    def test_put_promotes_uniflow_transports_after_success(self, monkeypatch) -> None:
        _FakeMultiTransportFactory.created_transports = []
        _set_uniflow_module(
            monkeypatch,
            _fake_uniflow_module(),
        )
        volume = _SuccessfulPutVolume()
        storage_volume_ref = _storage_volume_ref(
            default_transport_type=TransportType.TorchComms
        )
        storage_volume_ref.volume = volume
        buffer = TorchCommsTransportBuffer(storage_volume_ref)
        request = Request.from_tensor("key", torch.zeros(4))

        asyncio.run(buffer.put_to_storage_volume([request]))

        client_cache = storage_volume_ref.transport_context.get(cache_mod.UniflowCache)
        server_cache = volume.server_ctx.get(cache_mod.UniflowCache)

        assert client_cache.contains_transport(buffer.peer_key, -1)
        assert server_cache.contains_transport(buffer.peer_key, -1)
        assert server_cache._handshake_transports == {}

    def test_put_replaces_stale_server_cached_transport_after_success(
        self, monkeypatch
    ) -> None:
        _FakeMultiTransportFactory.created_transports = []
        _set_uniflow_module(monkeypatch, _fake_uniflow_module())
        volume = _SuccessfulPutVolume()
        storage_volume_ref = _storage_volume_ref(
            default_transport_type=TransportType.TorchComms
        )
        storage_volume_ref.volume = volume
        buffer = TorchCommsTransportBuffer(storage_volume_ref)
        stale = _FakeTransport()
        server_cache = volume.server_ctx.get(cache_mod.UniflowCache)
        server_cache.put_transport(buffer.peer_key, -1, stale)
        request = Request.from_tensor("key", torch.zeros(4))

        asyncio.run(buffer.put_to_storage_volume([request]))

        assert stale.shutdown_calls == 1
        assert server_cache.get_transport(buffer.peer_key, -1) is (
            _FakeMultiTransportFactory.created_transports[0]
        )

    def test_put_keeps_transfer_components_alive_until_completion(
        self, monkeypatch
    ) -> None:
        _FakeMultiTransportFactory.created_transports = []
        _set_uniflow_module(
            monkeypatch,
            _fake_uniflow_module(transfer_request_cls=_WeakTransferRequest),
        )
        volume = _SuccessfulPutVolume()
        storage_volume_ref = _storage_volume_ref(
            default_transport_type=TransportType.TorchComms
        )
        storage_volume_ref.volume = volume
        buffer = TorchCommsTransportBuffer(storage_volume_ref)
        request = Request.from_tensor("key", torch.zeros(4))

        asyncio.run(buffer.put_to_storage_volume([request]))

        assert volume.server_ctx.get(cache_mod.UniflowCache).contains_transport(
            buffer.peer_key, -1
        )

    def test_put_aborts_remote_handshake_when_pre_put_fails(self, monkeypatch) -> None:
        _FailingRegisterMultiTransportFactory.created_transports = []
        _set_uniflow_module(
            monkeypatch,
            _fake_uniflow_module(
                factory_cls=_FailingRegisterMultiTransportFactory,
            ),
        )
        volume = _SuccessfulPutVolume()
        storage_volume_ref = _storage_volume_ref(
            default_transport_type=TransportType.TorchComms
        )
        storage_volume_ref.volume = volume
        buffer = TorchCommsTransportBuffer(storage_volume_ref)
        request = Request.from_tensor("key", torch.zeros(4))

        with pytest.raises(RuntimeError, match="register failed"):
            asyncio.run(buffer.put_to_storage_volume([request]))

        client_cache = storage_volume_ref.transport_context.get(cache_mod.UniflowCache)
        server_cache = volume.server_ctx.get(cache_mod.UniflowCache)

        assert not client_cache.contains_transport(buffer.peer_key, -1)
        assert server_cache._transports == {}
        assert server_cache._handshake_transports == {}
