from __future__ import annotations

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import uuid
from dataclasses import dataclass
from logging import getLogger
from typing import Any, TYPE_CHECKING, cast

import torch

from torchstore.logging import LatencyTracker
from torchstore.transport.buffers import TransportBuffer
from torchstore.transport.torchcomms.cache import (
    UniflowCache,
    UniflowRegistration,
    shutdown_uniflow_transport,
    uniflow_transport_module,
)
from torchstore.transport.types import Request

if TYPE_CHECKING:
    from uniflow._core import (
        MultiTransport,
        MultiTransportFactory,
        RegisteredSegmentSpan,
        RemoteRegisteredSegment,
        RemoteRegisteredSegmentSpan,
        TransferRequest,
        UniflowFuture,
    )

    from torchstore.strategy import StorageVolumeRef
    from torchstore.transport.buffers import TransportContext


logger = getLogger(__name__)

# uniflow handshake occurs in two phases: topology and connect
# if there is partial failure, we need to abort the handshake
_HANDSHAKE_TOPOLOGY = "topology"
_HANDSHAKE_CONNECT = "connect"
_HANDSHAKE_ABORT = "abort"
_UNIFLOW_CPU_DEVICE_ID = -1


@dataclass
class UniflowConnectionState:
    """Per-device handshake state for the current request batch."""

    topology: bytes | None = None
    bind_info: bytes | None = None
    transport: MultiTransport | None = None


@dataclass
class UniflowContext:
    """Per-entry state for uniflow-backed TorchComms operations."""

    export_id: bytes | None = None  # Serialized
    tensor_ref: torch.Tensor | None = None  # LOCAL only
    shape: torch.Size | None = None  # Serialized
    dtype: torch.dtype | None = None  # Serialized
    is_object: bool = False  # Serialized
    objects: Any = None  # Serialized
    device_id: int = -1  # Serialized

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["tensor_ref"] = None
        return state


@dataclass
class _UniflowTransfer:
    """Owns objects backing a Uniflow TransferRequest until completion."""

    remote_segment: RemoteRegisteredSegment
    local_span: RegisteredSegmentSpan
    remote_span: RemoteRegisteredSegmentSpan
    request: TransferRequest


class TorchCommsTransportBuffer(TransportBuffer):
    """Uniflow-backed TorchComms transport for one client request batch.

    The same serialized buffer instance participates on both sides of a
    request: the client owns local tensors and local transports, while the
    storage volume receives serialized Uniflow context and owns its local
    transport endpoints.

    There are three lifetimes to keep distinct:
    - Request state on this buffer: selected device ids, exported segment ids,
      tensor refs, and transports created by the current handshake.
    - Long-lived state in ``UniflowCache``: peer keys, factories, connected
      transports, and tensor registrations that can be reused across requests.
    - Handshake-scoped server transports: created on the storage volume before
      connect completes, then either promoted to the long-lived cache after a
      successful transfer or discarded on handshake/transfer failure.

    A new connection is built in two RPC handshakes. The topology pass lets the
    storage volume create and bind a transport from the client topology. The
    connect pass lets the client create/connect its transport using the storage
    volume bind info, then tells the storage volume to connect its pending
    transport. Both sides only publish connected transports to the reusable
    cache after the data request succeeds, so failed requests do not leave
    half-used connections behind.

    Cleanup is split by ownership: client-side incomplete transports are shut
    down locally, remote handshake transports are explicitly aborted/discarded,
    and request-only tensor/export state is dropped from this buffer.
    """

    supports_batch_puts = True
    supports_batch_gets = True

    def __init__(self, storage_volume_ref: "StorageVolumeRef") -> None:
        super().__init__(storage_volume_ref)

        self.peer_key = storage_volume_ref.transport_context.get(
            UniflowCache
        ).peer_key(storage_volume_ref.volume_id)

        self.handshake_id = uuid.uuid4().hex
        self._connections: dict[int, UniflowConnectionState] = {}
        self._contexts: list[UniflowContext] = []
        self._handshake_phase: str | None = None
        self._remote_handshake_pending = False

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["storage_volume_ref"] = None
        # Native transports are process-local; only topology and bind info cross
        # the RPC boundary during handshake calls.
        state["_connections"] = {
            device_id: UniflowConnectionState(
                topology=connection.topology,
                bind_info=connection.bind_info,
            )
            for device_id, connection in self._connections.items()
        }
        return state

    def _setup_local_transport(self, request: Request) -> None:
        tensor = request.tensor_val
        device_id = (
            _UNIFLOW_CPU_DEVICE_ID
            if tensor is None or not tensor.is_contiguous()
            else UniflowCache.device_to_id(tensor.device)
        )
        uniflow_cache = self.storage_volume_ref.transport_context.get(UniflowCache)

        # transport already exists for this device
        if uniflow_cache.contains_transport(self.peer_key, device_id):
            return

        if device_id in self._connections:
            return

        self._connections[device_id] = UniflowConnectionState(
            topology=uniflow_cache.factory(device_id).get_topology()
        )

    def _get_sv_transport(
        self, ctx: "TransportContext", device_id: int
    ) -> MultiTransport:
        uniflow_cache = ctx.get(UniflowCache)
        if device_id in self._connections:
            # This request just performed a handshake for the device, so the
            # storage volume must use the not-yet-promoted server transport.
            if not uniflow_cache.contains_handshake_transport(
                self.handshake_id, device_id
            ):
                raise RuntimeError(
                    "Missing handshake-scoped TorchComms transport on storage volume "
                    f"for handshake_id={self.handshake_id}, device_id={device_id}"
                )
            return uniflow_cache.get_handshake_transport(
                self.handshake_id, device_id
            )

        if not uniflow_cache.contains_transport(self.peer_key, device_id):
            raise RuntimeError(
                "Missing cached TorchComms transport on storage volume "
                f"for peer_key={self.peer_key}, device_id={device_id}"
            )
        return uniflow_cache.get_transport(self.peer_key, device_id)

    def _get_sv_factory(self, ctx: "TransportContext") -> MultiTransportFactory:
        return ctx.get(UniflowCache).factory(_UNIFLOW_CPU_DEVICE_ID)

    def requires_handshake(self, requests: list[Request]) -> bool:
        for request in requests:
            if not request.is_object:
                self._setup_local_transport(request)
        return len(self._connections) > 0

    async def perform_handshake(
        self,
        _requests: list[Request],
        meta_requests: list[Request],
        latency_tracker: LatencyTracker | None = None,
    ) -> None:
        tracker = latency_tracker or LatencyTracker("handshake")
        try:
            self._handshake_phase = _HANDSHAKE_TOPOLOGY
            topology_results = await self.storage_volume_ref.volume.handshake.call_one(
                self, meta_requests
            )
            tracker.track_step("volume.handshake.topology.call")
            self._post_topology_handshake(topology_results)
            tracker.track_step("post_topology_handshake")

            self._handshake_phase = _HANDSHAKE_CONNECT
            connect_results = await self.storage_volume_ref.volume.handshake.call_one(
                self, meta_requests
            )
            tracker.track_step("volume.handshake.connect.call")
            self._post_connect_handshake(connect_results)
            self._remote_handshake_pending = True
            tracker.track_step("post_connect_handshake")
        except Exception:
            self._cleanup_local_handshake()
            await self._abort_remote_handshake(meta_requests)
            raise
        finally:
            self._handshake_phase = None

    def _cleanup_local_handshake(self) -> None:
        """Close client-side handshake transports that were not published."""
        for connection in self._connections.values():
            shutdown_uniflow_transport(connection.transport)
            connection.transport = None
            connection.bind_info = None

    async def _abort_remote_handshake(self, meta_requests: list[Request]) -> None:
        if not self._connections:
            return
        try:
            self._handshake_phase = _HANDSHAKE_ABORT
            await self.storage_volume_ref.volume.handshake.call_one(self, meta_requests)
            self._remote_handshake_pending = False
        except Exception:
            logger.warning(
                "Failed to abort pending torchcomms handshake %s",
                self.handshake_id,
                exc_info=True,
            )

    def _post_topology_handshake(
        self,
        handshake_results: list[tuple[int, bytes, bytes]],
    ) -> None:
        uniflow_cache = self.storage_volume_ref.transport_context.get(UniflowCache)
        current_transport = None

        try:
            for device_id, peer_topology, peer_bind in handshake_results:
                if device_id not in self._connections:
                    raise RuntimeError(
                        "TorchComms handshake returned unexpected device "
                        f"{device_id}"
                    )
                factory = uniflow_cache.factory(device_id)
                current_transport = cast(
                    "MultiTransport",
                    factory.create_transport(peer_topology).unwrap(),
                )
                local_bind = current_transport.bind().unwrap()
                current_transport.connect(peer_bind).unwrap()
                self._connections[device_id].bind_info = local_bind
                self._connections[device_id].transport = current_transport
                current_transport = None
        except Exception:
            shutdown_uniflow_transport(current_transport)
            self._cleanup_local_handshake()
            raise

    def _post_connect_handshake(self, handshake_results: list[int]) -> None:
        connected_devices = set(handshake_results)
        requested_devices = set(self._connections)
        if connected_devices != requested_devices:
            raise RuntimeError(
                "TorchComms handshake completed with unexpected device set: "
                f"{connected_devices} != {requested_devices}"
            )

    async def recv_handshake(
        self,
        ctx: "TransportContext",
        _entries: list[tuple[Request, Any]],
    ) -> list[Any]:
        if self._handshake_phase == _HANDSHAKE_TOPOLOGY:
            return self._recv_topology_handshake(ctx)

        if self._handshake_phase == _HANDSHAKE_CONNECT:
            return self._recv_connect_handshake(ctx)

        if self._handshake_phase == _HANDSHAKE_ABORT:
            return self._recv_abort_handshake(ctx)

        raise RuntimeError(f"Unknown TorchComms handshake phase: {self._handshake_phase}")

    def _recv_topology_handshake(
        self,
        ctx: "TransportContext",
    ) -> list[tuple[int, bytes, bytes]]:
        uniflow_cache = ctx.get(UniflowCache)
        factory = uniflow_cache.factory(_UNIFLOW_CPU_DEVICE_ID)
        results = []
        current_transport = None

        try:
            for device_id, connection in self._connections.items():
                if connection.topology is None:
                    raise RuntimeError(
                        "Missing TorchComms topology for handshake "
                        f"{self.handshake_id}, device_id={device_id}"
                    )
                current_transport = cast(
                    "MultiTransport",
                    factory.create_transport(connection.topology).unwrap(),
                )
                bind_info = current_transport.bind().unwrap()
                peer_topology = factory.get_topology()
                # Keep the server transport private to this handshake until
                # the data path succeeds; otherwise a failed request could
                # poison the reusable transport cache.
                uniflow_cache.put_handshake_transport(
                    self.handshake_id, device_id, current_transport
                )
                current_transport = None
                results.append((device_id, peer_topology, bind_info))
        except Exception:
            # clean up partial failure
            shutdown_uniflow_transport(current_transport)
            uniflow_cache.discard_handshake(self.handshake_id)
            raise

        return results

    def _recv_connect_handshake(self, ctx: "TransportContext") -> list[int]:
        uniflow_cache = ctx.get(UniflowCache)
        connected_devices = []

        try:
            for device_id, connection in self._connections.items():
                if connection.bind_info is None:
                    raise RuntimeError(
                        "Missing TorchComms bind info for handshake "
                        f"{self.handshake_id}, device_id={device_id}"
                    )
                transport = uniflow_cache.get_handshake_transport(
                    self.handshake_id, device_id
                )
                transport.connect(connection.bind_info).unwrap()
                connected_devices.append(device_id)
        except Exception:
            uniflow_cache.discard_handshake(self.handshake_id)
            raise

        return connected_devices

    def _recv_abort_handshake(self, ctx: "TransportContext") -> list[Any]:
        ctx.get(UniflowCache).discard_handshake(self.handshake_id)
        return []

    def _publish_local_transports(self) -> None:
        """Promote client transports after the data request succeeds."""
        uniflow_cache = self.storage_volume_ref.transport_context.get(UniflowCache)
        for device_id, connection in self._connections.items():
            transport = connection.transport
            if transport is None:
                continue
            if uniflow_cache.contains_transport(self.peer_key, device_id):
                shutdown_uniflow_transport(transport)
            else:
                uniflow_cache.put_transport(self.peer_key, device_id, transport)
            connection.transport = None
            connection.bind_info = None

    def _promote_server_transports(self, ctx: "TransportContext") -> None:
        """Promote server transports after the data request succeeds."""
        uniflow_cache = ctx.get(UniflowCache)
        for device_id in self._connections:
            if not uniflow_cache.contains_handshake_transport(
                self.handshake_id, device_id
            ):
                continue
            uniflow_cache.promote_handshake_transport(
                self.handshake_id,
                self.peer_key,
                device_id,
            )

    def _allocate_ctx(self, tensor: torch.Tensor) -> UniflowContext:
        self._assert_valid_tensor(tensor, tensor.dtype, tensor.shape)
        uniflow_cache = self.storage_volume_ref.transport_context.get(UniflowCache)
        device_id = UniflowCache.device_to_id(tensor.device)
        factory = uniflow_cache.factory(device_id)
        registration = uniflow_cache.get_or_register(tensor, factory)
        export_id = registration.registered_segment.export_id().unwrap()
        return UniflowContext(
            export_id=export_id,
            tensor_ref=tensor,
            shape=tensor.shape,
            dtype=tensor.dtype,
            device_id=device_id,
        )

    async def _post_request_success(self) -> None:
        self._publish_local_transports()
        self._remote_handshake_pending = False

    async def _pre_put_hook(self, requests: list[Request]) -> None:
        self._contexts = []
        for request in requests:
            if request.is_object:
                self._contexts.append(
                    UniflowContext(is_object=True, objects=request.objects)
                )
                continue

            tensor = request.tensor_val
            if not tensor.is_contiguous():
                logger.warning(
                    f"PUT called with non-contiguous tensor (key={request.key}), "
                    "creating a contiguous CPU copy"
                )
                tensor = tensor.cpu().contiguous()
            self._contexts.append(self._allocate_ctx(tensor))

    async def _pre_get_hook(self, requests: list[Request]) -> None:
        meta_requests = [req.meta_only() for req in requests if req.tensor_val is None]
        if meta_requests:
            meta_results = await self.storage_volume_ref.volume.get_meta.call_one(
                meta_requests
            )
        else:
            meta_results = []
        meta_iterator = iter(meta_results)

        self._contexts = []
        for request in requests:
            if request.tensor_val is not None:
                tensor_ref = request.tensor_val
            else:
                meta = next(meta_iterator)
                if isinstance(meta, str) or meta is None:
                    self._contexts.append(UniflowContext(is_object=True))
                    continue
                if request.tensor_slice is not None:
                    meta = (request.tensor_slice.local_shape, *meta[1:])
                tensor_ref = torch.zeros(
                    meta[0], dtype=meta[1], device=torch.device("cpu")
                )

            self._contexts.append(self._allocate_ctx(tensor_ref))

    def _transfer_request(
        self,
        local_registration: UniflowRegistration,
        remote_segment: RemoteRegisteredSegment,
    ) -> _UniflowTransfer:
        module = uniflow_transport_module()
        local_span = local_registration.registered_segment.span(
            0, local_registration.length
        )
        remote_span = remote_segment.span(0, local_registration.length)
        request = cast("TransferRequest", module.TransferRequest(local_span, remote_span))
        return _UniflowTransfer(
            remote_segment=remote_segment,
            local_span=local_span,
            remote_span=remote_span,
            request=request,
        )

    def _export_id(self, uniflow_ctx: UniflowContext) -> bytes:
        if uniflow_ctx.export_id is None:
            raise RuntimeError("Missing Uniflow export id for transfer")
        return uniflow_ctx.export_id

    def _await_transfer(
        self,
        future: UniflowFuture,
        _transfer: _UniflowTransfer,
    ) -> None:
        # Keep the spans and imported remote segment alive until Uniflow
        # completes; native requests can reference those backing objects.
        future.get().unwrap()

    async def handle_put_request(
        self,
        ctx: "TransportContext",
        entries: list[tuple[Request, Any]],
    ) -> list[Any]:
        factory = self._get_sv_factory(ctx)
        uniflow_cache = ctx.get(UniflowCache)
        results = []
        try:
            for entry, uniflow_ctx in zip(entries, self._contexts, strict=True):
                _, maybe_tensor = entry
                if uniflow_ctx.is_object:
                    results.append(uniflow_ctx.objects)
                    continue

                if maybe_tensor is None:
                    maybe_tensor = torch.zeros(
                        uniflow_ctx.shape,
                        dtype=uniflow_ctx.dtype,
                        device=torch.device("cpu"),
                    )
                self._assert_valid_tensor(
                    maybe_tensor, uniflow_ctx.dtype, uniflow_ctx.shape
                )

                transport = self._get_sv_transport(ctx, uniflow_ctx.device_id)
                local_registration = uniflow_cache.get_or_register(
                    maybe_tensor, factory
                )
                remote_segment = cast(
                    "RemoteRegisteredSegment",
                    factory.import_segment(self._export_id(uniflow_ctx)).unwrap(),
                )
                transfer = self._transfer_request(local_registration, remote_segment)
                self._await_transfer(transport.get([transfer.request]), transfer)
                results.append(maybe_tensor)
        except Exception:
            ctx.get(UniflowCache).discard_handshake(self.handshake_id)
            raise

        self._promote_server_transports(ctx)

        return results

    async def handle_get_request(
        self,
        ctx: "TransportContext",
        entries: list[tuple[Request, Any]],
    ) -> None:
        factory = self._get_sv_factory(ctx)
        uniflow_cache = ctx.get(UniflowCache)
        try:
            for entry, uniflow_ctx in zip(entries, self._contexts, strict=True):
                _, data = entry
                if not isinstance(data, torch.Tensor):
                    uniflow_ctx.is_object = True
                    uniflow_ctx.objects = data
                    continue

                tensor = data
                if not tensor.is_contiguous():
                    contiguous_buffer = torch.zeros_like(
                        tensor,
                        device="cpu",
                        memory_format=torch.contiguous_format,
                    )
                    contiguous_buffer.copy_(tensor)
                    tensor = contiguous_buffer

                self._assert_valid_tensor(tensor, uniflow_ctx.dtype, uniflow_ctx.shape)
                transport = self._get_sv_transport(ctx, uniflow_ctx.device_id)
                local_registration = uniflow_cache.get_or_register(tensor, factory)
                remote_segment = cast(
                    "RemoteRegisteredSegment",
                    factory.import_segment(self._export_id(uniflow_ctx)).unwrap(),
                )
                transfer = self._transfer_request(local_registration, remote_segment)
                self._await_transfer(transport.put([transfer.request]), transfer)
        except Exception:
            ctx.get(UniflowCache).discard_handshake(self.handshake_id)
            raise

        self._promote_server_transports(ctx)

    async def _handle_storage_volume_response(
        self, _requests: list[Request], transport_buffer: "TransportBuffer"
    ) -> list[Any]:
        results = []
        for client_ctx, sv_ctx in zip(
            self._contexts, transport_buffer._contexts, strict=True
        ):
            if sv_ctx.is_object:
                results.append(sv_ctx.objects)
            else:
                results.append(client_ctx.tensor_ref)
        return results

    async def drop(self) -> None:
        """Release request-local state without clearing reusable cache entries."""
        self._cleanup_local_handshake()
        if self._remote_handshake_pending:
            await self._abort_remote_handshake([])
        for ctx in self._contexts:
            ctx.export_id = None
            ctx.tensor_ref = None
        self._contexts.clear()
        self._connections.clear()
        self._remote_handshake_pending = False
        self._handshake_phase = None
