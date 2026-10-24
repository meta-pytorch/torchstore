# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""StorageVolume transport backed by ``torch.distributed._transport``.

Each client process keeps one connected endpoint per storage volume. The
volume performs one-sided reads (put) and writes (get) against memory the
client registered.

Transfers cannot be cancelled, so the volume always waits for submitted
transfers to finish before returning, even when its handler is cancelled. As
with the NIXL transport, a client whose request is cancelled may free memory
the volume is still transferring.

Limitations:
- Both sides register and unregister every tensor per request (~1.8 ms each
  with NIXL over UCX), so large state dicts pay seconds per transfer. A
  registration cache is a follow-up.
- If re-registering a buffer changes its backend metadata, or any request
  fails (including user errors and cancellation), the endpoint is replaced:
  a new transport on both sides plus a handshake RPC.
- Volumes hold a client's endpoints until that client reports them closed or
  the volume is reset.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import uuid
from dataclasses import dataclass
from functools import cache
from typing import Any, TYPE_CHECKING

import torch
from typing_extensions import override

from torchstore.transport.buffers import TransportBuffer, TransportCache
from torchstore.transport.types import Request

if TYPE_CHECKING:
    from torch.distributed._transport import Memory, RemoteBuffer, Transport

    from torchstore.logging import LatencyTracker
    from torchstore.strategy import StorageVolumeRef
    from torchstore.transport.buffers import TransportContext

logger = logging.getLogger(__name__)

# Name of a registered torch.distributed transport backend (e.g. "nixl").
# Setting it enables this transport.
ENV_TORCHSTORE_TORCH_TRANSPORT_BACKEND = os.environ.get(
    "TORCHSTORE_TORCH_TRANSPORT_BACKEND", ""
).lower()

# Endpoint (re)connections per request before giving up on descriptor
# conflicts.
_MAX_ENDPOINT_ATTEMPTS = 4
# Descriptors a client endpoint exposes before it is replaced, bounding the
# metadata both sides cache.
_MAX_EXPOSED = 16384

_RegistrationKey = tuple[int, int, str]


@cache
def torch_transport_available() -> bool:
    """Return whether the selected torch.distributed transport backend works.

    Creates and closes one transport, since registered backends may lack
    their native dependencies.
    """
    if not ENV_TORCHSTORE_TORCH_TRANSPORT_BACKEND:
        return False
    try:
        from torch.distributed._transport import available_transports, new_transport
    except ImportError:
        return False
    if ENV_TORCHSTORE_TORCH_TRANSPORT_BACKEND not in available_transports():
        return False
    try:
        new_transport(ENV_TORCHSTORE_TORCH_TRANSPORT_BACKEND).close()
    except Exception:
        logger.warning(
            "torch.distributed transport backend %r is unusable",
            ENV_TORCHSTORE_TORCH_TRANSPORT_BACKEND,
            exc_info=True,
        )
        return False
    return True


def _registration_key(tensor: torch.Tensor) -> _RegistrationKey:
    return (tensor.data_ptr(), tensor.nbytes, str(tensor.device))


class _Endpoint:
    """A connected transport and its refcounted memory registrations.

    Registrations hold their tensors, so each request unregisters its memory
    when it finishes. Backends alias registrations of the same byte range, so
    concurrent requests sharing a tensor share one registration.
    """

    def __init__(self, transport: Transport) -> None:
        self.transport = transport
        self.id = uuid.uuid4().hex
        # Requests (client) or transfer batches (volume) using the endpoint.
        self.active = 0
        self.retired = False
        self.closed = False
        self._registrations: dict[_RegistrationKey, tuple[Memory, int]] = {}
        # Descriptors exposed to the peer. Peers may cache metadata per
        # descriptor (NIXL rejects reloading a descriptor with new metadata),
        # so a range re-registered with different metadata, e.g. memory freed
        # and reallocated at the same address, needs a new endpoint. Values are
        # digests of the serialized descriptors.
        self._exposed: dict[_RegistrationKey, bytes] = {}

    def register(self, tensor: torch.Tensor) -> tuple[_RegistrationKey, Memory]:
        key = _registration_key(tensor)
        memory, count = self._registrations.get(key, (None, 0))
        if memory is None:
            # NIXL cannot register 0-dim tensors; register the same bytes as 1-dim.
            memory = self.transport.register_memory(tensor.reshape(-1))
        self._registrations[key] = (memory, count + 1)
        return key, memory

    def unregister(self, keys: list[_RegistrationKey]) -> None:
        # Closing the transport released every registration.
        if self.closed:
            return
        for key in keys:
            memory, count = self._registrations.pop(key)
            if count > 1:
                self._registrations[key] = (memory, count - 1)
                continue
            try:
                self.transport.unregister_memory(memory)
            except Exception:
                logger.error(
                    "Failed to unregister torch transport memory", exc_info=True
                )

    def expose(self, key: _RegistrationKey, memory: Memory) -> RemoteBuffer | None:
        """Describe registered memory for the peer; None if it conflicts."""
        remote_buffer = memory.to_remote_buffer()
        digest = hashlib.blake2b(remote_buffer.serialize(), digest_size=16).digest()
        if self._exposed.setdefault(key, digest) != digest:
            return None
        return remote_buffer

    def exposed_full(self) -> bool:
        return len(self._exposed) >= _MAX_EXPOSED

    def release(self) -> None:
        self.active -= 1
        self.close_if_idle()

    def retire(self) -> None:
        """Stop handing out this endpoint; close once it is idle."""
        self.retired = True
        self.close_if_idle()

    def close_if_idle(self) -> None:
        if self.retired and not self.active:
            self.close()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._registrations.clear()
        try:
            self.transport.close()
        except Exception:
            logger.error("Failed to close torch transport", exc_info=True)

    async def transfer(
        self, read: bool, transfers: list[tuple[torch.Tensor, RemoteBuffer]]
    ) -> None:
        """Run transfers and wait until all of them finished."""
        from torch.distributed._transport import wait_all

        keys, works = [], []
        self.active += 1
        try:
            for tensor, remote_buffer in transfers:
                key, memory = self.register(tensor)
                keys.append(key)
                if read:
                    work = self.transport.read(
                        memory.to_mutable_view(), remote_buffer, async_op=True
                    )
                else:
                    work = self.transport.write(
                        memory.to_view(), remote_buffer, async_op=True
                    )
                works.append(work)
        finally:
            # Drain submitted transfers even if a submission failed or the
            # caller is cancelled.
            waiter = asyncio.ensure_future(wait_all(works))

            def done(future: asyncio.Future[None]) -> None:
                if not future.cancelled() and future.exception() is not None:
                    logger.warning(
                        "torch transport transfer failed", exc_info=future.exception()
                    )
                self.unregister(keys)
                self.release()

            waiter.add_done_callback(done)
            try:
                await asyncio.shield(waiter)
            except asyncio.CancelledError:
                # Keep the RPC open so the client keeps its memory registered.
                while not waiter.done():
                    try:
                        await asyncio.wait([waiter])
                    except asyncio.CancelledError:
                        pass
                raise


class TorchTransportCache(TransportCache):
    """Process-local endpoints.

    Clients key endpoints by storage volume ID and keep retired endpoints
    until that volume learns they closed. Storage volumes key endpoints by
    ``(client_id, endpoint_id)`` and close each when its client reports it
    closed.
    """

    def __init__(self) -> None:
        self.client_id = uuid.uuid4().hex
        # Client side.
        self.endpoints: dict[str, _Endpoint] = {}
        # (volume_id, endpoint)
        self.retired: list[tuple[str, _Endpoint]] = []
        self._locks: dict[str, asyncio.Lock] = {}
        # Storage volume side.
        self.peers: dict[tuple[str, str], _Endpoint] = {}

    def lock(self, volume_id: str) -> asyncio.Lock:
        return self._locks.setdefault(volume_id, asyncio.Lock())

    def put(self, volume_id: str, endpoint: _Endpoint) -> None:
        previous = self.endpoints.get(volume_id)
        if previous is not None:
            self.retire(volume_id, previous)
        self.endpoints[volume_id] = endpoint

    def retire(self, volume_id: str, endpoint: _Endpoint) -> None:
        if self.endpoints.get(volume_id) is endpoint:
            del self.endpoints[volume_id]
        endpoint.retire()
        if (volume_id, endpoint) not in self.retired:
            self.retired.append((volume_id, endpoint))

    def closed_endpoint_ids(self, volume_id: str) -> list[str]:
        """Closed endpoint IDs to report to a storage volume."""
        return [e.id for v, e in self.retired if v == volume_id and e.closed]

    def forget(self, volume_id: str, endpoint_ids: list[str]) -> None:
        """Drop endpoints a storage volume has closed."""
        self.retired = [
            (v, e)
            for v, e in self.retired
            if v != volume_id or e.id not in endpoint_ids
        ]

    def close_peers(self, client_id: str, endpoint_ids: list[str]) -> None:
        """Close volume endpoints whose client side closed."""
        for endpoint_id in endpoint_ids:
            endpoint = self.peers.pop((client_id, endpoint_id), None)
            if endpoint is not None:
                endpoint.retire()

    @override
    def clear(self) -> None:
        # Endpoints with transfers in flight close once they finish.
        for endpoint in (
            *self.endpoints.values(),
            *(endpoint for _, endpoint in self.retired),
            *self.peers.values(),
        ):
            endpoint.retire()
        self.endpoints.clear()
        self.retired.clear()
        self._locks.clear()
        self.peers.clear()


@dataclass
class TorchTransportRequestContext:
    """Serializable state for one tensor or object in a request batch."""

    # TODO: send RemoteBuffer.serialize() bytes once torch can deserialize
    # them without knowing the backend class.
    remote_buffer: RemoteBuffer | None = None
    # Client-local; stripped from RPC state because the transport moves the data.
    tensor: torch.Tensor | None = None
    shape: torch.Size | None = None
    dtype: torch.dtype | None = None
    is_object: bool = False
    objects: Any = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["tensor"] = None
        return state


class TorchDistributedTransportBuffer(TransportBuffer):
    """Move StorageVolume tensors with a ``torch.distributed._transport`` backend."""

    supports_batch_puts = True
    supports_batch_gets = True

    def __init__(self, storage_volume_ref: StorageVolumeRef) -> None:
        super().__init__(storage_volume_ref)
        if not torch_transport_available():
            raise RuntimeError(
                "torch.distributed transport is unavailable. Set "
                "TORCHSTORE_TORCH_TRANSPORT_BACKEND to a backend listed by "
                "torch.distributed._transport.available_transports()."
            )
        self._backend = ENV_TORCHSTORE_TORCH_TRANSPORT_BACKEND
        self._client_id: str | None = None
        self._client_url: bytes | None = None
        self._endpoint_id: str | None = None
        # Client endpoints that closed; the volume closes its side.
        self._closed_endpoint_ids: list[str] = []
        self._contexts: list[TorchTransportRequestContext] = []
        # Client-local state, stripped from RPC state.
        self._pending: _Endpoint | None = None
        self._endpoint: _Endpoint | None = None
        self._keys: list[_RegistrationKey] = []
        self._succeeded = False

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["storage_volume_ref"] = None
        state["_pending"] = None
        state["_endpoint"] = None
        state["_keys"] = []
        return state

    def _client_cache(self) -> TorchTransportCache:
        return self.storage_volume_ref.transport_context.get(TorchTransportCache)

    def _report_closed(self) -> None:
        cache = self._client_cache()
        self._client_id = cache.client_id
        self._closed_endpoint_ids = cache.closed_endpoint_ids(
            self.storage_volume_ref.volume_id
        )

    @override
    def requires_handshake(self, requests: list[Request]) -> bool:
        return self.storage_volume_ref.volume_id not in self._client_cache().endpoints

    @override
    async def perform_handshake(
        self,
        requests: list[Request],
        meta_requests: list[Request],
        latency_tracker: LatencyTracker | None = None,
    ) -> None:
        volume_id = self.storage_volume_ref.volume_id
        cache = self._client_cache()
        async with cache.lock(volume_id):
            # A concurrent request may have connected while this one waited.
            if volume_id in cache.endpoints:
                return
            # recv_handshake ignores the requests; the volume would validate them
            # against stored data.
            await super().perform_handshake(requests, [], latency_tracker)

    @override
    async def _pre_handshake(self) -> None:
        from torch.distributed._transport import new_transport

        self._pending = _Endpoint(new_transport(self._backend))
        self._endpoint_id = self._pending.id
        self._client_url = self._pending.transport.bind()
        self._report_closed()

    @override
    async def recv_handshake(
        self,
        ctx: TransportContext,
        entries: list[tuple[Request, Any]],
    ) -> list[Any]:
        from torch.distributed._transport import new_transport

        assert self._client_id is not None and self._endpoint_id is not None
        assert self._client_url is not None
        cache = ctx.get(TorchTransportCache)
        cache.close_peers(self._client_id, self._closed_endpoint_ids)
        endpoint = _Endpoint(new_transport(self._backend))
        try:
            url = endpoint.transport.bind()
            endpoint.transport.connect(self._client_url)
        except BaseException:
            endpoint.close()
            raise
        cache.peers[(self._client_id, self._endpoint_id)] = endpoint
        return [url]

    @override
    async def _post_handshake(
        self,
        handshake_results: list[Any],
        requests: list[Request],
    ) -> None:
        assert self._pending is not None
        (volume_url,) = handshake_results
        self._pending.transport.connect(volume_url)
        self._client_cache().put(self.storage_volume_ref.volume_id, self._pending)
        self._pending = None

    async def _register_client_tensors(self) -> None:
        """Expose context tensors through the current endpoint."""
        volume_id = self.storage_volume_ref.volume_id
        self._synchronize_streams()
        for _ in range(_MAX_ENDPOINT_ATTEMPTS):
            endpoint = self._client_cache().endpoints.get(volume_id)
            if endpoint is None:
                # A failed request retired the endpoint after our handshake.
                await self.perform_handshake([], [])
                continue
            if endpoint.exposed_full() or not self._expose_contexts(endpoint):
                await self._replace_endpoint(endpoint)
                continue
            endpoint.active += 1
            self._endpoint = endpoint
            self._endpoint_id = endpoint.id
            self._report_closed()
            return
        raise RuntimeError(
            f"Failed to register torch transport memory for volume {volume_id} "
            f"after {_MAX_ENDPOINT_ATTEMPTS} endpoint attempts"
        )

    def _synchronize_streams(self) -> None:
        """Wait for queued kernels on client tensors; DMA ignores CUDA streams."""
        devices = {
            context.tensor.device
            for context in self._contexts
            if context.tensor is not None and context.tensor.is_cuda
        }
        for device in devices:
            torch.cuda.current_stream(device).synchronize()

    def _expose_contexts(self, endpoint: _Endpoint) -> bool:
        """Register context tensors; False if a descriptor conflicts."""
        keys = []
        try:
            for context in self._contexts:
                tensor = context.tensor
                # Empty tensors cannot be registered and have no bytes to move.
                if tensor is None or tensor.numel() == 0:
                    continue
                assert tensor.is_contiguous()
                key, memory = endpoint.register(tensor)
                keys.append(key)
                context.remote_buffer = endpoint.expose(key, memory)
                if context.remote_buffer is None:
                    endpoint.unregister(keys)
                    return False
        except BaseException:
            endpoint.unregister(keys)
            raise
        self._keys = keys
        return True

    async def _replace_endpoint(self, endpoint: _Endpoint) -> None:
        """Replace an endpoint whose peer may cache stale descriptor metadata."""
        volume_id = self.storage_volume_ref.volume_id
        cache = self._client_cache()
        async with cache.lock(volume_id):
            if cache.endpoints.get(volume_id) is not endpoint:
                # Another request already replaced it.
                return
            logger.info(
                "Replacing torch transport endpoint for volume %s: memory was "
                "re-registered with new metadata or too many descriptors were "
                "exposed",
                volume_id,
            )
            # Requests still using the old endpoint keep it open.
            cache.retire(volume_id, endpoint)
            await super().perform_handshake([], [])

    @override
    async def _pre_put_hook(self, requests: list[Request]) -> None:
        self._contexts = []
        for request in requests:
            if request.is_object:
                self._contexts.append(
                    TorchTransportRequestContext(
                        is_object=True, objects=request.objects
                    )
                )
                continue

            tensor = request.tensor_val
            assert tensor is not None
            if not tensor.is_contiguous():
                logger.warning(
                    "torch transport PUT received a non-contiguous tensor for "
                    "key=%s; staging a contiguous CPU copy",
                    request.key,
                )
                tensor = tensor.cpu().contiguous()
            self._contexts.append(
                TorchTransportRequestContext(
                    tensor=tensor, shape=tensor.shape, dtype=tensor.dtype
                )
            )
        await self._register_client_tensors()

    @override
    async def _pre_get_hook(self, requests: list[Request]) -> None:
        # Fetch metadata in a single batch, preserving order.
        meta_requests = [req.meta_only() for req in requests if req.tensor_val is None]
        meta_results = (
            await self.storage_volume_ref.volume.get_meta.call_one(meta_requests)
            if meta_requests
            else []
        )
        meta_iterator = iter(meta_results)

        self._contexts = []
        for request in requests:
            tensor = request.tensor_val
            if tensor is None:
                meta = next(meta_iterator)
                if isinstance(meta, str) or meta is None:
                    self._contexts.append(TorchTransportRequestContext(is_object=True))
                    continue
                shape, dtype = meta
                if request.tensor_slice is not None:
                    shape = request.tensor_slice.local_shape
                tensor = torch.empty(shape, dtype=dtype, device="cpu")
            elif not tensor.is_contiguous():
                # Transfers need contiguous memory; the result is returned in
                # the staging tensor.
                tensor = torch.empty(
                    tensor.shape, dtype=tensor.dtype, device=tensor.device
                )
            self._contexts.append(
                TorchTransportRequestContext(
                    tensor=tensor, shape=tensor.shape, dtype=tensor.dtype
                )
            )
        await self._register_client_tensors()

    async def _transfer(
        self,
        ctx: TransportContext,
        read: bool,
        transfers: list[tuple[torch.Tensor, RemoteBuffer]],
    ) -> None:
        assert self._client_id is not None
        cache = ctx.get(TorchTransportCache)
        cache.close_peers(self._client_id, self._closed_endpoint_ids)
        if not transfers:
            return
        endpoint = cache.peers.get((self._client_id, self._endpoint_id))
        if endpoint is None:
            raise RuntimeError(
                f"No torch transport endpoint for client {self._client_id}"
            )
        await endpoint.transfer(read, transfers)

    @override
    async def handle_put_request(
        self,
        ctx: TransportContext,
        entries: list[tuple[Request, Any]],
    ) -> list[Any]:
        """Called by storage volume. Read from the client's registered tensors."""
        results: list[Any] = []
        transfers = []
        for (_, tensor), context in zip(entries, self._contexts, strict=True):
            if context.is_object:
                results.append(context.objects)
                continue

            if tensor is None:
                tensor = torch.empty(context.shape, dtype=context.dtype, device="cpu")
            self._assert_valid_tensor(tensor, context.dtype, context.shape)
            if tensor.numel() != 0:
                assert context.remote_buffer is not None
                transfers.append((tensor, context.remote_buffer))
            results.append(tensor)
        await self._transfer(ctx, True, transfers)
        return results

    @override
    async def handle_get_request(
        self,
        ctx: TransportContext,
        entries: list[tuple[Request, Any]],
    ) -> None:
        """Called by storage volume. Write into the client's registered tensors."""
        transfers = []
        for (_, data), context in zip(entries, self._contexts, strict=True):
            if not isinstance(data, torch.Tensor):
                context.is_object = True
                context.objects = data
                continue

            self._assert_valid_tensor(
                data, context.dtype, context.shape, must_be_contiguous=False
            )
            if data.numel() != 0:
                assert context.remote_buffer is not None
                transfers.append((data.contiguous(), context.remote_buffer))
        await self._transfer(ctx, False, transfers)

    @override
    async def _handle_storage_volume_response(
        self,
        requests: list[Request],
        transport_buffer: TransportBuffer,
    ) -> list[Any]:
        assert isinstance(transport_buffer, TorchDistributedTransportBuffer)
        return [
            (
                volume_context.objects
                if volume_context.is_object
                else client_context.tensor
            )
            for client_context, volume_context in zip(
                self._contexts, transport_buffer._contexts, strict=True
            )
        ]

    @override
    async def _post_request_success(self) -> None:
        self._succeeded = True
        self._client_cache().forget(
            self.storage_volume_ref.volume_id, self._closed_endpoint_ids
        )

    async def drop(self) -> None:
        cache = self._client_cache()
        volume_id = self.storage_volume_ref.volume_id
        if self._pending is not None:
            # The handshake failed; the volume may have created its side.
            cache.retire(volume_id, self._pending)
            self._pending = None

        endpoint, succeeded = self._endpoint, self._succeeded
        self._endpoint, self._succeeded = None, False
        self._contexts = []
        if endpoint is None:
            return
        endpoint.unregister(self._keys)
        self._keys = []
        endpoint.release()
        if not succeeded:
            # The transport may be broken or the volume may have lost its
            # side; reconnect on the next request.
            cache.retire(volume_id, endpoint)
