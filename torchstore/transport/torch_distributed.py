# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""StorageVolume transport backed by ``torch.distributed._transport``.

Clients and storage volumes keep one endpoint per ``TransportContext``. A
volume connects to a client's endpoint on the first request with tensor bytes
and performs one-sided reads (put) and writes (get) against memory the client
registered. The backend reloads descriptor metadata a peer re-registered.

Transfers cannot be cancelled, so the volume always waits for submitted
transfers to finish before returning, even when its handler is cancelled. A
cancelled get keeps its destinations registered and alive until the volume
answers. The volume may write into them until then, so don't use or resize them
(``resize_``, ``set_``) before it answers. Meanwhile registering memory overlapping them
through the same ``TransportContext`` raises until it is cleared; other
contexts are not checked. If the volume call itself is cancelled (e.g. at event
loop shutdown), they are leaked, overlapping registrations raise until the
context is cleared and the endpoint stays open. A cancelled put does
not wait, so don't modify its tensors until the volume could have read them;
their registrations and staging copies may be released while the volume reads
them, partly overwriting the stored value. Clearing the cache closes an endpoint
once its requests finish. Endpoints are not thread-safe; use a
``TransportContext`` from one event loop.

Limitations:
- Both sides register and unregister every tensor per request (~1.8 ms each
  with NIXL over UCX), so large state dicts pay seconds per transfer. A
  registration cache is a follow-up.
- Volumes stay connected to every client endpoint they served until reset.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections import Counter
from dataclasses import dataclass
from functools import cache
from typing import Any, TYPE_CHECKING

import torch
from typing_extensions import override

from torchstore.transport.buffers import TransportBuffer, TransportCache
from torchstore.transport.types import Request

if TYPE_CHECKING:
    from torch.distributed._transport import Memory, RemoteBuffer, Transport

    from torchstore.strategy import StorageVolumeRef
    from torchstore.transport.buffers import TransportContext

logger = logging.getLogger(__name__)

# Name of a registered torch.distributed transport backend (e.g. "nixl").
# Setting it enables this transport.
ENV_TORCHSTORE_TORCH_TRANSPORT_BACKEND = os.environ.get(
    "TORCHSTORE_TORCH_TRANSPORT_BACKEND", ""
).lower()

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


# Destinations of cancelled volume calls; the volume may still write into them.
_leaked: list[list[torch.Tensor | None]] = []


def _view_key(tensor: torch.Tensor) -> _RegistrationKey:
    return (tensor.data_ptr(), tensor.nbytes, str(tensor.device))


def _overlaps(a: _RegistrationKey, b: _RegistrationKey) -> bool:
    return a[2] == b[2] and a[0] < b[0] + b[1] and b[0] < a[0] + a[1]


class _Endpoint:
    """A transport and its refcounted memory registrations.

    Registrations hold their tensors. Requests unregister their memory when
    they finish; concurrent requests sharing a tensor share one registration.
    """

    def __init__(self, transport: Transport) -> None:
        self.transport = transport
        # Requests (client) or transfer batches (volume) using the endpoint.
        self.active = 0
        self.retired = False
        self.closed = False
        self._registrations: dict[_RegistrationKey, tuple[Memory, int]] = {}
        # Destinations of cancelled gets the volume has not answered.
        self._abandoned: Counter[_RegistrationKey] = Counter()

    def register(self, tensor: torch.Tensor) -> tuple[_RegistrationKey, Memory]:
        key = _view_key(tensor)
        if any(_overlaps(key, abandoned) for abandoned in self._abandoned):
            raise RuntimeError(
                "torch transport memory overlaps the destination of a cancelled "
                "get the storage volume has not answered; use another tensor"
            )
        memory, count = self._registrations.get(key, (None, 0))
        if memory is None:
            # NIXL cannot register 0-dim tensors; register the same bytes as 1-dim.
            memory = self.transport.register_memory(tensor.reshape(-1))
        self._registrations[key] = (memory, count + 1)
        return key, memory

    def unregister(self, keys: list[_RegistrationKey]) -> None:
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

    def release(self) -> None:
        self.active -= 1
        self.close_if_idle()

    def release_when_answered(
        self,
        keys: list[_RegistrationKey],
        tensors: list[torch.Tensor | None],
        call: asyncio.Future[Any],
    ) -> None:
        """Release a cancelled get's destinations once its volume call finishes.

        Takes over the registrations of ``keys`` and one ``active`` reference.
        Until then the volume may still write into them, so ``register``
        refuses memory overlapping them and ``tensors`` are kept alive.
        """
        if not call.done():
            logger.warning(
                "Get cancelled; keeping %d torch transport destinations "
                "registered until the storage volume answers",
                len(keys),
            )
        self._abandoned.update(keys)

        def release(answer: asyncio.Future[Any]) -> None:
            if answer.cancelled():
                # The volume may still be writing; leak the destinations.
                logger.warning(
                    "Storage volume call cancelled; leaking %d torch transport "
                    "destinations",
                    len(keys),
                )
                _leaked.append(tensors)
                return
            answer.exception()
            self._abandoned.subtract(keys)
            self._abandoned = +self._abandoned
            self.unregister(keys)
            self.release()

        call.add_done_callback(release)

    def retire(self) -> None:
        """Close once idle; the cache no longer hands this endpoint out."""
        self.retired = True
        self.close_if_idle()

    def close_if_idle(self) -> None:
        if self.retired and not self.active:
            self.close()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
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
                # Keep the RPC open so a cancelled get keeps its destinations
                # registered.
                while not waiter.done():
                    try:
                        await asyncio.wait([waiter])
                    except asyncio.CancelledError:
                        pass
                raise


def _record_events(devices: set[torch.device]) -> list[torch.Event]:
    """Mark kernels queued on ``devices``; DMA ignores streams."""
    accelerator = torch.accelerator.current_accelerator()
    return [
        # Blocking events sleep instead of spinning in synchronize().
        torch.accelerator.current_stream(device).record_event(
            torch.Event(device, blocking=True)
        )
        for device in devices
        if accelerator is not None and device.type == accelerator.type
    ]


async def _wait_for(events: list[torch.Event]) -> None:
    """Wait for ``events`` without blocking the event loop."""
    pending = [event for event in events if not event.query()]
    if not pending:
        return

    def synchronize() -> None:
        for event in pending:
            event.synchronize()

    await asyncio.to_thread(synchronize)


def _close_created(creating: asyncio.Future[_Endpoint]) -> None:
    if creating.cancelled() or creating.exception() is not None:
        return
    creating.result().close()


class TorchTransportCache(TransportCache):
    """A client and a storage volume endpoint per ``TransportContext``.

    Clients send their endpoint's URL with each request carrying tensor bytes.
    Storage volumes connect their endpoint to each client's URL.
    """

    def __init__(self) -> None:
        # Client side.
        self.endpoint: _Endpoint | None = None
        self.url: bytes | None = None
        # Storage volume side.
        self.volume_endpoint: _Endpoint | None = None
        self.peers: set[bytes] = set()
        self._creating: asyncio.Future[_Endpoint] | None = None
        self._clears = 0

    def client_endpoint(self, backend: str) -> _Endpoint:
        from torch.distributed._transport import new_transport

        if self.endpoint is None:
            endpoint = _Endpoint(new_transport(backend))
            try:
                self.url = endpoint.transport.bind()
            except BaseException:
                endpoint.close()
                raise
            self.endpoint = endpoint
        return self.endpoint

    async def connect(self, backend: str, url: bytes) -> _Endpoint:
        """The storage volume's endpoint, connected to a client's URL.

        Creates the endpoint once, off the actor loop, and replaces it once
        the backend closes it (e.g. after a failed descriptor load).
        """
        from torch.distributed._transport import new_transport

        # Awaiting creation lets other requests break the endpoint; recheck.
        while (endpoint := self.volume_endpoint) is None or (
            self.peers and not endpoint.transport.connected()
        ):
            if endpoint is not None:
                logger.warning("Replacing broken torch transport endpoint")
                endpoint.retire()
                self.volume_endpoint = None
                self.peers.clear()
            if self._creating is None:
                # Creating a NIXL agent takes up to ~1.6 s.
                self._creating = asyncio.ensure_future(
                    asyncio.to_thread(lambda: _Endpoint(new_transport(backend)))
                )
            creating = self._creating
            clears = self._clears
            try:
                endpoint = await asyncio.shield(creating)
            except Exception:
                if self._creating is creating:
                    self._creating = None
                raise
            if self._creating is creating:
                self._creating = None
                self.volume_endpoint = endpoint
            elif self._clears != clears:
                # clear() closes it.
                raise RuntimeError("Transport cache cleared while connecting")
        if url not in self.peers:
            endpoint.transport.connect(url)
            self.peers.add(url)
        return endpoint

    @override
    def clear(self) -> None:
        # An endpoint with requests in flight closes once they finish.
        for endpoint in (self.endpoint, self.volume_endpoint):
            if endpoint is not None:
                endpoint.retire()
        if self._creating is not None:
            self._creating.add_done_callback(_close_created)
        self.endpoint = None
        self.volume_endpoint = None
        self.url = None
        self.peers.clear()
        self._creating = None
        self._clears += 1


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
        self._client_url: bytes | None = None
        self._contexts: list[TorchTransportRequestContext] = []
        # Client-local state, stripped from RPC state.
        self._endpoint: _Endpoint | None = None
        self._keys: list[_RegistrationKey] = []

    def __getstate__(self) -> dict[str, Any]:
        state = super().__getstate__()
        state["_endpoint"] = None
        state["_keys"] = []
        return state

    def _client_cache(self) -> TorchTransportCache:
        return self.storage_volume_ref.transport_context.get(TorchTransportCache)

    async def _register_client_tensors(self) -> None:
        """Expose context tensors through the client endpoint."""
        cache = self._client_cache()
        # Object-only requests need no endpoint.
        if not any(c.tensor is not None and c.tensor.numel() for c in self._contexts):
            return
        endpoint = cache.client_endpoint(self._backend)
        self._client_url = cache.url
        events = _record_events(
            {c.tensor.device for c in self._contexts if c.tensor is not None}
        )
        endpoint.active += 1
        self._endpoint = endpoint
        self._expose_contexts(endpoint)
        # Registration overlaps queued kernels; transfers must not.
        await _wait_for(events)

    def _expose_contexts(self, endpoint: _Endpoint) -> None:
        for context in self._contexts:
            tensor = context.tensor
            # Empty tensors cannot be registered and have no bytes to move.
            if tensor is None or tensor.numel() == 0:
                continue
            assert tensor.is_contiguous()
            key, memory = endpoint.register(tensor)
            self._keys.append(key)
            context.remote_buffer = memory.to_remote_buffer()

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
        cache = ctx.get(TorchTransportCache)
        if not transfers:
            return
        assert self._client_url is not None
        endpoint = await cache.connect(self._backend, self._client_url)
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
    async def _call_volume_get(self, meta_requests: list[Request]) -> TransportBuffer:
        if self._endpoint is None:
            return await super()._call_volume_get(meta_requests)
        call = asyncio.ensure_future(super()._call_volume_get(meta_requests))
        try:
            return await asyncio.shield(call)
        except asyncio.CancelledError:
            # A cancelled call does not stop the volume.
            if call.done() and not call.cancelled():
                raise
            endpoint, keys = self._take_endpoint()
            assert endpoint is not None
            # The volume may still write into the destinations; keep them
            # registered, which holds them, until it answers.
            tensors = [context.tensor for context in self._contexts]
            endpoint.release_when_answered(keys, tensors, call)
            raise

    def _take_endpoint(self) -> tuple[_Endpoint | None, list[_RegistrationKey]]:
        endpoint, keys = self._endpoint, self._keys
        self._endpoint, self._keys = None, []
        return endpoint, keys

    async def drop(self) -> None:
        self._contexts = []
        endpoint, keys = self._take_endpoint()
        if endpoint is not None:
            endpoint.unregister(keys)
            endpoint.release()
