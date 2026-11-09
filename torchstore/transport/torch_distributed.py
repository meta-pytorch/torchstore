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
cancelled request keeps its tensors alive until the volume answers. The volume
may still read or overwrite them until then, so don't use or resize
(``resize_``) them before it answers. Meanwhile registering memory overlapping a
cancelled get's destinations through the same ``TransportContext`` raises until
it is cleared; other contexts are not checked. If the volume call itself is
cancelled (e.g. at event loop shutdown), the tensors are leaked, overlapping
registrations raise until the context is cleared and the endpoint stays open
with all its registrations.
Clearing the cache closes an endpoint once its requests finish. Endpoints are
not thread-safe; use a ``TransportContext`` from one event loop.

Limitations:
- Memory stays registered (pinned) until every storage using it is freed, so
  only new tensors pay registration, e.g. get results without a destination.
  Freed memory stays registered until the next request or delete through
  the same ``TransportContext``.
- Do not move a registered storage's memory (``resize_``, ``share_memory_``):
  its old memory stays registered (pinned) until the storage is freed or the
  ``TransportContext`` is cleared.
- Volumes stay connected to every client endpoint they served until reset.
"""

from __future__ import annotations

import asyncio
import logging
import os
import weakref
from collections import Counter
from collections.abc import Awaitable
from dataclasses import dataclass
from functools import cache
from typing import Any, TYPE_CHECKING, TypeVar

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
_T = TypeVar("_T")

# Endpoints and storages of cancelled volume calls; the volume may still access
# them.
_leaked: list[tuple[_Endpoint, list[torch.UntypedStorage]]] = []


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


def _view_key(tensor: torch.Tensor) -> _RegistrationKey:
    return (tensor.data_ptr(), tensor.nbytes, str(tensor.device))


def _overlaps(a: _RegistrationKey, b: _RegistrationKey) -> bool:
    return a[2] == b[2] and a[0] < b[0] + b[1] and b[0] < a[0] + a[1]


def _alias(key: _RegistrationKey) -> torch.Tensor:
    """A flat byte tensor over a key's memory that does not keep it alive.

    Uses a private torch API.
    """
    data_ptr, nbytes, device = key
    storage = torch._C._construct_storage_from_data_pointer(
        data_ptr, torch.device(device), nbytes
    )
    return torch.empty(0, dtype=torch.uint8, device=device).set_(storage)


@dataclass
class _Registration:
    key: _RegistrationKey
    memory: Memory
    # (weakref, data_ptr) of storages using the memory. Distinct storages can
    # share memory (e.g. torch.from_numpy of one array); the pointer detects
    # storages resized to new memory.
    storages: list[tuple[weakref.ref[torch.UntypedStorage], int]]
    remote_buffer: RemoteBuffer | None = None

    def prune(self) -> bool:
        """Drop freed or moved storages; return whether any remain."""
        self.storages = [
            (ref, ptr)
            for ref, ptr in self.storages
            if (storage := ref()) is not None and storage.data_ptr() == ptr
        ]
        return bool(self.storages)


class _Endpoint:
    """A transport and its cached memory registrations.

    Registrations alias tensor memory without owning it and last until the
    storages using it are freed and swept or the endpoint closes, so tensors
    transferred repeatedly (stored values, weights updated in place) register
    once.
    """

    def __init__(self, transport: Transport) -> None:
        self.transport = transport
        # Requests (client) or transfer batches (volume) using the endpoint.
        self.active = 0
        self.retired = False
        self.closed = False
        # Also closes the transport if the endpoint is dropped uncleared.
        self._close = weakref.finalize(self, _close_transport, transport)
        # Closing at exit may wait for transfers a volume never finishes.
        self._close.atexit = False
        self._registrations: dict[_RegistrationKey, _Registration] = {}
        # Keys of freed storages, appended by weakref callbacks. Those may run
        # on any thread or inside a transport call, so sweep() unregisters
        # them later.
        self._dead: list[_RegistrationKey] = []
        # Destinations of cancelled gets the volume has not answered.
        self._abandoned: Counter[_RegistrationKey] = Counter()

    def sweep(self) -> None:
        """Unregister memory of freed storages."""
        while self._dead:
            registration = self._registrations.get(self._dead.pop())
            # Other storages, or new memory at the same key, may use it.
            if registration is not None and not registration.prune():
                self._unregister(registration)

    def register(self, tensor: torch.Tensor) -> _Registration:
        """Register a contiguous tensor's memory."""
        key = _view_key(tensor)
        if any(_overlaps(key, abandoned) for abandoned in self._abandoned):
            raise RuntimeError(
                "torch transport memory overlaps the destination of a cancelled "
                "get the storage volume has not answered; use another tensor"
            )
        storage = tensor.untyped_storage()
        registration = self._registrations.get(key)
        if registration is not None:
            if registration.prune():
                if all(ref() is not storage for ref, _ in registration.storages):
                    registration.storages.append(self._track(key, storage))
                return registration
            self._unregister(registration)

        # Registering an alias keeps the registration from holding the tensor,
        # and registers 0-dim tensors as 1-dim, which NIXL requires.
        memory = self.transport.register_memory(_alias(key))
        registration = _Registration(key, memory, [self._track(key, storage)])
        self._registrations[key] = registration
        return registration

    def _track(
        self, key: _RegistrationKey, storage: torch.UntypedStorage
    ) -> tuple[weakref.ref[torch.UntypedStorage], int]:
        dead = self._dead
        return weakref.ref(storage, lambda _: dead.append(key)), storage.data_ptr()

    def _unregister(self, registration: _Registration) -> None:
        del self._registrations[registration.key]
        try:
            self.transport.unregister_memory(registration.memory)
        except Exception:
            logger.error("Failed to unregister torch transport memory", exc_info=True)

    def expose(self, tensor: torch.Tensor) -> RemoteBuffer:
        """Register tensor and describe it for the peer."""
        registration = self.register(tensor)
        if registration.remote_buffer is None:
            registration.remote_buffer = registration.memory.to_remote_buffer()
        return registration.remote_buffer

    def release(self) -> None:
        self.active -= 1
        self.close_if_idle()

    def release_when_answered(
        self,
        keys: list[_RegistrationKey],
        storages: list[torch.UntypedStorage],
        call: asyncio.Future[Any],
    ) -> None:
        """Release a cancelled request once its volume call finishes.

        Until then the volume may still access its tensors, so ``storages``
        stay referenced and ``register`` refuses memory overlapping ``keys``.
        Takes over the request's endpoint reference and clears ``storages``
        once answered. If ``call`` is cancelled, the storages, the refused keys
        and the endpoint (with all its registrations) are kept forever.
        """
        if not call.done():
            logger.warning(
                "Request cancelled; the storage volume may still access its "
                "torch transport tensors until it answers"
            )
        self._abandoned.update(keys)

        def release(answer: asyncio.Future[Any]) -> None:
            if answer.cancelled():
                # The volume may still access them; leak the storages and keep
                # the endpoint open.
                logger.warning(
                    "Storage volume call cancelled; leaking its torch transport "
                    "tensors and endpoint with its registrations"
                )
                _leaked.append((self, storages))
                return
            answer.exception()
            storages.clear()
            self._abandoned.subtract(keys)
            self._abandoned = +self._abandoned
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
        # Closing the transport releases every registration.
        self._registrations.clear()
        self._close.detach()
        _close_transport(self.transport)

    async def transfer(
        self, read: bool, transfers: list[tuple[torch.Tensor, RemoteBuffer]]
    ) -> None:
        """Run transfers and wait until all of them finished."""
        from torch.distributed._transport import wait_all

        works = []
        self.active += 1
        try:
            for tensor, remote_buffer in transfers:
                memory = self.register(tensor).memory
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
                self.release()

            waiter.add_done_callback(done)
            try:
                await asyncio.shield(waiter)
            except asyncio.CancelledError:
                # Keep the RPC open so the client keeps its memory alive.
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


def _close_transport(transport: Transport) -> None:
    try:
        transport.close()
    except Exception:
        logger.error("Failed to close torch transport", exc_info=True)


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
        while self.volume_endpoint is None or self._volume_broken():
            if self.volume_endpoint is not None:
                logger.warning("Replacing broken torch transport endpoint")
                self.volume_endpoint.retire()
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
        endpoint = self.volume_endpoint
        if url not in self.peers:
            endpoint.transport.connect(url)
            self.peers.add(url)
        return endpoint

    def _volume_broken(self) -> bool:
        endpoint = self.volume_endpoint
        return (
            endpoint is not None
            and bool(self.peers)
            and not endpoint.transport.connected()
        )

    def sweep(self) -> None:
        """Unregister freed memory."""
        if self.endpoint is not None:
            self.endpoint.sweep()
        # A closed backend can't unregister; connect() replaces the endpoint.
        if self.volume_endpoint is not None and not self._volume_broken():
            self.volume_endpoint.sweep()

    @override
    def delete(self, keys: set[str]) -> None:
        # Deleted values may have freed registered tensors.
        self.sweep()

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
        # Storages of request tensors, kept alive while a cancelled request's
        # volume call may access them even if a tensor's data is reassigned.
        self._storages: list[torch.UntypedStorage] = []
        # Destination memory a cancelled get refuses to other requests.
        self._keys: list[_RegistrationKey] = []

    def __getstate__(self) -> dict[str, Any]:
        state = super().__getstate__()
        state["_endpoint"] = None
        state["_storages"] = []
        state["_keys"] = []
        return state

    def _client_cache(self) -> TorchTransportCache:
        return self.storage_volume_ref.transport_context.get(TorchTransportCache)

    async def _register_client_tensors(self) -> None:
        """Expose context tensors through the client endpoint."""
        cache = self._client_cache()
        cache.sweep()
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
        self._storages = [
            c.tensor.untyped_storage() for c in self._contexts if c.tensor is not None
        ]
        # Registration overlaps queued kernels; transfers must not.
        await _wait_for(events)

    def _expose_contexts(self, endpoint: _Endpoint) -> None:
        for context in self._contexts:
            tensor = context.tensor
            # Empty tensors cannot be registered and have no bytes to move.
            if tensor is None or tensor.numel() == 0:
                continue
            assert tensor.is_contiguous()
            context.remote_buffer = endpoint.expose(tensor)
            self._keys.append(_view_key(tensor))

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
            cache.sweep()
            return
        assert self._client_url is not None
        endpoint = await cache.connect(self._backend, self._client_url)
        cache.sweep()
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
    async def _call_volume_put(self, meta_requests: list[Request]) -> None:
        # A cancelled put keeps its sources alive, but unlike a get's
        # destinations they may be put again meanwhile, so don't refuse them.
        self._keys = []
        await self._shield(super()._call_volume_put(meta_requests))

    @override
    async def _call_volume_get(self, meta_requests: list[Request]) -> TransportBuffer:
        return await self._shield(super()._call_volume_get(meta_requests))

    async def _shield(self, coro: Awaitable[_T]) -> _T:
        """Await a volume call; if cancelled, keep its tensors until it answers."""
        call = asyncio.ensure_future(coro)
        try:
            return await asyncio.shield(call)
        except asyncio.CancelledError:
            # A cancelled call does not stop the volume.
            if call.done() and not call.cancelled():
                raise
            endpoint, keys, storages = self._take_endpoint()
            if endpoint is None:
                call.add_done_callback(lambda f: f.cancelled() or f.exception())
            else:
                endpoint.release_when_answered(keys, storages, call)
            raise

    def _take_endpoint(
        self,
    ) -> tuple[_Endpoint | None, list[_RegistrationKey], list[torch.UntypedStorage]]:
        endpoint, keys, storages = self._endpoint, self._keys, self._storages
        self._endpoint, self._keys, self._storages = None, [], []
        return endpoint, keys, storages

    async def drop(self) -> None:
        self._contexts = []
        endpoint, _, _ = self._take_endpoint()
        if endpoint is not None:
            endpoint.release()
