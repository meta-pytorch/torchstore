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
failed or cancelled request's storages are held until a later request from
the same actor to that volume succeeds, so don't resize them (``resize_``)
before then. This relies on the volume handling an actor's requests one at a
time, in order. Clearing the cache closes an endpoint once its requests
finish. Endpoints are not thread-safe; use a ``TransportContext`` from one
event loop.

Limitations:
- Memory stays registered (pinned) until every storage using it is freed, so
  only new tensors pay registration, e.g. get results without a destination.
  Freed memory stays registered until the endpoint's next request or delete.
- Each get stages its non-contiguous values through two CPU buffers of up to
  ``TORCHSTORE_TORCH_TRANSPORT_STAGING_BUFFER_BYTES``.
- Do not ``resize_`` registered storages; transfers may use the old memory,
  even if the storage regrows at the same address.
- Volumes stay connected to every client endpoint they served until reset.
"""

from __future__ import annotations

import asyncio
import logging
import os
import weakref
from dataclasses import dataclass
from functools import cache
from typing import Any, TYPE_CHECKING

import torch
from typing_extensions import override

from torchstore.transport.buffers import TransportBuffer, TransportCache
from torchstore.transport.staging import aligned, STAGING_ALIGNMENT, staging_pieces
from torchstore.transport.types import Request

if TYPE_CHECKING:
    from torch.distributed import Work
    from torch.distributed._transport import Memory, RemoteBuffer, Transport

    from torchstore.strategy import StorageVolumeRef
    from torchstore.transport.buffers import TransportContext

logger = logging.getLogger(__name__)

# Held storages of endpoints that failed to close; transfers may still use them.
_unclosed_held: list[Any] = []

# Name of a registered torch.distributed transport backend (e.g. "nixl").
# Setting it enables this transport.
ENV_TORCHSTORE_TORCH_TRANSPORT_BACKEND = os.environ.get(
    "TORCHSTORE_TORCH_TRANSPORT_BACKEND", ""
).lower()
# Maximum size of each of the two buffers that stage a get's non-contiguous
# tensors.
ENV_TORCHSTORE_TORCH_TRANSPORT_STAGING_BUFFER_BYTES = int(
    os.environ.get(
        "TORCHSTORE_TORCH_TRANSPORT_STAGING_BUFFER_BYTES", str(32 * 1024 * 1024)
    )
)

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
        from torch.distributed._transport import (
            available_transports,
            new_transport,
            RemoteBuffer,
        )
    except ImportError:
        return False
    # RemoteBuffer.slice marks the upstream stack adding descriptor reloads,
    # slices and multi-peer connect (pytorch/pytorch#202534-#202536); builds
    # with only part of it are unsupported.
    if not hasattr(RemoteBuffer, "slice"):
        logger.warning(
            "torch.distributed._transport lacks multi-peer connect and "
            "descriptor reloads; the torch transport needs a newer PyTorch"
        )
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


def _sender() -> str:
    from monarch.actor import context

    return str(context().actor_instance.actor_id)


def _view_key(tensor: torch.Tensor) -> _RegistrationKey:
    return (tensor.data_ptr(), tensor.nbytes, str(tensor.device))


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
    storages using it are freed or the endpoint closes, so tensors transferred
    repeatedly (stored values, weights updated in place) register once. A
    failed request's storages are held until a later request from the same
    actor to the same volume succeeds, or the endpoint closes.
    """

    def __init__(self, transport: Transport) -> None:
        self.transport = transport
        # Requests (client) or transfer batches (volume) using the endpoint.
        self.active = 0
        self.retired = False
        self.closed = False
        self._registrations: dict[_RegistrationKey, _Registration] = {}
        # Keys of freed storages, appended by weakref callbacks. Those may run
        # on any thread or inside a transport call, so sweep() unregisters
        # them later.
        self._dead: list[_RegistrationKey] = []
        # (volume_id, sender) -> [(failure, storages)] of failed requests the
        # volume may still transfer.
        self.held: dict[
            tuple[str, str], list[tuple[int, list[torch.UntypedStorage]]]
        ] = {}
        self.failures = 0

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

    def _unregister_if_current(self, registration: _Registration) -> None:
        # Closing the endpoint drops every registration.
        if self._registrations.get(registration.key) is registration:
            self._unregister(registration)

    def expose(self, tensor: torch.Tensor) -> RemoteBuffer:
        """Register tensor and describe it for the peer."""
        registration = self.register(tensor)
        if registration.remote_buffer is None:
            registration.remote_buffer = registration.memory.to_remote_buffer()
        return registration.remote_buffer

    def hold(self, peer: tuple[str, str], storages: list[torch.UntypedStorage]) -> None:
        self.failures += 1
        self.held.setdefault(peer, []).append((self.failures, storages))

    def release_held(self, peer: tuple[str, str], failures: int) -> None:
        """Drop storages of the first ``failures`` failed requests to
        ``peer``; a later request to it succeeded."""
        if held := [e for e in self.held.pop(peer, []) if e[0] > failures]:
            self.held[peer] = held
        self.sweep()

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
        # Closing the transport releases every registration.
        self._registrations.clear()
        try:
            self.transport.close()
        except Exception:
            # Transfers may still use held memory.
            logger.error("Failed to close torch transport", exc_info=True)
            _unclosed_held.append(self.held)
            return
        self.held.clear()

    async def read(self, transfers: list[tuple[torch.Tensor, RemoteBuffer]]) -> None:
        """Read into tensors and wait until all reads finished."""
        works = []
        self.active += 1
        try:
            for tensor, remote_buffer in transfers:
                view = self.register(tensor).memory.to_mutable_view()
                works.append(self.transport.read(view, remote_buffer, async_op=True))
        finally:
            await self._drain(works)

    async def write(
        self,
        transfers: list[tuple[torch.Tensor, RemoteBuffer]],
        staged: list[tuple[torch.Tensor, RemoteBuffer]],
    ) -> None:
        """Write contiguous tensors directly and non-contiguous tensors through
        two reused staging buffers, then wait until all writes finished.

        Copying every non-contiguous tensor with .contiguous() doubles peak
        memory for a large batch. Instead, pack row ranges into one buffer
        while the other buffer's transfers are in flight.
        """
        from torch.distributed._transport import wait_all

        buffer_bytes = ENV_TORCHSTORE_TORCH_TRANSPORT_STAGING_BUFFER_BYTES
        if staged and buffer_bytes < STAGING_ALIGNMENT:
            raise ValueError(
                "TORCHSTORE_TORCH_TRANSPORT_STAGING_BUFFER_BYTES must be at least "
                f"{STAGING_ALIGNMENT}, got {buffer_bytes}"
            )
        size = min(buffer_bytes, sum(aligned(t.nbytes) for t, _ in staged))
        buffers: list[tuple[torch.Tensor, _Registration] | None] = [None, None]
        direct: list[Work] = []
        works: list[list[Work]] = [[], []]
        index = 0
        offset = 0
        self.active += 1
        try:
            for tensor, remote_buffer in transfers:
                view = self.register(tensor).memory.to_view()
                direct.append(self.transport.write(view, remote_buffer, async_op=True))
            for source, remote_buffer in staged:
                remote_offset = 0
                for piece in staging_pieces(source, size):
                    nbytes = piece.nbytes
                    start = aligned(offset)
                    if buffers[index] is None or start + nbytes > size:
                        if buffers[index] is not None:
                            index ^= 1
                        # Reuse the other buffer once its transfers finished.
                        await wait_all(works[index])
                        works[index] = []
                        if buffers[index] is None:
                            buffer = torch.empty(size, dtype=torch.uint8, device="cpu")
                            buffers[index] = (buffer, self.register(buffer))
                        start = 0
                    buffer, registration = buffers[index]
                    local = buffer[start : start + nbytes]
                    local.view(piece.dtype).view(piece.shape).copy_(piece)
                    works[index].append(
                        self.transport.write(
                            registration.memory.to_view(start, nbytes),
                            remote_buffer.slice(remote_offset, nbytes),
                            async_op=True,
                        )
                    )
                    offset = start + nbytes
                    remote_offset += nbytes
        except BaseException:
            try:
                await self._drain(direct + works[0] + works[1])
            except Exception:
                pass  # _drain logged it; keep the original error.
            raise
        else:
            await self._drain(direct + works[0] + works[1])
        finally:
            # Staging buffers are freed on return.
            for entry in buffers:
                if entry is not None:
                    self._unregister_if_current(entry[1])

    async def _drain(self, works: list[Work]) -> None:
        """Wait for submitted transfers, even if the caller is cancelled, then
        release the endpoint."""
        from torch.distributed._transport import wait_all

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
            # Keep the RPC open so the client keeps its memory registered.
            while not waiter.done():
                try:
                    await asyncio.wait([waiter])
                except asyncio.CancelledError:
                    pass
            raise


class TorchTransportCache(TransportCache):
    """Process-local endpoints.

    Clients send their endpoint's URL with each request. Storage volumes
    connect their endpoint to each client's URL.
    """

    def __init__(self) -> None:
        # Client side.
        self.endpoint: _Endpoint | None = None
        self.url: bytes | None = None
        # Storage volume side.
        self.volume_endpoint: _Endpoint | None = None
        self.peers: set[bytes] = set()
        self._creating: asyncio.Future[_Endpoint] | None = None

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

        endpoint = self.volume_endpoint
        if endpoint is not None and self.peers and not endpoint.transport.connected():
            logger.warning("Replacing broken torch transport endpoint")
            endpoint.retire()
            self.volume_endpoint = None
            self.peers.clear()
        if self.volume_endpoint is None:
            if self._creating is None:
                # Creating a NIXL agent takes up to ~1.6 s.
                self._creating = asyncio.ensure_future(
                    asyncio.to_thread(lambda: _Endpoint(new_transport(backend)))
                )
            creating = self._creating
            try:
                endpoint = await asyncio.shield(creating)
            except Exception:
                if self._creating is creating:
                    self._creating = None
                raise
            if self._creating is creating:
                self._creating = None
                self.volume_endpoint = endpoint
            elif self.volume_endpoint is not endpoint:
                # clear() closes it.
                raise RuntimeError("Transport cache cleared while connecting")
        endpoint = self.volume_endpoint
        if url not in self.peers:
            endpoint.transport.connect(url)
            self.peers.add(url)
        return endpoint

    def sweep(self) -> None:
        """Unregister freed memory."""
        for endpoint in (self.endpoint, self.volume_endpoint):
            if endpoint is not None:
                endpoint.sweep()

    @override
    def delete(self, keys: set[str]) -> None:
        # Deleted values may have freed registered tensors.
        self.sweep()

    @override
    def clear(self) -> None:
        # An endpoint with transfers in flight closes once they finish. Closing
        # releases memory held for failed requests the volume may still use.
        for endpoint in (self.endpoint, self.volume_endpoint):
            if endpoint is not None:
                endpoint.retire()
        if self._creating is not None:
            self._creating.add_done_callback(
                lambda f: not f.cancelled()
                and f.exception() is None
                and f.result().close()
            )
        self.endpoint = None
        self.volume_endpoint = None
        self.url = None
        self.peers.clear()
        self._creating = None


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
        # Exposed storages; a tensor's storage may be swapped mid-request.
        self._storages: list[torch.UntypedStorage] = []
        # Endpoint failures before this request was sent.
        self._failures = 0
        self._succeeded = False

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["storage_volume_ref"] = None
        state["_endpoint"] = None
        state["_storages"] = []
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
        self._client_url, self._failures = cache.url, endpoint.failures
        self._synchronize_streams()
        endpoint.active += 1
        self._endpoint = endpoint
        self._expose_contexts(endpoint)
        self._storages = [
            c.tensor.untyped_storage() for c in self._contexts if c.tensor is not None
        ]

    def _synchronize_streams(self) -> None:
        """Wait for queued kernels on client tensors; DMA ignores CUDA streams."""
        devices = {
            context.tensor.device
            for context in self._contexts
            if context.tensor is not None and context.tensor.is_cuda
        }
        for device in devices:
            torch.cuda.current_stream(device).synchronize()

    def _expose_contexts(self, endpoint: _Endpoint) -> None:
        for context in self._contexts:
            tensor = context.tensor
            # Empty tensors cannot be registered and have no bytes to move.
            if tensor is None or tensor.numel() == 0:
                continue
            assert tensor.is_contiguous()
            context.remote_buffer = endpoint.expose(tensor)

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

    async def _volume_endpoint(self, ctx: TransportContext) -> _Endpoint:
        assert self._client_url is not None
        return await ctx.get(TorchTransportCache).connect(
            self._backend, self._client_url
        )

    @override
    async def handle_put_request(
        self,
        ctx: TransportContext,
        entries: list[tuple[Request, Any]],
    ) -> list[Any]:
        """Called by storage volume. Read from the client's registered tensors."""
        ctx.get(TorchTransportCache).sweep()
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
        if transfers:
            endpoint = await self._volume_endpoint(ctx)
            await endpoint.read(transfers)
        return results

    @override
    async def handle_get_request(
        self,
        ctx: TransportContext,
        entries: list[tuple[Request, Any]],
    ) -> None:
        """Called by storage volume. Write into the client's registered tensors."""
        ctx.get(TorchTransportCache).sweep()
        transfers = []
        staged = []
        for (_, data), context in zip(entries, self._contexts, strict=True):
            if not isinstance(data, torch.Tensor):
                context.is_object = True
                context.objects = data
                continue

            self._assert_valid_tensor(
                data, context.dtype, context.shape, must_be_contiguous=False
            )
            if data.numel() == 0:
                continue
            assert context.remote_buffer is not None
            if data.is_contiguous():
                transfers.append((data, context.remote_buffer))
            else:
                staged.append((data, context.remote_buffer))
        if transfers or staged:
            endpoint = await self._volume_endpoint(ctx)
            await endpoint.write(transfers, staged)

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

    async def drop(self) -> None:
        endpoint, succeeded, storages = self._endpoint, self._succeeded, self._storages
        self._endpoint, self._succeeded, self._storages = None, False, []
        self._contexts = []
        if endpoint is None:
            return
        # Monarch orders messages per sending actor only.
        peer = (self.storage_volume_ref.volume_id, _sender())
        if succeeded:
            endpoint.release_held(peer, self._failures)
        else:
            # The volume may still be transferring; keep the memory.
            endpoint.hold(peer, storages)
        endpoint.release()
