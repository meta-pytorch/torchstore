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
failed or cancelled request's tensors are held until a later request from the
same actor to that volume succeeds, so don't resize them (``resize_``,
``set_``) before then. This relies on the volume handling an actor's requests
one at a time, in order. Clearing the cache closes an endpoint once its
requests finish. Endpoints are not thread-safe; use a ``TransportContext``
from one event loop.

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


class _Endpoint:
    """A transport and its refcounted memory registrations.

    Registrations hold their tensors. Requests unregister their memory when
    they finish; concurrent requests sharing a tensor share one registration.
    A failed request's registrations are held until a later request from the
    same actor to the same volume succeeds, or the endpoint closes.
    """

    def __init__(self, transport: Transport) -> None:
        self.transport = transport
        # Requests (client) or transfer batches (volume) using the endpoint.
        self.active = 0
        self.retired = False
        self.closed = False
        self._registrations: dict[_RegistrationKey, tuple[Memory, int]] = {}
        # (volume_id, sender) -> [(failure, keys)] of failed requests' memory
        # the volume may still transfer.
        self.held: dict[tuple[str, str], list[tuple[int, list[_RegistrationKey]]]] = {}
        self.failures = 0

    def register(self, tensor: torch.Tensor) -> tuple[_RegistrationKey, Memory]:
        key = _view_key(tensor)
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

    def hold(self, peer: tuple[str, str], keys: list[_RegistrationKey]) -> None:
        self.failures += 1
        self.held.setdefault(peer, []).append((self.failures, keys))

    def release_held(self, peer: tuple[str, str], failures: int) -> None:
        """Unregister memory of the first ``failures`` failed requests to
        ``peer``; a later request to it succeeded."""
        held = self.held.pop(peer, [])
        self.unregister([key for f, keys in held if f <= failures for key in keys])
        if rest := [(f, keys) for f, keys in held if f > failures]:
            self.held[peer] = rest

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
        self.held.clear()
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
        self._keys: list[_RegistrationKey] = []
        # Endpoint failures before this request was sent.
        self._failures = 0
        self._succeeded = False

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["storage_volume_ref"] = None
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
        self._client_url, self._failures = cache.url, endpoint.failures
        self._synchronize_streams()
        endpoint.active += 1
        self._endpoint = endpoint
        self._expose_contexts(endpoint)

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
    async def _post_request_success(self) -> None:
        self._succeeded = True

    async def drop(self) -> None:
        endpoint, succeeded, keys = self._endpoint, self._succeeded, self._keys
        self._endpoint, self._succeeded, self._keys = None, False, []
        self._contexts = []
        if endpoint is None:
            return
        # Monarch orders messages per sending actor only.
        peer = (self.storage_volume_ref.volume_id, _sender())
        if succeeded:
            endpoint.unregister(keys)
            endpoint.release_held(peer, self._failures)
        else:
            # The volume may still be transferring; keep the memory registered,
            # which holds the tensors.
            endpoint.hold(peer, keys)
        endpoint.release()
