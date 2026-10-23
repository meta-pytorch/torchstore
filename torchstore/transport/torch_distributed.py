# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""StorageVolume transport backed by ``torch.distributed._transport``.

Each client process keeps one connected endpoint per storage volume. The
volume performs one-sided reads (put) and writes (get) against memory the
client registered for the request.

Each NIXL-style endpoint runs its own agent; with UCX each agent's progress
thread busy-polls a core even when idle.

Transports cannot cancel submitted DMA. When a request fails, the client
sends an abort through the handshake RPC before releasing its memory. The
volume refuses to start the request afterwards, waits for transfers already
submitted (up to the client's timeout, twice: once for the request and once
while draining), and reports whether its peer survived. If transfers never
finish, the abort fails, or the volume was reset while transfers ran, both
sides leak the request's registrations.

Tensors of a failed request may hold partial data, and DMA of a leaked get
may still modify its destination tensors later. A put stores into a new
tensor when the stored one may still be used by stale DMA.

The storage volume handles one message at a time, so an abort waiting for
transfers delays its other requests.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import threading
import uuid
from collections import OrderedDict
from collections.abc import Callable, Coroutine
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
_TIMEOUT_ENV = "TORCHSTORE_TORCH_TRANSPORT_TIMEOUT_S"

_RegistrationKey = tuple[int, int, str]
# (client_id, endpoint_id)
_Peer = tuple[str, str]

# Outcomes of aborting a failed request on the storage volume.
_KEPT = "kept"  # No transfer in flight; the peer is still usable.
_DROPPED = "dropped"  # Transfers finished; the volume dropped the peer.
_LEAKED = "leaked"  # Transfers may still run; memory must stay registered.

# Closed endpoint IDs a storage volume remembers, so a handshake that
# finishes after its client abandoned it does not leave a peer behind.
_MAX_CLOSED_PEERS = 4096
# Aborted request IDs a storage volume endpoint remembers. Requests aborted
# before they reached the volume are removed only if they arrive later.
_MAX_ABORTED_REQUESTS = 4096
# Requests whose transfers may still run when a storage volume was reset.
_MAX_LEAKED_REQUESTS = 4096
# Endpoint (re)connections per request before giving up on descriptor
# conflicts.
_MAX_ENDPOINT_ATTEMPTS = 4
# Descriptors a client endpoint exposes before it is replaced, bounding the
# metadata both sides cache.
_MAX_EXPOSED = 16384

# State that outlives a TransportContext reset, since DMA cannot be cancelled:
# tasks waiting for transfers, endpoints still registering memory transfers may
# use, tensors of leaked requests, leaked (client_id, endpoint_id, request_id)
# keys and closed storage volume peers.
_background: set[asyncio.Task[None]] = set()
_orphans: list[_Endpoint] = []
_leaked: list[list[TorchTransportRequestContext]] = []
_leaked_requests: OrderedDict[tuple[str, str, str], None] = OrderedDict()
_closed_peers: OrderedDict[_Peer, None] = OrderedDict()


def _remember(entries: OrderedDict[Any, None], key: Any, limit: int) -> None:
    entries[key] = None
    entries.move_to_end(key)
    while len(entries) > limit:
        entries.popitem(last=False)


def _prune_orphans() -> None:
    for endpoint in _orphans:
        # Retry endpoints that failed to close.
        endpoint.close_if_idle()
    _orphans[:] = [endpoint for endpoint in _orphans if not endpoint.closed]


def _all_completed(works: list[Any]) -> bool:
    # wait_all stops waiting for works whose future couldn't be obtained.
    try:
        return all(work.is_completed() for work in works)
    except Exception:
        logger.error("Failed to check torch transport transfers", exc_info=True)
        return False


def _run_in_background(coro: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
    task = asyncio.get_running_loop().create_task(coro)
    _background.add(task)
    task.add_done_callback(_background.discard)
    return task


def _transfer_timeout() -> float:
    value = os.environ.get(_TIMEOUT_ENV, "60")
    try:
        timeout = float(value)
    except ValueError:
        timeout = math.nan
    if not 0 < timeout < math.inf:
        raise ValueError(f"{_TIMEOUT_ENV} must be positive and finite, got {value!r}")
    return timeout


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
        _transfer_timeout()
    except ValueError:
        logger.error("torch.distributed transport disabled", exc_info=True)
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


def _resolve(outcome: asyncio.Future[str], result: str) -> None:
    # clear() resolves outstanding outcomes as leaked.
    if not outcome.done():
        outcome.set_result(result)


def _close(transport: Transport) -> None:
    try:
        transport.close()
    except Exception:
        logger.error("Failed to close torch transport", exc_info=True)


def _bind(backend: str) -> tuple[Transport, bytes]:
    from torch.distributed._transport import new_transport

    transport = new_transport(backend)
    try:
        return transport, transport.bind()
    except BaseException:
        _close(transport)
        raise


def _bind_and_connect(backend: str, peer_url: bytes) -> tuple[Transport, bytes]:
    transport, url = _bind(backend)
    try:
        transport.connect(peer_url)
    except BaseException:
        _close(transport)
        raise
    return transport, url


def _connect(transport: Transport, peer_url: bytes) -> tuple[Transport, bytes]:
    try:
        transport.connect(peer_url)
    except BaseException:
        _close(transport)
        raise
    return transport, peer_url


async def _setup_off_loop(
    fn: Callable[..., tuple[Transport, bytes]], *args: Any
) -> tuple[Transport, bytes]:
    """Run blocking transport setup in a thread.

    The thread keeps running if the caller is cancelled and closes the
    transport it creates, even if the event loop stopped.
    """
    lock = threading.Lock()
    abandoned = False

    def run() -> tuple[Transport, bytes] | None:
        transport, url = fn(*args)
        with lock:
            if not abandoned:
                return transport, url
        _close(transport)
        return None

    future = asyncio.ensure_future(asyncio.to_thread(run))
    try:
        result = await asyncio.shield(future)
    except asyncio.CancelledError:
        with lock:
            abandoned = True
        # run() may have returned before the flag was set.
        future.add_done_callback(_close_setup_result)
        raise
    assert result is not None
    return result


def _close_setup_result(
    future: asyncio.Future[tuple[Transport, bytes] | None],
) -> None:
    if future.cancelled() or future.exception() is not None:
        return
    result = future.result()
    if result is not None:
        _close(result[0])


class _Endpoint:
    """A connected transport and its refcounted memory registrations.

    Backends deduplicate registrations of the same byte range and unregistering
    any alias invalidates all of them, so concurrent requests sharing a tensor
    must not unregister it while another request still uses it.
    """

    def __init__(self, transport: Transport, endpoint_id: str) -> None:
        self.transport = transport
        self.id = endpoint_id
        self.retired = False
        self.closed = False
        self._registrations: dict[_RegistrationKey, tuple[Memory, int]] = {}
        # Descriptors exposed to the peer. Peers may cache metadata per
        # descriptor (NIXL rejects reloading a descriptor with new metadata),
        # so a range re-registered with different metadata, e.g. memory freed
        # and reallocated at the same address, needs a new endpoint.
        self._exposed: dict[_RegistrationKey, bytes] = {}
        # Storage volume side: requests aborted before they started.
        self._aborted: OrderedDict[str, None] = OrderedDict()

    def register(self, tensor: torch.Tensor) -> tuple[_RegistrationKey, Memory]:
        key = (
            tensor.data_ptr(),
            tensor.numel() * tensor.element_size(),
            str(tensor.device),
        )
        entry = self._registrations.get(key)
        if entry is None:
            entry = (self.transport.register_memory(tensor), 0)
        memory, count = entry
        self._registrations[key] = (memory, count + 1)
        return key, memory

    def unregister(self, key: _RegistrationKey) -> None:
        memory, count = self._registrations[key]
        if count > 1:
            self._registrations[key] = (memory, count - 1)
            return
        del self._registrations[key]
        if self.closed:
            return
        try:
            self.transport.unregister_memory(memory)
        finally:
            self.close_if_idle()

    def unregister_all(self, keys: list[_RegistrationKey]) -> None:
        """Unregister each key, logging failures so the rest are released."""
        for key in keys:
            try:
                self.unregister(key)
            except Exception:
                logger.error(
                    "Failed to unregister torch transport memory", exc_info=True
                )

    def expose(self, key: _RegistrationKey, remote_buffer: RemoteBuffer) -> bool:
        """Record a descriptor sent to the peer; False if it conflicts."""
        data = remote_buffer.serialize()
        return self._exposed.setdefault(key, data) == data

    def exposed_full(self) -> bool:
        return len(self._exposed) >= _MAX_EXPOSED

    def retire(self) -> None:
        """Stop handing out this endpoint; close once in-flight requests finish."""
        self.retired = True
        self.close_if_idle()

    def abort(self, request_id: str) -> None:
        """Refuse a request that has not started on the storage volume."""
        _remember(self._aborted, request_id, _MAX_ABORTED_REQUESTS)

    def take_aborted(self, request_id: str) -> bool:
        return self._aborted.pop(request_id, False) is None

    def overlaps(self, tensor: torch.Tensor) -> bool:
        """Whether a registration of this endpoint overlaps ``tensor``."""
        start = tensor.data_ptr()
        end = start + tensor.numel() * tensor.element_size()
        device = str(tensor.device)
        return any(
            key_device == device and ptr < end and start < ptr + nbytes
            for ptr, nbytes, key_device in self._registrations
        )

    def close_if_idle(self) -> None:
        if self.retired and not self._registrations:
            self.close()

    def close(self) -> None:
        """Close the transport; stays open (and unreported) if closing fails."""
        if self.closed:
            return
        try:
            self.transport.close()
        except Exception:
            logger.error("Failed to close torch transport", exc_info=True)
            return
        self.closed = True


class TorchTransportCache(TransportCache):
    """Process-local endpoints.

    Clients key endpoints by storage volume ID and keep retired endpoints
    until that volume learns they closed. Storage volumes key endpoints by
    ``(client_id, endpoint_id)`` and keep each until its client reports it
    closed, so requests sent before a client reconnects still find their
    peer.
    """

    def __init__(self) -> None:
        self.client_id = uuid.uuid4().hex
        # Client side.
        self.endpoints: dict[str, _Endpoint] = {}
        # (volume_id, endpoint)
        self.retired: list[tuple[str, _Endpoint]] = []
        # (volume_id, endpoint_id) of endpoints whose handshake failed after
        # the volume may have created its side.
        self.abandoned: list[tuple[str, str]] = []
        self._locks: dict[str, asyncio.Lock] = {}
        # Storage volume side.
        self.peers: dict[_Peer, _Endpoint] = {}
        self._quarantined: list[_Endpoint] = []
        # Outcomes of requests whose transfers started, until they succeed or
        # their client aborts them.
        self._outcomes: dict[tuple[str, str, str], asyncio.Future[str]] = {}

    def lock(self, key: str) -> asyncio.Lock:
        return self._locks.setdefault(key, asyncio.Lock())

    def put(self, key: str, endpoint: _Endpoint) -> None:
        previous = self.endpoints.get(key)
        if previous is not None:
            self.retire(key, previous)
        self.endpoints[key] = endpoint

    def retire(self, key: str, endpoint: _Endpoint) -> None:
        """Retire a client endpoint; it closes once its requests finish."""
        if self.endpoints.get(key) is endpoint:
            del self.endpoints[key]
        endpoint.retire()
        if (key, endpoint) not in self.retired:
            self.retired.append((key, endpoint))

    def abandon(self, key: str, endpoint_id: str) -> None:
        self.abandoned.append((key, endpoint_id))

    def closed_endpoint_ids(self, key: str) -> list[str]:
        """Endpoint IDs to report to storage volume ``key``."""
        _prune_orphans()
        endpoint_ids = [i for volume_id, i in self.abandoned if volume_id == key]
        for volume_id, endpoint in self.retired:
            if volume_id != key:
                continue
            # Retry endpoints that failed to close.
            endpoint.close_if_idle()
            if endpoint.closed:
                endpoint_ids.append(endpoint.id)
        return endpoint_ids

    def forget(self, key: str, endpoint_ids: list[str]) -> None:
        """Drop endpoints storage volume ``key`` has closed."""
        self.retired = [
            (v, e) for v, e in self.retired if v != key or e.id not in endpoint_ids
        ]
        self.abandoned = [
            (v, i) for v, i in self.abandoned if v != key or i not in endpoint_ids
        ]

    async def run_abort(self, abort: Coroutine[Any, Any, None]) -> None:
        """Run an abort to completion even if the caller is cancelled."""
        await asyncio.shield(_run_in_background(abort))

    def leak(
        self,
        key: str,
        endpoint: _Endpoint,
        contexts: list[TorchTransportRequestContext],
    ) -> None:
        """Keep a failed request's memory for DMA that may still run."""
        _leaked.append(contexts)
        self.retire(key, endpoint)
        # The endpoint may never close; stop rescanning it for reports.
        self.retired = [(v, e) for v, e in self.retired if e is not endpoint]
        if endpoint not in _orphans:
            _orphans.append(endpoint)

    def in_use_by_stale_transfer(self, tensor: torch.Tensor) -> bool:
        """Whether DMA of a timed-out request may still use ``tensor``."""
        return any(e.overlaps(tensor) for e in (*self._quarantined, *_orphans))

    def is_closed(self, peer: _Peer) -> bool:
        return peer in _closed_peers

    def close_peers(self, client_id: str, endpoint_ids: list[str]) -> None:
        """Retire volume endpoints whose client side closed."""
        for endpoint_id in endpoint_ids:
            peer = (client_id, endpoint_id)
            _remember(_closed_peers, peer, _MAX_CLOSED_PEERS)
            endpoint = self.peers.pop(peer, None)
            if endpoint is not None:
                endpoint.retire()
                self._quarantined.append(endpoint)
        if endpoint_ids:
            closed = set(endpoint_ids)
            self._outcomes = {
                key: outcome
                for key, outcome in self._outcomes.items()
                if not (key[0] == client_id and key[1] in closed and outcome.done())
            }
        for endpoint in self._quarantined:
            # Retry endpoints that failed to close.
            endpoint.close_if_idle()
        self._quarantined = [e for e in self._quarantined if not e.closed]
        _prune_orphans()

    def start(self, peer: _Peer, request_id: str) -> asyncio.Future[str]:
        """Track a request whose transfers are about to start."""
        outcome = asyncio.get_running_loop().create_future()
        self._outcomes[(*peer, request_id)] = outcome
        return outcome

    def finish(
        self, peer: _Peer, request_id: str, outcome: asyncio.Future[str]
    ) -> None:
        """Record a request whose transfers succeeded."""
        _resolve(outcome, _KEPT)
        key = (*peer, request_id)
        _leaked_requests.pop(key, None)
        if self._outcomes.get(key) is outcome:
            del self._outcomes[key]

    async def abort(self, peer: _Peer, request_id: str) -> str:
        """Stop a failed request and wait until its transfers stopped."""
        key = (*peer, request_id)
        outcome = self._outcomes.get(key)
        if outcome is not None:
            result = await asyncio.shield(outcome)
            self._outcomes.pop(key, None)
            return result
        if key in _leaked_requests:
            # Started before a reset; its transfers may still run.
            return _LEAKED
        endpoint = self.peers.get(peer)
        if endpoint is None:
            return _DROPPED
        # The request has not started; refuse it if it arrives later.
        endpoint.abort(request_id)
        return _KEPT

    def drop_peer(self, peer: _Peer, endpoint: _Endpoint) -> None:
        if self.peers.get(peer) is endpoint:
            del self.peers[peer]
        endpoint.retire()
        if endpoint not in self._quarantined:
            self._quarantined.append(endpoint)

    def quarantine(
        self,
        peer: _Peer,
        request_id: str,
        endpoint: _Endpoint,
        works: list[Any],
        registrations: list[_RegistrationKey],
        outcome: asyncio.Future[str],
        timeout: float,
    ) -> None:
        """Drop a peer whose transfers timed out or were cancelled.

        DMA may still use the registered memory, so registrations are released
        once the transfers finish, or leaked if they do not within a timeout.
        """
        from torch.distributed._transport import wait_all

        self.drop_peer(peer, endpoint)

        async def drain() -> None:
            try:
                await wait_all(works, timeout=timeout)
            except TimeoutError:
                pass
            except Exception:
                logger.warning("torch transport transfer failed", exc_info=True)
            except BaseException:
                # Cancelled, e.g. at shutdown.
                _resolve(outcome, _LEAKED)
                raise
            if not _all_completed(works):
                logger.error(
                    "torch transport transfers did not finish; leaking %d "
                    "registrations",
                    len(registrations),
                )
                _resolve(outcome, _LEAKED)
                return
            _leaked_requests.pop((*peer, request_id), None)
            endpoint.unregister_all(registrations)
            _resolve(outcome, _DROPPED)

        _run_in_background(drain())

    @override
    def clear(self) -> None:
        # Aborts and drains keep running: they release memory once transfers
        # finish.
        for key, outcome in self._outcomes.items():
            if not outcome.done() or outcome.result() == _LEAKED:
                _remember(_leaked_requests, key, _MAX_LEAKED_REQUESTS)
            _resolve(outcome, _LEAKED)
        endpoints = [
            *_orphans,
            *self.endpoints.values(),
            *(endpoint for _, endpoint in self.retired),
            *self.peers.values(),
            *self._quarantined,
        ]
        for endpoint in endpoints:
            # Endpoints with registrations close once transfers release them.
            endpoint.retire()
        _orphans[:] = [endpoint for endpoint in endpoints if not endpoint.closed]
        self.endpoints.clear()
        self.retired.clear()
        self.abandoned.clear()
        self._locks.clear()
        self.peers.clear()
        self._quarantined.clear()
        self._outcomes.clear()


@dataclass
class TorchTransportRequestContext:
    """Serializable state for one tensor or object in a request batch."""

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
        # The volume uses the client's timeout, so both agree on how long
        # DMA for a failed request may run.
        self._timeout = _transfer_timeout()
        self._client_id: str | None = None
        self._client_url: bytes | None = None
        self._endpoint_id: str | None = None
        self._request_id = uuid.uuid4().hex
        # Set when this buffer is sent to abort a failed request.
        self._abort = False
        # Client endpoints that closed; the volume closes its side.
        self._closed_endpoint_ids: list[str] = []
        self._contexts: list[TorchTransportRequestContext] = []
        # Client-local state, stripped from RPC state.
        self._pending_transport: Transport | None = None
        self._endpoint: _Endpoint | None = None
        self._registrations: list[_RegistrationKey] = []
        # Set once registration finished; the request may reach the volume
        # afterwards, so a failure needs an abort.
        self._sent = False
        self._succeeded = False

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["storage_volume_ref"] = None
        state["_pending_transport"] = None
        state["_endpoint"] = None
        state["_registrations"] = []
        return state

    def _client_cache(self) -> TorchTransportCache:
        return self.storage_volume_ref.transport_context.get(TorchTransportCache)

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
        cache = self._client_cache()
        # Set before the transport exists so drop() never abandons a live ID.
        self._endpoint_id = uuid.uuid4().hex
        self._pending_transport, self._client_url = await _setup_off_loop(
            _bind, self._backend
        )
        self._client_id = cache.client_id
        self._closed_endpoint_ids = cache.closed_endpoint_ids(
            self.storage_volume_ref.volume_id
        )

    @override
    async def recv_handshake(
        self,
        ctx: TransportContext,
        entries: list[tuple[Request, Any]],
    ) -> list[Any]:
        assert self._client_id is not None and self._endpoint_id is not None
        peer = (self._client_id, self._endpoint_id)
        if self._abort:
            return [await ctx.get(TorchTransportCache).abort(peer, self._request_id)]

        assert self._client_url is not None
        cache = ctx.get(TorchTransportCache)
        transport, url = await _setup_off_loop(
            _bind_and_connect, self._backend, self._client_url
        )
        if ctx.get(TorchTransportCache) is not cache:
            # The reset cleared the cache this peer would be added to.
            _close(transport)
            raise RuntimeError("torch transport was reset during the handshake")
        cache.close_peers(self._client_id, self._closed_endpoint_ids)
        if cache.is_closed(peer):
            _close(transport)
            raise RuntimeError("torch transport handshake was abandoned")
        cache.peers[peer] = _Endpoint(transport, self._endpoint_id)
        return [url]

    @override
    async def _post_handshake(
        self,
        handshake_results: list[Any],
        requests: list[Request],
    ) -> None:
        transport = self._pending_transport
        assert transport is not None and self._endpoint_id is not None
        (volume_url,) = handshake_results
        # _connect owns the transport now and closes it on failure.
        self._pending_transport = None
        try:
            await _setup_off_loop(_connect, transport, volume_url)
        except BaseException:
            # The volume created its side.
            self._client_cache().abandon(
                self.storage_volume_ref.volume_id, self._endpoint_id
            )
            raise
        if self._client_cache().client_id != self._client_id:
            # The volume keyed this peer by the client ID of the reset cache.
            _close(transport)
            raise RuntimeError("torch transport was reset during the handshake")
        self._client_cache().put(
            self.storage_volume_ref.volume_id, _Endpoint(transport, self._endpoint_id)
        )

    async def _register_client_tensors(self) -> None:
        """Register context tensors with the current endpoint."""
        volume_id = self.storage_volume_ref.volume_id
        self._synchronize_streams()
        for _ in range(_MAX_ENDPOINT_ATTEMPTS):
            # The client cache may have been reset during a handshake.
            cache = self._client_cache()
            endpoint = cache.endpoints.get(volume_id)
            if endpoint is None:
                # A failed request retired the endpoint after our handshake.
                await self.perform_handshake([], [])
                continue
            if endpoint.exposed_full():
                await self._replace_endpoint(endpoint)
                continue
            self._endpoint = endpoint
            self._client_id = cache.client_id
            self._endpoint_id = endpoint.id
            self._closed_endpoint_ids = cache.closed_endpoint_ids(volume_id)
            try:
                exposed = self._expose_contexts(endpoint)
            except Exception:
                # The transport may be broken.
                cache.retire(volume_id, endpoint)
                raise
            if exposed:
                self._sent = True
                return
            await self._replace_endpoint(endpoint)
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
        for context in self._contexts:
            tensor = context.tensor
            # Empty tensors cannot be registered and have no bytes to move.
            if tensor is None or tensor.numel() == 0:
                continue
            assert tensor.is_contiguous()
            key, memory = endpoint.register(tensor)
            self._registrations.append(key)
            context.remote_buffer = memory.to_remote_buffer()
            if not endpoint.expose(key, context.remote_buffer):
                self._release_registrations()
                return False
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
            # Requests still using the old endpoint keep its volume side.
            cache.retire(volume_id, endpoint)
            await super().perform_handshake([], [])

    def _release_registrations(self) -> None:
        if self._registrations:
            assert self._endpoint is not None
            self._endpoint.unregister_all(self._registrations)
        self._registrations = []
        for context in self._contexts:
            context.remote_buffer = None

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
        prepare: Callable[[], list[tuple[torch.Tensor, RemoteBuffer]]],
    ) -> None:
        """Validate the request with ``prepare`` and run its transfers."""
        assert self._client_id is not None and self._endpoint_id is not None
        cache = ctx.get(TorchTransportCache)
        cache.close_peers(self._client_id, self._closed_endpoint_ids)
        if all(context.remote_buffer is None for context in self._contexts):
            # Nothing to move; the client does not abort such requests.
            prepare()
            return

        peer = (self._client_id, self._endpoint_id)
        endpoint = cache.peers.get(peer)
        if endpoint is None:
            raise RuntimeError(
                f"No torch transport endpoint for client {self._client_id}"
            )
        if endpoint.take_aborted(self._request_id):
            raise RuntimeError("torch transport request was aborted by its client")

        # Track the request before validating it, so its client's abort
        # finds the outcome instead of marking the request aborted forever.
        outcome = cache.start(peer, self._request_id)
        try:
            transfers = prepare()
        except BaseException:
            # Nothing was submitted.
            _resolve(outcome, _KEPT)
            raise
        registrations: list[_RegistrationKey] = []
        works: list[Any] = []

        def quarantine() -> None:
            cache.quarantine(
                peer,
                self._request_id,
                endpoint,
                works,
                registrations,
                outcome,
                self._timeout,
            )

        try:
            await self._submit_and_wait(endpoint, read, transfers, registrations, works)
        except (TimeoutError, asyncio.CancelledError):
            quarantine()
            raise
        except Exception:
            if not _all_completed(works):
                quarantine()
                raise
            # Submitted transfers finished, but the connection may be broken.
            _leaked_requests.pop((*peer, self._request_id), None)
            cache.drop_peer(peer, endpoint)
            endpoint.unregister_all(registrations)
            _resolve(outcome, _DROPPED)
            raise
        except BaseException:
            # Transfers may still run; keep the memory quarantined.
            cache.drop_peer(peer, endpoint)
            _resolve(outcome, _LEAKED)
            raise
        endpoint.unregister_all(registrations)
        cache.finish(peer, self._request_id, outcome)

    async def _submit_and_wait(
        self,
        endpoint: _Endpoint,
        read: bool,
        transfers: list[tuple[torch.Tensor, RemoteBuffer]],
        registrations: list[_RegistrationKey],
        works: list[Any],
    ) -> None:
        from torch.distributed._transport import wait_all

        transport = endpoint.transport
        try:
            for tensor, remote_buffer in transfers:
                key, memory = endpoint.register(tensor)
                registrations.append(key)
                if read:
                    work = transport.read(
                        memory.to_mutable_view(), remote_buffer, async_op=True
                    )
                else:
                    work = transport.write(
                        memory.to_view(), remote_buffer, async_op=True
                    )
                works.append(work)
        finally:
            # Drain submitted transfers even if a later submission failed.
            await wait_all(works, timeout=self._timeout)

    @override
    async def handle_put_request(
        self,
        ctx: TransportContext,
        entries: list[tuple[Request, Any]],
    ) -> list[Any]:
        """Called by storage volume. Read from the client's registered tensors."""
        results: list[Any] = []
        cache = ctx.get(TorchTransportCache)

        def prepare() -> list[tuple[torch.Tensor, RemoteBuffer]]:
            transfers = []
            for (_, tensor), context in zip(entries, self._contexts, strict=True):
                if context.is_object:
                    results.append(context.objects)
                    continue

                if tensor is not None:
                    self._assert_valid_tensor(tensor, context.dtype, context.shape)
                if tensor is None or cache.in_use_by_stale_transfer(tensor):
                    # Store into a new tensor so stale DMA can't overwrite it.
                    tensor = torch.empty(
                        context.shape, dtype=context.dtype, device="cpu"
                    )
                if tensor.numel() != 0:
                    assert context.remote_buffer is not None
                    transfers.append((tensor, context.remote_buffer))
                results.append(tensor)
            return transfers

        await self._transfer(ctx, True, prepare)
        return results

    @override
    async def handle_get_request(
        self,
        ctx: TransportContext,
        entries: list[tuple[Request, Any]],
    ) -> None:
        """Called by storage volume. Write into the client's registered tensors."""

        def prepare() -> list[tuple[torch.Tensor, RemoteBuffer]]:
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
            return transfers

        await self._transfer(ctx, False, prepare)

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
        if self._pending_transport is not None:
            # The handshake failed before the endpoint was published; the
            # volume may have created its side.
            assert self._endpoint_id is not None
            cache.abandon(self.storage_volume_ref.volume_id, self._endpoint_id)
            _close(self._pending_transport)
            self._pending_transport = None

        endpoint, registrations = self._endpoint, self._registrations
        contexts = self._contexts
        self._endpoint = None
        self._registrations = []
        self._contexts = []
        if endpoint is None:
            return
        if self._succeeded or not self._sent or not registrations:
            endpoint.unregister_all(registrations)
            return
        await cache.run_abort(self._abort_request(endpoint, registrations, contexts))

    async def _abort_request(
        self,
        endpoint: _Endpoint,
        registrations: list[_RegistrationKey],
        contexts: list[TorchTransportRequestContext],
    ) -> None:
        """Release a failed request's memory once the volume stopped its DMA."""
        cache = self._client_cache()
        self._abort = True
        try:
            # The volume waits up to one timeout for the request's transfers
            # and up to another while draining them.
            (outcome,) = await asyncio.wait_for(
                self.storage_volume_ref.volume.handshake.call_one(self, []),
                timeout=3 * self._timeout,
            )
        except Exception:
            logger.error(
                "Failed to abort torch transport request; leaking %d registrations",
                len(registrations),
                exc_info=True,
            )
            outcome = _LEAKED
        if outcome == _LEAKED:
            # Keep the tensors and registrations for DMA that may still run.
            cache.leak(self.storage_volume_ref.volume_id, endpoint, contexts)
            return
        if outcome != _KEPT:
            # Reconnect on the next request.
            cache.retire(self.storage_volume_ref.volume_id, endpoint)
        endpoint.unregister_all(registrations)
