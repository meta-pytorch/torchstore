# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""NIXL transport for StorageVolume tensor transfers."""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import time
import uuid
import weakref
from dataclasses import dataclass
from enum import Enum
from functools import cache
from importlib.util import find_spec
from typing import Any, TYPE_CHECKING

import torch
from typing_extensions import override

from torchstore.transport.buffers import TransportBuffer, TransportCache
from torchstore.transport.types import Request

if TYPE_CHECKING:
    from torchstore.strategy import StorageVolumeRef
    from torchstore.transport.buffers import TransportContext

logger = logging.getLogger(__name__)

ENV_TORCHSTORE_NIXL_BACKEND = os.environ.get("TORCHSTORE_NIXL_BACKEND", "UCX")
ENV_TORCHSTORE_NIXL_ENABLED = os.environ.get("TORCHSTORE_NIXL_ENABLED", "0") == "1"
ENV_TORCHSTORE_NIXL_TIMEOUT_S = int(os.environ.get("TORCHSTORE_NIXL_TIMEOUT_S", "60"))


class _NixlTransferStatus(str, Enum):
    """Status strings returned by NIXL's Python transfer API."""

    PROCESSING = "PROC"
    DONE = "DONE"


class _NixlTransferOperation(str, Enum):
    """Operation strings accepted by NIXL's Python transfer API."""

    READ = "READ"
    WRITE = "WRITE"


@cache
def _load_nixl_api() -> tuple[Any, Any]:
    """Import NIXL only when constructing its transport."""
    try:
        from nixl._api import nixl_agent, nixl_agent_config
    except ImportError as error:
        raise RuntimeError("NIXL is not installed") from error
    return nixl_agent, nixl_agent_config


@cache
def nixl_available() -> bool:
    """Return whether the optional NIXL transport was explicitly enabled."""
    return ENV_TORCHSTORE_NIXL_ENABLED and find_spec("nixl") is not None


def _in_leaked_transfer(ctx: TransportContext, tensor: torch.Tensor) -> bool:
    """Return whether a retired agent's leaked transfer may use the tensor's storage."""
    storage = tensor.untyped_storage().data_ptr()
    return any(
        storage in cache.leaked_storages for cache in ctx.retired(NixlAgentCache)
    )


@dataclass
class _Registration:
    """Cached NIXL descriptors tied to the lifetime of their backing tensor storage."""

    descriptors: Any
    storage_ref: weakref.ReferenceType[Any]


class NixlAgentCache(TransportCache):
    """Process-local NIXL agent and reusable memory registrations."""

    def __init__(self) -> None:
        nixl_agent, nixl_agent_config = _load_nixl_api()

        self.backend = ENV_TORCHSTORE_NIXL_BACKEND
        name = (
            f"torchstore-{socket.gethostname()[:24]}-{os.getpid()}-"
            f"{uuid.uuid4().hex[:8]}"
        )
        config = nixl_agent_config(
            enable_prog_thread=True,
            enable_listen_thread=False,
            backends=[self.backend],
        )
        self.agent = nixl_agent(name, config)
        self._registrations: dict[tuple[int, int], _Registration] = {}
        self._remote_agents: dict[str, bytes] = {}
        # Unfinished transfer handles, with the tensors their DMA may still use.
        self.leaked: list[tuple[Any, list[torch.Tensor]]] = []
        self.leaked_storages: set[int] = set()

    def register(self, tensor: torch.Tensor) -> Any:
        """Register a tensor's whole storage once for the lifetime of that storage.

        Transfers build descriptors from the tensor itself, and NIXL accepts any
        descriptor inside a registered region, so one registration covers every
        view into the storage. Resharded gets write many views into one
        destination parameter (5.6k views over 1.5k parameters for Qwen3.5-27B
        pulled from FSDP4), and each registration costs ~1.8 ms over UCX/RoCE.
        """
        storage = tensor.untyped_storage()
        key = (storage.data_ptr(), storage.nbytes())
        existing = self._registrations.get(key)
        if existing is not None:
            return existing.descriptors

        # Not kept: a tensor over the storage would keep the storage alive.
        region = torch.empty(0, dtype=torch.uint8, device=tensor.device).set_(storage)
        descriptors = self.agent.register_memory(region, backends=[self.backend])
        storage_ref = weakref.ref(storage, lambda _ref, _key=key: self._evict(_key))
        self._registrations[key] = _Registration(descriptors, storage_ref)
        return descriptors

    def _evict(self, key: tuple[int, int]) -> None:
        registration = self._registrations.get(key)
        if registration is None:
            return
        self.agent.deregister_memory(registration.descriptors)
        self._registrations.pop(key, None)

    def leak(self, handle: Any, tensors: list[torch.Tensor]) -> None:
        """Keep an unfinished transfer and its memory alive with this cache.

        NIXL has no cancellation API: UCX keeps running RDMA after its handle
        is released, and its memory, registrations and remote agents must
        outlive it, so the cache is never cleared.
        """
        self.leaked.append((handle, tensors))
        self.leaked_storages.update(
            tensor.untyped_storage().data_ptr() for tensor in tensors
        )

    @override
    def busy(self) -> bool:
        return bool(self.leaked)

    def add_remote_agent(self, name: str, metadata: bytes) -> str:
        """Load new remote metadata, replacing an older copy when needed."""
        previous_metadata = self._remote_agents.get(name)
        if previous_metadata == metadata:
            return name
        if previous_metadata is not None:
            self.agent.remove_remote_agent(name)
        remote_name = self.agent.add_remote_agent(metadata)
        if isinstance(remote_name, bytes):
            remote_name = remote_name.decode()
        if remote_name != name:
            raise RuntimeError(
                f"NIXL metadata named agent {remote_name!r}, expected {name!r}"
            )
        self._remote_agents[name] = metadata
        return remote_name

    @override
    def clear(self) -> None:
        for key in list(self._registrations):
            self._evict(key)
        for remote_name in list(self._remote_agents):
            self.agent.remove_remote_agent(remote_name)
            self._remote_agents.pop(remote_name)


@dataclass
class NixlRequestContext:
    """Serializable state for one tensor or object in a request batch."""

    remote_descriptors: bytes | None = None
    # Client-local; stripped from RPC state because NIXL transfers its data directly.
    tensor: torch.Tensor | None = None
    shape: torch.Size | None = None
    dtype: torch.dtype | None = None
    is_object: bool = False
    objects: Any = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["tensor"] = None
        return state


class NixlTransportBuffer(TransportBuffer):
    """Move StorageVolume tensors with NIXL's UCX backend.

    NIXL cannot cancel transfers, so an agent with unfinished transfers and
    their memory are kept for the life of the transport context. The volume
    retires such an agent and uses a new one. Destinations of a failed
    request, including the stored value of a failed in-place PUT, hold
    undefined data. A failed GET's leaked transfers may still overwrite its
    destinations later, even after a later GET into them succeeds.
    """

    supports_batch_puts = True
    supports_batch_gets = True

    def __init__(self, storage_volume_ref: StorageVolumeRef) -> None:
        super().__init__(storage_volume_ref)
        if not nixl_available():
            raise RuntimeError(
                "NIXL transport is unavailable. Install nixl and set "
                "TORCHSTORE_NIXL_ENABLED=1."
            )
        _load_nixl_api()
        self._client_metadata: bytes | None = None
        self._client_agent_name: str | None = None
        self._contexts: list[NixlRequestContext] = []
        # GET destinations the volume may write into until the request succeeds.
        self._destinations: list[torch.Tensor] = []

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["storage_volume_ref"] = None
        state["_destinations"] = []
        return state

    def _client_cache(self) -> NixlAgentCache:
        return self.storage_volume_ref.transport_context.get(NixlAgentCache)

    def _register_client_tensor(self, tensor: torch.Tensor) -> NixlRequestContext:
        assert tensor.is_contiguous()
        context = NixlRequestContext(
            tensor=tensor,
            shape=tensor.shape,
            dtype=tensor.dtype,
        )
        # NIXL cannot create transfer descriptors for zero-byte tensors.
        if tensor.numel() == 0:
            return context

        cache = self._client_cache()
        cache.register(tensor)
        descriptors = cache.agent.get_xfer_descs(tensor)
        context.remote_descriptors = cache.agent.get_serialized_descs(descriptors)
        return context

    def _publish_client_metadata(self) -> None:
        if any(context.remote_descriptors is not None for context in self._contexts):
            cache = self._client_cache()
            self._client_agent_name = cache.agent.name
            self._client_metadata = cache.agent.get_agent_metadata()

    @override
    async def _pre_put_hook(self, requests: list[Request]) -> None:
        """Allocate RDMA memory for put (transport already set up)."""
        self._contexts = []
        for request in requests:
            if request.is_object:
                self._contexts.append(
                    NixlRequestContext(is_object=True, objects=request.objects)
                )
                continue

            tensor = request.tensor_val
            assert tensor is not None
            if not tensor.is_contiguous():
                logger.warning(
                    "NIXL PUT received a non-contiguous tensor for key=%s; "
                    "staging a contiguous CPU copy",
                    request.key,
                )
                tensor = tensor.cpu().contiguous()
            self._contexts.append(self._register_client_tensor(tensor))
        self._publish_client_metadata()

    @override
    async def _pre_get_hook(self, requests: list[Request]) -> None:
        """Fetch metadata if needed and allocate RDMA buffers."""
        # 1. fetch metadata in a single batch, preserving order
        meta_requests = [req.meta_only() for req in requests if req.tensor_val is None]
        meta_results = (
            await self.storage_volume_ref.volume.get_meta.call_one(meta_requests)
            if meta_requests
            else []
        )
        meta_iterator = iter(meta_results)

        # 2. build contexts
        self._contexts = []
        for request in requests:
            tensor = request.tensor_val
            if tensor is None:
                meta = next(meta_iterator)
                if isinstance(meta, str) or meta is None:
                    self._contexts.append(NixlRequestContext(is_object=True))
                    continue
                shape, dtype = meta
                if request.tensor_slice is not None:
                    shape = request.tensor_slice.local_shape
                tensor = torch.empty(shape, dtype=dtype, device="cpu")

            self._contexts.append(self._register_client_tensor(tensor))
        self._publish_client_metadata()
        self._destinations = [
            context.tensor
            for context in self._contexts
            if context.remote_descriptors is not None
        ]

    async def _transfer(
        self,
        ctx: TransportContext,
        operation: _NixlTransferOperation,
        transfers: list[tuple[Request, torch.Tensor, bytes]],
    ) -> None:
        if not transfers:
            return

        keys = [request.key for request, _, _ in transfers]
        transfer_groups: dict[tuple[Any, Any], tuple[Any, Any, list[torch.Tensor]]] = {}
        # Posted handles with the tensors each one transfers.
        handles: list[tuple[Any, list[torch.Tensor]]] = []
        dispatch_error: Exception | None = None
        cache: NixlAgentCache | None = None
        leaked = 0
        try:
            cache = ctx.get(NixlAgentCache)
            remote_agent = self._connect_client(cache)
            for _, tensor, serialized_remote_descs in transfers:
                cache.register(tensor)
                tensor_local_descs = cache.agent.get_xfer_descs(tensor)
                tensor_remote_descs = cache.agent.deserialize_descs(
                    serialized_remote_descs
                )

                # Each NIXL descriptor list must have one memory type per side.
                # Split mixed CPU/GPU batches while preserving cross-device pairs.
                group_key = (
                    tensor_local_descs.getType(),
                    tensor_remote_descs.getType(),
                )
                group = transfer_groups.get(group_key)
                if group is None:
                    transfer_groups[group_key] = (
                        tensor_local_descs,
                        tensor_remote_descs,
                        [tensor],
                    )
                else:
                    local_descs, remote_descs, group_tensors = group
                    for index in range(tensor_local_descs.descCount()):
                        local_descs.append(tensor_local_descs[index])
                    for index in range(tensor_remote_descs.descCount()):
                        remote_descs.append(tensor_remote_descs[index])
                    group_tensors.append(tensor)

            for local_descs, remote_descs, group_tensors in transfer_groups.values():
                handle = None
                try:
                    handle = cache.agent.initialize_xfer(
                        operation.value,
                        local_descs,
                        remote_descs,
                        remote_agent,
                        backends=[cache.backend],
                    )
                    cache.agent.transfer(handle)
                except Exception as error:
                    # NIXL may have posted part of a failed transfer, so leak it.
                    # Earlier handles may already be active. Wait for them below
                    # before propagating the dispatch error.
                    if handle is not None:
                        cache.leak(handle, group_tensors)
                        leaked += 1
                    dispatch_error = error
                    break
                handles.append((handle, group_tensors))

            deadline = time.monotonic() + ENV_TORCHSTORE_NIXL_TIMEOUT_S
            timed_out = 0
            cancelled = False
            check_error: Exception | None = None

            # All handles are already in flight, so waiting for them one at a time
            # does not serialize the transfers. NIXL has no cancellation API, so
            # handles that do not finish by the deadline are leaked with their
            # tensors instead of released: DMA may still use that memory.
            for handle, group_tensors in handles:
                while True:
                    try:
                        status = cache.agent.check_xfer_state(handle)
                    except Exception as error:
                        # NIXL raises for a failed transfer, and again on every
                        # later check of the handle, so stop polling it.
                        if check_error is None:
                            logger.warning(
                                "NIXL %s status check failed for keys=%r: %s",
                                operation.value,
                                keys,
                                error,
                            )
                            check_error = error
                        status = None
                        break
                    if status != _NixlTransferStatus.PROCESSING:
                        break
                    if time.monotonic() >= deadline:
                        break
                    try:
                        await asyncio.sleep(0.001)
                    except asyncio.CancelledError:
                        # Keep polling until the deadline so the transfer can
                        # finish and its handle be released.
                        cancelled = True
                if status == _NixlTransferStatus.DONE:
                    cache.agent.release_xfer_handle(handle)
                    continue
                cache.leak(handle, group_tensors)
                leaked += 1
                if status == _NixlTransferStatus.PROCESSING:
                    timed_out += 1
                elif status is not None and check_error is None:
                    check_error = RuntimeError(
                        f"Unexpected NIXL transfer status {status!r}"
                    )

            if cancelled:
                logger.warning(
                    "NIXL %s cancelled for keys=%r with %d unfinished transfers: %r",
                    operation.value,
                    keys,
                    leaked,
                    dispatch_error or check_error,
                )
                raise asyncio.CancelledError
            if dispatch_error is not None:
                raise dispatch_error
            if check_error is not None:
                raise check_error
            if timed_out:
                raise TimeoutError(
                    f"{timed_out} NIXL transfers did not finish within "
                    f"TORCHSTORE_NIXL_TIMEOUT_S={ENV_TORCHSTORE_NIXL_TIMEOUT_S}s"
                )
        except Exception as error:
            message = f"NIXL {operation.value} failed for keys={keys!r}"
            if leaked:
                message += (
                    f"; {leaked} unfinished transfers may still access their memory"
                )
            raise RuntimeError(message) from error
        finally:
            if leaked:
                assert cache is not None
                # Later transfers use a new agent, so they never replace the
                # remote agents or registrations that leaked transfers use.
                ctx.retire(cache)
                logger.warning(
                    "Retired NIXL agent %s with %d unfinished transfers",
                    cache.agent.name,
                    leaked,
                )

    def _connect_client(self, cache: NixlAgentCache) -> str:
        if self._client_agent_name is None or self._client_metadata is None:
            raise RuntimeError("NIXL request is missing client agent metadata")
        return cache.add_remote_agent(self._client_agent_name, self._client_metadata)

    @override
    async def handle_put_request(
        self,
        ctx: TransportContext,
        entries: list[tuple[Request, Any]],
    ) -> list[Any]:
        """Called by storage volume. Read from client's source RdmaMemory (put)."""
        results: list[Any] = []
        transfers: list[tuple[Request, torch.Tensor, bytes]] = []

        for (request, tensor), request_context in zip(
            entries, self._contexts, strict=True
        ):
            if request_context.is_object:
                results.append(request_context.objects)
                continue

            if tensor is not None and _in_leaked_transfer(ctx, tensor):
                # A leaked transfer may still write into the stored tensor.
                tensor = None
            if tensor is None:
                tensor = torch.empty(
                    request_context.shape,
                    dtype=request_context.dtype,
                    device="cpu",
                )
            self._assert_valid_tensor(
                tensor, request_context.dtype, request_context.shape
            )
            # Empty tensors have no bytes to move or NIXL descriptors to transfer.
            if tensor.numel() != 0:
                assert request_context.remote_descriptors is not None
                transfers.append((request, tensor, request_context.remote_descriptors))
            results.append(tensor)
        await self._transfer(ctx, _NixlTransferOperation.READ, transfers)
        return results

    @override
    async def handle_get_request(
        self,
        ctx: TransportContext,
        entries: list[tuple[Request, Any]],
    ) -> None:
        """Called by storage volume. Write to client's dest RdmaMemory (get)."""
        transfers: list[tuple[Request, torch.Tensor, bytes]] = []

        for (request, data), request_context in zip(
            entries, self._contexts, strict=True
        ):
            if not isinstance(data, torch.Tensor):
                request_context.is_object = True
                request_context.objects = data
                continue

            self._assert_valid_tensor(
                data,
                request_context.dtype,
                request_context.shape,
                must_be_contiguous=False,
            )
            source = data if data.is_contiguous() else data.contiguous()
            # Empty tensors have no bytes to move or NIXL descriptors to transfer.
            if source.numel() != 0:
                assert request_context.remote_descriptors is not None
                transfers.append((request, source, request_context.remote_descriptors))
        await self._transfer(ctx, _NixlTransferOperation.WRITE, transfers)

    @override
    async def _handle_storage_volume_response(
        self,
        requests: list[Request],
        transport_buffer: TransportBuffer,
    ) -> list[Any]:
        assert isinstance(transport_buffer, NixlTransportBuffer)
        results: list[Any] = []
        for client_context, volume_context in zip(
            self._contexts, transport_buffer._contexts, strict=True
        ):
            results.append(
                volume_context.objects
                if volume_context.is_object
                else client_context.tensor
            )
        return results

    @override
    async def _post_request_success(self) -> None:
        self._destinations = []

    async def drop(self) -> None:
        if self._destinations:
            # A failed or cancelled GET may leave volume transfers running, and
            # NIXL cannot cancel them, so keep their destinations allocated.
            self._client_cache().leak(None, self._destinations)
            logger.warning(
                "Keeping %d destinations of a failed NIXL GET",
                len(self._destinations),
            )
        self._destinations = []
        self._client_metadata = None
        self._contexts = []
