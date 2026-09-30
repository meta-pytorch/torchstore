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
from dataclasses import dataclass, field
from enum import Enum
from functools import cache
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any

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


@dataclass
class _Registration:
    """Cached NIXL descriptors tied to the lifetime of their backing tensor storage."""

    descriptors: Any
    storage_ref: weakref.ReferenceType[Any]
    xfer_descriptors: Any | None = None
    serialized_xfer_descriptors: bytes | None = None


@dataclass
class _PreparedTransferGroup:
    """Reusable NIXL preparation for one local/remote memory-type pair."""

    local_handle: Any
    remote_handle: Any
    indices: list[int]
    xfer_handles: dict[_NixlTransferOperation, Any] = field(default_factory=dict)
    operations_in_use: set[_NixlTransferOperation] = field(default_factory=set)


@dataclass
class _PreparedTransferLayout:
    """Publisher-side prepared handles for one stable client tensor layout."""

    local_signature: tuple[tuple[str, int, int], ...]
    groups: tuple[_PreparedTransferGroup, ...]
    requests: tuple[Request, ...]
    contexts: tuple[NixlRequestContext, ...]
    invalidated: bool = False


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
        self._client_layout_ids: dict[tuple[Any, ...], str] = {}
        self._published_client_layouts: set[tuple[str, str]] = set()
        self._client_transfer_tensors: dict[
            tuple[str, int, tuple[Any, ...]], tuple[str, torch.Tensor]
        ] = {}
        self._prepared_transfer_layouts: dict[
            tuple[str, str], _PreparedTransferLayout
        ] = {}
        self._publisher_staging_tensors: dict[
            tuple[str, str, int], tuple[str, torch.Tensor]
        ] = {}

    def register(self, tensor: torch.Tensor) -> Any:
        """Register a contiguous tensor once for the lifetime of its storage."""
        key = (tensor.data_ptr(), tensor.nbytes)
        existing = self._registrations.get(key)
        if existing is not None:
            return existing.descriptors

        descriptors = self.agent.register_memory(tensor, backends=[self.backend])
        storage_ref = weakref.ref(
            tensor.untyped_storage(), lambda _ref, _key=key: self._evict(_key)
        )
        self._registrations[key] = _Registration(descriptors, storage_ref)
        return descriptors

    def get_xfer_descriptors(self, tensor: torch.Tensor) -> Any:
        """Return transfer descriptors cached with the tensor registration."""
        key = (tensor.data_ptr(), tensor.nbytes)
        self.register(tensor)
        registration = self._registrations[key]
        if registration.xfer_descriptors is None:
            registration.xfer_descriptors = self.agent.get_xfer_descs(tensor)
        return registration.xfer_descriptors

    def get_serialized_xfer_descriptors(self, tensor: torch.Tensor) -> bytes:
        """Return a cached wire representation of a tensor's descriptors."""
        key = (tensor.data_ptr(), tensor.nbytes)
        descriptors = self.get_xfer_descriptors(tensor)
        registration = self._registrations[key]
        if registration.serialized_xfer_descriptors is None:
            registration.serialized_xfer_descriptors = self.agent.get_serialized_descs(
                descriptors
            )
        return registration.serialized_xfer_descriptors

    def copy_xfer_descriptors(self, tensor: torch.Tensor) -> Any:
        """Return a mutable request-local copy of cached tensor descriptors."""
        return self.agent.deserialize_descs(
            self.get_serialized_xfer_descriptors(tensor)
        )

    def get_client_layout_id(self, layout_key: tuple[Any, ...]) -> str:
        """Return a process-local stable ID for a client tensor layout."""
        layout_id = self._client_layout_ids.get(layout_key)
        if layout_id is None:
            layout_id = uuid.uuid4().hex
            self._client_layout_ids[layout_key] = layout_id
        return layout_id

    def get_client_transfer_tensor(
        self,
        volume_id: str,
        index: int,
        request: Request,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Keep the first destination for a logical request layout alive."""
        tensor_signature = (
            request.key,
            tuple(tensor.shape),
            str(tensor.dtype),
            str(tensor.device),
            request.tensor_slice,
        )
        cache_key = (volume_id, index, tensor_signature)
        cached = self._client_transfer_tensors.get(cache_key)
        if cached is None:
            self._client_transfer_tensors[cache_key] = (request.key, tensor)
            return tensor
        return cached[1]

    def client_layout_is_published(self, volume_id: str, layout_id: str) -> bool:
        return (volume_id, layout_id) in self._published_client_layouts

    def mark_client_layout_published(self, volume_id: str, layout_id: str) -> None:
        self._published_client_layouts.add((volume_id, layout_id))

    def mark_client_layout_missing(self, volume_id: str, layout_id: str) -> None:
        self._published_client_layouts.discard((volume_id, layout_id))

    def has_remote_agent(self, name: str) -> bool:
        return name in self._remote_agents

    def stage_publisher_tensor(
        self,
        remote_agent: str,
        layout_id: str,
        index: int,
        request_key: str,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Copy a source into a stable, reusable publisher-side NIXL buffer."""
        cache_key = (remote_agent, layout_id, index)
        cached = self._publisher_staging_tensors.get(cache_key)
        staging = cached[1] if cached is not None else None
        if (
            staging is None
            or staging.shape != tensor.shape
            or staging.dtype != tensor.dtype
            or staging.device != tensor.device
        ):
            staging = torch.empty_like(tensor, memory_format=torch.contiguous_format)
            self._publisher_staging_tensors[cache_key] = (request_key, staging)
        staging.copy_(tensor)
        return staging

    def get_prepared_transfer_layout(
        self,
        remote_agent: str,
        layout_id: str,
        local_signature: tuple[tuple[str, int, int], ...],
    ) -> _PreparedTransferLayout | None:
        layout = self._prepared_transfer_layouts.get((remote_agent, layout_id))
        if (
            layout is None
            or layout.invalidated
            or layout.local_signature != local_signature
        ):
            return None
        return layout

    def get_prepared_transfer_requests(
        self, remote_agent: str, layout_id: str
    ) -> tuple[list[Request], list[NixlRequestContext]] | None:
        layout = self._prepared_transfer_layouts.get((remote_agent, layout_id))
        if layout is None or layout.invalidated:
            return None
        contexts = [
            NixlRequestContext(
                shape=context.shape,
                dtype=context.dtype,
                is_object=context.is_object,
            )
            for context in layout.contexts
        ]
        return list(layout.requests), contexts

    def set_prepared_transfer_layout(
        self,
        remote_agent: str,
        layout_id: str,
        layout: _PreparedTransferLayout,
    ) -> None:
        key = (remote_agent, layout_id)
        previous = self._prepared_transfer_layouts.pop(key, None)
        if previous is not None:
            if any(group.operations_in_use for group in previous.groups):
                self._prepared_transfer_layouts[key] = previous
                raise RuntimeError("cannot replace an active NIXL descriptor layout")
            self._release_prepared_transfer_layout(previous)
        self._prepared_transfer_layouts[key] = layout

    def release_invalidated_transfer_layout(
        self,
        remote_agent: str,
        layout_id: str,
        layout: _PreparedTransferLayout,
    ) -> None:
        key = (remote_agent, layout_id)
        if (
            layout.invalidated
            and not any(group.operations_in_use for group in layout.groups)
            and self._prepared_transfer_layouts.get(key) is layout
        ):
            self._prepared_transfer_layouts.pop(key)
            self._release_prepared_transfer_layout(layout)

    def _release_prepared_transfer_layout(
        self, layout: _PreparedTransferLayout
    ) -> None:
        for group in layout.groups:
            for handle in group.xfer_handles.values():
                self.agent.release_xfer_handle(handle)
            self.agent.release_dlist_handle(group.local_handle)
            self.agent.release_dlist_handle(group.remote_handle)

    def _clear_prepared_transfer_layouts(self, remote_agent: str | None = None) -> None:
        for key, layout in list(self._prepared_transfer_layouts.items()):
            if remote_agent is None or key[0] == remote_agent:
                self._release_prepared_transfer_layout(layout)
                self._prepared_transfer_layouts.pop(key, None)

    def _invalidate_prepared_transfer_layouts(
        self, registration_key: tuple[int, int]
    ) -> None:
        for cache_key, layout in list(self._prepared_transfer_layouts.items()):
            if not any(
                (data_ptr, nbytes) == registration_key
                for _, data_ptr, nbytes in layout.local_signature
            ):
                continue
            if any(group.operations_in_use for group in layout.groups):
                layout.invalidated = True
            else:
                self._prepared_transfer_layouts.pop(cache_key)
                self._release_prepared_transfer_layout(layout)

    def _evict(self, key: tuple[int, int]) -> None:
        registration = self._registrations.get(key)
        if registration is None:
            return
        self._invalidate_prepared_transfer_layouts(key)
        self.agent.deregister_memory(registration.descriptors)
        self._registrations.pop(key, None)

    def add_remote_agent(self, name: str, metadata: bytes) -> str:
        """Load new remote metadata, replacing an older copy when needed."""
        previous_metadata = self._remote_agents.get(name)
        if previous_metadata == metadata:
            return name
        if previous_metadata is not None:
            self._clear_prepared_transfer_layouts(name)
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
    def delete(self, keys: set[str]) -> None:
        stale_layout_ids = {
            layout_id
            for layout_key, layout_id in self._client_layout_ids.items()
            if any(entry[0] in keys for entry in layout_key[1])
        }
        self._client_layout_ids = {
            layout_key: layout_id
            for layout_key, layout_id in self._client_layout_ids.items()
            if layout_id not in stale_layout_ids
        }
        self._published_client_layouts = {
            published
            for published in self._published_client_layouts
            if published[1] not in stale_layout_ids
        }
        self._client_transfer_tensors = {
            cache_key: entry
            for cache_key, entry in self._client_transfer_tensors.items()
            if entry[0] not in keys
        }
        for cache_key, layout in list(self._prepared_transfer_layouts.items()):
            if any(key in keys for key, _, _ in layout.local_signature):
                self._release_prepared_transfer_layout(layout)
                self._prepared_transfer_layouts.pop(cache_key, None)
        self._publisher_staging_tensors = {
            cache_key: entry
            for cache_key, entry in self._publisher_staging_tensors.items()
            if entry[0] not in keys
        }

    @override
    def clear(self) -> None:
        self._clear_prepared_transfer_layouts()
        self._client_transfer_tensors.clear()
        self._client_layout_ids.clear()
        self._published_client_layouts.clear()
        self._publisher_staging_tensors.clear()
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
    """Move StorageVolume tensors with NIXL's UCX backend."""

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
        self._descriptor_layout_id: str | None = None
        self._descriptor_cache_miss = False
        self._descriptor_request_compact = False
        self._server_request_received = False

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["storage_volume_ref"] = None
        if self._descriptor_request_compact and not self._server_request_received:
            state["_contexts"] = []
        return state

    def _client_cache(self) -> NixlAgentCache:
        return self.storage_volume_ref.transport_context.get(NixlAgentCache)

    def _register_client_tensor(
        self, tensor: torch.Tensor, *, include_descriptors: bool = True
    ) -> NixlRequestContext:
        assert tensor.is_contiguous()
        context = NixlRequestContext(
            tensor=tensor,
            shape=tensor.shape,
            dtype=tensor.dtype,
        )
        # NIXL cannot create transfer descriptors for zero-byte tensors.
        if tensor.numel() == 0 or not include_descriptors:
            return context

        cache = self._client_cache()
        context.remote_descriptors = cache.get_serialized_xfer_descriptors(tensor)
        return context

    def _publish_client_metadata(self) -> None:
        if self._descriptor_layout_id is not None or any(
            context.remote_descriptors is not None for context in self._contexts
        ):
            cache = self._client_cache()
            self._client_agent_name = cache.agent.name
            if (
                self._descriptor_layout_id is None
                or not cache.client_layout_is_published(
                    self.storage_volume_ref.volume_id, self._descriptor_layout_id
                )
            ):
                self._client_metadata = cache.agent.get_agent_metadata()

    def _configure_get_descriptor_layout(self, requests: list[Request]) -> None:
        tensor_layout = tuple(
            (
                request.key,
                tuple(context.tensor.shape),
                str(context.tensor.dtype),
                str(context.tensor.device),
                request.tensor_slice,
            )
            for request, context in zip(requests, self._contexts, strict=True)
            if context.tensor is not None and context.tensor.numel() != 0
        )
        if not tensor_layout:
            return

        cache = self._client_cache()
        self._descriptor_layout_id = cache.get_client_layout_id(("GET", tensor_layout))
        if not cache.client_layout_is_published(
            self.storage_volume_ref.volume_id, self._descriptor_layout_id
        ):
            self._restore_client_descriptors()

    def _restore_client_descriptors(self) -> None:
        for context in self._contexts:
            tensor = context.tensor
            if tensor is not None and tensor.numel() != 0:
                context.remote_descriptors = (
                    self._client_cache().get_serialized_xfer_descriptors(tensor)
                )

    @override
    def _storage_volume_requests(self, requests: list[Request]) -> list[Request]:
        if self._descriptor_layout_id is not None and all(
            context.remote_descriptors is None for context in self._contexts
        ):
            self._descriptor_request_compact = True
            return []
        return requests

    @override
    def resolve_get_requests(
        self,
        ctx: TransportContext,
        requests: list[Request],
    ) -> list[Request] | None:
        self._server_request_received = True
        if requests or self._descriptor_layout_id is None:
            return requests
        if self._client_agent_name is None:
            self._descriptor_cache_miss = True
            return None
        cached = ctx.get(NixlAgentCache).get_prepared_transfer_requests(
            self._client_agent_name, self._descriptor_layout_id
        )
        if cached is None:
            self._descriptor_cache_miss = True
            return None
        requests, self._contexts = cached
        return requests

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
        for index, request in enumerate(requests):
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

            if tensor.numel() != 0:
                tensor = self._client_cache().get_client_transfer_tensor(
                    self.storage_volume_ref.volume_id,
                    index,
                    request,
                    tensor,
                )

            self._contexts.append(
                self._register_client_tensor(tensor, include_descriptors=False)
            )
        self._configure_get_descriptor_layout(requests)
        self._publish_client_metadata()

    async def _transfer(
        self,
        ctx: TransportContext,
        operation: _NixlTransferOperation,
        transfers: list[tuple[Request, torch.Tensor, bytes | None]],
        request_layout: list[Request] | None = None,
    ) -> bool:
        if not transfers:
            return True

        cache = ctx.get(NixlAgentCache)
        remote_agent = self._connect_client(cache)
        local_signature = tuple(
            (request.key, tensor.data_ptr(), tensor.nbytes)
            for request, tensor, _ in transfers
        )
        prepared_layout = (
            cache.get_prepared_transfer_layout(
                remote_agent,
                self._descriptor_layout_id,
                local_signature,
            )
            if self._descriptor_layout_id is not None
            else None
        )
        if prepared_layout is None and any(
            remote_descriptors is None for _, _, remote_descriptors in transfers
        ):
            return False

        handles = []
        persistent_handles: list[tuple[_PreparedTransferGroup, Any]] = []
        temporary_handles = []
        dispatch_error: Exception | None = None
        try:
            if prepared_layout is None:
                transfer_groups: dict[tuple[Any, Any], tuple[Any, Any]] = {}
                for _, tensor, serialized_remote_descs in transfers:
                    assert serialized_remote_descs is not None
                    tensor_local_descs = cache.get_xfer_descriptors(tensor)
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
                        # The aggregate is mutated below, so clone the cached local
                        # descriptor list before using it as the accumulator.
                        transfer_groups[group_key] = (
                            cache.copy_xfer_descriptors(tensor),
                            tensor_remote_descs,
                        )
                    else:
                        local_descs, remote_descs = group
                        for index in range(tensor_local_descs.descCount()):
                            local_descs.append(tensor_local_descs[index])
                        for index in range(tensor_remote_descs.descCount()):
                            remote_descs.append(tensor_remote_descs[index])

                if self._descriptor_layout_id is not None:
                    prepared_groups = []
                    try:
                        for local_descs, remote_descs in transfer_groups.values():
                            if local_descs.descCount() != remote_descs.descCount():
                                raise RuntimeError(
                                    "NIXL local and remote descriptor counts differ"
                                )
                            local_handle = cache.agent.prep_xfer_dlist(
                                "NIXL_INIT_AGENT",
                                local_descs,
                                backends=[cache.backend],
                            )
                            try:
                                remote_handle = cache.agent.prep_xfer_dlist(
                                    remote_agent,
                                    remote_descs,
                                    backends=[cache.backend],
                                )
                            except Exception:
                                cache.agent.release_dlist_handle(local_handle)
                                raise
                            prepared_groups.append(
                                _PreparedTransferGroup(
                                    local_handle,
                                    remote_handle,
                                    list(range(local_descs.descCount())),
                                )
                            )
                    except Exception:
                        for group in prepared_groups:
                            cache.agent.release_dlist_handle(group.local_handle)
                            cache.agent.release_dlist_handle(group.remote_handle)
                        raise
                    prepared_layout = _PreparedTransferLayout(
                        local_signature,
                        tuple(prepared_groups),
                        tuple(
                            request_layout or (request for request, _, _ in transfers)
                        ),
                        tuple(
                            NixlRequestContext(
                                shape=context.shape,
                                dtype=context.dtype,
                                is_object=context.is_object,
                            )
                            for context in self._contexts
                        ),
                    )
                    cache.set_prepared_transfer_layout(
                        remote_agent,
                        self._descriptor_layout_id,
                        prepared_layout,
                    )
            try:
                if prepared_layout is not None:
                    for group in prepared_layout.groups:
                        handle = group.xfer_handles.get(operation)
                        if handle is None:
                            handle = cache.agent.make_prepped_xfer(
                                operation.value,
                                group.local_handle,
                                group.indices,
                                group.remote_handle,
                                group.indices,
                                backends=[cache.backend],
                            )
                            group.xfer_handles[operation] = handle
                        if operation in group.operations_in_use:
                            handle = cache.agent.make_prepped_xfer(
                                operation.value,
                                group.local_handle,
                                group.indices,
                                group.remote_handle,
                                group.indices,
                                backends=[cache.backend],
                            )
                            temporary_handles.append(handle)
                        else:
                            group.operations_in_use.add(operation)
                            persistent_handles.append((group, handle))
                        handles.append(handle)
                        cache.agent.transfer(handle)
                else:
                    for local_descs, remote_descs in transfer_groups.values():
                        handle = cache.agent.initialize_xfer(
                            operation.value,
                            local_descs,
                            remote_descs,
                            remote_agent,
                            backends=[cache.backend],
                        )
                        handles.append(handle)
                        cache.agent.transfer(handle)
            except Exception as error:
                # Earlier handles may already be active. Drain them below before
                # propagating the dispatch error.
                dispatch_error = error

            deadline = time.monotonic() + ENV_TORCHSTORE_NIXL_TIMEOUT_S
            timed_out = False
            cancelled = False
            failed = False
            check_error: Exception | None = None

            # All handles are already in flight, so waiting for them one at a time
            # does not serialize the transfers. NIXL has no cancellation API, so a
            # timeout is reported only after active operations reach a terminal
            # state. Returning earlier could free registered tensor storage while
            # DMA is still using it.
            for handle in handles:
                while True:
                    try:
                        status = cache.agent.check_xfer_state(handle)
                    except Exception as error:
                        # A status-query failure does not establish that the DMA has
                        # stopped. Remember the error, but keep polling so registered
                        # storage and the transfer handle remain alive until terminal.
                        if check_error is None:
                            check_error = error
                        status = _NixlTransferStatus.PROCESSING
                    if status != _NixlTransferStatus.PROCESSING:
                        break
                    timed_out |= time.monotonic() >= deadline
                    try:
                        await asyncio.sleep(0)
                    except asyncio.CancelledError:
                        # Keep polling so the RPC retains both sides' tensor storage.
                        cancelled = True
                if status != _NixlTransferStatus.DONE:
                    failed = True

            if cancelled:
                raise asyncio.CancelledError
            if dispatch_error is not None:
                raise dispatch_error
            if check_error is not None:
                raise check_error
            if failed:
                raise RuntimeError("NIXL transfer failed")
            if timed_out:
                raise TimeoutError("NIXL transfer timed out")
            return True
        except Exception as error:
            raise RuntimeError(
                f"NIXL {operation.value} failed for keys="
                f"{[request.key for request, _, _ in transfers]!r}"
            ) from error
        finally:
            for group, _ in persistent_handles:
                group.operations_in_use.discard(operation)
            # Concurrent uses get request-local handles. Cached handles remain
            # prepared and are reposted after each completed transfer.
            for handle in temporary_handles:
                cache.agent.release_xfer_handle(handle)
            if prepared_layout is None:
                # Legacy requests do not have a reusable layout.
                for handle in handles:
                    cache.agent.release_xfer_handle(handle)
            elif self._descriptor_layout_id is not None:
                cache.release_invalidated_transfer_layout(
                    remote_agent,
                    self._descriptor_layout_id,
                    prepared_layout,
                )

    def _connect_client(self, cache: NixlAgentCache) -> str:
        if self._client_agent_name is None:
            raise RuntimeError("NIXL request is missing the client agent name")
        if self._client_metadata is None:
            if cache.has_remote_agent(self._client_agent_name):
                return self._client_agent_name
            raise RuntimeError("NIXL request is missing uncached client agent metadata")
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
        transfers: list[tuple[Request, torch.Tensor, bytes | None]] = []

        for index, ((request, data), request_context) in enumerate(
            zip(entries, self._contexts, strict=True)
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
            if (
                self._client_agent_name is not None
                and self._descriptor_layout_id is not None
            ):
                source = ctx.get(NixlAgentCache).stage_publisher_tensor(
                    self._client_agent_name,
                    self._descriptor_layout_id,
                    index,
                    request.key,
                    data,
                )
            else:
                source = data if data.is_contiguous() else data.contiguous()
            # Empty tensors have no bytes to move or NIXL descriptors to transfer.
            if source.numel() != 0:
                transfers.append((request, source, request_context.remote_descriptors))
        self._descriptor_cache_miss = not await self._transfer(
            ctx,
            _NixlTransferOperation.WRITE,
            transfers,
            [request for request, _ in entries],
        )
        # Descriptor bytes are request-only data and need not be echoed back in
        # the response after the publisher has cached the prepared layout.
        for context in self._contexts:
            context.remote_descriptors = None

    @override
    async def _handle_storage_volume_response(
        self,
        requests: list[Request],
        transport_buffer: TransportBuffer,
    ) -> list[Any]:
        assert isinstance(transport_buffer, NixlTransportBuffer)
        if transport_buffer._descriptor_cache_miss:
            if self._descriptor_layout_id is None:
                raise RuntimeError("NIXL descriptor cache miss without a layout ID")
            cache = self._client_cache()
            cache.mark_client_layout_missing(
                self.storage_volume_ref.volume_id, self._descriptor_layout_id
            )
            self._descriptor_request_compact = False
            self._restore_client_descriptors()
            self._publish_client_metadata()
            transport_buffer = await self.storage_volume_ref.volume.get.call_one(
                self, [request.meta_only() for request in requests]
            )
            if transport_buffer._descriptor_cache_miss:
                raise RuntimeError("NIXL descriptor cache refill failed")
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
        if self._descriptor_layout_id is not None:
            self._client_cache().mark_client_layout_published(
                self.storage_volume_ref.volume_id, self._descriptor_layout_id
            )

    async def drop(self) -> None:
        self._client_metadata = None
        self._contexts = []
        self._descriptor_layout_id = None
