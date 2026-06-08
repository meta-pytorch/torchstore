# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import torch

try:
    from monarch.rdma import RDMAAction, RDMABuffer

    try:
        # monarch >= 0.4.0
        from monarch.rdma import is_ibverbs_available as monarch_rdma_available
    except ImportError:
        # monarch < 0.4.0
        from monarch.rdma import is_rdma_available as monarch_rdma_available
except ImportError:
    monarch_rdma_available = lambda: False

    def RDMABuffer(*args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "RDMABuffer is not available. This environment was likely not built with rdma support."
        )

    def RDMAAction(*args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "RDMAAction is not available. This environment was likely not built with rdma support."
        )


from torchstore.transport.buffers import TransportBuffer
from torchstore.transport.types import Request
from torchstore.utils import to_byte_view

if TYPE_CHECKING:
    from torchstore.strategy import StorageVolumeRef
    from torchstore.transport.buffers import TransportContext


def monarch_rdma_transport_available() -> bool:
    """Check if Monarch RDMA transport is available for use.

    Returns True if:
    - TORCHSTORE_RDMA_ENABLED environment variable is set to "1" (default)
    - The monarch RDMA library is available and functional
    """
    rdma_enabled = os.environ.get("TORCHSTORE_RDMA_ENABLED", "1") == "1"
    return rdma_enabled and monarch_rdma_available()


@dataclass
class RdmaContext:
    """Per-entry state for one request in a Monarch RDMA batch.

    The client registers ``tensor`` with an ``RDMABuffer`` and ships the
    buffer to the storage volume; the storage volume reads from (PUT) or
    writes into (GET) it via a batched ``RDMAAction``. ``tensor`` is local
    to the side that owns the memory and is stripped on serialization, so
    the storage volume only sees the remote buffer handle and the metadata.
    """

    rdma_buffer: Any = None  # serialized — SV reads/writes against it
    tensor: torch.Tensor | None = None  # LOCAL only — registered tensor
    shape: torch.Size | None = None  # serialized
    dtype: torch.dtype | None = None  # serialized
    is_object: bool = False  # serialized
    objects: Any = None  # serialized — carries non-tensor data

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["tensor"] = None
        return state


class MonarchRDMATransportBuffer(TransportBuffer):
    """Transport buffer that moves tensor data over Monarch RDMA.

    A whole put/get batch is registered as one context per request and
    transferred with a single ``RDMAAction``, so the storage volume
    issues all reads (or writes) for the batch as one submission rather
    than one round trip per tensor.

    Tensors must be contiguous; this transport registers them directly
    and does not stage contiguous copies.
    """

    supports_batch_puts = True
    supports_batch_gets = True

    def __init__(self, storage_volume_ref: "StorageVolumeRef"):
        super().__init__(storage_volume_ref)

        # One context per request, aligned with the request list.
        self._contexts: list[RdmaContext] = []

    def _allocate_ctx(self, tensor: torch.Tensor) -> RdmaContext:
        """Register a contiguous ``tensor`` for RDMA and return its context."""
        self._assert_valid_tensor(tensor, tensor.dtype, tensor.shape)
        return RdmaContext(
            rdma_buffer=RDMABuffer(to_byte_view(tensor)),
            tensor=tensor,
            shape=tensor.shape,
            dtype=tensor.dtype,
        )

    async def _pre_put_hook(self, requests: list[Request]) -> None:
        """Register a buffer per tensor request (client side)."""
        self._contexts = []
        for request in requests:
            if request.is_object:
                self._contexts.append(
                    RdmaContext(is_object=True, objects=request.objects)
                )
            else:
                self._contexts.append(self._allocate_ctx(request.tensor_val))

    async def _pre_get_hook(self, requests: list[Request]) -> None:
        """Allocate destination buffers per request (client side).

        For requests without an inplace destination we batch a single
        ``get_meta`` call to learn each tensor's shape/dtype before
        allocating.
        """
        meta_requests = [r.meta_only() for r in requests if r.tensor_val is None]
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
                self._contexts.append(self._allocate_ctx(request.tensor_val))
                continue

            meta = next(meta_iterator)
            if isinstance(meta, str) or meta is None:
                # objects don't get an RDMA buffer
                self._contexts.append(RdmaContext(is_object=True))
                continue

            # if we are fetching a tensor slice, the local shape is already known
            if request.tensor_slice is not None:
                meta = (request.tensor_slice.local_shape, *meta[1:])

            tensor = torch.empty(meta[0], dtype=meta[1], device=torch.device("cpu"))
            self._contexts.append(self._allocate_ctx(tensor))

    async def handle_put_request(
        self,
        ctx: "TransportContext",
        entries: list[tuple[Request, Any]],
    ) -> list[Any]:
        """Read each client buffer into local storage (storage volume side)."""
        action = RDMAAction()
        has_rdma_ops = False
        results: list[Any] = []

        for (request, current_object), rdma_ctx in zip(
            entries, self._contexts, strict=True
        ):
            if rdma_ctx.is_object:
                results.append(rdma_ctx.objects)
                continue

            tensor = current_object
            if tensor is None:
                # happens when we haven't seen this tensor / dtensor before
                tensor = torch.empty(
                    rdma_ctx.shape, dtype=rdma_ctx.dtype, device=torch.device("cpu")
                )

            self._assert_valid_tensor(tensor, rdma_ctx.dtype, rdma_ctx.shape)
            action.read_remote(to_byte_view(tensor), rdma_ctx.rdma_buffer)
            has_rdma_ops = True
            results.append(tensor)

        if has_rdma_ops:
            await action.submit()

        return results

    async def handle_get_request(
        self,
        ctx: "TransportContext",
        entries: list[tuple[Request, Any]],
    ) -> None:
        """Write stored data into each client buffer (storage volume side).

        The storage volume's view of ``is_object`` is authoritative: it
        mutates ``_contexts`` so the client response path can route
        non-tensor data correctly.
        """
        action = RDMAAction()
        has_rdma_ops = False

        for (request, data), rdma_ctx in zip(entries, self._contexts, strict=True):
            if not isinstance(data, torch.Tensor):
                rdma_ctx.is_object = True
                rdma_ctx.objects = data
                continue

            self._assert_valid_tensor(
                data, rdma_ctx.dtype, rdma_ctx.shape, must_be_contiguous=False
            )
            action.write_remote(rdma_ctx.rdma_buffer, to_byte_view(data))
            has_rdma_ops = True

        if has_rdma_ops:
            await action.submit()

    async def _handle_storage_volume_response(
        self, requests: list[Request], transport_buffer: "TransportBuffer"
    ) -> list[Any]:
        results: list[Any] = []
        for client_ctx, sv_ctx in zip(
            self._contexts, transport_buffer._contexts, strict=True
        ):
            if sv_ctx.is_object:
                results.append(sv_ctx.objects)
            else:
                results.append(client_ctx.tensor)
        return results

    async def drop(self) -> None:
        """Explicitly clean up RDMA buffers to prevent kernel memory leak.

        When RDMA buffers are created, they register memory regions with the RDMA
        hardware which pins pages in kernel memory. Without explicit cleanup, these
        pages remain pinned even after the Python objects are garbage collected,
        leading to a memory leak that manifests as unbounded Inactive(anon) growth.
        """
        for rdma_ctx in self._contexts:
            if rdma_ctx.rdma_buffer is None:
                continue
            try:
                await rdma_ctx.rdma_buffer.drop()
            except Exception as e:
                logging.warning(f"Failed to drop RDMA buffer during cleanup: {e}")
        self._contexts = []
