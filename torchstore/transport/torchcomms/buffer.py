# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from dataclasses import dataclass
from logging import getLogger
from typing import Any, TYPE_CHECKING

import torch

from torchstore.transport.buffers import TransportBuffer
from torchstore.transport.torchcomms.cache import RdmaMemoryCache, RdmaTransportCache
from torchstore.transport.types import Request

try:
    from torchcomms._transport import RdmaMemory
except ImportError:
    pass

logger = getLogger(__name__)

if TYPE_CHECKING:
    from torchstore.strategy import StorageVolumeRef
    from torchstore.transport.buffers import TransportContext


# I'd like caching of user owned pinned memory to be toggleable.
# yes, I understand we're introduce many "magic" vars. A follow up will
# be to provide a clear TorchStrategy level config for these switches
def _client_rdma_cache_enabled() -> bool:
    return os.environ.get("TORCHSTORE_CLIENT_RDMA_CACHE", "1") == "1"


@dataclass
class RdmaContext:
    """Per-entry state for TorchComms RDMA batch operations."""

    rdma_memory: Any = None  # LOCAL only — stripped in __getstate__
    rdma_remote_buffer: Any = None  # Serialized — SV uses this for RDMA read/write
    tensor_ref: torch.Tensor | None = None  # LOCAL only — GET destination tensor
    shape: torch.Size | None = None  # Serialized
    dtype: torch.dtype | None = None  # Serialized
    is_object: bool = False  # Serialized

    # These can be overwritten by the SV on GET paths.
    objects: Any = None  # Serialized — carries non-tensor data
    device_index: int = 0  # Serialized — identifies which transport/address to use

    def __getstate__(self):
        state = self.__dict__.copy()
        state["rdma_memory"] = None
        state["tensor_ref"] = None
        return state


class TorchCommsRdmaTransportBuffer(TransportBuffer):
    """
    Transport buffer implementation using TorchComms RDMA for efficient tensor transfer.
    """

    supports_batch_puts = True
    supports_batch_gets = True

    def __init__(self, storage_volume_ref: "StorageVolumeRef") -> None:
        super().__init__(storage_volume_ref)

        # {device_index: client_address}, used by SV to look up corresponding peer transport
        self.addresses: dict[int, bytes] = {}
        # device_indexes that need a new SV-side connection (handshake)
        self._devices_to_connect: set[int] = set()
        # {device_index: RdmaTransport}, client local transport lookup
        self._transports: dict[int, Any] = {}

        # Batch state – one context per processed request
        self._contexts: list[RdmaContext] = []

    def _setup_local_transport(self, tensor: torch.Tensor | None) -> None:
        """Ensure a local transport exists for the tensor's device."""
        device = tensor.device if tensor is not None else 0
        transport_cache = self.storage_volume_ref.transport_context.get(
            RdmaTransportCache
        )
        volume_id = self.storage_volume_ref.volume_id
        transport, address, is_new = transport_cache.get_or_create(volume_id, device)
        device_index = transport_cache.device_to_index(device)
        self._transports[device_index] = transport
        self.addresses[device_index] = address
        if is_new:
            self._devices_to_connect.add(device_index)

    def _get_sv_transport(self, ctx: "TransportContext", device_index: int) -> Any:
        """SV side: look up the transport for a given client device_index.

        The SV always uses CPU (device 0 NIC) as its RDMA device
        """
        client_addr = self.addresses[device_index]
        return ctx.get(RdmaTransportCache).get(client_addr, 0)[0]

    def requires_handshake(self, requests: list[Request]) -> bool:
        """Set up transports for all unique devices in the batch."""
        for request in requests:
            if not request.is_object:
                self._setup_local_transport(request.tensor_val)
        return len(self._devices_to_connect) > 0

    async def _post_handshake(
        self,
        handshake_results: list[Any],
        requests: list[Request],
    ) -> None:
        """Connect each local transport to its SV peer."""
        for device_index, sv_address in handshake_results:
            self._transports[device_index].connect(sv_address)

    async def recv_handshake(
        self,
        ctx: "TransportContext",
        entries: list[tuple[Request, Any]],
    ) -> list[Any]:
        """SV side: create a transport per new client device, connect, return SV addresses."""
        transport_cache = ctx.get(RdmaTransportCache)
        results = []
        for device_index in self._devices_to_connect:
            client_address = self.addresses[device_index]
            transport, sv_addr = transport_cache.put(client_address, device=0)
            transport.connect(client_address)
            results.append((device_index, sv_addr))
        return results

    def __getstate__(self) -> dict[str, Any]:
        """Serialize the state of the buffer, excluding non-serializable components."""
        state = self.__dict__.copy()
        state["storage_volume_ref"] = None
        state["_transports"] = {}
        return state

    def _allocate_ctx(self, tensor: torch.Tensor) -> RdmaContext:
        self._assert_valid_tensor(tensor, tensor.dtype, tensor.shape)
        if _client_rdma_cache_enabled():
            cache = self.storage_volume_ref.transport_context.get(RdmaMemoryCache)
            rdma_memory = cache.get_or_register(tensor)
        else:
            rdma_memory = RdmaMemory(tensor)
        return RdmaContext(
            rdma_memory=rdma_memory,
            rdma_remote_buffer=rdma_memory.to_remote_buffer(),
            tensor_ref=tensor,
            shape=tensor.shape,
            dtype=tensor.dtype,
            device_index=RdmaTransportCache.device_to_index(tensor.device),
        )

    async def _pre_put_hook(self, requests: list[Request]) -> None:
        """Allocate RDMA memory for put (transport already set up)."""
        self._contexts = []
        for request in requests:
            if request.is_object:
                self._contexts.append(
                    RdmaContext(is_object=True, objects=request.objects)
                )
            else:
                tensor = request.tensor_val
                if not tensor.is_contiguous():
                    logger.warning(
                        f"PUT called with non-contiguous tensor (key={request.key}), "
                        "creating a contiguous CPU copy"
                    )
                    # stage contiguous copy on CPU to avoid GPU OOM
                    tensor = tensor.cpu().contiguous()
                self._contexts.append(self._allocate_ctx(tensor))

    async def _pre_get_hook(self, requests: list[Request]) -> None:
        """Fetch metadata if needed and allocate RDMA buffers."""
        # 1. fetch metadata in a single batch, preserving order
        meta_requests = [req.meta_only() for req in requests if req.tensor_val is None]
        if meta_requests:
            meta_results = await self.storage_volume_ref.volume.get_meta.call_one(
                meta_requests
            )
        else:
            meta_results = []
        meta_iterator = iter(meta_results)

        # 2. build contexts
        self._contexts = []
        for request in requests:
            if request.tensor_val is not None:
                tensor_ref = request.tensor_val
            else:
                meta = next(meta_iterator)
                if isinstance(meta, str) or meta is None:
                    self._contexts.append(RdmaContext(is_object=True))
                    continue
                if request.tensor_slice is not None:
                    meta = (request.tensor_slice.local_shape, *meta[1:])
                tensor_ref = torch.zeros(
                    meta[0], dtype=meta[1], device=torch.device("cpu")
                )

            self._contexts.append(self._allocate_ctx(tensor_ref))

    async def handle_put_request(
        self,
        ctx: "TransportContext",
        entries: list[tuple[Request, Any]],
    ) -> list[Any]:
        """Called by storage volume. Read from client's source RdmaMemory (put)."""
        rdma_mem_cache = ctx.get(RdmaMemoryCache)

        results = []
        for entry, rdma_ctx in zip(entries, self._contexts, strict=True):
            _, maybe_tensor = entry
            if rdma_ctx.is_object:
                results.append(rdma_ctx.objects)
                continue

            transport = self._get_sv_transport(ctx, rdma_ctx.device_index)

            if maybe_tensor is None:
                maybe_tensor = torch.zeros(
                    rdma_ctx.shape, dtype=rdma_ctx.dtype, device=torch.device("cpu")
                )

            if rdma_ctx.rdma_remote_buffer is None:
                raise RuntimeError(
                    "Internal error: No remote RDMA memory reference found. cannot perform read"
                )
            self._assert_valid_tensor(maybe_tensor, rdma_ctx.dtype, rdma_ctx.shape)

            # TODO: replace sequential reads with true batch RDMA operations (coming to torchcomms)
            receiving_buffer = rdma_mem_cache.get_or_register(maybe_tensor)
            res = transport.read(
                receiving_buffer.to_mutable_view(), rdma_ctx.rdma_remote_buffer
            )
            if res != 0:
                raise RuntimeError(f"RDMA read failed: conn code {res}")
            results.append(maybe_tensor)

        return results

    async def handle_get_request(
        self,
        ctx: "TransportContext",
        entries: list[tuple[Request, Any]],
    ) -> None:
        """Called by storage volume. Write to client's dest RdmaMemory (get).

        Note: SV determination of is_object is authoritative and mutates _contexts
        sent back to the client.
        """
        rdma_mem_cache = ctx.get(RdmaMemoryCache)

        for entry, rdma_ctx in zip(entries, self._contexts, strict=True):
            _, data = entry
            if not isinstance(data, torch.Tensor):
                rdma_ctx.is_object = True
                rdma_ctx.objects = data
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
                # staging copy is lost, don't cache it
                rdma_memory = RdmaMemory(tensor)
            else:
                # stable tensor from SV, cache the registration
                rdma_memory = rdma_mem_cache.get_or_register(tensor)

            transport = self._get_sv_transport(ctx, rdma_ctx.device_index)

            if rdma_ctx.rdma_remote_buffer is None:
                raise RuntimeError(
                    "Internal error: No remote RDMA memory reference found. cannot perform read"
                )
            self._assert_valid_tensor(tensor, rdma_ctx.dtype, rdma_ctx.shape)
            # TODO: replace sequential writes with true batch RDMA operations (coming to torchcomms)

            res = transport.write(rdma_memory.to_view(), rdma_ctx.rdma_remote_buffer)
            if res != 0:
                raise RuntimeError(f"RDMA write failed: conn code {res}")

    async def _handle_storage_volume_response(
        self, requests: list[Request], transport_buffer: "TransportBuffer"
    ) -> list[Any]:
        """Extract data from response buffer on client side."""
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
        """Clean up any resources held by this buffer."""
        for ctx in self._contexts:
            ctx.rdma_remote_buffer = None
            ctx.rdma_memory = None
            ctx.tensor_ref = None
        self._contexts = []
        self._devices_to_connect = set()
