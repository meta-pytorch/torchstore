# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from enum import auto, Enum
from typing import Any, TYPE_CHECKING

import torch

from torchstore.transport.torchcomms.cache import RdmaTransportCache

try:
    from monarch.rdma import is_rdma_available as monarch_rdma_available, RDMABuffer
except ImportError:
    monarch_rdma_available = lambda: False

    def RDMABuffer(*args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "RDMABuffer is not available. This environment was likely not built with rdma support."
        )


if TYPE_CHECKING:
    from torchstore.transport.pipe import StorageVolumeRef


# Environment variable to control GPU Direct RDMA
# Set to "1" to enable GPU-to-GPU RDMA transfers (requires GPUDirect RDMA support)
# Set to "0" to force CPU-only transfers (safer, always works)
GPU_DIRECT_RDMA_ENABLED = os.environ.get("TORCHSTORE_GPU_DIRECT_RDMA", "1") == "1"

# Environment variable to enable GPU storage on storage volume side
# WARNING: This can cause OOM if trainer GPU doesn't have enough memory
# Set to "1" to enable (faster, but risk OOM)
# Set to "0" to disable (safer, uses CPU for storage)
GPU_STORAGE_ENABLED = os.environ.get("TORCHSTORE_GPU_STORAGE", "0") == "1"

logger = logging.getLogger(__name__)


# TODO: we no longer need to chunk with monararch rdma buffer. Setting large chunk size for now,
# but we should remove all chunking code
RDMA_CHUNK_SIZE_MB: int = int(
    os.environ.get("TORCHSTORE_RDMA_CHUNK_SIZE_MB", str(1024 * 32))
)


def rdma_available() -> bool:
    rdma_enabled = (
        os.environ.get("TORCHSTORE_RDMA_ENABLED", "1") == "1"
    )  # TODO: enable on this build
    return rdma_enabled and monarch_rdma_available()


class TransportContext:
    RDMA_TRANSPORT_CACHE = "rdma_transport_cache"

    def __init__(self):
        self.transport_context = {}

    def get_transport_context(self) -> dict[Any, Any]:
        return self.transport_context

    def get_rdma_transport_cache(self) -> RdmaTransportCache:
        if self.RDMA_TRANSPORT_CACHE not in self.transport_context:
            self.transport_context[self.RDMA_TRANSPORT_CACHE] = RdmaTransportCache()
        return self.transport_context[self.RDMA_TRANSPORT_CACHE]


class TransportType(Enum):
    MonarchRPC = auto()
    MonarchRDMA = auto()
    TorchCommsRDMA = auto()


class TransportBuffer:
    finalize: bool = False
    is_object: bool = False
    objects: Any | None = None
    requires_meta: bool = False

    def update(self, other_buffer: "TransportBuffer") -> None:
        self.finalize = other_buffer.finalize
        self.is_object = other_buffer.is_object
        self.objects = other_buffer.objects
        self.requires_meta = other_buffer.requires_meta

    def allocate(self, tensor_like: torch.Tensor | tuple) -> None:
        """Allocates internal buffers based on either an existing tensor
        or a tuple of (shape, dtype)
        """
        raise NotImplementedError()

    async def read_into(
        self, tensor: torch.Tensor | None, transport_context: TransportContext
    ) -> torch.Tensor:
        raise NotImplementedError()

    async def write_from(
        self, tensor: torch.Tensor | None, transport_context: TransportContext
    ) -> None:
        raise NotImplementedError()

    async def handshake(
        self, tensor: torch.Tensor, volume_ref: "StorageVolumeRef"
    ) -> None:
        """Establish a handshake with the remote volume, such as for RDMA."""
        pass

    async def recv_handshake(self, transport_context: TransportContext) -> Any | None:
        """Confirm a handshake initiated by the local client, and return some result."""
        pass

    async def drop(self) -> None:
        """Clean up any resources held by this buffer. Override in subclasses if needed."""
        pass

    def _assert_valid_tensor(
        self,
        tensor: torch.Tensor,
        dtype: torch.dtype,
        shape: torch.Size,
        must_be_contiguous=False,
    ) -> None:
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == dtype, f"{tensor.dtype} != {dtype}"
        assert tensor.shape == shape, f"{tensor.shape} != {shape}"
        assert not must_be_contiguous or tensor.is_contiguous()


class RDMATransportBuffer(TransportBuffer):
    # TODO: when we try this with rdma, I should be able to write rdma directly to the tensor
    # for now we utilize copies.
    # The major blocker for this is dealing with non-contiguous tensors
    requires_meta: bool = True

    def __init__(self) -> None:
        self.rdma_buffers: list[Any] | None = None
        self.tensor_refs: list[torch.Tensor] | None = None
        self.shape: torch.Size | None = None
        self.dtype: torch.dtype | None = None

    async def drop(self) -> None:
        """Explicitly clean up RDMA buffers to prevent kernel memory leak.

        When RDMA buffers are created, they register memory regions with the RDMA
        hardware which pins pages in kernel memory. Without explicit cleanup, these
        pages remain pinned even after the Python objects are garbage collected,
        leading to a memory leak that manifests as unbounded Inactive(anon) growth.
        """
        if self.rdma_buffers is not None:
            for rdma_buf in self.rdma_buffers:
                try:
                    # Drop the RDMA buffer to deregister the memory region
                    await rdma_buf.drop()
                except Exception as e:
                    # Log but don't raise - cleanup should be best-effort
                    logging.warning(f"Failed to drop RDMA buffer during cleanup: {e}")
            self.rdma_buffers = None
            self.tensor_refs = None

    def __getstate__(self) -> dict[str, Any]:
        # Any time that we serialize the transport buffer, the idea is
        # that tensors will be transported via tensor_enginer.RDMABuffer, so it makes
        # no sense to hold this reference when we are serializing
        state = self.__dict__.copy()
        state["tensor_refs"] = None
        return state

    def _create_byte_views_from_tensor(
        self, tensor: torch.Tensor
    ) -> list[torch.Tensor]:
        # handle scalar values
        if tensor.dim() == 0:
            tensor = tensor.unsqueeze(0)
        byte_view = tensor.view(torch.uint8).flatten()
        chunk_size = RDMA_CHUNK_SIZE_MB * 1024 * 1024
        tensor_chunks = torch.split(byte_view, chunk_size, dim=0)

        return tensor_chunks

    def _get_allocation_device(self) -> torch.device:
        """Determine the device to use for tensor allocation.

        Note: This is called from read_into() which can be invoked from:
        1. Storage volume side (put operation) - uses GPU only if GPU_STORAGE_ENABLED
        2. Client side (get operation) - uses GPU if GPU_DIRECT_RDMA_ENABLED

        For storage volume safety, GPU storage is disabled by default.
        Enable with TORCHSTORE_GPU_STORAGE=1 for maximum performance (but OOM risk).
        """
        if GPU_STORAGE_ENABLED and torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")

    def allocate(self, tensor_like: torch.Tensor | tuple) -> None:
        """Allocates internal buffers based on either an existing tensor
        or a tuple of (shape, dtype)

        This is called on the CLIENT side (for get operations).
        When GPU Direct RDMA is enabled, allocates on GPU for zero-copy transfers.
        """
        logging.debug("Allocating rdma buffer")

        if isinstance(tensor_like, str) or tensor_like is None:
            # tensor is just an object, nothing to allocate
            return
        elif isinstance(tensor_like, tuple):
            # Client-side allocation: use GPU if GPU Direct enabled
            if GPU_DIRECT_RDMA_ENABLED and torch.cuda.is_available():
                device = torch.device("cuda", torch.cuda.current_device())
            else:
                device = torch.device("cpu")
            tensor = torch.zeros(
                tensor_like[0], dtype=tensor_like[1], device=device
            )
        else:
            # we have an inplace tensor, preserve its device
            assert isinstance(tensor_like, torch.Tensor)
            tensor = torch.zeros_like(tensor_like, device=tensor_like.device)

        # store tensor meta
        self.shape = tensor.shape
        self.dtype = tensor.dtype
        self.dim = tensor.dim()

        self._assert_valid_tensor(tensor, self.dtype, self.shape)

        byte_view_chunks = self._create_byte_views_from_tensor(tensor)
        self.tensor_refs = [
            torch.zeros_like(chunk, device=tensor.device)
            for chunk in byte_view_chunks
        ]
        self.rdma_buffers = [RDMABuffer(chunk) for chunk in self.tensor_refs]

        chunk_sizes = set()
        for chunk in self.tensor_refs:
            chunk_sizes.add(chunk.shape)
        logging.debug(f"Allocted {len(self.rdma_buffers)} rdma buffers {chunk_sizes=}")

    def update(self, other_buffer: "TransportBuffer") -> None:
        super().update(other_buffer)

    # send
    async def read_into(
        self, tensor: torch.Tensor | None, transport_context: TransportContext
    ) -> torch.Tensor:
        if tensor is None:
            # allocate a tensor to return on appropriate device
            device = self._get_allocation_device()
            tensor = torch.zeros(
                self.shape, dtype=self.dtype, device=device
            )

        self._assert_valid_tensor(tensor, self.dtype, self.shape)
        assert self.rdma_buffers is not None

        chunked_byte_view = self._create_byte_views_from_tensor(tensor)

        # if we have tensor refs locally, we're still in the local case,
        # and we're just copying over our chunks into the tensor from
        # local memory
        if self.tensor_refs is not None:
            for idx, chunk in enumerate(chunked_byte_view):
                chunk.copy_(self.tensor_refs[idx])
            return tensor
        # else: we are in the remote case (in a different process), and must read from
        # the rdma buffer
        # TODO: gather instead of reading sequentially
        try:
            for idx, chunk in enumerate(chunked_byte_view):
                await self.rdma_buffers[idx].read_into(chunk)
        except Exception as e:
            logging.exception(
                f"Failed read_into, {tensor.shape=}, {tensor.dtype=}", exc_info=e
            )
            raise e

        return tensor

    # recv
    async def write_from(
        self, tensor: torch.Tensor | None, transport_context: TransportContext
    ) -> None:
        if tensor is None:
            return
        # source tensor does not have to be contiguous, it is copied into contiguous memory later in this function
        self._assert_valid_tensor(
            tensor, self.dtype, self.shape, must_be_contiguous=False
        )
        assert self.rdma_buffers is not None

        chunked_byte_view = self._create_byte_views_from_tensor(tensor)

        # if we have tensor refs locally, we're still in the local case,
        # and we're just copying over from the tensor into local memory
        if self.tensor_refs is not None:
            for idx, chunk in enumerate(chunked_byte_view):
                self.tensor_refs[idx].copy_(chunk)
            return
        # else: we are in the remote case (in a different process), and must read from
        # the rdma buffer
        # TODO: gather instead of reading sequentially
        for idx, chunk in enumerate(chunked_byte_view):
            await self.rdma_buffers[idx].write_from(chunk)


class MonarchTransportBuffer(TransportBuffer):
    """This interface is mostly a noop, intended to be used with Monarch's regular RPC.
    Not expected to be super fast, but always works.
    """

    finalize: bool = True

    def __init__(self) -> None:
        self.tensor: torch.Tensor | None = None

    def allocate(self, tensor_like: torch.Tensor | tuple) -> None:
        """In the case of using monarch comms, we don't do any allocation ahead of time"""
        return None

    # send
    async def read_into(
        self, tensor: torch.Tensor | None, transport_context: TransportContext
    ) -> torch.Tensor:
        if tensor is not None:
            # if there is a tensor here, likely this is the 'inplace' case,
            # and we should return back a ptr to the original tensor
            # (as opposed to the stored tensor, which we likely don't want to
            # keep around)
            tensor.copy_(self.tensor)
            return tensor

        return self.tensor

    # recv
    async def write_from(
        self, tensor: torch.Tensor | None, transport_context: TransportContext
    ) -> None:
        self.tensor = tensor

    def update(self, other_buffer: "TransportBuffer") -> None:
        super().update(other_buffer)
        self.tensor = other_buffer.tensor
