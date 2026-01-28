# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from typing import Any, TYPE_CHECKING

import torch

try:
    from monarch.rdma import is_rdma_available as monarch_rdma_available, RDMABuffer
except ImportError:
    monarch_rdma_available = lambda: False

    def RDMABuffer(*args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "RDMABuffer is not available. This environment was likely not built with rdma support."
        )


from torchstore.transport.buffers import TransportBuffer
from torchstore.transport.types import Request

if TYPE_CHECKING:
    from torchstore.strategy import StorageVolumeRef
    from torchstore.transport.buffers import TransportContext

MONARCH_RDMA_EAGER_D2H = os.environ.get("TORCHSTORE_MONARCH_RDMA_EAGER_D2H", "1") == "1"


def monarch_rdma_transport_available() -> bool:
    """Check if Monarch RDMA transport is available for use.

    Returns True if:
    - TORCHSTORE_RDMA_ENABLED environment variable is set to "1" (default)
    - The monarch RDMA library is available and functional
    """
    rdma_enabled = os.environ.get("TORCHSTORE_RDMA_ENABLED", "1") == "1"
    return rdma_enabled and monarch_rdma_available()


class MonarchRDMATransportBuffer(TransportBuffer):

    requires_handshake: bool = False

    # TODO: when we try this with rdma, I should be able to write rdma directly to the tensor
    # for now we utilize copies.
    # The major blocker for this is dealing with non-contiguous tensors

    def __init__(self, storage_volume_ref: "StorageVolumeRef"):
        super().__init__(storage_volume_ref)

        self.rdma_buffer: Any | None = None
        self.byte_view: torch.Tensor | None = None
        self.shape: torch.Size | None = None
        self.dtype: torch.dtype | None = None
        self.is_object: bool = False

    async def _pre_put_hook(self, request: Request) -> None:
        """Hook to perform any pre-put operations on the buffer."""

        if request.is_object:
            return
        self.allocate(request.tensor_val)

    async def _pre_get_hook(self, key, request: Request) -> None:
        """Hook to perform any pre-put operations on the buffer."""

        # keep request for later
        self.request = request

        # rdma buffer requires we have a pre-existing memory space locally
        # if the user has not provided a local tensor, we need to first
        # identify and allocate ahead of time
        meta = None
        if not request.tensor_val:
            meta = await self.storage_volume_ref.volume.get_meta.call_one(
                key, request.meta_only()
            )
            if isinstance(meta, str) or meta is None:
                return  # objects don't get handled

        self.allocate(meta or request.tensor_val)

    async def handle_put_request(
        self, ctx: "TransportContext", request: Request, current_object: Any
    ):
        if request.is_object:
            self.is_object = True
            return request.objects

        # current_object is now the extracted tensor (or None)
        tensor = current_object

        if tensor is None:
            # happens when we haven't seen this tensor / dtensor before
            tensor = torch.empty(
                self.shape, dtype=self.dtype, device=torch.device("cpu")
            )

        self._assert_valid_tensor(tensor, self.dtype, self.shape)

        byte_view = tensor.view(torch.uint8).flatten()
        await self.rdma_buffer.read_into(byte_view)

        return tensor

    async def handle_get_request(self, ctx: "TransportContext", data: Any):
        if not isinstance(data, torch.Tensor):
            self.is_object = True
            self.objects = data
            return

        tensor = data

        self._assert_valid_tensor(
            tensor, self.dtype, self.shape, must_be_contiguous=False
        )
        assert self.rdma_buffer is not None

        # Write directly from tensor byte view (no chunking)
        byte_view = tensor.view(torch.uint8).flatten()
        await self.rdma_buffer.write_from(byte_view)

    async def _handle_storage_volume_response(
        self, transport_buffer: TransportBuffer
    ) -> Any:
        if transport_buffer.is_object:
            return transport_buffer.objects

        # if we had to call .contiguous on the tensor during alloc, this assertion is
        # vioalted since .contiguous is a copy
        if self.request.tensor_val is not None:
            if self.request.tensor_val.data_ptr() != self.tensor.data_ptr():
                self.request.tensor_val.copy_(self.tensor)
            return self.request.tensor_val

        # self.byte_view already points to the byte view of self.tensor
        # so the data is already in self.tensor after the RDMA write completes
        return self.tensor

    async def drop(self) -> None:
        """Explicitly clean up RDMA buffers to prevent kernel memory leak.

        When RDMA buffers are created, they register memory regions with the RDMA
        hardware which pins pages in kernel memory. Without explicit cleanup, these
        pages remain pinned even after the Python objects are garbage collected,
        leading to a memory leak that manifests as unbounded Inactive(anon) growth.
        """
        if self.rdma_buffer is None:
            return
        try:
            await self.rdma_buffer.drop()
        except Exception as e:
            logging.warning(f"Failed to drop RDMA buffer during cleanup: {e}")
        self.rdma_buffer = None
        self.byte_view = None

    def __getstate__(self) -> dict[str, Any]:
        # Any time that we serialize the transport buffer, the idea is
        # that tensors will be transported via tensor_enginer.RDMABuffer, so it makes
        # no sense to hold this reference when we are serializing
        state = self.__dict__.copy()
        state["byte_view"] = None
        state["tensor"] = None
        state["request"] = None
        return state

    def allocate(self, tensor_like: torch.Tensor | tuple) -> None:
        """Allocates internal buffers based on either an existing tensor
        or a Tuple of (shape, dtype)
        """
        logging.debug("Allocating rdma buffer")

        if isinstance(tensor_like, tuple):
            # Happens only on get if we don't have an inplace tensor.
            # In that case, we know the size of the tensor from fetching metadata
            tensor = torch.empty(
                tensor_like[0], dtype=tensor_like[1], device=torch.device("cpu")
            )
        else:
            # note: .contiguous will return a copy if this tensor is not contiguous
            # that usually shows up during resharding cases
            assert isinstance(tensor_like, torch.Tensor)

            # monarch sometimes really doesn't like gpu tensors, so we convert to cpu
            # this makes things way slower, and hopefully will be fixed in the future
            if MONARCH_RDMA_EAGER_D2H:
                tensor = tensor_like.cpu()
            tensor = tensor_like.contiguous()

        self.tensor = tensor

        # store tensor meta
        self.shape = tensor.shape
        self.dtype = tensor.dtype
        self.dim = tensor.dim()

        self._assert_valid_tensor(tensor, self.dtype, self.shape)

        byte_view = tensor.view(torch.uint8).flatten()
        self.byte_view = byte_view
        self.rdma_buffer = RDMABuffer(byte_view)
