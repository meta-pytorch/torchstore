# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING, Union

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
from torchstore.transport.pipe import Request

if TYPE_CHECKING:
    from torchstore.transport.buffers import TransportContext
    from torchstore.transport.pipe import StorageVolumeRef


# TODO: we no longer need to chunk with monararch rdma buffer. Setting large chunk size for now,
# but we should remove all chunking code
RDMA_CHUNK_SIZE_MB: int = int(
    os.environ.get("TORCHSTORE_RDMA_CHUNK_SIZE_MB", str(1024 * 32))
)


class MonarchRDMATransportBuffer(TransportBuffer):

    requires_handshake: bool = False

    # # TODO: when we try this with rdma, I should be able to write rdma directly to the tensor
    # # for now we utilize copies.
    # # The major blocker for this is dealing with non-contiguous tensors
    # requires_meta: bool = True

    def __init__(self, storage_volume_ref: "StorageVolumeRef"):
        super().__init__(storage_volume_ref)

        self.rdma_buffers: Optional[List[Any]] = None
        self.tensor_refs: Optional[List[torch.Tensor]] = None
        self.shape: Optional[torch.Size] = None
        self.dtype: Optional[torch.dtype] = None

    def _pre_put_hook(self, request: Request) -> None:
        """Hook to perform any pre-put operations on the buffer."""
        self.allocate(request.tensor_val)

    async def handle_put_request(
        self, request: Request, current_object, storage_transport_context
    ):

        # TODO:
        # add support for writting diretly to current_object (inplace update)

        # TODO: clunky, this transport buffer should have something for
        # objects as well.
        if request.is_object:
            return request.objects

        # allocate new tensor to return
        tensor = torch.empty(self.shape, dtype=self.dtype, device=torch.device("cpu"))
        chunked_byte_view = self._create_byte_views_from_tensor(tensor)

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

    def __getstate__(self) -> Dict[str, Any]:
        # Any time that we serialize the transport buffer, the idea is
        # that tensors will be transported via tensor_enginer.RDMABuffer, so it makes
        # no sense to hold this reference when we are serializing
        state = self.__dict__.copy()
        state["tensor_refs"] = None
        return state

    def _create_byte_views_from_tensor(
        self, tensor: torch.Tensor
    ) -> List[torch.Tensor]:
        # handle scalar values
        if tensor.dim() == 0:
            tensor = tensor.unsqueeze(0)
        byte_view = tensor.view(torch.uint8).flatten()
        chunk_size = RDMA_CHUNK_SIZE_MB * 1024 * 1024
        tensor_chunks = torch.split(byte_view, chunk_size, dim=0)

        return tensor_chunks

    def allocate(self, tensor_like: Union[torch.Tensor, Tuple]) -> None:
        """Allocates internal buffers based on either an existing tensor
        or a Tuple of (shape, dtype)
        """
        logging.debug("Allocating rdma buffer")

        if isinstance(tensor_like, str) or tensor_like is None:
            # tensor is just an object, nothing to allocte
            return
        elif isinstance(tensor_like, Tuple):
            # Happens only on get if we don't have an inplace tensor.
            # In that case, we know the size of the tensor from fetching metadata
            tensor = torch.empty(
                tensor_like[0], dtype=tensor_like[1], device=torch.device("cpu")
            )
        else:
            # we have an tensor, allocate a copy
            # this copy is mostly to avoid contiguous tensors issues
            # that show up during resharding
            assert isinstance(tensor_like, torch.Tensor)
            tensor = torch.empty_like(tensor_like, device=torch.device("cpu"))

        # store tensor meta
        self.shape = tensor.shape
        self.dtype = tensor.dtype
        self.dim = tensor.dim()

        self._assert_valid_tensor(tensor, self.dtype, self.shape)

        byte_view_chunks = self._create_byte_views_from_tensor(tensor)
        self.tensor_refs = [
            torch.empty_like(chunk, device=torch.device("cpu"))
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
        self,
        tensor: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if tensor is None:
            # allocate a tensor to return
            tensor = torch.empty(
                self.shape, dtype=self.dtype, device=torch.device("cpu")
            )

        self._assert_valid_tensor(tensor, self.dtype, self.shape)
        assert self.rdma_buffers is not None

    # recv
    async def write_from(
        self, tensor: Optional[torch.Tensor], transport_context: "TransportContext"
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
