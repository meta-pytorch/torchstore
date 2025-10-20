# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

try:
    from monarch.rdma import is_rdma_available as monarch_rdma_available, RDMABuffer
except ImportError:
    monarch_rdma_available = lambda: False

    def RDMABuffer(*args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "RDMABuffer is not available. This environemnt was likely not built with rdma support."
        )


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


class TransportBuffer:
    finalize: bool = False
    is_object: bool = False
    objects: Optional[Any] = None
    requires_meta: bool = False

    def update(self, other_buffer: "TransportBuffer") -> None:
        self.finalize = other_buffer.finalize
        self.is_object = other_buffer.is_object
        self.objects = other_buffer.objects
        self.requires_meta = other_buffer.requires_meta

    def allocate(self, tensor_like: Union[torch.Tensor, Tuple]) -> None:
        """Allocates internal buffers based on either an existing tensor
        or a Tuple of (shape, dtype)
        """
        raise NotImplementedError()

    async def read_into(self, tensor: Optional[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError()

    async def write_from(self, tensor: Optional[torch.Tensor]) -> None:
        raise NotImplementedError()

    async def drop(self) -> None:
        """Clean up any resources held by this buffer. Override in subclasses if needed."""
        pass


class RDMATransportBuffer(TransportBuffer):
    # TODO: when we try this with rdma, I should be able to write rdma directly to the tensor
    # for now we utilize copies.
    # The major blocker for this is dealing with non-contiguous tensors
    requires_meta: bool = True

    def __init__(self) -> None:
        self.rdma_buffers: Optional[List[Any]] = None
        self.tensor_refs: Optional[List[torch.Tensor]] = None
        self.shape: Optional[torch.Size] = None
        self.dtype: Optional[torch.dtype] = None

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

    def _assert_valid_tensor(self, tensor: torch.Tensor) -> None:
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == self.dtype, f"{tensor.dtype} != {self.dtype}"
        assert tensor.shape == self.shape, f"{tensor.shape} != {self.shape}"
        assert tensor.is_contiguous()

    def allocate(self, tensor_like: Union[torch.Tensor, Tuple]) -> None:
        """Allocates internal buffers based on either an existing tensor
        or a Tuple of (shape, dtype)
        """
        logging.debug("Allocating rdma buffer")

        if isinstance(tensor_like, str) or tensor_like is None:
            # tensor is just an object, nothing to allocte
            return
        elif isinstance(tensor_like, Tuple):
            # we know the size of the tensor from fetching metadata
            tensor = torch.empty(
                tensor_like[0], dtype=tensor_like[1], device=torch.device("cpu")
            )
        else:
            # we have an inplace tensor, allocate a copy
            assert isinstance(tensor_like, torch.Tensor)
            tensor = torch.empty_like(tensor_like, device=torch.device("cpu"))

        # store tensor meta
        self.shape = tensor.shape
        self.dtype = tensor.dtype
        self.dim = tensor.dim()

        self._assert_valid_tensor(tensor)

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
    async def read_into(self, tensor: Optional[torch.Tensor] = None) -> torch.Tensor:
        if tensor is None:
            # allocate a tensor to return
            tensor = torch.empty(
                self.shape, dtype=self.dtype, device=torch.device("cpu")
            )

        self._assert_valid_tensor(tensor)
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
    async def write_from(self, tensor: Optional[torch.Tensor]) -> None:
        if tensor is None:
            return

        self._assert_valid_tensor(tensor)
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
        self.tensor: Optional[torch.Tensor] = None

    def allocate(self, tensor_like: Union[torch.Tensor, Tuple]) -> None:
        """In the case of using monarch comms, we don't do any allocation ahead of time"""
        return None

    # send
    async def read_into(self, tensor: Optional[torch.Tensor] = None) -> torch.Tensor:
        if tensor is not None:
            # if there is a tensor here, likely this is the 'inplace' case,
            # and we should return back a ptr to the original tensor
            # (as opposed to the stored tensor, which we likely don't want to
            # keep around)
            tensor.copy_(self.tensor)
            return tensor

        return self.tensor

    # recv
    async def write_from(self, tensor: Optional[torch.Tensor]) -> None:
        self.tensor = tensor

    def update(self, other_buffer: "TransportBuffer") -> None:
        super().update(other_buffer)
        self.tensor = other_buffer.tensor
