# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING, Union

import torch

try:
    from monarch.rdma import is_rdma_available as monarch_rdma_available, RDMABuffer
except ImportError:
    monarch_rdma_available = lambda: False

    def RDMABuffer(*args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "RDMABuffer is not available. This environment was likely not built with rdma support."
        )


from torchstore.logging import LatencyTracker
from torchstore.transport.buffers import TransportBuffer
from torchstore.transport.types import Request

if TYPE_CHECKING:
    from torchstore.strategy import StorageVolumeRef


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

        self.rdma_buffer: Optional[Any] = None
        self.byte_view: Optional[torch.Tensor] = None
        self.shape: Optional[torch.Size] = None
        self.dtype: Optional[torch.dtype] = None
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

        # rdma buffer requires we have a ore-existing memory space locally
        # if the user has not provided a local tensor, we need to first
        # identify and allocate  ahead of time
        meta = None
        if not request.tensor_val:
            meta = await self.storage_volume_ref.volume.get_meta.call_one(
                key, request.meta_only()
            )
            if isinstance(meta, str) or meta is None:
                return  # objects don't get handled

        self.allocate(meta or request.tensor_val)

    def _extract_existing_tensor(
        self, current_object: Any, request: Request
    ) -> torch.Tensor:
        """Extract existing tensor from current_object for in-place update.

        Uses fail-fast assertions to ensure type consistency between existing
        data and incoming request.

        Args:
            current_object: The existing stored data (Tensor or dict, NOT None)
            request: The incoming put request

        Returns:
            The existing tensor.

        Raises:
            AssertionError: If there's a type mismatch between existing data and request.
        """
        assert current_object is not None, "current_object must not be None"

        if isinstance(current_object, torch.Tensor):
            # Regular tensor - request must also be a regular tensor (no tensor_slice)
            assert (
                request.tensor_slice is None
            ), "Existing data is a regular tensor but incoming request has tensor_slice (DTensor)"
            return current_object

        if isinstance(current_object, dict):
            # Object dicts should never reach here - objects are handled by early return
            assert (
                "obj" not in current_object
            ), "Existing data is an object but request.is_object is False"
            # DTensor shard dict - incoming request must also be a DTensor
            assert (
                request.tensor_slice is not None
            ), "Existing data is DTensor shards but incoming request has no tensor_slice"
            # Look up by coordinates
            shard = current_object.get(request.tensor_slice.coordinates)
            if shard is not None and "tensor" in shard:
                return shard["tensor"]
            # Coordinates don't match - new shard, return None to allocate new
            return None

        raise AssertionError(f"Unexpected current_object type: {type(current_object)}")

    async def handle_put_request(
        self, request: Request, current_object, storage_transport_context
    ):
        if request.is_object:
            self.is_object = True
            return request.objects

        # Extract existing tensor for potential in-place update
        tensor = None
        if current_object is not None:
            tensor = self._extract_existing_tensor(current_object, request)

        if tensor is None:
            # happens when we haven't seen this tensor / dtensor before
            tensor = torch.empty(
                self.shape, dtype=self.dtype, device=torch.device("cpu")
            )
            print("allocating new tensor")

        self._assert_valid_tensor(tensor, self.dtype, self.shape)

        l = LatencyTracker("sv-put")
        byte_view = tensor.view(torch.uint8).flatten()
        l.track_step("byte view")
        await self.rdma_buffer.read_into(byte_view)
        l.track_step("read intoo")
        l.track_e2e()

        return tensor

    async def handle_get_request(self, data, storage_transport_context):
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
                raise RuntimeError(" extra clowny shit")
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
        if self.rdma_buffer is not None:
            try:
                await self.rdma_buffer.drop()
            except Exception as e:
                logging.warning(f"Failed to drop RDMA buffer during cleanup: {e}")
            self.rdma_buffer = None
            self.byte_view = None

    def __getstate__(self) -> Dict[str, Any]:
        # Any time that we serialize the transport buffer, the idea is
        # that tensors will be transported via tensor_enginer.RDMABuffer, so it makes
        # no sense to hold this reference when we are serializing
        state = self.__dict__.copy()
        state["byte_view"] = None
        state["tensor"] = None
        state["request"] = None
        return state

    def allocate(self, tensor_like: Union[torch.Tensor, Tuple]) -> None:
        """Allocates internal buffers based on either an existing tensor
        or a Tuple of (shape, dtype)
        """
        logging.debug("Allocating rdma buffer")

        if isinstance(tensor_like, Tuple):
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
            tensor = tensor_like.contiguous()
            # tensor = torch.empty_like(tensor_like, device=torch.device("cpu"))

        self.tensor = tensor

        # store tensor meta
        self.shape = tensor.shape
        self.dtype = tensor.dtype
        self.dim = tensor.dim()

        self._assert_valid_tensor(tensor, self.dtype, self.shape)

        byte_view = tensor.view(torch.uint8).flatten()
        self.byte_view = byte_view
        self.rdma_buffer = RDMABuffer(byte_view)
