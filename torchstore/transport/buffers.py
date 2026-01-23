# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, TYPE_CHECKING

import torch

from torchstore.logging import LatencyTracker
from torchstore.transport.torchcomms.cache import RdmaTransportCache

if TYPE_CHECKING:
    from torchstore.strategy import StorageVolumeRef
    from torchstore.transport.types import Request


class TransportContext:
    RDMA_TRANSPORT_CACHE = "rdma_transport_cache"

    def __init__(self):
        self.transport_context = {}

    def get_transport_context(self) -> Dict[Any, Any]:
        return self.transport_context

    def get_rdma_transport_cache(self) -> RdmaTransportCache:
        if self.RDMA_TRANSPORT_CACHE not in self.transport_context:
            self.transport_context[self.RDMA_TRANSPORT_CACHE] = RdmaTransportCache()
        return self.transport_context[self.RDMA_TRANSPORT_CACHE]


class TransportBuffer:
    """Abstract base class for transporting tensor data between clients and storage volumes.

    TransportBuffer provides the interface for moving tensor data across process boundaries
    in TorchStore's distributed architecture. Concrete implementations (e.g., MonarchRDMATransportBuffer)
    handle the actual data transport using different mechanisms (RDMA, RPC, etc.).

    Architecture Overview
    ---------------------
    TorchStore operates with a client-server model where:
    - **Client (local)**: The process calling `ts.put()` or `ts.get()`. Runs in the user's actor.
    - **StorageVolume (remote)**: A separate actor process that stores tensor data.

    The TransportBuffer is instantiated on the client side and serialized/sent to the
    StorageVolume. Methods are invoked on both sides during a put/get operation.

    Lifecycle: PUT Operation
    ------------------------
    1. Client creates TransportBuffer via TransportBufferFactory
    2. Client calls `put_to_storage_volume(key, request)` which:
       a. Invokes `_pre_put_hook(request)` [CLIENT] - allocate local buffers, prepare data
       b. Optionally performs handshake if `requires_handshake=True`
       c. Serializes self and sends to StorageVolume
    3. StorageVolume receives buffer and calls `handle_put_request(...)` [STORAGE VOLUME]
       - Reads data from transport mechanism (e.g., RDMA read) into local tensor
       - Returns the tensor to be stored
    4. Client calls `drop()` [CLIENT] - cleanup resources (e.g., deregister RDMA memory)

    Lifecycle: GET Operation
    ------------------------
    1. Client creates TransportBuffer via TransportBufferFactory
    2. Client calls `get_from_storage_volume(key, request)` which:
       a. Invokes `_pre_get_hook(key, request)` [CLIENT] - allocate receive buffers
       b. Optionally performs handshake if `requires_handshake=True`
       c. Serializes self and sends to StorageVolume
    3. StorageVolume receives buffer and calls `handle_get_request(...)` [STORAGE VOLUME]
       - Writes stored tensor data into the transport buffer (e.g., RDMA write)
       - Returns the buffer with data ready to be read
    4. Client calls `_handle_storage_volume_response(response)` [CLIENT]
       - Extracts tensor data from the response buffer
       - Copies into user's tensor if inplace, or returns new tensor
    5. Client calls `drop()` [CLIENT] - cleanup resources

    Methods Called on CLIENT (Local Process)
    ----------------------------------------
    - `__init__`: Initialize buffer with reference to target storage volume
    - `put_to_storage_volume`: Entry point for put operations
    - `get_from_storage_volume`: Entry point for get operations
    - `_pre_put_hook`: Prepare buffers before sending put request
    - `_pre_get_hook`: Prepare buffers before sending get request
    - `_handle_storage_volume_response`: Process response from storage volume
    - `drop`: Cleanup resources (CRITICAL for RDMA to prevent memory leaks)

    Methods Called on STORAGE VOLUME (Remote Process)
    -------------------------------------------------
    - `handle_handshake_request`: Exchange connection info (if requires_handshake=True)
    - `handle_put_request`: Receive tensor data and return it for storage
    - `handle_get_request`: Send stored tensor data back to client

    Implementing a Custom TransportBuffer
    -------------------------------------
    Subclasses must implement:
    - `handle_put_request`: How to receive data on the storage volume
    - `handle_get_request`: How to send data from the storage volume
    - `_handle_storage_volume_response`: How to extract data from response on client

    Optionally override:
    - `_pre_put_hook`: Custom buffer allocation for puts
    - `_pre_get_hook`: Custom buffer allocation for gets (may need metadata fetch)
    - `handle_handshake_request`: If `requires_handshake=True`
    - `drop`: Resource cleanup (especially important for RDMA buffers)

    Properties
    ----------
    requires_handshake : bool
        Property that returns True if a handshake is needed before put/get.
        Override this in subclasses to implement custom handshake logic.
        Default is False.

    Args
    ----
    storage_volume_ref : StorageVolumeRef
        Reference to the target storage volume, including actor handle and transport context.

    See Also
    --------
    MonarchRDMATransportBuffer : RDMA-based implementation from Monarch
    MonarchTransportBuffer : Simple RPC-based implementation (slower but always works).
    """

    def __init__(self, storage_volume_ref: "StorageVolumeRef"):
        self.storage_volume_ref = storage_volume_ref

    @property
    def requires_handshake(self) -> bool:
        """Determine if a handshake is needed before the operation.

        Override this property for custom handshake logic (e.g., cached connections).
        Default implementation returns False.
        """
        return False

    # Client-side interface. Called by the client to send/recv data to the storage volume.
    async def put_to_storage_volume(self, key, request: "Request"):
        l = LatencyTracker("put")
        try:
            # _give concrete implementation a chance to parse the request
            await self._pre_put_hook(request)
            l.track_step("_pre_put_hook")

            if self.requires_handshake:
                handshake_result = (
                    await self.storage_volume_ref.volume.handshake.call_one(self)
                )
                await self._post_handshake(handshake_result)
                l.track_step("handshake")

            await self.storage_volume_ref.volume.put.call(
                key, self, request.meta_only()
            )
            l.track_step("volume.put.call")
        finally:
            await self.drop()
            l.track_step("drop")
            l.track_e2e()

    async def get_from_storage_volume(self, key, request: "Request"):
        try:
            await self._pre_get_hook(key, request)

            if self.requires_handshake:
                handshake_result = (
                    await self.storage_volume_ref.volume.handshake.call_one(self)
                )
                await self._post_handshake(handshake_result)

            # when fetching data, we may need to handle the response from the storage volume
            # TODO: think of a good prefix to differentiate this between remote handlers
            response = await self._handle_storage_volume_response(
                await self.storage_volume_ref.volume.get.call_one(
                    key, self, request.meta_only()
                )
            )
        finally:
            await self.drop()

        return response

    async def _post_handshake(self, handshake_result: Any) -> None:
        """Process the result of a handshake on the client side.

        Called after the storage volume responds to a handshake request.
        Override this to handle handshake results (e.g., connecting to peer).
        """
        pass

    async def _drop(self, response: Any):
        pass

    async def _pre_put_hook(self, request: "Request"):
        pass

    async def _pre_get_hook(self, key: str, request: "Request"):
        pass

    async def _handle_storage_volume_response(self, response: Any) -> Any:
        raise NotImplementedError()

    # StorageVolume handlers -- must be implemented by concrete implementaiton
    # These methods are called by the StorageVolume on the remote side

    async def handle_handshake_request(self) -> None:
        # called on the storage volume side
        raise NotImplementedError()

    async def handle_put_request(
        self,
        ctx: "TransportContext",
        request: "Request",
        maybe_tensor,
    ) -> Any:
        # called on the storage volume side
        raise NotImplementedError()

    async def handle_get_request(self, ctx: "TransportContext", data) -> None:
        # called on the storage volume side
        raise NotImplementedError()

    # Helper methods
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
