# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, TYPE_CHECKING

import torch

from torchstore.logging import LatencyTracker
from torchstore.transport.torchcomms.cache import RdmaTransportCache
from torchstore.transport.types import KeyedRequest, Request

if TYPE_CHECKING:
    from torchstore.strategy import StorageVolumeRef
    from torchstore.transport.shared_memory import SharedMemoryCache


class TransportContext:
    RDMA_TRANSPORT_CACHE = "rdma_transport_cache"
    SHM_CACHE = "shm_cache"

    def __init__(self):
        self.transport_context = {}

    def get_transport_context(self) -> dict[Any, Any]:
        return self.transport_context

    def get_rdma_transport_cache(self) -> RdmaTransportCache:
        if self.RDMA_TRANSPORT_CACHE not in self.transport_context:
            self.transport_context[self.RDMA_TRANSPORT_CACHE] = RdmaTransportCache()
        return self.transport_context[self.RDMA_TRANSPORT_CACHE]

    def get_shm_cache(self) -> "SharedMemoryCache":
        """Get shared memory cache, lazily initializing if needed.

        This cache is used by both storage (for allocation) and client (for attachment).

        Note: Import is inside function to avoid cyclic import
        """
        from torchstore.transport.shared_memory import SharedMemoryCache

        if self.SHM_CACHE not in self.transport_context:
            self.transport_context[self.SHM_CACHE] = SharedMemoryCache()
        return self.transport_context[self.SHM_CACHE]


class TransportBuffer:
    """Abstract base class for transporting tensor data between clients and storage volumes.

    TransportBuffer provides the interface for moving tensor data across process boundaries
    in TorchStore's distributed architecture. Concrete implementations
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
    All put operations go through `put_to_storage_volume(entries)` which accepts a
    list of (key, request) tuples. The base class dispatches to `_put_entries`:

    - If `supports_batch_puts` is True (e.g., SharedMemory), the entire list is
      passed to `_put_entries` in a single call.
    - Otherwise, `_put_entries` is called once per entry with a single-element list.

    `_put_entries(entries)`:
      1. Optionally performs handshake if `requires_handshake(entries)` returns True
      2. Calls `_pre_put_hook(entries)` [CLIENT] - allocate local buffers, prepare data
      3. Sends to StorageVolume via `volume.put.call()`
      4. Calls `drop()` [CLIENT] - cleanup resources

    Lifecycle: GET Operation
    ------------------------
    1. Client creates TransportBuffer
    2. Client calls `get_from_storage_volume(key, request)` which:
       a. Optionally performs handshake
       b. Invokes `_pre_get_hook(key, request)` [CLIENT] - allocate receive buffers
       c. Serializes self and sends to StorageVolume
    3. StorageVolume receives buffer and calls `handle_get_request(...)` [STORAGE VOLUME]
    4. Client calls `_handle_storage_volume_response(response)` [CLIENT]
    5. Client calls `drop()` [CLIENT] - cleanup resources

    Methods Called on CLIENT (Local Process)
    ----------------------------------------
    - `__init__`: Initialize buffer with reference to target storage volume
    - `put_to_storage_volume`: Entry point for put operations (single or batch)
    - `get_from_storage_volume`: Entry point for get operations
    - `_pre_handshake`: Prepare for handshake (if requires_handshake returns True)
    - `_post_handshake`: Process handshake results (if requires_handshake returns True)
    - `_pre_put_hook`: Prepare buffers before sending put request
    - `_pre_get_hook`: Prepare buffers before sending get request
    - `_handle_storage_volume_response`: Process response from storage volume
    - `drop`: Cleanup resources (CRITICAL for RDMA to prevent memory leaks)

    Methods Called on STORAGE VOLUME (Remote Process)
    -------------------------------------------------
    - `recv_handshake`: Exchange connection info (if requires_handshake returns True)
    - `handle_put_request`: Receive tensor data and return it for storage
    - `handle_get_request`: Send stored tensor data back to client

    Implementing a Custom TransportBuffer
    -------------------------------------
    Subclasses must implement:
    - `handle_put_request`: How to receive data on the storage volume
    - `handle_get_request`: How to send data from the storage volume
    - `_handle_storage_volume_response`: How to extract data from response on client

    Optionally override:
    - `supports_batch_puts`: Set True if the transport can handle multiple entries at once
    - `requires_handshake`: Return True if a handshake is needed before put/get
    - `_pre_put_hook`: Custom buffer allocation for puts
    - `_pre_get_hook`: Custom buffer allocation for gets (may need metadata fetch)
    - `recv_handshake`: If `requires_handshake` returns True
    - `drop`: Resource cleanup (especially important for RDMA buffers)

    Attributes
    ----------
    supports_inplace_resharding : bool
        Whether this transport supports inplace resharding.
    supports_batch_puts : bool
        If True, `put_to_storage_volume` passes all entries to `_put_entries`
        in a single call. If False (default), entries are dispatched one at a time.

    Parameters
    ----------
    storage_volume_ref : StorageVolumeRef
        Reference to the target storage volume, including actor handle and transport context.

    """

    supports_inplace_resharding: bool = True
    supports_batch_puts: bool = False

    def __init__(self, storage_volume_ref: "StorageVolumeRef"):
        self.storage_volume_ref = storage_volume_ref

    def requires_handshake(self, entries: list[KeyedRequest]) -> bool:
        """Determine if a handshake is needed before the operation.

        Override this method for custom handshake logic (e.g., cached connections).
        This method may have side effects (e.g., allocating resources for the handshake).
        Default implementation returns False.

        Args:
            entries: List of KeyedRequest for the current operation.
        """
        return False

    # Client-side interface. Called by the client to send/recv data to the storage volume.
    async def put_to_storage_volume(self, entries: list[KeyedRequest]) -> None:
        if self.supports_batch_puts:
            await self._put_entries(entries)
        else:
            for entry in entries:
                await self._put_entries([entry])

    async def _put_entries(self, entries: list[KeyedRequest]) -> None:
        l = LatencyTracker("put")
        meta_requests = [e.meta_only() for e in entries]
        try:
            if self.requires_handshake(entries):
                await self._pre_handshake()
                l.track_step("pre_handshake")
                handshake_results = (
                    await self.storage_volume_ref.volume.handshake.call_one(
                        self, meta_requests
                    )
                )
                l.track_step("volume.handshake.call")
                await self._post_handshake(handshake_results, entries)
                l.track_step("post_handshake")

            await self._pre_put_hook(entries)
            l.track_step("_pre_put_hook")

            await self.storage_volume_ref.volume.put.call(self, meta_requests)
            l.track_step("volume.put.call")
        finally:
            await self.drop()
            l.track_step("drop")
            l.track_e2e()

    # batching not supported on get yet
    async def get_from_storage_volume(self, key, request: Request):
        try:
            entries = [KeyedRequest(key, request)]
            if self.requires_handshake(entries):
                await self._pre_handshake()
                handshake_results = (
                    await self.storage_volume_ref.volume.handshake.call_one(
                        self, [KeyedRequest(key, request.meta_only())]
                    )
                )
                await self._post_handshake(handshake_results, entries)

            await self._pre_get_hook(key, request)

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

    async def _pre_handshake(self) -> None:
        """Prepare for handshake on the client side.

        Called before the handshake request is sent to the storage volume.
        Override this to perform any setup needed prior to handshake
        (e.g., allocating resources, preparing connection info).
        """
        pass

    async def _post_handshake(
        self,
        handshake_results: list[Any],
        entries: list[KeyedRequest],
    ) -> None:
        """Process the results of a handshake on the client side.

        Called after the storage volume responds to a handshake request.
        Override this to handle handshake results (e.g., connecting to peer).

        Args:
            handshake_results: List of results from recv_handshake, one per entry.
            entries: The original KeyedRequest entries that were handshaked.
        """
        pass

    async def _drop(self, response: Any):
        pass

    async def _pre_put_hook(self, entries: list[KeyedRequest]):
        pass

    async def _pre_get_hook(self, key: str, request: Request):
        pass

    async def _handle_storage_volume_response(self, response: Any) -> Any:
        raise NotImplementedError()

    # StorageVolume handlers -- must be implemented by concrete implementaiton
    # These methods are called by the StorageVolume on the remote side

    async def recv_handshake(
        self,
        ctx: "TransportContext",
        entries: list[tuple[KeyedRequest, Any]],
    ) -> list[Any]:
        # called on the storage volume side
        raise NotImplementedError()

    async def handle_put_request(
        self,
        ctx: "TransportContext",
        entries: list[tuple[KeyedRequest, Any]],
    ) -> dict[str, Any]:
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
