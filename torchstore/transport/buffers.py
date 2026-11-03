# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import collections
import dataclasses
import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, TYPE_CHECKING, TypeVar

import torch

from torchstore.logging import LatencyTracker
from torchstore.transport.types import Request

if TYPE_CHECKING:
    from torchstore.strategy import StorageVolumeRef

logger = logging.getLogger(__name__)


class TransportCache(ABC):
    """Base class for per-transport caches stored in TransportContext."""

    def delete(self, keys: set[str]) -> None:
        """Delete per-key cache entries.

        Most transport caches are not keyed by TorchStore key, so their default
        behavior is a no-op.
        """
        return

    @abstractmethod
    def clear(self) -> None:
        ...


T = TypeVar("T", bound=TransportCache)


class TransportContext:
    """Generic type-keyed registry for per-transport caches.

    Each transport defines its own cache class (extending TransportCache)
    and accesses it via ctx.get(MyCacheType). Caches are lazily created
    on first access. Adding a new transport does NOT require modifying
    this class.
    """

    def __init__(self) -> None:
        self._caches: dict[type[TransportCache], TransportCache] = {}

    def get(self, cache_type: type[T]) -> T:
        """Get or lazily create a cache by its type (calls cache_type() if new).

        Threads that create one at once all get the cache stored first.
        """
        cache = self._caches.get(cache_type)
        if cache is None:
            cache = self._caches.setdefault(cache_type, cache_type())
        return cache  # type: ignore[return-value]

    def clear(self) -> None:
        """Clear all caches."""
        for cache in self._caches.values():
            cache.clear()
        self._caches.clear()

    def delete(self, keys: str | Iterable[str]) -> None:
        """Delete cache entries associated with one or more TorchStore keys."""
        key_set = {keys} if isinstance(keys, str) else set(keys)
        if not key_set:
            return
        for cache in self._caches.values():
            cache.delete(key_set)


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
    All put operations go through `put_to_storage_volume(requests)` which accepts a
    list of Request entries. The base class dispatches to `_put_requests`:

    - If `supports_batch_puts` is True (e.g., SharedMemory), the entire list is
      passed to `_put_requests` in a single call.
    - Otherwise, `_put_requests` is called once per entry with a single-element list.

    `_put_requests(requests)`:
    1. Runs `perform_handshake(...)` if `requires_handshake(requests)` returns True
    2. Calls `_pre_put_hook(requests)` [CLIENT] - allocate local buffers, prepare data
    3. Sends to StorageVolume via `volume.put.call()`
    4. Client calls `drop()` [CLIENT] - cleanup resources (e.g., deregister RDMA memory)

    Lifecycle: GET Operation
    ------------------------
    All get operations go through `get_from_storage_volume(requests)` which accepts a
    list of Request entries. The base class dispatches to `_get_requests`:

    - If `supports_batch_gets` is True (e.g., SharedMemory), the entire list is
      passed to `_get_requests` in a single call.
    - Otherwise, `_get_requests` is called once per entry with a single-element list.

    `_get_requests(requests)`:
      1. Runs `perform_handshake(...)` if `requires_handshake(requests)` returns True
      2. Calls `_pre_get_hook(requests)` [CLIENT] - save metadata for response handling
      3. Sends to StorageVolume via `_call_volume_get(meta_requests)`
      4. StorageVolume calls `handle_get_request(ctx, entries)` [STORAGE VOLUME]
      5. Client calls `_handle_storage_volume_response(requests, transport_buffer)` [CLIENT]
      6. Calls `drop()` [CLIENT] - cleanup resources

    Methods Called on CLIENT (Local Process)
    ----------------------------------------
    - `__init__`: Initialize buffer with reference to target storage volume
    - `put_to_storage_volume`: Entry point for put operations (single or batch)
    - `get_from_storage_volume`: Entry point for get operations
    - `perform_handshake`: Orchestrate handshake setup when required
    - `_pre_handshake`: Prepare for default one-RPC handshake
    - `_post_handshake`: Process default one-RPC handshake results
    - `_post_request_success`: Publish state only after the storage-volume call succeeds
    - `_pre_put_hook`: Prepare buffers before sending put request
    - `_pre_get_hook`: Prepare buffers before sending get request
    - `_handle_storage_volume_response`: Process response from storage volume
    - `drop`: Cleanup resources (CRITICAL for RDMA to prevent memory leaks)

    Methods Called on STORAGE VOLUME (Remote Process)
    -------------------------------------------------
    The base `__getstate__` sets `storage_volume_ref` to None in the copy the storage
    volume works on, in the reply and in any copy that pickle or the `copy` module makes
    through it. These methods use `ctx` and the fields the client set instead. A class in the
    MRO with its own `__reduce__`, `__reduce_ex__`, `__copy__` or `__deepcopy__` can bypass it and
    keep the handle. A state other than a dict, which only a hook of a later base class returns,
    passes through unchanged with the handle.

    - `recv_handshake`: Exchange connection info (if requires_handshake returns True)
    - `handle_put_request`: Receive tensor data and return list aligned with entries
    - `handle_get_request`: Send stored tensor data back to client
    - `handle_refused_put` / `handle_refused_get`: End the client's half of a request the store refuses

    Implementing a Custom TransportBuffer
    -------------------------------------
    Subclasses must implement:
    - `handle_put_request`: How to receive data on the storage volume (returns list aligned with entries)
    - `handle_get_request`: How to send data from the storage volume
    - `_handle_storage_volume_response`: How to extract data from response on client

    Optionally override:
    - `supports_batch_puts`: Set True if the transport can handle multiple entries at once for puts
    - `supports_batch_gets`: Set True if the transport can handle multiple entries at once for gets
    - `requires_handshake`: Return True if a handshake is needed before put/get
    - `perform_handshake`: Override for multi-step handshakes; default is one RPC
    - `_pre_put_hook`: Custom buffer allocation for puts
    - `_pre_get_hook`: Custom buffer allocation for gets (may need metadata fetch)
    - `_call_volume_get`: Custom handling of the get RPC (e.g. cancellation)
    - `recv_handshake`: StorageVolume side of the default handshake
    - `handle_refused_put`, `handle_refused_get`: End client halves that wait for the volume (e.g., Gloo)
    - `drop`: Resource cleanup (especially important for RDMA buffers). A buffer without one skips it
    - `__getstate__`: Start from `super().__getstate__()`, a copy of the instance dict, or of a dict
      the hook of a class between `TransportBuffer` and `object` returns, with the client-only volume
      handle set to None. That hook's other state comes back as it built it, handle included. A
      subclass that keeps fields in `__slots__` adds them there and restores them in its own `__setstate__`

    Attributes
    ----------
    supports_inplace_resharding : bool
        Whether this transport supports inplace resharding.
    supports_batch_puts : bool
        If True, `put_to_storage_volume` passes all requests to `_put_requests`
        in a single call. If False (default), requests are dispatched one at a time.
    supports_batch_gets : bool
        If True, `get_from_storage_volume` passes all requests to `_get_requests`
        in a single call. If False (default), requests are dispatched one at a time.
    casts_on_put : bool
        If True, the transport applies `Request.transfer_dtype` while copying put
        data. If False (default), `_put_requests` casts each dispatched batch
        before the transport sees it, so batched transports hold a cast copy of
        the whole batch.

    Parameters
    ----------
    storage_volume_ref : StorageVolumeRef
        Reference to the target storage volume, including actor handle and transport context.
        Copies the base `__getstate__` makes from a dict state hold None.

    """

    supports_inplace_resharding: bool = True

    # Transitionary period. These should eventually be TRUE for all transports.
    supports_batch_puts: bool = False
    supports_batch_gets: bool = False

    casts_on_put: bool = False

    def __init__(self, storage_volume_ref: "StorageVolumeRef"):
        self.storage_volume_ref = storage_volume_ref

    def __getstate__(self) -> Any:
        """Omit the client-only volume handle and the transport caches it holds from a dict state.

        Starts from the hook of a base class between this one and ``object`` in the MRO if one defines it,
        else from the instance dict. A dict state comes back as a copy with the handle set to None if set:
        an ``OrderedDict`` or ``defaultdict`` as one again and any other dict as a plain dict. Other states pass through.
        """
        mro = type(self).__mro__
        # The defining class comes from its cell because a reload of this module rebinds the global.
        later = mro[mro.index(__class__) + 1 : -1]  # type: ignore[name-defined]
        if any("__getstate__" in vars(base) for base in later):
            state = super().__getstate__()
        else:
            state = self.__dict__
        if isinstance(state, dict):
            if isinstance(state, collections.OrderedDict):
                state = collections.OrderedDict(state)
            elif isinstance(state, collections.defaultdict):
                state = collections.defaultdict(state.default_factory, state)
            else:
                state = dict(state)
            if "storage_volume_ref" in state:
                state["storage_volume_ref"] = None
        return state

    def requires_handshake(self, requests: list[Request]) -> bool:
        """Determine if a handshake is needed before the operation.

        Override this method for custom handshake logic (e.g., cached connections).
        This method may have side effects (e.g., allocating resources for the handshake).
        Default implementation returns False.

        Args:
            requests: List of Request for the current operation.
        """
        return False

    # Client-side interface. Called by the client to send/recv data to the storage volume.
    async def put_to_storage_volume(self, requests: list[Request]) -> None:
        if self.supports_batch_puts:
            await self._put_requests(requests)
        else:
            for request in requests:
                await self._put_requests([request])

    async def _put_requests(self, requests: list[Request]) -> None:
        l = LatencyTracker("put")
        if not self.casts_on_put:
            requests = _apply_transfer_dtypes(requests)
        meta_requests = [r.meta_only() for r in requests]
        raised = False
        try:
            if self.requires_handshake(requests):
                await self.perform_handshake(requests, meta_requests, l)

            await self._pre_put_hook(requests)
            l.track_step("_pre_put_hook")

            await self.storage_volume_ref.volume.put.call(self, meta_requests)
            l.track_step("volume.put.call")

            # Success-only hook: transports can promote staged state here,
            # while drop() still handles cleanup on both success and failure.
            await self._post_request_success()
            l.track_step("_post_request_success")
        except BaseException:
            raised = True
            raise
        finally:
            await self._drop_after_call(l, raised)

    async def get_from_storage_volume(self, requests: list[Request]) -> list[Any]:
        if self.supports_batch_gets:
            return await self._get_requests(requests)
        else:
            results = []
            for request in requests:
                results.extend(await self._get_requests([request]))
            return results

    async def _get_requests(self, requests: list[Request]) -> list[Any]:
        l = LatencyTracker("get")
        meta_requests = [r.meta_only() for r in requests]
        raised = False
        try:
            if self.requires_handshake(requests):
                await self.perform_handshake(requests, meta_requests, l)

            await self._pre_get_hook(requests)
            l.track_step("_pre_get_hook")

            response = await self._handle_storage_volume_response(
                requests, await self._call_volume_get(meta_requests)
            )
            l.track_step("volume.get.call")

            await self._post_request_success()
            l.track_step("_post_request_success")
        except BaseException:
            raised = True
            raise
        finally:
            await self._drop_after_call(l, raised)
        return response

    async def _call_volume_get(self, meta_requests: list[Request]) -> "TransportBuffer":
        return await self.storage_volume_ref.volume.get.call_one(self, meta_requests)

    async def _drop_after_call(
        self, latency_tracker: LatencyTracker, raised: bool
    ) -> None:
        """Run ``drop`` if the buffer has one, raising an ``Exception`` from it only when the call raised nothing.

        After a call that raised or was cancelled, such an ``Exception`` is logged at WARNING instead,
        so the call's own error or cancel propagates. A cancel or other ``BaseException`` from that
        ``drop`` propagates in its place.
        """
        try:
            # Looked up on each call so a drop from a mixin after this class or from __getattr__ runs.
            drop = getattr(self, "drop", None)
            if drop is not None:
                await drop()
        except Exception:
            if not raised:
                raise
            logger.warning(
                "%s.drop() failed after its call raised",
                type(self).__name__,
                exc_info=True,
            )
        else:
            latency_tracker.track_step("drop")
            latency_tracker.track_e2e()

    async def perform_handshake(
        self,
        requests: list[Request],
        meta_requests: list[Request],
        latency_tracker: LatencyTracker | None = None,
    ) -> None:
        """Run the transport handshake.

        The default implementation preserves the single-RPC handshake behavior.
        Transports with multi-stage setup can override this method.
        """
        tracker = latency_tracker or LatencyTracker("handshake")
        await self._pre_handshake()
        tracker.track_step("pre_handshake")

        handshake_results = await self.storage_volume_ref.volume.handshake.call_one(
            self, meta_requests
        )
        tracker.track_step("volume.handshake.call")

        await self._post_handshake(handshake_results, requests)
        tracker.track_step("post_handshake")

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
        requests: list[Request],
    ) -> None:
        """Process the result of a handshake on the client side.

        Called after the storage volume responds to a handshake request.
        Override this to handle handshake results (e.g., connecting to peer).
        """
        pass

    async def _post_request_success(self) -> None:
        """Run after the remote put/get succeeds but before request cleanup.

        Use this for committing request-scoped resources into longer-lived
        caches. ``drop()`` runs in ``finally`` and should still clean up any
        state that was not published here.
        """
        pass

    async def _pre_put_hook(self, requests: list[Request]):
        pass

    async def _pre_get_hook(self, requests: list[Request]):
        pass

    async def _handle_storage_volume_response(
        self, requests: list[Request], transport_buffer: "TransportBuffer"
    ) -> list[Any]:
        raise NotImplementedError()

    # StorageVolume handlers -- must be implemented by concrete implementaiton
    # These methods are called by the StorageVolume on the remote side

    async def recv_handshake(
        self,
        ctx: "TransportContext",
        entries: list[tuple[Request, Any]],
    ) -> list[Any]:
        # called on the storage volume side
        raise NotImplementedError()

    async def handle_put_request(
        self,
        ctx: "TransportContext",
        entries: list[tuple[Request, Any]],
    ) -> list[Any]:
        # called on the storage volume side
        raise NotImplementedError()

    async def handle_get_request(
        self,
        ctx: "TransportContext",
        entries: list[tuple[Request, Any]],
    ) -> None:
        # called on the storage volume side
        raise NotImplementedError()

    async def handle_refused_put(
        self,
        ctx: "TransportContext",
        requests: list[Request],
    ) -> None:
        """Called on the storage volume when the store's ``put`` refuses a request before ``handle_put_request``.

        A transport whose client half waits for the volume's half ends it here. The store logs an
        ``Exception`` from this hook at WARNING and still raises its refusal. The default does nothing.
        """
        pass

    async def handle_refused_get(
        self,
        ctx: "TransportContext",
        requests: list[Request],
    ) -> None:
        """Called on the storage volume when the store's ``get`` refuses a request before ``handle_get_request``.

        A transport whose client half waits for the volume's half ends it here. The store logs an
        ``Exception`` from this hook at WARNING and still raises its refusal. The default does nothing.
        """
        pass

    # Helper methods
    def _assert_valid_tensor(
        self,
        tensor: torch.Tensor,
        dtype: torch.dtype,
        shape: torch.Size,
        must_be_contiguous=True,
    ) -> None:
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == dtype, f"{tensor.dtype} != {dtype}"
        assert tensor.shape == shape, f"{tensor.shape} != {shape}"
        assert not must_be_contiguous or tensor.is_contiguous()


def _apply_transfer_dtypes(requests: list[Request]) -> list[Request]:
    """Return requests whose tensor_val is already in its transfer dtype."""
    cast_requests = []
    cuda_devices = set()
    for request in requests:
        if request.transfer_dtype is None or request.tensor_val is None:
            cast_requests.append(request)
            continue
        tensor = request.tensor_val.to(request.transfer_dtype)
        if tensor.is_cuda:
            cuda_devices.add(tensor.device)
        cast_requests.append(
            dataclasses.replace(request, tensor_val=tensor, transfer_dtype=None)
        )
    # Transports may read the cast tensors outside CUDA stream order (e.g. RDMA).
    for device in cuda_devices:
        torch.cuda.current_stream(device).synchronize()
    return cast_requests
