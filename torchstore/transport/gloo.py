# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import itertools
import socket
import uuid
from datetime import timedelta
from logging import getLogger
from typing import Any, TYPE_CHECKING

import portpicker
import torch
from torch.distributed import ProcessGroup, ProcessGroupGloo, Store, TCPStore

from torchstore.transport.buffers import TransportBuffer, TransportCache
from torchstore.transport.types import Request

if TYPE_CHECKING:
    from torchstore.transport.buffers import TransportContext
    from torchstore.transport.pipe import StorageVolumeRef
import os

import torch.distributed as dist

logger = getLogger(__name__)

# Global cache
_store_addrs: dict[
    str, tuple[str, int, str, bool]
] = {}  # volume_id -> (master_addr, master_port, store_key, tagged)
# Gives each tensor transfer to a volume that takes tags its own Gloo tag, so transfers in flight
# on one pair keep their own bytes. Tag 0 stays for clients and volumes before tags.
_tags = itertools.count()
# The largest tag torch accepts.
_MAX_TAG = 0x7FFFFFFF
# A volume's handshake reply when it moves each tensor on its buffer's tag.
_TAGS = "tags"


TORCHSTORE_GLOO_ENABLED = os.environ.get("TORCHSTORE_GLOO_ENABLED", "1") == "1"
TORCHSTORE_GLOO_INIT_TIMEOUT = int(
    os.environ.get("TORCHSTORE_GLOO_INIT_TIMEOUT", "120")
)


def gloo_available() -> bool:
    """Check if gloo transport is available and enabled.

    Returns True if:
    1. TORCHSTORE_GLOO_ENABLED environment variable is set to "1"
    2. torch.distributed is available
    3. gloo backend is available

    """
    return TORCHSTORE_GLOO_ENABLED and dist.is_gloo_available()


def _get_hostname() -> str:
    """Get the fully qualified hostname of the local machine.

    Uses FQDN to ensure hostname is resolvable from other nodes in multi-node setups.
    """
    return socket.getfqdn()


def _gloo_factory(
    store: Store,
    rank: int,
    world_size: int,
    timeout: timedelta,
    device: torch.device | None = None,
) -> ProcessGroup:
    """
    Create a ProcessGroup with Gloo backend.

    Based on:
    https://github.com/pytorch/pytorch/blob/92284fb2ff44f09a9c7df0d8cf6cac9903e376a4/torch/distributed/_dist2.py#L64
    """
    backend_class = ProcessGroupGloo(store, rank, world_size, timeout)
    backend_class._set_sequence_number_for_group()

    pg = ProcessGroup(store, rank, world_size)
    pg._set_default_backend(ProcessGroup.BackendType.GLOO)

    # Register devices
    if device is not None:
        pg._register_backend(device, ProcessGroup.BackendType.GLOO, backend_class)

    pg._register_backend(
        torch.device("cpu"), ProcessGroup.BackendType.GLOO, backend_class
    )
    if torch.cuda.is_available():
        pg._register_backend(
            torch.device("cuda"), ProcessGroup.BackendType.GLOO, backend_class
        )
    return pg


def _log_receive_failure(receive: asyncio.Task) -> None:
    """Log at WARNING the failure of a receive its get did not await."""
    if not receive.cancelled():
        try:
            receive.result()
        except Exception:
            logger.warning("A Gloo receive its get did not await failed", exc_info=True)


class GlooProcessGroupCache(TransportCache):
    """Cache for Gloo process groups, keyed by store_key."""

    def __init__(self) -> None:
        self._process_groups: dict[str, ProcessGroup] = {}

    def put(self, store_key: str, pg: ProcessGroup) -> None:
        self._process_groups[store_key] = pg

    def get(self, store_key: str) -> ProcessGroup:
        return self._process_groups[store_key]

    def clear(self) -> None:
        self._process_groups.clear()


class GlooTransportBuffer(TransportBuffer):
    """Transport buffer implementation using PyTorch distributed (gloo backend) for tensor transfer.

    This buffer creates a dedicated 2-process gloo process group between the client
    and storage volume for each connection using TCPStore for coordination.

    Prerequisites:
    - TORCHSTORE_GLOO_ENABLED=1 environment variable must be set
    - Network connectivity between client and storage (for TCPStore)
    """

    supports_inplace_resharding = False
    # The tag of the buffer's current tensor transfer, 0 for an object or a pair without tags. A class
    # default, as buffers that clients before tags pickle carry none and move on tag 0.
    tag: int = 0

    def __init__(self, storage_volume_ref: "StorageVolumeRef") -> None:
        super().__init__(storage_volume_ref)

        # Tensor metadata
        self.shape: torch.Size | None = None
        self.dtype: torch.dtype | None = None

        # Process group coordination
        self.master_addr: str | None = None
        self.master_port: int | None = None
        self.store_key: str | None = None  # Unique key for this connection

        # Set by the client when it moves an object, which posts no tensor half, and by the volume
        # when it answers a get in band in objects, which clients before tags read too.
        self.is_object: bool = False
        self.objects: Any = None  # A get's value when it comes back in band

        # TCPStore reference (kept alive until PG is created)
        self._tcp_store: TCPStore | None = None
        self._pg_task: asyncio.Task | None = None  # Background task for PG creation
        self._send_task: asyncio.Task | None = (
            None  # Background task for sending tensor
        )
        self._recv_task: asyncio.Task | None = (
            None  # Background task for receiving tensor
        )

    def requires_handshake(self, requests: list[Request]) -> bool:
        """Determine if a handshake is needed.

        Returns False if a cached connection already exists for this volume,
        True otherwise.
        """
        volume_id = self.storage_volume_ref.volume_id

        # Check if connection already exists
        if volume_id in _store_addrs:
            cached_addr = _store_addrs[volume_id]
            self.master_addr = cached_addr[0]
            self.master_port = cached_addr[1]
            self.store_key = cached_addr[2]
            return False

        return True

    async def _pre_handshake(self) -> None:
        """Prepare for handshake. Create TCPStore and start PG creation on
        client side (rank 0).

        The TCPStore is created here so it's ready to accept connections
        when the handshake RPC reaches the storage volume. The ProcessGroup
        creation is started as a background task so it runs concurrently with
        the handshake RPC, allowing both sides to rendezvous.
        """
        volume_id = self.storage_volume_ref.volume_id

        # Generate unique store key and find free port
        self.store_key = f"torchstore_gloo_{str(uuid.uuid4())[:8]}"
        self.master_addr = _get_hostname()
        self.master_port = portpicker.pick_unused_port()

        logger.info(
            f"Initiating gloo handshake with StorageVolume:[{volume_id}] "
            f"using TCPStore at {self.master_addr}:{self.master_port}"
        )

        # Create TCPStore as master (non-blocking with wait_for_workers=False)
        self._tcp_store = TCPStore(
            host_name=self.master_addr,
            port=self.master_port,
            world_size=2,
            is_master=True,
            timeout=timedelta(seconds=TORCHSTORE_GLOO_INIT_TIMEOUT),
            wait_for_workers=False,
        )

        # Start PG creation in background so it runs concurrently with handshake RPC
        tcp_store = self._tcp_store

        def create_pg():
            return _gloo_factory(
                store=tcp_store,
                rank=0,
                world_size=2,
                timeout=timedelta(seconds=TORCHSTORE_GLOO_INIT_TIMEOUT),
                device=torch.device("cpu"),
            )

        self._pg_task = asyncio.create_task(asyncio.to_thread(create_pg))

    def __getstate__(self) -> dict[str, Any]:
        """Serialize state, excluding non-serializable fields."""
        state = self.__dict__.copy()
        state["storage_volume_ref"] = None
        state["_tcp_store"] = None
        state["_pg_task"] = None
        state["_send_task"] = None
        state["_recv_task"] = None
        return state

    async def recv_handshake(
        self,
        ctx: "TransportContext",
        entries: list[tuple[Request, Any]],
    ) -> list[str]:
        """Called on storage volume side to set up the process group.

        Creates TCPStore and ProcessGroup on storage side (rank 1).

        transport_context: the TransportContext from the StorageVolume to which the pg will be added

        """
        logger.info(
            f"Storage volume setting up gloo process group with TCPStore at "
            f"{self.master_addr}:{self.master_port}"
        )

        # Create TCPStore and ProcessGroup on storage side (rank 1, worker)
        # Run in thread to avoid blocking event loop
        master_addr = self.master_addr
        master_port = self.master_port

        def create_pg():
            tcp_store = TCPStore(
                host_name=master_addr,
                port=master_port,
                world_size=2,
                is_master=False,
                timeout=timedelta(seconds=TORCHSTORE_GLOO_INIT_TIMEOUT),
            )
            return _gloo_factory(
                store=tcp_store,
                rank=1,
                world_size=2,
                timeout=timedelta(seconds=TORCHSTORE_GLOO_INIT_TIMEOUT),
                device=torch.device("cpu"),
            )

        pg = await asyncio.to_thread(create_pg)

        # Cache the process group
        ctx.get(GlooProcessGroupCache).put(self.store_key, pg)

        logger.info(
            f"Storage volume finished gloo process group setup for store_key={self.store_key}"
        )

        return [_TAGS]

    async def _post_handshake(
        self,
        handshake_results: list[Any],
        requests: list[Request],
    ) -> None:
        """Await ProcessGroup creation that was started in _pre_handshake.

        The PG creation was started concurrently with the handshake RPC,
        so we just await the result here.
        """
        volume_id = self.storage_volume_ref.volume_id

        # Await the PG creation that was started in _pre_handshake
        pg = await self._pg_task

        # Cache the connection info and process group. Volumes before tags answer [None].
        _store_addrs[volume_id] = (
            self.master_addr,
            self.master_port,
            self.store_key,
            _TAGS in handshake_results,
        )
        self.storage_volume_ref.transport_context.get(GlooProcessGroupCache).put(
            self.store_key, pg
        )

        # Clear references
        self._tcp_store = None
        self._pg_task = None

        logger.info(f"Finished gloo handshake with StorageVolume:[{volume_id}]")

    def _draw_tag(self) -> int:
        """Return a tag of its own for a tensor transfer if the cached pair takes tags, else 0."""
        if not _store_addrs[self.storage_volume_ref.volume_id][3]:
            return 0
        return next(_tags) % _MAX_TAG + 1

    async def _pre_put_hook(self, requests: list[Request]) -> None:
        """Start sending tensor before put RPC.

        Called after handshake completes, before put.call().
        Starts the send as a background task so it runs concurrently
        with the storage volume's recv in handle_put_request.
        """
        assert len(requests) == 1
        request = requests[0]
        # Cleared for each transfer, as the volume ends a half on any tag but 0 and an object has none.
        self.tag = 0

        # Check if this is an object (non-tensor) PUT
        if request.is_object:
            self.is_object = True
            return

        # Extract tensor from request
        if request.tensor_val is None:
            return

        tensor = request.tensor_val
        self.shape = tensor.shape
        self.dtype = tensor.dtype
        self.tag = self._draw_tag()

        # Start send in background - will run concurrently with put RPC
        self._send_task = asyncio.create_task(
            self._send_tensor(
                tensor,
                self.storage_volume_ref.transport_context,
                self.tag,
            )
        )

    async def handle_put_request(
        self,
        ctx: "TransportContext",
        entries: list[tuple[Request, Any]],
    ) -> list[Any]:
        """Called by storage volume. Receive tensor from client via gloo process group."""
        assert len(entries) == 1
        request, maybe_tensor = entries[0]

        if request.is_object:
            self.is_object = True
            return [request.objects]

        # Allocate destination tensor if needed
        tensor = maybe_tensor
        if tensor is None:
            tensor = torch.empty(
                self.shape, dtype=self.dtype, device=torch.device("cpu")
            )

        # Receive tensor from client via gloo
        tensor = await self._receive_tensor(tensor, ctx, self.tag)

        return [tensor]

    async def handle_refused_put(
        self,
        ctx: "TransportContext",
        requests: list[Request],
    ) -> None:
        """Called by storage volume. Receive the client's tensor into scratch so its send ends.

        Only for a send on a tag of its own, as on tag 0 the scratch could take another transfer's send.
        The scratch takes the tensor's full size, since a smaller receive would abort this process.
        """
        if self.tag:
            scratch = torch.empty(
                self.shape, dtype=self.dtype, device=torch.device("cpu")
            )
            await self._receive_tensor(scratch, ctx, self.tag)

    async def _pre_get_hook(self, requests: list[Request]) -> None:
        """Start receiving tensor before get RPC.

        Called after handshake completes, before get.call().
        Starts the recv as a background task so it runs concurrently
        with the storage volume's send in handle_get_request.
        """
        assert len(requests) == 1
        request = requests[0]
        # Cleared for each transfer, as the volume ends a half on any tag but 0 and an object has none.
        self.tag = 0
        # TODO: support in place get with receiving directly to request.tensor_val

        # Fetch metadata from storage volume
        meta = (
            await self.storage_volume_ref.volume.get_meta.call_one(
                [request.meta_only()]
            )
        )[0]

        # If request specifies a tensor slice, use its shape
        if request.tensor_slice is not None:
            self.shape = torch.Size(request.tensor_slice.local_shape)
            self.dtype = meta[1]
        else:
            if isinstance(meta, str) or meta is None:
                # It's an object, not a tensor
                self.is_object = True
                return
            # meta is (shape, dtype)
            self.shape = meta[0]
            self.dtype = meta[1]

        tensor = torch.empty(self.shape, dtype=self.dtype, device=torch.device("cpu"))
        self.tag = self._draw_tag()

        # Start recv in background - will run concurrently with get RPC
        # The storage volume will send the tensor in handle_get_request
        self._recv_task = asyncio.create_task(
            self._receive_tensor(
                tensor,
                self.storage_volume_ref.transport_context,
                self.tag,
            )
        )

    async def handle_get_request(
        self,
        ctx: "TransportContext",
        entries: list[tuple[Request, Any]],
    ) -> None:
        """Called by storage volume. Send tensor to client via gloo process group."""
        assert len(entries) == 1
        _, data = entries[0]
        # A get the client took for an object posted no receive, so its data goes in band.
        if not self.is_object:
            if (
                isinstance(data, torch.Tensor)
                and data.shape == self.shape
                and data.dtype == self.dtype
            ):
                # Send the tensor to client via gloo
                await self._send_tensor(data, ctx, self.tag)
                return
            # The data does not fit the client's receive, as when the key changed after the client's
            # metadata fetch or a slice runs past the tensor's end, so it goes in band and an empty
            # send ends the receive. A client before tags shares tag 0 among its transfers, so it gets
            # that send only where earlier volumes sent the data itself.
            if self.tag or isinstance(data, torch.Tensor):
                try:
                    await self._end_client_receive(ctx)
                except Exception:
                    logger.warning(
                        "Ending a Gloo get's receive failed, and the get's value still goes back in band",
                        exc_info=True,
                    )
        self.is_object = True
        self.objects = data

    async def handle_refused_get(
        self,
        ctx: "TransportContext",
        requests: list[Request],
    ) -> None:
        """Called by storage volume. End the client's receive with an empty send.

        Only for a receive on a tag of its own, as on tag 0 the send could reach another transfer's receive.
        """
        if self.tag:
            await self._end_client_receive(ctx)

    async def _end_client_receive(self, ctx: "TransportContext") -> None:
        # An empty send ends a receive of any size and can never be larger than the receive it meets.
        await self._send_tensor(torch.empty(0), ctx, self.tag)

    async def _handle_storage_volume_response(
        self, requests: list[Request], transport_buffer: "TransportBuffer"
    ) -> list[Any]:
        """Process the response from storage volume after get.

        The tensor data was received via gloo in the background recv task.
        """
        if transport_buffer.is_object:
            return [transport_buffer.objects]

        # Wait for the recv to complete
        if self._recv_task is not None:
            # Cleared before the await, as drop() treats a receive still set as one no get awaited.
            receive, self._recv_task = self._recv_task, None
            tensor = await receive
            if tensor is None:
                raise RuntimeError(
                    f"receive_tensor returned None (is_object={self.is_object}, "
                    f"shape={self.shape}, dtype={self.dtype})"
                )
            return [tensor]

        raise RuntimeError(f"No recv task available (is_object={self.is_object})")

    async def _receive_tensor(
        self, tensor: torch.Tensor, transport_context: "TransportContext", tag: int
    ) -> torch.Tensor:
        """Receive tensor via gloo. Waits for recv to complete before returning.

        Args:
            tensor: Pre-allocated destination tensor to receive into.
            transport_context: Transport context containing the process group.
            tag: The Gloo tag of the transfer.

        Returns:
            The received tensor (same object as input, with data overwritten).
        """
        # Gloo TCP transport requires CPU tensors for recv
        if tensor.device.type != "cpu":
            tensor = torch.zeros(
                tensor.shape, dtype=tensor.dtype, device=torch.device("cpu")
            )

        # Get process group from transport context
        pg = transport_context.get(GlooProcessGroupCache).get(self.store_key)

        # Determine remote rank based on our rank in the PG
        my_rank = pg.rank()
        remote_rank = 1 - my_rank

        logger.debug(
            f"receive_tensor: receiving tensor shape={tensor.shape} dtype={tensor.dtype} "
            f"from rank {remote_rank} (my_rank={my_rank}, store_key={self.store_key})"
        )

        # Run recv in thread and wait for completion
        def do_recv():
            work = pg.recv([tensor], srcRank=remote_rank, tag=tag)
            work.wait()

        await asyncio.to_thread(do_recv)

        logger.debug(f"receive_tensor: completed receiving tensor shape={tensor.shape}")

        return tensor

    async def _send_tensor(
        self, tensor: torch.Tensor, transport_context: "TransportContext", tag: int
    ) -> None:
        """Send tensor via gloo. Waits for send to complete before returning.

        Args:
            tensor: Tensor to send.
            transport_context: Transport context containing the process group.
            tag: The Gloo tag of the transfer.
        """
        # Gloo TCP transport requires CPU tensors
        if tensor.device.type != "cpu":
            tensor = tensor.cpu()

        # Gloo requires contiguous tensors (slices may be non-contiguous views)
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        # Get process group from transport context
        pg = transport_context.get(GlooProcessGroupCache).get(self.store_key)

        # Determine remote rank based on our rank in the PG
        my_rank = pg.rank()
        remote_rank = 1 - my_rank

        logger.debug(
            f"send_tensor: sending tensor shape={tensor.shape} dtype={tensor.dtype} "
            f"to rank {remote_rank} (my_rank={my_rank}, store_key={self.store_key})"
        )

        # Run send in thread and wait for completion
        def do_send():
            work = pg.send([tensor], dstRank=remote_rank, tag=tag)
            work.wait()

        await asyncio.to_thread(do_send)

        logger.debug(f"send_tensor: completed sending tensor shape={tensor.shape}")

    async def drop(self) -> None:
        """Clean up resources held by this buffer."""
        # Wait for send to complete if it was started
        if self._send_task is not None:
            await self._send_task
            self._send_task = None
        # A get whose call raised or was cancelled, or whose value came in band, did not await its
        # receive, which may wait out Gloo's timeout, so the receive's failure is logged once it ends.
        if self._recv_task is not None:
            self._recv_task.add_done_callback(_log_receive_failure)
            self._recv_task = None
        self.is_object = False
        self.objects = None
