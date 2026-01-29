# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import socket
import uuid
from datetime import timedelta
from logging import getLogger
from typing import Any, TYPE_CHECKING

import torch
from torch.distributed import ProcessGroup, ProcessGroupGloo, Store, TCPStore

from torchstore.transport.buffers import TransportBuffer

if TYPE_CHECKING:
    from torchstore.transport.buffers import TransportContext
    from torchstore.transport.pipe import StorageVolumeRef
    from torchstore.transport.types import Request
import os

import torch.distributed as dist

logger = getLogger(__name__)

# Global cache
_store_addrs: dict[
    str, tuple[str, int]
] = {}  # volume_id -> (master_addr, master_port, store_key)


def gloo_available() -> bool:
    """Check if gloo transport is available and enabled.

    Returns True if:
    1. USE_GLOO environment variable is set to "1"
    2. torch.distributed is available
    3. gloo backend is available

    """
    gloo_enabled = os.environ.get("USE_GLOO", "0") == "1"
    if not gloo_enabled:
        return False

    if not dist.is_available():
        return False

    if not dist.is_gloo_available():
        return False

    return True


def _find_free_port() -> int:
    """Find a free port on the local machine."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


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


class GlooTransportBuffer(TransportBuffer):
    """Transport buffer implementation using PyTorch distributed (gloo backend) for tensor transfer.

    This buffer creates a dedicated 2-process gloo process group between the client
    and storage volume for each connection using TCPStore for coordination.

    Prerequisites:
    - USE_GLOO=1 environment variable must be set
    - Network connectivity between client and storage (for TCPStore)
    """

    def __init__(self, storage_volume_ref: "StorageVolumeRef") -> None:
        super().__init__(storage_volume_ref)

        # Tensor metadata
        self.shape: torch.Size | None = None
        self.dtype: torch.dtype | None = None

        # Process group coordination
        self.master_addr: str | None = None
        self.master_port: int | None = None
        self.store_key: str | None = None  # Unique key for this connection

        # Flag for object (non-tensor) transport
        self.is_object: bool = False
        self.objects: Any = None  # For object transport in GET operations

        # TCPStore reference (kept alive until PG is created)
        self._tcp_store: TCPStore | None = None
        self._connection_exists: bool = (
            False  # Whether a handshake has already been performed
        )
        self._pg_task: asyncio.Task | None = None  # Background task for PG creation
        self._send_task: asyncio.Task | None = (
            None  # Background task for sending tensor
        )
        self._recv_task: asyncio.Task | None = (
            None  # Background task for receiving tensor
        )

    @property
    def requires_handshake(self) -> bool:
        """Only handshake if connection is not already cached."""
        return not self._connection_exists

    def __getstate__(self) -> dict[str, Any]:
        """Serialize state, excluding non-serializable fields."""
        state = self.__dict__.copy()
        state["storage_volume_ref"] = None
        state["_tcp_store"] = None
        state["_pg_task"] = None
        state["_send_task"] = None
        state["_recv_task"] = None
        return state

    async def _pre_handshake(self, request: "Request") -> None:
        """Prepare for handshake. Create TCPStore and start PG creation on
        client side (rank 0).

        The TCPStore is created here so it's ready to accept connections
        when the handshake RPC reaches the storage volume. The ProcessGroup
        creation is started as a background task so it runs concurrently with
        the handshake RPC, allowing both sides to rendezvous.
        """
        volume_id = self.storage_volume_ref.volume_id

        # Check if connection already exists
        if volume_id in _store_addrs:
            self._connection_exists = True
            cached_addr = _store_addrs[volume_id]
            self.master_addr = cached_addr[0]
            self.master_port = cached_addr[1]
            self.store_key = cached_addr[2]
            return

        # Generate unique store key and find free port
        self.store_key = f"torchstore_gloo_{str(uuid.uuid4())[:8]}"
        self.master_addr = _get_hostname()
        self.master_port = _find_free_port()

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
            timeout=timedelta(seconds=120),
            wait_for_workers=False,
        )

        # Start PG creation in background so it runs concurrently with handshake RPC
        tcp_store = self._tcp_store

        def create_pg():
            return _gloo_factory(
                store=tcp_store,
                rank=0,
                world_size=2,
                timeout=timedelta(seconds=120),
                device=torch.device("cpu"),
            )

        self._pg_task = asyncio.create_task(asyncio.to_thread(create_pg))

    async def recv_handshake(self, transport_context: "TransportContext") -> Any | None:
        """Called on storage volume side to set up the process group.

        Creates TCPStore and ProcessGroup on storage side (rank 1).

        transport_context: the TransportContext from the StorageVolume to which the pg will be added

        """
        ctx = transport_context.get_transport_context()

        if self.store_key in ctx:
            raise RuntimeError("this shouldnt happen")

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
                timeout=timedelta(seconds=120),
            )
            return _gloo_factory(
                store=tcp_store,
                rank=1,
                world_size=2,
                timeout=timedelta(seconds=120),
                device=torch.device("cpu"),
            )

        pg = await asyncio.to_thread(create_pg)

        # Cache the process group
        ctx[self.store_key] = pg

        logger.info(
            f"Storage volume finished gloo process group setup for store_key={self.store_key}"
        )

        return None

    async def _post_handshake(self, handshake_result: Any) -> None:
        """Await ProcessGroup creation that was started in _pre_handshake.

        The PG creation was started concurrently with the handshake RPC,
        so we just await the result here.
        """
        volume_id = self.storage_volume_ref.volume_id

        # Await the PG creation that was started in _pre_handshake
        pg = await self._pg_task

        # Cache the connection info and process group
        _store_addrs[volume_id] = (self.master_addr, self.master_port, self.store_key)
        self.storage_volume_ref.transport_context.get_transport_context()[
            self.store_key
        ] = pg

        # Clear references
        self._tcp_store = None
        self._pg_task = None

        logger.info(f"Finished gloo handshake with StorageVolume:[{volume_id}]")

    async def _pre_put_hook(self, request: "Request") -> None:
        """Start sending tensor before put RPC.

        Called after handshake completes, before put.call().
        Starts the send as a background task so it runs concurrently
        with the storage volume's recv in handle_put_request.
        """
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

        # Start send in background - will run concurrently with put RPC
        self._send_task = asyncio.create_task(
            self.send_tensor(
                tensor,
                self.storage_volume_ref.transport_context,
            )
        )

    async def handle_put_request(
        self,
        ctx: "TransportContext",
        request: "Request",
        maybe_tensor: torch.Tensor | None,
    ) -> Any:
        """Called by storage volume. Receive tensor from client via gloo process group."""
        if request.is_object:
            self.is_object = True
            return request.objects

        # Allocate destination tensor if needed
        tensor = maybe_tensor
        if tensor is None:
            tensor = torch.empty(
                self.shape, dtype=self.dtype, device=torch.device("cpu")
            )

        # Receive tensor from client via gloo
        tensor = await self.receive_tensor(tensor, ctx)

        return tensor

    async def _pre_get_hook(self, key: str, request: "Request") -> None:
        """Start receiving tensor before get RPC.

        Called after handshake completes, before get.call().
        Starts the recv as a background task so it runs concurrently
        with the storage volume's send in handle_get_request.
        """
        # If user provided a destination tensor, use it
        if request.tensor_val is not None:
            tensor = request.tensor_val
            self.shape = request.tensor_val.shape
            self.dtype = request.tensor_val.dtype
        else:
            # Need to fetch metadata to know shape/dtype for allocation
            meta = await self.storage_volume_ref.volume.get_meta.call_one(
                key, request.meta_only()
            )
            if isinstance(meta, str) or meta is None:
                # It's an object, not a tensor
                self.is_object = True
                return
            # meta is (shape, dtype)
            self.shape = meta[0]
            self.dtype = meta[1]
            tensor = torch.empty(
                self.shape, dtype=self.dtype, device=torch.device("cpu")
            )

        # Start recv in background - will run concurrently with get RPC
        # The storage volume will send the tensor in handle_get_request
        self._recv_task = asyncio.create_task(
            self.receive_tensor(
                tensor,
                self.storage_volume_ref.transport_context,
            )
        )

    async def handle_get_request(self, ctx: "TransportContext", data: Any) -> None:
        """Called by storage volume. Send tensor to client via gloo process group."""
        if not isinstance(data, torch.Tensor):
            self.is_object = True
            self.objects = data
            return

        # Send the tensor to client via gloo
        await self.send_tensor(data, ctx)

    async def _handle_storage_volume_response(
        self, transport_buffer: "TransportBuffer"
    ) -> Any:
        """Process the response from storage volume after get.

        The tensor data was received via gloo in the background recv task.
        """
        if transport_buffer.is_object:
            return transport_buffer.objects

        # Wait for the recv to complete
        if self._recv_task is not None:
            tensor = await self._recv_task
            self._recv_task = None
            if tensor is None:
                raise RuntimeError(
                    f"receive_tensor returned None (is_object={self.is_object}, "
                    f"shape={self.shape}, dtype={self.dtype})"
                )
            return tensor

        raise RuntimeError(f"No recv task available (is_object={self.is_object})")

    async def receive_tensor(
        self, tensor: torch.Tensor, transport_context: "TransportContext"
    ) -> torch.Tensor:
        """Receive tensor via gloo. Waits for recv to complete before returning.

        Args:
            tensor: Pre-allocated destination tensor to receive into.
            transport_context: Transport context containing the process group.

        Returns:
            The received tensor (same object as input, with data overwritten).
        """
        # Gloo TCP transport requires CPU tensors for recv
        if tensor.device.type != "cpu":
            tensor = torch.zeros(
                tensor.shape, dtype=tensor.dtype, device=torch.device("cpu")
            )

        # Get process group from transport context
        ctx = transport_context.get_transport_context()
        pg = ctx[self.store_key]

        # Determine remote rank based on our rank in the PG
        my_rank = pg.rank()
        remote_rank = 1 - my_rank

        logger.debug(
            f"receive_tensor: receiving tensor shape={tensor.shape} dtype={tensor.dtype} "
            f"from rank {remote_rank} (my_rank={my_rank}, store_key={self.store_key})"
        )

        # Run recv in thread and wait for completion
        def do_recv():
            work = pg.recv([tensor], srcRank=remote_rank, tag=0)
            work.wait()

        await asyncio.to_thread(do_recv)

        logger.debug(f"receive_tensor: completed receiving tensor shape={tensor.shape}")

        return tensor

    async def send_tensor(
        self, tensor: torch.Tensor, transport_context: "TransportContext"
    ) -> None:
        """Send tensor via gloo. Waits for send to complete before returning.

        Args:
            tensor: Tensor to send.
            transport_context: Transport context containing the process group.
        """
        # Gloo TCP transport requires CPU tensors
        if tensor.device.type != "cpu":
            tensor = tensor.cpu()

        # Gloo requires contiguous tensors (slices may be non-contiguous views)
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        # Get process group from transport context
        ctx = transport_context.get_transport_context()

        try:
            pg = ctx[self.store_key]
        except Exception as e:
            raise RuntimeError(f"ctx.keys(): {ctx.keys()}") from e

        # Determine remote rank based on our rank in the PG
        my_rank = pg.rank()
        remote_rank = 1 - my_rank

        logger.debug(
            f"send_tensor: sending tensor shape={tensor.shape} dtype={tensor.dtype} "
            f"to rank {remote_rank} (my_rank={my_rank}, store_key={self.store_key})"
        )

        # Run send in thread and wait for completion
        def do_send():
            work = pg.send([tensor], dstRank=remote_rank, tag=0)
            work.wait()

        await asyncio.to_thread(do_send)

        logger.debug(f"send_tensor: completed sending tensor shape={tensor.shape}")

    async def drop(self) -> None:
        """Clean up resources held by this buffer."""
        # Wait for send to complete if it was started
        if self._send_task is not None:
            await self._send_task
            self._send_task = None
