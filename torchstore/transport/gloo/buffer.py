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
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING, Union

import torch
from torch.distributed import ProcessGroup, ProcessGroupGloo, Store, TCPStore

from torchstore.transport.buffers import TransportBuffer

if TYPE_CHECKING:
    from torchstore.transport.buffers import TransportContext
    from torchstore.transport.pipe import StorageVolumeRef


logger = getLogger(__name__)

# Global cache
_store_addrs: Dict[str, Tuple[str, int]] = (
    {}
)  # volume_id -> (master_addr, master_port, store_key)


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
    device: Optional[torch.device] = None,
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
    - TORCHSTORE_GLOO_ENABLED=1 environment variable must be set
    - Network connectivity between client and storage (for TCPStore)
    """

    requires_meta: bool = True

    def __init__(self) -> None:
        # Tensor metadata
        self.shape: Optional[torch.Size] = None
        self.dtype: Optional[torch.dtype] = None

        # Process group coordination
        self.master_addr: Optional[str] = None
        self.master_port: Optional[int] = None
        self.store_key: Optional[str] = None  # Unique key for this connection

        # Local tensor reference
        self.tensor_ref: Optional[torch.Tensor] = None

        # Flag for object (non-tensor) transport
        self.is_object: bool = False

        self.transport_context: Optional[Dict[str, Any]] = None

    def __getstate__(self) -> Dict[str, Any]:
        """Serialize state, excluding non-serializable fields."""
        state = self.__dict__.copy()
        state["transport_context"] = None
        state["tensor_ref"] = None
        return state

    async def handshake(
        self, tensor: torch.Tensor, volume_ref: "StorageVolumeRef"
    ) -> None:
        """Establish a gloo process group with the storage volume.

        Uses TCPStore for coordination. Both sides create their ProcessGroups
        in parallel using non-blocking RPC.
        """
        volume_id = volume_ref.volume_id

        if volume_id not in _store_addrs:
            # Generate unique store key and find free port
            store_key = f"torchstore_gloo_{str(uuid.uuid4())[:8]}"
            master_addr = _get_hostname()
            master_port = _find_free_port()

            logger.info(
                f"Initiating gloo handshake with StorageVolume:[{volume_id}] "
                f"using TCPStore at {master_addr}:{master_port}"
            )

            self.master_addr = master_addr
            self.master_port = master_port
            self.store_key = store_key

            # Start storage-side setup via non-blocking RPC
            handshake_fut = volume_ref.volume.handshake.call(self)

            try:
                # Create TCPStore and ProcessGroup on client side (rank 0, master)
                # Run in thread to avoid blocking event loop
                def create_pg():
                    tcp_store = TCPStore(
                        host_name=master_addr,
                        port=master_port,
                        world_size=2,
                        is_master=True,
                        timeout=timedelta(seconds=120),
                    )
                    return _gloo_factory(
                        store=tcp_store,
                        rank=0,
                        world_size=2,
                        timeout=timedelta(seconds=120),
                        device=torch.device("cpu"),
                    )

                pg = await asyncio.to_thread(create_pg)

                # Cache the connection info
                _store_addrs[volume_id] = (master_addr, master_port, store_key)
                volume_ref.transport_context.get_transport_context()[store_key] = pg

            finally:
                # Wait for storage side to complete
                await handshake_fut

            logger.info(f"Finished gloo handshake with StorageVolume:[{volume_id}]")

        # Set connection info from cache (since handshake is only done the first time)
        cached_addr = _store_addrs[volume_id]
        self.master_addr = cached_addr[0]
        self.master_port = cached_addr[1]
        self.store_key = cached_addr[2]
        print(
            f"volume_ref.transport_context.get_transport_context().keys(): {volume_ref.transport_context.get_transport_context().keys()}"
        )
        self.transport_context = volume_ref.transport_context.get_transport_context()

    async def recv_handshake(
        self, transport_context: "TransportContext"
    ) -> Optional[Any]:
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
        self.transport_context = ctx

        logger.info(
            f"Storage volume finished gloo process group setup for store_key={self.store_key}"
        )

        return None

    def allocate(self, tensor_like: Union[torch.Tensor, Tuple]) -> None:
        """Allocates internal buffers based on either an existing tensor
        or a Tuple of (shape, dtype).
        """
        if isinstance(tensor_like, str) or tensor_like is None:
            self.is_object = True
            return
        elif isinstance(tensor_like, Tuple):
            self.shape = tensor_like[0]
            self.dtype = tensor_like[1]
        else:
            assert isinstance(tensor_like, torch.Tensor)
            self.shape = tensor_like.shape
            self.dtype = tensor_like.dtype

    def allocate_dest(self, tensor_like: Union[torch.Tensor, Tuple]) -> None:
        """Called by client for GET operations. Allocate buffer for receiving tensor."""
        self.allocate(tensor_like)
        if not self.is_object:
            if isinstance(tensor_like, Tuple):
                self.tensor_ref = torch.zeros(
                    tensor_like[0], dtype=tensor_like[1], device=torch.device("cpu")
                )
            else:
                # Gloo TCP transport requires CPU tensors for recv
                if tensor_like.device.type != "cpu":
                    self.tensor_ref = torch.zeros(
                        tensor_like.shape,
                        dtype=tensor_like.dtype,
                        device=torch.device("cpu"),
                    )
                else:
                    self.tensor_ref = tensor_like

    def allocate_source(self, tensor: Optional[torch.Tensor]) -> None:
        """Called by client for PUT operations. Prepare source tensor for sending."""
        if tensor is None:
            return

        self.shape = tensor.shape
        self.dtype = tensor.dtype
        self.tensor_ref = tensor

    async def read_into(
        self, tensor: Optional[torch.Tensor], transport_context: "TransportContext"
    ) -> torch.Tensor:
        """Receive tensor via gloo. Waits for recv to complete before returning."""
        if self.is_object:
            return None

        if tensor is None:
            if self.tensor_ref is not None:
                tensor = self.tensor_ref
            else:
                tensor = torch.zeros(
                    self.shape, dtype=self.dtype, device=torch.device("cpu")
                )

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
            f"read_into: receiving tensor shape={tensor.shape} dtype={tensor.dtype} "
            f"from rank {remote_rank} (my_rank={my_rank}, store_key={self.store_key})"
        )

        # Run recv in thread and wait for completion
        def do_recv():
            work = pg.recv([tensor], srcRank=remote_rank, tag=0)
            work.wait()

        await asyncio.to_thread(do_recv)

        logger.debug(f"read_into: completed receiving tensor shape={tensor.shape}")

        self.tensor_ref = tensor
        return tensor

    async def write_from(
        self, tensor: Optional[torch.Tensor], transport_context: "TransportContext"
    ) -> None:
        """Send tensor via gloo. Waits for send to complete before returning.
        # called on client rank 0
        # called on storage volume rank 1
        Args:
            tensor (Optional[torch.Tensor]): Tensor to send. If None, the tensor will be retrieved from the tensor_ref
            transport_context (TransportContext): Transport context from the client
        """

        if self.is_object:
            return

        if tensor is None:
            tensor = self.tensor_ref

        if tensor is None:
            return

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
            f"write_from: sending tensor shape={tensor.shape} dtype={tensor.dtype} "
            f"to rank {remote_rank} (my_rank={my_rank}, store_key={self.store_key})"
        )

        # Run send in thread and wait for completion
        def do_send():
            work = pg.send([tensor], dstRank=remote_rank, tag=0)
            work.wait()

        await asyncio.to_thread(do_send)

        logger.debug(f"write_from: completed sending tensor shape={tensor.shape}")

    async def drop(self) -> None:
        """Clean up resources held by this buffer."""
        self.tensor_ref = None
