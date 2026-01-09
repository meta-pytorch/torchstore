# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import uuid
from datetime import timedelta
from logging import getLogger
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING, Union

import torch
from torch.distributed import ProcessGroup, ProcessGroupGloo, Store
from torchstore.transport.buffers import TransportBuffer

if TYPE_CHECKING:
    from torchstore.transport.buffers import TransportContext
    from torchstore.transport.pipe import StorageVolumeRef


logger = getLogger(__name__)

# Global caches
_file_store_names: Dict[str, str] = {}  # volume_id -> file_store_name


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
    and storage volume for each connection using FileStore for coordination.

    Prerequisites:
    - TORCHSTORE_GLOO_ENABLED=1 environment variable must be set
    - Shared filesystem access between client and storage (for FileStore)
    """

    requires_meta: bool = True
    read_ahead: bool = False  # Disable read_ahead - we handle it differently

    def __init__(self) -> None:
        # Tensor metadata
        self.shape: Optional[torch.Size] = None
        self.dtype: Optional[torch.dtype] = None

        # Process group coordination
        self.file_store_name: Optional[str] = None
        self.transport_context: Optional[Dict[str, Any]] = None

        # Local tensor reference
        self.tensor_ref: Optional[torch.Tensor] = None

        # Flag for object (non-tensor) transport
        self.is_object: bool = False

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

        Uses FileStore for coordination. Both sides create their ProcessGroups
        in parallel using non-blocking RPC.
        """
        volume_id = volume_ref.volume_id

        if volume_id not in _file_store_names:
            # Generate unique file store path
            file_store_name = f"/tmp/torchstore_gloo_{str(uuid.uuid4())[:8]}"

            logger.info(
                f"Initiating gloo handshake with StorageVolume:[{volume_id}] "
                f"using file_store={file_store_name}"
            )

            self.file_store_name = file_store_name

            # Start storage-side setup via non-blocking RPC
            handshake_fut = volume_ref.volume.handshake.call(self)

            try:
                # Create FileStore and ProcessGroup on client side (rank 0)
                # Run in thread to avoid blocking event loop
                def create_pg():
                    file_store = torch.distributed.FileStore(file_store_name, 2)
                    return _gloo_factory(
                        store=file_store,
                        rank=0,
                        world_size=2,
                        timeout=timedelta(seconds=120),
                        device=torch.device("cpu"),
                    )

                pg = await asyncio.to_thread(create_pg)

                # Cache the file store name and process group
                _file_store_names[volume_id] = file_store_name
                volume_ref.transport_context.get_transport_context()[
                    file_store_name
                ] = pg

            finally:
                # Wait for storage side to complete
                await handshake_fut

            logger.info(f"Finished gloo handshake with StorageVolume:[{volume_id}]")

        # Set up instance state for this operation
        self.file_store_name = _file_store_names[volume_id]
        self.transport_context = volume_ref.transport_context.get_transport_context()

    async def recv_handshake(
        self, transport_context: "TransportContext"
    ) -> Optional[Any]:
        """Called on storage volume side to set up the process group.

        Creates FileStore and ProcessGroup on storage side (rank 1).
        """
        ctx = transport_context.get_transport_context()

        if self.file_store_name in ctx:
            logger.debug(
                f"Reusing existing gloo process group for file_store={self.file_store_name}"
            )
            self.transport_context = ctx
            return None

        logger.info(
            f"Storage volume setting up gloo process group with file_store={self.file_store_name}"
        )

        # Create FileStore and ProcessGroup on storage side (rank 1)
        # Run in thread to avoid blocking event loop
        file_store_name = self.file_store_name

        def create_pg():
            file_store = torch.distributed.FileStore(file_store_name, 2)
            return _gloo_factory(
                store=file_store,
                rank=1,
                world_size=2,
                timeout=timedelta(seconds=120),
                device=torch.device("cpu"),
            )

        pg = await asyncio.to_thread(create_pg)

        # Cache the process group
        ctx[self.file_store_name] = pg

        # Set instance state
        self.transport_context = ctx

        logger.info(
            f"Storage volume finished gloo process group setup for file_store={self.file_store_name}"
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
                self.tensor_ref = torch.empty(
                    tensor_like[0], dtype=tensor_like[1], device=torch.device("cpu")
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
                tensor = torch.empty(
                    self.shape, dtype=self.dtype, device=torch.device("cpu")
                )

        # Get process group from transport context
        ctx = transport_context.get_transport_context()
        pg = ctx[self.file_store_name]

        # Determine remote rank based on our rank in the PG
        my_rank = pg.rank()
        remote_rank = 1 - my_rank

        # Run recv in thread and wait for completion
        def do_recv():
            work = pg.recv([tensor], srcRank=remote_rank, tag=0)
            work.wait()

        await asyncio.to_thread(do_recv)

        self.tensor_ref = tensor
        return tensor

    async def write_from(
        self, tensor: Optional[torch.Tensor], transport_context: "TransportContext"
    ) -> None:
        """Send tensor via gloo. Waits for send to complete before returning."""
        if self.is_object:
            return

        if tensor is None:
            tensor = self.tensor_ref

        if tensor is None:
            return

        # Get process group from transport context
        ctx = transport_context.get_transport_context()
        pg = ctx[self.file_store_name]

        # Determine remote rank based on our rank in the PG
        my_rank = pg.rank()
        remote_rank = 1 - my_rank

        # Run send in thread and wait for completion
        def do_send():
            work = pg.send([tensor], dstRank=remote_rank, tag=0)
            work.wait()

        await asyncio.to_thread(do_send)

    async def drop(self) -> None:
        """Clean up resources held by this buffer."""
        self.tensor_ref = None
