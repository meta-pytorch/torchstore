# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import uuid
from datetime import timedelta
from logging import getLogger
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch.distributed import ProcessGroup, ProcessGroupGloo, Store

from torchstore.transport.buffers import TransportBuffer

# from torchstore.strategy import StorageVolumeRef

local_pgs: Dict[str, torch.distributed.ProcessGroup] = {}
file_store_names: Dict[str, str] = {}

logger = getLogger(__name__)


def _gloo_factory(
    store: Store,
    rank: int,
    world_size: int,
    timeout: timedelta,
    device: Optional[torch.device] = None,
    **kwargs: object,
) -> ProcessGroup:
    """
    From:
    https://github.com/pytorch/pytorch/blob/92284fb2ff44f09a9c7df0d8cf6cac9903e376a4/torch/distributed/_dist2.py#L64

    """

    assert len(kwargs) == 0, "Gloo backend received unexpected kwargs"

    backend_class = ProcessGroupGloo(store, rank, world_size, timeout)
    backend_class._set_sequence_number_for_group()

    pg = ProcessGroup(store, rank, world_size)
    pg._set_default_backend(ProcessGroup.BackendType.GLOO)

    # register devices
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


class TorchDistributedBuffer(TransportBuffer):
    requires_meta: bool = True
    read_ahead: bool = True

    def __init__(self) -> None:
        self.shape: Optional[torch.Size] = None
        self.dtype: Optional[torch.dtype] = None
        self.fut: Optional[torch.futures.Future] = None

    def __getstate__(self) -> Dict[str, Any]:
        # Any time that we serialize the transport buffer, the idea is
        # that tensors will be transported via tensor_enginer.RDMABuffer, so it makes
        # no sense to hold this reference when we are serializing
        state = self.__dict__.copy()
        state["fut"] = None
        state["transport_context"] = None
        return state

    # TODO: ensure this is only called once
    async def setup_comms(self, storage_volume):

        # transport context is actually stored in the strategy,
        # but is passed along here so we can cache PG's.

        # TODO: file store name is wrong
        if storage_volume.volume_id not in file_store_names:
            # TODO: TCPStore
            file_store_name = f"/tmp/lpasqualin/comms_test{str(uuid.uuid4())[:8]}"
            logger.info(
                f"Initiating pg handshake with StorageVolume:[{storage_volume.volume_id}]"
                f" using id={file_store_name}"
            )

            self.file_store_name = file_store_name
            handshake_fut = storage_volume.setup_comms.call(self)

            try:
                file_store = torch.distributed.FileStore(file_store_name, 2)
                pg = _gloo_factory(
                    store=file_store,
                    rank=0,
                    world_size=2,
                    timeout=timedelta(seconds=120),
                    device=torch.device("cpu"),
                )
                file_store_names[storage_volume.volume_id] = file_store_name
                storage_volume.transport_context[file_store_name] = pg
            finally:
                await handshake_fut

            logger.info(
                f"Finished pg handshake with StorageVolume:[{storage_volume.volume_id}]"
                f" using id={file_store_name}"
            )

        self.file_store_name = file_store_names[storage_volume.volume_id]
        self.transport_context = storage_volume.transport_context
        self.remote_rank = 1

    async def storage_volume_setup_comms(
        self, transport_context: Dict[str, Any]
    ) -> None:

        if self.file_store_name in transport_context:
            return

        file_store = torch.distributed.FileStore(self.file_store_name, 2)
        pg = _gloo_factory(
            store=file_store,
            rank=1,
            world_size=2,
            timeout=timedelta(seconds=120),
            device=torch.device("cpu"),
        )
        transport_context[self.file_store_name] = pg

    def allocate(self, tensor_like: Union[torch.Tensor, Tuple]) -> None:
        """Allocates internal buffers based on either an existing tensor
        or a Tuple of (shape, dtype)
        """
        if isinstance(tensor_like, str) or tensor_like is None:
            # tensor is just an object, nothing to alloctest
            self.is_object = True
            return
        elif isinstance(tensor_like, Tuple):
            # we know the size of the tensor from fetching metadata
            self.shape = tensor_like[0]
            self.dtype = tensor_like[1]
        else:
            # we have an inplace tensor, allocate a copy
            assert isinstance(tensor_like, torch.Tensor)
            self.shape = tensor_like.shape
            self.dtype = tensor_like.dtype

    # send
    async def read_into(self, tensor: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.is_object:
            return

        if tensor is None:
            tensor = torch.empty(self.shape, dtype=self.dtype)

        assert self.fut is None
        pg = self.transport_context[self.file_store_name]
        self.fut = pg.recv([tensor], srcRank=self.remote_rank, tag=0)

        return tensor

    # recv
    async def write_from(self, tensor: Optional[torch.Tensor]) -> None:
        if self.is_object:
            return

        assert self.fut is None
        pg = self.transport_context[self.file_store_name]
        self.fut = pg.send([tensor], dstRank=self.remote_rank, tag=0)

    async def finish(self):
        if self.fut is not None:
            while not self.fut.done():
                await asyncio.sleep(0.005)
