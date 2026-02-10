# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from enum import auto, Enum
from typing import TYPE_CHECKING

from torchstore.transport.buffers import TransportBuffer
from torchstore.transport.gloo import gloo_available, GlooTransportBuffer
from torchstore.transport.monarch_rdma import (
    monarch_rdma_transport_available,
    MonarchRDMATransportBuffer,
)
from torchstore.transport.monarch_rpc import MonarchRPCTransportBuffer
from torchstore.transport.shared_memory import (
    is_local_to_volume,
    SharedMemoryTransportBuffer,
    SHM_ENABLED,
)
from torchstore.transport.torchcomms.buffer import TorchCommsRdmaTransportBuffer
from torchstore.transport.torchcomms.cache import torchcomms_rdma_available
from torchstore.transport.types import Request, TensorSlice

if TYPE_CHECKING:
    from torchstore.strategy import StorageVolumeRef

logger = logging.getLogger(__name__)


class TransportType(Enum):
    Unset = auto()  # Default - lazily resolved based on availability
    MonarchRPC = auto()
    MonarchRDMA = auto()
    TorchCommsRDMA = auto()
    Gloo = auto()
    SharedMemory = auto()  # POSIX shared memory for same-host transfers


def get_available_transport(storage_volume_ref: "StorageVolumeRef") -> TransportType:
    """Determine the best available transport type for the given storage volume.

    Prefers SharedMemory for same-host transfers, then MonarchRDMA if available,
    otherwise falls back to MonarchRPC.
    """
    # Prefer SharedMemory for same-host transfers
    if SHM_ENABLED and is_local_to_volume(storage_volume_ref):
        return TransportType.SharedMemory

    # Fall back to RDMA if available
    if monarch_rdma_transport_available():
        return TransportType.MonarchRDMA
    elif torchcomms_rdma_available():
        return TransportType.TorchCommsRDMA
    elif gloo_available():
        return TransportType.Gloo

    return TransportType.MonarchRPC


def create_transport_buffer(storage_volume_ref: "StorageVolumeRef") -> TransportBuffer:
    transport_type = storage_volume_ref.default_transport_type

    if transport_type == TransportType.Unset:
        transport_type = get_available_transport(storage_volume_ref)

    logger.info(f"Creating transport buffer: {transport_type.name}")

    transport_map = {
        TransportType.MonarchRPC: MonarchRPCTransportBuffer,
        TransportType.MonarchRDMA: MonarchRDMATransportBuffer,
        TransportType.TorchCommsRDMA: TorchCommsRdmaTransportBuffer,
        TransportType.Gloo: GlooTransportBuffer,
        TransportType.SharedMemory: SharedMemoryTransportBuffer,
    }

    return transport_map[transport_type](storage_volume_ref)


__all__ = ["Request", "TensorSlice", "TransportType"]
