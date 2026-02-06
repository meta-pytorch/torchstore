# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from enum import auto, Enum
from typing import TYPE_CHECKING

from torchstore.transport.buffers import TransportBuffer
from torchstore.transport.cuda_ipc import (
    cuda_ipc_available,
    CudaIPCTransportBuffer,
)
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
from torchstore.transport.types import Request, TensorSlice

if TYPE_CHECKING:
    from torchstore.strategy import StorageVolumeRef


class TransportType(Enum):
    Unset = auto()  # Default - lazily resolved based on availability
    MonarchRPC = auto()
    MonarchRDMA = auto()
    TorchCommsRDMA = auto()
    CudaIPC = auto()  # CUDA IPC for intra-node GPU-direct transfers
    SharedMemory = auto()  # POSIX shared memory for same-host transfers


def get_available_transport(storage_volume_ref: "StorageVolumeRef") -> TransportType:
    """Determine the best available transport type for the given storage volume.

    Priority order:
    1. SharedMemory for same-host transfers
    2. MonarchRDMA (if InfiniBand available)
    3. CudaIPC (if single-node with CUDA)
    4. MonarchRPC (fallback)
    """
    # Prefer SharedMemory for same-host transfers
    if SHM_ENABLED and is_local_to_volume(storage_volume_ref):
        return TransportType.SharedMemory

    # Fall back to RDMA if available
    if monarch_rdma_transport_available():
        return TransportType.MonarchRDMA

    # Check CUDA IPC for GPU-direct transfers
    if cuda_ipc_available():
        return TransportType.CudaIPC

    return TransportType.MonarchRPC


def create_transport_buffer(storage_volume_ref: "StorageVolumeRef") -> TransportBuffer:
    transport_type = storage_volume_ref.default_transport_type

    if transport_type == TransportType.Unset:
        transport_type = get_available_transport(storage_volume_ref)

    transport_map = {
        TransportType.MonarchRPC: MonarchRPCTransportBuffer,
        TransportType.MonarchRDMA: MonarchRDMATransportBuffer,
        TransportType.TorchCommsRDMA: TorchCommsRdmaTransportBuffer,
        TransportType.CudaIPC: CudaIPCTransportBuffer,
        TransportType.SharedMemory: SharedMemoryTransportBuffer,
    }

    return transport_map[transport_type](storage_volume_ref)


__all__ = ["Request", "TensorSlice", "TransportType"]
