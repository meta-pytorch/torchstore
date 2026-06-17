# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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
from torchstore.transport.torchcomms.cache import (
    torchcomms_rdma_available,
    torchcomms_uniflow_available,
)
from torchstore.transport.torchcomms.uniflow_buffer import TorchCommsTransportBuffer
from torchstore.transport.types import Request, TensorSlice
from torchstore.transport.xccl import xccl_available, XcclTransportBuffer

if TYPE_CHECKING:
    from torchstore.strategy import StorageVolumeRef


class TransportType(Enum):
    Unset = auto()  # Default - lazily resolved based on availability
    MonarchRPC = auto()
    MonarchRDMA = auto()
    # Enum name is changed given uniflow supports more than just RDMA (i.e NVLink or TCP)
    TorchComms = auto()
    TorchCommsRDMA = TorchComms  # Backward compatible alias
    Gloo = auto()
    XCCL = auto()  # Intel oneCCL via torch.distributed; device-resident on XPU
    SharedMemory = auto()  # POSIX shared memory for same-host transfers


def get_available_transport(storage_volume_ref: "StorageVolumeRef") -> TransportType:
    """Determine the best available transport type for the given storage volume.

    Order: SharedMemory (same-host) > TorchComms (Uniflow RDMA/NVLink) >
    MonarchRDMA (ibverbs) > XCCL (XPU device-resident) > Gloo (TCP/CPU) >
    MonarchRPC (last resort).

    XCCL beats Gloo on XPU because it keeps tensors on device. Gloo is kept
    as the universal cross-platform fallback for non-XPU hosts and as a
    safety net if xccl init fails.
    """
    # Prefer SharedMemory for same-host transfers
    if SHM_ENABLED and is_local_to_volume(storage_volume_ref):
        return TransportType.SharedMemory

    # Fall back to RDMA if available (prefer TorchComms over Monarch RDMA)
    if torchcomms_uniflow_available() or torchcomms_rdma_available():
        return TransportType.TorchComms
    elif monarch_rdma_transport_available():
        return TransportType.MonarchRDMA
    elif xccl_available():
        return TransportType.XCCL
    elif gloo_available():
        return TransportType.Gloo

    return TransportType.MonarchRPC


def create_transport_buffer(storage_volume_ref: "StorageVolumeRef") -> TransportBuffer:
    transport_type = storage_volume_ref.default_transport_type

    if transport_type == TransportType.Unset:
        transport_type = get_available_transport(storage_volume_ref)

    if transport_type == TransportType.TorchComms:
        # Keep one public transport type while the backend migrates from the
        # legacy RDMA binding to Uniflow.
        if torchcomms_uniflow_available():
            return TorchCommsTransportBuffer(storage_volume_ref)
        if torchcomms_rdma_available():
            return TorchCommsRdmaTransportBuffer(storage_volume_ref)
        raise RuntimeError("TorchComms transport is not available.")

    transport_map = {
        TransportType.MonarchRPC: MonarchRPCTransportBuffer,
        TransportType.MonarchRDMA: MonarchRDMATransportBuffer,
        TransportType.Gloo: GlooTransportBuffer,
        TransportType.XCCL: XcclTransportBuffer,
        TransportType.SharedMemory: SharedMemoryTransportBuffer,
    }

    return transport_map[transport_type](storage_volume_ref)


__all__ = ["Request", "TensorSlice", "TransportType"]
