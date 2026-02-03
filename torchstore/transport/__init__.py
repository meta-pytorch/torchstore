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
from torchstore.transport.torchcomms.buffer import (
    TorchCommsRdmaTransportBuffer,
)
from torchstore.transport.torchcomms.cache import torchcomms_rdma_available
from torchstore.transport.types import Request, TensorSlice

if TYPE_CHECKING:
    from torchstore.strategy import StorageVolumeRef


class TransportType(Enum):
    Unset = auto()  # Default - lazily resolved based on availability
    MonarchRPC = auto()
    MonarchRDMA = auto()
    TorchCommsRDMA = auto()
    Gloo = auto()


def get_available_transport() -> TransportType:
    """Determine the best available transport type.

    Returns MonarchRDMA if available, otherwise falls back to MonarchRPC.
    """
    if monarch_rdma_transport_available():
        return TransportType.MonarchRDMA
    elif torchcomms_rdma_available():
        return TransportType.TorchCommsRDMA
    elif gloo_available():
        return TransportType.Gloo
    return TransportType.MonarchRPC


def create_transport_buffer(storage_volume_ref: "StorageVolumeRef") -> TransportBuffer:
    transport_type = storage_volume_ref.default_transport_type

    transport_map = {
        TransportType.MonarchRPC: MonarchRPCTransportBuffer,
        TransportType.MonarchRDMA: MonarchRDMATransportBuffer,
        TransportType.TorchCommsRDMA: TorchCommsRdmaTransportBuffer,
        TransportType.Gloo: GlooTransportBuffer,
    }

    return transport_map[transport_type](storage_volume_ref)


__all__ = ["Request", "TensorSlice", "TransportType"]
