# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from enum import auto, Enum
from typing import TYPE_CHECKING

from torchstore.transport.buffers import TransportBuffer
from torchstore.transport.monarch_rdma import MonarchRDMATransportBuffer

# from torchstore.transport.pipe import Pipe, Request, TensorSlice
from torchstore.transport.pipe import Request, TensorSlice

if TYPE_CHECKING:
    from torchstore.transport.pipe import StorageVolumeRef


class TransportType(Enum):
    MonarchRPC = auto()
    MonarchRDMA = auto()
    TorchCommsRDMA = auto()


def create_transport_buffer(storage_volume_ref: "StorageVolumeRef") -> TransportBuffer:
    transport_type = storage_volume_ref.default_transport_type

    transport_map = {
        # MonarchRPC: MonarchRDMATransportBuffer,
        TransportType.MonarchRDMA: MonarchRDMATransportBuffer,
        # TorchCommsRDMA: Torch #TODO
    }

    return transport_map[transport_type](storage_volume_ref)


__all__ = ["Request", "TensorSlice", "TransportType"]
