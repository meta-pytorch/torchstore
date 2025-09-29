# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from enum import auto, Enum

from torchstore.transport.buffers import MonarchTransportBuffer, RDMATransportBuffer

from torchstore.transport.pipe import Pipe, Request, TensorSlice
from torchstore.transport.torch_distributed_buffer import TorchDistributedBuffer


class TransportType(Enum):
    MonarchRPC = auto()
    MonarchRDMA = auto()
    TorchDistributed = auto()

    def buffer_cls(self):
        return {
            TransportType.MonarchRPC: MonarchTransportBuffer,
            TransportType.MonarchRDMA: RDMATransportBuffer,
            TransportType.TorchDistributed: TorchDistributedBuffer,
        }[self]


__all__ = ["Pipe", "Request", "TensorSlice", "TransportType"]
