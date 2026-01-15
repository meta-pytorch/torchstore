# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from functools import cache

import torch

try:
    from torchcomms._transport import RdmaTransport

    torchcomms_available = True
except ImportError:
    torchcomms_available = False


@cache
def torchcomms_rdma_available() -> bool:
    import os

    rdma_enabled = os.environ.get("USE_TORCHCOMMS_RDMA", "1") == "1"
    # (1) CommsRDMA flag is enabled (2) torchcomms lib is available (3) RDMA is supported
    return rdma_enabled and torchcomms_available and RdmaTransport.supported()


TransportAndAddress = tuple["RdmaTransport", bytes]


class RdmaTransportCache:
    def __init__(self) -> None:
        assert torchcomms_rdma_available(), "TorchComms RDMA is not available."
        # {key: {device: (transport, address)}}
        self.transports: dict[str, dict[int, TransportAndAddress]] = {}

    @classmethod
    def try_init(cls) -> "RdmaTransportCache | None":
        try:
            return cls()
        except Exception as e:
            logging.info(f"Failed to init RdmaTransportCache: {e}")
            return None

    def _device_to_index(self, device: torch.device | int) -> int:
        if isinstance(device, int):
            return device
        else:
            return 0 if device.type == "cpu" else device.index

    def put(self, key: str, device: torch.device | int) -> TransportAndAddress:
        index = self._device_to_index(device)
        transport = RdmaTransport(torch.device(index))

        if key not in self.transports:
            self.transports[key] = {}
        val = (transport, transport.bind())
        self.transports[key][index] = val
        return val

    def _get(self, key: str, device: torch.device | int) -> TransportAndAddress:
        index = self._device_to_index(device)
        return self.transports[key][index]

    def get(self, key: str, device: torch.device | int):
        if not self.contains(key, device):
            return self.put(key, device)
        return self._get(key, device)

    def contains(self, key: str, device: torch.device | int) -> bool:
        index = self._device_to_index(device)
        return key in self.transports and index in self.transports[key]
