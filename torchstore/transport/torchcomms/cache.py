# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import weakref
from functools import cache

import torch

from torchstore.transport.buffers import TransportCache

try:
    from torchcomms._transport import RdmaMemory, RdmaTransport

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


class RdmaTransportCache(TransportCache):
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

    @staticmethod
    def device_to_index(device: torch.device | int) -> int:
        if isinstance(device, int):
            return device
        else:
            return 0 if device.type == "cpu" else device.index

    def put(self, key: str, device: torch.device | int) -> TransportAndAddress:
        index = self.device_to_index(device)
        transport = RdmaTransport(torch.device(index))

        if key not in self.transports:
            self.transports[key] = {}
        val = (transport, transport.bind())
        self.transports[key][index] = val
        return val

    def get(self, key: str, device: torch.device | int) -> TransportAndAddress:
        index = self.device_to_index(device)
        return self.transports[key][index]

    def get_or_create(
        self, key: str, device: torch.device | int
    ) -> tuple["RdmaTransport", bytes, bool]:
        """Get or create a transport for (key, device). Returns (transport, address, is_new)"""
        if not self.contains(key, device):
            transport, address = self.put(key, device)
            return transport, address, True
        transport, address = self.get(key, device)
        return transport, address, False

    def contains(self, key: str, device: torch.device | int) -> bool:
        index = self.device_to_index(device)
        return key in self.transports and index in self.transports[key]

    def clear(self) -> None:
        self.transports.clear()


class RdmaMemoryCache(TransportCache):
    """Cache for RDMA memory registrations.

    Keyed on (data_ptr, nbytes). Avoids repeated ibv_reg_mr kernel calls for stable tensors.

    A weakref on each tensor's ``untyped_storage()`` auto-evicts the entry when
    the underlying memory is released (i.e. no more live tensors on the process that reference this
    storage). This makes the cache safe for both server-side (stable kv tensors)
    and client-side (user-owned, transient) use.
    """

    def __init__(self) -> None:
        # {(data_ptr, nbytes): RdmaMemory}
        self._cache: dict[tuple[int, int], "RdmaMemory"] = {}
        # parallel dict keeping the weakref alive so the callback can fire
        self._storage_refs: dict[tuple[int, int], weakref.ref] = {}

    def get_or_register(self, tensor: torch.Tensor) -> "RdmaMemory":
        key = (tensor.data_ptr(), tensor.nbytes)
        if key in self._cache:
            return self._cache[key]

        mem = RdmaMemory(tensor)
        self._cache[key] = mem
        self._storage_refs[key] = weakref.ref(
            tensor.untyped_storage(), lambda _ref, _k=key: self._evict(_k)
        )
        return mem

    def _evict(self, key: tuple[int, int]) -> None:
        self._cache.pop(key, None)
        self._storage_refs.pop(key, None)

    def clear(self) -> None:
        self._cache.clear()
        self._storage_refs.clear()
