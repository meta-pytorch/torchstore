from __future__ import annotations

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import uuid
import weakref
from dataclasses import dataclass
from functools import cache
from typing import Any, cast, TYPE_CHECKING

import torch

from torchstore.transport.buffers import TransportCache

if TYPE_CHECKING:
    from torchcomms._transport import RdmaMemory, RdmaTransport
    from uniflow._core import MultiTransport, MultiTransportFactory, RegisteredSegment

try:
    import torchcomms._transport as _torchcomms_transport
except ImportError:
    _torchcomms_transport = None

try:
    import uniflow._core as _uniflow_transport
except ImportError:
    _uniflow_transport = None


logger = logging.getLogger(__name__)


def torchcomms_enabled() -> bool:
    # Honor both USE_TORCHCOMMS and USE_TORCHCOMMS_RDMA for backward compatibility
    return (
        os.environ.get("USE_TORCHCOMMS", "1") == "1"
        and os.environ.get("USE_TORCHCOMMS_RDMA", "1") == "1"
    )


@cache
def torchcomms_uniflow_available() -> bool:
    if not torchcomms_enabled() or _uniflow_transport is None:
        return False

    try:
        support_result = _uniflow_transport.MultiTransportFactory.supported(
            _uniflow_transport.TransportType.RDMA
        )
        return bool(support_result.has_value())
    except Exception as e:
        logger.info(f"Uniflow transport is not available: {e}")
        return False


@cache
def torchcomms_rdma_available() -> bool:
    if not torchcomms_enabled() or _torchcomms_transport is None:
        return False

    return bool(_torchcomms_transport.RdmaTransport.supported())


def uniflow_transport_module() -> Any:
    if _uniflow_transport is None:
        raise RuntimeError("Uniflow transport is not available.")
    return _uniflow_transport


def shutdown_uniflow_transport(transport: MultiTransport | None) -> None:
    if transport is None:
        return
    try:
        transport.shutdown()
    except Exception:
        logger.warning("Failed to shutdown Uniflow transport", exc_info=True)


if TYPE_CHECKING:
    TransportAndAddress = tuple[RdmaTransport, bytes]
else:
    TransportAndAddress = tuple[Any, bytes]


# Legacy TorchComms RDMA support exists only during the Uniflow rollout.
# Remove it once Comms releases Uniflow in stable.
class RdmaTransportCache(TransportCache):
    def __init__(self) -> None:
        assert torchcomms_rdma_available(), "TorchComms RDMA is not available."
        # {key: {device: (transport, address)}}
        self.transports: dict[str, dict[int, TransportAndAddress]] = {}

    @classmethod
    def try_init(cls) -> RdmaTransportCache | None:
        try:
            return cls()
        except Exception as e:
            logger.info(f"Failed to init RdmaTransportCache: {e}")
            return None

    @staticmethod
    def device_to_index(device: torch.device | int) -> int:
        """Map devices to legacy RdmaTransport indices; CPU uses index 0."""
        if isinstance(device, int):
            return device
        if device.type == "cpu":
            return 0
        # CUDA, XPU, etc. — use the device index
        return int(device.index)

    def put(self, key: str, device: torch.device | int) -> TransportAndAddress:
        assert _torchcomms_transport is not None
        index = self.device_to_index(device)
        transport = cast(
            "RdmaTransport",
            _torchcomms_transport.RdmaTransport(torch.device(index)),
        )

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
    ) -> tuple[RdmaTransport, bytes, bool]:
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
        self._cache: dict[tuple[int, int], RdmaMemory] = {}
        # parallel dict keeping the weakref alive so the callback can fire
        self._storage_refs: dict[tuple[int, int], weakref.ref] = {}

    def get_or_register(self, tensor: torch.Tensor) -> RdmaMemory:
        key = (tensor.data_ptr(), tensor.nbytes)
        if key in self._cache:
            return self._cache[key]

        assert _torchcomms_transport is not None
        mem = cast("RdmaMemory", _torchcomms_transport.RdmaMemory(tensor))
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


@dataclass
class UniflowRegistration:
    registered_segment: RegisteredSegment
    length: int


class UniflowCache(TransportCache):
    """Process-local cache for reusable Uniflow state.

    This groups the Uniflow state whose lifetime outlives one serialized
    TransportBuffer:
    - peer keys identify stable client/SV transport pairs across requests
    - factories are reused per device for topology, transport, and registration
    - connected transports are reused after a successful handshake
    - handshake transports hold not-yet-promoted server transports until connect
    - registrations cache tensor memory registrations with weakref eviction

    The cache stores and shuts down objects, but the transport buffer decides
    when a handshake transport is safe to promote or must be discarded.
    """

    def __init__(self) -> None:
        assert torchcomms_uniflow_available(), "Uniflow transport is not available."
        self._peer_keys: dict[str, str] = {}
        self._factories: dict[int, MultiTransportFactory] = {}
        self._transports: dict[str, dict[int, MultiTransport]] = {}
        self._handshake_transports: dict[str, dict[int, MultiTransport]] = {}
        self._registrations: dict[tuple[int, int], UniflowRegistration] = {}
        self._storage_refs: dict[tuple[int, int], weakref.ref] = {}

    @staticmethod
    def device_to_id(device: torch.device | int) -> int:
        """Map devices to Uniflow ids; CPU uses -1, CUDA/XPU use device index."""
        if isinstance(device, int):
            return device
        return -1 if device.type == "cpu" else int(device.index)

    def peer_key(self, volume_id: str) -> str:
        """Return the stable key used to reuse transports for one volume."""
        if volume_id not in self._peer_keys:
            self._peer_keys[volume_id] = uuid.uuid4().hex
        return self._peer_keys[volume_id]

    def factory(self, device_id: int) -> MultiTransportFactory:
        """Return the per-device factory for topology, transports, and segments."""
        if device_id not in self._factories:
            self._factories[device_id] = cast(
                "MultiTransportFactory",
                uniflow_transport_module().MultiTransportFactory(device_id=device_id),
            )
        return self._factories[device_id]

    @staticmethod
    def _put_transport(
        bucket: dict[int, MultiTransport],
        device_id: int,
        transport: MultiTransport,
    ) -> MultiTransport:
        existing = bucket.get(device_id)
        if existing is not None and existing is not transport:
            shutdown_uniflow_transport(existing)
        bucket[device_id] = transport
        return transport

    @staticmethod
    def _shutdown_bucket(bucket: dict[int, MultiTransport]) -> None:
        for transport in bucket.values():
            shutdown_uniflow_transport(transport)

    def contains_transport(self, peer_key: str, device_id: int) -> bool:
        return peer_key in self._transports and device_id in self._transports[peer_key]

    def put_transport(
        self,
        peer_key: str,
        device_id: int,
        transport: MultiTransport,
    ) -> MultiTransport:
        """Cache a connected transport, closing any previous transport."""
        return self._put_transport(
            self._transports.setdefault(peer_key, {}),
            device_id,
            transport,
        )

    def get_transport(self, peer_key: str, device_id: int) -> MultiTransport:
        return self._transports[peer_key][device_id]

    def pop_transport(self, peer_key: str, device_id: int) -> MultiTransport:
        transport = self._transports[peer_key].pop(device_id)
        if not self._transports[peer_key]:
            self._transports.pop(peer_key, None)
        return transport

    def discard_transport(self, peer_key: str, device_id: int) -> None:
        if self.contains_transport(peer_key, device_id):
            shutdown_uniflow_transport(self.pop_transport(peer_key, device_id))

    def discard_all_transports(self, peer_key: str) -> None:
        self._shutdown_bucket(self._transports.pop(peer_key, {}))

    def contains_handshake_transport(
        self,
        handshake_id: str,
        device_id: int,
    ) -> bool:
        return (
            handshake_id in self._handshake_transports
            and device_id in self._handshake_transports[handshake_id]
        )

    def put_handshake_transport(
        self,
        handshake_id: str,
        device_id: int,
        transport: MultiTransport,
    ) -> MultiTransport:
        """Hold a server transport until the two-step handshake completes."""
        return self._put_transport(
            self._handshake_transports.setdefault(handshake_id, {}),
            device_id,
            transport,
        )

    def get_handshake_transport(
        self,
        handshake_id: str,
        device_id: int,
    ) -> MultiTransport:
        return self._handshake_transports[handshake_id][device_id]

    def pop_handshake_transport(
        self,
        handshake_id: str,
        device_id: int,
    ) -> MultiTransport:
        transport = self._handshake_transports[handshake_id].pop(device_id)
        if not self._handshake_transports[handshake_id]:
            self._handshake_transports.pop(handshake_id, None)
        return transport

    def promote_handshake_transport(
        self,
        handshake_id: str,
        peer_key: str,
        device_id: int,
    ) -> None:
        """Move a completed handshake transport into the reusable cache."""
        transport = self.pop_handshake_transport(handshake_id, device_id)
        self.put_transport(peer_key, device_id, transport)

    def discard_handshake(self, handshake_id: str) -> None:
        self._shutdown_bucket(self._handshake_transports.pop(handshake_id, {}))

    def get_or_register(
        self,
        tensor: torch.Tensor,
        factory: MultiTransportFactory,
    ) -> UniflowRegistration:
        """Register tensor memory once and evict when its storage is freed."""
        key = (tensor.data_ptr(), tensor.nbytes)
        if key in self._registrations:
            return self._registrations[key]

        segment = uniflow_transport_module().Segment.from_tensor(tensor)
        registered_segment = cast(
            "RegisteredSegment",
            factory.register_segment(segment).unwrap(),
        )
        registration = UniflowRegistration(
            registered_segment=registered_segment,
            length=tensor.nbytes,
        )
        self._registrations[key] = registration
        self._storage_refs[key] = weakref.ref(
            tensor.untyped_storage(), lambda _ref, _k=key: self._evict_registration(_k)
        )
        return registration

    def _evict_registration(self, key: tuple[int, int]) -> None:
        self._registrations.pop(key, None)
        self._storage_refs.pop(key, None)

    def clear(self) -> None:
        for peer_key in list(self._transports):
            self.discard_all_transports(peer_key)
        for handshake_id in list(self._handshake_transports):
            self.discard_handshake(handshake_id)
        self._peer_keys.clear()
        self._factories.clear()
        self._registrations.clear()
        self._storage_refs.clear()
