# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from collections.abc import Callable
from enum import auto, Enum
from typing import NamedTuple, TYPE_CHECKING

from torchstore.transport.buffers import TransportBuffer
from torchstore.transport.gloo import gloo_available, GlooTransportBuffer
from torchstore.transport.monarch_rpc import MonarchRPCTransportBuffer
from torchstore.transport.nixl import nixl_available, NixlTransportBuffer
from torchstore.transport.shared_memory import (
    is_local_to_volume,
    SharedMemoryTransportBuffer,
    SHM_ENABLED,
)
from torchstore.transport.torch_distributed import (
    torch_transport_available,
    TorchDistributedTransportBuffer,
)
from torchstore.transport.torchcomms.buffer import TorchCommsRdmaTransportBuffer
from torchstore.transport.torchcomms.cache import (
    torchcomms_rdma_available,
    torchcomms_uniflow_available,
)
from torchstore.transport.torchcomms.uniflow_buffer import TorchCommsTransportBuffer
from torchstore.transport.types import Request, TensorSlice

if TYPE_CHECKING:
    from torchstore.strategy import StorageVolumeRef


logger: logging.Logger = logging.getLogger(__name__)


class TransportType(Enum):
    Unset = auto()  # Default - lazily resolved based on availability
    MonarchRPC = auto()
    MonarchRDMA = auto()
    # Enum name is changed given uniflow supports more than just RDMA (i.e NVLink or TCP)
    TorchComms = auto()
    TorchCommsRDMA = TorchComms  # Backward compatible alias
    Gloo = auto()
    SharedMemory = auto()  # POSIX shared memory for same-host transfers
    NIXL = auto()
    TorchDistributed = auto()  # torch.distributed._transport backends


class _BackendTransport(NamedTuple):
    buffer: Callable[["StorageVolumeRef"], TransportBuffer]
    available: Callable[[], bool]


# Transports that the module providing them registers when it loads, which selection asks
# for at their place in its order. Only Monarch RDMA has such a place.
_BACKEND_TRANSPORT_TYPES = (TransportType.MonarchRDMA,)
_backend_transports: dict[TransportType, _BackendTransport] = {}


def register_backend_transport(
    transport_type: TransportType,
    buffer: Callable[["StorageVolumeRef"], TransportBuffer],
    available: Callable[[], bool],
    /,
) -> None:
    """Build ``buffer(ref)`` for ``transport_type``, which selection offers where ``available()``.

    ``torchstore.monarch.rdma`` registers Monarch RDMA when it loads. Raises ``ValueError``
    for a type that selection has no place for.
    """
    if transport_type not in _BACKEND_TRANSPORT_TYPES:
        raise ValueError(
            f"transport selection has no place for {transport_type}, only for "
            f"{', '.join(str(t) for t in _BACKEND_TRANSPORT_TYPES)}"
        )
    _backend_transports[transport_type] = _BackendTransport(buffer, available)


def _backend_transport_available(transport_type: TransportType) -> bool:
    entry = _backend_transports.get(transport_type)
    return entry is not None and entry.available()


def get_available_transport(storage_volume_ref: "StorageVolumeRef") -> TransportType:
    """Determine the best available transport type for the given storage volume.

    Prefers SharedMemory for same-host transfers, then explicitly enabled
    torch.distributed and NIXL transports, TorchComms, MonarchRDMA, and Gloo,
    before falling back to MonarchRPC.
    """
    # Prefer SharedMemory for same-host transfers
    if SHM_ENABLED and is_local_to_volume(storage_volume_ref):
        return TransportType.SharedMemory

    if torch_transport_available():
        return TransportType.TorchDistributed

    if nixl_available():
        return TransportType.NIXL

    # Fall back to RDMA if available (prefer TorchComms over Monarch RDMA)
    if torchcomms_uniflow_available() or torchcomms_rdma_available():
        return TransportType.TorchComms
    elif _backend_transport_available(TransportType.MonarchRDMA):
        return TransportType.MonarchRDMA
    elif gloo_available():
        return TransportType.Gloo

    return TransportType.MonarchRPC


def _log_transport_resolution(
    storage_volume_ref: "StorageVolumeRef", transport_type: TransportType
) -> None:
    logger.info(
        "[ts-transport] resolved=%s (torch_distributed=%s, nixl=%s, uniflow=%s, tc_rdma=%s, monarch_rdma=%s, gloo=%s, shm=%s)",
        transport_type.name,
        torch_transport_available(),
        nixl_available(),
        torchcomms_uniflow_available(),
        torchcomms_rdma_available(),
        _backend_transport_available(TransportType.MonarchRDMA),
        gloo_available(),
        SHM_ENABLED and is_local_to_volume(storage_volume_ref),
    )


def create_transport_buffer(storage_volume_ref: "StorageVolumeRef") -> TransportBuffer:
    transport_type = storage_volume_ref.default_transport_type

    if transport_type == TransportType.Unset:
        transport_type = get_available_transport(storage_volume_ref)

    _log_transport_resolution(storage_volume_ref, transport_type)

    if transport_type == TransportType.TorchComms:
        # Keep one public transport type while the backend migrates from the
        # legacy RDMA binding to Uniflow.
        if torchcomms_uniflow_available():
            return TorchCommsTransportBuffer(storage_volume_ref)
        if torchcomms_rdma_available():
            return TorchCommsRdmaTransportBuffer(storage_volume_ref)
        raise RuntimeError("TorchComms transport is not available.")

    if transport_type == TransportType.NIXL and not nixl_available():
        raise RuntimeError(
            "NIXL transport is not available. Install nixl and set "
            "TORCHSTORE_NIXL_ENABLED=1."
        )

    transport_map = {
        TransportType.MonarchRPC: MonarchRPCTransportBuffer,
        TransportType.Gloo: GlooTransportBuffer,
        TransportType.NIXL: NixlTransportBuffer,
        TransportType.TorchDistributed: TorchDistributedTransportBuffer,
        TransportType.SharedMemory: SharedMemoryTransportBuffer,
        **{type_: entry.buffer for type_, entry in _backend_transports.items()},
    }
    if transport_type not in transport_map:
        # Only Monarch RDMA, unregistered, is missing.
        raise RuntimeError(
            "Monarch RDMA transport is not registered in this process. TorchStore's Monarch "
            "backend registers it when it loads, as does importing torchstore.monarch.rdma "
            "or torchstore.transport.monarch_rdma."
        )

    return transport_map[transport_type](storage_volume_ref)


__all__ = ["Request", "TensorSlice", "TransportType"]
