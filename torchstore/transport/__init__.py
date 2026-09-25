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
from torchstore.transport.nixl import nixl_available, NixlTransportBuffer
from torchstore.transport.shared_memory import (
    is_local_to_volume,
    SharedMemoryTransportBuffer,
    SHM_ENABLED,
)
from torchstore.transport.neuron_efa import NeuronEFATransportBuffer
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
    # Trainium: RDMA into a vLLM-Neuron generator's HBM regions over EFA.
    #
    # Needed because the usual contract -- hand me a state_dict, I populate those tensors in
    # place -- has no effect on that receiver. vLLM-Neuron's compile path inlines model weights
    # as HLO constants, so writing nn.Parameter.data does not reach the running compiled NEFF.
    # The destination is therefore HBM regions described by a manifest the generator publishes,
    # not torch storage, and the generator itself acts as the StorageVolume.
    #
    # Never selected by get_available_transport: it is not a general-purpose transport and only
    # applies when the storage volume is a Neuron generator. Ask for it explicitly.
    NeuronEFA = auto()


def get_available_transport(storage_volume_ref: "StorageVolumeRef") -> TransportType:
    """Determine the best available transport type for the given storage volume.

    Prefers SharedMemory for same-host transfers, then an explicitly enabled
    NIXL transport, TorchComms, MonarchRDMA, and Gloo, before falling back to
    MonarchRPC.
    """
    # Prefer SharedMemory for same-host transfers
    if SHM_ENABLED and is_local_to_volume(storage_volume_ref):
        return TransportType.SharedMemory

    if nixl_available():
        return TransportType.NIXL

    # Fall back to RDMA if available (prefer TorchComms over Monarch RDMA)
    if torchcomms_uniflow_available() or torchcomms_rdma_available():
        return TransportType.TorchComms
    elif monarch_rdma_transport_available():
        return TransportType.MonarchRDMA
    elif gloo_available():
        return TransportType.Gloo

    return TransportType.MonarchRPC


def _log_transport_resolution(
    storage_volume_ref: "StorageVolumeRef", transport_type: TransportType
) -> None:
    logger.info(
        "[ts-transport] resolved=%s (nixl=%s, uniflow=%s, tc_rdma=%s, monarch_rdma=%s, gloo=%s, shm=%s)",
        transport_type.name,
        nixl_available(),
        torchcomms_uniflow_available(),
        torchcomms_rdma_available(),
        monarch_rdma_transport_available(),
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
        TransportType.MonarchRDMA: MonarchRDMATransportBuffer,
        TransportType.Gloo: GlooTransportBuffer,
        TransportType.NIXL: NixlTransportBuffer,
        TransportType.SharedMemory: SharedMemoryTransportBuffer,
        TransportType.NeuronEFA: NeuronEFATransportBuffer,
    }

    return transport_map[transport_type](storage_volume_ref)


__all__ = ["Request", "TensorSlice", "TransportType"]
