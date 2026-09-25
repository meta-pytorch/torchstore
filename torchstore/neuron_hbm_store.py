# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A storage volume backed by a Neuron generator's HBM.

Lives here rather than in ``torchstore/transport/`` so the transport layer does not have to
import the storage layer. See ``torchstore.transport.neuron_efa`` for the protocol this
implements and why the destination is HBM regions rather than torch storage.
"""

from typing import Any

from torchstore.storage_volume import StorageImpl
from torchstore.transport.buffers import TransportBuffer
from torchstore.transport.neuron_efa import NeuronEFAVolumeCache, NeuronGeneratorVolume
from torchstore.transport.types import Request


class NeuronHBMStore(StorageImpl):
    """A storage volume whose backing store is a Neuron generator's HBM.

    Host this on the generator and the transport works end to end. It does two things and
    nothing else: publish the generator into the ``TransportContext`` where the volume-side
    handlers look for it, and route ``handshake``/``put`` into those handlers.

    It deliberately does NOT implement ``get``, ``get_meta``, ``delete`` or ``delete_batch``.
    The contents of this volume are the policy the engine is currently serving. Reading it back
    is not supported (see the module docstring) and deleting it means tearing the engine down,
    so inheriting the base class's ``NotImplementedError`` is the correct behavior rather than
    an omission.

    Args:
        generator: the object implementing :class:`NeuronGeneratorVolume`. On Trainium this is
            the vLLM-Neuron generator actor; in tests it is a fake. The store does not care,
            which is what makes the transport testable without hardware.
    """

    def __init__(self, generator: NeuronGeneratorVolume) -> None:
        super().__init__()
        self.generator = generator
        # Registered once, at construction. The handlers are given only a TransportContext, so
        # this is the only channel through which they can reach the engine.
        self.transport_context.get(NeuronEFAVolumeCache).generator = generator

    async def handshake(
        self,
        transport_buffer: TransportBuffer,
        requests: list[Request],
    ) -> dict[str, Any]:
        """Publish this replica's HBM destinations. Called once per session."""
        pairs = [(request, None) for request in requests]
        return await transport_buffer.recv_handshake(self.transport_context, pairs)

    async def put(
        self,
        transport_buffer: TransportBuffer,
        requests: list[Request],
    ) -> dict[str, Any]:
        """Quiesce and RDMA a policy version into HBM. Does not commit.

        The return value carries the per-rank ACKs the client needs before it may release the
        trainer's source, so unlike most volumes this put is not fire-and-forget.
        """
        pairs = [(request, None) for request in requests]
        return await transport_buffer.handle_put_request(self.transport_context, pairs)
