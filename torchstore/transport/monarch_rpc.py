# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MonarchRPC transport buffer - simple RPC-based data transport.

This transport passes tensor data through Monarch's RPC serialization.
No special buffer management is needed - data is simply stored as
member variables and serialized/deserialized automatically.
"""

from typing import Any, TYPE_CHECKING

from torchstore.transport.buffers import TransportBuffer
from torchstore.transport.types import Request

if TYPE_CHECKING:
    from torchstore.strategy import StorageVolumeRef
    from torchstore.transport.buffers import TransportContext


class MonarchRPCTransportBuffer(TransportBuffer):
    """Simple RPC-based transport that passes data through serialization.

    Unlike RDMA transport which requires explicit memory registration,
    this transport simply stores data as member variables that are
    automatically serialized when the buffer is sent via RPC.
    """

    requires_handshake: bool = False

    def __init__(self, storage_volume_ref: "StorageVolumeRef"):
        super().__init__(storage_volume_ref)
        self.data: Any = None  # Carries data for both PUT and GET

    async def _pre_put_hook(self, request: Request) -> None:
        """Store data from request to be serialized with this buffer."""
        self.data = request.objects if request.is_object else request.tensor_val

    async def handle_put_request(
        self, ctx: "TransportContext", request: Request, current_object
    ) -> Any:
        """Return the data from the buffer to be stored."""
        return self.data

    async def handle_get_request(self, ctx: "TransportContext", data) -> None:
        """Store the data to be sent back to the client."""
        self.data = data

    async def _handle_storage_volume_response(
        self, transport_buffer: "TransportBuffer"
    ) -> Any:
        """Extract the data from the response buffer."""
        return transport_buffer.data

    async def drop(self) -> None:
        """Clean up stored references for RPC transport."""
        self.data = None
