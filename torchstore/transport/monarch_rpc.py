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


class MonarchRPCTransportBuffer(TransportBuffer):
    """Simple RPC-based transport that passes data through serialization.

    Unlike RDMA transport which requires explicit memory registration,
    this transport simply stores data as member variables that are
    automatically serialized when the buffer is sent via RPC.
    """

    requires_handshake: bool = False

    def __init__(self, storage_volume_ref: "StorageVolumeRef"):
        super().__init__(storage_volume_ref)
        self.data: Any = None

    async def handle_put_request(
        self, request: Request, current_object, storage_transport_context
    ) -> Any:
        """Return the data from the request to be stored.

        For RPC transport, data is already available in the request
        since it was serialized and sent via RPC.

        """
        if request.is_object:
            return request.objects
        return request.tensor_val

    async def handle_get_request(self, data, storage_transport_context) -> None:
        """Store the data to be sent back to the client.

        For RPC transport, we just store the data as a member variable.
        It will be serialized when this buffer is returned via RPC.
        """
        self.data = data

    async def _handle_storage_volume_response(
        self, transport_buffer: "TransportBuffer"
    ) -> Any:
        """Extract the data from the response buffer.

        The transport_buffer is the same instance that was modified
        on the storage volume side, with data stored in self.data.
        """
        return transport_buffer.data

    async def drop(self) -> None:
        """No cleanup needed for RPC transport."""
        self.data = None
