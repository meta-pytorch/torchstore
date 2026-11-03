# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The carrier: request/reply calls between TorchStore processes over TCP, without Monarch.

Internal and may change. A replacement behind ``serve``, ``call``, ``close``, ``stats``, ``wait_future`` and a server's
``dispatcher`` must keep FIFO and concurrent dispatch, deadlines, the errors below with ``VOLUME_CLOSED``, request
tensors sent from caller memory uncopied, replies copied once and fork safety.
"""

from torchstore.carrier.codec import (
    CallTimeoutError,
    CarrierAddress,
    CarrierError,
    CarrierProtocolError,
    JsonRecord,
    PeerLostError,
    RemoteError,
    StaleAddressError,
)
from torchstore.carrier.runtime import (
    CarrierClient,
    CarrierServer,
    close_all,
    close_scope,
    VOLUME_CLOSED,
    wait_future,
)

__all__ = [
    "CallTimeoutError",
    "CarrierAddress",
    "CarrierClient",
    "CarrierError",
    "CarrierProtocolError",
    "CarrierServer",
    "close_all",
    "close_scope",
    "JsonRecord",
    "PeerLostError",
    "RemoteError",
    "StaleAddressError",
    "VOLUME_CLOSED",
    "wait_future",
]
