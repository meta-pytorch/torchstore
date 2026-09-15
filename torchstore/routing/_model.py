# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Internal metadata and local actions for precomputed tensor routes."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Mapping, Tuple

from torchstore.transport.types import TensorSlice

__all__ = [
    "DestinationRoute",
    "KeyPlan",
    "LocalRouteTable",
    "KeyRegistration",
    "Registrations",
    "RankRole",
    "RouteEntry",
    "Transfer",
]


class RankRole(str, Enum):
    PUBLISHER = "publisher"
    REQUESTER = "requester"


@dataclass(frozen=True)
class KeyRegistration:
    """What one rank holds for one storage key"""

    tensor_slice: TensorSlice
    # Bytes per element of the transferred dtype
    element_size: int


# rank -> storage key -> what that rank holds for it
Registrations = Dict[str, Mapping[str, KeyRegistration]]


@dataclass(frozen=True)
class Transfer:
    """One fixed read from a source volume in global tensor coordinates."""

    source: str
    source_volume_id: str
    segment: TensorSlice
    nbytes: int


@dataclass(frozen=True)
class DestinationRoute:
    """Reads and relay coordination for one local requester slice."""

    destination_slice: TensorSlice
    transfers: Tuple[Transfer, ...]
    # Relay IDs carry their storage key, so plans built for different
    # state-dict namespaces cannot signal each other through a shared service.
    wait_for_relay_id: str | None = None
    notify_relay_id: str | None = None
    notify_peers: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.wait_for_relay_id is not None and self.notify_relay_id is not None:
            raise ValueError("a pull route cannot wait and notify")
        if bool(self.notify_peers) != (self.notify_relay_id is not None):
            raise ValueError("relay notification requires an ID and peers")


RouteEntry = Tuple[DestinationRoute, ...]


@dataclass(frozen=True)
class KeyPlan:
    """The slice one rank holds for a storage key, and the routes that fill it"""

    tensor_slice: TensorSlice
    # A publisher's slice is served to others, so its `routes` stay empty.
    routes: RouteEntry = ()


@dataclass(frozen=True)
class LocalRouteTable:
    """One rank's immutable route table, installed when layouts are exchanged."""

    rank: str
    volume_id: str
    role: RankRole
    keys: Mapping[str, KeyPlan] = field(default_factory=dict)

    def is_planned_tensor(self, key: str) -> bool:
        """Whether this rank has a planned slice for the key."""
        return key in self.keys

    def plan(self, key: str) -> KeyPlan:
        planned = self.keys.get(key)
        if planned is None:
            raise KeyError(f"rank {self.rank!r} has no metadata for key {key!r}")
        return planned

    def lookup(self, key: str) -> RouteEntry:
        return self.plan(key).routes
