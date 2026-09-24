# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Rank-local metadata and coordination for precomputed tensor routes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from torchstore.controller import ObjectType, StorageInfo
from torchstore.transport.types import Request, TensorSlice
from torchstore.utils import same_slice_geometry

from ._model import DestinationRoute, LocalRouteTable, RankRole

if TYPE_CHECKING:
    from .plan import RoutingPlan


@dataclass(frozen=True)
class _ResolvedRead:
    """Transport-neutral routes for one metadata-only read batch."""

    targets: Mapping[str, TensorSlice]
    routes: Mapping[str, DestinationRoute]


class RoutingDirectory:
    """Rank-local route metadata."""

    def __init__(self, rank: str) -> None:
        self.rank = rank
        self._routes: LocalRouteTable | None = None

    @property
    def routes(self) -> LocalRouteTable:
        if self._routes is None:
            raise RuntimeError(
                f"rank {self.rank!r} has no routes; register a state dict first"
            )
        return self._routes

    def install(self, plan: RoutingPlan) -> None:
        """Add one namespace's routes to this rank's table.

        Storage keys carry their namespace, so tables for different state dicts
        merge without collision.
        """
        table = plan._local(self.rank)
        if self._routes is None:
            self._routes = table
            return
        if (table.volume_id, table.role) != (self._routes.volume_id, self._routes.role):
            raise ValueError(
                f"rank {self.rank!r} cannot change volume or role between state dicts"
            )
        self._routes = replace(
            self._routes,
            keys={**self._routes.keys, **table.keys},
        )

    def _target_slice(self, request: Request) -> TensorSlice:
        entry = self.routes.keys.get(request.key)
        if entry is None:
            raise KeyError(f"rank {self.rank!r} does not request {request.key!r}")
        planned = entry.tensor_slice
        if request.tensor_slice is not None and not same_slice_geometry(
            planned, request.tensor_slice
        ):
            raise ValueError(
                f"requested slice for {request.key!r} does not match the plan"
            )
        return planned

    def resolve_get_batch(self, requests: Sequence[Request]) -> _ResolvedRead:
        if any(request.tensor_val is not None for request in requests):
            raise ValueError("resolve_get_batch requires metadata-only requests")
        if self.routes.role != RankRole.REQUESTER:
            raise RuntimeError("only requester clients get routed tensors")
        targets = {request.key: self._target_slice(request) for request in requests}
        routes = {}
        for request in requests:
            entries = self.routes.lookup(request.key)
            if len(entries) != 1:
                raise RuntimeError(
                    f"local route does not uniquely fill {request.key!r}"
                )
            routes[request.key] = entries[0]
        return _ResolvedRead(targets, routes)

    def locate_volumes(
        self,
        keys: list[str],
    ) -> dict[str, dict[str, StorageInfo]]:
        """Volumes holding each key, and the form it is stored in."""
        return {key: self._locate(key) for key in keys}

    def _locate(self, key: str) -> dict[str, StorageInfo]:
        if not self.routes.is_planned_tensor(key):
            return {
                source: StorageInfo(object_type=ObjectType.OBJECT)
                for source in self._object_volumes()
            }
        volumes: dict[str, StorageInfo] = {}
        for route in self.routes.lookup(key):
            for transfer in route.transfers:
                info = volumes.setdefault(
                    transfer.source_volume_id,
                    StorageInfo(object_type=ObjectType.TENSOR_SLICE),
                )
                info.tensor_slices.add(transfer.segment)
        return volumes

    def _object_volumes(self) -> set[str]:
        """Volumes this rank reads tensors from."""
        return {
            transfer.source_volume_id
            for entry in self.routes.keys.values()
            for route in entry.routes
            for transfer in route.transfers
        } or {self.routes.volume_id}
