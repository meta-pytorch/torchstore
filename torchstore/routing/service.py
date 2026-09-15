# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Lightweight readiness service for precomputed tensor routes."""

from __future__ import annotations

import asyncio

from monarch.actor import (  # type: ignore[import-untyped]
    Actor,
    concurrent_endpoint,
    endpoint,
    ProcMesh,
)

from torchstore.utils import spawn_actors

__all__ = ["RoutingService", "RoutingServiceGroup"]


class RoutingService(Actor):
    """Rank-local readiness service used by :class:`RoutingClient`.

    Tensor bytes move from the client straight to storage volumes; this actor
    only coordinates readiness between requester ranks. Each relay advances a
    private generation counter so completion from an older update cannot
    release a later one.
    """

    actor_name = "RoutingServices"

    def __init__(self, id_func) -> None:
        self._ready: dict[str, int] = {}
        self._conditions: dict[str, asyncio.Condition] = {}
        self.rank: str = id_func()

    @classmethod
    async def spawn(cls, mesh: ProcMesh, id_func) -> RoutingService:
        """Spawn one service per mesh rank, each naming itself via ``id_func``."""
        return await spawn_actors(
            1,
            cls,
            cls.actor_name,
            mesh,
            id_func=id_func,
        )

    @endpoint
    async def get_id(self) -> str:
        return self.rank

    def _condition(self, relay_id: str) -> asyncio.Condition:
        condition = self._conditions.get(relay_id)
        if condition is None:
            condition = asyncio.Condition()
            self._conditions[relay_id] = condition
        return condition

    @concurrent_endpoint
    async def wait_ready(
        self,
        generation: int,
        relay_id: str,
    ) -> None:
        """Wait locally until the source reports that its relay slice is stored."""
        condition = self._condition(relay_id)
        async with condition:
            await condition.wait_for(lambda: self._ready.get(relay_id, 0) >= generation)

    @concurrent_endpoint
    async def notify_ready(
        self,
        generation: int,
        relay_id: str,
    ) -> None:
        """Endpoint invoked by an ingress requester after its direct put."""
        condition = self._condition(relay_id)
        async with condition:
            self._ready[relay_id] = max(generation, self._ready.get(relay_id, 0))
            condition.notify_all()


class RoutingServiceGroup:
    """Resolves each rank's readiness service across several ProcMeshes.

    The mirror of ``MultiMeshStrategy`` for services: each service names itself
    at spawn, and this indexes them by that name so a rank can be handed
    handles to its relay peers. A coordinate alone cannot say which mesh a
    service is on, so the mesh is remembered per rank.
    """

    def __init__(self) -> None:
        self.rank_to_coord: dict[str, dict] = {}
        self.rank_to_mesh: dict[str, RoutingService] = {}

    async def set_services(self, *service_meshes: RoutingService) -> None:
        """Index every service mesh. Rank names must be unique across the set."""
        for mesh in service_meshes:
            for coord, rank in await mesh.get_id.call():
                if rank in self.rank_to_mesh:
                    raise ValueError(f"duplicate routing rank {rank!r}")
                self.rank_to_coord[rank] = coord
                self.rank_to_mesh[rank] = mesh

    @property
    def ranks(self) -> frozenset[str]:
        """Every participating rank, i.e. the routing roster."""
        return frozenset(self.rank_to_mesh)

    def get_service(self, rank: str) -> RoutingService:
        """Retrieve the service actor for a given rank."""
        return self.rank_to_mesh[rank].slice(**self.rank_to_coord[rank])
