# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import threading
import uuid
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Collection,
    Coroutine,
    Mapping,
)
from typing import Any

import torchstore as ts
from monarch.actor import get_or_spawn_controller, this_host
from torchstore.routing._model import KeyRegistration, RankRole, Registrations
from torchstore.routing.client import RoutingClient
from torchstore.routing.control_plane import ControlPlane
from torchstore.routing.coordinator import MonarchControlPlane, RoutingCoordinator
from torchstore.routing.layout_registry import LayoutRegistry
from torchstore.transport import TransportType
from torchstore.transport.types import TensorSlice


def tensor_slice(
    offsets: tuple[int, ...],
    local_shape: tuple[int, ...],
    *,
    global_shape: tuple[int, ...] = (8, 6),
    coordinates: tuple[int, ...] = (),
    mesh_shape: tuple[int, ...] = (),
) -> TensorSlice:
    """Build tensor-slice metadata with empty placement by default."""
    return TensorSlice(
        offsets=offsets,
        coordinates=coordinates,
        global_shape=global_shape,
        local_shape=local_shape,
        mesh_shape=mesh_shape,
    )


async def routing_clients(
    publishers: Mapping[str, Mapping[str, Any]],
    requesters: Mapping[str, Mapping[str, Any]],
) -> dict[str, RoutingClient]:
    """Spawn routing infrastructure without registering state-dict layouts."""
    publisher_mesh = this_host().spawn_procs(per_host={"procs": len(publishers)})
    store_name = f"routing-test-{uuid.uuid4()}"
    strategy = ts.LocalRankStrategy(TransportType.MonarchRPC)
    await ts.initialize(
        mesh=publisher_mesh,
        strategy=strategy,
        client_type=ts.ClientType.ROUTING,
        store_name=store_name,
    )
    coordinator = await get_or_spawn_controller(store_name, RoutingCoordinator)
    control_plane = MonarchControlPlane(coordinator)
    roles = {
        **{rank: RankRole.PUBLISHER for rank in publishers},
        **{rank: RankRole.REQUESTER for rank in requesters},
    }
    clients = {
        rank: RoutingClient(rank, role, control_plane, strategy)
        for rank, role in roles.items()
    }
    return clients


async def registered_clients(
    publishers: Mapping[str, Mapping[str, Any]],
    requesters: Mapping[str, Mapping[str, Any]],
    namespace: str = "model",
) -> dict[str, RoutingClient]:
    """Spawn clients and explicitly register layouts for low-level API tests."""
    clients = await routing_clients(publishers, requesters)
    await asyncio.gather(
        *(
            clients[rank].register_layout(state_dict, namespace)
            for rank, state_dict in publishers.items()
        ),
        *(
            clients[rank].get_layouts(state_dict, namespace)
            for rank, state_dict in requesters.items()
        ),
    )
    return clients


class HostedRegistry(ControlPlane):
    """Runs each call as a task on the registry's loop, like a concurrent endpoint."""

    def __init__(
        self, registry: LayoutRegistry, loop: asyncio.AbstractEventLoop
    ) -> None:
        self._registry = registry
        self._loop = loop

    async def _on_host(self, coroutine: Coroutine[Any, Any, Any]) -> Any:
        return await asyncio.wrap_future(
            asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        )

    async def init(
        self, publishers: Collection[str], strategy: ts.TorchStoreStrategy
    ) -> None:
        await self._on_host(self._registry.init(publishers, strategy))

    async def strategy(self) -> ts.TorchStoreStrategy:
        return await self._on_host(self._registry.strategy())

    async def register_layout(
        self, rank: str, key: str, registrations: Mapping[str, KeyRegistration]
    ) -> None:
        await self._on_host(self._registry.register_layout(rank, key, registrations))

    async def get_layouts(
        self, key: str, *, timeout_s: float | None = None
    ) -> Registrations:
        layouts = await self._on_host(
            self._registry.get_layouts(key, timeout_s=timeout_s)
        )
        # Callers own their layouts, as with Monarch's pickled replies.
        return {rank: dict(registrations) for rank, registrations in layouts.items()}


@contextlib.asynccontextmanager
async def registry_store() -> AsyncIterator[Callable[[], Awaitable[HostedRegistry]]]:
    """Host one ``LayoutRegistry`` on a thread's own event loop, without Monarch.

    Exit fails if the host loop has not stopped within 10 s.
    """
    hosting: concurrent.futures.Future = concurrent.futures.Future()

    async def host() -> None:
        stop = asyncio.Event()
        hosting.set_result((asyncio.get_running_loop(), stop))
        await stop.wait()

    thread = threading.Thread(target=asyncio.run, args=(host(),), daemon=True)
    thread.start()
    loop, stop = await asyncio.wrap_future(hosting)
    registry = LayoutRegistry()

    async def connect() -> HostedRegistry:
        return HostedRegistry(registry, loop)

    try:
        yield connect
    finally:
        # ``asyncio.run`` cancels any call still running on the host loop.
        loop.call_soon_threadsafe(stop.set)
        await asyncio.to_thread(thread.join, 10)
        assert not thread.is_alive(), "the registry's host loop did not stop"
