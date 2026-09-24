# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import importlib
import inspect
from collections import defaultdict
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

from torchstore.routing._model import RankRole
from torchstore.routing.client import RoutingClient
from torchstore.routing.coordinator import RoutingCoordinator
from torchstore.transport.types import TensorSlice
from torchstore.utils import slice_covers

client_module = importlib.import_module("torchstore.client")
routing_client_module = importlib.import_module("torchstore.routing.client")


def _endpoint_method(actor, name: str):
    descriptor = inspect.getattr_static(type(actor), name)
    method = getattr(descriptor._method, "__wrapped__", descriptor._method)
    return method.__get__(actor, type(actor))


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


class MemoryTransport:
    supports_inplace_resharding = True

    def __init__(
        self,
        volume_id: str,
        stores: defaultdict[str, dict[str, Any]],
    ) -> None:
        self.volume_id = volume_id
        self.stores = stores
        self.put_to_storage_volume = AsyncMock(side_effect=self._put)
        self.get_from_storage_volume = AsyncMock(side_effect=self._get)

    async def _put(self, requests) -> None:
        volume = self.stores[self.volume_id]
        for request in requests:
            if request.is_object:
                volume[request.key] = request.objects
            else:
                assert request.tensor_slice is not None
                assert request.tensor_val is not None
                volume[request.key] = (
                    request.tensor_slice,
                    request.tensor_val.detach().clone(),
                )

    async def _get(self, requests):
        volume = self.stores[self.volume_id]
        results = []
        for request in requests:
            stored = volume[request.key]
            if request.is_object:
                results.append(stored)
                continue

            stored_slice, tensor = stored
            requested_slice = request.tensor_slice
            assert requested_slice is not None
            assert slice_covers(stored_slice, requested_slice)
            indices = tuple(
                slice(offset - start, offset - start + size)
                for offset, start, size in zip(
                    requested_slice.offsets,
                    stored_slice.offsets,
                    requested_slice.local_shape,
                    strict=True,
                )
            )
            result = tensor[indices].clone()
            if request.tensor_val is not None:
                request.tensor_val.copy_(result)
                result = request.tensor_val
            results.append(result)
        return results


class TransportFactory:
    def __init__(self) -> None:
        self.stores: defaultdict[str, dict[str, Any]] = defaultdict(dict)
        self.transports: dict[str, MemoryTransport] = {}

    def __call__(self, volume_id: str) -> MemoryTransport:
        if volume_id not in self.transports:
            self.transports[volume_id] = MemoryTransport(volume_id, self.stores)
        return self.transports[volume_id]


class Strategy:
    def get_storage_volume(self, volume_id: str):
        return type("Volume", (), {"volume_id": volume_id})()


def transport_factory(monkeypatch) -> TransportFactory:
    factory = TransportFactory()

    def create_transport(volume):
        return factory(volume.volume_id)

    monkeypatch.setattr(client_module, "create_transport_buffer", create_transport)
    monkeypatch.setattr(
        routing_client_module, "create_transport_buffer", create_transport
    )
    return factory


def registered_clients(
    publishers: Mapping[str, Mapping[str, Any]],
    requesters: Mapping[str, Mapping[str, Any]],
    namespace: str = "model",
) -> dict[str, RoutingClient]:
    """Create clients and exchange their layouts through a real coordinator."""

    coordinator = RoutingCoordinator()
    coordinator_ref = SimpleNamespace(
        register_layouts=SimpleNamespace(
            call_one=_endpoint_method(coordinator, "register_layouts")
        )
    )
    roles = {
        **{rank: RankRole.PUBLISHER for rank in publishers},
        **{rank: RankRole.REQUESTER for rank in requesters},
    }
    clients = {
        rank: RoutingClient(rank, role, coordinator_ref, Strategy())
        for rank, role in roles.items()
    }
    state_dicts = publishers | requesters

    async def register() -> None:
        await _endpoint_method(coordinator, "init")(clients, Strategy())
        await asyncio.gather(
            *(
                client.register_state_dict_locally(state_dicts[rank], namespace)
                for rank, client in clients.items()
            )
        )

    asyncio.run(register())
    return clients
