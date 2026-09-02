# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Client-side execution of precomputed tensor routes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch
from torch.distributed.tensor import DTensor

from torchstore.client import LocalClient
from torchstore.logging import LatencyTracker
from torchstore.state_dict_utils import (
    _state_dict_mapping_key,
    _state_dict_storage_metadata,
)
from torchstore.strategy import TorchStoreStrategy
from torchstore.transport import create_transport_buffer
from torchstore.transport.types import Request
from torchstore.utils import same_slice_geometry

from ._model import KeyRegistration, RankRole
from .directory import RoutingDirectory
from .plan import RoutingPlan

if TYPE_CHECKING:
    from .coordinator import RoutingCoordinator

__all__ = ["RoutingClient"]


class RoutingClient(LocalClient):
    """A ``LocalClient`` backed by a rank-local precomputed directory."""

    def __init__(
        self,
        rank: str,
        role: RankRole,
        coordinator: "RoutingCoordinator",
        strategy: TorchStoreStrategy,
    ) -> None:
        super().__init__(RoutingDirectory(rank), strategy)
        self._role = role
        self._coordinator = coordinator
        # Storage key -> dtype selected during state-dict registration.
        self._wire_dtypes: dict[str, torch.dtype] = {}

    async def register_state_dict(
        self,
        state_dict: Mapping[str, Any],
        key: str,
        *,
        transfer_dtype: torch.dtype | None = None,
        preserve_dtype_keys: frozenset[str] = frozenset(),
    ) -> None:
        """Exchange this rank's layout for ``key`` and install its routes.

        Returns once every participant has registered ``key``. Only tensor
        geometry crosses the wire; the weights themselves never reach the
        coordinator. Each namespace has its own barrier, so a rank registers
        the state dicts it routes one at a time.
        """
        slices, element_sizes, dtypes, mapping = _state_dict_storage_metadata(
            state_dict,
            key,
            transfer_dtype=transfer_dtype,
            preserve_dtype_keys=preserve_dtype_keys,
        )
        self._wire_dtypes.update(dtypes)
        registrations = {
            name: KeyRegistration(tensor_slice, element_sizes[name])
            for name, tensor_slice in slices.items()
        }
        plan, services = await self._coordinator.register.call_one(
            rank=self._controller.rank,
            role=self._role,
            key=key,
            registrations=registrations,
        )
        self._controller.install(plan, services)
        await self._publish_mapping(key, mapping)

    @torch.no_grad()
    async def register_state_dict_locally(
        self,
        state_dict: Mapping[str, Any],
        key: str,
        *,
        transfer_dtype: torch.dtype | None = None,
        preserve_dtype_keys: frozenset[str] = frozenset(),
    ) -> None:
        """As :meth:`register_state_dict`, but plan here rather than centrally."""
        slices, element_sizes, dtypes, mapping = _state_dict_storage_metadata(
            state_dict,
            key,
            transfer_dtype=transfer_dtype,
            preserve_dtype_keys=preserve_dtype_keys,
        )
        self._wire_dtypes.update(dtypes)
        registrations = {
            name: KeyRegistration(tensor_slice, element_sizes[name])
            for name, tensor_slice in slices.items()
        }
        rank = self._controller.rank
        publishers, requesters, services = (
            await self._coordinator.register_layouts.call_one(
                rank=rank,
                role=self._role,
                key=key,
                registrations=registrations,
            )
        )
        self._controller.install(
            RoutingPlan.build_for(rank, publishers, requesters), services
        )
        await self._publish_mapping(key, mapping)

    async def _publish_mapping(self, key: str, mapping: Mapping[str, Any]) -> None:
        """Publish this ingress's mapping entries once during registration."""
        if self._role != RankRole.REQUESTER:
            return
        prefix = f"{key}/"
        direct_keys = {
            storage_key.removeprefix(prefix)
            for storage_key, entry in self._controller.routes.keys.items()
            if storage_key.startswith(prefix)
            and any(route.wait_for_relay_id is None for route in entry.routes)
        }
        if direct_keys:
            await self.put_batch(
                {
                    _state_dict_mapping_key(key): {
                        flat_key: mapping[flat_key] for flat_key in direct_keys
                    }
                }
            )

    @torch.no_grad()
    async def put_batch(self, entries: dict[str, torch.Tensor | Any]) -> None:
        """Put batches to the one volume this rank publishes through."""
        assert (
            isinstance(entries, dict) and entries
        ), "put_batch requires a non-empty dict"

        latency_tracker = LatencyTracker("put_batch")

        requests = []
        for key, value in entries.items():
            if isinstance(value, (torch.Tensor, DTensor)):
                request = Request.from_any(key, value)
                self._match_plan(request)
            else:
                request = Request.from_objects(key, value)
            requests.append(request)

        storage_volume_ref = self.strategy.get_storage_volume(
            self._controller.routes.volume_id
        )
        transport_buffer = create_transport_buffer(storage_volume_ref)
        latency_tracker.track_step("create transport buffer")

        await transport_buffer.put_to_storage_volume(requests)
        latency_tracker.track_step("put_to_storage_volume")
        latency_tracker.track_e2e()

    def _match_plan(self, request: Request) -> None:
        """Check one tensor against the slice this rank was planned to store."""
        planned = self._controller.routes.keys.get(request.key)
        if planned is None:
            raise KeyError(
                f"rank {self._controller.rank!r} does not store {request.key!r}"
            )
        expected = planned.tensor_slice
        assert request.tensor_val is not None
        wire_dtype = self._wire_dtypes.get(request.key)
        if wire_dtype is not None and request.tensor_val.dtype != wire_dtype:
            if not request.tensor_val.is_floating_point():
                raise ValueError(
                    f"published tensor for {request.key!r} has dtype "
                    f"{request.tensor_val.dtype}, expected {wire_dtype}"
                )
            request.tensor_val = request.tensor_val.to(wire_dtype)
        if request.tensor_slice is None:
            if tuple(request.tensor_val.shape) != tuple(expected.local_shape):
                raise ValueError(
                    f"published tensor for {request.key!r} has shape "
                    f"{tuple(request.tensor_val.shape)}, expected {expected.local_shape}"
                )
            request.tensor_slice = expected
        elif not same_slice_geometry(request.tensor_slice, expected):
            raise ValueError(
                f"published DTensor slice for {request.key!r} "
                "does not match the routing plan"
            )

    async def _fetch(self, requests: list[Request]) -> dict[str, Any]:
        """Read objects, then routed slices, relaying between peers as planned.

        Args:
            requests: Pre-built Request per key (may include tensor_slice).

        Returns:
            dict mapping each key to its raw fetched data.
        """
        tracker = LatencyTracker("routed_fetch")
        tensor_requests = []
        objects = {}
        for request in requests:
            if self._controller.routes.is_planned_tensor(request.key):
                tensor_requests.append(request)
            else:
                objects[request.key] = await self._fetch_object(request.key)
        tracker.track_step("objects")
        if objects:
            assert not tensor_requests, "cannot mix objects and routed tensors"
            tracker.track_e2e()
            return objects

        resolved = self._resolve_destinations(tensor_requests)
        tracker.track_step("resolve")

        by_key = {request.key: request for request in tensor_requests}
        direct = [by_key[key] for key in resolved.keys_for(relay=False)]
        published = await super()._fetch(direct) if direct else {}
        tracker.track_step("publisher")

        await self._relay(resolved, direct, published)
        tracker.track_step("relay_publish")

        peer = [by_key[key] for key in resolved.keys_for(relay=True)]
        if peer:
            await self._controller.wait_ready(resolved.relay_ids)
        relayed = await super()._fetch(peer) if peer else {}
        tracker.track_step("relay")

        tracker.track_e2e()
        return published | relayed

    def _resolve_destinations(self, requests: list[Request]) -> Any:
        """Look up each request's planned route, checking the caller's buffer."""
        resolved = self._controller.resolve_get_batch(
            [request.meta_only() for request in requests]
        )
        for request in requests:
            target = resolved.targets[request.key]
            if request.tensor_val is None:
                continue
            wire_dtype = self._wire_dtypes.get(request.key)
            if wire_dtype is not None and request.tensor_val.dtype != wire_dtype:
                raise ValueError(
                    f"destination for {request.key!r} has dtype "
                    f"{request.tensor_val.dtype}, expected {wire_dtype}"
                )
            if tuple(request.tensor_val.shape) != tuple(target.local_shape):
                raise ValueError(
                    f"destination for {request.key!r} has shape "
                    f"{tuple(request.tensor_val.shape)}, expected {target.local_shape}"
                )
        return resolved

    async def _relay(
        self, resolved: Any, direct: list[Request], published: dict[str, Any]
    ) -> None:
        """Store what this rank just read so its peers can read it from here."""
        if not resolved.notifications:
            return
        await self.put_batch(
            {
                request.key: published[request.key]
                for request in direct
                if request.key in published
            }
        )
        await self._controller.notify_ready(resolved.notifications)

    async def _fetch_object(self, key: str) -> Any:
        """Merge one object across every volume the plan names for it."""
        merged: dict[str, Any] = {}
        for volume_id in sorted(self._controller.locate_volumes([key])[key]):
            volume = self.strategy.get_storage_volume(volume_id)
            transport_buffer = create_transport_buffer(volume)
            [part] = await transport_buffer.get_from_storage_volume(
                [Request(key=key, is_object=True)]
            )
            merged.update(part)
        return merged

    async def _locate_volumes(self, keys: list[str]):
        return self._controller.locate_volumes(keys)

    @staticmethod
    def _unsupported(operation: str) -> NotImplementedError:
        return NotImplementedError(f"{operation} is unsupported by precomputed routing")

    async def keys(self, prefix: str | None = None) -> list[str]:
        raise self._unsupported("keys")

    async def delete(self, key: str) -> None:
        raise self._unsupported("delete")

    async def delete_batch(self, keys: list[str]) -> None:
        raise self._unsupported("delete_batch")

    async def exists(self, key: str) -> bool:
        raise self._unsupported("exists")
