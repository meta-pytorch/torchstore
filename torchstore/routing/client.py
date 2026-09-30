# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Client-side execution of precomputed tensor routes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch.distributed.tensor import DTensor

from torchstore.client import LocalClient
from torchstore.logging import LatencyTracker
from torchstore.state_dict_utils import (
    _state_dict_mapping_key,
    _state_dict_storage_metadata,
    DELIM,
    MAPPING,
)
from torchstore.strategy import TorchStoreStrategy
from torchstore.transport import create_transport_buffer
from torchstore.transport.types import Request
from torchstore.utils import same_slice_geometry
from ._model import KeyRegistration, RankRole
from .coordinator import RoutingCoordinator
from .directory import RoutingDirectory
from .plan import RoutingPlan

__all__ = ["RoutingClient"]


class RoutingClient(LocalClient):
    """A ``LocalClient`` backed by a rank-local precomputed directory."""

    def __init__(
        self,
        rank: str,
        role: RankRole,
        coordinator: RoutingCoordinator,
        strategy: TorchStoreStrategy,
    ) -> None:
        directory = RoutingDirectory(rank)
        super().__init__(directory, strategy)
        self._routing_directory = directory
        self._role = role
        self._coordinator = coordinator
        self._registered_layouts: set[str] = set()

    @torch.no_grad()
    async def register_layout(
        self,
        state_dict: Mapping[str, Any],
        key: str,
        *,
        transfer_dtype: torch.dtype | None = None,
        preserve_dtype_keys: frozenset[str] = frozenset(),
    ) -> None:
        """Register and install one publisher layout, once per state dict."""
        if key in self._registered_layouts:
            return
        if self._role != RankRole.PUBLISHER:
            raise RuntimeError("only publisher clients register layouts")
        slices, dtypes, _mapping = _state_dict_storage_metadata(
            state_dict,
            key,
            transfer_dtype=transfer_dtype,
            preserve_dtype_keys=preserve_dtype_keys,
        )
        registrations = {
            name: KeyRegistration(tensor_slice, dtype=dtypes[name])
            for name, tensor_slice in slices.items()
        }
        rank = self._routing_directory.rank
        self._routing_directory.install(RoutingPlan.for_publisher(rank, registrations))
        await self._coordinator.register_layout.call_one(
            rank=rank,
            key=key,
            registrations=registrations,
        )
        self._registered_layouts.add(key)

    @torch.no_grad()
    async def get_layouts(
        self,
        state_dict: Mapping[str, Any] | None,
        key: str,
    ) -> None:
        """Fetch publisher layouts and install one requester plan."""
        if key in self._registered_layouts:
            return
        if self._role != RankRole.REQUESTER:
            raise RuntimeError("only requester clients get layouts")
        if state_dict is None:
            raise ValueError(
                "user_state_dict is required for the first routed get_state_dict"
            )

        slices, dtypes, mapping = _state_dict_storage_metadata(state_dict, key)
        registrations = {
            name: KeyRegistration(tensor_slice, dtype=dtypes[name])
            for name, tensor_slice in slices.items()
        }
        rank = self._routing_directory.rank
        (
            publishers,
            requesters,
            services,
        ) = await self._coordinator.register_requester_layout.call_one(
            rank=rank,
            key=key,
            registrations=registrations,
        )
        self._routing_directory.install(
            RoutingPlan.for_requester(
                rank,
                registrations,
                publishers,
                requesters=requesters,
            ),
            services,
        )
        await self._publish_mapping(key, mapping)
        self._registered_layouts.add(key)

    async def _publish_mapping(self, key: str, mapping: Mapping[str, Any]) -> None:
        """Publish this ingress's mapping entries once during registration."""
        if self._role != RankRole.REQUESTER:
            return
        prefix = f"{key}/"
        direct_keys = {
            storage_key.removeprefix(prefix)
            for storage_key, entry in self._routing_directory.routes.keys.items()
            if storage_key.startswith(prefix)
            and entry.route is not None
            and entry.route.wait_for_relay_id is None
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
            self._routing_directory.routes.volume_id
        )
        transport_buffer = create_transport_buffer(storage_volume_ref)
        latency_tracker.track_step("create transport buffer")

        await transport_buffer.put_to_storage_volume(requests)
        latency_tracker.track_step("put_to_storage_volume")
        latency_tracker.track_e2e()

    def _match_plan(self, request: Request) -> None:
        """Check one tensor against the slice this rank was planned to store."""
        planned = self._routing_directory.routes.keys.get(request.key)
        if planned is None:
            raise KeyError(
                f"rank {self._routing_directory.rank!r} does not store {request.key!r}"
            )
        expected = planned.tensor_slice
        if request.tensor_slice is None:
            assert request.tensor_val is not None
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
            if self._routing_directory.routes.is_planned_tensor(request.key):
                tensor_requests.append(request)
            elif request.key.endswith(f"{DELIM}{MAPPING}"):
                objects[request.key] = await self._fetch_object(request.key)
            else:
                raise KeyError(
                    f"rank {self._routing_directory.rank!r} has no tensor route for "
                    f"{request.key!r}"
                )
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
            await self._routing_directory.wait_ready(resolved.relay_ids)
        relayed = await super()._fetch(peer) if peer else {}
        tracker.track_step("relay")

        tracker.track_e2e()
        return published | relayed

    def _resolve_destinations(self, requests: list[Request]) -> Any:
        """Look up each request's planned route, checking the caller's buffer."""
        resolved = self._routing_directory.resolve_get_batch(
            [request.meta_only() for request in requests]
        )
        for request in requests:
            target = resolved.targets[request.key]
            if request.tensor_val is not None and tuple(
                request.tensor_val.shape
            ) != tuple(target.local_shape):
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
        await self._routing_directory.notify_ready(resolved.notifications)

    async def _fetch_object(self, key: str) -> Any:
        """Merge a state-dict mapping and retain this requester's planned keys."""
        merged: dict[str, Any] = {}
        volumes = await self._directory.locate_volumes([key])
        for volume_id in sorted(volumes[key]):
            volume = self.strategy.get_storage_volume(volume_id)
            transport_buffer = create_transport_buffer(volume)
            [part] = await transport_buffer.get_from_storage_volume(
                [Request(key=key, is_object=True)]
            )
            if not isinstance(part, Mapping):
                raise TypeError(
                    f"routed object {key!r} from volume {volume_id!r} must be a "
                    f"mapping, got {type(part)}"
                )
            for flat_key, path in part.items():
                if flat_key in merged and merged[flat_key] != path:
                    raise ValueError(
                        f"conflicting state-dict mapping for {flat_key!r}: "
                        f"{merged[flat_key]!r} != {path!r}"
                    )
                merged[flat_key] = path

        namespace = key[: -len(f"{DELIM}{MAPPING}")]
        return {
            flat_key: path
            for flat_key, path in merged.items()
            if self._routing_directory.routes.is_planned_tensor(
                f"{namespace}{DELIM}{flat_key}"
            )
        }

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
