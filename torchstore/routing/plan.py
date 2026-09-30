# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Build, inspect, and distribute a tensor routing plan."""

from __future__ import annotations

import zlib
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from enum import Enum
from typing import TypeVar

import torch

from torchstore.transport.types import TensorSlice
from torchstore.utils import get_slice_intersection, get_slice_numel
from ._model import (
    DestinationRoute,
    KeyPlan,
    KeyRegistration,
    LocalRouteTable,
    RankRole,
    Registrations,
    Transfer,
)

__all__ = ["Balance", "KeyRegistration", "RoutingPlan"]

_Candidate = TypeVar("_Candidate")
_GeometryKey = tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
# One replication set: its geometry and the ranks reporting it.
_Group = tuple[_GeometryKey, list[tuple[str, TensorSlice]]]


def _geometry_key(tensor_slice: TensorSlice) -> _GeometryKey:
    return (
        tensor_slice.global_shape,
        tensor_slice.offsets,
        tensor_slice.local_shape,
    )


class Balance(str, Enum):
    """How to choose between ranks holding or wanting byte-identical data."""

    # Based on knowledge of global load
    LEAST_LOADED = "least_loaded"
    # Pure function of the key
    ROTATE = "rotate"

    @property
    def sequential(self) -> bool:
        """Whether a choice depends on the choices made before it."""
        return self is Balance.LEAST_LOADED


def _itself(candidate: str) -> str:
    """Name of a candidate that is already just a rank."""
    return candidate


class _LeastLoaded:
    """Hands each choice to whichever candidate has been given the fewest bytes."""

    def __init__(self) -> None:
        self._assigned: Counter[str] = Counter()

    def choose(
        self,
        key: str,
        index: int,
        candidates: Sequence[_Candidate],
        nbytes: int,
        name: Callable[[_Candidate], str] = _itself,
    ) -> _Candidate:
        chosen = min(
            candidates,
            key=lambda candidate: (self._assigned[name(candidate)], name(candidate)),
        )
        self._assigned[name(chosen)] += nbytes
        return chosen


class _Rotate:
    """Hands each choice to a candidate picked by hashing the key."""

    def choose(
        self,
        key: str,
        index: int,
        candidates: Sequence[_Candidate],
        nbytes: int,
        name: Callable[[_Candidate], str] = _itself,
    ) -> _Candidate:
        return candidates[(zlib.crc32(key.encode()) + index) % len(candidates)]


_BALANCERS = {Balance.LEAST_LOADED: _LeastLoaded, Balance.ROTATE: _Rotate}


def _geometry_slice(geometry: _GeometryKey) -> TensorSlice:
    """Inverse of :func:`_geometry_key`. Mesh placement is not part of a route."""
    global_shape, offsets, local_shape = geometry
    return TensorSlice(
        offsets=offsets,
        coordinates=(),
        global_shape=global_shape,
        local_shape=local_shape,
        mesh_shape=(),
    )


def _group_by_geometry(registrations: Registrations) -> dict[str, list[_Group]]:
    """
    Ranks reporting byte-identical data, in storage key then geometry order,
    with each group's members sorted too.

    Sorted for reproducibility and determinism for load balancing.

    On the requester side these are replication sets, one fetch to share:

        R0 wants "weights" rows 0-7  --+
                                       +--> same geometry, one group
        R1 wants "weights" rows 0-7  --+
        R2 wants "weights" rows 0-3  ----> different geometry, its own group

    On the publisher side they are interchangeable sources, so a tensor
    replicated across trainer DP collapses to one group.
    """
    grouped: defaultdict[
        str, defaultdict[_GeometryKey, list[tuple[str, TensorSlice]]]
    ] = defaultdict(lambda: defaultdict(list))
    for rank, items in registrations.items():
        for storage_key, item in items.items():
            geometry = _geometry_key(item.tensor_slice)
            grouped[storage_key][geometry].append((rank, item.tensor_slice))
    return {
        key: [
            (geometry, sorted(members))
            for geometry, members in sorted(grouped[key].items())
        ]
        for key in sorted(grouped)
    }


class _Builder:
    """Builds one requester's routes from its layout and all publishers."""

    def __init__(
        self,
        rank: str,
        registrations: Mapping[str, KeyRegistration],
        publishers: Registrations,
        requesters: Registrations | None,
        balance: Balance,
    ) -> None:
        self.rank = rank
        self.registrations = registrations
        self.publishers = publishers
        self.requesters = {**(requesters or {}), rank: registrations}
        self.balance = _BALANCERS[balance]()
        # storage_key -> wire dtype
        self.dtypes: dict[str, torch.dtype] = {}
        # storage_key -> [(published slice, ranks holding it)], in geometry order
        self.holders_by_key: dict[str, list[tuple[TensorSlice, list[str]]]] = {}
        # Slices repeat across keys and ranks far more often than they differ:
        # geometry carries no key, so every key of one shape shares a shard's
        # slice. They are immutable, so the plan stores one per geometry.
        self.slices: dict[_GeometryKey, TensorSlice] = {}

    def _slice(self, geometry: _GeometryKey) -> TensorSlice:
        """The one slice for this geometry, built the first time it is asked for."""
        held = self.slices.get(geometry)
        if held is None:
            held = self.slices[geometry] = _geometry_slice(geometry)
        return held

    def _element_size(self, key: str) -> int:
        """Derive byte width from the validated wire dtype for ``key``."""
        return self.dtypes[key].itemsize

    def _validate(self) -> None:
        """Check everything the ranks must agree on before any route is built."""
        if not self.publishers:
            raise ValueError("routing requires at least one publisher rank")

        published_keys = {
            key for registrations in self.publishers.values() for key in registrations
        }
        requested_keys = {
            key for registrations in self.requesters.values() for key in registrations
        }
        unpublished = requested_keys - published_keys
        if unpublished:
            raise KeyError(f"requester keys have no publisher: {sorted(unpublished)}")

        key_shapes: dict[str, tuple[int, ...]] = {}
        key_dtypes: dict[str, torch.dtype] = {}
        registrations_by_rank = [
            *self.publishers.values(),
            *self.requesters.values(),
        ]
        for registrations in registrations_by_rank:
            for key, item in registrations.items():
                global_shape = item.tensor_slice.global_shape
                if key_shapes.setdefault(key, global_shape) != global_shape:
                    raise ValueError(f"inconsistent global shape for key {key!r}")
                if key_dtypes.setdefault(key, item.dtype) != item.dtype:
                    raise ValueError(f"inconsistent dtype for key {key!r}")

    def _publisher_transfers(
        self,
        key: str,
        target_slice: TensorSlice,
    ) -> tuple[Transfer, ...]:
        """One read per publisher overlapping the target slice."""
        element_size = self._element_size(key)
        transfers: list[Transfer] = []
        covered = 0
        for index, (source_slice, ranks) in enumerate(self.holders_by_key[key]):
            # Target first, so the segment keeps its normalized coordinates.
            piece = get_slice_intersection(target_slice, source_slice)
            if piece is None:
                continue
            piece = self.slices.setdefault(_geometry_key(piece), piece)
            numel = get_slice_numel(piece)
            covered += numel
            nbytes = numel * element_size
            rank = self.balance.choose(key, index, ranks, nbytes)
            transfers.append(
                Transfer(
                    source=rank,
                    source_volume_id=rank,
                    segment=piece,
                    nbytes=nbytes,
                )
            )

        # Cheap stand-in for a full disjointness proof:
        # - Reject gaps that would leave part of the destination unfilled.
        # - Reject ordinary partial overlaps so concurrent volume fetches do not
        #   write the same destination region.
        # An overlap and a gap of equal size could still cancel out in this count.
        # TODO: A complete check would verify the actual union and disjointness.
        wanted = get_slice_numel(target_slice)
        if covered != wanted:
            raise ValueError(
                f"publisher metadata does not cover {key!r} exactly: {covered} of "
                f"{wanted} elements at offsets={target_slice.offsets}, "
                f"shape={target_slice.local_shape}"
            )
        return tuple(transfers)

    def build(self) -> LocalRouteTable:
        """Build the requester's route table from real slice metadata."""
        self._validate()

        for registrations in self.publishers.values():
            for storage_key, item in registrations.items():
                self.dtypes[storage_key] = item.dtype

        self.holders_by_key = {
            storage_key: [
                (self._slice(geometry), [rank for rank, _slice in members])
                for geometry, members in groups
            ]
            for storage_key, groups in _group_by_geometry(self.publishers).items()
        }

        # For each key, plan the local slice to fill and where to pull it from.
        #
        #   publishers                 requesters
        #   P0 [rows 0-3] --+
        #                   +--pull--> R0
        #   P1 [rows 4-7] --+           |
        #                               +--relay--> R1 (waits, then pulls)
        #
        #   R0: DestinationRoute(dest=rows0-7,
        #                        transfers=[P0->rows0-3, P1->rows4-7],
        #                        notify_relay_id="weights#0", notify_peers=("R1",))
        #   R1: DestinationRoute(dest=rows0-7, transfers=[R0->rows0-7],
        #                        wait_for_relay_id="weights#0")
        routes: dict[str, DestinationRoute] = {}

        for storage_key, groups in _group_by_geometry(self.requesters).items():
            for relay_index, (geometry, members) in enumerate(groups):
                local_slice = next(
                    (
                        member_slice
                        for member, member_slice in members
                        if member == self.rank
                    ),
                    None,
                )
                if local_slice is None:
                    continue

                target_slice = self._slice(geometry)
                target_bytes = get_slice_numel(target_slice) * self._element_size(
                    storage_key
                )
                # The ingress rank serves every peer, and how many peers there
                # are does not depend on which member is chosen.
                ingress_rank, ingress_slice = self.balance.choose(
                    storage_key,
                    relay_index,
                    members,
                    target_bytes * (len(members) - 1),
                    name=lambda member: member[0],
                )
                peers = tuple(member for member in members if member[0] != ingress_rank)

                relay_id = f"{storage_key}#{relay_index}" if peers else None

                if ingress_rank == self.rank:
                    routes[storage_key] = DestinationRoute(
                        destination_slice=ingress_slice,
                        transfers=self._publisher_transfers(storage_key, target_slice),
                        notify_relay_id=relay_id,
                        notify_peers=tuple(peer for peer, _slice in peers),
                    )
                else:
                    routes[storage_key] = DestinationRoute(
                        destination_slice=local_slice,
                        transfers=(
                            Transfer(
                                source=ingress_rank,
                                source_volume_id=ingress_rank,
                                segment=target_slice,
                                nbytes=target_bytes,
                            ),
                        ),
                        wait_for_relay_id=relay_id,
                    )

        return LocalRouteTable(
            rank=self.rank,
            volume_id=self.rank,
            role=RankRole.REQUESTER,
            keys={
                storage_key: KeyPlan(item.tensor_slice, routes.get(storage_key))
                for storage_key, item in sorted(self.registrations.items())
            },
        )


class RoutingPlan:
    """Immutable local routes produced once from global slice metadata."""

    def __init__(
        self,
        routes: LocalRouteTable,
    ) -> None:
        self.routes = routes

    @classmethod
    def for_publisher(
        cls,
        rank: str,
        registrations: Mapping[str, KeyRegistration],
    ) -> RoutingPlan:
        """Build a publisher plan using only that rank's local registrations."""
        return cls(
            LocalRouteTable(
                rank=rank,
                volume_id=rank,
                role=RankRole.PUBLISHER,
                keys={
                    storage_key: KeyPlan(item.tensor_slice)
                    for storage_key, item in sorted(registrations.items())
                },
            )
        )

    @classmethod
    def for_requester(
        cls,
        rank: str,
        registrations: Mapping[str, KeyRegistration],
        publishers: Registrations,
        balance: Balance = Balance.ROTATE,
        *,
        requesters: Registrations | None = None,
    ) -> RoutingPlan:
        """Build one requester plan from its layout and every publisher's."""
        if balance.sequential:
            raise ValueError(
                f"{balance.value!r} balances against a running total, so ranks "
                "only agree when one planner builds them all"
            )
        return cls(
            _Builder(
                rank,
                registrations,
                publishers,
                requesters,
                balance,
            ).build()
        )
