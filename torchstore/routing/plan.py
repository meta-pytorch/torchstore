# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Build, inspect, and distribute a tensor routing plan."""

from __future__ import annotations

import zlib
from collections import defaultdict
from collections.abc import Callable, Collection, Mapping, Sequence
from enum import Enum
from typing import DefaultDict, Dict, List, Tuple, TypeVar

from torchstore.transport.types import TensorSlice
from torchstore.utils import (
    get_slice_intersection,
    get_slice_numel,
)

from ._model import (
    DestinationRoute,
    KeyPlan,
    KeyRegistration,
    LocalRouteTable,
    RankRole,
    Registrations,
    RouteEntry,
    Transfer,
)

__all__ = ["Balance", "KeyRegistration", "RoutingPlan"]

_Candidate = TypeVar("_Candidate")
_GeometryKey = Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]
# One replication set: its geometry and the ranks reporting it.
_Group = Tuple[_GeometryKey, List[Tuple[str, TensorSlice]]]


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
        self._assigned: DefaultDict[str, int] = defaultdict(int)

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


def _group_by_geometry(registrations: Registrations) -> Dict[str, List[_Group]]:
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
    grouped: DefaultDict[
        str, DefaultDict[_GeometryKey, List[Tuple[str, TensorSlice]]]
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
    """Turns every rank's reported layout into per-rank route tables."""

    def __init__(
        self,
        publishers: Registrations,
        requesters: Registrations,
        balance: Balance,
    ) -> None:
        self.publishers = publishers
        published_keys = {
            storage_key
            for registrations in publishers.values()
            for storage_key in registrations
        }
        self.requesters = {
            rank: {
                storage_key: registration
                for storage_key, registration in registrations.items()
                if storage_key in published_keys
            }
            for rank, registrations in requesters.items()
        }
        self.by_rank = {**publishers, **self.requesters}
        self.balance = _BALANCERS[balance]()
        # storage_key -> element_size
        self.element_sizes: Dict[str, int] = {}
        # storage_key -> [(published slice, ranks holding it)], in geometry order
        self.holders_by_key: Dict[str, List[Tuple[TensorSlice, List[str]]]] = {}
        # Slices repeat across keys and ranks far more often than they differ:
        # geometry carries no key, so every key of one shape shares a shard's
        # slice. They are immutable, so the plan stores one per geometry.
        self.slices: Dict[_GeometryKey, TensorSlice] = {}

    def _slice(self, geometry: _GeometryKey) -> TensorSlice:
        """The one slice for this geometry, built the first time it is asked for."""
        held = self.slices.get(geometry)
        if held is None:
            held = self.slices[geometry] = _geometry_slice(geometry)
        return held

    def _validate(self, targets: Collection[str]) -> None:
        """Check everything the ranks must agree on before any route is built."""
        if not self.publishers:
            raise ValueError("routing requires at least one publisher rank")
        if not self.requesters:
            raise ValueError("routing requires at least one requester rank")

        unregistered = set(targets) - set(self.by_rank)
        if unregistered:
            raise KeyError(f"ranks did not register a layout: {unregistered}")

        key_shapes: Dict[str, Tuple[int, ...]] = {}
        key_sizes: Dict[str, int] = {}
        for registrations in self.by_rank.values():
            for key, item in registrations.items():
                element_size = int(item.element_size)
                if element_size <= 0:
                    raise ValueError(f"element size for {key!r} must be positive")
                global_shape = item.tensor_slice.global_shape
                if key_shapes.setdefault(key, global_shape) != global_shape:
                    raise ValueError(f"inconsistent global shape for key {key!r}")
                if key_sizes.setdefault(key, element_size) != element_size:
                    raise ValueError(f"inconsistent element size for key {key!r}")

    def _publisher_transfers(
        self,
        key: str,
        target: TensorSlice,
    ) -> Tuple[Transfer, ...]:
        """One read per publisher overlapping target."""
        element_size = self.element_sizes[key]
        transfers: List[Transfer] = []
        covered = 0
        for index, (source, ranks) in enumerate(self.holders_by_key[key]):
            # target first, so the segment keeps target's normalized coordinates.
            piece = get_slice_intersection(target, source)
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

        # Cheap stand-in for a full disjointness proof: a short read means the
        # publishers leave a gap, a long one means their slices overlap partially.
        wanted = get_slice_numel(target)
        if covered != wanted:
            raise ValueError(
                f"publisher metadata does not cover {key!r} exactly: {covered} of "
                f"{wanted} elements at offsets={target.offsets}, "
                f"shape={target.local_shape}"
            )
        return tuple(transfers)

    def build(self, targets: Collection[str]) -> Dict[str, LocalRouteTable]:
        """Build route tables for ``targets`` from real TorchStore slice metadata."""
        self._validate(targets)

        for registrations in self.publishers.values():
            for storage_key, item in registrations.items():
                self.element_sizes[storage_key] = item.element_size

        self.holders_by_key = {
            storage_key: [
                (self._slice(geometry), [rank for rank, _slice in members])
                for geometry, members in groups
            ]
            for storage_key, groups in _group_by_geometry(self.publishers).items()
        }

        # Per-rank plan: for each key, the local slices to fill and where to pull
        # them from
        # rank -> storage_key -> [destination_route]
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
        routes: DefaultDict[str, DefaultDict[str, List[DestinationRoute]]] = (
            defaultdict(lambda: defaultdict(list))
        )

        wanted = set(targets)
        for storage_key, groups in _group_by_geometry(self.requesters).items():
            for relay_index, (geometry, members) in enumerate(groups):
                if len(wanted) < len(self.by_rank) and wanted.isdisjoint(
                    member for member, _slice in members
                ):
                    continue  # no target wants this geometry

                target = self._slice(geometry)
                target_bytes = (
                    get_slice_numel(target) * self.element_sizes[storage_key]
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
                peers = tuple(
                    member for member in members if member[0] != ingress_rank
                )

                relay_id = f"{storage_key}#{relay_index}" if peers else None

                if ingress_rank in wanted:
                    routes[ingress_rank][storage_key].append(
                        DestinationRoute(
                            destination_slice=ingress_slice,
                            transfers=self._publisher_transfers(storage_key, target),
                            notify_relay_id=relay_id,
                            notify_peers=tuple(peer for peer, _slice in peers),
                        )
                    )
                for peer, peer_slice in peers:
                    if peer not in wanted:
                        continue
                    routes[peer][storage_key].append(
                        DestinationRoute(
                            destination_slice=peer_slice,
                            transfers=(
                                Transfer(
                                    source=ingress_rank,
                                    source_volume_id=ingress_rank,
                                    segment=target,
                                    nbytes=target_bytes,
                                ),
                            ),
                            wait_for_relay_id=relay_id,
                        )
                    )

        tables: Dict[str, LocalRouteTable] = {}
        for rank in sorted(wanted):
            publishes = rank in self.publishers
            tables[rank] = LocalRouteTable(
                rank=rank,
                volume_id=rank,
                role=RankRole.PUBLISHER if publishes else RankRole.REQUESTER,
                keys={
                    storage_key: KeyPlan(
                        item.tensor_slice, tuple(routes[rank][storage_key])
                    )
                    for storage_key, item in sorted(self.by_rank[rank].items())
                },
            )
        return tables


class RoutingPlan:
    """Immutable per-rank routes produced once from global slice metadata."""

    def __init__(
        self,
        routes: Mapping[str, LocalRouteTable],
    ) -> None:
        self._routes = dict(routes)

    @classmethod
    def build(
        cls,
        publishers: Registrations,
        requesters: Registrations,
        balance: Balance = Balance.LEAST_LOADED,
    ) -> "RoutingPlan":
        """Reconcile every rank's reported layout into every rank's plan."""
        by_rank = {**publishers, **requesters}
        return cls(_Builder(publishers, requesters, balance).build(by_rank))

    @classmethod
    def build_for(
        cls,
        rank: str,
        publishers: Registrations,
        requesters: Registrations,
        balance: Balance = Balance.ROTATE,
    ) -> "RoutingPlan":
        """Reconcile every rank's reported layout into ``rank``'s own plan.

        Raises:
            ValueError: for a balance that has to see every rank to be correct.
        """
        if balance.sequential:
            raise ValueError(
                f"{balance.value!r} balances against a running total, so ranks "
                "only agree when one planner builds them all"
            )
        return cls(_Builder(publishers, requesters, balance).build({rank}))

    @property
    def ranks(self) -> tuple[str, ...]:
        """Ranks with a local table in this plan."""
        return tuple(sorted(self._routes))

    def for_rank(self, rank: str) -> "RoutingPlan":
        """Return a distributable plan containing only ``rank``'s local table."""
        return RoutingPlan({rank: self._routes[rank]})

    def lookup(self, rank: str, key: str) -> RouteEntry:
        """Return the immutable local actions for ``rank`` and ``key``."""
        return self._routes[rank].lookup(key)

    def _local(self, rank: str) -> LocalRouteTable:
        return self._routes[rank]
