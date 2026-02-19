# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import socket
from logging import getLogger
from typing import Any

import torch
from monarch.actor import Actor, endpoint

from torchstore.transport.buffers import TransportBuffer, TransportContext
from torchstore.transport.types import KeyedRequest, Request, TensorSlice
from torchstore.utils import get_slice_intersection, spawn_actors

logger = getLogger(__name__)


FULL_TENSOR = "full_tensor"


class StorageVolume(Actor):
    """The remote logic for storage. Recieves remote put/get requests and handles them via the storage abstraction"""

    actor_name: str = "StorageVolumes"

    def __init__(
        self,
        id_func,
    ) -> None:
        self.store: StorageImpl = InMemoryStore()
        self.volume_id: str = id_func()

    @classmethod
    async def spawn(
        cls,
        num_volumes: int,
        mesh,
        *init_args: Any,
        **init_kwargs: Any,
    ) -> "StorageVolume":
        actors = await spawn_actors(
            num_volumes, cls, cls.actor_name, mesh, *init_args, **init_kwargs
        )

        return actors

    @endpoint
    async def get_id(self) -> tuple[str, str]:
        hostname = os.environ.get("HOSTNAME", socket.gethostname())
        return (self.volume_id, hostname)

    @endpoint
    async def handshake(
        self,
        transport_buffer: TransportBuffer,
        entries: list[KeyedRequest],
    ) -> list[Any]:
        return await self.store.handshake(transport_buffer, entries)

    @endpoint
    async def put(
        self,
        transport_buffer: TransportBuffer,
        entries: list[KeyedRequest],
    ) -> None:
        await self.store.put(transport_buffer, entries)

    @endpoint
    async def get(
        self, key: str, transport_buffer: TransportBuffer, request: Request
    ) -> TransportBuffer:
        return await self.store.get(key, transport_buffer, request)

    @endpoint
    async def get_meta(
        self,
        key: str,
        request: Request | None = None,
    ) -> tuple[torch.Size, torch.dtype] | str:
        return await self.store.get_meta(key, request)

    @endpoint
    async def delete(self, key: str) -> None:
        await self.store.delete(key)

    @endpoint
    async def reset(self) -> None:
        self.store.reset()


class StorageImpl:
    """Abstract base class for storage implementations."""

    def __init__(self) -> None:
        self.transport_context = TransportContext()

    async def put(
        self,
        transport_buffer: TransportBuffer,
        entries: list[KeyedRequest],
    ) -> None:
        """Store data in the storage backend."""
        raise NotImplementedError()

    async def get(
        self, key: str, transport_buffer: TransportBuffer, request: Request
    ) -> TransportBuffer:
        """Retrieve data from the storage backend."""
        raise NotImplementedError()

    async def get_meta(
        self, key: str, request: Request | None = None
    ) -> tuple[torch.Size, torch.dtype] | str:
        """Get metadata about stored data."""
        raise NotImplementedError()

    async def delete(self, key: str) -> None:
        """Delete data from the storage backend."""
        raise NotImplementedError()

    async def handshake(
        self,
        transport_buffer: TransportBuffer,
        entries: list[KeyedRequest],
    ) -> list[Any]:
        raise NotImplementedError()


class InMemoryStore(StorageImpl):
    """Local in memory storage."""

    def __init__(self) -> None:
        self.kv: dict[str, Any] = {}
        super().__init__()

    async def handshake(
        self,
        transport_buffer: TransportBuffer,
        entries: list[KeyedRequest],
    ) -> list[Any]:
        pairs = [
            (entry, self._extract_existing(entry.key, entry.request))
            for entry in entries
        ]
        return await transport_buffer.recv_handshake(self.transport_context, pairs)

    async def put(
        self,
        transport_buffer: TransportBuffer,
        entries: list[KeyedRequest],
    ) -> None:
        entries_with_current_obj = [
            (entry, self._extract_existing(entry.key, entry.request))
            for entry in entries
        ]
        results = await transport_buffer.handle_put_request(
            self.transport_context, entries_with_current_obj
        )
        for entry in entries:
            self._store(entry.key, entry.request, results[entry.key])

    def _store(self, key: str, request: "Request", data: Any) -> None:
        """Store data in kv, wrapping objects and handling DTensor shards."""
        if request.is_object:
            self.kv[key] = {"obj": data}
        elif request.tensor_slice is not None:
            self._handle_dtensor(key, request.tensor_slice, data)
        else:
            self.kv[key] = data

    def _extract_existing(self, key: str, request: "Request") -> torch.Tensor | None:
        """Extract existing tensor from storage for in-place update.

        Looks up the key in kv storage and extracts the tensor if it exists.
        Only asserts on type mismatches between existing data and incoming request.

        Args:
            key: The storage key to look up
            request: The incoming put request

        Returns:
            The existing tensor if found, None otherwise.

        Raises:
            AssertionError: If there's a type mismatch between existing data and request.
        """
        current_object = self.kv.get(key, None)

        if current_object is None:
            return None

        if isinstance(current_object, torch.Tensor):
            # Regular tensor - request must also be a regular tensor (no tensor_slice)
            assert (
                request.tensor_slice is None
            ), "Existing data is a regular tensor but incoming request has tensor_slice (DTensor)"
            return current_object

        if isinstance(current_object, dict):
            if "obj" in current_object:
                # Object dict - request must also be an object
                assert (
                    request.is_object
                ), "Existing data is an object but request.is_object is False"
                return None

            # DTensor shard dict - incoming request must also be a DTensor
            assert (
                request.tensor_slice is not None
            ), "Existing data is DTensor shards but incoming request has no tensor_slice"
            # Look up by coordinates
            shard = current_object.get(request.tensor_slice.coordinates)
            if shard is not None and "tensor" in shard:
                return shard["tensor"]
            # Coordinates don't match - new shard, return None to allocate new
            return None

        raise AssertionError(f"Unexpected current_object type: {type(current_object)}")

    def _handle_dtensor(
        self, key: str, tensor_slice: TensorSlice, tensor: torch.Tensor
    ) -> None:
        if key not in self.kv:
            self.kv[key] = {}

        self.kv[key][tensor_slice.coordinates] = {
            "slice": tensor_slice,
            "tensor": tensor,
        }

    def _extract_slice_from_tensor(
        self, tensor: torch.Tensor, tensor_slice: TensorSlice
    ) -> torch.Tensor:
        """Extract a slice from a full tensor.

        Args:
            tensor: The full stored tensor.
            tensor_slice: The slice specification to extract.

        Returns:
            The extracted tensor slice.
        """
        indices = []
        for dim in range(len(tensor_slice.global_shape)):
            start = tensor_slice.offsets[dim]
            end = start + tensor_slice.local_shape[dim]
            indices.append(slice(start, end))
        return tensor[tuple(indices)]

    def _get_sharded_tensor(self, request: Request, key: str) -> torch.Tensor | None:
        """
        Searches stored shards and returns one which completely contains the requested tensor slice

        Args:
            request: Request object containing the tensor_slice specification
            key: Storage key identifying the tensor shards to search.

        Returns:
            The extracted tensor slice if found completely within a stored shard,
            None otherwise.
        """
        for shard in self.kv[key].values():
            stored_slice = shard["slice"]
            stored_tensor = shard["tensor"]

            intersection_slice = get_slice_intersection(
                stored_slice, request.tensor_slice
            )

            # We don't want to visit the shard where requested tensor slice is not completely contained
            # in the stored tensor slice.
            if (
                intersection_slice is None
                or intersection_slice.local_shape != request.tensor_slice.local_shape
                or intersection_slice.offsets != request.tensor_slice.offsets
            ):
                continue

            # Extract the intersection from the stored tensor
            indices = []
            for dim in range(len(stored_slice.global_shape)):
                start = intersection_slice.offsets[dim] - stored_slice.offsets[dim]
                indices.append(
                    slice(
                        start,
                        start + intersection_slice.local_shape[dim],
                    )
                )
            extracted_tensor = stored_tensor[tuple(indices)]

            if extracted_tensor is not None:
                return extracted_tensor

    async def get(
        self, key: str, transport_buffer: TransportBuffer, request: Request
    ) -> TransportBuffer:
        if key not in self.kv:
            raise KeyError(f"Key '{key}' not found. {list(self.kv.keys())=}")

        data = self._get_data(request, key)
        await transport_buffer.handle_get_request(self.transport_context, data)

        return transport_buffer

    def _get_data(self, request, key):
        val = self.kv[key]
        if isinstance(val, dict) and "obj" in val:
            return val["obj"]

        # Full tensor stored - return it (possibly sliced)
        if isinstance(val, torch.Tensor):
            if request.tensor_slice is None:
                return val
            # User wants a slice of the full tensor - extract it
            return self._extract_slice_from_tensor(val, request.tensor_slice)

        # Must be sharded tensor dict - delegate to _get_sharded_tensor
        if request.tensor_slice is None:
            # TODO: currently, it seems we only support requested a subsection of a shard.
            # this needs to be made more general, such that we can request any region of the stored tensor
            # (with the most useful case really being even to fetch all shards at once, so we can recreate them locally)
            raise RuntimeError(
                f"Key '{key}' contains sharded tensor but no tensor_slice was requested"
            )

        extracted_tensor = self._get_sharded_tensor(request, key)

        if extracted_tensor is not None:
            return extracted_tensor

        raise RuntimeError(
            f"Tensor slice {request.tensor_slice} not found in any stored shards for {key}"
        )

    async def get_meta(
        self,
        key: str,
        request: Request | None = None,
    ) -> tuple[torch.Size, torch.dtype] | str:
        if key not in self.kv:
            raise KeyError(f"Key '{key}' not found. {list(self.kv.keys())=}")

        stored_object = self.kv[key]
        if isinstance(stored_object, torch.Tensor):
            return stored_object.shape, stored_object.dtype

        assert isinstance(stored_object, dict)
        if "obj" in stored_object:
            return "obj"

        if "tensor" in stored_object:
            return stored_object["tensor"].shape, stored_object["tensor"].dtype

        if request is not None and request.tensor_slice is not None:
            extracted_tensor = self._get_sharded_tensor(request, key)
            if extracted_tensor is not None:
                return extracted_tensor.shape, extracted_tensor.dtype

            raise KeyError(
                f"Could not find shard slice with {request.tensor_slice=}  Slices:{stored_object}"
            )

        raise RuntimeError(
            f"Unknown type for {key} type={type(stored_object)} {stored_object=}"
        )

    async def delete(self, key: str) -> None:
        if key not in self.kv:
            raise KeyError(f"Key '{key}' not found. {list(self.kv.keys())=}")
        del self.kv[key]

    def reset(self) -> None:
        self.kv = {}
