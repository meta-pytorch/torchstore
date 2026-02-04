# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import socket
from itertools import product
from logging import getLogger
from typing import Any

import torch
from monarch.actor import Actor, endpoint

from torchstore.transport.buffers import TransportBuffer, TransportContext
from torchstore.transport.types import Request, TensorSlice
from torchstore.utils import assemble_tensor, get_slice_intersection, spawn_actors

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
    async def handshake(self, transport_buffer: TransportBuffer) -> Any | None:
        return await self.store.handshake(transport_buffer)

    @endpoint
    async def put(
        self, key: str, transport_buffer: TransportBuffer, request: Request
    ) -> None:
        await self.store.put(key, transport_buffer, request)

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
        self, key: str, transport_buffer: TransportBuffer, request: Request
    ) -> TransportBuffer | None:
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

    async def handshake(self, transport_buffer: TransportBuffer) -> Any | None:
        raise NotImplementedError()


class InMemoryStore(StorageImpl):
    """Local in memory storage."""

    def __init__(self) -> None:
        self.kv: dict[str, Any] = {}
        super().__init__()

    async def handshake(self, transport_buffer: TransportBuffer) -> Any | None:
        return await transport_buffer.recv_handshake(self.transport_context)

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

    def _build_full_tensor(self, key: str) -> None:
        logger.debug(f"Building full tensor for {key}")
        # we can also consider in the future not requiring the full tensor to be
        # assembled, and instead only that the requested offsets are available
        # this is a performance optimization, but could be tricky to implement.
        assert self._has_full_tensor(key)

        # Early return if full tensor is already built
        if FULL_TENSOR in self.kv[key]:
            return

        # TODO: Utility fucntions may make more sense in a
        # a "PendingTensor" class and have these functions
        # defined there instead. should also totally simplify the logic here
        local_tensors = []
        global_offsets = []
        global_shape = None
        device_mesh_shape = None
        for shard in self.kv[key].values():

            local_tensors.append(shard["tensor"])
            tensor_shard = shard["slice"]

            global_offsets.append(tensor_shard.offsets)
            if global_shape is None:
                global_shape = tensor_shard.global_shape
            else:
                assert global_shape == tensor_shard.global_shape

            if device_mesh_shape is None:
                device_mesh_shape = tensor_shard.mesh_shape
            else:
                assert device_mesh_shape == tensor_shard.mesh_shape

        assert local_tensors and global_offsets and global_shape

        # TODO: doing it this way has peek 2x tensor size in memory :(
        full_tensor = assemble_tensor(
            local_tensors,
            global_shape,
            global_offsets,
        )

        self.kv[key] = {FULL_TENSOR: full_tensor}
        logger.debug(f"Finished full tensor for {key}")

    def _has_full_tensor(self, key: str) -> bool:
        if key not in self.kv:
            return False

        if FULL_TENSOR in self.kv[key]:
            return True

        # TODO: there's probably a smarter way to do this,
        # but for now we check that every "coordinate" in device mesh
        # has checked in a tensor shard, which _should_ imply all
        # pieces are received.
        mesh_shape = next(iter(self.kv[key].values()))["slice"].mesh_shape
        # iterate through all possible coordinates
        for coord in product(*(range(s) for s in mesh_shape)):
            if coord not in self.kv[key]:
                return False

        return True

    def _handle_dtensor(
        self, key: str, tensor_slice: TensorSlice, tensor: torch.Tensor
    ) -> None:
        if key not in self.kv:
            self.kv[key] = {}

        self.kv[key][tensor_slice.coordinates] = {
            "slice": tensor_slice,
            "tensor": tensor,
        }

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

    async def put(
        self, key: str, transport_buffer: TransportBuffer, request: Request
    ) -> None:
        # Extract existing tensor for potential in-place update
        current_obj = self._extract_existing(key, request)

        # fetch from remote
        data = await transport_buffer.handle_put_request(
            self.transport_context,
            request,
            current_obj,
        )

        # store locally
        if request.is_object:
            self.kv[key] = {"obj": data}
            return

        if request.tensor_slice is not None:
            # tensor is actually part of a DTensor
            self._handle_dtensor(key, request.tensor_slice, data)
            return

        self.kv[key] = data

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

        if request.tensor_slice is None:
            return self.kv[key]

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
        # Clean up shared memory segments if they exist
        shm_cache = self.transport_context.get_shm_cache()
        shm_cache.delete(key)

    def reset(self) -> None:
        self.kv = {}
        # Clean up all shared memory segments
        shm_cache = self.transport_context.get_shm_cache()
        shm_cache.reset()
