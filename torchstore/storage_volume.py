# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from itertools import product
from logging import getLogger
from typing import Any, Dict, Optional, Tuple, Union

import torch
from monarch.actor import Actor, endpoint

from torchstore.transport.buffers import TransportBuffer

from torchstore.transport.pipe import Request, TensorSlice
from torchstore.utils import assemble_global_tensor, spawn_actors

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
        cls, num_volumes: int, *init_args: Any, **init_kwargs: Any
    ) -> "StorageVolume":
        return await spawn_actors(
            num_volumes, cls, cls.actor_name, *init_args, **init_kwargs
        )

    @endpoint
    async def get_id(self) -> str:
        return self.volume_id

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
    async def get_meta(self, key: str) -> Union[Tuple[torch.Size, torch.dtype], str]:
        return await self.store.get_meta(key)


class StorageImpl:
    """Abstract base class for storage implementations."""

    async def put(
        self, key: str, transport_buffer: TransportBuffer, request: Request
    ) -> Optional[TransportBuffer]:
        """Store data in the storage backend."""
        raise NotImplementedError()

    async def get(
        self, key: str, transport_buffer: TransportBuffer, request: Request
    ) -> TransportBuffer:
        """Retrieve data from the storage backend."""
        raise NotImplementedError()

    async def get_meta(self, key: str) -> Union[Tuple[torch.Size, torch.dtype], str]:
        """Get metadata about stored data."""
        raise NotImplementedError()


class InMemoryStore(StorageImpl):
    """Local in memory storage."""

    def __init__(self) -> None:
        self.kv: Dict[str, Any] = {}

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
        full_tensor = assemble_global_tensor(
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

    async def put(
        self, key: str, transport_buffer: TransportBuffer, request: Request
    ) -> None:
        if request.is_object:
            self.kv[key] = {"obj": request.objects}
            return

        # since we pass tensor=None to the transport buffer,
        # we allocate on the fly
        tensor = await transport_buffer.read_into(tensor=None)
        if request.tensor_slice is not None:
            self._handle_dtensor(key, request.tensor_slice, tensor)
            return

        self.kv[key] = tensor

    async def get(
        self, key: str, transport_buffer: TransportBuffer, request: Request
    ) -> TransportBuffer:

        if key not in self.kv:
            raise KeyError(f"Key '{key}' not found. {list(self.kv.keys())=}")

        # TODO: clean up
        val = self.kv[key]
        if isinstance(val, dict) and "obj" in val:
            transport_buffer.is_object = True
            transport_buffer.objects = val["obj"]
            return transport_buffer

        if request.tensor_slice is None:
            await transport_buffer.write_from(self.kv[key])
            return transport_buffer

        for shard in self.kv[key].values():
            stored_slice = shard["slice"]
            stored_tensor = shard["tensor"]

            # Check if requested slice is contained within or intersects with stored slice
            extracted_tensor = self._extract_tensor_subset(
                stored_tensor, stored_slice, request.tensor_slice
            )

            if extracted_tensor is not None:
                await transport_buffer.write_from(extracted_tensor)
                return transport_buffer

        raise RuntimeError(
            f"Tensor slice {request.tensor_slice} not found in any stored shards for {key}"
        )

    def _extract_tensor_subset(
        self,
        stored_tensor: torch.Tensor,
        stored_slice: TensorSlice,
        requested_slice: TensorSlice,
    ) -> torch.Tensor | None:
        """
        Extract the intersection between stored slice and requested slice from the stored tensor.

        Args:
            stored_tensor: The tensor data that's stored locally
            stored_slice: The slice metadata for the stored tensor (describes what region it covers)
            requested_slice: The slice metadata for what the client wants

        Returns:
            The extracted tensor subset representing the intersection, or None if no overlap
        """
        # Ensure both slices have the same global shape
        if stored_slice.global_shape != requested_slice.global_shape:
            raise ValueError(
                f"Global shapes don't match: {stored_slice.global_shape=} (Stored) {requested_slice.global_shape=} (Requested)"
            )

        # Compute intersection bounds and extraction indices
        extract_indices = []
        intersection_shape = []

        for dim in range(len(stored_slice.global_shape)):
            # Stored slice boundaries in global coordinates
            stored_start = stored_slice.offsets[dim]
            stored_end = stored_start + stored_slice.local_shape[dim]

            # Requested slice boundaries in global coordinates
            requested_start = requested_slice.offsets[dim]
            requested_end = requested_start + requested_slice.local_shape[dim]

            # Compute intersection boundaries in global coordinates
            intersection_start = max(stored_start, requested_start)
            intersection_end = min(stored_end, requested_end)

            # Check if there's actually an intersection in this dimension
            if intersection_start >= intersection_end:
                return None  # No overlap

            # Convert intersection to local indices within the stored tensor
            local_start = intersection_start - stored_start
            local_end = intersection_end - stored_start

            extract_indices.append(slice(local_start, local_end))
            intersection_shape.append(local_end - local_start)

        # Extract the intersection portion from the stored tensor
        extracted_tensor = stored_tensor[tuple(extract_indices)]

        # Verify the extracted tensor has the expected intersection shape
        expected_shape = tuple(intersection_shape)
        if extracted_tensor.shape != expected_shape:
            raise RuntimeError(
                f"Extracted tensor shape {extracted_tensor.shape} doesn't match expected intersection shape {expected_shape}"
            )

        return extracted_tensor

    async def get_meta(self, key: str) -> Union[Tuple[torch.Size, torch.dtype], str]:
        if key not in self.kv:
            raise KeyError(f"Key '{key}' not found. {list(self.kv.keys())=}")

        val = self.kv[key]
        if isinstance(val, torch.Tensor):
            return val.shape, val.dtype

        assert isinstance(val, dict)
        if "obj" in val:
            return "obj"

        if "tensor" in val:
            return val["tensor"].shape, val["tensor"].dtype

        raise RuntimeError(f"Unknown type for {key} type={type(val)}")
