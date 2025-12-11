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

from torchstore.transport.buffers import TransportBuffer, TransportContext
from torchstore.transport.pipe import Request, TensorSlice
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
    async def get_id(self) -> str:
        return self.volume_id

    @endpoint
    async def handshake(self, transport_buffer: TransportBuffer) -> Optional[Any]:
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
        request: Optional[Request] = None,
    ) -> Union[Tuple[torch.Size, torch.dtype], str]:
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
    ) -> Optional[TransportBuffer]:
        """Store data in the storage backend."""
        raise NotImplementedError()

    async def get(
        self, key: str, transport_buffer: TransportBuffer, request: Request
    ) -> TransportBuffer:
        """Retrieve data from the storage backend."""
        raise NotImplementedError()

    async def get_meta(
        self, key: str, request: Optional[Request] = None
    ) -> Union[Tuple[torch.Size, torch.dtype], str]:
        """Get metadata about stored data."""
        raise NotImplementedError()

    async def delete(self, key: str) -> None:
        """Delete data from the storage backend."""
        raise NotImplementedError()

    async def handshake(self, transport_buffer: TransportBuffer) -> Optional[Any]:
        raise NotImplementedError()


class InMemoryStore(StorageImpl):
    """Local in memory storage."""

    def __init__(self) -> None:
        self.kv: Dict[str, Any] = {}
        super().__init__()

    async def handshake(self, transport_buffer: TransportBuffer) -> Optional[Any]:
        return await transport_buffer.recv_handshake(self.transport_context)

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

    def _get_sharded_tensor(self, request: Request, key: str) -> Optional[torch.Tensor]:
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

        if request.is_object:
            self.kv[key] = {"obj": request.objects}
            return

        # since we pass tensor=None to the transport buffer,
        # we allocate on the fly
        tensor = await transport_buffer.read_into(None, self.transport_context)
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
            await transport_buffer.write_from(self.kv[key], self.transport_context)
            return transport_buffer

        extracted_tensor = self._get_sharded_tensor(request, key)

        if extracted_tensor is not None:
            await transport_buffer.write_from(extracted_tensor, self.transport_context)
            return transport_buffer

        raise RuntimeError(
            f"Tensor slice {request.tensor_slice} not found in any stored shards for {key}"
        )

    async def get_meta(
        self,
        key: str,
        request: Optional[Request] = None,
    ) -> Union[Tuple[torch.Size, torch.dtype], str]:
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
