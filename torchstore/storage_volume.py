from itertools import product
from logging import getLogger
from typing import Any, Dict, Optional, Tuple, Union

import torch
from monarch.actor import Actor, endpoint
from torch.distributed.tensor import DTensor

from torchstore.transport.pipe import Request
from torchstore.utils import assemble_global_tensor, get_local_tensor, spawn_actors
from torchstore.transport import Pipe, Request, TensorSlice
from torchstore.controller import Controller

logger = getLogger(__name__)


FULL_TENSOR = "full_tensor"

class StorageVolume(Actor):
    """The remote logic for storage. Recieves remote put/get requests and handles them via the storage abstraction"""

    def __init__(self):
        self.store = InMemoryStore()

    @endpoint
    async def put(
        self,
        key: str,
        transport_buffer: torch.Tensor,
        request: Request
    ):
        await self.store.put(key, transport_buffer, request)

    @endpoint
    async def get(self, key: str, transport_buffer: torch.Tensor, request: Request):
        return await self.store.get(key, transport_buffer, request)

    @endpoint
    async def get_meta(self, key: str):
        return await self.store.get_meta(key)


class StorageImpl:
    async def put(
        self,
        key: str,
        transport_buffer: torch.Tensor,
        request: Request
    ):
        raise NotImplementedError()
    async def get(
        self,
        key: str,
        transport_buffer: torch.Tensor,
        request: Request
    ):
        raise NotImplementedError()
    async def get_meta(
        self,
        key: str
    ):
        raise NotImplementedError()

class InMemoryStore(StorageImpl): 
    """ Local in memory storage. 
    """
    def __init__(self):
        self.kv: Dict[str, Any] = {}

    def _build_full_tensor(self, key: str):
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

    def _handle_dtensor(self, key: str, tensor_slice: TensorSlice, tensor: torch.Tensor):
        if key not in self.kv:
            self.kv[key] = {}

        self.kv[key][tensor_slice.coordinates] = {
            "slice": tensor_slice,
            "tensor": tensor
        }

    async def put(
        self,
        key: str,
        transport_buffer: torch.Tensor,
        request: Request
    ):
        if request.is_object:
            self.kv[key] = {"obj": request.objects}
            return transport_buffer

        # since we pass tensor=None to the transport buffer,
        # we allocate on the fly
        tensor = await transport_buffer.read_into() 
        if request.tensor_slice is not None:
            self._handle_dtensor(key, request.tensor_slice, tensor)
            return  
    
        self.kv[key] = tensor

    async def get(
        self,
        key: str,
        transport_buffer: torch.Tensor,
        request: Request
    ):

        if key not in self.kv:
            raise KeyError(f"Key '{key}' not found. {list(self.kv.keys())=}")

        #TODO: clean up
        val = self.kv[key]
        if isinstance(val, dict) and "obj" in val:
            transport_buffer.is_object = True
            transport_buffer.objects = val["obj"]
            return transport_buffer

        if request.tensor_slice is None:
            await transport_buffer.write_from(self.kv[key])
            return transport_buffer

        # Maps to "reshard on load"
        # once all pieces are received.
        if FULL_TENSOR not in self.kv[key] and self._has_full_tensor(key):
            self._build_full_tensor(key)

        if not self._has_full_tensor(key):
            raise RuntimeError(
                f"Not ready to serve full tensor yet for {key}: {self.kv[key]=}"
            )

        logger.debug("Building local tensor")
        # TODO: should probably be a view
        local_tensor = get_local_tensor(
            self.kv[key][FULL_TENSOR],
            request.tensor_slice.local_shape,
            request.tensor_slice.offsets,
        )
        logger.debug("done local tensor")
        local_tensor = local_tensor.clone()
        await transport_buffer.write_from(local_tensor)

        return transport_buffer

    async def get_meta(
        self,
        key: str
    ):
        if key not in self.kv:
            raise KeyError(f"Key '{key}' not found. {list(self.kv.keys())=}")

        val = self.kv[key]
        if isinstance(val, dict) and "obj" in val:
            return "obj"

        t = val if isinstance(val, dict) and "FULL_TENSOR" in val else val
        return t.shape, t.dtype
