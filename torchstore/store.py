from itertools import product
from logging import getLogger
from threading import local
from typing import Any, Dict, Optional, Tuple, Union

import torch
from monarch.actor import Actor, endpoint
from torch.distributed.tensor import DTensor

from torchstore.utils import assemble_global_tensor, get_local_tensor, spawn_actors
from torchstore.transport import Pipe, Message, TensorSlice

import sys
import logging
logger = getLogger(__name__)

logger.root.setLevel(logging.INFO)
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
logger.root.addHandler(stdout_handler)

FULL_TENSOR = "full_tensor"



# @dataclass
# class DTensorPack:
#     offsets: Tuple
#     coordinates: Tuple
#     local_tensor: torch.Tensor
#     global_shape: Tuple
#     mesh_shape: Tuple

#     def __post_init__(self):
#         self.coordinates = tuple(self.coordinates)

class MultiProcessStore: # This is actually the local client
    """This class represents the local store, which exists on every process. Remote storage
    is handled by the client.
    """

    def __init__(self):
        self._client = None

    @classmethod
    async def create_store(cls):
        store = cls()
        await store.spawn()
        return store

    async def spawn(self):
        self._client = await spawn_actors(1, _MultiProcessClient, "MultiProcessStore")

    @property
    def client(self):
        assert self._client is not None, "Client not initialized, please instantiate this class with 'create_store'"
        return self._client

    @torch.no_grad
    async def put(self, key: str, value: Union[torch.Tensor, Any]):
        logger.info(f"Putting {key}")

        pipe = Pipe(self.client)
        message = Message.from_any(value)

        await pipe.put_to_storage_volume(key, message)

    @torch.no_grad
    async def get(self, key: str, inplace_tensor: Optional[torch.Tensor] = None):
        # TODO: when we try this with rdma, I should be able to write rdma directly to the tensor
        # for now we'll copy into it after fetching from the remote store

        logger.info(f"Fetching {key}")

        pipe = Pipe(self.client)
        message = Message.from_any(inplace_tensor)

        fetched_tensor = await pipe.get_from_storage_volume(key, message)
        return fetched_tensor if inplace_tensor is None else inplace_tensor


class _MultiProcessClient(Actor): # this is the storage volume
    """The remote logic for storage. Recieves remote put/get requests and handles them via the storage abstraction"""

    def __init__(self):
        self.store = CopyStore()

    @endpoint
    async def put(self, key: str, transport_buffer: torch.Tensor, message: Message):
        await self.store.put(key, transport_buffer, message)

    @endpoint
    async def get(self, key: str, transport_buffer: torch.Tensor, message: Message):
        return await self.store.get(key, transport_buffer, message)

    @endpoint
    async def get_meta(self, key: str):
        return await self.store.get_meta(key)



class CopyStore: # this just represents in memory. The alternative would be something like SSD
    # TODO: make functions atomic
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

    async def put(self, key: str, transport_buffer: torch.Tensor, message: Message):
    # def put(self, key: str, value: torch.Tensor): #TODO: value -> transport_buffer
        """ """
        # TODO: handle messagte with objects != None

        if message.is_object:
            self.kv[key] = {"obj": message.objects}
            return transport_buffer

        # since we pass tensor=None to the transport buffer,
        # we allocate on the fly
        tensor = await transport_buffer.read_into() 
        if message.tensor_slice is not None:
            self._handle_dtensor(key, message.tensor_slice, tensor)
            return  
    
        self.kv[key] = tensor

    async def get(self, key: str, transport_buffer: torch.Tensor, message: Message):

        if key not in self.kv:
            raise KeyError(f"Key '{key}' not found. {list(self.kv.keys())=}")

        #TODO: clean up
        val = self.kv[key]
        if isinstance(val, dict) and "obj" in val:
            transport_buffer.is_object = True
            transport_buffer.objects = val["obj"]
            return transport_buffer

        if message.tensor_slice is None:
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
            message.tensor_slice.local_shape, #TODO: remove tensor_val from messages by setting coordinates_only=True in msg cstrct
            message.tensor_slice.offsets,
        )
        logger.debug("done local tensor")
        local_tensor = local_tensor.clone()
        await transport_buffer.write_from(local_tensor)

        return transport_buffer

    async def get_meta(self, key: str):
        if key not in self.kv:
            raise KeyError(f"Key '{key}' not found. {list(self.kv.keys())=}")

        val = self.kv[key]
        if isinstance(val, dict) and "obj" in val:
            return "obj"

        t = val if isinstance(val, dict) and "FULL_TENSOR" in val else val
        return t.shape, t.dtype
