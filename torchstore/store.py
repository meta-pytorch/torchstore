from dataclasses import dataclass
from itertools import product
from logging import getLogger
from typing import Any, Dict, Optional, Tuple, Union

import torch
from monarch.actor import Actor, endpoint
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._utils import _compute_local_shape_and_global_offset

from torchstore.utils import assemble_global_tensor, get_local_tensor, spawn_actors


logger = getLogger(__name__)

FULL_TENSOR = "full_tensor"


@dataclass
class DTensorPack:
    offsets: Tuple
    coordinates: Tuple
    local_tensor: torch.Tensor
    global_shape: Tuple
    mesh_shape: Tuple

    def __post_init__(self):
        self.coordinates = tuple(self.coordinates)

class MultiProcessStore:
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
        logger.warn(f"Putting {key}")
        if isinstance(value, DTensor):
            coordinates = value.device_mesh.get_coordinate()
            _, offsets = _compute_local_shape_and_global_offset(
                value.shape,
                mesh_shape=value.device_mesh.shape,
                my_coordinate=coordinates,
                placements=value.placements,
            )

            # it's helpful representing the critical pieces of DTensor here
            # instead of serializing DTensor itself (which is possible)
            value = DTensorPack(
                offsets,
                coordinates,
                value._local_tensor,
                value.shape,
                value.device_mesh.shape,
            )

        await self.client.put.call(key, value)  # TODO: remove asyncio

    @torch.no_grad
    async def get(self, key: str, inplace_tensor: Optional[torch.Tensor] = None):
        # TODO: when we try this with rdma, I should be able to write rdma directly to the tensor
        # for now we'll copy into it after fetching from the remote store

        logger.warn(f"Fetching {key}")

        if isinstance(inplace_tensor, DTensor):
            coordinates = inplace_tensor.device_mesh.get_coordinate()
            _, offsets = _compute_local_shape_and_global_offset(
                inplace_tensor.shape,
                mesh_shape=inplace_tensor.device_mesh.shape,
                my_coordinate=coordinates,
                placements=inplace_tensor.placements,
            )
            # TODO: don't pass inplace_tensor in DTensorPack, we only use tensor.shape and will slow down comms
            dtensor_pack = DTensorPack(
                offsets, coordinates, inplace_tensor._local_tensor, None, None
            )
            fetched_tensor = await self.client.get.call_one(key, dtensor_pack)
            inplace_tensor._local_tensor.copy_(
                fetched_tensor
            )  # TODO: this is probably not allowed

        elif isinstance(inplace_tensor, torch.Tensor):
            fetched_tensor = await self.client.get.call_one(key)
            inplace_tensor.copy_(fetched_tensor)

            return inplace_tensor

        # call_one returns the value directly instead of the ValueMesh
        return await self.client.get.call_one(key)


class _MultiProcessClient(Actor):
    """The remote logic for storage. Recieves remote put/get requests and handles them via the storage abstraction"""

    def __init__(self):
        self.store = CopyStore()

    @endpoint
    async def put(self, key: str, value: torch.Tensor):
        self.store.put(key, value)

    @endpoint
    def get(self, key: str, dtensor_pack: Optional[DTensorPack] = None):
        return self.store.get(key, dtensor_pack)


class CopyStore:
    # TODO: make functions atomic
    def __init__(self):
        self.kv: Dict[str, Any] = {}

    def _build_full_tensor(self, key: str):
        logger.warn(f"Building full tensor for {key}")
        # we can also consider in the future not requiring the full tensor to be
        # assembled, and instead only that the requested offsets are available
        # this is a performance optimization, but could be tricky to implement.
        assert self._has_full_tensor(key)

        # Early return if full tensor is already built
        if FULL_TENSOR in self.kv[key]:
            return

        # TODO: DTensorPack is fullfilling too many purposes.
        # we should consider turning this into a "PendingTensor" class,
        # and having these functions defined there instead.
        # should also totally simplify the logic here
        local_tensors = []
        global_offsets = []
        global_shape = None
        device_mesh_shape = None
        for coordinate, tensor_shard in self.kv[key].items():
            if coordinate == FULL_TENSOR:
                continue

            local_tensors.append(tensor_shard.local_tensor)
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
        logger.warn(f"Finished full tensor for {key}")

    def _has_full_tensor(self, key: str) -> bool:
        if key not in self.kv:
            return False

        if FULL_TENSOR in self.kv[key]:
            return True

        # TODO: there's probably a smarter way to do this,
        # but for now we check that every "coordinate" in device mesh
        # has checked in a tensor shard, which _should_ imply all
        # pieces are received.
        mesh_shape = next(iter(self.kv[key].values())).mesh_shape
        # iterate through all possible coordinates
        for coord in product(*(range(s) for s in mesh_shape)):
            if coord not in self.kv[key]:
                return False

        return True

    def _handle_dtensor(self, key: str, value: DTensorPack):
        if key not in self.kv:
            self.kv[key] = {}

        self.kv[key][value.coordinates] = value

    def put(self, key: str, value: torch.Tensor):
        """ """
        if isinstance(value, DTensorPack):
            self._handle_dtensor(key, value)
            return
        elif isinstance(value, torch.Tensor):
            if key not in self.kv:
                self.kv[key] = torch.empty_like(value)
            # TODO: I am probably recieved a copy to beging with (unless RDMA is enabled), how can I tell?
            self.kv[key].copy_(value)
        else:
            self.kv[key] = value  # best of luck

    def get(self, key: str, dtensor_pack: Optional[DTensorPack] = None):
        if key not in self.kv:
            raise KeyError(f"Key '{key}' not found. {list(self.kv.keys())=}")

        if dtensor_pack is None:
            return self.kv[key]

        # experimenting with creating a full tensor representation on 'get'
        # once all pieces are received.
        # TODO: think more carefully about this
        if FULL_TENSOR not in self.kv[key] and self._has_full_tensor(key):
            self._build_full_tensor(key)

        if not self._has_full_tensor(key):
            raise RuntimeError(
                f"Not ready to serve full tensor yet for {key}: {self.kv[key]=}"
            )

        logger.warn("Building local tensor")
        # TODO: should probably be a view
        local_tensor = get_local_tensor(
            self.kv[key][FULL_TENSOR],
            dtensor_pack.local_tensor.shape,
            dtensor_pack.offsets,
        )
        logger.warn("done local tensor")

        return local_tensor
