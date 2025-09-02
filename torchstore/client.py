from itertools import product
from logging import getLogger
from typing import Any, Dict, Optional, Tuple, Union

import torch

from torchstore.transport.pipe import Request
from torchstore.utils import assemble_global_tensor, get_local_tensor, spawn_actors
from torchstore.transport import Pipe, Request, TensorSlice
from torchstore.controller import Controller

logger = getLogger(__name__)

class LocalClient: 
    """This class represents the local store, which exists on every process. Remote storage
    is handled by the client.
    """

    def __init__(self, storage_volumes, controller):
        self._storage_volumes = storage_volumes
        self._controller = controller
        self._id = None #TODO: think this through

    @torch.no_grad
    async def put(self, key: str, value: Union[torch.Tensor, Any]):
        logger.debug(f"Putting {key}")

        request = Request.from_any(value)
        # for now, we only write to one storage volume.
        # this will probably always be the case
        # we probably don't need a remote call for this case since
        # it will never be dynamic. e.g. it's always based on the 
        # PlacementStrategy defined during intiailization
        storage_volume = self._controller.select_storage_volume_for_put.call_one(
            request,
            key,
            self._id
        )

        pipe = Pipe(storage_volume)

        await pipe.put_to_storage_volume(key, request)
        await self._controller.notify.call(key, request, storage_volume)

    @torch.no_grad
    async def get(self, key: str, inplace_tensor: Optional[torch.Tensor] = None):
        logger.debug(f"Fetching {key}")
        request = Request.from_any(inplace_tensor)

        # multinode support here
        storage_volumes_to_tensor_silces = self._controller.select_storage_volume.call_one(
            request,
            self
        )
        partial_results = []
        for storage_volume_id, tensor_slices in storage_volumes_to_tensor_silces.items():
            storage_volume = self._get_storage_volume(storage_volume_id)
            
            # TODO:
            # - :( we should be able to get all of them at once
            # - we should be able to efficiently only fetch the regions we need 
            for tensor_slice in tensor_slices:
                request = Request.from_tensor_slice(
                    tensor_slice
                )
                pipe = Pipe(storage_volume)
                partial_results.append(
                   await pipe.get_from_storage_volume(key, request)
                )

        return self._merge(partial_results, inplace_tensor=inplace_tensor)


    def _merge(self, partial_results, inplace_tensor):
        fetched_tensor = partial_results[0]
        return fetched_tensor if inplace_tensor is None else inplace_tensor

    def _get_storage_volume(self, storage_volume_id):
        return self._storage_volumes[storage_volume_id]
