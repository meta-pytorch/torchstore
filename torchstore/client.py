from functools import partial
from itertools import product
from logging import getLogger
from typing import Any, Dict, Optional, Tuple, Union

import torch

from torchstore.transport.pipe import Request
from torchstore.utils import assemble_global_tensor, get_local_tensor, spawn_actors
from torchstore.transport import Pipe, Request, TensorSlice
from torchstore.controller import Controller, ObjectType

logger = getLogger(__name__)

class LocalClient: 
    """This class represents the local store, which exists on every process. Remote storage
    is handled by the client.
    """

    def __init__(
        self,
        controller,
        strategy,
    ):
        self._controller = controller
        self.strategy = strategy
        

    @torch.no_grad
    async def put(self, key: str, value: Union[torch.Tensor, Any]):
        logger.debug(f"Putting {key}")

        request = Request.from_any(value)
        # for now, we only write to one storage volume.
        # we probably don't need a remote call for this case since
        # it will never be dynamic. e.g. it's always based on the 
        # TorchstoreStrategy defined during intiailization
        storage_volume, volume_id = self.strategy.select_storage_volume()

        pipe = Pipe(storage_volume)

        await pipe.put_to_storage_volume(key, request)
        await self._controller.notify_put.call(key, request, volume_id)

    @torch.no_grad
    async def get(self, key: str, inplace_tensor: Optional[torch.Tensor] = None):
        logger.debug(f"Fetching {key}")
        request = Request.from_any(inplace_tensor)
        object_type = ObjectType.from_request(request)

        # multinode support here
        volume_map = await self._controller.locate_volumes.call_one(key, request)

        partial_results = []
        for volume_id, storage_info in volume_map.items():
            storage_volume = self.strategy.get_storage_volume(volume_id)
            pipe = Pipe(storage_volume)

            if object_type in (ObjectType.OBJECT, ObjectType.TENSOR_SLICE):
                fetched_tensor = await pipe.get_from_storage_volume(key, request)
                return fetched_tensor if inplace_tensor is None else inplace_tensor

            # else: dtensor
            # fetch from all storage volumes, something like this 
            # TODO: fix so we can request all tensor slices from a storage volume
            # at once, this is silly
            for tensor_slice in storage_info.tensor_slices:
                tensor_slice_request = Request.from_tensor_slice(tensor_slice)
                partial_results.append(
                    await pipe.get_from_storage_volume(key, tensor_slice_request)
                )

        return self._merge(partial_results, inplace_tensor=inplace_tensor)

    def _merge(self, partial_results, inplace_tensor):
        pass #TODO:
