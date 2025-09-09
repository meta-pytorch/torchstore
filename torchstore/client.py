# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from logging import getLogger
from typing import Any, Optional, Union

import torch

from torchstore.controller import ObjectType

from torchstore.transport import Pipe, Request
from torchstore.utils import assemble_global_tensor, get_local_tensor

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
        volume_map = await self._controller.locate_volumes.call_one(key)

        partial_results = []
        for volume_id, storage_info in volume_map.items():
            storage_volume = self.strategy.get_storage_volume(volume_id)
            pipe = Pipe(storage_volume)

            if object_type in (ObjectType.OBJECT, ObjectType.TENSOR):
                # TODO: in the future, we could intelligently select the best storage volume
                # but for now any should work.
                fetched_tensor = await pipe.get_from_storage_volume(key, request)
                return fetched_tensor if inplace_tensor is None else inplace_tensor

            # else: this is the dtensor / tensor slice case
            # fetch from all storage volumes, something like this
            # TODO: fix so we can request all tensor slices from a storage volume
            # at once, this is silly
            for tensor_slice in storage_info.tensor_slices:
                tensor_slice_request = Request.from_tensor_slice(tensor_slice)

                local_tensor = await pipe.get_from_storage_volume(
                    key, tensor_slice_request
                )
                partial_results.append((local_tensor, tensor_slice))

        assert partial_results, "No partial results found"
        assert request.tensor_slice is not None

        # build the entire tensor.
        # TODO: again, we should have better control over
        # rebuilding only the portion I need, but this is a good start

        local_tensors = []
        global_offsets = []
        global_shape = None
        device_mesh_shape = None
        for local_tensor, tensor_slice in partial_results:
            local_tensors.append(local_tensor)

            global_offsets.append(tensor_slice.offsets)
            if global_shape is None:
                global_shape = tensor_slice.global_shape
            else:
                assert global_shape == tensor_slice.global_shape

            if device_mesh_shape is None:
                device_mesh_shape = tensor_slice.mesh_shape
            else:
                assert device_mesh_shape == tensor_slice.mesh_shape

        full_tensor = assemble_global_tensor(
            local_tensors,
            global_shape,
            global_offsets,
        )

        fetched_tensor = get_local_tensor(
            full_tensor,
            request.tensor_slice.local_shape,
            request.tensor_slice.offsets,
        )
        # Pipe does not have support for inplace copies of fetched tensors yet,
        # so we just copy
        if inplace_tensor is not None:
            assert request.tensor_val is not None
            request.tensor_val.copy_(fetched_tensor)
            return inplace_tensor
        return fetched_tensor

    async def keys(self, prefix: str | None = None):
        """
        Get all keys that match the given prefix.

        This method retrieves all keys from the storage that start with the specified prefix.
        The prefix matching follows reverse domain name notation convention.

        Args:
            prefix (str): The prefix to match against stored keys.
                          For example, "xyz" matches "xyz.abc.def" but "xy" does not.

        Returns:
            List[str]: A list of keys that match the given prefix.
        """
        # Keys are synced across all storage volumes, so we just call one.
        return await self._controller.keys.call_one(prefix)
