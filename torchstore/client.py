# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from logging import getLogger
from typing import Any, Optional, Union

import torch

from torchstore.controller import ObjectType

from torchstore.transport import Pipe, Request, TensorSlice
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
    async def get(
        self,
        key: str,
        inplace_tensor: Optional[torch.Tensor] = None,
        tensor_slice_spec: Optional[TensorSlice] = None,
    ):
        logger.debug(f"Fetching {key}")

        if tensor_slice_spec is not None and inplace_tensor is not None:
            if tensor_slice_spec.local_shape != inplace_tensor.shape:
                raise ValueError(
                    f"Requested tensor slice shape {tensor_slice_spec.local_shape} does not match in-place tensor shape {inplace_tensor.shape}"
                )

        # When slicing, don't use inplace_tensor for the request because the transport
        # layer will try to load full tensor into slice-sized buffer (size mismatch)
        request_inplace = None if tensor_slice_spec is not None else inplace_tensor
        request = Request.from_any(request_inplace)
        object_type = ObjectType.from_request(request)

        # multinode support here
        volume_map = await self._controller.locate_volumes.call_one(key)

        if object_type in (ObjectType.OBJECT, ObjectType.TENSOR):
            # TODO: in the future, we could intelligently select the best storage volume
            # but for now any should work.
            volume_id, storage_info = volume_map.popitem()
            storage_volume = self.strategy.get_storage_volume(volume_id)
            pipe = Pipe(storage_volume)
            fetched_tensor = await pipe.get_from_storage_volume(key, request)
            # If user requested a specific slice, extract it
            if tensor_slice_spec is not None:
                if not isinstance(fetched_tensor, torch.Tensor):
                    raise ValueError(
                        "Cannot extract tensor slice from non-tensor object"
                    )
                sliced_tensor = get_local_tensor(
                    fetched_tensor,
                    tensor_slice_spec.local_shape,
                    tensor_slice_spec.offsets,
                )

                # Handle in-place operation for tensor slice
                if inplace_tensor is not None:
                    return inplace_tensor.copy_(sliced_tensor)
                return sliced_tensor

            return fetched_tensor if inplace_tensor is None else inplace_tensor

        # Handle the dtensor (slice) case
        partial_results = []
        for volume_id, storage_info in volume_map.items():
            storage_volume = self.strategy.get_storage_volume(volume_id)
            pipe = Pipe(storage_volume)

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

        # If user requested a specific slice, extract that instead of the DTensor's local portion
        if tensor_slice_spec is not None:
            fetched_tensor = get_local_tensor(
                full_tensor,
                tensor_slice_spec.local_shape,
                tensor_slice_spec.offsets,
            )
        else:
            # Normal DTensor case - extract the local portion for this process
            fetched_tensor = get_local_tensor(
                full_tensor,
                request.tensor_slice.local_shape,
                request.tensor_slice.offsets,
            )

        # Pipe does not have support for inplace copies of fetched tensors yet,
        # so we just copy
        if inplace_tensor is not None:
            inplace_tensor.copy_(fetched_tensor)
            return inplace_tensor
        return fetched_tensor

    async def exists(self, key: str) -> bool:
        """Check if a key exists in the distributed store.

        This is an efficient operation that only checks metadata at the controller level
        without retrieving the actual data.

        Args:
            key (str): The key to check for existence.

        Returns:
            bool: True if the key exists, False otherwise.
        """
        logger.debug(f"Checking existence of {key}")
        try:
            # Use the controller to check if key exists
            # This is efficient as it only checks metadata
            await self._controller.locate_volumes.call_one(key)
            return True
        except Exception as e:
            # Controller raises KeyError if key doesn't exist, but it comes wrapped
            # in an ActorError from the Monarch framework
            if "KeyError" in str(e) or "Unable to locate" in str(e):
                return False
            # Re-raise if it's a different kind of error
            raise e
