# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional, Union

import torch

import torchstore.state_dict_utils
from monarch.actor import get_or_spawn_controller
from torchstore.client import LocalClient
from torchstore.controller import Controller
from torchstore.storage_volume import StorageVolume
from torchstore.strategy import SingletonStrategy, TorchStoreStrategy


# I need to keep this somewhere, so here we go
DEFAULT_TORCHSTORE_NAME: str = "TorchStoreController"

# cache for local clients
_local_clent_map: Dict[str, LocalClient] = {}


async def initialize(
    num_storage_volumes: int = 1,
    strategy: Optional[TorchStoreStrategy] = None,
    store_name: str = DEFAULT_TORCHSTORE_NAME,
) -> None:
    """Initialize the TorchStore distributed storage system.

    Sets up storage volumes and controller. Must be called before any put/get operations.

    Args:
        num_storage_volumes (int): Number of storage volumes to create. Defaults to 1.
        strategy (TorchStoreStrategy, optional): Strategy for distributing tensors across volumes.
            Uses SingletonStrategy if None and num_storage_volumes=1.
        store_name (str): Unique name for this store instance. Defaults to DEFAULT_TORCHSTORE_NAME.

    Raises:
        RuntimeError: If num_storage_volumes > 1 but no strategy is provided.

    Example:
        >>> import torchstore as ts
        >>> await ts.initialize(num_storage_volumes=4, strategy=LocalRankStrategy()) # uses default namespace.
        >>> >>> await ts.initialize("my_custom_store")
    """
    if num_storage_volumes == 1 and strategy is None:
        strategy = SingletonStrategy()
    elif strategy is None:
        raise RuntimeError(
            "Must specify controller strategy if num_storage_volumes > 1"
        )

    # TODO: monarch doesn't support nested actors yet, so we need to spawn storage volumes here
    # ideally this is done in the controller.init
    storage_volumes = await StorageVolume.spawn(
        num_volumes=num_storage_volumes, id_func=strategy.get_volume_id
    )

    controller = await get_or_spawn_controller(
        store_name,
        Controller,
    )
    await controller.init.call(
        strategy=strategy,
        num_storage_volumes=num_storage_volumes,
        storage_volumes=storage_volumes,
    )


async def shutdown(store_name: str = DEFAULT_TORCHSTORE_NAME) -> None:
    """Shutdown and cleanup a TorchStore instance.

    Gracefully shuts down all storage volumes and controllers associated with the store.

    Args:
        store_name (str): Name of the store to shutdown. Defaults to DEFAULT_TORCHSTORE_NAME.

    Example:
        >>> import torchstore as ts
        >>> await ts.shutdown()  # Shutdown default store
        >>> await ts.shutdown("my_custom_store")
    """
    controller = await get_or_spawn_controller(store_name, Controller)
    await controller.teardown.call()
    global _local_clent_map
    _local_clent_map = {}


async def client(store_name: str = DEFAULT_TORCHSTORE_NAME) -> LocalClient:
    """Get a local client handle for interacting with the store.

    Returns a cached LocalClient instance that provides the interface for put/get operations.

    Args:
        store_name (str): Name of the store to get a client for. Defaults to DEFAULT_TORCHSTORE_NAME.

    Returns:
        LocalClient: A client instance for performing storage operations.

    Example:
        >>> store_client = await client()
        >>> await store_client.put("my_key", tensor)
    """
    if store_name in _local_clent_map:
        return _local_clent_map[store_name]

    controller = await get_or_spawn_controller(store_name, Controller)
    controller_strategy = await controller.get_controller_strategy.call_one()

    local_client = LocalClient(
        controller=controller,
        strategy=controller_strategy,
    )
    _local_clent_map[store_name] = local_client

    return local_client


async def put(
    key: str, value: Union[torch.Tensor, Any], store_name: str = DEFAULT_TORCHSTORE_NAME
) -> None:
    """Store a tensor or object in the distributed store.

    Args:
        key (str): Unique identifier for the stored value.
        value (torch.Tensor or Any): Tensor or object to store.
        store_name (str): Name of the store to use. Defaults to DEFAULT_TORCHSTORE_NAME.

    Example:
        >>> tensor = torch.randn(100, 100)
        >>> await put("my_tensor", tensor)
        >>> await put("my_object", {"data": [1, 2, 3]})
    """
    cl = await client(store_name)
    return await cl.put(key, value)


async def get(
    key: str,
    inplace_tensor: Optional[torch.Tensor] = None,
    store_name: str = DEFAULT_TORCHSTORE_NAME,
) -> Union[torch.Tensor, Any]:
    """Retrieve a tensor or object from the distributed store.

    Args:
        key (str): Unique identifier of the value to retrieve.
        inplace_tensor (torch.Tensor, optional): Pre-allocated tensor for in-place retrieval.
        store_name (str): Name of the store to use. Defaults to DEFAULT_TORCHSTORE_NAME.

    Returns:
        The stored tensor or object.

    Example:
        >>> tensor = await get("my_tensor")
        >>> obj = await get("my_object")
    """
    cl = await client(store_name)
    return await cl.get(key, inplace_tensor)


async def exists(key: str, store_name: str = DEFAULT_TORCHSTORE_NAME) -> bool:
    """Check if a key exists in the distributed store.

    Args:
        key (str): Unique identifier to check for existence.
        store_name (str): Name of the store to use. Defaults to DEFAULT_TORCHSTORE_NAME.

    Returns:
        bool: True if the key exists, False otherwise.

    Example:
        >>> if await exists("my_tensor"):
        ...     tensor = await get("my_tensor")
        >>>
        >>> # Check before storing
        >>> if not await exists("checkpoint_1"):
        ...     await put("checkpoint_1", model.state_dict())
    """
    cl = await client(store_name)
    return await cl.exists(key)


async def put_state_dict(
    state_dict: Dict[str, Any], key: str, store_name: str = DEFAULT_TORCHSTORE_NAME
) -> None:
    """Store a PyTorch model state_dict in the distributed store.

    Args:
        state_dict (dict): Model state_dict to store.
        key (str): Unique identifier for the state_dict.
        store_name (str): Name of the store to use. Defaults to DEFAULT_TORCHSTORE_NAME.

    Example:
        >>> model = torch.nn.Linear(10, 5)
        >>> await put_state_dict(model.state_dict(), "model_checkpoint")
    """
    cl = await client(store_name)
    await torchstore.state_dict_utils.put_state_dict(
        store=cl, state_dict=state_dict, key=key
    )


async def get_state_dict(
    key: str,
    user_state_dict: Optional[Dict[str, Any]] = None,
    strict: bool = True,
    store_name: str = DEFAULT_TORCHSTORE_NAME,
) -> Dict[str, Any]:
    """Retrieve a PyTorch model state_dict from the distributed store.

    Args:
        key (str): Unique identifier of the state_dict to retrieve.
        user_state_dict (dict, optional): Pre-existing state_dict to merge with.
        strict (bool): Whether to enforce strict loading. Defaults to True.
        store_name (str): Name of the store to use. Defaults to DEFAULT_TORCHSTORE_NAME.

    Returns:
        dict: The retrieved state_dict.

    Example:
        >>> state_dict = await get_state_dict("model_checkpoint")
        >>> model.load_state_dict(state_dict)
    """
    cl = await client(store_name)
    return await torchstore.state_dict_utils.get_state_dict(
        cl, key, user_state_dict, strict
    )
