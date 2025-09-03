from typing import Union, Any, Optional

import torch
from monarch.actor import get_or_spawn_controller


from torchstore.controller import Controller
from torchstore.client import LocalClient


# I need to keep this somewhere, so here we go
DEFAULT_TORCHSTORE_NAME = "TorchStoreController"

async def initialize_store(
    num_storage_volumes=1,
    strategy=None,
    store_name=DEFAULT_TORCHSTORE_NAME # must be unique per monarch context
):
    """Initializes the global store, and returns a local client.
    """

    if num_storage_volumes == 1 and strategy is None:
        #TODO: Singleton case
        raise NotImplementedError()
    elif strategy is None:
        raise RuntimeError("Must specify controller strategy if num_storage_volumes > 1")

    controller = await get_or_spawn_controller(
        store_name,
        Controller,
        strategy=strategy,
        num_storage_volumes=num_storage_volumes,
    )
    await controller.init.call()

_local_clent_map = {}
async def client(store_name=DEFAULT_TORCHSTORE_NAME):
    if store_name in _local_clent_map:
        return _local_clent_map[store_name]
        
    controller = await get_or_spawn_controller(store_name, Controller)
    controller_strategy = await controller.get_controller_strategy.call_one()       
    
    local_client = LocalClient(
        controller=controller,
        controller_strategy=controller_strategy,
    )
    _local_clent_map[store_name] = local_client

    return local_client


async def put(key: str, value: Union[torch.Tensor, Any], store_name=DEFAULT_TORCHSTORE_NAME):
    cl = await client(store_name)
    return await cl.put(key, value)

async def get(key: str, inplace_tensor: Optional[torch.Tensor] = None, store_name=DEFAULT_TORCHSTORE_NAME):
    cl = await client(store_name)
    return await cl.get(key, inplace_tensor)
