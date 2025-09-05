from typing import Union, Any, Optional

from tests import utils
import torch
from monarch.actor import get_or_spawn_controller


import torchstore.state_dict_utils
from torchstore.strategy import TorchStoreStrategy, SingletonStrategy
from torchstore.storage_volume import StorageVolume
from torchstore.controller import Controller
from torchstore.client import LocalClient


# I need to keep this somewhere, so here we go
DEFAULT_TORCHSTORE_NAME = "TorchStoreController"

# cache for local clients
_local_clent_map = {}

async def initialize(
    num_storage_volumes=1,
    strategy: Optional[TorchStoreStrategy]=None,
    store_name=DEFAULT_TORCHSTORE_NAME 
):
    """Initializes the global store, and returns a local client.
    """

    if num_storage_volumes == 1 and strategy is None:
        strategy = SingletonStrategy()
    elif strategy is None:
        raise RuntimeError("Must specify controller strategy if num_storage_volumes > 1")

    #TODO: monarch doesn't support nested actors yet, so we need to spawn storage volumes here
    # ideally this is done in the controller.init
    storage_volumes = await StorageVolume.spawn(
        num_volumes=num_storage_volumes,
        id_func=strategy.get_volume_id
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

async def teardown_store(store_name=DEFAULT_TORCHSTORE_NAME):
    controller = await get_or_spawn_controller(store_name, Controller)    
    await controller.teardown.call()
    global _local_clent_map
    _local_clent_map = {}

async def client(store_name=DEFAULT_TORCHSTORE_NAME):
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


async def put(key: str, value: Union[torch.Tensor, Any], store_name=DEFAULT_TORCHSTORE_NAME):
    cl = await client(store_name)
    return await cl.put(key, value)

async def get(key: str, inplace_tensor: Optional[torch.Tensor] = None, store_name=DEFAULT_TORCHSTORE_NAME):
    cl = await client(store_name)
    return await cl.get(key, inplace_tensor)

async def put_state_dict(state_dict, key:str, store_name: str=DEFAULT_TORCHSTORE_NAME):
    cl = await client(store_name)
    await torchstore.state_dict_utils.put_state_dict(store=cl,state_dict=state_dict, key=key)

async def get_state_dict(
    key: str, user_state_dict: Optional[dict] = None, strict:bool =True, store_name:str =DEFAULT_TORCHSTORE_NAME
):
    cl = await client(store_name)
    return await torchstore.state_dict_utils.get_state_dict(cl, key, user_state_dict, strict)
