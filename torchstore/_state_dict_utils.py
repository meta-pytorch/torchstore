import asyncio
from concurrent.futures import Future
from logging import getLogger
from time import sleep
from typing import Optional

import torch
from torch.distributed.checkpoint._nested_dict import (
    flatten_state_dict,
    unflatten_state_dict,
)

DELIM = "/"
MAPPING = "MAPPING"

logger = getLogger(__name__)

from concurrent.futures import ThreadPoolExecutor


def zo_push_state_dict(store, state_dict, key) -> Future[None]:
    """
    Run the put request in a separate thread. Calller should wait for the future to complete,
    before updating the state dict.
    """
    flattened_state_dict, mapping = flatten_state_dict(state_dict)

    def work_fn(store, sd, mapping, key):
        event_loop = asyncio.new_event_loop()

        async def put_wrapper():
            puts = []
            for flattend_key, value in sd.items():
                puts.append(store.put(f"{key}{DELIM}{flattend_key}", value))
            asyncio.gather(*puts)
            await store.put(f"{key}{DELIM}{MAPPING}", mapping)

        try:
            asyncio.set_event_loop(event_loop)
            event_loop.run_until_complete(put_wrapper())
        finally:
            logger.warning(f"[PUT_WRAPPER]Closing asyncio event loop")
            event_loop.close()

    with ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="PUSH_STATE_DICT_IO"
    ) as executor:
        return executor.submit(work_fn, store, flattened_state_dict, mapping, key)


async def push_state_dict(store, state_dict, key):
    """
    We have an option here. Either we can "flatten state dict", by turning state dict names into a single key,
    or I can actually just maintain the dictionary representation of the state dict, and we can allow some recursive behavior in the store.

    Overall, this might not even be something we want to solve for in the TorchStore, but I'm adding this utility so we can test sharding models.

    """
    flattened_state_dict, mapping = flatten_state_dict(state_dict)
    puts = []
    for flattened_key, value in flattened_state_dict.items():
        puts.append(store.put(f"{key}{DELIM}{flattened_key}", value))
    asyncio.gather(*puts)

    await store.put(f"{key}{DELIM}{MAPPING}", mapping)


async def get_state_dict(
    store, key, user_state_dict: Optional[dict] = None, strict=True
):
    """Unflatten the state dict from the store"""

    try:
        # Since the mapping is the last thing we write out, it also gaurantees the state dict is not pending
        fetched_mapping = await store.get(f"{key}{DELIM}{MAPPING}")
    except Exception as e:
        raise RuntimeError(
            f"Mapping is missing from the store. This most likely means there is no matching 'push' call for this key: {key=}"
        ) from e

    user_flattened_state_dict, user_mapping = (
        flatten_state_dict(user_state_dict)
        if user_state_dict is not None
        else ({}, None)
    )
    if strict and user_mapping is not None:
        pass
        # assert user_mapping == fetched_mapping

    fetched_state_dict = {}
    for flattened_key in fetched_mapping.keys():
        inplace_tensor = user_flattened_state_dict.get(flattened_key, None)
        logger.info(f"Fetching {flattened_key} with {inplace_tensor=}")
        fetched_state_dict[flattened_key] = await store.get(
            f"{key}{DELIM}{flattened_key}",
            inplace_tensor if isinstance(inplace_tensor, torch.Tensor) else None,
        )

    # # Prepare all the coroutines first
    # coros = []
    # keys = []
    # for flattened_key in fetched_mapping.keys():
    #     inplace_tensor = user_flattened_state_dict.get(flattened_key, None)
    #     keys.append(flattened_key)
    #     coros.append(
    #         store.get(
    #             f"{key}{DELIM}{flattened_key}",
    #             inplace_tensor if isinstance(inplace_tensor, torch.Tensor) else None,
    #         )
    #     )
    # # Run all requests concurrently
    # results = await asyncio.gather(*coros)
    # # Build the result dictionary
    # fetched_state_dict = dict(zip(keys, results))

    return unflatten_state_dict(fetched_state_dict, fetched_mapping)
