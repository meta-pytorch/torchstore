# pyre-unsafe
import asyncio
import time

from typing import Dict, Tuple

import torch
from torchstore import MultiProcessStore
from torchstore._state_dict_utils import (
    get_state_dict,
    push_state_dict,
    zo_push_state_dict,
)


# Run the example : python example/torchstore_benchmark.py


def _gen_workload(
    num_tensors: int,
) -> Tuple[Dict[str, torch.Tensor], int]:
    weight_sd = {}
    total_model_size_bytes = 0
    for idx in range(num_tensors):
        weight = torch.randn(8, 8 << 20, dtype=torch.float32)  # 512 bytes
        weight_sd[f"param_{idx}"] = weight
        total_model_size_bytes += weight.numel() * weight.element_size()

    return weight_sd, total_model_size_bytes // (1 << 20)


async def run_benchmark():
    for num_tensors in [1, 2, 4, 8, 64]:
        store = await MultiProcessStore.create_store()
        sd, model_size = _gen_workload(num_tensors)
        start_time = time.monotonic()
        # Baseline push state-dict routine.
        # await push_state_dict(store, state_dict=sd, key="v0")
        # ZO push state-dict routine.
        fut = zo_push_state_dict(store, state_dict=sd, key="v0")
        trainer_ret_time = time.monotonic()

        fut.result()
        data_movement_end_time = time.monotonic()
        
        print(
            f"total put size {model_size} MB, trainer block time: {trainer_ret_time- start_time:.2f},  toal time spent {data_movement_end_time - start_time:.2f}s"
        )


if __name__ == "__main__":
    asyncio.run(run_benchmark())
