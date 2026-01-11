# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# torchrun --nproc_per_node=2 example/send_tensor_gloo.py

import os
import time

import torch
import torch.distributed as dist


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    print(
        f"About to init process group: rank={rank} world_size={world_size}", flush=True
    )

    dist.init_process_group(backend="gloo")
    # Create a 1GB tensor (float32: 4 bytes per element)
    num_elements = 1024 * 1024 * 1024 // 4  # 1GB / 4 bytes
    tensor = torch.ones(num_elements, dtype=torch.float32)
    print("Created tensor")

    if dist.get_rank() == 0:
        # Send tensor to rank 1
        start_time = time.time()
        dist.send(tensor=tensor, dst=1)
        end_time = time.time()
        elapsed = end_time - start_time
        print(
            f"Rank 0 sent 1GB tensor to rank 1 in {elapsed:.4f} seconds ({1.0/elapsed:.2f} GB/s)"
        )
    elif dist.get_rank() == 1:
        # Prepare empty tensor to receive
        recv_tensor = torch.empty(num_elements, dtype=torch.float32)
        start_time = time.time()
        dist.recv(tensor=recv_tensor, src=0)
        end_time = time.time()
        elapsed = end_time - start_time
        print(
            f"Rank 1 received 1GB tensor from rank 0 in {elapsed:.4f} seconds ({1.0/elapsed:.2f} GB/s)"
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
