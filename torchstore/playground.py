import random
import asyncio

import torch

from monarch.tensor_engine import RDMABuffer
from monarch.actor import Actor, endpoint, this_host


class RDMATest(Actor):

    def __init__(self) -> None:
        self.other_actor = None

    @endpoint
    def set_other_actor(self, other_actor):
        self.other_actor = other_actor

    @endpoint
    async def send(self) -> None:
        
        shape = random.randint(1, 1000)
        tensor = torch.rand(shape)
        size_elem = tensor.numel() * tensor.element_size()
        print(f"allocating with {size_elem=}")

        byte_view = tensor.view(torch.uint8).flatten()
        buffer = RDMABuffer(byte_view)

        assert buffer.size() == size_elem, f"{buffer.size()} != {size_elem}"

        await self.other_actor.recv.call(buffer, tensor.shape, tensor.dtype)
        # await buffer.release_buffer()

    @endpoint
    async def recv(self, rdma_buffer, shape, dtype):
        tensor = torch.empty(shape, dtype=dtype)
        byte_view = tensor.view(torch.uint8).flatten()
        await rdma_buffer.read_into(byte_view)

async def main():

    mesh_0 = this_host().spawn_procs(per_host={"gpus": 1})
    actor_0 = mesh_0.spawn("rdma_test", RDMATest)
    
    mesh_1 = this_host().spawn_procs(per_host={"gpus": 1})
    actor_1 = mesh_1.spawn("rdma_test", RDMATest)

    while True:
        await actor_0.set_other_actor.call(actor_1)
        await actor_0.send.call()

if __name__ == "__main__":
    asyncio.run(main())
