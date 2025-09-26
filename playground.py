import asyncio
import os

import torch
from torch.distributed import FileStore
from monarch.actor import Actor, endpoint, this_host
import uuid


class TorchComms(Actor):
    
    @endpoint
    async def handshake(self, file_store_name):
        self.pg = torch.distributed.init_process_group(
            backend="gloo",
            init_method=f"file://{file_store_name}",
            rank=1,
            world_size=2,
        )

    @endpoint
    async def recv(self, shape, dtype):
        
        t = torch.zeros(shape, dtype=dtype, device=torch.device("cpu"))
        torch.distributed.recv(t, src=0, group=self.pg)
        print("recv", t)
        return t

    @endpoint
    async def finalize(self):
        self.torchcomm.finalize()


async def main():
    from torchcomms import new_comm
    proc_mesh = this_host().spawn_procs(per_host={"gpus": 1})
    actor = proc_mesh.spawn("torch_comms", TorchComms)

    os.environ["TORCHCOMM_RANK"] = "0"
    os.environ["TORCHCOMM_SIZE"] = "2"
    file_store_name = f"/tmp/lpasqualin/comms_test{str(uuid.uuid4())[:8]}"
    handshake_future = actor.handshake.call(file_store_name)
    pg = torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"file://{file_store_name}",
        rank=0,
        world_size=2,
    )
    
    print(f"client handshake")
    device=torch.device("cpu")
    
    await handshake_future
    print("finished handshake")
    
    import time
    while True:
        
        t = torch.ones(10, 10, device=device)
        print(f"exchanging tensor {t}")
        recv_fut = actor.recv.call(shape=t.shape, dtype=t.dtype)
        torch.distributed.send(t, dst=1, group=pg)
        # torchcomm.finalize()
        fetched = await recv_fut

        # f = actor.finalize.call() 
        # torchcomm.finalize()
        
        # await f
        
        print(f"{fetched=}")
    
    


if __name__ == "__main__":
    asyncio.run(main())
