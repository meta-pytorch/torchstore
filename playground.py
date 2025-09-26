import asyncio
import os

import torch
from torch.distributed import FileStore
from monarch.actor import Actor, endpoint, this_host
import uuid


class TorchComms(Actor):
    
    @endpoint
    async def handshake(self, file_store_name):
        print("starting handshake")
        from torchcomms import new_comm
        os.environ["TORCHCOMM_RANK"] = "1"
        os.environ["TORCHCOMM_SIZE"] = "2"
        device=torch.device("cuda")
        self.torchcomm = new_comm("ncclx", device, name="main_comm", store=FileStore(file_store_name, 2))

    @endpoint
    async def recv(self, shape, dtype):
        
        t = torch.zeros(shape, dtype=dtype, device=torch.device("cuda"))
        assert self.torchcomm.get_rank()
        self.torchcomm.recv(t, 0, async_op=False)
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
    file_store_name = f"comms_test{str(uuid.uuid4())[:8]}"
    handshake_future = actor.handshake.call(file_store_name)

    
    print(f"client handshake")
    device=torch.device("cuda")
    torchcomm = new_comm("ncclx", device, name="main_comm", store=FileStore(file_store_name, 2))
    
    await handshake_future
    print("finished handshake")
    
    import time
    while True:
        
        t = torch.ones(10, 10, device=device)
        print(f"exchanging tensor {t}")
        recv_fut = actor.recv.call(shape=t.shape, dtype=t.dtype)
        torchcomm.send(t, 1, async_op=False)
        # torchcomm.finalize()
        fetched = await recv_fut

        # f = actor.finalize.call() 
        # torchcomm.finalize()
        
        # await f
        
        print(f"{fetched=}")
        time.sleep(1)
    
    


if __name__ == "__main__":
    asyncio.run(main())
