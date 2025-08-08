# pyre-unsafe
import asyncio

from typing import List, Tuple

import torch
from monarch.actor import Actor, current_rank, endpoint, proc_mesh
from torchstore import MultiProcessStore


# Run the example : python example/torchstore_rl.py


class Learner(Actor):
    def __init__(self, store):
        self.store = store
        self.device = torch.device("cpu")
        self.model = torch.nn.Linear(4, 4, bias=False, device=self.device)
        self.optim = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-3,
            eps=1e-5,
        )

    async def store_weights(self):
        for k, v in self.model.state_dict().items():
            await self.store.put(k, v)

    @endpoint
    async def step(
        self,
        inputs: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # list(tensor, tensor) => list(tensor), list(tensor)
        inputs, rewards = zip(*inputs)

        # list(tensor) => tensor
        tensor = torch.stack(inputs).to(self.device)
        rewards = torch.stack(rewards).to(self.device)

        logits = self.model(tensor)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        loss = -torch.mean(rewards.detach().squeeze() * log_probs.sum(dim=[1, 2]))
        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optim.step()
        # put the weights in to torch.store
        await self.store_weights()
        print("[learner] weights: ", self.model.state_dict())
        return loss, rewards.sum()


class Generator(Actor):
    def __init__(self, store):
        self.store = store
        self.model = torch.nn.Linear(4, 4, bias=False, device="cuda")
        # self.weight_buffer = weight_buffer
        self.index = current_rank()["gpus"]

    @endpoint
    def update_weights(self):
        print(
            "[generator {}] original weights: {}".format(
                self.index, self.model.state_dict()
            )
        )
        state_dict = self.model.state_dict()
        for k, v in state_dict.items():
            asyncio.run(self.store.get(k, inplace_tensor=v))
        print(
            "[generator {}] new weights: {}".format(self.index, self.model.state_dict())
        )

    @endpoint
    async def generate(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = inputs.to("cuda")
        logits = self.model(inputs)
        reward = torch.sum(logits)
        return logits, reward


async def main():
    num_generators = 1
    learner_mesh = await proc_mesh(gpus=1, env={})
    gen_mesh = await proc_mesh(gpus=num_generators, env={})

    store = await MultiProcessStore.create_store()
    learner = await learner_mesh.spawn("learner", Learner, store)
    generators = await gen_mesh.spawn("generator", Generator, store)

    generation_stream = generators.generate.stream(torch.randn(4, 4, device="cuda"))
    for step in range(3):
        generations = [gen.get() for gen in generation_stream]
        loss, rewards = await learner.step.call_one(generations)
        print(f"step: {step}, loss: {loss}, rewards: {rewards}")
        generation_stream = generators.generate.stream(torch.randn(4, 4, device="cuda"))
        generators.update_weights.call().get()

    print("done")


asyncio.run(main())
