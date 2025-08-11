# pyre-unsafe
import asyncio

from typing import List, Tuple

import torch
from monarch.actor import Actor, current_rank, endpoint, proc_mesh
from torchstore import MultiProcessStore
from torchstore._state_dict_utils import get_state_dict, push_state_dict


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

    @endpoint
    async def step(
        self,
        inputs: List[Tuple[torch.Tensor, torch.Tensor]],
    ):
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
        print("[learner] weights: ", self.model.state_dict())
        # Put weights in to torch.store
        await push_state_dict(self.store, self.model.state_dict(), key="toy_app")


class Generator(Actor):
    def __init__(self, store):
        self.store = store
        self.model = torch.nn.Linear(4, 4, bias=False, device="cuda")
        self.index = current_rank()["gpus"]

    @endpoint
    async def update_weights(self):
        print(
            "[generator {}] original weights: {}".format(
                self.index, self.model.state_dict()
            )
        )
        # Fetch weights from torch.store
        await get_state_dict(
            self.store, key="toy_app", user_state_dict=self.model.state_dict()
        )
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
    num_learners = 1
    num_generators = 1

    # TODO: Show weights re-sharding usecase.
    learner_mesh = await proc_mesh(gpus=num_learners)
    gen_mesh = await proc_mesh(gpus=num_generators)

    store = await MultiProcessStore.create_store()

    learner = await learner_mesh.spawn("learner", Learner, store)
    generators = await gen_mesh.spawn("generator", Generator, store)

    generation_stream = generators.generate.stream(torch.randn(4, 4, device="cuda"))
    for _ in range(3):
        generations = [gen.get() for gen in generation_stream]
        await learner.step.call_one(generations)
        generation_stream = generators.generate.stream(torch.randn(4, 4, device="cuda"))
        await generators.update_weights.call_one()

    print("done")


asyncio.run(main())
