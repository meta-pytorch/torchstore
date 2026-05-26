# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
import asyncio
import os
from functools import partial

import torch
import torchstore as ts
from monarch.actor import Actor, current_rank, endpoint, shutdown_context, this_host

# Run the example : python example/torchstore_rl.py


def set_cuda_visible_devices(devices: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = devices


class Learner(Actor):
    def __init__(self):
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
        input_logit: torch.Tensor,
        input_reward: torch.Tensor,
    ):
        logits = self.model(input_logit)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        loss = -torch.mean(input_reward.detach().squeeze() * log_probs.sum(dim=[1, -2]))
        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optim.step()
        print("[learner] weights: ", self.model.state_dict())
        # Put weights in to torch.store
        await ts.put_state_dict(self.model.state_dict(), key="toy_app")


class Generator(Actor):
    def __init__(self):
        self.model = torch.nn.Linear(4, 4, bias=False, device="cuda")
        self.index = current_rank()["gpus"]

    @endpoint
    async def update_weights(self):
        print(f"[generator {self.index}] original weights: {self.model.state_dict()}")
        # Fetch weights from torch.store
        await ts.get_state_dict(key="toy_app", user_state_dict=self.model.state_dict())
        print(f"[generator {self.index}] new weights: {self.model.state_dict()}")

    @endpoint
    async def generate(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = inputs.to("cuda")
        logits = self.model(inputs)
        reward = torch.sum(logits)
        return logits, reward


async def main():
    """
    The example code shows how to use torchstore to share weights between
    trainer/learner and generator apps. The weights are shared synchronously
    between the two apps.
    """
    num_learners = 1
    num_generators = 1

    # TODO: Show weights re-sharding usecase.
    learner_mesh = this_host().spawn_procs(
        per_host={"gpus": num_learners},
        bootstrap=partial(set_cuda_visible_devices, "0"),
    )
    gen_mesh = this_host().spawn_procs(
        per_host={"gpus": num_generators},
        bootstrap=partial(set_cuda_visible_devices, "1"),
    )

    await ts.initialize()

    learner = learner_mesh.spawn("learner", Learner)
    generators = gen_mesh.spawn("generator", Generator)

    logits, reward = await generators.generate.call_one(
        torch.randn(4, 4, device="cuda")
    )
    for _ in range(3):
        await learner.step.call_one(logits, reward)
        logits, reward = await generators.generate.call_one(
            torch.randn(4, 4, device="cuda")
        )
        await generators.update_weights.call_one()

    print("done")
    await ts.shutdown()


asyncio.run(main())
shutdown_context().get(timeout=2.0)
