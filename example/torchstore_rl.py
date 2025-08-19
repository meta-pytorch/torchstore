# pyre-unsafe
import asyncio

import re
from typing import List, Tuple

import torch
from monarch.actor import Actor, current_rank, endpoint, proc_mesh
from pyre_extensions import none_throws
from torchstore import MultiProcessStore
from torchstore._state_dict_utils import get_state_dict, push_state_dict


# Run the example : python example/torchstore_rl.py

MAX_BUFFER_SIZE = 3


class WeightsTracker(Actor):
    def __init__(self) -> None:
        # FIFO tracking.
        # TODO: Synchronize the queue!!.
        self.weights_tracking_queue = asyncio.Queue(maxsize=MAX_BUFFER_SIZE)

    @endpoint
    async def mark_weights_ready(self, key: str):
        print(f"[weights_life_cycle_manager] weights are ready to consume: {key}")
        await self.weights_tracking_queue.put(key)
        print(
            f"[weights_life_cycle_manager] queue content : {self.weights_tracking_queue.qsize()}"
        )

    @endpoint
    async def get(self) -> str:
        ready_key = await asyncio.wait_for(
            self.weights_tracking_queue.get(), timeout=5.0
        )
        return ready_key


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
        self, weights_id: str, input_logit: torch.Tensor, input_reward: torch.Tensor
    ):
        logits = self.model(input_logit)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        loss = -torch.mean(input_reward.detach().squeeze() * log_probs.sum(dim=[1, -2]))
        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optim.step()
        print(
            f"[learner] weights_id : {weights_id}, weights: {self.model.state_dict()}"
        )
        # Put weights in to torch.store
        await push_state_dict(self.store, self.model.state_dict(), key=weights_id)


class Generator(Actor):
    def __init__(self, store):
        self.store = store
        self.model = torch.nn.Linear(4, 4, bias=False, device="cuda")
        self.index = current_rank()["gpus"]
        self.prev_weights_id = "bootstrap"

    @endpoint
    async def update_weights(self, weights_id: str):
        print(
            f"[generator {self.index}], prev. weights_id : {self.prev_weights_id},  prev. weights: {self.model.state_dict()}"
        )
        # Fetch weights from torch.store
        await get_state_dict(
            self.store, key=weights_id, user_state_dict=self.model.state_dict()
        )
        self.prev_weights_id = weights_id
        print(
            f"[generator {self.index}], new weights_id: {weights_id}, new weights: {self.model.state_dict()}"
        )

    @endpoint
    async def generate(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = inputs.to("cuda")
        logits = self.model(inputs)
        # await asyncio.sleep(2)  # Simulate extra work
        reward = torch.sum(logits)
        return logits, reward


async def run_learner(learner, tracker, queue):
    for weights_id in range(1, 4):
        # fetch generated input from the generator via queue.
        input_logit, input_reward = None, None
        try:
            # TODO: Move data via torchstore.
            input_logit, input_reward = await asyncio.wait_for(queue.get(), timeout=5.0)
        except TimeoutError:
            print("timeout waiting for logits and rewards from generator!")
        weights_id = f"v{weights_id}"
        # TODO: May be there should be a 'reserve_weights_ready' call to avoid OOMing torchstore.
        await learner.step.call_one(weights_id, input_logit, input_reward)
        # Notify weights are ready to consume.
        await tracker.mark_weights_ready.call_one(weights_id)


async def run_generator(generators, tracker, queue):
    for _ in range(4):
        await asyncio.sleep(2)  # Simulate extra work
        weights_id = await tracker.get.call_one()
        await generators.update_weights.call_one(weights_id)
        logits, reward = await generators.generate.call_one(
            torch.randn(4, 4, device="cuda")
        )
        # send the generated input to the learner via queue.
        await asyncio.wait_for(queue.put((logits, reward)), timeout=5.0)


async def main():
    """
    The example code shows how to use torchstore to share weights between
    trainer/learner and generator apps.
    """
    num_learners = 1
    num_generators = 1

    # TODO: Show weights re-sharding usecase.
    learner_mesh = await proc_mesh(gpus=num_learners)
    gen_mesh = await proc_mesh(gpus=num_generators)
    weight_tracker_mesh = await proc_mesh(
        gpus=1
    )  # TODO: We only need a CPU based service.

    queue = asyncio.Queue(
        maxsize=MAX_BUFFER_SIZE
    )  # rewards are still exchanged out of band to torchstore using a queue.
    store = await MultiProcessStore.create_store()

    learner = await learner_mesh.spawn("learner", Learner, store)
    generators = await gen_mesh.spawn("generator", Generator, store)
    weights_tracker = await weight_tracker_mesh.spawn("weights_tracker", WeightsTracker)

    # Bootstrapping the pipeline for off by one processing.
    # Disclaimer: I do not understand bootstrapping entirely.
    # I assume we somehow have access to two weights versions (persistent checkpoint?)
    # to bootstrap the off by one pipeline.
    cp_loaded_sd1 = {"weight": torch.randn(4, 4, device="cuda")}
    cp_loaded_sd2 = {"weight": torch.randn(4, 4, device="cuda")}

    logits, reward = await generators.generate.call_one(cp_loaded_sd1["weight"])
    await asyncio.wait_for(queue.put((logits, reward)), timeout=2.0)

    await push_state_dict(store, cp_loaded_sd2, key="v0")
    await weights_tracker.mark_weights_ready.call_one("v0")

    # Concurrent execution of leaner and generator.
    async with asyncio.TaskGroup() as tg:
        tg.create_task(run_learner(learner, weights_tracker, queue))
        tg.create_task(run_generator(generators, weights_tracker, queue))

    print("done")


asyncio.run(main())
