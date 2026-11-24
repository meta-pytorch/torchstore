# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Upstream torchft over torchstore's NIXL integration: train, kill a replica, heal.

A torchft Lighthouse and three replica processes, each on its own accelerator
(one NeuronCore, or one GPU). Every replica trains the same model with upstream
``torchft.Manager``; gradients go through ``ProcessGroupNixl``. Replica 2 dies in
the middle of a step. The survivors keep training as two replicas; the restarted
replica 2 rejoins, heals through ``NixlCheckpointTransport`` (model and Adam
state), and the run ends once all three have trained together past ``STEPS``.
Every replica must finish on the same step with bit-identical parameters, and
replica 2 must have healed.

Needs torchft, NIXL and a non-CPU accelerator with three devices; skipped
otherwise. On Trainium: ``TORCHSTORE_NIXL_BACKEND=LIBFABRIC pytest
tests/test_torchft_nixl.py``.
"""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

STEPS = 12
FAIL_STEP = 5
REPLICAS = 3
FAILING_REPLICA = 2
MAX_STEPS = 2000


def _accelerators() -> tuple[str, int]:
    """(device type, device count) without initializing a device runtime."""
    if importlib.util.find_spec("torch_neuronx") is not None:
        cores = os.environ.get("TORCHSTORE_TEST_NEURON_CORES")
        return "neuron", int(cores) if cores else _neuron_core_count()
    import torch

    if torch.cuda.is_available():
        return "cuda", torch.cuda.device_count()
    return "cpu", 0


def _neuron_core_count() -> int:
    try:
        out = subprocess.run(
            ["neuron-ls", "--json-output"], capture_output=True, text=True, timeout=60
        ).stdout
        return sum(int(d.get("nc_count", 0)) for d in json.loads(out))
    except Exception:
        return 0


def _requirements_missing() -> str:
    for module in ("torchft", "nixl"):
        if importlib.util.find_spec(module) is None and not (
            module == "nixl" and importlib.util.find_spec("nixl_cu12") is not None
        ):
            return f"{module} is not installed"
    kind, count = _accelerators()
    if kind == "cpu":
        return "needs a Neuron or CUDA accelerator"
    if count < REPLICAS:
        return f"needs {REPLICAS} {kind} devices, found {count}"
    return ""


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


@pytest.mark.skipif(bool(_requirements_missing()), reason=_requirements_missing())
def test_upstream_torchft_heals_a_killed_replica_over_nixl(tmp_path: Path) -> None:
    from torchft.coordination import LighthouseServer

    lighthouse = LighthouseServer(bind="[::]:0", min_replicas=2, join_timeout_ms=10_000)
    kind, _ = _accelerators()

    def launch(replica: int, attempt: int) -> subprocess.Popen:
        env = dict(os.environ)
        env.update(
            TORCHFT_LIGHTHOUSE=lighthouse.address(),
            TEST_REPLICA=str(replica),
            TEST_ATTEMPT=str(attempt),
            TEST_OUT=str(tmp_path),
            TEST_STORE_PORT=str(_free_port()),
            TEST_MANAGER_PORT=str(_free_port()),
        )
        if kind == "neuron":
            env["NEURON_RT_VISIBLE_CORES"] = str(replica)
        else:
            env["CUDA_VISIBLE_DEVICES"] = str(replica)
        log = open(tmp_path / f"replica{replica}.attempt{attempt}.log", "w")
        return subprocess.Popen(
            [sys.executable, __file__], env=env, stdout=log, stderr=subprocess.STDOUT
        )

    try:
        procs = {r: launch(r, 0) for r in range(REPLICAS)}
        restarted = False
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            codes = {r: p.poll() for r, p in procs.items()}
            if codes[FAILING_REPLICA] == 1 and not restarted:
                procs[FAILING_REPLICA] = launch(FAILING_REPLICA, 1)
                restarted = True
                continue
            if all(c is not None for c in codes.values()):
                break
            time.sleep(1)
        logs = {f.name: f.read_text()[-3000:] for f in sorted(tmp_path.glob("*.log"))}
        for p in procs.values():
            if p.poll() is None:
                p.kill()
        assert restarted, f"replica {FAILING_REPLICA} never failed: {logs}"
        assert all(p.returncode == 0 for p in procs.values()), logs
    finally:
        lighthouse.shutdown()

    import torch

    results = [torch.load(tmp_path / f"replica{r}.pt") for r in range(REPLICAS)]
    steps = [r["step"] for r in results]
    assert len(set(steps)) == 1 and steps[0] >= STEPS, steps
    # The killed replica must heal. A survivor may heal too: one still timing
    # out on the dead peer can fall a step behind the others' new quorum.
    assert results[FAILING_REPLICA]["healed"], [r["healed"] for r in results]
    for other in results[1:]:
        for a, b in zip(results[0]["params"], other["params"]):
            assert torch.equal(a, b), "replicas diverged after the heal"


def _replica_main() -> None:
    """One replica: upstream torchft Manager, NIXL gradients and healing."""
    from datetime import timedelta

    import torch
    from torch import nn
    from torch.distributed import TCPStore
    from torchft import Manager
    from torchstore.integrations.torchft import (
        NixlCheckpointTransport,
        ProcessGroupNixl,
    )

    replica = int(os.environ["TEST_REPLICA"])
    attempt = int(os.environ["TEST_ATTEMPT"])
    out = Path(os.environ["TEST_OUT"])
    store_port = int(os.environ["TEST_STORE_PORT"])
    store = TCPStore("localhost", store_port, is_master=True, wait_for_workers=False)

    pg = ProcessGroupNixl(timeout=timedelta(seconds=60))
    device = pg._device
    torch.manual_seed(0)  # identical initial weights: no step-0 heal needed
    model = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 16)).to(device)
    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-2)
    for p in params:  # create Adam's state so it is healed with the weights
        p.grad = torch.zeros_like(p)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    healed = [False]

    def state_dict() -> dict:
        return {"model": model.state_dict(), "optim": optimizer.state_dict()}

    def load_state_dict(state: dict) -> None:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optim"])
        healed[0] = True

    transport = NixlCheckpointTransport(pg, timeout=timedelta(seconds=120))
    manager = Manager(
        pg=pg,
        min_replica_size=2,
        load_state_dict=load_state_dict,
        state_dict=state_dict,
        replica_id=f"replica_{replica}",
        store_addr="localhost",
        store_port=store_port,
        rank=0,
        world_size=1,
        port=int(os.environ["TEST_MANAGER_PORT"]),
        lighthouse_addr=os.environ["TORCHFT_LIGHTHOUSE"],
        hostname="localhost",
        use_async_quorum=False,
        init_sync=False,
        checkpoint_transport=transport,
        timeout=timedelta(seconds=60),
        quorum_timeout=timedelta(seconds=300),
    )
    transport.set_state_dict(manager._manager_state_dict)

    loss_fn = nn.CrossEntropyLoss()
    # Stop once all replicas trained a step together past STEPS: the survivors
    # wait for the rejoiner instead of finishing without it.
    while manager.current_step() < MAX_STEPS:
        manager.start_quorum()
        step = manager.current_step()
        g = torch.Generator().manual_seed(1000 * replica + step)
        x = torch.randn(32, 256, generator=g).to(device)
        y = torch.randint(16, (32,), generator=g).to(device)
        loss_fn(model(x), y).backward()
        if replica == FAILING_REPLICA and attempt == 0 and step == FAIL_STEP:
            os._exit(1)  # die mid-step: the survivors' allreduce loses a peer
        flat = torch.cat([p.grad.reshape(-1) for p in params])
        manager.allreduce(flat).wait()
        offset = 0
        for p in params:
            n = p.numel()
            p.grad.copy_(flat[offset : offset + n].view_as(p))
            offset += n
        if manager.should_commit():
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        print(
            f"replica {replica} step {step} participants {manager.num_participants()}",
            flush=True,
        )
        if step >= STEPS and manager.num_participants() == REPLICAS:
            break
        time.sleep(0.2)

    torch.save(
        {
            "step": manager.current_step(),
            "healed": healed[0],
            "params": [p.detach().cpu() for p in params],
        },
        out / f"replica{replica}.pt",
    )
    manager.shutdown(wait=False)
    del store
    os._exit(0)  # NIXL's threads can crash a normal interpreter exit


if __name__ == "__main__":
    _replica_main()
