# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import json
import os
import socket
import tempfile
import unittest
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torchstore as ts

_WORLD_SIZE = 2
_MASTER_ADDR = "127.0.0.1"
_ENV_KEY = "env.tensor"
_EXPLICIT_OBJECT_KEY = "explicit.object"
_BATCH_KEYS = ("explicit.left", "explicit.right")


def _supports_monarch_spmd() -> bool:
    if not dist.is_available() or not dist.is_gloo_available():
        return False

    try:
        from monarch._src.spmd.host_mesh import host_mesh_from_store
    except ImportError:
        return False

    return callable(host_mesh_from_store)


def _reserve_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((_MASTER_ADDR, 0))
        return int(sock.getsockname()[1])


def _configure_spmd_env(rank: int, master_port: int) -> None:
    os.environ["HOSTNAME"] = socket.gethostname()
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(_WORLD_SIZE)
    os.environ["MASTER_ADDR"] = _MASTER_ADDR
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(_WORLD_SIZE)


def _strategy(strategy_name: str) -> ts.TorchStoreStrategy:
    if strategy_name == "host":
        return ts.HostStrategy()
    if strategy_name == "local_rank":
        return ts.LocalRankStrategy()
    raise ValueError(f"Unsupported strategy: {strategy_name}")


async def _run_noop_cycle(strategy_name: str, store_name: str) -> dict[str, bool]:
    await ts.initialize_spmd(strategy=_strategy(strategy_name), store_name=store_name)
    dist.barrier()
    await ts.shutdown(store_name=store_name)
    dist.barrier()
    return {"completed": True}


async def _run_env_cycle(
    rank: int, strategy_name: str, store_name: str
) -> dict[str, Any]:
    await ts.initialize_spmd(strategy=_strategy(strategy_name), store_name=store_name)
    missing_exists = await ts.exists(_ENV_KEY, store_name=store_name)

    if rank == 0:
        await ts.put(
            _ENV_KEY,
            torch.tensor([10, 20, 30], dtype=torch.int64),
            store_name=store_name,
        )

    dist.barrier()
    value = await ts.get(_ENV_KEY, store_name=store_name)
    keys = sorted(await ts.keys("env", store_name=store_name))
    dist.barrier()
    await ts.shutdown(store_name=store_name)
    dist.barrier()

    return {
        "keys": keys,
        "missing_exists": missing_exists,
        "tensor": value.tolist(),
    }


async def _run_explicit_cycle(
    rank: int,
    strategy_name: str,
    store_name: str,
    master_port: int,
) -> dict[str, Any]:
    env = ts.spmd.SPMDEnv(
        rank=rank,
        local_rank=rank,
        world_size=_WORLD_SIZE,
        local_world_size=_WORLD_SIZE,
        master_addr=_MASTER_ADDR,
        master_port=master_port,
    )
    await ts.initialize_spmd(
        strategy=_strategy(strategy_name),
        store_name=store_name,
        env=env,
    )
    missing_object = await ts.exists(_EXPLICIT_OBJECT_KEY, store_name=store_name)

    if rank == 1:
        await ts.put(
            _EXPLICIT_OBJECT_KEY,
            {"strategy": strategy_name, "writer_rank": rank},
            store_name=store_name,
        )
        await ts.put_batch(
            {
                _BATCH_KEYS[0]: torch.tensor([101, 102], dtype=torch.int64),
                _BATCH_KEYS[1]: torch.tensor([201, 202], dtype=torch.int64),
            },
            store_name=store_name,
        )

    dist.barrier()
    payload = await ts.get(_EXPLICIT_OBJECT_KEY, store_name=store_name)
    batch = await ts.get_batch(list(_BATCH_KEYS), store_name=store_name)
    keys = sorted(await ts.keys("explicit", store_name=store_name))
    object_exists = await ts.exists(_EXPLICIT_OBJECT_KEY, store_name=store_name)
    dist.barrier()
    await ts.shutdown(store_name=store_name)
    dist.barrier()

    return {
        "batch": {key: batch[key].tolist() for key in _BATCH_KEYS},
        "keys": keys,
        "missing_object": missing_object,
        "object_exists": object_exists,
        "payload": payload,
    }


async def _run_scenario(
    rank: int,
    strategy_name: str,
    scenario_name: str,
    store_name: str,
    master_port: int,
) -> dict[str, Any]:
    _configure_spmd_env(rank, master_port)

    if scenario_name == "noop":
        return {
            "rank": rank,
            "scenario": await _run_noop_cycle(strategy_name, store_name),
        }
    if scenario_name == "env":
        return {
            "rank": rank,
            "scenario": await _run_env_cycle(rank, strategy_name, store_name),
        }
    if scenario_name == "explicit":
        return {
            "rank": rank,
            "scenario": await _run_explicit_cycle(
                rank,
                strategy_name,
                store_name,
                master_port,
            ),
        }

    raise ValueError(f"Unsupported scenario: {scenario_name}")


def _worker(
    rank: int,
    strategy_name: str,
    scenario_name: str,
    store_name: str,
    pg_file: str,
    result_dir: str,
    master_port: int,
) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{pg_file}",
        rank=rank,
        world_size=_WORLD_SIZE,
    )
    try:
        results = asyncio.run(
            _run_scenario(
                rank,
                strategy_name,
                scenario_name,
                store_name,
                master_port,
            )
        )
        result_path = Path(result_dir) / f"rank_{rank}.json"
        result_path.write_text(json.dumps(results), encoding="utf-8")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_spmd_scenario(
    strategy_name: str,
    scenario_name: str,
) -> list[dict[str, Any]]:
    strategy_suffix = "h" if strategy_name == "host" else "lr"
    scenario_suffix = scenario_name[0]
    store_name = f"ts_{strategy_suffix}_{scenario_suffix}_{uuid.uuid4().hex[:8]}"
    master_port = _reserve_free_port()

    with tempfile.TemporaryDirectory() as temp_dir:
        pg_file = Path(temp_dir) / "dist_init"
        mp.spawn(
            _worker,
            args=(
                strategy_name,
                scenario_name,
                store_name,
                str(pg_file),
                temp_dir,
                master_port,
            ),
            nprocs=_WORLD_SIZE,
            join=True,
        )
        return [
            json.loads((Path(temp_dir) / f"rank_{rank}.json").read_text("utf-8"))
            for rank in range(_WORLD_SIZE)
        ]


class SPMDEnvTest(unittest.TestCase):
    def _env_vars(self, **overrides: str | int | None) -> dict[str, str]:
        env_vars: dict[str, str] = {
            "RANK": "3",
            "LOCAL_RANK": "1",
            "WORLD_SIZE": "8",
            "LOCAL_WORLD_SIZE": "2",
            "MASTER_ADDR": _MASTER_ADDR,
            "MASTER_PORT": "29500",
        }
        for key, value in overrides.items():
            if value is None:
                env_vars.pop(key, None)
            else:
                env_vars[key] = str(value)
        return env_vars

    def test_from_env_parses_torchrun_environment(self) -> None:
        expected = ts.spmd.SPMDEnv(
            rank=3,
            local_rank=1,
            world_size=8,
            local_world_size=2,
            master_addr=_MASTER_ADDR,
            master_port=29500,
        )

        with patch.dict(os.environ, self._env_vars(), clear=True):
            env = ts.spmd.SPMDEnv.from_env()

        self.assertEqual(env, expected)
        self.assertEqual(env.num_hosts, 4)
        self.assertEqual(env.group_rank, 1)

    def test_from_env_uses_explicit_master_defaults(self) -> None:
        with patch.dict(
            os.environ,
            self._env_vars(MASTER_ADDR=None, MASTER_PORT=None),
            clear=True,
        ):
            env = ts.spmd.SPMDEnv.from_env(
                master_addr="localhost",
                master_port=29600,
            )

        self.assertEqual(env.master_addr, "localhost")
        self.assertEqual(env.master_port, 29600)
        self.assertEqual(env.rank, 3)
        self.assertEqual(env.local_rank, 1)

    def test_from_env_rejects_non_divisible_topology(self) -> None:
        with patch.dict(
            os.environ,
            self._env_vars(WORLD_SIZE=3, LOCAL_WORLD_SIZE=2),
            clear=True,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "world_size \\(3\\) must be divisible by local_world_size \\(2\\)",
            ):
                ts.spmd.SPMDEnv.from_env()


@unittest.skipUnless(
    _supports_monarch_spmd(),
    "Monarch SPMD helpers are unavailable in this environment",
)
class SPMDTest(unittest.TestCase):
    def _assert_noop_results(self, results: list[dict[str, Any]]) -> None:
        self.assertCountEqual([result["rank"] for result in results], [0, 1])
        for result in results:
            self.assertTrue(result["scenario"]["completed"])

    def _assert_env_results(self, results: list[dict[str, Any]]) -> None:
        self.assertCountEqual([result["rank"] for result in results], [0, 1])
        for result in results:
            self.assertFalse(result["scenario"]["missing_exists"])
            self.assertEqual(result["scenario"]["tensor"], [10, 20, 30])
            self.assertEqual(result["scenario"]["keys"], [_ENV_KEY])

    def _assert_explicit_results(
        self,
        results: list[dict[str, Any]],
        strategy_name: str,
    ) -> None:
        self.assertCountEqual([result["rank"] for result in results], [0, 1])
        for result in results:
            self.assertFalse(result["scenario"]["missing_object"])
            self.assertTrue(result["scenario"]["object_exists"])
            self.assertEqual(
                result["scenario"]["payload"],
                {"strategy": strategy_name, "writer_rank": 1},
            )
            self.assertEqual(
                result["scenario"]["batch"],
                {
                    _BATCH_KEYS[0]: [101, 102],
                    _BATCH_KEYS[1]: [201, 202],
                },
            )
            self.assertCountEqual(
                result["scenario"]["keys"],
                [_EXPLICIT_OBJECT_KEY, *_BATCH_KEYS],
            )

    def _assert_strategy(self, strategy_name: str) -> None:
        scenarios = (
            ("noop", self._assert_noop_results),
            ("env", self._assert_env_results),
            (
                "explicit",
                lambda results: self._assert_explicit_results(results, strategy_name),
            ),
        )
        for scenario_name, assert_results in scenarios:
            with self.subTest(strategy=strategy_name, scenario=scenario_name):
                results = _run_spmd_scenario(strategy_name, scenario_name)
                assert_results(results)

    def test_host_strategy_spmd_lifecycle(self) -> None:
        self._assert_strategy("host")

    def test_local_rank_strategy_spmd_lifecycle(self) -> None:
        self._assert_strategy("local_rank")
