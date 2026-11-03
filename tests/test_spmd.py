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
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import cloudpickle
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torchstore as ts
import torchstore.api as _api
import torchstore.monarch.spmd as monarch_spmd
from torchstore.monarch.controller import Controller
from torchstore.monarch.spmd import _SPMDSession

from .backend_stub import install_backend_stub

_WORLD_SIZE = 2
_MASTER_ADDR = "127.0.0.1"
_ENV_KEY = "env.tensor"
_EXPLICIT_OBJECT_KEY = "explicit.object"
_BATCH_KEYS = ("explicit.left", "explicit.right")


def _supports_monarch_spmd() -> bool:
    if not dist.is_available() or not dist.is_gloo_available():
        return False

    try:
        from monarch.job.spmd import StoreJob
    except ImportError:
        return False

    return callable(StoreJob.from_store)


def _reserve_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((_MASTER_ADDR, 0))
        return int(sock.getsockname()[1])


def _configure_spmd_env(rank: int, master_port: int) -> None:
    os.environ["TORCHSTORE_HOSTNAME"] = socket.gethostname()
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


async def _run_noop_cycle(
    strategy_name: str,
    store_name: str,
) -> dict[str, bool]:
    configured = False

    def configure_job(_job: Any) -> None:
        nonlocal configured
        configured = True

    await ts.initialize_spmd(
        strategy=_strategy(strategy_name),
        store_name=store_name,
        configure_job=configure_job,
    )
    dist.barrier()
    await ts.shutdown(store_name=store_name)
    dist.barrier()
    return {"completed": True, "configured": configured}


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


def _two_host_env(*, rank: int) -> ts.spmd.SPMDEnv:
    """An SPMD environment of two hosts with two ranks each, as seen by ``rank``."""
    return ts.spmd.SPMDEnv(
        rank=rank,
        local_rank=rank % 2,
        world_size=4,
        local_world_size=2,
        master_addr=_MASTER_ADDR,
        master_port=0,
    )


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


class SPMDInitializeTest(unittest.IsolatedAsyncioTestCase):
    async def test_attach_failure_cleans_up_store_job(self) -> None:
        store_name = f"attach_failure_{uuid.uuid4().hex[:8]}"
        env = ts.spmd.SPMDEnv(
            rank=0,
            local_rank=0,
            world_size=1,
            local_world_size=1,
            master_addr=_MASTER_ADDR,
            master_port=_reserve_free_port(),
        )
        job = MagicMock()
        job.state.side_effect = RuntimeError("attach failed")

        with (
            patch.dict(os.environ),
            patch("torchstore.monarch.spmd.TCPStore"),
            patch("torchstore.monarch.spmd.StoreJob.from_store", return_value=job),
        ):
            with self.assertRaisesRegex(RuntimeError, "attach failed"):
                await ts.initialize_spmd(
                    strategy=ts.HostStrategy(),
                    store_name=store_name,
                    env=env,
                )

        job.kill.assert_called_once_with()
        self.assertNotIn(store_name, _api._spmd_state_map)

    async def test_a_failing_cleanup_logs_under_the_spmd_logger(self) -> None:
        """Log a cleanup that raises after a failed attach, under ``torchstore.spmd`` as before."""
        job = MagicMock()
        job.state.side_effect = RuntimeError("attach failed")
        job.kill.side_effect = RuntimeError("kill failed")

        with (
            patch.dict(os.environ),
            patch("torchstore.monarch.spmd.TCPStore"),
            patch("torchstore.monarch.spmd.StoreJob.from_store", return_value=job),
            self.assertLogs("torchstore.spmd", level="WARNING") as logs,
        ):
            with self.assertRaisesRegex(RuntimeError, "attach failed"):
                await ts.initialize_spmd(
                    strategy=ts.HostStrategy(),
                    store_name=f"cleanup_failure_{uuid.uuid4().hex[:8]}",
                    env=_two_host_env(rank=0),
                )

        self.assertEqual(
            [(record.name, record.getMessage()) for record in logs.records],
            [
                (
                    "torchstore.spmd",
                    "torchstore.spmd: job cleanup after init failure raised",
                )
            ],
        )

    async def test_initialize_refuses_a_strategy_before_reaching_the_backend(
        self,
    ) -> None:
        with pytest.MonkeyPatch.context() as monkeypatch:
            backend = install_backend_stub(monkeypatch)
            with self.assertRaisesRegex(RuntimeError, "explicit HostStrategy"):
                await ts.initialize_spmd(strategy=ts.TorchStoreStrategy())
        backend.initialize_spmd.assert_not_awaited()

    async def test_initialize_hands_the_backend_the_rest(self) -> None:
        env = ts.spmd.SPMDEnv(
            rank=0,
            local_rank=0,
            world_size=1,
            local_world_size=1,
            master_addr=_MASTER_ADDR,
            master_port=0,
        )
        strategy = ts.HostStrategy()
        with (
            pytest.MonkeyPatch.context() as monkeypatch,
            patch.dict(_api._spmd_state_map),
        ):
            backend = install_backend_stub(monkeypatch)
            await ts.initialize_spmd(
                strategy=strategy,
                store_name="store",
                env=env,
                rendezvous_timeout=timedelta(seconds=7),
                transport="tcp",
                monarch_port=7,
                configure_job=print,
            )
            registered = _api._spmd_state_map["store"]
        backend.initialize_spmd.assert_awaited_once_with(
            strategy,
            "store",
            env=env,
            rendezvous_timeout=timedelta(seconds=7),
            transport="tcp",
            monarch_port=7,
            configure_job=print,
        )
        self.assertIs(registered, backend.initialize_spmd.return_value)

    async def test_initialize_refuses_a_running_monarch_context(self) -> None:
        running = SimpleNamespace(_context=SimpleNamespace(get=lambda: object()))
        with (
            patch.dict(os.environ),
            patch.object(monarch_spmd, "actor_mesh", running),
            patch("torchstore.monarch.spmd.TCPStore") as tcp_store,
            patch("torchstore.monarch.spmd.StoreJob.from_store", return_value=None),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "requires a fresh Monarch context"
            ):
                await ts.initialize_spmd(
                    strategy=ts.HostStrategy(), env=_two_host_env(rank=1)
                )
        tcp_store.assert_not_called()

    async def _bootstrap_rank_zero(
        self, strategy: ts.TorchStoreStrategy
    ) -> tuple[Any, AsyncMock, str]:
        """Bootstrap rank 0 of two hosts with stand-ins, returning its host mesh and initialize."""
        store_name = f"configure_{uuid.uuid4().hex[:8]}"
        job = MagicMock()
        host_mesh = getattr(job.state.return_value, f"torchstore_{store_name}")
        controller = SimpleNamespace(store=store_name)
        configure_job = MagicMock()
        initialize = AsyncMock()
        lookup = AsyncMock(return_value=controller)
        with (
            patch.dict(os.environ),
            patch("torchstore.monarch.spmd.TCPStore") as tcp_store,
            patch("torchstore.monarch.spmd.StoreJob.from_store", return_value=job),
            patch("torchstore.monarch.spmd.get_or_spawn_controller", lookup),
            patch.object(_api, "initialize", initialize),
            patch.dict(_api._spmd_state_map),
        ):
            await ts.initialize_spmd(
                strategy=strategy,
                store_name=store_name,
                env=_two_host_env(rank=0),
                configure_job=configure_job,
            )
            session = _api._spmd_state_map[store_name]
        tcp_store.assert_called_once_with(
            _MASTER_ADDR, 0, 4, is_master=True, timeout=timedelta(seconds=120)
        )
        configure_job.assert_called_once_with(job)
        lookup.assert_awaited_once_with(store_name, Controller)
        self.assertIs(session.controller, controller)
        # Rank 0 broadcasts the controller handle to the other ranks.
        tcp_store.return_value.set.assert_called_once()
        key, value = tcp_store.return_value.set.call_args.args
        self.assertEqual(key, f"torchstore/spmd/{store_name}/controller")
        self.assertEqual(cloudpickle.loads(value), controller)
        return host_mesh, initialize, store_name

    async def test_rank_zero_spawns_a_volume_per_host_for_host_strategy(self) -> None:
        strategy = ts.HostStrategy()
        host_mesh, initialize, store_name = await self._bootstrap_rank_zero(strategy)
        host_mesh.spawn_procs.assert_called_once_with(name=f"{store_name}_spmd")
        initialize.assert_awaited_once_with(
            num_storage_volumes=2,
            strategy=strategy,
            store_name=store_name,
            mesh=host_mesh.spawn_procs.return_value,
        )

    async def test_rank_zero_spawns_a_volume_per_rank_for_local_rank_strategy(
        self,
    ) -> None:
        strategy = ts.LocalRankStrategy()
        host_mesh, initialize, store_name = await self._bootstrap_rank_zero(strategy)
        host_mesh.spawn_procs.assert_called_once_with(
            per_host={"gpus": 2}, name=f"{store_name}_spmd"
        )
        initialize.assert_awaited_once_with(
            num_storage_volumes=4,
            strategy=strategy,
            store_name=store_name,
            mesh=host_mesh.spawn_procs.return_value,
        )

    async def test_other_ranks_load_the_broadcast_controller_without_a_job(
        self,
    ) -> None:
        store_name = f"peer_{uuid.uuid4().hex[:8]}"
        controller = SimpleNamespace(store=store_name)
        configure_job = MagicMock()
        initialize = AsyncMock()
        with (
            patch.dict(os.environ),
            patch("torchstore.monarch.spmd.TCPStore") as tcp_store,
            patch("torchstore.monarch.spmd.StoreJob.from_store", return_value=None),
            patch.object(_api, "initialize", initialize),
            patch.dict(_api._spmd_state_map),
        ):
            tcp_store.return_value.get.return_value = cloudpickle.dumps(controller)
            await ts.initialize_spmd(
                strategy=ts.HostStrategy(),
                store_name=store_name,
                env=_two_host_env(rank=1),
                configure_job=configure_job,
            )
            session = _api._spmd_state_map[store_name]
        configure_job.assert_not_called()
        initialize.assert_not_awaited()
        tcp_store.return_value.get.assert_called_once_with(
            f"torchstore/spmd/{store_name}/controller"
        )
        self.assertEqual(session.controller, controller)
        self.assertFalse(session.is_primary)

    async def test_initialize_reads_the_environment_and_forwards_each_setting(
        self,
    ) -> None:
        """Read the topology from the environment, then hand each setting to the rendezvous and job."""
        store_name = f"env_{uuid.uuid4().hex[:8]}"
        topology = {
            "RANK": "1",
            "LOCAL_RANK": "1",
            "WORLD_SIZE": "4",
            "LOCAL_WORLD_SIZE": "2",
            "MASTER_ADDR": _MASTER_ADDR,
            "MASTER_PORT": "29512",
        }
        timeout = timedelta(seconds=7)
        with (
            patch.dict(os.environ, topology),
            patch("torchstore.monarch.spmd.TCPStore") as tcp_store,
            patch(
                "torchstore.monarch.spmd.StoreJob.from_store", return_value=None
            ) as from_store,
            patch.dict(_api._spmd_state_map),
        ):
            os.environ.pop("TORCHSTORE_HOSTNAME", None)
            tcp_store.return_value.get.return_value = cloudpickle.dumps(store_name)
            await ts.initialize_spmd(
                strategy=ts.HostStrategy(),
                store_name=store_name,
                rendezvous_timeout=timeout,
                transport="tcp",
                monarch_port=7,
            )
            hostname = os.environ.get("TORCHSTORE_HOSTNAME")
        tcp_store.assert_called_once_with(
            _MASTER_ADDR, 29512, 4, is_master=False, timeout=timeout
        )
        from_store.assert_called_once_with(
            tcp_store.return_value,
            monarch_port=7,
            name=f"torchstore_{store_name}",
            transport="tcp",
            rank=1,
            local_rank=1,
            world_size=4,
            local_world_size=2,
        )
        self.assertEqual(hostname, socket.gethostname())

    async def test_a_failure_after_the_attach_shuts_the_host_mesh_down(self) -> None:
        """Kill the job and shut the attached host mesh down when the store fails to start."""
        store_name = f"start_failure_{uuid.uuid4().hex[:8]}"
        job = MagicMock()
        host_mesh = getattr(job.state.return_value, f"torchstore_{store_name}")
        host_mesh.shutdown = AsyncMock()
        with (
            patch.dict(os.environ),
            patch("torchstore.monarch.spmd.TCPStore"),
            patch("torchstore.monarch.spmd.StoreJob.from_store", return_value=job),
            patch.object(
                _api, "initialize", AsyncMock(side_effect=RuntimeError("start failed"))
            ),
            patch.dict(_api._spmd_state_map),
        ):
            with self.assertRaisesRegex(RuntimeError, "start failed"):
                await ts.initialize_spmd(
                    strategy=ts.HostStrategy(),
                    store_name=store_name,
                    env=_two_host_env(rank=0),
                )
            self.assertNotIn(store_name, _api._spmd_state_map)
        job.kill.assert_called_once_with()
        host_mesh.shutdown.assert_awaited_once_with()

    @unittest.skipUnless(
        hasattr(monarch_spmd, "_store_job_import_error"),
        "this Monarch has StoreJob",
    )
    def test_the_store_job_stand_in_raises(self) -> None:
        """Raise where Monarch lacks ``StoreJob``, so no rank mistakes itself for a peer."""
        with self.assertRaisesRegex(RuntimeError, r"StoreJob\.from_store\(\) is not"):
            monarch_spmd.StoreJob.from_store()

    async def _shut_down_a_session(self, error: Exception | None) -> None:
        store_name = f"session_{uuid.uuid4().hex[:8]}"
        cached_during_teardown = []
        registered_during_teardown = []

        async def teardown() -> None:
            cached_during_teardown.append(store_name in _api._local_clent_map)
            registered_during_teardown.append(store_name in _api._spmd_state_map)
            if error is not None:
                raise error

        session = SimpleNamespace(shutdown=AsyncMock(side_effect=teardown))
        with (
            pytest.MonkeyPatch.context() as monkeypatch,
            patch.dict(_api._spmd_state_map, {store_name: session}),
            patch.dict(_api._local_clent_map, {store_name: MagicMock()}),
        ):
            backend = install_backend_stub(monkeypatch)
            if error is None:
                await ts.shutdown(store_name)
            else:
                with self.assertRaisesRegex(RuntimeError, "teardown failed"):
                    await ts.shutdown(store_name)
            self.assertNotIn(store_name, _api._spmd_state_map)
            self.assertNotIn(store_name, _api._local_clent_map)
        session.shutdown.assert_awaited_once_with()
        # Shutdown leaves an SPMD store's teardown to its session and looks up no controller.
        backend.controller.assert_not_awaited()
        self.assertEqual(cached_during_teardown, [True])
        self.assertEqual(registered_during_teardown, [False])

    async def test_shutdown_drops_an_spmd_session_then_its_client(self) -> None:
        """Drop the session first, then the client once the session's teardown returns."""
        await self._shut_down_a_session(None)

    async def test_shutdown_drops_the_client_when_the_teardown_fails(self) -> None:
        await self._shut_down_a_session(RuntimeError("teardown failed"))

    async def test_session_tears_the_store_down_once(self) -> None:
        """Tear the store, the job and the host mesh down once, and tell the peers it went well."""
        controller = SimpleNamespace(teardown=SimpleNamespace(call=AsyncMock()))
        rendezvous = MagicMock()
        job = MagicMock()
        host_mesh = SimpleNamespace(shutdown=AsyncMock())
        session = _SPMDSession(
            rendezvous=rendezvous,
            controller=controller,
            store_name="once",
            job=job,
            host_mesh=host_mesh,
            is_primary=True,
        )
        await session.shutdown()
        await session.shutdown()
        controller.teardown.call.assert_awaited_once_with()
        rendezvous.set.assert_called_once_with("torchstore/spmd/once/shutdown", "ok")
        job.kill.assert_called_once_with()
        host_mesh.shutdown.assert_awaited_once_with()

    async def test_session_does_not_retry_a_failed_teardown(self) -> None:
        error = RuntimeError("teardown failed")
        teardown = AsyncMock(side_effect=error)
        rendezvous = MagicMock()
        session = _SPMDSession(
            rendezvous=rendezvous,
            controller=SimpleNamespace(teardown=SimpleNamespace(call=teardown)),
            store_name="retry",
            job=None,
            host_mesh=None,
            is_primary=True,
        )
        with self.assertRaisesRegex(RuntimeError, "teardown failed"):
            await session.shutdown()
        await session.shutdown()
        teardown.assert_awaited_once_with()
        rendezvous.set.assert_called_once_with(
            "torchstore/spmd/retry/shutdown", repr(error)
        )

    async def test_session_runs_every_step_and_raises_the_first_failure(self) -> None:
        """Run each step after an earlier one fails, raise the first failure and run no step twice."""
        steps = ("teardown", "set", "kill", "mesh")
        for failing in [(step,) for step in steps] + [("teardown", "mesh")]:
            with self.subTest(failing=failing):
                errors = {step: RuntimeError(f"{step} failed") for step in failing}
                teardown = AsyncMock(side_effect=errors.get("teardown"))
                rendezvous = MagicMock()
                rendezvous.set.side_effect = errors.get("set")
                job = MagicMock()
                job.kill.side_effect = errors.get("kill")
                host_mesh = SimpleNamespace(
                    shutdown=AsyncMock(side_effect=errors.get("mesh"))
                )
                session = _SPMDSession(
                    rendezvous=rendezvous,
                    controller=SimpleNamespace(teardown=SimpleNamespace(call=teardown)),
                    store_name="steps",
                    job=job,
                    host_mesh=host_mesh,
                    is_primary=True,
                )
                with self.assertRaises(RuntimeError) as raised:
                    await session.shutdown()
                await session.shutdown()
                self.assertIs(raised.exception, errors[failing[0]])
                teardown.assert_awaited_once_with()
                rendezvous.set.assert_called_once()
                job.kill.assert_called_once_with()
                host_mesh.shutdown.assert_awaited_once_with()

    async def test_a_peer_session_reports_a_rendezvous_that_fails(self) -> None:
        session, rendezvous, _ = self._peer_session(b"ok")
        rendezvous.get.side_effect = TimeoutError("no answer")
        with self.assertRaisesRegex(
            RuntimeError, "Timed out waiting for TorchStore shutdown"
        ) as raised:
            await session.shutdown()
        self.assertIs(raised.exception.__cause__, rendezvous.get.side_effect)

    async def test_a_failing_host_mesh_cleanup_logs_and_keeps_the_init_failure(
        self,
    ) -> None:
        """Log a host mesh cleanup that raises after a failed start, and raise the start's failure."""
        store_name = f"mesh_cleanup_{uuid.uuid4().hex[:8]}"
        job = MagicMock()
        host_mesh = getattr(job.state.return_value, f"torchstore_{store_name}")
        host_mesh.shutdown = AsyncMock(side_effect=RuntimeError("mesh failed"))
        with (
            patch.dict(os.environ),
            patch("torchstore.monarch.spmd.TCPStore"),
            patch("torchstore.monarch.spmd.StoreJob.from_store", return_value=job),
            patch.object(
                _api, "initialize", AsyncMock(side_effect=RuntimeError("start failed"))
            ),
            patch.dict(_api._spmd_state_map),
            self.assertLogs("torchstore.spmd", level="WARNING") as logs,
        ):
            with self.assertRaisesRegex(RuntimeError, "start failed"):
                await ts.initialize_spmd(
                    strategy=ts.HostStrategy(),
                    store_name=store_name,
                    env=_two_host_env(rank=0),
                )
        job.kill.assert_called_once_with()
        self.assertEqual(
            [record.getMessage() for record in logs.records],
            ["torchstore.spmd: host mesh cleanup after init failure raised"],
        )

    def _peer_session(self, status: bytes) -> tuple[_SPMDSession, MagicMock, AsyncMock]:
        """A session on another rank, whose rendezvous answers rank 0's ``status``."""
        rendezvous = MagicMock()
        rendezvous.get.return_value = status
        teardown = AsyncMock()
        session = _SPMDSession(
            rendezvous=rendezvous,
            controller=SimpleNamespace(teardown=SimpleNamespace(call=teardown)),
            store_name="peer",
            job=None,
            host_mesh=None,
            is_primary=False,
        )
        return session, rendezvous, teardown

    async def test_a_peer_session_leaves_the_teardown_to_rank_zero(self) -> None:
        """Wait for rank 0's outcome and return on ``ok``, tearing nothing down itself."""
        session, rendezvous, teardown = self._peer_session(b"ok")
        await session.shutdown()
        rendezvous.get.assert_called_once_with("torchstore/spmd/peer/shutdown")
        rendezvous.set.assert_not_called()
        teardown.assert_not_awaited()

    async def test_a_peer_session_raises_rank_zeros_failure(self) -> None:
        session, _, teardown = self._peer_session(b"RuntimeError('boom')")
        with self.assertRaisesRegex(
            RuntimeError, r"shutdown failure - 'peer': RuntimeError\('boom'\)"
        ):
            await session.shutdown()
        teardown.assert_not_awaited()

    async def test_a_peer_shuts_its_session_down_after_a_failed_routing_client(
        self,
    ) -> None:
        """Take the name from a failed routing client's mark and shut the session down despite a later one."""
        import cloudpickle

        store_name = f"marked_{uuid.uuid4().hex[:8]}"
        env = ts.spmd.SPMDEnv(
            rank=1,
            local_rank=1,
            world_size=2,
            local_world_size=2,
            master_addr=_MASTER_ADDR,
            master_port=_reserve_free_port(),
        )
        rendezvous = MagicMock()
        rendezvous.get.return_value = cloudpickle.dumps("controller")
        with _api._selection_lock:
            _api._attempted.add(store_name)
        try:
            with (
                patch("torchstore.monarch.spmd.TCPStore", return_value=rendezvous),
                patch("torchstore.monarch.spmd.StoreJob.from_store", return_value=None),
            ):
                await ts.initialize_spmd(
                    strategy=ts.HostStrategy(), store_name=store_name, env=env
                )
            self.assertNotIn(store_name, _api._attempted)
            session = _api._spmd_state_map[store_name]
            with _api._selection_lock:
                # A failed client after the session is outranked by it at shutdown.
                _api._attempted.add(store_name)
            with patch.object(session, "shutdown", AsyncMock()) as shutdown:
                await ts.shutdown(store_name)
            shutdown.assert_awaited_once_with()
        finally:
            _api._spmd_state_map.pop(store_name, None)
            _api._attempted.discard(store_name)


@unittest.skipUnless(
    _supports_monarch_spmd(),
    "Monarch SPMD helpers are unavailable in this environment",
)
class SPMDTest(unittest.TestCase):
    def _assert_noop_results(self, results: list[dict[str, Any]]) -> None:
        self.assertCountEqual([result["rank"] for result in results], [0, 1])
        for result in results:
            self.assertTrue(result["scenario"]["completed"])
            self.assertEqual(
                result["scenario"]["configured"],
                result["rank"] == 0,
            )

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
