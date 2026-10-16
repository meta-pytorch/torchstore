# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""What ``import torchstore`` loads with Monarch absent, blocked, stubbed or installed.

Other cases run the code paths that import Monarch or cloudpickle only on first use.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import socket
import subprocess
import sys
import typing
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

pytestmark = pytest.mark.timeout(240)

# Each child process imports the tree that pytest imported.
_ROOT = str(Path(importlib.util.find_spec("torchstore").origin).parents[1])
_FRAME_LENGTH_DEFAULT = 910737418240
# Children import like a fresh process: without the frame length that this process's
# own TorchStore import exported, and with TorchStore's default RDMA setting.
_UNSET_IN_CHILDREN = ("HYPERACTOR_CODEC_MAX_FRAME_LENGTH", "TORCHSTORE_RDMA_ENABLED")

# Later Monarch-free modules join this list.
_MONARCH_FREE = [
    "torchstore",
    "torchstore.directory",
    "torchstore.logging",
    "torchstore.strategy",
    "torchstore.utils",
    "torchstore.transport",
    "torchstore.routing.client",
    "torchstore.routing.control_plane",
    "torchstore.routing.layout_registry",
    "torchstore.direct_weight_sync",
    "torchstore.transport.monarch_rdma",
    "torchstore.storage",
]
_ACTOR_MODULES = [
    "torchstore.controller",
    "torchstore.storage_volume",
    "torchstore.routing.coordinator",
]
_ENTRY_POINTS = [
    "initialize",
    "initialize_routing",
    "client",
    "client_routing",
    "shutdown",
    "initialize_spmd",
    "LocalRankStrategy.get_volume_id",
    "spawn_actors",
]

_BLOCKED_CHILD = """
import asyncio, importlib, json, os, sys

# Imports of either now raise. TorchStore gets cloudpickle only through torchmonarch.
sys.modules["monarch"] = sys.modules["cloudpickle"] = None
root, out, modules = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
sys.path.insert(0, root)


def error(call):
    try:
        call()
    except Exception as e:
        return f"{type(e).__name__}: {e}"
    return None


report = {"imports": {name: error(lambda: importlib.import_module(name)) for name in modules}}
import torchstore as ts

report["frame_env"] = os.environ.get("HYPERACTOR_CODEC_MAX_FRAME_LENGTH")
# Before any lazy name resolves, so only __dir__ can list spmd.
report["missing_from_dir"] = sorted(set(ts.__all__) - set(dir(ts)))
report["spmd_loaded"] = "torchstore.spmd" in sys.modules
import torchstore.utils
from torchstore.transport import monarch_rdma

report["unresolved"] = [name for name in ts.__all__ if error(lambda: getattr(ts, name))]
report["unknown"] = error(lambda: ts.no_such_name)
report["entry_points"] = {
    name: error(call)
    for name, call in {
        "initialize": lambda: asyncio.run(ts.initialize()),
        "initialize_routing": lambda: asyncio.run(
            ts.initialize(
                mesh=object(),
                strategy=ts.TorchStoreStrategy(),
                client_type=ts.ClientType.ROUTING,
            )
        ),
        "client": lambda: asyncio.run(ts.client()),
        "client_routing": lambda: asyncio.run(ts.client(role=ts.RankRole.PUBLISHER)),
        "shutdown": lambda: asyncio.run(ts.shutdown()),
        "initialize_spmd": lambda: asyncio.run(ts.initialize_spmd(strategy=ts.HostStrategy())),
        "LocalRankStrategy.get_volume_id": ts.LocalRankStrategy.get_volume_id,
        "spawn_actors": lambda: asyncio.run(torchstore.utils.spawn_actors(1, object, "probe")),
    }.items()
}
report["rdma_available"] = monarch_rdma.monarch_rdma_available()
report["rdma_buffer"] = error(lambda: monarch_rdma.RDMABuffer(None))
report["rdma_action"] = error(monarch_rdma.RDMAAction)
with open(out, "w") as f:
    json.dump(report, f)
"""

_INSTALLED_CHILD = """
import importlib, json, os, sys

root, out, target = sys.argv[1:]
sys.path.insert(0, root)
importlib.import_module(target)
report = {
    "loaded": [name for name in ("monarch.actor", "monarch.rdma", "monarch.job") if name in sys.modules],
    "modules": sorted(name for name in sys.modules if name.split(".")[0] == "monarch"),
    "env": os.environ.get("HYPERACTOR_CODEC_MAX_FRAME_LENGTH"),
}
from monarch.config import get_global_config

report["frame_length"] = get_global_config().get("codec_max_frame_length")
with open(out, "w") as f:
    json.dump(report, f)
"""

_TRANSFER_CHILD = """
import importlib, json, sys
from types import SimpleNamespace

root, out, preload = sys.argv[1:]
sys.path.insert(0, root)
if preload:
    importlib.import_module(preload)
from torchstore.transport import create_transport_buffer, TransportType
from torchstore.transport.monarch_rdma import monarch_rdma_transport_available

# Every new buffer logs whether Monarch RDMA is available, a Gloo one included.
ref = SimpleNamespace(
    default_transport_type=TransportType.Gloo,
    volume_id="v",
    volume_hostname="no-such-host",
    volume=None,
    transport_context=None,
)
create_transport_buffer(ref)
report = {"rdma_loaded": "monarch.rdma" in sys.modules}
report["available"] = monarch_rdma_transport_available()
with open(out, "w") as f:
    json.dump(report, f)
"""

_STUB_CHILD = """
import json, sys, types

root, out = sys.argv[1:]
sys.path.insert(0, root)
stub = sys.modules["monarch"] = types.ModuleType("monarch")  # Spec-less, like a test stub.
import torchstore

with open(out, "w") as f:
    json.dump({"kept_stub": sys.modules["monarch"] is stub}, f)
"""

_ABSENT_CHILD = """
import importlib.util, json, os, sys

root, out = sys.argv[1:]
sys.path.insert(0, root)


class Hidden:
    # Finds neither package, as when torchmonarch is not installed.
    def __init__(self, finder):
        self.finder = finder

    def find_spec(self, name, path=None, target=None):
        if name.partition(".")[0] in ("monarch", "cloudpickle"):
            return None
        return self.finder.find_spec(name, path, target)

    def __getattr__(self, attr):
        return getattr(self.finder, attr)


sys.meta_path[:] = [Hidden(finder) for finder in sys.meta_path]
assert importlib.util.find_spec("monarch") is None
assert importlib.util.find_spec("cloudpickle") is None
import torchstore as ts

report = {
    "unresolved": [name for name in ts.__all__ if not hasattr(ts, name)],
    "frame_env": os.environ.get("HYPERACTOR_CODEC_MAX_FRAME_LENGTH"),
}
with open(out, "w") as f:
    json.dump(report, f)
"""

_monarch_installed = pytest.mark.skipif(
    importlib.util.find_spec("monarch") is None, reason="Monarch is not installed"
)


def _run_child(
    script: str, out: Path, *args: str, extra_env: dict[str, str] | None = None
) -> dict[str, Any]:
    env = {k: v for k, v in os.environ.items() if k not in _UNSET_IN_CHILDREN}
    env.update(extra_env or {})
    child = subprocess.run(
        [sys.executable, "-c", script, _ROOT, str(out), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    return json.loads(out.read_text())


@pytest.fixture(scope="module")
def blocked(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    out = tmp_path_factory.mktemp("blocked") / "report.json"
    return _run_child(_BLOCKED_CHILD, out, json.dumps(_MONARCH_FREE + _ACTOR_MODULES))


@pytest.fixture(scope="module")
def installed(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict[str, Any]]:
    root = tmp_path_factory.mktemp("installed")
    return {
        target: _run_child(_INSTALLED_CHILD, root / f"{target}.json", target)
        for target in ("torchstore", "monarch")
    }


@pytest.mark.parametrize("module", _MONARCH_FREE)
def test_module_imports_with_monarch_blocked(blocked, module) -> None:
    """Import each Monarch-free module while every Monarch import fails."""
    assert blocked["imports"][module] is None


@pytest.mark.parametrize("module", _ACTOR_MODULES)
def test_actor_module_needs_monarch(blocked, module) -> None:
    """Fail to import the Actor modules, which import Monarch when they load."""
    assert blocked["imports"][module].startswith("ModuleNotFoundError: ")


def test_public_names_resolve_with_monarch_blocked(blocked) -> None:
    """List every ``__all__`` name while spmd is unloaded, then resolve each one."""
    assert blocked["spmd_loaded"] is False
    assert blocked["missing_from_dir"] == []
    assert blocked["unresolved"] == []
    assert blocked["unknown"] == (
        "AttributeError: module 'torchstore' has no attribute 'no_such_name'"
    )


def test_frame_length_default_is_exported_with_monarch_blocked(blocked) -> None:
    """Export the frame-length default, as a fresh import without Monarch does."""
    assert blocked["frame_env"] == str(_FRAME_LENGTH_DEFAULT)


@pytest.mark.parametrize("entry_point", _ENTRY_POINTS)
def test_entry_point_needs_monarch_only_when_called(blocked, entry_point) -> None:
    """Raise ``ModuleNotFoundError`` once a path that needs Monarch runs."""
    error = blocked["entry_points"][entry_point]
    assert error.startswith("ModuleNotFoundError: ") and "monarch" in error, error


def test_rdma_reports_unavailable_with_monarch_blocked(blocked) -> None:
    """Report Monarch RDMA as unavailable and refuse to build its buffers and actions."""
    assert blocked["rdma_available"] is False
    assert blocked["rdma_buffer"].startswith("NotImplementedError: RDMABuffer")
    assert blocked["rdma_action"].startswith("NotImplementedError: RDMAAction")


def test_import_succeeds_with_monarch_not_installed(tmp_path: Path) -> None:
    """Import with Monarch hidden from every finder, so the guard's ``find_spec`` returns None."""
    report = _run_child(_ABSENT_CHILD, tmp_path / "report.json")
    assert report["unresolved"] == []
    assert report["frame_env"] == str(_FRAME_LENGTH_DEFAULT)


def test_import_tolerates_a_stubbed_monarch(tmp_path: Path) -> None:
    """Import with a spec-less ``monarch`` stub in ``sys.modules``, as test harnesses install."""
    assert _run_child(_STUB_CHILD, tmp_path / "report.json")["kept_stub"] is True


def test_rdma_wrappers_forward_to_the_loaded_api(monkeypatch) -> None:
    """Build buffers and actions through the loaded API, and report its probe."""
    from torchstore.transport import monarch_rdma

    api = monarch_rdma._RdmaApi(
        action=lambda *args, **kwargs: ("action", args, kwargs),
        buffer=lambda *args, **kwargs: ("buffer", args, kwargs),
        available=lambda: True,
    )
    monkeypatch.setattr(monarch_rdma, "_rdma_api", lambda: api)
    assert monarch_rdma.RDMABuffer(1, size=2) == ("buffer", (1,), {"size": 2})
    assert monarch_rdma.RDMAAction(3) == ("action", (3,), {})
    assert monarch_rdma.monarch_rdma_available() is True
    monkeypatch.setattr(
        monarch_rdma, "_rdma_api", lambda: api._replace(available=lambda: False)
    )
    assert monarch_rdma.monarch_rdma_available() is False


def test_shared_types_resolve_their_type_hints() -> None:
    """Resolve the hints of the types the controller, client and routing directory share."""
    from torchstore.directory import ObjectType, StorageInfo
    from torchstore.transport.types import Request, TensorSlice

    assert typing.get_type_hints(StorageInfo) == {
        "object_type": ObjectType,
        "tensor_slices": set[TensorSlice | None],
    }
    assert typing.get_type_hints(ObjectType.from_request) == {
        "request": Request,
        "return": ObjectType,
    }


@_monarch_installed
def test_import_loads_no_monarch_actor_rdma_or_job_package(installed) -> None:
    """Load none of Monarch's actor, RDMA or job packages, nor any module ``import monarch`` does not."""
    torchstore, control = installed["torchstore"], installed["monarch"]
    assert torchstore["loaded"] == []
    assert sorted(set(torchstore["modules"]) - set(control["modules"])) == []


@_monarch_installed
def test_rdma_api_loads_from_monarch_on_first_use() -> None:
    """Resolve the RDMA buffer, action and probe to Monarch's own."""
    rdma = pytest.importorskip("monarch.rdma")
    from torchstore.transport import monarch_rdma

    api = monarch_rdma._rdma_api()
    assert api is not None
    assert (api.action, api.buffer) == (rdma.RDMAAction, rdma.RDMABuffer)
    assert api.available is rdma.is_ibverbs_available


@_monarch_installed
@pytest.mark.parametrize("rdma", ["default", "off"])
@pytest.mark.parametrize(
    "preload", ["monarch.actor", ""], ids=["actor_api", "no_actor_api"]
)
def test_transfer_probes_monarch_rdma_only_with_its_actor_api(
    tmp_path: Path, preload: str, rdma: str
) -> None:
    """Load ``monarch.rdma`` for a transfer only with RDMA on and Monarch's actor API loaded.

    Otherwise the transport never reports Monarch RDMA available.
    """
    extra_env = {"TORCHSTORE_RDMA_ENABLED": "0"} if rdma == "off" else {}
    report = _run_child(
        _TRANSFER_CHILD, tmp_path / "report.json", preload, extra_env=extra_env
    )
    assert report["rdma_loaded"] is (rdma == "default" and bool(preload))
    if rdma == "off" or not preload:
        assert report["available"] is False


@_monarch_installed
def test_frame_length_default_is_set_after_monarch_loads(installed) -> None:
    """Export the frame-length default only after Monarch has read its settings."""
    torchstore, control = installed["torchstore"], installed["monarch"]
    assert control["env"] is None
    assert isinstance(control["frame_length"], int)
    assert control["frame_length"] != _FRAME_LENGTH_DEFAULT
    assert torchstore["frame_length"] == control["frame_length"]
    assert torchstore["env"] == str(_FRAME_LENGTH_DEFAULT)


@_monarch_installed
def test_import_leaves_monarch_unloaded_with_the_frame_length_set(
    tmp_path: Path,
) -> None:
    """Load no Monarch module when the frame length is inherited, as TorchStore's children do."""
    frame_length = {"HYPERACTOR_CODEC_MAX_FRAME_LENGTH": str(_FRAME_LENGTH_DEFAULT)}
    report = _run_child(
        _INSTALLED_CHILD, tmp_path / "report.json", "torchstore", extra_env=frame_length
    )
    assert report["modules"] == []
    assert report["frame_length"] == _FRAME_LENGTH_DEFAULT


@_monarch_installed
@pytest.mark.parametrize("rank", [0, 1])
def test_spmd_initialize_imports_what_each_rank_uses(monkeypatch, rank: int) -> None:
    """Spawn and publish the controller on rank 0, and unpickle it on rank 1."""
    import cloudpickle
    import torchstore as ts
    import torchstore.api as store_api
    from torchstore.controller import Controller

    # initialize imports Monarch's spawner when called, so patching Monarch's module reaches it.
    assert not hasattr(ts.spmd, "get_or_spawn_controller")
    store_name = f"spmd_{uuid.uuid4().hex[:8]}"
    key = f"torchstore/spmd/{store_name}/controller"
    env = ts.spmd.SPMDEnv(
        rank=rank,
        local_rank=rank,
        world_size=2,
        local_world_size=2,
        master_addr="127.0.0.1",
        master_port=0,
    )
    store = MagicMock()
    store.get.return_value = cloudpickle.dumps("controller")
    spawn = AsyncMock(return_value="controller")
    # Restored after the test because initialize would otherwise export it for the session.
    monkeypatch.setenv("TORCHSTORE_HOSTNAME", socket.gethostname())
    monkeypatch.setattr("torchstore.spmd.TCPStore", MagicMock(return_value=store))
    monkeypatch.setattr(
        "torchstore.spmd.StoreJob.from_store",
        MagicMock(return_value=MagicMock() if rank == 0 else None),
    )
    monkeypatch.setattr("torchstore.api.initialize", AsyncMock())
    monkeypatch.setattr("monarch.actor.get_or_spawn_controller", spawn)
    try:
        asyncio.run(
            ts.initialize_spmd(
                strategy=ts.HostStrategy(), store_name=store_name, env=env
            )
        )
        controller = store_api._spmd_state_map[store_name].controller
    finally:
        store_api._spmd_state_map.pop(store_name, None)
    assert controller == "controller"
    if rank == 0:
        spawn.assert_awaited_once_with(store_name, Controller)
        store.set.assert_called_once_with(key, cloudpickle.dumps("controller"))
    else:
        spawn.assert_not_awaited()
        store.get.assert_called_once_with(key)
