# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""What ``import torchstore`` loads with Monarch blocked or installed, and which calls need it."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.timeout(240)

# Each child process imports the tree that pytest imported.
_ROOT = str(Path(importlib.util.find_spec("torchstore").origin).parents[1])
_FRAME_LENGTH_DEFAULT = 910737418240
# A launcher's own frame length, which importing TorchStore keeps.
_PRESET_FRAME_LENGTH = "1234567890"
# Children import like a fresh process, without the frame length that this process's own
# TorchStore import exported.
_UNSET_IN_CHILDREN = ("HYPERACTOR_CODEC_MAX_FRAME_LENGTH",)
# Modules that fail to load without Monarch. Every other module, the Monarch RDMA ones
# included, must load without it.
_NEEDS_MONARCH = {
    "torchstore.controller",
    "torchstore.monarch.backend",
    "torchstore.monarch.controller",
    "torchstore.monarch.coordinator",
    "torchstore.monarch.spmd",
    "torchstore.monarch.storage_volume",
    "torchstore.routing.coordinator",
    "torchstore.storage_volume",
}
# How a blocked import of one of the named packages fails, directly or through a submodule.
_BLOCKED = "import of ({0}) halted|No module named '({0})[.']"
_ENTRY_POINTS = [
    "initialize",
    "initialize_routing",
    "client",
    "shutdown",
    "initialize_spmd",
    "LocalRankStrategy.get_volume_id",
    "spawn_actors",
    "DirectWeightSyncSource.register",
]

_BLOCKED_CHILD = """
import asyncio, importlib, json, os, pkgutil, sys

# Imports of either now raise. TorchStore gets cloudpickle only through torchmonarch.
sys.modules["monarch"] = sys.modules["cloudpickle"] = None
root, out = sys.argv[1:]
sys.path.insert(0, root)


def error(call):
    try:
        call()
    except Exception as e:
        return f"{type(e).__name__}: {e}"
    return None


import torch
import torchstore as ts
import torchstore.utils
from torchstore.direct_weight_sync import DirectWeightSyncSource

report = {"frame_env": os.environ.get("HYPERACTOR_CODEC_MAX_FRAME_LENGTH")}
report["unresolved"] = [name for name in ts.__all__ if error(lambda: getattr(ts, name))]
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
        "shutdown": lambda: asyncio.run(ts.shutdown()),
        "initialize_spmd": lambda: asyncio.run(ts.initialize_spmd(strategy=ts.HostStrategy())),
        "LocalRankStrategy.get_volume_id": ts.LocalRankStrategy.get_volume_id,
        "spawn_actors": lambda: asyncio.run(torchstore.utils.spawn_actors(1, object, "probe")),
        "DirectWeightSyncSource.register": lambda: DirectWeightSyncSource().register(
            {}, rank=0
        ),
    }.items()
}
# Walked last, so the entry points above run with only what TorchStore itself loaded.
report["imports"] = {
    module.name: error(lambda: importlib.import_module(module.name))
    for module in pkgutil.walk_packages(ts.__path__, "torchstore.")
}
with open(out, "w") as f:
    json.dump(report, f)
"""

_INSTALLED_CHILD = """
import importlib, json, os, sys

root, out, targets = sys.argv[1:]
sys.path.insert(0, root)
for target in targets.split(","):
    importlib.import_module(target)
report = {
    "modules": sorted(name for name in sys.modules if name.split(".")[0] == "monarch"),
    "env": os.environ.get("HYPERACTOR_CODEC_MAX_FRAME_LENGTH"),
}
from monarch.config import get_global_config

report["frame_length"] = get_global_config().get("codec_max_frame_length")
with open(out, "w") as f:
    json.dump(report, f)
"""

_ATTRIBUTE_CHILD = """
import asyncio, json, sys

root, out = sys.argv[1:]
sys.path.insert(0, root)
import torchstore as ts
from monarch.actor import this_host


def old_paths():
    found = []
    for path in ("controller", "routing.coordinator", "storage_volume", "transport.monarch_rdma"):
        owner = ts
        for name in path.split("."):
            owner = getattr(owner, name, None)
        if owner is not None:
            found.append(path)
    return found


async def main():
    report = {"after_import": old_paths()}
    # As callers do: spawn the volumes' procs, initialize, then read an old path.
    mesh = this_host().spawn_procs(per_host={"procs": 1})
    await ts.initialize(mesh=mesh, strategy=ts.LocalRankStrategy())
    try:
        report["after_initialize"] = old_paths()
    finally:
        await ts.shutdown()
    return report


report = asyncio.run(main())
with open(out, "w") as f:
    json.dump(report, f)
"""

_FORK_CHILD = """
import asyncio, json, os, signal, sys, traceback

root, out = sys.argv[1:]
sys.path.insert(0, root)
import torch
import torchstore as ts


async def use_a_store():
    await ts.initialize(num_storage_volumes=1, store_name="forked")
    try:
        await ts.put("key", torch.arange(4), store_name="forked")
        return (await ts.get("key", store_name="forked")).tolist()
    finally:
        await ts.shutdown("forked")


# The child forks before any call has loaded Monarch, then uses TorchStore.
pid = os.fork()
if pid == 0:
    try:
        got = asyncio.run(use_a_store())
    except BaseException:
        traceback.print_exc()
        sys.exit(1)
    with open(out, "w") as f:
        json.dump({"got": got}, f)
    sys.exit(0)


def hung(signum, frame):
    os.kill(pid, signal.SIGKILL)
    sys.exit("the forked child hung")


signal.signal(signal.SIGALRM, hung)
signal.alarm(150)
status = os.waitstatus_to_exitcode(os.waitpid(pid, 0)[1])
signal.alarm(0)
sys.exit(status and f"the forked child failed with status {status}")
"""

_VOLUME_CHILD = """
import json, sys

root, out = sys.argv[1:]
sys.path.insert(0, root)
import torchstore.monarch.storage_volume
import torchstore.transport as transport

with open(out, "w") as f:
    json.dump({"rdma": transport.TransportType.MonarchRDMA in transport._backend_transports}, f)
"""

_monarch_installed = pytest.mark.skipif(
    importlib.util.find_spec("monarch") is None, reason="Monarch is not installed"
)


def _run_child(
    script: str, out: Path, *args: str, extra_env: dict[str, str] | None = None
) -> dict[str, Any]:
    env = {k: v for k, v in os.environ.items() if k not in _UNSET_IN_CHILDREN}
    env.update(extra_env or {})
    try:
        child = subprocess.run(
            [sys.executable, "-c", script, _ROOT, str(out), *args],
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        # Its partial output arrives as bytes, or None, even with text=True.
        output = b"".join(stream or b"" for stream in (error.stdout, error.stderr))
        pytest.fail(f"child timed out:\n{output.decode(errors='replace')}")
    assert child.returncode == 0, child.stdout + child.stderr
    return json.loads(out.read_text())


@pytest.fixture(scope="module")
def blocked(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    out = tmp_path_factory.mktemp("blocked") / "report.json"
    return _run_child(_BLOCKED_CHILD, out)


@pytest.fixture(scope="module")
def installed(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict[str, Any]]:
    """One child per import order, keyed by its comma-joined modules, and one preset."""
    root = tmp_path_factory.mktemp("installed")
    reports = {
        targets: _run_child(_INSTALLED_CHILD, root / f"{targets}.json", targets)
        for targets in ("torchstore", "monarch", "monarch,torchstore")
    }
    preset = {"HYPERACTOR_CODEC_MAX_FRAME_LENGTH": _PRESET_FRAME_LENGTH}
    reports["preset"] = _run_child(
        _INSTALLED_CHILD, root / "preset.json", "torchstore", extra_env=preset
    )
    return reports


def test_every_other_module_imports_with_monarch_blocked(blocked) -> None:
    """Import every module the package walk finds, but the Monarch ones, while Monarch fails."""
    failed = {
        module: error
        for module, error in blocked["imports"].items()
        if error is not None and module not in _NEEDS_MONARCH
    }
    assert failed == {}
    assert blocked["unresolved"] == []
    assert blocked["frame_env"] == str(_FRAME_LENGTH_DEFAULT)


def test_monarch_modules_fail_while_monarch_is_blocked(blocked) -> None:
    """Fail to import each module that needs Monarch, naming Monarch.

    The SPMD bootstrap may name cloudpickle first, which the child blocks too.
    """
    for module in sorted(_NEEDS_MONARCH):
        error = blocked["imports"][module] or ""
        named = "monarch|cloudpickle" if module.endswith(".spmd") else "monarch"
        assert error.startswith("ModuleNotFoundError: "), (module, error)
        assert re.search(_BLOCKED.format(named), error), (module, error)


@pytest.mark.parametrize("entry_point", _ENTRY_POINTS)
def test_entry_point_needs_monarch_only_when_called(blocked, entry_point) -> None:
    """Raise ``ModuleNotFoundError`` naming Monarch once a call needs the Monarch backend."""
    error = blocked["entry_points"][entry_point]
    assert error.startswith("ModuleNotFoundError: "), error
    assert re.search(_BLOCKED.format("monarch"), error), error


@_monarch_installed
def test_import_loads_no_monarch_module(installed) -> None:
    """Load no Monarch module on ``import torchstore``, where ``import monarch`` loads many."""
    assert installed["monarch"]["modules"]
    assert installed["torchstore"]["modules"] == []
    assert installed["preset"]["modules"] == []


@_monarch_installed
def test_monarch_loaded_after_torchstore_reads_the_exported_frame_length(
    installed,
) -> None:
    """Export the frame-length default before Monarch loads, so Monarch reads it too."""
    control, after = installed["monarch"], installed["torchstore"]
    assert control["env"] is None
    assert isinstance(control["frame_length"], int)
    assert control["frame_length"] != _FRAME_LENGTH_DEFAULT
    assert after["env"] == str(_FRAME_LENGTH_DEFAULT)
    assert after["frame_length"] == _FRAME_LENGTH_DEFAULT


@_monarch_installed
def test_monarch_loaded_before_torchstore_keeps_its_own_frame_length(installed) -> None:
    """Export the default for later processes while a Monarch loaded first keeps its own."""
    first = installed["monarch,torchstore"]
    assert first["env"] == str(_FRAME_LENGTH_DEFAULT)
    assert first["frame_length"] == installed["monarch"]["frame_length"]


@_monarch_installed
def test_a_preset_frame_length_is_kept(installed) -> None:
    """Keep a frame length set before the import, for TorchStore and Monarch alike."""
    preset = installed["preset"]
    assert preset["env"] == _PRESET_FRAME_LENGTH
    assert preset["frame_length"] == int(_PRESET_FRAME_LENGTH)


@_monarch_installed
def test_old_paths_are_attributes_once_initialize_loads_the_backend(
    tmp_path: Path,
) -> None:
    """Read old paths as attributes after ``ts.initialize`` on Monarch, not after the import."""
    report = _run_child(_ATTRIBUTE_CHILD, tmp_path / "report.json")
    assert report["after_import"] == []
    assert report["after_initialize"] == [
        "controller",
        "routing.coordinator",
        "storage_volume",
        "transport.monarch_rdma",
    ]


@_monarch_installed
def test_loading_a_storage_volume_registers_monarch_rdma(tmp_path: Path) -> None:
    """Offer Monarch RDMA wherever a volume handle loads, as in a worker handed a controller."""
    assert _run_child(_VOLUME_CHILD, tmp_path / "report.json") == {"rdma": True}


@_monarch_installed
def test_a_child_forked_after_the_import_can_use_torchstore(tmp_path: Path) -> None:
    """Initialize, put and get in a child forked after ``import torchstore``.

    Main loaded Monarch at the import, and such a child's first call hung.
    """
    assert _run_child(_FORK_CHILD, tmp_path / "report.json") == {"got": [0, 1, 2, 3]}
