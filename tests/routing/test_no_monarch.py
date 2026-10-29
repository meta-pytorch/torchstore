# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""What ``import torchstore`` loads with Monarch blocked or installed.

CI's Monarch-free job also runs this file without torchmonarch. Other cases run the code
paths that import Monarch or cloudpickle only on first use.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import os
import re
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
# Children import like a fresh process: without the frame length this process's own
# TorchStore import exported, and with TorchStore's default log level and RDMA setting.
_UNSET_IN_CHILDREN = (
    "HYPERACTOR_CODEC_MAX_FRAME_LENGTH",
    "TORCHSTORE_LOG_LEVEL",
    "TORCHSTORE_RDMA_ENABLED",
)
# Level, logger and message of the warning that TorchStore's first Monarch path logs.
_FRAME_WARNING = [
    "WARNING",
    "torchstore",
    "Warning: setting HYPERACTOR_CODEC_MAX_FRAME_LENGTH since this needs to be set"
    " to enable large RPC calls via Monarch",
]
# Monarch's actor log filter prefixes a message once for each handler it passes.
_ACTOR_PREFIXES = re.compile(r"^(\[actor=[^\]]*\] )+")
_IMPORT_TARGETS = [
    "torchstore",
    "torchstore.transport.monarch_rdma",
    "torchstore.direct_weight_sync",
]
# A launcher's own frame length, which importing TorchStore keeps.
_PRESET_FRAME_LENGTH = "1234567890"

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
    "torchstore.routing._gc",
    "torchstore.direct_weight_sync",
    "torchstore.transport.monarch_rdma",
    "torchstore.storage",
    "torchstore.volume_host",
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

# Starts each child that reports the frame-length warnings any logger emits.
_RECORD_FRAME_WARNINGS = """
import logging

frame_warnings = []


class FrameWarnings(logging.Handler):
    def emit(self, record):
        if "HYPERACTOR_CODEC_MAX_FRAME_LENGTH" in record.getMessage():
            frame_warnings.append([record.levelname, record.name, record.getMessage()])


logging.getLogger().addHandler(FrameWarnings())
"""

_BLOCKED_CHILD = (
    _RECORD_FRAME_WARNINGS
    + """
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
import torchstore.api as store_api
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
report["loaders"] = {
    "current_rank": error(store_api.current_rank),
    "get_or_spawn_controller": error(lambda: store_api.get_or_spawn_controller("probe", object)),
}
report["frame_warnings"] = frame_warnings
with open(out, "w") as f:
    json.dump(report, f)
"""
)

_INSTALLED_CHILD = (
    _RECORD_FRAME_WARNINGS
    + """
import importlib, json, os, sys

root, out, targets = sys.argv[1:]
sys.path.insert(0, root)
for target in targets.split(","):
    importlib.import_module(target)
report = {
    "modules": sorted(name for name in sys.modules if name.split(".")[0] == "monarch"),
    "env": os.environ.get("HYPERACTOR_CODEC_MAX_FRAME_LENGTH"),
    "frame_warnings": list(frame_warnings),
    "stdout_handler": any(
        getattr(handler, "stream", None) is sys.stdout
        for handler in logging.getLogger().handlers
    ),
}
from monarch.config import get_global_config

report["frame_length"] = get_global_config().get("codec_max_frame_length")
with open(out, "w") as f:
    json.dump(report, f)
"""
)

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

_FIRST_PATH_CHILD = (
    _RECORD_FRAME_WARNINGS
    + """
import json, sys

root, out = sys.argv[1:]
sys.path.insert(0, root)
import torchstore.api as store_api

report = {"at_import": list(frame_warnings)}
for _ in range(2):
    store_api.current_rank()
report["after_monarch_paths"] = list(frame_warnings)
with open(out, "w") as f:
    json.dump(report, f)
"""
)


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


def _frame_warnings(recorded: list[list[str]]) -> list[list[str]]:
    """Recorded frame-length warnings, without the actor prefixes Monarch adds."""
    return [
        [level, name, _ACTOR_PREFIXES.sub("", message)]
        for level, name, message in recorded
    ]


def _logged_frame_warnings(caplog: pytest.LogCaptureFixture) -> list[list[str]]:
    """The frame-length warnings among this process's captured log records."""
    return _frame_warnings(
        [
            [record.levelname, record.name, record.getMessage()]
            for record in caplog.records
            if "HYPERACTOR_CODEC_MAX_FRAME_LENGTH" in record.getMessage()
        ]
    )


@pytest.fixture(scope="module")
def blocked(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    out = tmp_path_factory.mktemp("blocked") / "report.json"
    return _run_child(_BLOCKED_CHILD, out, json.dumps(_MONARCH_FREE + _ACTOR_MODULES))


@pytest.fixture(scope="module")
def installed(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict[str, Any]]:
    """One child per import order, keyed by its comma-joined modules, and one preset."""
    root = tmp_path_factory.mktemp("installed")
    reports = {
        target: _run_child(_INSTALLED_CHILD, root / f"{target}.json", target)
        for target in (*_IMPORT_TARGETS, "monarch", "monarch,torchstore")
    }
    preset = {"HYPERACTOR_CODEC_MAX_FRAME_LENGTH": _PRESET_FRAME_LENGTH}
    reports["preset"] = _run_child(
        _INSTALLED_CHILD, root / "preset.json", "torchstore", extra_env=preset
    )
    return reports


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


def test_package_logger_stays_public() -> None:
    """Keep ``torchstore.logger``, the package's own logger."""
    import torchstore as ts

    assert ts.logger is logging.getLogger("torchstore")


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


def test_frame_length_warning_waits_for_monarch(blocked) -> None:
    """Log no frame-length warning while every path that needs Monarch fails to load it."""
    for loader, error in blocked["loaders"].items():
        assert error.startswith("ModuleNotFoundError: ") and "monarch" in error, loader
    assert blocked["frame_warnings"] == []


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


def test_unset_transport_takes_monarch_rdma_only_when_its_probe_says_so(
    monkeypatch,
) -> None:
    """Resolve an unset transport to Monarch RDMA once Monarch reports RDMA usable."""
    import types

    from torchstore import transport
    from torchstore.transport import monarch_rdma, TransportType

    actor_api = sys.modules.get("monarch.actor", types.ModuleType("monarch.actor"))
    monkeypatch.setitem(sys.modules, "monarch.actor", actor_api)
    monkeypatch.setenv("TORCHSTORE_RDMA_ENABLED", "1")
    monkeypatch.setattr(transport, "SHM_ENABLED", False)
    for name in ("nixl", "torchcomms_uniflow", "torchcomms_rdma"):
        monkeypatch.setattr(transport, f"{name}_available", lambda: False)
    monkeypatch.setattr(transport, "gloo_available", lambda: True)
    for available, expected in (
        (True, TransportType.MonarchRDMA),
        (False, TransportType.Gloo),
    ):
        api = monarch_rdma._RdmaApi(None, None, lambda available=available: available)
        monkeypatch.setattr(monarch_rdma, "_rdma_api", lambda api=api: api)
        assert transport.get_available_transport(object()) is expected


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
@pytest.mark.parametrize("target", _IMPORT_TARGETS)
def test_import_loads_no_monarch_module(installed, target: str) -> None:
    """Load no Monarch module and log no frame-length warning, where ``import monarch`` loads many."""
    assert installed["monarch"]["modules"]
    assert installed[target]["modules"] == []
    assert installed[target]["frame_warnings"] == []


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
@pytest.mark.parametrize("target", _IMPORT_TARGETS)
def test_monarch_loaded_after_torchstore_reads_the_exported_frame_length(
    installed, target: str
) -> None:
    """Export the frame-length default before Monarch loads, so Monarch reads it too."""
    control = installed["monarch"]
    assert control["env"] is None
    assert isinstance(control["frame_length"], int)
    assert control["frame_length"] != _FRAME_LENGTH_DEFAULT
    assert installed[target]["env"] == str(_FRAME_LENGTH_DEFAULT)
    assert installed[target]["frame_length"] == _FRAME_LENGTH_DEFAULT
    assert installed[target]["stdout_handler"] is True


@_monarch_installed
def test_monarch_loaded_before_torchstore_keeps_its_own_frame_length(installed) -> None:
    """Export the default for later processes while a Monarch loaded first keeps its own."""
    first = installed["monarch,torchstore"]
    assert first["env"] == str(_FRAME_LENGTH_DEFAULT)
    assert first["frame_length"] == installed["monarch"]["frame_length"]
    assert first["frame_length"] != _FRAME_LENGTH_DEFAULT


@_monarch_installed
def test_a_preset_frame_length_is_kept_without_logging_setup(installed) -> None:
    """Keep a frame length set before the import, and leave logging and warnings alone."""
    preset = installed["preset"]
    assert preset["env"] == _PRESET_FRAME_LENGTH
    assert preset["frame_length"] == int(_PRESET_FRAME_LENGTH)
    assert preset["frame_warnings"] == []
    assert preset["stdout_handler"] is False


@_monarch_installed
@pytest.mark.parametrize("inherited", [False, True], ids=["exported", "inherited"])
def test_frame_length_warning_logs_once_on_the_first_monarch_path(
    tmp_path: Path, inherited: bool
) -> None:
    """Log the frame-length warning on the first Monarch path, once, and only if this import exported the default."""
    frame_length = {"HYPERACTOR_CODEC_MAX_FRAME_LENGTH": str(_FRAME_LENGTH_DEFAULT)}
    report = _run_child(
        _FIRST_PATH_CHILD,
        tmp_path / "report.json",
        extra_env=frame_length if inherited else None,
    )
    assert report["at_import"] == []
    expected = [] if inherited else [_FRAME_WARNING]
    assert _frame_warnings(report["after_monarch_paths"]) == expected


@_monarch_installed
@pytest.mark.parametrize(
    "path",
    ["current_rank", "get_or_spawn_controller", "initialize", "initialize_routing"],
)
def test_each_monarch_path_logs_the_frame_length_warning_once(
    monkeypatch, caplog, path: str
) -> None:
    """Log the warning from each of the API's Monarch paths, and only on its first run.

    A standard initialize logs it before its first spawn, which fails here.
    """
    import torchstore as ts
    import torchstore.api as store_api
    from torchstore.routing.control_plane import ControlPlane
    from torchstore.storage_volume import StorageVolume

    monkeypatch.setattr(store_api, "_frame_length_warning_pending", True)
    monkeypatch.setattr(store_api, "_control_plane_map", {})
    monkeypatch.setattr(store_api, "_local_clent_map", {})
    monkeypatch.setattr("monarch.actor.current_rank", MagicMock())
    monkeypatch.setattr("monarch.actor.get_or_spawn_controller", MagicMock())
    monkeypatch.setattr(StorageVolume, "spawn", AsyncMock())
    caplog.set_level(logging.WARNING, logger="torchstore")

    def initialize() -> None:
        monkeypatch.setattr(
            StorageVolume, "spawn", AsyncMock(side_effect=RuntimeError("spawn failed"))
        )
        with pytest.raises(RuntimeError, match="spawn failed"):
            asyncio.run(
                ts.initialize(num_storage_volumes=2, strategy=ts.HostStrategy())
            )

    run = {
        "current_rank": store_api.current_rank,
        "get_or_spawn_controller": lambda: store_api.get_or_spawn_controller(
            "probe", StorageVolume
        ),
        "initialize": initialize,
        # Under its own control plane, a store's initialize makes no Monarch lookup.
        "initialize_routing": lambda: asyncio.run(
            ts.initialize(
                mesh=MagicMock(size=MagicMock(return_value=1)),
                strategy=MagicMock(set_storage_volumes=AsyncMock()),
                client_type=ts.ClientType.ROUTING,
                store_name=f"frame_{uuid.uuid4().hex[:8]}",
                control_plane=MagicMock(spec=ControlPlane),
            )
        ),
    }[path]
    for _ in range(2):
        run()
        assert _logged_frame_warnings(caplog) == [_FRAME_WARNING]


@_monarch_installed
@pytest.mark.parametrize("rank", [0, 1])
def test_spmd_initialize_imports_what_each_rank_uses(
    monkeypatch, caplog, rank: int
) -> None:
    """Spawn and publish the controller on rank 0, and unpickle it on rank 1.

    Each rank logs the frame-length warning once.
    """
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
    monkeypatch.setattr(store_api, "_frame_length_warning_pending", True)
    caplog.set_level(logging.WARNING, logger="torchstore")
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
    assert _logged_frame_warnings(caplog) == [_FRAME_WARNING]
