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
import logging
import os
import re
import socket
import subprocess
import sys
import threading
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
    "torchstore.api",
    "Warning: setting HYPERACTOR_CODEC_MAX_FRAME_LENGTH since this needs to be set"
    " to enable large RPC calls via Monarch",
]
# Monarch's actor log filter prefixes a message once for each handler it passes.
_ACTOR_PREFIXES = re.compile(r"^(\[actor=[^\]]*\] )+")
# The package, and the two modules whose own imports already skipped Monarch.
_IMPORT_TARGETS = [
    "torchstore",
    "torchstore.transport.monarch_rdma",
    "torchstore.direct_weight_sync",
]

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

root, out, target = sys.argv[1:]
sys.path.insert(0, root)
importlib.import_module(target)
report = {
    "modules": sorted(name for name in sys.modules if name.split(".")[0] == "monarch"),
    "env": os.environ.get("HYPERACTOR_CODEC_MAX_FRAME_LENGTH"),
    "frame_warnings": list(frame_warnings),
}
from monarch.config import get_global_config

report["frame_length"] = get_global_config().get("codec_max_frame_length")
with open(out, "w") as f:
    json.dump(report, f)
"""
)

# TitanRL's order on TorchTitan main: Monarch's actor API loads before TorchStore.
_MONARCH_FIRST_CHILD = (
    _RECORD_FRAME_WARNINGS
    + """
import json, os, sys

root, out = sys.argv[1:]
sys.path.insert(0, root)
import monarch.actor
import torchstore.api as store_api
from monarch.config import get_global_config

report = {
    "env": os.environ.get("HYPERACTOR_CODEC_MAX_FRAME_LENGTH"),
    "frame_length": get_global_config().get("codec_max_frame_length"),
    "at_import": list(frame_warnings),
}
store_api.current_rank()
report["frame_warnings"] = list(frame_warnings)
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

# Forks after TorchStore exported the default and before any Monarch path ran.
_FORKED_CHILD = (
    _RECORD_FRAME_WARNINGS
    + """
import json, os, signal, sys

root, out = sys.argv[1:]
sys.path.insert(0, root)
import torchstore.api as store_api

pid = os.fork()
if pid == 0:
    # A fork with Monarch's threads running can hang the child: end it before the test times out.
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.alarm(60)
    store_api.current_rank()
    with open(out + ".child", "w") as f:
        json.dump(frame_warnings, f)
    os._exit(0)
assert os.waitpid(pid, 0)[1] == 0
store_api.current_rank()
with open(out + ".child") as f:
    report = {"child": json.load(f), "parent": list(frame_warnings)}
with open(out, "w") as f:
    json.dump(report, f)
"""
)

_ROUTED_CHILD = (
    _RECORD_FRAME_WARNINGS
    + """
import asyncio, json, sys, uuid

root, out = sys.argv[1:]
sys.path.insert(0, root)
import torch
import torchstore as ts

report = {
    "modules": sorted(name for name in sys.modules if name.split(".")[0] == "monarch"),
    "at_import": list(frame_warnings),
}
# Monarch loads only now, as in a process that imports it after TorchStore.
from monarch.actor import this_host
from monarch.config import get_global_config
from torchstore.transport import TransportType

report["before_torchstore_paths"] = list(frame_warnings)


async def main():
    store_name = f"frame-length-{uuid.uuid4().hex[:8]}"
    mesh = this_host().spawn_procs(per_host={"procs": 1})
    try:
        await ts.initialize(
            mesh=mesh,
            strategy=ts.TorchStoreStrategy(TransportType.MonarchRPC),
            client_type=ts.ClientType.ROUTING,
            store_name=store_name,
        )
        weights = {"weight": torch.arange(6, dtype=torch.float32).view(2, 3)}
        # This process publishes as the mesh's only rank, then reads back as a requester.
        await ts.client(store_name, role=ts.RankRole.PUBLISHER)
        await ts.put_state_dict(weights, "sd", store_name=store_name)
        ts.reset_client(store_name)
        await ts.client(store_name, role=ts.RankRole.REQUESTER, group=0)
        pulled = await ts.get_state_dict("sd", {"weight": torch.zeros(2, 3)}, store_name=store_name)
        report["routed"] = torch.equal(pulled["weight"], weights["weight"])
    finally:
        await mesh.stop()


asyncio.run(main())
report["frame_length"] = get_global_config().get("codec_max_frame_length")
report["after_routed_sync"] = list(frame_warnings)
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
    with subprocess.Popen(
        [sys.executable, "-c", script, _ROOT, str(out), *args],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as child:
        try:
            stdout, stderr = child.communicate(timeout=180)
        except subprocess.TimeoutExpired:
            child.kill()
            stdout, stderr = child.communicate()
            pytest.fail(f"child timed out:\n{stdout}{stderr}")
        except BaseException:
            # Popen's exit waits for the child, so pytest's own timeout must kill it.
            child.kill()
            raise
    assert child.returncode == 0, stdout + stderr
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
    root = tmp_path_factory.mktemp("installed")
    return {
        target: _run_child(_INSTALLED_CHILD, root / f"{target}.json", target)
        for target in (*_IMPORT_TARGETS, "monarch")
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


def test_frame_length_warning_waits_for_monarch(blocked) -> None:
    """Log no frame-length warning while every path that needs Monarch fails to load it."""
    for loader, error in blocked["loaders"].items():
        assert error.startswith("ModuleNotFoundError: ") and "monarch" in error, loader
    assert blocked["frame_warnings"] == []


def test_import_succeeds_with_monarch_not_installed(tmp_path: Path) -> None:
    """Import with Monarch hidden from every finder, as when torchmonarch is not installed."""
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


@_monarch_installed
def test_monarch_loaded_before_torchstore_keeps_its_limit_and_exports_the_default(
    tmp_path: Path,
) -> None:
    """Keep a Monarch loaded first on its own limit, but export the default and log once."""
    report = _run_child(_MONARCH_FIRST_CHILD, tmp_path / "report.json")
    assert report["env"] == str(_FRAME_LENGTH_DEFAULT)
    assert report["frame_length"] != _FRAME_LENGTH_DEFAULT
    assert report["at_import"] == []
    assert _frame_warnings(report["frame_warnings"]) == [_FRAME_WARNING]


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
def test_a_forked_child_never_logs_the_frame_length_warning(tmp_path: Path) -> None:
    """Log the warning in the importing process only, not in a child forked before its first Monarch path."""
    report = _run_child(_FORKED_CHILD, tmp_path / "report.json")
    assert report["child"] == []
    assert _frame_warnings(report["parent"]) == [_FRAME_WARNING]


@_monarch_installed
@pytest.mark.parametrize(
    "path", ["current_rank", "get_or_spawn_controller", "initialize_routing"]
)
def test_each_monarch_path_logs_the_frame_length_warning_once(
    monkeypatch, caplog, path: str
) -> None:
    """Log the warning from each of the API's Monarch paths, and only on its first run."""
    import torchstore as ts
    import torchstore.api as store_api
    from torchstore.routing.control_plane import ControlPlane
    from torchstore.storage_volume import StorageVolume

    # The pending flag on entering and on leaving each section under the selection lock.
    sections: list[list[bool]] = []

    class RecordingLock:
        def __enter__(self) -> None:
            sections.append([store_api._frame_length_warning_pending])

        def __exit__(self, *exc_info: object) -> None:
            sections[-1].append(store_api._frame_length_warning_pending)

    # What this process had logged when routing initialization spawned its volumes.
    at_spawn: list[list[list[str]]] = []

    def record_spawn(*args: Any, **kwargs: Any) -> None:
        at_spawn.append(_logged_frame_warnings(caplog))

    monkeypatch.setattr(store_api, "_frame_length_warning_pending", True)
    monkeypatch.setattr(store_api, "_selection_lock", RecordingLock())
    monkeypatch.setattr(store_api, "_control_plane_map", {})
    monkeypatch.setattr(store_api, "_local_clent_map", {})
    monkeypatch.setattr("monarch.actor.current_rank", MagicMock())
    monkeypatch.setattr("monarch.actor.get_or_spawn_controller", MagicMock())
    monkeypatch.setattr(StorageVolume, "spawn", AsyncMock(side_effect=record_spawn))
    caplog.set_level(logging.WARNING, logger=store_api.__name__)
    run = {
        "current_rank": store_api.current_rank,
        "get_or_spawn_controller": lambda: store_api.get_or_spawn_controller(
            "probe", StorageVolume
        ),
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
    run()
    assert _logged_frame_warnings(caplog) == [_FRAME_WARNING]
    # Cleared under the selection lock. The interleaving test pins the check.
    assert [True, False] in sections
    if path == "initialize_routing":
        # Logged before the spawn, so a first initialize that fails there still logs it.
        assert at_spawn == [[_FRAME_WARNING]]
    run()
    assert _logged_frame_warnings(caplog) == [_FRAME_WARNING]
    assert store_api._frame_length_warning_pending is False


@_monarch_installed
@pytest.mark.parametrize("point", ["entering", "leaving"])
@pytest.mark.parametrize("path", ["current_rank", "get_or_spawn_controller"])
def test_two_threads_on_first_monarch_paths_log_the_warning_once(
    monkeypatch, caplog, path: str, point: str
) -> None:
    """Log once when another thread runs the same path as this one enters or leaves the lock."""
    import torchstore.api as store_api
    from torchstore.storage_volume import StorageVolume

    real = threading.Lock()
    others: list[threading.Thread] = []
    errors: list[BaseException] = []
    run = {
        "current_rank": store_api.current_rank,
        "get_or_spawn_controller": lambda: store_api.get_or_spawn_controller(
            "probe", StorageVolume
        ),
    }[path]

    def run_and_record() -> None:
        try:
            run()
        except BaseException as error:
            errors.append(error)

    def run_other() -> None:
        other = threading.Thread(target=run_and_record, daemon=True)
        others.append(other)
        other.start()
        other.join(timeout=30)
        assert not other.is_alive(), "the other thread is still on the selection lock"

    class InterleavingLock:
        def __enter__(self) -> None:
            if point == "entering" and not others:
                run_other()
            real.acquire()

        def __exit__(self, *exc_info: object) -> None:
            real.release()
            if point == "leaving" and not others:
                run_other()

    monkeypatch.setattr(store_api, "_frame_length_warning_pending", True)
    monkeypatch.setattr(store_api, "_selection_lock", InterleavingLock())
    monkeypatch.setattr("monarch.actor.current_rank", MagicMock())
    monkeypatch.setattr("monarch.actor.get_or_spawn_controller", MagicMock())
    caplog.set_level(logging.WARNING, logger=store_api.__name__)
    run()
    assert others and not others[0].is_alive()
    assert not errors, errors
    assert _logged_frame_warnings(caplog) == [_FRAME_WARNING]


@_monarch_installed
def test_routing_initialize_on_monarchs_plane_logs_before_its_volume_spawn(
    monkeypatch, caplog
) -> None:
    """Log before the volume spawn on Monarch's plane too, so a failed spawn still logs it."""
    import torchstore as ts
    import torchstore.api as store_api
    from torchstore.storage_volume import StorageVolume

    at_spawn: list[list[list[str]]] = []

    def record_spawn(*args: Any, **kwargs: Any) -> None:
        at_spawn.append(_logged_frame_warnings(caplog))

    coordinator = MagicMock()
    coordinator.init.call_one = AsyncMock()
    monkeypatch.setattr(store_api, "_frame_length_warning_pending", True)
    monkeypatch.setattr(store_api, "_control_plane_map", {})
    monkeypatch.setattr(store_api, "_local_clent_map", {})
    monkeypatch.setattr(
        "monarch.actor.get_or_spawn_controller", AsyncMock(return_value=coordinator)
    )
    monkeypatch.setattr(StorageVolume, "spawn", AsyncMock(side_effect=record_spawn))
    caplog.set_level(logging.WARNING, logger=store_api.__name__)
    asyncio.run(
        ts.initialize(
            mesh=MagicMock(size=MagicMock(return_value=1)),
            strategy=MagicMock(set_storage_volumes=AsyncMock()),
            client_type=ts.ClientType.ROUTING,
            store_name=f"frame_{uuid.uuid4().hex[:8]}",
        )
    )
    coordinator.init.call_one.assert_awaited_once()
    assert at_spawn == [[_FRAME_WARNING]]
    assert _logged_frame_warnings(caplog) == [_FRAME_WARNING]


@_monarch_installed
@pytest.mark.parametrize("spawn_fails", [False, True], ids=["spawned", "spawn_failed"])
def test_standard_initialize_logs_before_its_volume_spawn(
    monkeypatch, caplog, spawn_fails: bool
) -> None:
    """Log before a standard store spawns its volumes, so a failed spawn still logs it."""
    import torchstore as ts
    import torchstore.api as store_api
    from torchstore.storage_volume import StorageVolume

    at_spawn: list[list[list[str]]] = []

    def record_spawn(*args: Any, **kwargs: Any) -> MagicMock:
        at_spawn.append(_logged_frame_warnings(caplog))
        if spawn_fails:
            raise RuntimeError("volume spawn failed")
        return MagicMock()

    controller = MagicMock()
    controller.init.call = AsyncMock()
    monkeypatch.setattr(store_api, "_frame_length_warning_pending", True)
    monkeypatch.setattr(store_api, "_control_plane_map", {})
    monkeypatch.setattr(store_api, "_local_clent_map", {})
    monkeypatch.setattr(
        "monarch.actor.get_or_spawn_controller", AsyncMock(return_value=controller)
    )
    monkeypatch.setattr(StorageVolume, "spawn", AsyncMock(side_effect=record_spawn))
    caplog.set_level(logging.WARNING, logger=store_api.__name__)
    initialize = ts.initialize(
        num_storage_volumes=2,
        strategy=ts.LocalRankStrategy(),
        store_name=f"frame_{uuid.uuid4().hex[:8]}",
    )
    if spawn_fails:
        with pytest.raises(RuntimeError, match="volume spawn failed"):
            asyncio.run(initialize)
    else:
        asyncio.run(initialize)
        controller.init.call.assert_awaited_once()
    assert at_spawn == [[_FRAME_WARNING]]
    assert _logged_frame_warnings(caplog) == [_FRAME_WARNING]


@_monarch_installed
def test_routed_sync_runs_on_a_monarch_loaded_after_torchstore(tmp_path: Path) -> None:
    """Spawn and route on Monarch's control plane once Monarch has read the exported length."""
    report = _run_child(
        _ROUTED_CHILD,
        tmp_path / "report.json",
        extra_env={"TORCHSTORE_RDMA_ENABLED": "0"},
    )
    assert report["modules"] == []
    assert report["at_import"] == report["before_torchstore_paths"] == []
    assert report["routed"] is True
    assert report["frame_length"] == _FRAME_LENGTH_DEFAULT
    assert _frame_warnings(report["after_routed_sync"]) == [_FRAME_WARNING]


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
def test_spmd_initialize_imports_what_each_rank_uses(
    monkeypatch, caplog, rank: int
) -> None:
    """Spawn and publish the controller on rank 0, and unpickle it on rank 1.

    Each rank logs the frame-length warning once, on its process's first initialize and
    before the rendezvous.
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
    # What this process had logged when this rank reached the TCPStore rendezvous.
    at_rendezvous: list[list[list[str]]] = []

    def rendezvous(*args: Any, **kwargs: Any) -> MagicMock:
        at_rendezvous.append(_logged_frame_warnings(caplog))
        return store

    # Restored after the test because initialize would otherwise export it for the session.
    monkeypatch.setenv("TORCHSTORE_HOSTNAME", socket.gethostname())
    monkeypatch.setattr("torchstore.spmd.TCPStore", MagicMock(side_effect=rendezvous))
    monkeypatch.setattr(
        "torchstore.spmd.StoreJob.from_store",
        MagicMock(return_value=MagicMock() if rank == 0 else None),
    )
    monkeypatch.setattr("torchstore.api.initialize", AsyncMock())
    monkeypatch.setattr("monarch.actor.get_or_spawn_controller", spawn)
    monkeypatch.setattr(store_api, "_frame_length_warning_pending", True)
    caplog.set_level(logging.WARNING, logger=store_api.__name__)
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
    # Logged before the rendezvous, so a first initialize that times out still logs it.
    assert at_rendezvous == [[_FRAME_WARNING]]
    second = f"spmd_{uuid.uuid4().hex[:8]}"
    try:
        asyncio.run(
            ts.initialize_spmd(strategy=ts.HostStrategy(), store_name=second, env=env)
        )
    finally:
        store_api._spmd_state_map.pop(second, None)
    assert _logged_frame_warnings(caplog) == [_FRAME_WARNING]
    assert store_api._frame_length_warning_pending is False
