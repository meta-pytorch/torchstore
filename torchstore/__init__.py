# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
from typing import Any, TYPE_CHECKING

import torchstore.api as _api
from torchstore.api import (
    client,
    delete,
    delete_batch,
    exists,
    get,
    get_batch,
    get_state_dict,
    initialize,
    keys,
    put,
    put_batch,
    put_state_dict,
    reset_client,
    shutdown,
)
from torchstore.client import ClientType
from torchstore.logging import init_logging
from torchstore.routing._model import KeyRegistration, RankRole, Registrations
from torchstore.routing.control_plane import ControlPlane, ControlPlaneAddress
from torchstore.strategy import (
    ControllerStorageVolumes,
    HostStrategy,
    LocalRankStrategy,
    TorchStoreStrategy,
)

# Monarch reads the frame length when it first loads, which importing TorchStore never
# does, so a process that loads Monarch afterwards uses the default too, and one that
# loaded Monarch first keeps Monarch's own limit. Monarch entry points log the export.
if _api._export_frame_length_default():
    init_logging()


# Type checkers see these names but not __getattr__, so they still flag misspelled ones.
if TYPE_CHECKING:
    from torchstore import spmd
    from torchstore.spmd import initialize as initialize_spmd
else:

    def __getattr__(name: str) -> Any:
        """Load ``spmd`` on first use, since it imports Monarch's job API when it loads."""
        if name == "spmd":
            return importlib.import_module("torchstore.spmd")
        if name == "initialize_spmd":
            return importlib.import_module("torchstore.spmd").initialize
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), "initialize_spmd", "spmd"})


__all__ = [
    "initialize",
    "initialize_spmd",
    "init_logging",
    "put",
    "put_batch",
    "get",
    "get_batch",
    "delete",
    "delete_batch",
    "keys",
    "exists",
    "client",
    "ClientType",
    "RankRole",
    "ControlPlane",
    "ControlPlaneAddress",
    "KeyRegistration",
    "Registrations",
    "shutdown",
    "TorchStoreStrategy",
    "LocalRankStrategy",
    "HostStrategy",
    "ControllerStorageVolumes",
    "put_state_dict",
    "get_state_dict",
    "reset_client",
    "spmd",
]
