# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import os
import sys
from logging import getLogger
from typing import Any, TYPE_CHECKING

# Monarch reads its settings when first imported, so loading it before the
# frame-length default below keeps that default for spawned processes only.
# With the variable already set nothing is exported, so Monarch loads on first use.
# find_spec raises on a spec-less stub, so skip it once monarch is in sys.modules.
if (
    os.environ.get("HYPERACTOR_CODEC_MAX_FRAME_LENGTH") is None
    and "monarch" not in sys.modules
    and importlib.util.find_spec("monarch") is not None
):
    importlib.import_module("monarch")

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
from torchstore.routing._model import RankRole
from torchstore.strategy import (
    ControllerStorageVolumes,
    HostStrategy,
    LocalRankStrategy,
    TorchStoreStrategy,
)

if os.environ.get("HYPERACTOR_CODEC_MAX_FRAME_LENGTH", None) is None:
    init_logging()
    logger = getLogger(__name__)
    logger.warning(
        "Warning: setting HYPERACTOR_CODEC_MAX_FRAME_LENGTH since this needs to be set"
        " to enable large RPC calls via Monarch"
    )
    os.environ["HYPERACTOR_CODEC_MAX_FRAME_LENGTH"] = "910737418240"


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
