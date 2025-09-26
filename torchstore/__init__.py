# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from logging import getLogger

from torchstore.api import (
    client,
    delete,
    exists,
    get,
    get_state_dict,
    initialize,
    keys,
    put,
    put_state_dict,
    reset_client,
    shutdown,
)

from torchstore.logging import init_logging
from torchstore.strategy import (
    ControllerStorageVolumes,
    LocalRankStrategy,
    SingletonStrategy,
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


__all__ = [
    "initialize",
    "init_logging",
    "put",
    "get",
    "delete",
    "keys",
    "exists",
    "client",
    "shutdown",
    "TorchStoreStrategy",
    "LocalRankStrategy",
    "SingletonStrategy",
    "ControllerStorageVolumes",
    "put_state_dict",
    "get_state_dict",
    "reset_client",
]
