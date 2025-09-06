# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from logging import getLogger

from torchstore.logging import init_logging

if os.environ.get("HYPERACTOR_CODEC_MAX_FRAME_LENGTH", None) is None:
    init_logging()
    logger = getLogger(__name__)
    logger.warning(
        "Warning: setting HYPERACTOR_CODEC_MAX_FRAME_LENGTH since this needs to be set"
        " to enable large RPC calls via Monarch"
    )
    os.environ["HYPERACTOR_CODEC_MAX_FRAME_LENGTH"] = "910737418240"

from torchstore.store import MultiProcessStore


__all__ = ["MultiProcessStore"]
