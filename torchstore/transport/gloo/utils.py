# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch.distributed as dist


def gloo_available() -> bool:
    """Check if gloo transport is available and enabled.

    Returns True if:
    1. TORCHSTORE_GLOO_ENABLED environment variable is set to "1"
    2. torch.distributed is available
    3. gloo backend is available

    Note: Unlike the previous implementation, this no longer requires a
    pre-initialized process group. The GlooTransportBuffer creates its
    own dedicated process group per client-storage connection.
    """
    gloo_enabled = os.environ.get("TORCHSTORE_GLOO_ENABLED", "0") == "1"
    if not gloo_enabled:
        return False

    if not dist.is_available():
        return False

    if not dist.is_gloo_available():
        return False

    return True
