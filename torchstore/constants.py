# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

MONARCH_HOSTMESH_V1 = os.environ.get("MONARCH_HOSTMESH_V1", "1").lower() in (
    "1",
    "true",
)
