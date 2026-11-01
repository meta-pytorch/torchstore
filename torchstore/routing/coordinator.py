# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Old path of Monarch's routing coordinator, which lives in ``torchstore.monarch.coordinator``."""

from torchstore.monarch.coordinator import (  # noqa: F401
    _timed,
    logger,
    MonarchControlPlane,
    RoutingCoordinator,
)

__all__ = ["MonarchControlPlane", "RoutingCoordinator"]
