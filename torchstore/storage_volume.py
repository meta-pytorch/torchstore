# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Old path of the Monarch storage volume, which lives in ``torchstore.monarch.storage_volume``."""

from torchstore.monarch.storage_volume import (  # noqa: F401
    FULL_TENSOR,
    InMemoryStore,
    logger,
    StorageImpl,
    StorageVolume,
)

__all__ = ["FULL_TENSOR", "InMemoryStore", "StorageImpl", "StorageVolume"]
