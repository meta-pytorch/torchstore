# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TorchStore's Monarch implementations, which core modules reach through ``torchstore.backend``.

Monarch RDMA instead registers with ``torchstore.transport`` when it loads, and the old
module paths, such as ``torchstore.storage_volume``, re-export the moved names.
"""
