# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Utility to configure Monarch transport settings for tests."""

from monarch._rust_bindings.monarch_hyperactor.channel import ChannelTransport
from monarch._rust_bindings.monarch_hyperactor.config import configure


def apply_monarch_patch():
    """Apply Monarch configuration patch for MetaTLS with hostname."""
    configure(
        default_transport=ChannelTransport.MetaTlsWithHostname,
    )
