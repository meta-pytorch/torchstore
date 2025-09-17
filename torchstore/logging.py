# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys


def init_logging():
    log_level = os.environ.get("TORCHSTORE_LOG_LEVEL", "INFO").upper()
    print(f"xxxxx torchstore setting log level to {log_level}")
    logging.root.setLevel(log_level)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(log_level)
    logging.root.addHandler(stdout_handler)
