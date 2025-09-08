# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time
import logging
import os
import sys

logger = logging.getLogger(__name__)


def init_logging():
    log_level = os.environ.get("TORCHSTORE_LOG_LEVEL", "INFO").upper()
    logging.root.setLevel(log_level)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(log_level)
    logging.root.addHandler(stdout_handler)


class LatencyTracker:
    def __init__(self, name: str) -> None:
        self.name = name
        self.start_time = 0.0
        self.last_step = 0.0

    def track_step(self, step_name: str) -> None:
        now = time.perf_counter()
        logger.debug(f"{self.name}:{step_name} took {now - self.last_step} seconds")
        self.last_step = now
        
    def track_e2e(self) -> None:
        logger.debug(f"{self.name} took {time.time() - self.start_time} seconds")
