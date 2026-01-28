# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
import time


def init_logging():
    log_level = os.environ.get("TORCHSTORE_LOG_LEVEL", "INFO").upper()

    logging.root.setLevel(log_level)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(log_level)

    # Check if a StreamHandler to sys.stdout is already present
    for handler in logging.root.handlers:
        if (
            isinstance(handler, logging.StreamHandler)
            and getattr(handler, "stream", None) == sys.stdout
        ):
            # Already has a stdout handler, no need to add another
            return
    logging.root.addHandler(stdout_handler)


class LatencyTracker:
    def __init__(self, name: str) -> None:
        self.name = name
        self.last_step = self.start_time = time.perf_counter()

    def _format_throughput(self, elapsed: float, tensor) -> str:
        """Format throughput string if tensor is provided."""
        if tensor is None or elapsed <= 0:
            return ""
        nbytes = tensor.numel() * tensor.element_size()
        throughput_gbps = (nbytes / 1e9) / elapsed
        return f" ({throughput_gbps:.2f} GB/s)"

    def track_step(self, step_name: str, tensor=None) -> None:
        now = time.perf_counter()
        elapsed = now - self.last_step
        throughput = self._format_throughput(elapsed, tensor)
        logging.debug(f"{self.name}:{step_name} took {elapsed:.4f}s{throughput}")
        self.last_step = now

    def track_e2e(self) -> None:
        logging.debug(
            f"{self.name} took {time.perf_counter() - self.start_time} seconds"
        )
