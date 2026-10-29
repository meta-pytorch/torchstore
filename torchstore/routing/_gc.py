# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A process-wide pause of cyclic garbage collection around allocation-heavy work."""

from __future__ import annotations

import contextlib
import gc
import os
import threading
from collections.abc import Iterator

__all__ = ["gc_paused"]

# Threads whose open window found collection enabled, which a forked child resumes.
_resuming: set[int] = set()


@contextlib.contextmanager
def gc_paused() -> Iterator[None]:
    """Pause cyclic collection for every thread while this thread builds many objects.

    A window that found collection enabled enables it at exit, also on a raise, even inside
    another thread's window or a pause begun meanwhile, which may later restore it to off.
    No window disables it at exit. Hold only CPU-bound work: no await, wait or I/O.
    """
    thread = threading.get_ident()
    enabled = gc.isenabled()
    if enabled:
        _resuming.add(thread)
        gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()
            _resuming.discard(thread)


def _after_fork_in_child() -> None:
    # Windows open in other threads never close in a forked child.
    if _resuming:
        _resuming.clear()
        gc.enable()


os.register_at_fork(after_in_child=_after_fork_in_child)
