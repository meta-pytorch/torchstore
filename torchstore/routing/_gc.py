# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A process-wide pause of automatic cyclic garbage collection around allocation-heavy work."""

from __future__ import annotations

import gc
import os
from collections.abc import Callable
from typing import TypeVar

__all__ = ["call_gc_paused"]

_T = TypeVar("_T")

# A token per running call that found collection enabled, which a forked child resumes.
_resuming: set[object] = set()


def call_gc_paused(fn: Callable[..., _T], /, *args: object) -> _T:
    """Call ``fn(*args)`` with automatic cyclic collection paused for every thread.

    A call that found collection enabled enables it as it returns, also on a raise or an interrupt, even
    inside another thread's call or a pause begun meanwhile, which may later restore it to off. No call
    disables it as it returns, and ``gc.collect()`` still collects. Pass only CPU-bound work: no await or I/O.
    """
    if not gc.isenabled():
        return fn(*args)
    token = object()
    # Record, pause, call, resume and drop in one frame, so no interrupt lands between a pause and
    # the try that resumes it, as one could in a context manager's frames.
    try:
        _resuming.add(token)
        gc.disable()
        return fn(*args)
    finally:
        try:
            gc.enable()
        finally:
            _resuming.discard(token)


def _after_fork_in_child() -> None:
    # Calls running in other threads never return in a forked child, so any open one resumes there,
    # freezing what the parent held first so that no collection in the child finalizes it.
    if _resuming:
        _resuming.clear()
        gc.freeze()
        gc.enable()


os.register_at_fork(after_in_child=_after_fork_in_child)
