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

# Open windows, and whether any window opened since the count left 0 found collection enabled.
_lock = threading.Lock()
_windows: int = 0
_was_enabled: bool = False
# Set by the first window, so importing this module registers no at-fork hook.
_fork_hook_registered: bool = False
# Bumped by the fork hook, so a window open across a fork never closes in the child.
_forks: int = 0


@contextlib.contextmanager
def gc_paused() -> Iterator[None]:
    """Pause cyclic collection, for every thread, while this one builds many objects.

    The last of overlapping windows to close resumes collection if any of them found it enabled,
    even inside a pause other code began meanwhile, and a pause begun inside a window may later
    restore collection to off. A window holds CPU-bound work only: no await, wait or I/O.
    """
    global _windows, _was_enabled
    _register_fork_hook()
    forks, opened = _forks, False
    try:
        with _lock:
            # A pause that ends inside the first window leaves a later one to pause collection again.
            _was_enabled = gc.isenabled() or (_windows > 0 and _was_enabled)
            # Counted before collection pauses, so a fork never finds it paused with no window open.
            # No call runs between the count and its flag, so no signal handler can raise between them.
            _windows += 1
            opened = True
            gc.disable()
        yield
    finally:
        # In a forked child, the fork hook has already closed every window it inherited. The check
        # makes no call, after which a signal handler could raise before the close.
        if opened and forks == _forks:
            # CPython runs signal handlers inside a blocked lock acquire: retry until this window closes,
            # then raise the first error. A handler raising in contextlib's enter or exit instead, or on
            # Python 3.10 as this block starts after the body raised, leaves the window counted, as can a
            # second raise, which on Python 3.10 can also skip a `with` block's cleanup and hold the lock.
            interrupted: BaseException | None = None
            while opened:
                try:
                    with _lock:
                        try:
                            if _windows == 1 and _was_enabled:
                                gc.enable()
                        finally:
                            # Last, so a fork never finds collection paused with no window open.
                            _windows -= 1
                            opened = False
                except BaseException as error:
                    if interrupted is None:
                        interrupted = error
            if interrupted is not None:
                try:
                    raise interrupted
                finally:
                    # The error's traceback holds this frame, so drop the frame's reference to it.
                    interrupted = None


def _resume() -> None:
    """Close every window, resuming collection if any found it enabled. Call with ``_lock`` held.

    It never pauses collection, which would undo a resume that another pause made meanwhile.
    """
    global _windows
    if _was_enabled:
        gc.enable()
    # Cleared last, so a fork never finds collection paused with no window open.
    _windows = 0


def _register_fork_hook() -> None:
    """Register the at-fork reset before the first window opens.

    It is flagged only once registered, so no window opens without it. First windows that
    race may each register it, which is harmless: a second run finds no window open.
    """
    global _fork_hook_registered
    if not _fork_hook_registered:
        os.register_at_fork(after_in_child=_after_fork_in_child)
        _fork_hook_registered = True


def _after_fork_in_child() -> None:
    """Start a forked child with a fresh lock and no window open, resuming collection if one found it enabled."""
    global _lock, _forks
    _forks += 1
    # Windows open in other threads never close here, and such a thread may hold the lock.
    _lock = threading.Lock()
    with _lock:
        if _windows:
            _resume()
