# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the collection pause: nesting, exceptions, another pause and fork."""

from __future__ import annotations

import gc
import subprocess
import sys
import threading
from collections.abc import Iterator

import pytest
from torchstore.routing import _gc

# Loads the pause module alone by path, so no torch, and forks once beside another
# thread's open window, which found collection enabled. Exits with the child's status.
_FORK_CHILD = """
import gc, importlib.util, os, sys, threading

spec = importlib.util.spec_from_file_location("_gc", sys.argv[1])
_gc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_gc)
holding, release = threading.Event(), threading.Event()


def hold():
    with _gc.gc_paused():
        holding.set()
        release.wait()


threading.Thread(target=hold, daemon=True).start()
holding.wait()
pid = os.fork()
if pid == 0:
    os._exit(0 if gc.isenabled() else 1)
release.set()
sys.exit(os.waitstatus_to_exitcode(os.waitpid(pid, 0)[1]))
"""


@pytest.fixture
def collecting() -> Iterator[None]:
    """Start with collection enabled, end with no window recorded, and enable it again."""
    assert gc.isenabled()
    try:
        yield
        assert _gc._resuming == set()
    finally:
        gc.enable()


def _hold_a_window(entered: threading.Event, release: threading.Event) -> None:
    with _gc.gc_paused():
        entered.set()
        release.wait()


@pytest.mark.parametrize("enabled", [True, False], ids=["enabled", "disabled"])
def test_nested_windows_resume_collection_at_the_outermost_exit(
    collecting: None, enabled: bool
) -> None:
    """Keep collection paused until the outermost window closes, then leave it as it was."""
    if not enabled:
        gc.disable()
    with _gc.gc_paused():
        with _gc.gc_paused():
            assert not gc.isenabled()
        assert not gc.isenabled()
    assert gc.isenabled() is enabled


def test_a_window_never_pauses_collection_at_exit(collecting: None) -> None:
    """Leave collection enabled when a window that found it paused sees other code resume it."""
    gc.disable()
    with _gc.gc_paused():
        gc.enable()
    assert gc.isenabled()


@pytest.mark.parametrize("error_type", [ValueError, KeyboardInterrupt])
@pytest.mark.parametrize("enabled", [True, False], ids=["enabled", "disabled"])
def test_a_window_that_raises_still_closes(
    collecting: None, enabled: bool, error_type: type[BaseException]
) -> None:
    """Close a window that an exception leaves, inside one that stays open or as the last."""
    if not enabled:
        gc.disable()
    error = error_type("planning failed")
    with _gc.gc_paused():
        with pytest.raises(error_type) as caught:
            with _gc.gc_paused():
                raise error
        assert caught.value is error
        assert not gc.isenabled()
    assert gc.isenabled() is enabled
    with pytest.raises(error_type) as caught:
        with _gc.gc_paused():
            raise error
    assert caught.value is error
    assert gc.isenabled() is enabled


@pytest.mark.timeout(60)
def test_windows_in_two_threads_record_only_a_window_that_found_collection_on(
    collecting: None,
) -> None:
    """Resume collection when the window that found it on closes, inside another's window.

    Only that window is recorded for the fork hook, and the other one never resumes it.
    """
    first_in, first_out, second_in, second_out = (threading.Event() for _ in range(4))
    first = threading.Thread(target=_hold_a_window, args=(first_in, first_out))
    second = threading.Thread(target=_hold_a_window, args=(second_in, second_out))
    try:
        first.start()
        assert first_in.wait(10)
        second.start()
        assert second_in.wait(10)
        assert not gc.isenabled() and _gc._resuming == {first.ident}
        first_out.set()
        first.join(10)
        assert gc.isenabled() and _gc._resuming == set()
        # Collection that other code turns off stays off in a child and once all close.
        gc.disable()
        _gc._after_fork_in_child()
        assert not gc.isenabled()
        second_out.set()
        second.join(10)
        assert not gc.isenabled()
    finally:
        first_out.set()
        second_out.set()
        first.join(10)
        second.join(10)


@pytest.mark.parametrize("found_enabled", [True, False], ids=["enabled", "disabled"])
def test_the_fork_hook_resumes_collection_only_for_a_window_that_found_it_on(
    collecting: None, monkeypatch: pytest.MonkeyPatch, found_enabled: bool
) -> None:
    """Resume collection in a forked child only if an open window found it enabled.

    That window is another thread's, so it never closes in the child.
    """
    gc.disable()
    another_thread = threading.get_ident() + 1
    monkeypatch.setattr(_gc, "_resuming", {another_thread} if found_enabled else set())
    _gc._after_fork_in_child()
    assert gc.isenabled() is found_enabled
    assert _gc._resuming == set()


@pytest.mark.timeout(60)
def test_a_forked_child_runs_the_fork_hook() -> None:
    """Find collection enabled in a child forked beside another thread's open window."""
    child = subprocess.run(
        [sys.executable, "-c", _FORK_CHILD, _gc.__file__],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert child.returncode == 0, child.stdout + child.stderr
