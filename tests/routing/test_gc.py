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
from types import SimpleNamespace

import pytest
from torchstore.routing import _gc

# Loads the pause module alone by path, so no torch, and forks once beside another thread's
# running call, which found collection enabled and left a garbage cycle with a finalizer. The
# child exits 0 if collection is on and collecting finalizes nothing it inherited, else 1, and
# the parent exits 2 if its own collection was no longer paused after the fork.
_FORK_CHILD = """
import gc, importlib.util, os, sys, threading

spec = importlib.util.spec_from_file_location("_gc", sys.argv[1])
_gc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_gc)
holding, release = threading.Event(), threading.Event()
finalized = []


class Cycle:
    def __init__(self):
        self.cycle = self

    def __del__(self):
        finalized.append(self)


def hold():
    Cycle()
    holding.set()
    release.wait()


threading.Thread(target=_gc.call_gc_paused, args=(hold,)).start()
holding.wait()
pid = os.fork()
if pid == 0:
    gc.collect()
    os._exit(0 if gc.isenabled() and not finalized else 1)
# The hook acts in the child only: the parent's call is still running.
parent_paused = not gc.isenabled()
release.set()
status = os.waitstatus_to_exitcode(os.waitpid(pid, 0)[1])
sys.exit(status if parent_paused else 2)
"""


@pytest.fixture
def collecting(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Start with collection enabled and no call recorded, check none is left, enable it."""
    gc.enable()
    monkeypatch.setattr(_gc, "_resuming", set())
    try:
        yield
        assert _gc._resuming == set()
    finally:
        gc.enable()


def _hold(entered: threading.Event, release: threading.Event) -> None:
    entered.set()
    release.wait()


@pytest.mark.parametrize("enabled", [True, False], ids=["enabled", "disabled"])
def test_nested_calls_resume_collection_at_the_outermost_return(
    collecting: None, enabled: bool
) -> None:
    """Keep collection paused until the outermost call returns, then leave it as it was."""
    if not enabled:
        gc.disable()
    seen: list[bool] = []

    def outer() -> None:
        _gc.call_gc_paused(lambda: seen.append(gc.isenabled()))
        seen.append(gc.isenabled())

    _gc.call_gc_paused(outer)
    assert seen == [False, False]
    assert gc.isenabled() is enabled


def test_a_call_never_pauses_collection_as_it_returns(collecting: None) -> None:
    """Leave collection enabled when a call that found it paused sees other code resume it."""
    gc.disable()
    _gc.call_gc_paused(gc.enable)
    assert gc.isenabled()


def test_an_interrupt_as_a_call_pauses_collection_leaves_no_record(
    collecting: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Record a call before it pauses, so an interrupt right after the pause resumes and drops it."""
    recorded_on_pause: list[int] = []

    def interrupted() -> None:
        recorded_on_pause.append(len(_gc._resuming))
        gc.disable()
        raise KeyboardInterrupt

    stand_in = SimpleNamespace(
        isenabled=gc.isenabled, disable=interrupted, enable=gc.enable
    )
    monkeypatch.setattr(_gc, "gc", stand_in)
    with pytest.raises(KeyboardInterrupt):
        _gc.call_gc_paused(lambda: None)
    assert recorded_on_pause == [1]
    assert gc.isenabled() and _gc._resuming == set()


def test_an_interrupt_as_a_call_resumes_collection_still_drops_its_record(
    collecting: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resume collection while the call is still recorded, then drop it despite an interrupt."""
    recorded_on_resume: list[int] = []

    def interrupted() -> None:
        recorded_on_resume.append(len(_gc._resuming))
        gc.enable()
        raise KeyboardInterrupt

    stand_in = SimpleNamespace(
        isenabled=gc.isenabled, disable=gc.disable, enable=interrupted
    )
    monkeypatch.setattr(_gc, "gc", stand_in)
    with pytest.raises(KeyboardInterrupt):
        _gc.call_gc_paused(lambda: None)
    assert recorded_on_resume == [1]
    assert gc.isenabled() and _gc._resuming == set()


def test_a_call_drops_its_record_after_other_code_resumed_collection(
    collecting: None,
) -> None:
    """Drop a call's record as it returns even when other code resumed collection inside it."""
    _gc.call_gc_paused(gc.enable)
    assert gc.isenabled() and _gc._resuming == set()


def test_a_nested_call_that_found_collection_on_keeps_the_outer_record(
    collecting: None,
) -> None:
    """Keep an outer call's record after a nested one that also found collection on returns."""
    records: list[int] = []

    def outer() -> None:
        gc.enable()
        _gc.call_gc_paused(lambda: records.append(len(_gc._resuming)))
        records.append(len(_gc._resuming))

    _gc.call_gc_paused(outer)
    assert records == [2, 1]


@pytest.mark.parametrize("error_type", [ValueError, KeyboardInterrupt])
@pytest.mark.parametrize("enabled", [True, False], ids=["enabled", "disabled"])
def test_a_call_that_raises_still_resumes(
    collecting: None, enabled: bool, error_type: type[BaseException]
) -> None:
    """Resume after a call that raises, inside one that keeps running or as the last."""
    if not enabled:
        gc.disable()
    error = error_type("planning failed")

    def fail() -> None:
        raise error

    def outer() -> None:
        with pytest.raises(error_type) as caught:
            _gc.call_gc_paused(fail)
        assert caught.value is error
        assert not gc.isenabled()

    _gc.call_gc_paused(outer)
    assert gc.isenabled() is enabled
    with pytest.raises(error_type) as caught:
        _gc.call_gc_paused(fail)
    assert caught.value is error
    assert gc.isenabled() is enabled


@pytest.mark.timeout(60)
def test_calls_in_two_threads_record_only_a_call_that_found_collection_on(
    collecting: None,
) -> None:
    """Resume collection when the call that found it on returns, inside another's call.

    Only that call is recorded for the fork hook, and the other one never resumes it.
    """
    first_in, first_out, second_in, second_out = (threading.Event() for _ in range(4))
    first = threading.Thread(
        target=_gc.call_gc_paused, args=(_hold, first_in, first_out)
    )
    second = threading.Thread(
        target=_gc.call_gc_paused, args=(_hold, second_in, second_out)
    )
    try:
        first.start()
        assert first_in.wait(10)
        second.start()
        assert second_in.wait(10)
        assert not gc.isenabled() and len(_gc._resuming) == 1
        first_out.set()
        first.join(10)
        assert not first.is_alive()
        assert gc.isenabled() and _gc._resuming == set()
        # Collection that other code turns off stays off in a child and once all return.
        gc.disable()
        _gc._after_fork_in_child()
        assert not gc.isenabled()
        second_out.set()
        second.join(10)
        assert not second.is_alive()
        assert not gc.isenabled()
    finally:
        first_out.set()
        second_out.set()
        for thread in (first, second):
            if thread.ident is not None:
                thread.join(10)


@pytest.mark.parametrize("found_enabled", [True, False], ids=["enabled", "disabled"])
def test_the_fork_hook_resumes_collection_only_for_a_call_that_found_it_on(
    collecting: None, monkeypatch: pytest.MonkeyPatch, found_enabled: bool
) -> None:
    """Freeze what the parent held, then resume, only if a call running at the fork found it on."""
    calls: list[str] = []
    stand_in = SimpleNamespace(
        freeze=lambda: calls.append("freeze"), enable=lambda: calls.append("enable")
    )
    monkeypatch.setattr(_gc, "gc", stand_in)
    # The token of a call running at the fork, whichever thread made it.
    monkeypatch.setattr(_gc, "_resuming", {object()} if found_enabled else set())
    _gc._after_fork_in_child()
    assert calls == (["freeze", "enable"] if found_enabled else [])
    assert _gc._resuming == set()


@pytest.mark.timeout(60)
def test_a_forked_child_resumes_collection_but_never_finalizes_what_it_inherited() -> None:
    """Find collection on in a child forked beside another thread's call, with its garbage frozen.

    The parent finds it still paused after the fork, since the hook runs in the child only.
    """
    child = subprocess.run(
        [sys.executable, "-c", _FORK_CHILD, _gc.__file__],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert child.returncode == 0, child.stdout + child.stderr
