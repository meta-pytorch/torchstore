# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the process-wide collection pause: nesting, threads, exceptions and fork."""

from __future__ import annotations

import asyncio
import gc
import importlib.util
import json
import os
import signal
import subprocess
import sys
import threading
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from torchstore.routing import _gc

pytestmark = pytest.mark.timeout(120)

# Child processes import the tree that pytest imported.
_ROOT = str(Path(importlib.util.find_spec("torchstore").origin).parents[1])

# Records each at-fork hook as it is registered, from before TorchStore loads. Prints the
# pause's registrations after import, after the first window and after three more.
_HOOKS_CHILD = """
import json, os, sys, threading

sys.modules["monarch"] = None
sys.path.insert(0, sys.argv[1])
hooks = []
register_at_fork = os.register_at_fork


def recording(**kwargs):
    hooks.extend(f"{hook.__module__}.{hook.__qualname__}" for hook in kwargs.values())
    register_at_fork(**kwargs)


os.register_at_fork = recording
import torchstore
from torchstore.routing import _gc


def registered():
    return [hook for hook in hooks if hook.startswith("torchstore.routing._gc.")]


def window():
    with _gc.gc_paused():
        pass


counts = [registered()]
window()
counts.append(registered())
with _gc.gc_paused():
    window()
    thread = threading.Thread(target=window)
    thread.start()
    thread.join(10)
counts.append(registered())
print(json.dumps(counts), flush=True)
"""

# Forks children in one scenario, named by argv[2]. Prints their exit codes, then this
# process's collection state and open windows once its own windows have closed.
_FORK_CHILD = """
import gc, json, os, signal, sys, threading, traceback, weakref

# The pause needs no Monarch, and children forked here leave through os._exit.
sys.modules["monarch"] = None
sys.path.insert(0, sys.argv[1])
from torchstore.routing import _gc

scenario = sys.argv[2]


class Cycle:
    pass


def collects_garbage():
    # Whether a collection that this process's own allocations set off frees an unreachable cycle.
    cycle = Cycle()
    cycle.self = cycle
    freed = weakref.ref(cycle)
    del cycle
    kept = [[] for _ in range(100_000)]
    return freed() is None


def check_collection(enabled):
    # The child resumes collection if an inherited window found it enabled, and its own windows work.
    assert gc.isenabled() is enabled, "the child kept the parent's pause"
    assert collects_garbage() is enabled
    with _gc.gc_paused():
        assert not gc.isenabled()
    assert gc.isenabled() is enabled and _gc._windows == 0


def exit_code(pid):
    # A child stuck on an inherited lock, even inside fork's own hooks, is killed at the deadline.
    status = []
    waiter = threading.Thread(target=lambda: status.append(os.waitpid(pid, 0)[1]), daemon=True)
    waiter.start()
    waiter.join(10)
    if waiter.is_alive():
        os.kill(pid, signal.SIGKILL)
        waiter.join(10)
    return os.waitstatus_to_exitcode(status[0])


def in_child(check):
    pid = os.fork()
    if pid == 0:
        try:
            check()
            os._exit(0)
        except BaseException:
            traceback.print_exc()
            os._exit(1)
    return exit_code(pid)


def beside_a_window_or_the_held_lock():
    # Another thread's window is open, with collection disabled and then enabled, then another
    # thread holds the lock with no window open. A window registered the hook before, as in use.
    codes = []
    for enabled, held in ((False, "window"), (True, "window"), (False, "lock"), (True, "lock")):
        if enabled:
            gc.enable()
        else:
            gc.disable()
        holding, release = threading.Event(), threading.Event()

        def hold(held=held, holding=holding, release=release):
            with _gc.gc_paused() if held == "window" else _gc._lock:
                holding.set()
                release.wait(30)

        thread = threading.Thread(target=hold, daemon=True)
        thread.start()
        assert holding.wait(10)
        codes.append(in_child(lambda enabled=enabled: check_collection(enabled)))
        release.set()
        thread.join(10)
        assert not thread.is_alive() and gc.isenabled() is enabled
    gc.enable()
    return codes


def inside_its_own_windows():
    gc.enable()
    with _gc.gc_paused(), _gc.gc_paused():
        pid = os.fork()
        if pid == 0:
            resumed = gc.isenabled()
    if pid == 0:
        try:
            assert resumed, "the fork hook left this thread's windows paused"
            # Leaving the inherited windows must not unbalance the child's own, nor a second hook run.
            check_collection(True)
            _gc._after_fork_in_child()
            check_collection(True)
            os._exit(0)
        except BaseException:
            traceback.print_exc()
            os._exit(1)
    return [exit_code(pid)]


class Stalled:
    # Stands in for gc in the pause's module, stalling once on its first read, just after a pause
    # or just before a resume.
    def __init__(self, call):
        self.call, self.stalled, self.release = call, threading.Event(), threading.Event()

    def isenabled(self):
        self.stall("isenabled")
        return gc.isenabled()

    def disable(self):
        gc.disable()
        self.stall("disable")

    def enable(self):
        self.stall("enable")
        gc.enable()

    def stall(self, call):
        if call == self.call and not self.stalled.is_set():
            self.stalled.set()
            self.release.wait(30)


def mid_pause_or_mid_resume():
    def window():
        with _gc.gc_paused():
            pass

    codes = []
    for call in ("disable", "enable"):
        gc.enable()
        _gc.gc = stand_in = Stalled(call)
        thread = threading.Thread(target=window, daemon=True)
        thread.start()
        assert stand_in.stalled.wait(10)
        codes.append(in_child(lambda: check_collection(True)))
        stand_in.release.set()
        thread.join(10)
        _gc.gc = gc
        assert not thread.is_alive() and gc.isenabled()
    return codes


def as_the_first_window_registers_the_hook():
    # The process's first window stalls as it registers the hook, then a second window stalls
    # holding the lock. Neither fork may find the lock held without a registered hook.
    registering, release = threading.Event(), threading.Event()
    register_at_fork = os.register_at_fork

    def stalled_register(**hooks):
        if not registering.is_set():
            registering.set()
            release.wait(30)
        register_at_fork(**hooks)

    def window():
        with _gc.gc_paused():
            pass

    gc.enable()
    os.register_at_fork = stalled_register
    first = threading.Thread(target=window, daemon=True)
    first.start()
    assert registering.wait(10)
    codes = [in_child(lambda: check_collection(True))]
    _gc.gc = stand_in = Stalled("isenabled")
    second = threading.Thread(target=window, daemon=True)
    second.start()
    assert stand_in.stalled.wait(10)
    codes.append(in_child(lambda: check_collection(True)))
    stand_in.release.set()
    second.join(10)
    release.set()
    first.join(10)
    _gc.gc = gc
    os.register_at_fork = register_at_fork
    assert not first.is_alive() and not second.is_alive() and gc.isenabled()
    return codes


def after_a_pause_that_ended_inside_the_first_window():
    # A third thread's pause holds collection off as the first window opens and ends before the
    # second opens. Children fork with the first alone, both open and the second alone.
    entered = {name: threading.Event() for name in "AB"}
    leave = {name: threading.Event() for name in "AB"}

    def window(name):
        with _gc.gc_paused():
            entered[name].set()
            assert leave[name].wait(30)

    def pause(paused, end):
        # Pauses as torch's state-dict calls do, resuming if it found it enabled.
        enabled = gc.isenabled()
        gc.disable()
        paused.set()
        assert end.wait(30)
        if enabled:
            gc.enable()

    gc.enable()
    paused, end = threading.Event(), threading.Event()
    pausing = threading.Thread(target=pause, args=(paused, end), daemon=True)
    windows = {name: threading.Thread(target=window, args=(name,), daemon=True) for name in "AB"}
    pausing.start()
    assert paused.wait(10)
    windows["A"].start()
    assert entered["A"].wait(10) and not gc.isenabled()
    end.set()
    pausing.join(10)
    assert gc.isenabled(), "the pause did not resume collection"
    codes = [in_child(lambda: check_collection(True))]
    windows["B"].start()
    assert entered["B"].wait(10) and not gc.isenabled()
    codes.append(in_child(lambda: check_collection(True)))
    leave["A"].set()
    windows["A"].join(10)
    codes.append(in_child(lambda: check_collection(True)))
    leave["B"].set()
    windows["B"].join(10)
    assert not any(thread.is_alive() for thread in (pausing, *windows.values()))
    return codes


codes = globals()[scenario]()
print(json.dumps([codes, gc.isenabled(), _gc._windows]), flush=True)
"""


# A SIGINT reaches the main thread while its window's close waits for the lock another thread holds.
# Prints whether the close raised it, where the handler ran, and the state while waiting and after.
_INTERRUPTED_CLOSE_CHILD = """
import gc, json, signal, sys, threading, time

sys.modules["monarch"] = None
sys.path.insert(0, sys.argv[1])
from torchstore.routing import _gc

main = threading.main_thread().ident
window_code = _gc.gc_paused.__wrapped__.__code__
handled = []


def on_sigint(signum, frame):
    # Raises once, as Ctrl-C does. Later deliveries, from sends that crossed the first, only record.
    handled.append(frame is not None and frame.f_code is window_code)
    if len(handled) == 1:
        raise KeyboardInterrupt


def closing():
    frame = sys._current_frames().get(main)
    return frame is not None and frame.f_code is window_code


def hold(waiting):
    with _gc._lock:
        held.set()
        deadline = time.monotonic() + 30
        while not closing() and not done.is_set() and time.monotonic() < deadline:
            time.sleep(0.001)
        # A SIGINT that lands before the close starts to wait only marks its handler due: send again.
        while closing() and not handled and time.monotonic() < deadline:
            signal.pthread_kill(main, signal.SIGINT)
            time.sleep(0.01)
        waiting.append([_gc._windows, gc.isenabled()])


signal.signal(signal.SIGINT, on_sigint)
gc.enable()
held, done, waiting = threading.Event(), threading.Event(), []
window = _gc.gc_paused()
window.__enter__()
holder = threading.Thread(target=hold, args=(waiting,), daemon=True)
holder.start()
assert held.wait(30)
try:
    window.__exit__(None, None, None)
    raised = False
except KeyboardInterrupt:
    raised = True
done.set()
holder.join(30)
after = [_gc._windows, gc.isenabled(), _gc._lock.locked()]
with _gc.gc_paused():
    paused = not gc.isenabled()
later = [paused, gc.isenabled(), _gc._windows]
result = {"raised": raised, "handled": handled[:1], "waiting": waiting, "after": after, "later": later}
print(json.dumps(result), flush=True)
"""


def _run_child(script: str, *args: str) -> Any:
    # A session of its own, so a timeout also kills any child that it forked.
    child = subprocess.Popen(
        [sys.executable, "-c", script, _ROOT, *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = child.communicate(timeout=90)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        stdout, stderr = child.communicate()
        pytest.fail(f"child timed out:\n{stdout}{stderr}")
    assert child.returncode == 0, stdout + stderr
    return json.loads(stdout.splitlines()[-1])


class _WatchedLock:
    """A lock that counts the acquires that find it held, setting ``waited`` at the first.

    Once ``stall_exit`` is set, the next release stalls just after it until ``resume_exit`` is set.
    """

    def __init__(self) -> None:
        self.lock, self.waited = threading.Lock(), threading.Event()
        self.waits = 0
        self._counted = threading.Condition()
        self.stall_exit, self.exit_stalled = threading.Event(), threading.Event()
        self.resume_exit = threading.Event()

    def __enter__(self) -> None:
        if not self.lock.acquire(blocking=False):
            with self._counted:
                self.waits += 1
                self._counted.notify_all()
            self.waited.set()
            self.lock.acquire()

    def __exit__(self, *exc: object) -> None:
        self.lock.release()
        if self.stall_exit.is_set() and not self.exit_stalled.is_set():
            self.exit_stalled.set()
            assert self.resume_exit.wait(10)

    def wait_for_waits(self, waits: int, timeout: float) -> bool:
        """Wait until ``waits`` acquires in all have found the lock held."""
        with self._counted:
            return self._counted.wait_for(lambda: self.waits >= waits, timeout)


@pytest.fixture
def collecting() -> Iterator[None]:
    """Start with collection enabled and no window open, and enable collection again after."""
    assert gc.isenabled() and _gc._windows == 0
    try:
        yield
    finally:
        gc.enable()
    assert _gc._windows == 0


@pytest.mark.parametrize("enabled", [True, False], ids=["enabled", "disabled"])
def test_nested_windows_resume_collection_at_the_outermost_exit(
    collecting: None, enabled: bool
) -> None:
    """Keep collection paused until the outermost window closes, then leave it as it was."""
    if not enabled:
        gc.disable()
    with _gc.gc_paused():
        with _gc.gc_paused():
            assert not gc.isenabled() and _gc._windows == 2
        assert not gc.isenabled() and _gc._windows == 1
    assert gc.isenabled() is enabled and _gc._windows == 0


@pytest.mark.parametrize("enabled", [True, False], ids=["enabled", "disabled"])
def test_overlapping_windows_on_three_threads_resume_collection_at_the_last_exit(
    collecting: None, enabled: bool
) -> None:
    """Keep collection paused from the first window to the last, on three threads closing in order.

    The last resumes collection only if a window found it enabled, so it stays disabled if it was.
    """
    if not enabled:
        gc.disable()
    opened = [threading.Event() for _ in range(3)]
    leave = [threading.Event() for _ in range(3)]
    errors: list[BaseException] = []

    def window(i: int) -> None:
        try:
            with _gc.gc_paused(), _gc.gc_paused():
                opened[i].set()
                assert leave[i].wait(10)
        except BaseException as e:
            errors.append(e)

    threads = [
        threading.Thread(target=window, args=(i,), daemon=True) for i in range(3)
    ]
    states = []
    try:
        for thread, event in zip(threads, opened):
            thread.start()
            assert event.wait(10)
        assert _gc._windows == 6
        for thread, event in zip(threads, leave):
            event.set()
            thread.join(10)
            states.append(gc.isenabled())
    finally:
        for event in leave:
            event.set()
        for thread in threads:
            if thread.ident is not None:
                thread.join(10)
    assert not errors and not any(thread.is_alive() for thread in threads)
    assert states == [False, False, enabled]
    assert _gc._windows == 0


def test_the_last_window_keeps_the_resume_of_a_pause_that_ended_inside_it(
    collecting: None,
) -> None:
    """Leave collection enabled when another pause, open as the first window opened, ends inside it.

    Pausing collection at the window's close would leave it paused for good.
    """
    gc.disable()
    with _gc.gc_paused():
        gc.enable()
    assert gc.isenabled() and _gc._windows == 0


def test_no_window_close_pauses_collection_that_another_pause_resumed(
    collecting: None,
) -> None:
    """Leave collection enabled when another pause, open as two windows opened, ends inside both.

    A close that leaves a window open pausing collection again would leave it paused for good.
    """
    gc.disable()
    with _gc.gc_paused():
        with _gc.gc_paused():
            gc.enable()
        assert gc.isenabled() and _gc._windows == 1
    assert gc.isenabled() and _gc._windows == 0


def test_the_last_window_resumes_a_pause_that_a_later_window_made(
    collecting: None,
) -> None:
    """Resume collection that a later window paused after another pause, open at the first, ended.

    Recording only the first window's state would leave collection paused for good.
    """
    gc.disable()
    with _gc.gc_paused():
        gc.enable()
        with _gc.gc_paused():
            assert not gc.isenabled()
    assert gc.isenabled() and _gc._windows == 0


def test_the_last_window_resumes_collection_inside_a_pause_other_code_began(
    collecting: None,
) -> None:
    """Resume collection at the last window's close, even inside a pause other code began meanwhile.

    A guard that starts inside the window and restores the state it found, as vLLM's CUDA-graph
    capture guard does, then leaves collection paused.
    """
    with _gc.gc_paused():
        # The guard's pause begins here, and it finds collection paused.
        gc.disable()
    assert gc.isenabled() and _gc._windows == 0


@pytest.mark.parametrize("enabled", [True, False], ids=["enabled", "disabled"])
@pytest.mark.parametrize(
    "error",
    [ValueError("planning failed"), asyncio.CancelledError()],
    ids=["exception", "cancelled"],
)
def test_a_window_that_raises_still_closes(
    collecting: None, error: BaseException, enabled: bool
) -> None:
    """Close a window that an exception or a cancellation leaves, inside one that stays open or as the last."""
    if not enabled:
        gc.disable()
    with _gc.gc_paused():
        with pytest.raises(type(error)):
            with _gc.gc_paused():
                raise error
        assert not gc.isenabled() and _gc._windows == 1
    assert gc.isenabled() is enabled and _gc._windows == 0
    with pytest.raises(type(error)):
        with _gc.gc_paused():
            raise error
    assert gc.isenabled() is enabled and _gc._windows == 0


def test_a_window_opening_while_the_first_records_the_state_waits_for_it(
    collecting: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Make a second thread's window wait for the lock while the first records the collection state.

    The lock makes each open, like each close, one step that no other window's update can split.
    """
    recording, release, leave = threading.Event(), threading.Event(), threading.Event()
    calls: list[None] = []

    def isenabled() -> bool:
        calls.append(None)
        if len(calls) == 1:
            recording.set()
            assert release.wait(10)
        return gc.isenabled()

    fake = types.SimpleNamespace(
        isenabled=isenabled, enable=gc.enable, disable=gc.disable
    )
    monkeypatch.setattr(_gc, "gc", fake)
    watched = _WatchedLock()
    monkeypatch.setattr(_gc, "_lock", watched)
    opened = [threading.Event(), threading.Event()]
    errors: list[BaseException] = []

    def window(i: int) -> None:
        try:
            with _gc.gc_paused():
                opened[i].set()
                assert leave.wait(10)
        except BaseException as e:
            errors.append(e)

    threads = [
        threading.Thread(target=window, args=(i,), daemon=True) for i in range(2)
    ]
    try:
        threads[0].start()
        assert recording.wait(10)
        threads[1].start()
        assert watched.waited.wait(10)
        assert not opened[1].is_set()
        release.set()
        assert all(event.wait(10) for event in opened)
        assert not gc.isenabled()
    finally:
        release.set()
        leave.set()
        for thread in threads:
            if thread.ident is not None:
                thread.join(10)
    assert not errors and not any(thread.is_alive() for thread in threads)
    assert gc.isenabled() and _gc._windows == 0


def test_closing_windows_wait_while_another_thread_holds_the_lock(
    collecting: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Make two threads' windows that close together each wait for the lock, then resume once.

    Reading the count before the lock would let both closes see two windows and leave collection paused.
    """
    watched = _WatchedLock()
    monkeypatch.setattr(_gc, "_lock", watched)
    opened = [threading.Event(), threading.Event()]
    leave = threading.Event()
    errors: list[BaseException] = []

    def window(i: int) -> None:
        try:
            with _gc.gc_paused():
                opened[i].set()
                assert leave.wait(10)
        except BaseException as e:
            errors.append(e)

    threads = [
        threading.Thread(target=window, args=(i,), daemon=True) for i in range(2)
    ]
    try:
        for thread, event in zip(threads, opened):
            thread.start()
            assert event.wait(10)
        with watched:
            waits = watched.waits
            leave.set()
            assert watched.wait_for_waits(waits + 2, 10)
            assert not gc.isenabled() and _gc._windows == 2
    finally:
        leave.set()
        for thread in threads:
            if thread.ident is not None:
                thread.join(10)
    assert not errors and not any(thread.is_alive() for thread in threads)
    assert gc.isenabled() and _gc._windows == 0


def test_a_window_opening_while_the_last_resumes_collection_waits_for_it(
    collecting: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Make a window that opens while the last one resumes collection wait for the lock, then pause.

    Resuming after the lock would turn collection on inside the new window and drop it from the count.
    """
    resuming, release, leave = threading.Event(), threading.Event(), threading.Event()

    def enable() -> None:
        if not resuming.is_set():
            resuming.set()
            assert release.wait(10)
        gc.enable()

    fake = types.SimpleNamespace(
        isenabled=gc.isenabled, enable=enable, disable=gc.disable
    )
    monkeypatch.setattr(_gc, "gc", fake)
    watched = _WatchedLock()
    monkeypatch.setattr(_gc, "_lock", watched)
    opened = threading.Event()
    errors: list[BaseException] = []

    def window(stay: bool) -> None:
        try:
            with _gc.gc_paused():
                if stay:
                    opened.set()
                    assert leave.wait(10)
        except BaseException as e:
            errors.append(e)

    threads = [
        threading.Thread(target=window, args=(stay,), daemon=True)
        for stay in (False, True)
    ]
    try:
        threads[0].start()
        assert resuming.wait(10)
        threads[1].start()
        assert watched.waited.wait(10)
        assert not opened.is_set()
        release.set()
        assert opened.wait(10)
        assert not gc.isenabled() and _gc._windows == 1
    finally:
        release.set()
        leave.set()
        for thread in threads:
            if thread.ident is not None:
                thread.join(10)
    assert not errors and not any(thread.is_alive() for thread in threads)
    assert gc.isenabled() and _gc._windows == 0


@pytest.mark.parametrize("then", ["close", "open"])
def test_a_close_overtaken_just_after_it_releases_the_lock_resumes_once(
    collecting: None, monkeypatch: pytest.MonkeyPatch, then: str
) -> None:
    """Resume collection once when another thread closes or opens a window just after a close's release.

    Decrementing or clearing the count after the lock would leave collection paused for good.
    """
    watched = _WatchedLock()
    monkeypatch.setattr(_gc, "_lock", watched)
    opened = [threading.Event() for _ in range(3)]
    leave = [threading.Event() for _ in range(3)]
    errors: list[BaseException] = []

    def window(i: int) -> None:
        try:
            with _gc.gc_paused():
                opened[i].set()
                assert leave[i].wait(10)
        except BaseException as e:
            errors.append(e)

    threads = [
        threading.Thread(target=window, args=(i,), daemon=True) for i in range(3)
    ]
    other = 1 if then == "close" else 2
    try:
        for i in (0, 1) if then == "close" else (0,):
            threads[i].start()
            assert opened[i].wait(10)
        watched.stall_exit.set()
        leave[0].set()
        assert watched.exit_stalled.wait(10)
        if then == "open":
            threads[2].start()
            assert opened[2].wait(10)
        leave[other].set()
        threads[other].join(10)
        watched.resume_exit.set()
        threads[0].join(10)
    finally:
        watched.resume_exit.set()
        for event in leave:
            event.set()
        for thread in threads:
            if thread.ident is not None:
                thread.join(10)
    assert not errors and not any(thread.is_alive() for thread in threads)
    assert gc.isenabled() and _gc._windows == 0


@pytest.mark.parametrize(
    "edge", ["disable", "enable"], ids=["after-pause", "after-resume"]
)
def test_a_window_still_closes_when_a_signal_handler_raises_at_its_edge(
    collecting: None, monkeypatch: pytest.MonkeyPatch, edge: str
) -> None:
    """Close a window at once when an error leaves it right after it pauses or resumes collection.

    Signal handlers raise at the check after a call returns, as Ctrl-C does, so the window's
    fork check makes no call. A raise in contextlib's own enter or exit is not covered.
    """

    class HandlerError(Exception):
        """What a signal handler raises at the check after a call returns."""

    calls: list[None] = []

    def interrupted() -> None:
        getattr(gc, edge)()
        calls.append(None)
        # Raises on two calls at most, so a close that retried would still end, its calls counted.
        if len(calls) <= 2:
            raise HandlerError

    getpids: list[None] = []

    def getpid() -> int:
        getpids.append(None)
        return os.getpid()

    fake_gc = types.SimpleNamespace(
        isenabled=gc.isenabled, enable=gc.enable, disable=gc.disable
    )
    setattr(fake_gc, edge, interrupted)
    fake_os = types.SimpleNamespace(**{**vars(os), "getpid": getpid})
    try:
        with monkeypatch.context() as patch:
            patch.setattr(_gc, "gc", fake_gc)
            patch.setattr(_gc, "os", fake_os)
            with pytest.raises(HandlerError):
                with _gc.gc_paused():
                    pass
        left = (_gc._windows, gc.isenabled())
        with _gc.gc_paused():
            pass
        later = (_gc._windows, gc.isenabled())
    finally:
        # A window left counted would keep collection paused through later cases' windows.
        _gc._windows = 0
    assert left == (0, True), "the error left its window open"
    assert later == (0, True), "a later window left collection paused"
    assert len(calls) == 1, "the window paused or resumed collection twice"
    assert not getpids, "the window called os.getpid, after which a handler could raise"


def test_a_close_that_a_sigint_interrupts_while_it_waits_for_the_lock_still_closes() -> None:
    """Close a window whose wait for the lock, held by another thread, a SIGINT interrupts, then raise it.

    CPython runs the handler inside the blocked acquire, so the close retries until its window closes.
    """
    assert _run_child(_INTERRUPTED_CLOSE_CHILD) == {
        "raised": True,
        "handled": [True],
        "waiting": [[1, False]],
        "after": [0, True, False],
        "later": [True, True, 0],
    }


def test_the_first_window_registers_the_fork_hook_and_importing_registers_none() -> None:
    """Register the at-fork reset in the first window, not at import or in the windows after it.

    Importing TorchStore therefore adds none of this module's hooks.
    """
    hook = ["torchstore.routing._gc._after_fork_in_child"]
    assert _run_child(_HOOKS_CHILD) == [[], hook, hook]


@pytest.mark.parametrize(
    "scenario, codes",
    [
        ("beside_a_window_or_the_held_lock", 4),
        ("inside_its_own_windows", 1),
        ("mid_pause_or_mid_resume", 2),
        ("as_the_first_window_registers_the_hook", 2),
        ("after_a_pause_that_ended_inside_the_first_window", 3),
    ],
)
def test_a_forked_child_starts_with_a_fresh_lock_and_no_window_open(
    scenario: str, codes: int
) -> None:
    """Collect garbage in a child forked while a window is open, if a window found collection enabled.

    Children fork beside other threads' windows or the held lock, inside their own windows, mid-pause,
    mid-resume, mid-registration of the hook and after a pause ended inside a window. Their windows work.
    """
    assert _run_child(_FORK_CHILD, scenario) == [[0] * codes, True, 0]
