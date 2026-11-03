# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Blocking native calls run on daemon threads that a deadline bounds.

A call stuck in native code past its deadline keeps only its daemon thread, which ``asyncio.run``
never waits for and exit waits for only as long as its caller's exit hook allows. Internal and may change.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable
from typing import Any, TypeVar

__all__ = ["EXPIRED", "run_with_deadline", "settle"]

_T = TypeVar("_T")

# What a call's future holds once its deadline came first.
EXPIRED = object()


def settle(
    future: asyncio.Future[Any], result: Any, error: BaseException | None
) -> None:
    """Set ``future``'s result, or ``error`` if given, unless the future is already done.

    A future refuses ``StopIteration``. That error becomes the cause of a ``RuntimeError`` instead.
    """
    if future.done():
        return
    if error is None:
        future.set_result(result)
        return
    if type(error) is StopIteration:
        cause, error = error, RuntimeError("the native call raised StopIteration")
        error.__cause__ = cause
    future.set_exception(error)


def _run(
    loop: asyncio.AbstractEventLoop,
    future: asyncio.Future[Any],
    fn: Callable[[], Any],
) -> None:
    try:
        outcome = (fn(), None)
    except BaseException as error:
        # Its traceback would hold this frame, whose future holds the error, in a cycle.
        outcome = (None, error.with_traceback(None))
    try:
        loop.call_soon_threadsafe(settle, future, *outcome)
    except RuntimeError:
        # The loop closed after a cancel or the deadline. Nothing awaits this call.
        pass


async def run_with_deadline(
    fn: Callable[[], _T], deadline: float, what: str, thread_name: str
) -> _T:
    """Run ``fn`` on a daemon thread named ``thread_name``, raising ``TimeoutError`` at ``deadline``.

    ``fn``'s own error propagates without its traceback (a ``StopIteration`` as the cause of a
    ``RuntimeError``). A deadline that has passed, or is NaN, starts nothing.
    """
    timeout = deadline - time.monotonic()
    # Written so that a NaN deadline is refused too.
    if not timeout > 0:
        raise TimeoutError(f"{what} was not started: its deadline had passed")
    loop = asyncio.get_running_loop()
    future: asyncio.Future[Any] = loop.create_future()
    threading.Thread(
        target=_run, args=(loop, future, fn), name=thread_name, daemon=True
    ).start()
    # loop.time() is time.monotonic(), so the timer fires at the deadline. It settles the
    # future instead of cancelling it, so any CancelledError here is the caller's.
    timer = loop.call_at(deadline, settle, future, EXPIRED, None)
    try:
        result = await future
    finally:
        timer.cancel()
        # A kept error holds this frame, and the future holds what the call returned or raised.
        del future
    if result is EXPIRED:
        raise TimeoutError(f"{what} did not finish within {timeout:.1f} s")
    return result
