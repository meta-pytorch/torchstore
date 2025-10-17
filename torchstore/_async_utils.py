# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
from typing import Callable, cast, Generic, TypeVar

T = TypeVar("T")


class OnceCell(Generic[T]):
    """Poor man's version of tokio::sync::OnceCell, except it's not threadsafe (maybe it is because of GIL?).
    This is a cell that can be initialized exactly once."""

    def __init__(self):
        self._lock = asyncio.Lock()
        self._value: T | None = None
        self._initialized = False

    async def get_or_init(self, initializer) -> T:
        if self._initialized:
            return cast(T, self._value)

        async with self._lock:
            if not self._initialized:
                self._value = await initializer()
                self._initialized = True

        return cast(T, self._value)

    def get(self) -> T:
        if not self._initialized:
            raise ValueError("Value not initialized yet")
        return cast(T, self._value)


class SequentialExecutor:
    """A simple executor that runs tasks sequentially in the current event loop.
    This is mainly needed for RDMA operations, which will panic if concurrent requests are made (what the heck?).
    """

    def __init__(self):
        self._queue = asyncio.Queue()
        self._worker_task = None

    async def start_worker(self):
        self._worker_task = asyncio.create_task(self._worker())

    async def _worker(self):
        while True:
            try:
                func, args, kwargs, response = await self._queue.get()

                if response.cancelled():
                    continue  # Caller gave up

                try:
                    result = await func(*args, **kwargs)
                    response.set_result(result)
                except Exception as e:
                    response.set_exception(e)

            except Exception as outer_err:
                # Log or handle the error
                print(f"[SequentialExecutor] Worker crashed: {outer_err}")

    async def submit(self, func: Callable, *args, **kwargs) -> asyncio.Future:
        fut = asyncio.Future()
        await self._queue.put((func, args, kwargs, fut))
        return await fut
