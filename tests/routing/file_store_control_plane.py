# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A routing control plane over a directory of files, brought by the caller.

It uses only TorchStore's public names and the standard library, as a user's own
control plane would.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import pickle
import tempfile
from collections.abc import Collection, Mapping
from urllib.parse import quote

import torchstore as ts

# The coordinator's default, read at call time so tests can shorten it.
_LAYOUT_REGISTRATION_TIMEOUT_S = 5 * 60
_POLL_S = 0.01
_PUBLISHERS, _STRATEGY, _READY = "publishers", "strategy", "ready"


class FileStoreControlPlane(ts.ControlPlane):
    """Shares one routing store through a directory that every rank opens by its path.

    Worker threads link each value's file into place whole, so a killed writer leaves
    no partial value. Values are pickled, so only the job's own ranks may write there.
    Its deadline is checked only between file calls, so it suits a local directory.
    """

    # TorchStore's selection log lines name a control plane by its ``kind``.
    kind = "filestore"

    def __init__(self, path: str) -> None:
        self._path = path
        self._closed = False

    async def init(
        self, publishers: Collection[str], strategy: ts.TorchStoreStrategy
    ) -> None:
        """Create the directory, then write the publishers, the strategy and ``ready``.

        The directory must not exist yet, so a later call raises ``FileExistsError``.
        """
        values = {
            _PUBLISHERS: json.dumps(sorted(publishers)).encode(),
            _STRATEGY: pickle.dumps(strategy),
            _READY: b"1",
        }
        await asyncio.to_thread(self._init, values)

    async def strategy(self) -> ts.TorchStoreStrategy:
        """Decode a fresh copy of the strategy on every call."""
        return pickle.loads(await asyncio.to_thread(self._read, _STRATEGY))

    async def register_layout(
        self, rank: str, key: str, registrations: Mapping[str, ts.KeyRegistration]
    ) -> None:
        """Write a publisher's layout for ``key`` once. The first write wins."""
        value = pickle.dumps(dict(registrations))
        await asyncio.to_thread(self._register, rank, _layout_key(key, rank), value)

    async def get_layouts(
        self, key: str, *, timeout_s: float | None = None
    ) -> ts.Registrations:
        """Poll until every publisher's layout for ``key`` is written, then read them."""
        if timeout_s is None:
            timeout_s = _LAYOUT_REGISTRATION_TIMEOUT_S
        if math.isnan(timeout_s):
            raise ValueError("get_layouts timeout_s must not be NaN")
        timeout_s = timeout_s if timeout_s > 0 else 0
        publishers = await asyncio.to_thread(self._publishers)
        keys = {rank: _layout_key(key, rank) for rank in publishers}
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        while missing := await asyncio.to_thread(self._missing, keys):
            if loop.time() >= deadline:
                raise TimeoutError(
                    f"timed out after {timeout_s} seconds "
                    f"waiting for publisher layouts for {key!r}; "
                    f"missing publishers: {missing}"
                )
            await asyncio.sleep(_POLL_S)
        values = await asyncio.to_thread(self._read_all, list(keys.values()))
        return {rank: pickle.loads(value) for rank, value in zip(keys, values)}

    async def close(self) -> None:
        """Refuse later calls. ``ts.shutdown`` leaves a caller's control plane open."""
        self._closed = True

    def _directory(self) -> str:
        if self._closed:
            raise RuntimeError(f"the control plane on {self._path!r} is closed")
        return self._path

    def _file(self, key: str) -> str:
        return os.path.join(self._directory(), quote(key, safe=""))

    def _init(self, values: Mapping[str, bytes]) -> None:
        # Only the last part, so a missing parent fails here, and only the owner may
        # write: no earlier run's files or anyone else's land in the store.
        os.mkdir(self._directory(), 0o700)
        for key, value in values.items():
            self._write_once(key, value)

    def _write_once(self, key: str, value: bytes) -> None:
        target = self._file(key)
        descriptor, temporary = tempfile.mkstemp(dir=self._path)
        try:
            with open(descriptor, "wb") as file:
                file.write(value)
            # A link never replaces a file, so the first write wins and lands whole.
            os.link(temporary, target)
        except FileExistsError:
            pass
        finally:
            os.unlink(temporary)

    def _read(self, key: str) -> bytes:
        if not os.path.exists(self._file(_READY)):
            raise RuntimeError("FileStoreControlPlane.init has not run")
        return self._load(key)

    def _load(self, key: str) -> bytes:
        with open(self._file(key), "rb") as file:
            return file.read()

    def _publishers(self) -> list[str]:
        return json.loads(self._read(_PUBLISHERS))

    def _register(self, rank: str, layout_key: str, value: bytes) -> None:
        if rank not in self._publishers():
            raise KeyError(f"{rank!r} is not a publisher rank")
        self._write_once(layout_key, value)

    def _missing(self, keys: Mapping[str, str]) -> list[str]:
        return [
            rank for rank, key in keys.items() if not os.path.exists(self._file(key))
        ]

    def _read_all(self, keys: list[str]) -> list[bytes]:
        return [self._load(key) for key in keys]


def _layout_key(key: str, rank: str) -> str:
    return f"layout/{quote(key, safe='')}/{rank}"
