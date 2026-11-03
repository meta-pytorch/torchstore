# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""One rank of a routed sync on the service backend: a plain process given its identity as arguments.

Run as ``python -m tests.routing._routed_sync_worker --role driver|publisher|requester``. The driver's first
reply is its address, every other role's first stdin line. Then each command line gets one ``reply <json>`` line.
"""

import sys
from typing import cast

if __name__ == "__main__" and "--block-monarch" in sys.argv:
    # Before torch loads, so any import of either raises, as with torchmonarch uninstalled.
    blocked = cast("dict[str, object]", sys.modules)
    blocked["monarch"] = blocked["cloudpickle"] = None

import argparse
import asyncio
import collections
import dataclasses
import json
import logging
import os
import re
import threading
import time
import traceback
from collections.abc import Callable
from concurrent import futures
from typing import Any

import torch
import torchstore as ts
from torch.distributed.tensor import DTensor
from torchstore import carrier_volumes
from torchstore.carrier import runtime
from torchstore.carrier_volumes import VOLUME_ENDPOINTS
from torchstore.routing.service import ServiceAddress
from torchstore.storage import StorageVolumeCore
from torchstore.strategy import _installed_volume_host
from torchstore.transport import gloo, TransportType

from . import routed_sync_data as data

REPLY = "reply "
_STEP_TIMEOUT_S = 60.0
# Log lines that the parent's checks read. Transport lines are only counted.
_KEPT = (
    "routing control_plane=",
    "routing volume=",
    "carrier serve ",
    "carrier connect ",
    "carrier stats ",
)
_TRANSPORT = re.compile(r"\[ts-transport\] resolved=(\w+)")
# Modules that would show Monarch's runtime or an actor-based path at work.
_WATCHED = (
    "monarch.actor",
    "monarch.rdma",
    "monarch.job",
    "torchstore.monarch",
)


def reply(**fields: Any) -> None:
    print(REPLY + json.dumps(fields), flush=True)


def _error(error: BaseException) -> str:
    return "".join(traceback.format_exception_only(type(error), error)).strip()


class _Lines(logging.Handler):
    """Keeps TorchStore's selection, volume and carrier lines, and counts resolved transports."""

    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.kept: list[str] = []
        self.transports: collections.Counter[str] = collections.Counter()

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if match := _TRANSPORT.search(message):
            self.transports[match.group(1)] += 1
        elif message.startswith(_KEPT):
            self.kept.append(message)


def _count_calls() -> collections.Counter[str]:
    """Count each volume endpoint call that this process's volume cores take."""
    calls: collections.Counter[str] = collections.Counter()

    def counted(name: str, method: Callable[..., Any]) -> Callable[..., Any]:
        async def endpoint(self: StorageVolumeCore, *args: Any, **kwargs: Any) -> Any:
            calls[name] += 1
            return await method(self, *args, **kwargs)

        return endpoint

    for name in VOLUME_ENDPOINTS:
        setattr(
            StorageVolumeCore, name, counted(name, getattr(StorageVolumeCore, name))
        )
    return calls


def _address(line: str) -> ServiceAddress:
    """The driver's address, from its reply line as the parent or a person passes it on."""
    fields = json.loads(line[len(REPLY) :] if line.startswith(REPLY) else line)
    return ServiceAddress(**fields["address"])


class _Rank:
    """One routing rank in this process. Each store call runs on a fresh event loop."""

    def __init__(self, args: argparse.Namespace, lines: _Lines) -> None:
        self.args = args
        self.lines = lines
        self.calls: collections.Counter[str] = collections.Counter()
        self.hostnames: dict[str, str] = {}
        self.commands: dict[str, Callable[..., None]] = {"shutdown": self.shutdown}

    def start(self) -> None:
        raise NotImplementedError

    def serve(self) -> None:
        while line := sys.stdin.readline():
            command, *args = line.split() or [""]
            try:
                handler = self.commands.get(command)
                if handler is None:
                    raise ValueError(f"a {self.args.role} has no command {command!r}")
                handler(*args)
            except Exception as error:
                reply(error=_error(error))
            if command == "shutdown":
                return

    def connect(self, role: ts.RankRole, address: ServiceAddress | None) -> None:
        """Make this process's client from its arguments alone, then report it ready."""
        if role == ts.RankRole.PUBLISHER:
            self.calls = _count_calls()
        client = ts.client(
            self.args.store,
            role=role,
            group=self.args.group,
            rank=self.args.rank,
            address=address,
        )
        asyncio.run(client)
        # Serving the volume read its core's ID for the record. Only carrier calls count.
        self.calls.clear()
        reply(ready=os.getpid())

    def push(self, round_: str) -> None:
        state_dict = data.place(
            data.weights(int(round_)),
            data.PUBLISHER_DIMS,
            self.args.rank,
            data.PUBLISHER_RANKS,
        )
        put = ts.put_state_dict(state_dict, data.KEY, store_name=self.args.store)
        # A worker thread with a fresh event loop per push, as in TitanRL's trainer.
        asyncio.run(asyncio.to_thread(lambda: asyncio.run(put)))
        reply(pushed=int(round_))

    def shutdown(self) -> None:
        # Taken before the volume stops: stopping resets its core directly, not over the carrier.
        calls = dict(self.calls)
        pairs = self.volume_pairs()
        asyncio.run(ts.shutdown(self.args.store))
        reply(report={**self.report(), "calls": calls, "pairs": pairs})

    def volume_pairs(self) -> int:
        """How many Gloo pairs this process's volume keeps a half of, or 0 if it serves none."""
        volume = carrier_volumes._publishers.get(
            (self.args.store, f"publisher/{self.args.rank}")
        )
        core = None if volume is None else volume._core
        if core is None:
            return 0
        groups = core.store.transport_context.get(gloo.GlooProcessGroupCache)
        return len(groups._process_groups)

    def report(self) -> dict[str, Any]:
        return {
            "pid": os.getpid(),
            "torchstore": ts.__file__,
            "loaded": [name for name in _WATCHED if sys.modules.get(name) is not None],
            "lines": self.lines.kept,
            "transports": dict(self.lines.transports),
            "hostnames": self.hostnames,
        }


class _Driver(_Rank):
    """Initializes the store at start. Given ``--rank``, it also becomes that publisher on ``publish``."""

    def start(self) -> None:
        initialize = ts.initialize(
            num_publishers=self.args.num_publishers,
            strategy=ts.TorchStoreStrategy(TransportType[self.args.transport]),
            client_type=ts.ClientType.ROUTING,
            store_name=self.args.store,
            control_plane="service",
        )
        address = asyncio.run(initialize)
        assert isinstance(address, ServiceAddress)
        reply(address=dataclasses.asdict(address))
        self.commands["lookups"] = self.lookups
        if self.args.rank is not None:
            # Through the store it initialized, as a torchrun-style launcher's rank 0 can.
            self.commands.update(
                publish=lambda: self.connect(ts.RankRole.PUBLISHER, None),
                push=self.push,
            )

    def lookups(self, count: str) -> None:
        """Reply once the coordinator holds ``count`` calls, such as requesters' layout lookups that wait."""
        deadline = time.monotonic() + _STEP_TIMEOUT_S
        held = self.coordinator_calls()
        while held < int(count):
            assert time.monotonic() < deadline, f"the coordinator holds {held} calls"
            time.sleep(0.01)
            held = self.coordinator_calls()
        reply(lookups=held)

    def coordinator_calls(self) -> int:
        """How many calls the store's coordinator in this process runs now."""
        servers = runtime._runtime.servers if runtime._runtime else set()
        return sum(
            server.stats()["running"]
            for server in servers
            if server.address.service == "coordinator"
            and server.address.scope.startswith(f"{self.args.store}/")
        )


class _Publisher(_Rank):
    def start(self) -> None:
        self.commands["push"] = self.push
        self.connect(ts.RankRole.PUBLISHER, _address(sys.stdin.readline()))


class _Requester(_Rank):
    """Reads into the same buffers every round and checks each slice bitwise here."""

    def __init__(self, args: argparse.Namespace, lines: _Lines) -> None:
        super().__init__(args, lines)
        self.num_ranks, self.dims = data.REQUESTERS[args.group]
        zeros = {name: torch.zeros_like(t) for name, t in data.weights(0).items()}
        self.destination: dict[str, Any] = {
            **data.place(zeros, self.dims, args.rank, self.num_ranks),
            data.DESTINATION_ONLY: data.sentinel(),
        }
        self.exchange: futures.Future[None] | None = None
        self.commands.update(
            {
                "layouts": self.layouts,
                "pull": self.pull,
                "deadline": self.deadline,
                "pull-fault": self.pull_fault,
            }
        )

    def start(self) -> None:
        self.connect(ts.RankRole.REQUESTER, _address(sys.stdin.readline()))

    def layouts(self, wait_s: str = "0") -> None:
        """Start the layout exchange, a read's first step, then report whether it ends in ``wait_s``."""
        if self.exchange is None:
            started = threading.Event()
            exchange: futures.Future[None] = futures.Future()

            async def get_layouts() -> None:
                client = await ts.client(self.args.store)
                started.set()
                await client.get_layouts(self.destination, data.KEY, strict=False)

            def run() -> None:
                try:
                    exchange.set_result(asyncio.run(get_layouts()))
                except BaseException as error:
                    exchange.set_exception(error)
                finally:
                    started.set()

            threading.Thread(target=run, daemon=True).start()
            assert started.wait(_STEP_TIMEOUT_S), "the layout exchange did not start"
            self.exchange = exchange
        try:
            self.exchange.result(timeout=float(wait_s))
        except futures.TimeoutError:
            reply(layouts="waiting")
        else:
            reply(layouts="done")

    def _get(self) -> list[str]:
        if self.exchange is not None:
            self.exchange.result(timeout=_STEP_TIMEOUT_S)
        get = ts.get_state_dict(
            data.KEY, self.destination, strict=False, store_name=self.args.store
        )
        return sorted(asyncio.run(get))

    def pull(self, round_: str) -> None:
        expected = data.weights(int(round_))
        assert self._get() == sorted(expected)
        for name, tensor in expected.items():
            held = self.destination[name]
            held = held.to_local() if isinstance(held, DTensor) else held
            local = data.local(tensor, self.dims[name], self.args.rank, self.num_ranks)
            data.assert_bitwise_equal(held, local, f"round {round_}, {name}")
        assert torch.equal(self.destination[data.DESTINATION_ONLY], data.sentinel())
        reply(pulled=int(round_))

    def deadline(self, seconds: str) -> None:
        # Read as each call starts, so the next read uses it.
        runtime._CALL_TIMEOUT_S = float(seconds)
        reply(deadline=float(seconds))

    def pull_fault(self) -> None:
        """Time a read expected to fail within this process, so no timing includes an exit."""
        start = time.monotonic()
        try:
            self._get()
        except Exception as error:
            cause = error.__cause__
            reply(
                fault={
                    "error": type(error).__name__,
                    "message": str(error),
                    "cause": type(cause).__name__,
                    "cause_message": str(cause),
                    "elapsed_s": time.monotonic() - start,
                }
            )
        else:
            reply(fault=None)

    def shutdown(self) -> None:
        client = asyncio.run(ts.client(self.args.store))
        host = _installed_volume_host(client.strategy)
        assert host is not None
        # Before shutdown closes the host: each name comes from its volume's record.
        self.hostnames = {name: host.hostname(name) for name in host.volume_ids()}
        super().shutdown()


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--role", required=True, choices=("driver", "publisher", "requester")
    )
    parser.add_argument("--store", required=True)
    parser.add_argument("--num-publishers", type=int, default=data.PUBLISHER_RANKS)
    parser.add_argument("--transport", default=TransportType.Unset.name)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--group", type=int)
    parser.add_argument("--block-monarch", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse()
    lines = _Lines()
    # Children inherit the frame length that the parent's import exported, so the package
    # init sets up no logging here. The checks need these INFO lines, and warnings go on.
    logger = logging.getLogger("torchstore")
    logger.setLevel(logging.INFO)
    logger.addHandler(lines)
    warnings = logging.StreamHandler(sys.stderr)
    warnings.setLevel(logging.WARNING)
    logger.addHandler(warnings)
    roles: dict[str, type[_Rank]] = {
        "driver": _Driver,
        "publisher": _Publisher,
        "requester": _Requester,
    }
    rank = roles[args.role](args, lines)
    try:
        rank.start()
    except Exception as error:
        reply(error=_error(error))
    rank.serve()


if __name__ == "__main__":
    main()
