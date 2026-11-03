# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for storage volumes served over the carrier and reached through volume proxies."""

import asyncio
import collections
import contextlib
import dataclasses
import gc
import inspect
import json
import logging
import os
import pickle
import socket
import subprocess
import threading
import time
import traceback
import uuid
import weakref
from collections.abc import (
    Awaitable,
    Callable,
    Collection,
    Coroutine,
    Generator,
    Iterator,
)
from functools import partial
from types import coroutine, SimpleNamespace
from typing import Any, cast

import pytest
import torch
import torchstore.transport as transport
from torchstore import carrier_volumes
from torchstore.carrier import (
    CallTimeoutError,
    CarrierAddress,
    CarrierClient,
    CarrierProtocolError,
    CarrierServer,
    close_all,
    PeerLostError,
    RemoteError,
    runtime,
)
from torchstore.carrier_volumes import (
    decode_volume_record,
    encode_volume_record,
    serve_volume,
    VOLUME_ENDPOINTS,
    VolumeProxy,
)
from torchstore.storage import StorageVolumeCore
from torchstore.strategy import StorageVolumeRef
from torchstore.transport import gloo, TransportType
from torchstore.transport.buffers import TransportBuffer, TransportContext
from torchstore.transport.gloo import GlooProcessGroupCache, GlooTransportBuffer
from torchstore.transport.monarch_rpc import MonarchRPCTransportBuffer
from torchstore.transport.shared_memory import (
    allocate_shared_tensor,
    SharedMemoryCache,
    SharedMemoryDescriptor,
    SharedMemoryTransportBuffer,
    ShmContext,
)
from torchstore.transport.types import Request, TensorSlice

from . import test_carrier
from .routing.routed_sync_data import assert_bitwise_equal
from .test_carrier import (
    _assert_blocks_in_park,
    _Children,
    _counts,
    _in_thread,
    _read_line,
    _scope,
    _serve,
    _Target,
    _until,
)
from .test_volume_core import _TRANSPORT_CALLS
from .volume_meshes import get_through as _get, put_through as _put

pytestmark = [
    pytest.mark.timeout(120),
    pytest.mark.filterwarnings("error::pytest.PytestUnhandledThreadExceptionWarning"),
]
# For a case that parks a loop thread: a carrier runtime of its own, as test_carrier's cases use.
fresh_runtime = test_carrier.fresh_runtime

_HOLD_KEY = "held"
# What a second live client's tensor transfers raise.
_IN_USE = "another live client in this process holds the Gloo pair"

# Buffers checked only for what they carry never reach a volume.
_NO_REF: Any = None

_VOLUME_CHILD = """
import asyncio, atexit, sys, threading


@atexit.register
def report_dispatchers():
    # Registered before torchstore loads, so it runs after every exit hook torchstore registers.
    names = [thread.name for thread in threading.enumerate()]
    alive = sum(name.startswith("torchstore-carrier-dispatch") for name in names)
    print("dispatchers at exit:", alive, flush=True)


# Volume servers never need Monarch, so blocking it shows the child loads none.
sys.modules["monarch"] = None
sys.path.insert(0, sys.argv[1])
from torchstore.carrier_volumes import serve_volume
from torchstore.storage import StorageVolumeCore


class ServerOnlyError(Exception):
    pass


class Core(StorageVolumeCore):
    async def put(self, transport_buffer, requests):
        print("put", *(request.key for request in requests), flush=True)
        await super().put(transport_buffer, requests)

    async def reset(self):
        raise ServerOnlyError("raised where only the volume can import it")


async def main():
    _, record = await serve_volume(Core(sys.argv[2]), scope=sys.argv[3])
    print(record.decode(), flush=True)
    # "quit" or the end of input returns from main.
    await asyncio.get_running_loop().run_in_executor(None, sys.stdin.readline)


asyncio.run(main())
"""


class _InstrumentedCore(StorageVolumeCore):
    """A volume core that records its calls and how many overlap, and holds the named ones."""

    def __init__(
        self, held: Collection[str] = (), *, volume_id: str | None = None
    ) -> None:
        super().__init__(volume_id or f"volume-{uuid.uuid4()}")
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.held = set(held)
        self.release = threading.Event()
        self.active = 0
        self.max_active = 0
        for name in VOLUME_ENDPOINTS:
            setattr(self, name, partial(self._run, name, getattr(self, name)))

    async def _run(
        self,
        name: str,
        method: Callable[..., Awaitable[Any]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        self.calls.append((name, args))
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if name in self.held:
                await asyncio.get_running_loop().run_in_executor(
                    None, self.release.wait
                )
            # A call running alongside this one would start during these yields.
            for _ in range(3):
                await asyncio.sleep(0)
            return await method(*args, **kwargs)
        finally:
            self.active -= 1

    def names(self) -> list[str]:
        return [name for name, _ in self.calls]

    def keys(self, name: str) -> list[str]:
        """The first request's key in each call to ``name``, in call order."""
        return [args[1][0].key for called, args in self.calls if called == name]


class _Volumes:
    """Serves cores and builds references in this process, then frees all of it."""

    def __init__(self) -> None:
        self.scope = _scope()
        self.cores: list[StorageVolumeCore] = []
        self.servers: list[CarrierServer] = []
        self.refs: list[StorageVolumeRef] = []
        # Store keys of the pairs Gloo gave up during the test, in the order heard_retired heard them.
        self.retired: list[str] = []
        self._retiring = threading.Condition()

    async def serve(
        self, core: StorageVolumeCore | None = None, *, host: str | None = None
    ) -> tuple[Any, CarrierServer, bytes]:
        core = _InstrumentedCore() if core is None else core
        self.cores.append(core)
        server, record = await serve_volume(core, scope=self.scope, host=host)
        self.servers.append(server)
        if isinstance(core, _InstrumentedCore):
            core.calls.clear()
        return core, server, record

    def heard_retired(self, volume_id: str, store_key: str) -> None:
        """A Gloo retired listener, which hears of a pair after carrier volumes queued its key."""
        with self._retiring:
            self.retired.append(store_key)
            self._retiring.notify_all()

    async def wait_retired(self, count: int) -> list[str]:
        """Wait until ``count`` pairs retired and return the keys heard.

        An executor thread can let go of a pair's group a little after its transfer returns.
        """

        def wait() -> bool:
            with self._retiring:
                return self._retiring.wait_for(lambda: len(self.retired) >= count, 30)

        assert await asyncio.to_thread(wait), self.retired
        return list(self.retired)

    def ref(
        self, record: bytes, volume_id: str, transport_type: TransportType
    ) -> StorageVolumeRef:
        """A reference to the volume through a proxy, as a strategy would build one."""
        address, hostname = decode_volume_record(volume_id, record)
        ref = StorageVolumeRef(
            cast(Any, VolumeProxy(address)),
            volume_id,
            TransportContext(),
            transport_type,
            hostname,
        )
        self.refs.append(ref)
        return ref

    def close(self) -> None:
        """Run every teardown step, last registered first, though an earlier one raises."""
        with contextlib.ExitStack() as steps:
            # A test's frames and its failed calls' tracebacks can hold Gloo pairs in cycles.
            steps.callback(gc.collect)
            for core in self.cores:
                steps.callback(core.store.reset)
            for ref in self.refs:
                steps.callback(ref.transport_context.clear)
            # Tests share this process, and Gloo caches pairs per volume ID.
            steps.callback(gloo._forget_pairs, [ref.volume_id for ref in self.refs])
            steps.callback(self._close_all)
            for core in self.cores:
                if isinstance(core, _InstrumentedCore):
                    steps.callback(core.release.set)

    def _close_all(self) -> None:
        """Close every carrier client and server, then wait for each served volume's dispatcher to end.

        No wait follows a close that failed, which leaves that failure the one reported.
        """
        _in_thread(close_all(grace_s=0))
        current = runtime._runtime
        assert current is None or not (current.peers or current.servers)
        # A released handler ends its server's dispatcher thread when it returns.
        for server in self.servers:
            server.dispatcher.thread.join(30)
            assert not server.dispatcher.thread.is_alive(), "a dispatcher thread leaked"


@pytest.fixture(autouse=True)
def volumes(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Volumes]:
    # Torn down before monkeypatch, so handlers it releases see the test's patched timeouts.
    volumes = _Volumes()
    # Listens after carrier volumes' own listener, which has queued each key by then.
    listeners = [*gloo._retired_listeners, volumes.heard_retired]
    monkeypatch.setattr(gloo, "_retired_listeners", listeners)
    yield volumes
    volumes.close()


@pytest.fixture(autouse=True)
def _capture_warnings(caplog: pytest.LogCaptureFixture) -> None:
    # The log asserts hold whatever level TORCHSTORE_LOG_LEVEL gives the root logger.
    caplog.set_level(logging.WARNING)


def _serve_in_child(children: _Children) -> tuple[subprocess.Popen, str, bytes]:
    volume_id = f"volume-{uuid.uuid4()}"
    child = children.spawn(_VOLUME_CHILD, volume_id, _scope())
    return child, volume_id, _read_line(child, timeout_s=90).encode()


def _command(child: subprocess.Popen, command: str) -> None:
    stdin = child.stdin
    assert stdin is not None
    stdin.write(f"{command}\n")
    stdin.flush()


async def _hold(core: _InstrumentedCore, ref: StorageVolumeRef) -> asyncio.Future[Any]:
    """Start a delete that the volume holds until ``core.release`` is set, so later calls queue."""
    core.held.add("delete")
    holder = asyncio.ensure_future(ref.volume.delete.call_one(_HOLD_KEY))
    await _until(lambda: core.active == 1)
    return holder


async def _release(
    server: CarrierServer,
    core: _InstrumentedCore,
    holders: Collection[asyncio.Future[Any]],
    *,
    abandoned: bool = True,
    idle_timeout_s: float = 10,
) -> None:
    """Let the held calls end, after an abandoned queued call left the queue, then wait until idle."""
    if abandoned:
        # Released only then, so the abandoned call never runs.
        await _until(lambda: _counts(server)[1:] == (1, 0))
    core.release.set()
    await asyncio.gather(*holders, return_exceptions=True)
    await _until(lambda: _counts(server)[1:] == (0, 0), timeout_s=idle_timeout_s)


def _batch(value: float) -> list[Request]:
    """Two keys in one put, as a routed ``put_batch`` sends all of a publisher's keys."""
    return [
        Request.from_tensor("weights", torch.full((8,), value)),
        Request.from_tensor("bias", torch.full((4,), value)),
    ]


class _ForgottenAfterRead(dict):
    """A Gloo pair cache that another thread empties right after this thread first reads it."""

    def get(self, key: Any, default: Any = None) -> Any:
        found = super().get(key, default)
        super().pop(key, None)
        return found


async def _forget_and_queue(
    volumes: _Volumes, volume_id: str, ref: StorageVolumeRef
) -> None:
    """Forget the pair ``ref``'s context holds, then wait until carrier volumes queued its key."""
    [key] = ref.transport_context.get(GlooProcessGroupCache)._process_groups
    heard = len(volumes.retired)
    gloo._forget_pairs(volume_id, ref.transport_context)
    assert (await volumes.wait_retired(heard + 1))[heard:] == [key]


def _waiting() -> dict[CarrierAddress, list[str]]:
    """The retired keys waiting for each server whether or not a call has queued them under it.

    Reads snapshots another thread may extend. Unqueued keys go first so that one a call queues meanwhile still shows.
    """
    retiring = tuple(carrier_volumes._retiring)
    waiting = {
        where: [store_key for _, store_key in keys]
        for where, keys in tuple(carrier_volumes._retired.items())
    }
    for where, store_key, _ in retiring:
        waiting.setdefault(where, []).append(store_key)
    return waiting


def _queued(store_key: str) -> bool:
    """Whether ``store_key`` waits among the retired keys."""
    return any(store_key in keys for keys in _waiting().values())


def _retire_toward(server: CarrierAddress, store_key: str) -> None:
    """Give up ``store_key`` as the pair a proxy's handshake made with ``server``."""
    carrier_volumes._pair_servers[store_key] = server
    carrier_volumes._retire(server.service, store_key)


def _taken(server: CarrierAddress) -> list[str]:
    """The keys a call to ``server`` takes, oldest first."""
    return [store_key for _, store_key in carrier_volumes._take_retired(server)]


@coroutine
def _suspend() -> Generator[None, None, None]:
    """Suspend the coroutine awaiting this once for a test that drives that coroutine by hand."""
    yield


_ADDRESS = CarrierAddress("dial", 7, "publisher/0", "store/nonce", "0f1e")
_RECORD = encode_volume_record(_ADDRESS, "host-a")
_FIELDS = json.loads(_RECORD)
_NO_HOSTNAME = {name: value for name, value in _FIELDS.items() if name != "hostname"}


def test_volume_records_round_trip_as_versioned_json() -> None:
    """Encode a record as versioned JSON and decode it even with brackets in its store name."""
    assert decode_volume_record("publisher/0", _RECORD) == (_ADDRESS, "host-a")
    assert _FIELDS == {"v": 1, "hostname": "host-a", **json.loads(_ADDRESS.to_json())}
    # A store name's brackets sit inside strings, after an escaped quote too, so they never count.
    bracketed = CarrierAddress("dial", 7, "publisher/0", '"[{' * 50 + "/nonce", "0f1e")
    bracketed_data = encode_volume_record(bracketed, "host-a")
    assert decode_volume_record("publisher/0", bracketed_data) == (bracketed, "host-a")


@pytest.mark.parametrize(
    "bad, message",
    [
        (b"[1", "malformed record: .* JSONDecodeError"),
        (b"[1]", "malformed record: .* TypeError"),
        (b"[" * 100_000, "malformed record: .* over 64 brackets"),
        (json.dumps({**_FIELDS, "v": 2}).encode(), "version 2 record, not version 1"),
        # A version is an int, never a str, bool or float equal to one.
        (json.dumps({**_FIELDS, "v": "1"}).encode(), r"wrong type: \['v'\]"),
        (json.dumps({**_FIELDS, "v": True}).encode(), r"wrong type: \['v'\]"),
        (json.dumps({**_FIELDS, "v": 1.0}).encode(), r"wrong type: \['v'\]"),
        (json.dumps(_NO_HOSTNAME).encode(), "KeyError: 'hostname'"),
        (json.dumps({**_FIELDS, "port": "7"}).encode(), r"wrong type: \['port'\]"),
        (
            json.dumps({**_FIELDS, "hostname": 3}).encode(),
            r"wrong type: \['hostname'\]",
        ),
    ],
    ids=[
        "truncated",
        "not_an_object",
        "too_deep",
        "version_2",
        "version_str",
        "version_bool",
        "version_float",
        "no_hostname",
        "port_str",
        "hostname_int",
    ],
)
def test_a_malformed_volume_record_is_refused_by_its_volumes_name(
    bad: bytes, message: str
) -> None:
    """Refuse a record that is not this version's with an error naming the volume."""
    with pytest.raises(
        CarrierProtocolError, match=f"^volume 'publisher/0' .*{message}"
    ):
        decode_volume_record("publisher/0", bad)


def test_a_record_for_another_volume_is_refused() -> None:
    """Refuse a record whose server serves another volume with an error naming the volume asked for."""
    with pytest.raises(
        CarrierProtocolError,
        match="^volume 'publisher/1' has a record for service 'publisher/0'$",
    ):
        decode_volume_record("publisher/1", _RECORD)


def test_a_record_that_is_not_utf8_is_refused_without_its_scope() -> None:
    """Refuse a record that is not UTF-8 without showing it: its scope is the volume's only credential."""
    with pytest.raises(
        CarrierProtocolError, match="malformed record: .* UnicodeDecodeError"
    ) as raised:
        decode_volume_record("publisher/0", _RECORD.replace(b"host-a", b"host-\xff"))
    shown = "".join(traceback.format_exception(raised.type, raised.value, raised.tb))
    assert raised.value.__cause__ is None and "store/nonce" not in shown


_SHARD = TensorSlice((0,), (0,), (4,), (2,), (2,))


@pytest.mark.asyncio
async def test_a_record_dials_the_host_given_then_the_variable_then_this_hosts_name(
    monkeypatch: pytest.MonkeyPatch, volumes: _Volumes
) -> None:
    """Dial ``host``, else ``TORCHSTORE_CARRIER_HOST``, else this host, keeping the core's host name."""
    monkeypatch.delenv("TORCHSTORE_CARRIER_HOST", raising=False)
    # A locality name that does not resolve, so a record must not dial it.
    monkeypatch.setenv("TORCHSTORE_HOSTNAME", f"volume-host-{uuid.uuid4().hex}")
    served = [await volumes.serve()]
    monkeypatch.setenv("TORCHSTORE_CARRIER_HOST", "127.0.0.1")
    served += [await volumes.serve(), await volumes.serve(host="localhost")]
    dial_hosts = []
    for core, server, record in served:
        address, hostname = decode_volume_record(core.volume_id, record)
        assert address == server.address
        # Served under the caller's scope, the volume's only credential.
        assert address.scope == volumes.scope
        assert hostname == (await core.get_id())[1] == os.environ["TORCHSTORE_HOSTNAME"]
        assert await VolumeProxy(address).get_id.call_one() == (
            core.volume_id,
            hostname,
        )
        dial_hosts.append(address.host)
    assert dial_hosts == [socket.gethostname(), "127.0.0.1", "localhost"]


@pytest.mark.asyncio
async def test_a_volume_dialed_at_an_ip_address_still_uses_sharedmemory(
    monkeypatch: pytest.MonkeyPatch, volumes: _Volumes
) -> None:
    """Check locality against the record's host name, ``$TORCHSTORE_HOSTNAME`` first, not the address dialed."""
    monkeypatch.setattr(transport, "SHM_ENABLED", True)
    monkeypatch.setenv("TORCHSTORE_HOSTNAME", f"volume-host-{uuid.uuid4().hex}")
    core, _, record = await volumes.serve(host="127.0.0.1")
    address, hostname = decode_volume_record(core.volume_id, record)
    assert (address.host, hostname) == ("127.0.0.1", os.environ["TORCHSTORE_HOSTNAME"])
    ref = volumes.ref(record, core.volume_id, TransportType.Unset)
    assert transport.get_available_transport(ref) is TransportType.SharedMemory
    source = torch.arange(6.0)
    await _put(ref, "weights", source)
    assert_bitwise_equal(await _get(ref, "weights"), source)
    assert core.names() == _TRANSPORT_CALLS[TransportType.SharedMemory]


async def _proxy(volumes: _Volumes) -> tuple[Any, VolumeProxy, CarrierAddress, str]:
    """Serve a volume and return its core, a proxy to it, its server's address and its host's name."""
    core, _, record = await volumes.serve()
    address, hostname = decode_volume_record(core.volume_id, record)
    return core, VolumeProxy(address), address, hostname


@pytest.mark.asyncio
async def test_a_proxy_offers_each_endpoint_as_call_and_call_one(
    volumes: _Volumes,
) -> None:
    """Reach every endpoint as a Monarch slice of one volume does, and offer nothing else."""
    core, proxy, _, hostname = await _proxy(volumes)
    assert proxy.volume_id == core.volume_id
    public = {name for name in vars(StorageVolumeCore) if not name.startswith("_")}
    assert set(VOLUME_ENDPOINTS) == public
    for name in VOLUME_ENDPOINTS:
        endpoint = getattr(proxy, name)
        assert inspect.iscoroutinefunction(endpoint.call_one), name
        assert inspect.iscoroutinefunction(endpoint.call), name
    assert await proxy.get_id.call_one() == (core.volume_id, hostname)
    assert await proxy.get_id.call() == [(core.volume_id, hostname)]
    core.store.kv.update({"a": torch.ones(1), "b": torch.ones(1), "c": torch.ones(1)})
    meta = await proxy.get_meta.call_one(requests=[Request(key="a")])
    assert meta == [(torch.Size([1]), torch.float32)]
    assert await proxy.delete.call(key="a") == [None]
    assert await proxy.delete_batch.call(["b", "missing"]) == [None]
    assert await proxy.reset.call_one() is None
    assert core.store.kv == {}
    names = ["get_id", "get_id", "get_meta", "delete", "delete_batch", "reset"]
    assert core.names() == names
    for name in ("slice", "call", "size"):
        with pytest.raises(AttributeError):
            getattr(proxy, name)


@pytest.mark.asyncio
async def test_a_dropped_proxy_frees_its_handle_at_once(volumes: _Volumes) -> None:
    """Free a dropped proxy's handle with the cyclic GC off. Its connection stays until the carrier closes."""
    _, _, address, _ = await _proxy(volumes)
    gc.disable()
    try:
        dropped = weakref.ref(VolumeProxy(address)._client)
        assert dropped() is None
    finally:
        gc.enable()


@pytest.mark.parametrize("keyword", ["timeout_s", "endpoint", "retired_store_keys"])
@pytest.mark.asyncio
async def test_a_proxy_refuses_the_keywords_it_owns(
    volumes: _Volumes, keyword: str
) -> None:
    """Refuse the carrier client's own keywords and the proxy's: none reaches a volume."""
    core, proxy, _, _ = await _proxy(volumes)
    with pytest.raises(
        TypeError, match=rf"'get_id' got reserved keywords \['{keyword}'\]"
    ):
        await proxy.get_id.call_one(**{keyword: 1})
    assert core.names() == []


@pytest.mark.asyncio
async def test_a_proxy_does_not_pickle_but_a_buffer_holding_it_does(
    volumes: _Volumes,
) -> None:
    """Refuse to pickle a proxy, while a buffer holding it pickles without it."""
    core, proxy, _, hostname = await _proxy(volumes)
    with pytest.raises(TypeError, match="volume handles do not pickle"):
        pickle.dumps(proxy)
    ref = StorageVolumeRef(
        cast(Any, proxy),
        core.volume_id,
        TransportContext(),
        TransportType.MonarchRPC,
        hostname,
    )
    buffer = pickle.loads(pickle.dumps(MonarchRPCTransportBuffer(ref)))
    assert buffer.storage_volume_ref is None


@pytest.mark.asyncio
async def test_a_proxys_repr_names_its_volume_and_hides_its_scope(
    volumes: _Volumes,
) -> None:
    """Name the volume and where its server listens, but never the scope, the volume's only credential."""
    core, proxy, address, _ = await _proxy(volumes)
    assert f"{core.volume_id!r} at {address.host}:{address.port}" in repr(proxy)
    assert address.scope not in repr(proxy)


@pytest.mark.asyncio
async def test_closing_a_proxy_closes_only_its_own_handle(volumes: _Volumes) -> None:
    """Fail a closed proxy's later calls, while another handle and the volume keep serving."""
    core, proxy, address, hostname = await _proxy(volumes)
    other = VolumeProxy(address)
    await proxy.close()
    with pytest.raises(PeerLostError, match="is closed"):
        await proxy.get_id.call_one()
    assert core.names() == []
    assert await other.get_id.call_one() == (core.volume_id, hostname)


@pytest.mark.parametrize("transport_type", list(_TRANSPORT_CALLS))
@pytest.mark.asyncio
async def test_each_carrier_transport_moves_tensors_slices_and_objects(
    volumes: _Volumes, transport_type: TransportType
) -> None:
    """Put and get a tensor, a DTensor's shard and an object. A round trip makes only its calls."""
    core, server, record = await volumes.serve()
    ref = volumes.ref(record, core.volume_id, transport_type)
    source = torch.arange(12, dtype=torch.float32).view(3, 4)
    await _put(ref, "weights", source)
    assert_bitwise_equal(await _get(ref, "weights"), source)
    assert core.names() == _TRANSPORT_CALLS[transport_type]
    assert server.stats()["calls"] == len(_TRANSPORT_CALLS[transport_type])
    # The volume keeps its own copy of what it stores.
    assert core.store.kv["weights"].data_ptr() != source.data_ptr()
    meta = await ref.volume.get_meta.call_one([Request(key="weights")])
    assert meta == [(source.shape, source.dtype)]

    shard = TensorSlice((2, 0), (1,), (4, 6), (2, 6), (2,))
    local = (torch.arange(12.0).view(2, 6) / 7).bfloat16()
    buffer = transport.create_transport_buffer(ref)
    await buffer.put_to_storage_volume([Request("sharded", local, tensor_slice=shard)])
    buffer = transport.create_transport_buffer(ref)
    [fetched] = await buffer.get_from_storage_volume(
        [Request.from_tensor_slice("sharded", shard)]
    )
    assert_bitwise_equal(fetched, local)
    buffer = transport.create_transport_buffer(ref)
    await buffer.put_to_storage_volume([Request.from_objects("config", {"lr": 0.5})])
    assert await _get(ref, "config") == {"lr": 0.5}


@pytest.mark.asyncio
async def test_payloads_and_sharedmemory_gloo_rpc_buffers_rebuild_remotely() -> None:
    """Rebuild requests, slices, descriptors and each carrier transport's buffer on the server."""
    target = _Target()
    client = CarrierClient((await _serve(target, "echo")).address)
    tensor_slice = TensorSlice((2, 0), (1, 0), (4, 4), (2, 4), (2, 1))
    tensor = torch.arange(8.0).view(2, 4)
    request = Request(
        "k", tensor, tensor_slice=tensor_slice, objects=None, is_object=False
    )
    descriptor = SharedMemoryDescriptor.from_tensor(
        allocate_shared_tensor(torch.Size([2, 3]), torch.bfloat16)
    )
    shm = SharedMemoryTransportBuffer(_NO_REF)
    shm._contexts = [ShmContext(descriptor), ShmContext(objects={"a": 1}, use_rpc=True)]
    pair = GlooTransportBuffer(_NO_REF)
    pair.master_addr, pair.master_port, pair.store_key = "host", 29500, "gloo_1"
    pair.shape, pair.dtype = torch.Size([3]), torch.float16
    rpc = MonarchRPCTransportBuffer(_NO_REF)
    rpc.data = torch.arange(5)
    for value in (request, descriptor, shm, pair, rpc):
        await client.call("echo", value)

    received = dict(zip(("request", "descriptor", "shm", "pair", "rpc"), target.calls))
    assert (received["request"].key, received["request"].is_object) == ("k", False)
    assert received["request"].tensor_slice == tensor_slice
    assert isinstance(received["request"].tensor_slice.coordinates, tuple)
    assert_bitwise_equal(received["request"].tensor_val, tensor)
    assert received["descriptor"] == descriptor
    assert received["shm"]._contexts == shm._contexts
    assert vars(received["pair"]) == vars(pair)
    assert_bitwise_equal(received["rpc"].data, rpc.data)


@pytest.mark.asyncio
async def test_volume_errors_keep_their_type_and_text(volumes: _Volumes) -> None:
    """Raise each volume error at the caller as its own type and text, with its traceback as cause."""
    core, _, record = await volumes.serve()
    ref = volumes.ref(record, core.volume_id, TransportType.MonarchRPC)
    with pytest.raises(KeyError, match="Key 'missing' not found") as missing:
        await _get(ref, "missing")
    assert "Traceback" in str(missing.value.__cause__)
    await _put(ref, "whole", torch.ones(4))
    buffer = transport.create_transport_buffer(ref)
    with pytest.raises(AssertionError, match="Existing data is a regular tensor"):
        await buffer.put_to_storage_volume(
            [Request("whole", torch.ones(2), tensor_slice=_SHARD)]
        )
    buffer = transport.create_transport_buffer(ref)
    await buffer.put_to_storage_volume(
        [Request("sharded", torch.ones(2), tensor_slice=_SHARD)]
    )
    with pytest.raises(RuntimeError, match="contains sharded tensor"):
        await _get(ref, "sharded")


@pytest.mark.asyncio
async def test_an_error_only_the_volume_can_import_arrives_as_remote_error(
    volumes: _Volumes, children: _Children
) -> None:
    """Raise a ``RemoteError`` naming an error class that only the volume's process has."""
    _, volume_id, record = _serve_in_child(children)
    ref = volumes.ref(record, volume_id, TransportType.MonarchRPC)
    with pytest.raises(RemoteError, match="only the volume can import it") as raised:
        await ref.volume.reset.call_one()
    assert raised.value.remote_type == "__main__.ServerOnlyError"
    reply = await ref.volume.get_id.call_one()
    assert reply == (volume_id, ref.volume_hostname)


@pytest.mark.asyncio
async def test_a_volume_runs_one_call_at_a_time_and_a_connections_calls_in_send_order(
    volumes: _Volumes,
) -> None:
    """Never overlap two calls to one volume, and run the calls on one connection in send order.

    Three threads push as TitanRL's trainer does, each with an ``asyncio.run`` per call.
    """
    core, _, record = await volumes.serve()
    ref = volumes.ref(record, core.volume_id, TransportType.MonarchRPC)
    keys = [f"key-{i}" for i in range(6)]
    await asyncio.gather(
        *(_put(ref, key, torch.full((4,), i)) for i, key in enumerate(keys))
    )
    assert core.keys("put") == keys

    def push(thread: int) -> None:
        for i in range(5):
            asyncio.run(_put(ref, f"{thread}/{i}", torch.full((4,), thread * 10 + i)))

    await asyncio.gather(*(asyncio.to_thread(push, thread) for thread in range(3)))
    assert core.max_active == 1
    for thread in range(3):
        for i in range(5):
            stored = core.store.kv[f"{thread}/{i}"]
            assert_bitwise_equal(stored, torch.full((4,), thread * 10 + i))


@pytest.mark.asyncio
async def test_a_reply_waiting_to_be_sent_keeps_its_values_through_an_in_place_put(
    volumes: _Volumes,
) -> None:
    """Send a get's reply as it was when the get ran, though a Gloo put then wrote in place."""
    core, server, record = await volumes.serve()
    rpc = volumes.ref(record, core.volume_id, TransportType.MonarchRPC)
    pair = volumes.ref(record, core.volume_id, TransportType.Gloo)
    old, new = torch.zeros(16 << 20), torch.ones(16 << 20)
    await _put(pair, "weights", old)
    stored = core.store.kv["weights"]
    await server._set_sending(False)
    try:
        reading = asyncio.ensure_future(_get(rpc, "weights"))
        await _until(lambda: core.names()[-1:] == ["get"] and core.active == 0)
        writing = asyncio.ensure_future(_put(pair, "weights", new))
        await _until(lambda: core.names().count("put") == 2 and core.active == 0)
        assert core.store.kv["weights"] is stored and torch.equal(stored, new)
    finally:
        await server._set_sending(True)
    assert_bitwise_equal(await reading, old)
    await writing


class _OtherTransport(TransportBuffer):
    """A transport's buffer that no volume proxy carries, such as NIXL's or TorchComms'."""


@pytest.mark.parametrize("passed", ["positional", "keyword"])
@pytest.mark.asyncio
async def test_a_proxy_refuses_transfers_of_the_transports_it_does_not_carry(
    volumes: _Volumes, passed: str
) -> None:
    """Refuse a handshake, put or get of another transport or of no buffer before the volume sees it.

    A NIXL or TorchComms volume reads and writes caller memory, which a failed call frees at once.
    Such a refusal leaves the client's mappings of the call's keys alone.
    """
    core, _, record = await volumes.serve()
    ref = volumes.ref(record, core.volume_id, TransportType.MonarchRPC)
    mapped = ref.transport_context.get(SharedMemoryCache)
    mapped.allocate("k", torch.Size([4]), torch.float32)
    for endpoint in ("handshake", "put", "get"):
        call = getattr(ref.volume, endpoint).call_one
        buffer, requests = _OtherTransport(ref), [Request(key="k")]
        with pytest.raises(
            TypeError,
            match="carry SharedMemory, Gloo and MonarchRPC transfers, not _OtherTransport",
        ):
            if passed == "positional":
                await call(buffer, requests)
            else:
                await call(transport_buffer=buffer, requests=requests)
        with pytest.raises(TypeError, match="MonarchRPC transfers, not NoneType"):
            if passed == "positional":
                await call()
            else:
                await call(requests=requests)
    assert core.names() == []
    assert [key for key, _ in mapped._storages] == ["k"]


# Gloo's timeout in the failure cases: a stranded transfer ends by then.
_GLOO_TIMEOUT_S = 3


async def _in_own_loop(
    scenario: Callable[[], Coroutine[Any, Any, Any]]
) -> tuple[Any, float]:
    """Run ``scenario`` in an ``asyncio.run`` of its own, as TitanRL pushes, and time that run."""
    start = time.monotonic()
    result = await asyncio.to_thread(asyncio.run, scenario())
    return result, time.monotonic() - start


def _master_listening(port: int) -> bool:
    """Whether a Gloo pair's TCPStore master still accepts connections on ``port``."""
    try:
        socket.create_connection((gloo._get_hostname(), port), timeout=5).close()
    except OSError:
        return False
    return True


def _collected(*weakrefs: weakref.ref[Any]) -> Callable[[], bool]:
    """A condition that runs a cyclic GC, then holds once every referent is gone."""

    def condition() -> bool:
        gc.collect()
        return all(weak() is None for weak in weakrefs)

    return condition


async def _posted_receive(buffer: Any) -> asyncio.Task[Any]:
    """Wait until a get's receive has taken its pair's group and awaits Gloo, then return it.

    A carrier call whose reply came before it waited returns without yielding, so a get could
    fail and drop its pair before the receive's first step took the group.
    """
    receive = buffer._recv_task
    await _until(
        lambda: inspect.getcoroutinestate(receive.get_coro()) != inspect.CORO_CREATED
    )
    assert not receive.done(), receive
    return receive


def _deleting_after_receive_posted(
    ref: StorageVolumeRef, posted: list[tuple[int, asyncio.Task[Any]]]
) -> Callable[[Any, list[Request]], Coroutine[Any, Any, None]]:
    """A ``_pre_get_hook`` that keeps each get's master port and receive once posted, then deletes its key."""
    pre_get_hook = GlooTransportBuffer._pre_get_hook

    async def hook(buffer: Any, requests: list[Request]) -> None:
        await pre_get_hook(buffer, requests)
        posted.append((buffer.master_port, await _posted_receive(buffer)))
        await ref.volume.delete.call_one(requests[0].key)

    return hook


def _recording_halves(
    halves: list[tuple[int, asyncio.Task[Any]]],
) -> Callable[[Any], Coroutine[Any, Any, None]]:
    """A ``_pre_handshake`` that keeps each handshake's master port and client half, which a failed call's buffer drops."""
    pre_handshake = GlooTransportBuffer._pre_handshake

    async def hook(buffer: Any) -> None:
        await pre_handshake(buffer)
        halves.append((buffer.master_port, buffer._pg_task))

    return hook


@pytest.mark.asyncio
async def test_a_gloo_get_after_a_cancelled_one_pairs_anew_and_gets_its_own_bytes(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    volumes: _Volumes,
) -> None:
    """Pair anew after a Gloo get cancelled in the volume's queue, whose receive nothing feeds, then get own bytes.

    The next get runs in the same ``asyncio.run`` while that receive still waits, and that run then
    waits for the receive until Gloo's timeout.
    """
    monkeypatch.setattr(gloo, "TORCHSTORE_GLOO_INIT_TIMEOUT", _GLOO_TIMEOUT_S)
    core, server, record = await volumes.serve()
    ref = volumes.ref(record, core.volume_id, TransportType.Gloo)
    first, second = torch.arange(64.0), torch.arange(64.0) + 100
    await _put(ref, "first", first)
    await _put(ref, "second", second)
    stranded: list[asyncio.Task[Any]] = []
    pre_get_hook = GlooTransportBuffer._pre_get_hook

    holders: list[asyncio.Future[Any]] = []

    async def hold_once_the_receive_posted(buffer: Any, requests: Any) -> None:
        await pre_get_hook(buffer, requests)
        stranded.append(await _posted_receive(buffer))
        # So the get's call then queues behind a held one.
        holders.append(await _hold(core, ref))

    async def cancel_then_get_again() -> None:
        getting = asyncio.ensure_future(_get(ref, "first"))
        await _until(lambda: _counts(server)[1:] == (1, 1))
        getting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await getting
        # The cancelled get forgot its pair while its receive still waits.
        assert not stranded[0].done()
        assert core.volume_id not in gloo._store_addrs
        monkeypatch.setattr(GlooTransportBuffer, "_pre_get_hook", pre_get_hook)
        await _release(server, core, holders)
        again = await _get(ref, "second")
        # On a new pair, whose tags the stranded receive never matches either.
        assert not stranded[0].done()
        assert_bitwise_equal(again, second)

    monkeypatch.setattr(
        GlooTransportBuffer, "_pre_get_hook", hold_once_the_receive_posted
    )
    _, elapsed = await _in_own_loop(cancel_then_get_again)
    assert elapsed < _GLOO_TIMEOUT_S + 5
    assert core.names().count("handshake") == 2
    assert len(ref.transport_context.get(GlooProcessGroupCache)._process_groups) == 1
    # asyncio.run cancelled the receive still waiting, which must log nothing either.
    assert [r.getMessage() for r in caplog.records if r.name == "asyncio"] == []


@pytest.mark.asyncio
async def test_a_refused_gloo_get_forgets_its_pair_which_closes_and_leaves_the_volume_at_its_next_call(
    monkeypatch: pytest.MonkeyPatch, volumes: _Volumes
) -> None:
    """Forget a refused Gloo get's pair, close it without a cyclic GC once the get lets go, and drop its half at the next call.

    A call that never reaches the volume leaves that half to the call after it.
    """
    core, _, record = await volumes.serve()
    ref = volumes.ref(record, core.volume_id, TransportType.Gloo)
    await _put(ref, "lost", torch.zeros(8))
    client = ref.transport_context.get(GlooProcessGroupCache)._process_groups
    volume = core.store.transport_context.get(GlooProcessGroupCache)._process_groups
    [failed] = client
    pre_get_hook = GlooTransportBuffer._pre_get_hook
    posted: list[tuple[int, asyncio.Task[Any]]] = []
    lose_the_key = _deleting_after_receive_posted(ref, posted)
    monkeypatch.setattr(GlooTransportBuffer, "_pre_get_hook", lose_the_key)
    with pytest.raises(KeyError, match="Key 'lost' not found"):
        await _get(ref, "lost")
    monkeypatch.setattr(GlooTransportBuffer, "_pre_get_hook", pre_get_hook)
    [(port, receive)] = posted
    posted.clear()
    # The volume ended the refused get's receive with an empty send.
    await _until(receive.done)
    assert receive.exception() is None
    del receive
    gc.disable()
    try:
        # A finished receive's worker thread holds its pair a moment longer.
        await _until(lambda: _queued(failed) and not _master_listening(port))
    finally:
        gc.enable()
    assert client == {} and list(volume) == [failed]
    closed = VolumeProxy(cast(VolumeProxy, ref.volume)._client.address)
    await closed.close()
    with pytest.raises(PeerLostError, match="is closed"):
        await closed.get_id.call_one()
    assert list(volume) == [failed]
    assert await ref.volume.get_id.call_one() == (core.volume_id, ref.volume_hostname)
    assert volume == {}


@pytest.mark.parametrize("failure", ["get", "handshake"])
@pytest.mark.asyncio
async def test_a_failed_gloo_transfer_in_its_own_run_closes_its_pair_without_gc(
    monkeypatch: pytest.MonkeyPatch, volumes: _Volumes, failure: str
) -> None:
    """Close a failed get's or handshake's pair with GC off once its error left the call's ``asyncio.run``.

    The run, one per call as TitanRL pushes, cancels the stranded half as it ends.
    """
    monkeypatch.setattr(gloo, "TORCHSTORE_GLOO_INIT_TIMEOUT", _GLOO_TIMEOUT_S)
    core, server, record = await volumes.serve()
    ref = volumes.ref(record, core.volume_id, TransportType.Gloo)
    await _put(ref, "lost", torch.zeros(8))
    posted: list[tuple[int, asyncio.Task[Any]]] = []
    holders: list[asyncio.Future[Any]] = []
    call: Callable[[], Coroutine[Any, Any, Any]]
    if failure == "get":
        lose_the_key = _deleting_after_receive_posted(ref, posted)
        monkeypatch.setattr(GlooTransportBuffer, "_pre_get_hook", lose_the_key)
        call, expected = partial(_get, ref, "lost"), KeyError
    else:
        gloo._forget_pairs(core.volume_id, ref.transport_context)
        monkeypatch.setattr(
            GlooTransportBuffer, "_pre_handshake", _recording_halves(posted)
        )
        holders.append(await _hold(core, ref))
        monkeypatch.setattr(runtime, "_CALL_TIMEOUT_S", 2.0)
        call, expected = partial(_put, ref, "next", torch.ones(8)), CallTimeoutError

    def push() -> type[BaseException]:
        try:
            asyncio.run(call())
        except BaseException as e:
            return type(e)
        raise AssertionError("the call did not fail")

    gc.disable()
    try:
        start = time.monotonic()
        assert await asyncio.to_thread(push) is expected
        assert time.monotonic() - start < _GLOO_TIMEOUT_S + 5
        # Only the port: the half kept here would hold the pair.
        port = posted[0][0]
        posted.clear()
        await _until(lambda: not _master_listening(port), timeout_s=15)
    finally:
        gc.enable()
    monkeypatch.setattr(runtime, "_CALL_TIMEOUT_S", 300.0)
    await _release(server, core, holders, abandoned=bool(holders))


@pytest.mark.asyncio
async def test_a_stranded_gloo_half_ends_without_logging_an_unretrieved_error(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    volumes: _Volumes,
) -> None:
    """End a refused get's receive with an empty send, and retrieve the error a handshake's stranded half ends with.

    So no loop logs either. The halves end on the test's long-lived loop, as on a generator's.
    ``asyncio.run`` would cancel them instead.
    """
    monkeypatch.setattr(gloo, "TORCHSTORE_GLOO_INIT_TIMEOUT", _GLOO_TIMEOUT_S)
    core, server, record = await volumes.serve()
    ref = volumes.ref(record, core.volume_id, TransportType.Gloo)
    await _put(ref, "lost", torch.zeros(8))
    posted: list[tuple[int, asyncio.Task[Any]]] = []
    halves: list[tuple[int, asyncio.Task[Any]]] = []
    pre_get_hook = GlooTransportBuffer._pre_get_hook
    monkeypatch.setattr(
        GlooTransportBuffer,
        "_pre_get_hook",
        _deleting_after_receive_posted(ref, posted),
    )
    monkeypatch.setattr(
        GlooTransportBuffer, "_pre_handshake", _recording_halves(halves)
    )
    with pytest.raises(KeyError, match="Key 'lost' not found"):
        await _get(ref, "lost")
    monkeypatch.setattr(GlooTransportBuffer, "_pre_get_hook", pre_get_hook)
    # The failed get forgot the pair, so this put's handshake queues behind a held call.
    holder = await _hold(core, ref)
    pairing = transport.create_transport_buffer(ref)
    putting = asyncio.ensure_future(
        pairing.put_to_storage_volume([Request.from_tensor("next", torch.ones(8))])
    )
    await _until(lambda: _counts(server)[1:] == (1, 1))
    putting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await putting
    # The abandoned handshake never runs, so the volume never joins.
    await _release(server, core, [holder])
    receive, half = posted[0][1], halves[-1][1]
    await _until(lambda: receive.done() and half.done(), timeout_s=30)
    assert receive.exception() is None
    collected = [weakref.ref(receive), weakref.ref(half)]
    del receive, half, pairing, putting
    posted.clear()
    halves.clear()
    # A half whose error nothing retrieved logs it once collected.
    await _until(_collected(*collected))
    assert [r.getMessage() for r in caplog.records if r.name == "asyncio"] == []


@pytest.mark.asyncio
async def test_a_half_that_failed_before_its_call_ends_without_logging_an_unretrieved_error(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    volumes: _Volumes,
) -> None:
    """Retrieve a handshake half's Gloo error when the half failed before its queued call did."""
    monkeypatch.setattr(gloo, "TORCHSTORE_GLOO_INIT_TIMEOUT", 1)
    core, server, record = await volumes.serve()
    ref = volumes.ref(record, core.volume_id, TransportType.Gloo)
    holder = await _hold(core, ref)
    halves: list[tuple[int, asyncio.Task[Any]]] = []
    monkeypatch.setattr(
        GlooTransportBuffer, "_pre_handshake", _recording_halves(halves)
    )
    putting = asyncio.ensure_future(_put(ref, "next", torch.ones(8)))
    await _until(lambda: _counts(server)[1:] == (1, 1))
    # Gloo's 1 s timeout ends the half while its handshake still waits in the queue.
    await _until(lambda: halves[0][1].done())
    putting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await putting
    await _release(server, core, [holder])
    collected = weakref.ref(halves[0][1])
    halves.clear()
    await _until(_collected(collected))
    assert [r.getMessage() for r in caplog.records if r.name == "asyncio"] == []


@pytest.mark.asyncio
async def test_overlapping_gloo_transfers_of_one_client_keep_one_pair_through_failures(
    volumes: _Volumes,
) -> None:
    """Pair once each time overlapping transfers to a volume find no pair at first or after a failed put.

    Each forgotten pair goes from the client once its transfer ends and from the volume at the next call.
    """
    core, _, record = await volumes.serve()
    ref = volumes.ref(record, core.volume_id, TransportType.Gloo)
    core.store.kv["whole"] = torch.ones(4)
    client = ref.transport_context.get(GlooProcessGroupCache)._process_groups
    volume = core.store.transport_context.get(GlooProcessGroupCache)._process_groups
    for round_ in range(3):
        values = [torch.full((8,), float(round_ * 4 + i)) for i in range(4)]
        await asyncio.gather(*(_put(ref, f"k{i}", v) for i, v in enumerate(values)))
        for i, value in enumerate(values):
            assert_bitwise_equal(core.store.kv[f"k{i}"], value)
        if round_ < 2:
            # The volume drains the refused put's send and the proxy still forgets the pair it
            # used. The next round pairs anew.
            with pytest.raises(
                AssertionError, match="Existing data is a regular tensor"
            ):
                await transport.create_transport_buffer(ref).put_to_storage_volume(
                    [Request("whole", torch.ones(2), tensor_slice=_SHARD)]
                )
    address = cast(VolumeProxy, ref.volume)._client.address
    # One pairing a round. Every pair but the cached one retires, and only then are all their keys queued.
    assert core.names().count("handshake") == 3
    await volumes.wait_retired(2)
    assert len(_waiting().get(address, ())) == len(volume) - 1
    await ref.volume.get_id.call_one()
    assert (len(client), len(volume)) == (1, 1)
    gloo._forget_pairs(core.volume_id, ref.transport_context)
    await volumes.wait_retired(3)
    rpc = volumes.ref(record, core.volume_id, TransportType.MonarchRPC)
    await rpc.volume.get_id.call_one()
    assert (client, volume) == ({}, {})


@pytest.mark.parametrize("form", ["reserved", "keywords"])
@pytest.mark.asyncio
async def test_a_refused_or_keyword_transfer_cleans_up_as_a_failed_one(
    volumes: _Volumes, form: str
) -> None:
    """Clean up a transfer refused for a reserved keyword, or given its buffer by keyword, as any failed one.

    A Gloo get forgets its pair, and a SharedMemory put unmaps the segments the volume never attached.
    """
    core, _, record = await volumes.serve()
    pair = volumes.ref(record, core.volume_id, TransportType.Gloo)
    await _put(pair, "kept", torch.ones(4))
    buffer = GlooTransportBuffer(pair)
    assert not buffer.requires_handshake([Request(key="missing")])
    get = cast(VolumeProxy, pair.volume).get.call_one
    if form == "reserved":
        with pytest.raises(TypeError, match="reserved keywords"):
            await get(buffer, [Request(key="missing")], timeout_s=30)
    else:
        with pytest.raises(KeyError, match="Key 'missing' not found"):
            await get(transport_buffer=buffer, requests=[Request(key="missing")])
    assert core.volume_id not in gloo._store_addrs
    assert_bitwise_equal(await _get(pair, "kept"), torch.ones(4))
    assert core.names().count("handshake") == 2
    shm = volumes.ref(record, core.volume_id, TransportType.SharedMemory)
    mapped = shm.transport_context.get(SharedMemoryCache)
    mapped.allocate("weights", torch.Size([8]), torch.float32)
    requests = [Request(key="weights")]
    put = cast(VolumeProxy, shm.volume).put.call_one
    if form == "reserved":
        with pytest.raises(TypeError, match="reserved keywords"):
            await put(SharedMemoryTransportBuffer(shm), requests, endpoint="put")
    else:
        # The buffer carries no segment for the request, which the volume refuses.
        with pytest.raises(ValueError, match="zip"):
            await put(
                transport_buffer=SharedMemoryTransportBuffer(shm), requests=requests
            )
    assert list(mapped._storages) == []


@pytest.mark.asyncio
async def test_a_failed_call_on_another_pair_or_transport_keeps_the_gloo_pair(
    volumes: _Volumes,
) -> None:
    """Keep a volume's cached Gloo pair when a get on an older pair or another transport fails."""
    core, _, record = await volumes.serve()
    pair = volumes.ref(record, core.volume_id, TransportType.Gloo)
    await _put(pair, "kept", torch.ones(4))
    cached = gloo._store_addrs[core.volume_id]
    # A strategy's buffers share its proxy and transport context across transports.
    rpc = StorageVolumeRef(
        pair.volume,
        core.volume_id,
        pair.transport_context,
        TransportType.MonarchRPC,
        pair.volume_hostname,
    )
    with pytest.raises(KeyError, match="Key 'missing' not found"):
        await _get(rpc, "missing")
    older = GlooTransportBuffer(pair)
    older.store_key = "torchstore_gloo_older"
    with pytest.raises(KeyError, match="Key 'missing' not found"):
        await pair.volume.get.call_one(older, [Request(key="missing")])
    assert gloo._store_addrs[core.volume_id] == cached
    assert_bitwise_equal(await _get(pair, "kept"), torch.ones(4))
    assert core.names().count("handshake") == 1
    assert len(pair.transport_context.get(GlooProcessGroupCache)._process_groups) == 1


@pytest.mark.asyncio
async def test_a_gloo_transfer_pairs_anew_when_its_context_lost_the_cached_pair(
    volumes: _Volumes,
) -> None:
    """Pair anew instead of failing when the address cache names a pair its context no longer holds."""
    core, _, record = await volumes.serve()
    ref = volumes.ref(record, core.volume_id, TransportType.Gloo)
    await _put(ref, "weights", torch.ones(4))
    cached = gloo._store_addrs[core.volume_id]
    assert ref.transport_context.get(GlooProcessGroupCache).pop(cached[2]) is not None
    assert_bitwise_equal(await _get(ref, "weights"), torch.ones(4))
    assert core.names().count("handshake") == 2
    assert gloo._store_addrs[core.volume_id] != cached


@pytest.mark.parametrize("served", ["one", "same_id"])
@pytest.mark.asyncio
async def test_a_second_live_gloo_client_of_a_volume_id_raises_and_strands_no_pair(
    volumes: _Volumes, served: str
) -> None:
    """Refuse a second live client's Gloo transfer to a volume ID, of one volume or of two that share it.

    Two stores whose publishers share an ID do the latter. Pairing anew at every switch would strand
    each displaced pair's halves on the client and the volume.
    """
    volume_id = f"volume-{uuid.uuid4()}"
    cores = [StorageVolumeCore(volume_id) for _ in range(1 if served == "one" else 2)]
    records = [(await volumes.serve(core))[2] for core in cores]
    refs = [
        volumes.ref(records[i % len(records)], volume_id, TransportType.Gloo)
        for i in range(2)
    ]
    for round_ in range(3):
        await _put(refs[0], "weights", torch.full((4,), float(round_)))
        with pytest.raises(gloo.GlooPairInUseError, match=_IN_USE):
            await _put(refs[1], "weights", torch.ones(4))
    assert_bitwise_equal(await _get(refs[0], "weights"), torch.full((4,), 2.0))
    clients = [
        len(ref.transport_context.get(GlooProcessGroupCache)._process_groups)
        for ref in refs
    ]
    halves = [
        len(core.store.transport_context.get(GlooProcessGroupCache)._process_groups)
        for core in cores
    ]
    assert (clients, halves) == ([1, 0], [1, 0][: len(cores)])


@pytest.mark.asyncio
async def test_a_second_live_gloo_client_moves_objects_and_its_failed_ones_keep_the_pair(
    monkeypatch: pytest.MonkeyPatch, volumes: _Volumes
) -> None:
    """Move a second live Gloo client's objects, which never touch the pair, but refuse its tensors.

    Object transfers of that client that the volume fails leave the first client's pair cached.
    """
    core, _, record = await volumes.serve()
    first, second = (
        volumes.ref(record, core.volume_id, TransportType.Gloo) for _ in range(2)
    )
    await _put(first, "weights", torch.ones(4))
    cached = gloo._store_addrs[core.volume_id]

    async def put(key: str, value: Any) -> None:
        buffer = transport.create_transport_buffer(second)
        await buffer.put_to_storage_volume([Request.from_objects(key, value)])

    async def get(key: str) -> Any:
        buffer = transport.create_transport_buffer(second)
        [value] = await buffer.get_from_storage_volume(
            [Request(key=key, is_object=True)]
        )
        return value

    await put("config", {"lr": 0.5})
    assert await get("config") == {"lr": 0.5}
    with pytest.raises(gloo.GlooPairInUseError, match=_IN_USE):
        await _put(second, "weights", torch.zeros(4))

    async def failing(method: Any, buffer: Any, requests: list[Request]) -> Any:
        if requests[0].key == "config":
            raise ValueError("the volume failed this transfer")
        return await method(buffer, requests)

    monkeypatch.setattr(core.store, "put", partial(failing, core.store.put))
    monkeypatch.setattr(core.store, "get", partial(failing, core.store.get))
    with pytest.raises(ValueError, match="failed this transfer"):
        await put("config", {"lr": 1.0})
    with pytest.raises(ValueError, match="failed this transfer"):
        await get("config")
    assert gloo._store_addrs[core.volume_id] == cached
    assert_bitwise_equal(await _get(first, "weights"), torch.ones(4))
    assert core.names().count("handshake") == 1
    assert second.transport_context.get(GlooProcessGroupCache)._process_groups == {}


@pytest.mark.asyncio
async def test_a_failed_object_after_a_tensor_in_one_gloo_call_keeps_the_pair(
    monkeypatch: pytest.MonkeyPatch, volumes: _Volumes
) -> None:
    """Keep the cached pair when an object that follows a tensor through one buffer fails.

    One buffer moves each of a call's requests in turn, as a ``put_batch`` of a volume's keys does.
    """
    core, _, record = await volumes.serve()
    ref = volumes.ref(record, core.volume_id, TransportType.Gloo)
    await _put(ref, "weights", torch.ones(4))
    await transport.create_transport_buffer(ref).put_to_storage_volume(
        [Request.from_objects("config", {"lr": 0.5})]
    )
    cached = gloo._store_addrs[core.volume_id]

    async def failing(method: Any, buffer: Any, requests: list[Request]) -> Any:
        if requests[0].key == "config":
            raise ValueError("the volume failed this transfer")
        return await method(buffer, requests)

    monkeypatch.setattr(core.store, "put", partial(failing, core.store.put))
    buffer = transport.create_transport_buffer(ref)
    with pytest.raises(ValueError, match="failed this transfer"):
        await buffer.put_to_storage_volume(
            [
                Request.from_tensor("weights", torch.full((4,), 2.0)),
                Request.from_objects("config", {"lr": 1.0}),
            ]
        )
    assert gloo._store_addrs[core.volume_id] == cached
    groups = ref.transport_context.get(GlooProcessGroupCache)._process_groups
    assert list(groups) == [cached[2]]
    await _put(ref, "weights", torch.arange(4.0))
    assert core.names().count("handshake") == 1


class _CountsHalvesAtHandshake(_InstrumentedCore):
    """A volume core that records how many Gloo halves it holds as each handshake starts."""

    def __init__(self) -> None:
        super().__init__()
        self.halves_at_handshake: list[int] = []

    async def _run(
        self,
        name: str,
        method: Callable[..., Awaitable[Any]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if name == "handshake":
            groups = self.store.transport_context.get(GlooProcessGroupCache)
            self.halves_at_handshake.append(len(groups._process_groups))
        return await super()._run(name, method, *args, **kwargs)


@pytest.mark.asyncio
async def test_successive_gloo_clients_of_one_volume_leave_it_one_half(
    volumes: _Volumes,
) -> None:
    """Retire the pair of a client since collected as the next client of that volume pairs anew.

    So a client re-made while its carrier volume lives leaves that volume one half, not one per client.
    The volume never holds two: that half leaves before the next client's handshake runs.
    """
    core, _, record = await volumes.serve(_CountsHalvesAtHandshake())
    address, hostname = decode_volume_record(core.volume_id, record)
    volume = core.store.transport_context.get(GlooProcessGroupCache)._process_groups
    for value in range(3):
        client = StorageVolumeRef(
            cast(Any, VolumeProxy(address)),
            core.volume_id,
            TransportContext(),
            TransportType.Gloo,
            hostname,
        )
        await _put(client, "weights", torch.full((4,), float(value)))
        assert len(volume) == 1
        groups = client.transport_context.get(GlooProcessGroupCache)._process_groups
        [group] = groups.values()
        collected = threading.Event()
        weakref.finalize(group, collected.set)
        del client, groups, group
        # The executor thread that ran the put's send can let go of its group a little after the put returns.
        assert await asyncio.to_thread(collected.wait, 30)
    assert_bitwise_equal(core.store.kv["weights"], torch.full((4,), 2.0))
    assert core.names().count("handshake") == 3
    assert core.halves_at_handshake == [0, 0, 0]
    gloo._forget_pairs(core.volume_id)


@pytest.mark.asyncio
async def test_a_forgotten_gloo_pair_retires_to_its_own_volume_not_another_of_its_id(
    volumes: _Volumes,
) -> None:
    """Send a forgotten pair's key to the server holding its half, never another volume of its ID.

    Two stores whose publishers share an ID serve two such volumes in one process.
    """
    volume_id = f"volume-{uuid.uuid4()}"
    cores = [StorageVolumeCore(volume_id) for _ in range(2)]
    records = [(await volumes.serve(core))[2] for core in cores]
    pairing = volumes.ref(records[0], volume_id, TransportType.Gloo)
    other = volumes.ref(records[1], volume_id, TransportType.MonarchRPC)
    halves = [
        core.store.transport_context.get(GlooProcessGroupCache)._process_groups
        for core in cores
    ]
    await _put(pairing, "weights", torch.ones(4))
    assert [len(held) for held in halves] == [1, 0]
    await _forget_and_queue(volumes, volume_id, pairing)
    await other.volume.get_id.call_one()
    assert [len(held) for held in halves] == [1, 0]
    await pairing.volume.get_id.call_one()
    assert [len(held) for held in halves] == [0, 0]


@pytest.mark.asyncio
async def test_keys_a_failed_call_took_return_to_their_own_server_not_another_of_its_id(
    volumes: _Volumes,
) -> None:
    """Give a failed call's retired keys back to the server holding their halves, never another volume of its ID."""
    volume_id = f"volume-{uuid.uuid4()}"
    cores = [StorageVolumeCore(volume_id) for _ in range(2)]
    records = [(await volumes.serve(core))[2] for core in cores]
    pairing = volumes.ref(records[0], volume_id, TransportType.Gloo)
    other = volumes.ref(records[1], volume_id, TransportType.MonarchRPC)
    halves = [
        core.store.transport_context.get(GlooProcessGroupCache)._process_groups
        for core in cores
    ]
    await _put(pairing, "weights", torch.ones(4))
    await _forget_and_queue(volumes, volume_id, pairing)
    # A call that never reaches the volume takes the retired key, then gives it back.
    closed = VolumeProxy(cast(VolumeProxy, pairing.volume)._client.address)
    await closed.close()
    with pytest.raises(PeerLostError, match="is closed"):
        await closed.get_id.call_one()
    await other.volume.get_id.call_one()
    assert [len(held) for held in halves] == [1, 0]
    await pairing.volume.get_id.call_one()
    assert [len(held) for held in halves] == [0, 0]


@pytest.mark.parametrize("passed", ["positional", "keyword"])
@pytest.mark.asyncio
async def test_a_failed_handshakes_half_retires_to_its_own_volume_not_another_of_its_id(
    monkeypatch: pytest.MonkeyPatch, volumes: _Volumes, passed: str
) -> None:
    """Send a failed handshake's key to the server that ran it, never another volume of its ID.

    So too when the handshake's buffer comes by keyword.
    """
    monkeypatch.setattr(gloo, "TORCHSTORE_GLOO_INIT_TIMEOUT", 30)
    # Kept to the end: a client half freed as soon as it returns can close the pair before the volume's late
    # connect ends.
    halves: list[tuple[int, asyncio.Task[Any]]] = []
    monkeypatch.setattr(
        GlooTransportBuffer, "_pre_handshake", _recording_halves(halves)
    )
    held = _InstrumentedCore(held={"handshake"})
    _, server, record = await volumes.serve(held)
    _, _, other_record = await volumes.serve(StorageVolumeCore(held.volume_id))
    ref = volumes.ref(record, held.volume_id, TransportType.Gloo)
    other = volumes.ref(other_record, held.volume_id, TransportType.MonarchRPC)
    if passed == "keyword":
        handshake = cast(VolumeProxy, ref.volume).handshake
        monkeypatch.setattr(
            ref.volume,
            "handshake",
            SimpleNamespace(
                call_one=lambda buffer, requests: handshake.call_one(
                    transport_buffer=buffer, requests=requests
                )
            ),
        )
    putting = asyncio.ensure_future(_put(ref, "first", torch.ones(8)))
    await _until(lambda: held.names() == ["handshake"] and held.active == 1)
    putting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await putting
    held.release.set()
    await _until(lambda: server.stats()["running"] == 0, timeout_s=30)
    volume = held.store.transport_context.get(GlooProcessGroupCache)._process_groups
    [key] = volume
    assert len(volume) == 1
    await other.volume.get_id.call_one()
    assert len(volume) == 1
    await ref.volume.get_id.call_one()
    # The key retired to its server, which no longer needs naming.
    assert volume == {} and key not in carrier_volumes._pair_servers


@pytest.mark.asyncio
async def test_a_gloo_transfer_reuses_a_pair_another_thread_forgets_after_its_read(
    monkeypatch: pytest.MonkeyPatch, volumes: _Volumes
) -> None:
    """Reuse a pair whose address another thread forgot after this transfer read it."""
    core, _, record = await volumes.serve()
    ref = volumes.ref(record, core.volume_id, TransportType.Gloo)
    await _put(ref, "weights", torch.ones(4))
    pairs = {core.volume_id: gloo._store_addrs.pop(core.volume_id)}
    monkeypatch.setattr(gloo, "_store_addrs", _ForgottenAfterRead(pairs))
    assert_bitwise_equal(await _get(ref, "weights"), torch.ones(4))
    assert core.names().count("handshake") == 1


def test_queued_retired_keys_wait_for_a_bounded_number_of_servers_and_none_in_a_forked_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the queued keys of the newest ``_RETIRED_SERVERS`` servers, hand a server's keys over whole and give a child none.

    A process that re-makes stores leaves keys toward servers it never calls again.
    """
    # Restore every global the fork hook replaces and each live host's lock after the test.
    for host in list(carrier_volumes._hosts):
        monkeypatch.setattr(host, "_lock", host._lock)
    monkeypatch.setattr(carrier_volumes, "_retired_lock", threading.Lock())
    monkeypatch.setattr(carrier_volumes, "_retired", {})
    monkeypatch.setattr(carrier_volumes, "_retiring", collections.deque())
    monkeypatch.setattr(carrier_volumes, "_pair_servers", {})
    monkeypatch.setattr(carrier_volumes, "_publisher_lock", threading.Lock())
    monkeypatch.setattr(carrier_volumes, "_publishers", {})
    monkeypatch.setattr(carrier_volumes, "_inherited", [])
    monkeypatch.setattr(carrier_volumes, "_RETIRED_SERVERS", 2)
    servers = [dataclasses.replace(_ADDRESS, port=port) for port in (1, 2, 3, 4)]
    for n, server in enumerate(servers):
        carrier_volumes._pair_servers[f"torchstore_gloo_{n}"] = server
        carrier_volumes._retire("publisher/0", f"torchstore_gloo_{n}")
    assert _taken(servers[3]) == ["torchstore_gloo_3"]
    assert list(carrier_volumes._retired) == servers[1:3]
    lock = carrier_volumes._retired_lock
    carrier_volumes._pair_servers["torchstore_gloo_4"] = servers[0]
    carrier_volumes._retire("publisher/0", "torchstore_gloo_4")
    # A pair whose key never retired.
    carrier_volumes._pair_servers["torchstore_gloo_paired"] = servers[0]
    carrier_volumes._after_fork_in_child()
    assert carrier_volumes._retired_lock is not lock
    assert carrier_volumes._retired == carrier_volumes._pair_servers == {}
    assert not carrier_volumes._retiring


def test_a_call_takes_its_servers_keys_before_the_oldest_other_servers_lose_theirs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hand a call to the server whose keys waited longest those keys, then drop the oldest others past the bound."""
    monkeypatch.setattr(carrier_volumes, "_retired", {})
    monkeypatch.setattr(carrier_volumes, "_retiring", collections.deque())
    monkeypatch.setattr(carrier_volumes, "_pair_servers", {})
    monkeypatch.setattr(carrier_volumes, "_RETIRED_SERVERS", 2)
    servers = [dataclasses.replace(_ADDRESS, port=port) for port in (1, 2, 3, 4)]
    for n, server in enumerate(servers):
        _retire_toward(server, f"torchstore_gloo_{n}")
    assert _taken(servers[0]) == ["torchstore_gloo_0"]
    assert _waiting() == {
        servers[2]: ["torchstore_gloo_2"],
        servers[3]: ["torchstore_gloo_3"],
    }


def test_keys_given_up_toward_a_thousand_volumes_at_once_each_reach_their_volume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep each key given up toward 1,024 volumes at once until a call to its volume takes it.

    A client that paired with every volume of a large job gives up all those pairs together when it goes.
    """
    monkeypatch.setattr(carrier_volumes, "_retired", {})
    monkeypatch.setattr(carrier_volumes, "_retiring", collections.deque())
    monkeypatch.setattr(carrier_volumes, "_pair_servers", {})
    servers = [dataclasses.replace(_ADDRESS, port=port) for port in range(1, 1025)]
    for n, server in enumerate(servers):
        _retire_toward(server, f"torchstore_gloo_{n}")
    assert [_taken(server) for server in servers] == [
        [f"torchstore_gloo_{n}"] for n in range(1024)
    ]


@pytest.mark.asyncio
async def test_keys_a_failed_call_gives_back_keep_their_age(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drop the keys a failed call gave back before younger ones once more servers have keys waiting than the bound."""
    monkeypatch.setattr(carrier_volumes, "_retired", {})
    monkeypatch.setattr(carrier_volumes, "_retiring", collections.deque())
    monkeypatch.setattr(carrier_volumes, "_pair_servers", {})
    monkeypatch.setattr(carrier_volumes, "_RETIRED_SERVERS", 2)
    servers = [dataclasses.replace(_ADDRESS, port=port) for port in (1, 2, 3, 4)]
    _retire_toward(servers[0], "torchstore_gloo_0")
    _retire_toward(servers[1], "torchstore_gloo_1")

    class Stopped:
        """A carrier client of a stopped volume, whose calls fail once sent."""

        address = servers[0]

        def __init__(self) -> None:
            self.sent: list[str] = []

        async def call(
            self, endpoint: str, *, retired_store_keys: list[str], **kwargs: Any
        ) -> Any:
            self.sent.extend(retired_store_keys)
            raise RuntimeError("the volume stopped")

    stopped = Stopped()
    with pytest.raises(RuntimeError, match="the volume stopped"):
        await carrier_volumes._call(cast(CarrierClient, stopped), "get_id", (), {})
    assert stopped.sent == ["torchstore_gloo_0"]
    _retire_toward(servers[2], "torchstore_gloo_2")
    _retire_toward(servers[3], "torchstore_gloo_3")
    assert _taken(servers[3]) == ["torchstore_gloo_3"]
    assert _waiting() == {
        servers[1]: ["torchstore_gloo_1"],
        servers[2]: ["torchstore_gloo_2"],
    }


def test_a_call_that_starts_while_another_queues_its_servers_key_takes_that_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hand a key that one call queues to the call to its server that started meanwhile, and to that call alone.

    The second call waits for the queues' lock. Without it the second call took the key's queue before the key.
    """
    first, second = (dataclasses.replace(_ADDRESS, port=port) for port in (1, 2))
    queuing, moved = threading.Event(), threading.Event()
    lock = threading.Lock()
    threads: dict[str, threading.Thread] = {}

    class NotedLock:
        """The queues' lock, noting when the second call starts to wait for it."""

        def __enter__(self) -> None:
            if threading.current_thread() is threads["second"]:
                moved.set()
            lock.acquire()

        def __exit__(self, *exc_info: Any) -> None:
            lock.release()

    class Queues(dict):
        """The queued keys, holding the first call as it queues the key until the second call moved."""

        def setdefault(self, where: Any, default: Any = None) -> Any:
            queue = super().setdefault(where, default)
            if where == second and threading.current_thread() is threads["first"]:
                queuing.set()
                assert moved.wait(30)
            return queue

    monkeypatch.setattr(carrier_volumes, "_retired", Queues())
    monkeypatch.setattr(carrier_volumes, "_retiring", collections.deque())
    monkeypatch.setattr(carrier_volumes, "_pair_servers", {})
    monkeypatch.setattr(carrier_volumes, "_retired_lock", NotedLock())
    _retire_toward(second, "torchstore_gloo_raced")
    taken: dict[str, list[str]] = {}

    def take(name: str, server: CarrierAddress) -> None:
        taken[name] = _taken(server)
        moved.set()

    for name, server in (("first", first), ("second", second)):
        threads[name] = threading.Thread(target=take, args=(name, server), daemon=True)
    threads["first"].start()
    try:
        assert queuing.wait(30), "the first call never queued the key"
        threads["second"].start()
    except BaseException:
        moved.set()
        raise
    finally:
        for thread in threads.values():
            if thread.ident is not None:
                thread.join(30)
    assert taken == {"first": [], "second": ["torchstore_gloo_raced"]}
    assert _waiting() == {}


# Forks while a thread holds the retired keys' lock, with a key queued under its server, one given up and not yet
# queued, and a pair whose key never retired. The child takes the lock, which waits on the copy it inherited unless
# the fork hook replaced it, and checks that it holds none of its parent's keys or pairs. Prints its exit code.
_FORK_WITH_RETIRED_KEYS = """
import os, sys, threading

sys.modules["monarch"] = None
sys.path.insert(0, sys.argv[1])
from torchstore import carrier_volumes
from torchstore.carrier import CarrierAddress

assert carrier_volumes.__file__.startswith(sys.argv[1]), carrier_volumes.__file__
server = CarrierAddress("host", 1, "publisher/0", "store/nonce", "0f1e")
for store_key in ("torchstore_gloo_queued", "torchstore_gloo_waiting", "torchstore_gloo_paired"):
    carrier_volumes._pair_servers[store_key] = server
carrier_volumes._retire("publisher/0", "torchstore_gloo_queued")
# A call to another server queues that key under its own.
carrier_volumes._take_retired(CarrierAddress("host", 2, "publisher/1", "store/nonce", "0f1e"))
carrier_volumes._retire("publisher/0", "torchstore_gloo_waiting")
assert carrier_volumes._retired and carrier_volumes._retiring and carrier_volumes._pair_servers
held, release = threading.Event(), threading.Event()


def hold():
    with carrier_volumes._retired_lock:
        held.set()
        release.wait(30)


threading.Thread(target=hold, daemon=True).start()
assert held.wait(30)
pid = os.fork()
if pid == 0:
    fresh = carrier_volumes._retired_lock.acquire(timeout=5)
    forgot = not (carrier_volumes._retired or carrier_volumes._retiring or carrier_volumes._pair_servers)
    os._exit(0 if fresh and forgot else 1)
release.set()
_, status = os.waitpid(pid, 0)
print(os.waitstatus_to_exitcode(status), flush=True)
"""


def test_a_child_forked_while_a_thread_holds_the_retired_keys_lock_starts_with_none_of_its_parents(
    children: _Children,
) -> None:
    """Give a child forked while a thread holds the retired keys' lock a lock of its own and none of its parent's keys.

    Nor its parent's pairs: the parent's volume halves are not the child's to drop.
    """
    child = children.spawn(_FORK_WITH_RETIRED_KEYS)
    out, err = child.communicate(timeout=50)
    assert out.splitlines()[-1:] == ["0"], err[-2000:]


def test_a_pair_no_proxy_made_queues_its_key_nowhere(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Queue no key for a pair that no proxy's handshake made: no carrier volume holds its half."""
    monkeypatch.setattr(carrier_volumes, "_retired", {})
    monkeypatch.setattr(carrier_volumes, "_retiring", collections.deque())
    carrier_volumes._retire("volume-paired-elsewhere", "torchstore_gloo_elsewhere")
    assert _waiting() == {}


def test_keys_given_up_under_the_pair_cache_lock_never_wait_for_a_call_taking_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Give keys up on a thread holding Gloo's pair-cache lock while this one takes keys, then take that lock here.

    A GC closing an abandoned call here would take it. Giving a key up under the queues' lock deadlocked.
    """
    taken, unheld = "torchstore_gloo_taken", "torchstore_gloo_unheld"
    monkeypatch.setattr(carrier_volumes, "_retired", {})
    monkeypatch.setattr(carrier_volumes, "_retiring", collections.deque())
    monkeypatch.setattr(
        carrier_volumes, "_pair_servers", {taken: _ADDRESS, unheld: _ADDRESS}
    )
    monkeypatch.setattr(
        gloo, "_store_addrs", {_ADDRESS.service: ("host", 1, unheld, True)}
    )
    carrier_volumes._retire(_ADDRESS.service, taken)

    class Unanswered:
        """A carrier client whose calls never return."""

        address = _ADDRESS

        async def call(self, *args: Any, **kwargs: Any) -> Any:
            await _suspend()

    call = carrier_volumes._call(cast(CarrierClient, Unanswered()), "get_id", (), {})
    # It takes the queued key and then waits in the client's call.
    call.send(None)
    holding = threading.Event()

    def give_up() -> None:
        with gloo._cache_lock:
            holding.set()
            gloo.forget_unheld_pair(_ADDRESS.service)
            # As a GC closing the abandoned call does: it gives back the key it took.
            call.close()

    thread = threading.Thread(target=give_up, daemon=True)
    try:
        with carrier_volumes._retired_lock:
            thread.start()
            assert holding.wait(30)
            acquired = gloo._cache_lock.acquire(timeout=30)
            assert acquired, "giving a key up waited for the queues' lock"
            gloo._cache_lock.release()
    finally:
        thread.join(30)
    assert not thread.is_alive()
    # The key the call gave back is the older.
    assert _taken(_ADDRESS) == [taken, unheld]


@pytest.mark.parametrize("failure", ["expired", "cancelled"])
@pytest.mark.asyncio
async def test_a_failed_gloo_put_keeps_its_own_error_and_pairs_anew(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    volumes: _Volumes,
    failure: str,
) -> None:
    """End a Gloo put whose queued call expired or was cancelled with that error, once its send ends.

    The put waits for its send, so Gloo's timeout bounds it, and Gloo's error is only logged.
    """
    monkeypatch.setattr(gloo, "TORCHSTORE_GLOO_INIT_TIMEOUT", _GLOO_TIMEOUT_S)
    core, server, record = await volumes.serve()
    ref = volumes.ref(record, core.volume_id, TransportType.Gloo)
    await _put(ref, "first", torch.zeros(64))
    holder = await _hold(core, ref)
    if failure == "expired":
        monkeypatch.setattr(runtime, "_CALL_TIMEOUT_S", 2.0)

    async def fail_a_put() -> type[BaseException]:
        putting = asyncio.ensure_future(_put(ref, "second", torch.ones(64)))
        if failure == "cancelled":
            await _until(lambda: _counts(server)[1:] == (1, 1))
            putting.cancel()
        try:
            await putting
        except BaseException as e:
            return type(e)
        raise AssertionError("the put did not fail")

    pre_put_hook = GlooTransportBuffer._pre_put_hook
    sends: list[asyncio.Task[Any]] = []

    async def record_the_send(buffer: Any, requests: Any) -> None:
        await pre_put_hook(buffer, requests)
        sends.append(buffer._send_task)

    monkeypatch.setattr(GlooTransportBuffer, "_pre_put_hook", record_the_send)
    error, elapsed = await _in_own_loop(fail_a_put)
    monkeypatch.setattr(GlooTransportBuffer, "_pre_put_hook", pre_put_hook)
    expected = CallTimeoutError if failure == "expired" else asyncio.CancelledError
    assert error is expected
    assert elapsed < _GLOO_TIMEOUT_S + 5
    # The put returned only once Gloo timed out its send.
    assert isinstance(sends[0].exception(), RuntimeError)
    assert "GlooTransportBuffer.drop() failed after its call raised" in caplog.text
    assert core.volume_id not in gloo._store_addrs
    monkeypatch.setattr(runtime, "_CALL_TIMEOUT_S", 300.0)
    await _release(server, core, [holder])
    # The abandoned put never ran, and the next one pairs anew.
    assert core.keys("put") == ["first"]
    await _put(ref, "second", torch.ones(64))
    assert_bitwise_equal(await _get(ref, "second"), torch.ones(64))
    assert core.names().count("handshake") == 2
    assert len(ref.transport_context.get(GlooProcessGroupCache)._process_groups) == 1


@pytest.mark.asyncio
async def test_a_gloo_put_the_volume_refused_raises_its_error_once_drained_and_pairs_anew(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    volumes: _Volumes,
) -> None:
    """Raise the volume's error for a Gloo put it refused, once the volume drained its send.

    The proxy still forgets the pair the put used, so the next put pairs anew and stores its own bytes.
    """
    # Bounds the put if the volume ever stops draining refused sends.
    monkeypatch.setattr(gloo, "TORCHSTORE_GLOO_INIT_TIMEOUT", _GLOO_TIMEOUT_S)
    core, _, record = await volumes.serve()
    ref = volumes.ref(record, core.volume_id, TransportType.Gloo)
    await _put(ref, "whole", torch.ones(4))
    pre_put_hook = GlooTransportBuffer._pre_put_hook
    sends: list[asyncio.Task[Any]] = []

    async def record_the_send(buffer: Any, requests: Any) -> None:
        await pre_put_hook(buffer, requests)
        sends.append(buffer._send_task)

    monkeypatch.setattr(GlooTransportBuffer, "_pre_put_hook", record_the_send)
    buffer = transport.create_transport_buffer(ref)
    with pytest.raises(AssertionError, match="Existing data is a regular tensor"):
        await buffer.put_to_storage_volume(
            [Request("whole", torch.full((2,), 7.0), tensor_slice=_SHARD)]
        )
    monkeypatch.setattr(GlooTransportBuffer, "_pre_put_hook", pre_put_hook)
    # The volume took the tensor into scratch, so the send ended without an error.
    assert sends[0].done() and sends[0].exception() is None
    assert "drop() failed" not in caplog.text
    assert core.volume_id not in gloo._store_addrs
    await _put(ref, "other", torch.full((2,), 1.0))
    assert_bitwise_equal(core.store.kv["other"], torch.full((2,), 1.0))
    assert core.names().count("handshake") == 2


async def _held_once_done(task: asyncio.Task[Any]) -> None:
    """Wait for ``task``, then forever, so a wait on this outlasts ``task``."""
    await task
    await asyncio.Event().wait()


@pytest.mark.parametrize("failure", ["expired", "cancelled_pairing"])
@pytest.mark.asyncio
async def test_a_failed_gloo_handshake_raises_at_once_and_pairs_anew(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    volumes: _Volumes,
    failure: str,
) -> None:
    """Fail a Gloo handshake whose queued call expired at once, while the client's half still waits.

    Or cancel one while it waits for that half, which the cancel then cancels. Neither caches anything,
    and the next put pairs anew. An expired one's waiting half keeps a worker thread until Gloo's
    timeout, which the put's ``asyncio.run`` waits for.
    """
    monkeypatch.setattr(gloo, "TORCHSTORE_GLOO_INIT_TIMEOUT", _GLOO_TIMEOUT_S)
    core, server, record = await volumes.serve()
    ref = volumes.ref(record, core.volume_id, TransportType.Gloo)
    # A process's first Gloo pairing can take seconds on its own, so pair and forget first.
    await _put(ref, "warm", torch.ones(8))
    # The volume drops its half of the warm-up pair only once its group is freed.
    await _forget_and_queue(volumes, core.volume_id, ref)
    core.calls.clear()
    queued = failure == "expired"
    holders = [await _hold(core, ref)] if queued else []
    if failure == "expired":
        monkeypatch.setattr(runtime, "_CALL_TIMEOUT_S", 2.0)
    pre_handshake = GlooTransportBuffer._pre_handshake
    post_handshake = GlooTransportBuffer._post_handshake
    halves: list[tuple[int, asyncio.Task[Any]]] = []
    pairing: list[GlooTransportBuffer] = []

    async def hold_the_half(buffer: Any, *args: Any) -> None:
        # The cancel then lands where _post_handshake waits for the client's half.
        buffer._pg_task = asyncio.ensure_future(_held_once_done(buffer._pg_task))
        pairing.append(buffer)
        await post_handshake(buffer, *args)

    async def fail_a_put() -> tuple[type[BaseException], bool]:
        putting = asyncio.ensure_future(_put(ref, "first", torch.ones(8)))
        if failure == "cancelled_pairing":
            await _until(lambda: bool(pairing))
            putting.cancel()
        try:
            await putting
        except BaseException as e:
            return type(e), not halves[-1][1].done()
        raise AssertionError("the put did not fail")

    monkeypatch.setattr(
        GlooTransportBuffer, "_pre_handshake", _recording_halves(halves)
    )
    if failure == "cancelled_pairing":
        monkeypatch.setattr(GlooTransportBuffer, "_post_handshake", hold_the_half)
    (error, half_waiting), elapsed = await _in_own_loop(fail_a_put)
    monkeypatch.setattr(GlooTransportBuffer, "_pre_handshake", pre_handshake)
    monkeypatch.setattr(GlooTransportBuffer, "_post_handshake", post_handshake)
    monkeypatch.setattr(runtime, "_CALL_TIMEOUT_S", 300.0)
    assert error is (
        CallTimeoutError if failure == "expired" else asyncio.CancelledError
    )
    if failure == "expired":
        # Raised at once, while the client's half still waited for the volume's.
        assert half_waiting
    assert elapsed < _GLOO_TIMEOUT_S + 5
    assert core.volume_id not in gloo._store_addrs
    assert ref.transport_context.get(GlooProcessGroupCache)._process_groups == {}
    # A handshake run after the client's half gave up holds the volume until Gloo fails it.
    idle_timeout_s = 5 * _GLOO_TIMEOUT_S + 15
    await _release(
        server, core, holders, abandoned=queued, idle_timeout_s=idle_timeout_s
    )
    halves.clear()
    await _put(ref, "second", torch.arange(8.0))
    assert_bitwise_equal(await _get(ref, "second"), torch.arange(8.0))
    assert core.names().count("handshake") == (1 if queued else 2)
    assert len(ref.transport_context.get(GlooProcessGroupCache)._process_groups) == 1
    # The volume drops the warm-up pair and the failed one, also after a cancel that came once
    # its handshake returned.
    volume = core.store.transport_context.get(GlooProcessGroupCache)._process_groups
    assert len(volume) == 1
    assert [r.getMessage() for r in caplog.records if r.name == "asyncio"] == []


@pytest.mark.asyncio
async def test_a_handshake_cancelled_while_the_volume_runs_it_leaves_no_half_after_the_next_call(
    monkeypatch: pytest.MonkeyPatch, volumes: _Volumes
) -> None:
    """Drop the volume's half of a handshake whose caller gave up while it ran, at the next call.

    The client's half still waits, so the volume pairs with it and caches its half.
    """
    monkeypatch.setattr(gloo, "TORCHSTORE_GLOO_INIT_TIMEOUT", 30)
    core, server, record = await volumes.serve(_InstrumentedCore(held={"handshake"}))
    ref = volumes.ref(record, core.volume_id, TransportType.Gloo)
    putting = asyncio.ensure_future(_put(ref, "first", torch.ones(8)))
    await _until(lambda: core.names() == ["handshake"] and core.active == 1)
    putting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await putting
    core.release.set()
    await _until(lambda: server.stats()["running"] == 0, timeout_s=30)
    volume = core.store.transport_context.get(GlooProcessGroupCache)._process_groups
    assert len(volume) == 1
    assert await ref.volume.get_id.call_one() == (core.volume_id, ref.volume_hostname)
    assert volume == {}


@pytest.mark.asyncio
async def test_a_gloo_handshake_the_volume_refused_raises_at_once_then_logs_its_half(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    volumes: _Volumes,
) -> None:
    """Raise a handshake the volume refused at once. Its waiting half's failure is logged at WARNING, and its master closes."""
    monkeypatch.setattr(gloo, "TORCHSTORE_GLOO_INIT_TIMEOUT", _GLOO_TIMEOUT_S)
    core, _, record = await volumes.serve()
    ref = volumes.ref(record, core.volume_id, TransportType.Gloo)
    halves: list[tuple[int, asyncio.Task[Any]]] = []
    monkeypatch.setattr(
        GlooTransportBuffer, "_pre_handshake", _recording_halves(halves)
    )
    # As in a volume's process whose exit began, which starts no pairing.
    monkeypatch.setattr(gloo, "_exit_deadline", time.monotonic())
    buffer = transport.create_transport_buffer(ref)
    with pytest.raises(RuntimeError, match="was not started: exiting"):
        await buffer.put_to_storage_volume([Request.from_tensor("k", torch.ones(4))])
    port, half = halves[0]
    assert not half.done()

    def warned() -> list[logging.LogRecord]:
        return [
            r
            for r in caplog.records
            if r.name == gloo.__name__ and r.levelno == logging.WARNING
        ]

    # The half's done callback logs, a loop turn after the half ends.
    await _until(lambda: half.done() and bool(warned()), timeout_s=30)
    assert [r.getMessage() for r in warned()] == [
        "The client half of a Gloo pairing whose handshake failed then failed"
    ]
    # A kept record's traceback holds the half's frames, and so its master.
    for record_ in caplog.records:
        record_.exc_info = None
    collected = weakref.ref(half)
    del half, buffer
    halves.clear()
    await _until(_collected(collected))
    assert [r.getMessage() for r in caplog.records if r.name == "asyncio"] == []
    await _until(lambda: not _master_listening(port), timeout_s=15)


@pytest.mark.parametrize("failure", ["expired", "cancelled"])
@pytest.mark.asyncio
async def test_a_sharedmemory_put_the_volume_dropped_leaves_no_segment_mapped(
    monkeypatch: pytest.MonkeyPatch, volumes: _Volumes, failure: str
) -> None:
    """Unmap a batch put's new segments once its queued put call expires or is cancelled. The next put maps them."""
    core, server, record = await volumes.serve()
    ref = volumes.ref(record, core.volume_id, TransportType.SharedMemory)
    pre_put_hook = SharedMemoryTransportBuffer._pre_put_hook
    holders: list[asyncio.Future[Any]] = []

    async def hold_after_handshake(buffer: Any, requests: Any) -> None:
        await pre_put_hook(buffer, requests)
        holders.append(await _hold(core, ref))
        if failure == "expired":
            monkeypatch.setattr(runtime, "_CALL_TIMEOUT_S", 2.0)

    monkeypatch.setattr(
        SharedMemoryTransportBuffer, "_pre_put_hook", hold_after_handshake
    )
    buffer = transport.create_transport_buffer(ref)
    putting = asyncio.ensure_future(buffer.put_to_storage_volume(_batch(1.0)))
    if failure == "cancelled":
        await _until(lambda: _counts(server)[1:] == (1, 1))
        putting.cancel()
    expected = CallTimeoutError if failure == "expired" else asyncio.CancelledError
    with pytest.raises(expected):
        await putting
    monkeypatch.setattr(SharedMemoryTransportBuffer, "_pre_put_hook", pre_put_hook)
    monkeypatch.setattr(runtime, "_CALL_TIMEOUT_S", 300.0)
    mapped = ref.transport_context.get(SharedMemoryCache)._storages
    assert list(mapped) == []
    # Dropped before it ran, so the volume never attached the segments.
    await _release(server, core, holders)
    assert not {"weights", "bias"} & set(core.store.kv)
    await transport.create_transport_buffer(ref).put_to_storage_volume(_batch(2.0))
    assert_bitwise_equal(await _get(ref, "weights"), torch.full((8,), 2.0))
    assert_bitwise_equal(await _get(ref, "bias"), torch.full((4,), 2.0))
    assert sorted(key for key, _ in mapped) == ["bias", "weights"]
    assert core.keys("put") == ["weights"]


@pytest.mark.asyncio
async def test_a_sharedmemory_put_the_volume_refused_leaves_no_segment_mapped(
    monkeypatch: pytest.MonkeyPatch, volumes: _Volumes
) -> None:
    """Unmap a batch put's new segments once the volume refuses it, as when another writer changed a key."""
    core, _, record = await volumes.serve()
    ref = volumes.ref(record, core.volume_id, TransportType.SharedMemory)
    pre_put_hook = SharedMemoryTransportBuffer._pre_put_hook

    async def store_elsewhere_after_handshake(buffer: Any, requests: Any) -> None:
        await pre_put_hook(buffer, requests)
        core.store.kv["weights"] = torch.zeros(8)

    monkeypatch.setattr(
        SharedMemoryTransportBuffer, "_pre_put_hook", store_elsewhere_after_handshake
    )
    with pytest.raises(AssertionError) as refused:
        await transport.create_transport_buffer(ref).put_to_storage_volume(_batch(1.0))
    # The volume's segment check is a bare assert, so its traceback names it.
    assert "assert existing is not None" in str(refused.value.__cause__)
    assert list(ref.transport_context.get(SharedMemoryCache)._storages) == []


@pytest.mark.asyncio
async def test_a_failed_sharedmemory_get_keeps_the_clients_mappings(
    volumes: _Volumes,
) -> None:
    """Keep the client's mappings of a failed get's keys, which unlike a failed put's the volume still holds."""
    core, _, record = await volumes.serve()
    ref = volumes.ref(record, core.volume_id, TransportType.SharedMemory)
    await transport.create_transport_buffer(ref).put_to_storage_volume(_batch(1.0))
    mapped = ref.transport_context.get(SharedMemoryCache)._storages
    segments = set(mapped)
    with pytest.raises(KeyError, match="Key 'missing' not found"):
        await transport.create_transport_buffer(ref).get_from_storage_volume(
            [Request(key="weights"), Request(key="missing")]
        )
    assert set(mapped) == segments and len(segments) == 2


@pytest.mark.asyncio
async def test_a_sharedmemory_put_of_held_keys_writes_the_volumes_segments_in_place(
    volumes: _Volumes,
) -> None:
    """Write a put of keys the volume holds into its segments, which its handshake reply names."""
    core, _, record = await volumes.serve()
    ref = volumes.ref(record, core.volume_id, TransportType.SharedMemory)
    mapped = ref.transport_context.get(SharedMemoryCache)._storages
    await transport.create_transport_buffer(ref).put_to_storage_volume(_batch(1.0))
    stored = {key: tensor.data_ptr() for key, tensor in core.store.kv.items()}
    segments = set(mapped)
    await transport.create_transport_buffer(ref).put_to_storage_volume(_batch(2.0))
    # Attached, not allocated: the client maps no new segment and the volume keeps its own.
    assert {key: tensor.data_ptr() for key, tensor in core.store.kv.items()} == stored
    assert set(mapped) == segments and len(segments) == 2
    for key, size in (("weights", 8), ("bias", 4)):
        assert_bitwise_equal(await _get(ref, key), torch.full((size,), 2.0))
    assert core.names() == ["handshake", "put", "handshake", "put", "get", "get"]


@pytest.mark.asyncio
async def test_a_closed_volume_frees_its_server_and_core() -> None:
    """Free a closed volume that served a call, with its core and tensors, with the cyclic GC off."""
    gc.disable()
    try:
        core = StorageVolumeCore(f"volume-{uuid.uuid4()}")
        core.store.kv["weights"] = torch.zeros(1 << 20)
        server, record = await serve_volume(core, scope=_scope())
        proxy = VolumeProxy(decode_volume_record(core.volume_id, record).address)
        assert (await proxy.get_id.call_one())[0] == core.volume_id
        await proxy.close()
        freed = [weakref.ref(server), weakref.ref(core)]
        await server.close(grace_s=0)
        del server, core
        await _until(lambda: all(ref() is None for ref in freed))
    finally:
        gc.enable()


@pytest.mark.asyncio
async def test_a_volume_process_exits_within_gloos_timeout_while_a_handler_waits_in_gloo(
    monkeypatch: pytest.MonkeyPatch, volumes: _Volumes, children: _Children
) -> None:
    """Exit within Gloo's timeout plus 10 s while a volume handler waits in a Gloo receive.

    The volume closes first and its dispatcher thread ends, so finalization never stops it.
    """
    gloo_timeout_s = 5
    monkeypatch.setenv("TORCHSTORE_GLOO_INIT_TIMEOUT", str(gloo_timeout_s))
    monkeypatch.setattr(gloo, "TORCHSTORE_GLOO_INIT_TIMEOUT", gloo_timeout_s)
    child, volume_id, record = _serve_in_child(children)
    ref = volumes.ref(record, volume_id, TransportType.Gloo)
    await _put(ref, "first", torch.zeros(8))
    assert _read_line(child) == "put first"
    # A put whose send never starts leaves the volume waiting in its Gloo receive.
    stalled = GlooTransportBuffer(ref)
    tensor = torch.zeros(8)
    request = Request.from_tensor("second", tensor)
    assert not stalled.requires_handshake([request])
    stalled.shape, stalled.dtype = tensor.shape, tensor.dtype
    putting = asyncio.ensure_future(
        ref.volume.put.call_one(stalled, [request.meta_only()])
    )
    # Read off the loop, so the call above is sent meanwhile.
    assert await asyncio.to_thread(_read_line, child) == "put second"
    start = time.monotonic()
    _command(child, "quit")
    assert await asyncio.to_thread(child.wait, 30) == 0
    assert time.monotonic() - start < gloo_timeout_s + 10
    assert _read_line(child) == "dispatchers at exit: 0"
    with pytest.raises(Exception) as raised:
        await putting
    assert isinstance(raised.value, (PeerLostError, RuntimeError))


# Serves a volume whose handler pairs with a TCPStore that accepts and never replies, and keeps a
# forgotten pair's group as a transfer under way would. Exits with Gloo's exit bound at argv[2] seconds.
_PAIRING_AT_EXIT_CHILD = """
import asyncio, atexit, contextlib, json, socket, sys, time

seen = {"retired at exit": []}


@atexit.register
def report():
    # Registered before torchstore loads and so run after every exit hook torchstore registers.
    seen["exit hooks took"] = time.monotonic() - seen.pop("exit began")
    seen["dispatcher parked"] = served[0].dispatcher.parked
    print(json.dumps(seen), flush=True)


sys.modules["monarch"] = None
sys.path.insert(0, sys.argv[1])
import torch
from torchstore.carrier_volumes import decode_volume_record, serve_volume, VolumeProxy
from torchstore.storage import StorageVolumeCore
from torchstore.strategy import StorageVolumeRef
from torchstore.transport import gloo, TransportType
from torchstore.transport.buffers import TransportContext
from torchstore.transport.gloo import GlooProcessGroupCache, GlooTransportBuffer
from torchstore.transport.types import Request

gloo._EXIT_DRAIN_S = float(sys.argv[2])
# Each exit hook that takes the shared deadline. carrier_volumes and gloo look it up when called.
begin_exit = gloo.begin_exit
seen["exit deadlines"] = []
gloo.begin_exit = lambda: seen["exit deadlines"].append(begin_exit()) or seen["exit deadlines"][-1]
gloo.add_retired_listener(
    lambda volume_id, key: "exit began" in seen and seen["retired at exit"].append(key)
)
listener = socket.socket()
listener.bind(("127.0.0.1", 0))
listener.listen(8)
port = listener.getsockname()[1]
seen["port"] = port
# Kept open through finalization, which would otherwise reset the pairing's connection.
listener.detach()
served = []


class Group:
    pass


held = Group()


async def main():
    core = StorageVolumeCore("volume-at-exit")
    server, record = await serve_volume(core, scope="exit/nonce")
    served.append(server)
    address, hostname = decode_volume_record("volume-at-exit", record)
    ref = StorageVolumeRef(
        VolumeProxy(address), "volume-at-exit", TransportContext(), TransportType.Gloo, hostname
    )
    buffer = GlooTransportBuffer(ref)
    buffer.master_addr, buffer.master_port = "127.0.0.1", port
    buffer.store_key = "torchstore_gloo_stuck"
    request = Request.from_tensor("k", torch.ones(1)).meta_only()
    calling = asyncio.ensure_future(ref.volume.handshake.call_one(buffer, [request]))
    # A pairing refuses to start once exit begins unless it already counts itself among those exit waits for.
    while not gloo._pairings:
        await asyncio.sleep(0.01)
    # The volume's handler keeps waiting in the pairing.
    calling.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await calling
    ref.transport_context.get(GlooProcessGroupCache).put("torchstore_gloo_held", held)
    gloo._pair_groups["torchstore_gloo_held"] = held
    gloo._store_addrs["volume-held"] = ("host", 1, "torchstore_gloo_held", True)
    gloo.forget_pair("volume-held", "torchstore_gloo_held", ref.transport_context)


asyncio.run(main())
seen["exit began"] = time.monotonic()
"""


def test_a_volume_process_waits_out_one_gloo_exit_bound_while_a_handler_waits_in_a_pairing(
    monkeypatch: pytest.MonkeyPatch, children: _Children
) -> None:
    """Wait out one Gloo exit bound in all at exit while a volume's handler waits in a pairing that outlives it.

    The volume's exit hook closes the volume and parks its dispatcher, both hooks take the one deadline,
    Gloo's hook names the pairing still running at WARNING, and no retired listener hears of a pair whose
    group outlives the exit.
    """
    bound_s = 2.0
    # The pairing's own deadline, 4 Gloo timeouts and 10s more, stays far past the exit.
    monkeypatch.setenv("TORCHSTORE_GLOO_INIT_TIMEOUT", "60")
    child = children.spawn(_PAIRING_AT_EXIT_CHILD, str(bound_s))
    out, err = child.communicate(timeout=90)
    assert child.returncode == 0, err[-4000:]
    seen = json.loads(out.splitlines()[-1])
    assert seen["dispatcher parked"] and seen["retired at exit"] == [], seen
    # Both hooks wait until the one deadline the volumes' hook set rather than one bound after the other.
    deadlines = seen["exit deadlines"]
    assert len(deadlines) == 2 and deadlines[0] == deadlines[1], seen
    assert seen["exit hooks took"] >= bound_s, seen
    pairing = (
        f"torchstore-gloo pairing through the TCPStore at 127.0.0.1:{seen['port']}"
    )
    assert (
        f"Exiting with 1 Gloo pairings still running after {bound_s:g}s, which may abort"
        f" the process: {pairing}\n"
    ) in err


class _NotingClock:
    """Stands in for ``time`` in the module under test and notes the monotonic readings this thread takes."""

    def __init__(self) -> None:
        self.thread = threading.current_thread()
        self.readings: list[float] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(time, name)

    def monotonic(self) -> float:
        now = time.monotonic()
        if threading.current_thread() is self.thread:
            self.readings.append(now)
        return now


@pytest.mark.usefixtures("fresh_runtime")
@pytest.mark.parametrize("handler", ["ends_while_waited_for", "outlives_the_deadline"])
def test_the_volume_exit_hook_waits_for_a_running_handler_until_gloos_exit_deadline(
    monkeypatch: pytest.MonkeyPatch, volumes: _Volumes, handler: str
) -> None:
    """Wait at exit for a closed volume's running handler until Gloo's exit deadline and then park a late one's dispatcher.

    A handler that ends while the hook waits ends its dispatcher thread instead. The hook waits no longer than that
    deadline, runs on a loop of its own that it closes and leaves later hooks the main thread's current loop.
    """
    ends = handler == "ends_while_waited_for"
    # The deadline the hook sets refuses later pairings and must go once the case ends.
    monkeypatch.setattr(gloo, "_exit_deadline", None)
    monkeypatch.setattr(gloo, "_EXIT_DRAIN_S", 30.0 if ends else 0.1)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        core = _InstrumentedCore(held={"put"})
        server, record = loop.run_until_complete(serve_volume(core, scope=_scope()))
        monkeypatch.setattr(carrier_volumes, "_served", weakref.WeakSet([server]))
        ref = volumes.ref(record, core.volume_id, TransportType.MonarchRPC)
        putting = loop.create_task(_put(ref, "k", torch.ones(4)))
        loop.run_until_complete(_until(lambda: core.active == 1))
        dispatcher = server.dispatcher
        clock = _NotingClock()
        monkeypatch.setattr(carrier_volumes, "time", clock)
        # Each wait of the hook for the dispatcher, with the reading it was computed from.
        joins: list[tuple[float | None, float]] = []
        join = dispatcher.thread.join

        def noting_join(timeout: float | None = None) -> None:
            if threading.current_thread() is clock.thread:
                joins.append((timeout, clock.readings[-1]))
                if ends:
                    # The handler ends only once the hook waits for its thread.
                    core.release.set()
            join(timeout)

        monkeypatch.setattr(dispatcher.thread, "join", noting_join)
        hook_loops: list[asyncio.AbstractEventLoop] = []
        close = server.close

        async def close_noting_its_loop(**kwargs: Any) -> None:
            hook_loops.append(asyncio.get_running_loop())
            await close(**kwargs)

        monkeypatch.setattr(server, "close", close_noting_its_loop)
        try:
            carrier_volumes._close_at_exit()
            [(timeout, now)] = joins
            assert timeout == max(gloo._exit_deadline - now, 0.0)
            assert (dispatcher.parked, dispatcher.thread.is_alive()) == (not ends,) * 2
            if not ends:
                _assert_blocks_in_park(dispatcher)
            assert asyncio.get_event_loop_policy().get_event_loop() is loop
            [hook_loop] = hook_loops
            assert hook_loop is not loop and hook_loop.is_closed()
        finally:
            # The handler's executor thread ends even where a parked dispatcher never resumes the handler.
            core.release.set()
        if ends:
            # The handler finished its put after the closed volume had already failed the call.
            assert_bitwise_equal(core.store.kv["k"], torch.ones(4))
        with pytest.raises(PeerLostError, match="volume closed"):
            loop.run_until_complete(putting)
    finally:
        asyncio.set_event_loop(None)
        loop.close()
