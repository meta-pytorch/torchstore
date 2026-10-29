# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pickling tests for every transport buffer.

Clients pickle each buffer to its storage volume, so the copy must not carry
the client's volume handle or the transport caches that handle holds.
"""

import pickle
import threading

import pytest
import torch
import torchstore.transport.monarch_rdma as monarch_rdma
import torchstore.transport.nixl as nixl_transport
import torchstore.transport.torchcomms.cache as torchcomms_cache
from torchstore.strategy import StorageVolumeRef
from torchstore.transport import TransportType
from torchstore.transport.buffers import (
    TransportBuffer,
    TransportCache,
    TransportContext,
)
from torchstore.transport.monarch_rdma import MonarchRDMATransportBuffer
from torchstore.transport.monarch_rpc import MonarchRPCTransportBuffer
from torchstore.transport.types import Request

# Discovery must keep finding these. New subclasses are covered without listing.
EXPECTED_BUFFERS = {
    "GlooTransportBuffer",
    "MonarchRDMATransportBuffer",
    "MonarchRPCTransportBuffer",
    "NixlTransportBuffer",
    "SharedMemoryTransportBuffer",
    "TorchCommsRdmaTransportBuffer",
    "TorchCommsTransportBuffer",
}

CACHED_BYTES = 8 * 1024 * 1024
MAX_PICKLE_BYTES = 4 * 1024


def _buffer_classes() -> list[type[TransportBuffer]]:
    # Importing torchstore.transport loads every transport module.
    found: set[type[TransportBuffer]] = set()
    pending = [TransportBuffer]
    while pending:
        for subclass in pending.pop().__subclasses__():
            if subclass not in found:
                found.add(subclass)
                pending.append(subclass)
    buffers = [cls for cls in found if cls.__module__.startswith("torchstore.")]
    return sorted(buffers, key=lambda cls: cls.__name__)


BUFFERS = _buffer_classes()


class LockedVolume:
    """Volume handle that cannot be pickled."""

    def __init__(self) -> None:
        self.lock = threading.Lock()


class PlainVolume:
    """Volume handle that pickles to a few bytes."""


class LargeEntryCache(TransportCache):
    """Client cache holding plain tensors, so the case needs no shared memory."""

    def __init__(self) -> None:
        self.entries: dict[str, torch.Tensor] = {}

    def clear(self) -> None:
        self.entries.clear()


_RDMA_BUFFERS: dict[int, "SharedFakeRDMABuffer"] = {}


def _registered_rdma_buffer(key: int) -> "SharedFakeRDMABuffer":
    return _RDMA_BUFFERS[key]


class SharedFakeRDMABuffer:
    """Fake RDMABuffer whose pickled copies share its memory, as RDMA would."""

    def __init__(self, byte_view: torch.Tensor) -> None:
        self.mem = byte_view
        self.key = len(_RDMA_BUFFERS)
        _RDMA_BUFFERS[self.key] = self

    def __reduce__(self) -> tuple:
        return _registered_rdma_buffer, (self.key,)

    async def drop(self) -> None:
        pass


class FakeRDMAAction:
    """Fake RDMAAction that copies between registered views on submit."""

    def __init__(self) -> None:
        self.copies: list[tuple[torch.Tensor, torch.Tensor]] = []

    def read_remote(self, dst: torch.Tensor, src: SharedFakeRDMABuffer) -> None:
        self.copies.append((dst, src.mem))

    def write_remote(self, dst: SharedFakeRDMABuffer, src: torch.Tensor) -> None:
        self.copies.append((dst.mem, src))

    async def submit(self, *, timeout: int = 60) -> None:
        for dst, src in self.copies:
            dst.copy_(src)


def _volume_ref(volume: object) -> StorageVolumeRef:
    return StorageVolumeRef(
        volume, "volume-0", TransportContext(), TransportType.Unset, "volume-host"
    )


def _wire(buffer: TransportBuffer) -> TransportBuffer:
    return pickle.loads(pickle.dumps(buffer))


@pytest.fixture(autouse=True)
def optional_transports(monkeypatch):
    # The NIXL and Uniflow constructors only check availability. No test here
    # calls either library.
    monkeypatch.setattr(nixl_transport, "nixl_available", lambda: True)
    monkeypatch.setattr(nixl_transport, "_load_nixl_api", lambda: (None, None))
    monkeypatch.setattr(torchcomms_cache, "torchcomms_uniflow_available", lambda: True)


@pytest.fixture
def shared_fake_rdma(monkeypatch):
    monkeypatch.setattr(monarch_rdma, "RDMABuffer", SharedFakeRDMABuffer)
    monkeypatch.setattr(monarch_rdma, "RDMAAction", FakeRDMAAction)
    yield
    _RDMA_BUFFERS.clear()


@pytest.fixture
def locked_ref():
    ref = _volume_ref(LockedVolume())
    yield ref
    ref.transport_context.clear()


@pytest.fixture
def cached_ref():
    ref = _volume_ref(PlainVolume())
    cache = ref.transport_context.get(LargeEntryCache)
    cache.entries["weights"] = torch.zeros(CACHED_BYTES, dtype=torch.uint8)
    yield ref
    ref.transport_context.clear()


def test_expected_buffers_are_discovered():
    missing = EXPECTED_BUFFERS - {cls.__name__ for cls in BUFFERS}
    assert not missing, f"transport buffers no longer discovered: {sorted(missing)}"


# Process-local fields that a transport's own hook resets on purpose.
_RESET_ON_PICKLE = {
    "GlooTransportBuffer": {"_tcp_store", "_pg_task", "_send_task", "_recv_task"},
    "MonarchRPCTransportBuffer": {"inplace_tensor"},
    "TorchCommsRdmaTransportBuffer": {"_transports"},
    "TorchCommsTransportBuffer": {"_connections"},
}


@pytest.mark.parametrize("buffer_cls", BUFFERS, ids=lambda cls: cls.__name__)
def test_pickle_drops_unpicklable_volume_handle(buffer_cls, locked_ref):
    with pytest.raises(TypeError):
        pickle.dumps(locked_ref)
    buffer = buffer_cls(locked_ref)
    # Every other attribute holds a value, so a hook that also drops one shows.
    reset = _RESET_ON_PICKLE.get(buffer_cls.__name__, set())
    for name in vars(buffer).keys() - {"storage_volume_ref", *reset}:
        setattr(buffer, name, f"sentinel-{name}")

    copy = _wire(buffer)

    assert copy.storage_volume_ref is None
    # The volume still gets every other attribute, with its value.
    kept = {name: value for name, value in vars(copy).items() if name not in reset}
    sent = {name: value for name, value in vars(buffer).items() if name not in reset}
    assert kept == {**sent, "storage_volume_ref": None}
    # The client keeps using its buffer after sending it.
    assert buffer.storage_volume_ref is locked_ref


@pytest.mark.parametrize("buffer_cls", BUFFERS, ids=lambda cls: cls.__name__)
def test_pickle_excludes_client_transport_caches(buffer_cls, cached_ref):
    assert len(pickle.dumps(cached_ref)) > CACHED_BYTES
    buffer = buffer_cls(cached_ref)

    assert len(pickle.dumps(buffer)) < MAX_PICKLE_BYTES


@pytest.mark.parametrize("buffer_cls", BUFFERS, ids=lambda cls: cls.__name__)
def test_pickle_applies_rules_added_to_the_base_hook(buffer_cls, monkeypatch):
    base_getstate = TransportBuffer.__getstate__

    def getstate_with_later_rule(self: TransportBuffer) -> dict:
        state = base_getstate(self)
        state["client_only"] = None
        return state

    monkeypatch.setattr(TransportBuffer, "__getstate__", getstate_with_later_rule)
    buffer = buffer_cls(_volume_ref(PlainVolume()))
    buffer.client_only = "client"

    assert _wire(buffer).client_only is None
    assert buffer.client_only == "client"


def test_monarch_rpc_pickle_drops_inplace_destination():
    buffer = MonarchRPCTransportBuffer(_volume_ref(PlainVolume()))
    buffer.inplace_tensor = torch.zeros(CACHED_BYTES, dtype=torch.uint8)

    copy = _wire(buffer)

    assert copy.inplace_tensor is None
    assert buffer.inplace_tensor is not None


@pytest.mark.asyncio
async def test_monarch_rdma_volume_serves_pickled_buffers(locked_ref, shared_fake_rdma):
    source = torch.arange(16, dtype=torch.float32)
    put = [Request.from_tensor("weights", source)]
    client = MonarchRDMATransportBuffer(locked_ref)
    await client._pre_put_hook(put)
    [stored] = await _wire(client).handle_put_request(
        TransportContext(), [(put[0].meta_only(), None)]
    )
    assert torch.equal(stored, source)

    get = [Request.from_tensor("weights", torch.zeros(16))]
    client = MonarchRDMATransportBuffer(locked_ref)
    await client._pre_get_hook(get)
    volume_copy = _wire(client)
    await volume_copy.handle_get_request(
        TransportContext(), [(get[0].meta_only(), stored)]
    )
    [result] = await client._handle_storage_volume_response(get, _wire(volume_copy))
    assert torch.equal(result, source)
