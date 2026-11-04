# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pickling tests for every transport buffer.

Clients pickle each buffer to its storage volume, so the copy must not carry
the client's volume handle or the transport caches that handle holds.
"""

import copy
import functools
import pickle
import threading
from typing import Any

import pytest
import torch
import torchstore.transport.nixl as nixl_transport
import torchstore.transport.torch_distributed as torch_distributed_transport
import torchstore.transport.torchcomms.cache as torchcomms_cache
from torchstore.strategy import StorageVolumeRef
from torchstore.transport import TransportType
from torchstore.transport.buffers import (
    TransportBuffer,
    TransportCache,
    TransportContext,
)
from torchstore.transport.torchcomms.uniflow_buffer import UniflowConnectionState

# Discovery must keep finding these. A new subclass is covered without listing once its
# module loads with this file's imports.
EXPECTED_BUFFERS = {
    "GlooTransportBuffer",
    "MonarchRDMATransportBuffer",
    "MonarchRPCTransportBuffer",
    "NixlTransportBuffer",
    "SharedMemoryTransportBuffer",
    "TorchCommsRdmaTransportBuffer",
    "TorchCommsTransportBuffer",
    "TorchDistributedTransportBuffer",
    "XcclTransportBuffer",
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


class LargeEntryCache(TransportCache):
    """Client cache holding plain tensors, so the case needs no shared memory."""

    def __init__(self) -> None:
        self.entries: dict[str, torch.Tensor] = {}

    def clear(self) -> None:
        self.entries.clear()


def _volume_ref(volume: object) -> StorageVolumeRef:
    return StorageVolumeRef(
        volume, "volume-0", TransportContext(), TransportType.Unset, "volume-host"
    )


def _wire(buffer: TransportBuffer) -> TransportBuffer:
    return pickle.loads(pickle.dumps(buffer))


# Pickling sends a buffer to its volume, and copies made by the copy module match it.
_SENDS = {"pickle": _wire, "copy": copy.copy, "deepcopy": copy.deepcopy}


@pytest.fixture(autouse=True)
def optional_transports(monkeypatch):
    # The NIXL, Uniflow and torch.distributed constructors check availability, and NIXL's
    # also loads its API. No test here calls any of those libraries.
    monkeypatch.setattr(nixl_transport, "nixl_available", lambda: True)
    monkeypatch.setattr(nixl_transport, "_load_nixl_api", lambda: (None, None))
    monkeypatch.setattr(torchcomms_cache, "torchcomms_uniflow_available", lambda: True)
    monkeypatch.setattr(
        torch_distributed_transport, "torch_transport_available", lambda: True
    )


@pytest.fixture
def cached_ref():
    ref = _volume_ref(object())
    cache = ref.transport_context.get(LargeEntryCache)
    cache.entries["weights"] = torch.zeros(CACHED_BYTES, dtype=torch.uint8)
    yield ref
    ref.transport_context.clear()


def test_expected_buffers_are_discovered():
    missing = EXPECTED_BUFFERS - {cls.__name__ for cls in BUFFERS}
    assert not missing, f"transport buffers no longer discovered: {sorted(missing)}"


_connection = functools.partial(UniflowConnectionState, b"topology", b"bind-info")


# Process-local fields each transport's own hook resets, each building a fresh
# (client value, copy's value) pair. A TorchComms connection keeps the topology and bind
# info that the volume reads.
_RESET_ON_PICKLE = {
    "GlooTransportBuffer": {
        name: lambda: ("process-local", None)
        for name in ("_tcp_store", "_pg_task", "_send_task", "_recv_task")
    },
    "MonarchRPCTransportBuffer": {"inplace_tensor": lambda: ("process-local", None)},
    "NixlTransportBuffer": {"_leases": lambda: ({"process-local"}, set())},
    "TorchCommsRdmaTransportBuffer": {
        "_transports": lambda: ({0: "process-local"}, {})
    },
    "TorchCommsTransportBuffer": {
        "_connections": lambda: (
            {0: _connection("process-local")},
            {0: _connection(None)},
        )
    },
    "TorchDistributedTransportBuffer": {
        "_pending": lambda: ("process-local", None),
        "_endpoint": lambda: ("process-local", None),
        "_keys": lambda: (["process-local"], []),
    },
    "XcclTransportBuffer": {
        **{
            name: lambda: ("process-local", None)
            for name in ("_tcp_store", "_pg_task", "_send_task", "_recv_task")
        },
        "_batch_metas": lambda: (["process-local"], []),
    },
}


@pytest.mark.parametrize("send", list(_SENDS.values()), ids=list(_SENDS))
@pytest.mark.parametrize("buffer_cls", BUFFERS, ids=lambda cls: cls.__name__)
def test_copies_drop_volume_handle_and_reset_process_local_fields(buffer_cls, send):
    ref = _volume_ref(object())
    buffer = buffer_cls(ref)
    reset = _RESET_ON_PICKLE.get(buffer_cls.__name__, {})
    fields = vars(buffer).keys() - {"storage_volume_ref"}
    assert reset.keys() <= fields

    def state(side: int) -> dict[str, object]:
        # Fresh values on each call, so a hook that mutates the client's own shows.
        return {
            name: reset[name]()[side] if name in reset else f"sentinel-{name}"
            for name in fields
        }

    # Every attribute holds a value, so the equality shows a hook that drops or keeps
    # the wrong one, the handle included.
    for name, value in state(0).items():
        setattr(buffer, name, value)

    assert vars(send(buffer)) == {**state(1), "storage_volume_ref": None}
    # The client keeps using its buffer, every field as it set it, after sending it.
    assert vars(buffer) == {**state(0), "storage_volume_ref": ref}


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
    buffer = buffer_cls(_volume_ref(object()))
    buffer.client_only = "client"

    assert _wire(buffer).client_only is None
    assert buffer.client_only == "client"


class _DropsLock:
    """A pickling hook from a base class listed after a transport buffer class."""

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["lock"] = None
        return state


class _InstanceDictState:
    """A later base's hook that returns the instance dict itself, as Python's default does."""

    def __getstate__(self) -> dict:
        return self.__dict__


class _VersionedState:
    """A later base's hook whose state is a (fields, version) pair."""

    def __getstate__(self) -> tuple:
        return (self.__dict__.copy(), 1)

    def __setstate__(self, state: tuple) -> None:
        fields, _version = state
        self.__dict__.update(fields)


def _with_later_hook(buffer_cls: type, hook: type) -> Any:
    """Subclass ``buffer_cls`` with ``hook`` listed after it, at module level so pickle finds it."""
    name = buffer_cls.__name__ + "With" + hook.__name__.lstrip("_")
    mixed = type(name, (buffer_cls, hook), {"__module__": __name__})
    globals()[name] = mixed
    return mixed


_DROPS_LOCK_BUFFERS = [
    _with_later_hook(cls, _DropsLock) for cls in [TransportBuffer, *BUFFERS]
]


@pytest.mark.parametrize("send", list(_SENDS.values()), ids=list(_SENDS))
@pytest.mark.parametrize(
    "buffer_cls", _DROPS_LOCK_BUFFERS, ids=lambda cls: cls.__name__
)
def test_copies_start_from_a_later_base_hook(buffer_cls, send):
    """Run a later base's hook under the base hook and every transport's own hook."""
    ref = _volume_ref(object())
    buffer = buffer_cls(ref)
    buffer.lock = threading.Lock()

    copied = send(buffer)
    assert copied.lock is None and copied.storage_volume_ref is None
    assert buffer.lock is not None and buffer.storage_volume_ref is ref


def test_a_later_base_hooks_dict_is_copied_before_the_handle_drops():
    """Leave the client's own handle in place when a later base's hook returns its dict."""
    ref = _volume_ref(object())
    buffer = _with_later_hook(TransportBuffer, _InstanceDictState)(ref)

    assert _wire(buffer).storage_volume_ref is None
    assert buffer.storage_volume_ref is ref


def test_a_later_base_hooks_state_other_than_a_dict_passes_through():
    """Hand a state that is not a dict to the later base's restore as that hook built it."""
    buffer = _with_later_hook(TransportBuffer, _VersionedState)(_volume_ref(object()))
    buffer.field = "client"

    copied = vars(_wire(buffer))
    assert copied["field"] == "client"
    assert copied["storage_volume_ref"].volume_id == "volume-0"


class _NoHandle(TransportBuffer):
    """A subclass whose constructor sets no volume handle."""

    def __init__(self) -> None:
        self.field = "client"


@pytest.mark.parametrize("send", list(_SENDS.values()), ids=list(_SENDS))
def test_copies_add_no_handle_an_instance_never_had(send):
    """Add no handle to a copy of an instance whose constructor set none."""
    assert vars(send(_NoHandle())) == {"field": "client"}
