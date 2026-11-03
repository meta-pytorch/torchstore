# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pickling tests for every transport buffer.

Clients pickle each buffer to its storage volume, so the copy must not carry
the client's volume handle or the transport caches that handle holds.
"""

import collections
import copy
import functools
import pickle
import threading
import types
from typing import Any

import pytest
import torch
import torchstore.monarch.rdma  # noqa: F401
import torchstore.transport.nixl as nixl_transport
import torchstore.transport.torch_distributed as torch_distributed_transport
import torchstore.transport.torchcomms.cache as torchcomms_cache
from torch.fx.immutable_collections import immutable_dict
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
}

CACHED_BYTES = 8 * 1024 * 1024
MAX_PICKLE_BYTES = 4 * 1024


def _buffer_classes() -> list[type[TransportBuffer]]:
    # Importing torchstore.transport and torchstore.monarch.rdma loads every transport module.
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


@pytest.mark.parametrize("buffer_cls", BUFFERS, ids=lambda cls: cls.__name__)
def test_pickle_survives_a_reload_of_the_buffers_module(buffer_cls, monkeypatch):
    """Pickle a buffer built before a reload of the buffers module rebinds ``TransportBuffer``."""
    ref = _volume_ref(object())
    buffer = buffer_cls(ref)
    monkeypatch.setattr(
        "torchstore.transport.buffers.TransportBuffer", type("TransportBuffer", (), {})
    )

    assert _wire(buffer).storage_volume_ref is None
    assert buffer.storage_volume_ref is ref


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
        return (self.__dict__.copy(), 7)

    def __setstate__(self, state: tuple) -> None:
        fields, version = state
        self.__dict__.update(fields, version=version)


class _ObjectState:
    """A later base's hook whose state is an object that is neither a dict nor a container."""

    def __getstate__(self) -> types.SimpleNamespace:
        return types.SimpleNamespace(fields=self.__dict__.copy(), version=7)

    def __setstate__(self, state: types.SimpleNamespace) -> None:
        self.__dict__.update(state.fields, version=state.version)


class _OrderedState:
    """A later base's hook whose ``OrderedDict`` state its restore relies on."""

    def __getstate__(self) -> collections.OrderedDict:
        return collections.OrderedDict(self.__dict__)

    def __setstate__(self, state: collections.OrderedDict) -> None:
        state.move_to_end("storage_volume_ref")
        self.__dict__.update(state)


class _OrderedSubclass(collections.OrderedDict):
    """An ``OrderedDict`` subclass a later base's hook may return."""


class _OrderedSubclassState(_OrderedState):
    """A later base's hook whose state is an ``OrderedDict`` subclass."""

    def __getstate__(self) -> collections.OrderedDict:
        return _OrderedSubclass(self.__dict__)


def _unseen() -> str:
    """The default factory of the ``defaultdict`` states below, which no other code uses."""
    return "unseen"


class _DefaultState:
    """A later base's hook whose ``defaultdict`` state its restore relies on and whose factory it records."""

    def __getstate__(self) -> collections.defaultdict:
        return collections.defaultdict(_unseen, self.__dict__)

    def __setstate__(self, state: collections.defaultdict) -> None:
        self.__dict__.update(state, seen=state["seen"], factory=state.default_factory)


class _DefaultSubclass(collections.defaultdict):
    """A ``defaultdict`` subclass a later base's hook may return."""


class _DefaultSubclassState(_DefaultState):
    """A later base's hook whose state is a ``defaultdict`` subclass."""

    def __getstate__(self) -> collections.defaultdict:
        return _DefaultSubclass(_unseen, self.__dict__)


class _FrozenState:
    """A later base's hook whose state is ``torch.fx``'s immutable dict and whose restore records its type."""

    def __getstate__(self) -> immutable_dict:
        return immutable_dict(self.__dict__)

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state, state_type=type(state))


def _with_later_hook(buffer_cls: type, hook: type) -> Any:
    """Subclass ``buffer_cls`` with ``hook`` listed after it, at module level so pickle finds it."""
    name = buffer_cls.__name__ + "With" + hook.__name__.lstrip("_")
    mixed = type(name, (buffer_cls, hook), {"__module__": __name__})
    globals()[name] = mixed
    return mixed


_DROPS_LOCK_BUFFERS = [_with_later_hook(cls, _DropsLock) for cls in BUFFERS]
_LATER_HOOK_BUFFERS = {
    hook: _with_later_hook(TransportBuffer, hook)
    for hook in (
        _InstanceDictState,
        _VersionedState,
        _ObjectState,
        _OrderedState,
        _OrderedSubclassState,
        _DefaultState,
        _DefaultSubclassState,
        _FrozenState,
    )
}


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
    buffer = _LATER_HOOK_BUFFERS[_InstanceDictState](ref)

    assert _wire(buffer).storage_volume_ref is None
    assert buffer.storage_volume_ref is ref


@pytest.mark.parametrize(
    "dict_type",
    [collections.OrderedDict, functools.partial(collections.defaultdict, None)],
    ids=["OrderedDict", "defaultdict"],
)
def test_a_later_base_hooks_dict_subclass_is_copied_before_the_handle_drops(dict_type):
    """Leave the client's own handle in place when that dict is an ``OrderedDict`` or ``defaultdict``."""
    ref = _volume_ref(object())
    buffer = _LATER_HOOK_BUFFERS[_InstanceDictState](ref)
    buffer.__dict__ = dict_type(vars(buffer))

    assert _wire(buffer).storage_volume_ref is None
    assert buffer.storage_volume_ref is ref


@pytest.mark.parametrize(
    "hook", [_VersionedState, _ObjectState], ids=["pair", "object"]
)
def test_a_later_base_hooks_state_other_than_a_dict_passes_through(hook):
    """Hand a state that is not a dict to the later base's restore as that hook built it."""
    buffer = _LATER_HOOK_BUFFERS[hook](_volume_ref(object()))
    buffer.field = "client"

    copied = vars(_wire(buffer))
    assert (copied["field"], copied["version"]) == ("client", 7)
    assert copied["storage_volume_ref"].volume_id == "volume-0"


@pytest.mark.parametrize("send", list(_SENDS.values()), ids=list(_SENDS))
@pytest.mark.parametrize(
    "hook",
    [_OrderedState, _OrderedSubclassState, _DefaultState, _DefaultSubclassState],
    ids=["OrderedDict", "OrderedDict-subclass", "defaultdict", "defaultdict-subclass"],
)
def test_a_later_base_hooks_dict_keeps_its_type(hook, send):
    """Hand a later base's restore an ``OrderedDict`` or ``defaultdict`` state as one again with the handle dropped."""
    ref = _volume_ref(object())
    buffer = _LATER_HOOK_BUFFERS[hook](ref)

    assert send(buffer).storage_volume_ref is None
    assert buffer.storage_volume_ref is ref


@pytest.mark.parametrize("send", list(_SENDS.values()), ids=list(_SENDS))
@pytest.mark.parametrize(
    "hook",
    [_DefaultState, _DefaultSubclassState],
    ids=["defaultdict", "defaultdict-subclass"],
)
def test_a_later_base_hooks_defaultdict_keeps_its_default_factory(hook, send):
    """Hand a later base's restore its ``defaultdict`` with the default factory that hook gave it."""
    copied = send(_LATER_HOOK_BUFFERS[hook](_volume_ref(object())))
    assert (copied.factory, copied.seen) == (_unseen, "unseen")


@pytest.mark.parametrize("send", list(_SENDS.values()), ids=list(_SENDS))
def test_a_later_base_hooks_other_dict_type_comes_back_as_a_plain_dict(send):
    """Hand a later base's restore its state of another dict type as a plain dict with the handle dropped."""
    ref = _volume_ref(object())
    buffer = _LATER_HOOK_BUFFERS[_FrozenState](ref)

    copied = send(buffer)
    assert (copied.state_type, copied.storage_volume_ref) == (dict, None)
    assert buffer.storage_volume_ref is ref


class _Slotted(TransportBuffer):
    """A subclass that keeps a field in ``__slots__`` and has no pickling hook of its own."""

    __slots__ = ("seq",)


@pytest.mark.parametrize("send", list(_SENDS.values()), ids=list(_SENDS))
def test_a_slotted_subclass_drops_the_handle_and_its_slot(send):
    """Copy a slotted subclass from its instance dict alone, the handle set to None."""
    ref = _volume_ref(object())
    buffer = _Slotted(ref)
    buffer.seq = 2

    copied = send(buffer)
    assert copied.storage_volume_ref is None
    assert not hasattr(copied, "seq")
    assert (buffer.seq, buffer.storage_volume_ref) == (2, ref)


class _EmptyRestored(TransportBuffer):
    """A subclass whose constructor sets nothing and whose restore records its state."""

    def __init__(self) -> None:
        pass

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state, restored_from=dict(state))


@pytest.mark.parametrize("send", list(_SENDS.values()), ids=list(_SENDS))
def test_an_empty_instance_dict_restores_from_an_empty_state(send):
    """Hand the restore of an instance with an empty dict the empty state, where Python handed none."""
    assert vars(send(_EmptyRestored())) == {"restored_from": {}}


class _NoHandle(TransportBuffer):
    """A subclass whose constructor sets no volume handle."""

    def __init__(self) -> None:
        self.field = "client"


@pytest.mark.parametrize("send", list(_SENDS.values()), ids=list(_SENDS))
def test_copies_add_no_handle_an_instance_never_had(send):
    """Add no handle to a copy of an instance whose constructor set none."""
    assert vars(send(_NoHandle())) == {"field": "client"}
