# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The carrier's wire format: frames, pickled bodies with tensors out of band, and errors.

A frame is a 40-byte header, one 8-byte length per buffer, the body, then the buffers.
Receivers check the header and the lengths before allocating anything for the rest.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import copyreg
import ctypes
import io
import json
import logging
import math
import pickle
import re
import struct
import sys
import traceback
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from functools import cache, partial
from typing import Any, get_type_hints, TypeVar

import torch
from torch.utils._python_dispatch import (
    _len_torch_dispatch_stack,
    _pop_mode,
    _push_mode,
)

__all__ = [
    "byte_view",
    "CallTimeoutError",
    "CarrierAddress",
    "CarrierError",
    "CarrierProtocolError",
    "check_record_brackets",
    "decode_error",
    "decode_header",
    "decode_lengths",
    "dumps",
    "encode_error",
    "encode_frame",
    "FrameHeader",
    "HEADER_SIZE",
    "Hello",
    "JsonRecord",
    "Kind",
    "LENGTH_SIZE",
    "loads",
    "MAGIC",
    "MAX_BUFFERS",
    "MAX_FRAME_BYTES",
    "MAX_HANDSHAKE_FRAME_BYTES",
    "PeerLostError",
    "PROTOCOL_VERSION",
    "Reject",
    "RemoteError",
    "StaleAddressError",
    "Welcome",
]

logger = logging.getLogger("torchstore.carrier")

MAGIC = b"TSC1"
PROTOCOL_VERSION = 1
# The frame length the package init sets for Monarch's large RPC calls.
MAX_FRAME_BYTES = 910_737_418_240
MAX_HANDSHAKE_FRAME_BYTES = 64 * 1024
MAX_BUFFERS = 1 << 20
# Records are flat, and json's C scanner recurses once per opening bracket outside a string.
_MAX_RECORD_BRACKETS = 64
# Escapes go first, so every quote left opens or closes a string.
_ESCAPE = re.compile(r"\\.", re.DOTALL)
_STRING = re.compile(r'"[^"]*"')
# Quantized dtypes that keep several values a byte, and how many.
_PACKED_DTYPES = {torch.quint4x2: 2, torch.quint2x4: 4}
# The methods that return a compressed sparse layout's indices.
_COMPRESSED_INDICES = {
    torch.sparse_csr: ("crow_indices", "col_indices"),
    torch.sparse_bsr: ("crow_indices", "col_indices"),
    torch.sparse_csc: ("ccol_indices", "row_indices"),
    torch.sparse_bsc: ("ccol_indices", "row_indices"),
}

# Magic, kind, flags, reserved, buffer count, request id, remaining deadline in
# microseconds, body length, and a spare reserved field. A request's deadline counts
# from its arrival, so 0 means it has already expired.
_HEADER = struct.Struct("!4sBBHIQQQI")
HEADER_SIZE = _HEADER.size
LENGTH_SIZE = 8


class CarrierError(Exception):
    """Base class for the carrier's errors."""


if sys.version_info >= (3, 11):

    class CallTimeoutError(CarrierError, TimeoutError):
        """A call or a connection attempt outlived its deadline."""

else:

    class CallTimeoutError(CarrierError, TimeoutError, asyncio.TimeoutError):
        """A call or a connection attempt outlived its deadline.

        Python 3.10's ``asyncio.TimeoutError`` is not the builtin ``TimeoutError``.
        """


class PeerLostError(CarrierError, ConnectionError):
    """The connection is gone: the peer closed, died or could not be reached, or this process closed its side."""


class StaleAddressError(PeerLostError):
    """The server at an address no longer serves the address's scope or incarnation."""


class CarrierProtocolError(CarrierError):
    """Data that breaks the carrier's wire format."""


class RemoteError(CarrierError):
    """A remote error not rebuilt here, kept as its type name and text."""

    def __init__(self, remote_type: str, message: str) -> None:
        super().__init__(remote_type, message)

    @property
    def remote_type(self) -> str:
        return self.args[0]

    def __str__(self) -> str:
        remote_type, message = self.args
        return f"{remote_type}: {message}" if message else remote_type


class _RemoteTraceback(Exception):
    """The traceback of an error raised in another process, as its cause here."""

    def __str__(self) -> str:
        return f'\n"""\n{self.args[0]}"""'


def check_record_brackets(text: str) -> None:
    """Raise ValueError if ``text`` holds over 64 opening brackets outside its JSON strings.

    Python 3.10 bounds json's scanner's depth only by the recursion limit, which may exceed a thread's stack.
    """
    outside = _STRING.sub("", _ESCAPE.sub("", text))
    if outside.count("[") + outside.count("{") > _MAX_RECORD_BRACKETS:
        raise ValueError(f"over {_MAX_RECORD_BRACKETS} brackets")


_TRecord = TypeVar("_TRecord", bound="JsonRecord")


@cache
def _field_types(cls: type) -> dict[str, Any]:
    return get_type_hints(cls)


class JsonRecord:
    """Base of frozen dataclasses of ``str`` and ``int`` fields sent as JSON objects."""

    def to_json(self) -> str:
        """Return the fields as a JSON object."""
        return json.dumps(vars(self), sort_keys=True)

    @classmethod
    def from_json(cls: type[_TRecord], data: str | bytes) -> _TRecord:
        """Parse a JSON object that holds every field with its type, ignoring other keys.

        Raises:
            CarrierProtocolError: If ``data`` is not such an object.
        """
        hints = _field_types(cls)
        if not isinstance(data, (bytes, str)):
            raise CarrierProtocolError(
                f"not a {cls.__name__} object: {type(data).__name__}, not bytes or str"
            )
        try:
            text = data.decode() if isinstance(data, bytes) else data
            check_record_brackets(text)
            fields = json.loads(text)
            values = {name: fields[name] for name in hints}
            # An escaped lone surrogate parses as a str that does not encode.
            for value in values.values():
                if type(value) is str:
                    value.encode()
        except (ValueError, TypeError, KeyError) as e:
            # Never the repr or the cause: a UnicodeDecodeError holds the whole input, scope included.
            raise CarrierProtocolError(
                f"not a {cls.__name__} object: {type(e).__name__}: {e}"
            ) from None
        wrong = [name for name, hint in hints.items() if type(values[name]) is not hint]
        if wrong:
            raise CarrierProtocolError(
                f"{cls.__name__} fields of the wrong type: {wrong}"
            )
        return cls(**values)


@dataclass(frozen=True)
class CarrierAddress(JsonRecord):
    """Where a server listens, and the service, scope and incarnation it serves.

    The scope, ``{store_name}/{nonce}``, stays out of the repr and so out of logs.
    """

    host: str
    port: int
    service: str
    scope: str = field(repr=False)
    incarnation: str


@dataclass(frozen=True)
class Hello(JsonRecord):
    """A hello's body: the server's identity and the sender's ``host:pid``."""

    scope: str = field(repr=False)
    service: str
    incarnation: str
    peer: str
    version: int = PROTOCOL_VERSION


@dataclass(frozen=True)
class Welcome(Hello):
    """A welcome's body: a hello's fields and how many requests a client may leave unanswered, 0 for any."""

    window: int = 0


@dataclass(frozen=True)
class Reject(JsonRecord):
    """The body of a reject: why the server refused a hello."""

    reason: str
    version: int = PROTOCOL_VERSION


class Kind(IntEnum):
    """What a frame carries. Handshake frames come before all others on a connection.

    A client sends HELLO, then REQUESTs of a pickled endpoint, args and kwargs, and CANCELs naming a
    request id. A server answers WELCOME or REJECT, then one REPLY or ERROR per request id.
    """

    HELLO = 1
    WELCOME = 2
    REJECT = 3
    REQUEST = 4
    REPLY = 5
    ERROR = 6
    CANCEL = 7


_HANDSHAKE_KINDS = frozenset((Kind.HELLO, Kind.WELCOME, Kind.REJECT))


@dataclass(frozen=True)
class FrameHeader:
    """A frame header that passed its checks."""

    kind: Kind
    request_id: int
    deadline_us: int
    buffer_count: int
    body_len: int


def encode_frame(
    kind: Kind,
    request_id: int = 0,
    *,
    deadline_us: int = 0,
    body: bytes = b"",
    buffers: Sequence[memoryview] = (),
) -> list[bytes | memoryview]:
    """Return a frame as the chunks to send in order, without copying ``buffers``.

    Raises:
        CarrierProtocolError: If the frame has too many buffers or bytes for its kind.
        ValueError: If the request id or deadline is not an unsigned 64-bit integer.
    """
    values = (request_id, deadline_us)
    if not all(isinstance(value, int) and 0 <= value < 1 << 64 for value in values):
        raise ValueError(
            f"request id {request_id} and deadline {deadline_us} us must be unsigned"
            " 64-bit integers"
        )
    lengths = [memoryview(buffer).nbytes for buffer in buffers]
    prefix = HEADER_SIZE + LENGTH_SIZE * len(lengths)
    _check_frame(kind, len(lengths), prefix + len(body) + sum(lengths))
    header = _HEADER.pack(
        MAGIC, kind, 0, 0, len(lengths), request_id, deadline_us, len(body), 0
    )
    return [header + struct.pack(f"!{len(lengths)}Q", *lengths), body, *buffers]


def decode_header(data: bytes, *, handshake: bool) -> FrameHeader:
    """Check a frame header, admitting only small handshake frames during the handshake.

    Raises:
        CarrierProtocolError: If the header is malformed or its frame is not allowed.
    """
    if len(data) != HEADER_SIZE:
        raise CarrierProtocolError(
            f"a frame header is {HEADER_SIZE} bytes, not {len(data)}"
        )
    (
        magic,
        kind,
        flags,
        reserved,
        count,
        request_id,
        deadline_us,
        body_len,
        spare,
    ) = _HEADER.unpack(data)
    if magic != MAGIC:
        raise CarrierProtocolError(f"bad frame magic {magic!r}")
    try:
        kind = Kind(kind)
    except ValueError:
        raise CarrierProtocolError(f"unknown frame kind {kind}") from None
    if flags or reserved or spare:
        raise CarrierProtocolError("reserved frame header fields are not zero")
    if (kind in _HANDSHAKE_KINDS) != handshake:
        when = "before" if handshake else "after"
        raise CarrierProtocolError(f"{kind.name} frame {when} the handshake")
    _check_frame(kind, count, HEADER_SIZE + LENGTH_SIZE * count + body_len)
    return FrameHeader(kind, request_id, deadline_us, count, body_len)


def decode_lengths(header: FrameHeader, data: bytes) -> tuple[int, ...]:
    """Check the buffer lengths that follow ``header``, before any buffer is allocated.

    Raises:
        CarrierProtocolError: If ``data`` is not one length per buffer, or the whole
            frame exceeds the frame cap.
    """
    if len(data) != LENGTH_SIZE * header.buffer_count:
        raise CarrierProtocolError(
            f"expected {LENGTH_SIZE * header.buffer_count} bytes of buffer lengths,"
            f" not {len(data)}"
        )
    lengths = struct.unpack(f"!{header.buffer_count}Q", data)
    prefix = HEADER_SIZE + LENGTH_SIZE * header.buffer_count
    _check_frame(
        header.kind, header.buffer_count, prefix + header.body_len + sum(lengths)
    )
    return lengths


def _check_frame(kind: Kind, buffer_count: int, frame_bytes: int) -> None:
    if kind in _HANDSHAKE_KINDS:
        max_buffers, cap = 0, MAX_HANDSHAKE_FRAME_BYTES
    else:
        max_buffers, cap = MAX_BUFFERS, MAX_FRAME_BYTES
    if buffer_count > max_buffers:
        raise CarrierProtocolError(
            f"{kind.name} frame with {buffer_count} buffers, over the limit of {max_buffers}"
        )
    if frame_bytes > cap:
        raise CarrierProtocolError(
            f"{kind.name} frame of {frame_bytes} bytes, over the {cap}-byte cap"
        )


def byte_view(tensor: torch.Tensor) -> memoryview:
    """Return a writable view of a contiguous CPU tensor's bytes that keeps it alive.

    Raises:
        ValueError: If ``tensor`` is not a contiguous CPU tensor whose memory holds its values.
    """
    # The codec's own tensor ops reach no torch function mode.
    with torch._C.DisableTorchFunction():
        return _byte_view(tensor)


def _byte_view(tensor: torch.Tensor) -> memoryview:
    if tensor.device.type != "cpu":
        raise ValueError(f"need a contiguous CPU tensor, not one on {tensor.device}")
    if tensor.is_nested or tensor.layout != torch.strided:
        kind = "nested" if tensor.is_nested else str(tensor.layout)
        raise ValueError(f"need a contiguous CPU tensor, not a {kind} one")
    if not torch._C._has_storage(tensor):
        raise ValueError("need a contiguous CPU tensor, not one without storage")
    if not tensor.is_contiguous():
        raise ValueError("need a contiguous CPU tensor, not a non-contiguous one")
    _check_storage(tensor)
    if tensor.is_conj() or tensor.is_neg():
        raise ValueError(
            "need a contiguous CPU tensor, not a lazy conjugate or negative view"
        )
    per = _PACKED_DTYPES.get(tensor.dtype)
    if per and tensor.storage_offset():
        raise ValueError("need a contiguous CPU tensor, not a view into packed bytes")
    nbytes = -(-tensor.numel() // per) if per else tensor.nbytes
    view = (ctypes.c_ubyte * nbytes).from_address(tensor.data_ptr())
    # The ctypes view does not own the memory it exposes.
    view._tensor = tensor  # type: ignore[attr-defined]
    return memoryview(view).cast("B")


def _check_storage(tensor: torch.Tensor) -> None:
    if not tensor.numel():
        return
    per = _PACKED_DTYPES.get(tensor.dtype)
    if tensor.is_nested:
        end = _nested_end(tensor)
    elif per:
        # Torch's sub-byte kernels read a packed tensor's values as one run of whole bytes from its
        # storage offset, which they count in bytes, whatever its shape or strides.
        end = tensor.storage_offset() + -(-tensor.numel() // per)
    else:
        end = (tensor.storage_offset() + _span(tensor)) * tensor.element_size()
    # Zero tensors, and freed or shrunk storages, do not hold the bytes a tensor spans.
    if not tensor.data_ptr() or tensor.untyped_storage().nbytes() < end:
        raise ValueError("need a tensor whose storage holds all of its bytes")


def _nested_end(tensor: torch.Tensor) -> int:
    """The bytes from a strided nested tensor's buffer start to the end of its last-reaching component."""
    sizes = tensor._nested_tensor_size().tolist()
    strides = tensor._nested_tensor_strides().tolist()
    offsets = tensor._nested_tensor_storage_offsets().tolist()
    end = 0
    for size, stride, offset in zip(sizes, strides, offsets):
        if all(size):
            span = 1 + sum((n - 1) * step for n, step in zip(size, stride))
            end = max(end, offset + span)
    return end * tensor.element_size()


def _check_values(tensor: torch.Tensor, *, attributes: bool = True) -> None:
    """Raise ``ValueError`` unless every storage ``_value_parts`` finds holds all of its bytes, reaching no torch function mode.

    Without ``attributes`` it skips the attribute tensors of a tensor torch prints by its own values alone.
    """
    with torch._C.DisableTorchFunction():
        for part in _value_parts(tensor, {}, attributes):
            _check_storage(part)


def _value_parts(
    tensor: torch.Tensor, seen: dict[int, torch.Tensor], attributes: bool
) -> Iterator[torch.Tensor]:
    """Yield depth first each tensor whose own storage holds values of ``tensor`` or of the tensors it keeps.

    Those are tensors with storage but not meta or zero ones, sparse tensors' indices and values, and own-dispatch
    subclasses that hold their own bytes. Without ``attributes`` the walk skips the attributes and slots of tensors
    without their own dispatch. ``seen`` keeps each tensor the walk visits alive.
    """
    if id(tensor) in seen:
        return
    seen[id(tensor)] = tensor
    own_dispatch = (
        type(tensor).__torch_dispatch__ is not torch.Tensor.__torch_dispatch__
    )
    inner: list[torch.Tensor] = []
    if attributes or own_dispatch:
        inner = [
            value for value in _attributes(tensor) if isinstance(value, torch.Tensor)
        ]
    if tensor.layout == torch.sparse_coo:
        inner += [tensor._indices(), tensor._values()]
    elif tensor.layout in _COMPRESSED_INDICES:
        names = _COMPRESSED_INDICES[tensor.layout]
        inner += [*(getattr(tensor, name)() for name in names), tensor.values()]
    elif own_dispatch:
        flatten = getattr(tensor, "__tensor_flatten__", None)
        if flatten is not None:
            # DTensor also names its device mesh.
            parts = [getattr(tensor, name) for name in flatten()[0]]
            inner += [part for part in parts if isinstance(part, torch.Tensor)]
        if _holds_own_bytes(tensor):
            yield tensor
    elif torch._C._has_storage(tensor) and not (
        tensor.is_meta or tensor._is_zerotensor()
    ):
        yield tensor
    for part in inner:
        yield from _value_parts(part, seen, attributes)


def _attributes(obj: Any) -> list[Any]:
    """The values of ``obj``'s attributes and of the slots its class declares."""
    names = copyreg._slotnames(type(obj))  # type: ignore[attr-defined]
    return [*vars(obj).values(), *(getattr(obj, name, None) for name in names)]


def _holds_own_bytes(tensor: torch.Tensor) -> bool:
    """Whether a subclass with its own dispatch keeps values in its own storage. One that ``_make_subclass`` made does.

    A wrapper subclass's storage has a size but no memory, and some wrappers refuse access to it.
    """
    holds = False
    with contextlib.suppress(RuntimeError, NotImplementedError):
        holds = torch._C._has_storage(tensor) and (
            bool(tensor.data_ptr()) or not tensor.untyped_storage().nbytes()
        )
    return holds


def _span(tensor: torch.Tensor) -> int:
    """Return how many elements of storage a non-empty tensor spans, first to last."""
    sizes, strides = tensor.shape, tensor.stride()
    return 1 + sum((size - 1) * stride for size, stride in zip(sizes, strides))


@contextlib.contextmanager
def _outside_dispatch_modes() -> Iterator[None]:
    # Pre-dispatch modes sit in process-wide slots, so this thread only stops dispatching to them. Torch's own
    # tensor repr, which an error body's unpickling may run outside _TensorRepr, pops and re-pushes them,
    # turning this thread's key on.
    key = torch._C.DispatchKey.PreDispatch
    included = torch._C._dispatch_tls_is_dispatch_key_included(key)
    modes: list[Any] = [_pop_mode() for _ in range(_len_torch_dispatch_stack())]
    try:
        with torch._C._ExcludeDispatchKeyGuard(torch._C.DispatchKeySet(key)):
            yield
    finally:
        torch._C._dispatch_tls_set_dispatch_key_included(key, included)
        for mode in reversed(modes):
            _push_mode(mode)


# As in Monarch's unpickling, a caller's dispatch mode, such as FakeTensorMode, which
# refuses real tensors, must not see the codec's own tensor ops.
@_outside_dispatch_modes()
def dumps(obj: Any) -> tuple[bytes, list[memoryview]]:
    """Pickle ``obj``, sending each plain tensor's bytes, and ``bytes`` in a ``PickleBuffer``, out of band.

    Buffers alias contiguous CPU tensors, but for lazy conjugate, negative and zero ones, and copy others to CPU.
    Their dtype, shape, ``requires_grad`` and attributes survive.
    An object pickle refuses, such as a lambda or a lock, raises pickle's own error.

    Raises:
        ValueError: If a tensor's storage does not hold all of its bytes.
    """
    file = io.BytesIO()
    pickler = _Pickler(file)
    # An object's own reduction may print its tensors.
    with _TensorRepr():
        pickler.dump(obj)
    return file.getvalue(), pickler.buffers


@_outside_dispatch_modes()
def loads(body: bytes | memoryview, buffers: Sequence[torch.Tensor] = ()) -> Any:
    """Unpickle a body whose tensors become views of the received ``uint8`` buffers.

    A ``PickleBuffer`` over ``bytes`` arrives as a read-only ``memoryview``. Typed storages load on CPU, and as in Monarch a bare
    ``UntypedStorage`` fails here while a float8 typed one already fails to dump. Unpickling errors propagate unchanged.
    """
    views = [byte_view(buffer) for buffer in buffers]
    file = io.BytesIO(body) if type(body) is bytes else _BodyFile(body)
    # An object's own constructor may print its tensors.
    with _TensorRepr():
        return _Unpickler(file, buffers=views).load()


class _BodyFile:
    """A read-only file over a body, so unpickling a memoryview never copies all of it first."""

    def __init__(self, body: bytes | memoryview) -> None:
        self._view, self._pos = memoryview(body).cast("B"), 0

    def read(self, size: int = -1) -> bytes:
        end = len(self._view) if size < 0 else min(self._pos + size, len(self._view))
        data, self._pos = self._view[self._pos : end].tobytes(), end
        return data

    def readinto(self, target: Any) -> int:
        with memoryview(target) as view, view.cast("B") as out:
            count = min(len(out), len(self._view) - self._pos)
            out[:count] = self._view[self._pos : self._pos + count]
        self._pos += count
        return count

    def readline(self) -> bytes:
        start = self._pos
        while self._pos < len(self._view):
            chunk = self._view[self._pos : self._pos + 4096].tobytes()
            index = chunk.find(b"\n")
            if index >= 0:
                self._pos += index + 1
                break
            self._pos += len(chunk)
        return self._view[start : self._pos].tobytes()


class _FieldsUnpickler(pickle.Unpickler):
    """Unpickles an error frame body's fields, a tuple of str, bytes and None, refusing any global it names."""

    def find_class(self, module: str, name: str) -> Any:
        raise pickle.UnpicklingError(f"it names a global, {module}.{name}")


class _Unpickler(pickle.Unpickler):
    """Unpickles with torch's storage loader and sparse rebuild replaced by the codec's."""

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) == ("torch.storage", "_load_from_bytes"):
            return _load_storage
        if (module, name) == ("torch._utils", "_rebuild_sparse_tensor"):
            return _rebuild_sparse_tensor
        return super().find_class(module, name)


class _ByReference(pickle.Pickler):
    """Pickles as stdlib pickle does, but passes tensors and typed storages by reference."""

    def __init__(self, file: io.BytesIO) -> None:
        super().__init__(file, protocol=5)
        self.objects: list[Any] = []

    def persistent_id(self, obj: Any) -> int | None:
        if isinstance(obj, (torch.Tensor, torch.TypedStorage)):
            # Torch's own reduction sends only the bytes a storage still holds, and its rebuild then fails or makes up the rest.
            if isinstance(obj, torch.Tensor):
                _check_values(obj)
            self.objects.append(obj)
            return len(self.objects) - 1
        return None


class _InPlaceUnpickler(_Unpickler):
    """Unpickles a ``_ByReference`` pickle, taking its tensors and storages back as they are."""

    def __init__(self, file: io.BytesIO, objects: list[Any]) -> None:
        super().__init__(file)
        self._objects = objects

    def persistent_load(self, pid: Any) -> Any:
        return self._objects[pid]


def _load_storage(data: bytes) -> Any:
    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)


def _rebuild_sparse_tensor(layout: torch.layout, data: tuple[Any, ...]) -> torch.Tensor:
    # torch's own rebuild keeps each tensor in a list that only torch.load's end checks and
    # clears. Here its parts have arrived, so the constructor checks it as the global setting
    # says, passed explicitly as torch warns otherwise. An explicit device keeps out a caller's
    # default device, such as torch.device("meta").
    checked = torch.sparse.check_sparse_tensor_invariants.is_enabled()
    # The codec's own tensor ops reach no torch function mode, loads' _TensorRepr included.
    with torch._C.DisableTorchFunction():
        if layout == torch.sparse_coo:
            indices, values, size, is_coalesced = data
            return torch.sparse_coo_tensor(
                indices,
                values,
                size,
                is_coalesced=is_coalesced,
                device=values.device,
                check_invariants=checked,
            )
        return torch.sparse_compressed_tensor(
            *data, layout=layout, device=data[2].device, check_invariants=checked
        )


class _Pickler(pickle.Pickler):
    """Pickles with protocol 5, sending plain tensors and ``PickleBuffer``s over ``bytes`` out of band."""

    def __init__(self, file: io.BytesIO) -> None:
        self.buffers: list[memoryview] = []
        # Held until pickling ends, so no other buffer can reuse an id.
        self._tensor_buffers: dict[int, pickle.PickleBuffer] = {}
        callback = partial(_keep_in_band, self._tensor_buffers, self.buffers)
        super().__init__(file, protocol=5, buffer_callback=callback)

    def reducer_override(self, obj: Any) -> Any:  # type: ignore[override]
        # A PickleBuffer arrives as a read-only 1-D byte memoryview, which pickle cannot send
        # back. Other views keep pickle's TypeError, as they would arrive as flat bytes.
        if (
            type(obj) is memoryview
            and obj.readonly
            and obj.format == "B"
            and obj.ndim == 1
            and obj.contiguous
        ):
            # Out of band only over a whole bytes object itself, never a view, for the reason the
            # tensors' ctypes arrays avoid one. A slice stays in the body.
            whole = type(obj.obj) is bytes and obj.nbytes == len(obj.obj)
            return memoryview, (pickle.PickleBuffer(obj.obj if whole else obj),)
        if not issubclass(type(obj), torch.Tensor):
            return NotImplemented
        # The codec's own tensor ops reach no torch function mode, dumps' _TensorRepr included.
        with torch._C.DisableTorchFunction():
            return self._reduce_tensor(obj)

    def _reduce_tensor(self, obj: torch.Tensor) -> Any:
        # Torch pickles sparse, quantized, nested and meta tensors and subclasses, and
        # Parameters reach here through their data.
        if type(obj) is not torch.Tensor or (
            obj.layout != torch.strided
            or obj.is_quantized
            or obj.is_nested
            or obj.is_meta
        ):
            # Torch's own reduction sends only the bytes a storage still holds, and its rebuild then fails or makes up the rest.
            _check_values(obj)
            return NotImplemented
        # Copying a tensor whose storage lost its bytes would read memory it does not own.
        if not obj._is_zerotensor():
            _check_storage(obj)
        data, strides = obj.detach(), None
        span = _span(obj) if obj.numel() else 0
        # A tensor that spans fewer storage elements than it has sends only that span, and its strides.
        if span < obj.numel():
            data, strides = data.as_strided((span,), (1,)), obj.stride()
        data = data.resolve_conj().resolve_neg()
        # Copied as bytes, which any dtype of their width shares: torch's sub-byte dtypes have no copy kernel.
        if data.element_size() == 1:
            data = data.view(torch.uint8)
        data = data.cpu().contiguous()
        # A span still in the caller's memory is copied too, as for any other non-contiguous tensor.
        if strides is not None and data.data_ptr() == obj.data_ptr():
            data = data.clone()
        # A zero tensor has no memory to view, so its zeros are materialized.
        if data._is_zerotensor():
            data = data.clone()
        view = byte_view(data).obj
        # A copy made here cannot change later, so a server's reply copy skips it.
        view._private = data.data_ptr() != obj.data_ptr()  # type: ignore[attr-defined]
        # Over the ctypes array: Python 3.10 crashes collecting a cycle that holds a
        # memoryview while another memoryview, such as the frame's raw buffer, exports it.
        buffer = pickle.PickleBuffer(view)
        self._tensor_buffers[id(buffer)] = buffer
        state = vars(obj) or None
        args = (buffer, obj.dtype, tuple(obj.shape), obj.requires_grad, state, strides)
        return _rebuild_tensor, args


def _keep_in_band(
    tensor_buffers: dict[int, pickle.PickleBuffer],
    out: list[memoryview],
    buffer: pickle.PickleBuffer,
) -> bool:
    if id(buffer) not in tensor_buffers:
        # Bytes never change, so a reply pickled once can go to every caller uncopied.
        with memoryview(buffer) as view:
            if type(view.obj) is not bytes:
                return True
    out.append(buffer.raw())
    return False


def _rebuild_tensor(
    data: memoryview,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    requires_grad: bool,
    state: dict[str, Any] | None,
    strides: tuple[int, ...] | None,
) -> torch.Tensor:
    # The codec's own tensor ops reach no torch function mode, loads' _TensorRepr included.
    with torch._C.DisableTorchFunction():
        # loads hands out each received tensor as its byte_view, which holds it.
        received = data.obj._tensor.view(dtype)  # type: ignore[union-attr]
        # An expanded tensor arrives as the storage it spans, with its strides.
        if strides is None:
            tensor = received.reshape(shape)
        else:
            tensor = received.as_strided(shape, strides)
        tensor.requires_grad_(requires_grad)
    if state:
        vars(tensor).update(state)
    return tensor


_CANCELLATIONS = (asyncio.CancelledError, concurrent.futures.CancelledError)
# A coroutine turns a raised StopIteration into RuntimeError, and StopAsyncIteration
# raised in __anext__ ends the caller's async for without an error.
_CONTROL_FLOW = (StopIteration, StopAsyncIteration)


@_outside_dispatch_modes()
def encode_error(error: BaseException) -> bytes:
    """Return an error frame's body: ``error``'s type name, text and traceback, and itself.

    The error itself travels only if it rebuilds here intact, and a text or traceback that fails becomes a placeholder.
    """
    pickled = None
    try:
        # Its reduction and constructor are its own code, which may print its tensors.
        with _TensorRepr():
            candidate = pickle.dumps(error, protocol=5)
            if _rebuilt_intact(error, _rebuilt_sharing_tensors(error)):
                pickled = candidate
    except Exception as e:
        logger.debug(
            "carrier sends %s as text: %s: %s",
            _type_name(type(error)),
            _type_name(type(e)),
            _text(e),
        )
    try:
        with _TensorRepr():
            remote_traceback = "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            )
    except Exception:
        remote_traceback = f"<{_type_name(type(error))} whose traceback failed>\n"
    return pickle.dumps(
        (_type_name(type(error)), _text(error), remote_traceback, pickled), protocol=5
    )


@_outside_dispatch_modes()
def decode_error(body: bytes | memoryview) -> Exception:
    """Rebuild a remote ``Exception``, or return a ``RemoteError`` with its type and text.

    Cancellations, ``StopIteration`` and ``StopAsyncIteration``, ``BaseException`` types and errors not rebuilt intact become one.

    Raises:
        CarrierProtocolError: If ``body`` is not an error frame body.
    """
    try:
        fields = _FieldsUnpickler(io.BytesIO(body)).load()
    except Exception as e:
        # Capped and without its cause: the error can quote much of the body, which may be as long as a frame.
        reason = f"{type(e).__name__}: {_text(e)}"[:200]
        raise CarrierProtocolError(f"not an error frame body: {reason}") from None
    # No subclass gets here: the fields unpickler refuses the class that building one names.
    if not (
        isinstance(fields, tuple)
        and len(fields) == 4
        and all(isinstance(field, str) for field in fields[:3])
        and (fields[3] is None or isinstance(fields[3], bytes))
    ):
        raise CarrierProtocolError("not an error frame body: wrong fields or types")
    remote_type, text, remote_traceback, pickled = fields
    error = None
    if pickled is not None:
        try:
            with _TensorRepr():
                error = _Unpickler(io.BytesIO(pickled)).load()
        except Exception as e:
            logger.debug(
                "carrier cannot rebuild remote %s: %s: %s",
                remote_type,
                _type_name(type(e)),
                _text(e),
            )
    # Re-raised in a caller, a rebuilt BaseException such as CancelledError, or the
    # concurrent.futures one that asyncio converts to it, would look like its own.
    if not (
        isinstance(error, Exception)
        and not isinstance(error, _CANCELLATIONS)
        and not isinstance(error, _CONTROL_FLOW)
        and _type_name(type(error)) == remote_type
    ):
        error = RemoteError(remote_type, text)
    # Past the class's own __setattr__, which may refuse assignment, as a frozen dataclass's does.
    object.__setattr__(error, "__cause__", _RemoteTraceback(remote_traceback))
    return error


def _rebuilt_sharing_tensors(error: BaseException) -> Any:
    """Rebuild ``error`` through its own reduction, sharing its tensors, so no storage is loaded, allocated or read.

    A bare ``UntypedStorage``, which torch's own rebuild fails to load, is rebuilt and fails here too.
    """
    file = io.BytesIO()
    pickler = _ByReference(file)
    pickler.dump(error)
    return _InPlaceUnpickler(io.BytesIO(file.getvalue()), pickler.objects).load()


def _rebuilt_intact(error: BaseException, rebuilt: Any) -> bool:
    """Whether a rebuilt copy of ``error`` has its args and state, or else its text. Decoding checks its type.

    Equal args and state make an equal error even where a set among them prints in another order.
    """
    try:
        if _same(rebuilt.args, error.args) and _same(_state(rebuilt), _state(error)):
            return True
    except Exception:
        pass  # A NumPy array among them, say, has no single truth value.
    return _text(rebuilt) == _text(error)


def _state(error: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return what besides its args pickling may drop: its attributes and its slots."""
    names = copyreg._slotnames(type(error))  # type: ignore[attr-defined]
    return vars(error), {name: getattr(error, name, None) for name in names}


def _same(a: Any, b: Any) -> bool:
    """Whether ``a`` equals ``b``, item by item through plain tuples, lists and dicts, where a NaN matches a NaN.

    A copy that shares the error's tensors matches them by identity, so their values are never read.
    """
    if a is b:
        return True
    if type(a) in (tuple, list) and type(b) is type(a):
        return len(a) == len(b) and all(map(_same, a, b))
    if type(a) is dict and type(b) is dict:
        return a.keys() == b.keys() and all(_same(a[key], b[key]) for key in a)
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a):
        return math.isnan(b)
    # Comparing a tensor that a copy's own reduction rebuilt is one of the codec's own ops.
    with torch._C.DisableTorchFunction():
        return bool(a == b)


def _type_name(cls: type) -> str:
    # As in a class's repr, a class without a str __module__, as type() makes under
    # globals without __name__, goes by its bare name.
    module = getattr(cls, "__module__", None)
    if not isinstance(module, str) or module == "builtins":
        return cls.__qualname__
    return f"{module}.{cls.__qualname__}"


class _TensorRepr(torch.overrides.TorchFunctionMode):
    """Prints tensors as torch does, but without popping the process-wide pre-dispatch modes.

    Torch's repr pops them while it prints, so another thread's export would lose its ops. Tensors
    formatted, as f-strings and DTensor's repr do, and the reprs a repr nests come back here too.
    """

    def __init__(self) -> None:
        super().__init__()
        # Reprs in progress, whose own ops reach no other torch function mode.
        self._printing = 0

    def __torch_function__(
        self, func: Any, types: Any, args: Any = (), kwargs: Any = None
    ) -> Any:
        kwargs = kwargs or {}
        prints = func is torch.Tensor.__format__ or func is torch.Tensor.__repr__
        if prints:
            # Printing a tensor whose storage lost its bytes would read memory it does not own.
            _check_values(args[0], attributes=False)
        if func is torch.Tensor.__format__ and not _formats_as_item(args[0]):
            # Torch's own fallback prints str(tensor), which must reach this mode.
            with self:
                return object.__format__(*args)
        if func is torch.Tensor.__repr__:
            self._printing += 1
            try:
                # As torch's own _str prints, but leaving the dispatch modes in place. Its functorch
                # guard changes no text, as _str_intern unwraps a functorch tensor itself.
                with self, torch.no_grad():
                    return torch._tensor_str._str_intern(args[0], **kwargs)
            finally:
                self._printing -= 1
        if self._printing:
            with torch._C.DisableTorchFunction():
                return func(*args, **kwargs)
        return func(*args, **kwargs)


def _formats_as_item(tensor: torch.Tensor) -> bool:
    """Whether torch formats a tensor as its item, printing no tensor."""
    with torch._C.DisableTorchFunction():
        return tensor.dim() == 0 and not tensor.is_meta and type(tensor) is torch.Tensor


def _text(error: BaseException) -> str:
    try:
        with _TensorRepr():
            text = str(error)
    except Exception:
        return f"<{_type_name(type(error))} whose str() failed>"
    # A __str__ may return a str subclass, which need not pickle or import in the caller.
    return str.__str__(text)
