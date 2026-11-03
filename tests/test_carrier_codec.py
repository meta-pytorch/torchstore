# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the carrier's frames, bodies and errors, which need no sockets."""

import asyncio
import concurrent.futures
import contextlib
import dataclasses
import errno
import gc
import inspect
import io
import json
import logging
import math
import os
import pickle
import queue
import struct
import subprocess
import sys
import threading
import traceback
import types
import weakref
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, cast

import pytest
import torch
import torch.distributed as dist
import torchstore
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch.utils._python_dispatch import (
    _get_current_dispatch_mode_stack,
    TorchDispatchMode,
)
from torchstore.carrier import (
    CallTimeoutError,
    CarrierAddress,
    CarrierError,
    CarrierProtocolError,
    codec,
    PeerLostError,
    RemoteError,
    StaleAddressError,
)
from torchstore.carrier.codec import Hello, Kind, Reject

pytestmark = pytest.mark.timeout(120)

# The child process imports the tree that pytest imported.
_ROOT = str(Path(torchstore.__file__).parents[1])
_QUANTIZED = {torch.qint8, torch.quint8, torch.qint32, torch.quint4x2, torch.quint2x4}
_DTYPES = sorted(
    {value for value in vars(torch).values() if isinstance(value, torch.dtype)}
    - _QUANTIZED,
    key=str,
)
_ONE_BYTE_DTYPES = [dtype for dtype in _DTYPES if dtype.itemsize == 1]
_HANDSHAKE_KINDS = (Kind.HELLO, Kind.WELCOME, Kind.REJECT)

# Imports the codec from the tree in argv[1] and lands buffers in new uint8 tensors as a receiver does.
_CHILD_RECEIVES = """
import sys

sys.path.insert(0, sys.argv[1])
import torch
from torchstore.carrier import codec


def received(buffers):
    tensors = [torch.empty(buffer.nbytes, dtype=torch.uint8) for buffer in buffers]
    for buffer, target in zip(buffers, tensors):
        codec.byte_view(target)[:] = buffer
    return tensors
"""


_NO_NUMPY_CHILD = (
    """
import sys

sys.modules["numpy"] = None  # Importing NumPy now raises ImportError.
"""
    + _CHILD_RECEIVES
    + """
tensor = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)
body, buffers = codec.dumps(tensor)
decoded = codec.loads(body, received(buffers))
assert decoded.dtype == torch.bfloat16 and torch.equal(decoded, tensor), decoded
"""
)


_CYCLE_CHILD = """
import gc, sys

sys.path.insert(0, sys.argv[1])
import torch
from torchstore.carrier import codec


class Cycle:
    pass


body, buffers = codec.dumps(torch.ones(1 << 16))
# Like a failed send's frames and chunk lists, made after the buffers they hold.
cycle = Cycle()
cycle.self, cycle.chunks = cycle, [body, *buffers]
del body, buffers, cycle
gc.collect()
"""


_DEEP_RECORD_CHILD = """
import sys, threading

sys.path.insert(0, sys.argv[1])
from torchstore.carrier import codec
from torchstore.carrier.codec import CarrierProtocolError, Hello

# Python 3.10's json scanner recurses on the C stack, which this limit no longer guards.
sys.setrecursionlimit(100_000)
# A thread stack this small overflows if either payload reaches the scanner.
threading.stack_size(2 << 20)
limit = codec.MAX_HANDSHAKE_FRAME_BYTES - codec.HEADER_SIZE
refused = []


def parse(payload):
    try:
        Hello.from_json(payload)
    except CarrierProtocolError as e:
        refused.append(str(e))


# Off the main thread, as the I/O thread parses a hello: nested arrays, then nested objects.
for payload in (b"[" * limit, b'{"":' * (limit // 4)):
    thread = threading.Thread(target=parse, args=(payload,))
    thread.start()
    thread.join()
assert refused == ["not a Hello object: ValueError: over 64 brackets"] * 2, refused
"""


_SET_ERROR_CHILD = """
import math, sys

sys.modules["monarch"] = None  # The codec never needs Monarch.
sys.path.insert(0, sys.argv[1])
import torch
from torchstore.carrier import codec

keys = {"weights", "bias", "scale", "zero_point"}
# Beside a tensor or a NaN, args that compare unequal or not at all would leave the set's print order to decide.
extras = [(), (torch.arange(3.0),), (float("nan"),), ([float("nan")],), ()]
if sys.argv[2] == "encode":
    bodies = []
    for i, extra in enumerate(extras):
        try:
            raise KeyError("unknown keys", keys, *extra)
        except KeyError as e:
            if i == 4:
                e.loss = float("nan")  # A NaN among its attributes.
            bodies.append(codec.encode_error(e))
    for body in bodies:
        print("body", body.hex(), flush=True)
else:
    bodies = [bytes.fromhex(line.split()[1]) for line in sys.stdin if line.startswith("body ")]
for i, body in enumerate(bodies):
    decoded = codec.decode_error(body)
    assert type(decoded) is KeyError and decoded.args[:2] == ("unknown keys", keys), repr(decoded)
    extra = decoded.args[2:]
    if i % 5 == 0:
        assert extra == (), repr(decoded)
    elif i % 5 == 1:
        assert len(extra) == 1 and torch.equal(extra[0], torch.arange(3.0)), repr(decoded)
    elif i % 5 == 2:
        assert len(extra) == 1 and math.isnan(extra[0]), repr(decoded)
    elif i % 5 == 3:
        assert len(extra) == 1 and math.isnan(extra[0][0]), repr(decoded)
    else:
        assert extra == () and math.isnan(decoded.loss), repr(decoded)
print("decoded", len(bodies), flush=True)
"""


# Torch warns once per process when a sparse tensor is built without saying whether to check it.
_SPARSE_WARNING_CHILD = (
    _CHILD_RECEIVES
    + """
import warnings

dense = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
with warnings.catch_warnings(record=True) as caught:
    for tensor in (dense.to_sparse(), dense.to_sparse_csr()):
        body, buffers = codec.dumps(tensor)
        assert torch.equal(codec.loads(body, received(buffers)).to_dense(), dense)
invariants = [str(warning.message) for warning in caught if "invariant" in str(warning.message)]
assert not invariants, invariants
"""
)


class _CustomError(Exception):
    """An error class that both ends import."""


class _TwoArgError(Exception):
    """An error whose constructor cannot take its own ``args`` back."""

    def __init__(self, key: str, reason: str) -> None:
        super().__init__(f"{key}: {reason}")


class _PrefixError(Exception):
    """An error whose constructor rewrites its text, so a rebuilt copy reads differently."""

    def __init__(self, message: str) -> None:
        super().__init__(f"prefix {message}")


class _AliasError(Exception):
    """An error that pickles as another class, as a class renamed across versions would."""

    def __reduce__(self) -> tuple[Any, ...]:
        return ValueError, self.args


class _CheckedError(Exception):
    """An error whose constructor refuses its own text, so rebuilding it raises ``_CustomError``, which no narrowed catch names."""

    def __init__(self, code: int) -> None:
        if not isinstance(code, int):
            raise _CustomError(f"code must be an int, not {code!r}")
        super().__init__(f"code {code}")


class _SlotError(Exception):
    """An error that keeps its shard in a slot, which ``BaseException``'s reduction drops though its args survive."""

    __slots__ = ("shard",)

    def __init__(self, message: str, shard: int = -1) -> None:
        super().__init__(message)
        self.shard = shard

    def __str__(self) -> str:
        return f"{self.args[0]} (shard {self.shard})"


class _LossyReduceError(Exception):
    """An error whose own reduction drops its shard though its args survive."""

    def __init__(self, message: str, shard: int = -1) -> None:
        super().__init__(message)
        self.shard = shard

    def __reduce__(self) -> tuple[Any, ...]:
        return _LossyReduceError, (self.args[0],)

    def __str__(self) -> str:
        return f"{self.args[0]} (shard {self.shard})"


class _TruncatingError(Exception):
    """An error whose own reduction drops its trailing args, which its text prints."""

    def __reduce__(self) -> tuple[Any, ...]:
        return type(self), self.args[:1]


class _ListifyingError(Exception):
    """An error whose own reduction rebuilds its tuple argument as a list of the same items."""

    def __reduce__(self) -> tuple[Any, ...]:
        return type(self), (list(self.args[0]),)


class _DroppedAttrError(Exception):
    """An error whose own reduction drops its attributes, one of which its text reads."""

    def __init__(self, message: str, shard: int | None = None) -> None:
        super().__init__(message)
        if shard is not None:
            self.shard = shard

    def __reduce__(self) -> tuple[Any, ...]:
        return type(self), self.args

    def __str__(self) -> str:
        return f"{self.args[0]} (shard {getattr(self, 'shard', '?')})"


class _ChildSlotError(_SlotError):
    """An error whose shard sits in its parent's slot, beside a slot of its own."""

    __slots__ = ("rank",)

    def __init__(self, message: str, shard: int = -1, rank: int = -1) -> None:
        super().__init__(message, shard)
        self.rank = rank


class _ReduceFormatsError(Exception):
    """An error whose own reduction formats its tensor, as a reduction that logs might."""

    def __reduce__(self) -> tuple[Any, ...]:
        return type(self), self.args, {"note": f"bad shard {self.args[0]}"}


class _SenderRefusedError(Exception):
    """An error whose class refuses to rebuild while ``refusing`` is set, as on a sender whose rebuild fails."""

    refusing = False

    def __init__(self, *args: Any) -> None:
        if _SenderRefusedError.refusing:
            raise _CustomError("cannot rebuild here")
        super().__init__(*args)


class _Incomparable:
    """An object whose ``==`` raises ``_CustomError``, which no narrowed catch names, as a NumPy array's truth value raises."""

    def __eq__(self, other: object) -> bool:
        raise _CustomError("not comparable")

    __hash__ = object.__hash__

    def __repr__(self) -> str:
        return "_Incomparable()"


class _BadStrError(Exception):
    """An error whose ``str()`` raises ``_CustomError``, which no narrowed catch names."""

    def __str__(self) -> str:
        raise _CustomError("no text")


class _BadReduceError(Exception):
    """An error whose reduction raises ``_CustomError``, as a broken ``__reduce__`` might."""

    def __reduce__(self) -> tuple[Any, ...]:
        raise _CustomError("cannot reduce")


def _refuse_rebuild() -> None:
    raise ValueError("cannot rebuild", torch.arange(3.0))


class _UnrebuildableError(Exception):
    """An error whose rebuild raises an error holding a tensor, which a debug line then prints."""

    def __reduce__(self) -> tuple[Any, ...]:
        return _refuse_rebuild, ()


class _BadTextSyntaxError(SyntaxError):
    """A ``SyntaxError`` whose traceback formatting raises an error no narrowed catch would name."""

    @property
    def text(self) -> str:  # type: ignore[override]
        raise _CustomError("no source line")


class _TextError(Exception):
    """An error whose ``str()`` returns its argument as it is, which may be a str subclass."""

    def __str__(self) -> str:
        return self.args[0]


class _Text(str):
    """A str subclass, which an error body's fields must not be."""


class _LazyTextError(Exception):
    """An error that formats its tensor only when printed, as an f-string does."""

    def __str__(self) -> str:
        return f"bad shard {self.args[0]}"


class _CloningError(Exception):
    """An error whose own reduction rebuilds its tensor as a clone so a rebuilt copy holds equal values in a new tensor."""

    def __reduce__(self) -> tuple[Any, ...]:
        return type(self), (self.args[0].clone(),)


class _InitFormatsError(Exception):
    """An error that keeps its tensor as its argument and formats it when built, so rebuilding it prints the tensor."""

    def __init__(self, tensor: torch.Tensor) -> None:
        super().__init__(tensor)
        self.note = f"bad shard {tensor}"


class _SummingError(Exception):
    """An error whose ``str()`` computes with the tensor it holds."""

    def __str__(self) -> str:
        return f"column sums {self.args[0].sum(dim=0).tolist()}"


@dataclass(frozen=True)
class _FrozenError(Exception):
    """An error whose class refuses attribute assignment, as a frozen dataclass does."""


@dataclass(frozen=True)
class _FrozenFieldError(Exception):
    """A frozen dataclass error with a field, which ``BaseException`` would restore by the assignment it refuses."""

    shard: int


class _ImmutableError(Exception):
    """An error whose ``__setattr__`` refuses every assignment."""

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable")


class _CancelledException(asyncio.CancelledError, Exception):
    """A cancellation that is also an ``Exception``, so only the cancellation check refuses it."""


class _Subclass(torch.Tensor):
    """A tensor subclass that both ends import."""


class _Wrapper(torch.Tensor):
    """A wrapper subclass whose repr formats its inner tensor, as DTensor's does."""

    @staticmethod
    def __new__(cls, inner: torch.Tensor) -> "_Wrapper":
        return torch.Tensor._make_wrapper_subclass(cls, inner.shape, dtype=inner.dtype)

    def __init__(self, inner: torch.Tensor) -> None:
        self.wrapped = inner

    def __repr__(self) -> str:  # type: ignore[override]
        return f"_Wrapper(inner={self.wrapped})"

    @classmethod
    def __torch_dispatch__(  # type: ignore[override]
        cls, func: Any, types: Any, args: Any = (), kwargs: Any = None
    ) -> Any:
        raise NotImplementedError(func)


class _ContentsWrapper(torch.Tensor):
    """A wrapper subclass whose repr passes its inner values to torch's repr, as torch's own logging tensor does."""

    @staticmethod
    def __new__(cls, inner: torch.Tensor) -> "_ContentsWrapper":
        return torch.Tensor._make_wrapper_subclass(cls, inner.shape, dtype=inner.dtype)

    def __init__(self, inner: torch.Tensor) -> None:
        self.wrapped = inner

    def __repr__(self) -> str:  # type: ignore[override]
        return super().__repr__(tensor_contents=f"wrapped={self.wrapped.tolist()}")

    @classmethod
    def __torch_dispatch__(  # type: ignore[override]
        cls, func: Any, types: Any, args: Any = (), kwargs: Any = None
    ) -> Any:
        raise NotImplementedError(func)


class _Flattened(torch.Tensor):
    """A wrapper subclass that keeps its inner tensor in a slot and names it and a non-tensor in ``__tensor_flatten__``.

    DTensor names its local tensor and its device mesh that way.
    """

    __slots__ = ("inner",)
    mesh: object

    @staticmethod
    def __new__(cls, inner: torch.Tensor) -> "_Flattened":
        wrapper = torch.Tensor._make_wrapper_subclass(
            cls, inner.shape, dtype=inner.dtype
        )
        wrapper.inner = inner
        wrapper.mesh = object()
        return wrapper

    def __tensor_flatten__(self) -> tuple[list[str], None]:
        return ["inner", "mesh"], None

    def __repr__(self) -> str:  # type: ignore[override]
        return f"_Flattened(inner={self.inner})"

    @classmethod
    def __torch_dispatch__(  # type: ignore[override]
        cls, func: Any, types: Any, args: Any = (), kwargs: Any = None
    ) -> Any:
        raise NotImplementedError(func)


class _SlotWrapper(torch.Tensor):
    """A wrapper subclass that keeps its inner tensor in a slot without naming it anywhere. Torch's ``LoggingTensor`` does so."""

    __slots__ = ("elem",)
    elem: torch.Tensor

    @staticmethod
    def __new__(cls, elem: torch.Tensor) -> "_SlotWrapper":
        wrapper = torch.Tensor._make_wrapper_subclass(cls, elem.shape, dtype=elem.dtype)
        wrapper.elem = elem
        return wrapper

    def __repr__(self) -> str:  # type: ignore[override]
        return super().__repr__(tensor_contents=f"{self.elem}")

    @classmethod
    def __torch_dispatch__(  # type: ignore[override]
        cls, func: Any, types: Any, args: Any = (), kwargs: Any = None
    ) -> Any:
        raise NotImplementedError(func)


class _OwnDispatch(torch.Tensor):
    """A subclass with its own ``__torch_dispatch__`` that ``_make_subclass`` makes over a tensor's own storage."""

    @classmethod
    def __torch_dispatch__(  # type: ignore[override]
        cls, func: Any, types: Any, args: Any = (), kwargs: Any = None
    ) -> Any:
        raise NotImplementedError(func)


class _Slotted(torch.Tensor):
    """A plain subclass that keeps an extra tensor in a slot."""

    __slots__ = ("extra",)
    extra: torch.Tensor


class _Blob:
    """An object that pickles its bytes through a pickle buffer, as NumPy arrays do."""

    def __init__(self, data: bytes | bytearray) -> None:
        self.data = bytes(data)

    def __reduce__(self) -> tuple[Any, ...]:
        return _Blob, (pickle.PickleBuffer(bytearray(self.data)),)


class _Gate:
    """Pickles only once its test lets it, holding the codec mid-call meanwhile."""

    inside = threading.Event()
    proceed = threading.Event()

    def __reduce__(self) -> tuple[Any, ...]:
        _Gate.inside.set()
        assert _Gate.proceed.wait(60)
        return _Gate, ()


class _Printing:
    """Prints a tensor with torch's own repr while it pickles, as an object that logs might."""

    def __reduce__(self) -> tuple[Any, ...]:
        repr(torch.arange(2.0))
        return _Printing, ()


class _PrintingPastTheRepr:
    """Prints a tensor with torch's own printer, past ``Tensor.__repr__`` and so past the codec's printer, while it pickles."""

    def __reduce__(self) -> tuple[Any, ...]:
        torch._tensor_str._str(torch.arange(2.0))
        return _PrintingPastTheRepr, ()


class _TorchOps(torch.overrides.TorchFunctionMode):
    """Records each torch function with the file and function of the frame that called it.

    The walk passes torch's override frames, ``__torch_function__`` frames and the called function's own frame so a
    torch function defined in Python (such as ``Tensor.storage``) is credited to its caller.
    """

    def __init__(self) -> None:
        super().__init__()
        self.ops: list[tuple[str, str, Any]] = []

    def __torch_function__(
        self, func: Any, types: Any, args: Any = (), kwargs: Any = None
    ) -> Any:
        own = getattr(inspect.unwrap(func), "__code__", None)
        frame: types.FrameType | None = sys._getframe(1)
        while frame is not None and (
            frame.f_code.co_filename == torch.overrides.__file__
            or frame.f_code.co_name == "__torch_function__"
            or frame.f_code is own
        ):
            frame = frame.f_back
        if frame is not None:
            self.ops.append((frame.f_code.co_filename, frame.f_code.co_name, func))
        return func(*args, **(kwargs or {}))

    def codec_ops(self) -> list[str]:
        """The ops the codec's own frames called, but for the reduction hook that pickle calls on a tensor from C."""
        return [
            f"{name}: {func}"
            for file, name, func in self.ops
            if file == codec.__file__ and func is not torch.Tensor.__reduce_ex__
        ]


@contextlib.contextmanager
def _shrunk(storage: torch.UntypedStorage, nbytes: int) -> Iterator[None]:
    """Shrink ``storage`` for a block and give its bytes back as the block leaves, before a report with -l prints it.

    A restore that fails after the block failed is logged so the block's own failure stays the one reported.
    """
    full = storage.nbytes()
    storage.resize_(nbytes)
    try:
        yield
    except BaseException:
        try:
            storage.resize_(full)
        except Exception:
            logging.getLogger(__name__).exception("restoring a shrunk storage failed")
        raise
    storage.resize_(full)


def _run_child(script: str) -> None:
    """Run ``script`` in a child process that imports this tree, and fail unless it exits with status 0."""
    child = subprocess.run(
        [sys.executable, "-c", script, _ROOT],
        capture_output=True,
        text=True,
        timeout=100,
        check=False,
    )
    assert child.returncode == 0, child.stdout + child.stderr


class _Recording(TorchDispatchMode):
    """Runs each op unchanged and records it, so a test can stack two dispatch modes."""

    def __init__(self) -> None:
        super().__init__()
        self.ops: list[Any] = []

    def __torch_dispatch__(
        self, func: Any, types: Any, args: Any = (), kwargs: Any = None
    ) -> Any:
        self.ops.append(func)
        return func(*args, **(kwargs or {}))


@dataclass
class _Record:
    name: str
    values: list[int]


def _frame(kind: Kind, request_id: int = 0, **kwargs: Any) -> bytes:
    chunks = codec.encode_frame(kind, request_id, **kwargs)
    return b"".join(bytes(chunk) for chunk in chunks)


def _receive(
    data: bytes, *, handshake: bool = False
) -> tuple[codec.FrameHeader, bytes, list[torch.Tensor]]:
    """Parse one frame in a receiver's order, landing each buffer in a new uint8 tensor."""
    header = codec.decode_header(data[: codec.HEADER_SIZE], handshake=handshake)
    start = codec.HEADER_SIZE + codec.LENGTH_SIZE * header.buffer_count
    lengths = codec.decode_lengths(header, data[codec.HEADER_SIZE : start])
    body = data[start : start + header.body_len]
    start += header.body_len
    buffers = []
    for length in lengths:
        buffers.append(torch.empty(length, dtype=torch.uint8))
        codec.byte_view(buffers[-1])[:] = data[start : start + length]
        start += length
    assert start == len(data)
    return header, body, buffers


def _round_trip(obj: Any) -> Any:
    body, buffers = codec.dumps(obj)
    _, body, received = _receive(_frame(Kind.REPLY, body=body, buffers=buffers))
    return codec.loads(body, received)


def _raised(error: BaseException) -> BaseException:
    """Raise and catch ``error``, so it carries a traceback as a handler's error does."""
    try:
        raise error
    except BaseException as e:
        return e


def _server_only(monkeypatch: pytest.MonkeyPatch, cls: type, module_name: str) -> None:
    """Make ``cls`` importable from ``module_name`` until the test deletes that module."""
    module = types.ModuleType(module_name)
    cls.__module__, cls.__qualname__ = module_name, cls.__name__
    vars(module)[cls.__name__] = cls
    monkeypatch.setitem(sys.modules, module_name, module)


def _raw_header(
    *,
    magic: bytes = codec.MAGIC,
    kind: int = Kind.REQUEST,
    flags: int = 0,
    reserved: int = 0,
    count: int = 0,
    request_id: int = 0,
    deadline_us: int = 0,
    body_len: int = 0,
    spare: int = 0,
) -> bytes:
    fields = (flags, reserved, count, request_id, deadline_us, body_len, spare)
    return struct.pack("!4sBBHIQQQI", magic, kind, *fields)


# Creating some of these tensors warns that their support is experimental or deprecated.
@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("dtype", _DTYPES, ids=str)
def test_every_dtype_round_trips(dtype: torch.dtype) -> None:
    """Round-trip every non-quantized dtype's bytes out of band, bfloat16, bool, complex and float8 included."""
    tensor = torch.empty((2, 3, 4), dtype=dtype)
    raw = tensor.view(torch.uint8)
    raw.copy_((torch.arange(raw.numel()) * 37 % 256).reshape(raw.shape))
    assert [buffer.nbytes for buffer in codec.dumps(tensor)[1]] == [tensor.nbytes]
    decoded = _round_trip(tensor)
    assert (decoded.dtype, decoded.shape) == (dtype, tensor.shape)
    assert torch.equal(decoded.view(torch.uint8), raw)


@pytest.mark.parametrize(
    "tensor",
    [
        torch.tensor(2.5),
        torch.empty(0, 3),
        torch.ones(2, 0, dtype=torch.int64),
        torch.arange(10.0)[::2][:0],
        torch.zeros(4, 4)[:0, ::2],
    ],
    ids=["0-d", "empty", "empty-int64", "empty-step", "empty-strided-2d"],
)
def test_zero_dim_and_empty_tensors_round_trip(tensor: torch.Tensor) -> None:
    """Round-trip a 0-d tensor in a 4-byte buffer, and empty tensors in empty buffers."""
    _, buffers = codec.dumps(tensor)
    assert [buffer.nbytes for buffer in buffers] == [tensor.nbytes]
    decoded = _round_trip(tensor)
    assert (decoded.dtype, decoded.shape) == (tensor.dtype, tensor.shape)
    assert torch.equal(decoded, tensor)


def test_requires_grad_survives() -> None:
    """Rebuild tensors that require grad, a leaf or an op's result, as leaves that require grad."""
    leaf = torch.ones(3, requires_grad=True)
    obj = {"leaf": leaf, "result": leaf * 2, "plain": torch.ones(3)}
    assert [buffer.nbytes for buffer in codec.dumps(obj)[1]] == [12, 12, 12]
    decoded = _round_trip(obj)
    assert decoded["leaf"].requires_grad and decoded["leaf"].is_leaf
    assert decoded["result"].requires_grad and decoded["result"].is_leaf
    assert torch.equal(decoded["result"].detach(), torch.full((3,), 2.0))
    assert not decoded["plain"].requires_grad


def test_a_buffer_keeps_no_autograd_graph_alive() -> None:
    """Buffer an op's result detached, so its graph and the tensors that graph saved go once the caller drops it."""
    intermediate = torch.ones(3, requires_grad=True) * 3
    result = intermediate.sin()
    saved = weakref.ref(intermediate)
    del intermediate
    buffers = codec.dumps(result)[1]
    del result
    assert saved() is None and [buffer.nbytes for buffer in buffers] == [12]


def test_repeated_references_travel_once() -> None:
    """Send a tensor referenced three times as one buffer, and rebuild one shared tensor."""
    tensor = torch.arange(4.0)
    obj = {"a": tensor, "b": [tensor, tensor]}
    assert len(codec.dumps(obj)[1]) == 1
    decoded = _round_trip(obj)
    assert decoded["a"] is decoded["b"][0] is decoded["b"][1]
    assert torch.equal(decoded["a"], tensor)


@pytest.mark.parametrize(
    "view, nbytes, strides",
    [
        (lambda base: base[10:20], 40, (1,)),
        (lambda base: base.view(10, 10)[:, 3], 40, (1,)),
        (lambda base: base[:3].expand(4, 3), 12, (0, 1)),
        (lambda base: base[99:].expand(8), 4, (0,)),
        (lambda base: base[:1].expand(1 << 28), 4, (0,)),
        (lambda base: base[:5].unfold(0, 3, 1), 20, (1, 1)),
        (lambda base: base.view(10, 10).t(), 400, (10, 1)),
    ],
    ids=[
        "slice",
        "column",
        "expanded",
        "expanded-last",
        "expanded-huge",
        "windows",
        "transposed",
    ],
)
def test_a_view_encodes_only_its_own_bytes(
    view: Callable[[torch.Tensor], torch.Tensor],
    nbytes: int,
    strides: tuple[int, ...],
) -> None:
    """Send at most a view's elements, a permuted one as a contiguous copy, and an expanded one only the storage it spans."""
    tensor = view(torch.arange(100, dtype=torch.float32))
    _, buffers = codec.dumps(tensor)
    assert [buffer.nbytes for buffer in buffers] == [nbytes]
    decoded = _round_trip(tensor)
    assert decoded.stride() == strides and torch.equal(decoded, tensor)


def test_conjugate_and_negative_views_send_their_values() -> None:
    """Send the values of contiguous lazy conjugate and negative views, not their storage's."""
    conj = torch.tensor([1 + 2j, 3 - 4j]).conj()
    neg = torch.tensor([3 - 4j]).conj().imag
    assert conj.is_conj() and conj.is_contiguous()
    assert neg.is_neg() and neg.is_contiguous()
    decoded_conj, decoded_neg = _round_trip([conj, neg])
    assert torch.equal(decoded_conj, torch.tensor([1 - 2j, 3 + 4j]))
    assert torch.equal(decoded_neg, torch.tensor([4.0]))
    assert not (decoded_conj.is_conj() or decoded_neg.is_neg())


def test_cpu_tensors_travel_uncopied() -> None:
    """Alias a CPU tensor's memory in its buffer, keeping its bytes out of the body."""
    tensor = torch.zeros(1 << 22)
    body, buffers = codec.dumps(tensor)
    tensor.fill_(7.0)
    assert len(body) < 1024 and [buffer.nbytes for buffer in buffers] == [1 << 24]
    assert bytes(buffers[0]) == bytes(codec.byte_view(torch.full((1 << 22,), 7.0)))


@pytest.mark.parametrize(
    "view",
    [lambda base: base.expand(3, 4), lambda base: base.unfold(0, 2, 1)],
    ids=["expanded", "windows"],
)
def test_an_expanded_or_overlapping_tensor_travels_as_a_copy(
    view: Callable[[torch.Tensor], torch.Tensor],
) -> None:
    """Send the span of an expanded or overlapping view as a copy, as other non-contiguous tensors are sent."""
    base = torch.arange(4.0)
    tensor = view(base)
    expected = tensor.clone()
    body, buffers = codec.dumps(tensor)
    base.add_(100)
    _, body, received = _receive(_frame(Kind.REPLY, body=body, buffers=buffers))
    assert torch.equal(codec.loads(body, received), expected)


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("dtype", _ONE_BYTE_DTYPES, ids=str)
@pytest.mark.parametrize(
    "view",
    [
        lambda base: base[:1].expand(5),
        lambda base: base[:6].unfold(0, 3, 1),
        lambda base: base.view(4, 6).t(),
    ],
    ids=["expanded", "windows", "transposed"],
)
def test_one_byte_dtypes_travel_byte_for_byte_in_any_layout(
    view: Callable[[torch.Tensor], torch.Tensor], dtype: torch.dtype
) -> None:
    """Send an expanded, overlapping or transposed tensor of each 1-byte dtype byte for byte, sub-byte ones included."""
    raw = (torch.arange(24) * 37 % 256).to(torch.uint8)
    tensor = view(raw.view(dtype))
    decoded = _round_trip(tensor)
    assert (decoded.dtype, decoded.shape) == (dtype, tensor.shape)
    assert torch.equal(decoded.view(torch.uint8), view(raw))


def test_decoded_tensors_view_their_received_buffers() -> None:
    """Rebuild each tensor, an expanded one too, as a view of its received buffer, so receiving copies nothing."""
    body, buffers = codec.dumps([torch.arange(6.0), torch.arange(3.0).expand(2, 3)])
    _, body, received = _receive(_frame(Kind.REPLY, body=body, buffers=buffers))
    decoded = codec.loads(body, received)
    assert [item.data_ptr() for item in decoded] == [buf.data_ptr() for buf in received]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_tensors_decode_on_cpu() -> None:
    """Decode CUDA tensors on CPU, a plain subclass's through torch's own reduction, and in errors, among their args too."""
    tensor = torch.arange(6.0, device="cuda").reshape(2, 3)
    labeled = tensor.clone()
    vars(labeled)["label"] = "weights"
    decoded = _round_trip([tensor, labeled, tensor.as_subclass(_Subclass)])
    assert [item.device.type for item in decoded] == ["cpu"] * 3
    assert all(
        torch.equal(item.as_subclass(torch.Tensor), tensor.cpu()) for item in decoded
    )
    assert decoded[1].label == "weights" and type(decoded[2]) is _Subclass
    error = ValueError("shard")
    vars(error)["tensor"] = tensor
    decoded_error = codec.decode_error(codec.encode_error(_raised(error)))
    assert type(decoded_error) is ValueError
    assert vars(decoded_error)["tensor"].device.type == "cpu"
    # Printed here, its tensor would lose the device its text names.
    in_args = codec.decode_error(codec.encode_error(_raised(ValueError("s", tensor))))
    assert type(in_args) is ValueError and in_args.args[1].device.type == "cpu"
    assert torch.equal(in_args.args[1], tensor.cpu())


def test_parameters_send_their_data_out_of_band() -> None:
    """Rebuild a Parameter as one, its data reaching the codec through its own reduction."""
    param = torch.nn.Parameter(torch.arange(3.0))
    assert len(codec.dumps(param)[1]) == 1
    decoded = _round_trip(param)
    assert type(decoded) is torch.nn.Parameter and decoded.requires_grad
    assert torch.equal(decoded.detach(), param.detach())


def test_tensors_with_attributes_send_only_their_own_bytes() -> None:
    """Rebuild a tensor's attributes, tensors among them, sending only the tensor's own bytes."""
    labeled = torch.arange(1 << 18, dtype=torch.float32)[:4]
    vars(labeled).update(label="weights", scale=torch.ones(2))
    body, buffers = codec.dumps(labeled)
    assert len(body) < 1024 and [buffer.nbytes for buffer in buffers] == [16, 8]
    decoded = _round_trip(labeled)
    assert torch.equal(decoded, labeled) and decoded.label == "weights"
    assert vars(decoded).keys() == {"label", "scale"}
    assert torch.equal(decoded.scale, torch.ones(2))


def test_zero_tensors_send_their_zeros() -> None:
    """Send a zero tensor, which has no memory of its own, as real zeros, an expanded one as just the zeros it spans."""
    zeros = torch._efficientzerotensor(2, 3)
    with pytest.raises(ValueError, match="storage holds all of its bytes"):
        codec.byte_view(zeros)
    decoded = _round_trip(zeros)
    assert torch.equal(decoded, torch.zeros(2, 3)) and not decoded._is_zerotensor()
    expanded = torch._efficientzerotensor(1).expand(4, 3)
    assert [buffer.nbytes for buffer in codec.dumps(expanded)[1]] == [4]
    decoded = _round_trip(expanded)
    assert decoded.stride() == (0, 0) and torch.equal(decoded, torch.zeros(4, 3))


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    "make, nbytes",
    [
        (lambda: torch.arange(16, dtype=torch.float32).view(4, 4).t(), 0),
        (lambda: torch.arange(8.0)[4:], 16),
        (lambda: torch.arange(16.0).as_subclass(_Subclass), 8),
        (lambda: torch.quantize_per_tensor(torch.ones(16), 0.5, 2, torch.qint8), 8),
    ],
    ids=["freed-transposed", "shrunk-past-offset", "shrunk-subclass", "shrunk-qint8"],
)
def test_tensors_whose_storage_lost_their_bytes_are_refused(
    make: Callable[[], torch.Tensor], nbytes: int
) -> None:
    """Refuse a tensor whose storage was freed or shrunk, subclassed or quantized too, before copying or reducing it."""
    tensor = make()
    tensor.untyped_storage().resize_(nbytes)
    with pytest.raises(ValueError, match="storage holds all of its bytes"):
        codec.dumps(tensor)


def _q4(values: torch.Tensor) -> torch.Tensor:
    return torch.quantize_per_tensor(values, 1.0, 0, torch.quint4x2)


def _part(make: Callable[[], torch.Tensor], name: str) -> Callable[[], Any]:
    """A maker of a tensor and the storage of the part it gets from method or attribute ``name``, itself for ``""``."""

    def made() -> tuple[torch.Tensor, torch.UntypedStorage]:
        tensor = make()
        part = getattr(tensor, name) if name else tensor
        held = cast(torch.Tensor, part() if callable(part) else part)
        return tensor, held.untyped_storage()

    return made


_COO = partial(
    torch.sparse_coo_tensor, torch.tensor([[0, 1, 2, 3]]), torch.ones(4), (4,)
)
_COMPRESSED = {
    "csr": (lambda: torch.eye(4).to_sparse_csr(), "crow_indices", "col_indices"),
    "csc": (lambda: torch.eye(4).to_sparse_csc(), "ccol_indices", "row_indices"),
    "bsr": (lambda: torch.eye(4).to_sparse_bsr((2, 2)), "crow_indices", "col_indices"),
    "bsc": (lambda: torch.eye(4).to_sparse_bsc((2, 2)), "ccol_indices", "row_indices"),
}
# Each holds its values in the storage of the part named: its own, a sparse part's or an inner tensor's.
_LOST_BYTES = {
    "quint4x2": _part(lambda: _q4(torch.arange(16.0)), ""),
    "coo-indices": _part(_COO, "_indices"),
    "coo-values": _part(_COO, "_values"),
    **{
        f"{layout}-{name}": _part(make, name)
        for layout, (make, *indices) in _COMPRESSED.items()
        for name in (*indices, "values")
    },
    "wrapper": _part(lambda: _Wrapper(torch.arange(16.0)), "wrapped"),
    "flattened": _part(lambda: _Flattened(torch.arange(16.0)), "inner"),
    "slot-wrapper": _part(lambda: _SlotWrapper(torch.arange(16.0)), "elem"),
    "own-dispatch": _part(
        lambda: torch.Tensor._make_subclass(_OwnDispatch, torch.arange(16.0)), ""
    ),
    "nested": _part(
        lambda: torch.nested.nested_tensor([torch.arange(8.0), torch.arange(8.0)]), ""
    ),
}


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("make", _LOST_BYTES.values(), ids=_LOST_BYTES.keys())
def test_packed_sparse_nested_and_subclassed_tensors_whose_storage_lost_bytes_are_refused(
    make: Callable[[], tuple[torch.Tensor, torch.UntypedStorage]],
) -> None:
    """Refuse a tensor whose values' storage shrank and send an error holding it as placeholder text that prints none of it.

    That storage is a packed tensor's, an own-dispatch subclass's or a nested tensor's own, any part of a sparse one's,
    or the inner tensor's a wrapper flattens to or keeps in an attribute or a slot.
    """
    tensor, storage = make()
    with _shrunk(storage, storage.nbytes() // 4):
        with pytest.raises(ValueError, match="storage holds all of its bytes"):
            codec.dumps(tensor)
        body = codec.encode_error(ValueError("bad", tensor))
        assert pickle.loads(body)[3] is None
        text = "ValueError: <ValueError whose str() failed>"
        assert str(codec.decode_error(body)) == text


def _parameter_with_a_freed_attribute() -> tuple[torch.Tensor, torch.UntypedStorage]:
    """A Parameter that keeps a tensor attribute like the padded one FSDP1's flat parameter keeps after resharding."""
    parameter = torch.nn.Parameter(torch.arange(8.0))
    padded = vars(parameter)["_full_param_padded"] = torch.arange(16.0)
    return parameter, padded.untyped_storage()


def _slotted() -> tuple[torch.Tensor, torch.UntypedStorage]:
    tensor = torch.arange(8.0).as_subclass(_Slotted)
    tensor.extra = torch.arange(16.0)
    return tensor, tensor.extra.untyped_storage()


@pytest.mark.parametrize(
    "make", [_parameter_with_a_freed_attribute, _slotted], ids=["attribute", "slot"]
)
def test_a_tensor_whose_attribute_lost_its_bytes_is_refused_but_printed(
    make: Callable[[], tuple[torch.Tensor, torch.UntypedStorage]],
) -> None:
    """Refuse to pickle a tensor whose attribute or slot tensor lost its bytes, but print it.

    Torch's repr reads only the tensor's own values.
    """
    tensor, storage = make()
    with _shrunk(storage, 0):
        with pytest.raises(ValueError, match="storage holds all of its bytes"):
            codec.dumps(tensor)
        error = ValueError("hook failed on", tensor)
        body = codec.encode_error(error)
        assert pickle.loads(body)[3] is None
        assert str(codec.decode_error(body)) == f"ValueError: {error}"


def test_a_freed_own_dispatch_subclass_is_refused() -> None:
    """Refuse an own-dispatch subclass whose own storage was freed, which a wrapper's sized storage without memory is not."""
    tensor = torch.Tensor._make_subclass(_OwnDispatch, torch.arange(16.0))
    with _shrunk(tensor.untyped_storage(), 0):
        with pytest.raises(ValueError, match="storage holds all of its bytes"):
            codec.dumps(tensor)


def test_the_printer_checks_a_wrappers_inner_tensors_but_not_a_plain_tensors_attributes() -> None:
    """Skip a plain tensor's attributes but check a wrapper's inner tensors before printing it. Its own repr may read them.

    This wrapper's repr reads its inner values with ``tolist`` before torch's repr. Its text is thus the placeholder
    only because the wrapper's own print is checked.
    """
    wrapper = _ContentsWrapper(torch.arange(16.0))
    plain, freed = _parameter_with_a_freed_attribute()
    with _shrunk(wrapper.wrapped.untyped_storage(), 16), _shrunk(freed, 0):
        with pytest.raises(ValueError, match="storage holds all of its bytes"):
            codec._check_values(wrapper, attributes=False)
        codec._check_values(plain, attributes=False)
        text = pickle.loads(codec.encode_error(ValueError("bad", wrapper)))[1]
        assert text == "<ValueError whose str() failed>"


def test_a_tensor_holding_two_sparse_tensors_is_checked_through_both(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check both sparse tensors a tensor holds and keep each part the walk made alive until the walk ends.

    A part let go could leave its id to a part made later, which the walk would then skip as already seen.
    """
    tensor = torch.arange(4.0)
    vars(tensor).update(
        first=torch.eye(4).to_sparse_csr(), second=torch.eye(4).to_sparse_csr()
    )
    checked: list[weakref.ref[torch.Tensor]] = []
    check = codec._check_storage

    def recording(part: torch.Tensor) -> None:
        assert all(
            earlier() is not None for earlier in checked
        ), "the walk let a part go"
        checked.append(weakref.ref(part))
        check(part)

    monkeypatch.setattr(codec, "_check_storage", recording)
    codec._check_values(tensor)
    # The tensor itself, then each sparse tensor's two indices and its values.
    assert len(checked) == 7
    with _shrunk(vars(tensor)["second"].values().untyped_storage(), 4):
        assert pickle.loads(codec.encode_error(ValueError("bad", tensor)))[3] is None


def test_an_error_holding_a_tensor_that_holds_itself_keeps_its_type() -> None:
    """Rebuild an error holding a tensor that keeps itself as an attribute, which the storage walk visits once."""
    tensor = torch.arange(4.0)
    vars(tensor)["me"] = tensor
    decoded = codec.decode_error(codec.encode_error(_raised(ValueError("bad", tensor))))
    assert type(decoded) is ValueError and torch.equal(decoded.args[1], tensor)


@pytest.fixture
def dtensor() -> Iterator[DTensor]:
    """A DTensor sharded over a one-rank mesh in this process on torch's fake process group, which opens no socket."""
    started = not dist.is_initialized()
    if started:
        dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=1)
    try:
        mesh = DeviceMesh("cpu", [0])
        yield DTensor.from_local(torch.arange(8.0), mesh, [Shard(0)])
    finally:
        if started:
            dist.destroy_process_group()


def test_dtensors_round_trip_and_errors_holding_them_keep_their_type(
    dtensor: DTensor,
) -> None:
    """Send a DTensor and a Parameter of one, whose flattened names include its device mesh, and an error holding one."""
    decoded = _round_trip({"w": dtensor, "p": torch.nn.Parameter(dtensor)})
    assert type(decoded["w"]) is DTensor
    assert torch.equal(decoded["w"].to_local(), dtensor.to_local())
    assert torch.equal(decoded["p"].to_local(), dtensor.to_local())
    assert isinstance(decoded["p"], torch.nn.Parameter)
    error = _raised(ValueError("bad shard", dtensor))
    rebuilt = codec.decode_error(codec.encode_error(error))
    assert type(rebuilt) is ValueError and str(rebuilt) == str(error)


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    "make",
    [
        lambda: _q4(torch.arange(4.0)).view(4, 1),
        lambda: _q4(torch.arange(4.0)).unsqueeze(-1),
        lambda: _q4(torch.arange(4.0).reshape(1, 4)).t(),
        lambda: _q4(torch.arange(9.0).reshape(3, 3)),
        lambda: _q4(torch.tensor(3.0)),
    ],
    ids=["view", "unsqueezed", "transposed", "odd-rows", "0-d"],
)
def test_intact_packed_quantized_tensors_of_any_shape_round_trip(
    make: Callable[[], torch.Tensor],
) -> None:
    """Send a packed quantized tensor of any shape whose storage holds the bytes torch's sub-byte kernels read.

    Those kernels read its values as one run of whole bytes, which a reshaped view shares with its base.
    """
    tensor = make()
    decoded = _round_trip(tensor)
    assert decoded.shape == tensor.shape
    assert torch.equal(decoded.dequantize(), tensor.dequantize())


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    "make, nbytes",
    [
        (lambda: _q4(torch.arange(16.0).reshape(4, 4))[2:], None),
        (lambda: _q4(torch.arange(16.0))[4:], None),
        (lambda: _q4(torch.arange(9.0)), 4),
    ],
    ids=["row-slice", "element-slice", "odd-count-short-a-byte"],
)
def test_packed_quantized_tensors_torch_would_read_past_their_storage_are_refused(
    make: Callable[[], torch.Tensor], nbytes: int | None
) -> None:
    """Refuse a packed quantized tensor whose storage, shrunk or not, ends before the bytes torch's sub-byte kernels read.

    They count a view's storage offset in bytes so a slice of whole values reads past its storage.
    """
    tensor = make()
    storage = tensor.untyped_storage()
    with _shrunk(storage, storage.nbytes() if nbytes is None else nbytes):
        with pytest.raises(ValueError, match="storage holds all of its bytes"):
            codec.dumps(tensor)
        body = codec.encode_error(ValueError("bad", tensor))
        assert pickle.loads(body)[3] is None
        assert str(codec.decode_error(body)).endswith("whose str() failed>")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_shrunk_cuda_tensors_are_refused_before_their_copy() -> None:
    """Refuse a shrunk CUDA tensor before its copy to CPU reads past its storage."""
    tensor = torch.arange(16.0, device="cuda")
    tensor.untyped_storage().resize_(8)
    with pytest.raises(ValueError, match="storage holds all of its bytes"):
        codec.dumps(tensor)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_torch_reduces_other_tensors() -> None:
    """Leave sparse, quantized, meta and nested tensors and tensor subclasses to torch."""
    subclass = torch.arange(3.0).as_subclass(_Subclass)
    sparse = torch.sparse_coo_tensor(
        torch.tensor([[0, 2]]), torch.tensor([1.0, 2.0]), (3,)
    )
    nested = torch.nested.nested_tensor([torch.ones(2), torch.zeros(3)])
    quantized = torch.quantize_per_tensor(torch.ones(3), 0.5, 2, torch.qint8)
    meta = torch.empty(2, 3, device="meta")
    decoded = _round_trip([subclass, sparse, nested, quantized, meta])
    subclass2, sparse2, nested2, quantized2, meta2 = decoded
    assert type(subclass2) is _Subclass and torch.equal(subclass2, subclass)
    assert sparse2.layout == torch.sparse_coo
    assert torch.equal(sparse2.to_dense(), sparse.to_dense())
    assert nested2.is_nested
    assert all(map(torch.equal, nested2.unbind(), nested.unbind()))
    assert (quantized2.q_scale(), quantized2.q_zero_point()) == (0.5, 2)
    assert torch.equal(quantized2.int_repr(), quantized.int_repr())
    assert meta2.is_meta and meta2.shape == (2, 3)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_decoded_sparse_tensors_are_left_for_no_later_load() -> None:
    """Free decoded sparse tensors, malformed ones too, which torch would hold for a later ``torch.load`` to check."""
    malformed = torch.sparse_coo_tensor(
        torch.tensor([[5]]), torch.tensor([1.0]), (3,), check_invariants=False
    )
    error = ValueError("shard")
    vars(error)["tensor"] = malformed
    decoded = [
        _round_trip(malformed),
        _round_trip(torch.eye(2).to_sparse_csr()),
        vars(codec.decode_error(codec.encode_error(_raised(error))))["tensor"],
    ]
    alive = [weakref.ref(tensor) for tensor in decoded]
    del decoded
    gc.collect()
    assert [ref() for ref in alive] == [None] * 3
    file = io.BytesIO()
    torch.save(torch.ones(2), file)
    file.seek(0)
    assert torch.equal(torch.load(file, weights_only=True), torch.ones(2))


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_encoding_an_error_leaves_no_sparse_tensor_for_a_later_load() -> None:
    """Leave nothing for a later ``torch.load`` to check once an error holding a malformed sparse tensor is encoded."""
    malformed = torch.sparse_coo_tensor(
        torch.tensor([[5]]), torch.tensor([1.0]), (3,), check_invariants=False
    )
    error = ValueError("shard")
    vars(error)["tensor"] = malformed
    codec.encode_error(_raised(error))
    file = io.BytesIO()
    torch.save(torch.ones(2), file)
    file.seek(0)
    assert torch.equal(torch.load(file, weights_only=True), torch.ones(2))


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_sparse_tensors_keep_their_layout_and_coalescing() -> None:
    """Rebuild each sparse layout with its values and a coalesced tensor as one, checking invariants when torch's setting asks."""
    dense = torch.tensor([[1.0, 2.0, 0.0, 0.0], [0.0, 0.0, 3.0, 0.0]])
    coo = _round_trip(dense.to_sparse().coalesce())
    assert coo.is_coalesced() and torch.equal(
        coo.indices(), dense.to_sparse().indices()
    )
    duplicates = torch.sparse_coo_tensor(
        torch.tensor([[0, 0, 2]]), torch.tensor([1.0, 2.0, 5.0]), (3,)
    )
    uncoalesced = _round_trip(duplicates)
    assert not uncoalesced.is_coalesced()
    assert torch.equal(uncoalesced.coalesce().values(), torch.tensor([3.0, 5.0]))
    layouts = [
        dense.to_sparse_csc(),
        dense.to_sparse_bsr((1, 2)),
        dense.to_sparse_bsc((1, 2)),
    ]
    for tensor in layouts:
        decoded = _round_trip(tensor)
        assert decoded.layout == tensor.layout and torch.equal(
            decoded.to_dense(), dense
        )
    malformed = [
        torch.sparse_coo_tensor(
            torch.tensor([[5]]), torch.tensor([1.0]), (3,), check_invariants=False
        ),
        torch.sparse_csr_tensor(
            torch.tensor([0, 1, 1]),
            torch.tensor([7]),
            torch.tensor([1.0]),
            (2, 3),
            check_invariants=False,
        ),
    ]
    for tensor, message in zip(malformed, ["found index 5", "col_indices < ncols"]):
        assert _round_trip(tensor).layout == tensor.layout
        body, buffers = codec.dumps(tensor)
        _, body, received = _receive(_frame(Kind.REPLY, body=body, buffers=buffers))
        with torch.sparse.check_sparse_tensor_invariants():
            with pytest.raises(RuntimeError, match=message):
                codec.loads(body, received)


def test_decoding_sparse_tensors_never_warns_that_checks_are_off() -> None:
    """Decode COO and CSR tensors in a fresh process without torch's warning that invariant checks are implicitly off."""
    _run_child(_SPARSE_WARNING_CHILD)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_sparse_tensors_decode_on_cpu_under_a_default_device() -> None:
    """Rebuild sparse tensors, in an error too, on CPU with their values, whatever default device the caller set."""
    dense = torch.tensor([[1.0, 2.0, 0.0, 0.0], [0.0, 0.0, 3.0, 0.0]])
    error = ValueError("shard")
    vars(error)["tensor"] = dense.to_sparse()
    error_body = codec.encode_error(_raised(error))
    for tensor in (
        dense.to_sparse(),
        dense.to_sparse_csr(),
        dense.to_sparse_bsc((1, 2)),
    ):
        body, buffers = codec.dumps(tensor)
        _, body, received = _receive(_frame(Kind.REPLY, body=body, buffers=buffers))
        with torch.device("meta"):
            decoded = codec.loads(body, received)
        assert decoded.device.type == "cpu" and torch.equal(decoded.to_dense(), dense)
    with torch.device("meta"):
        decoded_error = codec.decode_error(error_body)
    assert type(decoded_error) is ValueError
    assert torch.equal(vars(decoded_error)["tensor"].to_dense(), dense)


def test_other_pickle_buffers_stay_in_band() -> None:
    """Keep pickle buffers that are not the codec's in the body, so every buffer is a tensor."""
    blob = _Blob(b"\x00\x01\x02")
    assert len(codec.dumps([blob, torch.ones(2)])[1]) == 1
    decoded_blob, decoded_tensor = _round_trip([blob, torch.ones(2)])
    assert decoded_blob.data == blob.data and torch.equal(decoded_tensor, torch.ones(2))


def test_plain_objects_round_trip() -> None:
    """Round-trip what calls carry besides tensors: containers, bytes, None and dataclasses."""
    obj = (
        "put",
        ({"key": "w", "data": b"\x00\xff", "record": _Record("r", [1, 2])}, None),
        {"timeout": 1.5},
    )
    body, buffers = codec.dumps(obj)
    assert buffers == []
    assert codec.loads(body) == obj == _round_trip(obj)


def test_codec_runs_outside_the_callers_dispatch_modes() -> None:
    """Encode and decode real tensors, and an error holding a Parameter, under a ``FakeTensorMode``, as Monarch's unpickling can."""
    tensors = [torch.arange(6.0).reshape(2, 3), torch.arange(6.0).reshape(3, 2).t()]
    error = ValueError("shard", torch.nn.Parameter(tensors[0]))
    with FakeTensorMode():
        body, buffers = codec.dumps(tensors)
        error_body = codec.encode_error(_raised(error))
    _, body, received = _receive(_frame(Kind.REPLY, body=body, buffers=buffers))
    with FakeTensorMode():
        decoded = codec.loads(body, received)
        decoded_error = codec.decode_error(error_body)
    assert [type(item) for item in decoded] == [torch.Tensor] * 2
    assert all(map(torch.equal, decoded, tensors))
    assert type(decoded_error) is ValueError
    assert type(decoded_error.args[1]) is torch.nn.Parameter
    assert torch.equal(decoded_error.args[1].detach(), tensors[0])


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_a_callers_torch_function_mode_sees_none_of_the_codecs_own_ops() -> None:
    """Keep the codec's own tensor ops from a caller's torch function mode: copies, rebuilds, checks, comparisons and prints.

    Each input reaches its own bypass: a sparse tensor's dumps and loads (its parts go as plain tensors), an error whose
    reduction clones its tensor, an error whose text prints one and a byte view.
    """
    sparse = torch.tensor([[1.0, 0.0], [0.0, 2.0]]).to_sparse()
    body, buffers = codec.dumps(sparse)
    _, body, received = _receive(_frame(Kind.REPLY, body=body, buffers=buffers))
    cloning = _raised(_CloningError(torch.arange(3.0)))
    printing = _raised(_LazyTextError(torch.arange(3.0)))
    recording = _TorchOps()
    with recording:
        codec.dumps(sparse)
        codec.loads(body, received)
        codec.encode_error(cloning)
        codec.encode_error(printing)
        codec.byte_view(torch.ones(2))
    assert recording.codec_ops() == []
    printed = [
        func for file, _, func in recording.ops if file == torch._tensor_str.__file__
    ]
    assert printed == []


def test_a_callers_torch_function_mode_sees_torchs_own_reduction_and_rebuild() -> None:
    """Let the ops of torch's own reduction and rebuild of a tensor subclass reach a caller's torch function mode."""
    subclass = torch.arange(3.0).as_subclass(_Subclass)
    body, buffers = codec.dumps(subclass)
    _, body, received = _receive(_frame(Kind.REPLY, body=body, buffers=buffers))
    dumping, loading = _TorchOps(), _TorchOps()
    with dumping:
        codec.dumps(subclass)
    with loading:
        decoded = codec.loads(body, received)
    assert type(decoded) is _Subclass
    assert torch.Tensor.__reduce_ex__ in [func for *_, func in dumping.ops]
    rebuilt = [func for file, _, func in loading.ops if file == torch._utils.__file__]
    assert rebuilt, loading.ops


def test_the_codec_leaves_another_threads_export_alone() -> None:
    """Trace an export's op while another thread is mid-codec, as pre-dispatch modes sit in process-wide slots."""
    _Gate.inside.clear()
    _Gate.proceed.clear()
    outcome: dict[str, Any] = {}

    def run_codec() -> None:
        try:
            body, buffers = codec.dumps([torch.arange(4.0), _Gate()])
            received = [torch.empty(b.nbytes, dtype=torch.uint8) for b in buffers]
            for buffer, target in zip(buffers, received):
                codec.byte_view(target)[:] = buffer
            outcome["decoded"] = codec.loads(body, received)[0]
            codec.decode_error(codec.encode_error(ValueError("x")))
        except BaseException as e:
            outcome["error"] = e
        outcome["pre_dispatch"] = torch._C._dispatch_tls_is_dispatch_key_included(
            torch._C.DispatchKey.PreDispatch
        )

    class Model(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            thread = threading.Thread(target=run_codec)
            thread.start()
            assert _Gate.inside.wait(60)
            y = x * 2
            _Gate.proceed.set()
            thread.join(60)
            assert not thread.is_alive()
            return y + 1

    exported = torch.export.export(Model(), (torch.ones(3),), strict=False)
    ops = [str(n.target) for n in exported.graph.nodes if n.op == "call_function"]
    assert ops == ["aten.mul.Tensor", "aten.add.Tensor"]
    assert "error" not in outcome, outcome.get("error")
    assert torch.equal(outcome["decoded"], torch.arange(4.0))
    assert outcome["pre_dispatch"] is False


def test_a_codec_thread_that_printed_tensors_keeps_later_exports_whole() -> None:
    """Keep a later export's ops after the codec's thread printed tensors inside codec calls during an earlier one.

    Torch's own printer pops the process-wide pre-dispatch modes and pushes them back, turning the printing thread's
    key on, and an object's reduction can reach it past the codec's printer.
    """
    jobs: queue.Queue[Callable[[], Any] | None] = queue.Queue()
    results: queue.Queue[Any] = queue.Queue()

    def serve() -> None:
        for job in iter(jobs.get, None):
            try:
                results.put(job())
            except BaseException as e:
                results.put(e)

    def run(job: Callable[[], Any]) -> Any:
        jobs.put(job)
        result = results.get(timeout=60)
        assert not isinstance(result, BaseException), result
        return result

    tensor = torch.arange(3.0)
    error = _raised(ValueError("bad shard", tensor))
    body = codec.encode_error(error)
    # Not an error body: its own unpickling would format a tensor, which the codec refuses to load.
    formatting = pickle.dumps(_InitFormatsError(tensor))
    key = torch._C.DispatchKey.PreDispatch
    seen: list[bool] = []

    def refused(data: bytes) -> type[BaseException]:
        with pytest.raises(CarrierProtocolError) as caught:
            codec.decode_error(data)
        return caught.type

    class Prints(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            run(lambda: (codec.encode_error(error), codec.decode_error(body)))
            run(partial(codec.dumps, _Printing()))
            run(partial(codec.dumps, _PrintingPastTheRepr()))
            run(partial(refused, formatting))
            seen.append(
                run(partial(torch._C._dispatch_tls_is_dispatch_key_included, key))
            )
            return x * 2

    class Traces(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            jobs.put(partial(codec.dumps, [tensor, _Gate()]))
            assert _Gate.inside.wait(60)
            try:
                y = x * 2
            finally:
                _Gate.proceed.set()
            result = results.get(timeout=60)
            assert not isinstance(result, BaseException), result
            return y + 1

    _Gate.inside.clear()
    _Gate.proceed.clear()
    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        torch.export.export(Prints(), (torch.ones(3),), strict=False)
        exported = torch.export.export(Traces(), (torch.ones(3),), strict=False)
    finally:
        jobs.put(None)
        thread.join(60)
    ops = [str(n.target) for n in exported.graph.nodes if n.op == "call_function"]
    assert seen == [False] and ops == ["aten.mul.Tensor", "aten.add.Tensor"]


def test_the_codec_prints_an_errors_tensors_without_emptying_an_exports_modes(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Trace an export's op while another thread's codec is printing a tensor in an error.

    The encoded error does not pickle, so its debug line comes first, and must not print the tensor either.
    """
    caplog.set_level(logging.DEBUG, logger="torchstore.carrier")
    tensor = torch.arange(3.0)
    error = _raised(ValueError("bad shard", tensor, threading.Lock()))
    inside, proceed = threading.Event(), threading.Event()
    printing: list[int] = []
    outcome: list[Any] = []
    str_intern = torch._tensor_str._str_intern

    def held(*args: Any, **kwargs: Any) -> str:
        if threading.get_ident() in printing and not inside.is_set():
            inside.set()
            assert proceed.wait(60)
        return str_intern(*args, **kwargs)

    def run() -> None:
        printing.append(threading.get_ident())
        try:
            outcome.append(type(codec.encode_error(error)))
        except BaseException as e:
            outcome.append(e)
        finally:
            inside.set()

    class Model(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            thread = threading.Thread(target=run)
            thread.start()
            assert inside.wait(60)
            try:
                y = x * 2
            finally:
                proceed.set()
                thread.join(60)
            assert not thread.is_alive()
            return y + 1

    monkeypatch.setattr(torch._tensor_str, "_str_intern", held)
    exported = torch.export.export(Model(), (torch.ones(3),), strict=False)
    ops = [str(n.target) for n in exported.graph.nodes if n.op == "call_function"]
    assert ops == ["aten.mul.Tensor", "aten.add.Tensor"]
    assert outcome == [bytes]


def _export_while(
    call: Callable[[], Any], monkeypatch: pytest.MonkeyPatch
) -> tuple[list[str], list[Any]]:
    """Export a model that runs ``call`` on another thread, held where it prints with the pre-dispatch slots empty."""
    inside, proceed = threading.Event(), threading.Event()
    printing: list[int] = []
    outcome: list[Any] = []
    str_intern = torch._tensor_str._str_intern

    def held(*args: Any, **kwargs: Any) -> str:
        empty = not torch._ops._len_torch_dispatch_stack_pre_dispatch()
        if threading.get_ident() in printing and empty and not inside.is_set():
            inside.set()
            assert proceed.wait(60)
        return str_intern(*args, **kwargs)

    def run() -> None:
        printing.append(threading.get_ident())
        try:
            outcome.append(call())
        except BaseException as e:
            outcome.append(e)
        finally:
            inside.set()

    class Model(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            thread = threading.Thread(target=run)
            thread.start()
            try:
                assert inside.wait(60)
                y = x * 2
            finally:
                proceed.set()
                thread.join(60)
            assert not thread.is_alive()
            return y + 1

    monkeypatch.setattr(torch._tensor_str, "_str_intern", held)
    exported = torch.export.export(Model(), (torch.ones(3),), strict=False)
    ops = [str(n.target) for n in exported.graph.nodes if n.op == "call_function"]
    return ops, outcome


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("call", ["encode_error", "decode_error"])
@pytest.mark.parametrize(
    "make",
    [
        _LazyTextError,
        lambda t: ValueError("bad shard", _Wrapper(t)),
        lambda t: ValueError("bad shard", torch.nested.nested_tensor([t, t[:2]])),
        # Torch formats these 0-d tensors with its own repr, not as their items.
        lambda t: _LazyTextError(torch.nn.Parameter(torch.tensor(2.5))),
        lambda t: _LazyTextError(torch.tensor(2.5).as_subclass(_Subclass)),
        lambda t: _LazyTextError(torch.tensor(2.5, device="meta")),
        # A one-element tensor that is not 0-d prints as a tensor too.
        lambda t: _LazyTextError(torch.tensor([2.5])),
        _InitFormatsError,
        _ReduceFormatsError,
    ],
    ids=[
        "f-string",
        "subclass-repr",
        "nested",
        "0d-parameter",
        "0d-subclass",
        "0d-meta",
        "1-element",
        "rebuild-formats",
        "reduce-formats",
    ],
)
def test_the_codec_never_empties_an_exports_modes_to_print_an_error(
    monkeypatch: pytest.MonkeyPatch,
    call: str,
    make: Callable[[torch.Tensor], Exception],
) -> None:
    """Trace an export's op while the codec prints an error, holding it if the process-wide pre-dispatch slots empty.

    Decoding prints only through an error's own constructor, so its other cases check only that each error comes back as itself.
    """
    error = _raised(make(torch.arange(3.0)))
    body = codec.encode_error(error)
    if call == "encode_error":
        ops, outcome = _export_while(
            lambda: type(codec.encode_error(error)), monkeypatch
        )
    else:
        ops, outcome = _export_while(
            lambda: type(codec.decode_error(body)), monkeypatch
        )
    assert ops == ["aten.mul.Tensor", "aten.add.Tensor"]
    assert outcome == [bytes if call == "encode_error" else type(error)]


@pytest.mark.parametrize("call", ["dumps", "loads"])
def test_dumps_and_loads_never_empty_an_exports_modes_to_print(
    monkeypatch: pytest.MonkeyPatch, call: str
) -> None:
    """Trace an export's op while dumps runs a reduction that prints a tensor, or loads a constructor that does."""
    tensor = torch.arange(3.0)
    if call == "dumps":
        ops, outcome = _export_while(
            lambda: len(codec.dumps(_ReduceFormatsError(tensor))[1]), monkeypatch
        )
    else:
        body, buffers = codec.dumps(_InitFormatsError(tensor))
        _, body, received = _receive(_frame(Kind.REPLY, body=body, buffers=buffers))
        ops, outcome = _export_while(
            lambda: len(codec.loads(body, received).args), monkeypatch
        )
    assert ops == ["aten.mul.Tensor", "aten.add.Tensor"]
    assert outcome == [1]


def test_the_codec_stays_out_of_a_pre_dispatch_trace_on_its_own_thread() -> None:
    """Keep the codec's ops out of a pre-dispatch trace on its own thread, as when traced code pickles."""
    tensor = torch.arange(6.0).reshape(2, 3).t()
    received = [
        torch.empty(b.nbytes, dtype=torch.uint8) for b in codec.dumps(tensor)[1]
    ]
    decoded: list[Any] = []

    def traced(x: torch.Tensor) -> torch.Tensor:
        body, buffers = codec.dumps(tensor)
        for buffer, target in zip(buffers, received):
            codec.byte_view(target)[:] = buffer
        decoded.append(codec.loads(body, received))
        decoded.append(codec.decode_error(codec.encode_error(ValueError("x", tensor))))
        return x * 2

    graph = make_fx(traced, pre_dispatch=True)(torch.ones(3)).graph
    ops = [str(n.target) for n in graph.nodes if n.op == "call_function"]
    assert ops == ["aten.mul.Tensor"]
    assert torch.equal(decoded[0], tensor)
    assert type(decoded[1]) is ValueError and torch.equal(decoded[1].args[1], tensor)


def test_the_codec_restores_the_callers_modes_in_their_order() -> None:
    """Leave the caller's two dispatch modes as they were after every codec call, failed ones too, having seen none of its ops."""
    tensor = torch.arange(4.0)
    body, buffers = codec.dumps(tensor)
    received = [torch.empty(b.nbytes, dtype=torch.uint8) for b in buffers]
    for buffer, target in zip(buffers, received):
        codec.byte_view(target)[:] = buffer
    error_body = codec.encode_error(ValueError("x"))
    outer, inner = _Recording(), _Recording()
    with outer, inner:
        for call in (
            partial(codec.dumps, tensor),
            partial(codec.loads, body, received),
            partial(codec.encode_error, ValueError("x")),
            partial(codec.decode_error, error_body),
        ):
            call()
            assert _get_current_dispatch_mode_stack() == [outer, inner]
        for call, error_type in (
            (partial(codec.dumps, threading.Lock()), TypeError),
            (partial(codec.decode_error, b"not a pickle"), CarrierProtocolError),
        ):
            with pytest.raises(error_type):
                call()
            assert _get_current_dispatch_mode_stack() == [outer, inner]
    assert outer.ops == inner.ops == []


def test_round_trips_without_numpy() -> None:
    """Round-trip a bfloat16 tensor in a child process where importing NumPy fails."""
    _run_child(_NO_NUMPY_CHILD)


def test_collecting_a_cycle_that_holds_frame_buffers_is_safe() -> None:
    """Collect a garbage cycle that holds a frame's buffers, which must not crash Python 3.10."""
    _run_child(_CYCLE_CHILD)


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    "make, message",
    [
        (lambda: torch.zeros(4, 4)[:, 0], "not a non-contiguous one"),
        (lambda: torch.zeros(2, device="meta"), "not one on meta"),
        (
            lambda: torch.tensor([1 + 2j]).conj(),
            "not a lazy conjugate or negative view",
        ),
        (
            lambda: torch.tensor([3 - 4j]).conj().imag,
            "not a lazy conjugate or negative view",
        ),
        (lambda: torch.eye(2).to_sparse_csr(), "not a torch.sparse_csr one"),
        (
            lambda: torch.nested.nested_tensor([torch.ones(2), torch.ones(3)]),
            "not a nested one",
        ),
        (lambda: torch.zeros(3).to_mkldnn(), "not a torch._mkldnn one"),
    ],
    ids=["non-contiguous", "meta", "conj", "neg", "csr", "nested", "mkldnn"],
)
def test_byte_view_needs_a_contiguous_cpu_tensor(
    make: Callable[[], torch.Tensor], message: str
) -> None:
    """Refuse a non-contiguous, sparse, nested or MKL-DNN tensor, one off the CPU, or a lazy conjugate or negative view.

    The error names which.
    """
    with pytest.raises(ValueError, match=f"^need a contiguous CPU tensor, {message}$"):
        codec.byte_view(make())


def test_byte_view_refuses_a_storageless_tensor() -> None:
    """Refuse a functorch batched tensor, which has no storage."""
    refused: list[str] = []

    def view(tensor: torch.Tensor) -> torch.Tensor:
        with pytest.raises(ValueError) as caught:
            codec.byte_view(tensor)
        refused.append(str(caught.value))
        return tensor

    torch.func.vmap(view)(torch.ones(2, 3))
    assert refused == ["need a contiguous CPU tensor, not one without storage"]


def test_byte_view_refuses_a_tensor_whose_storage_shrank() -> None:
    """Refuse to view a tensor whose storage shrank under it, which would expose memory the tensor does not own."""
    tensor = torch.arange(16.0)
    with _shrunk(tensor.untyped_storage(), 8):
        with pytest.raises(ValueError, match="storage holds all of its bytes"):
            codec.byte_view(tensor)


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    "dtype, shape, nbytes",
    [
        (torch.quint4x2, (16,), 8),
        (torch.quint4x2, (9,), 5),
        (torch.quint4x2, (3, 3), 5),
        (torch.quint2x4, (16,), 4),
    ],
    ids=["quint4x2", "quint4x2-odd", "quint4x2-rows", "quint2x4"],
)
def test_byte_view_views_packed_bytes_but_no_view_into_them(
    dtype: torch.dtype, shape: tuple[int, ...], nbytes: int
) -> None:
    """View a packed quantized tensor as the whole bytes its values take, and refuse a view into them that its storage holds.

    Those bytes are one run whatever its rows. So a 3 by 3 tensor of 4-bit values takes 5 of the 6 bytes its storage holds.
    """
    values = torch.arange(float(math.prod(shape))).reshape(shape)
    packed = torch.quantize_per_tensor(values, 1.0, 0, dtype)
    assert len(codec.byte_view(packed)) == nbytes
    with pytest.raises(ValueError, match="not a view into packed bytes"):
        codec.byte_view(packed.view(-1)[2:4])


def test_byte_view_keeps_its_tensor_alive() -> None:
    """Keep a viewed tensor alive while its view lives, so the view never reads freed memory."""
    tensor = torch.arange(4, dtype=torch.uint8)
    alive = weakref.ref(tensor)
    view = codec.byte_view(tensor)
    del tensor
    gc.collect()
    assert alive() is not None and bytes(view) == bytes([0, 1, 2, 3])
    del view
    gc.collect()
    assert alive() is None


@pytest.mark.parametrize(
    "error",
    [
        KeyError("w"),
        ValueError("bad value"),
        RuntimeError("boom"),
        AssertionError("x != y"),
        TimeoutError("slow"),
        OSError(errno.ENOENT, "No such file", "/missing"),
        _CustomError("custom"),
    ],
    ids=lambda error: type(error).__name__,
)
def test_remote_errors_keep_their_type_and_text(error: Exception) -> None:
    """Rebuild an error that pickles with its type and text, caused by its remote traceback."""
    decoded = codec.decode_error(codec.encode_error(_raised(error)))
    assert type(decoded) is type(error)
    assert str(decoded) == str(error) and decoded.args == error.args
    cause = str(decoded.__cause__)
    assert type(decoded.__cause__).__name__ == "_RemoteTraceback"
    assert "in _raised\n    raise error\n" in cause
    assert cause.endswith(f"{codec._type_name(type(error))}: {error}\n" + '"""')


def test_the_remote_traceback_is_the_whole_formatted_chain() -> None:
    """Send the remote error's whole formatted traceback, chained causes included, as its cause."""
    try:
        try:
            raise OSError("disk gone")
        except OSError as inner:
            raise KeyError("w") from inner
    except KeyError as e:
        error = e
    expected = "".join(
        traceback.format_exception(type(error), error, error.__traceback__)
    )
    cause = str(codec.decode_error(codec.encode_error(error)).__cause__)
    assert "OSError: disk gone" in cause and cause == f'\n"""\n{expected}"""'


@pytest.mark.parametrize(
    "error",
    [
        PeerLostError("volume closed"),
        StaleAddressError("stale incarnation"),
        CallTimeoutError("call timed out after 300 s"),
        CarrierProtocolError("bad frame magic"),
        RemoteError("KeyError", "'w'"),
    ],
    ids=lambda error: type(error).__name__,
)
def test_carrier_errors_round_trip(error: CarrierError) -> None:
    """Rebuild the carrier's own errors, which a server may send in place of a call's error."""
    decoded = codec.decode_error(codec.encode_error(_raised(error)))
    assert type(decoded) is type(error)
    assert str(decoded) == str(error) and decoded.args == error.args


def test_unpicklable_errors_become_remote_errors(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Send an error of a local class, holding a lock, or whose reduction raises, as its type name and text."""

    class LocalError(Exception):
        pass

    caplog.set_level(logging.DEBUG, logger="torchstore.carrier")
    for error, remote_type in [
        (LocalError("local"), f"{__name__}.{LocalError.__qualname__}"),
        (ValueError(threading.Lock()), "ValueError"),
        (_BadReduceError("reduce"), codec._type_name(_BadReduceError)),
    ]:
        decoded = codec.decode_error(codec.encode_error(_raised(error)))
        assert type(decoded) is RemoteError and decoded.remote_type == remote_type
        assert str(decoded) == f"{remote_type}: {error}"
        assert "in _raised" in str(decoded.__cause__)
    assert caplog.text.count("carrier sends") == 3


def test_errors_only_the_server_can_rebuild_become_remote_errors(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Send an error of a class this process lacks, or whose class here refuses it, as its type name and text, logging why."""

    class ServerOnlyError(Exception):
        pass

    class RefusingError(Exception):
        def __init__(self, *args: Any) -> None:
            raise _CustomError("refused here")

    _server_only(monkeypatch, ServerOnlyError, "carrier_server_only")
    body = codec.encode_error(_raised(ServerOnlyError("only on the server")))
    monkeypatch.delitem(sys.modules, "carrier_server_only")
    caplog.set_level(logging.DEBUG, logger="torchstore.carrier")
    decoded = codec.decode_error(body)
    assert type(decoded) is RemoteError
    assert str(decoded) == "carrier_server_only.ServerOnlyError: only on the server"
    assert "cannot rebuild remote carrier_server_only.ServerOnlyError" in caplog.text
    # Only this end's rebuild raises, so only its catch can turn _CustomError, which no narrowed catch names, into RemoteError.
    RefusingError.__name__ = "ServerOnlyError"
    _server_only(monkeypatch, RefusingError, "carrier_server_only")
    decoded = codec.decode_error(body)
    assert type(decoded) is RemoteError
    assert str(decoded) == "carrier_server_only.ServerOnlyError: only on the server"


def test_debug_lines_print_a_failures_tensors_without_torchs_repr(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Log why each end could not rebuild an error, printing the failure's tensor, but never through torch's repr."""
    caplog.set_level(logging.DEBUG, logger="torchstore.carrier")
    pickled = pickle.dumps(_UnrebuildableError("x"), protocol=5)
    name = codec._type_name(_UnrebuildableError)
    body = pickle.dumps((name, "x", "remote traceback\n", pickled), protocol=5)
    printed: list[Any] = []
    torch_str = torch._tensor_str._str

    def recorded(*args: Any, **kwargs: Any) -> str:
        printed.append(args)
        return torch_str(*args, **kwargs)

    # Torch's repr empties the process-wide pre-dispatch slots, as _TensorRepr does not.
    monkeypatch.setattr(torch._tensor_str, "_str", recorded)
    assert (
        type(codec.decode_error(codec.encode_error(_UnrebuildableError("x"))))
        is RemoteError
    )
    assert type(codec.decode_error(body)) is RemoteError
    assert printed == []
    assert caplog.text.count("cannot rebuild', tensor([0., 1., 2.])") == 2


@pytest.mark.parametrize(
    "error",
    [
        _TwoArgError("w", "missing"),
        _CheckedError(3),
        _PrefixError("text"),
        _AliasError("renamed"),
        _FrozenFieldError(3),
        _SlotError("lost", 3),
        _LossyReduceError("lost", 3),
        _TruncatingError("shard", 3),
        _DroppedAttrError("lost", 3),
        _ChildSlotError("lost", 3),
        _ListifyingError((1, 2)),
    ],
    ids=lambda error: type(error).__name__,
)
def test_errors_that_rebuild_differently_become_remote_errors(error: Exception) -> None:
    """Keep the remote type and text when rebuilding fails or would change either."""
    decoded = codec.decode_error(codec.encode_error(_raised(error)))
    assert type(decoded) is RemoteError
    assert decoded.remote_type == codec._type_name(type(error))
    assert str(decoded) == f"{decoded.remote_type}: {error}"


def test_an_error_holding_a_set_keeps_its_type_under_any_hash_seeds() -> None:
    """Rebuild an error holding a set, alone or beside a tensor or a NaN, in a list or an attribute too, under any hash seed."""

    def run(seeds: range, mode: str, data: str = "") -> list[str]:
        children = [
            subprocess.Popen(
                [sys.executable, "-c", _SET_ERROR_CHILD, _ROOT, mode],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env={**os.environ, "PYTHONHASHSEED": str(seed)},
            )
            for seed in seeds
        ]
        try:
            outputs = [child.communicate(data, timeout=100) for child in children]
        finally:
            for child in children:
                child.kill()
                child.wait()
        for child, (stdout, stderr) in zip(children, outputs):
            assert child.returncode == 0, stdout + stderr
        return [stdout for stdout, _ in outputs]

    # Each sender also decodes its own error.
    bodies = "".join(run(range(4), "encode"))
    assert run(range(4, 5), "decode", bodies)[0].splitlines()[-1] == "decoded 20"


def test_an_error_holding_a_tensor_keeps_its_type_under_other_print_options() -> None:
    """Rebuild an error holding a tensor where torch's print options would print it otherwise."""
    error = _raised(ValueError("bad shard", torch.tensor([1.23456789])))
    body = codec.encode_error(error)
    with torch._tensor_str.printoptions(precision=2):
        decoded = codec.decode_error(body)
    assert type(decoded) is ValueError and torch.equal(decoded.args[1], error.args[1])


def test_an_error_whose_tensor_another_thread_writes_keeps_its_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rebuild an error whose tensor is written right after the sender's trial rebuild, which shares it."""
    tensor = torch.arange(3.0)
    error = _raised(ValueError("bad shard", tensor))
    load = codec._InPlaceUnpickler.load

    def load_then_write(self: Any) -> Any:
        rebuilt = load(self)
        tensor.add_(1)
        return rebuilt

    monkeypatch.setattr(codec._InPlaceUnpickler, "load", load_then_write)
    decoded = codec.decode_error(codec.encode_error(error))
    assert type(decoded) is ValueError
    assert torch.equal(decoded.args[1], torch.arange(3.0))


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_judging_an_errors_rebuild_loads_no_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Judge an error's rebuild on a copy sharing its tensors and storages, so encoding loads no storage, as a GPU's might not fit.

    A bare ``UntypedStorage`` does not survive torch's own pickling, so the storage is a ``TypedStorage``.
    """
    tensor = torch.arange(1 << 20, dtype=torch.float32)[:4]
    other, storage = torch.ones(2, 2), torch.arange(8.0).storage()
    error = _raised(ValueError("bad shard", tensor, other, storage))

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("encoding loaded a storage")

    monkeypatch.setattr(torch, "load", refuse)
    body = codec.encode_error(error)
    monkeypatch.undo()
    decoded = codec.decode_error(body)
    assert type(decoded) is ValueError and torch.equal(decoded.args[1], tensor)
    assert torch.equal(decoded.args[2], other)
    assert decoded.args[3].tolist() == storage.tolist()


# Torch prints a tensor of up to 1,000 items whole, reading past a shrunk storage instead of failing to slice it.
@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    "kind, numel, nbytes",
    [
        ("tensor", 1 << 20, 0),
        ("parameter", 1 << 20, 0),
        ("subclass", 16, 8),
        ("qint8", 16, 8),
    ],
    ids=["freed-tensor", "freed-parameter", "shrunk-subclass", "shrunk-qint8"],
)
def test_an_error_holding_a_tensor_whose_storage_lost_its_bytes_travels_as_text(
    kind: str, numel: int, nbytes: int
) -> None:
    """Send as text alone an error holding a tensor whose storage was freed (as FSDP frees a resharded Parameter) or shrunk.

    Torch's rebuild would fail or make up the bytes its reduction did not send. Torch's repr raises only for a freed
    tensor large enough to print summarized (as these freed ones are) and otherwise reads a null pointer or past a shrunk storage.
    """
    tensor = torch.arange(numel, dtype=torch.float32)
    if kind == "parameter":
        tensor = torch.nn.Parameter(tensor)
    elif kind == "subclass":
        tensor = tensor.as_subclass(_Subclass)
    elif kind == "qint8":
        tensor = torch.quantize_per_tensor(tensor, 0.5, 2, torch.qint8)
    with _shrunk(tensor.untyped_storage(), nbytes):
        body = codec.encode_error(_raised(ValueError("bad", tensor)))
        assert pickle.loads(body)[3] is None  # no copy of the error itself
        decoded = codec.decode_error(body)
        assert type(decoded) is RemoteError
        assert str(decoded) == "ValueError: <ValueError whose str() failed>"


def test_printing_an_errors_tensor_records_no_autograd() -> None:
    """Print an error's tensor as torch's own repr does, with no op saved for autograd.

    A saved-tensor hook such as an offloading one would otherwise run for each op the print makes.
    """
    saved: list[torch.Size] = []

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        saved.append(tensor.shape)
        return tensor

    leaf = torch.arange(6.0, requires_grad=True)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        decoded = codec.decode_error(codec.encode_error(_raised(_LazyTextError(leaf))))
    assert saved == [] and str(decoded) == str(_LazyTextError(leaf))


def test_printing_an_errors_functorch_tensor_shows_its_value() -> None:
    """Print a functorch tensor in an error's text as torch's own repr does."""
    texts: list[tuple[str, str]] = []

    def printed(tensor: torch.Tensor) -> torch.Tensor:
        error = _LazyTextError(tensor)
        texts.append((codec._text(error), str(error)))
        return tensor

    torch.func.vmap(printed)(torch.ones(2, 3))
    assert len(texts) == 1 and texts[0][0] == texts[0][1], texts


def test_an_error_holding_a_bare_untyped_storage_travels_as_text() -> None:
    """Send an error holding a bare ``UntypedStorage`` as text alone, as torch's own rebuild cannot load one."""
    body = codec.encode_error(_raised(ValueError("bad", torch.UntypedStorage(8))))
    assert pickle.loads(body)[3] is None


def test_an_error_formatting_a_shrunk_0d_tensor_travels_as_text() -> None:
    """Send an error whose text formats a 0-d tensor whose storage shrank, which torch formats as its item, as text."""
    tensor = torch.tensor(7, dtype=torch.int64)
    with _shrunk(tensor.untyped_storage(), 4):
        decoded = codec.decode_error(
            codec.encode_error(_raised(_LazyTextError(tensor)))
        )
        name = codec._type_name(_LazyTextError)
        assert type(decoded) is RemoteError
        assert str(decoded) == f"{name}: <{name} whose str() failed>"


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("dtype", [torch.quint4x2, torch.quint2x4], ids=str)
def test_an_error_holding_a_packed_quantized_tensor_keeps_its_type(
    dtype: torch.dtype,
) -> None:
    """Rebuild an error holding a quantized tensor that packs 2 or 4 values a byte, which a storage check by items would refuse."""
    packed = torch.quantize_per_tensor(torch.arange(8.0), 0.5, 2, dtype)
    error = _raised(ValueError("bad shard", packed))
    decoded = codec.decode_error(codec.encode_error(error))
    assert type(decoded) is ValueError and str(decoded) == str(error)
    assert torch.equal(decoded.args[1].int_repr(), packed.int_repr())


def test_an_error_the_sender_cannot_rebuild_travels_as_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Send an error whose rebuild fails on the sender as its type name and text, though the receiver could rebuild it."""
    error = _raised(_SenderRefusedError("shard 3"))
    monkeypatch.setattr(_SenderRefusedError, "refusing", True)
    body = codec.encode_error(error)
    monkeypatch.setattr(_SenderRefusedError, "refusing", False)
    decoded = codec.decode_error(body)
    assert type(decoded) is RemoteError
    assert str(decoded) == f"{codec._type_name(_SenderRefusedError)}: shard 3"


def test_an_error_whose_args_do_not_compare_is_judged_by_its_text() -> None:
    """Rebuild an error whose args raise a non-tensor error when compared, judging the copy by its text."""
    error = _raised(ValueError("bad shard", _Incomparable()))
    decoded = codec.decode_error(codec.encode_error(error))
    assert type(decoded) is ValueError and str(decoded) == str(error)


@pytest.mark.parametrize(
    "error",
    [
        asyncio.CancelledError("handler cancelled"),
        concurrent.futures.CancelledError("executor shut down"),
        KeyboardInterrupt(),
        StopIteration(5),
        StopAsyncIteration("done"),
        _CancelledException("handler cancelled"),
    ],
    ids=lambda error: codec._type_name(type(error)),
)
def test_cancellations_and_base_exceptions_become_remote_errors(
    error: BaseException,
) -> None:
    """Never rebuild a ``BaseException``, ``CancelledError`` or iteration stop, which a caller takes as its own.

    A cancellation that is also an ``Exception`` comes back as ``RemoteError`` too.
    """
    decoded = codec.decode_error(codec.encode_error(_raised(error)))
    assert type(decoded) is RemoteError and isinstance(decoded, Exception)
    remote_type = codec._type_name(type(error))
    assert decoded.remote_type == remote_type
    assert str(decoded) == (f"{remote_type}: {error}" if str(error) else remote_type)
    assert "in _raised" in str(decoded.__cause__)


def test_an_error_whose_text_fails_still_travels() -> None:
    """Send an error whose ``str()`` raises with a placeholder text, and rebuild it."""
    decoded = codec.decode_error(codec.encode_error(_raised(_BadStrError())))
    assert type(decoded) is _BadStrError
    assert (
        codec._text(decoded) == f"<{codec._type_name(_BadStrError)} whose str() failed>"
    )
    assert "in _raised" in str(decoded.__cause__)


def test_an_error_whose_text_is_a_str_subclass_travels_as_plain_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Send the text of an error whose ``str()`` returns a str subclass, local or server-only, as a plain ``str``.

    One subclass's ``__str__`` returns itself, which ``str()`` would pass on unchanged.
    """

    class LocalText(str):
        pass

    class SelfText(str):
        def __str__(self) -> str:
            return self

    class ServerText(str):
        pass

    _server_only(monkeypatch, ServerText, "carrier_server_text")
    texts = [LocalText("local text"), SelfText("self text"), ServerText("server text")]
    bodies = [codec.encode_error(_raised(_TextError(text))) for text in texts]
    monkeypatch.delitem(sys.modules, "carrier_server_text")
    for body, text in zip(bodies, texts):
        decoded = codec.decode_error(body)
        assert type(decoded) is RemoteError and type(decoded.args[1]) is str
        assert str(decoded) == f"{codec._type_name(_TextError)}: {text}"


@pytest.mark.parametrize(
    "error", [_FrozenError(), _ImmutableError("nope")], ids=["frozen", "immutable"]
)
def test_an_error_that_refuses_attributes_is_rebuilt_with_its_cause(
    error: Exception,
) -> None:
    """Rebuild an error whose class refuses attribute assignment, still caused by its remote traceback."""
    decoded = codec.decode_error(codec.encode_error(_raised(error)))
    assert type(decoded) is type(error) and str(decoded) == str(error)
    assert "in _raised" in str(decoded.__cause__) and decoded.__suppress_context__


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_errors_holding_tensors_keep_their_text() -> None:
    """Send an error holding a tensor with its text as torch prints it, formatted, nested, computed with, wrapped or zero too."""
    tensor = torch.arange(3.0)
    unpicklable = ValueError("bad shard", tensor, threading.Lock())
    decoded = codec.decode_error(codec.encode_error(_raised(unpicklable)))
    assert type(decoded) is RemoteError and str(decoded) == f"ValueError: {unpicklable}"
    for error in (
        unpicklable,
        _SummingError(torch.arange(6.0).reshape(2, 3)),
        _LazyTextError(tensor),
        _LazyTextError(torch.tensor(2.5)),
        _LazyTextError(torch.nn.Parameter(tensor)),
        ValueError("bad shard", torch.nested.nested_tensor([tensor, tensor[:2]])),
        _LazyTextError(_ContentsWrapper(tensor)),
        ValueError("bad shard", torch._efficientzerotensor(3)),
    ):
        cause = str(codec.decode_error(codec.encode_error(_raised(error))).__cause__)
        assert cause.endswith(f"{codec._type_name(type(error))}: {error}\n" + '"""')


def test_an_error_of_a_class_without_a_module_travels_by_its_bare_name() -> None:
    """Send an error of a class without ``__module__``, as ``type()`` makes under globals without ``__name__``."""
    cls = eval("type('ModulelessError', (Exception,), {})", {})
    assert not hasattr(cls, "__module__")
    decoded = codec.decode_error(codec.encode_error(_raised(cls("no module"))))
    assert type(decoded) is RemoteError
    assert str(decoded) == "ModulelessError: no module"
    assert "ModulelessError" in str(decoded.__cause__)


@pytest.mark.parametrize(
    "error",
    [
        SyntaxError("bad", ("f.py", 1, 1, b"x = (")),  # type: ignore[call-overload]
        _BadTextSyntaxError("bad", ("f.py", 1, 1, "x = (")),
    ],
    ids=["bytes-text", "text-raises"],
)
def test_an_error_whose_traceback_fails_to_format_still_travels(
    error: SyntaxError,
) -> None:
    """Send an error whose traceback does not format, as a ``SyntaxError`` holding bytes does, with a placeholder."""
    decoded = codec.decode_error(codec.encode_error(_raised(error)))
    assert type(decoded) is type(error) and str(decoded) == str(error)
    placeholder = f"<{codec._type_name(type(error))} whose traceback failed>"
    assert placeholder in str(decoded.__cause__)


def test_malformed_error_bodies_are_refused() -> None:
    """Refuse an error frame body that does not hold an encoded error, or holds other fields or types, a str subclass too."""
    bodies = [b"not a pickle"] + [
        pickle.dumps(fields)
        for fields in (
            ("a", "b"),
            ["a", "b", "c", None],
            (1, "b", "c", None),
            ("a", 2, "c", None),
            ("a", "b", 3, None),
            ("a", _Text("b"), "c", None),
            ("a", "b", "c", 5),
        )
    ]
    for body in bodies:
        with pytest.raises(CarrierProtocolError, match="not an error frame body"):
            codec.decode_error(body)


_loaded: list[str] = []


def _load_records() -> str:
    _loaded.append("loaded")
    return "a"


class _RecordsItsLoad:
    """Unpickles by calling a global that records the call. Any object that a body names would run its own code like this."""

    def __reduce__(self) -> tuple[Any, ...]:
        return _load_records, ()


def test_an_error_body_that_names_a_global_is_refused_before_loading_it() -> None:
    """Refuse an error body whose fields name a global (which they never need) without running the global."""
    _loaded.clear()
    with pytest.raises(CarrierProtocolError, match="names a global"):
        codec.decode_error(pickle.dumps((_RecordsItsLoad(), "text", "traceback", None)))
    assert _loaded == []


def test_an_error_body_that_does_not_unpickle_is_refused_with_a_capped_reason() -> None:
    """Cap the reason for refusing an error body, without its cause, as its unpickling error can quote the body at length."""
    with pytest.raises(CarrierProtocolError) as info:
        codec.decode_error(b"cbuiltins\n" + b"x" * 100_000 + b"\n.")
    reason = str(info.value)
    assert reason.startswith("not an error frame body: UnpicklingError: ")
    assert len(reason) == len("not an error frame body: ") + 200
    assert info.value.__cause__ is None and info.value.__suppress_context__
    printed = "".join(traceback.format_exception(info.type, info.value, info.tb))
    assert "x" * 1000 not in printed and len(printed) < 2000


def test_error_classes() -> None:
    """Catch a call timeout as either timeout type, and a lost peer as a connection error."""
    for timeout_type in (TimeoutError, asyncio.TimeoutError):
        with pytest.raises(timeout_type):
            raise CallTimeoutError("call timed out after 300 s")
    assert issubclass(StaleAddressError, PeerLostError)
    assert issubclass(PeerLostError, ConnectionError)
    error_types = (CallTimeoutError, PeerLostError, CarrierProtocolError, RemoteError)
    assert all(issubclass(error_type, CarrierError) for error_type in error_types)
    assert str(RemoteError("KeyboardInterrupt", "")) == "KeyboardInterrupt"


def test_wire_constants() -> None:
    """Pin the magic, version and kind numbers, which peers of other versions read."""
    assert (codec.MAGIC, codec.PROTOCOL_VERSION) == (b"TSC1", 1)
    assert {kind.name: kind.value for kind in Kind} == {
        "HELLO": 1,
        "WELCOME": 2,
        "REJECT": 3,
        "REQUEST": 4,
        "REPLY": 5,
        "ERROR": 6,
        "CANCEL": 7,
    }


def test_header_layout() -> None:
    """Lay out the 40-byte header as ``!4sBBHIQQQI``, then each buffer's length in bytes, however wide its items."""
    buffers: list[memoryview[Any]] = [
        memoryview(b"ab"),
        memoryview(b"cde"),
        memoryview(bytearray(16)).cast("d"),
    ]
    chunks = codec.encode_frame(
        Kind.REQUEST, 9, deadline_us=5_000_000, body=b"body", buffers=buffers
    )
    header = _raw_header(count=3, request_id=9, deadline_us=5_000_000, body_len=4)
    assert codec.HEADER_SIZE == 40 and len(header) == 40
    assert chunks[0] == header + struct.pack("!3Q", 2, 3, 16)
    assert chunks[1] == b"body" and chunks[2] is buffers[0] and chunks[3] is buffers[1]


@pytest.mark.parametrize("kind", list(Kind), ids=lambda kind: kind.name)
def test_every_kind_round_trips(kind: Kind) -> None:
    """Read back each kind's request id, deadline, body and buffers, handshake kinds first."""
    handshake = kind in _HANDSHAKE_KINDS
    buffers = [] if handshake else [codec.byte_view(torch.arange(3, dtype=torch.uint8))]
    data = _frame(kind, 2**64 - 1, deadline_us=12, body=b"{}", buffers=buffers)
    header, body, received = _receive(data, handshake=handshake)
    assert header == codec.FrameHeader(kind, 2**64 - 1, 12, len(buffers), 2)
    assert body == b"{}"
    assert [buffer.tolist() for buffer in received] == [[0, 1, 2]] * len(buffers)


@pytest.mark.parametrize(
    "fields, message",
    [
        ({"magic": b"TSC2"}, "bad frame magic"),
        ({"kind": 0}, "unknown frame kind 0"),
        ({"kind": 8}, "unknown frame kind 8"),
        ({"flags": 1}, "reserved frame header fields"),
        ({"reserved": 1}, "reserved frame header fields"),
        ({"spare": 1}, "reserved frame header fields"),
        ({"count": codec.MAX_BUFFERS + 1}, "over the limit of 1048576"),
        ({"body_len": 2**63}, "over the 910737418240-byte cap"),
    ],
    ids=[
        "magic",
        "kind-0",
        "kind-8",
        "flags",
        "reserved",
        "spare",
        "count",
        "body",
    ],
)
def test_bad_headers_are_refused(fields: dict[str, Any], message: str) -> None:
    """Refuse a malformed header, or one announcing too much, before anything is allocated."""
    with pytest.raises(CarrierProtocolError, match=message):
        codec.decode_header(_raw_header(**fields), handshake=False)


@pytest.mark.parametrize(
    "lengths",
    [(2**63,), (codec.MAX_FRAME_BYTES // 2, codec.MAX_FRAME_BYTES // 2)],
    ids=["one", "sum"],
)
def test_oversized_buffer_lengths_are_refused(lengths: tuple[int, ...]) -> None:
    """Refuse buffer lengths over the frame cap, alone or together, before any allocation."""
    header = codec.decode_header(_raw_header(count=len(lengths)), handshake=False)
    with pytest.raises(CarrierProtocolError, match="over the 910737418240-byte cap"):
        codec.decode_lengths(header, struct.pack(f"!{len(lengths)}Q", *lengths))


def test_truncated_headers_and_lengths_are_refused() -> None:
    """Refuse a header or a length table of the wrong size, as malformed data."""
    header = codec.decode_header(_raw_header(count=2), handshake=False)
    for data in (b"", _raw_header()[:-1], _raw_header() + b"\0"):
        with pytest.raises(CarrierProtocolError, match="a frame header is 40 bytes"):
            codec.decode_header(data, handshake=False)
    for data in (struct.pack("!Q", 1), struct.pack("!3Q", 1, 2, 3)):
        with pytest.raises(CarrierProtocolError, match="expected 16 bytes of buffer"):
            codec.decode_lengths(header, data)


@pytest.mark.parametrize(
    "request_id, deadline_us",
    [(-1, 0), (1 << 64, 0), (1.0, 0), (0, -1), (0, 1 << 64)],
    ids=["negative-id", "huge-id", "float-id", "negative-deadline", "huge-deadline"],
)
def test_ids_and_deadlines_must_fit_the_header(
    request_id: Any, deadline_us: Any
) -> None:
    """Refuse a request id or remaining deadline that is not an unsigned 64-bit integer, rather than let struct raise."""
    with pytest.raises(ValueError, match="must be unsigned 64-bit integers"):
        codec.encode_frame(Kind.REQUEST, request_id, deadline_us=deadline_us)


def test_a_frame_may_fill_the_caps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Accept a frame at the byte cap and buffer limit, both read at call time, but no more."""
    monkeypatch.setattr(codec, "MAX_FRAME_BYTES", 100)
    monkeypatch.setattr(codec, "MAX_BUFFERS", 1)
    buffer = codec.byte_view(torch.zeros(48, dtype=torch.uint8))
    data = _frame(Kind.REPLY, body=b"body", buffers=[buffer])
    assert len(data) == 100
    _receive(data)
    with pytest.raises(CarrierProtocolError, match="REPLY frame of 101 bytes"):
        codec.encode_frame(Kind.REPLY, body=b"body!", buffers=[buffer])
    header = codec.decode_header(
        _raw_header(kind=Kind.REPLY, count=1, body_len=5), handshake=False
    )
    with pytest.raises(CarrierProtocolError, match="REPLY frame of 101 bytes"):
        codec.decode_lengths(header, struct.pack("!Q", 48))
    with pytest.raises(CarrierProtocolError, match="REPLY frame of 101 bytes"):
        codec.decode_header(
            _raw_header(kind=Kind.REPLY, count=1, body_len=53), handshake=False
        )
    with pytest.raises(CarrierProtocolError, match="2 buffers, over the limit of 1"):
        codec.encode_frame(Kind.REPLY, buffers=[memoryview(b"a"), memoryview(b"b")])
    with pytest.raises(CarrierProtocolError, match="2 buffers, over the limit of 1"):
        codec.decode_header(_raw_header(kind=Kind.REPLY, count=2), handshake=False)


def test_frames_before_the_handshake_are_small_handshake_frames() -> None:
    """Accept only handshake frames before the handshake, with no buffers and at most 64 KiB."""
    limit = codec.MAX_HANDSHAKE_FRAME_BYTES - codec.HEADER_SIZE
    header = codec.decode_header(
        _raw_header(kind=Kind.HELLO, body_len=limit), handshake=True
    )
    assert header.body_len == limit
    for raw, message in [
        (_raw_header(kind=Kind.HELLO, body_len=limit + 1), "of 65537 bytes, over"),
        (_raw_header(kind=Kind.REQUEST), "REQUEST frame before the handshake"),
        (_raw_header(kind=Kind.WELCOME, count=1), "over the limit of 0"),
    ]:
        with pytest.raises(CarrierProtocolError, match=message):
            codec.decode_header(raw, handshake=True)
    with pytest.raises(CarrierProtocolError, match="HELLO frame after the handshake"):
        codec.decode_header(_raw_header(kind=Kind.HELLO), handshake=False)
    with pytest.raises(CarrierProtocolError, match="over the 65536-byte cap"):
        codec.encode_frame(Kind.HELLO, body=b"x" * limit + b"x")
    with pytest.raises(CarrierProtocolError, match="over the limit of 0"):
        codec.encode_frame(Kind.REJECT, buffers=[memoryview(b"x")])


def test_handshake_bodies_are_json() -> None:
    """Send the protocol version, scope, service, incarnation and sender's ``host:pid`` as JSON, in frozen records."""
    hello = Hello("store/nonce", "publisher/0", "9f2c", "host-a:123")
    fields = {"scope": "store/nonce", "service": "publisher/0", "incarnation": "9f2c"}
    assert json.loads(hello.to_json()) == {**fields, "peer": "host-a:123", "version": 1}
    _, body, _ = _receive(
        _frame(Kind.HELLO, body=hello.to_json().encode()), handshake=True
    )
    assert Hello.from_json(body) == hello
    reject = Reject("stale incarnation")
    assert json.loads(reject.to_json()) == {"reason": "stale incarnation", "version": 1}
    assert Reject.from_json(reject.to_json()) == reject
    for record in (hello, reject):
        with pytest.raises(dataclasses.FrozenInstanceError):
            record.version = 2  # type: ignore[misc]
    future = {**fields, "peer": "host-b:7", "version": 2, "added_later": True}
    assert Hello.from_json(json.dumps(future)).version == 2


def test_carrier_address_round_trips_through_json() -> None:
    """Round-trip an address through JSON as a frozen key that hashes alike, and keep its scope out of reprs."""
    address = CarrierAddress("host-a", 29500, "publisher/0", "store/nonce", "9f2c")
    assert CarrierAddress.from_json(address.to_json()) == address
    assert {address: "connection"}[
        CarrierAddress.from_json(address.to_json())
    ] == "connection"
    with pytest.raises(dataclasses.FrozenInstanceError):
        address.port = 1  # type: ignore[misc]
    assert json.loads(address.to_json()) == {
        "host": "host-a",
        "port": 29500,
        "service": "publisher/0",
        "scope": "store/nonce",
        "incarnation": "9f2c",
    }
    assert "nonce" not in repr(address) + repr(Hello("store/nonce", "s", "i", "p"))


def test_records_look_up_their_field_types_once() -> None:
    """Look a record class's field types up once, as every hello, welcome and address parse needs them."""
    codec._field_types.cache_clear()
    record = Hello("store/nonce", "publisher/0", "9f2c", "host-a:123").to_json()
    for _ in range(3):
        Hello.from_json(record)
    info = codec._field_types.cache_info()
    assert (info.misses, info.hits) == (1, 2)


def test_malformed_json_records_are_refused() -> None:
    """Refuse a record that is not JSON."""
    with pytest.raises(CarrierProtocolError, match="CarrierAddress"):
        CarrierAddress.from_json(b"not json")


@pytest.mark.parametrize(
    "wrap", [bytearray, memoryview], ids=lambda wrap: wrap.__name__
)
def test_records_parse_from_bytes_or_str_only(wrap: Callable[[bytes], Any]) -> None:
    """Refuse a record of any other type, though its bytes hold a valid record."""
    data = wrap(
        Hello("store/nonce", "publisher/0", "9f2c", "host-a:123").to_json().encode()
    )
    message = f"^not a Hello object: {type(data).__name__}, not bytes or str$"
    with pytest.raises(CarrierProtocolError, match=message):
        Hello.from_json(data)


def test_malformed_json_records_never_echo_their_input() -> None:
    """Keep a refused record's bytes, which may hold a scope, out of the error: undecodable, of a wrong type or short a field."""
    record = Hello("store/NONCE-0123", "publisher/0", "9f2c", "host-a:123").to_json()
    fields = json.loads(record)
    missing = json.dumps(
        {name: value for name, value in fields.items() if name != "peer"}
    )
    for data in (
        record.encode() + b"\x80",
        record[:-1].encode() + b"\xff" * 60_000,
        # An escaped lone surrogate parses as a str that cannot be encoded.
        json.dumps({**fields, "peer": "\ud800"}),
        json.dumps({**fields, "version": "1"}),
        # A bool is an int to isinstance, but not a version.
        json.dumps({**fields, "version": True}),
        missing,
        "[]",
    ):
        with pytest.raises(
            CarrierProtocolError, match="^(not a Hello|Hello fields)"
        ) as refused:
            Hello.from_json(data)
        assert "NONCE" not in str(refused.value) and len(str(refused.value)) < 200
        # Raised from None where it replaces another error, so a traceback prints only its own line.
        assert refused.value.__cause__ is None
        assert refused.value.__context__ is None or refused.value.__suppress_context__


@pytest.mark.parametrize(
    "scope",
    ["w" + "[" * 100 + "/nonce", 'w"' + "{" * 100 + "/nonce"],
    ids=["brackets", "brackets-after-a-quote"],
)
def test_brackets_in_strings_never_count_toward_the_cap(scope: str) -> None:
    """Parse records whose scope holds 100 brackets, after an escaped quote too, as json never nests on them."""
    address = CarrierAddress("host-a", 29500, "publisher/0", scope, "9f2c")
    assert CarrierAddress.from_json(address.to_json()) == address
    hello = Hello(scope, "publisher/0", "9f2c", "host-a:123")
    assert Hello.from_json(hello.to_json().encode()) == hello


def test_records_with_over_64_brackets_outside_strings_are_refused() -> None:
    """Parse a record with 64 opening brackets outside its strings, its own brace included, and refuse one with 65."""
    hello = Hello("store/nonce", "publisher/0", "9f2c", "host-a:123")

    def nested(depth: int) -> str:
        return hello.to_json()[:-1] + ', "x": ' + "[" * depth + "]" * depth + "}"

    assert Hello.from_json(nested(63)) == hello
    message = "^not a Hello object: ValueError: over 64 brackets$"
    with pytest.raises(CarrierProtocolError, match=message):
        Hello.from_json(nested(64))


def test_a_deeply_nested_record_is_refused_under_a_raised_recursion_limit() -> None:
    """Refuse a hello of nested brackets on a thread under a raised recursion limit, which crashed Python 3.10."""
    _run_child(_DEEP_RECORD_CHILD)
