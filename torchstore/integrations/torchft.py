# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""torchft fault tolerance over NIXL, for any device TorchStore's NIXL transport moves.

Upstream torchft takes any process group and any checkpoint transport, so the
device- and fabric-specific pieces live here instead of in torchft:

* ``ProcessGroupNixl``: upstream ``torchft.ProcessGroupGloo`` (the control plane)
  whose ``allreduce`` of a device tensor moves the bytes with NIXL one-sided
  WRITEs. On AWS Neuron that is LIBFABRIC (EFA) straight into HBM.
* ``NixlCheckpointTransport``: upstream ``CheckpointTransport`` for healing. The
  source registers its state tensors' storages and publishes their addresses
  (plus NIXL partial metadata and upstream's pickled state-dict layout) on the
  generation's store; the recovering replica registers its own tensors (in place
  when given a template) and READs everything with one NIXL transfer per memory
  type. No staging copies.

Gradient allreduce: all-to-all. Each rank registers, once at construction, a send
slot and two receive slots per possible peer (``2 * max_world_size`` slots). A
tensor goes in slot-sized chunks: copy into the send slot, synchronize the
device, WRITE into slot ``[my_rank][seq % 2]`` of every peer with notification
``g<uuid>:<seq>:<rank>``, wait for every peer, sum in rank order (so every
replica gets a bit-identical result). Two slots per peer suffice: a peer writes
chunk seq+1 only after it received this rank's chunk seq.

Lessons carried over from the torchft fork's NIXL port (measured on trn3, Qwen3,
4 replicas x 16 ranks: 5.27 s/step vs 6.63 s for torch-neuronx native RDMA):

* stage with a same-dtype copy (the send slot viewed as the gradient's dtype):
  copying through a uint8 view of the source ran at ~1.2 GB/s (3 s per step);
* create the agent and register memory at construction, on the main thread: a
  rejoiner creating it in reconfigure stalled every replica ~7 s, and loading
  NIXL's native module on torchft's quorum thread segfaulted under Monarch;
* never remove a remote agent: that cancels a dead peer's pending write and the
  cancellation lands on the next peer's first transfer (and LIBFABRIC cannot
  reconnect a removed agent).
"""

from __future__ import annotations

import base64
import json
import logging
import os
import pickle
import threading
import time
import uuid as _uuid
from datetime import timedelta
from typing import Any, Callable, Generic, Optional, TypeVar, Union

import torch
from torch.distributed import PrefixStore
from torch.distributed.distributed_c10d import AllreduceOptions, ReduceOp, Work

from torchstore.transport import neuron_memory

try:  # upstream torchft
    from torchft.checkpointing.pg_transport import (
        _DTensorMeta,
        _StateDictMeta,
        _TensorMeta,
    )
    from torchft.checkpointing.transport import CheckpointTransport
    from torchft.process_group import (
        _rendezvous,
        create_store_client,
        ProcessGroupGloo,
        ReconfigureOptions,
    )
    from torchft.work import _DummyWork
except ImportError as _error:  # pragma: no cover - torchft is optional for torchstore
    raise ImportError("torchstore.integrations.torchft needs torchft") from _error

logger = logging.getLogger(__name__)

_MAX_WORLD_SIZE = int(os.environ.get("TORCHSTORE_TORCHFT_MAX_WORLD_SIZE", "4"))
_GRAD_SLOT_BYTES = int(
    os.environ.get("TORCHSTORE_TORCHFT_GRAD_SLOT_BYTES", str(64 << 20))
)
_BACKEND = os.environ.get("TORCHSTORE_NIXL_BACKEND", "LIBFABRIC")
_OP_TIMEOUT_S = float(os.environ.get("TORCHSTORE_TORCHFT_OP_TIMEOUT_S", "30"))
_WARMUP_TIMEOUT_S = float(os.environ.get("TORCHSTORE_TORCHFT_WARMUP_TIMEOUT_S", "300"))
_ALIGN = 64


def _load_nixl() -> tuple[Any, Any]:
    from torchstore.transport.nixl import _load_nixl_api

    return _load_nixl_api()


def _synchronize(device: torch.device) -> None:
    if (
        neuron_memory.NEURON_DEVICE_TYPES
        and device.type in neuron_memory.NEURON_DEVICE_TYPES
    ):
        import torch_neuronx

        torch_neuronx.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


class _Link:
    """One rank's NIXL endpoint: a registered arena of receive slots and a send slot."""

    def __init__(self, device: torch.device, slot_bytes: int, max_world: int) -> None:
        nixl_agent, nixl_agent_config = _load_nixl()
        self.slot_bytes = slot_bytes
        self.max_world = max_world
        total = slot_bytes * (1 + 2 * max_world)
        self.arena = torch.zeros(total, dtype=torch.uint8, device=device)
        _synchronize(device)
        self.name = f"torchstore-torchft-{os.getpid()}-{_uuid.uuid4().hex[:8]}"
        self.agent = nixl_agent(
            self.name,
            nixl_agent_config(
                enable_prog_thread=True, enable_listen_thread=False, backends=[]
            ),
        )
        self.agent.create_backend(_BACKEND, {})
        self.agent.register_memory(
            (
                neuron_memory.neuron_reg_descs(self.agent, self.arena)
                if neuron_memory.is_neuron(self.arena)
                else self.agent.get_reg_descs(self.arena)
            ),
            backends=[_BACKEND],
        )
        self.va = (
            neuron_memory.neuron_va(self.arena)
            if neuron_memory.is_neuron(self.arena)
            else self.arena.data_ptr()
        )
        self.dev = device.index or 0
        self.mem = "VRAM" if device.type != "cpu" else "DRAM"
        self.seen: set[bytes] = set()
        self.known: set[str] = set()
        logger.info(
            "torchstore torchft: NIXL agent %s, %d B registered on %s (%d slots of %d B)",
            self.name,
            total,
            device,
            2 * max_world,
            slot_bytes,
        )

    def slot_off(self, writer: int, parity: int) -> int:
        return self.slot_bytes * (1 + 2 * writer + parity)

    def info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "meta": base64.b64encode(self.agent.get_agent_metadata()).decode(),
            "va": self.va,
            "dev": self.dev,
            "slot_bytes": self.slot_bytes,
            "max_world": self.max_world,
        }

    def connect(self, info: dict[str, Any]) -> dict[str, Any]:
        if info["slot_bytes"] != self.slot_bytes or info["max_world"] != self.max_world:
            raise RuntimeError(
                "NIXL peers disagree on TORCHSTORE_TORCHFT_* slot settings"
            )
        if info["name"] not in self.known:  # never removed: see module docstring
            self.agent.add_remote_agent(base64.b64decode(info["meta"]))
            self.known.add(info["name"])
        return info

    def stage(self, x: torch.Tensor) -> int:
        n = x.numel() * x.element_size()
        self.arena[:n].view(x.dtype).copy_(x.reshape(-1))  # same-dtype copy: fast path
        return n

    def write(self, peer: dict[str, Any], offset: int, nbytes: int, tag: bytes) -> Any:
        local = self.agent.get_xfer_descs([(self.va, nbytes, self.dev)], self.mem)
        remote = self.agent.get_xfer_descs(
            [(peer["va"] + offset, nbytes, peer["dev"])], self.mem
        )
        handle = self.agent.initialize_xfer("WRITE", local, remote, peer["name"], tag)
        if self.agent.transfer(handle) == "ERR":
            self.agent.release_xfer_handle(handle)
            raise RuntimeError(f"NIXL write {tag!r}: ERR")
        return handle

    def finish(self, handles: list[Any], deadline: float) -> None:
        try:
            for handle in handles:
                state = self.agent.check_xfer_state(handle)
                while state not in ("DONE", "ERR"):
                    if time.monotonic() > deadline:
                        raise TimeoutError("NIXL write timed out")
                    state = self.agent.check_xfer_state(handle)
                if state != "DONE":
                    raise RuntimeError(f"NIXL write: {state}")
        finally:
            for handle in handles:
                try:
                    self.agent.release_xfer_handle(handle)
                except Exception:
                    pass

    def wait(self, tags: set[bytes], deadline: float, retired: threading.Event) -> None:
        while not tags <= self.seen:
            for messages in self.agent.get_new_notifs().values():
                self.seen.update(messages)
            if tags <= self.seen:
                break
            if retired.is_set():
                raise RuntimeError("NIXL wait abandoned: process group reconfigured")
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"NIXL: {len(tags - self.seen)} notification(s) missing"
                )
            time.sleep(0.0002)
        self.seen -= tags


class ProcessGroupNixl(ProcessGroupGloo):
    """Upstream torchft Gloo control plane; device-tensor allreduce over NIXL."""

    def __init__(
        self,
        timeout: timedelta = timedelta(seconds=60),
        device: Optional[torch.device] = None,
        max_world_size: int = _MAX_WORLD_SIZE,
        grad_slot_bytes: int = _GRAD_SLOT_BYTES,
    ) -> None:
        super().__init__(timeout=timeout)
        self._nixl_timeout = timeout
        if device is None:
            device = _default_device()
        self._device = device
        slot = -(-grad_slot_bytes // _ALIGN) * _ALIGN
        # Main thread, at construction (see module docstring).
        self._link = _Link(device, slot, max_world_size)
        self._op_lock = threading.Lock()
        self._generation: Optional[dict[str, Any]] = None

    def get_reconfigure_handle(self) -> str:
        """Upstream's handle, with a store host other pods can reach.

        Upstream advertises ``socket.gethostname()``; on Kubernetes that is the bare
        pod name, which other pods cannot resolve. ``TORCHFT_ADVERTISED_HOST`` (the
        pod IP) replaces it when set.
        """
        handle = json.loads(super().get_reconfigure_handle())
        host = os.environ.get("TORCHFT_ADVERTISED_HOST")
        if host:
            handle["store"] = f"{host}:{handle['store'].rsplit(':', 1)[1]}"
        return json.dumps(handle)

    def reconfigure(self, opts: ReconfigureOptions) -> Work:
        self._retire()
        work = super().reconfigure(opts)  # Gloo control group on the rendezvous store
        rendezvous = _rendezvous(opts, self._handle_id())
        if rendezvous.world_size > self._link.max_world:
            raise RuntimeError(
                f"{rendezvous.world_size} replicas > TORCHSTORE_TORCHFT_MAX_WORLD_SIZE="
                f"{self._link.max_world}"
            )
        timeout = opts.timeout or self._nixl_timeout
        store = PrefixStore(
            f"nixl/{rendezvous.uuid}",
            create_store_client(rendezvous.store_addr, timeout=timeout),
        )
        store.set(str(rendezvous.rank), json.dumps(self._link.info()))
        keys = [str(r) for r in range(rendezvous.world_size)]
        store.wait(keys, timeout)
        peers = {
            r: self._link.connect(json.loads(bytes(store.get(str(r))).decode()))
            for r in range(rendezvous.world_size)
            if r != rendezvous.rank
        }
        self._generation = {
            "tag": str(rendezvous.uuid & 0xFFFFFFFF),
            "rank": rendezvous.rank,
            "world": rendezvous.world_size,
            "peers": peers,
            "seq": 0,
            "retired": threading.Event(),
            "store": store,
        }
        return work

    def _retire(self) -> None:
        generation, self._generation = self._generation, None
        if generation is not None:
            generation["retired"].set()
            dead = generation["tag"].encode()
            self._link.seen = {
                m for m in self._link.seen if m.split(b":")[0][1:] != dead
            }

    def abort(self, errored: bool = True) -> None:
        self._retire()
        super().abort(errored)

    def shutdown(self) -> None:
        self._retire()
        super().shutdown()

    def allreduce(
        self, tensors: list[torch.Tensor], opts: Union[AllreduceOptions, ReduceOp]
    ) -> Work:
        if not any(t.device.type != "cpu" for t in tensors):
            return super().allreduce(tensors, opts)
        if len(tensors) != 1:
            raise RuntimeError("ProcessGroupNixl allreduce takes one device tensor")
        reduce_op = opts.reduceOp if isinstance(opts, AllreduceOptions) else opts
        reduce_op = getattr(reduce_op, "op", reduce_op)  # ReduceOp object -> RedOpType
        if reduce_op not in (ReduceOp.SUM, ReduceOp.AVG, ReduceOp.MAX):
            raise RuntimeError(f"ProcessGroupNixl does not support {reduce_op}")
        generation = self._generation
        if generation is None:
            raise RuntimeError("ProcessGroupNixl is not configured")
        tensor = tensors[0]
        if generation["world"] > 1:
            # The first allreduce of a generation waits for its slowest member
            # (first-step compile, a rejoiner's warm-up); later ones fail fast, so a
            # dead peer costs one skipped step and a new quorum, not a hang.
            budget = _OP_TIMEOUT_S if generation.get("warm") else _WARMUP_TIMEOUT_S
            deadline = time.monotonic() + budget  # not capped by the PG timeout
            with self._op_lock:
                self._allreduce(generation, tensor, reduce_op, deadline)
            generation["warm"] = True
        return _DummyWork(tensor)

    def _allreduce(
        self, g: dict[str, Any], tensor: torch.Tensor, op: ReduceOp, deadline: float
    ) -> None:
        link = self._link
        flat = tensor.reshape(-1)
        if not flat.is_contiguous():
            raise RuntimeError("ProcessGroupNixl allreduce needs a contiguous tensor")
        per_chunk = link.slot_bytes // tensor.element_size()
        rank, world = g["rank"], g["world"]
        for start in range(0, flat.numel(), per_chunk):
            part = flat[start : start + per_chunk]
            seq = g["seq"]
            g["seq"] += 1
            parity = seq % 2
            nbytes = link.stage(part)
            _synchronize(tensor.device)
            tag = f"g{g['tag']}:{seq}:".encode()
            handles = [
                link.write(
                    peer, link.slot_off(rank, parity), nbytes, tag + str(rank).encode()
                )
                for peer in g["peers"].values()
            ]
            link.finish(handles, deadline)
            link.wait(
                {tag + str(p).encode() for p in g["peers"]}, deadline, g["retired"]
            )
            # Same rank order on every replica: bit-identical results.
            parts = [
                (
                    link.arena[:nbytes].view(tensor.dtype)
                    if r == rank
                    else link.arena[
                        link.slot_off(r, parity) : link.slot_off(r, parity) + nbytes
                    ].view(tensor.dtype)
                )
                for r in range(world)
            ]
            part.copy_(parts[0])
            for x in parts[1:]:
                if op == ReduceOp.MAX:
                    torch.maximum(part, x, out=part)
                else:
                    part.add_(x)
            # No sync: the next chunk's stage+sync orders these reads before any
            # peer can reuse this parity slot.
        if op == ReduceOp.AVG:
            tensor.div_(world)

    def getBackendName(self) -> str:
        return "torchstore-nixl-gloo"


def _default_device() -> torch.device:
    try:
        import torch_neuronx

        return torch.device("neuron", torch_neuronx.current_device())
    except ImportError:
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")


T = TypeVar("T")


def _region_of(t: torch.Tensor) -> tuple[str, tuple[int, int, int]]:
    """(memory type, (address, nbytes, device id)) of a contiguous tensor."""
    if neuron_memory.is_neuron(t):
        return "VRAM", (
            neuron_memory.neuron_va(t),
            t.nbytes,
            neuron_memory.neuron_device_id(t),
        )
    if t.device.type == "cuda":
        return "VRAM", (t.data_ptr(), t.nbytes, t.device.index or 0)
    return "DRAM", (t.data_ptr(), t.nbytes, 0)


def _regions(tensors: list[torch.Tensor]) -> dict[str, list[tuple[int, int, int]]]:
    """Regions grouped by memory type (for registration)."""
    by_type: dict[str, list[tuple[int, int, int]]] = {}
    for t in tensors:
        mem, entry = _region_of(t)
        by_type.setdefault(mem, []).append(entry)
    return by_type


def _storage_bytes(tensor: torch.Tensor) -> torch.Tensor:
    """A uint8 tensor over ``tensor``'s whole storage.

    Upstream's ``_cast_tensor`` uses ``torch.tensor(storage)``, which torch-neuronx
    does not support ("could not determine the shape of object type ..."); a view
    set onto the storage works on every device.
    """
    return torch.empty(0, dtype=torch.uint8, device=tensor.device).set_(
        tensor.untyped_storage()
    )


def _prepare(state_dict: object, step: int) -> tuple[Any, list[torch.Tensor]]:
    """Upstream's ``_prepare_state_dict`` (same metadata), storages via ``_storage_bytes``."""
    from torch.distributed.tensor import DTensor
    from torch.utils._pytree import tree_flatten_with_path

    leaves, treespec = tree_flatten_with_path(state_dict)
    paths, non_tensor_leaves, tensors = [], [], []

    def prep(t: torch.Tensor) -> Any:
        tensors.append(_storage_bytes(t))
        return _TensorMeta(
            shape=t.shape,
            dtype=t.dtype,
            storage_offset=int(t.storage_offset()),
            stride=t.stride(),
            nbytes=t.untyped_storage().nbytes(),
        )

    for path, v in leaves:
        paths.append(path)
        if isinstance(v, DTensor):
            non_tensor_leaves.append(
                _DTensorMeta(local=prep(v._local_tensor), spec=v._spec)
            )
        elif isinstance(v, torch.Tensor):
            non_tensor_leaves.append(prep(v))
        else:
            non_tensor_leaves.append(v)
    return (
        _StateDictMeta(
            step=step,
            treespec=treespec,
            paths=paths,
            non_tensor_leaves=non_tensor_leaves,
        ),
        tensors,
    )


class NixlCheckpointTransport(CheckpointTransport[T], Generic[T]):
    """Heal a recovering replica by READing the source's state over NIXL."""

    def __init__(
        self,
        pg: ProcessGroupNixl,
        timeout: timedelta = timedelta(seconds=120),
        state_dict: Optional[Callable[[], object]] = None,
    ) -> None:
        self._pg = pg
        self._timeout = timeout
        self._state_dict = state_dict

    def set_state_dict(self, state_dict: Callable[[], object]) -> None:
        """In-place receive into this callable's tensors (set once the Manager exists)."""
        self._state_dict = state_dict

    def metadata(self) -> str:
        return "<torchstore-nixl>"

    def _store_and_rank(self) -> tuple[Any, int, dict[str, Any]]:
        g = self._pg._generation
        if g is None:
            raise RuntimeError(
                "NixlCheckpointTransport: process group is not configured"
            )
        return g["store"], g["rank"], g

    def _register(self, tensors: list[torch.Tensor]) -> list[Any]:
        agent = self._pg._link.agent
        regs = []
        for mem, entries in _regions(tensors).items():
            regs.append(
                agent.register_memory(
                    agent.get_reg_descs([(a, n, d, "") for a, n, d in entries], mem),
                    backends=[_BACKEND],
                )
            )
        return regs

    def _deregister(self, regs: list[Any]) -> None:
        for reg in regs:
            try:
                self._pg._link.agent.deregister_memory(reg)
            except Exception as error:
                logger.warning("NixlCheckpointTransport: deregister failed: %s", error)

    def send_checkpoint(
        self,
        dst_ranks: list[int],
        step: int,
        state_dict: T,
        timeout: Optional[timedelta] = None,
    ) -> None:
        timeout = timeout or self._timeout
        store, rank, _ = self._store_and_rank()
        t0 = time.monotonic()
        meta, tensors = _prepare(state_dict, step)
        tensors = [t for t in tensors if t.nbytes > 0]
        _synchronize(self._pg._device)
        regs = self._register(tensors)
        try:
            agent = self._pg._link.agent
            info = json.dumps(
                {
                    "meta": base64.b64encode(pickle.dumps(meta)).decode(),
                    "regions": [(*_region_of(t)[1], _region_of(t)[0]) for t in tensors],
                    "md": [
                        base64.b64encode(
                            agent.get_partial_agent_metadata(r, False, [_BACKEND])
                        ).decode()
                        for r in regs
                    ],
                }
            )
            for dst in dst_ranks:
                store.set(f"ckpt/{step}/{rank}/{dst}", info)
            store.wait([f"ckpt/{step}/{rank}/{dst}/done" for dst in dst_ranks], timeout)
        finally:
            self._deregister(regs)
        logger.info(
            "NixlCheckpointTransport: sent step %d to %s: %d tensors, %d B in %.2f s",
            step,
            dst_ranks,
            len(tensors),
            sum(t.nbytes for t in tensors),
            time.monotonic() - t0,
        )

    def recv_checkpoint(
        self,
        src_rank: int,
        metadata: str,
        step: int,
        timeout: Optional[timedelta] = None,
    ) -> T:
        from torch.distributed.tensor import DTensor
        from torch.utils._pytree import tree_flatten_with_path, tree_unflatten

        timeout = timeout or self._timeout
        store, rank, g = self._store_and_rank()
        t0 = time.monotonic()
        key = f"ckpt/{step}/{src_rank}/{rank}"
        store.wait([key], timeout)
        info = json.loads(bytes(store.get(key)).decode())
        agent = self._pg._link.agent
        for md in info["md"]:
            try:
                agent.add_remote_agent(base64.b64decode(md))
            except Exception as error:  # a section this agent already holds
                if "NOT_ALLOWED" not in f"{type(error).__name__} {error}":
                    raise
        meta = pickle.loads(base64.b64decode(info["meta"]))
        if meta.step != step:
            raise RuntimeError(
                f"NixlCheckpointTransport: got step {meta.step}, expected {step}"
            )

        template = dict(
            tree_flatten_with_path(self._state_dict() if self._state_dict else {})[0]
        )
        device = self._pg._device
        locals_: list[torch.Tensor] = []
        values: list[object] = []
        for path, v in zip(meta.paths, meta.non_tensor_leaves):
            local_meta = v.local if isinstance(v, _DTensorMeta) else v
            if not isinstance(local_meta, _TensorMeta):
                values.append(v)
                continue
            inplace = template.get(path)
            if isinstance(inplace, DTensor):
                inplace = inplace._local_tensor
            if (
                isinstance(inplace, torch.Tensor)
                and inplace.untyped_storage().nbytes() == local_meta.nbytes
            ):
                storage = _storage_bytes(inplace)
            else:
                storage = torch.empty(
                    local_meta.nbytes, dtype=torch.uint8, device=device
                )
            if local_meta.nbytes > 0:
                locals_.append(storage)
            t = torch.as_strided(
                storage.view(local_meta.dtype),
                size=local_meta.shape,
                stride=local_meta.stride,
                storage_offset=local_meta.storage_offset,
            )
            values.append(
                DTensor(t, v.spec, requires_grad=False)
                if isinstance(v, _DTensorMeta)
                else t
            )

        remote = info["regions"]
        if len(remote) != len(locals_):
            raise RuntimeError(
                f"NixlCheckpointTransport: {len(remote)} source tensors, {len(locals_)} here"
            )
        regs = self._register(locals_)
        handles = []
        try:
            groups: dict[tuple[str, str], tuple[list, list]] = {}
            for t, (a, n, d, rmem) in zip(locals_, remote):
                if t.nbytes != n:
                    raise RuntimeError("NixlCheckpointTransport: tensor size mismatch")
                lmem, (la, ln, ld) = _region_of(t)
                group = groups.setdefault((lmem, rmem), ([], []))
                group[0].append((la, ln, ld))
                group[1].append((a, n, d))
            deadline = time.monotonic() + timeout.total_seconds()
            src_name = g["peers"][src_rank]["name"]
            for (lmem, rmem), (local, rem) in groups.items():
                h = agent.initialize_xfer(
                    "READ",
                    agent.get_xfer_descs(local, lmem),
                    agent.get_xfer_descs(rem, rmem),
                    src_name,
                )
                if agent.transfer(h) == "ERR":
                    agent.release_xfer_handle(h)
                    raise RuntimeError("NixlCheckpointTransport: READ ERR")
                handles.append(h)
            self._pg._link.finish(handles, deadline)
            handles = []
            _synchronize(device)
        finally:
            store.set(f"{key}/done", "1")  # release the source either way
            self._deregister(regs)
        logger.info(
            "NixlCheckpointTransport: received step %d from %d: %d tensors, %d B in %.2f s",
            step,
            src_rank,
            len(locals_),
            sum(t.nbytes for t in locals_),
            time.monotonic() - t0,
        )
        return tree_unflatten(values, meta.treespec)

    def shutdown(self, wait: bool = True) -> None:
        pass
