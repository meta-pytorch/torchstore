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
* (``NixlCheckpointTransport``, healing: see below in this module.)

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
import threading
import time
import uuid as _uuid
from datetime import timedelta
from typing import Any, Optional, Union

import torch
from torch.distributed import PrefixStore
from torch.distributed.distributed_c10d import AllreduceOptions, ReduceOp, Work

from torchstore.transport import neuron_memory

try:  # upstream torchft
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
_GRAD_SLOT_BYTES = int(os.environ.get("TORCHSTORE_TORCHFT_GRAD_SLOT_BYTES", str(64 << 20)))
_BACKEND = os.environ.get("TORCHSTORE_NIXL_BACKEND", "LIBFABRIC")
_ALIGN = 64


def _load_nixl() -> tuple[Any, Any]:
    from torchstore.transport.nixl import _load_nixl_api

    return _load_nixl_api()


def _synchronize(device: torch.device) -> None:
    if neuron_memory.NEURON_DEVICE_TYPES and device.type in neuron_memory.NEURON_DEVICE_TYPES:
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
            nixl_agent_config(enable_prog_thread=True, enable_listen_thread=False, backends=[]),
        )
        self.agent.create_backend(_BACKEND, {})
        self.agent.register_memory(
            neuron_memory.neuron_reg_descs(self.agent, self.arena)
            if neuron_memory.is_neuron(self.arena)
            else self.agent.get_reg_descs(self.arena),
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
            self.name, total, device, 2 * max_world, slot_bytes,
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
            raise RuntimeError("NIXL peers disagree on TORCHSTORE_TORCHFT_* slot settings")
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
        remote = self.agent.get_xfer_descs([(peer["va"] + offset, nbytes, peer["dev"])], self.mem)
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
                raise TimeoutError(f"NIXL: {len(tags - self.seen)} notification(s) missing")
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
        }
        return work

    def _retire(self) -> None:
        generation, self._generation = self._generation, None
        if generation is not None:
            generation["retired"].set()
            dead = generation["tag"].encode()
            self._link.seen = {m for m in self._link.seen if m.split(b":")[0][1:] != dead}

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
            deadline = time.monotonic() + self._nixl_timeout.total_seconds()
            with self._op_lock:
                self._allreduce(generation, tensor, reduce_op, deadline)
        return _DummyWork(tensor)

    def _allreduce(self, g: dict[str, Any], tensor: torch.Tensor, op: ReduceOp, deadline: float) -> None:
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
                link.write(peer, link.slot_off(rank, parity), nbytes, tag + str(rank).encode())
                for peer in g["peers"].values()
            ]
            link.finish(handles, deadline)
            link.wait({tag + str(p).encode() for p in g["peers"]}, deadline, g["retired"])
            # Same rank order on every replica: bit-identical results.
            parts = [
                link.arena[:nbytes].view(tensor.dtype)
                if r == rank
                else link.arena[link.slot_off(r, parity) : link.slot_off(r, parity) + nbytes].view(
                    tensor.dtype
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
