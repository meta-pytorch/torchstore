# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""XCCL transport for TorchStore on Intel XPU.

Mirrors gloo.py but uses ProcessGroupXCCL and keeps tensors on XPU,
avoiding the two device-host copies that gloo would incur.
"""

import asyncio
import os
import socket
import uuid
from datetime import timedelta
from logging import getLogger
from typing import Any, TYPE_CHECKING

import portpicker
import torch
import torch.distributed as dist
from torch.distributed import PrefixStore, ProcessGroup, Store, TCPStore

from torchstore.transport.buffers import TransportBuffer, TransportCache
from torchstore.transport.types import Request

if TYPE_CHECKING:
    from torchstore.transport.buffers import TransportContext
    from torchstore.transport.pipe import StorageVolumeRef


logger = getLogger(__name__)

_store_addrs: dict[str, tuple[str, int, str]] = {}

TORCHSTORE_XCCL_ENABLED = os.environ.get("TORCHSTORE_XCCL_ENABLED", "1") == "1"
TORCHSTORE_XCCL_INIT_TIMEOUT = int(
    os.environ.get("TORCHSTORE_XCCL_INIT_TIMEOUT", "120")
)


def xccl_available() -> bool:
    """True iff xccl backend and at least one XPU device are available."""
    if not TORCHSTORE_XCCL_ENABLED:
        return False
    if not hasattr(dist, "is_xccl_available"):
        return False
    try:
        if not dist.is_xccl_available():
            return False
    except Exception:
        return False
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        return False
    try:
        return torch.xpu.device_count() > 0
    except Exception:
        return False


def _get_hostname() -> str:
    return socket.getfqdn()


def _xpu_device() -> torch.device:
    """Pick the XPU tile this process should use (LOCAL_RANK or xpu:0)."""
    if "LOCAL_RANK" in os.environ:
        try:
            return torch.device(f"xpu:{int(os.environ['LOCAL_RANK'])}")
        except (TypeError, ValueError):
            pass
    return torch.device("xpu:0")


def _xccl_factory(
    store: Store,
    rank: int,
    world_size: int,
    timeout: timedelta,
    device: torch.device,
    group_name: str,
) -> ProcessGroup:
    """Construct a 2-rank ProcessGroup with XCCL backend bound to ``device``.

    Builds the PG directly from the supplied Store (no global state) so
    each TorchStore transfer gets its own private PG. Options must carry
    global_ranks_in_group and group_name, otherwise oneCCL spins up a
    separate internal KVS and the ranks fail to meet.
    """
    from torch.distributed import ProcessGroupXCCL

    if device.type == "xpu":
        torch.xpu.set_device(device)

    prefix = f"xccl/{device.type}:{device.index if device.index is not None else 0}/"
    backend_prefix_store = PrefixStore(prefix, store)

    options = ProcessGroupXCCL.Options()
    options.global_ranks_in_group = list(range(world_size))
    options.group_name = group_name
    options._timeout = timeout

    backend_class = ProcessGroupXCCL(backend_prefix_store, rank, world_size, options)

    pg = ProcessGroup(backend_prefix_store, rank, world_size)
    pg._set_default_backend(ProcessGroup.BackendType.XCCL)
    pg._register_backend(device, ProcessGroup.BackendType.XCCL, backend_class)
    pg._set_group_name(group_name)
    return pg


class XcclProcessGroupCache(TransportCache):
    def __init__(self) -> None:
        self._process_groups: dict[str, ProcessGroup] = {}

    def put(self, store_key: str, pg: ProcessGroup) -> None:
        self._process_groups[store_key] = pg

    def get(self, store_key: str) -> ProcessGroup:
        return self._process_groups[store_key]

    def clear(self) -> None:
        self._process_groups.clear()


class XcclTransportBuffer(TransportBuffer):
    """Device-resident transport using ProcessGroupXCCL (oneCCL).

    Same handshake/send/recv protocol as GlooTransportBuffer but tensors
    stay on XPU throughout.
    """

    supports_inplace_resharding = False

    def __init__(self, storage_volume_ref: "StorageVolumeRef") -> None:
        super().__init__(storage_volume_ref)
        self.shape: torch.Size | None = None
        self.dtype: torch.dtype | None = None
        self.master_addr: str | None = None
        self.master_port: int | None = None
        self.store_key: str | None = None
        self.is_object: bool = False
        self.objects: Any = None
        self._tcp_store: TCPStore | None = None
        self._pg_task: asyncio.Task | None = None
        self._send_task: asyncio.Task | None = None
        self._recv_task: asyncio.Task | None = None

    def requires_handshake(self, requests: list[Request]) -> bool:
        volume_id = self.storage_volume_ref.volume_id
        if volume_id in _store_addrs:
            cached_addr = _store_addrs[volume_id]
            self.master_addr = cached_addr[0]
            self.master_port = cached_addr[1]
            self.store_key = cached_addr[2]
            return False
        return True

    async def _pre_handshake(self) -> None:
        volume_id = self.storage_volume_ref.volume_id
        self.store_key = f"torchstore_xccl_{str(uuid.uuid4())[:8]}"
        self.master_addr = _get_hostname()
        self.master_port = portpicker.pick_unused_port()

        logger.info(
            f"[pid={os.getpid()}] xccl handshake with StorageVolume:[{volume_id}] "
            f"TCPStore at {self.master_addr}:{self.master_port}"
        )

        self._tcp_store = TCPStore(
            host_name=self.master_addr,
            port=self.master_port,
            world_size=2,
            is_master=True,
            timeout=timedelta(seconds=TORCHSTORE_XCCL_INIT_TIMEOUT),
            wait_for_workers=False,
        )

        tcp_store = self._tcp_store
        device = _xpu_device()
        group_name = self.store_key

        def create_pg():
            return _xccl_factory(
                store=tcp_store,
                rank=0,
                world_size=2,
                timeout=timedelta(seconds=TORCHSTORE_XCCL_INIT_TIMEOUT),
                device=device,
                group_name=group_name,
            )

        self._pg_task = asyncio.create_task(asyncio.to_thread(create_pg))

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["storage_volume_ref"] = None
        state["_tcp_store"] = None
        state["_pg_task"] = None
        state["_send_task"] = None
        state["_recv_task"] = None
        return state

    async def recv_handshake(
        self,
        ctx: "TransportContext",
        entries: list[tuple[Request, Any]],
    ) -> list[None]:
        """Storage-side: build the PG as rank 1."""
        logger.info(
            f"[pid={os.getpid()}] xccl PG setup at "
            f"{self.master_addr}:{self.master_port}"
        )

        master_addr = self.master_addr
        master_port = self.master_port
        device = _xpu_device()
        group_name = self.store_key

        def create_pg():
            tcp_store = TCPStore(
                host_name=master_addr,
                port=master_port,
                world_size=2,
                is_master=False,
                timeout=timedelta(seconds=TORCHSTORE_XCCL_INIT_TIMEOUT),
            )
            return _xccl_factory(
                store=tcp_store,
                rank=1,
                world_size=2,
                timeout=timedelta(seconds=TORCHSTORE_XCCL_INIT_TIMEOUT),
                device=device,
                group_name=group_name,
            )

        pg = await asyncio.to_thread(create_pg)
        ctx.get(XcclProcessGroupCache).put(self.store_key, pg)
        return [None]

    async def _post_handshake(
        self,
        handshake_results: list[Any],
        requests: list[Request],
    ) -> None:
        volume_id = self.storage_volume_ref.volume_id
        pg = await self._pg_task

        _store_addrs[volume_id] = (self.master_addr, self.master_port, self.store_key)
        self.storage_volume_ref.transport_context.get(XcclProcessGroupCache).put(
            self.store_key, pg
        )
        self._tcp_store = None
        self._pg_task = None
        logger.info(f"xccl handshake done with StorageVolume:[{volume_id}]")

    async def _pre_put_hook(self, requests: list[Request]) -> None:
        assert len(requests) == 1
        request = requests[0]

        if request.is_object:
            self.is_object = True
            return
        if request.tensor_val is None:
            return

        tensor = request.tensor_val
        self.shape = tensor.shape
        self.dtype = tensor.dtype

        self._send_task = asyncio.create_task(
            self._send_tensor(tensor, self.storage_volume_ref.transport_context)
        )

    async def handle_put_request(
        self,
        ctx: "TransportContext",
        entries: list[tuple[Request, Any]],
    ) -> list[Any]:
        assert len(entries) == 1
        request, maybe_tensor = entries[0]

        if request.is_object:
            self.is_object = True
            return [request.objects]

        tensor = maybe_tensor
        if tensor is None:
            tensor = torch.empty(self.shape, dtype=self.dtype, device=_xpu_device())

        tensor = await self._receive_tensor(tensor, ctx)
        return [tensor]

    async def _pre_get_hook(self, requests: list[Request]) -> None:
        assert len(requests) == 1
        request = requests[0]

        meta = (
            await self.storage_volume_ref.volume.get_meta.call_one(
                [request.meta_only()]
            )
        )[0]

        if request.tensor_slice is not None:
            self.shape = torch.Size(request.tensor_slice.local_shape)
            self.dtype = meta[1]
        else:
            if isinstance(meta, str) or meta is None:
                self.is_object = True
                return
            self.shape = meta[0]
            self.dtype = meta[1]

        tensor = torch.empty(self.shape, dtype=self.dtype, device=_xpu_device())
        self._recv_task = asyncio.create_task(
            self._receive_tensor(tensor, self.storage_volume_ref.transport_context)
        )

    async def handle_get_request(
        self,
        ctx: "TransportContext",
        entries: list[tuple[Request, Any]],
    ) -> None:
        assert len(entries) == 1
        _, data = entries[0]
        if not isinstance(data, torch.Tensor):
            self.is_object = True
            self.objects = data
            return
        await self._send_tensor(data, ctx)

    async def _handle_storage_volume_response(
        self, requests: list[Request], transport_buffer: "TransportBuffer"
    ) -> list[Any]:
        if transport_buffer.is_object:
            return [transport_buffer.objects]

        if self._recv_task is not None:
            tensor = await self._recv_task
            self._recv_task = None
            if tensor is None:
                raise RuntimeError(
                    f"receive_tensor returned None (is_object={self.is_object}, "
                    f"shape={self.shape}, dtype={self.dtype})"
                )
            return [tensor]

        raise RuntimeError(f"No recv task available (is_object={self.is_object})")

    async def _receive_tensor(
        self, tensor: torch.Tensor, transport_context: "TransportContext"
    ) -> torch.Tensor:
        target = _xpu_device()
        if tensor.device != target:
            tensor = torch.empty(tensor.shape, dtype=tensor.dtype, device=target)

        pg = transport_context.get(XcclProcessGroupCache).get(self.store_key)
        my_rank = pg.rank()
        remote_rank = 1 - my_rank

        def do_recv():
            work = pg.recv([tensor], srcRank=remote_rank, tag=0)
            work.wait()
            torch.xpu.synchronize(target)

        await asyncio.to_thread(do_recv)
        return tensor

    async def _send_tensor(
        self, tensor: torch.Tensor, transport_context: "TransportContext"
    ) -> None:
        target = _xpu_device()
        if tensor.device != target:
            tensor = tensor.to(target)
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        pg = transport_context.get(XcclProcessGroupCache).get(self.store_key)
        my_rank = pg.rank()
        remote_rank = 1 - my_rank

        def do_send():
            work = pg.send([tensor], dstRank=remote_rank, tag=0)
            work.wait()
            torch.xpu.synchronize(target)

        await asyncio.to_thread(do_send)

    async def drop(self) -> None:
        if self._send_task is not None:
            await self._send_task
            self._send_task = None
        self.is_object = False
        self.objects = None
