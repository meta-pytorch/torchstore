# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""XCCL transport for TorchStore on Intel XPU.

Mirrors gloo.py one-for-one but constructs a ``ProcessGroupXCCL`` instead
of ``ProcessGroupGloo`` and keeps tensors device-resident on ``xpu``.

Why a separate transport, not just a flag on the gloo path:

- Gloo stages every transfer through CPU. For a Qwen3-0.6B+LoRA
  ``state_dict`` (~600 MB), that's two device↔host copies plus TCP per
  transfer.
- xccl uses oneCCL's Level Zero queues directly, so the bytes never
  leave XPU memory.

This transport sits above gloo in the selection ladder; gloo remains
the universal fallback for non-XPU hosts.
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


# Same shape as gloo.py: per-volume cache of (master_addr, master_port, store_key)
_store_addrs: dict[str, tuple[str, int, str]] = {}


TORCHSTORE_XCCL_ENABLED = os.environ.get("TORCHSTORE_XCCL_ENABLED", "1") == "1"
TORCHSTORE_XCCL_INIT_TIMEOUT = int(
    os.environ.get("TORCHSTORE_XCCL_INIT_TIMEOUT", "120")
)


def xccl_available() -> bool:
    """Return True iff the XCCL transport is usable on this host.

    Conditions:
    1. ``TORCHSTORE_XCCL_ENABLED`` is "1" (default).
    2. ``torch.distributed.is_xccl_available()`` reports True.
    3. ``torch.xpu.is_available()`` and at least one XPU device is visible.
    """
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


def _xccl_factory(
    store: Store,
    rank: int,
    world_size: int,
    timeout: timedelta,
    device: torch.device,
    group_name: str,
) -> ProcessGroup:
    """Construct a 2-rank ProcessGroup with the XCCL backend bound to ``device``.

    Mirrors ``_gloo_factory``: builds the C++ backend directly from the
    supplied ``Store`` (no global state) and wraps it in a ``ProcessGroup``.
    Avoids ``dist.init_process_group`` because it sets process-global state
    and cannot be called more than once per process — each TorchStore
    transfer needs its own private PG.

    Avoids constructing ``ProcessGroupXCCL(store, rank, world_size)`` without
    ``Options``: with no ``global_ranks_in_group``/``group_name`` set, oneCCL
    spins up its own internal KVS server and the two ranks fail to meet
    (``Unregister Barrier request!``).

    ``group_name`` MUST be identical on both ranks. It namespaces oneCCL's
    internal KVS rendezvous; when it differs (e.g. each rank picks a fresh
    ``uuid4()``), each rank starts its own KVS and the first collective
    fails with ``Unregister Barrier request!``. We pass the shared
    ``store_key`` from the TCPStore handshake — both ranks see the same
    value through ``XcclTransportBuffer.__getstate__`` / pickling.

    ``device`` must name a specific XPU tile (``xpu:N``), not the bare
    ``xpu`` type — oneCCL needs to know which tile to drive.
    """
    from torch.distributed import ProcessGroupXCCL

    # Bind the device before init so oneCCL picks the right tile.
    if device.type == "xpu":
        torch.xpu.set_device(device)

    # Match torch's own backend init: scope the store under the device name
    # so multiple PGs can reuse a single underlying TCPStore without key
    # collisions.
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
    """Cache for XCCL process groups, keyed by store_key."""

    def __init__(self) -> None:
        self._process_groups: dict[str, ProcessGroup] = {}

    def put(self, store_key: str, pg: ProcessGroup) -> None:
        self._process_groups[store_key] = pg

    def get(self, store_key: str) -> ProcessGroup:
        return self._process_groups[store_key]

    def clear(self) -> None:
        self._process_groups.clear()


class XcclTransportBuffer(TransportBuffer):
    """Transport buffer using ``ProcessGroupXCCL`` for device-resident transfer.

    Mirrors ``GlooTransportBuffer``: a fresh 2-rank PG per (client,
    storage_volume) pair, coordinated via a ``TCPStore``. Differences
    from gloo:

    - Bound to ``xpu:0`` (or the local XPU rank if set), not ``cpu``.
    - ``send``/``recv`` keep tensors on XPU; no ``.cpu()`` staging.
    - Receive-side dest tensor is allocated on XPU.

    Prerequisites:
    - ``TORCHSTORE_XCCL_ENABLED=1`` (default).
    - ``torch.xpu.is_available()`` and ``dist.is_xccl_available()``.
    - oneCCL env block (``FI_PROVIDER=tcp``, ``CCL_ATL_TRANSPORT=ofi``,
      etc.) — see ``run_spmd_xpu.sh`` for the canonical set.
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

    @staticmethod
    def _xpu_device() -> torch.device:
        """Pick the XPU tile this process should use.

        Honors ``LOCAL_RANK`` if set, otherwise falls back to ``xpu:0``.
        oneCCL needs a concrete tile, not bare ``xpu``.
        """
        if "LOCAL_RANK" in os.environ:
            try:
                return torch.device(f"xpu:{int(os.environ['LOCAL_RANK'])}")
            except (TypeError, ValueError):
                pass
        return torch.device("xpu:0")

    def requires_handshake(self, requests: list[Request]) -> bool:
        """Skip handshake if we already have a cached PG for this volume."""
        volume_id = self.storage_volume_ref.volume_id
        if volume_id in _store_addrs:
            cached_addr = _store_addrs[volume_id]
            self.master_addr = cached_addr[0]
            self.master_port = cached_addr[1]
            self.store_key = cached_addr[2]
            return False
        return True

    async def _pre_handshake(self) -> None:
        """Create TCPStore, start client-side PG creation as a background task."""
        volume_id = self.storage_volume_ref.volume_id

        self.store_key = f"torchstore_xccl_{str(uuid.uuid4())[:8]}"
        self.master_addr = _get_hostname()
        self.master_port = portpicker.pick_unused_port()

        logger.info(
            f"[pid={os.getpid()}] Initiating xccl handshake with StorageVolume:[{volume_id}] "
            f"using TCPStore at {self.master_addr}:{self.master_port}"
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
        device = self._xpu_device()
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
        """Storage-side: build the PG as rank 1 and stash it in the context cache."""
        logger.info(
            f"[pid={os.getpid()}] Storage volume setting up xccl process group with TCPStore at "
            f"{self.master_addr}:{self.master_port}"
        )

        master_addr = self.master_addr
        master_port = self.master_port
        device = self._xpu_device()
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

        logger.info(
            f"Storage volume finished xccl process group setup for "
            f"store_key={self.store_key}"
        )
        return [None]

    async def _post_handshake(
        self,
        handshake_results: list[Any],
        requests: list[Request],
    ) -> None:
        """Client-side: await the background PG creation, cache it."""
        volume_id = self.storage_volume_ref.volume_id
        pg = await self._pg_task

        _store_addrs[volume_id] = (self.master_addr, self.master_port, self.store_key)
        self.storage_volume_ref.transport_context.get(XcclProcessGroupCache).put(
            self.store_key, pg
        )

        self._tcp_store = None
        self._pg_task = None

        logger.info(f"Finished xccl handshake with StorageVolume:[{volume_id}]")

    async def _pre_put_hook(self, requests: list[Request]) -> None:
        """Start sending tensor before put RPC."""
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
            self._send_tensor(
                tensor,
                self.storage_volume_ref.transport_context,
            )
        )

    async def handle_put_request(
        self,
        ctx: "TransportContext",
        entries: list[tuple[Request, Any]],
    ) -> list[Any]:
        """Storage-side put: receive into a fresh XPU tensor."""
        assert len(entries) == 1
        request, maybe_tensor = entries[0]

        if request.is_object:
            self.is_object = True
            return [request.objects]

        tensor = maybe_tensor
        if tensor is None:
            tensor = torch.empty(
                self.shape, dtype=self.dtype, device=self._xpu_device()
            )

        tensor = await self._receive_tensor(tensor, ctx)
        return [tensor]

    async def _pre_get_hook(self, requests: list[Request]) -> None:
        """Start receiving tensor before get RPC."""
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

        tensor = torch.empty(self.shape, dtype=self.dtype, device=self._xpu_device())

        self._recv_task = asyncio.create_task(
            self._receive_tensor(
                tensor,
                self.storage_volume_ref.transport_context,
            )
        )

    async def handle_get_request(
        self,
        ctx: "TransportContext",
        entries: list[tuple[Request, Any]],
    ) -> None:
        """Storage-side get: send the stored tensor back to the client."""
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
        """Receive a tensor on the bound XPU tile, no CPU staging.

        Caller may pass a tensor on a different device; we land on the
        local XPU regardless to keep the pg.recv on the registered backend.
        """
        target = self._xpu_device()
        if tensor.device != target:
            tensor = torch.empty(tensor.shape, dtype=tensor.dtype, device=target)

        pg = transport_context.get(XcclProcessGroupCache).get(self.store_key)
        my_rank = pg.rank()
        remote_rank = 1 - my_rank

        logger.debug(
            f"xccl recv: shape={tensor.shape} dtype={tensor.dtype} "
            f"device={tensor.device} from rank={remote_rank} "
            f"my_rank={my_rank} store_key={self.store_key}"
        )

        def do_recv():
            work = pg.recv([tensor], srcRank=remote_rank, tag=0)
            work.wait()
            torch.xpu.synchronize(target)

        await asyncio.to_thread(do_recv)
        logger.debug(f"xccl recv: completed shape={tensor.shape}")
        return tensor

    async def _send_tensor(
        self, tensor: torch.Tensor, transport_context: "TransportContext"
    ) -> None:
        """Send a tensor from XPU directly. No CPU staging."""
        # If the source tensor lives on a foreign device (CPU, another XPU
        # tile), move it onto the bound tile first; oneCCL needs the tensor
        # on the device the PG is registered against.
        target = self._xpu_device()
        if tensor.device != target:
            tensor = tensor.to(target)
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        pg = transport_context.get(XcclProcessGroupCache).get(self.store_key)
        my_rank = pg.rank()
        remote_rank = 1 - my_rank

        logger.debug(
            f"xccl send: shape={tensor.shape} dtype={tensor.dtype} "
            f"device={tensor.device} to rank={remote_rank} "
            f"my_rank={my_rank} store_key={self.store_key}"
        )

        def do_send():
            work = pg.send([tensor], dstRank=remote_rank, tag=0)
            work.wait()
            torch.xpu.synchronize(target)

        await asyncio.to_thread(do_send)
        logger.debug(f"xccl send: completed shape={tensor.shape}")

    async def drop(self) -> None:
        if self._send_task is not None:
            await self._send_task
            self._send_task = None
        self.is_object = False
        self.objects = None
