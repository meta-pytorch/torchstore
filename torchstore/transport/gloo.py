# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import atexit
import functools
import itertools
import socket
import threading
import time
import uuid
import weakref
from collections.abc import Awaitable, Callable, Iterable
from concurrent import futures
from datetime import timedelta
from logging import getLogger
from typing import Any, TYPE_CHECKING

import portpicker
import torch
from torch.distributed import ProcessGroup, ProcessGroupGloo, Store, TCPStore

from torchstore import native_calls
from torchstore.transport.buffers import TransportBuffer, TransportCache
from torchstore.transport.types import Request

if TYPE_CHECKING:
    from torchstore.logging import LatencyTracker
    from torchstore.transport.buffers import TransportContext
    from torchstore.transport.pipe import StorageVolumeRef
import os

import torch.distributed as dist

logger = getLogger(__name__)

# Global cache
_store_addrs: dict[
    str, tuple[str, int, str, bool]
] = {}  # volume_id -> (master_addr, master_port, store_key, tagged)
# Gives each tensor transfer to a volume that takes tags its own Gloo tag, so transfers in flight
# on one pair keep their own bytes. Tag 0 stays for clients and volumes before tags.
_tags = itertools.count()
# The largest tag torch accepts.
_MAX_TAG = 0x7FFFFFFF
# A volume's handshake reply when it moves each tensor on its buffer's tag.
_TAGS = "tags"
# The client group cache holding each cached pair, by store key, so another live client is refused.
_pair_owners: "weakref.WeakValueDictionary[str, GlooProcessGroupCache]" = (
    weakref.WeakValueDictionary()
)
# Each cached pair's client group by store key. A pair given up retires only once no client or
# transfer holds that group.
_pair_groups: "weakref.WeakValueDictionary[str, ProcessGroup]" = (
    weakref.WeakValueDictionary()
)
# The pairing in flight for each client group cache and volume ID. That client's other transfers wait
# for it and reuse its pair. Replaced in a forked child where no such pairing ever ends.
_pairing_now: "weakref.WeakKeyDictionary[GlooProcessGroupCache, dict[str, futures.Future[None]]]" = (
    weakref.WeakKeyDictionary()
)
# Told the volume ID and store key of each pair this process gives up, as add_retired_listener says.
_retired_listeners: list[Callable[[str, str], None]] = []
# Held while a volume's cached pair changes. Reentrant, as a GC may close an abandoned call's
# coroutine, which forgets its pair, on whatever thread holds the lock.
_cache_lock = threading.RLock()
# Volumes' pairings, each on a daemon thread, which exit waits for: one returning while the interpreter
# finalizes aborts it.
_pairings: set[threading.Thread] = set()
_pairings_lock = threading.Lock()
# When exit stops waiting for running pairings, set as exit begins. No pairing starts after that.
_exit_deadline: float | None = None
_EXIT_DRAIN_S = 5.0
# A volume's pairing gives up after this many Gloo timeouts and seconds, past torch's own bounds: its
# connect to a client that never listens retries with backoff past its timeout, by up to about 4s in
# probes at 1s and 2s timeouts, and making its group takes up to 1 more timeout.
_PAIRING_TIMEOUTS = 4
_PAIRING_SLACK_S = 10.0


TORCHSTORE_GLOO_ENABLED = os.environ.get("TORCHSTORE_GLOO_ENABLED", "1") == "1"
TORCHSTORE_GLOO_INIT_TIMEOUT = int(
    os.environ.get("TORCHSTORE_GLOO_INIT_TIMEOUT", "120")
)


class GlooPairInUseError(RuntimeError):
    """Another live client in this process holds the Gloo pair a tensor transfer needs."""


def gloo_available() -> bool:
    """Check if gloo transport is available and enabled.

    Returns True if:
    1. TORCHSTORE_GLOO_ENABLED environment variable is set to "1"
    2. torch.distributed is available
    3. gloo backend is available

    """
    return TORCHSTORE_GLOO_ENABLED and dist.is_gloo_available()


def _get_hostname() -> str:
    """Get the fully qualified hostname of the local machine.

    Uses FQDN to ensure hostname is resolvable from other nodes in multi-node setups.
    """
    return socket.getfqdn()


def _gloo_factory(
    store: Store,
    rank: int,
    world_size: int,
    timeout: timedelta,
    device: torch.device | None = None,
) -> ProcessGroup:
    """
    Create a ProcessGroup with Gloo backend.

    Based on:
    https://github.com/pytorch/pytorch/blob/92284fb2ff44f09a9c7df0d8cf6cac9903e376a4/torch/distributed/_dist2.py#L64
    """
    backend_class = ProcessGroupGloo(store, rank, world_size, timeout)
    backend_class._set_sequence_number_for_group()

    pg = ProcessGroup(store, rank, world_size)
    pg._set_default_backend(ProcessGroup.BackendType.GLOO)

    # Register devices
    if device is not None:
        pg._register_backend(device, ProcessGroup.BackendType.GLOO, backend_class)

    pg._register_backend(
        torch.device("cpu"), ProcessGroup.BackendType.GLOO, backend_class
    )
    if torch.cuda.is_available():
        pg._register_backend(
            torch.device("cuda"), ProcessGroup.BackendType.GLOO, backend_class
        )
    return pg


def _log_unawaited_failure(task: asyncio.Task, message: str) -> None:
    """Log at WARNING the failure of ``task``, which no call awaited, as ``message``."""
    try:
        error = task.exception()
    except asyncio.CancelledError:
        # Read all the same: from Python 3.11 the task holds its cancel and that traceback's frames until read.
        return
    if error is not None:
        # Read through exception(), as result() would add this frame, which holds the task, to
        # the traceback the task keeps.
        logger.warning(message, exc_info=error)


def _log_receive_failure(receive: asyncio.Task) -> None:
    """Log at WARNING the failure of a receive its get did not await."""
    _log_unawaited_failure(receive, "A Gloo receive its get did not await failed")


def _log_half_failure(half: asyncio.Task) -> None:
    """Log at WARNING the failure of the client's half of a pairing whose handshake call failed."""
    _log_unawaited_failure(
        half, "The client half of a Gloo pairing whose handshake failed then failed"
    )


class GlooProcessGroupCache(TransportCache):
    """Cache for Gloo process groups, keyed by store_key."""

    def __init__(self) -> None:
        self._process_groups: dict[str, ProcessGroup] = {}

    def put(self, store_key: str, pg: ProcessGroup) -> None:
        self._process_groups[store_key] = pg

    def get(self, store_key: str) -> ProcessGroup:
        return self._process_groups[store_key]

    def lookup(self, store_key: str) -> ProcessGroup | None:
        return self._process_groups.get(store_key)

    def pop(self, store_key: str) -> ProcessGroup | None:
        return self._process_groups.pop(store_key, None)

    def clear(self) -> None:
        self._process_groups.clear()


def _holder(store_key: str) -> GlooProcessGroupCache | None:
    """The live client group cache that still holds the pair under ``store_key``, if any."""
    owner = _pair_owners.get(store_key)
    return owner if owner is not None and owner.lookup(store_key) is not None else None


def _held_elsewhere(volume_id: str) -> GlooPairInUseError:
    return GlooPairInUseError(
        "another live client in this process holds the Gloo pair cached for volume ID"
        f" {volume_id!r}, so this client cannot move tensors to it. Drop that client and"
        " any error kept from its calls first"
    )


def add_retired_listener(listener: Callable[[str, str], None]) -> None:
    """Call ``listener(volume_id, store_key)`` for each pair this process gives up, once nothing here holds its group.

    Given up means forgotten, found unheld, displaced or refused, or a client half failing after its handshake returned.
    Listeners run on any thread, in GC finalizers or under the pair-cache lock: they must not block or take locks.
    """
    _retired_listeners.append(listener)


def _tell_retired(volume_id: str, store_key: str) -> None:
    for listener in _retired_listeners:
        listener(volume_id, store_key)


def _retire_once_unheld(
    volume_id: str, store_key: str, group: ProcessGroup | None
) -> None:
    """Tell the listeners of a pair once nothing holds its ``group``, at once if there is none."""
    if group is None:
        _tell_retired(volume_id, store_key)
    else:
        weakref.finalize(group, _tell_retired, volume_id, store_key).atexit = False


def cached_store_key(volume_id: str) -> str | None:
    """The store key of the pair this process caches for ``volume_id``, if any."""
    cached = _store_addrs.get(volume_id)
    return None if cached is None else cached[2]


def forget_pair(
    volume_id: str,
    store_key: str,
    transport_context: "TransportContext | None" = None,
) -> None:
    """Uncache ``volume_id``'s pair if this process caches it under ``store_key``. The next transfer then pairs anew.

    Given a client context, drop its group under ``store_key`` too. Without one the client holding the group keeps it
    and both halves until it drops it. Listeners hear of a pair this call uncached once nothing holds its group.
    """
    with _cache_lock:
        cached = _store_addrs.get(volume_id)
        # pop, as a GC closing an abandoned call here may have forgotten the pair already.
        forgot = (
            cached is not None
            and cached[2] == store_key
            and _store_addrs.pop(volume_id, None) is not None
        )
    if transport_context is not None:
        transport_context.get(GlooProcessGroupCache).pop(store_key)
    if forgot:
        _retire_once_unheld(volume_id, store_key, _pair_groups.get(store_key))


def _forget_pairs(
    volume_ids: str | Iterable[str], transport_context: "TransportContext | None" = None
) -> None:
    """Forget the pair this process caches for each of ``volume_ids``, as ``forget_pair`` does. For tests."""
    if isinstance(volume_ids, str):
        volume_ids = (volume_ids,)
    for volume_id in volume_ids:
        cached = _store_addrs.get(volume_id)
        if cached is not None:
            forget_pair(volume_id, cached[2], transport_context)


def forget_unheld_pair(volume_id: str) -> None:
    """Forget ``volume_id``'s cached pair unless a live client holds it.

    Retired listeners hear of it once no transfer holds its group.
    """
    with _cache_lock:
        cached = _store_addrs.get(volume_id)
        if cached is None or _holder(cached[2]) is not None:
            return
        _store_addrs.pop(volume_id, None)
    _retire_once_unheld(volume_id, cached[2], _pair_groups.get(cached[2]))


def _counted(pair: Callable[[], ProcessGroup], what: str) -> ProcessGroup:
    """Run ``pair`` unless exit has begun, counting its thread among those exit waits for."""
    current = threading.current_thread()
    with _pairings_lock:
        if _exit_deadline is not None:
            raise RuntimeError(f"{what} was not started: exiting")
        _pairings.add(current)
    try:
        return pair()
    finally:
        with _pairings_lock:
            _pairings.discard(current)


async def _pair_with_deadline(
    pair: Callable[[], ProcessGroup], what: str
) -> ProcessGroup:
    """Run a volume's ``pair`` on a daemon thread, raising ``TimeoutError`` past torch's own bounds.

    Torch's TCPStore client waits for its first reply with no deadline, and making the group can wait as long on
    a client frozen after the connect. Exit waits for the thread only until ``begin_exit``'s deadline.
    """
    deadline = (
        time.monotonic()
        + _PAIRING_TIMEOUTS * TORCHSTORE_GLOO_INIT_TIMEOUT
        + _PAIRING_SLACK_S
    )
    return await native_calls.run_with_deadline(
        functools.partial(_counted, pair, what),
        deadline,
        what,
        f"torchstore-gloo {what}",
    )


def begin_exit() -> float:
    """Refuse new pairings from now on and return when exit stops waiting for running ones.

    The first call sets that deadline ``_EXIT_DRAIN_S`` ahead. Later calls return the same one, which the exit
    hooks that wait on volumes' handlers and pairings share.
    """
    global _exit_deadline
    with _pairings_lock:
        if _exit_deadline is None:
            _exit_deadline = time.monotonic() + _EXIT_DRAIN_S
        return _exit_deadline


@atexit.register
def _drain_pairings() -> None:
    """Refuse new pairings, then give live ones until the exit deadline to return before finalization."""
    deadline = begin_exit()
    with _pairings_lock:
        live = list(_pairings)
    for thread in live:
        thread.join(max(deadline - time.monotonic(), 0.0))
    running = sorted(thread.name for thread in live if thread.is_alive())
    if running:
        logger.warning(
            "Exiting with %d Gloo pairings still running after %gs, which may abort the"
            " process: %s",
            len(running),
            _EXIT_DRAIN_S,
            ", ".join(running),
        )


def _after_fork_in_child() -> None:
    # A thread that held a lock or ran a pairing at the fork never releases the child's copy. The cached
    # pairs are the parent's. None retires because their volume halves still serve the parent's clients.
    # A child forked during or after the parent's exit hooks is a live process.
    global _cache_lock, _pairings_lock, _pairing_now, _exit_deadline
    _cache_lock = threading.RLock()
    _pairings_lock = threading.Lock()
    _pairing_now = weakref.WeakKeyDictionary()
    _exit_deadline = None
    _store_addrs.clear()
    _pair_owners.clear()
    _pair_groups.clear()


os.register_at_fork(after_in_child=_after_fork_in_child)


class GlooTransportBuffer(TransportBuffer):
    """Transport buffer implementation using PyTorch distributed (gloo backend) for tensor transfer.

    This buffer creates a dedicated 2-process gloo process group between the client
    and storage volume for each connection using TCPStore for coordination.

    Prerequisites:
    - TORCHSTORE_GLOO_ENABLED=1 environment variable must be set
    - Network connectivity between client and storage (for TCPStore)
    """

    supports_inplace_resharding = False
    # The tag of the buffer's current tensor transfer, 0 for an object or a pair without tags, and the
    # group it holds. Class defaults, as buffers that clients before tags pickle carry neither field.
    tag: int = 0
    _pg: ProcessGroup | None = None

    def __init__(self, storage_volume_ref: "StorageVolumeRef") -> None:
        super().__init__(storage_volume_ref)

        # Tensor metadata
        self.shape: torch.Size | None = None
        self.dtype: torch.dtype | None = None

        # Process group coordination
        self.master_addr: str | None = None
        self.master_port: int | None = None
        self.store_key: str | None = None  # Unique key for this connection
        # The client's group and whether its volume takes tags, held from when the transfer finds
        # its pair, so a dropped pair spares a transfer already under way.
        self._pg: ProcessGroup | None = None
        self._takes_tags = False
        # Set once the client's half of the current transfer has begun, before it can post.
        self._half_started = False

        # Set by the client when it moves an object, which posts no tensor half, and by the volume
        # when it answers a get in band in objects, which clients before tags read too.
        self.is_object: bool = False
        self.objects: Any = None  # A get's value when it comes back in band

        # TCPStore reference (kept alive until PG is created)
        self._tcp_store: TCPStore | None = None
        self._pg_task: asyncio.Task | None = None  # Background task for PG creation
        self._send_task: asyncio.Task | None = (
            None  # Background task for sending tensor
        )
        self._recv_task: asyncio.Task | None = (
            None  # Background task for receiving tensor
        )

    def requires_handshake(self, requests: list[Request]) -> bool:
        """Return whether to pair: never for objects alone, else unless this buffer's context holds its volume's cached pair.

        A cached pair that no live client holds is forgotten first, as ``forget_unheld_pair`` does.

        Raises:
            GlooPairInUseError: If a request not marked as an object needs the pair while another live
                client in this process holds it.
        """
        volume_id = self.storage_volume_ref.volume_id
        # One buffer moves a call's requests in turn, so the last request's pair is not this one's.
        self.master_addr = self.master_port = self.store_key = None
        if all(request.is_object for request in requests):
            # Objects travel in the call itself and never touch the pair.
            return False

        # Read once and pair anew if either half is gone: another thread may forget the pair.
        cached_addr = _store_addrs.get(volume_id)
        if cached_addr is None:
            return True
        context = self.storage_volume_ref.transport_context
        pg = context.get(GlooProcessGroupCache).lookup(cached_addr[2])
        if pg is None:
            if _holder(cached_addr[2]) is not None:
                # Pairing anew would strand that client's pair on both sides at every switch.
                raise _held_elsewhere(volume_id)
            forget_unheld_pair(volume_id)
            return True
        self.master_addr, self.master_port, self.store_key = cached_addr[:3]
        self._pg, self._takes_tags = pg, cached_addr[3]
        return False

    async def perform_handshake(
        self,
        requests: list[Request],
        meta_requests: list[Request],
        latency_tracker: "LatencyTracker | None" = None,
    ) -> None:
        """Pair with the volume or wait for this client's pairing with it in flight and reuse that pair.

        A transfer that waited pairs once that pairing ends if it left this client no pair to reuse. Transfers through a
        carrier volume proxy to a volume whose handshakes never return so fail one call timeout after another.
        """
        context = self.storage_volume_ref.transport_context
        volume_id = self.storage_volume_ref.volume_id
        while True:
            with _cache_lock:
                pairings = _pairing_now.setdefault(
                    context.get(GlooProcessGroupCache), {}
                )
                running = pairings.get(volume_id)
                if running is None:
                    if not self.requires_handshake(requests):
                        return
                    mine = pairings[volume_id] = futures.Future()
                    # wrap_future cancels a pending future whose waiter is cancelled but never a running one.
                    mine.set_running_or_notify_cancel()
                    break
            await asyncio.wrap_future(running)
        try:
            # Sent without requests: the volume's half never reads them, and a store that checked them here would
            # refuse before pairing and leave this client's half waiting. The put or get checks them and ends the
            # half of a transfer it refuses.
            await super().perform_handshake(requests, [], latency_tracker)
        finally:
            with _cache_lock:
                del pairings[volume_id]
            mine.set_result(None)

    async def _pre_handshake(self) -> None:
        """Prepare for handshake. Create TCPStore and start PG creation on
        client side (rank 0).

        The TCPStore is created here so it's ready to accept connections
        when the handshake RPC reaches the storage volume. The ProcessGroup
        creation is started as a background task so it runs concurrently with
        the handshake RPC, allowing both sides to rendezvous.
        """
        volume_id = self.storage_volume_ref.volume_id

        # Generate unique store key and find free port
        self.store_key = f"torchstore_gloo_{uuid.uuid4().hex}"
        self.master_addr = _get_hostname()
        self.master_port = portpicker.pick_unused_port()

        logger.info(
            f"Initiating gloo handshake with StorageVolume:[{volume_id}] "
            f"using TCPStore at {self.master_addr}:{self.master_port}"
        )

        # Create TCPStore as master (non-blocking with wait_for_workers=False)
        self._tcp_store = TCPStore(
            host_name=self.master_addr,
            port=self.master_port,
            world_size=2,
            is_master=True,
            timeout=timedelta(seconds=TORCHSTORE_GLOO_INIT_TIMEOUT),
            wait_for_workers=False,
        )

        # Start PG creation in background so it runs concurrently with handshake RPC
        tcp_store = self._tcp_store

        def create_pg():
            return _gloo_factory(
                store=tcp_store,
                rank=0,
                world_size=2,
                timeout=timedelta(seconds=TORCHSTORE_GLOO_INIT_TIMEOUT),
                device=torch.device("cpu"),
            )

        self._pg_task = asyncio.create_task(asyncio.to_thread(create_pg))

    def __getstate__(self) -> dict[str, Any]:
        """Serialize state, excluding non-serializable fields."""
        state = super().__getstate__()
        state["_tcp_store"] = None
        state["_pg_task"] = None
        state["_send_task"] = None
        state["_recv_task"] = None
        state["_pg"] = None
        return state

    async def recv_handshake(
        self,
        ctx: "TransportContext",
        entries: list[tuple[Request, Any]],
    ) -> list[str]:
        """Called on storage volume side to set up the process group.

        Creates TCPStore and ProcessGroup on storage side (rank 1).

        transport_context: the TransportContext from the StorageVolume to which the pg will be added

        """
        logger.info(
            f"Storage volume setting up gloo process group with TCPStore at "
            f"{self.master_addr}:{self.master_port}"
        )

        # Create TCPStore and ProcessGroup on storage side (rank 1, worker), off the event loop
        # and under a deadline of their own, as torch's client can wait for a frozen client forever.
        master_addr = self.master_addr
        master_port = self.master_port
        timeout = timedelta(seconds=TORCHSTORE_GLOO_INIT_TIMEOUT)

        def pair() -> ProcessGroup:
            tcp_store = TCPStore(
                host_name=master_addr,
                port=master_port,
                world_size=2,
                is_master=False,
                timeout=timeout,
            )
            return _gloo_factory(
                store=tcp_store,
                rank=1,
                world_size=2,
                timeout=timeout,
                device=torch.device("cpu"),
            )

        pg = await _pair_with_deadline(
            pair, f"pairing through the TCPStore at {master_addr}:{master_port}"
        )

        # Cache the process group
        ctx.get(GlooProcessGroupCache).put(self.store_key, pg)

        logger.info(
            f"Storage volume finished gloo process group setup for store_key={self.store_key}"
        )

        return [_TAGS]

    async def _post_handshake(
        self,
        handshake_results: list[Any],
        requests: list[Request],
    ) -> None:
        """Await the client's half that ``_pre_handshake`` started, then cache the pair.

        Volumes before tags answer ``[None]``, so their pair moves every tensor on tag 0.

        Raises:
            GlooPairInUseError: If another live client in this process cached its pair to this
                volume ID first.
        """
        volume_id = self.storage_volume_ref.volume_id

        # Cleared before the await, as drop() treats a half still set as one no handshake awaited.
        half, self._pg_task = self._pg_task, None
        try:
            pg = await half
        except BaseException:
            # The volume made its half as the handshake returned, and nothing here reaches it now.
            _tell_retired(volume_id, self.store_key)
            raise
        finally:
            # A failed half's traceback holds this frame, which would tie the task into a cycle.
            del half

        # Cache the process group, its owner, then the connection info that names it, unless
        # another live client cached its pair to this volume ID first.
        takes_tags = _TAGS in handshake_results
        groups = self.storage_volume_ref.transport_context.get(GlooProcessGroupCache)
        with _cache_lock:
            displaced = _store_addrs.get(volume_id)
            taken = displaced is not None and _holder(displaced[2]) is not None
            if not taken:
                groups.put(self.store_key, pg)
                _pair_owners[self.store_key] = groups
                _pair_groups[self.store_key] = pg
                _store_addrs[volume_id] = (
                    self.master_addr,
                    self.master_port,
                    self.store_key,
                    takes_tags,
                )
        if taken:
            # The pair stays with that client and this new one retires at once. A kept error's
            # traceback holds this frame, which must not keep the new group.
            del pg
            _tell_retired(volume_id, self.store_key)
            raise _held_elsewhere(volume_id)
        self._pg, self._takes_tags = pg, takes_tags
        if displaced is not None:
            # Left by a client that is gone or lost it: a client pairs only while it holds no pair.
            # It retires once no transfer holds its group.
            _retire_once_unheld(volume_id, displaced[2], _pair_groups.get(displaced[2]))

        logger.info(f"Finished gloo handshake with StorageVolume:[{volume_id}]")

    def _draw_tag(self) -> int:
        """Return a tag of its own for a tensor transfer if the pair takes tags, else 0."""
        if not self._takes_tags:
            return 0
        return next(_tags) % _MAX_TAG + 1

    async def _pre_put_hook(self, requests: list[Request]) -> None:
        """Start sending tensor before put RPC.

        Called after handshake completes, before put.call().
        Starts the send as a background task so it runs concurrently
        with the storage volume's recv in handle_put_request.
        """
        assert len(requests) == 1
        request = requests[0]
        # Cleared for each transfer, as the volume ends a half on any tag but 0 and an object has none.
        self.tag = 0

        # Check if this is an object (non-tensor) PUT
        if request.is_object:
            self.is_object = True
            return

        # Extract tensor from request
        if request.tensor_val is None:
            return

        tensor = request.tensor_val
        self.shape = tensor.shape
        self.dtype = tensor.dtype
        self.tag = self._draw_tag()

        # Start send in background - will run concurrently with put RPC
        self._send_task = asyncio.create_task(
            self._run_half(
                self._send_tensor,
                tensor,
                self.storage_volume_ref.transport_context,
                self.tag,
            )
        )

    async def handle_put_request(
        self,
        ctx: "TransportContext",
        entries: list[tuple[Request, Any]],
    ) -> list[Any]:
        """Called by storage volume. Receive tensor from client via gloo process group."""
        assert len(entries) == 1
        request, maybe_tensor = entries[0]

        if request.is_object:
            self.is_object = True
            return [request.objects]

        # Allocate destination tensor if needed
        tensor = maybe_tensor
        if tensor is None:
            tensor = torch.empty(
                self.shape, dtype=self.dtype, device=torch.device("cpu")
            )

        # Receive tensor from client via gloo
        tensor = await self._receive_tensor(tensor, ctx, self.tag)

        return [tensor]

    async def handle_refused_put(
        self,
        ctx: "TransportContext",
        requests: list[Request],
    ) -> None:
        """Called by storage volume. Receive the client's tensor into scratch so its send ends.

        Only for a send on a tag of its own, as on tag 0 the scratch could take another transfer's send.
        The scratch takes the tensor's full size, since a smaller receive would abort this process.
        """
        if self.tag:
            scratch = torch.empty(
                self.shape, dtype=self.dtype, device=torch.device("cpu")
            )
            await self._receive_tensor(scratch, ctx, self.tag)

    async def _pre_get_hook(self, requests: list[Request]) -> None:
        """Start receiving tensor before get RPC.

        Called after handshake completes, before get.call().
        Starts the recv as a background task so it runs concurrently
        with the storage volume's send in handle_get_request.
        """
        assert len(requests) == 1
        request = requests[0]
        # Cleared for each transfer, as the volume ends a half on any tag but 0 and an object has none.
        self.tag = 0
        if request.is_object:
            # A get marked as an object holds no pair, so the volume answers in band with whatever
            # the key holds by then.
            self.is_object = True
            return
        # TODO: support in place get with receiving directly to request.tensor_val

        # Fetch metadata from storage volume
        meta = (
            await self.storage_volume_ref.volume.get_meta.call_one(
                [request.meta_only()]
            )
        )[0]

        # If request specifies a tensor slice, use its shape
        if request.tensor_slice is not None:
            self.shape = torch.Size(request.tensor_slice.local_shape)
            self.dtype = meta[1]
        else:
            if isinstance(meta, str) or meta is None:
                # It's an object, not a tensor
                self.is_object = True
                return
            # meta is (shape, dtype)
            self.shape = meta[0]
            self.dtype = meta[1]

        tensor = torch.empty(self.shape, dtype=self.dtype, device=torch.device("cpu"))
        self.tag = self._draw_tag()

        # Start recv in background - will run concurrently with get RPC
        # The storage volume will send the tensor in handle_get_request
        self._recv_task = asyncio.create_task(
            self._run_half(
                self._receive_tensor,
                tensor,
                self.storage_volume_ref.transport_context,
                self.tag,
            )
        )

    async def handle_get_request(
        self,
        ctx: "TransportContext",
        entries: list[tuple[Request, Any]],
    ) -> None:
        """Called by storage volume. Send tensor to client via gloo process group."""
        assert len(entries) == 1
        _, data = entries[0]
        # A get the client took for an object posted no receive, so its data goes in band.
        if not self.is_object:
            if (
                isinstance(data, torch.Tensor)
                and data.shape == self.shape
                and data.dtype == self.dtype
            ):
                # Send the tensor to client via gloo
                await self._send_tensor(data, ctx, self.tag)
                return
            # The data does not fit the client's receive, as when the key changed after the client's
            # metadata fetch, so it goes in band and an empty send ends the receive. A client before
            # tags shares tag 0 among its transfers, so it gets that send only where earlier volumes
            # sent the data itself.
            if self.tag or isinstance(data, torch.Tensor):
                try:
                    await self._end_client_receive(ctx)
                except Exception:
                    logger.warning(
                        "Ending a Gloo get's receive failed, and the get's value still goes back in band",
                        exc_info=True,
                    )
        self.is_object = True
        self.objects = data

    async def handle_refused_get(
        self,
        ctx: "TransportContext",
        requests: list[Request],
    ) -> None:
        """Called by storage volume. End the client's receive with an empty send.

        Only for a receive on a tag of its own, as on tag 0 the send could reach another transfer's receive.
        """
        if self.tag:
            await self._end_client_receive(ctx)

    async def _end_client_receive(self, ctx: "TransportContext") -> None:
        # An empty send ends a receive of any size and can never be larger than the receive it meets.
        await self._send_tensor(torch.empty(0), ctx, self.tag)

    async def _handle_storage_volume_response(
        self, requests: list[Request], transport_buffer: "TransportBuffer"
    ) -> list[Any]:
        """Process the response from storage volume after get.

        The tensor data was received via gloo in the background recv task.
        """
        if transport_buffer.is_object:
            return [transport_buffer.objects]

        # Wait for the recv to complete
        if self._recv_task is not None:
            # Cleared before the await, as drop() treats a receive still set as one no get awaited.
            receive, self._recv_task = self._recv_task, None
            try:
                tensor = await receive
            finally:
                # A failed receive's traceback holds this frame, which would tie the task into a cycle.
                del receive
            if tensor is None:
                raise RuntimeError(
                    f"receive_tensor returned None (is_object={self.is_object}, "
                    f"shape={self.shape}, dtype={self.dtype})"
                )
            return [tensor]

        raise RuntimeError(f"No recv task available (is_object={self.is_object})")

    async def _receive_tensor(
        self, tensor: torch.Tensor, transport_context: "TransportContext", tag: int
    ) -> torch.Tensor:
        """Receive tensor via gloo. Waits for recv to complete before returning.

        Args:
            tensor: Pre-allocated destination tensor to receive into.
            transport_context: Transport context containing the process group.
            tag: The Gloo tag of the transfer.

        Returns:
            The received tensor (same object as input, with data overwritten).
        """
        # Gloo TCP transport requires CPU tensors for recv
        if tensor.device.type != "cpu":
            tensor = torch.zeros(
                tensor.shape, dtype=tensor.dtype, device=torch.device("cpu")
            )

        # Get process group from transport context
        pg = self._group(transport_context)

        # Determine remote rank based on our rank in the PG
        my_rank = pg.rank()
        remote_rank = 1 - my_rank

        logger.debug(
            f"receive_tensor: receiving tensor shape={tensor.shape} dtype={tensor.dtype} "
            f"from rank {remote_rank} (my_rank={my_rank}, store_key={self.store_key})"
        )

        # Run recv in thread and wait for completion
        def do_recv():
            work = pg.recv([tensor], srcRank=remote_rank, tag=tag)
            work.wait()

        await asyncio.to_thread(do_recv)

        logger.debug(f"receive_tensor: completed receiving tensor shape={tensor.shape}")

        return tensor

    async def _send_tensor(
        self, tensor: torch.Tensor, transport_context: "TransportContext", tag: int
    ) -> None:
        """Send tensor via gloo. Waits for send to complete before returning.

        Args:
            tensor: Tensor to send.
            transport_context: Transport context containing the process group.
            tag: The Gloo tag of the transfer.
        """
        # Gloo TCP transport requires CPU tensors
        if tensor.device.type != "cpu":
            tensor = tensor.cpu()

        # Gloo requires contiguous tensors (slices may be non-contiguous views)
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        # Get process group from transport context
        pg = self._group(transport_context)

        # Determine remote rank based on our rank in the PG
        my_rank = pg.rank()
        remote_rank = 1 - my_rank

        logger.debug(
            f"send_tensor: sending tensor shape={tensor.shape} dtype={tensor.dtype} "
            f"to rank {remote_rank} (my_rank={my_rank}, store_key={self.store_key})"
        )

        # Run send in thread and wait for completion
        def do_send():
            work = pg.send([tensor], dstRank=remote_rank, tag=tag)
            work.wait()

        await asyncio.to_thread(do_send)

        logger.debug(f"send_tensor: completed sending tensor shape={tensor.shape}")

    async def _run_half(self, half: Callable[..., Awaitable[Any]], *args: Any) -> Any:
        """Run the client's half of the current transfer, noting that it began before it can post."""
        self._half_started = True
        return await half(*args)

    def _group(self, transport_context: "TransportContext") -> ProcessGroup:
        """The process group this transfer holds, else the one its context caches, as on a volume."""
        if self._pg is not None:
            return self._pg
        return transport_context.get(GlooProcessGroupCache).get(self.store_key)

    async def drop(self) -> None:
        """Clean up resources held by this buffer."""
        # Taken out first, so a later drop never awaits or re-raises a failed send.
        send_task, self._send_task = self._send_task, None
        receive, self._recv_task = self._recv_task, None
        half, self._pg_task = self._pg_task, None
        if not self._half_started:
            # The call failed before the loop ran the transfer's half, which so never posts.
            for task in (send_task, receive):
                if task is not None:
                    task.cancel()
            send_task = receive = None
        try:
            if send_task is not None:
                await send_task
        finally:
            # A failed send's traceback holds this frame, which would tie the task into a cycle.
            del send_task
            # A get whose call raised or was cancelled, or whose value came in band, did not await its
            # receive, which may wait out Gloo's timeout, so the receive's failure is logged once it ends.
            if receive is not None:
                receive.add_done_callback(_log_receive_failure)
            # Likewise the client's half of a pairing whose handshake call failed.
            if half is not None:
                half.add_done_callback(_log_half_failure)
            del receive, half
            self._pg = self._tcp_store = None
            self._half_started = False
            self.is_object = False
            self.objects = None
