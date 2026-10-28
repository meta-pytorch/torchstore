# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Direct RDMA weight sync — zero-copy weight transfer between actors.

Instead of routing tensor data through a StorageVolume (two hops),
this module lets the destination read directly from the source's
registered memory via one-sided RDMA reads (one hop).
Reads use NIXL when ``TORCHSTORE_NIXL_ENABLED=1``, else Monarch RDMA.

TorchStore is still used for metadata exchange (RDMA handle registration),
but the actual weight data travels directly source → destination.

Typical usage (RL trainer → generator weight sync):

    # Source side (trainer, called once at setup):
    source = DirectWeightSyncSource()
    handles = source.register(model.state_dict(), rank=dist.get_rank())
    await store.publish_rdma_handles(key, handles)

    # After each optimizer.step():
    source.refresh()

    # Destination side (generator):
    dest = DirectWeightSyncDest()
    all_handles = await store.get_rdma_handles(key)
    await dest.pull(all_handles, model.state_dict())
"""

import asyncio
import logging
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import torch

from torchstore.transport.nixl import (
    _NixlTransferOperation,
    _transfer_tensors,
    nixl_available,
    NixlAgentCache,
)
from torchstore.transport.types import Request, TensorSlice
from torchstore.utils import get_slice_intersection, get_slice_numel, to_byte_view

logger = logging.getLogger(__name__)

RDMA_KEY_PREFIX = "policy_rdma"


@dataclass(frozen=True)
class NixlSourceAgent:
    """A source rank's NIXL agent.

    One instance is shared by all of a rank's handles, so pickling the
    rank's handle dict sends the metadata once.
    """

    name: str
    metadata: bytes


@dataclass(frozen=True)
class NixlBuffer:
    """Serialized NIXL descriptors of a registered source buffer."""

    agent: NixlSourceAgent
    descriptors: bytes


@dataclass
class RDMAWeightHandle:
    """Serializable handle for direct RDMA access to a weight shard.

    The ``rdma_buffer`` field is a :class:`NixlBuffer` or a Monarch
    ``RDMABuffer``; both pickle across Monarch actors. The destination
    reads from it with one-sided RDMA reads.
    """

    rdma_buffer: object | None  # NixlBuffer | monarch.rdma.RDMABuffer
    tensor_slice: TensorSlice  # shard position in the global tensor
    source_rank: int


def _request_to_slice(req: Request, param: torch.Tensor) -> TensorSlice:
    """Get the TensorSlice from a Request, synthesizing one for plain tensors."""
    if req.tensor_slice is not None:
        return req.tensor_slice
    # Plain (non-sharded) tensor: covers the entire global shape at offset 0
    shape = tuple(param.shape)
    ndim = len(shape)
    return TensorSlice(
        offsets=tuple(0 for _ in range(ndim)),
        coordinates=tuple(0 for _ in range(ndim)),
        global_shape=shape,
        local_shape=shape,
        mesh_shape=tuple(1 for _ in range(ndim)),
    )


def group_handles(
    per_rank: Iterable[Mapping[str, RDMAWeightHandle]],
) -> dict[str, list[RDMAWeightHandle]]:
    """Merge per-rank handle dicts into per-param lists of handles."""
    all_handles: dict[str, list[RDMAWeightHandle]] = defaultdict(list)
    for rank_handles in per_rank:
        for name, handle in rank_handles.items():
            all_handles[name].append(handle)
    return all_handles


# ---------------------------------------------------------------------------
# Source side (trainer)
# ---------------------------------------------------------------------------


class DirectWeightSyncSource:
    """Manages RDMA handle registration and staging buffer refresh.

    RDMA handles point directly at param memory (true zero-copy). The
    optimizer updates weights in-place, so RDMA reads always see fresh
    values without any copy.

    When ``transfer_dtype`` is set, a staging buffer is allocated per
    param in the target dtype. Call :meth:`refresh` after each optimizer
    step to re-cast from the source tensors into the staging buffers.
    """

    def __init__(self) -> None:
        self._handles: dict[str, RDMAWeightHandle] = {}
        # name → (staging_buffer, source_local_tensor)
        self._staging: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._nixl: NixlAgentCache | None = None
        # Registered views; NIXL descriptors don't keep their memory alive.
        self._nixl_views: list[torch.Tensor] = []

    def register(
        self,
        state_dict: dict[str, torch.Tensor],
        rank: int,
        transfer_dtype: torch.dtype | None = None,
    ) -> dict[str, RDMAWeightHandle]:
        """Create RDMA handles for every parameter in *state_dict*.

        Args:
            state_dict: Model state dict to register.
            rank: This process's rank.
            transfer_dtype: If set, cast all params to this dtype for
                transfer. A staging buffer is allocated for every param
                so that :meth:`refresh` can re-cast from the original
                source tensors after optimizer updates.


        Returns a dict of serializable :class:`RDMAWeightHandle` objects
        that can be stored in TorchStore or sent to the destination actor.
        """
        use_nixl = nixl_available()
        if use_nixl:
            if self._nixl is None:
                self._nixl = NixlAgentCache()
        else:
            from monarch.rdma import RDMABuffer

        handles: dict[str, RDMAWeightHandle] = {}
        nixl_views: dict[str, torch.Tensor] = {}
        num_staged = 0

        for name, param in state_dict.items():
            # Use TorchStore's Request to extract local tensor + shard metadata
            req = Request.from_any(name, param)
            local_tensor = req.tensor_val
            tensor_slice = _request_to_slice(req, param)

            if transfer_dtype is not None:
                # Always stage: cast to transfer dtype for RDMA transfer.
                # refresh() will re-copy from the original source tensor.
                buf_tensor = local_tensor.to(transfer_dtype).contiguous()
                self._staging[name] = (buf_tensor, local_tensor)
                num_staged += 1
            else:
                assert (
                    local_tensor.is_contiguous()
                ), f"Expected contiguous tensor for key={name}, strides={local_tensor.stride()}"
                buf_tensor = local_tensor

            rdma_buf = None
            if buf_tensor.numel() > 0:
                byte_view = to_byte_view(buf_tensor)
                if use_nixl:
                    self._nixl.register(byte_view)
                    nixl_views[name] = byte_view
                else:
                    rdma_buf = RDMABuffer(byte_view)

            handles[name] = RDMAWeightHandle(
                rdma_buffer=rdma_buf,
                tensor_slice=tensor_slice,
                source_rank=rank,
            )

        if nixl_views:
            # The metadata carries the registrations' remote keys, so export it
            # only after registering every buffer.
            agent = self._nixl.agent
            source_agent = NixlSourceAgent(agent.name, agent.get_agent_metadata())
            for name, byte_view in nixl_views.items():
                handles[name].rdma_buffer = NixlBuffer(
                    source_agent,
                    agent.get_serialized_descs(agent.get_xfer_descs(byte_view)),
                )
            self._nixl_views.extend(nixl_views.values())

        self._handles = handles
        logger.info(
            f"Registered {len(handles)} RDMA handles "
            f"({num_staged} staged, {len(handles) - num_staged} direct)"
        )
        return handles

    def refresh(self) -> int:
        """Re-cast source params into their dtype-cast staging buffers.

        Only needed when ``transfer_dtype`` was set during registration.
        Without dtype casting, RDMA handles point directly at param memory
        that the optimizer updates in-place, so no refresh is needed.

        Returns the number of staging buffers refreshed.
        """
        for staging_buf, source_tensor in self._staging.values():
            staging_buf.copy_(source_tensor)
        return len(self._staging)

    async def cleanup(self) -> None:
        """Release all RDMA registrations."""
        for handle in self._handles.values():
            if handle.rdma_buffer is not None and not isinstance(
                handle.rdma_buffer, NixlBuffer
            ):
                await handle.rdma_buffer.drop()
        if self._nixl is not None:
            self._nixl.clear()
        self._handles.clear()
        self._staging.clear()
        self._nixl_views.clear()


# ---------------------------------------------------------------------------
# Destination side (generator)
# ---------------------------------------------------------------------------


@dataclass
class _TransferOp:
    """A single pre-computed RDMA read operation.

    Fields:
        rdma_buffer: Source RDMA handle to read from.
        dest_byte_view: Byte view of the buffer that RDMA writes into.
        dest_tensor: Destination param tensor for post-read copy.
            None when RDMA writes directly into param memory (zero-copy).
        recv_buffer: Temporary buffer for RDMA read when we can't write
            directly into dest. None for zero-copy ops.
        src_slices: Slices into recv_buffer for the overlap region.
            None for exact-match ops (copy entire buffer).
        dest_slices: Slices into dest_tensor for the overlap region.
            None for exact-match ops (copy entire buffer).
    """

    rdma_buffer: object
    dest_byte_view: torch.Tensor
    dest_tensor: torch.Tensor | None = None
    recv_buffer: torch.Tensor | None = None
    src_slices: tuple[slice, ...] | None = None
    dest_slices: tuple[slice, ...] | None = None


def _plan_signature(
    all_handles: dict[str, list[RDMAWeightHandle]],
    dest_state_dict: dict[str, torch.Tensor],
) -> tuple:
    """Identify the destination memory a plan was built for."""
    dests = []
    for name, param in dest_state_dict.items():
        if name not in all_handles:
            continue
        local = Request.from_any(name, param).tensor_val
        dests.append(
            (
                name,
                local.data_ptr(),
                local.untyped_storage().nbytes(),
                local.dtype,
                local.shape,
                local.stride(),
            )
        )
    return tuple(dests)


class DirectWeightSyncDest:
    """Pulls weights via direct RDMA reads from source handles.

    On the first :meth:`pull`, a transfer plan is computed and cached.
    Subsequent pulls into the same destination memory skip all metadata
    work and just execute RDMA reads + post-read copies. The plan is
    rebuilt when the handles or the destination tensors change, e.g. when
    pulling a subset of the state dict or into reallocated storage.
    Receive buffers are allocated on GPU (same device as dest) to avoid
    CPU-GPU copies.
    """

    def __init__(self) -> None:
        self._plan: list[_TransferOp] | None = None
        self._plan_handles: dict[str, list[RDMAWeightHandle]] | None = None
        self._plan_signature: tuple | None = None
        self._nixl: NixlAgentCache | None = None

    async def _read_nixl(
        self,
        source_agent: NixlSourceAgent,
        transfers: list[tuple[torch.Tensor, bytes]],
    ) -> None:
        """Read every buffer of one source agent in a batched NIXL transfer."""
        if self._nixl is None:
            self._nixl = NixlAgentCache()
        remote_agent = self._nixl.add_remote_agent(
            source_agent.name, source_agent.metadata
        )
        await _transfer_tensors(
            self._nixl, _NixlTransferOperation.READ, remote_agent, transfers
        )

    def _build_plan(
        self,
        all_handles: dict[str, list[RDMAWeightHandle]],
        dest_state_dict: dict[str, torch.Tensor],
    ) -> list[_TransferOp]:
        """Build the transfer plan.

        For each destination parameter, finds overlapping source shards
        and creates a _TransferOp describing how to read from each one.

        Two cases per (source, dest) pair:
          1. Exact match -> zero-copy RDMA directly into param memory
          2. Partial overlap (resharding) -> RDMA into recv buf, slice copy
        """
        ops: list[_TransferOp] = []

        for name, param in dest_state_dict.items():
            handles = all_handles.get(name)
            if not handles:
                continue

            # Extract dest shard metadata using Request
            dest_req = Request.from_any(name, param)
            dest_tensor = dest_req.tensor_val
            dest_slice = _request_to_slice(dest_req, param)

            # Deduplicate: for replicated params, all trainer ranks have
            # identical shards. Only read from the first matching one.
            seen: set[tuple] = set()
            covered = 0

            for handle in handles:
                # Use get_slice_intersection to find overlap
                intersection = get_slice_intersection(handle.tensor_slice, dest_slice)
                if intersection is None:
                    continue
                if 0 in intersection.local_shape:
                    continue
                if handle.rdma_buffer is None:
                    raise RuntimeError(
                        f"Missing RDMA buffer for nonempty tensor slice {name}"
                    )

                # Skip duplicate regions (replicated params)
                region_key = (intersection.offsets, intersection.local_shape)
                if region_key in seen:
                    continue
                seen.add(region_key)
                covered += get_slice_numel(intersection)

                is_exact = (
                    handle.tensor_slice.offsets == dest_slice.offsets
                    and handle.tensor_slice.local_shape == dest_slice.local_shape
                )
                if is_exact:
                    assert dest_tensor.is_contiguous(), (
                        f"Expected contiguous dest tensor for "
                        f"key={name}, strides={dest_tensor.stride()}"
                    )
                    # Zero-copy: RDMA directly into model parameter
                    ops.append(
                        _TransferOp(
                            rdma_buffer=handle.rdma_buffer,
                            dest_byte_view=to_byte_view(dest_tensor),
                        )
                    )
                else:
                    # Partial overlap (different TP), need resharding.
                    # Read the full source shard, then copy the overlap region.
                    recv = torch.empty(
                        handle.tensor_slice.local_shape,
                        dtype=dest_tensor.dtype,
                        device=dest_tensor.device,
                    )
                    # Convert global intersection offsets to shard-local offsets
                    ndim = len(intersection.offsets)
                    src_local = tuple(
                        intersection.offsets[d] - handle.tensor_slice.offsets[d]
                        for d in range(ndim)
                    )
                    dest_local_off = tuple(
                        intersection.offsets[d] - dest_slice.offsets[d]
                        for d in range(ndim)
                    )
                    ops.append(
                        _TransferOp(
                            rdma_buffer=handle.rdma_buffer,
                            dest_byte_view=to_byte_view(recv),
                            dest_tensor=dest_tensor,
                            recv_buffer=recv,
                            src_slices=tuple(
                                slice(o, o + s)
                                for o, s in zip(src_local, intersection.local_shape)
                            ),
                            dest_slices=tuple(
                                slice(o, o + s)
                                for o, s in zip(
                                    dest_local_off, intersection.local_shape
                                )
                            ),
                        )
                    )

            # Same check as the routed plan: a gap would leave uninitialized
            # destination memory.
            wanted = get_slice_numel(dest_slice)
            if covered != wanted:
                raise ValueError(
                    f"source handles do not cover {name!r} exactly: {covered} of "
                    f"{wanted} elements at offsets={dest_slice.offsets}, "
                    f"shape={dest_slice.local_shape}"
                )

        logger.debug(f"Built transfer plan with {len(ops)} RDMA ops")
        return ops

    async def pull(
        self,
        all_handles: dict[str, list[RDMAWeightHandle]],
        dest_state_dict: dict[str, torch.Tensor],
    ) -> None:
        """Pull weights from source handles into *dest_state_dict*.

        The transfer plan is built on the first call and reused while the
        handles and destination memory stay the same.
        All RDMA reads are issued concurrently for maximum throughput.
        Run one pull at a time per instance: each pull deregisters every
        NIXL destination when it finishes.

        Args:
            all_handles: Per-param list of handles from all source ranks.
            dest_state_dict: Destination model state dict to write into
                (parameters are updated in-place).

        Raises:
            ValueError: If the handles do not cover a destination tensor
                that has handles exactly.
        """
        signature = _plan_signature(all_handles, dest_state_dict)
        if (
            self._plan is None
            or all_handles is not self._plan_handles
            or signature != self._plan_signature
        ):
            self._plan = self._build_plan(all_handles, dest_state_dict)
            self._plan_handles = all_handles
            self._plan_signature = signature
        plan = self._plan

        # RDMA writes are not stream ordered: wait for queued kernels that may
        # still use this memory (e.g. storage the caching allocator reused).
        for device in {
            op.dest_byte_view.device
            for op in plan
            if op.dest_byte_view.device.type == "cuda"
        }:
            torch.cuda.synchronize(device)

        # Issue all RDMA reads concurrently, one NIXL batch per source agent.
        monarch_reads = []
        nixl_reads: dict[str, tuple[NixlSourceAgent, list]] = {}
        for op in plan:
            if not isinstance(op.rdma_buffer, NixlBuffer):
                monarch_reads.append(op.rdma_buffer.read_into(op.dest_byte_view))
                continue
            agent = op.rdma_buffer.agent
            _, transfers = nixl_reads.setdefault(agent.name, (agent, []))
            transfers.append((op.dest_byte_view, op.rdma_buffer.descriptors))
        # return_exceptions: deregister only after every read has finished.
        try:
            results = await asyncio.gather(
                *monarch_reads,
                *(
                    self._read_nixl(agent, transfers)
                    for agent, transfers in nixl_reads.values()
                ),
                return_exceptions=True,
            )
            for result in results:
                if isinstance(result, BaseException):
                    raise result
        finally:
            # Destinations may be freed and reallocated at the same address
            # before the next pull (e.g. FSDP resharding), and a registration
            # over the old allocation silently drops writes. Registering costs
            # ~0.15 ms per storage for GPU memory.
            if self._nixl is not None:
                self._nixl.deregister_all()

        # Post-read copies (only for resharded ops). no_grad: dests may be
        # parameters that require grad.
        with torch.no_grad():
            for op in plan:
                if op.dest_tensor is None:
                    continue  # zero-copy, already in place
                # Resharded: copy overlapping region
                # recv_buffer contains the full source shard
                # op.recv_buffer[op.src_slices] extracts the overlap from the source
                # op.dest_tensor[op.dest_slices] targets where it goes in the dest
                op.dest_tensor[op.dest_slices].copy_(op.recv_buffer[op.src_slices])

        # Callers may consume the destination from other streams or threads,
        # which the copies above are not ordered with.
        for device in {
            op.dest_tensor.device
            for op in plan
            if op.dest_tensor is not None and op.dest_tensor.device.type == "cuda"
        }:
            torch.cuda.current_stream(device).synchronize()
