# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Direct RDMA weight sync — zero-copy weight transfer between actors.

Instead of routing tensor data through a StorageVolume (two hops),
this module lets the destination read directly from the source's GPU
memory via one-sided RDMA reads (one hop).

TorchStore is still used for metadata exchange (RDMA handle registration),
but the actual weight data travels directly source → destination.

Typical usage (RL trainer → generator weight sync):

    # Source side (trainer, called once at setup):
    source = DirectWeightSyncSource()
    handles = source.register(model.state_dict(), rank=dist.get_rank())
    await ts.put(f"{RDMA_KEY_PREFIX}/rank_{rank}", handles)

    # After each optimizer.step():
    source.refresh()

    # Destination side (generator):
    dest = DirectWeightSyncDest()
    # First call fetches + caches handles, subsequent calls reuse them
    await dest.pull(all_handles, model.state_dict())
"""

import asyncio
import logging
from dataclasses import dataclass

import torch

from torchstore.transport.types import Request, TensorSlice
from torchstore.utils import get_slice_intersection, to_byte_view

logger = logging.getLogger(__name__)

RDMA_KEY_PREFIX = "policy_rdma"


@dataclass
class RDMAWeightHandle:
    """Serializable handle for direct RDMA access to a weight shard.

    The ``rdma_buffer`` field is a Monarch ``RDMABuffer`` which is
    natively pickle-able across Monarch actors.  After deserialization
    on the destination, calling ``rdma_buffer.read_into(byte_view)``
    performs a one-sided RDMA read from the source's memory.
    """

    rdma_buffer: object  # monarch.rdma.RDMABuffer
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
        from monarch.rdma import RDMABuffer

        handles: dict[str, RDMAWeightHandle] = {}
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

            # Register the contiguous buffer with RDMA
            rdma_buf = RDMABuffer(to_byte_view(buf_tensor))

            handles[name] = RDMAWeightHandle(
                rdma_buffer=rdma_buf,
                tensor_slice=tensor_slice,
                source_rank=rank,
            )

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
            await handle.rdma_buffer.drop()
        self._handles.clear()
        self._staging.clear()


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


class DirectWeightSyncDest:
    """Pulls weights via direct RDMA reads from source handles.

    On the first :meth:`pull`, a transfer plan is computed and cached.
    Subsequent pulls skip all metadata work and just execute RDMA reads
    + post-read copies.  Receive buffers are allocated on GPU (same
    device as dest) to avoid CPU-GPU copies.
    """

    def __init__(self) -> None:
        self._plan: list[_TransferOp] | None = None

    def _build_plan(
        self,
        all_handles: dict[str, list[RDMAWeightHandle]],
        dest_state_dict: dict[str, torch.Tensor],
    ) -> list[_TransferOp]:
        """Build the transfer plan (called once on first pull).

        For each destination parameter, finds overlapping source shards
        and creates a _TransferOp describing how to read from each one.

        Three cases per (source, dest) pair:
          1. Exact match + contiguous dest -> zero-copy RDMA into param
          2. Exact match + non-contiguous dest -> RDMA into recv buf, full copy
          3. Partial overlap (resharding) -> RDMA into recv buf, slice copy
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

            for handle in handles:
                # Use get_slice_intersection to find overlap
                intersection = get_slice_intersection(handle.tensor_slice, dest_slice)
                if intersection is None:
                    continue

                # Skip duplicate regions (replicated params)
                region_key = (intersection.offsets, intersection.local_shape)
                if region_key in seen:
                    continue
                seen.add(region_key)

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

        logger.info(f"Built transfer plan with {len(ops)} RDMA ops")
        return ops

    async def pull(
        self,
        all_handles: dict[str, list[RDMAWeightHandle]],
        dest_state_dict: dict[str, torch.Tensor],
    ) -> None:
        """Pull weights from source handles into *dest_state_dict*.

        On the first call the transfer plan is built and cached.
        All RDMA reads are issued concurrently for maximum throughput.

        Args:
            all_handles: Per-param list of handles from all source ranks.
            dest_state_dict: Destination model state dict to write into
                (parameters are updated in-place).
        """
        if self._plan is None:
            self._plan = self._build_plan(all_handles, dest_state_dict)

        # Issue all RDMA reads concurrently
        await asyncio.gather(
            *(op.rdma_buffer.read_into(op.dest_byte_view) for op in self._plan)
        )

        # Post-read copies (only for non-contiguous or resharded ops)
        for op in self._plan:
            if op.dest_tensor is None:
                continue  # zero-copy, already in place
            if op.src_slices is not None:
                # Resharded: copy overlapping region
                # recv_buffer contains the full source shard
                # op.recv_buffer[op.src_slices] extracts the overlap from the source
                # op.dest_tensor[op.dest_slices] targets where it goes in the dest
                op.dest_tensor[op.dest_slices].copy_(op.recv_buffer[op.src_slices])
            else:
                # Non-contiguous exact match: full copy
                # recv_buffer is the contiguous version of what we want in the
                # dest_tensor, after the RDMA read
                op.dest_tensor.copy_(op.recv_buffer)
