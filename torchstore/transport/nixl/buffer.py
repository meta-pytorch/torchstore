# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

import torch

from torchstore.logging import init_logging, LatencyTracker
from torchstore.transport.buffers import TransportBuffer

try:
    from nixl._api import nixl_agent, nixl_agent_config

    NIXL_AVAILABLE = True
except ImportError:
    NIXL_AVAILABLE = False


if TYPE_CHECKING:
    from torchstore.transport.buffers import TransportContext
    from torchstore.transport.pipe import StorageVolumeRef


def nixl_available() -> bool:
    """Check if NIXL is available and enabled."""
    nixl_enabled = os.environ.get("TORCHSTORE_NIXL_ENABLED", "0") == "1"
    return nixl_enabled and NIXL_AVAILABLE


def _ensure_pinned(tensor: torch.Tensor) -> torch.Tensor:
    """Ensure a CPU tensor is pinned for NIXL RDMA transfers.

    NIXL's UCX backend requires pinned (page-locked) memory for CPU tensors.
    GPU tensors are returned as-is.
    """
    if tensor.device.type == "cuda":
        return tensor

    # Ensure CUDA is initialized for pinned memory to work with NIXL
    if torch.cuda.is_available() and not torch.cuda.is_initialized():
        torch.cuda.init()

    # If already pinned, return as-is
    if tensor.is_pinned():
        return tensor

    # Create a pinned copy
    pinned = torch.empty_like(tensor, pin_memory=True)
    pinned.copy_(tensor)
    return pinned


class NixlTransportBuffer(TransportBuffer):
    """Transport buffer implementation using NIXL for efficient tensor transfer."""

    requires_meta: bool = True

    def __init__(self) -> None:
        init_logging()

        # Transport context (stored during handshake for use in allocation)
        self._transport_context: "TransportContext | None" = None

        # Local tensor reference
        self.tensor_ref: torch.Tensor | None = None

        # NIXL handles
        self.reg_descs: Any | None = None  # Memory registration descriptors
        self.serialized_descs: bytes | None = None  # Serialized tensor descriptors
        self.agent_metadata: bytes | None = None  # Agent metadata for remote connection

        # Tensor metadata
        self.shape: Optional[torch.Size] = None
        self.dtype: Optional[torch.dtype] = None

    def _get_or_create_agent(
        self, transport_context: "TransportContext"
    ) -> "nixl_agent":
        """Get or create a NIXL agent from the transport context."""
        NIXL_AGENT_KEY = "nixl_agent"
        ctx = transport_context.get_transport_context()

        if NIXL_AGENT_KEY not in ctx:
            # Get backend from environment (default to UCX)
            backend = os.environ.get("TORCHSTORE_NIXL_BACKEND", "UCX")
            # Simple config - just specify backends, no listening port needed
            config = nixl_agent_config(backends=[backend])

            # Use a unique agent name
            import uuid

            agent_name = f"torchstore-{uuid.uuid4()}"
            ctx[NIXL_AGENT_KEY] = nixl_agent(agent_name, config)

        return ctx[NIXL_AGENT_KEY]

    async def handshake(
        self, tensor: torch.Tensor, volume_ref: "StorageVolumeRef"
    ) -> None:
        """
        Establish a NIXL connection with the storage volume.

        Client-side: Creates agent and calls storage's recv_handshake to ensure
        storage agent is initialized.
        """
        latency_tracker = LatencyTracker("nixl_handshake")

        # Store transport context for use in allocation methods
        self._transport_context = volume_ref.transport_context

        # Create client-side agent
        agent = self._get_or_create_agent(volume_ref.transport_context)
        latency_tracker.track_step("get_agent")

        # Get our agent metadata for the storage side to connect back
        self.agent_metadata = agent.get_agent_metadata()
        latency_tracker.track_step("get_agent_metadata")

        # Call storage's handshake to ensure storage agent is created
        await volume_ref.volume.handshake.call_one(self)
        latency_tracker.track_step("storage_handshake")

        latency_tracker.track_e2e()

    async def recv_handshake(
        self, transport_context: "TransportContext"
    ) -> Optional[Any]:
        """
        Confirm a handshake initiated by the local client.

        Storage-volume side: Creates agent and returns its metadata.
        """
        # Create agent for storage side
        agent = self._get_or_create_agent(transport_context)

        # Return agent metadata for the client to use
        return {
            "agent_metadata": agent.get_agent_metadata(),
        }

    def __getstate__(self) -> Dict[str, Any]:
        """
        Serialize the state of the buffer.
        Excludes non-serializable NIXL handles but includes serialized descriptors
        and agent metadata.
        """
        state = self.__dict__.copy()
        state["tensor_ref"] = None
        state["reg_descs"] = None
        state["_transport_context"] = None
        return state

    def _allocate(self, tensor: torch.Tensor) -> None:
        """Allocate and register memory with NIXL."""
        assert (
            self._transport_context is not None
        ), "Must call handshake before allocate"
        self._assert_valid_tensor(tensor, self.dtype, self.shape)

        # Ensure tensor is contiguous for RDMA
        if not tensor.is_contiguous():
            raise RuntimeError("Tensor must be contiguous for NIXL registration")

        agent = self._get_or_create_agent(self._transport_context)

        print(
            f"DEBUG _allocate: tensor shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}"
        )

        # Register memory with NIXL
        self.reg_descs = agent.register_memory([tensor])
        if not self.reg_descs:
            raise RuntimeError("NIXL memory registration failed")

        print(f"DEBUG _allocate: reg_descs={self.reg_descs}")

        # Get serialized descriptors using trim()
        self.serialized_descs = agent.get_serialized_descs(self.reg_descs.trim())
        print(f"DEBUG _allocate: serialized_descs len={len(self.serialized_descs)}")

    def allocate_dest(self, tensor_like: torch.Tensor | Tuple) -> None:
        """Called by the local client. Allocate memory for destination tensor (get)."""
        if isinstance(tensor_like, str) or tensor_like is None:
            return
        elif isinstance(tensor_like, Tuple):
            # Ensure CUDA is initialized for pinned memory to work with NIXL
            if torch.cuda.is_available() and not torch.cuda.is_initialized():
                torch.cuda.init()
            # For CPU tensors, allocate with pin_memory for NIXL RDMA
            self.tensor_ref = torch.empty(
                tensor_like[0], dtype=tensor_like[1], device=torch.device("cpu"), pin_memory=True
            )
            self.shape, self.dtype = tensor_like
        else:
            assert isinstance(tensor_like, torch.Tensor)
            # Ensure CPU tensors are pinned for NIXL
            self.tensor_ref = _ensure_pinned(tensor_like)
            self.shape, self.dtype = tensor_like.shape, tensor_like.dtype

        self._allocate(self.tensor_ref)

    def allocate_source(self, tensor: Optional[torch.Tensor]) -> None:
        """Called by the local client. Allocate memory for source tensor (put)."""
        if tensor is None:
            return

        self.shape = tensor.shape
        self.dtype = tensor.dtype
        # Ensure CPU tensors are pinned for NIXL
        self.tensor_ref = _ensure_pinned(tensor)

        self._allocate(self.tensor_ref)

    async def read_into(
        self, tensor: Optional[torch.Tensor], transport_context: "TransportContext"
    ) -> torch.Tensor:
        """Called by the remote storage volume. Read from client's source memory (put)."""
        latency_tracker = LatencyTracker("nixl_read_into")

        # Debug: verify serialized data is available
        if self.serialized_descs is None:
            raise RuntimeError(
                "serialized_descs is None - buffer not properly initialized"
            )
        if self.agent_metadata is None:
            raise RuntimeError(
                "agent_metadata is None - buffer not properly initialized"
            )

        print(f"DEBUG read_into: shape={self.shape}, dtype={self.dtype}")
        print(f"DEBUG read_into: serialized_descs len={len(self.serialized_descs)}")
        print(f"DEBUG read_into: agent_metadata len={len(self.agent_metadata)}")

        if tensor is None:
            # Ensure CUDA is initialized for pinned memory to work with NIXL
            if torch.cuda.is_available() and not torch.cuda.is_initialized():
                torch.cuda.init()
            # Use pinned memory for CPU tensors to enable NIXL RDMA
            tensor = torch.zeros(
                self.shape, dtype=self.dtype, device=torch.device("cpu"), pin_memory=True
            )
        else:
            # Ensure existing CPU tensors are pinned
            tensor = _ensure_pinned(tensor)
        latency_tracker.track_step("allocate_tensor")

        self._assert_valid_tensor(tensor, self.dtype, self.shape)

        agent = self._get_or_create_agent(transport_context)
        latency_tracker.track_step("get_agent")

        # Register local memory for receiving
        local_reg_descs = agent.register_memory([tensor])
        if not local_reg_descs:
            raise RuntimeError("NIXL memory registration failed for read_into")
        latency_tracker.track_step("register_memory")

        # Deserialize remote descriptors
        remote_descs = agent.deserialize_descs(self.serialized_descs)
        if not remote_descs:
            raise RuntimeError("Failed to deserialize remote descriptors")
        latency_tracker.track_step("deserialize_remote_descs")

        # Add remote agent using its metadata
        remote_name = agent.add_remote_agent(self.agent_metadata)
        if not remote_name:
            raise RuntimeError("Failed to add remote agent")
        latency_tracker.track_step("add_remote_agent")

        print(
            f"DEBUG read_into: local tensor shape={tensor.shape}, device={tensor.device}"
        )
        print(f"DEBUG read_into: remote_name={remote_name}")
        print(f"DEBUG read_into: local_reg_descs.getType()={local_reg_descs.getType()}")
        print(f"DEBUG read_into: remote_descs.getType()={remote_descs.getType()}")

        xfer_handle = None
        try:
            # Initialize READ transfer
            xfer_handle = agent.initialize_xfer(
                "READ",
                local_reg_descs.trim(),
                remote_descs,
                remote_name,
                "notif",
            )
            latency_tracker.track_step("initialize_xfer")

            state = agent.transfer(xfer_handle)
            if state == "ERR":
                raise RuntimeError("NIXL transfer posting failed")
            latency_tracker.track_step("start_transfer")

            # Wait for transfer completion
            while True:
                state = agent.check_xfer_state(xfer_handle)
                if state == "ERR":
                    raise RuntimeError("NIXL transfer failed")
                elif state == "DONE":
                    break
                time.sleep(0.001)  # Avoid busy waiting
            latency_tracker.track_step("wait_transfer")

        finally:
            # Cleanup
            if xfer_handle:
                agent.release_xfer_handle(xfer_handle)
            agent.remove_remote_agent(remote_name)
            agent.deregister_memory(local_reg_descs)
            latency_tracker.track_step("cleanup")

        latency_tracker.track_e2e()
        return tensor

    async def write_from(
        self, tensor: Optional[torch.Tensor], transport_context: "TransportContext"
    ) -> None:
        """Called by the remote storage volume. Write to client's dest memory (get)."""
        if tensor is None:
            return

        latency_tracker = LatencyTracker("nixl_write_from")

        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        latency_tracker.track_step("ensure_contiguous")

        # Ensure CPU tensors are pinned for NIXL
        tensor = _ensure_pinned(tensor)

        self._assert_valid_tensor(tensor, self.dtype, self.shape)

        agent = self._get_or_create_agent(transport_context)
        latency_tracker.track_step("get_agent")

        # Register local memory for sending (pass as list per NIXL API)
        local_reg_descs = agent.register_memory([tensor])
        if not local_reg_descs:
            raise RuntimeError("NIXL memory registration failed for write_from")
        latency_tracker.track_step("register_memory")

        # Deserialize remote descriptors (client's destination)
        remote_descs = agent.deserialize_descs(self.serialized_descs)
        latency_tracker.track_step("deserialize_remote_descs")

        # Add remote agent using its metadata
        remote_name = agent.add_remote_agent(self.agent_metadata)
        latency_tracker.track_step("add_remote_agent")

        xfer_handle = None
        try:
            # Initialize WRITE transfer
            xfer_handle = agent.initialize_xfer(
                "WRITE",
                local_reg_descs.trim(),
                remote_descs,
                remote_name,
                "notif",
            )
            latency_tracker.track_step("initialize_xfer")

            state = agent.transfer(xfer_handle)
            if state == "ERR":
                raise RuntimeError("NIXL transfer posting failed")
            latency_tracker.track_step("start_transfer")

            # Wait for transfer completion
            while True:
                state = agent.check_xfer_state(xfer_handle)
                if state == "ERR":
                    raise RuntimeError("NIXL transfer failed")
                elif state == "DONE":
                    break
                time.sleep(0.001)  # Avoid busy waiting
            latency_tracker.track_step("wait_transfer")

        finally:
            # Cleanup
            if xfer_handle:
                agent.release_xfer_handle(xfer_handle)
            agent.remove_remote_agent(remote_name)
            agent.deregister_memory(local_reg_descs)
            latency_tracker.track_step("cleanup")

        latency_tracker.track_e2e()

    async def drop(self) -> None:
        """Clean up NIXL resources."""
        # Deregister memory if we have reg_descs and transport_context
        if self.reg_descs is not None and self._transport_context is not None:
            try:
                agent = self._get_or_create_agent(self._transport_context)
                agent.deregister_memory(self.reg_descs)
            except Exception:
                pass  # Best effort cleanup

        self.tensor_ref = None
        self.reg_descs = None
        self.serialized_descs = None
        self.agent_metadata = None
