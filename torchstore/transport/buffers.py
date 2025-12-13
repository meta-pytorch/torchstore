# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from functools import cache
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING, Union

import torch

try:
    from monarch.rdma import is_rdma_available as monarch_rdma_available, RDMABuffer
except ImportError:
    monarch_rdma_available = lambda: False

    def RDMABuffer(*args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "RDMABuffer is not available. This environment was likely not built with rdma support."
        )


try:
    from torchcomms._transport import RdmaMemory, RdmaRemoteBuffer, RdmaTransport

    torchcomms_available = True
except ImportError:
    torchcomms_available = False

if TYPE_CHECKING:
    from torchstore.transport.pipe import StorageVolumeRef


# TODO: we no longer need to chunk with monararch rdma buffer. Setting large chunk size for now,
# but we should remove all chunking code
RDMA_CHUNK_SIZE_MB: int = int(
    os.environ.get("TORCHSTORE_RDMA_CHUNK_SIZE_MB", str(1024 * 32))
)


def rdma_available() -> bool:
    rdma_enabled = (
        os.environ.get("TORCHSTORE_RDMA_ENABLED", "1") == "1"
    )  # TODO: enable on this build
    return rdma_enabled and monarch_rdma_available()


@cache
def torchcomms_rdma_available() -> bool:
    rdma_enabled = os.environ.get("USE_TORCHCOMMS_RDMA", "0") == "1"
    # (1) CommsRDMA flag is enabled (2) torchcomms lib is available (3) RDMA is supported
    return rdma_enabled and torchcomms_available and RdmaTransport.supported()


TransportAndAddress = Tuple["RdmaTransport", bytes]


class RdmaTransportCache:
    def __init__(self) -> None:
        assert torchcomms_rdma_available(), "TorchComms RDMA is not available."
        # {key: {device: (transport, address)}}
        self.transports: Dict[str, Dict[int, TransportAndAddress]] = {}

    @classmethod
    def try_init(cls) -> Optional["RdmaTransportCache"]:
        try:
            return cls()
        except Exception as e:
            logging.info(f"Failed to init RdmaTransportCache: {e}")
            return None

    def _device_to_index(self, device: torch.device | int) -> int:
        if isinstance(device, int):
            return device
        else:
            return 0 if device.type == "cpu" else device.index

    def put(self, key: str, device: torch.device | int) -> TransportAndAddress:
        index = self._device_to_index(device)
        transport = RdmaTransport(torch.device(index))

        if key not in self.transports:
            self.transports[key] = {}
        val = (transport, transport.bind())
        self.transports[key][index] = val
        return val

    def _get(self, key: str, device: torch.device | int) -> TransportAndAddress:
        index = self._device_to_index(device)
        return self.transports[key][index]

    def get(self, key: str, device: torch.device | int):
        if not self.contains(key, device):
            return self.put(key, device)
        return self._get(key, device)

    def contains(self, key: str, device: torch.device | int) -> bool:
        index = self._device_to_index(device)
        return key in self.transports and index in self.transports[key]


class TransportContext:
    RDMA_TRANSPORT_CACHE = "rdma_transport_cache"

    def __init__(self):
        self.transport_context = {}
        self.transport_context[
            self.RDMA_TRANSPORT_CACHE
        ] = RdmaTransportCache.try_init()

    def get_transport_context(self) -> Dict[Any, Any]:
        return self.transport_context

    def get_rdma_transport_cache(self) -> RdmaTransportCache:
        return self.transport_context.get(self.RDMA_TRANSPORT_CACHE, None)


class TransportBuffer:
    finalize: bool = False
    is_object: bool = False
    objects: Optional[Any] = None
    requires_meta: bool = False

    def update(self, other_buffer: "TransportBuffer") -> None:
        self.finalize = other_buffer.finalize
        self.is_object = other_buffer.is_object
        self.objects = other_buffer.objects
        self.requires_meta = other_buffer.requires_meta

    def allocate(self, tensor_like: Union[torch.Tensor, Tuple]) -> None:
        """Allocates internal buffers based on either an existing tensor
        or a Tuple of (shape, dtype)
        """
        raise NotImplementedError()

    async def read_into(
        self, tensor: Optional[torch.Tensor], transport_context: TransportContext
    ) -> torch.Tensor:
        raise NotImplementedError()

    async def write_from(
        self, tensor: Optional[torch.Tensor], transport_context: TransportContext
    ) -> None:
        raise NotImplementedError()

    async def handshake(
        self, tensor: torch.Tensor, volume_ref: "StorageVolumeRef"
    ) -> None:
        """Establish a handshake with the remote volume, such as for RDMA."""
        pass

    async def recv_handshake(
        self, transport_context: TransportContext
    ) -> Optional[Any]:
        """Confirm a handshake initiated by the local client, and return some result."""
        pass

    async def drop(self) -> None:
        """Clean up any resources held by this buffer. Override in subclasses if needed."""
        pass

    def _assert_valid_tensor(
        self,
        tensor: torch.Tensor,
        dtype: torch.dtype,
        shape: torch.Size,
        must_be_contiguous=False,
    ) -> None:
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == dtype, f"{tensor.dtype} != {dtype}"
        assert tensor.shape == shape, f"{tensor.shape} != {shape}"
        assert not must_be_contiguous or tensor.is_contiguous()


class TorchCommsRdmaTransportBuffer(TransportBuffer):
    requires_meta: bool = True

    def __init__(self) -> None:
        # local client's rdmatransport address. used by storage volume to retrieve cached peer transport.
        self.address: bytes | None = None

        self.tensor_ref: torch.Tensor | None = (
            None  # reference to local client's destination tensor
        )
        self.rdma_memory: RdmaMemory | None = (
            None  # must be kept alive until transport is done
        )
        self.rdma_remote_buffer: RdmaRemoteBuffer | None = (
            None  # remote reference of rdma memory
        )

        self.shape: Optional[torch.Size] = None
        self.dtype: Optional[torch.dtype] = None

    async def handshake(
        self, tensor: torch.Tensor, volume_ref: "StorageVolumeRef"
    ) -> None:
        """
        Establish an RDMA handshake with the storage volume, and save the local RdmaTransport address.
        """
        device = tensor.device if tensor is not None else 0
        transport_cache = volume_ref.transport_context.get_rdma_transport_cache()
        connection_exists = transport_cache.contains(volume_ref.volume_id, device)
        local_transport, self.address = transport_cache.get(
            volume_ref.volume_id, device
        )

        if connection_exists:
            return

        peer_addr = await volume_ref.volume.handshake.call_one(self)
        local_transport.connect(peer_addr)

    async def recv_handshake(
        self, transport_context: TransportContext
    ) -> Optional[Any]:
        """
        Confirm a handshake initiated by the local client.
        """
        transport_cache = transport_context.get_rdma_transport_cache()
        transport, addr = transport_cache.put(self.address, device=0)
        transport.connect(self.address)
        return addr

    def __getstate__(self) -> Dict[str, Any]:
        """
        Serialize the state of the buffer, including RdmaRemoteBuffer but excluding the RdmaMemory and local dest tensor ref.
        """
        state = self.__dict__.copy()
        state["rdma_memory"] = None
        state["tensor_ref"] = None
        return state

    def _allocate(self, tensor: torch.Tensor) -> None:
        self._assert_valid_tensor(tensor, self.dtype, self.shape)
        self.rdma_memory = RdmaMemory(tensor)
        self.rdma_remote_buffer = self.rdma_memory.to_remote_buffer()

    def allocate_dest(self, tensor_like: Union[torch.Tensor, Tuple]) -> None:
        """Called by the local client. Allocate RdmaMemory for the destination tensor (get)."""
        if isinstance(tensor_like, str) or tensor_like is None:
            return
        elif isinstance(tensor_like, Tuple):
            self.tensor_ref = torch.empty(
                tensor_like[0], dtype=tensor_like[1], device=torch.device("cpu")
            )
            self.shape, self.dtype = tensor_like
        else:
            assert isinstance(tensor_like, torch.Tensor)
            self.tensor_ref = tensor_like
            self.shape, self.dtype = tensor_like.shape, tensor_like.dtype

        self._allocate(self.tensor_ref)

    # TODO @amirafzali: add test case and support for non-contiguous input
    def allocate_source(self, tensor: Optional[torch.Tensor]) -> None:
        """Called by the local client. Allocate RdmaMemory for the source tensor (put)."""
        if tensor is None:
            return

        self.shape = tensor.shape
        self.dtype = tensor.dtype

        self._allocate(tensor)

    async def read_into(
        self, tensor: Optional[torch.Tensor], transport_context: TransportContext
    ) -> torch.Tensor:
        """Called by the remote storage volume. Read from the local client's source RdmaMemory (put)"""
        if tensor is None:
            tensor = torch.empty(
                self.shape, dtype=self.dtype, device=torch.device("cpu")
            )

        assert self.rdma_remote_buffer is not None
        self._assert_valid_tensor(tensor, self.dtype, self.shape)

        transport_cache = transport_context.get_rdma_transport_cache()
        transport = transport_cache.get(self.address, 0)[0]

        receiving_buffer = RdmaMemory(tensor)
        res = transport.read(
            receiving_buffer.to_mutable_view(), self.rdma_remote_buffer
        )
        assert res == 0, f"RDMA read failed: conn code {res}"

        return tensor

    async def write_from(
        self, tensor: Optional[torch.Tensor], transport_context: TransportContext
    ) -> None:
        """Called by the remote storage volume. Write to the local client's dest RdmaMemory (get)"""
        if tensor is None:
            return

        if not tensor.is_contiguous():
            contiguous_buffer = torch.empty(
                tensor.shape,
                dtype=tensor.dtype,
                device="cpu",
                memory_format=torch.contiguous_format,
                pin_memory=True,
            )
            contiguous_buffer.copy_(tensor)
            tensor = contiguous_buffer

        assert self.rdma_remote_buffer is not None
        self._assert_valid_tensor(tensor, self.dtype, self.shape)
        rdma_memory = RdmaMemory(tensor)

        transport_cache = transport_context.get_rdma_transport_cache()
        transport, _ = transport_cache.get(self.address, 0)
        res = transport.write(rdma_memory.to_view(), self.rdma_remote_buffer)
        assert res == 0, f"RDMA write failed: conn code {res}"

    async def drop(self) -> None:
        del self.rdma_remote_buffer
        del self.rdma_memory
        self.tensor_ref = None


class RDMATransportBuffer(TransportBuffer):
    # TODO: when we try this with rdma, I should be able to write rdma directly to the tensor
    # for now we utilize copies.
    # The major blocker for this is dealing with non-contiguous tensors
    requires_meta: bool = True

    def __init__(self) -> None:
        self.rdma_buffers: Optional[List[Any]] = None
        self.tensor_refs: Optional[List[torch.Tensor]] = None
        self.shape: Optional[torch.Size] = None
        self.dtype: Optional[torch.dtype] = None

    async def drop(self) -> None:
        """Explicitly clean up RDMA buffers to prevent kernel memory leak.

        When RDMA buffers are created, they register memory regions with the RDMA
        hardware which pins pages in kernel memory. Without explicit cleanup, these
        pages remain pinned even after the Python objects are garbage collected,
        leading to a memory leak that manifests as unbounded Inactive(anon) growth.
        """
        if self.rdma_buffers is not None:
            for rdma_buf in self.rdma_buffers:
                try:
                    # Drop the RDMA buffer to deregister the memory region
                    await rdma_buf.drop()
                except Exception as e:
                    # Log but don't raise - cleanup should be best-effort
                    logging.warning(f"Failed to drop RDMA buffer during cleanup: {e}")
            self.rdma_buffers = None
            self.tensor_refs = None

    def __getstate__(self) -> Dict[str, Any]:
        # Any time that we serialize the transport buffer, the idea is
        # that tensors will be transported via tensor_enginer.RDMABuffer, so it makes
        # no sense to hold this reference when we are serializing
        state = self.__dict__.copy()
        state["tensor_refs"] = None
        return state

    def _create_byte_views_from_tensor(
        self, tensor: torch.Tensor
    ) -> List[torch.Tensor]:
        # handle scalar values
        if tensor.dim() == 0:
            tensor = tensor.unsqueeze(0)
        byte_view = tensor.view(torch.uint8).flatten()
        chunk_size = RDMA_CHUNK_SIZE_MB * 1024 * 1024
        tensor_chunks = torch.split(byte_view, chunk_size, dim=0)

        return tensor_chunks

    def allocate(self, tensor_like: Union[torch.Tensor, Tuple]) -> None:
        """Allocates internal buffers based on either an existing tensor
        or a Tuple of (shape, dtype)
        """
        logging.debug("Allocating rdma buffer")

        if isinstance(tensor_like, str) or tensor_like is None:
            # tensor is just an object, nothing to allocte
            return
        elif isinstance(tensor_like, Tuple):
            # we know the size of the tensor from fetching metadata
            tensor = torch.empty(
                tensor_like[0], dtype=tensor_like[1], device=torch.device("cpu")
            )
        else:
            # we have an inplace tensor, allocate a copy
            assert isinstance(tensor_like, torch.Tensor)
            tensor = torch.empty_like(tensor_like, device=torch.device("cpu"))

        # store tensor meta
        self.shape = tensor.shape
        self.dtype = tensor.dtype
        self.dim = tensor.dim()

        self._assert_valid_tensor(tensor, self.dtype, self.shape)

        byte_view_chunks = self._create_byte_views_from_tensor(tensor)
        self.tensor_refs = [
            torch.empty_like(chunk, device=torch.device("cpu"))
            for chunk in byte_view_chunks
        ]
        self.rdma_buffers = [RDMABuffer(chunk) for chunk in self.tensor_refs]

        chunk_sizes = set()
        for chunk in self.tensor_refs:
            chunk_sizes.add(chunk.shape)
        logging.debug(f"Allocted {len(self.rdma_buffers)} rdma buffers {chunk_sizes=}")

    def update(self, other_buffer: "TransportBuffer") -> None:
        super().update(other_buffer)

    # send
    async def read_into(
        self, tensor: Optional[torch.Tensor], transport_context: TransportContext
    ) -> torch.Tensor:
        if tensor is None:
            # allocate a tensor to return
            tensor = torch.empty(
                self.shape, dtype=self.dtype, device=torch.device("cpu")
            )

        self._assert_valid_tensor(tensor, self.dtype, self.shape)
        assert self.rdma_buffers is not None

        chunked_byte_view = self._create_byte_views_from_tensor(tensor)

        # if we have tensor refs locally, we're still in the local case,
        # and we're just copying over our chunks into the tensor from
        # local memory
        if self.tensor_refs is not None:
            for idx, chunk in enumerate(chunked_byte_view):
                chunk.copy_(self.tensor_refs[idx])
            return tensor
        # else: we are in the remote case (in a different process), and must read from
        # the rdma buffer
        # TODO: gather instead of reading sequentially
        try:
            for idx, chunk in enumerate(chunked_byte_view):
                await self.rdma_buffers[idx].read_into(chunk)
        except Exception as e:
            logging.exception(
                f"Failed read_into, {tensor.shape=}, {tensor.dtype=}", exc_info=e
            )
            raise e

        return tensor

    # recv
    async def write_from(
        self, tensor: Optional[torch.Tensor], transport_context: TransportContext
    ) -> None:
        if tensor is None:
            return
        # source tensor does not have to be contiguous, it is copied into contiguous memory later in this function
        self._assert_valid_tensor(
            tensor, self.dtype, self.shape, must_be_contiguous=False
        )
        assert self.rdma_buffers is not None

        chunked_byte_view = self._create_byte_views_from_tensor(tensor)

        # if we have tensor refs locally, we're still in the local case,
        # and we're just copying over from the tensor into local memory
        if self.tensor_refs is not None:
            for idx, chunk in enumerate(chunked_byte_view):
                self.tensor_refs[idx].copy_(chunk)
            return
        # else: we are in the remote case (in a different process), and must read from
        # the rdma buffer
        # TODO: gather instead of reading sequentially
        for idx, chunk in enumerate(chunked_byte_view):
            await self.rdma_buffers[idx].write_from(chunk)


class MonarchTransportBuffer(TransportBuffer):
    """This interface is mostly a noop, intended to be used with Monarch's regular RPC.
    Not expected to be super fast, but always works.
    """

    finalize: bool = True

    def __init__(self) -> None:
        self.tensor: Optional[torch.Tensor] = None

    def allocate(self, tensor_like: Union[torch.Tensor, Tuple]) -> None:
        """In the case of using monarch comms, we don't do any allocation ahead of time"""
        return None

    # send
    async def read_into(
        self, tensor: Optional[torch.Tensor], transport_context: TransportContext
    ) -> torch.Tensor:
        if tensor is not None:
            # if there is a tensor here, likely this is the 'inplace' case,
            # and we should return back a ptr to the original tensor
            # (as opposed to the stored tensor, which we likely don't want to
            # keep around)
            tensor.copy_(self.tensor)
            return tensor

        return self.tensor

    # recv
    async def write_from(
        self, tensor: Optional[torch.Tensor], transport_context: TransportContext
    ) -> None:
        self.tensor = tensor

    def update(self, other_buffer: "TransportBuffer") -> None:
        super().update(other_buffer)
        self.tensor = other_buffer.tensor
