# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""NIXL memory descriptors for AWS Neuron (Trainium/Inferentia) device tensors.

NIXL builds descriptors from a torch tensor with ``data_ptr()`` and maps CUDA to
``VRAM`` and CPU to ``DRAM``. A torch-neuronx tensor's ``data_ptr()`` is not its
device address, so Neuron tensors are described explicitly instead: the Neuron
runtime address of the tensor (``nrt_tensor_get_va``), its size, and the Neuron
device index, as ``VRAM``. NIXL's LIBFABRIC backend (EFA) then moves Neuron HBM
directly; build NIXL with that plugin and set ``TORCHSTORE_NIXL_BACKEND=LIBFABRIC``.

Proven with torchft's NIXL process group on trn3 (EFA, Neuron HBM, 23.8 GB/s for
one 16.6 GB multi-descriptor WRITE).
"""

from __future__ import annotations

import ctypes
from functools import cache
from typing import Any

import torch

NEURON_DEVICE_TYPES = ("neuron", "privateuseone")


def is_neuron(tensor: torch.Tensor) -> bool:
    return tensor.device.type in NEURON_DEVICE_TYPES


@cache
def _nrt() -> Any:
    nrt = ctypes.CDLL("libnrt.so.1")
    nrt.nrt_tensor_get_va.restype = ctypes.c_void_p
    nrt.nrt_tensor_get_va.argtypes = [ctypes.c_void_p]
    return nrt


def _capsule_va(tensor: torch.Tensor) -> int:
    import torch_neuronx

    capsule = torch_neuronx._C._get_nrt_tensor_capsule(tensor)
    api = ctypes.pythonapi
    api.PyCapsule_GetName.restype = ctypes.c_char_p
    api.PyCapsule_GetName.argtypes = [ctypes.py_object]
    api.PyCapsule_GetPointer.restype = ctypes.c_void_p
    api.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    pointer = api.PyCapsule_GetPointer(capsule, api.PyCapsule_GetName(capsule))
    return int(_nrt().nrt_tensor_get_va(pointer))


def neuron_va(tensor: torch.Tensor) -> int:
    """The Neuron runtime device address of ``tensor``'s first byte.

    The runtime resolves the tensor's allocation; for a view it may have no
    entry of its own, so the storage's base tensor plus the view's offset is
    used instead.
    """
    try:
        return _capsule_va(tensor) + tensor.storage_offset() * tensor.element_size()
    except RuntimeError:
        base = torch.empty(0, dtype=torch.uint8, device=tensor.device).set_(
            tensor.untyped_storage()
        )
        return _capsule_va(base) + tensor.storage_offset() * tensor.element_size()


def neuron_device_id(tensor: torch.Tensor) -> int:
    return tensor.device.index or 0


def neuron_reg_descs(agent: Any, region: torch.Tensor) -> Any:
    """Registration descriptors for a Neuron tensor (usually a whole-storage region)."""
    return agent.get_reg_descs(
        [(neuron_va(region), region.nbytes, neuron_device_id(region), "")], "VRAM"
    )


def neuron_xfer_descs(agent: Any, tensor: torch.Tensor) -> Any:
    """Transfer descriptors for a contiguous Neuron tensor."""
    return agent.get_xfer_descs(
        [(neuron_va(tensor), tensor.nbytes, neuron_device_id(tensor))], "VRAM"
    )


def register_memory(agent: Any, region: torch.Tensor, backends: list[str]) -> Any:
    """``agent.register_memory`` that also understands Neuron tensors."""
    if is_neuron(region):
        return agent.register_memory(neuron_reg_descs(agent, region), backends=backends)
    return agent.register_memory(region, backends=backends)


def get_xfer_descs(agent: Any, tensor: torch.Tensor) -> Any:
    """``agent.get_xfer_descs`` that also understands Neuron tensors."""
    if is_neuron(tensor):
        return neuron_xfer_descs(agent, tensor)
    return agent.get_xfer_descs(tensor)
