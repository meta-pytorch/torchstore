# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""NIXL transport on AWS Neuron: descriptors by device address, LIBFABRIC peers kept."""

from types import SimpleNamespace

import pytest

import torchstore.transport.nixl as nixl_transport
from torchstore.transport import neuron_memory


class RecordingAgent:
    def __init__(self, name="volume") -> None:
        self.name = name
        self.calls = []
        self.fail_partial_with = None

    def get_reg_descs(self, regions, memory_type=None):
        self.calls.append(("get_reg_descs", regions, memory_type))
        return ("reg", tuple(regions), memory_type)

    def get_xfer_descs(self, descs, memory_type=None):
        self.calls.append(("get_xfer_descs", descs, memory_type))
        return ("xfer", descs, memory_type)

    def register_memory(self, descs, backends=()):
        self.calls.append(("register_memory", descs, list(backends)))
        return descs

    def create_backend(self, backend, params):
        self.calls.append(("create_backend", backend))

    def add_remote_agent(self, metadata):
        self.calls.append(("add_remote_agent", metadata))
        if metadata.startswith(b"partial") and self.fail_partial_with is not None:
            raise self.fail_partial_with
        return metadata.split(b":")[1].decode()

    def remove_remote_agent(self, name):
        self.calls.append(("remove_remote_agent", name))


def neuron_tensor(nbytes=4096, index=3):
    """A duck-typed Neuron tensor: only what neuron_memory reads."""
    return SimpleNamespace(device=SimpleNamespace(type="neuron", index=index), nbytes=nbytes)


def test_neuron_tensor_registers_by_device_address(monkeypatch):
    monkeypatch.setattr(neuron_memory, "neuron_va", lambda t: 0x7F0000001000)
    agent = RecordingAgent()
    tensor = neuron_tensor()

    neuron_memory.register_memory(agent, tensor, ["LIBFABRIC"])
    neuron_memory.get_xfer_descs(agent, tensor)

    assert ("get_reg_descs", [(0x7F0000001000, 4096, 3, "")], "VRAM") in agent.calls
    assert ("get_xfer_descs", [(0x7F0000001000, 4096, 3)], "VRAM") in agent.calls
    assert agent.calls[1][0] == "register_memory" and agent.calls[1][2] == ["LIBFABRIC"]


def test_non_neuron_tensor_goes_to_nixl_unchanged():
    agent = RecordingAgent()
    cpu = SimpleNamespace(device=SimpleNamespace(type="cpu", index=None))
    neuron_memory.get_xfer_descs(agent, cpu)
    assert agent.calls == [("get_xfer_descs", cpu, None)]


@pytest.fixture
def libfabric_cache(monkeypatch):
    agent = RecordingAgent()
    monkeypatch.setattr(
        nixl_transport,
        "_load_nixl_api",
        lambda: (lambda name, config: agent, lambda **kwargs: kwargs),
    )
    monkeypatch.setattr(nixl_transport, "ENV_TORCHSTORE_NIXL_BACKEND", "LIBFABRIC")
    return nixl_transport.NixlAgentCache(), agent


def test_libfabric_keeps_a_known_peer_and_loads_only_new_registrations(libfabric_cache):
    cache, agent = libfabric_cache
    cache.add_remote_agent("client", b"full:client:v1")
    cache.add_remote_agent("client", b"full:client:v2", [b"partial:client:a"])

    assert ("remove_remote_agent", "client") not in agent.calls
    assert ("add_remote_agent", b"partial:client:a") in agent.calls
    assert ("add_remote_agent", b"full:client:v2") not in agent.calls


def test_libfabric_ignores_a_repeated_section(libfabric_cache):
    cache, agent = libfabric_cache
    agent.fail_partial_with = RuntimeError("NIXL_ERR_NOT_ALLOWED")
    cache.add_remote_agent("client", b"full:client:v1")
    cache.add_remote_agent("client", b"full:client:v2", [b"partial:client:a"])
    assert cache._remote_agents["client"] == b"full:client:v2"


def test_libfabric_raises_other_partial_errors(libfabric_cache):
    cache, agent = libfabric_cache
    agent.fail_partial_with = RuntimeError("NIXL_ERR_BACKEND")
    cache.add_remote_agent("client", b"full:client:v1")
    with pytest.raises(RuntimeError, match="NIXL_ERR_BACKEND"):
        cache.add_remote_agent("client", b"full:client:v2", [b"partial:client:a"])
