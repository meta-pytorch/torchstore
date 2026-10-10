# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for how the gloo transport binds its master TCPStore.

``GlooTransportBuffer._pre_handshake`` creates the master ``TCPStore`` with
``port=0``, so the OS assigns a free port as part of the bind, and sends the
bound port to the storage volume in the handshake RPC.
"""

import asyncio
import pickle
from datetime import timedelta

import pytest
import torch
import torchstore.transport.gloo as gloo
from torch.distributed import TCPStore
from torchstore.transport.buffers import TransportContext
from torchstore.transport.gloo import GlooTransportBuffer


class MockStorageVolumeRef:
    def __init__(self) -> None:
        self.volume_id = "test_volume"
        self.transport_context = TransportContext()


def _connect_worker(port: int) -> TCPStore:
    return TCPStore(
        host_name="127.0.0.1",
        port=port,
        world_size=2,
        is_master=False,
        timeout=timedelta(seconds=30),
    )


@pytest.fixture(autouse=True)
def loopback(monkeypatch):
    """Bind on loopback and start each test with an empty connection cache."""
    monkeypatch.setattr(gloo, "_get_hostname", lambda: "127.0.0.1")
    monkeypatch.setattr(gloo, "_store_addrs", {})


@pytest.fixture
def fake_pg_factory(monkeypatch):
    """Stub ProcessGroup creation for tests where no rank 1 joins.

    Returns the list of stores the factory was called with.
    """
    created = []

    def _factory(store, rank, world_size, timeout, device=None):
        created.append(store)
        return object()

    monkeypatch.setattr(gloo, "_gloo_factory", _factory)
    return created


class TestMasterStorePort:
    @pytest.mark.asyncio
    async def test_master_store_binds_os_assigned_port(
        self, monkeypatch, fake_pg_factory
    ):
        store_kwargs = []

        def recording_tcp_store(**kwargs):
            store_kwargs.append(kwargs)
            return TCPStore(**kwargs)

        monkeypatch.setattr(gloo, "TCPStore", recording_tcp_store)
        buffer = GlooTransportBuffer(MockStorageVolumeRef())

        await buffer._pre_handshake()
        await buffer._pg_task

        # The OS chooses the port during the bind; no port is chosen beforehand.
        assert len(store_kwargs) == 1
        assert store_kwargs[0]["is_master"]
        assert store_kwargs[0]["port"] == 0
        # The port sent to the storage volume is the port the store is bound
        # to, and a worker dialing it reaches this store.
        assert buffer.master_port == buffer._tcp_store.port
        assert buffer.master_port > 0
        remote = pickle.loads(pickle.dumps(buffer))
        assert remote.master_port == buffer.master_port
        _connect_worker(remote.master_port).set("probe", "ok")
        assert buffer._tcp_store.get("probe") == b"ok"
        assert fake_pg_factory == [buffer._tcp_store]

    @pytest.mark.asyncio
    async def test_concurrent_handshakes_get_distinct_live_ports(self, fake_pg_factory):
        buffers = [GlooTransportBuffer(MockStorageVolumeRef()) for _ in range(4)]

        await asyncio.gather(*(b._pre_handshake() for b in buffers))
        await asyncio.gather(*(b._pg_task for b in buffers))

        ports = [b.master_port for b in buffers]
        assert len(set(ports)) == len(ports)
        # Each store serves the port it reported.
        for i, buffer in enumerate(buffers):
            assert buffer._tcp_store.port == buffer.master_port
            _connect_worker(buffer.master_port).set("id", str(i))
            assert buffer._tcp_store.get("id") == str(i).encode()

    @pytest.mark.asyncio
    async def test_peer_rendezvous_and_transfer(self, monkeypatch):
        """End to end with a real gloo ProcessGroup on both sides."""
        # Bound the rendezvous so a regression fails instead of hanging.
        monkeypatch.setattr(gloo, "TORCHSTORE_GLOO_INIT_TIMEOUT", 20)
        ref = MockStorageVolumeRef()
        buffer = GlooTransportBuffer(ref)

        await buffer._pre_handshake()
        bound_port = buffer._tcp_store.port
        # The handshake RPC ships a pickled copy of the buffer to the volume.
        remote = pickle.loads(pickle.dumps(buffer))
        volume_ctx = TransportContext()
        await remote.recv_handshake(volume_ctx, [])
        await buffer._post_handshake([None], [])

        assert gloo._store_addrs[ref.volume_id][1] == bound_port
        sent = torch.arange(16, dtype=torch.float32)
        received = torch.zeros(16, dtype=torch.float32)
        await asyncio.gather(
            buffer._send_tensor(sent, ref.transport_context),
            remote._receive_tensor(received, volume_ctx),
        )
        assert torch.equal(received, sent)
