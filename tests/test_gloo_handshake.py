# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The Gloo handshake with both halves in this process."""

import pickle
import socket

import pytest
import torch
import torchstore.transport.gloo as gloo
from torchstore.strategy import StorageVolumeRef
from torchstore.transport.buffers import TransportContext
from torchstore.transport.gloo import GlooProcessGroupCache, GlooTransportBuffer

pytestmark = [
    pytest.mark.skipif(not gloo.gloo_available(), reason="Gloo transport is off"),
    pytest.mark.asyncio,
    pytest.mark.timeout(60),
]


@pytest.fixture(autouse=True)
def _short_gloo_timeout(monkeypatch):
    # Bounds how long teardown waits for a pairing thread that a failed case leaves behind.
    monkeypatch.setattr(gloo, "TORCHSTORE_GLOO_INIT_TIMEOUT", 10)


def _client(volume_id: str) -> GlooTransportBuffer:
    return GlooTransportBuffer(StorageVolumeRef(None, volume_id, None, None))


async def test_two_handshakes_pair_through_the_ports_their_stores_bound(
    monkeypatch, request
):
    """Another socket binds the port each client store asks for just before that store binds."""
    real_store = gloo.TCPStore

    def store_after_another_bind(**kwargs):
        if kwargs["is_master"]:
            other = socket.socket()
            request.addfinalizer(other.close)
            other.bind(("", kwargs["port"]))
        return real_store(**kwargs)

    monkeypatch.setattr(gloo, "TCPStore", store_after_another_bind)
    clients = [_client("volume-0"), _client("volume-1")]
    for client in clients:
        await client._pre_handshake()
    # The copies the handshake RPC carries to the volumes.
    halves = [pickle.loads(pickle.dumps(client)) for client in clients]
    bound = [client._tcp_store.port for client in clients]
    assert [half.master_port for half in halves] == bound
    assert bound[0] != bound[1]

    sent = [torch.full((4,), 1.0), torch.full((4,), 2.0)]
    for client, half, tensor in zip(clients, halves, sent):
        volume_ctx = TransportContext()
        request.addfinalizer(volume_ctx.clear)
        await half.recv_handshake(volume_ctx, [])
        client_group = await client._pg_task
        volume_group = volume_ctx.get(GlooProcessGroupCache).get(client.store_key)
        request.addfinalizer(client_group.shutdown)
        request.addfinalizer(volume_group.shutdown)
        received = torch.zeros(4)
        works = [
            client_group.send([tensor], dstRank=1, tag=0),
            volume_group.recv([received], srcRank=0, tag=0),
        ]
        for work in works:
            work.wait()
        assert torch.equal(received, tensor)
