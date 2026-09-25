# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the NeuronEFA transport.

These run anywhere: no Trainium, no EFA, no vLLM. The transport itself never touches
tensors -- the bytes move by RDMA between the trainer's source slab and the generator's HBM,
outside this object -- so what is testable is exactly what is worth testing: the ORDER of the
bracket, and every check that stands between a caller and silently writing to the wrong place.

The fakes record a call log. Most assertions are about that log, because on this protocol
ordering IS correctness: committing before the trainer releases its source lets the engine
serve a request from weights whose source is about to be freed.
"""

import inspect
import re

import pytest
from torchstore.transport import TransportType
from torchstore.transport.buffers import TransportContext
from torchstore.transport.neuron_efa import (
    NeuronEFATransportBuffer,
    NeuronEFAVolumeCache,
    NeuronGeneratorVolume,
    NeuronWeightSource,
)
from torchstore.transport.types import Request

SOURCE_GENERATION = "gen-0d6d202f"
LAYOUT = "ca8611202e7b6b0512c961909c242f36151ec3febef073d165949ec57250ddc5"
CONSUMER_IDS = [f"generator-0-rank-{rank}" for rank in range(8)]


class FakeTrainer:
    """The client side of the bracket: pins, fences and frees source regions."""

    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.attempt_ids: list[str] = []
        self.released_acks: list[list] = []
        self.attempt_id_to_mint = "attempt-7cc2b357"
        self.manifests_to_mint: list[dict] | None = None

    async def prepare_version(self, version, replica_index):
        self.calls.append(f"prepare(v{version},r{replica_index})")
        self.attempt_ids.append(self.attempt_id_to_mint)
        manifests = self.manifests_to_mint
        if manifests is None:
            manifests = [{"rank": rank, "bytes": 3692820480} for rank in range(8)]
        return self.attempt_id_to_mint, manifests

    async def begin_pull(self, version, attempt_id, consumer_ids, replica_index):
        self.calls.append(f"begin_pull(v{version},{attempt_id},n={len(consumer_ids)})")

    async def release_version(self, version, attempt_id, consumer_acks, replica_index):
        self.calls.append(f"release(v{version},{attempt_id})")
        self.released_acks.append(list(consumer_acks))


class FakeGenerator:
    """The volume: its HBM regions are the storage."""

    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.committed_version = -1
        self.source_generation = SOURCE_GENERATION
        self.layout_sha256 = LAYOUT
        self.state = "ACTIVE"
        self.put_raises: Exception | None = None

    async def ts_handshake(self, source_generation, current_policy_version=-1):
        self.calls.append(
            f"ts_handshake({source_generation},v={current_policy_version})"
        )
        return {
            "model_id": "Qwen/Qwen3-14B",
            "manifests": [{"rank": rank} for rank in range(8)],
            "transport": "neuron_efa",
            "provider": "efa",
            "library_sha256": "9a96d0eaf033",
            "metadata": {
                "schema_version": 1,
                "tp_size": 8,
                "layout_sha256": self.layout_sha256,
                "source_generation": self.source_generation,
                "policy_version": current_policy_version,
            },
        }

    async def ts_put(self, manifests, policy_version, attempt_id, slot=0):
        self.calls.append(f"ts_put(v{policy_version},{attempt_id},slot={slot})")
        if self.put_raises is not None:
            raise self.put_raises
        if policy_version != self.committed_version + 1:
            raise RuntimeError(
                f"non-consecutive policy version {policy_version}; "
                f"committed is {self.committed_version}"
            )
        self.state = "QUIESCED"
        return {
            "ok": True,
            "state": "QUIESCED",
            "consumer_acks": [{"rank": rank, "ok": True} for rank in range(8)],
            "logical_bytes": 29542563840,
        }

    async def ts_commit(self, policy_version, attempt_id):
        self.calls.append(f"ts_commit(v{policy_version},{attempt_id})")
        if self.state != "QUIESCED":
            raise RuntimeError(
                f"commit requires a quiesced engine, state is {self.state}"
            )
        if policy_version != self.committed_version + 1:
            raise RuntimeError(f"non-consecutive commit {policy_version}")
        self.committed_version = policy_version
        self.state = "ACTIVE"
        return {
            "ok": True,
            "state": "ACTIVE",
            "committed_ranks": list(range(8)),
        }


class FakeVolumeEndpoint:
    """Stands in for a monarch endpoint: ``volume.put.call(buffer, requests)``."""

    def __init__(self, volume: "FakeStorageVolume", name: str) -> None:
        self._volume = volume
        self._name = name

    async def call(self, transport_buffer, requests):
        return await self._volume.dispatch(self._name, transport_buffer, requests)

    async def call_one(self, transport_buffer, requests):
        return await self._volume.dispatch(self._name, transport_buffer, requests)


class FakeStorageVolume:
    """A StorageImpl that hosts a Neuron generator.

    Mirrors what the real one does: publish the generator into the TransportContext, then let
    the buffer's volume-side handlers reach it from there.
    """

    def __init__(self, generator: FakeGenerator) -> None:
        self.transport_context = TransportContext()
        self.transport_context.get(NeuronEFAVolumeCache).generator = generator
        self.handshake = FakeVolumeEndpoint(self, "handshake")
        self.put = FakeVolumeEndpoint(self, "put")
        self.get = FakeVolumeEndpoint(self, "get")

    async def dispatch(self, name, transport_buffer, requests):
        pairs = [(request, None) for request in requests]
        if name == "handshake":
            return await transport_buffer.recv_handshake(self.transport_context, pairs)
        if name == "put":
            return await transport_buffer.handle_put_request(
                self.transport_context, pairs
            )
        return await transport_buffer.handle_get_request(self.transport_context, pairs)


class FakeStorageVolumeRef:
    # A hostname that is never the local one: the SharedMemory locality probe runs during
    # transport resolution, and a generator's HBM is not reachable as local shared memory
    # even when the pod happens to share a host with the trainer.
    volume_hostname = "not-this-host.invalid"

    def __init__(self, volume: FakeStorageVolume) -> None:
        self.volume = volume


def build(version=0, consumer_ids=None, committed_version=-1):
    """Wire a buffer to a fake trainer and generator. Returns (buffer, trainer, gen, calls)."""
    calls: list[str] = []
    trainer = FakeTrainer(calls)
    generator = FakeGenerator(calls)
    generator.committed_version = committed_version
    buffer = NeuronEFATransportBuffer(
        FakeStorageVolumeRef(FakeStorageVolume(generator))
    )
    buffer.bind_sync(
        source=trainer,
        generator=generator,
        source_generation=SOURCE_GENERATION,
        policy_version=version,
        consumer_ids=CONSUMER_IDS if consumer_ids is None else consumer_ids,
        replica_index=0,
    )
    return buffer, trainer, generator, calls


# ----------------------------------------------------------------------
# The happy path, and the ordering that makes it correct
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_full_bracket_runs_in_protocol_order():
    buffer, trainer, generator, calls = build(version=0)

    await buffer.put_to_storage_volume([Request(key="policy")])

    assert calls == [
        "ts_handshake(gen-0d6d202f,v=-1)",
        "prepare(v0,r0)",
        "begin_pull(v0,attempt-7cc2b357,n=8)",
        "ts_put(v0,attempt-7cc2b357,slot=0)",
        "release(v0,attempt-7cc2b357)",
        "ts_commit(v0,attempt-7cc2b357)",
    ]
    assert generator.committed_version == 0
    assert generator.state == "ACTIVE"


@pytest.mark.asyncio
async def test_source_is_released_before_the_generator_commits():
    """The single most important ordering property in this protocol.

    Committing first would let the engine admit a request against weights whose source the
    trainer is about to free.
    """
    buffer, _trainer, _generator, calls = build(version=0)

    await buffer.put_to_storage_volume([Request(key="policy")])

    assert calls.index("release(v0,attempt-7cc2b357)") < calls.index(
        "ts_commit(v0,attempt-7cc2b357)"
    )


@pytest.mark.asyncio
async def test_source_is_pinned_and_fenced_before_any_pull():
    """Nothing may read a source that is not yet pinned, or be fenced before it exists."""
    buffer, _trainer, _generator, calls = build(version=0)

    await buffer.put_to_storage_volume([Request(key="policy")])

    assert (
        calls.index("prepare(v0,r0)")
        < calls.index("begin_pull(v0,attempt-7cc2b357,n=8)")
        < calls.index("ts_put(v0,attempt-7cc2b357,slot=0)")
    )


@pytest.mark.asyncio
async def test_release_carries_the_consumer_acks_from_the_pull():
    """The refcount is what lets a mesh share one exported version."""
    buffer, trainer, _generator, _calls = build(version=0)

    await buffer.put_to_storage_volume([Request(key="policy")])

    assert len(trainer.released_acks) == 1
    assert len(trainer.released_acks[0]) == 8


@pytest.mark.asyncio
async def test_consecutive_versions_advance_the_generator():
    calls: list[str] = []
    trainer = FakeTrainer(calls)
    generator = FakeGenerator(calls)
    ref = FakeStorageVolumeRef(FakeStorageVolume(generator))

    for version in (0, 1, 2):
        trainer.attempt_id_to_mint = f"attempt-{version}"
        buffer = NeuronEFATransportBuffer(ref)
        buffer.bind_sync(
            source=trainer,
            generator=generator,
            source_generation=SOURCE_GENERATION,
            policy_version=version,
            consumer_ids=CONSUMER_IDS,
            session_bound=version > 0,
        )
        await buffer.put_to_storage_volume([Request(key="policy")])

    assert generator.committed_version == 2
    # Bound ONCE. Rebinding with a later version is not idempotent on the engine; it
    # demands cold actor reconstruction, so a per-version handshake would refuse to sync.
    assert [c for c in calls if c.startswith("ts_handshake")] == [
        "ts_handshake(gen-0d6d202f,v=-1)"
    ]


# ----------------------------------------------------------------------
# Every guard that stands between a caller and writing to the wrong place
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_skipped_version_is_rejected():
    """Versions are a sync sequence. A gap means a publish was lost."""
    buffer, _trainer, _generator, _calls = build(version=5, committed_version=-1)

    with pytest.raises(RuntimeError, match="non-consecutive policy version"):
        await buffer.put_to_storage_volume([Request(key="policy")])


@pytest.mark.asyncio
async def test_generation_mismatch_is_caught_before_any_bytes_move():
    """A generator bound to another trainer's export must not be written to."""
    buffer, _trainer, generator, calls = build(version=0)
    generator.source_generation = "gen-somebody-else"

    with pytest.raises(RuntimeError, match="different export generation"):
        await buffer.put_to_storage_volume([Request(key="policy")])

    assert not any(call.startswith("ts_put") for call in calls)


@pytest.mark.asyncio
async def test_missing_layout_digest_is_rejected():
    """Without it a TP layout mismatch would not be detected before bytes move."""
    buffer, _trainer, generator, calls = build(version=0)
    generator.layout_sha256 = ""

    with pytest.raises(RuntimeError, match="no layout_sha256"):
        await buffer.put_to_storage_volume([Request(key="policy")])

    assert not any(call.startswith("ts_put") for call in calls)


@pytest.mark.asyncio
async def test_trainer_without_attempt_id_is_rejected():
    buffer, trainer, _generator, calls = build(version=0)
    trainer.attempt_id_to_mint = ""

    with pytest.raises(RuntimeError, match="no attempt_id"):
        await buffer.put_to_storage_volume([Request(key="policy")])

    assert not any(call.startswith("ts_put") for call in calls)


@pytest.mark.asyncio
async def test_trainer_without_manifests_is_rejected():
    buffer, trainer, _generator, _calls = build(version=0)
    trainer.manifests_to_mint = []

    with pytest.raises(RuntimeError, match="no source manifests"):
        await buffer.put_to_storage_volume([Request(key="policy")])


@pytest.mark.asyncio
async def test_failed_pull_does_not_release_the_source_or_commit():
    """If the pull failed the source must stay pinned: the generator may still hold it."""
    buffer, _trainer, generator, calls = build(version=0)
    generator.put_raises = RuntimeError("rank 3 transfer failed")

    with pytest.raises(RuntimeError, match="rank 3 transfer failed"):
        await buffer.put_to_storage_volume([Request(key="policy")])

    assert not any(call.startswith("release") for call in calls)
    assert not any(call.startswith("ts_commit") for call in calls)


@pytest.mark.asyncio
async def test_unbound_buffer_refuses_to_transfer():
    generator = FakeGenerator([])
    buffer = NeuronEFATransportBuffer(
        FakeStorageVolumeRef(FakeStorageVolume(generator))
    )

    with pytest.raises(RuntimeError, match="bind_sync must be called"):
        await buffer.put_to_storage_volume([Request(key="policy")])


@pytest.mark.asyncio
async def test_volume_without_a_registered_generator_is_a_clear_error():
    calls: list[str] = []
    volume = FakeStorageVolume(FakeGenerator(calls))
    volume.transport_context.get(NeuronEFAVolumeCache).generator = None
    buffer = NeuronEFATransportBuffer(FakeStorageVolumeRef(volume))
    buffer.bind_sync(
        source=FakeTrainer(calls),
        generator=FakeGenerator(calls),
        source_generation=SOURCE_GENERATION,
        policy_version=0,
        consumer_ids=CONSUMER_IDS,
    )

    with pytest.raises(RuntimeError, match="no Neuron generator is registered"):
        await buffer.put_to_storage_volume([Request(key="policy")])


# ----------------------------------------------------------------------
# bind_sync argument validation
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"consumer_ids": []}, "consumer_ids must name this replica's ranks"),
        (
            {"consumer_ids": ["generator-0-rank-0", "generator-0-rank-0"]},
            "contains duplicates",
        ),
        ({"source_generation": ""}, "source_generation must be a non-empty string"),
        ({"policy_version": -1}, "policy_version must be non-negative"),
        ({"replica_index": -1}, "replica_index must be non-negative"),
    ],
)
def test_bind_sync_rejects_bad_arguments(kwargs, message):
    calls: list[str] = []
    generator = FakeGenerator(calls)
    buffer = NeuronEFATransportBuffer(
        FakeStorageVolumeRef(FakeStorageVolume(generator))
    )
    args = {
        "source": FakeTrainer(calls),
        "generator": generator,
        "source_generation": SOURCE_GENERATION,
        "policy_version": 0,
        "consumer_ids": CONSUMER_IDS,
    }
    args.update(kwargs)

    with pytest.raises((ValueError, TypeError), match=re.escape(message)):
        buffer.bind_sync(**args)


def test_bind_sync_rejects_a_bool_version():
    """bool is an int in Python; a version of True would silently mean 1."""
    calls: list[str] = []
    generator = FakeGenerator(calls)
    buffer = NeuronEFATransportBuffer(
        FakeStorageVolumeRef(FakeStorageVolume(generator))
    )

    with pytest.raises(TypeError, match="policy_version must be an int"):
        buffer.bind_sync(
            source=FakeTrainer(calls),
            generator=generator,
            source_generation=SOURCE_GENERATION,
            policy_version=True,
            consumer_ids=CONSUMER_IDS,
        )


# ----------------------------------------------------------------------
# Write-only, and registration
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_get_is_refused_with_an_actionable_message():
    buffer, _trainer, _generator, _calls = build(version=0)

    with pytest.raises(NotImplementedError, match="write-only"):
        await buffer.get_from_storage_volume([Request(key="policy")])


def test_resharding_is_not_claimed():
    """The manifests pin one exact TP layout, so there is nothing to reshard."""
    assert NeuronEFATransportBuffer.supports_inplace_resharding is False


def test_batching_is_not_claimed():
    """Batching separate keys would break the quiesce/commit bracket."""
    assert NeuronEFATransportBuffer.supports_batch_puts is False
    assert NeuronEFATransportBuffer.supports_batch_gets is False


def test_resolves_only_when_asked_for_explicitly():
    """Requesting it by name works; nothing else should ever select it.

    It is not a general-purpose transport -- it applies only when the storage volume is a
    Neuron generator -- so it must stay out of get_available_transport's preference order.
    """
    from torchstore.transport import create_transport_buffer

    class Ref(FakeStorageVolumeRef):
        default_transport_type = TransportType.NeuronEFA

    ref = Ref(FakeStorageVolume(FakeGenerator([])))
    assert isinstance(create_transport_buffer(ref), NeuronEFATransportBuffer)

    source = inspect.getsource(
        __import__(
            "torchstore.transport", fromlist=["get_available_transport"]
        ).get_available_transport
    )
    assert "NeuronEFA" not in source


def test_fakes_satisfy_the_declared_protocols():
    """If the Protocols drift from what the generator actually provides, fail here."""
    calls: list[str] = []
    assert isinstance(FakeTrainer(calls), NeuronWeightSource)
    assert isinstance(FakeGenerator(calls), NeuronGeneratorVolume)


@pytest.mark.asyncio
async def test_session_bound_buffer_skips_the_handshake():
    """Every version after the first. Rebinding is not idempotent on the engine."""
    calls: list[str] = []
    trainer = FakeTrainer(calls)
    generator = FakeGenerator(calls)
    generator.committed_version = 41
    buffer = NeuronEFATransportBuffer(
        FakeStorageVolumeRef(FakeStorageVolume(generator))
    )
    buffer.bind_sync(
        source=trainer,
        generator=generator,
        source_generation=SOURCE_GENERATION,
        policy_version=42,
        consumer_ids=CONSUMER_IDS,
        session_bound=True,
    )

    await buffer.put_to_storage_volume([Request(key="policy")])

    assert not any(call.startswith("ts_handshake") for call in calls)
    assert generator.committed_version == 42


# ----------------------------------------------------------------------
# NeuronHBMStore: the real storage volume, not the test double
# ----------------------------------------------------------------------
class RealStoreRef:
    """A StorageVolumeRef whose volume is the shipped NeuronHBMStore."""

    volume_hostname = "not-this-host.invalid"

    def __init__(self, generator):
        from torchstore.neuron_hbm_store import NeuronHBMStore

        self.volume = _StoreEndpoints(NeuronHBMStore(generator))


class _StoreEndpoints:
    def __init__(self, store):
        self.handshake = _StoreEndpoint(store, "handshake")
        self.put = _StoreEndpoint(store, "put")


class _StoreEndpoint:
    def __init__(self, store, name):
        self._store, self._name = store, name

    async def call(self, transport_buffer, requests):
        return await getattr(self._store, self._name)(transport_buffer, requests)

    async def call_one(self, transport_buffer, requests):
        return await getattr(self._store, self._name)(transport_buffer, requests)


@pytest.mark.asyncio
async def test_full_bracket_through_the_shipped_storage_volume():
    """End to end against NeuronHBMStore, including the handshake.

    This is the path a user outside this repo takes, and it is the one that exercises
    recv_handshake -- the production caller binds its session separately and passes
    session_bound=True, so this test is the only coverage that handler has.
    """
    calls: list[str] = []
    trainer = FakeTrainer(calls)
    generator = FakeGenerator(calls)
    buffer = NeuronEFATransportBuffer(RealStoreRef(generator))
    buffer.bind_sync(
        source=trainer,
        generator=generator,
        source_generation=SOURCE_GENERATION,
        policy_version=0,
        consumer_ids=CONSUMER_IDS,
    )

    await buffer.put_to_storage_volume([Request(key="policy")])

    assert calls == [
        "ts_handshake(gen-0d6d202f,v=-1)",
        "prepare(v0,r0)",
        "begin_pull(v0,attempt-7cc2b357,n=8)",
        "ts_put(v0,attempt-7cc2b357,slot=0)",
        "release(v0,attempt-7cc2b357)",
        "ts_commit(v0,attempt-7cc2b357)",
    ]
    assert generator.committed_version == 0


@pytest.mark.asyncio
async def test_store_refuses_get_and_delete():
    """Its contents are the policy the engine is serving: not readable, not deletable."""
    from torchstore.neuron_hbm_store import NeuronHBMStore

    store = NeuronHBMStore(FakeGenerator([]))
    for coro in (
        store.get(None, []),
        store.get_meta([]),
        store.delete("k"),
        store.delete_batch(["k"]),
    ):
        with pytest.raises(NotImplementedError):
            await coro


def test_store_registers_the_generator_for_the_volume_side_handlers():
    """The TransportContext is the only channel those handlers have to the engine."""
    from torchstore.neuron_hbm_store import NeuronHBMStore

    generator = FakeGenerator([])
    store = NeuronHBMStore(generator)
    assert (
        store.transport_context.get(NeuronEFAVolumeCache).require_generator()
        is generator
    )
