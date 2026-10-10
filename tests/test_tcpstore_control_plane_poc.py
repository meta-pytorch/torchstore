# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import gc
import unittest
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from threading import Barrier, Event, Lock
from typing import cast
from unittest.mock import patch

from torch.distributed import Store, TCPStore
from torchstore.tcpstore_control_plane import (
    AmbiguousMutationError,
    C10dGenerationCoordinator,
    ConflictError,
    connect_tcpstore_generation_coordinator,
    ConnectionUnusableError,
    GenerationManifest,
    NotReadyError,
    ObjectRef,
    PublishDisposition,
    PublishResult,
    SessionConfig,
    TCPStoreEndpoint,
    UnsupportedStoreError,
    ValidationError,
)


class C10dGenerationCoordinatorPocTest(unittest.TestCase):
    def setUp(self) -> None:
        self._control_planes: list[C10dGenerationCoordinator] = []
        self.server = TCPStore(
            host_name="127.0.0.1",
            port=0,
            world_size=None,
            is_master=True,
            timeout=timedelta(seconds=5),
            wait_for_workers=False,
        )
        self.config = SessionConfig(
            "poc-run", "trainer", ("inference-0", "inference-1")
        )
        self.endpoint = TCPStoreEndpoint(
            "127.0.0.1", self.server.port, self.config.run_id, 5000
        )
        self.publisher = self._connect(self.endpoint, self.config, "trainer")
        self.consumers = tuple(
            self._connect(self.endpoint, self.config, consumer_id)
            for consumer_id in self.config.consumer_ids
        )
        self.publisher.create_session()
        for consumer in self.consumers:
            consumer.attach()

    def tearDown(self) -> None:
        for control_plane in reversed(self._control_planes):
            control_plane.close()
        self._control_planes.clear()
        del self.consumers
        del self.publisher
        del self.server
        gc.collect()

    def _connect(
        self,
        endpoint: TCPStoreEndpoint,
        config: SessionConfig,
        participant_id: str,
    ) -> C10dGenerationCoordinator:
        control_plane = connect_tcpstore_generation_coordinator(
            endpoint, config, participant_id
        )
        self._control_planes.append(control_plane)
        return control_plane

    def _owned_store(self, control_plane: C10dGenerationCoordinator) -> Store:
        store = control_plane._store
        if store is None:
            self.fail("test control-plane handle is closed")
        return store

    def test_real_tcpstore_clients_complete_metadata_only_lifecycle(self) -> None:
        manifest = self._publish(1, 42)

        discovered = tuple(consumer.discover() for consumer in self.consumers)
        with self.assertRaises(NotReadyError):
            self.publisher.retire(manifest)
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = tuple(
                executor.submit(consumer.acknowledge_applied, manifest)
                for consumer in self.consumers
            )
            for future in futures:
                future.result()
        self.publisher.retire(manifest)
        self.publisher.retire(manifest)

        self.assertEqual((manifest, manifest), discovered)
        self.assertTrue(self.publisher.is_retired(manifest))
        self.assertIsNone(self.consumers[0].discover(after_generation=1))

    def test_publish_retry_is_idempotent_and_conflict_is_rejected(self) -> None:
        manifest = self._publish(1, 42)
        retry = self.publisher.publish(1, 42, self._objects(1))

        self.assertEqual(manifest, retry.manifest)
        self.assertEqual(PublishDisposition.COMMITTED_CURRENT, retry.disposition)
        with self.assertRaisesRegex(ConflictError, "immutable"):
            self.publisher.publish(
                1,
                43,
                (ObjectRef("weight", "rdma://trainer/other", 16),),
            )

    def test_concurrent_publishers_commit_exactly_one_manifest(self) -> None:
        other_publisher = self._connect(self.endpoint, self.config, "trainer")
        other_publisher.attach()
        barrier = Barrier(2)

        def publish(version: int) -> PublishResult:
            control_plane = self.publisher if version == 42 else other_publisher
            barrier.wait(timeout=5)
            return control_plane.publish(
                1,
                version,
                (ObjectRef("weight", f"rdma://trainer/{version}", 16),),
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = tuple(executor.submit(publish, version) for version in (42, 43))
        successes = tuple(
            future.result() for future in futures if future.exception() is None
        )
        conflicts = tuple(
            future.exception() for future in futures if future.exception() is not None
        )

        self.assertEqual(1, len(successes))
        self.assertEqual(1, len(conflicts))
        self.assertIsInstance(conflicts[0], ConflictError)
        self.assertEqual(successes[0].manifest, self.consumers[0].discover())

    def test_lost_head_response_is_safe_to_retry_exactly(self) -> None:
        self.publisher._store = cast(
            Store, _ApplyThenFailStore(self._owned_store(self.publisher), "head")
        )

        with self.assertRaises(AmbiguousMutationError):
            self._publish(1, 42)
        with self.assertRaisesRegex(ConnectionUnusableError, "reconnect"):
            self.publisher.discover()
        self.publisher = self._connect(self.endpoint, self.config, "trainer")
        self.publisher.attach()
        manifest = self.consumers[0].discover()
        if manifest is None:
            self.fail("lost response did not publish the generation")
        for consumer in self.consumers:
            consumer.acknowledge_applied(manifest)
        self.publisher.retire(manifest)
        self._publish(2, 43)

        retry = self.publisher.publish(1, 42, self._objects(1))

        self.assertEqual(manifest, retry.manifest)
        self.assertEqual(
            PublishDisposition.ALREADY_COMMITTED_HISTORICAL,
            retry.disposition,
        )

    def test_read_failure_serializes_and_poisons_concurrent_operations(self) -> None:
        manifest = self._publish(1, 42)
        consumer = self.consumers[0]
        failing_store = _FailReadWhenPeerStartsStore(self._owned_store(consumer))
        consumer._store = cast(Store, failing_store)

        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(consumer.discover)
            self.assertTrue(failing_store.first_read_started.wait(timeout=5))

            def discover_concurrently() -> GenerationManifest | None:
                failing_store.peer_started.set()
                return consumer.discover()

            second = executor.submit(discover_concurrently)
            with self.assertRaises(ConnectionUnusableError):
                first.result()
            with self.assertRaises(ConnectionUnusableError):
                second.result()

        self.assertEqual(1, failing_store.check_calls)
        consumer = self._connect(self.endpoint, self.config, "inference-0")
        consumer.attach()
        self.assertEqual(manifest, consumer.discover())

    def test_lost_ack_and_retirement_responses_are_safe_to_retry(self) -> None:
        manifest = self._publish(1, 42)
        first_consumer = self.consumers[0]
        first_consumer._store = cast(
            Store,
            _ApplyThenFailStore(
                self._owned_store(first_consumer),
                f"ack/{manifest.generation:020d}/inference-0",
            ),
        )

        with self.assertRaises(AmbiguousMutationError):
            first_consumer.acknowledge_applied(manifest)
        first_consumer = self._connect(self.endpoint, self.config, "inference-0")
        first_consumer.attach()
        first_consumer.acknowledge_applied(manifest)
        self.consumers[1].acknowledge_applied(manifest)
        self.publisher._store = cast(
            Store,
            _ApplyThenFailStore(
                self._owned_store(self.publisher),
                f"retired/{manifest.generation:020d}",
            ),
        )

        with self.assertRaises(AmbiguousMutationError):
            self.publisher.retire(manifest)
        self.publisher = self._connect(self.endpoint, self.config, "trainer")
        self.publisher.attach()
        self.publisher.retire(manifest)

        self.assertTrue(self.publisher.is_retired(manifest))

    def test_session_generation_count_is_explicitly_bounded(self) -> None:
        with self.assertRaisesRegex(ValidationError, "supported range"):
            self._publish(17, 42)

    def test_ack_and_retirement_retries_survive_head_advance(self) -> None:
        first = self._publish(1, 42)
        for consumer in self.consumers:
            consumer.acknowledge_applied(first)
        self.publisher.retire(first)
        self._publish(2, 43)

        self.consumers[0].acknowledge_applied(first)
        self.publisher.retire(first)

        self.assertTrue(self.publisher.is_retired(first))

    def test_ack_retry_reconciles_concurrent_head_advance(self) -> None:
        first = self._publish(1, 42)
        retrying_consumer = self.consumers[0]
        racing_consumer = self._connect(self.endpoint, self.config, "inference-0")
        racing_consumer.attach()
        require_current = retrying_consumer._require_current

        def complete_and_advance(manifest: GenerationManifest) -> None:
            racing_consumer.acknowledge_applied(manifest)
            self.consumers[1].acknowledge_applied(manifest)
            self.publisher.retire(manifest)
            self._publish(2, 43)
            require_current(manifest)

        with patch.object(
            retrying_consumer,
            "_require_current",
            side_effect=complete_and_advance,
        ):
            retrying_consumer.acknowledge_applied(first)

        self.assertTrue(self.publisher.is_retired(first))

    def test_retire_retry_reconciles_concurrent_head_advance(self) -> None:
        first = self._publish(1, 42)
        for consumer in self.consumers:
            consumer.acknowledge_applied(first)
        racing_publisher = self._connect(self.endpoint, self.config, "trainer")
        racing_publisher.attach()
        require_current = self.publisher._require_current

        def complete_and_advance(manifest: GenerationManifest) -> None:
            racing_publisher.retire(manifest)
            self._publish(2, 43)
            require_current(manifest)

        with patch.object(
            self.publisher,
            "_require_current",
            side_effect=complete_and_advance,
        ):
            self.publisher.retire(first)

        self.assertTrue(self.publisher.is_retired(first))

    def test_generation_advance_waits_for_retirement(self) -> None:
        first = self._publish(1, 42)

        with self.assertRaisesRegex(NotReadyError, "not retired"):
            self._publish(2, 43)
        for consumer in self.consumers:
            consumer.acknowledge_applied(first)
        self.publisher.retire(first)
        second = self._publish(2, 43)

        self.assertEqual(2, second.generation)
        self.assertEqual(second, self.consumers[0].discover(after_generation=1))

    def test_session_must_exist_and_generations_must_be_contiguous(self) -> None:
        config = SessionConfig("new-run", "trainer", ("inference",))
        participant = self._connect(
            TCPStoreEndpoint("127.0.0.1", self.server.port, config.run_id),
            config,
            "inference",
        )

        with self.assertRaisesRegex(NotReadyError, "does not exist"):
            participant.attach()
        with self.assertRaisesRegex(ConflictError, "first generation"):
            self.publisher.publish(2, 42, self._objects(2))
        first = self._publish(1, 42)
        for consumer in self.consumers:
            consumer.acknowledge_applied(first)
        self.publisher.retire(first)

        with self.assertRaisesRegex(ConflictError, "exactly one"):
            self.publisher.publish(3, 43, self._objects(3))

    def test_namespace_and_membership_are_isolated(self) -> None:
        other_config = SessionConfig("other-run", "trainer", ("inference",))
        other = self._connect(
            TCPStoreEndpoint("127.0.0.1", self.server.port, "other-run"),
            other_config,
            "trainer",
        )

        other.create_session()

        self.assertIsNone(other.discover())
        with self.assertRaisesRegex(ValidationError, "fixed membership"):
            connect_tcpstore_generation_coordinator(
                self.endpoint, self.config, "stranger"
            )
        with self.assertRaisesRegex(ValidationError, "namespace"):
            connect_tcpstore_generation_coordinator(
                TCPStoreEndpoint("127.0.0.1", self.server.port, "wrong-run"),
                self.config,
                "inference-0",
            )
        with self.assertRaisesRegex(ValidationError, "TCPStoreEndpoint"):
            connect_tcpstore_generation_coordinator(
                cast(TCPStoreEndpoint, object()), self.config, "inference-0"
            )
        with self.assertRaisesRegex(ValidationError, "SessionConfig"):
            connect_tcpstore_generation_coordinator(
                self.endpoint,
                cast(SessionConfig, object()),
                "inference-0",
            )

    def test_attach_rejects_a_different_fixed_configuration(self) -> None:
        mismatched = SessionConfig(
            self.config.run_id, "other-trainer", ("inference-0",)
        )
        participant = self._connect(self.endpoint, mismatched, "other-trainer")

        with self.assertRaisesRegex(ConflictError, "does not match"):
            participant.attach()

    def test_attach_rejects_noncanonical_stored_configuration(self) -> None:
        config_key = f"{self.config.run_id}/config"
        self.assertTrue(self.server.delete_key(config_key))
        self.server.set(
            config_key,
            " " + self.config.to_bytes().decode("ascii"),
        )
        participant = self._connect(self.endpoint, self.config, "inference-0")

        with self.assertRaisesRegex(ConflictError, "not canonical"):
            participant.attach()

    def test_roles_and_exact_handles_are_enforced(self) -> None:
        manifest = self._publish(1, 42)
        other = GenerationManifest("other-run", 1, 42, manifest.objects)

        with self.assertRaisesRegex(ValidationError, "publisher"):
            self.consumers[0].publish(2, 43, manifest.objects)
        with self.assertRaisesRegex(ValidationError, "consumer"):
            self.publisher.acknowledge_applied(manifest)
        with self.assertRaisesRegex(ConflictError, "another session"):
            self.consumers[0].acknowledge_applied(other)
        with self.assertRaisesRegex(ValidationError, "GenerationManifest"):
            self.consumers[0].acknowledge_applied(cast(GenerationManifest, object()))

    def test_missing_committed_manifest_fails_closed(self) -> None:
        manifest = self._publish(1, 42)
        self.consumers[0].acknowledge_applied(manifest)
        manifest_key = f"{self.config.run_id}/manifest/{manifest.generation:020d}"
        self.assertTrue(self.server.delete_key(manifest_key))

        operations = (
            lambda: self._publish(1, 42),
            self.consumers[0].discover,
            lambda: self.consumers[0].acknowledge_applied(manifest),
            lambda: self.publisher.retire(manifest),
            lambda: self.publisher.is_retired(manifest),
        )
        for operation in operations:
            with self.subTest(operation=operation):
                with self.assertRaises(ConflictError):
                    operation()

    def test_conflicting_acknowledgement_fails_closed(self) -> None:
        manifest = self._publish(1, 42)
        acknowledgement_key = (
            f"{self.config.run_id}/ack/{manifest.generation:020d}/inference-0"
        )
        self.server.set(acknowledgement_key, "b" * 64)
        self.consumers[1].acknowledge_applied(manifest)

        with self.assertRaisesRegex(ConflictError, "acknowledgement"):
            self.publisher.retire(manifest)

    def test_noncanonical_committed_manifest_fails_closed(self) -> None:
        manifest = self._publish(1, 42)
        manifest_key = f"{self.config.run_id}/manifest/{manifest.generation:020d}"
        self.assertTrue(self.server.delete_key(manifest_key))
        self.server.set(manifest_key, " " + manifest.to_bytes().decode("utf-8"))

        with self.assertRaisesRegex(ConflictError, "not canonical"):
            self.consumers[0].discover()

    def test_noncanonical_committed_head_fails_closed(self) -> None:
        self._publish(1, 42)
        head_key = f"{self.config.run_id}/head"
        head = self.server.get(head_key)
        self.assertTrue(self.server.delete_key(head_key))
        self.server.set(head_key, " " + head.decode("utf-8"))

        with self.assertRaisesRegex(ConflictError, "not canonical"):
            self.consumers[0].discover()

    def test_conflicting_retirement_marker_fails_closed(self) -> None:
        manifest = self._publish(1, 42)
        retired_key = f"{self.config.run_id}/retired/{manifest.generation:020d}"
        self.server.set(retired_key, "b" * 64)

        with self.assertRaisesRegex(ConflictError, "retirement marker"):
            self.publisher.is_retired(manifest)
        with self.assertRaisesRegex(ConflictError, "retirement marker"):
            self.publisher.retire(manifest)

    def test_committed_head_is_smaller_than_a_large_manifest(self) -> None:
        objects = tuple(
            ObjectRef(
                f"model.weight.{index}",
                f"rdma://trainer/generation-1/{index}",
                16,
            )
            for index in range(100)
        )
        publication = self.publisher.publish(1, 42, objects)
        manifest = publication.manifest
        head = self.server.get(f"{self.config.run_id}/head")

        self.assertLess(len(head), len(manifest.to_bytes()))
        self.assertEqual(manifest, self.consumers[0].discover())

    def test_close_is_idempotent_and_terminal(self) -> None:
        consumer = self.consumers[0]

        consumer.close()
        consumer.close()

        with self.assertRaisesRegex(ConnectionUnusableError, "closed"):
            consumer.discover()

    def test_store_without_compare_set_is_rejected_without_poisoning(self) -> None:
        consumer = self._connect(self.endpoint, self.config, "inference-0")
        consumer._store = cast(
            Store, _UnsupportedCompareSetStore(self._owned_store(consumer))
        )

        for _ in range(2):
            with self.assertRaisesRegex(UnsupportedStoreError, "compare_set"):
                consumer.attach()

    def _publish(self, generation: int, version: int) -> GenerationManifest:
        return self.publisher.publish(
            generation, version, self._objects(generation)
        ).manifest

    def _objects(self, generation: int) -> tuple[ObjectRef, ...]:
        return (
            ObjectRef(
                "model.weight",
                f"rdma://trainer/generation-{generation}",
                16,
                "a" * 64,
            ),
        )


class _ApplyThenFailStore:
    def __init__(self, store: Store, failing_key: str) -> None:
        self._store = store
        self._failing_key = failing_key
        self._failed = False

    def check(self, keys: list[str]) -> bool:
        return self._store.check(keys)

    def get(self, key: str) -> bytes:
        return self._store.get(key)

    def compare_set(self, key: str, expected: bytes, desired: bytes) -> bytes:
        compare_set = cast(
            Callable[[str, bytes, bytes], bytes], self._store.compare_set
        )
        reply = compare_set(key, expected, desired)
        if key == self._failing_key and not self._failed:
            self._failed = True
            raise RuntimeError("injected lost mutation response")
        return reply


class _FailReadWhenPeerStartsStore:
    def __init__(self, store: Store) -> None:
        self._store = store
        self._counter_lock = Lock()
        self.first_read_started = Event()
        self.peer_started = Event()
        self.check_calls = 0

    def check(self, keys: list[str]) -> bool:
        with self._counter_lock:
            self.check_calls += 1
            first = self.check_calls == 1
        if first:
            self.first_read_started.set()
            if not self.peer_started.wait(timeout=5):
                raise AssertionError("concurrent reader did not start")
        raise RuntimeError("injected read failure")

    def get(self, key: str) -> bytes:
        return self._store.get(key)

    def compare_set(self, key: str, expected: bytes, desired: bytes) -> bytes:
        compare_set = cast(
            Callable[[str, bytes, bytes], bytes], self._store.compare_set
        )
        return compare_set(key, expected, desired)


class _UnsupportedCompareSetStore:
    def __init__(self, store: Store) -> None:
        self._store = store

    def check(self, keys: list[str]) -> bool:
        return self._store.check(keys)

    def get(self, key: str) -> bytes:
        return self._store.get(key)

    def compare_set(self, key: str, expected: bytes, desired: bytes) -> bytes:
        raise NotImplementedError("compare_set is unsupported")
