# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import gc
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from threading import Barrier

from torch.distributed import TCPStore
from torchstore.tcpstore_control_plane import (
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
    ValidationError,
)


class C10dGenerationCoordinatorPocTest(unittest.TestCase):
    def setUp(self) -> None:
        self._coordinators: list[C10dGenerationCoordinator] = []
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
        for coordinator in reversed(self._coordinators):
            coordinator.close()
        self._coordinators.clear()
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
        coordinator = connect_tcpstore_generation_coordinator(
            endpoint, config, participant_id
        )
        self._coordinators.append(coordinator)
        return coordinator

    def test_real_tcpstore_clients_publish_and_discover_generation(self) -> None:
        manifest = self._publish(1, 42)

        discovered = tuple(consumer.discover() for consumer in self.consumers)

        self.assertEqual((manifest, manifest), discovered)
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
            coordinator = self.publisher if version == 42 else other_publisher
            barrier.wait(timeout=5)
            return coordinator.publish(
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

    def test_session_generation_count_is_explicitly_bounded(self) -> None:
        with self.assertRaisesRegex(ValidationError, "supported range"):
            self._publish(17, 42)

    def test_session_must_exist_and_first_generation_is_one(self) -> None:
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

    def test_another_generation_requires_completion(self) -> None:
        self._publish(1, 42)

        with self.assertRaisesRegex(ConflictError, "completion"):
            self._publish(2, 43)

    def test_namespace_and_membership_are_isolated(self) -> None:
        other_config = SessionConfig("other-run", "trainer", ("inference",))
        other_endpoint = TCPStoreEndpoint(
            "127.0.0.1", self.server.port, other_config.run_id, 5000
        )
        other = self._connect(other_endpoint, other_config, "trainer")

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

    def test_missing_committed_manifest_fails_closed(self) -> None:
        manifest = self._publish(1, 42)
        manifest_key = f"{self.config.run_id}/manifest/{manifest.generation:020d}"
        self.assertTrue(self.server.delete_key(manifest_key))

        with self.assertRaisesRegex(ConflictError, "missing"):
            self.consumers[0].discover()

    def test_noncanonical_committed_manifest_fails_closed(self) -> None:
        manifest = self._publish(1, 42)
        manifest_key = f"{self.config.run_id}/manifest/{manifest.generation:020d}"
        self.assertTrue(self.server.delete_key(manifest_key))
        self.server.set(manifest_key, " " + manifest.to_bytes().decode("utf-8"))

        with self.assertRaisesRegex(ConflictError, "not canonical"):
            self.consumers[0].discover()

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
