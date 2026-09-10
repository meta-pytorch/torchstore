# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest
from typing import cast

from torchstore.tcpstore_control_plane import (
    GenerationManifest,
    ObjectRef,
    SessionConfig,
    TCPStoreEndpoint,
    ValidationError,
)


class C10dGenerationCoordinatorTypesTest(unittest.TestCase):
    def test_endpoint_round_trip_is_canonical(self) -> None:
        endpoint = TCPStoreEndpoint("trainer.example", 1234, "run-a", 5000)

        encoded = endpoint.to_bytes()

        self.assertEqual(endpoint, TCPStoreEndpoint.from_bytes(encoded))
        self.assertEqual(encoded, TCPStoreEndpoint.from_bytes(encoded).to_bytes())

    def test_session_membership_is_canonical_and_fixed(self) -> None:
        config = SessionConfig("run-a", "trainer", ("inference-b", "inference-a"))

        self.assertEqual(("inference-a", "inference-b"), config.consumer_ids)
        self.assertEqual(config, SessionConfig.from_bytes(config.to_bytes()))

    def test_manifest_round_trip_is_canonical_and_digest_stable(self) -> None:
        manifest = GenerationManifest(
            run_id="run-a",
            generation=7,
            application_version=42,
            objects=(
                ObjectRef("z.weight", "rdma://trainer/z", 64),
                ObjectRef("a.weight", "rdma://trainer/a", 32, "a" * 64),
            ),
        )

        decoded = GenerationManifest.from_bytes(manifest.to_bytes())

        self.assertEqual(manifest, decoded)
        self.assertEqual(manifest.digest, decoded.digest)
        self.assertEqual(
            ("a.weight", "z.weight"),
            tuple(item.logical_key for item in decoded.objects),
        )

    def test_unknown_duplicate_and_noncanonical_fields_fail_closed(self) -> None:
        endpoint = TCPStoreEndpoint("localhost", 1234, "run-a")
        unknown = endpoint.to_bytes()[:-1] + b',"unknown":1}'
        duplicate = b'{"host":"a","host":"b"}'

        with self.assertRaises(ValidationError):
            TCPStoreEndpoint.from_bytes(unknown)
        with self.assertRaisesRegex(ValidationError, "duplicate"):
            TCPStoreEndpoint.from_bytes(duplicate)
        with self.assertRaises(ValidationError):
            GenerationManifest.from_bytes(b'{"generation":1.0}')

    def test_invalid_membership_and_objects_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValidationError, "unique"):
            SessionConfig("run-a", "trainer", ("consumer", "consumer"))
        with self.assertRaisesRegex(ValidationError, "cannot also"):
            SessionConfig("run-a", "trainer", ("trainer",))
        with self.assertRaisesRegex(ValidationError, "unique"):
            GenerationManifest(
                "run-a",
                1,
                1,
                (
                    ObjectRef("weight", "rdma://left", 1),
                    ObjectRef("weight", "rdma://right", 1),
                ),
            )
        with self.assertRaisesRegex(ValidationError, "SHA-256"):
            ObjectRef("weight", "rdma://trainer", 1, "not-a-digest")

    def test_direct_construction_rejects_incorrect_container_types(self) -> None:
        with self.assertRaisesRegex(ValidationError, "tuple"):
            SessionConfig("run-a", "trainer", cast(tuple[str, ...], "consumer"))
        with self.assertRaisesRegex(ValidationError, "ObjectRef"):
            GenerationManifest(
                "run-a",
                1,
                1,
                cast(tuple[ObjectRef, ...], ("not-an-object-reference",)),
            )
        with self.assertRaisesRegex(ValidationError, "SHA-256"):
            ObjectRef("weight", "rdma://trainer", 1, cast(str, 1))
        with self.assertRaisesRegex(ValidationError, "bounded"):
            GenerationManifest(
                "run-a",
                1,
                1,
                tuple(
                    ObjectRef(str(index), "rdma://trainer", 1) for index in range(4097)
                ),
            )

    def test_deeply_nested_json_raises_validation_error(self) -> None:
        nested = b"[" * 1000 + b"0" + b"]" * 1000

        with self.assertRaises(ValidationError):
            TCPStoreEndpoint.from_bytes(nested)
