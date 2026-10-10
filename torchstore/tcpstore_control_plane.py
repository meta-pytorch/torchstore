# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Minimal scheduler-neutral metadata contract for a TCPStore proof of concept."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import cast, NoReturn, TypeAlias


SCHEMA_VERSION = 1
MAX_DOCUMENT_BYTES = 1024 * 1024
MAX_ENDPOINT_BYTES = 4096
MAX_OBJECTS = 4096
MAX_CONSUMERS = 4096
MAX_GENERATIONS_PER_SESSION = 16
MAX_INTEGER = (1 << 63) - 1
_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_CHECKSUM_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_JsonObject: TypeAlias = dict[str, object]


class ControlPlaneError(Exception):
    """Base class for proof-of-concept control-plane failures."""


class ValidationError(ControlPlaneError, ValueError):
    """A wire value or operation argument is invalid."""


class ConflictError(ControlPlaneError):
    """An immutable value conflicts with existing state."""


class NotReadyError(ControlPlaneError):
    """A lifecycle precondition has not been reached."""


@dataclass(frozen=True, slots=True)
class TCPStoreEndpoint:
    """Connection data distributed by an external scheduler."""

    host: str
    port: int
    namespace: str
    timeout_milliseconds: int = 30_000
    security_mode: str = "trusted_network_only"

    def __post_init__(self) -> None:
        _validate_text(self.host, "host", 255)
        if self.host in {"0.0.0.0", "::"}:
            raise ValidationError("host must be a connectable address")
        _validate_integer(self.port, "port", minimum=1, maximum=65_535)
        _validate_identifier(self.namespace, "namespace")
        _validate_integer(
            self.timeout_milliseconds,
            "timeout_milliseconds",
            minimum=1,
            maximum=86_400_000,
        )
        if self.security_mode != "trusted_network_only":
            raise ValidationError("the POC supports trusted networks only")

    def to_bytes(self) -> bytes:
        return _encode_document(
            {
                "host": self.host,
                "namespace": self.namespace,
                "port": self.port,
                "schema_version": SCHEMA_VERSION,
                "security_mode": self.security_mode,
                "timeout_milliseconds": self.timeout_milliseconds,
            },
            "TCPStore endpoint",
            MAX_ENDPOINT_BYTES,
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> TCPStoreEndpoint:
        document = _decode_document(data, "TCPStore endpoint", MAX_ENDPOINT_BYTES)
        _require_fields(
            document,
            {
                "host",
                "namespace",
                "port",
                "schema_version",
                "security_mode",
                "timeout_milliseconds",
            },
            "TCPStore endpoint",
        )
        _require_schema_version(document)
        return cls(
            host=_expect_string(document["host"], "host"),
            port=_expect_integer(document["port"], "port"),
            namespace=_expect_string(document["namespace"], "namespace"),
            timeout_milliseconds=_expect_integer(
                document["timeout_milliseconds"], "timeout_milliseconds"
            ),
            security_mode=_expect_string(document["security_mode"], "security_mode"),
        )


@dataclass(frozen=True, slots=True)
class SessionConfig:
    """Fixed membership for one externally orchestrated handoff session."""

    run_id: str
    publisher_id: str
    consumer_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _validate_identifier(self.run_id, "run_id")
        _validate_identifier(self.publisher_id, "publisher_id")
        if not isinstance(self.consumer_ids, tuple):
            raise ValidationError("consumer_ids must be a tuple")
        if not self.consumer_ids or len(self.consumer_ids) > MAX_CONSUMERS:
            raise ValidationError("consumer_ids must be a bounded nonempty tuple")
        for consumer_id in self.consumer_ids:
            _validate_identifier(consumer_id, "consumer_id")
        consumers = tuple(sorted(self.consumer_ids))
        if len(consumers) != len(set(consumers)):
            raise ValidationError("consumer_ids must be unique")
        if self.publisher_id in consumers:
            raise ValidationError("publisher_id cannot also be a consumer")
        object.__setattr__(self, "consumer_ids", consumers)

    def to_bytes(self) -> bytes:
        return _encode_document(
            {
                "consumer_ids": list(self.consumer_ids),
                "publisher_id": self.publisher_id,
                "run_id": self.run_id,
                "schema_version": SCHEMA_VERSION,
            },
            "session config",
            MAX_DOCUMENT_BYTES,
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> SessionConfig:
        document = _decode_document(data, "session config", MAX_DOCUMENT_BYTES)
        _require_fields(
            document,
            {"consumer_ids", "publisher_id", "run_id", "schema_version"},
            "session config",
        )
        _require_schema_version(document)
        consumer_values = _expect_list(document["consumer_ids"], "consumer_ids")
        return cls(
            run_id=_expect_string(document["run_id"], "run_id"),
            publisher_id=_expect_string(document["publisher_id"], "publisher_id"),
            consumer_ids=tuple(
                _expect_string(value, "consumer_id") for value in consumer_values
            ),
        )


@dataclass(frozen=True, slots=True)
class ObjectRef:
    """Opaque data-plane location published through the control plane."""

    logical_key: str
    location: str
    size_bytes: int
    checksum: str | None = None

    def __post_init__(self) -> None:
        _validate_text(self.logical_key, "logical_key", 256)
        _validate_text(self.location, "location", 1024)
        _validate_integer(self.size_bytes, "size_bytes", minimum=0, maximum=MAX_INTEGER)
        if self.checksum is not None:
            if (
                not isinstance(self.checksum, str)
                or _CHECKSUM_PATTERN.fullmatch(self.checksum) is None
            ):
                raise ValidationError("checksum must be a lowercase SHA-256 digest")

    def _to_document(self) -> _JsonObject:
        return {
            "checksum": self.checksum,
            "location": self.location,
            "logical_key": self.logical_key,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def _from_document(cls, document: _JsonObject) -> ObjectRef:
        _require_fields(
            document,
            {"checksum", "location", "logical_key", "size_bytes"},
            "object reference",
        )
        checksum = document["checksum"]
        if checksum is not None:
            checksum = _expect_string(checksum, "checksum")
        return cls(
            logical_key=_expect_string(document["logical_key"], "logical_key"),
            location=_expect_string(document["location"], "location"),
            size_bytes=_expect_integer(document["size_bytes"], "size_bytes"),
            checksum=checksum,
        )


@dataclass(frozen=True, slots=True)
class GenerationManifest:
    """One complete immutable generation of opaque object references."""

    run_id: str
    generation: int
    application_version: int
    objects: tuple[ObjectRef, ...]

    def __post_init__(self) -> None:
        _validate_identifier(self.run_id, "run_id")
        _validate_integer(
            self.generation,
            "generation",
            minimum=1,
            maximum=MAX_GENERATIONS_PER_SESSION,
        )
        _validate_integer(
            self.application_version,
            "application_version",
            minimum=0,
            maximum=MAX_INTEGER,
        )
        if not isinstance(self.objects, tuple):
            raise ValidationError("objects must be a tuple of ObjectRef values")
        if not self.objects or len(self.objects) > MAX_OBJECTS:
            raise ValidationError("objects must be a bounded nonempty tuple")
        if not all(isinstance(item, ObjectRef) for item in self.objects):
            raise ValidationError("objects must be a tuple of ObjectRef values")
        objects = tuple(sorted(self.objects, key=lambda item: item.logical_key))
        keys = tuple(item.logical_key for item in objects)
        if len(keys) != len(set(keys)):
            raise ValidationError("manifest logical keys must be unique")
        object.__setattr__(self, "objects", objects)
        self.to_bytes()

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    def to_bytes(self) -> bytes:
        return _encode_document(
            {
                "application_version": self.application_version,
                "generation": self.generation,
                "objects": [item._to_document() for item in self.objects],
                "run_id": self.run_id,
                "schema_version": SCHEMA_VERSION,
            },
            "generation manifest",
            MAX_DOCUMENT_BYTES,
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> GenerationManifest:
        document = _decode_document(data, "generation manifest", MAX_DOCUMENT_BYTES)
        _require_fields(
            document,
            {
                "application_version",
                "generation",
                "objects",
                "run_id",
                "schema_version",
            },
            "generation manifest",
        )
        _require_schema_version(document)
        object_values = _expect_list(document["objects"], "objects")
        return cls(
            run_id=_expect_string(document["run_id"], "run_id"),
            generation=_expect_integer(document["generation"], "generation"),
            application_version=_expect_integer(
                document["application_version"], "application_version"
            ),
            objects=tuple(
                ObjectRef._from_document(_expect_object(item, "object reference"))
                for item in object_values
            ),
        )


def _encode_document(document: _JsonObject, label: str, maximum_bytes: int) -> bytes:
    try:
        encoded = json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as error:
        raise ValidationError(f"{label} cannot be encoded") from error
    if not encoded or len(encoded) > maximum_bytes:
        raise ValidationError(f"{label} exceeds its encoded size limit")
    return encoded


def _decode_document(data: bytes, label: str, maximum_bytes: int) -> _JsonObject:
    if not isinstance(data, bytes) or not data or len(data) > maximum_bytes:
        raise ValidationError(f"{label} is empty or exceeds its size limit")
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_float=_reject_json_number,
            parse_constant=_reject_json_number,
        )
    except ValidationError:
        raise
    except (json.JSONDecodeError, ValueError, RecursionError) as error:
        raise ValidationError(f"{label} is not valid JSON") from error
    return _expect_object(value, label)


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> _JsonObject:
    document: _JsonObject = {}
    for key, value in pairs:
        if key in document:
            raise ValidationError(f"duplicate field: {key}")
        document[key] = value
    return document


def _reject_json_number(value: str) -> NoReturn:
    raise ValidationError(f"unsupported JSON number: {value}")


def _require_schema_version(document: _JsonObject) -> None:
    if _expect_integer(document["schema_version"], "schema_version") != SCHEMA_VERSION:
        raise ValidationError("unsupported schema version")


def _require_fields(document: _JsonObject, expected: set[str], label: str) -> None:
    if set(document) != expected:
        raise ValidationError(f"{label} has missing or unknown fields")


def _expect_object(value: object, name: str) -> _JsonObject:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValidationError(f"{name} must be an object")
    return cast(_JsonObject, value)


def _expect_list(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValidationError(f"{name} must be a list")
    return value


def _expect_string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"{name} must be a string")
    return value


def _expect_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{name} must be an integer")
    return value


def _validate_identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or _IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise ValidationError(f"{name} is not a valid identifier")


def _validate_text(value: str, name: str, maximum_length: int) -> None:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > maximum_length
        or any(ord(character) < 32 for character in value)
    ):
        raise ValidationError(f"{name} is not valid bounded text")


def _validate_integer(value: int, name: str, *, minimum: int, maximum: int) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or value > maximum
    ):
        raise ValidationError(f"{name} is outside its supported range")


__all__ = [
    "ConflictError",
    "ControlPlaneError",
    "GenerationManifest",
    "NotReadyError",
    "ObjectRef",
    "SessionConfig",
    "TCPStoreEndpoint",
    "ValidationError",
]
