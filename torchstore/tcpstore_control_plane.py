# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Scheduler-neutral metadata contract and TCPStore control-plane proof."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from threading import Lock
from types import TracebackType
from typing import cast, NoReturn, TypeAlias

from torch.distributed import PrefixStore, Store, TCPStore


SCHEMA_VERSION = 1
MAX_DOCUMENT_BYTES = 1024 * 1024
MAX_ENDPOINT_BYTES = 4096
MAX_HEAD_BYTES = 512
MAX_OBJECTS = 4096
MAX_CONSUMERS = 4096
MAX_GENERATIONS_PER_SESSION = 16
MAX_INTEGER = (1 << 63) - 1
_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_CHECKSUM_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_JsonObject: TypeAlias = dict[str, object]
_CAS_CAPABILITY_KEY = "capability/compare-set"
_CAS_CAPABILITY_VALUE = b"supported"
_CONFIG_KEY = "config"
_HEAD_KEY = "head"


class ControlPlaneError(Exception):
    """Base class for proof-of-concept control-plane failures."""


class ValidationError(ControlPlaneError, ValueError):
    """A wire value or operation argument is invalid."""


class ConflictError(ControlPlaneError):
    """An immutable value conflicts with existing state."""


class AmbiguousMutationError(ControlPlaneError):
    """A Store mutation may have applied before its response failed."""


class ConnectionUnusableError(ControlPlaneError):
    """A Store connection must be replaced before another operation."""


class NotReadyError(ControlPlaneError):
    """A lifecycle precondition has not been reached."""


class UnsupportedStoreError(ControlPlaneError):
    """The supplied c10d Store lacks a required atomic primitive."""


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


class PublishDisposition(str, Enum):
    COMMITTED_CURRENT = "committed_current"
    ALREADY_COMMITTED_HISTORICAL = "already_committed_historical"


@dataclass(frozen=True, slots=True)
class PublishResult:
    manifest: GenerationManifest
    disposition: PublishDisposition


@dataclass(frozen=True, slots=True)
class _GenerationHead:
    run_id: str
    generation: int
    manifest_digest: str

    def __post_init__(self) -> None:
        _validate_identifier(self.run_id, "run_id")
        _validate_integer(
            self.generation,
            "generation",
            minimum=1,
            maximum=MAX_GENERATIONS_PER_SESSION,
        )
        if (
            not isinstance(self.manifest_digest, str)
            or _CHECKSUM_PATTERN.fullmatch(self.manifest_digest) is None
        ):
            raise ValidationError("manifest_digest must be a lowercase SHA-256 digest")

    def to_bytes(self) -> bytes:
        return _encode_document(
            {
                "generation": self.generation,
                "manifest_digest": self.manifest_digest,
                "run_id": self.run_id,
                "schema_version": SCHEMA_VERSION,
            },
            "generation head",
            MAX_HEAD_BYTES,
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> _GenerationHead:
        document = _decode_document(data, "generation head", MAX_HEAD_BYTES)
        _require_fields(
            document,
            {"generation", "manifest_digest", "run_id", "schema_version"},
            "generation head",
        )
        _require_schema_version(document)
        return cls(
            run_id=_expect_string(document["run_id"], "run_id"),
            generation=_expect_integer(document["generation"], "generation"),
            manifest_digest=_expect_string(
                document["manifest_digest"], "manifest_digest"
            ),
        )


class C10dGenerationCoordinator:
    """Fixed-membership lifecycle requiring TCPStore compare-set semantics.

    Each instance must exclusively own its injected Store connection.
    """

    def __init__(
        self,
        store: Store,
        config: SessionConfig,
        participant_id: str,
    ) -> None:
        if not isinstance(store, Store):
            raise ValidationError("store must implement torch.distributed.Store")
        if not isinstance(config, SessionConfig):
            raise ValidationError("config must be a SessionConfig")
        _validate_identifier(participant_id, "participant_id")
        members = {config.publisher_id, *config.consumer_ids}
        if participant_id not in members:
            raise ValidationError("participant_id is not in fixed membership")
        self._store: Store | None = PrefixStore(config.run_id, store)
        self._config = config
        self._participant_id = participant_id
        self._opened = False
        self._poisoned = False
        self._closed = False
        self._io_lock = Lock()

    def __enter__(self) -> C10dGenerationCoordinator:
        self._require_usable()
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Make this handle terminal and release its Store reference."""
        with self._io_lock:
            self._opened = False
            self._closed = True
            self._store = None

    def create_session(self) -> SessionConfig:
        """Create the immutable session configuration as the publisher."""
        self._require_publisher()
        self._require_compare_set()
        self._put_immutable(_CONFIG_KEY, self._config.to_bytes())
        self._opened = True
        return self._config

    def attach(self) -> SessionConfig:
        """Attach only when the stored configuration matches exactly."""
        encoded = self._read(_CONFIG_KEY)
        if encoded is None:
            raise NotReadyError("control-plane session does not exist")
        stored = SessionConfig.from_bytes(encoded)
        if encoded != stored.to_bytes():
            raise ConflictError("session configuration is not canonical")
        if stored != self._config:
            raise ConflictError("session configuration does not match")
        self._require_compare_set()
        self._opened = True
        return stored

    def publish(
        self,
        generation: int,
        application_version: int,
        objects: Sequence[ObjectRef],
    ) -> PublishResult:
        """Publish one complete manifest, then atomically advance the head."""
        self._require_open()
        self._require_publisher()
        if not isinstance(objects, Sequence) or isinstance(
            objects, (str, bytes, bytearray)
        ):
            raise ValidationError("objects must be a bounded sequence")
        if not objects or len(objects) > MAX_OBJECTS:
            raise ValidationError("objects must be a bounded nonempty sequence")
        manifest = GenerationManifest(
            self._config.run_id,
            generation,
            application_version,
            tuple(objects),
        )
        desired = manifest.to_bytes()
        manifest_key = _manifest_key(generation)
        stored = self._read(manifest_key)
        if stored is not None and stored != desired:
            raise ConflictError("generation manifest is immutable")
        committed_head = self._read_committed_head()
        current_head = None
        committed = None
        if committed_head is not None:
            current_head, committed = committed_head
            if committed == manifest:
                return PublishResult(manifest, PublishDisposition.COMMITTED_CURRENT)
            if generation < committed.generation and stored == desired:
                return PublishResult(
                    manifest,
                    PublishDisposition.ALREADY_COMMITTED_HISTORICAL,
                )
        self._validate_next_generation(committed, generation)
        self._put_immutable(manifest_key, desired)
        desired_head = _GenerationHead(
            run_id=self._config.run_id,
            generation=generation,
            manifest_digest=manifest.digest,
        ).to_bytes()
        reply = self._compare_set(_HEAD_KEY, current_head or b"", desired_head)
        if reply == desired_head:
            return PublishResult(manifest, PublishDisposition.COMMITTED_CURRENT)
        observed = self._read_committed_head()
        if observed is not None and observed[0] == desired_head:
            return PublishResult(manifest, PublishDisposition.COMMITTED_CURRENT)
        if observed is not None and observed[1].generation > generation:
            return PublishResult(
                manifest,
                PublishDisposition.ALREADY_COMMITTED_HISTORICAL,
            )
        raise ConflictError("generation head changed concurrently")

    def discover(self, *, after_generation: int = 0) -> GenerationManifest | None:
        """Return the committed head without exposing unpublished manifests."""
        self._require_open()
        _validate_integer(
            after_generation,
            "after_generation",
            minimum=0,
            maximum=MAX_INTEGER,
        )
        committed_head = self._read_committed_head()
        if committed_head is None:
            return None
        _, manifest = committed_head
        if manifest.generation <= after_generation:
            return None
        return manifest

    def acknowledge_applied(self, manifest: GenerationManifest) -> None:
        """Record one fixed consumer's application of the exact head."""
        self._require_open()
        if self._participant_id not in self._config.consumer_ids:
            raise ValidationError("only a fixed consumer may acknowledge")
        self._require_session(manifest)
        key = _ack_key(manifest.generation, self._participant_id)
        expected = manifest.digest.encode("ascii")
        existing = self._read(key)
        if existing == expected:
            self._require_immutable_manifest(manifest)
            return
        if existing is not None:
            raise ConflictError("consumer acknowledgement is immutable")
        try:
            self._require_current(manifest)
        except ConflictError:
            existing = self._read(key)
            if existing == expected:
                self._require_immutable_manifest(manifest)
                return
            if existing is not None:
                raise ConflictError("consumer acknowledgement is immutable") from None
            raise
        self._put_immutable(key, expected)

    def retire(self, manifest: GenerationManifest) -> None:
        """Retire only after every fixed consumer applied the exact manifest."""
        self._require_open()
        self._require_publisher()
        if self.is_retired(manifest):
            return
        try:
            self._require_current(manifest)
        except ConflictError:
            if self.is_retired(manifest):
                return
            raise
        expected = manifest.digest.encode("ascii")
        for consumer_id in self._config.consumer_ids:
            acknowledgement = self._read(_ack_key(manifest.generation, consumer_id))
            if acknowledgement is None:
                raise NotReadyError("not every fixed consumer has acknowledged")
            if acknowledgement != expected:
                raise ConflictError("consumer acknowledgement conflicts")
        self._put_immutable(_retired_key(manifest.generation), expected)

    def is_retired(self, manifest: GenerationManifest) -> bool:
        """Return whether the exact generation has a retirement marker."""
        self._require_open()
        self._require_session(manifest)
        self._require_immutable_manifest(manifest)
        stored = self._read(_retired_key(manifest.generation))
        if stored is None:
            return False
        if stored != manifest.digest.encode("ascii"):
            raise ConflictError("generation retirement marker conflicts")
        return True

    def _validate_next_generation(
        self, current: GenerationManifest | None, generation: int
    ) -> None:
        if current is None:
            if generation != 1:
                raise ConflictError("the first generation must be one")
            return
        if generation != current.generation + 1:
            raise ConflictError("generation must advance by exactly one")
        if not self.is_retired(current):
            raise NotReadyError("the current generation is not retired")

    def _require_current(self, manifest: GenerationManifest) -> None:
        self._require_session(manifest)
        committed_head = self._read_committed_head()
        if committed_head is None or committed_head[1] != manifest:
            raise ConflictError("manifest is not the exact committed head")

    def _read_committed_head(
        self,
    ) -> tuple[bytes, GenerationManifest] | None:
        encoded = self._read(_HEAD_KEY)
        if encoded is None:
            return None
        head = _GenerationHead.from_bytes(encoded)
        if encoded != head.to_bytes():
            raise ConflictError("generation head is not canonical")
        if head.run_id != self._config.run_id:
            raise ConflictError("generation head belongs to another session")
        manifest_data = self._read(_manifest_key(head.generation))
        if manifest_data is None:
            raise ConflictError("committed generation manifest is missing")
        manifest = GenerationManifest.from_bytes(manifest_data)
        self._require_session(manifest)
        if manifest_data != manifest.to_bytes():
            raise ConflictError("committed generation manifest is not canonical")
        if (
            manifest.generation != head.generation
            or manifest.digest != head.manifest_digest
        ):
            raise ConflictError("generation head does not match immutable manifest")
        return encoded, manifest

    def _require_immutable_manifest(self, manifest: GenerationManifest) -> None:
        self._require_session(manifest)
        stored = self._read(_manifest_key(manifest.generation))
        if stored != manifest.to_bytes():
            raise ConflictError("generation manifest does not match immutable data")

    def _require_session(self, manifest: GenerationManifest) -> None:
        if not isinstance(manifest, GenerationManifest):
            raise ValidationError("manifest must be a GenerationManifest")
        if manifest.run_id != self._config.run_id:
            raise ConflictError("manifest belongs to another session")

    def _require_publisher(self) -> None:
        if self._participant_id != self._config.publisher_id:
            raise ValidationError("only the fixed publisher may perform this operation")

    def _require_open(self) -> None:
        self._require_usable()
        if not self._opened:
            raise NotReadyError("create or attach the session first")

    def _read(self, key: str) -> bytes | None:
        with self._io_lock:
            self._require_usable()
            store = self._store
            if store is None:
                raise ConnectionUnusableError("control-plane handle is closed")
            try:
                if not store.check([key]):
                    return None
                value = store.get(key)
            except (OSError, RuntimeError) as error:
                self._poisoned = True
                raise ConnectionUnusableError(
                    "TCPStore read failed; reconnect before retrying"
                ) from error
        if not isinstance(value, bytes) or not value:
            raise ConflictError("TCPStore returned invalid metadata")
        return value

    def _put_immutable(self, key: str, value: bytes) -> None:
        reply = self._compare_set(key, b"", value)
        if reply != value:
            raise ConflictError(f"immutable metadata conflict at {key}")

    def _require_compare_set(self) -> None:
        reply = self._compare_set(_CAS_CAPABILITY_KEY, b"", _CAS_CAPABILITY_VALUE)
        if reply != _CAS_CAPABILITY_VALUE:
            raise ConflictError("compare-set capability marker conflicts")

    def _compare_set(self, key: str, expected: bytes, desired: bytes) -> bytes:
        with self._io_lock:
            self._require_usable()
            store = self._store
            if store is None:
                raise ConnectionUnusableError("control-plane handle is closed")
            compare_set = cast(Callable[[str, bytes, bytes], bytes], store.compare_set)
            try:
                reply = compare_set(key, expected, desired)
            except NotImplementedError as error:
                raise UnsupportedStoreError(
                    "c10d Store does not implement compare_set"
                ) from error
            except (OSError, RuntimeError) as error:
                self._poisoned = True
                raise AmbiguousMutationError(
                    f"TCPStore mutation outcome is unknown at {key}"
                ) from error
        if not isinstance(reply, bytes) or not reply:
            raise ConflictError("TCPStore compare_set returned invalid metadata")
        return reply

    def _require_usable(self) -> None:
        if self._closed:
            raise ConnectionUnusableError("control-plane handle is closed")
        if self._poisoned:
            raise ConnectionUnusableError(
                "TCPStore connection is ambiguous; reconnect before retrying"
            )


def connect_tcpstore_generation_coordinator(
    endpoint: TCPStoreEndpoint,
    config: SessionConfig,
    participant_id: str,
) -> C10dGenerationCoordinator:
    """Connect one participant using scheduler-distributed endpoint data."""
    if not isinstance(endpoint, TCPStoreEndpoint):
        raise ValidationError("endpoint must be a TCPStoreEndpoint")
    if not isinstance(config, SessionConfig):
        raise ValidationError("config must be a SessionConfig")
    _validate_identifier(participant_id, "participant_id")
    if participant_id not in {config.publisher_id, *config.consumer_ids}:
        raise ValidationError("participant_id is not in fixed membership")
    if endpoint.namespace != config.run_id:
        raise ValidationError("endpoint namespace must match the session run_id")
    store = TCPStore(
        host_name=endpoint.host,
        port=endpoint.port,
        world_size=None,
        is_master=False,
        timeout=timedelta(milliseconds=endpoint.timeout_milliseconds),
        wait_for_workers=False,
    )
    return C10dGenerationCoordinator(store, config, participant_id)


def _manifest_key(generation: int) -> str:
    return f"manifest/{generation:020d}"


def _ack_key(generation: int, consumer_id: str) -> str:
    return f"ack/{generation:020d}/{consumer_id}"


def _retired_key(generation: int) -> str:
    return f"retired/{generation:020d}"


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
    "AmbiguousMutationError",
    "C10dGenerationCoordinator",
    "ConnectionUnusableError",
    "ConflictError",
    "ControlPlaneError",
    "connect_tcpstore_generation_coordinator",
    "GenerationManifest",
    "NotReadyError",
    "ObjectRef",
    "PublishDisposition",
    "PublishResult",
    "SessionConfig",
    "TCPStoreEndpoint",
    "UnsupportedStoreError",
    "ValidationError",
]
