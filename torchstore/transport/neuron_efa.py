# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TorchStore transport that RDMAs a policy into a vLLM-Neuron generator's HBM over EFA.

Why this exists
---------------
TorchStore's usual contract is: hand me a ``state_dict``, I populate those tensors in place.
On Trainium that operation has no effect. vLLM-Neuron's compile path inlines model weights
into the compiled graph as HLO constants, so writing ``nn.Parameter.data`` does not change
what the running NEFF executes. The weights ARE torch tensors -- the model is a
``torch.compile`` wrapper, not an NxDI-traced artifact -- but they are not the bytes the
graph reads.

So the destination is not torch storage. It is a set of HBM regions that the generator
describes in a manifest, and the write is an RDMA over EFA into those regions, bracketed by a
quiesce/commit handshake so the engine never reads half-updated weights.

Architecture mapping
--------------------
TorchStore is client <-> StorageVolume. The Neuron weight sync is trainer -> generator. These
reconcile if **the generator IS the StorageVolume**, backed by its own HBM rather than host
memory. The generator exposes the three operations this transport needs (see
``NeuronGeneratorVolume``); a ``StorageImpl`` on the generator publishes that object into the
``TransportContext`` as a ``NeuronEFAVolumeCache`` so the volume-side handlers can reach it.

One shape does NOT fit, and it is the reason this file is not a thin wrapper
-------------------------------------------------------------------------------
The protocol is client -> volume -> **client** -> volume:

  0. generator binds a refit session and publishes its HBM manifests (volume, ONCE)
  1. trainer prepares version N, pinning and fencing its source regions (client)
  2. generator quiesces and RDMA-pulls into HBM, staying quiesced     (volume)
  3. trainer releases its source, once every rank has ACKed           (client)
  4. generator commits and resumes admitting requests                 (volume)

Step 0 is the handshake and happens once per session, not once per version -- binding is not
idempotent across versions. Steps 1-4 repeat per version.

Step 3 must precede step 4: committing first would let the engine serve a request from
weights whose source is about to be freed, and releasing after commit would mean the source
was still pinned while generation had already resumed on it. But ``_put_requests`` gives one
client-side post-success hook (``_post_request_success``) and no second volume round-trip
after it. So step 4 cannot be reached through ``storage_volume_ref``.

The transport therefore takes a DIRECT reference to the generator for the commit, bound by
the caller alongside the trainer-side source. That is not a workaround around TorchStore --
it is this protocol needing an addressable receiver for a step that happens after the store's
put has already completed. ``recv_handshake`` and ``handle_put_request`` still run volume-side
through the normal path.

What this transport does NOT do
-------------------------------
- It is never chosen by ``get_available_transport``. It only applies when the storage volume
  is a Neuron generator, so it must be requested explicitly.
- It does not support ``get``. The flow is one-directional: the trainer publishes a policy
  version, the generator consumes it. Nothing reads a policy back out of a generator, and
  pretending otherwise would invite a caller to rely on semantics that were never tested.
- It does not support resharding. The manifests pin an exact TP layout, and both sides
  validate ``layout_sha256`` before any bytes move.

Correctness notes worth keeping
-------------------------------
The handshake is refcounted per consumer: the trainer releases its source storage only after
every consumer has ACKed, which is what allows a mesh of generators to share one exported
version. Policy versions must be strictly CONSECUTIVE -- the receiver enforces
``version == current + 1`` on both apply and commit -- so a skipped publish is rejected, and
the version is a sync sequence number rather than a trainer step.

A session is bound to ONE generator replica's receive destinations, so a mesh is driven one
replica at a time. Two replicas pulling concurrently would leave an ACK with no unambiguous
owner.
"""

from logging import getLogger
from typing import Any, Protocol, runtime_checkable, TYPE_CHECKING

from torchstore.transport.buffers import (
    TransportBuffer,
    TransportCache,
    TransportContext,
)
from torchstore.transport.types import Request

if TYPE_CHECKING:
    from torchstore.strategy import StorageVolumeRef

logger = getLogger(__name__)


@runtime_checkable
class NeuronGeneratorVolume(Protocol):
    """What a Neuron generator must provide to act as a storage volume.

    Deliberately three calls, matching the three points where the engine's state changes.
    ``ts_put`` does NOT commit: the trainer has to release its source in between, so
    collapsing put and commit is precisely the bug this split prevents.
    """

    async def ts_handshake(
        self,
        source_generation: str,
        current_policy_version: int = -1,
    ) -> dict[str, Any]:
        """Bind a refit session and publish this replica's per-rank HBM manifests."""
        ...

    async def ts_put(
        self,
        manifests: list[dict[str, Any]],
        policy_version: int,
        attempt_id: str,
        slot: int = 0,
    ) -> dict[str, Any]:
        """Quiesce generation and take the RDMA pull. Stays quiesced on return."""
        ...

    async def ts_commit(
        self,
        policy_version: int,
        attempt_id: str,
    ) -> dict[str, Any]:
        """Publish the pulled weights and resume admitting requests."""
        ...


@runtime_checkable
class NeuronWeightSource(Protocol):
    """The trainer side of the bracket: what pins, fences and frees the source regions.

    Separate from the volume because these run on the client and the transport must not
    assume they are co-located with anything.
    """

    async def prepare_version(
        self,
        version: int,
        replica_index: int,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Pin source regions for ``version``. Returns ``(attempt_id, manifests)``.

        The attempt id is minted HERE, not by the generator: it identifies one export of one
        version to one replica, and every later call in the bracket is checked against it so
        a manifest from an abandoned attempt can never be used.
        """
        ...

    async def begin_pull(
        self,
        version: int,
        attempt_id: str,
        consumer_ids: list[str],
        replica_index: int,
    ) -> None:
        """Fence the source and authorize exactly these consumers to read it."""
        ...

    async def release_version(
        self,
        version: int,
        attempt_id: str,
        consumer_acks: list[Any],
        replica_index: int,
    ) -> None:
        """Release the source, once every authorized consumer has ACKed its pull."""
        ...


class NeuronEFAVolumeCache(TransportCache):
    """Carries the generator object to the volume-side handlers.

    ``recv_handshake`` and ``handle_put_request`` are handed only a ``TransportContext``, so
    this is how the generator reaches them. Populated once by the ``StorageImpl`` running on
    the generator.
    """

    def __init__(self) -> None:
        self.generator: NeuronGeneratorVolume | None = None

    def clear(self) -> None:
        # Deliberately does NOT drop the generator. The engine outlives any single transfer,
        # and clearing a cache is a per-request cleanup -- dropping the reference here would
        # break every subsequent sync for no benefit.
        return

    def require_generator(self) -> NeuronGeneratorVolume:
        if self.generator is None:
            raise RuntimeError(
                "no Neuron generator is registered on this storage volume; the StorageImpl "
                "running on the generator must set "
                "transport_context.get(NeuronEFAVolumeCache).generator before any transfer"
            )
        return self.generator


class NeuronEFATransportBuffer(TransportBuffer):
    """Publishes a policy version into a Neuron generator's HBM over EFA.

    The buffer carries only metadata. The bytes move by RDMA between the trainer's exported
    source slab and the generator's manifest-described HBM regions, never through this object
    and never through host memory -- which is the whole reason for the transport.

    Bind the sync with :meth:`bind_sync` before use. The base class constructs transports with
    only a ``storage_volume_ref``, so the trainer source, the generator handle and the version
    cannot be constructor arguments.
    """

    # The manifests pin one exact TP layout and both sides check layout_sha256 before any
    # bytes move, so there is nothing for TorchStore to reshard.
    supports_inplace_resharding: bool = False

    # One policy version is published per call, as a whole. Batching separate keys would
    # break the quiesce/commit bracket that keeps the engine from reading a half-updated
    # policy.
    supports_batch_puts: bool = False
    supports_batch_gets: bool = False

    def __init__(self, storage_volume_ref: "StorageVolumeRef") -> None:
        super().__init__(storage_volume_ref)
        self._source: NeuronWeightSource | None = None
        self._generator: NeuronGeneratorVolume | None = None
        self._source_generation: str | None = None
        self._consumer_ids: tuple[str, ...] = ()
        self._replica_index: int = 0
        self._slot: int = 0
        self._session_bound: bool = False
        self.policy_version: int | None = None

        # Filled in as the bracket progresses. Kept as instance state because the buffer is
        # what travels to the volume and back.
        self.attempt_id: str | None = None
        self.manifests: list[dict[str, Any]] | None = None
        self.volume_handshake: dict[str, Any] | None = None
        self.apply_report: dict[str, Any] | None = None
        self.commit_report: dict[str, Any] | None = None

    def bind_sync(
        self,
        *,
        source: NeuronWeightSource,
        generator: NeuronGeneratorVolume,
        source_generation: str,
        policy_version: int,
        consumer_ids: list[str] | tuple[str, ...],
        replica_index: int = 0,
        slot: int = 0,
        session_bound: bool = False,
    ) -> None:
        """Bind one version, for one generator replica.

        ``consumer_ids`` must name that replica's ranks exactly. They are checked on the wire
        by the receiver, so a set belonging to another replica is rejected rather than
        silently writing to the wrong destinations.

        ``session_bound`` says the refit session for this replica has already been established
        by an earlier sync. Pass True for every version after the first: binding is a ONE-TIME
        operation on the engine, and repeating it with a different version is not idempotent --
        it requires cold actor reconstruction. Getting this wrong does not corrupt weights, it
        refuses to sync at all.
        """
        if not isinstance(policy_version, int) or isinstance(policy_version, bool):
            raise TypeError(f"policy_version must be an int, got {policy_version!r}")
        if policy_version < 0:
            raise ValueError(
                f"policy_version must be non-negative, got {policy_version}"
            )
        if not isinstance(source_generation, str) or not source_generation:
            raise ValueError("source_generation must be a non-empty string")
        if not consumer_ids:
            raise ValueError(
                "consumer_ids must name this replica's ranks; an empty set would authorize "
                "no reader and the pull would hang"
            )
        if len(set(consumer_ids)) != len(consumer_ids):
            raise ValueError(
                f"consumer_ids contains duplicates: {list(consumer_ids)!r}"
            )
        if replica_index < 0:
            raise ValueError(f"replica_index must be non-negative, got {replica_index}")

        self._source = source
        self._generator = generator
        self._source_generation = source_generation
        self.policy_version = policy_version
        self._consumer_ids = tuple(consumer_ids)
        self._replica_index = replica_index
        self._slot = slot
        self._session_bound = session_bound

    # ------------------------------------------------------------------
    # Client side
    # ------------------------------------------------------------------
    def requires_handshake(self, requests: list[Request]) -> bool:
        """Once per session, not once per version.

        The handshake is the engine BINDING a refit session and publishing its HBM
        destinations. That is a one-time operation: repeating it with a different version is
        not idempotent and demands cold actor reconstruction. So the first sync handshakes and
        every later one is told ``session_bound=True``.

        There is still nothing the client could compute for itself -- unlike an RDMA transport
        over host memory there is no local buffer to register -- which is why the first sync
        cannot skip it.
        """
        return not self._session_bound

    async def _pre_handshake(self) -> None:
        """Nothing to allocate locally.

        The counterpart RDMA transports register a local buffer here. There is none: the
        source is the trainer's exported EFA slab and the destination is the generator's HBM,
        so no host memory participates in the transfer.
        """
        return None

    async def _post_handshake(
        self,
        handshake_results: list[Any],
        requests: list[Request],
    ) -> None:
        """Validate the layout the generator published, before anything is pinned to it.

        The response carries the per-rank HBM manifests and the layout digests. It does NOT
        carry an attempt id -- that is minted per version by the trainer in ``_pre_put_hook``
        -- so this does not look for one.
        """
        result = self._single_handshake_result(handshake_results)
        self.volume_handshake = result

        metadata = result.get("metadata")
        if not isinstance(metadata, dict):
            raise RuntimeError(
                "Neuron generator handshake carried no metadata; layout cannot be validated"
            )
        if not result.get("manifests"):
            raise RuntimeError(
                "Neuron generator handshake carried no manifests; the receive destinations "
                "are unknown"
            )
        if not metadata.get("layout_sha256"):
            raise RuntimeError(
                "Neuron generator handshake carried no layout_sha256; a TP layout mismatch "
                "would not be detected before bytes move"
            )
        volume_generation = metadata.get("source_generation")
        if volume_generation != self._source_generation:
            raise RuntimeError(
                "Neuron generator is bound to a different export generation: volume has "
                f"{volume_generation!r}, this sync is {self._source_generation!r}. The "
                "session must be rebuilt; reusing it would write into regions described by "
                "another trainer's manifests"
            )
        self._session_bound = True

    async def _pre_put_hook(self, requests: list[Request]) -> None:
        """Pin the trainer's source for this version, then authorize the reader.

        Per VERSION, unlike the handshake. Both steps live here because the order between them
        is required and neither may outlive a failed put: nothing can be authorized to read a
        source that is not yet pinned, and a fence taken before the regions exist would name
        the wrong addresses.
        """
        source = self._require_source()
        version = self._require_version()

        attempt_id, manifests = await source.prepare_version(
            version, self._replica_index
        )
        if not isinstance(attempt_id, str) or not attempt_id:
            raise RuntimeError(
                f"trainer returned no attempt_id for version {version}; without one a stale "
                "manifest could be used for this transfer"
            )
        if not manifests:
            raise RuntimeError(
                f"trainer returned no source manifests for version {version}"
            )
        self.attempt_id = attempt_id
        self.manifests = list(manifests)

        await source.begin_pull(
            version,
            attempt_id,
            list(self._consumer_ids),
            self._replica_index,
        )

    async def _post_request_success(self) -> None:
        """Release the trainer's source, then let the generator commit and resume.

        This is the second half of the bracket and the ORDER is the correctness property.
        Release first, because every rank has completed and fenced its pull while generation
        is still quiesced. Commit second, because only then is it safe for the engine to
        admit requests against the new weights.

        Commit goes to the generator directly rather than through ``storage_volume_ref``:
        TorchStore's put has already returned by this point, so there is no second volume
        round-trip left in the lifecycle. See the module docstring.
        """
        report = self.apply_report
        if report is None:
            raise RuntimeError(
                "Neuron EFA put reported success without an apply report; refusing to "
                "release the trainer's source, which would free regions the generator may "
                "still be reading"
            )
        version = self._require_version()
        attempt_id = self._require_attempt_id()

        await self._require_source().release_version(
            version,
            attempt_id,
            list(report.get("consumer_acks") or []),
            self._replica_index,
        )
        self.commit_report = await self._require_generator().ts_commit(
            policy_version=version,
            attempt_id=attempt_id,
        )

    async def drop(self) -> None:
        """Release request-scoped state.

        Runs in a ``finally``, so it must not assume the bracket completed. Nothing is
        unregistered here: the source regions are the trainer's to free (``release_version``)
        and the HBM regions are the engine's. Dropping either from here would be reaching
        across an ownership boundary on a path that also runs after a failure.
        """
        self.manifests = None
        self.volume_handshake = None

    # ------------------------------------------------------------------
    # Storage volume side
    # ------------------------------------------------------------------
    async def recv_handshake(
        self,
        ctx: "TransportContext",
        entries: list[tuple[Request, Any]],
    ) -> dict[str, Any]:
        """Generator side: publish this replica's HBM destinations.

        Delegates to the generator rather than reimplementing the session logic. That code
        produces the per-rank manifests, the transport attestation and the layout digests, and
        it is what has been validated on hardware -- a second implementation here is exactly
        what would drift from it.
        """
        generator = ctx.get(NeuronEFAVolumeCache).require_generator()
        version = self.policy_version
        return await generator.ts_handshake(
            self._require_source_generation(),
            -1 if version is None else version - 1,
        )

    async def handle_put_request(
        self,
        ctx: "TransportContext",
        entries: list[tuple[Request, Any]],
    ) -> dict[str, Any]:
        """Generator side: quiesce and RDMA into HBM. Does NOT commit.

        Generation must be quiesced before any byte lands and every rank's pull must be
        fenced before the trainer releases its source. Commit is deliberately left to the
        client, after that release -- see ``_post_request_success``.
        """
        generator = ctx.get(NeuronEFAVolumeCache).require_generator()
        manifests = self.manifests
        if not manifests:
            raise RuntimeError(
                "Neuron EFA put arrived with no source manifests; the handshake did not "
                "complete and there is nothing to pull"
            )
        report = await generator.ts_put(
            manifests,
            self._require_version(),
            self._require_attempt_id(),
            self._slot,
        )
        self.apply_report = report
        return report

    # ------------------------------------------------------------------
    # Unsupported directions
    # ------------------------------------------------------------------
    async def _pre_get_hook(self, requests: list[Request]) -> None:
        raise NotImplementedError(
            "NeuronEFA transport is write-only: a policy version is published from the "
            "trainer to the generator and never read back. Use the trainer's own state_dict "
            "as the source of truth instead of getting from the generator."
        )

    async def handle_get_request(
        self,
        ctx: "TransportContext",
        entries: list[tuple[Request, Any]],
    ) -> list[Any]:
        raise NotImplementedError(
            "NeuronEFA transport is write-only; see _pre_get_hook."
        )

    async def _handle_storage_volume_response(
        self,
        requests: list[Request],
        transport_buffer: "TransportBuffer",
    ) -> list[Any]:
        raise NotImplementedError(
            "NeuronEFA transport is write-only; see _pre_get_hook."
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _single_handshake_result(handshake_results: Any) -> dict[str, Any]:
        """One volume, one response.

        ``handshake.call_one`` may hand back the dict itself or a one-element list depending
        on the volume, so accept both and reject anything else rather than guessing.
        """
        result = handshake_results
        if isinstance(result, list):
            if len(result) != 1:
                raise RuntimeError(
                    "Neuron generator handshake expected exactly one response, got "
                    f"{len(result)}. A session binds to ONE replica's destinations, so "
                    "several responses mean several replicas were addressed at once"
                )
            result = result[0]
        if not isinstance(result, dict):
            raise RuntimeError(
                f"Neuron generator handshake expected a dict, got {type(result)!r}"
            )
        return result

    def _require_source(self) -> NeuronWeightSource:
        if self._source is None:
            raise RuntimeError(
                "bind_sync must be called before the transfer: no trainer source"
            )
        return self._source

    def _require_generator(self) -> NeuronGeneratorVolume:
        if self._generator is None:
            raise RuntimeError(
                "bind_sync must be called before the transfer: no generator"
            )
        return self._generator

    def _require_source_generation(self) -> str:
        if self._source_generation is None:
            raise RuntimeError(
                "bind_sync must be called before the transfer: no generation"
            )
        return self._source_generation

    def _require_version(self) -> int:
        if self.policy_version is None:
            raise RuntimeError(
                "bind_sync must be called before the transfer: no version"
            )
        return self.policy_version

    def _require_attempt_id(self) -> str:
        if not self.attempt_id:
            raise RuntimeError(
                "no attempt_id: the handshake has not run, so nothing authorizes this transfer"
            )
        return self.attempt_id
