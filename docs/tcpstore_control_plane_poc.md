# TCPStore Generation-Coordinator Proof of Concept

Status: reviewable metadata-control proof; not an end-to-end weight-sync path.

## Decision

The public type is `C10dGenerationCoordinator`, not
`TCPStoreControlPlane`. The narrower name reflects what this implementation
does: coordinate immutable generations, consumer acknowledgements, and
retirement over a c10d Store.

The coordinator does not schedule processes, launch jobs, transfer tensors,
or implement the rest of TorchStore's Monarch-backed control path. A separate
factory connects it to TCPStore using endpoint data supplied by an external
launcher.

## Control-Plane Boundary

```text
external launcher
  ├── starts TCPStore and distributes a fresh session configuration
  ├── starts the trainer-side publisher
  └── starts the fixed inference-side consumers

publisher
  └── publishes one immutable manifest and commits its digest-bound head

consumers
  └── discover the head, apply data externally, and acknowledge its digest

publisher
  └── retires the generation after every configured consumer acknowledges
```

The launcher distributes endpoint data out of band. TCPStore stores only
membership, descriptor, generation, and lifecycle metadata.
`ObjectRef.location` is intentionally opaque so a data plane such as Mooncake
can interpret it without introducing Mooncake into the coordinator.

## Mooncake Mapping

The proposed Mooncake control/data-plane design maps to this contract as
follows:

| Paste concept | POC representation | Status |
| --- | --- | --- |
| Weight version `V` | Contiguous `generation` plus opaque `application_version` | Implemented with split semantics |
| Tensor FQN | Provider-owned `(source_rank, FQN[, segment])` logical key | Encoding deferred |
| Remote span length | `ObjectRef.size_bytes` | Implemented |
| Mooncake peer and address | Versioned provider-owned `ObjectRef.location` | Encoding deferred |
| Ready marker | Atomic digest-bound generation head | Implemented |
| Consumer completion | Exact digest acknowledgement and retirement | Implemented |
| Distributed sharding metadata | Immutable external object reference or a future typed record | Deferred |

The POC has one publisher. A trainer leader must therefore gather every
trainer rank's descriptors before publishing a complete manifest, or a later
protocol must add multi-publisher assembly. Rank must be part of each logical
key when the same FQN is present on multiple source ranks.

The manifest is limited to 4,096 references and 1 MiB; logical keys are limited
to 256 characters and locations to 1,024. A larger peer table or distributed
metadata document needs a bounded chunked catalog or an immutable external
object reference. That object must be content-addressed, retained through
retirement, and verified before use.

## What This Proves

- Bounded canonical records can cross independent TCPStore client connections.
- A manifest is discoverable only after an atomic compare-set commits its head.
- Exact retries are idempotent and conflicting or corrupt records fail closed.
- Consumer acknowledgements gate retirement and the next contiguous generation.
- The isolated Buck target has no dependency on Monarch APIs.

The tests use a real TCPStore with distinct connections in one process. They
do not prove independent-job, multi-host, tensor-transfer, or resharding
behavior. The normal TorchStore package import and existing APIs remain
Monarch-backed in this focused proof.

## Required Assumptions

- Every session uses a fresh, never-reused run ID and TCPStore namespace.
- Membership is fixed, with exactly one live process per participant ID.
- Adding an inference replica requires a newly declared session.
- Each coordinator exclusively owns its Store connection.
- Participants are cooperative and communicate on a trusted network.
- Participant failure or the 16-generation limit ends the session.
- The external launcher owns endpoint distribution and server availability.

These assumptions prevent a replacement process or stale acknowledgement from
being mistaken for current progress. Production restart support requires
incarnation fencing or leases.

## Data-Plane Safety Requirement

Manifest immutability does not make referenced memory immutable. A Mooncake
integration must keep each advertised source buffer stable until all consumers
acknowledge and the generation retires, or publish from generation-owned
double-buffered storage. Otherwise a one-sided pull can race the next optimizer
update even though the TCPStore metadata remains valid.

A consumer must acknowledge only after transfer, validation, resharding, and
safe model activation all succeed. Partial application must fail closed and
must not return an inference worker to service.

The Mooncake adapter must use a canonical versioned descriptor and validate
the source-rank, session, address, span, and sharding-metadata relationship;
the coordinator deliberately treats the location as untyped text.

## Deferred Integration

- Mooncake Transfer Engine initialization, handshakes, registration, and reads.
- A typed peer/address descriptor and `torch_checkpointing` metadata contract.
- `DefaultResharder` load-plan execution and strided-copy handling.
- Trainer-rank aggregation, blocking waits, and multi-process barriers.
- TorchTitan pause, activation, post-load, and resume wiring.
- Cross-process, cross-host, RDMA, correctness, and performance qualification.
- Authentication, restart recovery, dynamic membership, and long-lived cleanup.
