# NeuronEFA transport: storage volumes backed by accelerator HBM

A transport for a storage volume whose backing store is not host memory or torch storage, but
an inference engine's HBM, written by RDMA over EFA. It exists for RL weight sync on Trainium:
publishing a new policy from a TorchTitan trainer into a running vLLM-Neuron generator.

It is never selected automatically. Ask for it by name:

```python
ref.default_transport_type = TransportType.NeuronEFA
```

## Why a separate transport

TorchStore's usual contract is: hand me a `state_dict`, I populate those tensors in place. On
Trainium that has no effect. vLLM-Neuron's compile path inlines model weights into the compiled
graph as HLO constants, so writing `nn.Parameter.data` does not change what the running NEFF
executes. The weights *are* torch tensors — the model is a `torch.compile` wrapper — but they
are not the bytes the graph reads.

So the destination is a set of HBM regions the generator describes in a manifest, and the write
is an RDMA into those regions, bracketed by a quiesce/commit handshake so the engine never
serves a request from half-updated weights.

## The two things you implement

The transport talks to two objects. Both are `Protocol`s, so there is no base class to inherit
and `isinstance` works for a quick self-check.

`NeuronGeneratorVolume` — the receiver, on the generator:

| method | does |
|---|---|
| `ts_handshake(source_generation, current_policy_version)` | bind a refit session, publish per-rank HBM manifests and layout digests |
| `ts_put(manifests, policy_version, attempt_id, slot)` | quiesce generation, take the RDMA pull, **stay quiesced** |
| `ts_commit(policy_version, attempt_id)` | publish the pulled weights, resume admitting requests |

`NeuronWeightSource` — the sender, on the trainer:

| method | does |
|---|---|
| `prepare_version(version, replica_index)` | pin source regions, return `(attempt_id, manifests)` |
| `begin_pull(version, attempt_id, consumer_ids, replica_index)` | fence the source, authorize exactly these consumers |
| `release_version(version, attempt_id, consumer_acks, replica_index)` | release the source, after every consumer ACKed |

## Wiring it up

On the generator, host a `NeuronHBMStore`. It publishes your generator into the
`TransportContext` where the volume-side handlers look for it:

```python
from torchstore.neuron_hbm_store import NeuronHBMStore

store = NeuronHBMStore(my_generator)   # my_generator: NeuronGeneratorVolume
```

On the client, bind one version for one generator replica and put:

```python
from torchstore.transport.neuron_efa import NeuronEFATransportBuffer

buffer = NeuronEFATransportBuffer(storage_volume_ref)
buffer.bind_sync(
    source=my_trainer,                 # NeuronWeightSource
    generator=my_generator,            # NeuronGeneratorVolume, for the commit
    source_generation=generation_uuid,
    policy_version=version,
    consumer_ids=[f"generator-0-rank-{r}" for r in range(8)],
    replica_index=0,
    session_bound=version > first_version,
)
await buffer.put_to_storage_volume([Request(key=f"policy/v{version}")])
```

## Four constraints that are not optional

**1. The order is the correctness property.** The bracket is
client → volume → **client** → volume:

```
0. ts_handshake          bind session, publish HBM manifests      (volume, ONCE)
1. prepare_version       pin source regions                       (client)
   begin_pull            fence, authorize consumers               (client)
2. ts_put                quiesce, RDMA pull, stay quiesced        (volume)
3. release_version       free the source, after all ACKs          (client)
4. ts_commit             publish, resume                          (volume)
```

Step 3 must precede step 4. Commit first and the engine can serve a request from weights whose
source is about to be freed; release after commit and the source stays pinned while generation
has already resumed on it.

`_put_requests` offers one client-side post-success hook and no second volume round-trip after
it, which is why `bind_sync` takes a **direct** generator reference for the commit: by then
TorchStore's put has already returned. This is the one place the transport reaches outside
`storage_volume_ref`, and it is deliberate.

**2. The handshake is once per session, not once per version.** Binding a refit session is not
idempotent across versions — rebinding with a later version requires cold actor reconstruction.
Pass `session_bound=True` for every version after the first. Getting this wrong refuses to sync
rather than corrupting anything, which is the failure direction to prefer.

**3. Versions must be strictly consecutive.** The receiver enforces `version == current + 1` on
both apply and commit, so a skipped publish is rejected. The version is a *sync sequence
number*, not a trainer step — with `weight_sync_interval > 1` the two diverge immediately.

**4. One replica at a time.** A session owns one replica's receive destinations. Two replicas
pulling concurrently leave an ACK with no unambiguous owner. Drive a generator mesh
sequentially, one `bind_sync` and one put per replica.

## What it does not do

- **No `get`.** One-directional: the trainer publishes, the generator consumes. Nothing reads a
  policy back out of a generator. Use the trainer's own `state_dict` as the source of truth.
- **No resharding.** The manifests pin an exact TP layout and both sides check `layout_sha256`
  before any bytes move.
- **No delete or reset.** The volume's contents are the policy the engine is serving; clearing
  them means engine teardown, not a store operation.
- **No batching.** One policy version per put, as a whole — batching separate keys would break
  the quiesce/commit bracket.

## Testing without hardware

The transport never touches tensors; the bytes move outside it. So the fakes in
`tests/test_neuron_efa_transport.py` cover the whole protocol on any machine — the call
ordering, non-consecutive versions, generation mismatch, a missing layout digest, a failed pull
leaving the source pinned, and `get` refusal. Start there when implementing either Protocol.

## A production caller

`torchtitan/experiments/rl/grpo.py` in AWS's NeuronTorchTitan drives this for GRPO on
Qwen3-14B: a trainer of 8 actors at TP=8 publishing 29.5 GB per sync into a two-pod generator
mesh, ~1.1 s per replica.
