# Precomputed tensor routing

Plan once at setup. Every update is a local lookup, not a controller round trip.

## Using it

```
setup
─────
trainer ranks (publishers)                     generation ranks (requesters)
                            ┌────────────┐
           layout ─────────▶│ TorchStore │ ◀───────── layout
     local table ◀──────────│   client   │──────────▶ local table
                            └────────────┘
updates
───────
                         ┌───────────────┐
publisher ─────put──────▶│ volume / RDMA │──────get──────▶ requester
                         └───────────────┘                    │
                         ┌───────────────┐                    │ relay
other requesters ◀─get───│ volume / RDMA │◀───────────────────┤
        ▲                └───────────────┘                    │
        └──────────────────── notify ─────────────────────────┘
```

- The application coordinator calls `initialize`; TorchStore installs a barrier.
- The application coordinator broadcasts `register_state_dict` to its ranks;
  each TorchStore client registers its `layout` (tensor geometry metadata) and
  waits.
- When the last layout arrives, the barrier completes and every TorchStore client
  has a fixed `local route table`.
- Each routing client records the wire dtype selected at registration. Publishers
  cast floating tensors to that dtype, while `preserve_dtype_keys` keeps selected
  buffers in their registered dtype.
- For every `put`/`get`, each rank consults that table locally.
- Some requesters fetch from publishers, then `relay` what they fetched to other
  requesters.

```python
# At startup on application coordinator.
await ts.initialize(mesh=trainer_mesh, relay_meshes=[gen_mesh], strategy=strategy)

# Once per publisher rank.
c = await ts.client(role="publisher")
await c.register_state_dict(
    model.state_dict(),
    "weights",
    transfer_dtype=torch.bfloat16,
    preserve_dtype_keys=frozenset(buffer_names),
)

# Once per requester rank. Its state dict declares the destination dtypes.
c = await ts.client(role="requester", group=0)
await c.register_state_dict(model.state_dict(), "weights")

# Every update.
await ts.put_state_dict(model.state_dict(), "weights")   # publisher
await ts.get_state_dict("weights", model.state_dict())   # requester
```

## Two places to plan

We can build the routing table either on the coordinator or on each rank

```
1. on coordinator                          2. on each rank
─────────────────                          ────────────────
rank ──layout──▶ coordinator               rank ──layout──▶ coordinator
                      │ barrier                                 │ barrier
                      ▼                                         ▼
              build(all ranks)                            (build nothing)
              O(QG²/DP), LEAST_LOADED                           │
                      │                    everyone's layouts───┘
  ◀───my table────────┘                     │
                                            ▼
                                      build_for(me)
                                      O(QG), ROTATE, on every rank at once
```

### 1. Build the plan at the coordinator

- Once the application calls `register_state_dict` on every rank, each TorchStore
  client calls `register` on the TorchStore coordinator.
- After all ranks have called `register`, the coordinator builds every table.
- The coordinator assigns routes by balancing traffic across publishers and
  requesters (`LEAST_LOADED`).
- The coordinator returns each client its table.

### 2. Build the plan on each rank

- Once the application calls `register_state_dict_locally` on every rank, each
  client calls `register_layouts`.
- After all ranks have called `register_layouts`, the TorchStore coordinator
  returns all layouts to each client.
- Each client assigns routes by deterministically spreading traffic across
  publishers and requesters (`ROTATE`).

### In both cases

- Either path leaves every client with a fixed `local route table`; only where
  and how it is built changes.
- `Q` is tensor keys, `G` requester ranks, and `DP` requester replicas.

Measured on `kimi-k2`: 5,203 keys, FSDP 256 publishing to TP 128 × DP 1, a
transformer layout, 305k transfers per rank.

| Where built             | On coordinator                    | On each rank               |
| ----------------------- | --------------------------------- | -------------------------- |
| load balance            | fewest bytes assigned so far      | hash-based rotation        |
| build CPU               | **238 s**                         | **6.6 s**                  |
| time complexity         | `O(QG²/DP)`                       | `O(QG)`                    |
| space complexity        | `O(QG²/DP)`                       | `O(QG)`                    |
| peak                    | ~6.5 GB, in the coordinator       | **520 MiB**, on each rank  |
| steady                  | 6 GB coordinator + 50 MB a rank   | 50 MB a rank               |
| each `get`/`put` lookup | 1 ms                              | 1 ms                       |

All benchmark numbers are measured.

- Prebuilt, distributed route tables reduce setup planning time to zero.
- Trainer or generator topology changes require only rerunning the registration
  barrier, which should be straightforward to add.

## What it fixes

### Relay replicated reads

Three requester replicas need the same shard `A`:

```
       before                         after
     ┌─A──▶ G0
T0 ──┼─A──▶ G1                 T0 ──A──▶ G0 ──A──▶ G1
     └─A──▶ G2                           └────A──▶ G2
```

`G1` and `G2` still fetch shard `A`, but from `G0` instead of `T0`.

### Balance publisher traffic

Different shards are spread across interchangeable publisher replicas:

```
       before                         after
     ┌─A──▶ G0                  T0 ──A──▶ G0
T0 ──┼─B──▶ G1                  T1 ──B──▶ G1
     └─C──▶ G2                  T2 ──C──▶ G2
```

`T0`–`T2` hold the same data; `A`–`C` are different requested shards.

### Transfer only shard intersections

Different source and destination sharding transfers only their intersections:

```
before
──────
T0 rows 0–3 ──┐
              ├──all-gather──▶ full weights ──┬──▶ G0 cols 0–3
T1 rows 4–7 ──┘                               └──▶ G1 cols 4–7

after
─────
T0 rows 0–3 ──┬──intersection────────▶ G0 cols 0–3
              └──intersection────────▶ G1 cols 4–7
T1 rows 4–7 ──┬──intersection────────▶ G0 cols 0–3
              └──intersection────────▶ G1 cols 4–7
```

No rank gathers the full weights.

### Use one key for a distributed state dict

Different publishers may own different tensors:

```
             before                              after
T0 ──▶ weights_0 {A, B} ─┐               T0 ──▶ weights {A, B} ─┐
T1 ──▶ weights_1 {C, D} ─┼──▶ G          T1 ──▶ weights {C, D} ─┼──▶ G
T2 ──▶ weights_2 {E, F} ─┘               T2 ──▶ weights {E, F} ─┘
      enumerate and merge keys                  one key
```

The requester asks for one state dict; TorchStore finds each tensor at its publisher.
