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

- The application coordinator calls `initialize`.
- Each publisher's first `put_state_dict` lazily registers its `layout`
  (tensor geometry metadata) and returns without waiting for requesters.
- Each requester's first `get_state_dict` lazily registers its layout, waits
  for the layouts needed to plan routes, and installs a fixed local route table.
- Each routing client records the wire dtype selected at registration. Publishers
  cast floating tensors to that dtype before publishing them.
- For every `put`/`get`, each rank consults that table locally.
- Some requesters fetch from publishers, then `relay` what they fetched to other
  requesters.

```python
# At startup on application coordinator.
await ts.initialize(mesh=trainer_mesh, relay_meshes=[gen_mesh], strategy=strategy)

# On each publisher rank. The first put also registers its layout.
await ts.client(role="publisher")
await ts.put_state_dict(
    model.state_dict(),
    "weights",
    transfer_dtype=torch.bfloat16,
)

# On each requester rank. The first get registers its destination layout.
await ts.client(role="requester", group=0)
await ts.get_state_dict("weights", model.state_dict())
```

## Build each plan locally

- A publisher builds its local table from its own state dict, registers that
  layout with the coordinator, and returns without waiting.
- A requester registers its layout and waits until the layouts needed for
  requester relays are available.
- The coordinator returns metadata and relay-service handles, not a routing
  plan. Each requester builds its one local table deterministically (`ROTATE`).
- Every client is left with one fixed `local route table`.
- `Q` is tensor keys, `G` requester ranks, and `DP` requester replicas.

Measured on `kimi-k2`: 5,203 keys, FSDP 256 publishing to TP 128 × DP 1, a
transformer layout, 305k transfers per rank.

| Metric                  | Per rank                    |
| ----------------------- | --------------------------- |
| load balance            | hash-based rotation         |
| build CPU               | **6.6 s**                   |
| time complexity         | `O(QG)`                     |
| space complexity        | `O(QG)`                     |
| peak                    | **520 MiB**                 |
| steady                  | 50 MB                       |
| each `get`/`put` lookup | 1 ms                        |

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

## Class structure and call flow

- `ts.initialize()` is the user-facing entry point- users pass the handles to all actors that can store data
- Setup `StorageVolume` using `MultiMeshStrategy` - stores the RPC handle to all the storage volume actors across all monarch meshes
- Setup `RoutingServiceGroup` - similar to `MultiMeshStrategy` it stores a handle to all `RoutingService`'s which is a RPC service that allows for notifications between generators
- Spawn `RoutingCoordinator` - it stores tensor layouts and forwards the
  metadata requesters need to build their local routing tables.

```
ts.initialize(meshes, ...)
        ├──▶ StorageVolume.spawn(...)
        ├──▶ RoutingService.spawn(...) ──▶ RoutingServiceGroup.set_services(...)
        └──▶ RoutingCoordinator.init(...)
```

- After initialization, all actors call `ts.client(role=...)` - role is "publisher" or "requester" which return `RoutingClient`
- `RoutingClient` inherits from the original client for torchstore to avoid implementing similar things, it's function is unchanged
- Users call `put_state_dict` or `get_state_dict`; their first call lazily
  invokes the matching routing-client layout hook.
  - Publishers call `RoutingClient.register_layout()`, which builds their
    local table and calls `RoutingCoordinator.register_layout()` without waiting.
  - Requesters call `RoutingClient.get_layouts()`, which receives layouts and
    relay services from `RoutingCoordinator.register_requester_layout()` and
    builds their local table.
- After client has the routing table, it calls `RoutingDirectory.install(...)` which stores this local routing table
- The client builds tables with `RoutingPlan.for_publisher()` or
  `RoutingPlan.for_requester()`.

```
put_state_dict(...) / get_state_dict(...)
        │
        ├──▶ RoutingClient.register_layout(...)
        │          └──▶ RoutingCoordinator.register_layout(...)
        │
        └──▶ RoutingClient.get_layouts(...)
                   └──▶ RoutingCoordinator.register_requester_layout(...)
                    │
                    ▼
             RoutingPlan.for_requester(...)
                    │
                    ▼
             RoutingDirectory.install(...)
```

- Every update, publishers call `put_state_dict()`, requesters call `get_state_dict()`
- `put_state_dict()` calls `RoutingClient.put_batch(...)` that interacts with  `StorageVolume` to store tensors
- `get_state_dict()` calls `RoutingClient._fetch(...)` that does the following things
  - Calls `RoutingDirectory.resolve_get_batch(...)` to figure out which volume it should get the tensors from
  - If plan says it should fetch from another requester, it uses `RoutingService.wait_ready()` to wait for a notification
  - Uses `StorageVolume` to actually fetch the batch locally when it receives notification (or just fetches from the publisher directly if plan says so)
  - If the plan says that it should act as a "relay", it stores the tensor in its own `RoutingDirectory` and notifies other requestors using `RoutingService.notify_ready()`

```
ts.put_state_dict(...) ──▶ RoutingClient.put_batch(...) ──▶ StorageVolume

ts.get_state_dict(...) ──▶ RoutingClient._fetch(...)
                                      │
                                      ├──▶ RoutingDirectory.resolve_get_batch(...)
                                      ├──▶ StorageVolume
                                      └──▶ RoutingDirectory.notify_ready(...) / wait_ready(...)
                                               └──▶ RoutingService.notify_ready(...) / wait_ready(...)
                                                    (only when requesters relay data)
```
