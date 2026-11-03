# TorchStore

Distributed, asynchronous tensor storage for PyTorch, built on [Monarch](https://github.com/meta-pytorch/monarch).

TorchStore makes it easy to share tensors and model weights across distributed processes. Store and retrieve PyTorch tensors (including DTensors and arbitrary Python objects), exchange `state_dict`s between actors for workflows like reinforcement learning weight sync, and reshard data across different device meshes — all with automatic transport selection that picks the fastest available path based on topology and hardware availability.

Key Features:
- **Async put/get API** with batch operations (`put_batch`/`get_batch`) and key management (`delete`, `exists`, `keys`)
- **DTensor-aware storage** with tensor-slice retrieval and resharding across different layouts
- **`state_dict` exchange** for checkpoint-style weight sync between actors
- **Direct RDMA weight sync** for zero-copy GPU-to-GPU model weight transfer (one-hop, no intermediate storage for cases like synchronous RL)
- **Automatic transport selection** — POSIX shared memory for same-host, RDMA when available, with Gloo and Monarch RPC fallbacks
- **Configurable storage strategies** — `LocalRankStrategy` (per-rank volumes), `HostStrategy` (per-host), extensible for your specific use case

> **Note:** TorchStore supports both Monarch-native launches and SPMD entry points (`torchrun` / `torchx`). See [SPMD Usage](#spmd-usage-torchrun--torchx) below for the latter.


> ⚠️ **Early Development Warning** TorchStore is currently in an experimental
> stage. You should expect bugs, incomplete features, and APIs that may change
> in future versions. The project welcomes bugfixes, but to make sure things are
> well coordinated you should discuss any significant change before starting the
> work. It's recommended that you signal your intention to contribute in the
> issue tracker, either by filing a new issue or by claiming an existing one.

## Installation

### uv (Recommended)
```bash
uv venv --python 3.12
source .venv/bin/activate

git clone git@github.com:meta-pytorch/torchstore.git
cd torchstore

# Core development install.
uv sync --extra dev

# CUDA 12.8 + torchcomms 0.2.0 from the PyTorch cu128 index.
uv sync --extra dev --extra cu128

# CUDA 13.0 + torchcomms 0.2.0 from the PyTorch cu130 index.
uv sync --extra dev --extra cu130
```

### pip

If you prefer `pip`, use the matching PyTorch CUDA index explicitly:

```bash
conda create -n torchstore python=3.12
conda activate torchstore

# Base install
pip install -e .

# Development install with CUDA 12.8 wheels
pip install -e '.[dev,cu128]' --extra-index-url https://download.pytorch.org/whl/cu128

# Development install with CUDA 13.0 wheels
pip install -e '.[dev,cu130]' --extra-index-url https://download.pytorch.org/whl/cu130
```

### Regular Installation

```bash
pip install 'torchstore[cu128] @ git+https://github.com/meta-pytorch/torchstore.git' \
  --extra-index-url https://download.pytorch.org/whl/cu128
```

> **Performance:** For the best transfer speeds, install with a CUDA extra (`cu128` or `cu130`) to include `torchcomms`.

Once installed, you can import it in your Python code:

```python
import torchstore
```

## Usage

### Reusing keys

Ordinary tensors can overwrite an existing key with a different shape or dtype.
Shared-memory transfers reuse compatible allocations and retire obsolete cached
handles when a replacement succeeds or a reader attaches to a new handle. This
also bounds cached allocations when keys cycle through delete/put operations.

Callers must finish consuming a key before deleting or recycling it. The
shared-memory lock serializes PUT/GET operations sharing one client cache; it
does not coordinate independent clients or DELETE. Overwrites do not provide
atomic snapshots or batch rollback. Changing a DTensor's shard shape, dtype, or
layout under an existing key is unsupported; delete the old key before storing
a different layout.

TorchStore APIs are called from within Monarch actors. Each actor interacts
with the store through the module-level `ts.*` functions.

```python
import asyncio

import torch

from monarch.actor import Actor, current_rank, endpoint

import torchstore as ts
from torchstore.utils import spawn_actors


WORLD_SIZE = 4


# In monarch, Actors are the way we represent multi-process/node applications. For additional details, see:
# https://github.com/meta-pytorch/monarch
class ExampleActor(Actor):
    def __init__(self, world_size=WORLD_SIZE):
        self.rank = current_rank().rank
        self.world_size = world_size

    @endpoint
    async def store_tensor(self):
        t = torch.tensor([self.rank])
        await ts.put(f"{self.rank}_tensor", t)

    @endpoint
    async def print_tensor(self):
        other_rank = (self.rank + 1) % self.world_size
        t = await ts.get(f"{other_rank}_tensor")
        print(f"Rank=[{self.rank}] Fetched {t} from {other_rank=}")


async def main():

    # Create a store instance
    await ts.initialize()

    actors = await spawn_actors(WORLD_SIZE, ExampleActor, "example_actors")

    # Calls "store_tensor" on each actor instance
    await actors.store_tensor.call()
    await actors.print_tensor.call()

    await ts.shutdown()


if __name__ == "__main__":
    asyncio.run(main())

# Expected output
# [0] [2] Rank=[2] Fetched tensor([3]) from other_rank=3
# [0] [0] Rank=[0] Fetched tensor([1]) from other_rank=1
# [0] [3] Rank=[3] Fetched tensor([0]) from other_rank=0
# [0] [1] Rank=[1] Fetched tensor([2]) from other_rank=2

```

### Importing TorchStore and Monarch

`import torchstore` loads no Monarch module. TorchStore's Monarch backend loads Monarch
on the first call that goes through it, such as `ts.initialize`, `ts.initialize_spmd` or
a process's first `ts.client`, which data calls such as `ts.put` make. Volumes of
`LocalRankStrategy` and of routing stores load it as they start, to read their rank, and
so does direct weight sync's RDMA registration. A worker that builds a `LocalClient` from
a controller it was handed loads Monarch when it unpickles the controller, but not the
backend.

Monarch reads its `HYPERACTOR_*` settings when it loads, and others such as `MONARCH_*`
and `OTEL_PYTHON_CONTEXT`. So in a process that has not imported Monarch yet, settings you
set after importing TorchStore apply too. Monarch also sets `HYPERACTOR_EXECUTION_ID`
when it loads, so start child processes that must share it after that first call. On
ROCm, torchmonarch 0.7.0 preloads Triton's library when it loads to keep Triton from
crashing, which works only before Triton loads, so there import Monarch first.

If `HYPERACTOR_CODEC_MAX_FRAME_LENGTH` is unset, importing TorchStore exports
`910737418240` for it. A process that loads Monarch afterwards reads that limit. In one
that imported Monarch first, the first TorchStore call that loads TorchStore's Monarch
backend, such as `ts.initialize` or `ts.client`, hands Monarch that limit through
`monarch.config.configure`, unless Monarch holds a limit other than its own, the variable
no longer holds TorchStore's value, or the value it held when Monarch loaded outranks
`configure`. Processes that never load the backend keep the limit Monarch started with,
such as one that only hosts `HostStrategy` volumes, or one that builds a `LocalClient`
from a controller it was handed and registers no direct RDMA buffer, while
`LocalRankStrategy` and routing store volumes load it as they start. Where this process's
own import exported the default and it is in effect, the backend's load, not the import,
logs TorchStore's warning about it through the `torchstore` logger. A process that
inherited or preset the value logs nothing. The import no longer creates that logger, so a
logging configuration run after the import still lets the warning through.

Connections Monarch opened before that call keep the limit they started with, so a process
that imported Monarch first makes that call before it spawns or calls actors, or sets the
variable before importing Monarch. A driver that imported Monarch first and passes
`ts.initialize(mesh=...)` has spawned that mesh by then, so it sets the variable instead.

Set the variable before importing either one, in every process that loads TorchStore,
those on other hosts included, to choose the limit yourself, Monarch's own (`10737418240`
in torchmonarch 0.6.0 and 0.7.0) included. Otherwise a process whose own import exported
the default raises Monarch's own limit to it once it loads the backend, even one its
parent pushed. The processes a driver starts afterwards get the driver's limit only where
their environment lacks the variable, since one they inherit outranks what the driver
pushes. A Monarch `configured()` block in which a process that imported Monarch first
makes that call restores Monarch's own limit when it exits.
`monarch.config.configure(codec_max_frame_length=...)` also chooses a limit other than
Monarch's own, but only in a process that imports Monarch first, since the variable
outranks it once Monarch reads it. TorchStore reads Monarch's config and then configures
it, and Monarch offers no way to do both at once, so configure Monarch from no other thread
while the call that loads the backend runs.

Monarch shuts down at exit through exit hooks, one registered when it loads and one at its
first use outside an actor, such as `this_host()`, `current_rank()` or TorchStore's first
call that starts actors, as `ts.initialize` does. Exit hooks run in reverse order of
registration, so register one that calls TorchStore after `ts.initialize` returns. In a
process that loads Monarch before exit, a hook registered before Monarch loads (such as
right after `import torchstore`) runs after Monarch's shutdown. Its TorchStore calls then
raise pyo3's `PanicException`, which derives from `BaseException` and so escapes
`except Exception`. A hook that makes the process's first TorchStore call loads Monarch
during exit instead. Python runs no exit hook registered that late. Monarch then never
shuts down at exit and the hook's calls work as on main.

Monarch code that core modules held now lives under `torchstore.monarch`, such as the
controller, storage volume and routing coordinator modules, Monarch RDMA and
`torchstore.spmd`'s bootstrap. Its log records keep the logger names they had, such as
`torchstore.routing.coordinator`, and the old module paths, such as
`torchstore.storage_volume`, still re-export every name the old modules defined except
the coordinator's `_Layouts` and `_LAYOUT_REGISTRATION_TIMEOUT_S`, which moved to
`torchstore.routing.layout_registry`. Code that used those modules changes in four ways:

- Import each name from the module that defines it, or a Monarch name from Monarch, but
  `InMemoryStore` and `StorageImpl` from `torchstore.storage_volume`, which loads Monarch.
  The module that now defines them is internal. The old paths no longer offer names their
  modules only imported, such as `spawn_actors` and Monarch's `Actor` and `endpoint` on
  `torchstore.storage_volume`. `torchstore.api`, `torchstore.spmd`, `torchstore.strategy`,
  `torchstore.transport` and `torchstore.utils` no longer offer the names that went with
  their Monarch code, such as `torchstore.api`'s actor classes and `torchstore.spmd`'s
  bootstrap. A star import of `torchstore.controller`, `torchstore.storage_volume` or
  `torchstore.transport.monarch_rdma` now gives only the names in its `__all__`
- Patch a moved name in the module whose code reads it, such as
  `torchstore.monarch.rdma.monarch_rdma_available` instead of
  `torchstore.transport.monarch_rdma.monarch_rdma_available`, since a patch through an
  old path no longer reaches that code. Each volume's store and host name now come from
  `torchstore.storage`, which is internal and may change, so set `TORCHSTORE_HOSTNAME` to
  choose the host name, and treat a patch of `torchstore.storage.InMemoryStore` or
  `torchstore.storage.get_local_hostname` as a patch of internals. Patching an attribute
  of a moved class, such as a method of `InMemoryStore`, still works through any path
  that offers the class
- `import torchstore` no longer loads the old paths or `torchstore.storage_utils`.
  TorchStore's Monarch backend loads them when it loads, so before that, or in a process
  that never loads it, import a module, as in `import torchstore.transport.monarch_rdma`,
  before reading it as an attribute
- `StorageVolume` now subclasses `StorageVolumeCore`, which is internal, and shadows the
  core's methods with its endpoints. `InMemoryStore` and `StorageImpl` now name
  `torchstore.storage` as their `__module__` and in their repr and pickles. Older
  TorchStore versions cannot load those pickles. Importing the volume module or its old
  path also loads `torchstore.storage`

`typing.get_type_hints` on `ts.initialize` and `ts.initialize_spmd` needs `ProcMesh` from
`monarch.actor` and `StoreJob` from `torchstore.monarch.spmd` in `localns`, since their
annotations name these only for type checkers.

Monarch RDMA joins transport selection once `torchstore.monarch.rdma` has loaded, which
happens wherever a storage volume handle loads, such as when the Monarch backend loads or
a worker unpickles a controller. Selection then looks up `monarch_rdma_transport_available`
and `MonarchRDMATransportBuffer` in that module each time it runs, so patch them there.

Because Monarch and TorchStore's Monarch modules now load later, these things that happened
at `import torchstore` happen then instead:

- A `logging.config.dictConfig` or `fileConfig` run in between no longer disables the
  loggers of TorchStore's Monarch modules, so their records reach your handlers. Monarch's
  job loggers, such as `monarch._src.job.job`, get Monarch's own level, stderr handler and
  `propagate=False` when the backend loads them, which replaces a level or `propagate` you
  set on them before that, so configure them after it
- TorchStore's Monarch backend binds `monarch.actor.get_or_spawn_controller` and
  `current_rank` when it loads. It takes `monarch.rdma.RDMABuffer` from
  `torchstore.monarch.rdma`, which binds it as that module loads. That happens with the
  backend or earlier through an old path, the storage volume module or an unpickled
  handle. A patch of one of these names that is active at its binding stays bound after
  the patch ends. A patch made afterwards does not reach the backend
- Monarch's import-time warning filters, such as showing its RDMA fallback warning once,
  now take precedence over filters you set for those warnings after `import torchstore`,
  and a `warnings.catch_warnings` block around the call that loads Monarch drops them when
  it exits. Set such filters after that call
- The RDMA fallback warning that direct weight sync's registration raises now names
  `torchstore.monarch.backend` as its module, so a filter on module
  `torchstore.direct_weight_sync` must name `torchstore.monarch.backend` instead
- Monarch's import runs on the first call's thread, so that call blocks its event loop
  while Monarch imports, where `import torchstore` took that time on main. To keep that
  pause out of a running loop, `import torchstore.monarch.backend` before starting it

### SPMD Usage (torchrun / torchx)

If your job is launched with `torchrun` or `torchx` rather than from inside a
Monarch runtime, use `ts.initialize_spmd()` instead of `ts.initialize()`. It
reads the standard `RANK` / `WORLD_SIZE` / `MASTER_ADDR` / `MASTER_PORT` env
vars (via `ts.spmd.SPMDEnv.from_env()`) and bootstraps a fresh Monarch context for
you. Cross-rank coordination is the caller's responsibility, typically through `torch.distributed`.

```python
import asyncio
import os

import torch
import torch.distributed as dist
import torchstore as ts


async def main() -> None:
    rank = int(os.environ["RANK"])
    dist.init_process_group("gloo")

    await ts.initialize_spmd(strategy=ts.LocalRankStrategy())
    try:
        if rank == 0:
            await ts.put("demo", torch.tensor([123], dtype=torch.int64))
            print(f"[rank=0] wrote", flush=True)

        dist.barrier()  # wait for rank 0's write before reading
        value = await ts.get("demo")
        print(f"[rank={rank}] got {value}", flush=True)
    finally:
        dist.barrier()  # wait for readers before tearing down storage
        await ts.shutdown()
        dist.destroy_process_group()


if __name__ == "__main__":
    asyncio.run(main())
```

Launch with:

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=2 your_script.py
```

To pass an explicit SPMD config instead of reading env vars, construct
`ts.spmd.SPMDEnv(...)` directly and pass it via `env=`. For a complete runnable
example, see [`example/torchstore_spmd.py`](example/torchstore_spmd.py).

Use `configure_job=` to enable Monarch Job API components before the rank 0
client attaches to the store-backed host mesh. The callback runs only on global
rank 0:

```python
def configure_job(job):
    job.enable_telemetry()
    job.remote_mount(
        "/path/to/local/source",
        mntpoint="/path/on/workers",
        python_exe=None,
    )


await ts.initialize_spmd(
    strategy=ts.LocalRankStrategy(),
    configure_job=configure_job,
)
```

### API Overview

Beyond `put`/`get`, TorchStore exposes batch operations and key-management helpers:

```python
# Batch put/get for efficient multi-key transfers
await ts.put_batch({"key1": tensor1, "key2": tensor2})
results = await ts.get_batch(["key1", "key2"])

# Key management
await ts.exists("key1")        # True
keys = await ts.keys("key")    # ["key1", "key2"] — prefix search
await ts.delete("key1")

# In-place retrieval — writes directly into a pre-allocated tensor
pre_allocated = torch.empty(100, 100)
await ts.get("my_tensor", inplace_tensor=pre_allocated)
```

### Resharding Support with DTensor

TorchStore makes it easy to fetch arbitrary slices of a distributed tensor and
to reshard between different meshes. For a full DTensor example, see
[`example/dtensor.py`](example/dtensor.py). For end-to-end resharding coverage,
see [`tests/test_resharding_basic.py`](tests/test_resharding_basic.py) and
[`tests/test_resharding_ext.py`](tests/test_resharding_ext.py).


```python

class DTensorActor(Actor):
    """
    Example pseudo-code for an Actor utilizing DTensor support.

    See example/dtensor.py for the full actor definition.
    """

    @endpoint
    async def do_put(self):
        # Typical dtensor boiler-plate
        self.initialize_distributed()
        device_mesh = init_device_mesh("cpu", self.mesh_shape)
        tensor = self.original_tensor.to("cpu")
        dtensor = distribute_tensor(tensor, device_mesh, placements=self.placements)

        print(f"Calling put with {dtensor=}")
        # This will place only the local shard into TorchStore
        await ts.put(self.shared_key, dtensor)

    @endpoint
    async def do_get(self):
        # Typical dtensor boiler-plate
        self.initialize_distributed()
        device_mesh = init_device_mesh("cpu", self.mesh_shape)
        tensor = self.original_tensor.to("cpu")
        dtensor = distribute_tensor(tensor, device_mesh, placements=self.placements)

        # TorchStore will use the metadata in the local dtensor to only fetch tensor data
        # which belongs to the local shard.
        fetched_tensor = await ts.get(self.shared_key, dtensor)
        print(fetched_tensor)
```

### State Dict Sync

TorchStore supports sharded `state_dict` exchange between actors, making it straightforward
to synchronize model weights (e.g. learner → generator in RL workflows):

```python
# Learner: publish weights after each training step
await ts.put_state_dict(model.state_dict(), "v0")

# Generator: pull weights into its own model
await ts.get_state_dict("v0", user_state_dict=serving_model.state_dict())
```

For a sample learner/generator example, see
[`example/torchstore_rl.py`](example/torchstore_rl.py).

#### Direct RDMA Weight Sync

The default (buffered) path already uses RDMA when available. When your use
case calls for it, `direct_rdma=True` bypasses the intermediate StorageVolume
entirely — the destination reads directly from the source's GPU memory via
one-sided RDMA.

```python
await ts.put_state_dict(model.state_dict(), "policy", direct_rdma=True)
await ts.get_state_dict("policy", user_state_dict=model.state_dict(), direct_rdma=True)
```

`transfer_dtype` can cast weights for transfer (e.g. float32 master weights
transferred as bfloat16).

## Routing Control Planes

Routing stores (`client_type=ts.ClientType.ROUTING`) sync state dicts from a publisher
mesh, such as RL trainers, to requester meshes, such as generators: publishers register
their layouts and requesters plan their reads from them. Ranks share the publisher set,
the strategy and these layouts through a control plane, by default `"monarch"`, which
spawns TorchStore's routing coordinator as a Monarch actor:

```python
# Driver: spawns a storage volume on each trainer rank, then sets up the control plane
await ts.initialize(
    mesh=trainer_mesh,
    strategy=ts.TorchStoreStrategy(),
    client_type=ts.ClientType.ROUTING,
    store_name="weights",
    control_plane="monarch",  # the default
)
# Each trainer rank, before its state-dict calls
await ts.client(store_name="weights", role=ts.RankRole.PUBLISHER)
# Each generator rank, before its state-dict calls (group is its mesh's index)
await ts.client(store_name="weights", role=ts.RankRole.REQUESTER, group=group)
```

Initialize each routing store once per job unless you start it over as described below.
TorchStore does not check this on any plane.
Every process keeps its cached client for a store, and on a bring-your-own plane its
record of that plane (passed or dialed), until its own `ts.shutdown` for the store. A
routing `ts.initialize` of the store also drops the client cached in the process that
calls it, and replaces that process's record with the plane it initializes, or drops the
record on `"monarch"`. A forked child drops each record of a plane dialed through an
address, with that plane's client, so the child passes the address again.
`ts.reset_client` drops only the client, so the next `ts.client`, which passes the rank's
role (and group) again, reuses the recorded plane. To start a store over:

- On the `"monarch"` plane, call `ts.initialize` again, which makes the coordinator drop
  the publishers, strategy and layouts it held and drops a client cached in the calling
  process. Then call `ts.reset_client` in every rank, and `ts.client` with its role (and
  group) before its next state-dict call, which otherwise looks the store up as a
  standard store and raises `AttributeError`
- On a bring-your-own plane, call `ts.shutdown` in the driver and in every rank, then
  `ts.initialize` with a plane holding fresh state, and pass that plane or its address
  to each rank's next `ts.client`

Give standard and routing stores different names on every plane, since a process keeps
one client per store name. In one process, whether from one event loop or from several
threads, never overlap a `ts.initialize` or `ts.shutdown` with another call for its store,
nor a first `ts.client` given `address` or `control_plane` with another `ts.client` for
the store. The exception is concurrent first `ts.client` calls through one address on one
event loop, which share its first connection and that connection's client, or a new client
on it when a `ts.reset_client` drops that client in between. Concurrent data calls and
role-less first clients for one store still work as before.

Make no routing `ts.client` or state-dict call in the driver on either plane. Outside an
actor every routing client gets rank 0 of its mesh, so a driver would publish as
`publisher/0` and write into that rank's volume.

`ts.initialize` and each new routing client log one INFO selection line naming the
control plane, its address, store, rank and pid, such as
`routing control_plane=monarch address=- store=weights rank=publisher/0 pid=1234`.
The `"monarch"` plane also logs one INFO line per call with its duration, such as
`routing control_plane=monarch init took 0.0042s`. A process gets a logging handler from
`import torchstore` only when that import exported `HYPERACTOR_CODEC_MAX_FRAME_LENGTH`.
Ranks in Monarch actors that inherit the variable, as local ones spawned after the
driver's import do, start with none, so call `ts.init_logging()` in each rank before its first
`ts.client` to see these lines, and in the driver too when a launcher sets that variable.

On the `"monarch"` plane, `ts.shutdown` cannot shut down a routing store: it looks the
name up as a standard store's controller and raises `AttributeError`. The store's
routing coordinator runs in the driver's process and keeps the strategy and every layout
it collected until that process exits or the next `ts.initialize` starts it over. This
is a known limitation, so skip `ts.shutdown` for a routing store on that plane and stop
its meshes when the job is done.

A routing store's volumes outlive `ts.shutdown` on every control plane and, unlike a
standard store's, keep every tensor published to them until `mesh` stops. Each
`ts.initialize` spawns a new volume on every rank of `mesh` beside the old ones, so stop
`mesh`, or use a new one, before you initialize a routing store on it again. Publish
each state dict under one key on every push: routing clients cannot delete, so each new
key adds tensors to the volumes, layouts to the plane and routes to every rank for good.
A key's layout is fixed at its first push, so keep its names, shapes, sharding and dtypes
on every push. Routed reads neither wait for a push nor detect one, so start each pull
after every publisher's `put_state_dict` for that push has returned, and the next push
after every pull of the key has returned.

A requester's routed `get_state_dict` that passes entries a publisher has and no earlier
call through its client planned turns off automatic garbage collection for the whole
process while it plans their routes, a few seconds for a whole state dict at 512
publishers, and then turns it back on if it was on. A strict call that passes only
entries no publisher has also turns it off and back on before it raises `KeyError`. In a
call with `strict=False`, entries no publisher has stay unplanned without turning it off.
A client built after `ts.reset_client` plans its entries again.
When such calls overlap in two threads, the one that turned collection off turns it back
on as it finishes, even while the other still plans. Other threads see collection off in
that window, and a `gc.disable()` they call meanwhile is undone when the window closes.
So a guard that restores the collection state it found, such as vLLM's CUDA-graph
capture, can see collection resume, run automatic collections inside it, and then leave
it off for the rest of the process. Keep such syncs out of that work, since turning
collection back on afterwards cannot undo collections that ran inside it. A process that
exits while another thread plans skips Python's collection before teardown, so let
planning finish first.
Importing TorchStore also registers an at-fork hook that turns collection on in a child
forked while a plan that found collection on runs in any thread, after freezing
everything the child inherited. That holds even where other code turned collection back
on before the fork. Until the child calls `gc.unfreeze()`, its collections skip what it
inherited, even through `gc.collect()`, and `gc.get_objects()` and `gc.get_referrers()`
leave those objects out. So a `subprocess` `preexec_fn` that runs in it finds collection
on but finalizes none of the garbage cycles the parent left, and a leak hunt or reference
walk in such a child calls `gc.unfreeze()` first.

### Bring Your Own Control Plane

To run this coordination on your own service, subclass `ts.ControlPlane`:

```python
from collections.abc import Collection, Mapping

import torchstore as ts


class MyControlPlane(ts.ControlPlane):
    kind = "mine"  # its name in log lines (the class name if unset)

    async def init(
        self, publishers: Collection[str], strategy: ts.TorchStoreStrategy
    ) -> None:
        """Fix the publisher set and the volume-resolution strategy."""
        ...

    async def register_layout(
        self, rank: str, key: str, registrations: Mapping[str, ts.KeyRegistration]
    ) -> None:
        """Record one publisher's layout. Idempotent per rank, never blocks."""
        ...

    async def get_layouts(self, key: str) -> ts.Registrations:
        """Every publisher layout for ``key``. Blocks until the set is complete."""
        ...

    async def strategy(self) -> ts.TorchStoreStrategy:
        """Return a strategy that resolves the registered publishers' volumes."""
        ...

    async def close(self) -> None:
        """Optional: release the connection of a plane an address's connect() returned."""
        ...
```

Your plane must keep these rules:

- `ts.initialize` calls `init` once, on the driver's instance. Until then, make
  `register_layout`, `get_layouts` and `strategy` raise on every instance, but let `close`
  release what the plane holds, since TorchStore closes planes dialed before `init`
- Give each `ts.initialize` plane state that no earlier run wrote, such as a new
  directory or key prefix: TorchStore never clears a plane, so leftover state would hand
  ranks an earlier run's layouts or volumes
- A rank's first layout per key wins. Registering a rank outside the publisher set raises
- `get_layouts` keys each publisher's layout by its rank exactly as `register_layout`
  received it
- Put `init has not run`, `not a publisher rank` or `missing publishers: [...]` in the
  matching error's message, since callers match errors by these texts
- TorchStore calls the skeleton's methods, and an address's `connect()`, with no timeout,
  so give each call a deadline of your own past which it raises. `get_layouts`, which
  waits for every publisher (Monarch's coordinator waits 300 s), raises naming the
  missing ones
- `strategy()` returns, on every rank, the strategy `init` received or a copy of it. It
  holds the store's Monarch volume handles, which can be pickled
- Unpickling can run arbitrary code, so keep anything you pickle where only the job's
  ranks can write
- TorchStore may call your plane from any thread or event loop, and a child forked after
  you passed the plane in keeps using it, so hold no loop-bound state and keep the plane
  working after fork. Each store needs its own instance
- Keep the skeleton's parameter names: TorchStore passes `init` and `register_layout`
  their arguments by keyword
- TorchStore may close a plane that `connect()` returned while calls on any thread or
  event loop, the caller's own included, are still in flight, so make those calls fail
  rather than hang

The driver passes an instance to
`ts.initialize(..., client_type=ts.ClientType.ROUTING, control_plane=plane)`. Each rank
passes `ts.client(role=..., control_plane=...)` its own instance on the same service, or
passes `address=` a `ts.ControlPlaneAddress` from the driver: a frozen dataclass subclass
whose `store_name` names the store, whose async `connect()` returns a connected plane and
whose `__str__` names the address in log lines without whitespace, such as `host:port`.
Equal addresses count as one, as the dataclass compares them, so an actor call's copy
reuses the connection. `ts.client` raises `TypeError` for an address class that the
dataclass decorator skipped, or processed with `eq=False` without an `__eq__` of its own,
since a class that inherits its base's `__eq__` compares only the base's fields. Pass a
rank's instance once: later `ts.client` calls in the process return the cached client,
which refuses any other instance, even a copy an actor call delivers. That cache holds
one client per store and process whatever its role, so give each process at most one
routing rank per store.

TorchStore closes each plane `connect()` returned, through its async `close()`, when
`ts.shutdown` releases this process's record of the store (its volumes stay, as above),
when a later selection replaces that record, or when the `ts.client` call that dialed it
fails before recording it or finds that another call recorded the same address first. If
that call is cancelled, or fails with an error that is no `Exception`, TorchStore starts
the close without waiting for it. A call cancelled after it recorded the plane, while the
connection it replaced closes, keeps the plane recorded until a shutdown or selection
closes it, and a call whose coroutine is closed while the dialed plane's `strategy()`
runs closes nothing, which leaves that plane open. Return a new plane from each
`connect()`, one whose `close()` leaves the others working. A `connect()` that returns
something other than a `ts.ControlPlane` makes that `ts.client` raise `TypeError`.
TorchStore closes a plane `connect()` returned even when you pass it back in as
`control_plane=`, and never closes another plane you passed in. Close that one yourself
only after this process's `ts.shutdown` for the store, since `ts.reset_client` keeps it
recorded for the next `ts.client`.

`ts.shutdown` raises any error from the `close()` it runs, `RuntimeError` if that close
ends cancelled or in `GeneratorExit`, or `TimeoutError` after 5s, leaving the close running
on the event loop. A failed close whose caller is cancelled, even in the loop step the
close fails in, is logged, as is every failed close that other calls start, unless it ends
cancelled with no call left waiting for it. Give each blocking call inside `close()` a
timeout of its own.

Only the coordination is yours: `ts.initialize` still spawns the store's volumes as
Monarch actors on `mesh` and installs them in the strategy, and `ts.client` takes each
rank's id from Monarch's `current_rank()`, which follows the actor's context, not the
process. So make each rank's first `ts.client` call in an endpoint called on all of `mesh`
or of a requester mesh, since a call through a slice gets the rank's position in that
slice, or in an `asyncio.to_thread` call that endpoint makes, which copies the context.
A bring-your-own control plane therefore needs `torchmonarch`.

The contract tests' own bring-your-own plane, a worked example over a local directory
rather than a plane to deploy, is
[`tests/routing/file_store_control_plane.py`](tests/routing/file_store_control_plane.py).
It passes the same contract tests as Monarch's coordinator
([`tests/routing/test_coordinator.py`](tests/routing/test_coordinator.py)), which also
call `get_layouts` with a keyword-only `timeout_s` to bound each wait, and
[`tests/routing/test_routed_sync.py`](tests/routing/test_routed_sync.py) syncs weights
through it.

## Transport Backends

TorchStore automatically selects the best available transport for each transfer. No
configuration is needed — the selection happens at runtime:

| Priority | Transport | When used |
|----------|-----------|-----------|
| 1 | **POSIX Shared Memory** | Client and storage volume are on the same host |
| 2 | **TorchComms RDMA** | Cross-host, RDMA supported by Uniflow or `torchcomms`, `USE_TORCHCOMMS` and `USE_TORCHCOMMS_RDMA` unset or `1` |
| 3 | **Monarch RDMA** | Cross-host, ibverbs available to `monarch.rdma`, `TORCHSTORE_RDMA_ENABLED` unset or `1`, `torchstore.monarch.rdma` loaded |
| 4 | **Gloo** | Cross-host fallback via collective transport |
| 5 | **Monarch RPC** | Universal fallback, always available |

Setting `TORCHSTORE_TORCH_TRANSPORT_BACKEND` to a backend listed by
`torch.distributed._transport.available_transports()` (e.g. `nixl`) enables the
torch.distributed transport, preferred after shared memory and over NIXL. Requires
a PyTorch build with `torch.distributed._transport`. The variable is read at import
time. This transport registers every tensor per request, so large state dicts are
slower than with NIXL until registrations are cached.

To force a specific transport, pass `default_transport_type` when constructing a
strategy:

```python
from torchstore.transport import TransportType

strategy = ts.LocalRankStrategy(default_transport_type=TransportType.Gloo)
await ts.initialize(num_storage_volumes=N, strategy=strategy)
```

A custom transport subclasses `torchstore.transport.buffers.TransportBuffer`. Clients
pickle each buffer to its storage volume, and the base class's `__getstate__` sets
`storage_volume_ref` to None in that copy, in the reply and in copies made with the
`copy` module, so volume-side code reads its `ctx` argument and the fields the client set.
A subclass hook extends `super().__getstate__()`, a copy of the instance dict. Where a
class after `TransportBuffer` in the subclass's method resolution order defines its own
hook, such as a mixin listed after the transport class, the base hook starts from that
hook's state instead, copying a dict and passing anything else through unchanged, handle
included. A subclass that keeps fields in `__slots__` adds them there and restores them in
its own `__setstate__`. Otherwise slot fields that no hook of the subclass returns are
lost in copies, and a `__setstate__` that unpacks the base hook's state as Python's
(dict, slots) pair fails.

## Testing

Pytest is used for testing.

For a quick installation smoke test, run:
`uv run pytest -q tests/test_store.py -k 'test_basic and MonarchRPC'`

This selects the Monarch RPC parametrizations of `test_basic` without needing
the extra transport env-var overrides.

For a more verbose test run with logs, use:
`TORCHSTORE_LOG_LEVEL=DEBUG uv run pytest -vs --log-cli-level=DEBUG tests/test_store.py::test_basic`

## License

TorchStore is BSD-3 licensed, as found in the [LICENSE](LICENSE) file.
