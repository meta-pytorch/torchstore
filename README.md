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
`910737418240` for it. A process that loads Monarch afterwards uses that limit, which
also outranks `monarch.config.configure(codec_max_frame_length=...)`, while one that
imported Monarch first keeps Monarch's own limit. Set the variable before importing
either one to choose the limit yourself.

Monarch shuts down at exit through exit hooks, one registered when it loads and one at its
first use outside an actor, such as `this_host()`, `current_rank()` or TorchStore's first
call that starts actors, as `ts.initialize` does. Exit hooks run in reverse order of
registration, so register one that calls TorchStore after `ts.initialize` returns. A hook
registered before Monarch loads, such as right after `import torchstore`, runs after
Monarch's shutdown, and its TorchStore calls raise pyo3's `PanicException`, which derives
from `BaseException`, so `except Exception` does not catch it.

Monarch code that core modules held now lives under `torchstore.monarch`, such as the
controller, storage volume and routing coordinator modules, Monarch RDMA and
`torchstore.spmd`'s bootstrap. Its log records keep the logger names they had, such as
`torchstore.routing.coordinator`, and the old module paths, such as
`torchstore.storage_volume`, still re-export every name the old modules defined except
the coordinator's `_Layouts` and `_LAYOUT_REGISTRATION_TIMEOUT_S`, which moved to
`torchstore.routing.layout_registry`. Code that used those modules changes in four ways:

- Import each name from the module that defines it, or a Monarch name from Monarch, but
  `InMemoryStore` and `StorageImpl` from `torchstore.storage_volume`, since the module that
  now defines them is internal. The old paths no longer offer names their modules only
  imported, such as `spawn_actors` and Monarch's `Actor` and `endpoint` on
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
- `StorageVolume` now subclasses `StorageVolumeCore` and shadows the core's methods with
  its endpoints, so it passes `isinstance` checks for the core while code calls the core's
  methods through `StorageVolumeCore`. Pickles of `InMemoryStore` and `StorageImpl` name
  `torchstore.storage`, which older TorchStore versions cannot load, and importing the
  volume module or its old path also loads `torchstore.storage`

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
- TorchStore's Monarch backend binds Monarch names such as
  `monarch.actor.get_or_spawn_controller` and `monarch.rdma.RDMABuffer` when it loads, so a
  patch of one that is active then stays bound after the patch ends, and a patch made
  afterwards does not reach the backend
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
