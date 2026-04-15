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

> **Note:** TorchStore requires distributed jobs to be launched with [Monarch](https://github.com/meta-pytorch/monarch). Direct SPMD support is planned.


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
| 2 | **Monarch RDMA** | Cross-host, `monarch.rdma` available |
| 3 | **TorchComms RDMA** | Cross-host, `torchcomms` installed |
| 4 | **Gloo** | Cross-host fallback via collective transport |
| 5 | **Monarch RPC** | Universal fallback, always available |

To force a specific transport, pass `default_transport_type` when constructing a
strategy:

```python
from torchstore.transport import TransportType

strategy = ts.LocalRankStrategy(default_transport_type=TransportType.Gloo)
await ts.initialize(num_storage_volumes=N, strategy=strategy)
```

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
