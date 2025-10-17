# TorchStore

A storage solution for PyTorch tensors with distributed tensor support.

TorchStore provides a distributed, asynchronous tensor storage system built on top of
Monarch actors. It enables efficient storage and retrieval of PyTorch tensors across
multiple processes and nodes with support for various transport mechanisms including
RDMA when available.

Key Features:
- Distributed tensor storage with configurable storage strategies
- Asynchronous put/get operations for tensors and arbitrary objects
- Support for PyTorch state_dict serialization/deserialization
- Multiple transport backends (RDMA, regular TCP) for optimal performance
- Flexible storage volume management and sharding strategies

Note: Although this may change in the future, TorchStore only supports multi-processing/multi-node jobs launched with Monarch.
For more information on what Monarch is, see https://github.com/meta-pytorch/monarch?tab=readme-ov-file#monarch-


> ⚠️ **Early Development Warning** TorchStore is currently in an experimental
> stage. You should expect bugs, incomplete features, and APIs that may change
> in future versions. The project welcomes bugfixes, but to make sure things are
> well coordinated you should discuss any significant change before starting the
> work. It's recommended that you signal your intention to contribute in the
> issue tracker, either by filing a new issue or by claiming an existing one.

## Installation

### Env Setup
```bash
conda create -n torchstore python=3.12
pip install torch

git clone git@github.com:meta-pytorch/monarch.git
python monarch/scripts/install_nightly.py

git clone git@github.com:meta-pytorch/torchstore.git
cd torchstore
pip install -e .
```


### Development Installation

To install the package in development mode:

```bash
# Clone the repository
git clone https://github.com/your-username/torchstore.git
cd torchstore

# Install in development mode
pip install -e .

# Install development dependencies
pip install -e '.[dev]'

# NOTE: It's common to run into libpytorch issues. A good workaround is to export:
# export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
```



### Regular Installation

To install the package directly from the repository:

```bash
pip install git+https://github.com/your-username/torchstore.git
```

Once installed, you can import it in your Python code:

```python
import torchstore
```

## Usage

```python
import asyncio

import torch

from monarch.actor import Actor, current_rank, endpoint

import torchstore as ts
from torchstore.utils import spawn_actors


WORLD_SIZE = 4


# In monarch, Actors are the way we represent multi-process/node applications. For additional details, see:
# https://github.com/meta-pytorch/monarch?tab=readme-ov-file#monarch-
class ExampleActor(Actor):
    def __init__(self, world_size=WORLD_SIZE):
        self.rank = current_rank().rank
        self.world_size = WORLD_SIZE

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

if  __name__ == "__main__":
    asyncio.run(main())

# Expected output
# [0] [2] Rank=[2] Fetched tensor([3]) from other_rank=3
# [0] [0] Rank=[0] Fetched tensor([1]) from other_rank=1
# [0] [3] Rank=[3] Fetched tensor([0]) from other_rank=0
# [0] [1] Rank=[1] Fetched tensor([2]) from other_rank=2

```

### Resharding Support with DTensor

TorchStore makes it easy to fetch arbitraty slices of any Distributed Tensor.
For a full DTensor example, see [examples/dtensor.py](https://github.com/meta-pytorch/torchstore/blob/main/example/dtensor.py)


```python

class DTensorActor(Actor):
    """
    Example pseudo-code for an Actor utilizing DTensor support

    Full actor definition in [examples/dtensor.py](https://github.com/meta-pytorch/torchstore/blob/main/example/dtensor.py)
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

        # Torchstore will use the metadata in the local dtensor to only fetch tensor data
        # which belongs to the local shard.
        fetched_tensor = await ts.get(self.shared_key, dtensor)
        print(fetched_tensor)

# checkout out tests/test_resharding.py for more e2e examples with resharding DTensor.
```

# Testing

Pytest is used for testing. For an examples of how to run tests (and get logs), see:
`TORCHSTORE_LOG_LEVEL=DEBUG pytest -vs --log-cli-level=DEBUG tests/test_models.py::test_basic

## License

Torchstore is BSD-3 licensed, as found in the [LICENSE](LICENSE) file.
