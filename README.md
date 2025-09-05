# TorchStore

A storage solution for PyTorch tensors with distributed tensor support.

# Under Construction!

Nothing to see here yet, but check back soon

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

Note: Setup currently assumes you have a working conda environment with both torch & monarch (this is currently a todo). For now the fastest way of setting up is going through [this](https://www.internalfb.com/wiki/Monarch/Monarch_xlformers_integration/Running_Monarch_on_Conda/#how-to-run-monarch) guide.

Protop: Install finetine conda & use the 'local' option for the latest packges

## Usage

```python
import torch
import asyncio
import torchstore as ts

async def main():

    # Create a store instance
    store = await ts.initialize()

    # Store a tensor
    await ts.put("my_tensor", torch.randn(3, 4))

    # Retrieve a tensor
    tensor = await ts.get("my_tensor")


if __name__ == "__main__":
    asyncio.run(main())

```

### Resharding Support with DTensor

```python
import torchstore as ts
from torch.distributed._tensor import distribute_tensor, Replicate, Shard
from torch.distributed.device_mesh import init_device_mesh

async def place_dtensor_in_store():
    device_mesh = init_device_mesh("cpu", (4,))
    tensor = torch.arange(4)
    dtensor = distribute_tensor(tensor, device_mesh, placements=[Shard(1)])

    # Store a tensor
    await ts.put("my_tensor", dtensor)


async def fetch_dtensor_from_store()
    # You can now fetch arbitrary shards of this tensor from any rank e.g.
    device_mesh = init_device_mesh("cpu", (2,2))
    tensor = torch.rand(4)
    dtensor = distribute_tensor(
        tensor,
        device_mesh,
        placements=[Replicate(), Shard(0)]
    )

    # This line copies the previously stored dtensor into local memory.
    await ts.get("my_tensor", dtensor)

def run_in_parallel(func):
    # just for demonstrative purposes
    return func

if __name__ == "__main__":
    ts.initialize()
    run_in_parallel(place_dtensor_in_store)
    run_in_parallel(fetch_dtensor_from_store)
    ts.shutdown()

# checkout out tests/test_resharding.py for more e2e examples with resharding DTensor.
```

# Contributing Guidelines

1. Build in public -- TorchStore should be OSS first, giving users a clear vision on where we want to go and how to help us get there.

2. Build the Bicycle, not the super car -- Develop value iterively, instead of trying to ship everything at once.

3. Work backwards from use cases, and leave tests!
