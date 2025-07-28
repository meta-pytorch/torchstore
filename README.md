# TorchStore

A storage solution for PyTorch tensors with distributed tensor support.

# Under Construction!

Nothing to see here yet, but check back soon

## Installation

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
from torchstore import MultiProcessStore
```

## Usage

```python
import torch
import asyncio
from torchstore import MultiProcessStore

async def main():

    # Create a store instance
    store = await MultiProcessStore.create_store()

    # Store a tensor
    await store.put("my_tensor", torch.randn(3, 4))

    # Retrieve a tensor
    tensor = await store.get("my_tensor")


if __name__ == "__main__":
    asyncio.run(main())

# checkout out tests/test_resharding.py for more examples with resharding DTensor!

```

## Usage

```python
import torch
import asyncio
from torchstore import MultiProcessStore

async def main():

    # Create a store instance
    store = await MultiProcessStore.create_store()

    # Store a tensor
    await store.put("my_tensor", torch.randn(3, 4))

    # Retrieve a tensor
    tensor = await store.get("my_tensor")


if __name__ == "__main__":
    asyncio.run(main())

# checkout out tests/test_resharding.py for more examples with resharding DTensor!

```

### Resharding Support with DTensor
```python
from torchstore import MultiProcessStore
from torch.distributed._tensor import distribute_tensor, Replicate, Shard
from torch.distributed.device_mesh import init_device_mesh

async def place_dtensor_in_store():
    device_mesh = init_device_mesh("cpu", (4,))
    tensor = torch.arange(4)
    dtensor = distribute_tensor(tensor, device_mesh, placements=[Shard(1)])

    # Create a store instance
    store = await MultiProcessStore.create_store()

    # Store a tensor
    await store.put("my_tensor", dtensor)


async def fetch_dtensor_from_store()
    # You can now fetch arbitrary shards of this tensor from any rank e.g.
    device_mesh = init_device_mesh("cpu", (2,2))
    tensor = original_tensor = torch.rand(4)
    dtensor = distribute_tensor(
        tensor,
        device_mesh,
        placements=[Replicate(), Shard(0)]
    )

    # This line copies the previously stored dtensor into local memory.
    await store.get("my_tensor", dtensor)
```

# Contributing Guidelines

1. Build in public -- TorchStore should be OSS first, giving users a clear vision on where we want to go and how to help us get there.

2. Build the Bicycle, not the super car -- Develop value iterively, instead of trying to ship everything at once.

3. Work backwards from use cases, and leave tests!
