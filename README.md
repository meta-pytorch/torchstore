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


# Contributing Guidelines

1. Build in public -- TorchStore should be OSS first, giving users a clear vision on where we want to go and how to help us get there.

2. Build the Bicycle, not the super car -- Develop value iterively, instead of trying to ship everything at once.

3. Work backwards from use cases, and leave tests!
