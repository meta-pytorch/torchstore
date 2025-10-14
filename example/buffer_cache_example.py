#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Example demonstrating TransportBufferCache usage for improved performance.

This example shows how to use TransportBufferCache to avoid expensive RDMA
buffer allocations when saving model checkpoints repeatedly during training.
"""

import asyncio
import time

import torch
import torchstore as ts


class Trainer:
    """Example trainer class that saves checkpoints periodically."""

    def __init__(self, model):
        self.model = model
        self.optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        # Create a buffer cache to reuse across checkpoints
        self.state_dict_cache = ts.TransportBufferCache()

    async def save_checkpoint(self, epoch: int):
        """Save a checkpoint using the cached buffers for improved performance."""
        state_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
        }

        # The first call allocates buffers and caches them
        # Subsequent calls reuse the cached buffers, avoiding expensive RDMA allocation
        await ts.put_state_dict(
            state_dict, f"checkpoint_epoch_{epoch}", cache=self.state_dict_cache
        )

    async def train(self, num_epochs: int = 10):
        """Simulate training loop with periodic checkpointing."""
        for epoch in range(num_epochs):
            # Simulate training step
            loss = self._train_step()
            print(f"Epoch {epoch}: loss = {loss:.4f}")

            # Save checkpoint
            start = time.perf_counter()
            await self.save_checkpoint(epoch)
            elapsed = time.perf_counter() - start
            print(f"  Checkpoint saved in {elapsed:.4f}s")

    def _train_step(self):
        """Simulate a training step."""
        inputs = torch.randn(32, 10)
        targets = torch.randn(32, 5)
        outputs = self.model(inputs)
        loss = torch.nn.functional.mse_loss(outputs, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


async def main():
    """Main example function."""
    print("=" * 80)
    print("TorchStore TransportBufferCache Example")
    print("=" * 80)

    # Initialize TorchStore
    print("\nInitializing TorchStore...")
    await ts.initialize()

    try:
        # Create a simple model
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 5),
        )

        # Example 1: Basic tensor caching
        print("\n" + "=" * 80)
        print("Example 1: Basic Tensor Caching")
        print("=" * 80)

        cache = ts.TransportBufferCache()
        tensor = torch.randn(1000, 1000)

        print("\nFirst put (allocates buffer):")
        start = time.perf_counter()
        await ts.put("my_tensor", tensor, cache=cache)
        elapsed_first = time.perf_counter() - start
        print(f"  Time: {elapsed_first:.4f}s")

        print("\nSecond put (reuses cached buffer):")
        tensor = torch.randn(1000, 1000)  # Different data, same shape
        start = time.perf_counter()
        await ts.put("my_tensor", tensor, cache=cache)
        elapsed_second = time.perf_counter() - start
        print(f"  Time: {elapsed_second:.4f}s")

        if elapsed_first > elapsed_second:
            speedup = elapsed_first / elapsed_second
            print(f"\n✓ Speedup with cache: {speedup:.2f}x")
        else:
            print("\n  Note: Speedup is most noticeable with RDMA enabled")

        # Example 2: Training with periodic checkpointing
        print("\n" + "=" * 80)
        print("Example 2: Training Loop with Cached Checkpoints")
        print("=" * 80)

        trainer = Trainer(model)
        await trainer.train(num_epochs=5)

        print("\n✓ Training completed with cached checkpoints!")

        # Example 3: Cache management
        print("\n" + "=" * 80)
        print("Example 3: Cache Management")
        print("=" * 80)

        # Check cache size
        print(f"\nCache has {len(trainer.state_dict_cache._buffers)} buffers")

        # Clear specific checkpoint
        print("\nClearing cache for checkpoint_epoch_0...")
        trainer.state_dict_cache.remove("checkpoint_epoch_0/model/0.weight")
        print(f"Cache now has {len(trainer.state_dict_cache._buffers)} buffers")

        # Clear all cache
        print("\nClearing entire cache...")
        trainer.state_dict_cache.clear()
        print(f"Cache now has {len(trainer.state_dict_cache._buffers)} buffers")

        print("\n✓ Cache management completed!")

        # Example 4: Without cache (for comparison)
        print("\n" + "=" * 80)
        print("Example 4: Without Cache (existing behavior)")
        print("=" * 80)

        print("\nSaving without cache:")
        start = time.perf_counter()
        await ts.put_state_dict(
            trainer.model.state_dict(), "checkpoint_no_cache"
        )
        elapsed = time.perf_counter() - start
        print(f"  Time: {elapsed:.4f}s")
        print("\n✓ Existing API still works without cache parameter!")

    finally:
        print("\n" + "=" * 80)
        print("Shutting down TorchStore...")
        await ts.shutdown()
        print("✓ Done!")
        print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
