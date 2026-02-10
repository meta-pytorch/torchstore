# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from logging import getLogger

import pytest
import torch
import torchstore as ts
from monarch.actor import Actor, current_rank, endpoint
from torchstore.logging import init_logging
from torchstore.utils import spawn_actors

from .utils import main, transport_plus_strategy_params

init_logging()
logger = getLogger(__name__)


@pytest.mark.parametrize(*transport_plus_strategy_params())
@pytest.mark.asyncio
async def test_put_batch_empty(strategy_params, transport_type):
    """Test that _put_batch with an empty dict is a no-op."""

    class EmptyBatchActor(Actor):
        def __init__(self):
            init_logging()
            self.rank = current_rank().rank
            os.environ["LOCAL_RANK"] = str(self.rank)

        @endpoint
        async def do_test(self):
            local_client = await ts.client()
            # Should return without error
            await local_client._put_batch({})
            # Store should have no keys
            return await ts.keys()

    volume_world_size, strategy = strategy_params
    await ts.initialize(
        num_storage_volumes=volume_world_size, strategy=strategy(transport_type)
    )
    actor = await spawn_actors(volume_world_size, EmptyBatchActor, "empty_batch_actor")
    try:
        results = await actor.do_test.call()
        for _, keys in results:
            assert keys == [], f"Expected no keys after empty batch, got {keys}"
    finally:
        await ts.shutdown()


@pytest.mark.parametrize(*transport_plus_strategy_params())
@pytest.mark.asyncio
async def test_put_batch_mixed_values(strategy_params, transport_type):
    """Test _put_batch with a mix of tensor and non-tensor values."""

    class MixedBatchActor(Actor):
        def __init__(self):
            init_logging()
            self.rank = current_rank().rank
            os.environ["LOCAL_RANK"] = str(self.rank)

        @endpoint
        async def do_test(self):
            local_client = await ts.client()

            tensor_val = torch.tensor([1.0, 2.0, 3.0])
            object_val = {"hello": "world", "count": 42}

            items = {
                "batch/tensor_key": tensor_val,
                "batch/object_key": object_val,
            }
            await local_client._put_batch(items)

            # Verify both items can be retrieved
            fetched_tensor = await ts.get("batch/tensor_key")
            fetched_object = await ts.get("batch/object_key")

            return fetched_tensor, fetched_object, tensor_val, object_val

    volume_world_size, strategy = strategy_params
    await ts.initialize(
        num_storage_volumes=volume_world_size, strategy=strategy(transport_type)
    )
    actor = await spawn_actors(1, MixedBatchActor, "mixed_batch_actor")
    try:
        (
            fetched_tensor,
            fetched_object,
            expected_tensor,
            expected_object,
        ) = await actor.do_test.call_one()
        assert torch.equal(
            fetched_tensor, expected_tensor
        ), f"Tensor mismatch: {fetched_tensor} != {expected_tensor}"
        assert (
            fetched_object == expected_object
        ), f"Object mismatch: {fetched_object} != {expected_object}"
    finally:
        await ts.shutdown()


@pytest.mark.parametrize(*transport_plus_strategy_params())
@pytest.mark.asyncio
async def test_put_batch_multiple_tensors(strategy_params, transport_type):
    """Test _put_batch correctly stores and retrieves multiple tensors."""

    class MultipleTensorBatchActor(Actor):
        def __init__(self):
            init_logging()
            self.rank = current_rank().rank
            os.environ["LOCAL_RANK"] = str(self.rank)

        @endpoint
        async def do_test(self):
            local_client = await ts.client()

            num_tensors = 10
            items = {
                f"multi/tensor_{i}": torch.tensor([float(i)] * 5)
                for i in range(num_tensors)
            }
            await local_client._put_batch(items)

            # Verify all tensors
            results = {}
            for key in items:
                results[key] = await ts.get(key)
            return results, items

    volume_world_size, strategy = strategy_params
    await ts.initialize(
        num_storage_volumes=volume_world_size, strategy=strategy(transport_type)
    )
    actor = await spawn_actors(1, MultipleTensorBatchActor, "multi_tensor_actor")
    try:
        fetched, expected = await actor.do_test.call_one()
        for key in expected:
            assert torch.equal(
                fetched[key], expected[key]
            ), f"Mismatch for {key}: {fetched[key]} != {expected[key]}"
    finally:
        await ts.shutdown()


@pytest.mark.parametrize(*transport_plus_strategy_params())
@pytest.mark.asyncio
async def test_notify_put_batch_registers_keys(strategy_params, transport_type):
    """Test that notify_put_batch correctly registers keys at the controller,
    equivalent to multiple individual notify_put calls."""

    class NotifyBatchActor(Actor):
        def __init__(self):
            init_logging()
            self.rank = current_rank().rank
            os.environ["LOCAL_RANK"] = str(self.rank)

        @endpoint
        async def do_test(self):
            # Store multiple tensors via _put_batch (which uses notify_put_batch)
            local_client = await ts.client()
            items = {f"notify_test/key_{i}": torch.randn(4) for i in range(5)}
            await local_client._put_batch(items)

            # Verify all keys are visible via the controller
            all_keys = await ts.keys()
            for key in items:
                assert (
                    key in all_keys
                ), f"Key {key} not found in controller keys {all_keys}"

            # Also verify they can be individually fetched (controller knows about them)
            for key, expected_val in items.items():
                fetched = await ts.get(key)
                assert torch.equal(fetched, expected_val), f"Mismatch for {key}"

            return True

    volume_world_size, strategy = strategy_params
    await ts.initialize(
        num_storage_volumes=volume_world_size, strategy=strategy(transport_type)
    )
    actor = await spawn_actors(1, NotifyBatchActor, "notify_batch_actor")
    try:
        result = await actor.do_test.call_one()
        assert result is True
    finally:
        await ts.shutdown()


if __name__ == "__main__":
    main(__file__)
