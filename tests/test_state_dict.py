# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import math
import os
import tempfile
from logging import getLogger
from typing import Union

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
import torchstore as ts
from monarch.actor import Actor, current_rank, endpoint
from torch.distributed.checkpoint._nested_dict import flatten_state_dict
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
)
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor, Replicate, Shard
from torchstore.state_dict_utils import TensorReference, TorchStoreStateDict
from torchstore.utils import spawn_actors

from .utils import main, set_transport_type, transport_plus_strategy_params

logger = getLogger(__name__)


MODEL_LINER_LENGTH = 10


def _setup_process_group():
    """Set up minimal distributed environment for DTensor testing."""

    if not dist.is_initialized():
        # Set minimal environment variables for single process
        import os

        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault(
            "MASTER_PORT", "29501"
        )  # Different port to avoid conflicts
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")

        # Initialize single-process group
        dist.init_process_group(
            backend="gloo",  # CPU backend
            rank=0,
            world_size=1,
        )
        return True


class UnitModule(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.l1 = nn.Linear(MODEL_LINER_LENGTH, MODEL_LINER_LENGTH, device=device)
        self.seq = nn.Sequential(
            nn.ReLU(),
            nn.Linear(MODEL_LINER_LENGTH, MODEL_LINER_LENGTH, device=device),
            nn.ReLU(),
        )
        self.l2 = nn.Linear(MODEL_LINER_LENGTH, MODEL_LINER_LENGTH, device=device)

    def forward(self, x):
        return self.l2(self.seq(self.l1(x)))


class CompositeParamModel(nn.Module):
    """
    ref:
    https://github.com/pytorch/pytorch/blob/e2c9d8d6414927ce754bbc40b767edf103cf16da/torch/testing/_internal/common_dist_composable.py#L52
    """

    def __init__(self, device: Union[torch.device, str] = "cpu"):
        super().__init__()
        if isinstance(device, str):
            device = torch.device(device)

        self.l = nn.Linear(MODEL_LINER_LENGTH, MODEL_LINER_LENGTH, device=device)
        self.u1 = UnitModule(device)
        self.u2 = UnitModule(device)
        self.p = nn.Parameter(
            torch.randn((MODEL_LINER_LENGTH, MODEL_LINER_LENGTH), device=device)
        )
        # TODO: buffers are failing atm, because they are not DTensors and thus
        # have unique values on each rank. This isn't necessarily a bug,
        # but it makes it a little harder to compare against a DCP checkpoint directly
        # self.register_buffer(
        #     "buffer", torch.randn((MODEL_LINER_LENGTH, MODEL_LINER_LENGTH), device=device), persistent=True
        # )

    def forward(self, x):
        a = self.u2(self.u1(self.l(x)))
        b = self.p
        return torch.mm(a, b)


class DCPParityTest(Actor):
    """Since DCP is known to have resharding support, this test uses DCP as a proxy to confirm
    correctness of torchstore resharding.
    """

    torchstore_checkpoint_fn: str = "torchstore_checkpoint.pt"

    def __init__(self, mesh_shape, dcp_checkpoint_fn, file_store_name):
        self.mesh_shape = mesh_shape
        self.world_size = math.prod(mesh_shape)
        self.dcp_checkpoint_fn = dcp_checkpoint_fn
        self.file_store_name = file_store_name
        self.rank = current_rank().rank
        # needed for LocalRankStrategy
        os.environ["LOCAL_RANK"] = str(self.rank)

    def rlog(self, msg):
        logger.info(f"rank: {self.rank} {msg}")

    def build_model_optimizer(self):
        mesh_dim_names = ["dp", "tp"] if len(self.mesh_shape) == 2 else None
        device_mesh = init_device_mesh(
            "cpu", self.mesh_shape, mesh_dim_names=mesh_dim_names
        )

        model = CompositeParamModel()
        model = fully_shard(model, mesh=device_mesh)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        return model, optimizer

    def initialize_distributed(self):
        self.rlog(f"Initialize process group using {self.file_store_name=} ")
        torch.distributed.init_process_group(
            backend="gloo",
            rank=self.rank,
            world_size=self.world_size,
            init_method=f"file://{self.file_store_name}",
        )

        # this barrier is more to make sure torch.distibuted is working
        self.rlog("barrrer")
        torch.distributed.barrier()

    @endpoint
    async def do_put(self):
        self.initialize_distributed()

        torch.manual_seed(0)

        model, optimizer = self.build_model_optimizer()
        for _ in range(5):
            optimizer.zero_grad()
            loss = model(torch.randn(8, MODEL_LINER_LENGTH)).sum()
            loss.backward()
            optimizer.step()

        state_dict = {
            "model": get_model_state_dict(model),
            "optimizer": get_optimizer_state_dict(model, optimizer),
        }

        dcp.save(state_dict, checkpoint_id=self.dcp_checkpoint_fn)
        await ts.put_state_dict(state_dict, "v0")

    @endpoint
    async def do_get(self):
        self.initialize_distributed()

        model, optimizer = self.build_model_optimizer()
        state_dict = {
            "model": get_model_state_dict(model),
            "optimizer": get_optimizer_state_dict(model, optimizer),
        }
        dcp_state_dict = copy.deepcopy(state_dict)
        dcp.load(dcp_state_dict, checkpoint_id=self.dcp_checkpoint_fn)

        torchstore_state_dict = copy.deepcopy(state_dict)
        await ts.get_state_dict("v0", torchstore_state_dict)

        return dcp_state_dict, torchstore_state_dict


@pytest.mark.parametrize(*transport_plus_strategy_params())
@pytest.mark.asyncio
async def test_state_dict(strategy_params, transport_type):
    set_transport_type(transport_type)

    class Trainer(Actor):
        # Monarch RDMA does not work outside of an actor, so we need
        # to wrapp this test first
        # TODO: assert this within rdma buffer
        def __init__(self) -> None:
            self.rank = current_rank().rank
            # needed for LocalRankStrategy
            os.environ["LOCAL_RANK"] = str(self.rank)

        @endpoint
        async def do_test(self):
            model = CompositeParamModel()
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

            for _ in range(5):
                optimizer.zero_grad()
                loss = model(torch.randn(8, MODEL_LINER_LENGTH)).sum()
                loss.backward()
                optimizer.step()

            state_dict = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            await ts.put_state_dict(state_dict, "v0")
            print(state_dict)
            fetched_state_dict = await ts.get_state_dict("v0")
            return state_dict, fetched_state_dict

    _, strategy = strategy_params
    await ts.initialize(num_storage_volumes=1, strategy=strategy)
    trainer = await spawn_actors(1, Trainer, "trainer")
    try:
        state_dict, fetched_state_dict = await trainer.do_test.call_one()
    finally:
        await ts.shutdown()
    _assert_equal_state_dict(state_dict, fetched_state_dict)


@pytest.mark.skip("TODO(kaiyuan-li@): fix this test")
@pytest.mark.parametrize(*transport_plus_strategy_params())
@pytest.mark.asyncio
async def test_dcp_sharding_parity(strategy_params, transport_type):
    set_transport_type(transport_type)

    for save_mesh_shape, get_mesh_shape in [
        ((2,), (4,)),
        ((4,), (2,)),
        ((2, 2), (4,)),
        ((2,), (2, 4)),
        ((4, 2), (2, 4)),
    ]:
        save_world_size = math.prod(save_mesh_shape)
        get_world_size = math.prod(get_mesh_shape)
        logger.info(
            f"Testing -- save_mesh_shape: {save_mesh_shape} get_mesh_shape: {get_mesh_shape}"
        )

        _, strategy = strategy_params
        await ts.initialize(
            num_storage_volumes=save_world_size if strategy is not None else 1,
            strategy=strategy,
        )
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                dcp_checkpoint_fn = os.path.join(tmpdir, "dcp_checkpoint.pt")

                save_world = await spawn_actors(
                    save_world_size,
                    DCPParityTest,
                    "save_world",
                    mesh_shape=save_mesh_shape,
                    dcp_checkpoint_fn=dcp_checkpoint_fn,
                    file_store_name=os.path.join(tmpdir, "save_world"),
                )
                await save_world.do_put.call()

                get_world = await spawn_actors(
                    get_world_size,
                    DCPParityTest,
                    "get_world",
                    mesh_shape=get_mesh_shape,
                    dcp_checkpoint_fn=dcp_checkpoint_fn,
                    file_store_name=os.path.join(tmpdir, "get_world"),
                )
                value_mesh = await get_world.do_get.call()
                for coord, val in value_mesh:
                    try:
                        dcp_state_dict, torchstore_state_dict = val
                        _assert_equal_state_dict(dcp_state_dict, torchstore_state_dict)
                    except Exception as e:
                        raise AssertionError(
                            f"Assertion failed on rank {coord.rank} ({save_mesh_shape=} {get_mesh_shape=}): {e}"
                        ) from e
        finally:
            # TODO: Investigate monarch bug with proc_mesh.stop()
            # await save_world._proc_mesh.stop()
            # await get_world._proc_mesh.stop()
            await ts.shutdown()


def _assert_equal_state_dict(state_dict1, state_dict2):
    flattened_state_dict_1, _ = flatten_state_dict(state_dict1)
    flattened_state_dict_2, _ = flatten_state_dict(state_dict2)

    assert len(flattened_state_dict_1) == len(
        flattened_state_dict_2
    ), f"{flattened_state_dict_1.keys()=}\n{flattened_state_dict_2.keys()=}"
    for key in flattened_state_dict_1:
        assert key in flattened_state_dict_2
        if isinstance(flattened_state_dict_1[key], torch.Tensor):
            t1, t2 = flattened_state_dict_1[key], flattened_state_dict_2[key]
            if isinstance(t1, DTensor):
                t1 = t1._local_tensor
            if isinstance(t2, DTensor):
                t2 = t2._local_tensor

            assert torch.equal(t1, t2), (
                f"{key=} {flattened_state_dict_1[key]=} {t1.shape=} {flattened_state_dict_2[key]=} {t2.shape=}",
            )
        else:
            assert (
                flattened_state_dict_1[key] == flattened_state_dict_2[key]
            ), f"{key=} {flattened_state_dict_1[key]=} {flattened_state_dict_2[key]=}"


def _verify_reconstructed_state_dict(flattened_original, flattened_reconstructed):
    """Utility function to verify reconstructed state dict matches original."""
    for key, original_value in flattened_original.items():
        reconstructed_value = flattened_reconstructed[key]

        if hasattr(original_value, "_local_tensor"):  # DTensor check
            # Should be reconstructed as DTensor
            assert hasattr(
                reconstructed_value, "_local_tensor"
            ), f"Expected DTensor for {key}"

            # Verify local tensor data matches
            assert torch.equal(
                original_value._local_tensor, reconstructed_value._local_tensor
            ), f"Local tensor data mismatch for {key}"

            # Verify global shape matches
            assert (
                original_value.shape == reconstructed_value.shape
            ), f"Global shape mismatch for {key}"

            # Verify placements match
            assert (
                original_value.placements == reconstructed_value.placements
            ), f"Placements mismatch for {key}"

        elif isinstance(original_value, torch.Tensor):
            # Regular tensors should remain the same
            assert torch.equal(
                original_value, reconstructed_value
            ), f"Regular tensor mismatch for {key}"
        else:
            # Non-tensor values should be preserved
            assert (
                original_value == reconstructed_value
            ), f"Non-tensor value mismatch for {key}"


def test_torchstore_state_dict():
    """Test TorchStoreStateDict class with various tensor types and reconstruction."""

    # Create a state dict with various tensor types and shapes
    original_state_dict = {
        # Scalar tensor (0D)
        "scalar": torch.tensor(42.5, dtype=torch.float32),
        # 1D tensors with different dtypes
        "vector_float": torch.randn(10, dtype=torch.float32),
        "vector_int": torch.randint(0, 100, (5,), dtype=torch.int64),
        "vector_half": torch.randn(8, dtype=torch.float16),
        # 2D tensors with different dtypes
        "matrix_float": torch.randn(3, 4, dtype=torch.float32),
        "matrix_double": torch.randn(2, 3, dtype=torch.float64),
        "matrix_int": torch.randint(-50, 50, (4, 2), dtype=torch.int32),
        # Nested structure
        "model": {
            "layer1": {
                "weight": torch.randn(5, 3, dtype=torch.float32),
                "bias": torch.randn(5, dtype=torch.float32),
            },
            "layer2": {
                "weight": torch.randn(2, 5, dtype=torch.float32),
                "bias": torch.randn(2, dtype=torch.float32),
            },
        },
        # Mixed with non-tensor data
        "metadata": {
            "epoch": 10,
            "learning_rate": 0.001,
            "optimizer_state": torch.randn(3, 3, dtype=torch.float32),
        },
        # List with tensors (note: flattened state dict doesn't preserve list structure)
        "layer_weights": [
            torch.randn(2, 2, dtype=torch.float32),
            torch.tensor(123, dtype=torch.int32),
        ],
    }

    # Create TorchStoreStateDict
    torchstore_state_dict = TorchStoreStateDict.from_state_dict(original_state_dict)

    # Verify blob properties
    blob = torchstore_state_dict.tensor_blob
    assert blob.dtype == torch.uint8, f"Expected uint8 blob, got {blob.dtype}"
    assert blob.dim() == 1, f"Expected 1D blob, got {blob.dim()}D"

    # 1. Flatten original state dict
    original_flattened, _ = flatten_state_dict(original_state_dict)

    # 2. Verify keys match between original flattened and torchstore flattened state dict
    assert set(original_flattened.keys()) == set(
        torchstore_state_dict.flattened_state_dict.keys()
    ), "Keys don't match between original and torchstore flattened state dicts"

    # 3. Verify tensor references and calculate total size
    # _verify_tensor_references(torchstore_state_dict, original_flattened)

    # Calculate total size for blob verification
    total_size = 0
    for key, original_value in original_flattened.items():
        if isinstance(original_value, torch.Tensor):
            tensor_to_size = (
                original_value._local_tensor
                if hasattr(original_value, "_local_tensor")
                else original_value
            )
            total_size += tensor_to_size.numel() * tensor_to_size.element_size()

    # Verify tensor blob size matches total size
    assert (
        len(blob) == total_size
    ), f"Tensor blob size {len(blob)} doesn't match expected total size {total_size}"

    # Reconstruct the state dict
    reconstructed_state_dict = torchstore_state_dict.to_state_dict()

    # Compare flattened versions - simpler than recursive comparison
    original_flattened, original_mapping = flatten_state_dict(original_state_dict)
    reconstructed_flattened, reconstructed_mapping = flatten_state_dict(
        reconstructed_state_dict
    )

    # Verify mappings are identical (structure preserved)
    assert (
        original_mapping == reconstructed_mapping
    ), "State dict structure mappings don't match"

    # Verify keys match
    assert set(original_flattened.keys()) == set(
        reconstructed_flattened.keys()
    ), "Flattened keys don't match"

    # Verify reconstruction using utility function
    _verify_reconstructed_state_dict(original_flattened, reconstructed_flattened)


def _verify_tensor_references(torchstore_state_dict, flattened_original):
    """Utility function to verify TensorReference objects in flattened state dict."""
    for key, original_value in flattened_original.items():
        torchstore_value = torchstore_state_dict.flattened_state_dict[key]

        if isinstance(original_value, torch.Tensor):
            if hasattr(original_value, "_local_tensor"):  # DTensor check
                # DTensor should be converted to TensorReference with tensor_slice
                assert isinstance(torchstore_value, TensorReference)
                assert (
                    torchstore_value.tensor_slice is not None
                ), f"DTensor at {key} should have tensor_slice"
                assert (
                    torchstore_value.device_mesh is not None
                ), f"DTensor at {key} should have device_mesh"
                assert (
                    torchstore_value.placements is not None
                ), f"DTensor at {key} should have placements"

                # Verify local tensor metadata
                local_tensor = original_value._local_tensor
                assert torchstore_value.shape == tuple(local_tensor.shape)
                assert torchstore_value.dtype == local_tensor.dtype
            else:
                # Regular tensor should not have tensor_slice
                assert isinstance(torchstore_value, TensorReference)
                assert (
                    torchstore_value.tensor_slice is None
                ), f"Regular tensor at {key} should not have tensor_slice"
                assert torchstore_value.shape == tuple(original_value.shape)
                assert torchstore_value.dtype == original_value.dtype


def test_torchstore_state_dict_with_dtensor():
    """Test TorchStoreStateDict with DTensor support."""
    _setup_process_group()

    # Create single-device mesh (CPU only)
    device_mesh = DeviceMesh("cpu", [0])

    # Create DTensor from local tensor
    local_tensor = torch.arange(4 * 6, dtype=torch.float32).reshape(4, 6)
    dtensor = DTensor.from_local(local_tensor, device_mesh, [Replicate()])

    # Create state dict with DTensor and regular tensor
    original_state_dict = {
        "regular_tensor": torch.randn(3, 3),
        "dtensor": dtensor,
        "nested": {
            "another_dtensor": DTensor.from_local(
                torch.ones(2, 3), device_mesh, [Shard(0)]
            ),
            "metadata": {"test": "value"},
        },
    }

    # Test serialization
    torchstore_state_dict = TorchStoreStateDict.from_state_dict(original_state_dict)

    # Verify DTensor metadata is preserved using utility function
    flattened_original, _ = flatten_state_dict(original_state_dict)
    _verify_tensor_references(torchstore_state_dict, flattened_original)

    # Test deserialization
    reconstructed_state_dict = torchstore_state_dict.to_state_dict()

    # Verify reconstruction using utility function
    flattened_reconstructed, _ = flatten_state_dict(reconstructed_state_dict)
    _verify_reconstructed_state_dict(flattened_original, flattened_reconstructed)

    dist.destroy_process_group()


class TorchStoreStateDictDTensorActor(Actor):
    """Actor for testing TorchStoreStateDict with distributed DTensors."""

    def __init__(self, mesh_shape, original_tensor, file_store_name):
        self.rank = current_rank().rank
        self.mesh_shape = mesh_shape
        self.world_size = math.prod(mesh_shape)
        self.original_tensor = original_tensor
        self.file_store_name = file_store_name
        os.environ["LOCAL_RANK"] = str(self.rank)

    def initialize_distributed(self):
        torch.distributed.init_process_group(
            backend="gloo",
            rank=self.rank,
            world_size=self.world_size,
            init_method=f"file://{self.file_store_name}",
        )
        torch.distributed.barrier()

    @endpoint
    async def test_state_dict_with_dtensor(self):
        from torch.distributed._tensor import distribute_tensor

        self.initialize_distributed()

        device_mesh = init_device_mesh("cpu", self.mesh_shape)

        # Create DTensors with different placements
        tensor1 = self.original_tensor.clone()
        tensor2 = torch.arange(12, dtype=torch.float32).reshape(3, 4)

        dtensor_sharded = distribute_tensor(tensor1, device_mesh, placements=[Shard(0)])
        dtensor_replicated = distribute_tensor(
            tensor2, device_mesh, placements=[Replicate()]
        )

        # Create state dict with DTensors and regular tensors
        original_state_dict = {
            "sharded_dtensor": dtensor_sharded,
            "replicated_dtensor": dtensor_replicated,
            "regular_tensor": torch.randn(2, 2),
            "nested": {
                "weight": dtensor_sharded,
                "epoch": 5,
            },
        }

        # Test TorchStoreStateDict serialization
        torchstore_sd = TorchStoreStateDict.from_state_dict(original_state_dict)

        # Verify blob is created
        assert torchstore_sd.tensor_blob.dtype == torch.uint8
        assert torchstore_sd.tensor_blob.dim() == 1

        # Verify DTensor metadata is preserved
        flattened_original, _ = flatten_state_dict(original_state_dict)
        for key, value in flattened_original.items():
            ref = torchstore_sd.flattened_state_dict.get(key)
            if isinstance(value, DTensor):
                assert isinstance(ref, TensorReference)
                assert ref.tensor_slice is not None
                assert ref.device_mesh is not None
                assert ref.placements is not None

        # Test reconstruction
        reconstructed_sd = torchstore_sd.to_state_dict()
        flattened_reconstructed, _ = flatten_state_dict(reconstructed_sd)

        # Verify data integrity
        for key, original_value in flattened_original.items():
            reconstructed_value = flattened_reconstructed[key]
            if isinstance(original_value, DTensor):
                original_local = original_value._local_tensor
                if isinstance(reconstructed_value, DTensor):
                    reconstructed_local = reconstructed_value._local_tensor
                else:
                    reconstructed_local = reconstructed_value
                assert torch.equal(
                    original_local, reconstructed_local
                ), f"Mismatch for {key} on rank {self.rank}"
            elif isinstance(original_value, torch.Tensor):
                assert torch.equal(original_value, reconstructed_value)
            else:
                assert original_value == reconstructed_value

        return self.rank, "success"

    @endpoint
    async def destroy_process_group(self):
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize(*transport_plus_strategy_params())
@pytest.mark.asyncio
async def test_torchstore_state_dict_dtensor_distributed(
    strategy_params, transport_type
):
    """Test TorchStoreStateDict with DTensors across multiple distributed actors."""
    set_transport_type(transport_type)

    num_actors = 2
    mesh_shape = (num_actors,)
    # Tensor shape must be divisible by num_actors for Shard(0)
    original_tensor = torch.arange(num_actors * 4 * 3, dtype=torch.float32).reshape(
        num_actors * 4, 3
    )

    _, strategy = strategy_params
    await ts.initialize(
        num_storage_volumes=num_actors if strategy is not None else 1,
        strategy=strategy,
    )

    actors = None
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            actors = await spawn_actors(
                num_actors,
                TorchStoreStateDictDTensorActor,
                "dtensor_state_dict_test",
                mesh_shape=mesh_shape,
                original_tensor=original_tensor,
                file_store_name=os.path.join(tmpdir, "pg_store"),
            )

            results = await actors.test_state_dict_with_dtensor.call()
            for coord, (rank, status) in results:
                assert status == "success", f"Actor rank {rank} failed"

    finally:
        if actors is not None:
            await actors.destroy_process_group.call()
        await ts.shutdown()


if __name__ == "__main__":
    main(__file__)
