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
import torch.distributed.checkpoint as dcp
import torch.nn as nn

import torchstore as ts

from monarch.actor import Actor, current_rank, endpoint
from torch.distributed.checkpoint._nested_dict import flatten_state_dict
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor
from torchstore.state_dict_utils import (
    generate_tensor_blob,
    reconstruct_state_dict_from_tensor_blob,
    TensorReference,
)
from torchstore.utils import spawn_actors

from .utils import main, transport_plus_strategy_params

logger = getLogger(__name__)


MODEL_LINER_LENGTH = 10


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
async def test_state_dict(strategy_params, use_rdma):
    os.environ["TORCHSTORE_RDMA_ENABLED"] = "1" if use_rdma else "0"

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
async def test_dcp_sharding_parity(strategy_params, use_rdma):
    os.environ["TORCHSTORE_RDMA_ENABLED"] = "1" if use_rdma else "0"

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
            await save_world._proc_mesh.stop()
            await get_world._proc_mesh.stop()
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


def test_generate_tensor_blob():
    """Test generate_tensor_blob with various tensor types and reconstruction."""

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
        # List with tensors
        "tensor_list": [
            torch.randn(2, 2, dtype=torch.float32),
            torch.tensor(123, dtype=torch.int32),
        ],
    }

    # Generate tensor blob
    modified_state_dict, blob = generate_tensor_blob(original_state_dict)

    # Verify blob properties
    assert blob.dtype == torch.uint8, f"Expected uint8 blob, got {blob.dtype}"
    assert blob.dim() == 1, f"Expected 1D blob, got {blob.dim()}D"

    # Calculate expected blob size
    expected_size = 0

    def calculate_expected_size(obj):
        nonlocal expected_size
        if isinstance(obj, torch.Tensor):
            expected_size += obj.numel() * obj.element_size()
        elif isinstance(obj, dict):
            for v in obj.values():
                calculate_expected_size(v)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                calculate_expected_size(item)

    calculate_expected_size(original_state_dict)
    assert (
        len(blob) == expected_size
    ), f"Expected blob size {expected_size}, got {len(blob)}"

    # Verify that tensors are replaced with TensorReference objects
    def verify_tensor_references(obj, path=""):
        if isinstance(obj, TensorReference):
            assert obj.shape is not None, f"TensorReference at {path} missing shape"
            assert obj.dtype is not None, f"TensorReference at {path} missing dtype"
            assert (
                obj.offset >= 0
            ), f"TensorReference at {path} has invalid offset {obj.offset}"
            assert (
                obj.size > 0
            ), f"TensorReference at {path} has invalid size {obj.size}"
        elif isinstance(obj, dict):
            for k, v in obj.items():
                verify_tensor_references(v, f"{path}.{k}" if path else k)
        elif isinstance(obj, (list, tuple)):
            for i, item in enumerate(obj):
                verify_tensor_references(item, f"{path}[{i}]")
        elif isinstance(obj, torch.Tensor):
            raise AssertionError(f"Found unreplaced tensor at {path}")

    verify_tensor_references(modified_state_dict)

    # Verify that non-tensor data is preserved
    assert modified_state_dict["metadata"]["epoch"] == 10
    assert modified_state_dict["metadata"]["learning_rate"] == 0.001

    # Reconstruct the state dict
    reconstructed_state_dict = reconstruct_state_dict_from_tensor_blob(
        modified_state_dict, blob
    )

    # Verify reconstruction matches original
    def compare_state_dicts(original, reconstructed, path=""):
        if isinstance(original, torch.Tensor):
            assert isinstance(reconstructed, torch.Tensor), f"Expected tensor at {path}"
            assert (
                original.shape == reconstructed.shape
            ), f"Shape mismatch at {path}: {original.shape} vs {reconstructed.shape}"
            assert (
                original.dtype == reconstructed.dtype
            ), f"Dtype mismatch at {path}: {original.dtype} vs {reconstructed.dtype}"
            assert torch.equal(original, reconstructed), f"Values mismatch at {path}"
        elif isinstance(original, dict):
            assert isinstance(reconstructed, dict), f"Expected dict at {path}"
            assert set(original.keys()) == set(
                reconstructed.keys()
            ), f"Key mismatch at {path}"
            for k in original.keys():
                compare_state_dicts(
                    original[k], reconstructed[k], f"{path}.{k}" if path else k
                )
        elif isinstance(original, (list, tuple)):
            assert type(original) == type(reconstructed), f"Type mismatch at {path}"
            assert len(original) == len(reconstructed), f"Length mismatch at {path}"
            for i, (orig_item, recon_item) in enumerate(zip(original, reconstructed)):
                compare_state_dicts(orig_item, recon_item, f"{path}[{i}]")
        else:
            assert (
                original == reconstructed
            ), f"Value mismatch at {path}: {original} vs {reconstructed}"

    compare_state_dicts(original_state_dict, reconstructed_state_dict)

    print("✅ test_generate_tensor_blob passed!")
    print(
        f"   Processed {len([x for x in str(modified_state_dict) if 'TensorReference' in str(x)])} tensors"
    )
    print(f"   Blob size: {len(blob)} bytes ({len(blob) / 1024:.1f} KB)")


def test_generate_tensor_blob_edge_cases():
    """Test edge cases for generate_tensor_blob."""

    # Test empty state dict
    empty_dict = {}
    modified, blob = generate_tensor_blob(empty_dict)
    assert modified == {}
    assert len(blob) == 0
    reconstructed = reconstruct_state_dict_from_tensor_blob(modified, blob)
    assert reconstructed == {}

    # Test state dict with no tensors
    no_tensors = {"a": 1, "b": {"c": "hello", "d": [1, 2, 3]}}
    modified, blob = generate_tensor_blob(no_tensors)
    assert modified == no_tensors
    assert len(blob) == 0
    reconstructed = reconstruct_state_dict_from_tensor_blob(modified, blob)
    assert reconstructed == no_tensors

    # Test scalar tensor edge case
    scalar_dict = {"scalar": torch.tensor(3.14159)}
    modified, blob = generate_tensor_blob(scalar_dict)
    assert isinstance(modified["scalar"], TensorReference)
    assert modified["scalar"].shape == ()  # Empty tuple for scalar
    reconstructed = reconstruct_state_dict_from_tensor_blob(modified, blob)
    assert torch.equal(scalar_dict["scalar"], reconstructed["scalar"])

    # Test different dtypes
    dtype_dict = {
        "bool": torch.tensor([True, False, True]),
        "uint8": torch.randint(0, 255, (5,), dtype=torch.uint8),
        "int8": torch.randint(-128, 127, (3,), dtype=torch.int8),
        "int16": torch.randint(-1000, 1000, (4,), dtype=torch.int16),
        "bfloat16": torch.randn(3, dtype=torch.bfloat16),
    }

    modified, blob = generate_tensor_blob(dtype_dict)
    reconstructed = reconstruct_state_dict_from_tensor_blob(modified, blob)

    for key in dtype_dict:
        assert torch.equal(
            dtype_dict[key], reconstructed[key]
        ), f"Mismatch for dtype {key}"

    print("✅ test_generate_tensor_blob_edge_cases passed!")


if __name__ == "__main__":
    # Run our new tests
    test_generate_tensor_blob()
    test_generate_tensor_blob_edge_cases()

    # Run existing tests
    main(__file__)
