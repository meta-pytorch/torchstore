# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from logging import getLogger

import pytest

import torch

from torchstore.utils import assemble_tensor, get_local_tensor


logger = getLogger(__name__)


def _test_get_local_tensor(global_tensor, test_cases):
    """
    Given a global_tensor, assert we can correctly extract local tensor slices
    given local_shape and global_offset.

    global_tensor (torch.Tensor): The source tensor to extract slices from
    test_cases (list): List of test case tuples, where each tuple contains:
        - expected_result (list): Expected tensor values
        - local_shape (tuple): Shape of the local tensor slice to extract
        - global_offset (tuple): Starting offset coordinates in the global tensor
    """

    for i, (expected, shape, offset) in enumerate(test_cases):
        result = get_local_tensor(global_tensor, shape, offset)
        assert torch.equal(result, torch.tensor(expected)), f"Test case {i} failed"


@pytest.mark.asyncio
async def test_1d_get_local_tensor():
    global_tensor = torch.tensor([0, 1, 2, 3, 4])
    test_cases = [
        # Each test case is a tuple of (expected_result, local_shape, global_offset)
        ([0], (1,), (0,)),
        ([1], (1,), (1,)),
        ([2], (1,), (2,)),
        ([3], (1,), (3,)),
    ]
    _test_get_local_tensor(global_tensor, test_cases)


@pytest.mark.asyncio
async def test_2d_get_local_tensor():
    # Test cases from:
    # https://github.com/pytorch/pytorch/blob/42ff6a4a5c4e0d77bd18fcc5426622f1b8f20add/torch/distributed/tensor/_utils.py#L73
    global_tensor = torch.tensor([[0, 1, 2, 3, 4], [10, 11, 12, 13, 14]])
    test_cases = [
        # Each test case is a tuple of (expected_result, local_shape, global_offset)
        ([[0, 1], [10, 11]], (2, 2), (0, 0)),
        ([[2], [12]], (2, 1), (0, 2)),
        ([[3], [13]], (2, 1), (0, 3)),
        ([[4], [14]], (2, 1), (0, 4)),
    ]
    _test_get_local_tensor(global_tensor, test_cases)


def _test_assemble_tensor(local_tensors, global_shape, global_offsets, expected_output):
    assembled_tensor = assemble_tensor(local_tensors, global_shape, global_offsets)
    assert torch.equal(
        assembled_tensor,
        expected_output,
    ), f"{assembled_tensor=} != {expected_output=}"


@pytest.mark.asyncio
async def test_1d_assemble_tensor():
    _test_assemble_tensor(
        local_tensors=[
            torch.tensor([0]),
            torch.tensor([1]),
            torch.tensor([2]),
            torch.tensor([3]),
        ],
        global_shape=(4,),
        global_offsets=[(0,), (1,), (2,), (3,)],
        expected_output=torch.tensor([0, 1, 2, 3]),
    )


@pytest.mark.asyncio
async def test_1d_assemble_tensor_slice():
    _test_assemble_tensor(
        local_tensors=[
            torch.tensor([1]),
            torch.tensor([2]),
        ],
        global_shape=(4,),
        global_offsets=[(1,), (2,)],
        expected_output=torch.tensor([1, 2]),
    )


@pytest.mark.asyncio
async def test_2d_assemble_tensor():
    _test_assemble_tensor(
        local_tensors=[
            torch.tensor([[0, 1], [10, 11]]),
            torch.tensor([[2], [12]]),
            torch.tensor([[3], [13]]),
            torch.tensor([[4], [14]]),
        ],
        global_shape=(2, 5),
        global_offsets=[(0, 0), (0, 2), (0, 3), (0, 4)],
        expected_output=torch.tensor([[0, 1, 2, 3, 4], [10, 11, 12, 13, 14]]),
    )


@pytest.mark.asyncio
async def test_2d_assemble_tensor_slice():
    _test_assemble_tensor(
        local_tensors=[
            torch.tensor([[0, 1], [10, 11]]),
            torch.tensor([[2], [12]]),
            torch.tensor([[20, 21, 22]]),
        ],
        global_shape=(100, 100),
        global_offsets=[(1, 1), (1, 3), (3, 1)],
        expected_output=torch.tensor([[0, 1, 2], [10, 11, 12], [20, 21, 22]]),
    )


def test_assemble_tensor_empty_list_assertion():
    """Test that assemble_tensor raises assertion error for empty local_tensors list."""
    with pytest.raises(AssertionError):
        assemble_tensor([], (4,), [])


def test_assemble_tensor_overlapping_tensors():
    """Test that assemble_tensor raises assertion error when tensors overlap."""
    # Create overlapping tensors: tensor regions overlap
    _test_assemble_tensor(
        local_tensors=[
            torch.tensor(
                [[1, 2], [3, 4]]
            ),  # Shape (2,2) at offset (0,0) -> covers (0,0) to (1,1)
            torch.tensor(
                [[5, 6], [7, 8]]
            ),  # Shape (2,2) at offset (1,0) -> covers (1,0) to (2,1) - overlaps at (1,0) and (1,1)
        ],
        global_offsets=[
            (0, 0),
            (1, 0),
        ],  # Second tensor starts at (1,0), causing overlap
        global_shape=(10, 10),
        expected_output=torch.tensor([[1, 2], [5, 6], [7, 8]]),
    )


def test_assemble_tensor_with_gaps():
    """Test that assemble_tensor raises assertion error when tensors have gaps."""
    # Create tensors with gaps - this should now fail because sizes don't match
    local_tensors = [
        torch.tensor([1, 2]),  # At positions (0, 1) - 2 elements
        torch.tensor([7, 8]),  # At positions (5, 6) - 2 elements
    ]
    global_offsets = [(0,), (5,)]  # Gap between positions 2 and 5
    global_shape = (10,)

    # This should now fail the assertion because:
    # - Total local tensor elements: 2 + 2 = 4
    # - Target tensor size: 7 (from offset 0 to offset 7)
    # - Since 4 != 7, assertion fails
    with pytest.raises(
        AssertionError, match="Local tensor sizes doesn't match target tensor"
    ):
        assemble_tensor(local_tensors, global_shape, global_offsets)


def test_assemble_tensor_size_mismatch():
    """Test that assemble_tensor raises assertion error when total size doesn't match target."""
    # Create a scenario where local tensor elements don't fill the target exactly
    local_tensors = [
        torch.tensor([[1, 2, 3]]),  # Shape (1,3) = 3 elements
        torch.tensor([[4], [5]]),  # Shape (2,1) = 2 elements
    ]
    global_offsets = [(0, 0), (1, 0)]  # No overlap
    global_shape = (10, 10)

    # This should raise an assertion error because:
    # - Total local tensor elements: 3 + 2 = 5
    # - Target tensor size: 2x3 = 6 (from offset (0,0) to (2,3))
    # - Since 5 != 6, assertion fails
    with pytest.raises(
        AssertionError, match="Local tensor sizes doesn't match target tensor"
    ):
        assemble_tensor(local_tensors, global_shape, global_offsets)


def test_assemble_tensor_perfect_fit():
    """Test that assemble_tensor works when local tensors perfectly fill the target."""
    # Create tensors that perfectly fill a 2x2 target without gaps or overlaps
    local_tensors = [
        torch.tensor([[1, 2]]),  # Shape (1,2) at (0,0)
        torch.tensor([[3, 4]]),  # Shape (1,2) at (1,0)
    ]
    global_offsets = [(0, 0), (1, 0)]
    global_shape = (10, 10)

    # This should work because:
    # - Total local tensor elements: 2 + 2 = 4
    # - Target tensor size: 2x2 = 4 (from offset (0,0) to (2,2))
    # - Since 4 == 4, assertion passes
    result = assemble_tensor(local_tensors, global_shape, global_offsets)
    expected = torch.tensor([[1, 2], [3, 4]])
    assert torch.equal(result, expected)
