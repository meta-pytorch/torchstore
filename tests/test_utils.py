import unittest
from logging import getLogger

import torch

from torchstore.utils import assemble_global_tensor, get_local_tensor


logger = getLogger(__name__)


class TestGetLocalTensor(unittest.TestCase):
    def _test_get_local_tensor(self, global_tensor, test_cases):
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

    def test_1d_get_local_tensor(self):
        global_tensor = torch.tensor([0, 1, 2, 3, 4])
        test_cases = [
            # Each test case is a tuple of (expected_result, local_shape, global_offset)
            ([0], (1,), (0,)),
            ([1], (1,), (1,)),
            ([2], (1,), (2,)),
            ([3], (1,), (3,)),
        ]
        self._test_get_local_tensor(global_tensor, test_cases)

    def test_2d_get_local_tensor(self):
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
        self._test_get_local_tensor(global_tensor, test_cases)


class TestAssembleGlobalTensor(unittest.TestCase):
    def _test_assemble_global_tensor(
        self, local_tensors, global_shape, global_offsets, expected_output
    ):
        assembled_tensor = assemble_global_tensor(
            local_tensors, global_shape, global_offsets
        )
        assert torch.equal(
            assembled_tensor,
            expected_output,
        ), f"{assembled_tensor=} != {expected_output=}"

    def test_1d_assemble_global_tensor(self):
        self._test_assemble_global_tensor(
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

    def test_2d_assemble_global_tensor(self):
        self._test_assemble_global_tensor(
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


if __name__ == "__main__":
    unittest.main()
