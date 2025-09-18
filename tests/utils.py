# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from itertools import product

import pytest
import torchstore as ts


def main(file):
    ts.init_logging()
    pytest.main([file])


def transport_plus_strategy_params():
    strategies = [
        (2, ts.LocalRankStrategy()),
        (1, None),  # singleton
    ]
    rdma_options = [True, False]

    return "strategy_params, use_rdma", list(product(strategies, rdma_options))
