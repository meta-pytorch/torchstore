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
    rdma_options = [False]  # , True] broken on my build

    return "strategy_params, use_rdma", list(product(strategies, rdma_options))
