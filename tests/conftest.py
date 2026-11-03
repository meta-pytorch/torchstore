# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fixtures that more than one test module uses."""

from collections.abc import Iterator

import pytest

from .test_carrier import _Children


@pytest.fixture
def children() -> Iterator[_Children]:
    children = _Children()
    yield children
    children.kill_all()
