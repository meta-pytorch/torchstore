# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The key metadata that TorchStore's controller, client and directories share."""

import dataclasses
import typing

import pytest
from torchstore.directory import ObjectType, StorageInfo, VolumeDirectory
from torchstore.transport.types import Request, TensorSlice


def _shard(index: int) -> TensorSlice:
    return TensorSlice(
        offsets=(index,),
        coordinates=(index,),
        global_shape=(4,),
        local_shape=(1,),
        mesh_shape=(4,),
    )


def test_update_adds_the_other_infos_slices() -> None:
    """Keep the recorded slices and add the other info's.

    The controller does so when another put of a key reaches a volume that holds it.
    """
    info = StorageInfo(ObjectType.TENSOR_SLICE, {_shard(0), _shard(1)})
    info.update(StorageInfo(ObjectType.TENSOR_SLICE, {_shard(1), _shard(2)}))
    assert info.tensor_slices == {_shard(0), _shard(1), _shard(2)}


def test_update_refuses_another_object_type() -> None:
    """Fail an assert before touching the recorded slices when the other info's type differs.

    The check is an assert, as on main, so python -O skips it.
    """
    info = StorageInfo(ObjectType.TENSOR_SLICE, {_shard(0)})
    with pytest.raises(AssertionError, match="change storage type"):
        info.update(StorageInfo(ObjectType.TENSOR, {None}))
    assert info.tensor_slices == {_shard(0)}


def test_shared_types_keep_their_runtime_annotations() -> None:
    """Evaluate the annotations when the classes load, and resolve the quoted ones."""
    assert StorageInfo.__annotations__ == {
        "object_type": ObjectType,
        "tensor_slices": set[TensorSlice | None],
    }
    assert [field.type for field in dataclasses.fields(StorageInfo)] == [
        ObjectType,
        set[TensorSlice | None],
    ]
    assert ObjectType.from_request.__annotations__ == {
        "request": Request,
        "return": "ObjectType",
    }
    assert typing.get_type_hints(VolumeDirectory.locate_volumes) == {
        "keys": list[str],
        "missing_ok": bool,
        "require_fully_committed": bool,
        "return": dict[str, dict[str, StorageInfo]],
    }
    assert typing.get_type_hints(StorageInfo.update) == {
        "other_storage_info": StorageInfo
    }
