# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""How the routing client and the Monarch adapter take their control plane."""

from __future__ import annotations

from typing import Any

import pytest
import torch
import torchstore as ts
from torchstore.routing._model import RankRole
from torchstore.routing.client import RoutingClient
from torchstore.routing.control_plane import ControlPlane
from torchstore.routing.coordinator import MonarchControlPlane


class _Reached(Exception):
    """Carries the call a control plane received."""


class _InterfaceOnly(ControlPlane):
    """A plane with exactly the interface's signatures that reports each call."""

    async def init(self, publishers: Any, strategy: Any) -> None:
        raise _Reached(("init",))

    async def register_layout(self, rank: str, key: str, registrations: Any) -> None:
        raise _Reached(("register_layout", rank, key, sorted(registrations)))

    async def get_layouts(self, key: str) -> Any:
        raise _Reached(("get_layouts", key))

    async def strategy(self) -> Any:
        raise _Reached(("strategy",))


def test_routing_client_refuses_a_coordinator_handle() -> None:
    """Refuse anything but a control plane, naming the adapter for an actor handle."""
    with pytest.raises(TypeError, match=r"MonarchControlPlane\(coordinator\)"):
        RoutingClient(
            "publisher/0", RankRole.PUBLISHER, object(), ts.TorchStoreStrategy()
        )


@pytest.mark.asyncio
async def test_routing_client_calls_only_the_interface() -> None:
    """Publish and read layouts with the interface's own arguments, no timeout included."""
    state_dict = {"w": torch.zeros(4)}
    publisher = RoutingClient(
        "publisher/0", RankRole.PUBLISHER, _InterfaceOnly(), ts.TorchStoreStrategy()
    )
    with pytest.raises(_Reached) as reached:
        await publisher.register_layout(state_dict, "model")
    assert reached.value.args == (
        ("register_layout", "publisher/0", "model", ["model/w"]),
    )
    requester = RoutingClient(
        "requester/0/0", RankRole.REQUESTER, _InterfaceOnly(), ts.TorchStoreStrategy()
    )
    with pytest.raises(_Reached) as reached:
        await requester.get_layouts(state_dict, "model")
    assert reached.value.args == (("get_layouts", "model"),)


def test_monarch_adapter_takes_its_actor_positionally() -> None:
    """Take the coordinator only by position, so its parameter name can change."""
    with pytest.raises(TypeError, match="coordinator"):
        MonarchControlPlane(coordinator=object())
