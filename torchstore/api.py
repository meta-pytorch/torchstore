# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
from collections.abc import Mapping, Sequence
from functools import partial
from typing import TYPE_CHECKING, Any, overload

import torch
from monarch.actor import current_rank, get_or_spawn_controller, ProcMesh
from torch.distributed.tensor import DTensor

import torchstore.state_dict_utils
from torchstore.client import LocalClient
from torchstore.controller import Controller
from torchstore.routing._model import RankRole
from torchstore.routing.client import RoutingClient
from torchstore.routing.coordinator import RoutingCoordinator
from torchstore.routing.service import RoutingService, RoutingServiceGroup
from torchstore.storage_volume import StorageVolume
from torchstore.strategy import (
    ControllerStorageVolumes,
    MultiMeshStrategy,
    TorchStoreStrategy,
)
from torchstore.transport.types import TensorSlice

if TYPE_CHECKING:
    from torchstore.spmd import _SPMDSession

# I need to keep this somewhere, so here we go
DEFAULT_TORCHSTORE_NAME: str = "torchstore"

# cache for local clients
_local_clent_map: dict[str, LocalClient] = {}


# SPMD-initialized stores register their session here
_spmd_state_map: dict[str, "_SPMDSession"] = {}


def _routing_volume_id(namespace: str) -> str:
    """Qualify the monarch rank with its mesh's namespace."""
    return f"{namespace}/{current_rank().rank}"


def _routing_namespace(role: RankRole, group: int | None) -> str:
    """The namespace of one participating mesh."""
    if role == RankRole.PUBLISHER:
        if group is not None:
            raise ValueError("publishers are a single mesh and take no group")
        return role.value
    if group is None:
        raise ValueError("requesters must pass the index of their relay mesh")
    return f"{role.value}/{group}"


async def initialize(
    num_storage_volumes: int = 1,
    strategy: TorchStoreStrategy | None = None,
    store_name: str = DEFAULT_TORCHSTORE_NAME,
    mesh: ProcMesh | None = None,
    relay_meshes: Sequence[ProcMesh] | None = None,
) -> None:
    """Initialize the TorchStore distributed storage system.

    Sets up storage volumes and controller. Must be called before any put/get operations.

    Args:
        num_storage_volumes (int): Number of storage volumes to create. Defaults to 1.
        strategy (TorchStoreStrategy, optional): Strategy for distributing tensors across volumes.
            Uses ControllerStorageVolumes if None and num_storage_volumes=1.
        store_name (str): Unique name for this store instance. Defaults to DEFAULT_TORCHSTORE_NAME.
        mesh (ProcMesh, optional): Monarch ProcMesh on which to spawn StorageVolumes
        relay_meshes: Requester ProcMeshes. When set, the store runs in routing
            mode: ``mesh`` publishes, each relay mesh requests, and every
            participant calls ``client(state_dict=...)`` once to exchange
            layouts and receive its precomputed routes.

    Raises:
        RuntimeError: If num_storage_volumes > 1 but no strategy is provided.

    Example:
        >>> import torchstore as ts
        >>> await ts.initialize(num_storage_volumes=4, strategy=LocalRankStrategy()) # uses default namespace.
        >>> >>> await ts.initialize("my_custom_store")
    """
    if relay_meshes is not None:
        if strategy is None or mesh is None:
            raise RuntimeError("routing mode requires both mesh and strategy")
        await _initialize_routing(mesh, relay_meshes, strategy, store_name)
        return

    if num_storage_volumes == 1 and strategy is None:
        strategy = ControllerStorageVolumes()
    elif strategy is None:
        raise RuntimeError(
            "Must specify controller strategy if num_storage_volumes > 1"
        )

    # TODO: monarch doesn't support nested actors yet, so we need to spawn storage volumes here
    # ideally this is done in the controller.init
    if isinstance(strategy, ControllerStorageVolumes):
        storage_volumes = await get_or_spawn_controller(
            "storage_volume_controller",
            StorageVolume,
            id_func=strategy.get_volume_id,
        )
    else:
        storage_volumes = await StorageVolume.spawn(
            num_volumes=num_storage_volumes,
            mesh=mesh,
            id_func=strategy.get_volume_id,
        )

    controller = await _controller(store_name)
    await controller.init.call(
        strategy=strategy,
        num_storage_volumes=num_storage_volumes,
        storage_volumes=storage_volumes,
    )


async def _initialize_routing(
    mesh: ProcMesh,
    relay_meshes: Sequence[ProcMesh],
    strategy: TorchStoreStrategy,
    store_name: str,
) -> None:
    """Spawn the volumes, services and coordinator that routing mode needs."""
    if not relay_meshes:
        raise RuntimeError("routing mode requires at least one relay mesh")
    if not isinstance(strategy, MultiMeshStrategy):
        raise RuntimeError(
            "routing mode needs a MultiMeshStrategy: publisher and relay "
            f"volumes span several ProcMeshes, which {type(strategy).__name__} "
            "cannot index"
        )

    namespaces = [_routing_namespace(RankRole.PUBLISHER, None)] + [
        _routing_namespace(RankRole.REQUESTER, group)
        for group in range(len(relay_meshes))
    ]
    meshes = [mesh, *relay_meshes]
    volumes = await asyncio.gather(
        *(
            StorageVolume.spawn(
                1, m, id_func=partial(_routing_volume_id, namespace)
            )
            for m, namespace in zip(meshes, namespaces, strict=True)
        )
    )
    await strategy.set_storage_volumes(*volumes)

    services = RoutingServiceGroup()
    await services.set_services(
        *await asyncio.gather(
            *(
                RoutingService.spawn(m, id_func=partial(_routing_volume_id, namespace))
                for m, namespace in zip(meshes, namespaces, strict=True)
            )
        )
    )

    coordinator = await get_or_spawn_controller(store_name, RoutingCoordinator)
    await coordinator.init.call_one(services=services, strategy=strategy)


async def shutdown(store_name: str = DEFAULT_TORCHSTORE_NAME) -> None:
    """Shutdown and cleanup a TorchStore instance.

    Gracefully shuts down all storage volumes and controllers associated with the
    store. For SPMD-initialized stores, this delegates to the session cleanup
    path created by ``torchstore.spmd.initialize()``, which also tears down the
    Monarch host mesh and the per-host worker subprocess each local rank 0 spawned.

    Args:
        store_name (str): Name of the store to shutdown. Defaults to DEFAULT_TORCHSTORE_NAME.

    Example:
        >>> import torchstore as ts
        >>> await ts.shutdown()  # Shutdown default store
        >>> await ts.shutdown("my_custom_store")
    """
    session = _spmd_state_map.get(store_name)
    if session is not None:
        await session.shutdown()
        return

    controller = await _controller(store_name)
    try:
        await controller.teardown.call()
    finally:
        reset_client(store_name)


def reset_client(store_name: str = DEFAULT_TORCHSTORE_NAME) -> None:
    """Reset the cached or installed client for a given store."""
    _local_clent_map.pop(store_name, None)


async def _controller(store_name: str = DEFAULT_TORCHSTORE_NAME) -> Controller:
    """Get a controller handle for interacting with the store."""
    session = _spmd_state_map.get(store_name)
    if session is not None:
        return session.controller
    return await get_or_spawn_controller(store_name, Controller)


async def client(
    store_name: str = DEFAULT_TORCHSTORE_NAME,
    *,
    role: RankRole | str | None = None,
    group: int | None = None,
) -> LocalClient:
    """Get a local client handle for interacting with the store.

    Returns a cached LocalClient instance that provides the interface for put/get operations.

    Args:
        store_name (str): Name of the store to get a client for. Defaults to DEFAULT_TORCHSTORE_NAME.
        role: Set on a rank of a store initialized with ``relay_meshes``, to get
            a routing client: ``"publisher"`` for ranks on ``mesh``,
            ``"requester"`` for ranks on a relay mesh. Call
            ``register_state_dict`` on the result before using it.
        group: Index of this rank's mesh in ``initialize(relay_meshes=...)``.
            Required for requesters, rejected for publishers.

    Returns:
        LocalClient: A client instance for performing storage operations.

    Example:
        >>> store_client = await client()
        >>> await store_client.put("my_key", tensor)
    """
    if store_name in _local_clent_map:
        return _local_clent_map[store_name]

    if role is not None:
        role = RankRole(role)
        coordinator = await get_or_spawn_controller(store_name, RoutingCoordinator)
        local_client = RoutingClient(
            _routing_volume_id(_routing_namespace(role, group)),
            role,
            coordinator,
            await coordinator.strategy.call_one(),
        )
    else:
        controller = await _controller(store_name)
        controller_strategy = await controller.get_controller_strategy.call_one()
        local_client = LocalClient(
            controller=controller,
            strategy=controller_strategy,
        )
    _local_clent_map[store_name] = local_client

    return local_client


async def put(
    key: str, value: torch.Tensor | Any, store_name: str = DEFAULT_TORCHSTORE_NAME
) -> None:
    """Store a tensor or object in the distributed store.

    Args:
        key (str): Unique identifier for the stored value.
        value (torch.Tensor or Any): Tensor or object to store.
        store_name (str): Name of the store to use. Defaults to DEFAULT_TORCHSTORE_NAME.

    Example:
        >>> tensor = torch.randn(100, 100)
        >>> await put("my_tensor", tensor)
        >>> await put("my_object", {"data": [1, 2, 3]})
    """
    cl = await client(store_name)
    return await cl.put(key, value)


async def put_batch(
    entries: dict[str, torch.Tensor | Any],
    store_name: str = DEFAULT_TORCHSTORE_NAME,
) -> None:
    """Store multiple key-value pairs in a single batched operation.

    Args:
        entries: Dict mapping keys to values to store.
        store_name (str): Name of the store to use. Defaults to DEFAULT_TORCHSTORE_NAME.

    Example:
        >>> t1 = torch.randn(100, 100)
        >>> t2 = torch.randn(50, 50)
        >>> await put_batch({"key1": t1, "key2": t2})
    """
    cl = await client(store_name)
    return await cl.put_batch(entries)


async def get(
    key: str,
    inplace_tensor: torch.Tensor | None = None,
    tensor_slice_spec: TensorSlice | None = None,
    store_name: str = DEFAULT_TORCHSTORE_NAME,
) -> torch.Tensor | Any:
    """Retrieve a tensor or object from the distributed store.

    Args:
        key (str): Unique identifier of the value to retrieve.
        inplace_tensor (torch.Tensor, optional): Pre-allocated tensor for in-place retrieval.
        tensor_slice_spec (TensorSlice, optional): Specification for which slice of the tensor to retrieve.
        store_name (str): Name of the store to use. Defaults to DEFAULT_TORCHSTORE_NAME.

    Returns:
        The stored tensor, tensor slice, or object.

    Raises:
        KeyError: If the key does not exist.

    Example:
        >>> # Get full tensor
        >>> tensor = await get("my_tensor")

        >>> # Get specific slice
        >>> from torchstore.transport.pipe import TensorSlice
        >>> slice_spec = TensorSlice(
        ...     offsets=(10, 20),
        ...     coordinates=(),
        ...     global_shape=(1000, 1000),
        ...     local_shape=(50, 100),
        ...     mesh_shape=()
        ... )
        >>> tensor_slice = await get("my_tensor", tensor_slice_spec=slice_spec)

        >>> # In-place retrieval
        >>> pre_allocated_tensor = torch.empty(100, 100)
        >>> await get("my_tensor", inplace_tensor=pre_allocated_tensor)

        >>> # In-place slice retrieval (copies slice into pre-allocated tensor)
        >>> slice_buffer = torch.empty(50, 100)
        >>> await get("my_tensor", inplace_tensor=slice_buffer, tensor_slice_spec=slice_spec)
    """
    cl = await client(store_name)
    return await cl.get(key, inplace_tensor, tensor_slice_spec)


@overload
async def get_batch(
    keys: list[str],
    store_name: str = DEFAULT_TORCHSTORE_NAME,
) -> dict[str, Any]:
    ...


@overload
async def get_batch(
    keys: dict[str, torch.Tensor | DTensor | None],
    store_name: str = DEFAULT_TORCHSTORE_NAME,
) -> dict[str, Any]:
    ...


async def get_batch(
    keys: list[str] | dict[str, torch.Tensor | DTensor | None],
    store_name: str = DEFAULT_TORCHSTORE_NAME,
) -> dict[str, Any]:
    """Retrieve multiple keys from the distributed store in a single batched operation.

    All-or-nothing: if any key is missing, the entire batch raises
    and no partial results are returned.

    Args:
        keys: Either a list of keys to retrieve, or a dict mapping keys to
            optional pre-allocated tensors for in-place retrieval.
        store_name (str): Name of the store to use. Defaults to DEFAULT_TORCHSTORE_NAME.

    Returns:
        dict mapping each key to its fetched data.

    Raises:
        KeyError: If any key does not exist.
    """
    cl = await client(store_name)
    return await cl.get_batch(keys)


async def delete(
    key: str,
    *,
    store_name: str = DEFAULT_TORCHSTORE_NAME,
) -> None:
    """Delete a key from the distributed store.

    Args:
        key (str): Unique identifier of the value to delete.

    Keyword Args:
        store_name (str): Name of the store to use. Defaults to DEFAULT_TORCHSTORE_NAME.

    Example:
        >>> await delete("my_tensor")
    """
    cl = await client(store_name=store_name)
    return await cl.delete(key)


async def delete_batch(
    keys: list[str],
    *,
    store_name: str = DEFAULT_TORCHSTORE_NAME,
) -> None:
    """Delete multiple keys from the distributed store.

    Missing keys are ignored to make cleanup retries idempotent.

    Args:
        keys: Unique identifiers of the values to delete.

    Keyword Args:
        store_name (str): Name of the store to use. Defaults to DEFAULT_TORCHSTORE_NAME.

    Example:
        >>> await delete_batch(["tensor_a", "tensor_b"])
    """
    cl = await client(store_name=store_name)
    return await cl.delete_batch(keys)


async def keys(
    prefix: str | None = None,
    *,
    store_name: str = DEFAULT_TORCHSTORE_NAME,
) -> list[str]:
    """
    Get all keys that match the given prefix.

    This method retrieves all keys from the storage that start with the specified prefix.

    Args:
        prefix (str): The prefix to match against stored keys.


    Returns:
        List[str]: A list of keys that match the given prefix.

    Example:
        >>> keys = await keys("my_prefix")
    """
    cl = await client(store_name=store_name)
    return await cl.keys(prefix)


async def exists(key: str, store_name: str = DEFAULT_TORCHSTORE_NAME) -> bool:
    """Check if a key exists in the distributed store.

    Args:
        key (str): Unique identifier to check for existence.
        store_name (str): Name of the store to use. Defaults to DEFAULT_TORCHSTORE_NAME.

    Returns:
        bool: True if the key exists, False otherwise.

    Example:
        >>> if await exists("my_tensor"):
        ...     tensor = await get("my_tensor")
        >>>
        >>> # Check before storing
        >>> if not await exists("checkpoint_1"):
        ...     await put("checkpoint_1", model.state_dict())
    """
    cl = await client(store_name)
    return await cl.exists(key)


async def put_state_dict(
    state_dict: dict[str, Any] | None,
    key: str,
    store_name: str = DEFAULT_TORCHSTORE_NAME,
    direct_rdma: bool = False,
    transfer_dtype: "torch.dtype | None" = None,
) -> None:
    """Store a PyTorch model state_dict in the distributed store.

    Args:
        state_dict (dict or None): Model state_dict to store.  Required on the
            first call.  When ``direct_rdma=True``, may be ``None`` on
            subsequent calls to skip the (potentially expensive)
            ``model.state_dict()`` construction — only staging-buffer refresh
            is performed.
        key (str): Unique identifier for the state_dict.
        store_name (str): Name of the store to use. Defaults to DEFAULT_TORCHSTORE_NAME.
        direct_rdma (bool): If True, register RDMA handles pointing at the
            caller's GPU memory instead of copying data to a StorageVolume.
            First call registers handles; subsequent calls refresh staging
            buffers for non-contiguous params.
        transfer_dtype (torch.dtype, optional): If set, cast floating-point
            weights to this dtype for transfer. Allows the source to keep
            higher-precision master weights (e.g. float32) while transferring
            in a lower precision (e.g. bfloat16).

    Example:
        >>> model = torch.nn.Linear(10, 5)
        >>> await put_state_dict(model.state_dict(), "model_checkpoint")
    """
    cl = await client(store_name)
    await torchstore.state_dict_utils.put_state_dict(
        store=cl,
        state_dict=state_dict,
        key=key,
        direct_rdma=direct_rdma,
        transfer_dtype=transfer_dtype,
    )


async def get_state_dict(
    key: str,
    user_state_dict: dict[str, Any] | None = None,
    strict: bool = True,
    store_name: str = DEFAULT_TORCHSTORE_NAME,
    direct_rdma: bool = False,
) -> dict[str, Any]:
    """Retrieve a PyTorch model state_dict from the distributed store.

    Args:
        key (str): Unique identifier of the state_dict to retrieve.
        user_state_dict (dict, optional): Pre-existing state_dict to merge with.
            Required when ``direct_rdma=True`` (destination tensors for in-place writes).
        strict (bool): Whether to enforce strict loading. Defaults to True.
        store_name (str): Name of the store to use. Defaults to DEFAULT_TORCHSTORE_NAME.
        direct_rdma (bool): If True, pull weights directly from the source's
            GPU memory via one-sided RDMA reads. Handles are fetched from
            TorchStore on the first call and cached for subsequent calls.

    Returns:
        dict: The retrieved state_dict.

    Example:
        >>> state_dict = await get_state_dict("model_checkpoint")
        >>> model.load_state_dict(state_dict)
    """
    cl = await client(store_name)
    return await torchstore.state_dict_utils.get_state_dict(
        cl, key, user_state_dict, strict, direct_rdma=direct_rdma
    )
