import os
from logging import getLogger
from torchstore.controller import Controller
from torchstore.storage_volume import StorageVolume
from torchstore.utils import spawn_actors
from torchstore.logging import init_logging

if os.environ.get("HYPERACTOR_CODEC_MAX_FRAME_LENGTH", None) is None:
    init_logging()
    logger = getLogger(__name__)
    logger.warning(
        "Warning: setting HYPERACTOR_CODEC_MAX_FRAME_LENGTH since this needs to be set"
        " to enable large RPC calls via Monarch"
    )
    os.environ["HYPERACTOR_CODEC_MAX_FRAME_LENGTH"] = "910737418240"

async def create_store(num_hosts=1) -> "LocalClient":
    """Initializes the global store, and returns a local client.
    """
    controller = await spawn_actors(1, Controller, "Controller")
    storage_volumes = await spawn_actors(num_hosts, StorageVolume, "StorageVolume")

    await controller.set_storage_volumes(storage_volumes)
    await storage_volumes.set_controller(controller)

    return LocalClient(storage_volumes, controller)

__all__ = ["create_store", "init_logging"]
