import os
from logging import getLogger

from torchstore.logging import init_logging
from torchstore.api import (
    initialize,
    put,
    get,
    client,
    teardown_store
)
from torchstore.strategy import LocalRankStrategy, SingletonStrategy

if os.environ.get("HYPERACTOR_CODEC_MAX_FRAME_LENGTH", None) is None:
    init_logging()
    logger = getLogger(__name__)
    logger.warning(
        "Warning: setting HYPERACTOR_CODEC_MAX_FRAME_LENGTH since this needs to be set"
        " to enable large RPC calls via Monarch"
    )
    os.environ["HYPERACTOR_CODEC_MAX_FRAME_LENGTH"] = "910737418240"



__all__ = [
    "initialize",
    "init_logging",
    "put",
    "get",
    "client",
    "teardown_store",
    "LocalRankStrategy",
    "SingletonStrategy"
]
