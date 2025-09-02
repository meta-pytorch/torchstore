import os
from logging import getLogger

from torchstore.logging import init_logging

if os.environ.get("HYPERACTOR_CODEC_MAX_FRAME_LENGTH", None) is None:
    init_logging()
    logger = getLogger(__name__)
    logger.warning(
        "Warning: setting HYPERACTOR_CODEC_MAX_FRAME_LENGTH since this needs to be set"
        " to enable large RPC calls via Monarch"
    )
    os.environ["HYPERACTOR_CODEC_MAX_FRAME_LENGTH"] = "910737418240"

from torchstore.store import get, put


__all__ = ["put", "get"]
