import os
import sys
import logging
def init_logging():
    log_level = os.environ.get("TORCHSTORE_LOG_LEVEL", "INFO")
    logging.root.setLevel(log_level)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(log_level)
    logging.root.addHandler(stdout_handler)
