"""Logging module.

Usage:
    import qcog_python_client.log as qcoglogger

    # __name__ refers to the module name
    child_logger = qcoglogger.getChild(__name__)
    child_logger.info("This is a log message")
"""

import logging
import os
import sys

levels_map = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

QCOG_LOG_LEVL = os.getenv("QCOG_LOG_LEVL", "INFO")

# If no log level is recognized, default to INFO
if QCOG_LOG_LEVL not in levels_map:
    QCOG_LOG_LEVL = "INFO"

# Convert the string representing the log level
# to the corresponding logging level
level = levels_map[QCOG_LOG_LEVL]

# Set the logging level and the handlers
logging.basicConfig(
    level=level,
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)

qcoglogger = logging.getLogger("qcog")
