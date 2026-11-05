"""The library logger.

provenance-kit logs through ``logging.getLogger("provenance_kit")`` and its
children (``provenance_kit.image``, ``provenance_kit.tools``, ...). A
``NullHandler`` is attached to the root library logger, and the library never
configures handlers, levels or formatters: that is the application's job.

Logging rule: messages at INFO level or above never contain media bytes,
watermark payloads or identifiers, keys or other secrets, or the caller's
source paths. DEBUG messages may contain truncated tool output and temporary
paths, but still never secrets or payloads.
"""

from __future__ import annotations

import logging

__all__ = ["LOGGER_NAME", "get_logger"]

LOGGER_NAME = "provenance_kit"

logging.getLogger(LOGGER_NAME).addHandler(logging.NullHandler())


def get_logger(name: str | None = None) -> logging.Logger:
    """Returns the library logger or one of its children.

    Args:
        name: A module name. ``"provenance_kit.video.frames"`` and
            ``"video.frames"`` both give the child logger
            ``provenance_kit.video.frames``. ``None`` gives the library logger.

    Returns:
        The logger.
    """
    if not name or name == LOGGER_NAME:
        return logging.getLogger(LOGGER_NAME)
    if name.startswith(LOGGER_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{LOGGER_NAME}.{name}")
