"""Type aliases shared across the package (internal)."""

from __future__ import annotations

import os
from typing import TypeAlias

__all__ = ["BytesLike", "StrPath"]

#: A filesystem path given as text or as a path-like object.
StrPath: TypeAlias = str | os.PathLike[str]

#: Raw bytes accepted without copying.
BytesLike: TypeAlias = bytes | bytearray | memoryview
