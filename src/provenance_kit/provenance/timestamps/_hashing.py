"""SHA-256 of files and streams, in chunks (internal module; import from the package)."""

from __future__ import annotations

import hashlib
import os
from typing import BinaryIO

from provenance_kit._typing import StrPath

__all__ = ["DEFAULT_CHUNK_BYTES", "sha256_file", "sha256_stream"]

#: The read size: 1 MiB, as python-opentimestamps' ``CryptOp.hash_fd`` uses.
DEFAULT_CHUNK_BYTES = 1024 * 1024


def sha256_stream(stream: BinaryIO, *, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> bytes:
    """Hashes a binary stream from its current position to the end.

    Memory use is one chunk, so any size can be hashed; the time is linear
    in the size. The stream is not closed.

    Args:
        stream: A readable binary stream (``read(n)`` returning bytes).
        chunk_bytes: The read size.

    Returns:
        The 32-byte SHA-256 digest, ready for :meth:`OpenTimestamper.stamp`.

    Raises:
        TypeError: If the stream returns something other than bytes.
        ValueError: If ``chunk_bytes`` is not positive.
        OSError: If reading fails.
    """
    size: object = chunk_bytes
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("chunk_bytes must be a positive int")
    hasher = hashlib.sha256()
    while True:
        chunk = stream.read(chunk_bytes)
        if not chunk:
            return hasher.digest()
        if not isinstance(chunk, (bytes, bytearray, memoryview)):
            raise TypeError("the stream must be opened in binary mode")
        hasher.update(chunk)


def sha256_file(path: StrPath, *, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> bytes:
    """Hashes a file in chunks.

    Args:
        path: The file to hash.
        chunk_bytes: The read size.

    Returns:
        The 32-byte SHA-256 digest.

    Raises:
        ValueError: If ``chunk_bytes`` is not positive.
        OSError: If the file cannot be read (the error, raised by the
            operating system, names the path).
    """
    with open(os.fspath(path), "rb") as stream:
        return sha256_stream(stream, chunk_bytes=chunk_bytes)
