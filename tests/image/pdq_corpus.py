"""A deterministic corpus of generated RGB images for the PDQ tests.

The images are built with integer arithmetic only (a SplitMix64-style mixer,
nearest-neighbour upsampling, integer box blurs and integer gradients), so they
are bit-identical on every platform and NumPy version. That lets the tests
compare against hashes that were computed once with Meta's pure-Python PDQ
reference implementation (see ``pdq_vectors.py``).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

_MASK64 = (1 << 64) - 1
_GOLDEN = 0x9E3779B97F4A7C15
_MIX1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX2 = np.uint64(0x94D049BB133111EB)

#: (name, width, height, kind, seed). Sizes cover every Jarosz window size
#: from 1 to 5 (width or height up to 128, 129-256, ..., 513-640), images
#: smaller than PDQ's 64 x 64 block, and non-square shapes.
CORPUS_SPECS: tuple[tuple[str, int, int, str, int], ...] = (
    ("blocks-64x64", 64, 64, "blocks", 1),
    ("smooth-64x64", 64, 64, "smooth", 2),
    ("noise-48x40", 48, 40, "noise", 3),
    ("shapes-96x80", 96, 80, "shapes", 4),
    ("smooth-129x100", 129, 100, "smooth", 5),
    ("blocks-200x150", 200, 150, "blocks", 6),
    ("shapes-257x190", 257, 190, "shapes", 7),
    ("smooth-300x300", 300, 300, "smooth", 8),
    ("noise-333x257", 333, 257, "noise", 9),
    ("smooth-400x240", 400, 240, "smooth", 10),
    ("shapes-512x384", 512, 384, "shapes", 11),
    ("blocks-640x360", 640, 360, "blocks", 12),
    ("smooth-150x520", 150, 520, "smooth", 13),
    ("shapes-31x200", 31, 200, "shapes", 14),
)


def _mix(index: NDArray[np.uint64], seed: int) -> NDArray[np.uint64]:
    offset = np.uint64((seed * _GOLDEN) & _MASK64)
    z = index + offset
    z = (z ^ (z >> np.uint64(30))) * _MIX1
    z = (z ^ (z >> np.uint64(27))) * _MIX2
    return z ^ (z >> np.uint64(31))


def _random_bytes(shape: tuple[int, ...], seed: int) -> NDArray[np.uint8]:
    count = int(np.prod(shape))
    values = _mix(np.arange(count, dtype=np.uint64), seed) >> np.uint64(56)
    return values.astype(np.uint8).reshape(shape)


def _upsample(small: NDArray[np.uint8], width: int, height: int) -> NDArray[np.uint8]:
    rows = (np.arange(height) * small.shape[0]) // height
    cols = (np.arange(width) * small.shape[1]) // width
    return small[np.ix_(rows, cols)]


def _box_blur(image: NDArray[np.uint8], radius: int) -> NDArray[np.uint8]:
    work = image.astype(np.int64)
    for axis in (0, 1):
        padded = np.concatenate(
            [
                np.repeat(work.take([0], axis=axis), radius, axis=axis),
                work,
                np.repeat(work.take([-1], axis=axis), radius, axis=axis),
            ],
            axis=axis,
        )
        cumulative = np.cumsum(padded, axis=axis)
        zero_shape = list(cumulative.shape)
        zero_shape[axis] = 1
        cumulative = np.concatenate([np.zeros(zero_shape, dtype=np.int64), cumulative], axis=axis)
        size = work.shape[axis]
        upper = cumulative.take(np.arange(2 * radius + 1, 2 * radius + 1 + size), axis=axis)
        lower = cumulative.take(np.arange(size), axis=axis)
        work = (upper - lower) // (2 * radius + 1)
    return work.astype(np.uint8)


def _shapes(width: int, height: int, seed: int) -> NDArray[np.uint8]:
    y = np.arange(height, dtype=np.int64)[:, None]
    x = np.arange(width, dtype=np.int64)[None, :]
    image = np.empty((height, width, 3), dtype=np.int64)
    image[..., 0] = (x * 255) // max(width - 1, 1)
    image[..., 1] = (y * 255) // max(height - 1, 1)
    image[..., 2] = ((x + y) * 127) // max(width + height - 2, 1)
    params = _random_bytes((6, 7), seed).astype(np.int64)
    for p in params:
        x0, y0 = (p[0] * width) // 256, (p[1] * height) // 256
        x1 = min(width, x0 + 1 + (p[2] * width) // 512)
        y1 = min(height, y0 + 1 + (p[3] * height) // 512)
        image[y0:y1, x0:x1] = p[4:7]
    return image.astype(np.uint8)


def make_image(width: int, height: int, kind: str, seed: int) -> NDArray[np.uint8]:
    """Builds one corpus image as an ``(height, width, 3)`` ``uint8`` array."""
    if kind == "noise":
        return _random_bytes((height, width, 3), seed)
    if kind == "blocks":
        return _upsample(_random_bytes((9, 12, 3), seed), width, height)
    if kind == "smooth":
        coarse = _upsample(_random_bytes((14, 18, 3), seed), width, height)
        return _box_blur(coarse, max(1, min(width, height) // 24))
    if kind == "shapes":
        return _shapes(width, height, seed)
    raise ValueError(f"unknown corpus kind {kind!r}")


def corpus() -> list[tuple[str, NDArray[np.uint8]]]:
    """Returns every corpus image with its name."""
    return [(name, make_image(w, h, kind, seed)) for name, w, h, kind, seed in CORPUS_SPECS]
