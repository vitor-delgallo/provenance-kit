"""NumPy port of Meta's PDQ reference hasher (internal).

Ported from the pure-Python reference implementation in Meta's ThreatExchange
repository, ``pdq/python/pdqhashing/hasher/pdq_hasher.py`` and
``pdq/python/pdqhashing/utils/matrix.py``
(https://github.com/facebook/ThreatExchange/tree/main/pdq/python, commit
``a9a36d4552875c385c4b40f2be847db2b613ed44``).

The port keeps the reference's arithmetic in ``float64`` and in the same
order (luma weights, the four-phase running-sum box filter, decimation, the
16 x 64 DCT accumulated term by term, Torben's median), vectorized across the
other image axis. For the same RGB input it therefore reproduces the
reference hash bit for bit; see ``tests/image/test_hashing_pdq.py``. Meta's
C++ code (wrapped by the ``pdqhash`` package) computes in ``float32``, so it
can differ by a few bits on coefficients that sit next to the median (and by
many more on images with a side under 64 pixels; see
:mod:`provenance_kit.image.hashing`).

The original code carries this notice, which must be kept with the port:

    BSD License

    For ThreatExchange software

    Copyright (c) Meta Platforms, Inc. and affiliates.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

     * Redistributions of source code must retain the above copyright notice,
       this list of conditions and the following disclaimer.

     * Redistributions in binary form must reproduce the above copyright
       notice, this list of conditions and the following disclaimer in the
       documentation and/or other materials provided with the distribution.

     * Neither the name Facebook nor the names of its contributors may be used
       to endorse or promote products derived from this software without
       specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
    THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
    PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
    CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
    EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
    PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Modifications for provenance-kit (MIT): vectorized with NumPy; only the
hash and quality computation is kept (no file reading, no dihedral variants);
images with a side under ``MIN_HASHABLE_DIM`` (5) pixels get the all-zero
hash with quality 0, as in Meta's C++ hasher.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

__all__ = ["pdq_hash_rgb"]

_LUMA_FROM_R = 0.299
_LUMA_FROM_G = 0.587
_LUMA_FROM_B = 0.114
_DCT_MATRIX_SCALE_FACTOR = math.sqrt(2.0 / 64.0)
_NUM_JAROSZ_XY_PASSES = 2
_JAROSZ_WINDOW_SIZE_DIVISOR = 128
# Meta's C++ hasher (and so pdqhash) returns the all-zero hash with quality 0
# when a side is shorter than this (MIN_HASHABLE_DIM); the port does the same,
# so that the two backends agree on such images.
_MIN_HASHABLE_DIM = 5
_ZERO_HASH = bytes(32)


def _compute_dct_matrix() -> NDArray[np.float64]:
    # math.cos, not numpy.cos, so the matrix matches the reference exactly.
    matrix = np.empty((16, 64), dtype=np.float64)
    for i in range(16):
        for j in range(64):
            matrix[i, j] = _DCT_MATRIX_SCALE_FACTOR * math.cos(
                (math.pi / 2 / 64.0) * (i + 1) * (2 * j + 1)
            )
    matrix.flags.writeable = False
    return matrix


_DCT_MATRIX = _compute_dct_matrix()


def pdq_hash_rgb(rgb: NDArray[np.uint8]) -> tuple[bytes, int]:
    """Computes the PDQ hash of an RGB image.

    Args:
        rgb: ``uint8`` array of shape ``(height, width, 3)``, height and width
            at least 1.

    Returns:
        The 32-byte hash (canonical PDQ order: bit ``k`` of the reference
        ``Hash256`` is bit ``k % 8`` of byte ``31 - k // 8``) and the quality
        score, 0 to 100. Images with a side under 5 pixels get the all-zero
        hash and quality 0, as in Meta's C++ hasher.
    """
    rows, cols = rgb.shape[0], rgb.shape[1]
    if rows < _MIN_HASHABLE_DIM or cols < _MIN_HASHABLE_DIM:
        return _ZERO_HASH, 0
    luma = (
        _LUMA_FROM_R * rgb[:, :, 0].astype(np.float64)
        + _LUMA_FROM_G * rgb[:, :, 1].astype(np.float64)
        + _LUMA_FROM_B * rgb[:, :, 2].astype(np.float64)
    )
    window_along_rows = _jarosz_window_size(cols)
    window_along_cols = _jarosz_window_size(rows)
    for _ in range(_NUM_JAROSZ_XY_PASSES):
        # Box filter along each row (window from the width), then along each
        # column (window from the height), as boxAlongRowsFloat /
        # boxAlongColsFloat do.
        luma = _box_filter_axis0(np.ascontiguousarray(luma.T), window_along_rows).T
        luma = _box_filter_axis0(np.ascontiguousarray(luma), window_along_cols)
    block = _decimate(luma)
    quality = _quality(block)
    dct = _dct_64_to_16(block)
    median = _torben(dct.ravel())
    bits = (dct > median).ravel()  # index k = i * 16 + j, as setBit(i * 16 + j)
    return np.packbits(bits[::-1]).tobytes(), quality


def _jarosz_window_size(dimension: int) -> int:
    return (dimension + _JAROSZ_WINDOW_SIZE_DIVISOR - 1) // _JAROSZ_WINDOW_SIZE_DIVISOR


def _box_filter_axis0(values: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    """box1DFloat applied to every column at once (the reference's four phases)."""
    length = values.shape[0]
    half = (window + 2) // 2
    phase_1 = half - 1
    phase_2 = window - half + 1
    phase_3 = length - window
    phase_4 = half - 1
    out = np.empty_like(values)
    total = np.zeros(values.shape[1:], dtype=np.float64)
    count = 0
    left = right = index = 0
    for _ in range(phase_1):
        total += values[right]
        count += 1
        right += 1
    for _ in range(phase_2):
        total += values[right]
        count += 1
        out[index] = total / count
        right += 1
        index += 1
    for _ in range(phase_3):
        total += values[right]
        total -= values[left]
        out[index] = total / count
        left += 1
        right += 1
        index += 1
    for _ in range(phase_4):
        total -= values[left]
        count -= 1
        out[index] = total / count
        left += 1
        index += 1
    return out


def _decimate(values: NDArray[np.float64]) -> NDArray[np.float64]:
    rows, cols = values.shape
    steps = np.arange(64, dtype=np.float64) + 0.5
    row_index = ((steps * rows) / 64).astype(np.int64)
    col_index = ((steps * cols) / 64).astype(np.int64)
    return values[np.ix_(row_index, col_index)]


def _quality(block: NDArray[np.float64]) -> int:
    vertical = np.trunc(((block[:-1, :] - block[1:, :]) * 100) / 255)
    horizontal = np.trunc(((block[:, :-1] - block[:, 1:]) * 100) / 255)
    gradient_sum = int(np.abs(vertical).sum()) + int(np.abs(horizontal).sum())
    return min(gradient_sum // 90, 100)


def _dct_64_to_16(block: NDArray[np.float64]) -> NDArray[np.float64]:
    matrix = _DCT_MATRIX
    # T = D A, accumulated over k in order: T[i][j] += D[i][k] * A[k][j].
    partial = np.zeros((16, 64), dtype=np.float64)
    for k in range(64):
        partial += np.multiply.outer(matrix[:, k], block[k, :])
    # B = T D^t: B[i][j] += T[i][k] * D[j][k].
    result = np.zeros((16, 16), dtype=np.float64)
    for k in range(64):
        result += np.multiply.outer(partial[:, k], matrix[:, k])
    return result


def _torben(values: NDArray[np.float64]) -> float:
    """Torben's median (MatrixUtil.torben): the ``(n + 1) // 2``-th smallest value."""
    count = values.size
    half = (count + 1) // 2
    # np.min / np.max rather than the methods: NumPy 2.5's stubs reject
    # ndarray.min() without arguments under mypy --strict.
    low = float(np.min(values))
    high = float(np.max(values))
    # The reference loops until the bracket settles, which takes a few dozen
    # rounds for finite input; the cap only guards against a pathological
    # input that would never settle.
    for _ in range(4 * count + 64):
        guess = (low + high) / 2
        below = values < guess
        above = values > guess
        less = int(np.count_nonzero(below))
        greater = int(np.count_nonzero(above))
        equal = count - less - greater
        max_below = max(low, float(np.max(values[below]))) if less else low
        min_above = min(high, float(np.min(values[above]))) if greater else high
        if less <= half and greater <= half:
            if less >= half:
                return max_below
            if less + equal >= half:
                return guess
            return min_above
        if less > greater:
            high = max_below
        else:
            low = min_above
    return float(np.partition(values, half - 1)[half - 1])
