"""Band-limited resampling of in-memory PCM (internal).

AudioSeal works at 16 kHz, so the watermark backend converts in memory (the
backends never call ffmpeg, ARCHITECTURE.md decision 2). The method is
windowed-sinc interpolation: each output sample is a weighted sum of the
input samples around its position, with a low-pass sinc kernel (cutoff at
``rolloff`` times the lower Nyquist frequency, ``zeros`` zero crossings on
each side) tapered by a Hann window, and normalized to unit DC gain.

* When the reduced rate ratio ``up/down`` is small (all the common rates:
  8, 11.025, 16, 22.05, 24, 32, 44.1, 48, 88.2, 96, 176.4 and 192 kHz), a
  polyphase table of ``up`` kernels is precomputed and applied as a matrix
  product over strided windows of the input, in blocks of bounded memory.
* Otherwise each output sample's kernel is evaluated directly (slower, same
  result).

Edges are zero-padded. The output has ``ceil(n * new_rate / old_rate)``
samples.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = ["resample", "resampled_length"]

_ZEROS = 24
_ROLLOFF = 0.945
_MAX_TABLE = 4_000_000  # polyphase table entries (up x taps)
_BLOCK_ROWS = 8192
_DIRECT_BLOCK = 4096


def resampled_length(length: int, old_rate: int, new_rate: int) -> int:
    """Returns the output length for ``length`` input samples.

    Args:
        length: Input samples.
        old_rate: Input rate in hertz.
        new_rate: Output rate in hertz.

    Returns:
        ``ceil(length * new_rate / old_rate)``.
    """
    return -(-length * new_rate // old_rate)


def resample(
    signal: npt.NDArray[np.floating], old_rate: int, new_rate: int
) -> npt.NDArray[np.float32]:
    """Resamples a 1-D signal.

    Args:
        signal: The samples (1-D).
        old_rate: Input rate in hertz (positive).
        new_rate: Output rate in hertz (positive).

    Returns:
        The resampled ``float32`` signal.

    Raises:
        ValueError: If the signal is not 1-D or a rate is not positive.
    """
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("resample expects a 1-D signal")
    if old_rate < 1 or new_rate < 1:
        raise ValueError("sample rates must be positive")
    divisor = math.gcd(old_rate, new_rate)
    up, down = new_rate // divisor, old_rate // divisor
    if up == down or x.size == 0:
        return x.astype(np.float32)
    cutoff = 0.5 * min(1.0, up / down) * _ROLLOFF  # cycles per input sample
    half_width = _ZEROS / (2.0 * cutoff)  # kernel half-width in input samples
    width = math.ceil(half_width)
    plan = _Plan(
        up=up,
        down=down,
        cutoff=cutoff,
        half_width=half_width,
        width=width,
        n_out=resampled_length(x.size, old_rate, new_rate),
    )
    out = _polyphase(x, plan) if up * (2 * width + down) <= _MAX_TABLE else _direct(x, plan)
    return out.astype(np.float32)


@dataclass(frozen=True, slots=True)
class _Plan:
    up: int
    down: int
    cutoff: float
    half_width: float
    width: int
    n_out: int


def _kernel(
    tau: npt.NDArray[np.float64], cutoff: float, half_width: float
) -> npt.NDArray[np.float64]:
    window = np.where(np.abs(tau) < half_width, np.cos(np.pi * tau / (2.0 * half_width)) ** 2, 0.0)
    return 2.0 * cutoff * np.sinc(2.0 * cutoff * tau) * window


def _polyphase(x: npt.NDArray[np.float64], plan: _Plan) -> npt.NDArray[np.float64]:
    up, down, width, n_out = plan.up, plan.down, plan.width, plan.n_out
    cutoff, half_width = plan.cutoff, plan.half_width
    # Output sample q * up + p sits at input position q * down + p * down / up.
    # Its taps are the input samples q * down + o for o in [-width, width + down).
    taps = 2 * width + down
    offsets = np.arange(-width, width + down, dtype=np.float64)
    phases = np.arange(up, dtype=np.float64) * down / up
    table = _kernel(offsets[None, :] - phases[:, None], cutoff, half_width)
    table /= table.sum(axis=1, keepdims=True)
    blocks = -(-n_out // up)
    padded = np.zeros(width + blocks * down + taps, dtype=np.float64)
    padded[width : width + x.size] = x
    windows = np.lib.stride_tricks.sliding_window_view(padded, taps)[::down][:blocks]
    out = np.empty(blocks * up, dtype=np.float64)
    for start in range(0, blocks, _BLOCK_ROWS):
        stop = min(start + _BLOCK_ROWS, blocks)
        # table @ windows.T rather than windows @ table.T: the same products,
        # but OpenBLAS handles this shape well for short signals too (a few
        # dozen blocks), where the other one was up to 40 times slower when
        # PyTorch's threads compete for the cores (the alignment search scores
        # hundreds of 0.5 s excerpts).
        out[start * up : stop * up] = (table @ windows[start:stop].T).T.reshape(-1)
    return out[:n_out]


def _direct(x: npt.NDArray[np.float64], plan: _Plan) -> npt.NDArray[np.float64]:
    up, down, width, n_out = plan.up, plan.down, plan.width, plan.n_out
    cutoff, half_width = plan.cutoff, plan.half_width
    padded = np.concatenate((np.zeros(width + 1), x, np.zeros(width + 1)))
    out = np.empty(n_out, dtype=np.float64)
    taps = np.arange(-width, width + 1)
    for start in range(0, n_out, _DIRECT_BLOCK):
        index = np.arange(start, min(start + _DIRECT_BLOCK, n_out))
        position = index * down / up
        base = np.floor(position).astype(np.int64)
        sample_index = base[:, None] + taps[None, :]
        weights = _kernel(sample_index - position[:, None], cutoff, half_width)
        weights /= weights.sum(axis=1, keepdims=True)
        out[index] = np.sum(padded[sample_index + width + 1] * weights, axis=1)
    return out
