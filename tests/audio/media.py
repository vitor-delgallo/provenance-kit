"""Synthetic test audio generated with ffmpeg's lavfi sources and NumPy (integration tests)."""

from __future__ import annotations

import io
import wave
from pathlib import Path

import numpy as np

from provenance_kit.tools import ExternalTools, Tool

BASE = ["-nostdin", "-hide_banner", "-loglevel", "error"]


def song(k: int, rate: int = 44100) -> str:
    """A non-periodic test signal: two wandering tones plus a little noise.

    Different ``k`` give different (but structurally similar) signals, which
    makes "unrelated audio" a hard case for the fingerprint thresholds.
    """
    return (
        f"aevalsrc=0.25*sin(2*PI*({300 + 40 * k}+200*sin(2*PI*0.{13 + k}*t)"
        f"*sin(2*PI*0.0{71 + k}*t))*t)"
        f"+0.2*sin(2*PI*({500 + 30 * k}+300*sin(2*PI*0.0{31 + k}*t))*t)"
        f"+0.05*random({k}):s={rate}"
    )


def music(seconds: float, rate: int, seed: int) -> np.ndarray:
    """A non-periodic mono "song" (float64 samples): four wobbling tones plus noise.

    Unlike :func:`song`, different seeds give tones in unrelated ranges with
    unrelated modulations, so windows of two seeds do not resemble each
    other (the fingerprint false-positive tests use it as unrelated music).
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * rate)
    t = np.arange(n) / rate
    out = np.zeros(n)
    for _ in range(4):
        base = rng.uniform(150, 1500)
        wobble = rng.uniform(50, 400) * np.sin(
            2 * np.pi * rng.uniform(0.05, 0.4) * t + rng.uniform(0, 6)
        )
        phase = 2 * np.pi * np.cumsum(base + wobble) / rate
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(0.2, 3) * t + rng.uniform(0, 6))
        out += 0.15 * envelope * np.sin(phase)
    return out + 0.03 * rng.standard_normal(n)


def wav(samples: np.ndarray, rate: int) -> bytes:
    """16-bit WAV bytes of float samples shaped ``(n,)`` or ``(channels, n)``."""
    matrix = np.atleast_2d(np.asarray(samples, dtype=np.float64))
    pcm = np.clip(np.round(matrix.T * 32767), -32768, 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(matrix.shape[0])
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(pcm.tobytes())
    return buffer.getvalue()


#: Broadband, speech-like test signal for AudioSeal: pink noise with a slow
#: amplitude envelope plus a wandering tone.
BROADBAND = (
    "aevalsrc=0.3*sin(2*PI*(300+200*sin(2*PI*0.13*t)*sin(2*PI*0.071*t))*t)+0.05*random(0):s={rate}"
)
#: Sparse tonal content (AudioSeal's message bits are weaker on it).
TONES = (
    "aevalsrc=0.3*sin(2*PI*440*t)*(0.5+0.5*sin(2*PI*3*t))"
    "+0.2*sin(2*PI*1250*t)*(t-floor(t)):s={rate}"
)


def ffmpeg(tools: ExternalTools, args: list[str], *, stdin: bytes | None = None) -> bytes:
    """Runs ffmpeg with the common arguments and returns its standard output."""
    result = tools.run(
        Tool.FFMPEG, [*BASE, *args], stdin=stdin, timeout_seconds=300, max_stdout_bytes=1 << 30
    )
    return result.stdout


def to_file(tools: ExternalTools, args: list[str], path: Path) -> bytes:
    """Runs ffmpeg writing ``path`` (for containers that need a seekable output)."""
    ffmpeg(tools, [*args, "-y", str(path)])
    return path.read_bytes()


def lavfi(expression: str, seconds: float) -> list[str]:
    """Input arguments for a lavfi expression."""
    return ["-f", "lavfi", "-i", expression, "-t", str(seconds)]


def encode_pcm(
    tools: ExternalTools, samples: np.ndarray, rate: int, codec_args: list[str]
) -> bytes:
    """Encodes float32 (channels, n) PCM read from a pipe (the tests' encoder callable).

    ``codec_args`` ends with the output: ``pipe:1`` is appended unless it
    already names a file (``-y <path>``).
    """
    channels = samples.shape[0]
    data = np.ascontiguousarray(samples.T, dtype="<f4").tobytes()
    return ffmpeg(
        tools,
        [
            "-f",
            "f32le",
            "-ar",
            str(rate),
            "-ac",
            str(channels),
            "-i",
            "pipe:0",
            *codec_args,
            *([] if "-y" in codec_args else ["pipe:1"]),
        ],
        stdin=data,
    )
