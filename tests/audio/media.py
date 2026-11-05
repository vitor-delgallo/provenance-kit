"""Synthetic test audio generated with ffmpeg's lavfi sources and NumPy (integration tests)."""

from __future__ import annotations

import io
import wave
from pathlib import Path
from typing import Any

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


def melody(
    seconds: float, rate: int, seed: int, *, noise_snr_db: float | None = None
) -> np.ndarray:
    """Structured mono "music" (float64): a scale melody, bass, pad chords and drums.

    Each seed picks a tempo (80 to 140 BPM), a key and a mode. The bass line
    and the kick drum make it bass-heavy, the hard case for bass shelving in
    the fingerprint measurements of ``docs/audio.md`` (which used seeds 400
    to 403, and 500 to 503 with ``noise_snr_db=20``).
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * rate)
    out = np.zeros(n)
    beat = 60.0 / rng.uniform(80, 140)
    root = rng.integers(45, 57)  # MIDI note
    major = rng.random() < 0.5
    scale = np.array([0, 2, 4, 5, 7, 9, 11] if major else [0, 2, 3, 5, 7, 8, 10])

    def hz(midi: float) -> float:
        return 440.0 * 2 ** ((midi - 69) / 12)

    def note(start: float, duration: float, freq: float, level: float, **shape: Any) -> None:
        harmonics = shape.get("harmonics", (1.0, 0.5, 0.25))
        attack = shape.get("attack", 0.01)
        first, last = int(start * rate), min(n, int((start + duration + 0.08) * rate))
        if first >= n:
            return
        t = np.arange(last - first) / rate
        envelope = np.minimum(1.0, t / attack) * np.exp(-t / max(duration, 0.05) * 1.5)
        tone = sum(h * np.sin(2 * np.pi * freq * (k + 1) * t) for k, h in enumerate(harmonics))
        out[first:last] += level * envelope * tone

    degree, step, position = 7, beat / 2, 0.0
    while position < seconds:
        chord = scale[rng.integers(0, 7)]
        for k in (0, 2, 4):  # pad chord for the bar
            index = (np.searchsorted(scale, chord) + k) % 7
            note(
                position,
                4 * beat,
                hz(root + 12 + scale[index]),
                0.04,
                harmonics=(1.0, 0.3),
                attack=0.2,
            )
        note(position, 2 * beat, hz(root - 12 + chord), 0.12, harmonics=(1.0, 0.4))  # bass
        note(position + 2 * beat, 2 * beat, hz(root - 12 + chord), 0.10, harmonics=(1.0, 0.4))
        for s in range(8):  # melody in eighth notes, with rests
            if rng.random() < 0.8:
                degree = int(np.clip(degree + rng.integers(-2, 3), 0, 20))
                octave, d = divmod(degree, 7)
                length = step * rng.choice([1, 1, 2])
                note(position + s * step, length, hz(root + 12 * (1 + octave) + scale[d]), 0.10)
        for b in range(4):  # kick on 1 and 3, noise snare on 2 and 4, hi-hats on eighths
            start = int((position + b * beat) * rate)
            if start >= n:
                break
            length = min(n - start, int(0.25 * rate))
            t = np.arange(length) / rate
            if b % 2 == 0:
                sweep = np.sin(2 * np.pi * (50 + 80 * np.exp(-t * 30)) * t)
                out[start : start + length] += 0.3 * np.exp(-t * 18) * sweep
            else:
                out[start : start + length] += 0.12 * np.exp(-t * 25) * rng.standard_normal(length)
            for h in (0, 1):
                hat = int((position + b * beat + h * step) * rate)
                hat_length = min(n - hat, int(0.05 * rate)) if hat < n else 0
                if hat_length > 0:
                    hiss = np.diff(rng.standard_normal(hat_length), prepend=0.0)
                    decay = np.exp(-np.arange(hat_length) / rate * 80)
                    out[hat : hat + hat_length] += 0.03 * decay * hiss
        position += 4 * beat
    out *= 0.7 / np.max(np.abs(out))
    if noise_snr_db is not None:
        noise = rng.standard_normal(n)
        noise *= np.sqrt(np.mean(out**2) / 10 ** (noise_snr_db / 10)) / np.std(noise)
        out = out + noise
        out *= 0.8 / np.max(np.abs(out))
    return out


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
