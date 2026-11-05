"""Synthetic test media generated with ffmpeg's lavfi sources (integration tests)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from provenance_kit.tools import ExternalTools, Tool

BASE = ["-nostdin", "-hide_banner", "-loglevel", "error"]

#: Content that changes quickly and has texture (Conway's game of life,
#: blurred): PDQ hashes of consecutive seconds differ by about 50 bits.
LIFE = (
    "life=size={w}x{h}:rate={r}:mold=10:ratio=0.3:seed={seed}"
    ":death_color=#C83232:life_color=#00ff00,gblur=sigma=2"
)
#: A textured, slowly changing source (VideoSeal embeds well into it).
MANDELBROT = "mandelbrot=size={w}x{h}:rate={r}"
#: ffmpeg's standard test pattern (flat areas: hard for VideoSeal).
TESTSRC2 = "testsrc2=size={w}x{h}:rate={r}"


def ffmpeg(tools: ExternalTools, args: list[str], *, stdin: bytes | None = None) -> bytes:
    """Runs ffmpeg with the common arguments and returns its standard output."""
    result = tools.run(
        Tool.FFMPEG, [*BASE, *args], stdin=stdin, timeout_seconds=300, max_stdout_bytes=1 << 30
    )
    return result.stdout


def to_file(tools: ExternalTools, args: list[str], path: Path) -> bytes:
    """Runs ffmpeg writing ``path`` (seekable output, e.g. MP4 with moov at the end)."""
    ffmpeg(tools, [*args, "-y", str(path)])
    return path.read_bytes()


def lavfi(source: str, *, width: int, height: int, rate: int = 25, seed: int = 7) -> list[str]:
    """Input arguments for a lavfi source template."""
    return ["-f", "lavfi", "-i", source.format(w=width, h=height, r=rate, seed=seed)]


def raw_frames(
    tools: ExternalTools, source: str, *, width: int, height: int, rate: int, count: int
) -> list[np.ndarray]:
    """Decoded RGB frames straight from a lavfi source."""
    out = ffmpeg(
        tools,
        [
            *lavfi(source, width=width, height=height, rate=rate),
            "-frames:v",
            str(count),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
    )
    array = np.frombuffer(out, dtype=np.uint8).reshape(-1, height, width, 3)
    return [np.array(frame) for frame in array]


def encode_frames(
    tools: ExternalTools,
    frames: list[np.ndarray],
    codec_args: list[str],
    *,
    rate: int,
) -> bytes:
    """Encodes RGB frames through a pipe (the encoder callable of the tests)."""
    height, width = frames[0].shape[:2]
    data = b"".join(np.ascontiguousarray(f).tobytes() for f in frames)
    return ffmpeg(
        tools,
        [
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(rate),
            "-i",
            "pipe:0",
            *codec_args,
            "pipe:1",
        ],
        stdin=data,
    )


#: Codec settings available in LGPL ffmpeg builds (no x264/x265).
H264 = [
    "-c:v",
    "libopenh264",
    "-pix_fmt",
    "yuv420p",
    "-f",
    "mp4",
    "-movflags",
    "frag_keyframe+empty_moov",
]
VP9 = [
    "-c:v",
    "libvpx-vp9",
    "-deadline",
    "realtime",
    "-cpu-used",
    "8",
    "-pix_fmt",
    "yuv420p",
    "-f",
    "webm",
]


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """Peak signal-to-noise ratio in dB between two uint8 images or stacks."""
    error = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if error == 0 else float(10 * np.log10(255.0**2 / error))
