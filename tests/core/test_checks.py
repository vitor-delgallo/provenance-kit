"""Argument validation helpers (``provenance_kit._checks``) and their public call sites."""

from __future__ import annotations

import math
import re
import sys
from collections.abc import Callable

import numpy as np
import pytest

from provenance_kit._checks import check_int, check_number

HUGE = 10**400  # an int far beyond the float range: float(HUGE) raises OverflowError


def test_check_int_accepts_ints_within_bounds():
    assert check_int("n", 5, minimum=0, maximum=10) == 5
    assert check_int("n", HUGE) == HUGE  # ints are compared exactly, never converted
    assert check_int("n", -HUGE, maximum=0) == -HUGE


@pytest.mark.parametrize("value", [True, 1.0, "1", None])
def test_check_int_refuses_other_types(value):
    with pytest.raises(TypeError, match="n must be an int"):
        check_int("n", value)


@pytest.mark.parametrize(
    ("value", "bounds"),
    [(-1, {"minimum": 0}), (11, {"maximum": 10}), (HUGE, {"maximum": 10}), (-HUGE, {"minimum": 0})],
    ids=["below", "above", "huge-above", "huge-below"],
)
def test_check_int_refuses_values_out_of_bounds(value, bounds):
    with pytest.raises(ValueError, match="n must be between"):
        check_int("n", value, **bounds)


def test_check_number_returns_a_float():
    result = check_number("x", 3, minimum=0.0, maximum=10.0)
    assert result == 3.0
    assert type(result) is float


@pytest.mark.parametrize("value", [True, "1", None, 1j])
def test_check_number_refuses_other_types(value):
    with pytest.raises(TypeError, match="x must be a number"):
        check_number("x", value)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_check_number_refuses_non_finite_floats(value):
    with pytest.raises(ValueError, match="x must be finite"):
        check_number("x", value)


def test_check_number_bounds():
    assert check_number("x", 0.0, minimum=0.0) == 0.0
    with pytest.raises(ValueError, match="x must be above 0"):
        check_number("x", 0.0, minimum=0.0, exclusive_minimum=True)
    with pytest.raises(ValueError, match="x must be at least 1"):
        check_number("x", 0.5, minimum=1.0)
    with pytest.raises(ValueError, match="x must be at most 1"):
        check_number("x", 1.5, maximum=1.0)


@pytest.mark.parametrize(
    ("value", "bounds", "message"),
    [
        (HUGE, {"maximum": 1.0}, "x must be at most 1"),
        (HUGE, {"minimum": 0.0, "maximum": 60.0}, "x must be at most 60"),
        (-HUGE, {"minimum": 0.0}, "x must be at least 0"),
        (-HUGE, {"minimum": 0.0, "exclusive_minimum": True}, "x must be above 0"),
        # No bound on the overflowing side: the float range is the bound.
        (HUGE, {}, f"x must be at most {sys.float_info.max:g}"),
        (HUGE, {"minimum": 0.0}, f"x must be at most {sys.float_info.max:g}"),
        (-HUGE, {}, f"x must be at least {-sys.float_info.max:g}"),
        (-HUGE, {"maximum": 1.0}, f"x must be at least {-sys.float_info.max:g}"),
    ],
    ids=[
        "max",
        "range",
        "min",
        "exclusive-min",
        "no-bound",
        "min-only",
        "neg-no-bound",
        "neg-max-only",
    ],
)
def test_check_number_refuses_ints_beyond_the_float_range_with_value_error(value, bounds, message):
    # float(10**400) raises OverflowError, which used to escape every public
    # parameter validated here; it must be the same ValueError as any other
    # out-of-range value.
    with pytest.raises(ValueError, match=re.escape(message)):
        check_number("x", value, **bounds)


def test_check_number_accepts_large_ints_that_fit_a_float():
    assert check_number("x", 10**300) == float(10**300)
    with pytest.raises(ValueError, match="x must be at most 1e\\+09"):
        check_number("x", 10**300, maximum=1e9)


# ---------------------------------------------------------------- public call sites


def _audio_fingerprint():
    from provenance_kit.audio.fingerprint import AudioFingerprint

    return AudioFingerprint(values=(1, 2, 3), duration_seconds=1.0)


def _video_fingerprint():
    from provenance_kit.video.fingerprint import VideoFingerprint

    return VideoFingerprint(fps=1.0, hash_bits=256, hashes=(0,), qualities=(100,))


def _audio_fingerprint_duration() -> None:
    from provenance_kit.audio.fingerprint import AudioFingerprint

    AudioFingerprint(values=(), duration_seconds=HUGE)


def _overlap_threshold() -> None:
    from provenance_kit.audio.fingerprint import overlap_threshold

    overlap_threshold(16, max_bit_error_rate=HUGE)


def _compare_audio_fingerprints() -> None:
    from provenance_kit.audio.fingerprint import compare_fingerprints

    fp = _audio_fingerprint()
    compare_fingerprints(fp, fp, max_bit_error_rate=HUGE)


def _compare_audio_fingerprints_overlap() -> None:
    from provenance_kit.audio.fingerprint import compare_fingerprints

    fp = _audio_fingerprint()
    compare_fingerprints(fp, fp, min_overlap_seconds=HUGE)


def _fingerprint_audio() -> None:
    from provenance_kit.audio.fingerprint import fingerprint_audio

    fingerprint_audio(b"RIFF", max_duration_seconds=HUGE)


def _decode_pcm() -> None:
    from provenance_kit.audio.pcm import decode_pcm

    decode_pcm(b"RIFF", max_duration_seconds=HUGE)


def _alignment_search() -> None:
    from provenance_kit.audio.watermark import AlignmentSearch

    AlignmentSearch(window_seconds=HUGE)


def _detect_audio_watermark() -> None:
    from provenance_kit.audio.watermark import detect_watermark

    detect_watermark(np.zeros((1, 16_000), dtype=np.float32), 16_000, threshold=HUGE)


def _audioseal_backend() -> None:
    from provenance_kit.audio.watermark import AudioSealBackend

    AudioSealBackend(strength=HUGE)  # refused before PyTorch is imported


def _iter_video_frames() -> None:
    from provenance_kit.video.frames import iter_frames

    iter_frames(b"\x00" * 16, fps=HUGE)


def _match_video_fingerprints() -> None:
    from provenance_kit.video.fingerprint import match_fingerprints

    fp = _video_fingerprint()
    match_fingerprints(fp, fp, min_match_fraction=HUGE)


def _detect_video_watermark() -> None:
    from provenance_kit.video.watermark import detect_watermark

    detect_watermark([], max_p_value=HUGE)


def _videoseal_backend() -> None:
    from provenance_kit.video.watermark import VideoSealBackend

    VideoSealBackend(strength=HUGE)  # refused before PyTorch is imported


def _trustmark_backend() -> None:
    from provenance_kit.image.watermark import TrustMarkBackend

    TrustMarkBackend(download_timeout_seconds=HUGE)  # refused before PyTorch is imported


def _match_video_fingerprints_seconds() -> None:
    from provenance_kit.video.fingerprint import match_fingerprints

    fp = _video_fingerprint()
    match_fingerprints(fp, fp, max_gap_seconds=HUGE)


# The validators below do not use _checks; each catches the overflow itself.


def _media_limits() -> None:
    from provenance_kit.limits import MediaLimits

    MediaLimits(subprocess_timeout_seconds=HUGE)


_SPEC_ARGS = {
    "name": "model.bin",
    "url": "https://example.org/model.bin",
    "sha256": "0" * 64,
    "size": 1,
}


def _download_model_file() -> None:
    from provenance_kit.modelfiles import ModelFile, download_model_file

    # Refused before the directory is touched or the network is used.
    download_model_file(ModelFile(**_SPEC_ARGS), "unused", timeout_seconds=HUGE)


def _ensure_model_files() -> None:
    from provenance_kit.modelfiles import ModelFile, ensure_model_files

    ensure_model_files([ModelFile(**_SPEC_ARGS)], "unused", timeout_seconds=HUGE)


def _run_tool() -> None:
    from provenance_kit.tools import run_tool

    run_tool(sys.executable, ["-c", "pass"], timeout_seconds=HUGE)  # refused before starting


class _FakeImageBackend:
    payload_bits = 61

    def embed(self, image, bits, *, strength):  # pragma: no cover - never reached
        raise AssertionError("the strength must be refused first")

    def extract(self, image):  # pragma: no cover - never reached
        raise AssertionError("not used")


def _image_watermarker() -> None:
    from provenance_kit.image.watermark import ImageWatermarker

    ImageWatermarker(_FakeImageBackend(), strength=HUGE)


def _trustmark_embed() -> None:
    from PIL import Image

    from provenance_kit.image.watermark import TrustMarkBackend, TrustMarkEncoding

    # The constructor needs PyTorch and the model files; embed checks its
    # arguments before any model call, so a backend with only its encoding
    # set reaches the strength check.
    backend = object.__new__(TrustMarkBackend)
    backend._encoding = TrustMarkEncoding.BCH_5
    bits = "0" * backend.payload_bits
    backend.embed(Image.new("RGB", (8, 8)), bits, strength=HUGE)


def _open_timestamper() -> None:
    from provenance_kit.provenance.timestamps import OpenTimestamper

    OpenTimestamper(timeout_seconds=HUGE)


# Each call and the parameter its ValueError must name.
PUBLIC_CALLS: dict[str, tuple[Callable[[], None], str]] = {
    "audio.fingerprint.AudioFingerprint": (_audio_fingerprint_duration, "duration_seconds"),
    "audio.fingerprint.overlap_threshold": (_overlap_threshold, "max_bit_error_rate"),
    "audio.fingerprint.compare_fingerprints(max_bit_error_rate)": (
        _compare_audio_fingerprints,
        "max_bit_error_rate",
    ),
    "audio.fingerprint.compare_fingerprints(min_overlap_seconds)": (
        _compare_audio_fingerprints_overlap,
        "min_overlap_seconds",
    ),
    "audio.fingerprint.fingerprint_audio": (_fingerprint_audio, "max_duration_seconds"),
    "audio.pcm.decode_pcm": (_decode_pcm, "max_duration_seconds"),
    "audio.watermark.AlignmentSearch": (_alignment_search, "window_seconds"),
    "audio.watermark.detect_watermark": (_detect_audio_watermark, "threshold"),
    "audio.watermark.AudioSealBackend": (_audioseal_backend, "strength"),
    "video.frames.iter_frames": (_iter_video_frames, "fps"),
    "video.fingerprint.match_fingerprints": (_match_video_fingerprints, "min_match_fraction"),
    "video.watermark.detect_watermark": (_detect_video_watermark, "max_p_value"),
    "video.watermark.VideoSealBackend": (_videoseal_backend, "strength"),
    "image.watermark.TrustMarkBackend": (_trustmark_backend, "download_timeout_seconds"),
    "video.fingerprint.match_fingerprints(max_gap_seconds)": (
        _match_video_fingerprints_seconds,
        "max_gap_seconds",
    ),
    "limits.MediaLimits": (_media_limits, "MediaLimits.subprocess_timeout_seconds"),
    "modelfiles.download_model_file": (_download_model_file, "timeout_seconds"),
    "modelfiles.ensure_model_files": (_ensure_model_files, "timeout_seconds"),
    "tools.run_tool": (_run_tool, "timeout_seconds"),
    "image.watermark.ImageWatermarker": (_image_watermarker, "strength"),
    "image.watermark.TrustMarkBackend.embed": (_trustmark_embed, "strength"),
    "provenance.timestamps.OpenTimestamper": (_open_timestamper, "timeout_seconds"),
}


@pytest.mark.parametrize(("call", "parameter"), PUBLIC_CALLS.values(), ids=PUBLIC_CALLS.keys())
def test_public_numeric_parameters_refuse_overflowing_ints_with_value_error(call, parameter):
    # The message names the parameter, so the refusal comes from its own check.
    with pytest.raises(ValueError, match=rf"^{re.escape(parameter)} must be"):
        call()
