from __future__ import annotations

import dataclasses

import pytest

from provenance_kit.errors import MediaError, MediaLimitExceededError
from provenance_kit.limits import MediaLimits


def test_defaults_are_valid_and_documented_values():
    limits = MediaLimits()
    assert limits.max_bytes == 256 * 1024 * 1024
    assert limits.max_pixels == 64_000_000
    assert limits.max_width == limits.max_height == 16_384
    assert limits.max_duration_seconds == 10_800.0
    assert limits.max_frames == 648_000
    assert limits.max_audio_channels == 8
    assert limits.max_sample_rate == 192_000
    assert limits.subprocess_timeout_seconds == 600.0
    # Pillow refuses images above twice its 89,478,485-pixel warning threshold;
    # the default must stay under the warning itself.
    assert limits.max_pixels < 89_478_485


def test_is_frozen_and_replaceable():
    limits = MediaLimits()
    with pytest.raises(dataclasses.FrozenInstanceError):
        limits.max_bytes = 1  # type: ignore[misc]
    bigger = dataclasses.replace(limits, max_bytes=2**31)
    assert bigger.max_bytes == 2**31
    assert bigger.max_pixels == limits.max_pixels


@pytest.mark.parametrize("field", [f.name for f in dataclasses.fields(MediaLimits)])
@pytest.mark.parametrize("value", [0, -1])
def test_non_positive_values_are_refused(field, value):
    with pytest.raises(ValueError, match=field):
        MediaLimits(**{field: value})


@pytest.mark.parametrize("field", ["max_bytes", "max_pixels", "max_frames", "max_sample_rate"])
@pytest.mark.parametrize("value", [1.5, "10", True, None])
def test_integer_fields_refuse_other_types(field, value):
    with pytest.raises(TypeError, match=field):
        MediaLimits(**{field: value})


@pytest.mark.parametrize("field", ["max_duration_seconds", "subprocess_timeout_seconds"])
def test_float_fields(field):
    assert getattr(MediaLimits(**{field: 5}), field) == 5
    with pytest.raises(ValueError):
        MediaLimits(**{field: float("inf")})
    with pytest.raises(ValueError):
        MediaLimits(**{field: float("nan")})
    with pytest.raises(TypeError):
        MediaLimits(**{field: True})


def test_check_bytes():
    limits = MediaLimits(max_bytes=100)
    limits.check_bytes(100)
    with pytest.raises(MediaLimitExceededError) as info:
        limits.check_bytes(101)
    assert info.value.limit == "max_bytes"
    assert info.value.maximum == 100
    assert info.value.actual == 101
    assert str(info.value) == "media exceeds max_bytes: 101 > 100"
    assert isinstance(info.value, MediaError)


def test_check_dimensions_checks_each_side_and_the_area():
    limits = MediaLimits(max_width=100, max_height=50, max_pixels=4000)
    limits.check_dimensions(80, 50)
    with pytest.raises(MediaLimitExceededError, match="max_width"):
        limits.check_dimensions(101, 1)
    with pytest.raises(MediaLimitExceededError, match="max_height"):
        limits.check_dimensions(1, 51)
    with pytest.raises(MediaLimitExceededError, match="max_pixels"):
        limits.check_dimensions(100, 41)


def test_check_duration_refuses_non_finite_values():
    limits = MediaLimits(max_duration_seconds=10)
    limits.check_duration(10.0)
    for bad in (10.5, float("inf"), float("nan")):
        with pytest.raises(MediaLimitExceededError, match="max_duration_seconds"):
            limits.check_duration(bad)


def test_check_frames_and_audio():
    limits = MediaLimits(max_frames=10, max_audio_channels=2, max_sample_rate=48_000)
    limits.check_frames(10)
    limits.check_audio(2, 48_000)
    with pytest.raises(MediaLimitExceededError, match="max_frames"):
        limits.check_frames(11)
    with pytest.raises(MediaLimitExceededError, match="max_audio_channels"):
        limits.check_audio(3, 8000)
    with pytest.raises(MediaLimitExceededError, match="max_sample_rate"):
        limits.check_audio(1, 96_000)
