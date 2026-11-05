from __future__ import annotations

from dataclasses import dataclass

import pytest

from provenance_kit.detection import WatermarkDetection
from provenance_kit.payload import PayloadCodec


def test_found_with_bits_and_payload():
    codec = PayloadCodec(16, version_bits=2)
    bits = codec.encode(77, version=1)
    result = WatermarkDetection(found=True, confidence=0.98, bits=bits, payload=codec.decode(bits))
    assert result.payload is not None
    assert result.payload.identifier == 77


def test_not_found_helper():
    result = WatermarkDetection.not_found(confidence=0.1)
    assert not result.found
    assert result.bits is None
    assert result.payload is None


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"found": 1}, TypeError),
        ({"found": True, "confidence": 1.5}, ValueError),
        ({"found": True, "confidence": -0.1}, ValueError),
        ({"found": True, "confidence": float("nan")}, ValueError),
        ({"found": False, "bits": "01"}, ValueError),
        ({"found": True, "bits": ""}, ValueError),
        ({"found": True, "bits": "012"}, ValueError),
    ],
)
def test_validation(kwargs, error):
    with pytest.raises(error):
        WatermarkDetection(**kwargs)


def test_subclasses_can_add_fields():
    @dataclass(frozen=True, slots=True, kw_only=True)
    class FrameDetection(WatermarkDetection):
        frames: tuple[float, ...]

    result = FrameDetection(found=True, confidence=0.9, frames=(0.8, 1.0))
    assert result.frames == (0.8, 1.0)
    with pytest.raises(AttributeError):
        result.found = False  # type: ignore[misc]
