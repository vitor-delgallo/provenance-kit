from __future__ import annotations

import random

import numpy as np
import pytest

from provenance_kit.errors import PayloadError, ProvenanceKitError
from provenance_kit.payload import MAX_TOTAL_BITS, DecodedPayload, PayloadCodec
from tests.support import SEED

BACKEND_SIZES = [16, 40, 61, 68, 75, 96, 256, 1024]


@pytest.mark.parametrize("total_bits", BACKEND_SIZES)
def test_round_trip_for_every_backend_size(total_bits):
    version_bits = 2 if total_bits <= 16 else 4
    codec = PayloadCodec(total_bits, version_bits=version_bits)
    rand = random.Random(SEED + total_bits)
    for _ in range(50):
        identifier = rand.randrange(codec.capacity)
        version = rand.randrange(codec.max_version + 1)
        bits = codec.encode(identifier, version=version)
        assert len(bits) == total_bits
        assert set(bits) <= {"0", "1"}
        decoded = codec.decode(bits)
        assert (decoded.version, decoded.identifier) == (version, identifier)
        assert decoded.valid


def test_layout_puts_the_version_first_most_significant_bit_first():
    codec = PayloadCodec(8, version_bits=3)
    assert codec.encode(0b00101, version=0b110) == "110" + "00101"


def test_capacity_trade_off():
    codec = PayloadCodec(16, version_bits=4, check_bits=2)
    assert codec.identifier_bits == 10
    assert codec.capacity == 1024
    assert codec.max_identifier == 1023
    assert codec.max_version == 15
    assert PayloadCodec(16, version_bits=0).capacity == 65536
    assert PayloadCodec(16, version_bits=0).max_version == 0


def test_boundary_identifiers():
    codec = PayloadCodec(40)
    assert codec.decode(codec.encode(0)).identifier == 0
    assert codec.decode(codec.encode(codec.max_identifier)).identifier == codec.max_identifier


def test_identifier_that_does_not_fit_is_a_payload_error():
    codec = PayloadCodec(16, version_bits=4)
    with pytest.raises(PayloadError, match="does not fit in 12 bits"):
        codec.encode(4096)
    assert issubclass(PayloadError, ProvenanceKitError)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"identifier": -1}, ValueError),
        ({"identifier": True}, TypeError),
        ({"identifier": 1.0}, TypeError),
        ({"identifier": "7"}, TypeError),
        ({"identifier": 1, "version": 16}, ValueError),
        ({"identifier": 1, "version": -1}, ValueError),
        ({"identifier": 1, "version": False}, TypeError),
    ],
)
def test_caller_mistakes_in_encode(kwargs, error):
    codec = PayloadCodec(61, version_bits=4)
    identifier = kwargs.pop("identifier")
    with pytest.raises(error):
        codec.encode(identifier, **kwargs)


@pytest.mark.parametrize(
    ("args", "kwargs", "error"),
    [
        ((0,), {}, ValueError),
        ((MAX_TOTAL_BITS + 1,), {}, ValueError),
        ((16,), {"version_bits": 16}, ValueError),
        ((16,), {"version_bits": 8, "check_bits": 8}, ValueError),
        ((16,), {"version_bits": -1}, ValueError),
        ((512,), {"check_bits": 257}, ValueError),
        ((16.0,), {}, TypeError),
        ((True,), {}, TypeError),
        ((16,), {"versions": "12"}, TypeError),
        ((16,), {"versions": []}, ValueError),
        ((16,), {"versions": [16]}, ValueError),
    ],
)
def test_invalid_configurations(args, kwargs, error):
    with pytest.raises(error):
        PayloadCodec(*args, **kwargs)


def test_versions_restrict_encoding_and_flag_unknown_versions():
    codec = PayloadCodec(16, version_bits=2, versions=[1, 2])
    assert codec.versions == frozenset({1, 2})
    with pytest.raises(ValueError, match="not one of"):
        codec.encode(5, version=0)
    unknown = codec.decode("00" + "0" * 14)
    assert unknown.version == 0
    assert not unknown.version_known
    assert not unknown.valid
    with pytest.raises(PayloadError, match="unknown version"):
        unknown.require_valid()
    known = codec.decode(codec.encode(5, version=2))
    assert known.require_valid() is known


def test_check_bits_reject_most_random_garbage():
    codec = PayloadCodec(61, version_bits=4, check_bits=8)
    rand = random.Random(SEED)
    trials = 4000
    valid = sum(codec.decode(format(rand.getrandbits(61), "061b")).valid for _ in range(trials))
    # Expected rate 2**-8: about 16 in 4000.
    assert valid < 60
    decoded = codec.decode(codec.encode(123, version=3))
    assert decoded.check_ok is True
    assert decoded.valid


def test_a_flipped_bit_breaks_the_check():
    codec = PayloadCodec(256, version_bits=4, check_bits=16)
    bits = codec.encode(2**200 + 12345, version=1)
    for position in (0, 3, 100, 239, 255):
        flipped = bits[:position] + ("1" if bits[position] == "0" else "0") + bits[position + 1 :]
        decoded = codec.decode(flipped)
        assert decoded.check_ok is False
        with pytest.raises(PayloadError, match="check bits"):
            decoded.require_valid()


def test_without_check_bits_check_ok_is_none():
    decoded = PayloadCodec(16).decode("0" * 16)
    assert decoded.check_ok is None
    assert decoded.valid


def test_decode_never_raises_for_well_formed_garbage():
    rand = random.Random(SEED)
    codecs = [
        PayloadCodec(16, version_bits=0),
        PayloadCodec(40, version_bits=4, check_bits=6, versions={1}),
        PayloadCodec(256, version_bits=8, check_bits=32),
    ]
    for codec in codecs:
        for _ in range(500):
            result = codec.decode(
                format(rand.getrandbits(codec.total_bits), f"0{codec.total_bits}b")
            )
            assert isinstance(result, DecodedPayload)


def test_decode_accepts_detector_outputs():
    codec = PayloadCodec(16, version_bits=2)
    bits = codec.encode(1000, version=1)
    as_ints = [int(c) for c in bits]
    assert codec.decode(as_ints).identifier == 1000
    assert codec.decode([bool(b) for b in as_ints]).identifier == 1000
    assert codec.decode(np.array(as_ints, dtype=np.float32)).identifier == 1000
    assert codec.decode(np.array(as_ints, dtype=bool)).identifier == 1000


@pytest.mark.parametrize(
    ("bits", "error"),
    [
        ("0" * 15, ValueError),
        ("0" * 17, ValueError),
        ("0" * 15 + "2", ValueError),
        ([0] * 15 + [0.5], ValueError),
        ([0] * 15 + [float("nan")], ValueError),
        (b"\x00\x00", TypeError),
        (12345, TypeError),
    ],
)
def test_malformed_bits_are_caller_mistakes(bits, error):
    with pytest.raises(error):
        PayloadCodec(16).decode(bits)


def test_codec_is_immutable_and_comparable():
    codec = PayloadCodec(61, version_bits=4)
    assert codec == PayloadCodec(61, version_bits=4)
    with pytest.raises(AttributeError):
        codec.total_bits = 40  # type: ignore[misc]
