from __future__ import annotations

import numpy as np
import pytest

from provenance_kit.bits import (
    bit_error_rate,
    bits_to_bytes,
    bits_to_int,
    bytes_to_bits,
    hamming_distance,
    int_to_bits,
    normalize_bits,
)


def test_normalize_bits_accepts_strings_and_iterables():
    assert normalize_bits("0101") == "0101"
    assert normalize_bits([0, 1, 0, 1]) == "0101"
    assert normalize_bits((False, True)) == "01"
    assert normalize_bits(np.array([1.0, 0.0])) == "10"
    assert normalize_bits(np.array([1, 0], dtype=np.int64)) == "10"
    assert normalize_bits("") == ""
    assert normalize_bits("11", length=2) == "11"


@pytest.mark.parametrize("bad", ["012", "01 ", [0, 2], [0.5], [None], [np.array([0, 1])]])
def test_normalize_bits_rejects_non_binary_values(bad):
    with pytest.raises(ValueError):
        normalize_bits(bad)


def test_normalize_bits_rejects_bytes_and_non_iterables():
    with pytest.raises(TypeError, match="bytes_to_bits"):
        normalize_bits(b"\x01")
    with pytest.raises(TypeError):
        normalize_bits(7)  # type: ignore[arg-type]


def test_normalize_bits_checks_length():
    with pytest.raises(ValueError, match="expected 3 bits, got 2"):
        normalize_bits("01", length=3)


def test_int_conversions():
    assert int_to_bits(5, 4) == "0101"
    assert bits_to_int("0101") == 5
    assert bits_to_int([1] * 64) == 2**64 - 1
    with pytest.raises(ValueError):
        int_to_bits(16, 4)
    with pytest.raises(ValueError):
        int_to_bits(-1, 4)
    with pytest.raises(ValueError):
        int_to_bits(0, 0)
    with pytest.raises(TypeError):
        int_to_bits(True, 4)
    with pytest.raises(ValueError):
        bits_to_int("")


def test_byte_conversions():
    assert bytes_to_bits(b"\x80\x01") == "1000000000000001"
    assert bits_to_bytes("1000000000000001") == b"\x80\x01"
    assert bytes_to_bits(b"") == ""
    assert bits_to_bytes("") == b""
    assert bytes_to_bits(b"\x00\x00") == "0" * 16
    with pytest.raises(ValueError, match="multiple of 8"):
        bits_to_bytes("101")


def test_hamming_distance_for_each_kind():
    assert hamming_distance(0b1011, 0b0001) == 2
    assert hamming_distance(b"\xff\x00", b"\x0f\x00") == 4
    assert hamming_distance(bytearray(b"\x01"), memoryview(b"\x03")) == 1
    assert hamming_distance("1100", "1010") == 2
    assert hamming_distance(2**256 - 1, 0) == 256


def test_hamming_distance_rejects_mismatches():
    with pytest.raises(TypeError):
        hamming_distance(1, b"\x01")
    with pytest.raises(TypeError):
        hamming_distance(True, 1)
    with pytest.raises(ValueError):
        hamming_distance(b"\x01", b"\x01\x02")
    with pytest.raises(ValueError):
        hamming_distance("01", "011")
    with pytest.raises(ValueError):
        hamming_distance(-1, 1)


def test_bit_error_rate():
    assert bit_error_rate("1111", "1111") == 0.0
    assert bit_error_rate("1111", "1110") == 0.25
    assert bit_error_rate([1, 0], [0, 1]) == 1.0
    with pytest.raises(ValueError):
        bit_error_rate("", "")
    with pytest.raises(ValueError):
        bit_error_rate("11", "1")
