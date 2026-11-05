"""Bit-string helpers shared by watermark payloads and perceptual hashes.

The canonical in-memory form of a bit sequence in provenance-kit is a ``str``
of ``"0"`` and ``"1"`` characters, most significant bit first. It is what
TrustMark accepts in binary mode, it compares and hashes cheaply, and it
prints legibly in tests. Hashes are also handled as non-negative ``int``
values with a known bit width (bit 0 of the string is the most significant
bit of the integer) or as big-endian ``bytes``.

All functions raise ``TypeError`` or ``ValueError`` for malformed input; they
never raise anything else.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TypeAlias

__all__ = [
    "BitsLike",
    "bit_error_rate",
    "bits_to_bytes",
    "bits_to_int",
    "bytes_to_bits",
    "hamming_distance",
    "int_to_bits",
    "normalize_bits",
]

#: Inputs accepted wherever a bit sequence is expected: a ``"0"``/``"1"``
#: string, or an iterable of values equal to 0 or 1 (ints, bools, floats such as
#: ``0.0``/``1.0``, NumPy scalars, or a flattened NumPy array).
BitsLike: TypeAlias = str | Iterable[object]

_BITS = frozenset("01")


def normalize_bits(bits: BitsLike, *, length: int | None = None) -> str:
    """Converts a bit sequence to the canonical ``"0"``/``"1"`` string.

    Args:
        bits: The bits, as a string or an iterable of 0/1 values.
        length: If given, the exact number of bits required.

    Returns:
        The bits as a string, most significant first.

    Raises:
        TypeError: If ``bits`` is ``bytes`` (ambiguous: use
            :func:`bytes_to_bits`) or not iterable.
        ValueError: If a value is not 0 or 1, or the length is wrong.
    """
    if isinstance(bits, (bytes, bytearray, memoryview)):
        raise TypeError("bits must not be bytes; convert them with bytes_to_bits()")
    if isinstance(bits, str):
        text = bits
        if not _BITS.issuperset(text):
            raise ValueError("a bit string may contain only '0' and '1'")
    else:
        try:
            iterator = iter(bits)
        except TypeError:
            raise TypeError(
                f"bits must be a str or an iterable, got {type(bits).__name__}"
            ) from None
        chars: list[str] = []
        for value in iterator:
            chars.append(_bit_char(value))
        text = "".join(chars)
    if length is not None and len(text) != length:
        raise ValueError(f"expected {length} bits, got {len(text)}")
    return text


def _bit_char(value: object) -> str:
    # Equality covers int, bool, float 0.0/1.0 and NumPy scalars alike; NaN
    # compares unequal to both and is rejected.
    try:
        if value == 1:
            return "1"
        if value == 0:
            return "0"
    except (TypeError, ValueError):
        pass
    raise ValueError("each bit must be 0 or 1")


def bits_to_int(bits: BitsLike) -> int:
    """Interprets a bit sequence as an unsigned big-endian integer.

    Args:
        bits: The bits, most significant first. Must not be empty.

    Returns:
        The integer value.

    Raises:
        ValueError: If the bits are empty or malformed.
    """
    text = normalize_bits(bits)
    if not text:
        raise ValueError("cannot convert an empty bit sequence to an integer")
    return int(text, 2)


def int_to_bits(value: int, width: int) -> str:
    """Formats an unsigned integer as a fixed-width bit string.

    Args:
        value: A non-negative integer below ``2**width``.
        width: The number of bits, at least 1.

    Returns:
        The bit string, most significant bit first, zero-padded.

    Raises:
        TypeError: If ``value`` or ``width`` is not an ``int`` (``bool`` is
            refused).
        ValueError: If ``width`` < 1 or ``value`` does not fit.
    """
    _check_int("value", value)
    _check_int("width", width)
    if width < 1:
        raise ValueError("width must be at least 1")
    if value < 0 or value >> width:
        raise ValueError(f"value does not fit in {width} unsigned bits")
    return format(value, f"0{width}b")


def bytes_to_bits(data: bytes | bytearray | memoryview) -> str:
    """Converts bytes to a bit string, most significant bit of byte 0 first.

    Args:
        data: The bytes.

    Returns:
        A string of ``8 * len(data)`` bits.
    """
    raw = bytes(data)
    if not raw:
        return ""
    return format(int.from_bytes(raw, "big"), f"0{8 * len(raw)}b")


def bits_to_bytes(bits: BitsLike) -> bytes:
    """Packs a bit string whose length is a multiple of 8 into bytes.

    Args:
        bits: The bits, most significant first.

    Returns:
        The packed bytes.

    Raises:
        ValueError: If the bits are malformed or not a whole number of bytes.
    """
    text = normalize_bits(bits)
    if len(text) % 8:
        raise ValueError("the number of bits must be a multiple of 8")
    if not text:
        return b""
    return int(text, 2).to_bytes(len(text) // 8, "big")


def hamming_distance(
    a: int | bytes | bytearray | memoryview | str,
    b: int | bytes | bytearray | memoryview | str,
) -> int:
    """Counts the bits that differ between two values of the same kind.

    Accepted pairs: two non-negative ``int`` values, two byte strings of the
    same length, or two bit strings of the same length.

    Args:
        a: The first value.
        b: The second value.

    Returns:
        The number of differing bits.

    Raises:
        TypeError: If the two values are not of the same kind.
        ValueError: If lengths differ, an int is negative, or a bit string is
            malformed.
    """
    if isinstance(a, bool) or isinstance(b, bool):
        raise TypeError("bool is not a hash value")
    if isinstance(a, int) and isinstance(b, int):
        if a < 0 or b < 0:
            raise ValueError("hash integers must be non-negative")
        return (a ^ b).bit_count()
    if isinstance(a, (bytes, bytearray, memoryview)) and isinstance(
        b, (bytes, bytearray, memoryview)
    ):
        left, right = bytes(a), bytes(b)
        if len(left) != len(right):
            raise ValueError("byte strings must have the same length")
        return (int.from_bytes(left, "big") ^ int.from_bytes(right, "big")).bit_count()
    if isinstance(a, str) and isinstance(b, str):
        left_bits, right_bits = normalize_bits(a), normalize_bits(b)
        if len(left_bits) != len(right_bits):
            raise ValueError("bit strings must have the same length")
        return sum(x != y for x, y in zip(left_bits, right_bits, strict=True))
    raise TypeError("both values must be ints, byte strings, or bit strings")


def bit_error_rate(expected: BitsLike, actual: BitsLike) -> float:
    """Returns the fraction of bits that differ, in ``[0.0, 1.0]``.

    Used to verify a watermark read-back against the embedded payload.

    Args:
        expected: The embedded bits.
        actual: The extracted bits, of the same length.

    Returns:
        ``hamming_distance / length``.

    Raises:
        ValueError: If the sequences are empty, malformed, or differ in length.
    """
    left = normalize_bits(expected)
    right = normalize_bits(actual, length=len(left))
    if not left:
        raise ValueError("cannot compare empty bit sequences")
    return hamming_distance(left, right) / len(left)


def _check_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
