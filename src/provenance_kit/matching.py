"""Hamming-distance matching of fixed-width binary hashes against a list.

Perceptual hashes (pHash, PDQ) and per-frame video hashes are compared by
Hamming distance. :func:`hamming_matches` scans a list of reference hashes
linearly, which is fast enough for lists of up to a few hundred thousand
entries (Python's ``int.bit_count`` does the heavy lifting). Modules that need
larger lists can build an index and still return :class:`HammingMatch`.
"""

from __future__ import annotations

import heapq
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Generic, TypeVar

__all__ = ["HammingMatch", "hamming_matches"]

K = TypeVar("K")


@dataclass(frozen=True, slots=True)
class HammingMatch(Generic[K]):
    """A reference hash within the distance threshold of a query.

    Attributes:
        key: The caller's identifier for the reference entry.
        distance: The Hamming distance in bits.
        bits: The hash width in bits.
    """

    key: K
    distance: int
    bits: int

    @property
    def similarity(self) -> float:
        """``1 - distance / bits``: 1.0 for identical hashes."""
        return 1.0 - self.distance / self.bits


def hamming_matches(
    query: int,
    references: Iterable[tuple[K, int]],
    *,
    bits: int,
    max_distance: int,
    limit: int | None = None,
) -> list[HammingMatch[K]]:
    """Finds the references within ``max_distance`` bits of ``query``.

    Args:
        query: The query hash as a non-negative integer below ``2**bits``.
        references: ``(key, hash)`` pairs. Hashes must have the same width as
            the query.
        bits: The hash width, for example 64 for pHash or 256 for PDQ.
        max_distance: The inclusive distance threshold, from 0 to ``bits``.
        limit: If given, return at most this many matches (the closest ones).

    Returns:
        The matches, closest first; ties keep the order of ``references``.

    Raises:
        TypeError: If a hash or parameter is not an ``int``.
        ValueError: If a hash does not fit in ``bits`` or a parameter is out of
            range.
    """
    _check_int("bits", bits)
    _check_int("max_distance", max_distance)
    if bits < 1:
        raise ValueError("bits must be at least 1")
    if not 0 <= max_distance <= bits:
        raise ValueError("max_distance must be between 0 and bits")
    if limit is not None:
        _check_int("limit", limit)
        if limit < 1:
            raise ValueError("limit must be at least 1")
    query_value = _check_hash("query", query, bits)

    found: list[tuple[int, int, K]] = []
    for index, (key, value) in enumerate(references):
        distance = (query_value ^ _check_hash("reference hash", value, bits)).bit_count()
        if distance <= max_distance:
            found.append((distance, index, key))
    ordered = (
        heapq.nsmallest(limit, found, key=_order)
        if limit is not None
        else sorted(found, key=_order)
    )
    return [HammingMatch(key=key, distance=distance, bits=bits) for distance, _, key in ordered]


def _order(item: tuple[int, int, object]) -> tuple[int, int]:
    return item[0], item[1]


def _check_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    return value


def _check_hash(name: str, value: object, bits: int) -> int:
    number = _check_int(name, value)
    if number < 0 or number >> bits:
        raise ValueError(f"{name} does not fit in {bits} unsigned bits")
    return number
