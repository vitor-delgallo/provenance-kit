from __future__ import annotations

import random

import pytest

from provenance_kit.matching import HammingMatch, hamming_matches
from tests.support import SEED


def test_matches_are_filtered_and_sorted_closest_first():
    references = [("far", 0b1111_0000), ("exact", 0b1010_1010), ("near", 0b1010_1011)]
    matches = hamming_matches(0b1010_1010, references, bits=8, max_distance=2)
    assert [m.key for m in matches] == ["exact", "near"]
    assert [m.distance for m in matches] == [0, 1]
    assert matches[0].similarity == 1.0
    assert matches[1].similarity == pytest.approx(7 / 8)


def test_ties_keep_input_order_and_limit_keeps_the_closest():
    references = [("a", 0b0001), ("b", 0b0010), ("c", 0b0000), ("d", 0b0100)]
    matches = hamming_matches(0, references, bits=4, max_distance=4, limit=3)
    assert [m.key for m in matches] == ["c", "a", "b"]


def test_large_list_with_pdq_width():
    rand = random.Random(SEED)
    query = rand.getrandbits(256)
    references = [(i, rand.getrandbits(256)) for i in range(20_000)]
    near = query ^ 0b111  # 3 bits away
    references.append(("planted", near))
    matches = hamming_matches(query, references, bits=256, max_distance=31)
    assert [m.key for m in matches] == ["planted"]
    assert matches[0] == HammingMatch(key="planted", distance=3, bits=256)


@pytest.mark.parametrize(
    ("query", "references", "kwargs", "error"),
    [
        (0, [], {"bits": 0, "max_distance": 0}, ValueError),
        (0, [], {"bits": 8, "max_distance": 9}, ValueError),
        (0, [], {"bits": 8, "max_distance": -1}, ValueError),
        (0, [], {"bits": 8, "max_distance": 1, "limit": 0}, ValueError),
        (256, [], {"bits": 8, "max_distance": 1}, ValueError),
        (-1, [], {"bits": 8, "max_distance": 1}, ValueError),
        (0, [("x", 256)], {"bits": 8, "max_distance": 1}, ValueError),
        (0, [("x", "00")], {"bits": 8, "max_distance": 1}, TypeError),
        (0.0, [], {"bits": 8, "max_distance": 1}, TypeError),
        (0, [], {"bits": True, "max_distance": 1}, TypeError),
    ],
)
def test_invalid_inputs(query, references, kwargs, error):
    with pytest.raises(error):
        hamming_matches(query, references, **kwargs)
