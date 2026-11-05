"""Unit tests of the VideoSeal repetition code and its detection statistic."""

from __future__ import annotations

import numpy as np
import pytest

from provenance_kit.video.watermark import RepetitionCode


def hard_scores(channel: str) -> np.ndarray:
    return np.array([1.0 if c == "1" else -1.0 for c in channel])


def random_bits(rng: np.random.Generator, n: int) -> str:
    return "".join(rng.choice(["0", "1"], size=n))


def test_layout_sizes():
    assert RepetitionCode().data_bits == 64
    assert RepetitionCode(copies=3).data_bits == 85
    assert RepetitionCode(copies=3).padding_bits == 1
    assert RepetitionCode(copies=1).data_bits == 256


def test_round_trip_without_noise(rng):
    for copies in (1, 2, 3, 4, 8):
        code = RepetitionCode(copies=copies)
        payload = random_bits(rng, code.data_bits)
        channel = code.encode(payload)
        assert len(channel) == 256
        decoded = code.decode(hard_scores(channel))
        assert decoded.bits == payload
        assert decoded.agreement == 1.0


def test_whitening_spreads_low_entropy_payloads():
    code = RepetitionCode(copies=4)
    channel = code.encode("0" * 64)
    copies = [channel[i * 64 : (i + 1) * 64] for i in range(4)]
    assert len(set(copies)) == 4  # every copy has its own mask
    assert 96 < channel.count("1") < 160
    plain = RepetitionCode(copies=4, whiten=False).encode("0" * 64)
    assert plain == "0" * 256


def test_majority_vote_corrects_minority_errors(rng):
    code = RepetitionCode(copies=5)
    payload = random_bits(rng, code.data_bits)
    scores = hard_scores(code.encode(payload))
    # Flip two of the five copies of every data bit.
    for bit in range(code.data_bits):
        for copy in rng.choice(5, size=2, replace=False):
            scores[copy * code.data_bits + bit] *= -1
    decoded = code.decode(scores)
    assert decoded.bits == payload
    assert decoded.agreement == pytest.approx(0.6)


def test_soft_vote_uses_magnitudes(rng):
    code = RepetitionCode(copies=2)
    payload = random_bits(rng, code.data_bits)
    scores = hard_scores(code.encode(payload)) * 3.0
    scores[: code.data_bits] *= -0.1  # copy 0 is wrong but unsure
    assert code.decode(scores).bits == payload


def test_agreement_is_measured_against_the_soft_decoded_bits():
    # Data bit 0: three weak "0" votes and one strong "1" vote, so the soft
    # sum decodes 1 while the hard majority says 0. Data bit 1: all agree on 1.
    code = RepetitionCode(copies=4, total_bits=8, whiten=False)
    scores = np.array([-0.1, 1.0, -0.1, 1.0, -0.1, 1.0, 5.0, 1.0])
    decoded = code.decode(scores)
    assert decoded.bits == "11"
    # Bit 0: 1 of 4 copies agree with the decoded 1; bit 1: 4 of 4.
    assert decoded.agreement == pytest.approx(5 / 8)
    # The p-value stays the calibrated hard-majority statistic: 3 + 4 copies.
    majority = RepetitionCode(copies=4, total_bits=8, whiten=False).decode(
        np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0])
    )
    assert majority.agreement == pytest.approx(7 / 8)
    assert decoded.p_value == pytest.approx(majority.p_value)


def test_agreement_with_whitening_matches_the_unwhitened_layout(rng):
    code = RepetitionCode(copies=4)
    payload = random_bits(rng, code.data_bits)
    scores = hard_scores(code.encode(payload)) * rng.uniform(0.1, 2.0, size=256)
    # Make copy 3 of every bit wrong but very confident: the soft vote flips
    # each bit to copy 3's value, so only 1 of 4 copies agrees with it.
    scores[3 * 64 : 4 * 64] *= -1000.0
    decoded = code.decode(scores)
    assert decoded.bits == "".join("1" if c == "0" else "0" for c in payload)
    assert decoded.agreement == pytest.approx(0.25)


def test_p_value_is_tiny_for_a_clean_watermark(rng):
    code = RepetitionCode(copies=4)
    decoded = code.decode(hard_scores(code.encode(random_bits(rng, 64))))
    assert decoded.p_value < 1e-30


def test_p_value_is_calibrated_on_random_scores(rng):
    code = RepetitionCode(copies=4)
    p_values = np.array([code.decode(rng.normal(size=256)).p_value for _ in range(2000)])
    assert np.mean(p_values <= 0.01) <= 0.02
    assert 0.05 <= np.mean(p_values <= 0.1) <= 0.13
    assert np.all((p_values >= 0) & (p_values <= 1))


def test_p_value_is_calibrated_on_biased_scores(rng):
    # Unmarked content can push every channel the same way; whitening keeps
    # the copies' agreement at chance level.
    code = RepetitionCode(copies=4)
    p_values = [code.decode(np.abs(rng.normal(size=256)) + 0.5).p_value for _ in range(200)]
    assert np.mean(np.array(p_values) <= 0.01) <= 0.03


def test_single_copy_has_no_statistic(rng):
    decoded = RepetitionCode(copies=1).decode(rng.normal(size=256))
    assert decoded.p_value == 1.0


@pytest.mark.parametrize(
    "kwargs", [{"copies": 0}, {"copies": 257}, {"total_bits": 0}, {"total_bits": 5000}]
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        RepetitionCode(**kwargs)


def test_invalid_types():
    with pytest.raises(TypeError):
        RepetitionCode(copies=2.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        RepetitionCode(whiten=1)  # type: ignore[arg-type]


def test_encode_and_decode_validate_lengths():
    code = RepetitionCode(copies=4)
    with pytest.raises(ValueError):
        code.encode("01")
    with pytest.raises(ValueError):
        code.decode([1.0] * 255)
    with pytest.raises(ValueError):
        code.decode([float("nan")] * 256)
