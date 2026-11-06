"""Chromaprint audio fingerprints through the ``fpcalc`` binary.

:func:`fingerprint_audio` decodes untrusted media with ffmpeg (limits
enforced, see :mod:`provenance_kit.audio.pcm`) to mono 16-bit PCM at
11,025 Hz, Chromaprint's working rate, and streams it into
``fpcalc -raw -json`` through standard input. fpcalc therefore only ever
parses raw PCM, never the untrusted container. The result is an
:class:`AudioFingerprint`: the raw 32-bit sub-fingerprints (one every
1365/11025 s, about 0.124 s) and the duration, which is taken from the PCM
sample count because fpcalc reports ``0.0`` when it reads standard input.

:meth:`AudioFingerprint.to_bytes` gives a compact, versioned binary form for
a database column (4 bytes per sub-fingerprint plus a 20-byte header, about
116 KiB per hour); :meth:`AudioFingerprint.from_bytes` validates it strictly.

:func:`compare_fingerprints` aligns two fingerprints by **bit error rate**:
for every offset it compares the overlapping sub-fingerprints bit by bit
(all offsets at once with FFT cross-correlations of the 32 bit planes, so
long references are cheap). A clip cut from a longer recording is found at
its offset. Low-information overlaps cannot match: digital silence
(:data:`SILENCE_VALUE`) is left out of the comparison, an overlap needs at
least :data:`MIN_COMPARED_ITEMS` compared sub-fingerprints (audio shorter
than about 4.6 s never matches), its agreement must be well above the
chance level of the compared pairs' own bit statistics
(:data:`MAX_CHANCE_ERROR_FRACTION`; stationary noise and steady tones agree
by chance at bit error rates as low as 0.15; in this test a run of
identical pairs, as two steady drones give, counts once) and it must hold
:data:`MIN_DISTINCT_VALUES` distinct values.

The bit error rate threshold depends on the overlap
(:func:`overlap_threshold`): :data:`DEFAULT_MAX_BIT_ERROR_RATE` (0.20) for
overlaps of 20 s or more of compared span, down to 0.11 at 5 s or less,
linear in between. Signals that sound alike over a few seconds share short
windows at bit error rates of 0.10 to 0.20 (windows of the test suite's
synthetic songs matched another song 36 % of the time at 5 s with a flat
0.20; 0.2 % with this rule), while long genuine overlaps need the room: a
+6 dB bass shelf on bass-heavy music gives up to 0.14 on full tracks.
Measured numbers (13 degradations, 6 clip lengths, false matches by window
length) are in ``docs/audio.md``; in short, every degradation measured is
found on clips of 15 s and longer, and on 5 to 10 s clips all but bass
shelving of bass-heavy music and a few Opus-encoded synthetic chirps.

Not handled: tempo or pitch changes, and heavy mixing with other audio;
Chromaprint is not designed to survive them.

Thread safety: the functions are thread-safe; each call runs its own tools.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from provenance_kit import _ffmpeg
from provenance_kit._checks import check_int, check_number
from provenance_kit._logging import get_logger
from provenance_kit._typing import BytesLike, StrPath

# The PCM module's private internals: fingerprints decode through the same
# probe, limits and ffmpeg arguments.
from provenance_kit.audio.pcm import (
    _AUDIO_CONTAINERS,
    PcmAudio,
    _duration_cap_bytes,
    _pcm_args,
    _probe_staged,
)
from provenance_kit.errors import (
    ExternalToolError,
    MediaDecodeError,
    MediaLimitExceededError,
)
from provenance_kit.limits import MediaLimits
from provenance_kit.tools import ExternalTools, Tool, ToolProcess

__all__ = [
    "CHROMAPRINT_ALGORITHM",
    "CHROMAPRINT_SAMPLE_RATE",
    "DEFAULT_MAX_BIT_ERROR_RATE",
    "DEFAULT_MIN_OVERLAP_SECONDS",
    "FORMAT_VERSION",
    "ITEM_SECONDS",
    "LONG_OVERLAP_SECONDS",
    "MAX_CHANCE_ERROR_FRACTION",
    "MAX_COMPARISON_SECONDS",
    "MAX_FINGERPRINT_ITEMS",
    "MIN_COMPARED_ITEMS",
    "MIN_DISTINCT_VALUES",
    "SHORT_OVERLAP_FACTOR",
    "SHORT_OVERLAP_SECONDS",
    "SILENCE_VALUE",
    "AudioFingerprint",
    "AudioMatch",
    "compare_fingerprints",
    "fingerprint_audio",
    "fingerprint_pcm",
    "overlap_threshold",
]

_log = get_logger(__name__)

#: Version of the binary layout written by :meth:`AudioFingerprint.to_bytes`.
FORMAT_VERSION = 1

#: The Chromaprint algorithm used (fpcalc's default, the one AcoustID uses).
CHROMAPRINT_ALGORITHM = 2

#: The sample rate Chromaprint works at; PCM is decoded at this rate.
CHROMAPRINT_SAMPLE_RATE = 11_025

#: Time between consecutive sub-fingerprints: a hop of 1365 samples at
#: 11,025 Hz (4096-sample frames with two-thirds overlap).
ITEM_SECONDS = 1365 / CHROMAPRINT_SAMPLE_RATE

#: Default threshold on the bit error rate of a **long** overlap (at least
#: :data:`LONG_OVERLAP_SECONDS` of compared span); shorter overlaps are held
#: to a stricter one (:func:`overlap_threshold`). Measured on full tracks and
#: 30 s clips of 16 generated pieces after 13 degradations (MP3 64 and
#: 96 kb/s, AAC 64 kb/s, Opus 32 and 48 kb/s, 22.05 kHz, volume and bass or
#: treble shelving at +-6 dB, mono): at most 0.144 (bass +6 dB on bass-heavy
#: music), 0.097 without the bass shelving (``docs/audio.md``).
DEFAULT_MAX_BIT_ERROR_RATE = 0.20

#: Overlaps whose compared span (compared sub-fingerprints x
#: :data:`ITEM_SECONDS`) is at most this many seconds must reach
#: :data:`SHORT_OVERLAP_FACTOR` x ``max_bit_error_rate`` (0.11 by default).
SHORT_OVERLAP_SECONDS = 5.0

#: From this compared span on, ``max_bit_error_rate`` applies unchanged; the
#: threshold rises linearly from the short one in between.
LONG_OVERLAP_SECONDS = 20.0

#: The fraction of ``max_bit_error_rate`` that short overlaps must reach.
#: Short windows of different signals that sound alike agree far more often
#: than long ones: windows of the test suite's synthetic songs matched
#: another song 36 % (5 s span), 21 % (8 s) and 12 % (10 s) of the time at a
#: flat 0.20, and 0.2 %, 0.0 % and 0.0 % with this rule (1,000 windows
#: each; ``docs/audio.md``).
SHORT_OVERLAP_FACTOR = 0.55

#: Default minimum overlap for an offset to be considered (seconds), or the
#: whole shorter fingerprint if it is shorter, but never fewer than
#: :data:`MIN_COMPARED_ITEMS` compared sub-fingerprints.
DEFAULT_MIN_OVERLAP_SECONDS = 5.0

#: The sub-fingerprint Chromaprint (algorithm 2) gives digital silence: every
#: frame of an all-zero signal quantizes the same way. Pairs where either side
#: holds it are left out of the comparison, as AcoustID's ``acoustid_compare``
#: does: silence carries no information about which recording it is.
SILENCE_VALUE = 627_964_279

#: The fewest compared sub-fingerprints (silence excluded) an offset needs,
#: whatever ``min_overlap_seconds`` says: 16 items span 2 s of hops, so a
#: query needs at least 4.6 s of audio (Chromaprint's first item needs
#: 2.7 s). Below it the best of thousands of offsets passes the bit error
#: rate threshold by chance: measured with unrelated 2.7 to 3.6 s clips (1
#: to 8 items) against a 6-minute reference, 90 % down to 1 % of them did.
MIN_COMPARED_ITEMS = 16

#: An offset matches only if its bit error rate is below this fraction of its
#: chance error rate (:attr:`AudioMatch.chance_bit_error_rate`): the agreement
#: must be well above what the bits' own statistics give unrelated signals.
#: Both rates are taken over the compared pairs, with a run of identical
#: pairs counted once (:attr:`AudioMatch.chance_test_bit_error_rate`).
#: Measured: genuine matches stay at or below 0.43 of chance (0.29 with 5 s
#: of overlap or more); independent recordings of stationary noise (white,
#: pink, brown, blue, violet, velvet, hiss, a -60 dBFS noise floor) are at
#: 0.82 to 0.94 of it at their best offset, although their bit error rate
#: (0.13 to 0.30) is as low as that of a genuine match, and the ends of two
#: steady drones (hum, sawtooth, organ) at 0.60 to 0.76. 0.6 would leave more
#: room for genuine short clips but let more windows of the suite's
#: similar-structure songs through (``docs/audio.md``).
MAX_CHANCE_ERROR_FRACTION = 0.5

#: The fewest distinct sub-fingerprint values each side of a matching overlap
#: must hold (silence excluded). A steady tone or hum yields one value over
#: and over, which agrees perfectly with itself and says nothing.
MIN_DISTINCT_VALUES = MIN_COMPARED_ITEMS // 2

#: The longest fingerprint accepted (about 34 hours).
MAX_FINGERPRINT_ITEMS = 1_000_000

#: The largest ``min_overlap_seconds`` and ``max_offset_seconds`` that
#: :func:`compare_fingerprints` accepts: the span of the longest fingerprint
#: (:data:`MAX_FINGERPRINT_ITEMS` x :data:`ITEM_SECONDS`, about 34.4 hours).
#: No overlap or offset can be longer, so a larger value could only mean
#: "the whole fingerprint" or "any offset", which this value already says.
MAX_COMPARISON_SECONDS = MAX_FINGERPRINT_ITEMS * ITEM_SECONDS

_MAGIC = b"PKAF"
_HEADER = struct.Struct(">4sBBHdI")  # magic, version, algorithm, reserved, duration, count
_BITS = 32
_MAX_FPCALC_OUTPUT_PER_ITEM = 12  # "4294967295," is 11 characters
_FPCALC_ARGS = (
    "-raw",
    "-json",
    "-format",
    "s16le",
    "-rate",
    str(CHROMAPRINT_SAMPLE_RATE),
    "-channels",
    "1",
    "-length",
    "0",
    "-",
)
# Chromaprint needs 4096 + 19 * 1365 samples for its first sub-fingerprint.
_MIN_SAMPLES = 4096 + 19 * 1365
# The distinct-value check (_diverse) runs on at most this many offsets that
# pass the other tests, best first: it bounds the cost of a long
# low-information overlap. When more pass, they are first narrowed to the
# offsets whose overlap holds MIN_DISTINCT_VALUES distinct values on each side
# at all (_may_be_diverse, linear time), so the cap only matters when more
# than 64 better offsets hold enough values in their overlap but not in their
# compared pairs, which takes silence on one side hiding the other's values.
_MAX_DIVERSITY_CHECKS = 64
# _chance_test counts the pairs directly for at most this many
# offsets and pairs in all; beyond, FFT correlations give every offset at once.
_DIRECT_CHANCE_OFFSETS = 16
_DIRECT_CHANCE_BUDGET = 1 << 22


@dataclass(frozen=True, slots=True)
class AudioFingerprint:
    """A raw Chromaprint fingerprint.

    Attributes:
        values: The 32-bit sub-fingerprints (unsigned), one every
            :data:`ITEM_SECONDS` seconds. Empty for audio shorter than about
            2.7 seconds.
        duration_seconds: The duration of the fingerprinted audio.
        algorithm: The Chromaprint algorithm (only 2 is produced).

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If a value is out of range.
    """

    values: tuple[int, ...]
    duration_seconds: float
    algorithm: int = CHROMAPRINT_ALGORITHM

    def __post_init__(self) -> None:
        """Validates the fields."""
        values = tuple(self.values)
        if len(values) > MAX_FINGERPRINT_ITEMS:
            raise ValueError(f"a fingerprint holds at most {MAX_FINGERPRINT_ITEMS} values")
        for value in values:
            check_int("fingerprint value", value, minimum=0, maximum=2**_BITS - 1)
        object.__setattr__(self, "values", values)
        duration = check_number("duration_seconds", self.duration_seconds, minimum=0.0)
        object.__setattr__(self, "duration_seconds", duration)
        check_int("algorithm", self.algorithm, minimum=0, maximum=255)

    def __len__(self) -> int:
        """Returns the number of sub-fingerprints."""
        return len(self.values)

    def to_bytes(self) -> bytes:
        """Serializes to the compact versioned binary form.

        Layout (big-endian): ``b"PKAF"``, format version (1 byte), algorithm
        (1 byte), two reserved zero bytes, duration in seconds (IEEE 754
        double), count (4 bytes), then the values as unsigned 32-bit integers.

        Returns:
            The serialized fingerprint.
        """
        header = _HEADER.pack(
            _MAGIC, FORMAT_VERSION, self.algorithm, 0, self.duration_seconds, len(self.values)
        )
        return header + np.asarray(self.values, dtype=">u4").tobytes()

    @classmethod
    def from_bytes(cls, data: BytesLike) -> AudioFingerprint:
        """Parses the form written by :meth:`to_bytes`, validating it strictly.

        Args:
            data: The serialized fingerprint.

        Returns:
            The fingerprint.

        Raises:
            TypeError: If ``data`` is not bytes-like.
            ValueError: If ``data`` is not a valid serialized fingerprint.
        """
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("data must be bytes-like")
        raw = bytes(data)
        if len(raw) < _HEADER.size:
            raise ValueError("not an audio fingerprint: too short")
        magic, version, algorithm, reserved, duration, count = _HEADER.unpack_from(raw)
        if magic != _MAGIC:
            raise ValueError("not an audio fingerprint: bad magic")
        if version != FORMAT_VERSION:
            raise ValueError(f"unsupported audio fingerprint format version {version}")
        if reserved != 0 or count > MAX_FINGERPRINT_ITEMS:
            raise ValueError("invalid audio fingerprint header")
        if len(raw) != _HEADER.size + 4 * count:
            raise ValueError("invalid audio fingerprint: wrong length")
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("invalid audio fingerprint: bad duration")
        values = np.frombuffer(raw, dtype=">u4", offset=_HEADER.size, count=count)
        return cls(
            values=tuple(int(v) for v in values), duration_seconds=duration, algorithm=algorithm
        )


@dataclass(frozen=True, slots=True)
class AudioMatch:
    """The best alignment of a query fingerprint against a reference.

    Attributes:
        is_match: Whether some offset passed every test of
            :func:`compare_fingerprints` (then the fields below describe the
            best passing offset; otherwise the best-scoring eligible one).
        bit_error_rate: Fraction of differing bits over the compared pairs
            at that offset (1.0 when no offset could be compared).
        similarity: ``1 - bit_error_rate``.
        offset_seconds: Where the query starts in the reference:
            ``reference_time = query_time + offset_seconds`` (negative when
            the query starts before the reference).
        offset_items: The same offset in sub-fingerprints.
        overlap_items: Sub-fingerprint pairs compared at that offset: the
            overlap without the pairs where either side is silence
            (:data:`SILENCE_VALUE`).
        overlap_seconds: ``overlap_items`` in seconds.
        query_coverage: ``overlap_items / len(query)``.
        chance_bit_error_rate: The bit error rate two unrelated signals with
            the same bit statistics would show at that offset:
            ``mean over bits of p_q (1 - p_r) + p_r (1 - p_q)``, where ``p``
            is how often the bit is 1 on each side of the **chance-test
            pairs**: the compared pairs, with a run of identical pairs
            counted once (a pair whose two codes both repeat the previous
            item's is left out). About 0.42 for music, 0.16 to 0.32 for
            stationary noise, 0.0 for a constant value; 0.0 when nothing was
            compared.
        chance_test_bit_error_rate: The bit error rate over the same
            chance-test pairs, which the chance test compares with
            ``MAX_CHANCE_ERROR_FRACTION * chance_bit_error_rate``. Equal to
            ``bit_error_rate`` unless consecutive pairs repeat (steady
            sounds); two steady drones that end at the same time agree at
            0.018 over all their pairs but 0.18 to 0.29 over these (1.0 when
            nothing was compared).
        bit_error_threshold: The bit error rate threshold that applied at
            that offset: :func:`overlap_threshold` of ``overlap_items``
            (stricter for short overlaps).
    """

    is_match: bool
    bit_error_rate: float
    similarity: float
    offset_seconds: float
    offset_items: int
    overlap_items: int
    overlap_seconds: float
    query_coverage: float
    chance_bit_error_rate: float
    chance_test_bit_error_rate: float
    bit_error_threshold: float


def overlap_threshold(
    compared_items: int, *, max_bit_error_rate: float = DEFAULT_MAX_BIT_ERROR_RATE
) -> float:
    """Returns the bit error rate an overlap of ``compared_items`` must not exceed.

    The compared span is ``compared_items * ITEM_SECONDS``. Up to
    :data:`SHORT_OVERLAP_SECONDS` (5 s) the threshold is
    :data:`SHORT_OVERLAP_FACTOR` x ``max_bit_error_rate`` (0.11 by default);
    from :data:`LONG_OVERLAP_SECONDS` (20 s) on it is ``max_bit_error_rate``
    (0.20); in between it rises linearly (0.155 at 12.5 s). Short windows of
    different signals that sound alike agree by chance far more often than
    long ones, so a short overlap needs a closer agreement.

    Args:
        compared_items: Compared sub-fingerprint pairs (silence excluded).
        max_bit_error_rate: The threshold for long overlaps, in ``[0, 1]``.

    Returns:
        The threshold.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an argument is out of range.
    """
    check_int("compared_items", compared_items, minimum=0)
    check_number("max_bit_error_rate", max_bit_error_rate, minimum=0.0, maximum=1.0)
    return float(_thresholds(np.array([compared_items]), max_bit_error_rate)[0])


def _thresholds(
    compared: npt.NDArray[np.int64], max_bit_error_rate: float
) -> npt.NDArray[np.float64]:
    span = compared * ITEM_SECONDS
    ramp = np.clip(
        (span - SHORT_OVERLAP_SECONDS) / (LONG_OVERLAP_SECONDS - SHORT_OVERLAP_SECONDS), 0.0, 1.0
    )
    factor = SHORT_OVERLAP_FACTOR + (1.0 - SHORT_OVERLAP_FACTOR) * ramp
    result: npt.NDArray[np.float64] = max_bit_error_rate * factor
    return result


def fingerprint_audio(
    data: BytesLike,
    *,
    max_duration_seconds: float | None = None,
    tools: ExternalTools | None = None,
    limits: MediaLimits | None = None,
    temp_dir: StrPath | None = None,
    timeout_seconds: float | None = None,
) -> AudioFingerprint:
    """Fingerprints the first audio stream of untrusted media.

    Args:
        data: The encoded media (untrusted); audio files or video containers.
        max_duration_seconds: Fingerprint at most this much audio from the
            start (truncation, not an error); ``None`` for all of it.
        tools: Where ffmpeg, ffprobe and fpcalc are. Defaults to
            ``ExternalTools.from_env()``.
        limits: The limits to enforce. Defaults to ``MediaLimits()``.
        temp_dir: Directory for a private temporary file, if the container
            needs one.
        timeout_seconds: Timeout of each tool run. Defaults to
            ``limits.subprocess_timeout_seconds``.

    Returns:
        The fingerprint (empty for audio shorter than about 2.7 s).

    Raises:
        TypeError: If ``data`` is not bytes-like.
        ValueError: If ``max_duration_seconds`` is out of range.
        MediaLimitExceededError: If the media exceeds ``limits``.
        UnsupportedMediaError: If the media holds no audio stream.
        MediaDecodeError: If the media cannot be decoded or fpcalc's output is
            unusable.
        BackendUnavailableError: If a binary is missing.
        ExternalToolTimeoutError: If a tool times out.
    """
    limits = MediaLimits() if limits is None else limits
    tools = ExternalTools.from_env() if tools is None else tools
    if max_duration_seconds is not None:
        check_number(
            "max_duration_seconds", max_duration_seconds, minimum=0.0, exclusive_minimum=True
        )
    media_type = _ffmpeg.check_media_input(data, limits, _AUDIO_CONTAINERS)
    timeout = limits.subprocess_timeout_seconds if timeout_seconds is None else timeout_seconds
    with _ffmpeg.staged_input(data, media_type, temp_dir=temp_dir) as staged:
        info = _probe_staged(
            staged, media_type, tools=tools, limits=limits, timeout_seconds=timeout
        )
        args = _pcm_args(
            staged,
            info,
            limits=limits,
            sample_format="s16le",
            sample_rate=CHROMAPRINT_SAMPLE_RATE,
            channels=1,
            max_duration_seconds=max_duration_seconds,
        )
        # The PCM streams into fpcalc and is never held in memory, so only the
        # duration bounds it (max_decoded_audio_bytes is about memory).
        cap = _duration_cap_bytes(
            limits,
            bytes_per_frame=2,
            sample_rate=CHROMAPRINT_SAMPLE_RATE,
            max_duration_seconds=max_duration_seconds,
        )
        decoder: ToolProcess | None = None
        try:
            # ffmpeg's PCM output is piped straight into fpcalc; the output cap
            # on ffmpeg bounds what fpcalc can be fed.
            with tools.stream(
                Tool.FFMPEG, args, stdin=staged.stdin, timeout_seconds=timeout, max_stdout_bytes=cap
            ) as decoder:
                values = _run_fpcalc(tools, decoder, timeout, limits)
        except ExternalToolError as exc:
            if decoder is not None and decoder.output_limit_exceeded:
                raise MediaLimitExceededError(
                    "max_duration_seconds",
                    limits.max_duration_seconds,
                    decoder.bytes_read / (2 * CHROMAPRINT_SAMPLE_RATE),
                ) from exc
            if exc.tool == Tool.FPCALC.value:
                raise
            raise _ffmpeg.as_media_error(exc, "audio decoding") from exc
    decoded = decoder.bytes_read
    samples = decoded // 2
    if max_duration_seconds is None and decoded > cap - 2:
        raise MediaLimitExceededError(
            "max_duration_seconds",
            limits.max_duration_seconds,
            samples / CHROMAPRINT_SAMPLE_RATE,
        )
    return AudioFingerprint(values=values, duration_seconds=samples / CHROMAPRINT_SAMPLE_RATE)


def fingerprint_pcm(
    audio: PcmAudio,
    *,
    tools: ExternalTools | None = None,
    timeout_seconds: float | None = None,
) -> AudioFingerprint:
    """Fingerprints audio that is already decoded.

    The samples are mixed down to mono and converted to 16-bit PCM at their
    own rate; fpcalc resamples to 11,025 Hz itself.

    Args:
        audio: The decoded audio.
        tools: Where fpcalc is. Defaults to ``ExternalTools.from_env()``.
        timeout_seconds: The fpcalc timeout. Defaults to
            ``MediaLimits().subprocess_timeout_seconds``.

    Returns:
        The fingerprint.

    Raises:
        TypeError: If ``audio`` is not a :class:`~provenance_kit.audio.pcm.PcmAudio`.
        MediaDecodeError: If fpcalc's output is unusable.
        BackendUnavailableError: If fpcalc is missing.
        ExternalToolTimeoutError: If fpcalc times out.
    """
    if not isinstance(audio, PcmAudio):
        raise TypeError("audio must be a PcmAudio")
    tools = ExternalTools.from_env() if tools is None else tools
    limits = MediaLimits()
    timeout = limits.subprocess_timeout_seconds if timeout_seconds is None else timeout_seconds
    mono = audio.samples.mean(axis=0)
    pcm = np.clip(np.round(mono * 32767.0), -32768, 32767).astype("<i2").tobytes()
    args = ["-raw", "-json", "-format", "s16le", "-rate", str(audio.sample_rate)]
    args += ["-channels", "1", "-length", "0", "-"]
    min_samples = math.ceil(_MIN_SAMPLES * audio.sample_rate / CHROMAPRINT_SAMPLE_RATE)
    values = _fpcalc_values(tools, args, pcm, timeout, short=audio.frames < min_samples)
    return AudioFingerprint(values=values, duration_seconds=audio.duration_seconds)


def compare_fingerprints(
    query: AudioFingerprint,
    reference: AudioFingerprint,
    *,
    max_bit_error_rate: float = DEFAULT_MAX_BIT_ERROR_RATE,
    min_overlap_seconds: float = DEFAULT_MIN_OVERLAP_SECONDS,
    max_offset_seconds: float | None = None,
) -> AudioMatch:
    """Aligns ``query`` with ``reference`` and decides whether they share audio.

    Every offset is compared over the overlapping sub-fingerprints at once
    (FFT cross-correlations of the 32 bit planes). Pairs where either side is
    digital silence (:data:`SILENCE_VALUE`) are left out of both the bit count
    and the overlap. An offset is **eligible** when at least
    ``min_overlap_seconds`` of pairs are compared (or all the informative
    items of the shorter fingerprint, if it is shorter), and never fewer than
    :data:`MIN_COMPARED_ITEMS`. An eligible offset **passes** when

    1. its bit error rate is at most :func:`overlap_threshold` of its compared
       pairs: ``max_bit_error_rate`` for overlaps of 20 s or more of compared
       span, down to 0.55 x ``max_bit_error_rate`` at 5 s or less (0.11 to
       0.20 by default), because short windows of signals that sound alike
       agree by chance;
    2. its bit error rate is below :data:`MAX_CHANCE_ERROR_FRACTION` times its
       chance error rate (:attr:`AudioMatch.chance_bit_error_rate`), which
       rejects stationary noise, steady tones and other content whose bits
       agree by chance. Both rates are taken over the compared pairs, with a
       run of identical pairs counted once
       (:attr:`AudioMatch.chance_test_bit_error_rate`): the long run of one
       code that two steady drones share would otherwise make the few items
       where they end look related;
    3. each side of the compared pairs holds at least
       :data:`MIN_DISTINCT_VALUES` distinct values.

    The result is a match when **any** offset passes; the reported one is
    the passing offset with the highest binomial z-score,
    ``(0.5 - bit_error_rate) * sqrt(compared_bits)`` (a long low-error
    overlap rather than a short one, which matters for repetitive content).
    When none passes, the eligible offset with the highest z-score is
    reported, with ``is_match=False``. Ties go to the larger overlap, then
    the smaller absolute offset. So a genuine short overlap (an upload that
    starts with the last seconds of the reference) is found even when a long
    unrelated overlap scores a higher z-score.

    Test 3 runs on at most 64 offsets that pass tests 1 and 2, best first,
    which bounds its cost. When more pass, they are first narrowed, in linear
    time, to the offsets whose overlap holds :data:`MIN_DISTINCT_VALUES`
    distinct values on each side (silence aside). An alternation of two
    codes that passes tests 1 and 2 at hundreds of offsets is removed there,
    so "any offset passes" holds unless more than 64 better offsets hold
    enough distinct values in their overlap but not in their compared pairs,
    which takes silence on one side hiding the other side's values.

    Args:
        query: The fingerprint to look up (for example a clip).
        reference: The fingerprint to search in.
        max_bit_error_rate: Threshold on the bit error rate of long
            overlaps, in ``[0, 1]``; shorter overlaps are held to a
            proportionally stricter one (:func:`overlap_threshold`).
        min_overlap_seconds: Minimum compared overlap, in seconds (at least
            :data:`MIN_COMPARED_ITEMS` sub-fingerprints in any case), at
            most :data:`MAX_COMPARISON_SECONDS`.
        max_offset_seconds: If given, only offsets with
            ``abs(offset_seconds) <= max_offset_seconds`` are considered; at
            most :data:`MAX_COMPARISON_SECONDS`.

    Returns:
        The alignment; ``is_match`` is false when nothing passes, including
        when a fingerprint has fewer than :data:`MIN_COMPARED_ITEMS`
        informative sub-fingerprints (audio shorter than about 4.6 s).

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the algorithms differ or a threshold is out of range
            (including durations above :data:`MAX_COMPARISON_SECONDS`).
    """
    for name, value in (("query", query), ("reference", reference)):
        if not isinstance(value, AudioFingerprint):
            raise TypeError(f"{name} must be an AudioFingerprint")
    if query.algorithm != reference.algorithm:
        raise ValueError("the fingerprints use different Chromaprint algorithms")
    check_number("max_bit_error_rate", max_bit_error_rate, minimum=0.0, maximum=1.0)
    min_overlap = check_number(
        "min_overlap_seconds", min_overlap_seconds, minimum=0.0, maximum=MAX_COMPARISON_SECONDS
    )
    max_offset: int | None = None
    if max_offset_seconds is not None:
        seconds = check_number(
            "max_offset_seconds", max_offset_seconds, minimum=0.0, maximum=MAX_COMPARISON_SECONDS
        )
        max_offset = math.floor(seconds / ITEM_SECONDS + 1e-9)

    nothing = AudioMatch(
        is_match=False,
        bit_error_rate=1.0,
        similarity=0.0,
        offset_seconds=0.0,
        offset_items=0,
        overlap_items=0,
        overlap_seconds=0.0,
        query_coverage=0.0,
        chance_bit_error_rate=0.0,
        chance_test_bit_error_rate=1.0,
        bit_error_threshold=float(_thresholds(np.array([0]), max_bit_error_rate)[0]),
    )
    q = np.asarray(query.values, dtype=np.uint32)
    r = np.asarray(reference.values, dtype=np.uint32)
    q_informative = int(np.count_nonzero(q != SILENCE_VALUE))
    r_informative = int(np.count_nonzero(r != SILENCE_VALUE))
    needed = max(
        MIN_COMPARED_ITEMS,
        min(math.ceil(min_overlap / ITEM_SECONDS - 1e-9), q_informative, r_informative),
    )
    if min(q_informative, r_informative) < needed:
        return nothing
    errors, compared = _offset_counts(q, r)
    offsets = np.arange(-(q.size - 1), r.size)
    eligible = compared >= needed
    if max_offset is not None:
        eligible &= np.abs(offsets) <= max_offset
    if not eligible.any():
        return nothing
    candidates = np.flatnonzero(eligible)
    bits = _BITS * compared[candidates]
    rates = errors[candidates] / bits
    # The binomial z-score of the agreement: among offsets that pass, it
    # prefers long overlaps with a low error rate over short overlaps that
    # are low by chance (or by the periodicity of the content).
    scores = (0.5 - rates) * np.sqrt(bits)
    order = np.lexsort((np.abs(offsets[candidates]), -compared[candidates], -scores))
    ranked = candidates[order]
    thresholds = _thresholds(compared[candidates], max_bit_error_rate)
    passing = ranked[(rates <= thresholds)[order]]
    best, matched = int(ranked[0]), False
    if passing.size:
        chances, test_errors, test_pairs = _chance_test(q, r, offsets[passing])
        passing = passing[test_errors < MAX_CHANCE_ERROR_FRACTION * chances * _BITS * test_pairs]
        if passing.size > _MAX_DIVERSITY_CHECKS:
            # A long low-information overlap (an alternation of two codes)
            # can pass the other tests at hundreds of offsets: keep only the
            # offsets whose overlap holds enough distinct values at all.
            passing = passing[_may_be_diverse(q, r, offsets[passing])]
        for index in passing[:_MAX_DIVERSITY_CHECKS]:
            if _diverse(q, r, int(offsets[index])):
                best, matched = int(index), True
                break
    chance, test_error_count, test_pair_count = _chance_test_at(q, r, int(offsets[best]))
    rate = float(errors[best] / (_BITS * compared[best]))
    overlap = int(compared[best])
    offset = int(offsets[best])
    return AudioMatch(
        is_match=matched,
        bit_error_rate=rate,
        similarity=1.0 - rate,
        offset_seconds=offset * ITEM_SECONDS,
        offset_items=offset,
        overlap_items=overlap,
        overlap_seconds=overlap * ITEM_SECONDS,
        query_coverage=overlap / q.size,
        chance_bit_error_rate=chance,
        chance_test_bit_error_rate=(
            test_error_count / (_BITS * test_pair_count) if test_pair_count else 1.0
        ),
        bit_error_threshold=float(_thresholds(np.array([overlap]), max_bit_error_rate)[0]),
    )


# --------------------------------------------------------------------------
# Internals
# --------------------------------------------------------------------------


def _run_fpcalc(
    tools: ExternalTools,
    decoder: ToolProcess,
    timeout: float,
    limits: MediaLimits,
) -> tuple[int, ...]:
    """Runs fpcalc on the decoder's output, piped in (see :class:`ToolProcess`)."""
    max_items = math.ceil(limits.max_duration_seconds / ITEM_SECONDS) + 64
    result = tools.run(
        Tool.FPCALC,
        list(_FPCALC_ARGS),
        stdin=decoder,
        timeout_seconds=timeout,
        max_stdout_bytes=max_items * _MAX_FPCALC_OUTPUT_PER_ITEM + 4096,
        check=False,
    )
    if result.returncode != 0:
        if decoder.bytes_read // 2 < _MIN_SAMPLES:
            return ()  # too short for a single sub-fingerprint
        raise MediaDecodeError(f"fpcalc failed (exit status {result.returncode})")
    return _parse_fpcalc(result.stdout, max_items)


def _fpcalc_values(
    tools: ExternalTools,
    args: Sequence[str],
    pcm: bytes,
    timeout: float,
    *,
    short: bool,
) -> tuple[int, ...]:
    max_items = len(pcm) // 2 + 64
    result = tools.run(
        Tool.FPCALC,
        list(args),
        stdin=pcm,
        timeout_seconds=timeout,
        max_stdout_bytes=max_items * _MAX_FPCALC_OUTPUT_PER_ITEM + 4096,
        check=False,
    )
    if result.returncode != 0:
        if short:
            return ()
        raise MediaDecodeError(f"fpcalc failed (exit status {result.returncode})")
    return _parse_fpcalc(result.stdout, max_items)


def _parse_fpcalc(raw: bytes, max_items: int) -> tuple[int, ...]:
    document = _ffmpeg.load_json_object(raw, "fingerprint")
    values = document.get("fingerprint")
    if not isinstance(values, list) or len(values) > max_items:
        raise MediaDecodeError("fingerprint: fpcalc returned no usable fingerprint")
    parsed: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**_BITS:
            raise MediaDecodeError("fingerprint: fpcalc returned an invalid value")
        parsed.append(value)
    return tuple(parsed)


def _offset_counts(
    q: npt.NDArray[np.uint32], r: npt.NDArray[np.uint32]
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Bit errors and compared pairs for every offset, via FFT correlation.

    Offset ``d`` (index ``d + len(q) - 1``) pairs query item ``i`` with
    reference item ``i + d``. Each bit plane is mapped to ``±1``, and to 0
    where the item is silence, so that pairs involving silence add nothing;
    the cross-correlation of two planes is (agreements - disagreements) at
    every offset, and the 32 planes are summed in the frequency domain. The
    correlation of the two non-silence masks counts the compared pairs.
    """
    n_q, n_r = q.size, r.size
    q_ok = (q != SILENCE_VALUE).astype(np.float64)
    r_ok = (r != SILENCE_VALUE).astype(np.float64)
    size = _fft_size(n_q, n_r)
    spectrum = np.zeros(size // 2 + 1, dtype=np.complex128)
    for bit in range(_BITS):
        q_plane = (((q >> bit) & 1).astype(np.float64) * 2.0 - 1.0) * q_ok
        r_plane = (((r >> bit) & 1).astype(np.float64) * 2.0 - 1.0) * r_ok
        spectrum += np.conj(np.fft.rfft(q_plane, size)) * np.fft.rfft(r_plane, size)
    pairs = np.conj(np.fft.rfft(q_ok, size)) * np.fft.rfft(r_ok, size)
    compared = np.clip(np.rint(_by_offset(pairs, n_q, n_r, size)), 0, None).astype(np.int64)
    errors = np.rint((_BITS * compared - _by_offset(spectrum, n_q, n_r, size)) / 2.0)
    return np.clip(errors.astype(np.int64), 0, _BITS * compared), compared


def _fft_size(n_q: int, n_r: int) -> int:
    """A power of two that holds every offset's correlation without wrapping."""
    return 1 << (n_q + n_r - 1).bit_length()


def _by_offset(
    spectrum: npt.NDArray[np.complex128], n_q: int, n_r: int, size: int
) -> npt.NDArray[np.float64]:
    """The correlation of a cross spectrum, indexed by offset (``d + n_q - 1``)."""
    correlation = np.fft.irfft(spectrum, size)
    # correlation[k] = sum_i q[i] r[i + k]; negative k wrap around to the end.
    return np.concatenate((correlation[size - (n_q - 1) :], correlation[:n_r]))


def _chance_test(
    q: npt.NDArray[np.uint32], r: npt.NDArray[np.uint32], offsets: npt.NDArray[np.int64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """The chance test's numbers at each offset: chance error rate, bit errors, pairs.

    All three are taken over the **chance-test pairs**: the compared pairs
    (neither side silence), with a run of identical pairs counted once (a
    pair whose two codes both repeat the previous item's is left out). The
    chance error rate is the mean over the 32 bits of ``p_q (1 - p_r) + p_r
    (1 - p_q)``, where ``p_q`` and ``p_r`` are how often the bit is 1 on each
    side of those pairs.

    Both choices keep content that carries no information from looking
    related: taking each side's own non-silent items instead of the compared
    pairs let a query that ends in silence borrow the bit statistics of the
    reference's music facing that silence (round-2 finding M01), and the 180
    or so identical pairs of two steady drones made the 20 items where both
    drones end look far better than chance.

    A few short overlaps are counted directly; otherwise every offset is
    computed at once with FFT correlations (:func:`_chance_test_all`).
    """
    n_q, n_r = q.size, r.size
    spans = np.minimum(n_q, n_r - offsets) - np.maximum(0, -offsets)
    if offsets.size <= _DIRECT_CHANCE_OFFSETS and int(spans.sum()) <= _DIRECT_CHANCE_BUDGET:
        rows = [_chance_test_at(q, r, int(d)) for d in offsets]
        return (
            np.array([row[0] for row in rows], dtype=np.float64),
            np.array([row[1] for row in rows], dtype=np.int64),
            np.array([row[2] for row in rows], dtype=np.int64),
        )
    chance, errors, pairs = _chance_test_all(q, r)
    index = offsets + (n_q - 1)
    return chance[index], errors[index], pairs[index]


def _chance_test_at(
    q: npt.NDArray[np.uint32], r: npt.NDArray[np.uint32], offset: int
) -> tuple[float, int, int]:
    """:func:`_chance_test` at one offset, counted directly."""
    start = max(0, -offset)
    stop = min(q.size, r.size - offset)
    q_part = q[start:stop]
    r_part = r[start + offset : stop + offset]
    kept = (q_part != SILENCE_VALUE) & (r_part != SILENCE_VALUE)
    kept[1:] &= (q_part[1:] != q_part[:-1]) | (r_part[1:] != r_part[:-1])
    q_part, r_part = q_part[kept], r_part[kept]
    pairs = int(q_part.size)
    if not pairs:
        return 0.0, 0, 0
    total = 0.0
    errors = 0
    differ = q_part ^ r_part
    for bit in range(_BITS):
        p_q = int(np.count_nonzero((q_part >> bit) & 1)) / pairs
        p_r = int(np.count_nonzero((r_part >> bit) & 1)) / pairs
        total += p_q * (1.0 - p_r) + p_r * (1.0 - p_q)
        errors += int(np.count_nonzero((differ >> bit) & 1))
    return total / _BITS, errors, pairs


def _chance_test_all(
    q: npt.NDArray[np.uint32], r: npt.NDArray[np.uint32]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """:func:`_chance_test` at every offset (indexed like :func:`_offset_counts`).

    The weight of the pair ``(i, i + d)`` is ``ok_q[i] ok_r[i + d] - rep_q[i]
    rep_r[i + d]``, where ``ok`` marks non-silent items and ``rep`` the
    non-silent items equal to the item before them (a repeated pair is then
    compared and weighs 0). Each weighted sum over the pairs is the
    difference of two correlations: the pairs, a bit's ones on each side, and
    the pairs where both sides hold a 1 (which give the errors). Without
    repeats on one side, no pair repeats and those correlations are skipped.
    """
    n_q, n_r = q.size, r.size
    size = _fft_size(n_q, n_r)
    q_ok = q != SILENCE_VALUE
    r_ok = r != SILENCE_VALUE
    q_rep = np.zeros(n_q, dtype=np.bool_)
    q_rep[1:] = (q[1:] == q[:-1]) & q_ok[1:]
    r_rep = np.zeros(n_r, dtype=np.bool_)
    r_rep[1:] = (r[1:] == r[:-1]) & r_ok[1:]
    repeats = bool(q_rep.any() and r_rep.any())

    def spectrum(values: npt.NDArray[np.bool_], *, repeated: bool = False) -> Any:
        if repeated and not repeats:
            return 0.0
        return np.fft.rfft(values.astype(np.float64), size)

    q_ok_f, q_rep_f = np.conj(spectrum(q_ok)), np.conj(spectrum(q_rep, repeated=True))
    r_ok_f, r_rep_f = spectrum(r_ok), spectrum(r_rep, repeated=True)

    def weighted(q_ok_part: Any, q_rep_part: Any, r_ok_part: Any, r_rep_part: Any) -> Any:
        correlation = _by_offset(q_ok_part * r_ok_part - q_rep_part * r_rep_part, n_q, n_r, size)
        return np.clip(np.rint(correlation), 0, None)

    pairs = weighted(q_ok_f, q_rep_f, r_ok_f, r_rep_f)
    total = np.zeros(n_q + n_r - 1, dtype=np.float64)
    ones = np.zeros(n_q + n_r - 1, dtype=np.float64)
    both = np.zeros(size // 2 + 1, dtype=np.complex128)
    for bit in range(_BITS):
        q_bit = ((q >> bit) & 1).astype(np.bool_)
        r_bit = ((r >> bit) & 1).astype(np.bool_)
        q_ones_ok = np.conj(spectrum(q_bit & q_ok))
        q_ones_rep = np.conj(spectrum(q_bit & q_rep, repeated=True))
        r_ones_ok = spectrum(r_bit & r_ok)
        r_ones_rep = spectrum(r_bit & r_rep, repeated=True)
        a = weighted(q_ones_ok, q_ones_rep, r_ok_f, r_rep_f)  # query ones over the pairs
        b = weighted(q_ok_f, q_rep_f, r_ones_ok, r_ones_rep)  # reference ones
        both += q_ones_ok * r_ones_ok - q_ones_rep * r_ones_rep
        # p_q (1 - p_r) + p_r (1 - p_q), times pairs**2.
        total += (a + b) * pairs - 2.0 * a * b
        ones += a + b
    agree = np.clip(np.rint(_by_offset(both, n_q, n_r, size)), 0, None)
    errors = np.clip(ones - 2.0 * agree, 0, None).astype(np.int64)
    squared = np.maximum(pairs, 1.0) ** 2
    chance: npt.NDArray[np.float64] = total / (squared * _BITS)
    return chance, errors, pairs.astype(np.int64)


def _may_be_diverse(
    q: npt.NDArray[np.uint32], r: npt.NDArray[np.uint32], offsets: npt.NDArray[np.int64]
) -> npt.NDArray[np.bool_]:
    """Whether each side's overlap holds :data:`MIN_DISTINCT_VALUES` distinct values.

    A filter before :func:`_diverse`: it counts each side's non-silent items
    over the whole overlap, so it can only overcount the distinct values of
    the compared pairs (where silence on the other side removes pairs), never
    reject an offset that :func:`_diverse` accepts.
    """
    q0 = np.maximum(0, -offsets)
    q1 = np.minimum(q.size, r.size - offsets)
    return _holds_distinct(q, q0, q1) & _holds_distinct(r, q0 + offsets, q1 + offsets)


def _holds_distinct(
    values: npt.NDArray[np.uint32], starts: npt.NDArray[np.int64], stops: npt.NDArray[np.int64]
) -> npt.NDArray[np.bool_]:
    """Whether ``values[start:stop]`` holds :data:`MIN_DISTINCT_VALUES` non-silent values.

    Linear in the number of items: consecutive equal values (and silence) are
    collapsed into runs, and one sweep over the runs finds, for each run, how
    far a range starting there must reach to hold enough distinct values.
    """
    result = np.zeros(starts.size, dtype=np.bool_)
    informative = values != SILENCE_VALUE
    kept = values[informative]
    if not kept.size:
        return result
    new_run = np.ones(kept.size, dtype=np.bool_)
    new_run[1:] = kept[1:] != kept[:-1]
    run_of = np.cumsum(new_run) - 1
    reach = _distinct_reach(kept[new_run].tolist(), MIN_DISTINCT_VALUES)
    # Non-silent items before each index: the kept items of values[start:stop]
    # are kept[before[start]:before[stop]].
    before = np.concatenate(([0], np.cumsum(informative)))
    first, last = before[starts], before[stops]
    some = last > first
    result[some] = run_of[last[some] - 1] + 1 >= reach[run_of[first[some]]]
    return result


def _distinct_reach(runs: list[int], count: int) -> npt.NDArray[np.int64]:
    """For each start, the smallest end with ``count`` distinct values in ``runs[start:end]``.

    ``len(runs) + 1`` where no end does. A two-pointer sweep: the end only
    moves forward.
    """
    size = len(runs)
    reach = np.full(size, size + 1, dtype=np.int64)
    seen: dict[int, int] = {}
    end = 0
    for start, value in enumerate(runs):
        while len(seen) < count and end < size:
            added = runs[end]
            seen[added] = seen.get(added, 0) + 1
            end += 1
        if len(seen) < count:
            break  # later starts see a subset of what is left
        reach[start] = end
        left = seen[value] - 1
        if left:
            seen[value] = left
        else:
            del seen[value]
    return reach


def _diverse(q: npt.NDArray[np.uint32], r: npt.NDArray[np.uint32], offset: int) -> bool:
    """Whether both sides of the compared pairs hold enough distinct values."""
    start = max(0, -offset)
    stop = min(q.size, r.size - offset)
    q_part = q[start:stop]
    r_part = r[start + offset : stop + offset]
    compared = (q_part != SILENCE_VALUE) & (r_part != SILENCE_VALUE)
    return (
        np.unique(q_part[compared]).size >= MIN_DISTINCT_VALUES
        and np.unique(r_part[compared]).size >= MIN_DISTINCT_VALUES
    )
