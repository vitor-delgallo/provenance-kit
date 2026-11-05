"""Video fingerprints: perceptual frame hashes at a fixed sampling rate.

A :class:`VideoFingerprint` is the sequence of PDQ hashes (256 bits each, by
default) of frames sampled at a fixed rate (``fps``, 1 per second by default),
plus each hash's quality score. Frame ``k`` was sampled ``k / fps`` seconds
after the start of the video. :meth:`VideoFingerprint.to_bytes` gives a
compact, versioned binary form for a database column (33 bytes per sampled
frame plus a 20-byte header: about 116 KiB for an hour at 1 frame per
second); :meth:`VideoFingerprint.from_bytes` validates it strictly.

:func:`match_fingerprints` aligns a query against a reference and looks for a
**matched segment**: a stretch of time over which the aligned frames match.
For every temporal offset it walks the aligned frame pairs in order; a pair
matches when its Hamming distance is at most ``max_distance``, and a segment
is a run of matched pairs that may skip at most ``max_gap_seconds`` of
unmatched pairs at a time. The decision depends on the segment only, never on
the length of either video:

* **duration**: the segment spans at least ``min_matched_seconds`` (5 s);
* **distinct frames**: at least ``min_distinct_frames`` (3) of its matched
  frames are distinct pictures. Consecutive near-identical samples (a still
  image, an end card, a frozen frame: each within ``static_distance`` bits of
  the first sample of its run) count as one, on both sides, so a static
  card shared by two unrelated videos is one matched frame, not three;
* **density**: at least ``min_match_fraction`` (half) of the segment's
  compared pairs match.

So a 15-second excerpt is found whatever surrounds it in either video (an
intro, an outro, a compilation), and a few seconds of shared boilerplate are
not a match. The result reports the segment's start and end in both videos.
The query may be a clip cut from anywhere in the reference (or extend past
either end of it).

Robustness (measured in ``tests/video/test_fingerprint_integration.py``, see
``docs/video.md`` for the numbers):

* **Re-encoding** (other codec, lower bit rate) and **resizing** (any size:
  frames are scaled to a fixed square before hashing) keep the per-frame PDQ
  distance far below the default threshold.
* **Frame-rate changes** are handled by construction: sampling is by time,
  not by frame number, so a 25 fps and a 30 fps encode of the same video are
  sampled at the same instants (to within one source frame).
* **Trimming** is handled by the offset search. A clip whose cut points fall
  between two sampling instants is sampled up to ``0.5 / fps`` seconds away
  from the reference's instants; fast motion then raises the distances, which
  is why the default threshold has room to spare. Sampling more densely
  (``fps=2``) reduces the effect.
* **Not handled:** speed changes (a clip played at another speed matches only
  over short stretches), mirroring, rotation by the uploader, cropping,
  letterboxing, picture-in-picture and heavy overlays. PDQ is not designed to
  survive them.

Frame hashing goes through the :class:`FrameHasher` Protocol. The default
implementation is the PDQ hasher of :mod:`provenance_kit.image.hashing`
(the core install's NumPy port, or ``pdqhash`` with extra ``pdq-native``),
loaded lazily; tests and applications can pass any object
with a ``hash_image(PIL.Image.Image)`` method returning ``value`` (bytes) and
``quality`` (0 to 100).

Thread safety: the functions and the immutable fingerprints are thread-safe,
provided the hasher is (the default PDQ hasher is).
"""

from __future__ import annotations

import dataclasses
import importlib
import math
import struct
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
import numpy.typing as npt

from provenance_kit._checks import check_int, check_number
from provenance_kit._logging import get_logger
from provenance_kit._typing import BytesLike, StrPath
from provenance_kit.errors import BackendUnavailableError
from provenance_kit.limits import MediaLimits
from provenance_kit.tools import ExternalTools
from provenance_kit.video.frames import MAX_SAMPLING_FPS, FrameSampling, VideoFrame, iter_frames

if TYPE_CHECKING:
    from PIL import Image

__all__ = [
    "DEFAULT_FRAME_SIZE",
    "DEFAULT_MAX_DISTANCE",
    "DEFAULT_MAX_GAP_SECONDS",
    "DEFAULT_MIN_DISTINCT_FRAMES",
    "DEFAULT_MIN_MATCHED_SECONDS",
    "DEFAULT_MIN_MATCH_FRACTION",
    "DEFAULT_MIN_QUALITY",
    "DEFAULT_STATIC_DISTANCE",
    "FORMAT_VERSION",
    "MAX_FINGERPRINT_FRAMES",
    "FrameHash",
    "FrameHasher",
    "VideoFingerprint",
    "VideoMatch",
    "default_frame_hasher",
    "fingerprint_frames",
    "fingerprint_video",
    "match_fingerprints",
]

_log = get_logger(__name__)

#: Version of the binary layout written by :meth:`VideoFingerprint.to_bytes`.
FORMAT_VERSION = 1

#: Frames are scaled to this square size (pixels) before hashing.
DEFAULT_FRAME_SIZE = 256

#: Default per-frame threshold: aligned frames whose PDQ hashes differ in at
#: most this many of 256 bits count as matched. It is above the 31 bits Meta
#: recommends for PDQ image matching because aligned video samples also differ
#: by the sampling-phase offset of a trimmed clip (measured: median 6 to 10
#: bits after re-encoding, scaling or a frame-rate change, with one sample of
#: 40 at 44 bits after halving the frame rate; median 24, maximum 34 for a
#: clip cut 0.4 s off the sampling grid of fast-moving content), while
#: unrelated frames measure about 128 (minimum 102 in the same test). A match
#: also needs a matched segment (see the thresholds below).
DEFAULT_MAX_DISTANCE = 40

#: Frames whose hash quality is below this are ignored when matching (PDQ's
#: guidance: lower-quality hashes come from nearly featureless images such as
#: black or flat frames, which match each other spuriously).
DEFAULT_MIN_QUALITY = 50

#: A match needs a segment with at least this many distinct matched frames
#: (static runs counted once) ...
DEFAULT_MIN_DISTINCT_FRAMES = 3

#: ... spanning at least this many seconds (first to last matched sample,
#: inclusive: 5 samples at 1 fps) ...
DEFAULT_MIN_MATCHED_SECONDS = 5.0

#: ... in which at least this fraction of the compared pairs matched.
DEFAULT_MIN_MATCH_FRACTION = 0.5

#: A segment survives at most this many seconds of consecutive unmatched
#: pairs (an occasional noisy sample, a scene cut sampled off the grid).
DEFAULT_MAX_GAP_SECONDS = 2.0

#: Consecutive samples within this many bits of the first sample of their run
#: are one static picture (measured within one encode: 0 to 2 bits for a still
#: end card, 4 to 14 for a still scene with heavy grain; moving content 40 or
#: more between samples one second apart).
DEFAULT_STATIC_DISTANCE = 16

#: The longest fingerprint :meth:`VideoFingerprint.from_bytes` accepts.
MAX_FINGERPRINT_FRAMES = 1_000_000

_MAGIC = b"PKVF"
_HEADER = struct.Struct(">4sBBHdI")  # magic, version, hash bytes, reserved, fps, count
_MAX_HASH_BYTES = 64
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)
_EPSILON = 1e-9


class FrameHash(Protocol):
    """A perceptual hash of one frame, as returned by a :class:`FrameHasher`."""

    @property
    def value(self) -> bytes:
        """The hash bits, big-endian (most significant bit first)."""
        ...

    @property
    def quality(self) -> int:
        """The hasher's quality score, 0 (featureless) to 100."""
        ...


class FrameHasher(Protocol):
    """What the fingerprint needs from an image hasher (PDQ by default)."""

    def hash_image(self, image: Image.Image) -> FrameHash:
        """Hashes one RGB image.

        Args:
            image: The frame as a Pillow RGB image.

        Returns:
            The hash and its quality.
        """
        ...


def default_frame_hasher() -> FrameHasher:
    """Returns the default frame hasher: PDQ from :mod:`provenance_kit.image.hashing`.

    Returns:
        A new ``PdqHasher`` (``backend="auto"``: ``pdqhash`` when installed,
        else the NumPy reference port).

    Raises:
        BackendUnavailableError: If the image hashing module or its PDQ
            hasher cannot be loaded.
    """
    try:
        module = importlib.import_module("provenance_kit.image.hashing")
    except ImportError as exc:
        raise BackendUnavailableError.missing_extra(
            "Video fingerprints",
            "video",
            reason="provenance_kit.image.hashing could not be imported",
        ) from exc
    hasher_class = getattr(module, "PdqHasher", None)
    if hasher_class is None:
        raise BackendUnavailableError.missing_extra(
            "Video fingerprints", "video", reason="the PDQ hasher is not available"
        )
    return cast(FrameHasher, hasher_class())


@dataclass(frozen=True, slots=True)
class VideoFingerprint:
    """Perceptual hashes of frames sampled at a fixed rate.

    Attributes:
        fps: The sampling rate in frames per second; frame ``k`` was sampled
            at ``k / fps`` seconds.
        hash_bits: The width of each hash in bits (256 for PDQ), a multiple
            of 8.
        hashes: The hashes as non-negative integers below ``2**hash_bits``.
        qualities: The hashes' quality scores (0 to 100), one per hash.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If a field is out of range or the sequences differ in
            length.
    """

    fps: float
    hash_bits: int
    hashes: tuple[int, ...]
    qualities: tuple[int, ...]

    def __post_init__(self) -> None:
        """Validates the fields."""
        if isinstance(self.fps, bool) or not isinstance(self.fps, (int, float)):
            raise TypeError("fps must be a number")
        if not math.isfinite(self.fps) or not 0 < self.fps <= MAX_SAMPLING_FPS:
            raise ValueError(f"fps must be positive and at most {MAX_SAMPLING_FPS:g}")
        object.__setattr__(self, "fps", float(self.fps))
        if isinstance(self.hash_bits, bool) or not isinstance(self.hash_bits, int):
            raise TypeError("hash_bits must be an int")
        if not 8 <= self.hash_bits <= 8 * _MAX_HASH_BYTES or self.hash_bits % 8:
            raise ValueError(f"hash_bits must be a multiple of 8 from 8 to {8 * _MAX_HASH_BYTES}")
        hashes, qualities = tuple(self.hashes), tuple(self.qualities)
        if len(hashes) != len(qualities):
            raise ValueError("hashes and qualities must have the same length")
        if len(hashes) > MAX_FINGERPRINT_FRAMES:
            raise ValueError(f"a fingerprint holds at most {MAX_FINGERPRINT_FRAMES} frames")
        for value in hashes:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError("hashes must be ints")
            if value < 0 or value >> self.hash_bits:
                raise ValueError(f"a hash does not fit in {self.hash_bits} bits")
        for quality in qualities:
            if isinstance(quality, bool) or not isinstance(quality, int):
                raise TypeError("qualities must be ints")
            if not 0 <= quality <= 100:
                raise ValueError("qualities must be between 0 and 100")
        object.__setattr__(self, "hashes", hashes)
        object.__setattr__(self, "qualities", qualities)

    def __len__(self) -> int:
        """Returns the number of sampled frames."""
        return len(self.hashes)

    @property
    def duration_seconds(self) -> float:
        """The time span covered: ``len(self) / fps``."""
        return len(self.hashes) / self.fps

    @property
    def timestamps(self) -> tuple[float, ...]:
        """The sampling instant of each hash, in seconds."""
        return tuple(k / self.fps for k in range(len(self.hashes)))

    def hex_hashes(self) -> list[str]:
        """Returns the hashes as zero-padded lowercase hex strings."""
        digits = self.hash_bits // 4
        return [format(value, f"0{digits}x") for value in self.hashes]

    def to_bytes(self) -> bytes:
        """Serializes to the compact versioned binary form.

        Layout (big-endian): ``b"PKVF"``, format version (1 byte), hash size
        in bytes (1 byte), two reserved zero bytes, ``fps`` (IEEE 754 double),
        frame count (4 bytes), then every hash, then every quality byte.

        Returns:
            The serialized fingerprint.
        """
        hash_bytes = self.hash_bits // 8
        header = _HEADER.pack(_MAGIC, FORMAT_VERSION, hash_bytes, 0, self.fps, len(self.hashes))
        body = b"".join(value.to_bytes(hash_bytes, "big") for value in self.hashes)
        return header + body + bytes(self.qualities)

    @classmethod
    def from_bytes(cls, data: BytesLike) -> VideoFingerprint:
        """Parses the form written by :meth:`to_bytes`.

        The input is validated strictly (magic, version, sizes, ranges), so a
        corrupted or foreign value raises instead of producing a fingerprint.

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
            raise ValueError("not a video fingerprint: too short")
        magic, version, hash_bytes, reserved, fps, count = _HEADER.unpack_from(raw)
        if magic != _MAGIC:
            raise ValueError("not a video fingerprint: bad magic")
        if version != FORMAT_VERSION:
            raise ValueError(f"unsupported video fingerprint format version {version}")
        if not 1 <= hash_bytes <= _MAX_HASH_BYTES or reserved != 0:
            raise ValueError("invalid video fingerprint header")
        if count > MAX_FINGERPRINT_FRAMES:
            raise ValueError("invalid video fingerprint: too many frames")
        expected = _HEADER.size + count * (hash_bytes + 1)
        if len(raw) != expected:
            raise ValueError("invalid video fingerprint: wrong length")
        body_end = _HEADER.size + count * hash_bytes
        hashes = tuple(
            int.from_bytes(raw[offset : offset + hash_bytes], "big")
            for offset in range(_HEADER.size, body_end, hash_bytes)
        )
        return cls(
            fps=fps, hash_bits=8 * hash_bytes, hashes=hashes, qualities=tuple(raw[body_end:])
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class VideoMatch:
    """The best matched segment of a query fingerprint against a reference.

    The counts, the similarity and the times describe the **segment** (see
    :func:`match_fingerprints`): the best stretch of matched frames at the
    reported offset. When no aligned pair matches at any offset, there is no
    segment: the offset is then the one whose aligned frames are closest on
    average, ``matched_frames``, ``distinct_frames`` and the times are 0, and
    ``compared_frames`` and ``similarity`` describe the whole overlap there.

    Attributes:
        is_match: The decision: the segment has at least
            ``min_distinct_frames`` distinct matched frames, spans at least
            ``min_matched_seconds`` and has a ``match_fraction`` of at least
            ``min_match_fraction``.
        similarity: Mean of ``1 - distance / hash_bits`` over the compared
            pairs (1.0 for identical frames, about 0.5 for unrelated ones);
            0.0 when nothing could be compared.
        offset_seconds: Where the query starts in the reference:
            ``reference_time = query_time + offset_seconds``. Negative when
            the query starts before the reference.
        offset_frames: The same offset in sampled frames.
        matched_frames: Matched pairs in the segment.
        distinct_frames: Matched frames in the segment after counting each
            run of near-identical samples once, on each side; the smaller of
            the two counts.
        compared_frames: Pairs in the segment (from its first to its last
            matched pair) where both hashes had usable quality.
        match_fraction: ``matched_frames / compared_frames`` (0.0 if none).
        matched_seconds: The segment's duration, from its first to its last
            matched sample inclusive: ``(last - first + 1) / fps``.
        query_start_seconds: Where the segment starts in the query.
        query_end_seconds: Where it ends in the query (exclusive: the last
            matched sample plus one sampling interval).
        reference_start_seconds: Where the segment starts in the reference.
        reference_end_seconds: Where it ends in the reference (exclusive).
        query_coverage: ``compared_frames`` over the query's usable frames:
            how much of the query the segment covers.
    """

    is_match: bool
    similarity: float
    offset_seconds: float
    offset_frames: int
    matched_frames: int
    distinct_frames: int
    compared_frames: int
    match_fraction: float
    matched_seconds: float
    query_start_seconds: float
    query_end_seconds: float
    reference_start_seconds: float
    reference_end_seconds: float
    query_coverage: float


def fingerprint_frames(
    frames: Iterable[VideoFrame | npt.NDArray[np.uint8]],
    *,
    fps: float,
    hasher: FrameHasher | None = None,
    frame_size: int = DEFAULT_FRAME_SIZE,
) -> VideoFingerprint:
    """Fingerprints frames that were already decoded at a fixed rate.

    Args:
        frames: Frames sampled at ``fps`` from the start of the video, as
            :class:`~provenance_kit.video.frames.VideoFrame` objects or RGB
            ``uint8`` arrays shaped ``(height, width, 3)``.
        fps: The rate at which the frames were sampled.
        hasher: The frame hasher. Defaults to :func:`default_frame_hasher`.
        frame_size: Frames are resized to ``frame_size`` x ``frame_size``
            before hashing (only when they have another size).

    Returns:
        The fingerprint.

    Raises:
        TypeError: If a frame or argument has the wrong type.
        ValueError: If a frame has the wrong shape, or the hasher returns
            hashes of inconsistent size or an invalid quality.
        BackendUnavailableError: If the default hasher is not available.
    """
    from PIL import Image  # noqa: PLC0415 - Pillow is a core dependency

    check_int("frame_size", frame_size, minimum=8, maximum=4096)
    hasher = default_frame_hasher() if hasher is None else hasher
    hashes: list[int] = []
    qualities: list[int] = []
    hash_bytes: int | None = None
    for item in frames:
        array = item.image if isinstance(item, VideoFrame) else item
        _check_frame(array)
        image = Image.fromarray(np.ascontiguousarray(array))
        if image.size != (frame_size, frame_size):
            image = image.resize((frame_size, frame_size), Image.Resampling.BICUBIC)
        result = hasher.hash_image(image)
        value = bytes(result.value)
        if hash_bytes is None:
            if not 1 <= len(value) <= _MAX_HASH_BYTES:
                raise ValueError("the frame hasher returned a hash of unsupported size")
            hash_bytes = len(value)
        elif len(value) != hash_bytes:
            raise ValueError("the frame hasher returned hashes of different sizes")
        quality = _hash_quality(result.quality)
        hashes.append(int.from_bytes(value, "big"))
        qualities.append(quality)
    bits = 256 if hash_bytes is None else 8 * hash_bytes
    return VideoFingerprint(
        fps=fps, hash_bits=bits, hashes=tuple(hashes), qualities=tuple(qualities)
    )


def fingerprint_video(
    data: BytesLike,
    *,
    fps: float = 1.0,
    hasher: FrameHasher | None = None,
    frame_size: int = DEFAULT_FRAME_SIZE,
    tools: ExternalTools | None = None,
    limits: MediaLimits | None = None,
    temp_dir: StrPath | None = None,
    timeout_seconds: float | None = None,
) -> VideoFingerprint:
    """Decodes untrusted video at ``fps`` frames per second and fingerprints it.

    Frames are decoded by :func:`~provenance_kit.video.frames.iter_frames`
    (which enforces ``limits``) directly at ``frame_size`` x ``frame_size``,
    one at a time.

    Args:
        data: The encoded video (untrusted).
        fps: The sampling rate. Compare only fingerprints with the same rate.
        hasher: The frame hasher. Defaults to :func:`default_frame_hasher`.
        frame_size: The square size frames are scaled to before hashing.
        tools: Where the binaries are. Defaults to ``ExternalTools.from_env()``.
        limits: The limits to enforce. Defaults to ``MediaLimits()``.
        temp_dir: Directory for a private temporary file, if the container
            needs one.
        timeout_seconds: Timeout of each tool run.

    Returns:
        The fingerprint.

    Raises:
        TypeError: See :func:`~provenance_kit.video.frames.iter_frames`.
        ValueError: See :func:`~provenance_kit.video.frames.iter_frames`.
        MediaLimitExceededError: If the media exceeds ``limits``.
        UnsupportedMediaError: If the media is not a supported video.
        MediaDecodeError: If the media cannot be decoded.
        BackendUnavailableError: If a binary or the default hasher is missing.
        ExternalToolTimeoutError: If a tool times out.
    """
    check_int("frame_size", frame_size, minimum=8, maximum=4096)
    hasher = default_frame_hasher() if hasher is None else hasher
    frames = iter_frames(
        data,
        sampling=FrameSampling.FPS,
        fps=fps,
        size=(frame_size, frame_size),
        tools=tools,
        limits=limits,
        temp_dir=temp_dir,
        timeout_seconds=timeout_seconds,
    )
    return fingerprint_frames(frames, fps=fps, hasher=hasher, frame_size=frame_size)


def match_fingerprints(
    query: VideoFingerprint,
    reference: VideoFingerprint,
    *,
    max_distance: int = DEFAULT_MAX_DISTANCE,
    min_quality: int = DEFAULT_MIN_QUALITY,
    min_distinct_frames: int = DEFAULT_MIN_DISTINCT_FRAMES,
    min_matched_seconds: float = DEFAULT_MIN_MATCHED_SECONDS,
    min_match_fraction: float = DEFAULT_MIN_MATCH_FRACTION,
    max_gap_seconds: float = DEFAULT_MAX_GAP_SECONDS,
    static_distance: int = DEFAULT_STATIC_DISTANCE,
    max_offset_seconds: float | None = None,
) -> VideoMatch:
    """Finds the best matched segment of ``query`` in ``reference``.

    For every temporal offset (``reference_time = query_time + offset``), the
    aligned pairs of usable frames (quality at least ``min_quality`` on both
    sides) are walked in time order. A pair matches when its Hamming distance
    is at most ``max_distance``. A **segment** starts at a matched pair and
    extends over the following pairs until more than ``max_gap_seconds`` of
    consecutive pairs fail to match; it ends at its last matched pair. Each
    segment is described by:

    * its matched pairs and its compared pairs (from its first to its last
      matched pair), whose ratio is its ``match_fraction``;
    * its duration, from the first to the last matched sample inclusive;
    * its distinct frames: each fingerprint is split into runs of
      near-identical consecutive samples (each within ``static_distance``
      bits of the run's first sample; low-quality samples are skipped), and
      the segment's matched pairs are counted once per run on each side; the
      smaller count is kept. A still end card is one distinct frame however
      many seconds it lasts.

    A segment **qualifies** when it has at least ``min_distinct_frames``
    distinct frames, lasts at least ``min_matched_seconds`` and has a
    ``match_fraction`` of at least ``min_match_fraction``. None of this
    depends on the length of either fingerprint, so an excerpt is found
    whatever surrounds it. The result is the best segment over all offsets,
    ranked by: qualifying first, then more distinct frames, more matched
    pairs, a higher match fraction, a lower mean distance, a smaller absolute
    offset; on one offset, the earliest of equal segments.

    Cost: one Hamming distance per pair of frames (``len(query) *
    len(reference)``), computed one row at a time (memory proportional to
    the longer fingerprint); ``max_offset_seconds`` restricts the offsets
    reported.

    Args:
        query: The fingerprint to look up (for example a clip).
        reference: The fingerprint to search in.
        max_distance: The per-frame Hamming threshold, 0 to ``hash_bits``.
        min_quality: Frames with a lower hash quality are ignored.
        min_distinct_frames: Distinct matched frames required for
            ``is_match`` (at least 1).
        min_matched_seconds: Segment duration required for ``is_match``.
        min_match_fraction: Fraction of the segment's compared pairs that
            must match, 0 to 1.
        max_gap_seconds: The longest run of unmatched pairs a segment may
            skip, in seconds (``floor(max_gap_seconds * fps)`` samples).
        static_distance: Consecutive samples within this many bits of the
            first sample of their run count as one picture, 0 to
            ``hash_bits``. 0 counts only bit-identical samples as static.
        max_offset_seconds: If given, only offsets with
            ``abs(offset_seconds) <= max_offset_seconds`` are considered.

    Returns:
        The best segment (see :class:`VideoMatch`). When no aligned pair
        matches, ``is_match`` is false and the result describes the offset
        whose aligned frames are closest on average (among offsets with at
        least ``min_distinct_frames`` compared pairs); when there is no such
        offset either, the counts are zero.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the fingerprints use different sampling rates or hash
            sizes, or a threshold is out of range.
    """
    _require_fingerprint("query", query)
    _require_fingerprint("reference", reference)
    if not math.isclose(query.fps, reference.fps, rel_tol=1e-9):
        raise ValueError("the fingerprints were sampled at different rates")
    if query.hash_bits != reference.hash_bits:
        raise ValueError("the fingerprints use different hash sizes")
    bits = query.hash_bits
    fps = query.fps
    check_int("max_distance", max_distance, minimum=0, maximum=bits)
    check_int("min_quality", min_quality, minimum=0, maximum=100)
    check_int("min_distinct_frames", min_distinct_frames, minimum=1)
    min_seconds = check_number("min_matched_seconds", min_matched_seconds, minimum=0.0)
    check_number("min_match_fraction", min_match_fraction, minimum=0.0, maximum=1.0)
    gap_seconds = check_number("max_gap_seconds", max_gap_seconds, minimum=0.0)
    check_int("static_distance", static_distance, minimum=0, maximum=bits)
    max_offset: int | None = None
    if max_offset_seconds is not None:
        seconds = check_number("max_offset_seconds", max_offset_seconds, minimum=0.0)
        max_offset = math.floor(seconds * fps + _EPSILON)
    rule = _Rule(
        max_distance=max_distance,
        max_gap=math.floor(gap_seconds * fps + _EPSILON),
        min_distinct=min_distinct_frames,
        min_span=max(1, math.ceil(min_seconds * fps - _EPSILON)),
        min_fraction=float(min_match_fraction),
    )

    q_usable = np.array([quality >= min_quality for quality in query.qualities], dtype=bool)
    r_usable = np.array([quality >= min_quality for quality in reference.qualities], dtype=bool)
    usable_query = int(q_usable.sum())
    if usable_query == 0 or not r_usable.any():
        return _no_match()

    # Scan the rows of the shorter fingerprint ("a") against the longer one
    # ("b"); the diagonal of a pair (a, b) is b - a + len(a) - 1.
    swapped = len(query) > len(reference)
    a_fp, b_fp = (reference, query) if swapped else (query, reference)
    a_usable, b_usable = (r_usable, q_usable) if swapped else (q_usable, r_usable)
    scan = _scan(
        _Side.of(a_fp, a_usable, static_distance),
        _Side.of(b_fp, b_usable, static_distance),
        rule,
    )
    shift = np.arange(scan.size, dtype=np.int64) - (len(a_fp) - 1)  # b index - a index
    offsets = -shift if swapped else shift  # reference index - query index
    allowed = np.ones(scan.size, dtype=bool)
    if max_offset is not None:
        allowed = np.abs(offsets) <= max_offset

    found = np.flatnonzero(scan.best_valid & allowed)
    if found.size == 0:
        return _closest_overlap(
            scan,
            offsets=offsets,
            allowed=allowed,
            min_compared=min_distinct_frames,
            usable_query=usable_query,
            bits=bits,
            fps=fps,
        )
    # np.lexsort sorts by the last key first.
    order = np.lexsort(
        (
            offsets[found],
            np.abs(offsets[found]),
            scan.best_mean[found],
            -scan.best_fraction[found],
            -scan.best_matched[found],
            -scan.best_distinct[found],
            ~scan.best_passes[found],
        )
    )
    best = int(found[order[0]])
    offset = int(offsets[best])
    first, last = int(scan.best_first[best]), int(scan.best_last[best])
    if swapped:  # first and last are reference indices
        first, last = first - offset, last - offset
    matched = int(scan.best_matched[best])
    compared = int(scan.best_compared[best])
    return VideoMatch(
        is_match=bool(scan.best_passes[best]),
        similarity=1.0 - int(scan.best_distance[best]) / (compared * bits),
        offset_seconds=offset / fps,
        offset_frames=offset,
        matched_frames=matched,
        distinct_frames=int(scan.best_distinct[best]),
        compared_frames=compared,
        match_fraction=matched / compared,
        matched_seconds=(last - first + 1) / fps,
        query_start_seconds=first / fps,
        query_end_seconds=(last + 1) / fps,
        reference_start_seconds=(first + offset) / fps,
        reference_end_seconds=(last + offset + 1) / fps,
        query_coverage=compared / usable_query,
    )


# --------------------------------------------------------------------------
# Internals
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Rule:
    """The segment thresholds, in samples."""

    max_distance: int
    max_gap: int
    min_distinct: int
    min_span: int
    min_fraction: float


class _Scan:
    """Per-diagonal state of the segment scan.

    ``cur_*`` describe the segment in progress on each diagonal, ``best_*``
    the best closed segment, ``overlap_*`` every usable pair (for the result
    when nothing matches).
    """

    def __init__(self, size: int) -> None:
        self.size = size
        self.cur_open = np.zeros(size, dtype=bool)
        self.cur_first = np.zeros(size, dtype=np.int64)
        self.cur_last = np.zeros(size, dtype=np.int64)
        self.cur_gap = np.zeros(size, dtype=np.int64)
        self.cur_matched = np.zeros(size, dtype=np.int64)
        self.cur_compared = np.zeros(size, dtype=np.int64)  # including trailing misses
        self.cur_distance = np.zeros(size, dtype=np.int64)
        self.cur_compared_at_last = np.zeros(size, dtype=np.int64)
        self.cur_distance_at_last = np.zeros(size, dtype=np.int64)
        self.cur_runs_a = np.zeros(size, dtype=np.int64)
        self.cur_runs_b = np.zeros(size, dtype=np.int64)
        self.cur_run_a = np.zeros(size, dtype=np.int64)
        self.cur_run_b = np.zeros(size, dtype=np.int64)
        self.best_valid = np.zeros(size, dtype=bool)
        self.best_passes = np.zeros(size, dtype=bool)
        self.best_distinct = np.zeros(size, dtype=np.int64)
        self.best_matched = np.zeros(size, dtype=np.int64)
        self.best_compared = np.zeros(size, dtype=np.int64)
        self.best_distance = np.zeros(size, dtype=np.int64)
        self.best_fraction = np.zeros(size, dtype=np.float64)
        self.best_mean = np.zeros(size, dtype=np.float64)
        self.best_first = np.zeros(size, dtype=np.int64)
        self.best_last = np.zeros(size, dtype=np.int64)
        self.overlap_compared = np.zeros(size, dtype=np.int64)
        self.overlap_distance = np.zeros(size, dtype=np.int64)

    def miss(self, index: npt.NDArray[np.intp], dist: npt.NDArray[np.int64], rule: _Rule) -> None:
        """Unmatched pairs on open segments: longer gaps; too long a gap closes."""
        self.cur_gap[index] += 1
        self.cur_compared[index] += 1
        self.cur_distance[index] += dist
        self.close(index[self.cur_gap[index] > rule.max_gap], rule)

    def hit(
        self,
        index: npt.NDArray[np.intp],
        dist: npt.NDArray[np.int64],
        *,
        row: int,
        run_a: int,
        run_b: npt.NDArray[np.int64],
    ) -> None:
        """Matched pairs: they extend the open segments and start the others."""
        extend = self.cur_open[index]
        grow = index[extend]
        self.cur_matched[grow] += 1
        self.cur_compared[grow] += 1
        self.cur_distance[grow] += dist[extend]
        self.cur_compared_at_last[grow] = self.cur_compared[grow]
        self.cur_distance_at_last[grow] = self.cur_distance[grow]
        self.cur_last[grow] = row
        self.cur_gap[grow] = 0
        self.cur_runs_a[grow] += self.cur_run_a[grow] != run_a
        self.cur_run_a[grow] = run_a
        self.cur_runs_b[grow] += self.cur_run_b[grow] != run_b[extend]
        self.cur_run_b[grow] = run_b[extend]
        new = index[~extend]
        self.cur_open[new] = True
        self.cur_first[new] = row
        self.cur_last[new] = row
        self.cur_gap[new] = 0
        self.cur_matched[new] = 1
        self.cur_compared[new] = 1
        self.cur_distance[new] = dist[~extend]
        self.cur_compared_at_last[new] = 1
        self.cur_distance_at_last[new] = dist[~extend]
        self.cur_runs_a[new] = 1
        self.cur_runs_b[new] = 1
        self.cur_run_a[new] = run_a
        self.cur_run_b[new] = run_b[~extend]

    def close(self, index: npt.NDArray[np.intp], rule: _Rule) -> None:
        """Ends the segments in progress on the diagonals ``index``."""
        if index.size == 0:
            return
        matched = self.cur_matched[index]
        compared = self.cur_compared_at_last[index]
        distance = self.cur_distance_at_last[index]
        distinct = np.minimum(self.cur_runs_a[index], self.cur_runs_b[index])
        span = self.cur_last[index] - self.cur_first[index] + 1
        fraction = matched / compared
        mean = distance / compared
        passes = (
            (distinct >= rule.min_distinct)
            & (span >= rule.min_span)
            & (fraction >= rule.min_fraction - _EPSILON)
        )
        better = ~self.best_valid[index] | _ranks_higher(
            (passes, distinct, matched, fraction, -mean),
            (
                self.best_passes[index],
                self.best_distinct[index],
                self.best_matched[index],
                self.best_fraction[index],
                -self.best_mean[index],
            ),
        )
        chosen = index[better]
        self.best_valid[chosen] = True
        self.best_passes[chosen] = passes[better]
        self.best_distinct[chosen] = distinct[better]
        self.best_matched[chosen] = matched[better]
        self.best_compared[chosen] = compared[better]
        self.best_distance[chosen] = distance[better]
        self.best_fraction[chosen] = fraction[better]
        self.best_mean[chosen] = mean[better]
        self.best_first[chosen] = self.cur_first[chosen]
        self.best_last[chosen] = self.cur_last[chosen]
        self.cur_open[index] = False


def _ranks_higher(
    new: Sequence[npt.ArrayLike], old: Sequence[npt.ArrayLike]
) -> npt.NDArray[np.bool_]:
    """Element-wise lexicographic ``new > old`` (higher keys rank first)."""
    keys = [
        (np.asarray(n, dtype=np.float64), np.asarray(o, dtype=np.float64))
        for n, o in zip(new, old, strict=True)
    ]
    result = np.zeros(keys[0][0].shape, dtype=bool)
    tied = np.ones(keys[0][0].shape, dtype=bool)
    for new_key, old_key in keys:
        result |= tied & (new_key > old_key)
        tied &= new_key == old_key
    return result


@dataclass(frozen=True, slots=True)
class _Side:
    """One fingerprint prepared for the scan."""

    hashes: npt.NDArray[np.uint8]
    usable: npt.NDArray[np.bool_]
    runs: npt.NDArray[np.int64]

    @classmethod
    def of(
        cls, fingerprint: VideoFingerprint, usable: npt.NDArray[np.bool_], static_distance: int
    ) -> _Side:
        return cls(
            hashes=_hash_matrix(fingerprint.hashes, fingerprint.hash_bits),
            usable=usable,
            runs=_static_runs(fingerprint.hashes, usable, static_distance),
        )


def _scan(a: _Side, b: _Side, rule: _Rule) -> _Scan:
    """Finds the best segment on every diagonal, one row of ``a`` at a time."""
    n_a, n_b = a.hashes.shape[0], b.hashes.shape[0]
    scan = _Scan(n_a + n_b - 1)
    distances = _RowDistances(b.hashes)
    for row in range(n_a):
        if not a.usable[row]:
            continue
        start = n_a - 1 - row  # the diagonal of the pair (row, 0)
        window = slice(start, start + n_b)
        dist = distances.to(a.hashes[row])
        scan.overlap_compared[window] += b.usable
        scan.overlap_distance[window] += np.where(b.usable, dist, 0)
        hit = b.usable & (dist <= rule.max_distance)
        miss = np.flatnonzero(b.usable & ~hit & scan.cur_open[window])
        if miss.size:
            scan.miss(miss + start, dist[miss], rule)
        columns = np.flatnonzero(hit)
        if columns.size:
            scan.hit(
                columns + start,
                dist[columns],
                row=row,
                run_a=int(a.runs[row]),
                run_b=b.runs[columns],
            )
    scan.close(np.flatnonzero(scan.cur_open), rule)
    return scan


class _RowDistances:
    """Hamming distances from one hash to every row of ``b``."""

    def __init__(self, b: npt.NDArray[np.uint8]) -> None:
        self._bytes = np.ascontiguousarray(b)
        native = hasattr(np, "bitwise_count") and b.shape[1] % 8 == 0
        self._words = self._bytes.view(np.uint64) if native else None

    def to(self, row: npt.NDArray[np.uint8]) -> npt.NDArray[np.int64]:
        if self._words is not None:
            words = np.ascontiguousarray(row).view(np.uint64)
            native: npt.NDArray[np.int64] = np.bitwise_count(self._words ^ words).sum(
                axis=1, dtype=np.int64
            )
            return native
        table: npt.NDArray[np.int64] = _POPCOUNT[self._bytes ^ row].sum(axis=1, dtype=np.int64)
        return table


def _static_runs(
    hashes: Sequence[int], usable: npt.NDArray[np.bool_], static_distance: int
) -> npt.NDArray[np.int64]:
    """The run number of each usable sample (``-1`` for unusable ones).

    A run continues while samples stay within ``static_distance`` bits of its
    first sample; anchoring on the first sample keeps a slow pan from chaining
    into one run.
    """
    runs = np.full(len(hashes), -1, dtype=np.int64)
    run = -1
    anchor: int | None = None
    for position, value in enumerate(hashes):
        if not usable[position]:
            continue
        if anchor is None or (value ^ anchor).bit_count() > static_distance:
            run += 1
            anchor = value
        runs[position] = run
    return runs


def _no_match() -> VideoMatch:
    return VideoMatch(
        is_match=False,
        similarity=0.0,
        offset_seconds=0.0,
        offset_frames=0,
        matched_frames=0,
        distinct_frames=0,
        compared_frames=0,
        match_fraction=0.0,
        matched_seconds=0.0,
        query_start_seconds=0.0,
        query_end_seconds=0.0,
        reference_start_seconds=0.0,
        reference_end_seconds=0.0,
        query_coverage=0.0,
    )


def _closest_overlap(
    scan: _Scan,
    *,
    offsets: npt.NDArray[np.int64],
    allowed: npt.NDArray[np.bool_],
    min_compared: int,
    usable_query: int,
    bits: int,
    fps: float,
) -> VideoMatch:
    """The result when nothing matched: the offset whose frames are closest."""
    eligible = np.flatnonzero(allowed & (scan.overlap_compared >= min_compared))
    if eligible.size == 0:
        return _no_match()
    mean = scan.overlap_distance[eligible] / scan.overlap_compared[eligible]
    order = np.lexsort((offsets[eligible], np.abs(offsets[eligible]), mean))
    best = int(eligible[order[0]])
    count = int(scan.overlap_compared[best])
    offset = int(offsets[best])
    return dataclasses.replace(
        _no_match(),
        similarity=1.0 - int(scan.overlap_distance[best]) / (count * bits),
        offset_seconds=offset / fps,
        offset_frames=offset,
        compared_frames=count,
        query_coverage=count / usable_query,
    )


def _hash_matrix(hashes: Sequence[int], bits: int) -> npt.NDArray[np.uint8]:
    width = bits // 8
    raw = b"".join(value.to_bytes(width, "big") for value in hashes)
    return np.frombuffer(raw, dtype=np.uint8).reshape(len(hashes), width)


def _check_frame(array: object) -> None:
    if not isinstance(array, np.ndarray):
        raise TypeError("frames must be VideoFrame objects or NumPy arrays")
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("frames must be uint8 arrays shaped (height, width, 3)")
    if array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError("frames must not be empty")


def _require_fingerprint(name: str, value: object) -> None:
    if not isinstance(value, VideoFingerprint):
        raise TypeError(f"{name} must be a VideoFingerprint")


def _hash_quality(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 100:
        raise ValueError("the frame hasher returned an invalid quality")
    return value
