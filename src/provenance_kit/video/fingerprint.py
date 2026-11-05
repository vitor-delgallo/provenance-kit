"""Video fingerprints: perceptual frame hashes at a fixed sampling rate.

A :class:`VideoFingerprint` is the sequence of PDQ hashes (256 bits each, by
default) of frames sampled at a fixed rate (``fps``, 1 per second by default),
plus each hash's quality score. Frame ``k`` was sampled ``k / fps`` seconds
after the start of the video. :meth:`VideoFingerprint.to_bytes` gives a
compact, versioned binary form for a database column (33 bytes per sampled
frame plus a 20-byte header: about 116 KiB for an hour at 1 frame per
second); :meth:`VideoFingerprint.from_bytes` validates it strictly.

:func:`match_fingerprints` aligns a query against a reference: for every
temporal offset it counts the aligned frame pairs whose Hamming distance is at
most ``max_distance``, and keeps the offset with the most matched frames.
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
(extra ``video``), loaded lazily; tests and applications can pass any object
with a ``hash_image(PIL.Image.Image)`` method returning ``value`` (bytes) and
``quality`` (0 to 100).

Thread safety: the functions and the immutable fingerprints are thread-safe,
provided the hasher is (the default PDQ hasher is).
"""

from __future__ import annotations

import importlib
import math
import struct
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
import numpy.typing as npt

from provenance_kit._logging import get_logger
from provenance_kit._typing import BytesLike, StrPath
from provenance_kit.errors import BackendUnavailableError
from provenance_kit.limits import MediaLimits
from provenance_kit.tools import ExternalTools
from provenance_kit.video._checks import check_int, check_number
from provenance_kit.video.frames import MAX_SAMPLING_FPS, FrameSampling, VideoFrame, iter_frames

if TYPE_CHECKING:
    from PIL import Image

__all__ = [
    "DEFAULT_FRAME_SIZE",
    "DEFAULT_MAX_DISTANCE",
    "DEFAULT_MIN_MATCHED_FRAMES",
    "DEFAULT_MIN_MATCH_FRACTION",
    "DEFAULT_MIN_QUALITY",
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
#: bits after re-encoding, scaling or a frame-rate change; median 30, maximum
#: 34 for a clip cut 0.4 s off the sampling grid of fast-moving content), while
#: unrelated frames measure about 128 (minimum 106 in the same test). A match
#: also needs several frames (see the thresholds below).
DEFAULT_MAX_DISTANCE = 40

#: Frames whose hash quality is below this are ignored when matching (PDQ's
#: guidance: lower-quality hashes come from nearly featureless images such as
#: black or flat frames, which match each other spuriously).
DEFAULT_MIN_QUALITY = 50

#: A match needs at least this many matched frames ...
DEFAULT_MIN_MATCHED_FRAMES = 3

#: ... and at least this fraction of the compared frames matched.
DEFAULT_MIN_MATCH_FRACTION = 0.5

#: The longest fingerprint :meth:`VideoFingerprint.from_bytes` accepts.
MAX_FINGERPRINT_FRAMES = 1_000_000

_MAGIC = b"PKVF"
_HEADER = struct.Struct(">4sBBHdI")  # magic, version, hash bytes, reserved, fps, count
_MAX_HASH_BYTES = 64
_CHUNK_CELLS = 1 << 21  # distance-matrix cells computed per block (bounds memory)
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)


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


@dataclass(frozen=True, slots=True)
class VideoMatch:
    """The best temporal alignment of a query fingerprint against a reference.

    Attributes:
        is_match: The decision: at least ``min_matched_frames`` matched frames
            and a ``match_fraction`` of at least ``min_match_fraction``.
        similarity: Mean of ``1 - distance / hash_bits`` over the compared
            frame pairs at the best offset (1.0 for identical frames, about 0.5
            for unrelated ones); 0.0 when nothing could be compared.
        offset_seconds: Where the query starts in the reference:
            ``reference_time = query_time + offset_seconds``. Negative when
            the query starts before the reference.
        offset_frames: The same offset in sampled frames.
        matched_frames: Aligned pairs within ``max_distance``.
        compared_frames: Aligned pairs where both hashes had usable quality.
        match_fraction: ``matched_frames / compared_frames`` (0.0 if none).
        query_coverage: ``compared_frames`` over the query's usable frames:
            how much of the query overlaps the reference at that offset.
    """

    is_match: bool
    similarity: float
    offset_seconds: float
    offset_frames: int
    matched_frames: int
    compared_frames: int
    match_fraction: float
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
    min_matched_frames: int = DEFAULT_MIN_MATCHED_FRAMES,
    min_match_fraction: float = DEFAULT_MIN_MATCH_FRACTION,
    max_offset_seconds: float | None = None,
) -> VideoMatch:
    """Finds the temporal offset that best aligns ``query`` with ``reference``.

    Every offset at which at least ``min_matched_frames`` usable frame pairs
    overlap is scored by its number of matched pairs (Hamming distance at most
    ``max_distance``); ties go to the higher match fraction, then the lower
    mean distance, then the smaller absolute offset. Cost: one Hamming
    distance per pair of usable frames (``len(query) * len(reference)``),
    computed in blocks of bounded memory; ``max_offset_seconds`` restricts
    the offsets considered.

    Args:
        query: The fingerprint to look up (for example a clip).
        reference: The fingerprint to search in.
        max_distance: The per-frame Hamming threshold, 0 to ``hash_bits``.
        min_quality: Frames with a lower hash quality are ignored.
        min_matched_frames: Matched frames required for ``is_match`` (and
            overlapping usable pairs required for an offset to be scored).
        min_match_fraction: Matched fraction required for ``is_match``.
        max_offset_seconds: If given, only offsets with
            ``abs(offset_seconds) <= max_offset_seconds`` are scored.

    Returns:
        The best alignment. When no offset can be scored, ``is_match`` is
        false and the counts are zero.

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
    check_int("max_distance", max_distance, minimum=0, maximum=bits)
    check_int("min_quality", min_quality, minimum=0, maximum=100)
    check_int("min_matched_frames", min_matched_frames, minimum=1)
    check_number("min_match_fraction", min_match_fraction, minimum=0.0, maximum=1.0)
    max_offset: int | None = None
    if max_offset_seconds is not None:
        seconds = check_number("max_offset_seconds", max_offset_seconds, minimum=0.0)
        max_offset = math.floor(seconds * query.fps + 1e-9)

    q_usable = np.array([quality >= min_quality for quality in query.qualities], dtype=bool)
    r_usable = np.array([quality >= min_quality for quality in reference.qualities], dtype=bool)
    usable_query = int(q_usable.sum())
    nothing = VideoMatch(
        is_match=False,
        similarity=0.0,
        offset_seconds=0.0,
        offset_frames=0,
        matched_frames=0,
        compared_frames=0,
        match_fraction=0.0,
        query_coverage=0.0,
    )
    if usable_query == 0 or not r_usable.any():
        return nothing

    matched, compared, distance_sum = _diagonal_counts(
        _hash_matrix(query.hashes, bits),
        _hash_matrix(reference.hashes, bits),
        q_usable,
        r_usable,
        max_distance,
    )
    n_query = len(query)
    offsets = np.arange(matched.size) - (n_query - 1)
    eligible = compared >= min_matched_frames
    if max_offset is not None:
        eligible &= np.abs(offsets) <= max_offset
    if not eligible.any():
        return nothing

    candidates = np.flatnonzero(eligible)
    fraction = matched[candidates] / compared[candidates]
    mean_distance = distance_sum[candidates] / compared[candidates]
    # np.lexsort sorts by the last key first.
    order = np.lexsort(
        (np.abs(offsets[candidates]), mean_distance, -fraction, -matched[candidates])
    )
    best = int(candidates[order[0]])
    best_matched = int(matched[best])
    best_compared = int(compared[best])
    best_fraction = best_matched / best_compared
    offset_frames = int(offsets[best])
    return VideoMatch(
        is_match=best_matched >= min_matched_frames and best_fraction >= min_match_fraction,
        similarity=1.0 - float(distance_sum[best]) / (best_compared * bits),
        offset_seconds=offset_frames / query.fps,
        offset_frames=offset_frames,
        matched_frames=best_matched,
        compared_frames=best_compared,
        match_fraction=best_fraction,
        query_coverage=best_compared / usable_query,
    )


# --------------------------------------------------------------------------
# Internals
# --------------------------------------------------------------------------


def _hash_matrix(hashes: Sequence[int], bits: int) -> npt.NDArray[np.uint8]:
    width = bits // 8
    raw = b"".join(value.to_bytes(width, "big") for value in hashes)
    return np.frombuffer(raw, dtype=np.uint8).reshape(len(hashes), width)


def _diagonal_counts(
    q: npt.NDArray[np.uint8],
    r: npt.NDArray[np.uint8],
    q_usable: npt.NDArray[np.bool_],
    r_usable: npt.NDArray[np.bool_],
    max_distance: int,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Per-offset matched / compared counts and distance sums.

    Offset ``d`` (index ``d + len(q) - 1``) pairs query frame ``i`` with
    reference frame ``i + d``.
    """
    n_q, n_r = q.shape[0], r.shape[0]
    n_offsets = n_q + n_r - 1
    matched = np.zeros(n_offsets, dtype=np.int64)
    compared = np.zeros(n_offsets, dtype=np.int64)
    distance_sum = np.zeros(n_offsets, dtype=np.float64)
    rows = max(1, _CHUNK_CELLS // max(n_r * q.shape[1], 1))
    columns = np.arange(n_r)
    for start in range(0, n_q, rows):
        stop = min(start + rows, n_q)
        block = _POPCOUNT[q[start:stop, None, :] ^ r[None, :, :]].sum(axis=2)
        valid = q_usable[start:stop, None] & r_usable[None, :]
        index = columns[None, :] - np.arange(start, stop)[:, None] + (n_q - 1)
        flat_index = index.ravel()
        flat_valid = valid.ravel()
        matched += np.bincount(
            flat_index,
            weights=(flat_valid & (block.ravel() <= max_distance)).astype(np.float64),
            minlength=n_offsets,
        ).astype(np.int64)
        compared += np.bincount(
            flat_index, weights=flat_valid.astype(np.float64), minlength=n_offsets
        ).astype(np.int64)
        distance_sum += np.bincount(
            flat_index,
            weights=np.where(flat_valid, block.ravel(), 0).astype(np.float64),
            minlength=n_offsets,
        )
    return matched, compared, distance_sum


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
