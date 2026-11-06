"""Perceptual image hashes (PDQ, pHash, dHash) and known-hash-list matching.

Three hashes, for finding copies of an image after re-encoding, resizing or
small edits (not crops: removing 2 % of each side already moves PDQ past its
threshold on most images, see ``docs/image.md``):

* **PDQ** (Meta), 256 bits plus a quality score. :class:`PdqHasher` runs one
  of two backends behind :class:`PdqBackend`: a NumPy port of Meta's
  pure-Python reference implementation that needs nothing beyond the core
  install (the default backend), and the ``pdqhash`` package (Meta's C++
  code, extra ``pdq-native``), which ``"auto"`` prefers when it is
  installed. Both give the same hash for the same pixels, to within a few
  bits, on images at least 64 pixels on each side; see "Backend agreement"
  below.
* **pHash** (DCT) and **dHash** (gradient), 64 bits each by default, through
  the ``imagehash`` package (extra ``image-hash``): :func:`phash`,
  :func:`dhash`.

:class:`ImageFingerprint` bundles the three, :func:`compare_fingerprints`
compares two of them against :class:`MatchThresholds`, and :class:`HashList`
matches a hash against a large list of known hashes (for example a list
shared by a partner organization).

**PDQ bit order.** :attr:`PdqHash.value` is ``bytes.fromhex(h)``, where ``h``
is the canonical 64-digit hex form that Meta's tools print and exchange
(``Hash256.toHexString``). Bit ``k`` of the reference ``Hash256`` (``k = 16 *
row + column`` of the 16 x 16 DCT coefficients; set when that coefficient is
above the median) is bit ``k % 8`` (least significant = 0) of byte ``31 - k //
8``; equivalently, ``int.from_bytes(value, "big")`` has bit ``k`` at ``2**k``.

**Pre-scaling.** Both of Meta's reference hashers shrink an image to at most
512 x 512 before PDQ's blur filter, which is "prohibitively expensive for
larger images" (``pdq/cpp/io/pdqio.cpp``; the Python reference calls
``Image.thumbnail((512, 512))``). :class:`PdqHasher` does the same by default
(``max_side=512``, aspect ratio kept, Pillow's ``thumbnail``), so both
backends see identical pixels and the cost is bounded, with one deliberate
difference: the shorter side is never shrunk below 64 pixels (see
:class:`PdqHasher`), because PDQ is numerically unstable on thinner images.
``max_side=None`` hashes at full resolution, as ``python-threatexchange``'s
``pdq_from_bytes`` does; the two settings usually differ by a few bits only.

**Backend agreement.** The NumPy port reproduces Meta's pure-Python
reference bit for bit on the generated test corpus (``tests/image``), except
that an image with a side under 5 pixels gets the C++ hasher's all-zero hash
with quality 0 (see below). On
ThreatExchange's published test images, both backends return the published
hashes exactly (distance 0) for every image with quality >= 50. ``pdqhash``
computes in ``float32``, the port in ``float64``, so DCT coefficients that
sit next to the median can come out on different sides of it:

* both sides at least 64 pixels as hashed (every image at least 64 x 64,
  whatever its aspect ratio, thanks to the pre-scaling rule): over 3,000
  generated images, 2,302 of the 2,307 reliable pairs were identical and the
  other five within 2 bits. The tests assert :data:`PDQ_BACKEND_MAX_DISTANCE`
  (10 bits), far below the match threshold;
* a side of 5 to 63 pixels: no bound. Many coefficients nearly tie (the
  64 x 64 block repeats rows or columns), so the backends measured up to 21
  bits apart (426 pairs; worst at sides of 8 and 16 pixels) and 132 bits on
  one 854 x 6 image, and even one backend's hash moves a median of 64 to 86
  bits after a JPEG re-encode when the side is under 20 pixels. Do not rely
  on PDQ for such images, whatever their quality;
* a side under 5 pixels: both return the all-zero hash with quality 0
  (Meta's ``MIN_HASHABLE_DIM``).

The quality scores are computed from the same truncated gradients and agree
on almost every image, but not always: they differed by 1 on 3 of the
3,000 images, so an image at the threshold (49 or 50) can be reliable with
one backend only.

**Thresholds.** Meta's PDQ README recommends treating hashes at Hamming
distance ``<= 31`` as matching and discarding hashes with quality ``<= 49``
(``python-threatexchange`` rejects quality ``< 50``): :data:`PDQ_MATCH_DISTANCE`
and :data:`PDQ_MIN_QUALITY`. The pHash and dHash defaults
(:data:`PHASH_MATCH_DISTANCE`, :data:`DHASH_MATCH_DISTANCE`) are this
library's conservative starting points, not an external standard: calibrate
them on your own content. :func:`compare_fingerprints` uses pHash and dHash
only with enough set and clear bits (:attr:`MatchThresholds.min_bit_balance`;
flat images look alike to every perceptual hash). When PDQ is unreliable
(quality below 50 on a side: night scenes, light sketches, dark paintings),
pHash decides instead, and a match also needs PDQ to agree, because
unrelated minimal images (one shape on a plain background) can have
identical pHashes; such decisions are flagged as weaker evidence
(:attr:`FingerprintComparison.low_quality_fallback`).

Thread safety: hashing functions, :class:`PdqHasher` and :class:`HashList`
queries are thread-safe (they keep no mutable state after construction).

Inputs: the hashers take decoded :class:`PIL.Image.Image` objects. Decode
untrusted bytes with :func:`provenance_kit.image.loading.load_image` first,
which enforces :class:`~provenance_kit.limits.MediaLimits`.
"""

from __future__ import annotations

import enum
from collections.abc import Collection, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeAlias, TypeVar

import numpy as np
from PIL import Image

from provenance_kit._logging import get_logger
from provenance_kit._optional import import_optional
from provenance_kit.errors import BackendUnavailableError
from provenance_kit.image import _modes
from provenance_kit.image._pdq_numpy import pdq_hash_rgb
from provenance_kit.matching import HammingMatch

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "BINARY_MIN_BALANCE",
    "DHASH_MATCH_DISTANCE",
    "PDQ_BACKEND_MAX_DISTANCE",
    "PDQ_BITS",
    "PDQ_DEFAULT_MAX_SIDE",
    "PDQ_MATCH_DISTANCE",
    "PDQ_MIN_QUALITY",
    "PHASH_MATCH_DISTANCE",
    "BinaryHash",
    "FingerprintComparison",
    "HashAlgorithm",
    "HashList",
    "HashValue",
    "ImageFingerprint",
    "MatchThresholds",
    "PdqBackend",
    "PdqBackendName",
    "PdqHash",
    "PdqHasher",
    "UndecidedReason",
    "compare_fingerprints",
    "dhash",
    "fingerprint_image",
    "phash",
]

_log = get_logger(__name__)

#: Width of a PDQ hash in bits.
PDQ_BITS = 256
_PDQ_BYTES = PDQ_BITS // 8

#: Meta's recommended match threshold: Hamming distance <= 31 bits.
PDQ_MATCH_DISTANCE = 31

#: Meta's recommended minimum quality: hashes with quality <= 49 are discarded.
PDQ_MIN_QUALITY = 50

#: Images whose longer side exceeds this are shrunk before PDQ (see the module
#: docstring), as Meta's reference hashers do.
PDQ_DEFAULT_MAX_SIDE = 512

#: The largest distance between the two PDQ backends' hashes of the same image
#: that the test suite accepts, for images with quality >= PDQ_MIN_QUALITY on
#: both backends whose sides are both at least 64 pixels as hashed. The
#: measured maximum is 2 bits; thinner images have no bound (see the module
#: docstring and docs/image.md).
PDQ_BACKEND_MAX_DISTANCE = 10

#: Pre-scaling never shrinks the shorter side below this (PDQ's 64 x 64 block).
_PDQ_MIN_SCALED_SIDE = 64

#: An elongated image keeps its shorter side at 64 px only while its longer
#: side stays within this multiple of ``max_side`` (16,384 px by default, the
#: default ``MediaLimits.max_width``), which bounds the cost.
_PDQ_MAX_ELONGATION = 32

#: Default for :attr:`MatchThresholds.min_bit_balance`: a pHash or dHash
#: decides only if at least 1/8 of its bits are set and 1/8 clear.
BINARY_MIN_BALANCE = 0.125

#: Default pHash threshold (64-bit hash): a starting point, calibrate it.
PHASH_MATCH_DISTANCE = 10

#: Default dHash threshold (64-bit hash): a starting point, calibrate it.
DHASH_MATCH_DISTANCE = 10

_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")


# ---------------------------------------------------------------- hash values


@dataclass(frozen=True, slots=True)
class PdqHash:
    """A 256-bit PDQ hash and its quality score.

    Attributes:
        value: Exactly 32 bytes in the canonical PDQ order (see the module
            docstring): ``value.hex()`` is the hex form Meta's tools print.
        quality: The PDQ quality score, 0 to 100. Below
            :data:`PDQ_MIN_QUALITY` the hash describes a mostly featureless
            image and should not be matched.

    Raises:
        TypeError: If ``value`` is not ``bytes`` or ``quality`` not an ``int``.
        ValueError: If ``value`` is not 32 bytes or ``quality`` is outside
            0 to 100.
    """

    value: bytes
    quality: int

    def __post_init__(self) -> None:
        """Validates the fields."""
        if not isinstance(self.value, bytes):
            raise TypeError(f"value must be bytes, got {type(self.value).__name__}")
        if len(self.value) != _PDQ_BYTES:
            raise ValueError(f"a PDQ hash is {_PDQ_BYTES} bytes, got {len(self.value)}")
        if isinstance(self.quality, bool) or not isinstance(self.quality, int):
            raise TypeError(f"quality must be an int, got {type(self.quality).__name__}")
        if not 0 <= self.quality <= 100:
            raise ValueError("quality must be between 0 and 100")

    @property
    def bits(self) -> int:
        """The hash width, always :data:`PDQ_BITS`."""
        return PDQ_BITS

    def hex(self) -> str:
        """Returns the canonical form: 64 lowercase hex digits."""
        return self.value.hex()

    @classmethod
    def from_hex(cls, text: str, quality: int = 100) -> PdqHash:
        """Parses the canonical 64-digit hex form.

        Args:
            text: 64 hex digits (either case), as printed by Meta's tools.
            quality: The quality score to attach. Defaults to 100 because
                lists of known hashes normally contain only hashes that passed
                the quality filter; pass the real value when you have it.

        Returns:
            The hash.

        Raises:
            TypeError: If ``text`` is not a ``str``.
            ValueError: If ``text`` is not 64 hex digits, or ``quality`` is
                out of range.
        """
        return cls(_parse_hex(text, PDQ_BITS), quality)

    def to_int(self) -> int:
        """Returns the hash as a non-negative integer below ``2**256``."""
        return int.from_bytes(self.value, "big")

    def distance(self, other: PdqHash) -> int:
        """Returns the Hamming distance to another PDQ hash.

        Raises:
            TypeError: If ``other`` is not a :class:`PdqHash`.
        """
        if not isinstance(other, PdqHash):
            raise TypeError("other must be a PdqHash")
        return (self.to_int() ^ other.to_int()).bit_count()

    def is_reliable(self, min_quality: int = PDQ_MIN_QUALITY) -> bool:
        """Whether the quality is at least ``min_quality`` (default 50)."""
        return self.quality >= min_quality


@dataclass(frozen=True, slots=True)
class BinaryHash:
    """A fixed-width binary hash, such as a 64-bit pHash or dHash.

    Serialized as lowercase hex, zero-padded to ``ceil(bits / 4)`` digits;
    for :func:`phash` and :func:`dhash` this is the same string
    ``imagehash`` prints (``str(ImageHash)``), so values can be exchanged
    with it.

    Attributes:
        value: The hash as a non-negative integer below ``2**bits``. Bit 0
            of the hash in ``imagehash``'s row-major order is the most
            significant bit.
        bits: The width in bits, at least 1.

    Raises:
        TypeError: If a field is not an ``int``.
        ValueError: If ``bits`` < 1 or ``value`` does not fit.
    """

    value: int
    bits: int

    def __post_init__(self) -> None:
        """Validates the fields."""
        for name in ("value", "bits"):
            field_value = getattr(self, name)
            if isinstance(field_value, bool) or not isinstance(field_value, int):
                raise TypeError(f"{name} must be an int")
        if self.bits < 1:
            raise ValueError("bits must be at least 1")
        if self.value < 0 or self.value >> self.bits:
            raise ValueError(f"value does not fit in {self.bits} unsigned bits")

    def hex(self) -> str:
        """Returns the hash as lowercase hex, ``ceil(bits / 4)`` digits."""
        return format(self.value, f"0{-(-self.bits // 4)}x")

    @classmethod
    def from_hex(cls, text: str, *, bits: int | None = None) -> BinaryHash:
        """Parses the hex form.

        Args:
            text: Hex digits (either case).
            bits: The width; ``None`` takes ``4 * len(text)``. The text must
                have ``ceil(bits / 4)`` digits.

        Returns:
            The hash.

        Raises:
            TypeError: If ``text`` is not a ``str``.
            ValueError: If ``text`` is not hex, has the wrong length or does
                not fit in ``bits``.
        """
        width = 4 * len(text) if bits is None else bits
        _check_bits(width)
        return cls(int.from_bytes(_parse_hex(text, width), "big"), width)

    def distance(self, other: BinaryHash) -> int:
        """Returns the Hamming distance to a hash of the same width.

        Raises:
            TypeError: If ``other`` is not a :class:`BinaryHash`.
            ValueError: If the widths differ.
        """
        if not isinstance(other, BinaryHash):
            raise TypeError("other must be a BinaryHash")
        if other.bits != self.bits:
            raise ValueError("hashes of different widths cannot be compared")
        return (self.value ^ other.value).bit_count()

    def to_bytes(self) -> bytes:
        """Returns the hash as big-endian bytes, ``ceil(bits / 8)`` long."""
        return self.value.to_bytes((self.bits + 7) // 8, "big")


def _parse_hex(text: object, bits: int) -> bytes:
    """Parses ``ceil(bits / 4)`` hex digits into ``ceil(bits / 8)`` big-endian bytes."""
    if not isinstance(text, str):
        raise TypeError(f"a hash in hex must be a str, got {type(text).__name__}")
    digits = -(-bits // 4)
    if len(text) != digits or not _HEX_DIGITS.issuperset(text):
        raise ValueError(f"expected {digits} hex digits")
    value = int(text, 16)
    if value >> bits:
        raise ValueError(f"hash does not fit in {bits} bits")
    return value.to_bytes((bits + 7) // 8, "big")


# ---------------------------------------------------------------- PDQ


class PdqBackendName(enum.StrEnum):
    """PDQ implementation selection for :class:`PdqHasher`."""

    #: ``pdqhash`` if it imports, else the NumPy port.
    AUTO = "auto"
    #: Meta's C++ implementation through the ``pdqhash`` package.
    PDQHASH = "pdqhash"
    #: The NumPy port of Meta's Python reference (no extra needed).
    NUMPY = "numpy"


class PdqBackend(Protocol):
    """What :class:`PdqHasher` needs from a PDQ implementation."""

    @property
    def name(self) -> str:
        """A short backend name, for logs and diagnostics."""
        ...

    def hash_rgb(self, rgb: NDArray[np.uint8]) -> tuple[bytes, int]:
        """Hashes an RGB image.

        Args:
            rgb: ``uint8`` array of shape ``(height, width, 3)``.

        Returns:
            The 32-byte hash in canonical order and the quality (0 to 100).
        """
        ...


class _NumpyPdqBackend:
    name = PdqBackendName.NUMPY.value

    def hash_rgb(self, rgb: NDArray[np.uint8]) -> tuple[bytes, int]:
        return pdq_hash_rgb(rgb)


class _PdqhashBackend:
    name = PdqBackendName.PDQHASH.value

    def __init__(self) -> None:
        module = import_optional("pdqhash", extra="pdq-native", feature="PDQ with pdqhash")
        self._compute = module.compute

    def hash_rgb(self, rgb: NDArray[np.uint8]) -> tuple[bytes, int]:
        # pdqhash returns the 256 bits with bit 255 first, which is exactly
        # the canonical byte order once packed. It reads the buffer as
        # row-major whatever the array's strides, so a transposed or
        # Fortran-ordered view must be copied first.
        vector, quality = self._compute(np.ascontiguousarray(rgb))
        bits = np.asarray(vector, dtype=np.uint8)
        return np.packbits(bits).tobytes(), int(quality)


def _resolve_pdq_backend(backend: str | PdqBackend) -> PdqBackend:
    if not isinstance(backend, str):
        if not callable(getattr(backend, "hash_rgb", None)):
            raise TypeError("backend must be a backend name or an object with hash_rgb()")
        return backend
    try:
        choice = PdqBackendName(backend.strip().lower())
    except ValueError:
        raise ValueError('backend must be "auto", "pdqhash" or "numpy"') from None
    if choice is PdqBackendName.NUMPY:
        return _NumpyPdqBackend()
    if choice is PdqBackendName.PDQHASH:
        return _PdqhashBackend()
    try:
        return _PdqhashBackend()
    except BackendUnavailableError:
        _log.debug("pdqhash is not available; using the NumPy PDQ port")
        return _NumpyPdqBackend()


class PdqHasher:
    """Computes PDQ hashes with ``pdqhash`` or the built-in NumPy port.

    Example::

        hasher = PdqHasher()  # "auto": pdqhash if installed, else NumPy
        h = hasher.hash_image(loaded.image)
        if h.is_reliable():
            matches = known_hashes.match(h)

    Construct one hasher and reuse it. Hashing a 512 x 512 image takes about
    8 ms with the ``pdqhash`` backend and 15 ms with the NumPy port; the most
    elongated shape the pre-scaling keeps (16,384 x 64) up to 30 and 140 ms
    (see ``docs/image.md`` for the measurements).

    Thread safety: safe to share between threads.

    Args:
        backend: ``"auto"`` (default), ``"pdqhash"``, ``"numpy"``, or an
            object implementing :class:`PdqBackend` (for tests or other
            implementations). Backends always receive C-contiguous pixels.
        max_side: Shrink images whose longer side exceeds this many pixels
            before hashing, keeping the aspect ratio (Pillow ``thumbnail``),
            like Meta's reference hashers, with one difference: the shorter
            side is never shrunk below 64 pixels (PDQ's block size), so an
            elongated image such as a 4000 x 100 banner becomes 2560 x 64
            rather than 512 x 12, and an image whose shorter side is already
            under 64 pixels is not shrunk at all. The longer side is capped
            at ``32 * max_side`` (16,384 pixels by default) to bound the cost,
            which only affects images more elongated than that. ``None``
            hashes at full resolution (cost and memory grow with the pixel
            count: the NumPy port holds about 24 bytes per pixel).

    Raises:
        TypeError: If ``backend`` or ``max_side`` has the wrong type.
        ValueError: If ``backend`` is not a known name or ``max_side`` < 64.
        BackendUnavailableError: If ``backend="pdqhash"`` and the package is
            not installed.
    """

    def __init__(
        self,
        backend: str | PdqBackend = "auto",
        *,
        max_side: int | None = PDQ_DEFAULT_MAX_SIDE,
    ) -> None:
        """Selects the backend (see the class docstring)."""
        if max_side is not None:
            if isinstance(max_side, bool) or not isinstance(max_side, int):
                raise TypeError("max_side must be an int or None")
            if max_side < 64:
                raise ValueError("max_side must be at least 64 (PDQ works on a 64 x 64 grid)")
        self._backend = _resolve_pdq_backend(backend)
        self._max_side = max_side

    @property
    def backend(self) -> str:
        """The name of the backend in use (``"pdqhash"`` or ``"numpy"``)."""
        return self._backend.name

    @property
    def max_side(self) -> int | None:
        """The pre-scaling bound, or ``None`` for full resolution."""
        return self._max_side

    def hash_image(self, image: Image.Image) -> PdqHash:
        """Hashes a decoded image.

        The image is converted to 8-bit RGB first (alpha is ignored, as in
        Meta's tools; 16-bit modes are scaled to 8 bits) and shrunk according
        to ``max_side``. An image with a side under 5 pixels gets the all-zero
        hash with quality 0, as in Meta's C++ hasher.

        Args:
            image: The image, at least 1 x 1 pixels.

        Returns:
            The hash and its quality.

        Raises:
            TypeError: If ``image`` is not a Pillow image.
            ValueError: If the image is empty or its mode cannot be converted.
        """
        if not isinstance(image, Image.Image):
            raise TypeError(f"image must be a PIL.Image.Image, got {type(image).__name__}")
        if image.width < 1 or image.height < 1:
            raise ValueError("cannot hash an empty image")
        rgb = _modes.to_rgb(image)
        box = _thumbnail_box(rgb.width, rgb.height, self._max_side)
        if box is not None:
            work = rgb.copy() if rgb is image else rgb
            work.thumbnail((box, box))
            rgb = work
        return self._hash_array(np.asarray(rgb))

    def hash_rgb(self, rgb: NDArray[np.uint8]) -> PdqHash:
        """Hashes an RGB frame given as an array (for example a video frame).

        Args:
            rgb: ``uint8`` array of shape ``(height, width, 3)``, with any
                strides (a transposed, rotated or Fortran-ordered view is
                copied to C order first). Frames larger than ``max_side`` are
                shrunk exactly as :meth:`hash_image` would.

        Returns:
            The hash and its quality.

        Raises:
            TypeError: If ``rgb`` is not a ``uint8`` NumPy array.
            ValueError: If the shape is not ``(height, width, 3)`` or it is
                empty.
        """
        candidate: object = rgb
        if not isinstance(candidate, np.ndarray) or candidate.dtype != np.uint8:
            raise TypeError("rgb must be a uint8 numpy array")
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("rgb must have shape (height, width, 3)")
        if rgb.shape[0] < 1 or rgb.shape[1] < 1:
            raise ValueError("cannot hash an empty image")
        if _thumbnail_box(rgb.shape[1], rgb.shape[0], self._max_side) is not None:
            return self.hash_image(Image.fromarray(np.ascontiguousarray(rgb)))
        return self._hash_array(rgb)

    def _hash_array(self, rgb: NDArray[np.uint8]) -> PdqHash:
        value, quality = self._backend.hash_rgb(np.ascontiguousarray(rgb))
        return PdqHash(bytes(value), int(quality))


def _thumbnail_box(width: int, height: int, max_side: int | None) -> int | None:
    """The ``thumbnail((box, box))`` bound for pre-scaling, or ``None`` to keep the size.

    Meta's rule (a ``max_side`` box), except that the shorter side never
    drops below ``_PDQ_MIN_SCALED_SIDE`` and the longer side never exceeds
    ``_PDQ_MAX_ELONGATION * max_side``.
    """
    if max_side is None:
        return None
    longer, shorter = max(width, height), min(width, height)
    if longer <= max_side:
        return None
    # The longer side at which the shorter one is exactly 64 px (rounded up,
    # so that Pillow's rounding of the shorter side cannot give 63).
    keeps_short_side = -(-_PDQ_MIN_SCALED_SIDE * longer // shorter)
    box = min(max(max_side, keeps_short_side), _PDQ_MAX_ELONGATION * max_side)
    return box if box < longer else None


# ---------------------------------------------------------------- pHash / dHash


def _imagehash() -> Any:
    return import_optional("imagehash", extra="image-hash", feature="pHash and dHash")


def _check_hash_size(hash_size: int) -> None:
    if isinstance(hash_size, bool) or not isinstance(hash_size, int):
        raise TypeError("hash_size must be an int")
    if not 2 <= hash_size <= 64:
        raise ValueError("hash_size must be between 2 and 64")


def _to_binary_hash(result: Any) -> BinaryHash:
    flat = np.asarray(result.hash, dtype=bool).ravel()
    return BinaryHash(
        int.from_bytes(np.packbits(flat).tobytes(), "big") >> (-flat.size % 8), flat.size
    )


def phash(image: Image.Image, *, hash_size: int = 8) -> BinaryHash:
    """Computes the DCT perceptual hash (pHash) with ``imagehash``.

    The image is converted to 8-bit greyscale first (alpha ignored, 16-bit
    modes scaled), then ``imagehash.phash`` resizes it to ``4 * hash_size``
    pixels square and keeps the low-frequency DCT signs.

    Args:
        image: The decoded image.
        hash_size: The hash is ``hash_size ** 2`` bits (default 64).

    Returns:
        The hash; ``hex()`` equals ``str(imagehash.phash(...))``.

    Raises:
        TypeError: If ``image`` is not a Pillow image.
        ValueError: If ``hash_size`` is out of range (2 to 64) or the image is
            empty.
        BackendUnavailableError: If ``imagehash`` is not installed (extra
            ``image-hash``).
    """
    _check_hash_size(hash_size)
    luma = _luma_for_hashing(image)
    return _to_binary_hash(_imagehash().phash(luma, hash_size=hash_size))


def dhash(image: Image.Image, *, hash_size: int = 8) -> BinaryHash:
    """Computes the horizontal difference hash (dHash) with ``imagehash``.

    Args:
        image: The decoded image.
        hash_size: The hash is ``hash_size ** 2`` bits (default 64).

    Returns:
        The hash; ``hex()`` equals ``str(imagehash.dhash(...))``.

    Raises:
        TypeError: If ``image`` is not a Pillow image.
        ValueError: If ``hash_size`` is out of range (2 to 64) or the image is
            empty.
        BackendUnavailableError: If ``imagehash`` is not installed (extra
            ``image-hash``).
    """
    _check_hash_size(hash_size)
    luma = _luma_for_hashing(image)
    return _to_binary_hash(_imagehash().dhash(luma, hash_size=hash_size))


def _luma_for_hashing(image: Image.Image) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise TypeError(f"image must be a PIL.Image.Image, got {type(image).__name__}")
    if image.width < 1 or image.height < 1:
        raise ValueError("cannot hash an empty image")
    return _modes.to_luma(image)


# ---------------------------------------------------------------- fingerprints


class HashAlgorithm(enum.StrEnum):
    """The perceptual hashes in an :class:`ImageFingerprint`."""

    PDQ = "pdq"
    PHASH = "phash"
    DHASH = "dhash"


@dataclass(frozen=True, slots=True)
class ImageFingerprint:
    """The perceptual hashes of one image; absent hashes are ``None``.

    Attributes:
        pdq: The PDQ hash and quality.
        phash: The 64-bit pHash.
        dhash: The 64-bit dHash.
    """

    pdq: PdqHash | None = None
    phash: BinaryHash | None = None
    dhash: BinaryHash | None = None

    def __post_init__(self) -> None:
        """Validates the field types."""
        if self.pdq is not None and not isinstance(self.pdq, PdqHash):
            raise TypeError("pdq must be a PdqHash")
        for name in ("phash", "dhash"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, BinaryHash):
                raise TypeError(f"{name} must be a BinaryHash")


def fingerprint_image(
    image: Image.Image,
    *,
    algorithms: Collection[HashAlgorithm | str] = (
        HashAlgorithm.PDQ,
        HashAlgorithm.PHASH,
        HashAlgorithm.DHASH,
    ),
    pdq_hasher: PdqHasher | None = None,
) -> ImageFingerprint:
    """Computes several perceptual hashes of one image.

    Args:
        image: The decoded image.
        algorithms: Which hashes to compute (default: all three).
        pdq_hasher: The PDQ hasher to use; ``None`` creates
            ``PdqHasher()``. Pass a shared one when hashing many images.

    Returns:
        The fingerprint, with ``None`` for hashes not requested.

    Raises:
        TypeError: If ``image`` is not a Pillow image.
        ValueError: If an algorithm name is unknown or the image is empty.
        BackendUnavailableError: If pHash or dHash is requested and
            ``imagehash`` is not installed.
    """
    try:
        wanted = {HashAlgorithm(a) for a in algorithms}
    except ValueError:
        raise ValueError('algorithms must be among "pdq", "phash" and "dhash"') from None
    pdq_value = None
    if HashAlgorithm.PDQ in wanted:
        pdq_value = (pdq_hasher or PdqHasher()).hash_image(image)
    return ImageFingerprint(
        pdq=pdq_value,
        phash=phash(image) if HashAlgorithm.PHASH in wanted else None,
        dhash=dhash(image) if HashAlgorithm.DHASH in wanted else None,
    )


@dataclass(frozen=True, slots=True)
class MatchThresholds:
    """Thresholds for :func:`compare_fingerprints`.

    Attributes:
        pdq_max_distance: PDQ matches at Hamming distance ``<=`` this
            (Meta's recommendation: 31 of 256 bits).
        pdq_min_quality: PDQ is used only when both hashes have at least this
            quality (Meta's recommendation: 50).
        phash_max_distance: pHash matches at distance ``<=`` this (default 10
            of 64 bits; a starting point, calibrate it).
        dhash_max_distance: dHash matches at distance ``<=`` this (default 10
            of 64 bits; a starting point, calibrate it).
        min_bit_balance: pHash and dHash decide only when, on both sides, at
            least this fraction of the hash's bits are set and at least this
            fraction are clear (default :data:`BINARY_MIN_BALANCE`, 1/8: 8 of
            64 bits). Flat or smoothly shaded images give nearly all-zero or
            all-one hashes that are close to each other whatever the picture;
            ``imagehash``'s pHash of an image without ties always has half its
            bits set. 0 disables the check; at most 0.5.
        allow_phash_fallback: When both fingerprints have PDQ hashes but one
            is below ``pdq_min_quality``, let informative pHashes decide
            (see :func:`compare_fingerprints`). ``False`` leaves every such
            pair undecided (:attr:`UndecidedReason.LOW_QUALITY`).

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If a field is negative, a quality is above 100, or
            ``min_bit_balance`` is not between 0 and 0.5.
    """

    pdq_max_distance: int = PDQ_MATCH_DISTANCE
    pdq_min_quality: int = PDQ_MIN_QUALITY
    phash_max_distance: int = PHASH_MATCH_DISTANCE
    dhash_max_distance: int = DHASH_MATCH_DISTANCE
    min_bit_balance: float = BINARY_MIN_BALANCE
    allow_phash_fallback: bool = True

    def __post_init__(self) -> None:
        """Validates the fields."""
        for name in (
            "pdq_max_distance",
            "pdq_min_quality",
            "phash_max_distance",
            "dhash_max_distance",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int")
            if value < 0:
                raise ValueError(f"{name} must not be negative")
        if self.pdq_min_quality > 100:
            raise ValueError("pdq_min_quality must be at most 100")
        balance: object = self.min_bit_balance
        if isinstance(balance, bool) or not isinstance(balance, (int, float)):
            raise TypeError("min_bit_balance must be a number")
        if not 0.0 <= balance <= 0.5:  # also false for NaN
            raise ValueError("min_bit_balance must be between 0 and 0.5")
        if not isinstance(self.allow_phash_fallback, bool):
            raise TypeError("allow_phash_fallback must be a bool")


class UndecidedReason(enum.StrEnum):
    """Why :func:`compare_fingerprints` made no decision (``decided_by is None``)."""

    #: Both fingerprints have PDQ hashes and at least one is below
    #: ``pdq_min_quality``, and pHash could not decide in its place: a pHash
    #: is missing or uninformative (``min_bit_balance``: a flat or smoothly
    #: shaded image), the fallback is disabled, or the pHashes match while
    #: the PDQ hashes do not (unrelated minimal images can share a pHash).
    LOW_QUALITY = "low-quality"
    #: There is no PDQ pair, and every pHash / dHash pair has a hash below
    #: ``min_bit_balance`` (a flat or smoothly shaded image).
    LOW_INFORMATION = "low-information"
    #: The fingerprints have no hash type in common.
    NO_COMMON_HASH = "no-common-hash"


@dataclass(frozen=True, slots=True)
class FingerprintComparison:
    """The outcome of :func:`compare_fingerprints`.

    Attributes:
        pdq_distance: PDQ Hamming distance, or ``None`` if either side has no
            PDQ hash.
        pdq_reliable: Whether both PDQ hashes meet the quality threshold
            (``None`` without PDQ hashes).
        phash_distance: pHash distance, or ``None``.
        dhash_distance: dHash distance, or ``None``.
        decided_by: The hash that decided :attr:`matched`, or ``None`` when
            there was no decision (see :attr:`undecided_reason`).
        matched: The decision; ``False`` when there was none. When both
            fingerprints have PDQ hashes and both are reliable, PDQ decides.
            When they are not, pHash decides if both pHashes are informative
            (``min_bit_balance``): a pHash distance above
            ``phash_max_distance`` is no match, and a closer pHash is a
            match only if the PDQ distance is also within
            ``pdq_max_distance`` (otherwise there is no decision). Only when
            a side has no PDQ hash do pHash, then dHash decide, each only
            with informative hashes on both sides. Applications with another
            policy use the distances.
        undecided_reason: Why there was no decision, or ``None`` when there
            was one.
        low_quality_fallback: ``True`` when the PDQ pair was unreliable and
            pHash decided in its place (``decided_by`` is
            :attr:`HashAlgorithm.PHASH`, ``pdq_reliable`` is ``False``). This
            is weaker evidence than a reliable PDQ decision; applications
            that need PDQ-grade evidence treat it as undecided.
    """

    pdq_distance: int | None
    pdq_reliable: bool | None
    phash_distance: int | None
    dhash_distance: int | None
    decided_by: HashAlgorithm | None
    matched: bool
    undecided_reason: UndecidedReason | None = None
    low_quality_fallback: bool = False


def compare_fingerprints(
    a: ImageFingerprint,
    b: ImageFingerprint,
    *,
    thresholds: MatchThresholds | None = None,
) -> FingerprintComparison:
    """Compares two fingerprints hash by hash.

    The decision rule (see :class:`FingerprintComparison`):

    * with PDQ hashes on both sides, both reliable: PDQ decides;
    * with PDQ hashes on both sides, one unreliable (quality below
      ``pdq_min_quality``): pHash decides if both pHashes pass the
      ``min_bit_balance`` information check (and ``allow_phash_fallback``):
      no match above ``phash_max_distance``; a match at or below it if the
      PDQ distance is within ``pdq_max_distance`` as well, else no decision.
      The result is flagged ``low_quality_fallback``. Otherwise nothing
      decides (:attr:`UndecidedReason.LOW_QUALITY`). Measured on artwork-like
      images whose PDQ quality is below 50 (``docs/image.md``): JPEG, WebP
      and resized copies match (night scenes 100 %, light pencil sketches
      98 %, dark paintings 88 %, the rest undecided), while unrelated images
      of the same style match in at most 0.06 % of pairs (pHash alone: up to
      6 %);
    * without a PDQ pair: pHash and then dHash decide, each only if both of
      its hashes pass the information check.

    Args:
        a: One fingerprint.
        b: The other.
        thresholds: The thresholds; ``None`` uses ``MatchThresholds()``.

    Returns:
        The distances and the decision (see :class:`FingerprintComparison`).

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the pHash or dHash widths differ.
    """
    pair: tuple[object, object] = (a, b)
    if not all(isinstance(item, ImageFingerprint) for item in pair):
        raise TypeError("a and b must be ImageFingerprint instances")
    limits = thresholds if thresholds is not None else MatchThresholds()
    pdq_distance = pdq_reliable = None
    if a.pdq is not None and b.pdq is not None:
        pdq_distance = a.pdq.distance(b.pdq)
        pdq_reliable = a.pdq.is_reliable(limits.pdq_min_quality) and b.pdq.is_reliable(
            limits.pdq_min_quality
        )
    phash_distance = (
        a.phash.distance(b.phash) if a.phash is not None and b.phash is not None else None
    )
    dhash_distance = (
        a.dhash.distance(b.dhash) if a.dhash is not None and b.dhash is not None else None
    )
    decided_by: HashAlgorithm | None = None
    matched = fallback = False
    reason: UndecidedReason | None = None
    if pdq_distance is not None:
        if pdq_reliable:
            decided_by, matched = HashAlgorithm.PDQ, pdq_distance <= limits.pdq_max_distance
        elif (
            limits.allow_phash_fallback
            and phash_distance is not None
            and _informative(a.phash, b.phash, limits)
            and (
                phash_distance > limits.phash_max_distance
                or pdq_distance <= limits.pdq_max_distance
            )
        ):
            # pHash decides for the unreliable PDQ pair; a pHash match must
            # not be contradicted by PDQ (minimal images share pHashes).
            decided_by, fallback = HashAlgorithm.PHASH, True
            matched = phash_distance <= limits.phash_max_distance
        else:
            reason = UndecidedReason.LOW_QUALITY
    elif phash_distance is not None and _informative(a.phash, b.phash, limits):
        decided_by, matched = HashAlgorithm.PHASH, phash_distance <= limits.phash_max_distance
    elif dhash_distance is not None and _informative(a.dhash, b.dhash, limits):
        decided_by, matched = HashAlgorithm.DHASH, dhash_distance <= limits.dhash_max_distance
    elif phash_distance is not None or dhash_distance is not None:
        reason = UndecidedReason.LOW_INFORMATION
    else:
        reason = UndecidedReason.NO_COMMON_HASH
    return FingerprintComparison(
        pdq_distance=pdq_distance,
        pdq_reliable=pdq_reliable,
        phash_distance=phash_distance,
        dhash_distance=dhash_distance,
        decided_by=decided_by,
        matched=matched,
        undecided_reason=reason,
        low_quality_fallback=fallback,
    )


def _informative(a: BinaryHash | None, b: BinaryHash | None, limits: MatchThresholds) -> bool:
    """Whether both hashes have enough set and clear bits (``min_bit_balance``)."""
    for value in (a, b):
        if value is None:
            return False
        ones = value.value.bit_count()
        if min(ones, value.bits - ones) < limits.min_bit_balance * value.bits:
            return False
    return True


# ---------------------------------------------------------------- hash lists

_K = TypeVar("_K")

#: A hash given to :class:`HashList`: a :class:`PdqHash`, a
#: :class:`BinaryHash`, big-endian ``bytes``, a non-negative ``int``, or a hex
#: ``str``.
HashValue: TypeAlias = "PdqHash | BinaryHash | bytes | int | str"

_CHUNK_ROWS = 1 << 14
_POPCOUNT_TABLE = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


class _Scanner:
    """Per-query scratch buffers for the XOR / popcount scan (about 600 KB)."""

    def __init__(self, words_per_hash: int) -> None:
        self.xor = np.empty((_CHUNK_ROWS, words_per_hash), dtype=np.uint64)
        self.native = hasattr(np, "bitwise_count")  # NumPy >= 2.0
        width = words_per_hash if self.native else 8 * words_per_hash
        self.counts = np.empty((_CHUNK_ROWS, width), dtype=np.uint8)
        self.distances = np.empty(_CHUNK_ROWS, dtype=np.uint32)

    def distances_to(
        self, chunk: NDArray[np.uint64], query: NDArray[np.uint64]
    ) -> NDArray[np.uint32]:
        rows = chunk.shape[0]
        xor = self.xor[:rows]
        np.bitwise_xor(chunk, query, out=xor)
        counts = self.counts[:rows]
        if self.native:
            np.bitwise_count(xor, out=counts)
        else:
            np.take(_POPCOUNT_TABLE, xor.view(np.uint8), out=counts)
        # Adding columns one by one is several times faster than sum(axis=1)
        # on such narrow rows.
        distances = self.distances[:rows]
        distances[:] = counts[:, 0]
        for column in range(1, counts.shape[1]):
            distances += counts[:, column]
        return distances


def _popcount_rows(words: NDArray[np.uint64]) -> NDArray[np.int64]:
    """Population count of each row (used by tests to check the scanner)."""
    scanner = _Scanner(words.shape[1])
    zero = np.zeros(words.shape[1], dtype=np.uint64)
    parts = [
        scanner.distances_to(words[start : start + _CHUNK_ROWS], zero).astype(np.int64)
        for start in range(0, words.shape[0], _CHUNK_ROWS)
    ]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)


class HashList(Generic[_K]):
    """An in-memory list of known hashes, searched by Hamming distance.

    Built for lists shared by other organizations (for example PDQ hash lists
    of known material): every query compares against every entry with a
    vectorized XOR and population count, so results are exact (no
    approximate index). One PDQ query against 1,000,000 hashes takes about
    15 ms on one CPU core (NumPy 2; the NumPy 1 fallback uses a lookup table
    and is a few times slower).

    Memory: the hashes are packed into 64-bit words, ``8 * ceil(bits / 64)``
    bytes per entry (32 MB per million PDQ hashes), plus the keys as given
    (pass ``keys=range(n)`` to :meth:`from_packed` or :meth:`from_hex` for
    positional keys that cost nothing). A query allocates under 1 MB of
    scratch space regardless of the list size.

    Example::

        known = HashList.from_hex(partner_hex_lines, keys=partner_ids)
        for match in known.match(pdq_hasher.hash_image(image)):
            print(match.key, match.distance)

    Thread safety: immutable after construction; queries are thread-safe.

    Args:
        entries: ``(key, hash)`` pairs; see :data:`HashValue`.
        bits: The hash width (default 256, PDQ). Every hash must have it.

    Raises:
        TypeError: If a hash has an unsupported type.
        ValueError: If a hash has the wrong width or ``bits`` < 1.
    """

    def __init__(self, entries: Iterable[tuple[_K, HashValue]], *, bits: int = PDQ_BITS) -> None:
        """Packs the entries (see the class docstring)."""
        nbytes = _check_bits(bits)
        keys: list[_K] = []
        packed = bytearray()
        for key, value in entries:
            keys.append(key)
            packed += _hash_bytes(value, bits, nbytes)
        self._init(np.frombuffer(bytes(packed), dtype=np.uint8), keys, bits)

    @classmethod
    def from_packed(
        cls,
        data: bytes | bytearray | memoryview | NDArray[np.uint8],
        *,
        keys: Sequence[_K],
        bits: int = PDQ_BITS,
    ) -> HashList[_K]:
        """Builds a list from hashes packed back to back.

        Args:
            data: ``len(keys) * ceil(bits / 8)`` bytes: each hash big-endian
                (for PDQ, :attr:`PdqHash.value`), or a ``uint8`` array of
                shape ``(len(keys), ceil(bits / 8))`` (one hash per row) or
                of that many elements in one dimension.
            keys: One key per hash, in order. ``range(n)`` gives positional
                keys without storing them.
            bits: The hash width.

        Returns:
            The list.

        Raises:
            TypeError: If ``data`` is an array whose dtype is not ``uint8``.
            ValueError: If the sizes do not agree, or an array has another
                shape (a transposed ``(nbytes, n)`` array is refused rather
                than silently reinterpreted).
        """
        nbytes = _check_bits(bits)
        expected = len(keys) * nbytes
        if isinstance(data, np.ndarray):
            if data.dtype != np.uint8:
                raise TypeError(f"a packed hash array must be uint8, got {data.dtype}")
            if data.shape not in ((len(keys), nbytes), (expected,)):
                raise ValueError(
                    f"a packed hash array must have shape ({len(keys)}, {nbytes}) "
                    f"or ({expected},), got {data.shape}"
                )
            flat = np.ascontiguousarray(data).reshape(-1)
        else:
            flat = np.frombuffer(bytes(data), dtype=np.uint8)
        if flat.size != expected:
            raise ValueError(f"expected {len(keys)} hashes of {nbytes} bytes")
        instance = cls.__new__(cls)
        instance._init(flat, keys, bits)
        return instance

    @classmethod
    def from_hex(
        cls,
        hashes: Iterable[str],
        *,
        keys: Sequence[_K],
        bits: int = PDQ_BITS,
    ) -> HashList[_K]:
        """Builds a list from hex strings (for example one PDQ hash per line).

        Args:
            hashes: ``ceil(bits / 4)`` hex digits each, in the same order
                as ``keys``. Surrounding whitespace is ignored.
            keys: One key per hash.
            bits: The hash width.

        Returns:
            The list.

        Raises:
            ValueError: If a hash is malformed (the message gives its
                position) or the counts differ.
        """
        _check_bits(bits)
        packed = bytearray()
        count = 0
        for position, text in enumerate(hashes):
            try:
                item: object = text
                packed += _parse_hex(item.strip() if isinstance(item, str) else item, bits)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"hash at position {position}: {exc}") from None
            count += 1
        if count != len(keys):
            raise ValueError(f"got {count} hashes for {len(keys)} keys")
        return cls.from_packed(bytes(packed), keys=keys, bits=bits)

    def _init(self, flat: NDArray[np.uint8], keys: Sequence[_K], bits: int) -> None:
        nbytes = (bits + 7) // 8
        words = -(-nbytes // 8)
        count = len(keys)
        padded = np.zeros((count, words * 8), dtype=np.uint8)
        if count:
            padded[:, :nbytes] = flat.reshape(count, nbytes)
        self._words: NDArray[np.uint64] = padded.view(np.uint64)
        self._words.flags.writeable = False
        self._keys: Sequence[_K] = keys
        self._bits = bits
        self._nbytes = nbytes

    @property
    def bits(self) -> int:
        """The hash width in bits."""
        return self._bits

    @property
    def nbytes(self) -> int:
        """Bytes used by the packed hashes (keys not included)."""
        return int(self._words.nbytes)

    def __len__(self) -> int:
        """The number of hashes."""
        return int(self._words.shape[0])

    def __iter__(self) -> Iterator[_K]:
        """Iterates over the keys."""
        return iter(self._keys)

    def match(
        self,
        query: HashValue,
        *,
        max_distance: int | None = None,
        limit: int | None = None,
    ) -> list[HammingMatch[_K]]:
        """Finds the entries within ``max_distance`` bits of ``query``.

        A low-quality PDQ query (``not query.is_reliable()``) still runs; the
        caller decides whether to trust its matches.

        Args:
            query: The hash to look up, of this list's width.
            max_distance: Inclusive threshold, 0 to ``bits``. ``None`` means
                :data:`PDQ_MATCH_DISTANCE` (31) for 256-bit lists and is an
                error for other widths.
            limit: Return at most this many (the closest).

        Returns:
            The matches, closest first; ties keep list order. Empty when
            nothing is close enough.

        Raises:
            TypeError: If ``query`` has an unsupported type.
            ValueError: If the width or a parameter is wrong.
        """
        threshold = self._threshold(max_distance)
        if limit is not None:
            if isinstance(limit, bool) or not isinstance(limit, int):
                raise TypeError("limit must be an int")
            if limit < 1:
                raise ValueError("limit must be at least 1")
        query_words = self._query_words(query)
        scanner = _Scanner(self._words.shape[1])
        found_distances: list[NDArray[np.uint32]] = []
        found_rows: list[NDArray[np.int64]] = []
        for start in range(0, len(self), _CHUNK_ROWS):
            distances = scanner.distances_to(self._words[start : start + _CHUNK_ROWS], query_words)
            rows = np.flatnonzero(distances <= threshold)
            if rows.size:
                found_distances.append(distances[rows].copy())
                found_rows.append(rows + start)
        if not found_rows:
            return []
        all_distances = np.concatenate(found_distances)
        all_rows = np.concatenate(found_rows)
        order = np.lexsort((all_rows, all_distances))
        if limit is not None:
            order = order[:limit]
        return [
            HammingMatch(
                key=self._keys[int(all_rows[i])], distance=int(all_distances[i]), bits=self._bits
            )
            for i in order
        ]

    def nearest(self, query: HashValue, *, count: int = 1) -> list[HammingMatch[_K]]:
        """Returns the ``count`` closest entries, whatever their distance.

        Raises:
            TypeError: If ``query`` has an unsupported type.
            ValueError: If the width is wrong or ``count`` < 1.
        """
        return self.match(query, max_distance=self._bits, limit=count)

    def _threshold(self, max_distance: int | None) -> int:
        if max_distance is None:
            if self._bits != PDQ_BITS:
                raise ValueError("max_distance is required for hash lists that are not PDQ")
            return PDQ_MATCH_DISTANCE
        if isinstance(max_distance, bool) or not isinstance(max_distance, int):
            raise TypeError("max_distance must be an int")
        if not 0 <= max_distance <= self._bits:
            raise ValueError("max_distance must be between 0 and bits")
        return max_distance

    def _query_words(self, query: HashValue) -> NDArray[np.uint64]:
        raw = _hash_bytes(query, self._bits, self._nbytes)
        padded = np.zeros(self._words.shape[1] * 8, dtype=np.uint8)
        padded[: self._nbytes] = np.frombuffer(raw, dtype=np.uint8)
        return padded.view(np.uint64)


def _check_bits(bits: int) -> int:
    if isinstance(bits, bool) or not isinstance(bits, int):
        raise TypeError("bits must be an int")
    if bits < 1:
        raise ValueError("bits must be at least 1")
    return (bits + 7) // 8


def _hash_bytes(value: object, bits: int, nbytes: int) -> bytes:
    if isinstance(value, PdqHash):
        if bits != PDQ_BITS:
            raise ValueError(f"a PDQ hash does not fit a {bits}-bit list")
        return value.value
    if isinstance(value, BinaryHash):
        if value.bits != bits:
            raise ValueError(f"a {value.bits}-bit hash does not fit a {bits}-bit list")
        return value.to_bytes()
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        if len(raw) != nbytes:
            raise ValueError(f"expected {nbytes} bytes per hash, got {len(raw)}")
        if bits % 8 and raw[0] >> (bits % 8):
            raise ValueError(f"hash does not fit in {bits} bits")
        return raw
    if isinstance(value, bool):
        raise TypeError("bool is not a hash value")
    if isinstance(value, int):
        if value < 0 or value >> bits:
            raise ValueError(f"hash does not fit in {bits} unsigned bits")
        return value.to_bytes(nbytes, "big")
    if isinstance(value, str):
        return _parse_hex(value, bits)
    raise TypeError(f"unsupported hash type {type(value).__name__}")
