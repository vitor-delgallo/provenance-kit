"""Media type detection from the leading bytes ("magic numbers").

The type of untrusted media is always detected from its content, never from a
file name, an extension or a client-supplied MIME type. :func:`sniff_media_type`
recognizes the containers that provenance-kit's backends handle.

What it detects is the *container*. An MP4 file may hold only audio and an Ogg
file may hold video: a backend that cares must still probe the streams (for
example with ffprobe) after sniffing, and must still enforce
:class:`~provenance_kit.limits.MediaLimits` on what the probe reports.
"""

from __future__ import annotations

import enum
from collections.abc import Collection

from provenance_kit._typing import BytesLike
from provenance_kit.errors import UnsupportedMediaError

__all__ = [
    "SNIFF_BYTES",
    "MediaKind",
    "MediaType",
    "require_media_type",
    "sniff_media_type",
]

#: How many leading bytes :func:`sniff_media_type` looks at. Passing more is
#: harmless; passing fewer may fail to recognize some formats.
SNIFF_BYTES = 4096


class MediaKind(enum.StrEnum):
    """The broad category of a media type."""

    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class MediaType(enum.StrEnum):
    """A detected container format; the value is its MIME type."""

    JPEG = "image/jpeg"
    PNG = "image/png"
    GIF = "image/gif"
    WEBP = "image/webp"
    TIFF = "image/tiff"
    BMP = "image/bmp"
    HEIC = "image/heic"
    HEIF = "image/heif"
    AVIF = "image/avif"
    JXL = "image/jxl"
    MP4 = "video/mp4"
    QUICKTIME = "video/quicktime"
    WEBM = "video/webm"
    MATROSKA = "video/x-matroska"
    AVI = "video/x-msvideo"
    MPEG_TS = "video/mp2t"
    WAV = "audio/wav"
    MP3 = "audio/mpeg"
    AAC = "audio/aac"
    FLAC = "audio/flac"
    OGG = "audio/ogg"
    M4A = "audio/mp4"
    AIFF = "audio/aiff"

    @property
    def mime(self) -> str:
        """The MIME type, for example ``"image/jpeg"``."""
        return self.value

    @property
    def kind(self) -> MediaKind:
        """The broad category, from the MIME type's top-level type."""
        return MediaKind(self.value.split("/", 1)[0])


_HEIC_BRANDS = frozenset({b"heic", b"heix", b"heim", b"heis", b"hevc", b"hevx", b"hevm", b"hevs"})
_HEIF_BRANDS = frozenset({b"mif1", b"msf1", b"mif2"})
_AVIF_BRANDS = frozenset({b"avif", b"avis"})
_BMFF_IMAGE_BRANDS: tuple[tuple[frozenset[bytes], MediaType], ...] = (
    (_AVIF_BRANDS, MediaType.AVIF),
    (_HEIC_BRANDS, MediaType.HEIC),
    (_HEIF_BRANDS, MediaType.HEIF),
)
_BMFF_MAJOR = {
    b"qt  ": MediaType.QUICKTIME,
    b"M4A ": MediaType.M4A,
    b"M4B ": MediaType.M4A,
    b"M4P ": MediaType.M4A,
}
_BMFF_UNSUPPORTED_MAJOR = frozenset({b"crx "})  # Canon raw images
_BMP_DIB_SIZES = frozenset({12, 16, 40, 52, 56, 64, 108, 124})
_TS_PACKET = 188


def sniff_media_type(data: BytesLike) -> MediaType | None:
    """Detects the container format from the leading bytes.

    Args:
        data: The start of the media; at least :data:`SNIFF_BYTES` bytes are
            recommended (the whole content is fine, only the head is read).

    Returns:
        The detected type, or ``None`` if the bytes match no supported format.
        Never raises for any byte content.

    Raises:
        TypeError: If ``data`` is not bytes-like.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError(f"data must be bytes-like, got {type(data).__name__}")
    head = bytes(memoryview(data).cast("B")[:SNIFF_BYTES])
    return (
        _sniff_image(head)
        or _sniff_riff_family(head)
        or _sniff_iso_bmff(head)
        or _sniff_ebml(head)
        or _sniff_audio(head)
        or _sniff_mpeg_ts(head)
    )


def require_media_type(
    data: BytesLike,
    *,
    allowed: Collection[MediaType] | None = None,
    kinds: Collection[MediaKind] | None = None,
) -> MediaType:
    """Detects the media type and checks that it is acceptable.

    Args:
        data: The start of the media (see :func:`sniff_media_type`).
        allowed: The acceptable types, or ``None`` for any.
        kinds: The acceptable kinds, or ``None`` for any.

    Returns:
        The detected type.

    Raises:
        UnsupportedMediaError: If the type is not recognized or not accepted.
            The message names the detected MIME type only.
    """
    detected = sniff_media_type(data)
    if detected is None:
        raise UnsupportedMediaError("unrecognized media type")
    if allowed is not None and detected not in allowed:
        raise UnsupportedMediaError(f"media type {detected.mime} is not accepted here")
    if kinds is not None and detected.kind not in kinds:
        raise UnsupportedMediaError(
            f"{detected.kind.value} media ({detected.mime}) is not accepted here"
        )
    return detected


_IMAGE_SIGNATURES: tuple[tuple[tuple[bytes, ...], MediaType], ...] = (
    ((b"\xff\xd8\xff",), MediaType.JPEG),
    ((b"\x89PNG\r\n\x1a\n",), MediaType.PNG),
    ((b"GIF87a", b"GIF89a"), MediaType.GIF),
    ((b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"), MediaType.TIFF),
    ((b"\xff\x0a", b"\x00\x00\x00\x0cJXL \r\n\x87\n"), MediaType.JXL),
)


def _sniff_image(head: bytes) -> MediaType | None:
    for prefixes, media_type in _IMAGE_SIGNATURES:
        if head.startswith(prefixes):
            return media_type
    if (
        head.startswith(b"BM")
        and len(head) >= 18
        and head[6:10] == b"\x00\x00\x00\x00"
        and int.from_bytes(head[14:18], "little") in _BMP_DIB_SIZES
    ):
        return MediaType.BMP
    return None


def _sniff_riff_family(head: bytes) -> MediaType | None:
    if len(head) < 12:
        return None
    tag, form = head[:4], head[8:12]
    if tag == b"RIFF":
        return {b"WEBP": MediaType.WEBP, b"WAVE": MediaType.WAV, b"AVI ": MediaType.AVI}.get(form)
    if tag in {b"RF64", b"BW64"} and form == b"WAVE":
        return MediaType.WAV
    if tag == b"FORM" and form in {b"AIFF", b"AIFC"}:
        return MediaType.AIFF
    return None


def _sniff_iso_bmff(head: bytes) -> MediaType | None:
    if len(head) < 16 or head[4:8] != b"ftyp":
        return None
    box_size = int.from_bytes(head[:4], "big")
    if box_size < 16:  # 8-byte box header + major brand + minor version
        return None
    end = min(box_size, len(head))
    major = head[8:12]
    brands = {major} | {head[i : i + 4] for i in range(16, end - 3, 4)}
    # Image brands win: HEIF/AVIF files also list generic brands.
    for image_brands, media_type in _BMFF_IMAGE_BRANDS:
        if brands & image_brands:
            return media_type
    if major in _BMFF_UNSUPPORTED_MAJOR:
        return None
    return _BMFF_MAJOR.get(major, MediaType.MP4)


def _sniff_ebml(head: bytes) -> MediaType | None:
    if not head.startswith(b"\x1a\x45\xdf\xa3"):
        return None
    # The DocType element (ID 0x4282) follows in the EBML header.
    window = head[4:64]
    if b"\x42\x82" in window:
        if b"webm" in window:
            return MediaType.WEBM
        if b"matroska" in window:
            return MediaType.MATROSKA
    return None


def _sniff_audio(head: bytes) -> MediaType | None:
    if head.startswith(b"fLaC"):
        return MediaType.FLAC
    if head.startswith(b"OggS"):
        return MediaType.OGG
    if head.startswith(b"ID3"):
        return MediaType.MP3
    if len(head) >= 3 and head[0] == 0xFF:
        second, third = head[1], head[2]
        if second & 0xF6 == 0xF0:  # ADTS: 12-bit sync, layer bits 00
            return MediaType.AAC
        layer = (second >> 1) & 0b11
        bitrate = third >> 4
        sample_rate = (third >> 2) & 0b11
        if second & 0xE0 == 0xE0 and layer != 0 and bitrate != 0xF and sample_rate != 0b11:
            return MediaType.MP3
    return None


def _sniff_mpeg_ts(head: bytes) -> MediaType | None:
    packets = min(len(head) // _TS_PACKET, 3)
    if packets >= 2 and all(head[i * _TS_PACKET] == 0x47 for i in range(packets)):
        return MediaType.MPEG_TS
    return None
