"""Media types accepted by the C2PA module and their detection (internal)."""

from __future__ import annotations

from provenance_kit._logging import get_logger
from provenance_kit.errors import UnsupportedMediaError
from provenance_kit.mediatype import SNIFF_BYTES, MediaType, sniff_media_type

__all__ = ["SUPPORTED_MEDIA_TYPES", "parse_media_type", "resolve_media_type"]

_log = get_logger(__name__)

#: The media types that :func:`~provenance_kit.provenance.c2pa.sign_asset` and
#: :func:`~provenance_kit.provenance.c2pa.read_manifest` accept: the containers
#: that :mod:`provenance_kit.mediatype` detects and that c2pa-python 0.38
#: (c2pa-rs 0.91) can both sign and read. The test suite checks this set
#: against the library's own ``Builder.get_supported_mime_types()`` and
#: ``Reader.get_supported_mime_types()``.
SUPPORTED_MEDIA_TYPES: frozenset[MediaType] = frozenset(
    {
        MediaType.JPEG,
        MediaType.PNG,
        MediaType.GIF,
        MediaType.WEBP,
        MediaType.TIFF,
        MediaType.HEIC,
        MediaType.HEIF,
        MediaType.AVIF,
        MediaType.JXL,
        MediaType.MP4,
        MediaType.QUICKTIME,
        MediaType.AVI,
        MediaType.WAV,
        MediaType.MP3,
        MediaType.FLAC,
        MediaType.M4A,
    }
)

# ISO BMFF brands share one c2pa-rs asset handler, so a declared type from
# this group is accepted for content sniffed as any other member.
_BMFF = frozenset(
    {
        MediaType.MP4,
        MediaType.QUICKTIME,
        MediaType.M4A,
        MediaType.HEIC,
        MediaType.HEIF,
        MediaType.AVIF,
    }
)

# Alternative spellings that c2pa-python lists as supported.
_ALIASES = {
    "image/jpg": MediaType.JPEG,
    "image/x-webp": MediaType.WEBP,
    "audio/wave": MediaType.WAV,
    "audio/x-wav": MediaType.WAV,
    "audio/vnd.wave": MediaType.WAV,
    "audio/mp3": MediaType.MP3,
    "audio/x-mp3": MediaType.MP3,
    "audio/mpeg3": MediaType.MP3,
    "audio/x-flac": MediaType.FLAC,
    "audio/x-m4a": MediaType.M4A,
    "video/avi": MediaType.AVI,
    "video/msvideo": MediaType.AVI,
    "application/mp4": MediaType.MP4,
}


def parse_media_type(value: str | MediaType) -> MediaType:
    """Parses a declared MIME type.

    Args:
        value: A :class:`MediaType` or a MIME string; parameters after ``;``
            and letter case are ignored.

    Returns:
        The media type.

    Raises:
        TypeError: If ``value`` is not a str.
        UnsupportedMediaError: If the type is not one the module accepts.
    """
    if isinstance(value, MediaType):
        media_type = value
    else:
        if not isinstance(value, str):
            raise TypeError(f"mime_type must be a str, got {type(value).__name__}")
        key = value.split(";", 1)[0].strip().lower()
        try:
            media_type = MediaType(key)
        except ValueError:
            alias = _ALIASES.get(key)
            if alias is None:
                raise UnsupportedMediaError(
                    "the declared media type is not supported for C2PA"
                ) from None
            media_type = alias
    if media_type not in SUPPORTED_MEDIA_TYPES:
        raise UnsupportedMediaError(f"media type {media_type.mime} is not supported for C2PA")
    return media_type


def resolve_media_type(data: bytes, declared: str | MediaType | None) -> MediaType:
    """Detects the container from the bytes and reconciles it with a declared type.

    The bytes decide: the declared type is accepted only when it names the
    detected container (or another ISO BMFF brand, which c2pa-rs handles
    alike). A FLAC stream behind an ID3v2 tag, which is how c2pa-rs embeds
    manifests in FLAC, is detected as FLAC rather than MP3.

    Args:
        data: The media.
        declared: The caller's MIME type, or ``None``.

    Returns:
        The declared type when given, else the detected one.

    Raises:
        TypeError: If ``declared`` is neither a str, a MediaType nor ``None``.
        UnsupportedMediaError: If the content is not recognized or not
            supported, or does not match the declared type.
    """
    sniffed = sniff_media_type(data)
    if sniffed is None:
        raise UnsupportedMediaError("unrecognized media type")
    if sniffed is MediaType.MP3 and _id3_wrapped_type(data) is MediaType.FLAC:
        sniffed = MediaType.FLAC
    if sniffed not in SUPPORTED_MEDIA_TYPES:
        raise UnsupportedMediaError(f"media type {sniffed.mime} is not supported for C2PA")
    if declared is None:
        return sniffed
    media_type = parse_media_type(declared)
    if media_type is not sniffed and not (media_type in _BMFF and sniffed in _BMFF):
        _log.debug("declared %s, detected %s", media_type.mime, sniffed.mime)
        raise UnsupportedMediaError(
            f"the declared media type {media_type.mime} does not match the content ({sniffed.mime})"
        )
    return media_type


def _id3_wrapped_type(data: bytes) -> MediaType | None:
    """Sniffs what follows a leading ID3v2 tag, if there is one."""
    if len(data) < 10 or data[:3] != b"ID3":
        return None
    size_bytes = data[6:10]
    if any(byte & 0x80 for byte in size_bytes):  # synchsafe integers have bit 7 clear
        return None
    size = 0
    for byte in size_bytes:
        size = (size << 7) | byte
    footer = 10 if data[5] & 0x10 else 0
    start = 10 + size + footer
    if start >= len(data):
        return None
    return sniff_media_type(data[start : start + SNIFF_BYTES])
