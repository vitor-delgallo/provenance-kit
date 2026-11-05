"""Safe decoding of untrusted image bytes.

:func:`load_image` turns encoded bytes from an untrusted source into a
:class:`PIL.Image.Image` in a predictable mode, and refuses anything that is
too large before decoding pixels:

1. ``limits.check_bytes`` on the input length.
2. The type comes from the leading bytes
   (:func:`provenance_kit.mediatype.require_media_type`), never from a name.
   Only the Pillow plugin for that type parses the bytes: a file that sniffs
   as PNG is never handed to the TIFF parser.
3. The header is parsed without decoding pixels; then the dimensions
   (``limits.check_dimensions``, also for the rotated size when EXIF says
   so) and, for animated images, the frame count are checked. Pillow's own
   process-global decompression-bomb check (``Image.MAX_IMAGE_PIXELS``) is
   not involved: these limits trigger first and are per call.
4. Pixels are decoded. Any decoder failure (truncated data, corrupt
   streams, a Pillow bug triggered by crafted input) becomes
   :class:`~provenance_kit.errors.MediaDecodeError`.
5. The EXIF orientation is applied, the ICC profile is kept on the result
   (and optionally applied, ``to_srgb=True``), and the mode is normalized to
   8-bit ``RGB`` or ``RGBA`` (:class:`ImageMode`). 16-bit images are scaled
   to 8 bits.

Formats (:data:`DEFAULT_IMAGE_TYPES`): JPEG, PNG (including APNG), GIF,
WebP and AVIF (when the installed Pillow has AVIF support, as the official
Pillow 11.3+ wheels do). TIFF and BMP decode when passed in ``allowed``; they
are off by default to keep the parsers exposed to strangers few.

**HEIC/HEIF** (common for phone uploads) is not decoded by Pillow. No
permissively licensed decoder package is available (``pillow-heif``'s
binary wheels are GPL-2.0 as a whole because they bundle the x265 encoder;
the LGPL-only ``pi-heif`` is discontinued and ships a libheif with a known
heap overflow; see ``docs/THIRD_PARTY.md``). Applications that accept a
decoder's license pass it through ``openers``::

    import pillow_heif  # installed and license-reviewed by the application

    loaded = load_image(
        data,
        openers={
            MediaType.HEIC: pillow_heif.HeifImageFile,
            MediaType.HEIF: pillow_heif.HeifImageFile,
        },
    )

An opener receives a binary stream and returns a Pillow image whose header is
parsed but whose pixels are not decoded yet (a Pillow ``ImageFile`` class
works); the limits are then checked exactly as for built-in formats.

**Animated images** (GIF, APNG, animated WebP, AVIF sequences):
:func:`load_image` returns the first frame; :func:`iter_frames` yields every
frame as it is displayed (after disposal and blending), one at a time. The
declared frame count is checked against ``max_frames`` before any frame is
decoded (default :data:`DEFAULT_MAX_IMAGE_FRAMES`; ``MediaLimits.max_frames``
is sized for video and is also applied). Counting GIF frames scans the whole
file, which is linear in its size.

Thread safety: the functions are thread-safe; they share no state.
"""

from __future__ import annotations

import enum
import io
from collections.abc import Callable, Collection, Iterator, Mapping
from dataclasses import dataclass
from typing import IO

from PIL import (
    BmpImagePlugin,
    GifImagePlugin,
    Image,
    ImageCms,
    JpegImagePlugin,
    PngImagePlugin,
    TiffImagePlugin,
    WebPImagePlugin,
    features,
)

from provenance_kit._logging import get_logger
from provenance_kit._typing import BytesLike
from provenance_kit.errors import (
    MediaDecodeError,
    MediaLimitExceededError,
    UnsupportedMediaError,
)
from provenance_kit.image import _modes
from provenance_kit.limits import MediaLimits
from provenance_kit.mediatype import MediaKind, MediaType, require_media_type

__all__ = [
    "DEFAULT_IMAGE_TYPES",
    "DEFAULT_MAX_IMAGE_FRAMES",
    "ImageInfo",
    "ImageMode",
    "ImageOpener",
    "LoadedImage",
    "iter_frames",
    "load_image",
    "probe_image",
]

_log = get_logger(__name__)

#: The image types accepted when ``allowed`` is not given.
DEFAULT_IMAGE_TYPES: frozenset[MediaType] = frozenset(
    {MediaType.JPEG, MediaType.PNG, MediaType.GIF, MediaType.WEBP, MediaType.AVIF}
)

#: Default cap on the frames of an animated image: about 3 minutes at
#: 25 frames per second, far above ordinary animated GIFs and stickers.
DEFAULT_MAX_IMAGE_FRAMES = 5_000

#: Builds a Pillow image from a binary stream, parsing only the header.
ImageOpener = Callable[[IO[bytes]], Image.Image]

_BUILTIN_OPENERS: dict[MediaType, ImageOpener] = {
    MediaType.JPEG: JpegImagePlugin.JpegImageFile,
    MediaType.PNG: PngImagePlugin.PngImageFile,
    MediaType.GIF: GifImagePlugin.GifImageFile,
    MediaType.TIFF: TiffImagePlugin.TiffImageFile,
    MediaType.BMP: BmpImagePlugin.BmpImageFile,
}
if features.check("webp"):
    _BUILTIN_OPENERS[MediaType.WEBP] = WebPImagePlugin.WebPImageFile
try:  # Pillow 11.2+; decoding needs a build with libavif (official wheels 11.3+)
    from PIL import AvifImagePlugin
except ImportError:  # pragma: no cover - depends on the Pillow version
    pass
else:
    if getattr(AvifImagePlugin, "SUPPORTED", False):
        _BUILTIN_OPENERS[MediaType.AVIF] = AvifImagePlugin.AvifImageFile

_ORIENTATION_TAG = 0x0112
_TRANSPOSE: dict[int, Image.Transpose] = {
    2: Image.Transpose.FLIP_LEFT_RIGHT,
    3: Image.Transpose.ROTATE_180,
    4: Image.Transpose.FLIP_TOP_BOTTOM,
    5: Image.Transpose.TRANSPOSE,
    6: Image.Transpose.ROTATE_270,
    7: Image.Transpose.TRANSVERSE,
    8: Image.Transpose.ROTATE_90,
}


class ImageMode(enum.StrEnum):
    """The pixel layout :func:`load_image` returns."""

    #: ``RGBA`` when the image has an alpha channel or a transparent palette
    #: entry, else ``RGB``.
    AUTO = "auto"
    #: Always ``RGB``; alpha is dropped (colour values are kept as stored).
    RGB = "rgb"
    #: Always ``RGBA``; opaque images get a constant 255 alpha.
    RGBA = "rgba"


@dataclass(frozen=True, slots=True)
class ImageInfo:
    """What the header says about an image (nothing here is decoded pixels).

    Attributes:
        media_type: The type detected from the bytes.
        width: Stored width in pixels (before applying the orientation).
        height: Stored height in pixels.
        mode: Pillow's mode for the stored pixels, for example ``"RGB"``,
            ``"P"``, ``"CMYK"`` or ``"I;16"``.
        frame_count: Number of frames (1 for still images).
        orientation: EXIF orientation, 1 to 8 (1 when absent or invalid).
        has_alpha: Whether the image has alpha or palette transparency.
        icc_profile: The embedded ICC profile bytes, if any. Untrusted.
    """

    media_type: MediaType
    width: int
    height: int
    mode: str
    frame_count: int
    orientation: int
    has_alpha: bool
    icc_profile: bytes | None

    @property
    def oriented_size(self) -> tuple[int, int]:
        """``(width, height)`` after the EXIF orientation is applied."""
        if self.orientation in (5, 6, 7, 8):
            return self.height, self.width
        return self.width, self.height


@dataclass(frozen=True, slots=True)
class LoadedImage:
    """A decoded image and what was learned while decoding it.

    Attributes:
        image: An independent, fully decoded image in ``RGB`` or ``RGBA``,
            orientation applied.
        info: The header information.
        frame_index: Which frame this is (0 for still images).
        duration_ms: The frame's display duration, for animated images.
        srgb_converted: Whether the ICC profile was applied (``to_srgb=True``
            and the conversion succeeded).
    """

    image: Image.Image
    info: ImageInfo
    frame_index: int = 0
    duration_ms: float | None = None
    srgb_converted: bool = False


def probe_image(
    data: BytesLike,
    *,
    limits: MediaLimits | None = None,
    allowed: Collection[MediaType] | None = None,
    openers: Mapping[MediaType, ImageOpener] | None = None,
    max_frames: int | None = None,
) -> ImageInfo:
    """Parses and checks an image header without decoding pixels.

    Args:
        data: The encoded image (untrusted).
        limits: Resource limits; ``None`` uses ``MediaLimits()``.
        allowed: Accepted types; ``None`` uses :data:`DEFAULT_IMAGE_TYPES`
            plus the types in ``openers``.
        openers: Extra decoders by type (see the module docstring).
        max_frames: Frame-count cap; ``None`` uses
            :data:`DEFAULT_MAX_IMAGE_FRAMES`.

    Returns:
        The header information.

    Raises:
        TypeError: If ``data`` is not bytes-like.
        ValueError: If ``max_frames`` < 1.
        UnsupportedMediaError: If the type is unrecognized or not accepted.
        MediaLimitExceededError: If a limit is exceeded.
        MediaDecodeError: If the header cannot be parsed.
    """
    raw, checked = _prepare(data, limits, allowed, openers, max_frames)
    image, info = _open_checked(raw, *checked)
    image.close()
    return info


def load_image(
    data: BytesLike,
    *,
    limits: MediaLimits | None = None,
    mode: ImageMode | str = ImageMode.AUTO,
    allowed: Collection[MediaType] | None = None,
    openers: Mapping[MediaType, ImageOpener] | None = None,
    max_frames: int | None = None,
    to_srgb: bool = False,
) -> LoadedImage:
    """Decodes an untrusted image (the first frame of an animated one).

    Args:
        data: The encoded image (untrusted).
        limits: Resource limits; ``None`` uses ``MediaLimits()``.
        mode: The output layout (:class:`ImageMode`).
        allowed: Accepted types; ``None`` uses :data:`DEFAULT_IMAGE_TYPES`
            plus the types in ``openers``.
        openers: Extra decoders by type, for example for HEIC.
        max_frames: Frame-count cap; ``None`` uses
            :data:`DEFAULT_MAX_IMAGE_FRAMES`.
        to_srgb: Convert to sRGB with the embedded ICC profile (LittleCMS).
            Without a profile, or if the profile cannot be used, the pixels
            are converted without colour management and
            :attr:`LoadedImage.srgb_converted` is false.

    Returns:
        The decoded image.

    Raises:
        TypeError: If ``data`` is not bytes-like.
        ValueError: If ``mode`` is unknown or ``max_frames`` < 1.
        UnsupportedMediaError: If the type is unrecognized or not accepted.
        MediaLimitExceededError: If a limit is exceeded.
        MediaDecodeError: If the image cannot be decoded.
    """
    output_mode = _check_mode(mode)
    raw, checked = _prepare(data, limits, allowed, openers, max_frames)
    image, info = _open_checked(raw, *checked)
    try:
        return _finish(image, info, 0, output_mode, to_srgb, original=image)
    finally:
        image.close()


def iter_frames(
    data: BytesLike,
    *,
    limits: MediaLimits | None = None,
    mode: ImageMode | str = ImageMode.AUTO,
    allowed: Collection[MediaType] | None = None,
    openers: Mapping[MediaType, ImageOpener] | None = None,
    max_frames: int | None = None,
    to_srgb: bool = False,
) -> Iterator[LoadedImage]:
    """Decodes every frame of an untrusted image, one at a time.

    Still images yield one frame. Frames come out as displayed (Pillow
    applies GIF disposal and APNG/WebP blending), each as an independent
    image, so memory stays at about two frames whatever the frame count. The
    declared count is checked before the first frame is decoded; stop
    iterating to decode fewer frames.

    Args:
        data: The encoded image (untrusted).
        limits: Resource limits; ``None`` uses ``MediaLimits()``.
        mode: The output layout (:class:`ImageMode`).
        allowed: Accepted types; ``None`` uses :data:`DEFAULT_IMAGE_TYPES`
            plus the types in ``openers``.
        openers: Extra decoders by type.
        max_frames: Frame-count cap; ``None`` uses
            :data:`DEFAULT_MAX_IMAGE_FRAMES`.
        to_srgb: Apply the ICC profile (see :func:`load_image`).

    Yields:
        One :class:`LoadedImage` per frame, in order.

    Raises:
        TypeError: If ``data`` is not bytes-like.
        ValueError: If ``mode`` is unknown or ``max_frames`` < 1.
        UnsupportedMediaError: If the type is unrecognized or not accepted.
        MediaLimitExceededError: If a limit is exceeded (also for a frame
            that turns out larger than the header said).
        MediaDecodeError: If a frame cannot be decoded.
    """
    output_mode = _check_mode(mode)
    raw, checked = _prepare(data, limits, allowed, openers, max_frames)
    image, info = _open_checked(raw, *checked)
    return _frames(image, info, checked[1], output_mode, to_srgb)


# ---------------------------------------------------------------- internals

_Checked = tuple[MediaType, MediaLimits, int, Mapping[MediaType, ImageOpener]]


def _frames(
    image: Image.Image,
    info: ImageInfo,
    limits: MediaLimits,
    output_mode: ImageMode,
    to_srgb: bool,
) -> Iterator[LoadedImage]:
    try:
        for index in range(info.frame_count):
            if index:
                try:
                    image.seek(index)
                except EOFError:
                    return  # fewer frames than declared
                except (MemoryError, RecursionError):
                    raise
                except Exception as exc:
                    raise MediaDecodeError(f"{info.media_type.mime} frame is corrupt") from exc
                limits.check_dimensions(*image.size)
            yield _finish(image, info, index, output_mode, to_srgb, original=image)
    finally:
        image.close()


def _check_mode(mode: ImageMode | str) -> ImageMode:
    try:
        return ImageMode(mode)
    except ValueError:
        raise ValueError('mode must be "auto", "rgb" or "rgba"') from None


def _prepare(
    data: BytesLike,
    limits: MediaLimits | None,
    allowed: Collection[MediaType] | None,
    openers: Mapping[MediaType, ImageOpener] | None,
    max_frames: int | None,
) -> tuple[bytes, _Checked]:
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError(f"data must be bytes-like, got {type(data).__name__}")
    limits = limits if limits is not None else MediaLimits()
    if not isinstance(limits, MediaLimits):
        raise TypeError("limits must be a MediaLimits")
    frame_cap = DEFAULT_MAX_IMAGE_FRAMES if max_frames is None else max_frames
    if isinstance(frame_cap, bool) or not isinstance(frame_cap, int):
        raise TypeError("max_frames must be an int")
    if frame_cap < 1:
        raise ValueError("max_frames must be at least 1")
    extra = dict(openers) if openers else {}
    entries: list[tuple[object, object]] = list(extra.items())
    for key, opener in entries:
        if not isinstance(key, MediaType) or not callable(opener):
            raise TypeError("openers must map MediaType values to callables")
    limits.check_bytes(memoryview(data).nbytes)
    raw = bytes(data)
    accepted = frozenset(allowed) if allowed is not None else DEFAULT_IMAGE_TYPES | extra.keys()
    media_type = require_media_type(raw, kinds={MediaKind.IMAGE})
    if media_type not in accepted:
        raise UnsupportedMediaError(f"media type {media_type.mime} is not accepted here")
    return raw, (media_type, limits, frame_cap, extra)


def _builtin_opener(media_type: MediaType) -> ImageOpener:
    # Pillow's plugin classes parse the header in their constructor. Using
    # them directly selects exactly one parser and skips Image.open's global
    # MAX_IMAGE_PIXELS check, so the caller's limits decide.
    opener = _BUILTIN_OPENERS.get(media_type)
    if opener is not None:
        return opener
    if media_type is MediaType.WEBP:
        raise UnsupportedMediaError("this Pillow build cannot decode WebP")
    if media_type is MediaType.AVIF:
        raise UnsupportedMediaError(
            "this Pillow build cannot decode AVIF (Pillow 11.3 or later wheels can)"
        )
    raise UnsupportedMediaError(
        f"no decoder for {media_type.mime}: pass one through the openers argument"
    )


def _open_checked(
    raw: bytes,
    media_type: MediaType,
    limits: MediaLimits,
    frame_cap: int,
    extra: Mapping[MediaType, ImageOpener],
) -> tuple[Image.Image, ImageInfo]:
    opener = extra.get(media_type) or _builtin_opener(media_type)
    try:
        image = opener(io.BytesIO(raw))
    except (MemoryError, RecursionError):
        raise
    except Exception as exc:
        raise MediaDecodeError(f"{media_type.mime} header could not be parsed") from exc
    if not isinstance(image, Image.Image):
        raise TypeError("an image opener must return a PIL.Image.Image")
    try:
        width, height = image.size
        limits.check_dimensions(width, height)
        if width < 1 or height < 1:
            raise MediaDecodeError(f"{media_type.mime} declares an empty image")
        frame_count = _frame_count(image, media_type, frame_cap)
        limits.check_frames(frame_count)
        orientation = _orientation(image)
        if orientation in (5, 6, 7, 8):
            limits.check_dimensions(height, width)
        icc = image.info.get("icc_profile")
        info = ImageInfo(
            media_type=media_type,
            width=width,
            height=height,
            mode=image.mode,
            frame_count=frame_count,
            orientation=orientation,
            has_alpha=_modes.has_alpha(image),
            icc_profile=bytes(icc) if isinstance(icc, (bytes, bytearray)) and icc else None,
        )
    except BaseException:
        image.close()
        raise
    return image, info


def _frame_count(image: Image.Image, media_type: MediaType, frame_cap: int) -> int:
    try:
        count = int(getattr(image, "n_frames", 1))
    except (MemoryError, RecursionError):
        raise
    except Exception as exc:
        raise MediaDecodeError(f"{media_type.mime} frames could not be counted") from exc
    if count > frame_cap:
        raise MediaLimitExceededError("max_frames", frame_cap, count)
    return max(count, 1)


def _orientation(image: Image.Image) -> int:
    try:
        value = image.getexif().get(_ORIENTATION_TAG)
    except (MemoryError, RecursionError):
        raise
    except Exception:
        # Corrupt metadata must not block the pixels: treat as unrotated.
        _log.debug("ignoring unreadable EXIF metadata")
        return 1
    return value if isinstance(value, int) and value in _TRANSPOSE else 1


def _finish(
    frame: Image.Image,
    info: ImageInfo,
    index: int,
    output_mode: ImageMode,
    to_srgb: bool,
    *,
    original: Image.Image,
) -> LoadedImage:
    try:
        frame.load()
    except (MemoryError, RecursionError):
        raise
    except Exception as exc:
        raise MediaDecodeError(f"{info.media_type.mime} pixel data could not be decoded") from exc
    duration = frame.info.get("duration") if info.frame_count > 1 else None
    try:
        result = frame
        converted = False
        if to_srgb and info.icc_profile:
            result, converted = _apply_icc(result, info.icc_profile)
        keep_alpha = output_mode is ImageMode.RGBA or (
            output_mode is ImageMode.AUTO and _modes.has_alpha(result)
        )
        result = _modes.to_rgba(result) if keep_alpha else _modes.to_rgb(result)
        if info.orientation in _TRANSPOSE:
            result = result.transpose(_TRANSPOSE[info.orientation])
        if result is original:
            result = result.copy()
    except (MemoryError, RecursionError):
        raise
    except Exception as exc:
        raise MediaDecodeError(f"{info.media_type.mime} pixels could not be converted") from exc
    result.info.pop("exif", None)  # the orientation is applied; drop the stale tag
    return LoadedImage(
        image=result,
        info=info,
        frame_index=index,
        duration_ms=float(duration) if isinstance(duration, (int, float)) else None,
        srgb_converted=converted,
    )


def _apply_icc(image: Image.Image, icc: bytes) -> tuple[Image.Image, bool]:
    work = image
    if work.mode not in ("RGB", "RGBA", "CMYK", "L"):
        work = _modes.to_rgba(work) if _modes.has_alpha(work) else _modes.to_rgb(work)
    output = "RGBA" if work.mode == "RGBA" else "RGB"
    try:
        source = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        target = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
        converted = ImageCms.profileToProfile(work, source, target, outputMode=output)
    except (MemoryError, RecursionError):
        raise
    except Exception:
        _log.debug("ICC profile could not be applied; converting without colour management")
        return image, False
    if converted is None:  # pragma: no cover - only with inPlace=True
        return image, False
    return converted, True
