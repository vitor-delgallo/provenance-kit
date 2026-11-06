"""Resource limits for untrusted media.

Every function that reads media accepts a ``limits: MediaLimits | None``
argument (``None`` means ``MediaLimits()``, the defaults below). The limits are
checked **before** full decoding whenever the value can be known from a header
or a probe, so that a small file that declares a huge decoded size (a
decompression bomb) is refused cheaply.

The defaults are deliberately conservative for media uploaded by strangers.
Applications that process trusted or larger media should build their own
instance, for example ``dataclasses.replace(MediaLimits(), max_bytes=2**31)``.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

from provenance_kit.errors import MediaLimitExceededError

__all__ = ["MediaLimits"]

_MIB = 1024 * 1024


@dataclass(frozen=True, slots=True)
class MediaLimits:
    """Upper bounds applied to one media item.

    Each default and the reason for it:

    * ``max_bytes`` = 256 MiB. Bounds the encoded input that a backend may hold
      in memory or pipe into a tool. It covers high-resolution photos (rarely
      above 100 MiB) and several minutes of high-bitrate video, while keeping a
      single request from exhausting a worker's memory.
    * ``max_pixels`` = 64,000,000. Bounds the decoded image or frame. It stays
      below Pillow's decompression-bomb threshold (about 89.5 million pixels),
      covers 50-megapixel phone sensors and 61-megapixel full-frame cameras,
      and caps an RGB float32 tensor at about 770 MB, which is what the
      watermarking models allocate.
    * ``max_width`` / ``max_height`` = 16,384. Refuses degenerate shapes (for
      example 1 x 60,000,000) that pass the pixel limit but break resampling
      and model tiling. 16,384 is the WebP and common GPU texture ceiling and
      fits 16K panoramas.
    * ``max_duration_seconds`` = 10,800 (3 hours). Bounds the decode work for
      video and audio. It covers feature films and long recordings.
    * ``max_frames`` = 648,000 (3 hours at 60 frames per second). Bounds the
      number of frames one operation decodes or extracts, which matters when a
      container declares an absurd frame rate for a short duration.
    * ``max_image_frames`` = 5,000. Bounds the frames of one animated image
      (GIF, APNG, animated WebP, AVIF sequences): about 3 minutes at 25 frames
      per second, far above ordinary animated GIFs and stickers. Each frame is
      a full canvas once composited, so the video-sized ``max_frames`` would
      let a small file demand hours of decoding.
    * ``max_audio_channels`` = 8. Covers 7.1 surround; refuses channel-count
      bombs that multiply memory use.
    * ``max_sample_rate`` = 192,000 Hz. The highest rate in common studio use;
      refuses declared rates that inflate decoded size.
    * ``max_decoded_audio_bytes`` = 1 GiB. Bounds the decoded PCM that one
      call holds in memory: duration x output sample rate x output channels
      x bytes per sample (4 for the float32 of
      :func:`~provenance_kit.audio.pcm.decode_pcm`). The duration, rate and
      channel limits above are separate and multiply to about 66 GB (3 hours
      of 8-channel 192 kHz float32), which a 16 KB FLAC of silence can
      request: compressed silence decodes at ratios above 7,000. 1 GiB holds
      46 minutes of 48 kHz stereo, 3 hours of 16 kHz mono (what the
      AudioSeal models use) and is in line with the ~770 MB that
      ``max_pixels`` allows for an image tensor. Decode long media at a
      lower rate or channel count, or raise the limit. ``decode_pcm``'s
      peak is this plus a few MiB of read buffers.
    * ``subprocess_timeout_seconds`` = 600 (10 minutes). Bounds one external
      tool invocation (ffmpeg, ffprobe, fpcalc) so that a hung or pathological
      decode is killed. Long videos processed in one call may need more.

    Raises:
        TypeError: If a field has the wrong type (``bool`` is not accepted as
            an integer).
        ValueError: If a field is not a positive, finite number.
    """

    max_bytes: int = 256 * _MIB
    max_pixels: int = 64_000_000
    max_width: int = 16_384
    max_height: int = 16_384
    max_duration_seconds: float = 3 * 3600.0
    max_frames: int = 648_000
    max_image_frames: int = 5_000
    max_audio_channels: int = 8
    max_sample_rate: int = 192_000
    max_decoded_audio_bytes: int = 1024 * _MIB
    subprocess_timeout_seconds: float = 600.0

    def __post_init__(self) -> None:
        """Validates every field."""
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if field.type in (float, "float"):
                _check_positive_float(field.name, value)
            else:
                _check_positive_int(field.name, value)

    # ------------------------------------------------------------ checks

    def check_bytes(self, size: int) -> None:
        """Checks an encoded size in bytes.

        Args:
            size: The number of bytes, from the stream length or a declared
                content length.

        Raises:
            MediaLimitExceededError: If ``size`` exceeds ``max_bytes``.
        """
        if size > self.max_bytes:
            raise MediaLimitExceededError("max_bytes", self.max_bytes, size)

    def check_dimensions(self, width: int, height: int) -> None:
        """Checks decoded image or frame dimensions.

        Call this with the dimensions read from the header, before decoding
        pixel data.

        Args:
            width: Width in pixels.
            height: Height in pixels.

        Raises:
            MediaLimitExceededError: If the width, the height or their product
                exceeds the corresponding limit.
        """
        if width > self.max_width:
            raise MediaLimitExceededError("max_width", self.max_width, width)
        if height > self.max_height:
            raise MediaLimitExceededError("max_height", self.max_height, height)
        pixels = width * height
        if pixels > self.max_pixels:
            raise MediaLimitExceededError("max_pixels", self.max_pixels, pixels)

    def check_duration(self, seconds: float) -> None:
        """Checks a declared or measured duration.

        Args:
            seconds: The duration in seconds. NaN and infinity are treated as
                exceeding the limit, because a container that declares them
                cannot be bounded.

        Raises:
            MediaLimitExceededError: If the duration exceeds
                ``max_duration_seconds`` or is not finite.
        """
        if not math.isfinite(seconds) or seconds > self.max_duration_seconds:
            raise MediaLimitExceededError(
                "max_duration_seconds", self.max_duration_seconds, seconds
            )

    def check_frames(self, count: int) -> None:
        """Checks a number of video frames.

        Args:
            count: The declared, estimated or running frame count.

        Raises:
            MediaLimitExceededError: If ``count`` exceeds ``max_frames``.
        """
        if count > self.max_frames:
            raise MediaLimitExceededError("max_frames", self.max_frames, count)

    def check_audio(self, channels: int, sample_rate: int) -> None:
        """Checks an audio stream's layout.

        Args:
            channels: The number of channels.
            sample_rate: The sample rate in hertz.

        Raises:
            MediaLimitExceededError: If either value exceeds its limit.
        """
        if channels > self.max_audio_channels:
            raise MediaLimitExceededError("max_audio_channels", self.max_audio_channels, channels)
        if sample_rate > self.max_sample_rate:
            raise MediaLimitExceededError("max_sample_rate", self.max_sample_rate, sample_rate)

    def check_decoded_audio_bytes(self, size: int) -> None:
        """Checks the size of decoded PCM, declared or counted.

        Call this with the size computed from the probe (see
        :meth:`decoded_audio_bytes`) **before** decoding, and again with the
        running byte count while reading the decoder's output, because
        containers can under-declare their duration.

        Args:
            size: The decoded size in bytes.

        Raises:
            MediaLimitExceededError: If ``size`` exceeds
                ``max_decoded_audio_bytes``.
        """
        if size > self.max_decoded_audio_bytes:
            raise MediaLimitExceededError(
                "max_decoded_audio_bytes", self.max_decoded_audio_bytes, size
            )

    @staticmethod
    def decoded_audio_bytes(
        seconds: float, *, sample_rate: int, channels: int, bytes_per_sample: int = 4
    ) -> int:
        """Returns the size of ``seconds`` of decoded PCM.

        Args:
            seconds: The duration (declared, or the caller's truncation).
            sample_rate: The output sample rate in hertz.
            channels: The output channel count.
            bytes_per_sample: 4 for float32 (the default), 2 for 16-bit PCM.

        Returns:
            ``ceil(seconds * sample_rate) * channels * bytes_per_sample``.

        Raises:
            ValueError: If ``seconds`` is negative or not finite.
        """
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError(f"seconds must be finite and non-negative, got {seconds}")
        return math.ceil(seconds * sample_rate) * channels * bytes_per_sample


def _check_positive_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"MediaLimits.{name} must be an int, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"MediaLimits.{name} must be positive, got {value}")


def _check_positive_float(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"MediaLimits.{name} must be a number, got {type(value).__name__}")
    try:
        finite = math.isfinite(value)
    except OverflowError:  # an int beyond the float range: too long to quote
        raise ValueError(f"MediaLimits.{name} must be positive and finite") from None
    if not finite or value <= 0:
        raise ValueError(f"MediaLimits.{name} must be positive and finite, got {value}")
