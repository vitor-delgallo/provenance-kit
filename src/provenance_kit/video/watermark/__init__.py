"""Invisible video watermarks: Meta VideoSeal (extra ``video-watermark``).

Backends work on **decoded frames in memory** (ARCHITECTURE.md, decision 2):
RGB ``uint8`` NumPy arrays shaped ``(height, width, 3)``. They never encode;
:func:`embed_verified` takes the application's encoder as a callable and
checks the watermark after that real encoding.

* :class:`VideoSealBackend` loads the official VideoSeal v1.0 TorchScript
  model (256 bits) from a pinned, SHA-256-verified file. It works offline
  from a pre-provisioned directory; downloading happens only with
  ``download=True`` or :func:`download_videoseal_model`.
* :class:`RepetitionCode` repeats a shorter payload across the 256 bits and
  decodes it by soft majority vote, which also yields a calibrated detection
  p-value (decision 8).
* :func:`embed_watermark`, :func:`detect_watermark` and
  :func:`embed_verified` hold the domain logic and accept any
  :class:`VideoWatermarkBackend`.

Example::

    from provenance_kit import PayloadCodec
    from provenance_kit.video.watermark import (
        RepetitionCode,
        VideoSealBackend,
        detect_watermark,
        embed_watermark,
    )

    backend = VideoSealBackend(model_dir="/srv/models/videoseal")
    code = RepetitionCode(copies=4)  # 64 data bits
    codec = PayloadCodec(code.data_bits, version_bits=4, check_bits=12)
    marked = embed_watermark(frames, codec.encode(42), backend=backend, repetition=code)
    ...  # encode `marked` with ffmpeg
    detection = detect_watermark(decoded_frames, backend=backend, codec=codec, repetition=code)

Robustness numbers measured with real codecs are in ``docs/video.md``.

Thread safety: :class:`VideoSealBackend` can be shared between threads (model
calls are serialized); the functions are thread-safe given a thread-safe
backend.
"""

from __future__ import annotations

from provenance_kit.video.watermark._api import (
    DEFAULT_MAX_P_VALUE,
    DEFAULT_MIN_MEAN_ABS_SCORE,
    FrameEncoder,
    VerifiedVideoEmbedding,
    detect_watermark,
    embed_verified,
    embed_watermark,
)
from provenance_kit.video.watermark._videoseal import (
    VIDEOSEAL_MODEL,
    VIDEOSEAL_PAYLOAD_BITS,
    VideoSealBackend,
    download_videoseal_model,
)
from provenance_kit.video.watermark._weights import MODEL_DIR_ENV, ModelFile
from provenance_kit.video.watermark.base import (
    Frame,
    VideoWatermarkBackend,
    VideoWatermarkDetection,
)
from provenance_kit.video.watermark.repetition import RepetitionCode, RepetitionDecoded

__all__ = [
    "DEFAULT_MAX_P_VALUE",
    "DEFAULT_MIN_MEAN_ABS_SCORE",
    "MODEL_DIR_ENV",
    "VIDEOSEAL_MODEL",
    "VIDEOSEAL_PAYLOAD_BITS",
    "Frame",
    "FrameEncoder",
    "ModelFile",
    "RepetitionCode",
    "RepetitionDecoded",
    "VerifiedVideoEmbedding",
    "VideoSealBackend",
    "VideoWatermarkBackend",
    "VideoWatermarkDetection",
    "detect_watermark",
    "download_videoseal_model",
    "embed_verified",
    "embed_watermark",
]
