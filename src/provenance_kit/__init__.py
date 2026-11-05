"""provenance-kit: building blocks for content provenance and media tracking.

The top-level package is light: it imports only the standard library. Each
capability lives in a subpackage and loads its heavy dependencies lazily:

* :mod:`provenance_kit.image`: invisible (TrustMark) and visible watermarks,
  perceptual hashes (pHash, PDQ).
* :mod:`provenance_kit.video`: frame extraction (ffmpeg), frame-hash
  fingerprints, invisible watermarks (VideoSeal).
* :mod:`provenance_kit.audio`: Chromaprint fingerprints (fpcalc), invisible
  watermarks (AudioSeal).
* :mod:`provenance_kit.provenance`: C2PA manifests and OpenTimestamps proofs.

Shared foundations: :mod:`~provenance_kit.errors`,
:mod:`~provenance_kit.limits`, :mod:`~provenance_kit.payload`,
:mod:`~provenance_kit.tools`, :mod:`~provenance_kit.tempfiles`,
:mod:`~provenance_kit.mediatype`, :mod:`~provenance_kit.bits`,
:mod:`~provenance_kit.matching`, :mod:`~provenance_kit.detection` and
:mod:`~provenance_kit.device`.
"""

from __future__ import annotations

from provenance_kit import _logging as _logging  # installs the NullHandler
from provenance_kit.errors import (
    BackendUnavailableError,
    ExternalToolError,
    ExternalToolTimeoutError,
    MediaDecodeError,
    MediaError,
    MediaLimitExceededError,
    ModelLoadError,
    PayloadError,
    ProvenanceKitError,
    UnsupportedMediaError,
    WatermarkBackendError,
)
from provenance_kit.limits import MediaLimits
from provenance_kit.payload import DecodedPayload, PayloadCodec

__version__ = "0.1.0"

__all__ = [
    "BackendUnavailableError",
    "DecodedPayload",
    "ExternalToolError",
    "ExternalToolTimeoutError",
    "MediaDecodeError",
    "MediaError",
    "MediaLimitExceededError",
    "MediaLimits",
    "ModelLoadError",
    "PayloadCodec",
    "PayloadError",
    "ProvenanceKitError",
    "UnsupportedMediaError",
    "WatermarkBackendError",
    "__version__",
]
