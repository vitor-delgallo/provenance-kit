"""Invisible audio watermarks: Meta AudioSeal (extra ``audio-watermark``).

Backends work on **decoded PCM in memory** (ARCHITECTURE.md, decision 2):
``float32`` NumPy arrays shaped ``(channels, samples)`` plus a sample rate.
They never encode; :func:`embed_verified` takes the application's encoder as
a callable and checks the watermark after that real encoding.

* :class:`AudioSealBackend` loads the 16-bit base generator and detector from
  pinned, SHA-256-verified checkpoints (MIT-licensed weights). It works
  offline from a pre-provisioned directory; downloading happens only with
  ``download=True`` or :func:`download_audioseal_models`. Audio at any rate
  is resampled to the models' 16 kHz in memory.
* :func:`embed_watermark`, :func:`detect_watermark` and
  :func:`embed_verified` hold the domain logic and accept any
  :class:`AudioWatermarkBackend`.

The payload is 16 bits with no room for error correction (decision 8). The
detector's ``found`` is not evidence on its own (unmarked impulse trains
score up to 0.97): give the :class:`~provenance_kit.payload.PayloadCodec`
check bits, for example ``PayloadCodec(16, version_bits=0, check_bits=4)``
(4,096 identifiers, random bits pass the check with probability 1/16), and
rely on ``detection.verified`` (found and the check bits match).

Example::

    from provenance_kit import PayloadCodec
    from provenance_kit.audio.pcm import decode_pcm
    from provenance_kit.audio.watermark import AudioSealBackend, detect_watermark, embed_watermark

    backend = AudioSealBackend(model_dir="/srv/models/audioseal")
    codec = PayloadCodec(16, version_bits=0, check_bits=4)
    audio = decode_pcm(data)
    marked = embed_watermark(audio.samples, audio.sample_rate, codec.encode(42), backend=backend)
    detection = detect_watermark(marked, audio.sample_rate, backend=backend, codec=codec)
    if detection.verified:  # found, and the check bits match
        identifier = detection.payload.identifier

Alignment matters: the models are sensitive to sample-level time shifts, so
publish in containers that record the encoder delay (AAC in MP4/M4A, MP3
files with the LAME/Info tag, Ogg Opus, FLAC) and check the real pipeline
with :func:`embed_verified`. For audio that went through a pipeline that
does not record the delay (raw ADTS AAC, MP3 written to a pipe), pass
``alignment=AlignmentSearch()`` (with the codec) to :func:`detect_watermark`:
an opt-in, bounded and slower search over sample offsets, whose shift is used
only when it is corroborated (by the check bits; without them, only to
refine a signal already found unshifted). Robustness numbers measured with
real codecs are in ``docs/audio.md``.

Errors: a model that fails while it runs, or a backend that returns
malformed scores or PCM, raises
:class:`~provenance_kit.errors.WatermarkBackendError` (re-exported here, as
in the image and video watermark modules).

Thread safety: :class:`AudioSealBackend` can be shared between threads
(model calls are serialized); the functions are thread-safe given a
thread-safe backend.
"""

from __future__ import annotations

from provenance_kit.audio.watermark._api import (
    DEFAULT_DETECTION_THRESHOLD,
    MAX_ALIGNMENT_CANDIDATES,
    AlignmentSearch,
    AudioEncoder,
    VerifiedAudioEmbedding,
    detect_watermark,
    embed_verified,
    embed_watermark,
)
from provenance_kit.audio.watermark._audioseal import (
    AUDIOSEAL_DETECTOR,
    AUDIOSEAL_GENERATOR,
    AUDIOSEAL_PAYLOAD_BITS,
    AUDIOSEAL_REVISION,
    AUDIOSEAL_SAMPLE_RATE,
    AudioSealBackend,
    download_audioseal_models,
)
from provenance_kit.audio.watermark.base import (
    AudioScores,
    AudioWatermarkBackend,
    AudioWatermarkDetection,
    Samples,
)
from provenance_kit.errors import WatermarkBackendError
from provenance_kit.modelfiles import MODEL_DIR_ENV, ModelFile

__all__ = [
    "AUDIOSEAL_DETECTOR",
    "AUDIOSEAL_GENERATOR",
    "AUDIOSEAL_PAYLOAD_BITS",
    "AUDIOSEAL_REVISION",
    "AUDIOSEAL_SAMPLE_RATE",
    "DEFAULT_DETECTION_THRESHOLD",
    "MAX_ALIGNMENT_CANDIDATES",
    "MODEL_DIR_ENV",
    "AlignmentSearch",
    "AudioEncoder",
    "AudioScores",
    "AudioSealBackend",
    "AudioWatermarkBackend",
    "AudioWatermarkDetection",
    "ModelFile",
    "Samples",
    "VerifiedAudioEmbedding",
    "WatermarkBackendError",
    "detect_watermark",
    "download_audioseal_models",
    "embed_verified",
    "embed_watermark",
]
