"""Audio provenance: PCM decoding, Chromaprint fingerprints, invisible watermarks.

Modules (see ``docs/audio.md``):

* ``provenance_kit.audio.pcm``: decoding of untrusted audio to float32 PCM
  through ffmpeg pipes (binaries: ffmpeg, ffprobe).
* ``provenance_kit.audio.fingerprint``: Chromaprint fingerprints through the
  ``fpcalc`` binary, and their comparison by bit error rate (extra ``audio``).
* ``provenance_kit.audio.watermark``: Meta AudioSeal invisible watermark on
  decoded PCM (extra ``audio-watermark``).

This file stays import-light: it must not import the submodules eagerly.
"""
