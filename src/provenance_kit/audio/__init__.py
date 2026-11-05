"""Audio provenance: Chromaprint fingerprints and invisible watermarks.

Planned modules (see ``docs/ARCHITECTURE.md`` for ownership):

* ``provenance_kit.audio.fingerprint``: Chromaprint fingerprints through the
  ``fpcalc`` binary, and their comparison (extra ``audio``).
* ``provenance_kit.audio.watermark``: Meta AudioSeal invisible watermark
  (extra ``audio-watermark``).

This file stays import-light: it must not import the submodules eagerly.
"""
