"""Video provenance: frame extraction, fingerprints, invisible watermarks.

Planned modules (see ``docs/ARCHITECTURE.md`` for ownership):

* ``provenance_kit.video.frames``: probing and frame extraction through the
  ffprobe and ffmpeg binaries.
* ``provenance_kit.video.fingerprint``: per-frame hash sequences with temporal
  alignment (extra ``video``).
* ``provenance_kit.video.watermark``: Meta VideoSeal invisible watermark
  (extra ``video-watermark``).

This file stays import-light: it must not import the submodules eagerly.
"""
