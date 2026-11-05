"""Video provenance: probing, frame extraction, fingerprints, invisible watermarks.

Modules (see ``docs/video.md``):

* ``provenance_kit.video.probe``: ffprobe probing of untrusted video into a
  validated ``VideoInfo`` (binaries: ffprobe).
* ``provenance_kit.video.frames``: frame extraction through ffmpeg pipes, at
  a fixed rate, at key frames or every frame, with timestamps (binaries:
  ffmpeg, ffprobe).
* ``provenance_kit.video.fingerprint``: PDQ frame-hash sequences with
  temporal alignment (extra ``video``).
* ``provenance_kit.video.watermark``: Meta VideoSeal invisible watermark on
  decoded frames (extra ``video-watermark``).

This file stays import-light: it must not import the submodules eagerly.
"""
