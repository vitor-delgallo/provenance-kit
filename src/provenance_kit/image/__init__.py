"""Image provenance: invisible and visible watermarks, perceptual hashes.

Modules (see ``docs/image.md``):

* ``provenance_kit.image.loading``: safe decoding of untrusted image bytes
  (limits before decoding, type from content, EXIF orientation, ICC profile).
* ``provenance_kit.image.watermark``: Adobe TrustMark invisible watermark with
  read-back verification and strength escalation (extra ``image-watermark``).
* ``provenance_kit.image.visible``: tiled visible text watermark (Pillow only).
* ``provenance_kit.image.hashing``: PDQ (a built-in NumPy port, or
  ``pdqhash`` with extra ``pdq-native``), pHash and dHash perceptual hashes
  (extra ``image-hash``), fingerprints and known-hash-list matching.

This file stays import-light: it must not import the submodules eagerly.
"""
