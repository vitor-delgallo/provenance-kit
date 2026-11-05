"""Image provenance: invisible and visible watermarks, perceptual hashes.

Modules (see ``docs/image.md``):

* ``provenance_kit.image.loading``: safe decoding of untrusted image bytes
  (limits before decoding, type from content, EXIF orientation, ICC profile).
* ``provenance_kit.image.watermark``: Adobe TrustMark invisible watermark with
  read-back verification and strength escalation (extra ``image-watermark``).
* ``provenance_kit.image.visible``: tiled visible text watermark (Pillow only).
* ``provenance_kit.image.hashing``: PDQ (``pdqhash`` or a built-in NumPy
  port), pHash and dHash perceptual hashes, fingerprints and known-hash-list
  matching (extra ``image-hash`` for ``pdqhash`` and ``imagehash``).

This file stays import-light: it must not import the submodules eagerly.
"""
