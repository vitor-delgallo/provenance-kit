"""Image provenance: invisible and visible watermarks, perceptual hashes.

Planned modules (see ``docs/ARCHITECTURE.md`` for ownership):

* ``provenance_kit.image.watermark``: Adobe TrustMark invisible watermark with
  read-back verification and strength escalation (extra ``image-watermark``).
* ``provenance_kit.image.visible``: tiled visible text watermark (Pillow only).
* ``provenance_kit.image.hashing``: pHash and PDQ perceptual hashes and hash
  list matching (extra ``image-hash``).

This file stays import-light: it must not import the submodules eagerly.
"""
