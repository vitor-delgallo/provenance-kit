# Image module

`provenance_kit.image` has four public modules:

| Module | What it does | Needs |
|---|---|---|
| `provenance_kit.image.loading` | decodes untrusted image bytes safely | core install |
| `provenance_kit.image.hashing` | PDQ, pHash and dHash; fingerprints; known-hash lists | core install for PDQ (NumPy port); extra `image-hash` for `pdqhash`, pHash and dHash |
| `provenance_kit.image.watermark` | invisible watermark with read-back verification (TrustMark) | extra `image-watermark` and the model files |
| `provenance_kit.image.visible` | visible text tiled diagonally across an image | core install |

Importing any of them is cheap: PyTorch, TrustMark, imagehash and pdqhash load
only when a backend that needs them is constructed or called.

The numbers below were measured on 2026-11-05 on the reference Windows machine
(CPU only, Python 3.11, Pillow 13.0, NumPy 2.4, PyTorch 2.15 CPU, trustmark
0.9.2, pdqhash 0.2.8).

## Loading untrusted images

```python
from provenance_kit import MediaLimits
from provenance_kit.image.loading import iter_frames, load_image, probe_image

loaded = load_image(data, limits=MediaLimits())  # first frame, RGB or RGBA
loaded.image            # PIL.Image.Image, orientation applied, 8-bit
loaded.info.media_type  # MediaType.JPEG, ... (from the bytes)
loaded.info.icc_profile # bytes or None (untrusted)

for frame in iter_frames(gif_bytes):  # every frame, as displayed
    ...
```

What `load_image` guarantees, in order:

1. `limits.check_bytes(len(data))`.
2. The type is detected from the leading bytes; only that type's Pillow plugin
   parses the data (no fallback through other parsers). Accepted by default:
   JPEG, PNG/APNG, GIF, WebP, AVIF (official Pillow 11.3+ wheels have AVIF).
   TIFF and BMP need `allowed=`.
3. Header only: dimensions (`limits.check_dimensions`, also for the size after
   an EXIF rotation) and the frame count of animated images (default cap
   `DEFAULT_MAX_IMAGE_FRAMES` = 5,000, plus `limits.max_frames`) are checked
   before any pixel is decoded. Pillow's global `MAX_IMAGE_PIXELS` check is
   bypassed (plugins are constructed directly), so the caller's limits decide
   and no `DecompressionBombWarning` is emitted.
4. Every decoder failure becomes `MediaDecodeError`; unsupported or refused
   types `UnsupportedMediaError`; limits `MediaLimitExceededError`.
5. EXIF orientation applied; ICC profile kept (`to_srgb=True` applies it with
   LittleCMS, falling back to unmanaged conversion if the profile is unusable);
   mode normalized: `ImageMode.AUTO` gives `RGBA` when the image has alpha or a
   transparent palette entry, else `RGB`. 16-bit images are scaled to 8 bits
   (Pillow's `convert` would clip them). CMYK is converted without colour
   management unless `to_srgb=True` and the file has a profile.

`probe_image` performs steps 1 to 3 only. The tests feed truncated files,
header-only decompression bombs (PNG `IHDR` 100,000 px wide, 10,000 x 10,000,
a 65,535 x 65,535 GIF screen), wrong magic (MP4, WAV, text), a GIF with 5,001
real frames, an APNG declaring 10 million frames, a CMYK/L/1-bit/LA/RGBA/
palette/16-bit mix, and 480 randomly mutated JPEG/PNG/GIF/WebP files; only
`MediaError` subclasses escape.

### HEIC / HEIF

Pillow does not decode HEIC, and no permissively licensed decoder package
exists today:

* `pillow-heif` 1.8.1: Python code BSD-3-Clause, but its binary wheels are
  **GPL-2.0 as a whole** (they bundle the x265 encoder; libheif and libde265
  are LGPL-3.0), per its `LICENSES_bundled.txt`.
* `pi-heif` 1.4.0, the decode-only LGPL variant, is **discontinued**, and its
  wheels bundle libheif 1.23.0, affected by a heap buffer overflow on crafted
  HEIC files (CVE-2026-84383); the pillow-heif README says to migrate.

provenance-kit therefore ships no HEIF extra. Applications that accept a
decoder's license plug it in: any Pillow `ImageFile` class works as an opener,
and the limits are checked exactly as for built-in formats.

```python
import pillow_heif  # installed and license-reviewed by the application
from provenance_kit.mediatype import MediaType

openers = {MediaType.HEIC: pillow_heif.HeifImageFile, MediaType.HEIF: pillow_heif.HeifImageFile}
loaded = load_image(data, openers=openers)
```

## Perceptual hashes

```python
from provenance_kit.image.hashing import HashList, PdqHasher, fingerprint_image

hasher = PdqHasher()                  # "auto": pdqhash if installed, else NumPy
h = hasher.hash_image(loaded.image)   # PdqHash(value=<32 bytes>, quality=0..100)
h.hex()                               # canonical 64-digit hex, as Meta's tools print

known = HashList.from_hex(partner_lines, keys=partner_ids)
matches = known.match(h) if h.is_reliable() else []   # distance <= 31 by default
```

### PDQ

* `PdqHash.value` is `bytes.fromhex(<canonical hex>)`. Bit `k` of Meta's
  `Hash256` (`k = 16 * row + column` of the 16 x 16 DCT block, set when that
  coefficient exceeds the median) is bit `k % 8` of byte `31 - k // 8`.
* **Pre-scaling.** Both of Meta's reference hashers shrink images to at most
  512 x 512 first (the C++ `pdqio.cpp` calls the two-pass blur "prohibitively
  expensive for larger images"; the Python reference calls
  `Image.thumbnail((512, 512))`). `PdqHasher(max_side=512)` does the same by
  default. `max_side=None` hashes at full resolution like
  `python-threatexchange`'s `pdq_from_bytes`.
* **Thresholds.** Meta's PDQ README: distance `<= 31` matches, discard
  quality `<= 49`; `python-threatexchange` rejects quality `< 50`
  (`PDQ_MATCH_DISTANCE`, `PDQ_MIN_QUALITY`).
* **Backends.** `pdqhash` (Meta's C++, `float32`) or a NumPy port of Meta's
  pure-Python reference (`float64`, ThreatExchange commit `a9a36d45`, BSD
  notice kept in `_pdq_numpy.py`).

Validation:

| Check | Result |
|---|---|
| NumPy port vs Meta's pure-Python reference, 14 generated images (all Jarosz window sizes 1 to 5, images under 64 px, non-square) | identical hashes and qualities (committed vectors, unit test) |
| `pdqhash` vs the same reference vectors | identical |
| ThreatExchange published vectors (`pdq_test.py`, 18 images: photos, rotations and flips, LabelMe subset), both backends, default `max_side=512` | distance 0 on all 17 images with quality >= 50; 10 on the one with quality 3 |
| Same at full resolution (`max_side=None`) | up to 12 bits (photos 1,004 to 1,600 px) |
| `pdqhash` vs NumPy port, 1,200 generated images (963 with quality >= 50) | 958 identical; 2, 2, 2, 7 and 8 bits on the other five (all with a side under 64 px); quality always equal |

The tests assert `PDQ_BACKEND_MAX_DISTANCE` = 10 bits between backends. Cost
for one image: 9 ms (`pdqhash`) or 19 ms (NumPy) at 512 x 512; 41 or 44 ms for
a 4000 x 3000 image with the default pre-scaling, 0.5 or 0.8 s without it.

The ThreatExchange images are not committed (their licensing is not stated
separately from the repository's code license, and several exceed the size
budget for fixtures): `pytest --run-network` downloads them once at the
pinned commit into `.cache/threatexchange-pdq/`, checking each file's SHA-256.

### pHash, dHash, fingerprints

`phash(image)` and `dhash(image)` wrap `imagehash` and return a 64-bit
`BinaryHash` whose `hex()` equals `str(imagehash.phash(...))`.
`fingerprint_image` computes all three; `compare_fingerprints` decides with
PDQ when both PDQ hashes are reliable, else pHash, else dHash.
`MatchThresholds` defaults: PDQ 31 / quality 50 (Meta), pHash 10, dHash 10
(this library's starting points; there is no external standard, calibrate on
your content).

### Known-hash lists

`HashList` holds hashes packed in 64-bit words and scans them with a
vectorized XOR and population count, so matches are exact. Memory: 32 bytes
per PDQ hash (32 MB per million) plus the keys (`keys=range(n)` costs
nothing); a query needs under 1 MB of scratch. One query against 1,000,000 PDQ
hashes: 12 ms. Building from packed bytes: 10 ms per million; from hex lines:
about 2 s per million.

## Invisible watermark (TrustMark)

```python
from provenance_kit.image.watermark import (
    ImageWatermarker, TrustMarkBackend, download_trustmark_models,
)

# Deployment time (network, once): pinned, verified, atomic.
download_trustmark_models("Q", model_dir="/srv/models/trustmark")

# Worker (offline):
marker = ImageWatermarker(TrustMarkBackend(model_dir="/srv/models/trustmark"))
result = marker.embed_verified(image, identifier=123_456, encode=to_webp_q80)
if not result.verified:
    ...  # publish result.encoded anyway and flag the asset
detection = marker.detect_encoded(published_bytes)
if detection.found and detection.payload.valid:
    detection.payload.identifier
```

### Capacity

TrustMark's models carry 100 bits: payload + `7 * t` BCH parity bits (GF(2^7),
polynomial 137) + 4 schema bits. From `trustmark/datalayer.py` 0.9.2
(`schemaCapacity`, `raw_payload_split`) and checked against it in a test:

| `TrustMarkEncoding` | schema | payload bits | corrects |
|---|---|---|---|
| `BCH_SUPER` | 0 | 40 | 8 |
| `BCH_5` (default) | 1 | 61 | 5 |
| `BCH_4` | 2 | 68 | 4 |
| `BCH_3` | 3 | 75 | 3 |

`ImageWatermarker` packs identifiers with
`PayloadCodec(payload_bits, version_bits=4, check_bits=8)` by default (49
identifier bits with `BCH_5`).

### Model files (decision 5)

* `MODEL_FILES` pins the size and SHA-256 of the config, encoder and decoder
  of each variant (Q, P, B, C); each file's MD5 also matched the checksums in
  `trustmark` 0.9.2. The remover and box-detector models are not used.
* `model_dir` defaults to `$PROVENANCE_KIT_MODEL_DIR/trustmark` or a per-user
  cache directory, and is refused if it is inside `site-packages`.
* The constructor never downloads (unless `download=True`); missing or altered
  files raise `ModelLoadError` naming the file, not the path.
* Files are read once, hashed, and the models are built from those bytes in
  memory; checkpoints load with `torch.load(weights_only=True)`; config files
  may only name classes in the `trustmark` package and the two decoder
  architectures that do not fetch pretrained weights.
* `trustmark`'s `print()` calls are redirected to DEBUG logging (the tests
  assert empty stdout and stderr).

Cost (CPU): first construction about 5 s (importing PyTorch, TorchVision and
Lightning dominates); another backend in the same process about 0.6 s. The
weights take about as much memory as the files (65 MB for Q). Embedding takes
0.15 s at 512 x 384, 0.25 s at 1600 x 1000 and 0.7 s at 3200 x 2000 (the
full-size residual dominates); detection 0.2 to 0.3 s at any size (the decoder
sees 245 x 245 pixels). Thread safety: model calls are serialized with a lock;
share one backend.

### Measured verification

`embed_verified` embeds, runs the caller's encoder, decodes the bytes with
`load_image` and reads the mark back, escalating through `DEFAULT_STRENGTHS`
= (1.0, 1.5, 2.0, 3.0). Variant Q, `BCH_5`, Pillow encoders, 12 images (6
photos from ThreatExchange's test data at 34 x 42 to 3200 x 2008 px, 6
generated):

| Encoder | strength 0.5 | 1.0 | 1.5 | 2.0 | 3.0 | default schedule |
|---|---|---|---|---|---|---|
| WebP q80 | 1/12 | 9/12 | 10/12 | 11/12 | 11/12 | 11/12 (all at 1.0) |
| WebP q60 | 1/12 | 9/12 | 10/12 | 10/12 | 9/12 | 11/12 (1.0; one 256 px photo at 1.5, a flat grey image at 3.0) |
| JPEG q85 | 2/12 | 11/12 | 11/12 | 11/12 | 10/12 | 11/12 (all at 1.0) |
| JPEG q70 | 2/12 | 11/12 | 11/12 | 11/12 | 11/12 | 11/12 (all at 1.0) |

* The image that never verifies is 34 x 42 px: too small to hold a mark.
* Recovery is not monotonic in strength (some images fail at 3.0 after
  passing at 2.0) and depends on the payload bits, which is why verification
  is done per asset rather than with a fixed strength.
* PSNR (watermarked vs original, before encoding): 36 to 45 dB at 1.0, 33 to
  41 at 1.5, 30 to 39 at 2.0, 27 to 35 at 3.0 on the photos; up to 51 dB on
  flat images.
* Unmarked images: 0 of 72 detected (BCH decode in the configured schema).
  TrustMark accepts any schema the bits claim; the backend only reports marks
  in its own encoding, and the default codec's 8 check bits reject about 255
  in 256 of the remaining random codewords.
* Variant P (PSNR about 48 dB on photos) cannot hold a mark on very smooth
  content: on a blurred generated image it fails even before encoding, with
  both this adapter and `trustmark`'s own `TrustMark` class. C works on all
  test images (PSNR 36 to 48 dB).

## Visible watermark

```python
from provenance_kit.image.visible import tile_text

preview = tile_text(image, "viewer 1234 - 2026-11-05", opacity=0.35, angle=30)
```

The text is drawn on a rotated brick-pattern grid covering the whole image,
with an optional dark outline (`shadow=True`) for legibility on light and dark
areas. `size` is the font height as a fraction of the shorter side (default
0.04, at least 10 px); `spacing` the gap in font heights (default 2). Same
size and mode out (RGB, RGBA, L, LA directly; CMYK, YCbCr, LAB, HSV, RGBX
through RGB; P back onto its palette, transparency kept; 1 through L).
Deterministic for given Pillow and FreeType versions. 0.3 s for a 4000 x 3000
image.

Fonts: pass `font=` (a path to a TTF/OTF you may use). The default is Pillow's
embedded subset of Aileron Regular (CC0, Basic Latin only; other scripts
render as boxes) when Pillow has FreeType; otherwise Pillow's bitmap font,
scaled. provenance-kit bundles no font.
