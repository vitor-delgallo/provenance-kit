# provenance-kit

[![CI](https://github.com/vitor-delgallo/provenance-kit/actions/workflows/ci.yml/badge.svg)](https://github.com/vitor-delgallo/provenance-kit/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)

Building blocks for content provenance and media tracking in Python:
invisible and visible watermarks, perceptual hashes, audio and video
fingerprints, C2PA manifests and OpenTimestamps proofs.

provenance-kit is a library of small, typed, well-tested pieces that a media
pipeline (for example a background worker) composes as it needs. It does not
generate derivatives, store anything or impose an application model. It is
built for **untrusted media**: resource limits are checked before decoding,
types are detected from content, external tools run without a shell under
timeouts, and errors never leak media or secrets.

> **Status: pre-alpha (0.1.0).** The foundation (errors, limits, tool runner,
> payload codec, media type detection, hash matching) is implemented and
> tested. The capability modules below are **planned** and their examples
> show the intended shape, not a released API.

## Install

Requires Python 3.11 or later.

```sh
pip install provenance-kit                      # core: NumPy and Pillow only
pip install "provenance-kit[image-hash,c2pa]"   # pick the capabilities you need
pip install "provenance-kit[all]"               # everything
```

| Extra | Provides | Installs |
|---|---|---|
| `image-watermark` | TrustMark invisible image watermark | `trustmark`, `torch` |
| `image-hash` | pHash and PDQ perceptual hashes | `imagehash`, `pdqhash` |
| `video` | frame extraction, frame-hash fingerprints | `image-hash` extra; needs `ffmpeg`/`ffprobe` binaries |
| `video-watermark` | VideoSeal invisible video watermark | `torch` |
| `audio` | Chromaprint fingerprints | nothing; needs the `fpcalc` binary |
| `audio-watermark` | AudioSeal invisible audio watermark | `audioseal`, `torch` |
| `c2pa` | C2PA manifest signing, reading and validation | `c2pa-python` |
| `timestamps` | OpenTimestamps stamping, upgrading and inspection | `opentimestamps` |
| `all` | all of the above | |

For CPU-only machines, install PyTorch from the
[PyTorch CPU index](https://download.pytorch.org/whl/cpu) first. On Linux,
`pdqhash` builds from source and needs a C++ compiler. The licenses of every
dependency, model and binary are in [docs/THIRD_PARTY.md](docs/THIRD_PARTY.md).

A missing extra or binary raises `BackendUnavailableError`, whose message says
what to install or configure.

## Foundation (available now)

```python
from provenance_kit import MediaLimits, PayloadCodec
from provenance_kit.mediatype import MediaKind, require_media_type
from provenance_kit.tools import ExternalTools, Tool

# Watermark payloads: a format version and an identifier in N bits.
codec = PayloadCodec(61, version_bits=4, check_bits=8)  # TrustMark BCH_5 capacity
bits = codec.encode(123_456, version=1)                  # "0001..." (61 characters)
decoded = codec.decode(bits)
assert decoded.valid and decoded.identifier == 123_456

# Untrusted media: enforce limits and detect the type from the bytes.
limits = MediaLimits()
with open("upload.bin", "rb") as f:
    data = f.read(limits.max_bytes + 1)  # never read more than the limit allows
limits.check_bytes(len(data))
media_type = require_media_type(data, kinds={MediaKind.IMAGE})

# External tools: no shell, pipes, timeouts, sanitized errors.
tools = ExternalTools.from_env()  # PROVENANCE_KIT_FFPROBE, or PATH
result = tools.run(
    Tool.FFPROBE,
    ["-hide_banner", "-loglevel", "error", "-protocol_whitelist", "pipe",
     "-print_format", "json", "-show_streams", "-i", "pipe:0"],
    stdin=data,
    timeout_seconds=30,
)
```

## Capabilities (planned)

The examples show the intended usage; names may change before release.

### Image: invisible watermark (TrustMark)

```python
from provenance_kit.image.watermark import TrustMarkWatermarker  # planned

marker = TrustMarkWatermarker(device="cpu")
marked = marker.embed(image, codec.encode(42))      # verifies the read-back,
detection = marker.detect(marked, codec=codec)      # escalates strength if needed
```

### Image: visible tiled watermark

```python
from provenance_kit.image.visible import tile_text  # planned

preview = tile_text(image, "sample", opacity=0.25, angle=30)
```

### Image: perceptual hashes (pHash, PDQ)

```python
from provenance_kit.image.hashing import pdq_hash, match_hashes  # planned

h = pdq_hash(image)                       # 256-bit hash and quality score
matches = match_hashes(h, known_hashes, max_distance=31)
```

### Video: frames, fingerprints, watermark (ffmpeg, VideoSeal)

```python
from provenance_kit.video.fingerprint import fingerprint_video, align  # planned

fp = fingerprint_video(data, tools=tools, fps=1)    # frame-hash sequence
alignment = align(fp, reference_fp)                 # offset and similarity
```

### Audio: fingerprint and watermark (Chromaprint, AudioSeal)

```python
from provenance_kit.audio.fingerprint import chromaprint  # planned

fp = chromaprint(data, tools=tools)
```

### C2PA manifests

```python
from provenance_kit.provenance.c2pa import sign_manifest, read_manifest  # planned

signed = sign_manifest(data, manifest, signer=kms_signer)  # key stays in the KMS/HSM
report = read_manifest(signed)                              # manifest + validation status
```

### OpenTimestamps

```python
from provenance_kit.provenance.timestamps import stamp, upgrade, inspect  # planned

proof = stamp(sha256_digest, calendars=[...])   # submitted to several calendars
proof = upgrade(proof)                           # once Bitcoin attestations exist
info = inspect(proof)
```

## Documentation

* [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): package map, dependency,
  error, security, threading and testing rules (binding for contributors).
* [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md): environment setup, binaries,
  checks and tests.
* [docs/THIRD_PARTY.md](docs/THIRD_PARTY.md): licenses of code, model weights
  and binaries.
* [SECURITY.md](SECURITY.md): reporting vulnerabilities and the threat model.
* [CHANGELOG.md](CHANGELOG.md).

## License

MIT. See [LICENSE](LICENSE). Optional dependencies, models and binaries have
their own licenses: see [docs/THIRD_PARTY.md](docs/THIRD_PARTY.md).
