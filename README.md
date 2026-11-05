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
> payload codec, media type detection, hash matching, model files) and the
> capability modules below (image, video, audio, C2PA, OpenTimestamps) are
> implemented and tested, with measured results in [docs/](docs/). The API
> may still change before 1.0; see [CHANGELOG.md](CHANGELOG.md).

## Install

Requires Python 3.11 or later.

```sh
pip install provenance-kit                      # core: NumPy and Pillow only
pip install "provenance-kit[image-hash,c2pa]"   # pick the capabilities you need
pip install "provenance-kit[all]"               # every capability (not pdq-native)
```

| Extra | Provides | Installs |
|---|---|---|
| `image-watermark` | TrustMark invisible image watermark | `trustmark`, `torch` |
| `image-hash` | pHash and dHash perceptual hashes (PDQ needs no extra) | `imagehash` |
| `pdq-native` | the C++ PDQ backend, preferred when installed (optional) | `pdqhash` |
| `video` | frame extraction, frame-hash fingerprints | nothing; needs the `ffmpeg`/`ffprobe` binaries |
| `video-watermark` | VideoSeal invisible video watermark | `torch` |
| `audio` | Chromaprint fingerprints | nothing; needs the `fpcalc` binary |
| `audio-watermark` | AudioSeal invisible audio watermark | `audioseal`, `torch` |
| `c2pa` | C2PA manifest signing, reading and validation | `c2pa-python` |
| `timestamps` | OpenTimestamps stamping, upgrading and inspection | `opentimestamps` |
| `all` | all of the above except `pdq-native` | |

For CPU-only machines, install PyTorch from the
[PyTorch CPU index](https://download.pytorch.org/whl/cpu) first. PDQ hashes
work with the core install (a NumPy port of Meta's reference code);
`pdq-native` adds Meta's C++ code through `pdqhash`, which has no Linux wheels
and builds from source with a C++ compiler there, so `all` leaves it out. The
licenses of every dependency, model and binary are in
[docs/THIRD_PARTY.md](docs/THIRD_PARTY.md).

Model weights (TrustMark, VideoSeal, AudioSeal) are pinned by size and
SHA-256 and never downloaded implicitly. Each backend reads them from its
`model_dir` argument, by default `$PROVENANCE_KIT_MODEL_DIR/<backend>` or,
without the variable, `provenance-kit/models/<backend>` in the per-user cache
directory (`provenance_kit.modelfiles`).

A missing extra or binary raises `BackendUnavailableError`, whose message says
what to install or configure.

## Foundation

```python
from provenance_kit import MediaLimits, PayloadCodec
from provenance_kit.mediatype import MediaKind, require_media_type
from provenance_kit.tools import ExternalTools, Tool

# Watermark payloads: a format version and an identifier in N bits.
codec = PayloadCodec(61, version_bits=4, check_bits=8)  # TrustMark BCH_5 capacity
bits = codec.encode(123_456, version=1)  # "0001..." (61 characters)
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
    [
        "-hide_banner",
        "-loglevel",
        "error",
        "-protocol_whitelist",
        "pipe",
        "-print_format",
        "json",
        "-show_streams",
        "-i",
        "pipe:0",
    ],
    stdin=data,
    timeout_seconds=30,
)
```

## Capabilities

Each module has its own page under [docs/](docs/) with the details,
thresholds and measured results.

### Image

The image module is available; [docs/image.md](docs/image.md) has the
details and the measured results.

```python
import io

from provenance_kit.image.loading import load_image
from provenance_kit.image.hashing import HashList, PdqHasher
from provenance_kit.image.visible import tile_text
from provenance_kit.image.watermark import (
    ImageWatermarker,
    TrustMarkBackend,
    download_trustmark_models,
)

# Untrusted bytes: limits before decoding, type from content, orientation applied.
image = load_image(data, limits=limits).image

# Perceptual hash (PDQ; pdqhash if installed, else a built-in NumPy port)
# against a list of known hashes shared by a partner organization.
pdq = PdqHasher().hash_image(image)  # PdqHash(value=32 bytes, quality)
known = HashList.from_hex(partner_hex_lines, keys=partner_ids)
matches = known.match(pdq) if pdq.is_reliable() else []  # distance <= 31

# Invisible watermark: models pinned by SHA-256, downloaded only on request.
download_trustmark_models("Q", model_dir="/srv/models/trustmark")  # once, at deploy
marker = ImageWatermarker(TrustMarkBackend(model_dir="/srv/models/trustmark"))


def to_webp(img):
    buffer = io.BytesIO()
    img.save(buffer, "WEBP", quality=80)
    return buffer.getvalue()


result = marker.embed_verified(image, 123_456, encode=to_webp)  # escalates strength
publish(result.encoded)  # verified or not
detection = marker.detect_encoded(result.encoded)
assert detection.found and detection.payload.valid

# Visible, viewer-identifying overlay burned into the pixels.
preview = tile_text(image, "viewer 1234", opacity=0.35, angle=30)
```

HEIC/HEIF is not decoded out of the box: no permissively licensed decoder
package exists today. `load_image(..., openers=...)` accepts one the
application has chosen (see docs/image.md).

### Video: frames, fingerprints, watermark

Probing and frame extraction through ffprobe/ffmpeg pipes, PDQ frame-hash
fingerprints with temporal alignment, and the VideoSeal watermark on decoded
frames. Details, thresholds and measured robustness:
[docs/video.md](docs/video.md).

```python
from provenance_kit import PayloadCodec
from provenance_kit.video.fingerprint import fingerprint_video, match_fingerprints
from provenance_kit.video.frames import iter_frames
from provenance_kit.video.watermark import RepetitionCode, VideoSealBackend, embed_verified

fp = fingerprint_video(data, tools=tools)  # PDQ per second; fp.to_bytes() to store
match = match_fingerprints(clip_fp, fp)  # is_match, offset_seconds, matched_frames

backend = VideoSealBackend(model_dir="/srv/models/videoseal")  # strength 0.5 by default
code = RepetitionCode(copies=4)  # 64-bit payload, majority vote
codec = PayloadCodec(code.data_bits, version_bits=4, check_bits=12)
frames = (f.image for f in iter_frames(data, sampling="all", tools=tools))
result = embed_verified(
    frames, codec.encode(42), encoder=my_encoder, backend=backend, repetition=code, codec=codec
)
```

### Audio: PCM, fingerprint and watermark

PCM decoding through ffmpeg pipes, Chromaprint fingerprints (`fpcalc`) with
offset alignment by bit error rate, and the AudioSeal watermark on decoded
PCM. Details, thresholds and measured robustness:
[docs/audio.md](docs/audio.md).

```python
from provenance_kit.audio.fingerprint import compare_fingerprints, fingerprint_audio
from provenance_kit.audio.pcm import decode_pcm
from provenance_kit.audio.watermark import AudioSealBackend, detect_watermark

fp = fingerprint_audio(data, tools=tools)  # fp.to_bytes() to store
match = compare_fingerprints(clip_fp, fp)  # is_match, bit_error_rate, offset_seconds

audio = decode_pcm(data, tools=tools)  # float32 (channels, samples)
backend = AudioSealBackend(model_dir="/srv/models/audioseal")
detection = detect_watermark(audio.samples, audio.sample_rate, backend=backend)
```

### C2PA manifests

Sign C2PA manifests into JPEG, PNG, WebP, GIF, TIFF, HEIF/AVIF, JPEG XL,
MP4/MOV, AVI, WAV, MP3, FLAC and M4A, and read and validate them. The signer
is a Protocol, so the private key can stay in a KMS or HSM (DER ECDSA
signatures are converted for you); `LocalKeySigner` holds a PEM key for
development. Reading never touches the network.

```python
from provenance_kit.provenance.c2pa import (
    ActionType,
    DigitalSourceType,
    GeneratorInfo,
    ManifestAction,
    ManifestSpec,
    TrainingMiningPolicy,
    ValidationState,
    read_manifest,
    sign_asset,
)

spec = ManifestSpec(
    claim_generator=GeneratorInfo("my-pipeline", "1.0"),
    actions=(
        ManifestAction(
            action=ActionType.CREATED, digital_source_type=DigitalSourceType.DIGITAL_CAPTURE
        ),
        ManifestAction(action=ActionType.PUBLISHED),
    ),
    authors=("Jane Doe",),  # cawg.metadata dc:creator
    training_mining=TrainingMiningPolicy.not_allowed(),  # cawg.training-mining
)
signed = sign_asset(jpeg_bytes, "image/jpeg", spec, signer=kms_signer)

report = read_manifest(untrusted_bytes, trust_anchors_pem=trust_list)
report.state  # UNSIGNED, INVALID, VALID or TRUSTED
report.training_mining_not_allowed  # AI training, generative training, mining, inference
```

Labels, specification sources and the signature encodings are documented in
[docs/provenance.md](docs/provenance.md).

### OpenTimestamps

```python
from provenance_kit.provenance.timestamps import OpenTimestamper, sha256_file

ots = OpenTimestamper()  # the official client's calendars; 2 must answer
proof = ots.stamp(sha256_file("master.mov"))  # only a nonced commitment is sent
proof = ots.upgrade(proof)  # hours later: Bitcoin attestations
info = ots.info(proof)  # pending calendars, block heights
result = ots.verify(proof, digest, block_source)  # offline, with your block headers
open("master.mov.ots", "wb").write(proof.to_bytes())  # standard .ots format
```

Calendar responses and proofs are treated as untrusted (https only, deadlines,
size caps, same-origin redirects, an allow-list for upgrades).

## Documentation

* [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): package map, dependency,
  error, security, threading and testing rules (binding for contributors).
* [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md): environment setup, binaries,
  checks and tests.
* [docs/THIRD_PARTY.md](docs/THIRD_PARTY.md): licenses of code, model weights
  and binaries.
* [docs/image.md](docs/image.md), [docs/video.md](docs/video.md),
  [docs/audio.md](docs/audio.md) and [docs/provenance.md](docs/provenance.md):
  the capability modules, thresholds and measured robustness.
* [SECURITY.md](SECURITY.md): reporting vulnerabilities and the threat model.
* [CHANGELOG.md](CHANGELOG.md).

## License

MIT. See [LICENSE](LICENSE). Optional dependencies, models and binaries have
their own licenses: see [docs/THIRD_PARTY.md](docs/THIRD_PARTY.md).
