# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Until 1.0, minor versions may change the API.

## [0.1.0] - 2026-11-05

First release, published on GitHub only (tag `v0.1.0`); the package is not on
PyPI. Requires Python 3.11 or later.

### Added

#### Project and packaging

- `src/` layout and hatchling build. License expression
  `MIT AND BSD-3-Clause`: the project is MIT, and the PDQ NumPy port keeps
  Meta's ThreatExchange BSD-3-Clause license; both texts (`LICENSE`,
  `LICENSES/ThreatExchange-BSD-3-Clause.txt`) ship in the wheel and the sdist.
- Extras per capability: `image-watermark`, `image-hash`, `pdq-native`,
  `video`, `video-watermark`, `audio`, `audio-watermark`, `c2pa`,
  `timestamps`, `all` (everything except `pdq-native`, which needs a C++
  compiler on Linux) and `dev`. The core install depends on NumPy and Pillow
  only; heavy dependencies are imported lazily, inside the backend that
  needs them.
- ruff, `mypy --strict` and pytest with the `integration` and `network`
  markers. CI on Linux and Windows with Python 3.11 and 3.13 (lint, format,
  types, unit tests), plus a Linux job (Python 3.11 and 3.13) that runs the
  unit and integration tests against FFmpeg 8.1 (BtbN's LGPL build n8.1.3-27 from a month-end
  release, pinned by SHA-256 and cached under its digest), fpcalc 1.6.2
  (pinned by SHA-256) and the extras that need no model weights; there
  `PROVENANCE_KIT_TEST_REQUIRE` makes a missing or broken extra or binary
  fail the job instead of skipping its tests.
- Documentation: the contributor contract (`docs/ARCHITECTURE.md`),
  development guide, third-party license review (`docs/THIRD_PARTY.md`), one
  page per capability with thresholds and measured results, security policy
  and threat model.

#### Foundation (`provenance_kit`)

- `errors`: `ProvenanceKitError`, `BackendUnavailableError` (names the extra
  or binary to install), `MediaError` with `MediaDecodeError`,
  `UnsupportedMediaError` and `MediaLimitExceededError`, `ExternalToolError`
  and `ExternalToolTimeoutError`, `ModelLoadError`, `WatermarkBackendError`
  (a watermark model failed while it ran, or a backend returned malformed
  output, in the image, video and audio modules alike) and `PayloadError`.
  Messages are safe to log.
- `limits.MediaLimits`: validated resource limits for untrusted media (bytes,
  pixels, dimensions, image frames, duration, frames, channels, sample rate,
  decoded audio size, subprocess time). `max_decoded_audio_bytes` (1 GiB)
  bounds duration x rate x channels of decoded PCM, which the separate limits
  alone would let reach about 66 GB.
- `tools`: `ExternalTools` binary lookup and a shell-free runner (`run_tool`,
  `stream_tool`) with pipes, timeouts that kill the process tree, output caps,
  and one tool's `ToolProcess` as another's standard input.
- `tempfiles` (private, always-deleted temporary files), `payload`
  (`PayloadCodec`: format version, identifier and optional check bits),
  `mediatype` (type from content, including FLAC behind an ID3v2 tag),
  `bits`, `matching` (`HammingMatch`), `detection` (`WatermarkDetection`)
  and `device` (`"cpu"`, `"cuda"`, `"auto"`).
- `modelfiles`: one model-file convention for every backend. Files are
  pinned by HTTPS URL, size and SHA-256 (`ModelFile`) and verified before
  loading; the default directory is `$PROVENANCE_KIT_MODEL_DIR/<backend>` or
  `provenance-kit/models/<backend>` in the per-user cache, never inside
  `site-packages`; downloads are opt-in, HTTPS-only, size-capped, time-limited
  (`DEFAULT_DOWNLOAD_TIMEOUT_SECONDS`, 60 s per read, plus a total deadline
  per file of `max(10 x the read timeout, size / DOWNLOAD_MIN_BYTES_PER_SECOND)`
  at 64 KiB/s, enforced by a watchdog that shuts the connection down) and
  atomic.
- Private shared layers for the video and audio modules: `_ffmpeg` (input
  staging through a pipe or a private temporary file, the mandatory ffmpeg and
  ffprobe arguments, and defensive parsing of ffprobe's JSON: a bare `NaN` or
  `Infinity` literal refuses the report, while non-finite field values and
  integers too large for a float are ignored) and `_checks` (argument
  validation; the numeric options it checks refuse an integer too large for a
  float with `ValueError`, like any other out-of-range value).

#### Image (`provenance_kit.image`)

- `loading`: `load_image`, `iter_frames`, `probe_image` for untrusted images.
  Limits are checked from the header before decoding (dimensions, rotated
  size, `MediaLimits.max_image_frames`), the type comes from the content and
  only that type's Pillow parser runs; EXIF orientation applied, ICC profile
  kept or applied, 8-bit RGB/RGBA output, `MediaDecodeError` for every
  decoder failure. JPEG, PNG/APNG, GIF, WebP and AVIF; HEIC/HEIF through
  caller-supplied openers.
- `hashing`: `PdqHasher` / `PdqHash` with a NumPy port of Meta's reference
  (the default, bit-exact on the reference vectors) or `pdqhash` when the
  `pdq-native` extra is installed; strided arrays (rotations, transposes)
  are hashed as displayed, pre-scaling keeps the shorter side at 64 px or
  more, and sides under 5 px give the zero hash with quality 0, as in Meta's
  C++ code. pHash and dHash via `imagehash`. `ImageFingerprint` and
  `compare_fingerprints`: PDQ decides when both hashes are reliable; when one
  is below quality 50 (night scenes, light sketches, dark paintings), pHash
  decides in its place, and a pHash match also needs the PDQ distance within
  its threshold, because unrelated minimal images share pHashes (measured:
  88 to 100 % of copies matched, at most 0.06 % of unrelated same-style
  pairs). Such decisions are flagged `low_quality_fallback=True`;
  `MatchThresholds(allow_phash_fallback=False)` turns the fallback off.
  Blank canvases and flat gradients stay undecided
  (`undecided_reason=LOW_QUALITY`): pHash and dHash decide only past an
  information gate (`MatchThresholds.min_bit_balance`). Crops are not
  handled (a 2 % crop per side is missed most of the time; see
  `docs/image.md`). `MatchThresholds` starts from Meta's PDQ defaults
  (distance 31, quality 50). `HashList`, an exact vectorized matcher for a
  million hashes or more, whose `from_packed` refuses arrays of the wrong
  shape or type.
- `watermark`: `ImageWatermarker` (payload codec, `detect`, `detect_encoded`,
  and `embed_verified(image, identifier, *, encoder=...)`, which runs the
  application's encoder, reads the mark back and escalates the strength;
  `VerifiedEmbedding`'s repr leaves out the image and the encoded bytes) and
  `TrustMarkBackend` (variants Q, P, B and C, BCH modes, SHA-256-pinned model
  files loaded offline from `model_dir`, `download=True` with
  `download_timeout_seconds`, no console output: `print` is silenced in
  TrustMark's own modules, never in `builtins`).
  `download_trustmark_models(model_dir=None, *, variant="Q",
  timeout_seconds=60.0)` returns the model directory.
- `visible`: `tile_text`, a deterministic diagonal tiled text watermark
  (opacity, angle, spacing, relative size, font, shadow); palette images keep
  their transparency, and opaque pixels are never mapped onto a transparent
  palette entry; custom fonts keep Pillow's layout engine (Raqm where
  libraqm is available), so pixels can differ between platforms.

#### Video (`provenance_kit.video`)

- `probe`: `probe_video` / `VideoInfo` with defensive ffprobe parsing,
  limits on the declared values, and the display transform: rotation and
  mirroring (`VideoInfo.mirrored`, for example a phone's "flip" edit).
- `frames`: `iter_frames` / `extract_frames` through an ffmpeg pipe, at a
  fixed rate (the frame labelled `t` is the one shown at `t`), at key frames
  or all frames, with timestamps and bounded memory, frames oriented exactly
  as ffmpeg's autorotation would; a private temporary file only for MP4 with
  a trailing `moov` box.
- `fingerprint`: `VideoFingerprint` (PDQ per sampled frame, versioned binary
  form, `fps` from 0.001 to 240), `fingerprint_video` and
  `match_fingerprints`, which looks for a matched **segment**: consecutive
  matches at most 2 s apart whatever lies between them (a longer gap ends
  the segment, black and flat frames included, and such frames count against
  it), at least 5 s from first to last match with at least half of its
  sampling instants matched, and 3 distinct pictures on each side, where
  frames within the 40-bit match threshold of a picture already counted in
  the segment are that picture (`static_distance` defaults to
  `max_distance`). Still, grainy or blinking end cards and short stings
  padded with black are not a match. `VideoMatch` reports where the segment
  is in both videos, its exact distinct-picture count and `match_fraction`
  over its span; out-of-range rates and time arguments above 1e9 s raise
  `ValueError`.
- `watermark`: `VideoSealBackend` (the official TorchScript v1.0 model, 256
  bits, SHA-256-pinned; default strength 0.5, measured in `docs/video.md`,
  `strength=None` keeps the model's 0.2; `model_dir`, `download`,
  `download_timeout_seconds`, and a `model_path` / `sha256` override; the
  model is loaded from the bytes whose SHA-256 was checked, never re-read
  from the path), `download_videoseal_models(model_dir=None, *,
  timeout_seconds=60.0)` returning the model directory, `RepetitionCode` (whitened soft majority
  vote with a calibrated detection p-value), `embed_watermark`,
  `detect_watermark` and `embed_verified(frames, bits, *, encoder=...)`.

#### Audio (`provenance_kit.audio`)

- `pcm`: `decode_pcm` / `probe_audio`, float32 PCM through an ffmpeg pipe.
  The decoded size (`max_decoded_audio_bytes`) is refused from the probe
  before ffmpeg starts and counted while reading. Samples are de-interleaved
  chunk by chunk into one buffer whose size is fixed before the decode (the
  declared duration plus 1 % and 1 s; media that declares none, or holds
  more, is counted by a first decode and read into an exact buffer by a
  second). A call therefore holds about one copy of the result and never
  more than `max_decoded_audio_bytes` plus a few MiB of read buffers. The
  float downmix is normalized (stereo to mono is the mean). The public API is
  `decode_pcm`, `probe_audio`, `AudioInfo` and `PcmAudio`; the helpers shared
  with the fingerprint module are private.
- `fingerprint`: Chromaprint `AudioFingerprint` via `fpcalc -raw` fed by an
  ffmpeg pipe (duration from the PCM, versioned binary form) and
  `compare_fingerprints`, which aligns by bit error rate (FFT) and leaves
  Chromaprint's silence code out of the bit count and the overlap. A match
  needs at least 16 compared items, 8 distinct values on each side, and a bit
  error rate below half the chance rate (`AudioMatch.chance_bit_error_rate`).
  Both rates are taken over the compared pairs, with a run of identical pairs
  counted once (`AudioMatch.chance_test_bit_error_rate`), so independent
  noise or steady drones followed by silence on one side and music on the
  other do not match. A match is reported when any eligible offset passes,
  even behind hundreds of offsets of a low-diversity alternation.
  `min_overlap_seconds` and `max_offset_seconds` accept at most
  `MAX_COMPARISON_SECONDS` (about 34.4 hours).
- Audio fingerprints: the bit error rate threshold of `compare_fingerprints`
  depends on the overlap length. It is `max_bit_error_rate` (0.20) from 20 s
  of compared span, 0.11 (0.55 x 0.20) up to 5 s, and linear in between
  (`overlap_threshold()`, `SHORT_OVERLAP_SECONDS`, `LONG_OVERLAP_SECONDS`,
  `SHORT_OVERLAP_FACTOR`; the threshold that applied is
  `AudioMatch.bit_error_threshold`). Windows of 5 to 10 s of different songs
  that sound alike stay at or below 0.3 % false matches, and every measured
  degradation (MP3 and AAC at 64 kb/s, Opus at 32 and 48 kb/s, resampling,
  volume, EQ, mono) is found on clips of 15 s and longer.
- `watermark`: `AudioSealBackend` (16-bit base checkpoints pinned by
  revision and SHA-256 and unpickled, through an allowlist, from the verified
  in-memory bytes; in-memory resampling to 16 kHz; `model_dir`, `download`,
  `download_timeout_seconds`, and per-checkpoint path / SHA-256 overrides),
  `download_audioseal_models(model_dir=None, *,
  timeout_seconds=60.0)` returning the model directory, `embed_watermark`,
  `detect_watermark` and `embed_verified(samples, rate, bits, *, encoder=...)`.
  `AudioWatermarkDetection.verified` (found **and** the payload's check bits
  match) is the evidence; `found` alone is not, because unmarked click trains
  fool the detector. `verified` is 2^-c evidence for c check bits: a found
  unmarked signal passes with that probability (up to 2 x 2^-c with the
  search). The recommended codec is `PayloadCodec(16, version_bits=0,
  check_bits=8)` (256 identifiers); measured rates are in `docs/audio.md`.
  `AlignmentSearch`, an opt-in search over sample offsets for audio shifted
  by an unsignaled encoder delay, uses a shift only when the shifted
  detection is found and corroborated.

#### Provenance (`provenance_kit.provenance`)

- `c2pa`: `sign_asset` and `read_manifest` on c2pa-python 0.38 (image, video
  and audio containers). `ManifestSpec` (actions, `cawg.metadata` authors,
  `cawg.training-mining`, `c2pa.soft-binding`, ingredients) with
  `ManifestSpec.placements` choosing created or gathered assertions (C2PA 2.3,
  section 10.2.2; authors are gathered by default). The `C2paSigner` Protocol
  lets the key stay in a KMS or HSM, with DER-to-`r || s` conversion
  (`ecdsa_der_to_raw`); `LocalKeySigner` for development; optional TSA.
  Signing refuses a certificate whose key usage lacks `digitalSignature`
  (C2PA 2.3, section 14.5.1); the chain must be PEM blocks labelled exactly
  `CERTIFICATE`, and the C2PA library receives it re-encoded from the
  certificates checked, so the checked leaf is the signing leaf; malformed
  chains and extensions raise `ValueError`. A chain with `openssl pkcs12`
  Bag Attributes text is refused. `ManifestReport` with unsigned, invalid,
  valid and trusted states; `training_mining_not_allowed` treats
  `constrained` as `notAllowed`, as CAWG TDM 1.1 requires
  (`all_declared_not_allowed` keeps the literal view). Network access, remote
  manifest fetching and thumbnails are off.
- `timestamps`: `OpenTimestamper` (stamp, upgrade, info, and offline verify
  against a caller-supplied `BitcoinBlockSource`) with the official client's
  calendars and `min_calendars` 2 (`DEFAULT_MIN_CALENDARS`; a value above the
  number of calendars is refused); `TimestampProof` in the standard `.ots`
  format (`from_bytes(..., max_proof_bytes=...)`); a hardened calendar HTTP
  client: https only; one deadline for the whole exchange, name resolution,
  connect, TLS handshake, headers and body included, enforced by a watchdog
  on Windows and Linux alike, so the calling thread is free at the deadline
  (a name lookup still running is left to a daemon thread, at most 32 at
  once); size caps; bodies shorter than their `Content-Length` refused;
  same-origin redirects only. `sha256_file` and `sha256_stream`.

[0.1.0]: https://github.com/vitor-delgallo/provenance-kit/releases/tag/v0.1.0
