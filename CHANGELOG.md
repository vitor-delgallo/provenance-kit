# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Until 1.0, minor versions may change the API.

## [Unreleased]

### Added

- Project foundation: `src/` layout, hatchling build, optional-dependency
  extras per capability, ruff, mypy `--strict`, pytest markers
  (`integration`, `network`) and CI on Linux and Windows with Python 3.11
  and 3.13. Requires Python 3.11 or later.
- `provenance_kit.errors`: exception hierarchy (`ProvenanceKitError`,
  `BackendUnavailableError`, `MediaError` and subclasses,
  `ExternalToolError`, `ExternalToolTimeoutError`, `ModelLoadError`,
  `PayloadError`).
- `provenance_kit.limits.MediaLimits`: validated resource limits for untrusted
  media, with documented defaults.
- `provenance_kit.tools`: `ExternalTools` binary lookup and a shell-free
  runner (`run_tool`, `stream_tool`) with pipes, timeouts, process-tree kill
  and output caps.
- `provenance_kit.tempfiles`: private temporary files and directories that are
  always deleted.
- `provenance_kit.payload`: `PayloadCodec` packing a format version, an
  identifier and optional check bits into a watermark payload.
- `provenance_kit.mediatype`: media type detection from leading bytes.
- `provenance_kit.bits`, `provenance_kit.matching`,
  `provenance_kit.detection`, `provenance_kit.device`: shared bit helpers,
  Hamming matching, the common watermark detection result and device
  selection.
- Documentation: architecture contract, development guide, third-party
  license review, security policy.
- `provenance_kit.image.loading`: safe decoding of untrusted images
  (`load_image`, `iter_frames`, `probe_image`): limits checked from the header
  before decoding (dimensions, rotated size, frame count), type from content,
  one Pillow parser per type, EXIF orientation, ICC profile kept or applied,
  8-bit RGB/RGBA output, `MediaDecodeError` for every decoder failure; JPEG,
  PNG/APNG, GIF, WebP, AVIF; HEIC/HEIF through caller-supplied openers.
- `provenance_kit.image.hashing`: `PdqHasher` / `PdqHash` with a NumPy port
  of Meta's reference (the default; `pdqhash`, extra `pdq-native`, is
  preferred when installed) (bit-exact on the reference vectors, equal to
  ThreatExchange's published hashes), pHash/dHash via `imagehash`,
  `ImageFingerprint`, `compare_fingerprints`, `MatchThresholds` (Meta's PDQ
  defaults: distance 31, quality 50) and `HashList`, an exact vectorized
  matcher for lists of a million hashes or more.
- `provenance_kit.image.watermark`: `ImageWatermarker` (payload codec,
  detection, `embed_verified` with the caller's encoder and strength
  escalation) and `TrustMarkBackend` with SHA-256-pinned model files, offline
  loading, opt-in download (`download_trustmark_models`) and no console
  output.
- `provenance_kit.image.visible`: `tile_text`, a deterministic diagonal tiled
  text watermark with opacity, angle, spacing, relative size, font and shadow.
- `provenance_kit.provenance.c2pa`: `sign_asset` and `read_manifest` on
  c2pa-python 0.38 (image, video and audio containers), `ManifestSpec`
  (actions, `cawg.metadata` authors, `cawg.training-mining`,
  `c2pa.soft-binding`, ingredients), the `C2paSigner` Protocol for KMS/HSM
  keys with DER-to-`r || s` conversion (`ecdsa_der_to_raw`), `LocalKeySigner`,
  optional TSA, and `ManifestReport` with unsigned/invalid/valid/trusted
  states. Network access, remote manifest fetching and thumbnails are off.
- `provenance_kit.provenance.timestamps`: `OpenTimestamper` (stamp, upgrade,
  info, verify against a caller-supplied `BitcoinBlockSource`) with the
  official client's calendars, `TimestampProof` in the standard `.ots`
  format, a hardened calendar HTTP client and `sha256_file`/`sha256_stream`.
  Documentation in `docs/provenance.md`.
- `provenance_kit.video`: `probe_video` / `VideoInfo` (defensive ffprobe
  parsing, limits, rotation), `iter_frames` / `extract_frames` (ffmpeg pipe,
  fixed-rate, key-frame or all frames with timestamps, bounded memory, private
  temporary file only for MP4 with a trailing `moov`), `VideoFingerprint`
  (PDQ per sampled frame, versioned binary form) with `match_fingerprints`
  (temporal alignment, clips), and `VideoSealBackend` (official TorchScript
  v1.0 model, SHA-256-pinned, opt-in download) with `RepetitionCode`
  (whitened soft majority vote and a calibrated detection p-value),
  `embed_watermark`, `detect_watermark` and `embed_verified`. Documentation
  and measured robustness in `docs/video.md`.
- `provenance_kit.audio`: `decode_pcm` / `probe_audio` (float32 PCM through an
  ffmpeg pipe, limits counted while decoding), Chromaprint `AudioFingerprint`
  via `fpcalc -raw -json` on standard input (duration from the PCM, versioned
  binary form) with `compare_fingerprints` (FFT bit-error-rate alignment), and
  `AudioSealBackend` (16-bit base checkpoints pinned by revision and SHA-256,
  allowlist unpickling, in-memory resampling to 16 kHz) with
  `embed_watermark`, `detect_watermark` and `embed_verified`. Documentation and
  measured robustness in `docs/audio.md`.
- `provenance_kit.modelfiles` (consolidation of the three watermark
  backends): `ModelFile`, `default_model_dir` (`$PROVENANCE_KIT_MODEL_DIR/<backend>`,
  else `provenance-kit/models/<backend>` in the per-user cache directory),
  `model_directory`, `verify_model_file`, `read_model_file`,
  `find_model_file`, `download_model_file` and `ensure_model_files`.
  TrustMark, VideoSeal and AudioSeal use it.
- `provenance_kit.tools`: a `ToolProcess` can be the `stdin` of another tool
  (piping), with `ToolProcess.bytes_read` and
  `ToolProcess.output_limit_exceeded`.
- `MediaLimits.max_image_frames` (default 5,000).
- `provenance_kit.provenance.c2pa`: `AssertionPlacement` and
  `ManifestSpec.placements` (`AssertionPlacements`): each assertion is
  written as created or gathered (C2PA 2.3, section 10.2.2).
- AudioSeal detection: an opt-in alignment search over sample offsets
  (`AlignmentSearch`, `detect_watermark(..., alignment=...)`).

### Changed

- The ffmpeg/ffprobe plumbing of the video and audio modules (input staging,
  mandatory arguments, defensive ffprobe JSON parsing) and their argument
  checks live in one private core module each (`provenance_kit._ffmpeg`,
  `provenance_kit._checks`).
- One default model directory for every backend (see `modelfiles` above);
  VideoSeal and AudioSeal used `~/.cache/provenance-kit/<backend>` before. A
  `model_dir` inside `site-packages` is refused for every backend, and the
  download functions keep valid files and replace missing or invalid ones.
- `provenance_kit.mediatype` detects FLAC behind an ID3v2 tag (as c2pa-rs
  writes signed FLAC) instead of MP3; the C2PA module's own check is gone.
- `image.loading` caps animated images with `MediaLimits.max_image_frames`;
  `DEFAULT_MAX_IMAGE_FRAMES` is removed. The `max_frames` argument still
  overrides it per call.
- Packaging: `pdqhash` moved from `image-hash` to its own extra,
  `pdq-native` (it needs a C++ compiler on Linux); `image-hash` installs
  `imagehash` only, `video` needs no Python packages, and `all` does not
  include `pdq-native`.
- `OpenTimestamper(min_calendars=...)` defaults to 2, the official client's
  `-m` (was 1); `DEFAULT_MIN_CALENDARS`.
- C2PA: the authors (`cawg.metadata`) are written as a gathered assertion by
  default; actions, training-mining and soft binding stay created.
- VideoSeal: the default strength is 0.5 (`VIDEOSEAL_DEFAULT_STRENGTH`; was
  the model's 0.2), measured in `docs/video.md`; `strength=None` keeps 0.2.
- TrustMark: `print` is silenced in every loaded `trustmark` module (still a
  module-global name only; `builtins.print` is never patched).
