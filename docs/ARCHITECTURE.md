# Architecture and contributor contract

This document is binding for everyone who adds code to provenance-kit. When a
rule here and a habit of yours disagree, the rule wins; when a rule is wrong,
change this document in the same pull request as the code.

provenance-kit is a toolkit of building blocks for content provenance and
media tracking: invisible and visible watermarks, perceptual hashes,
fingerprints, C2PA manifests and OpenTimestamps proofs. It is a library, not
an application: it does not generate derivatives (resizing, transcoding, HLS),
store anything, talk to databases, or contain product-specific logic. Nothing
in the code, docs or tests names a product or a company's application.

Everything is written in English: code, identifiers, docstrings, comments,
messages and docs.

Section 11 records the maintainers' decisions for v0.1.0; read it before
starting a module.

## 1. Package map

```
src/provenance_kit/
  __init__.py      version, re-exports of errors / MediaLimits / PayloadCodec
  errors.py        exception hierarchy
  limits.py        MediaLimits (resource limits for untrusted media)
  tools.py         ExternalTools, run_tool, stream_tool (ffmpeg, ffprobe, fpcalc)
  tempfiles.py     private_temp_file, private_temp_dir
  payload.py       PayloadCodec, DecodedPayload (watermark payload layout)
  detection.py     WatermarkDetection (common detection result)
  bits.py          bit-string helpers, hamming_distance, bit_error_rate
  matching.py      HammingMatch, hamming_matches (hash-list matching)
  mediatype.py     sniff_media_type, require_media_type (type from bytes)
  device.py        parse_device, resolve_device ("cpu" / "cuda" / "auto")
  _logging.py      library logger, get_logger
  _optional.py     import_optional (lazy import of optional dependencies)
  _typing.py       StrPath, BytesLike

  image/           owner: image agents
    watermark      TrustMark invisible watermark, read-back verification,
                   strength escalation                      extra: image-watermark
    visible        tiled visible text watermark (Pillow)    core install
    hashing        pHash (imagehash), PDQ (pdqhash), hash-list matching
                                                            extra: image-hash
  video/           owner: video agents
    frames         ffprobe probing, frame extraction via ffmpeg pipes
                                                            binaries: ffmpeg, ffprobe
    fingerprint    frame-hash sequences, temporal alignment extra: video
    watermark      Meta VideoSeal invisible watermark       extra: video-watermark
  audio/           owner: audio agents
    fingerprint    Chromaprint via fpcalc, comparison       extra: audio (binary: fpcalc)
    pcm            PCM decode through ffmpeg pipes             binary: ffmpeg
    watermark      Meta AudioSeal invisible watermark       extra: audio-watermark
  provenance/      owner: provenance agents
    c2pa           sign with a pluggable signer, read and validate
                                                            extra: c2pa
    timestamps     OpenTimestamps stamp / upgrade / inspect extra: timestamps
```

Ownership rules for parallel work:

* The foundation modules (top level of `provenance_kit/`) and `tests/core/`
  are owned by the foundation. Module agents do not edit them; if you need a
  change, say so in your report and work around it locally.
* Each name in the map (for example `image.watermark`) may be a module
  (`watermark.py`) or a subpackage (`watermark/`). Inside a subpackage, put
  backend adapters in private modules (`_trustmark.py`) and the Protocols in
  `base.py` or `_protocols.py`.
* Subpackage `__init__.py` files (`image/`, `video/`, `audio/`,
  `provenance/`) stay docstring-only and never import submodules eagerly. You
  may update the docstring line that describes your module.
* Shared files that several agents touch are edited **append-only, in your own
  section**: your extra's line in `pyproject.toml`, your section in
  `README.md`, one bullet under `## [Unreleased]` in `CHANGELOG.md`, your row
  in `docs/THIRD_PARTY.md`.
* Dependencies between subpackages go one way only: `video` may use `image`
  (frame hashes); nothing else crosses subpackages. During parallel work,
  depend on a Protocol you define (for example a `FrameHasher`) rather than on
  another agent's unfinished class, and test with a fake.
* The top-level package does not re-export module APIs: users import
  `provenance_kit.image.hashing`, which keeps `import provenance_kit` cheap.

## 2. Dependency rules

* The core install depends on `numpy` and `Pillow` only, and the foundation
  modules import neither (they import only the standard library).
* Every heavy dependency belongs to an extra (`pyproject.toml`) and is
  imported **lazily**, inside the backend that needs it, through
  `provenance_kit._optional.import_optional(module, extra=..., feature=...)`.
  A missing or broken dependency becomes `BackendUnavailableError`, whose
  message names the extra to install.
* Importing any public module must not import a heavy dependency. Type-only
  imports go under `if TYPE_CHECKING:`. Each module agent adds a test that
  calls `tests.support.assert_imports_lightly([...its modules...])`.
* A missing external binary is reported with
  `BackendUnavailableError.missing_binary(...)`; binaries are located with
  `ExternalTools.resolve()` only.
* The mypy configuration already ignores missing type stubs for the heavy
  packages. Wrap untyped third-party objects behind typed adapters; do not
  let `Any` leak into public signatures.
* Pin extras to the range you tested: `>=tested,<next-minor` for 0.x packages.

## 3. Backends and Protocols

Every ML-model or external-tool feature sits behind a `typing.Protocol`, so
that tests use fakes and applications can swap implementations. The concrete
adapters wrap TrustMark, VideoSeal, AudioSeal, c2pa-python, opentimestamps,
imagehash/pdqhash, ffmpeg and fpcalc.

The pattern (names are illustrative; each module defines its own):

```python
class ImageWatermarkBackend(Protocol):
    """What the image watermark module needs from a model."""

    @property
    def payload_bits(self) -> int: ...

    def embed(self, image: Image.Image, bits: str, *, strength: float) -> Image.Image: ...

    def extract(self, image: Image.Image) -> WatermarkDetection: ...


class TrustMarkBackend:  # implements ImageWatermarkBackend
    def __init__(self, *, variant: TrustMarkVariant = TrustMarkVariant.Q,
                 device: str = "cpu", model_dir: StrPath | None = None) -> None:
        trustmark = import_optional("trustmark", extra="image-watermark",
                                    feature="TrustMark watermarking")
        ...
```

Rules:

* The public, high-level API (functions or classes with domain logic such as
  read-back verification, strength escalation, temporal alignment) takes a
  backend argument typed with the Protocol, and creates the default adapter
  only when none is given.
* Protocols are small: only what the domain logic calls. Mark them
  `@runtime_checkable` only if you actually `isinstance`-check them.
* Adapters convert third-party types and exceptions at the boundary: nothing
  from `torch`, `c2pa` or `opentimestamps` appears in a public signature,
  return value or raised exception type.
* The C2PA signer is a Protocol whose implementation receives the bytes to
  sign and returns the signature, so the private key can stay in a KMS or HSM.
  An in-process key implementation is a convenience, not the only path.

## 4. Data conventions

So that modules written in parallel fit together:

| Thing                    | Representation                                                        |
|--------------------------|-----------------------------------------------------------------------|
| Encoded (untrusted) media| `bytes` (or a binary file object where streaming matters)             |
| Decoded image            | `PIL.Image.Image`; backends convert the mode they need (usually RGB)  |
| Video frame              | `numpy.ndarray`, `uint8`, shape `(height, width, 3)`, RGB, C-contiguous |
| Audio samples            | `numpy.ndarray`, `float32`, shape `(channels, samples)`, range [-1, 1], plus `sample_rate: int` |
| Watermark payload bits   | `str` of `"0"`/`"1"`, most significant first (`provenance_kit.bits`)  |
| Payload meaning          | `PayloadCodec` (version + identifier [+ check bits])                  |
| Detection result         | `WatermarkDetection` or a frozen `kw_only` subclass of it             |
| Binary hash              | non-negative `int` plus its width in bits; serialized as lowercase hex, zero-padded to `bits / 4` digits |
| Hash-list match          | `HammingMatch` (`matching.hamming_matches` or an index returning the same type) |
| Times and durations      | `float` seconds                                                       |
| Paths                    | `StrPath`: a `str` or an `os.PathLike[str]`                           |

## 5. Error handling

The hierarchy lives in `provenance_kit.errors`:

```
ProvenanceKitError
  BackendUnavailableError      missing optional dependency or binary (names the fix)
  MediaError
    MediaDecodeError           corrupt, truncated, undecodable media
    UnsupportedMediaError      valid but unsupported type, codec or layout
    MediaLimitExceededError    exceeds MediaLimits (attributes: limit, maximum, actual)
  ExternalToolError            a subprocess failed (attributes: tool, returncode, stderr_tail)
    ExternalToolTimeoutError   ... or timed out and was killed
  ModelLoadError               weights could not be downloaded, verified or loaded
  PayloadError                 identifier does not fit; invalid decoded payload on require_valid()
```

Rules:

* **Caller mistakes raise `ValueError` or `TypeError`**: wrong types,
  out-of-range arguments, inconsistent configuration. Document them in the
  `Raises:` section.
* **Media and environment failures raise a `ProvenanceKitError` subclass.**
  Module-specific exceptions are allowed in the module (for example
  `SigningError(ProvenanceKitError)` or `CalendarError(ProvenanceKitError)`)
  and must subclass the closest class above. Do not add classes to
  `errors.py` from a module task.
* **Detection does not raise when nothing is found.** It returns a result
  (`found=False`, a confidence, no payload). Lookups return empty lists.
  Validation of a C2PA manifest returns a result with a status and the
  validation errors; it raises only when the input cannot be processed at all.
* **Untrusted input never escapes as an arbitrary exception.** At the adapter
  boundary, catch what the third-party library raises for bad input
  (including broad `Exception` around decoders and parsers) and re-raise as
  `MediaDecodeError` / `UnsupportedMediaError` with `raise ... from exc`.
  Let `MemoryError`, `KeyboardInterrupt` and `SystemExit` propagate.
* **Messages are safe to log**: no media bytes, no payload bits or
  identifiers, no keys, certificates or tokens, no caller file paths, no
  subprocess stderr. Put only names, counts, limits and exit codes in them.

## 6. Security rules for untrusted media

Assume every byte of media comes from an attacker.

**Limits before decoding.** Every function that reads media takes
`limits: MediaLimits | None = None`. Check what can be known up front before
decoding pixels or samples:

* bytes: `limits.check_bytes(len(data))`; when reading a stream, read at most
  `max_bytes + 1` bytes and fail if you got more;
* images: `Image.open()` parses only the header; call
  `limits.check_dimensions(*image.size)` (and check `n_frames` for animated
  formats) **before** `load()`, `convert()` or any pixel access. Never change
  the process-global `PIL.Image.MAX_IMAGE_PIXELS`; our default limits are
  below Pillow's own bomb threshold, so ours trigger first;
* video and audio: probe with ffprobe first and check duration, dimensions,
  frame count, channels and sample rate; then decode with explicit caps
  (`-t`, `-frames:v`) and count frames and samples while reading, because
  headers can lie.

**Type from bytes, not names.** Use `mediatype.require_media_type()` on the
leading bytes and pick the decoder from the result. Never trust an extension,
a file name or a client-supplied MIME type. Sniffing identifies the
container only: probe the streams before relying on their kind.

**No secrets or media in messages or logs.** See the logging rule (section 9).

**Subprocesses.** Always through `provenance_kit.tools` (`run_tool`,
`stream_tool`, `ExternalTools.run/stream`):

* argument lists only, never a shell; the executable comes from
  `ExternalTools.resolve()` (absolute path; `PATH` lookup ignores the current
  directory);
* media in and out through pipes (`pipe:0` / `pipe:1`) whenever the format
  allows;
* every ffmpeg/ffprobe call passes `-nostdin` (input on `pipe:0` still works),
  `-hide_banner`, `-loglevel error`, and
  `-protocol_whitelist pipe` (or `file,pipe` when reading a temporary file).
  This blocks playlists and container tricks (HLS, concat, `subfile`) that
  make ffmpeg open URLs or local files named inside the media;
* pass the input demuxer explicitly (`-f mov`, `-f matroska`, ...) when the
  sniffed type determines it, and refuse playlist demuxers;
* when a path must be given, use an absolute temporary path prefixed with
  `file:` so that a name can never be parsed as an option or a protocol;
* every call has a timeout (default `limits.subprocess_timeout_seconds`); the
  runner kills the whole process tree on timeout. The timeout covers the whole
  run, including the time the caller spends between reads;
* cap captured output (`max_stdout_bytes`); stream large outputs with
  `stream_tool` and `read_exact` instead of capturing them.

**Temporary files.** Avoid them. When unavoidable, use
`tempfiles.private_temp_file` / `private_temp_dir`, pass through the caller's
`temp_dir: StrPath | None = None` argument (an application may point it at a
RAM disk), and never keep them past the call.

**Model weights.** Load weights only from pinned sources: pin a SHA-256 for
every file your adapter downloads or accepts, verify before loading, and raise
`ModelLoadError` on mismatch. Accept a caller `model_dir` so production can
pre-provision weights and run offline; never download during a request unless
the caller allowed it. Prefer `torch.load(..., weights_only=True)` or
TorchScript; never unpickle a file that did not pass the hash check.

**Network.** Only the timestamps module talks to the network, and only to
send digests (never media) to calendars the caller configured, with timeouts
and response-size caps.

## 7. Threading, model loading and devices

* A model-backed backend loads its model **once per instance**, in its
  constructor (preferred: failures surface at startup) or lazily on first use
  behind a `threading.Lock`. Construct backends once and reuse them.
* Every backend class documents its thread safety in a `Thread safety:`
  paragraph. Default for PyTorch-backed backends: safe to share between
  threads, because model calls are serialized with an internal lock; stateful
  modes (AudioSeal streaming) are not shareable and say so. Pure functions
  (hashing, payload coding, matching) are thread-safe.
* Inference runs under `torch.inference_mode()` (or `no_grad()`) with the
  model in `eval()` mode. The library never calls process-global setters
  such as `torch.set_num_threads` or `torch.manual_seed`; applications do.
* Device selection: constructors take `device: str = "cpu"`. Validate with
  `device.parse_device`, resolve with `device.resolve_device(device, torch)`
  after importing torch. `"auto"` picks CUDA when available; explicit
  `"cuda"` without CUDA raises `BackendUnavailableError`. Outputs are moved
  back to CPU NumPy/Pillow objects before they leave the adapter.
* Functions that use randomness take a `seed` or a `numpy.random.Generator`.

## 8. Naming, typing and docstrings

* Python 3.11+ (`requires-python = ">=3.11"`), `from __future__ import annotations` in every module,
  type hints everywhere; `mypy --strict` passes on `src/`.
* Docstrings: **Google style** (`Args:`, `Returns:`, `Raises:`, `Yields:`,
  `Attributes:`), enforced by ruff's pydocstyle rules. Every public module,
  class and function has one. Say what is untrusted and what is thread-safe.
* Results are **frozen dataclasses** (`@dataclass(frozen=True, slots=True)`;
  add `kw_only=True` when subclasses may add fields). Validate in
  `__post_init__`.
* Algorithm and mode choices are **enums** with string values
  (`class HashAlgorithm(enum.StrEnum): PHASH = "phash"`), not free strings.
* Public names are listed in `__all__`; private modules and helpers start with
  `_`. Keyword-only arguments (`*`) after the first one or two positional ones.
* Formatting and lint: `ruff format` and `ruff check` (configured in
  `pyproject.toml`, line length 100, LF line endings).

## 9. Logging

* Get loggers with `provenance_kit._logging.get_logger(__name__)`; all are
  children of `provenance_kit`, which has a `NullHandler`. The library never
  configures handlers, levels or formats.
* At INFO and above: no media bytes, no payload bits or identifiers, no keys or
  secrets, no caller source paths, no subprocess output. DEBUG may include
  truncated tool output and temporary paths, never secrets or payloads.
* Use lazy `%`-style arguments (`log.debug("x=%s", x)`), not f-strings.

## 10. Testing

* **Unit tests** (the default) use fakes for every Protocol and need no heavy
  dependency, binary, model or network. They run in CI on Linux and Windows,
  Python 3.11 and 3.13: `pytest -m "not integration and not network"`.
* **Integration tests** are marked `@pytest.mark.integration` (usually
  `pytestmark = pytest.mark.integration` in a `test_<module>_integration.py`
  file). They obtain dependencies through `tests.support.require_module(...)`,
  `tests.support.require_tool(...)` or the `ffmpeg_path` / `ffprobe_path` /
  `fpcalc_path` fixtures, which **skip with a clear reason** when something is
  missing.
* **Network tests** are marked `@pytest.mark.network` and run only with
  `--run-network`. A test that downloads model weights is both `integration`
  and `network`.
* **Determinism**: use `tests.support.SEED` and the `rng` fixture; never
  depend on wall-clock time or test order.
* **Test data is generated at test time**: images with Pillow/NumPy, audio and
  video with ffmpeg's `lavfi` sources, keys and certificates with
  `cryptography`. Never commit keys, certificates, model weights or large
  binaries. If a tiny fixture is unavoidable, keep it under 100 KB and
  document its origin.
* Layout: `tests/<area>/test_<module>.py` and
  `tests/<area>/test_<module>_integration.py`, where `<area>` is `image`,
  `video`, `audio` or `provenance`. The test directories are packages, so
  file names only need to be unique within a directory.
* Every module adds an `assert_imports_lightly` test for its public modules.

## 11. Decisions (v0.1.0)

Decisions taken by the maintainers on the questions raised while building the
foundation. They are binding for the v0.1.0 module work; where they are more
specific than the sections above, they win.

1. **Python 3.11 or later.** `requires-python = ">=3.11"` (Python 3.10 reached
   end of life in October 2026, and `opentimestamps` 0.5.0 needs 3.11). CI
   runs on Linux and Windows with Python 3.11 and 3.13. Use 3.11 features
   freely (`enum.StrEnum`, `typing.Self`, exception groups); do not add
   version special-casing.

2. **Video and audio watermarks work on decoded media in memory.**
   * The VideoSeal and AudioSeal backends take decoded media and return
     watermarked decoded media: video frames as NumPy arrays (section 4:
     `uint8`, `(height, width, 3)`, RGB), audio as a `float32` NumPy array
     shaped `(channels, samples)` plus `sample_rate`. They never encode.
     Encoding is the application's job.
   * Read-back verification mirrors the image design: an `embed_verified`-style
     function takes a **caller-supplied encoder callable** (decoded media in,
     encoded bytes out), runs it on the watermarked output, decodes the result
     and detects the mark, so verification covers the real encoding the
     application will publish.
   * Decoding helpers belong to the library: `video.frames` decodes frames and
     `audio.pcm` decodes PCM through ffmpeg pipes, with `MediaLimits` checks
     (section 6).

3. **PDQ has two backends behind one Protocol.**
   * `pdqhash`, when it is installed (C++ reference code, fast).
   * A NumPy port of Meta's BSD-licensed reference implementation
     (ThreatExchange `pdq/python`), validated against ThreatExchange's
     published test vectors. Keep the BSD notice with the ported code.
   * Backend selection `auto` (the default) picks `pdqhash` if it imports,
     else the NumPy port.
   * Tests cross-check the two backends on a generated image corpus and
     assert the hashes agree within a small Hamming distance; the bound is
     measured and documented in the module and in the test.

4. **VideoSeal: the official TorchScript v1.0 model (256 bits) is enough for
   v0.1** (`https://dl.fbaipublicfiles.com/videoseal/y_256b_img.jit`). The
   `videoseal` PyPI package is not a dependency; PixelSeal and ChunkySeal are
   out of scope for v0.1.

5. **TrustMark adapter: offline, pinned, quiet.**
   * The model directory is configurable and never inside `site-packages`.
   * The SHA-256 of every model file is pinned in the adapter and verified
     before loading; a mismatch raises `ModelLoadError`.
   * No progress output on stdout or stderr: the adapter suppresses the
     upstream `print()` calls and reports through logging only.
   * Downloading is opt-in and explicit (a separate call or an explicit flag);
     the default never touches the network.
   * Loading from a pre-provisioned directory works fully offline.

6. **C2PA callback signer format.** The provenance module agent confirms by a
   test whether `c2pa-python`'s callback signer expects ECDSA signatures as raw
   `r || s` or as DER, documents the answer, and converts in the adapter so
   that signer implementations (KMS, HSM, in-process keys) state which format
   they return.

7. **OpenTimestamps calendars.** The defaults are the same calendars as the
   official `opentimestamps-client`; the module agent verifies the list from
   that project's source and cites it. The calendar list is always
   configurable by the caller.

8. **Error correction.**
   * AudioSeal (16 bits) has no room for error correction: rely on the
     detector's confidence plus the optional `PayloadCodec` check-bit field.
   * VideoSeal (256 bits): the video module may add a simple, documented
     repetition code with majority vote over the payload bits, and must
     measure the resulting bit accuracy in an integration test.

9. **Project metadata.** Repository and project URLs:
   `https://github.com/vitor-delgallo/provenance-kit`. Author:
   `Vitor Delgallo` (no e-mail address in the metadata).
