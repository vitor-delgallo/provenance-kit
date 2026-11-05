# Audio: PCM decoding, Chromaprint fingerprints and the AudioSeal watermark

`provenance_kit.audio` decodes untrusted audio through `ffmpeg`, fingerprints
it with Chromaprint's `fpcalc`, and embeds or detects Meta's AudioSeal
invisible watermark on decoded PCM.

| Module | What it does | Needs |
|---|---|---|
| `provenance_kit.audio.pcm` | `probe_audio` → `AudioInfo`; `decode_pcm` → `PcmAudio` (float32, `(channels, samples)`) | `ffmpeg`, `ffprobe` |
| `provenance_kit.audio.fingerprint` | `AudioFingerprint`, `fingerprint_audio`, `fingerprint_pcm`, `compare_fingerprints` | extra `audio` + `fpcalc` |
| `provenance_kit.audio.watermark` | `AudioSealBackend`, `embed_watermark`, `detect_watermark`, `embed_verified` | extra `audio-watermark` |

## PCM decoding

```python
from provenance_kit.audio.pcm import decode_pcm

audio = decode_pcm(data, sample_rate=16000, channels=1)  # resampled and downmixed by ffmpeg
audio.samples  # float32, shape (1, n), values in [-1, 1]
audio.sample_rate  # 16000
```

Any supported container works, including the audio track of a video (MP4,
Matroska/WebM, AVI, MPEG-TS); the first audio stream is decoded. A FLAC
stream behind an ID3v2 tag (how C2PA signing stores the manifest in FLAC) is
detected and decoded as FLAC. The same input rules as the video modules
apply, from the same code (the core's private `provenance_kit._ffmpeg`): type
sniffed from the bytes, explicit demuxer, `-protocol_whitelist pipe`,
defensive parsing of ffprobe's report, media through a pipe except MP4/M4A
files whose `moov` box follows the media data, which go through a private
temporary file because ffmpeg must seek.
Limits are checked on the byte size, on the declared channels, sample rate and
duration, on the requested layout, and while decoding: ffmpeg runs with `-t`
one second past `max_duration_seconds` and the samples are counted, so media
that under-declares its duration (FLAC or Matroska written to a pipe declare
none) raises `MediaLimitExceededError`. `max_duration_seconds=` on the call
truncates on purpose instead. Downmixing uses ffmpeg's default matrix (stereo
to mono sums the channels at -3 dB each).

## Chromaprint fingerprints

```python
from provenance_kit.audio.fingerprint import (
    AudioFingerprint,
    compare_fingerprints,
    fingerprint_audio,
)

reference = fingerprint_audio(track_bytes)
stored = reference.to_bytes()  # 4 bytes per 0.124 s + 20-byte header
match = compare_fingerprints(fingerprint_audio(clip_bytes), AudioFingerprint.from_bytes(stored))
match.is_match, match.bit_error_rate, match.offset_seconds
```

The media is decoded by ffmpeg (limits enforced) to mono 16-bit PCM at
11,025 Hz, Chromaprint's working rate, and ffmpeg's output is piped straight
into `fpcalc -raw -json -length 0 -` (the runner feeds one tool's
`ToolProcess` to the other's standard input; ffmpeg's output cap, set from
`max_duration_seconds`, bounds what fpcalc receives). fpcalc only ever sees
raw PCM, never the untrusted container. Its report is validated (a list of
unsigned 32-bit integers). `fpcalc` reports a duration of 0.0 when it reads
standard input, so the duration comes from the PCM sample count. Audio
shorter than about 2.7 s yields an empty fingerprint. `fingerprint_pcm`
fingerprints already decoded audio.

**Serialized form** (`to_bytes`, big-endian): `b"PKAF"`, format version 1,
Chromaprint algorithm (2), 2 reserved bytes, duration as a double, count,
then the values as unsigned 32-bit integers.

**Comparison** is by bit error rate over the overlapping 32-bit
sub-fingerprints. All offsets are evaluated at once with FFT
cross-correlations of the 32 bit planes. The best offset maximizes the
binomial z-score of the agreement, `(0.5 - BER) * sqrt(compared bits)`:
for a clip inside the reference that is the lowest error rate, and with
varying overlaps it prefers a long low-error overlap to a short lucky one
(which matters for repetitive content). Offsets with less than
`min_overlap_seconds` (5 s, or the whole shorter fingerprint) of overlap are
ignored. `is_match` is `bit_error_rate <= 0.20` (`DEFAULT_MAX_BIT_ERROR_RATE`).

**Measured robustness** (60 s synthetic test signals, 44.1 kHz stereo, fpcalc
1.6.2, LGPL ffmpeg 8.1; `tests/audio/test_fingerprint_integration.py`):

| Variant | Offset found | Bit error rate |
|---|---|---|
| MP3 64 kb/s | 0 s | 0.039 to 0.040 |
| AAC 32 kb/s, mono, 22.05 kHz | 0 s | 0.065 to 0.070 |
| Opus 24 kb/s, trimmed 17.3 s to 37.3 s | 17.33 s | 0.069 to 0.079 |
| AAC 96 kb/s, trimmed 41.5 s to 49.5 s | 41.48 s | 0.002 to 0.003 |
| Volume x0.3, low-pass 4 kHz, MP3 | 0 s | 0.042 to 0.044 |
| Unrelated signals of similar structure (12 pairs) | none | 0.316 to 0.365 |
| Unrelated pink noise | none | 0.435 |

Offsets are found to within one sub-fingerprint (0.124 s). Not handled:
tempo or pitch changes and heavy mixing with other audio.

## AudioSeal watermark

```python
from provenance_kit import PayloadCodec
from provenance_kit.audio.watermark import AudioSealBackend, detect_watermark, embed_verified

backend = AudioSealBackend(model_dir="/srv/models/audioseal")
codec = PayloadCodec(16, version_bits=0, check_bits=4)  # 4,096 identifiers
result = embed_verified(
    audio.samples,
    audio.sample_rate,
    codec.encode(42),
    encoder=my_encoder,
    backend=backend,
    codec=codec,
)
detection = detect_watermark(samples, rate, backend=backend, codec=codec)
detection.found, detection.confidence, detection.payload
```

**Checkpoints.** The 16-bit base generator and detector of the
`facebook/audioseal` Hugging Face repository, pinned to revision
`3c19eba53390776cf2cc9ed5f6c9ac67ce72ecba` (2025-12-09; model card
`license: mit`) and by SHA-256:

| File | Size | SHA-256 |
|---|---|---|
| `generator_base.pth` | 58,805,980 bytes | `7a845b5fbe9364a63a3909d8ab3fe064d13a76ae4c2e983573e08c69b7b51748` |
| `detector_base.pth` | 34,667,641 bytes | `8a78e8a83584113523e161fc599fcab10fd0e94c04d2eb9d2fa1e9ec91ab69d9` |

URL pattern: `https://huggingface.co/facebook/audioseal/resolve/<revision>/<file>`.
The hashes equal the Git LFS object IDs published by the Hugging Face API and
were recomputed after download. The streaming checkpoints added in December
2025 are not used. `model_dir` defaults to `$PROVENANCE_KIT_MODEL_DIR/audioseal`
or, without the variable, `provenance-kit/models/audioseal` in the per-user
cache directory (`provenance_kit.modelfiles`), and is refused inside
`site-packages`; nothing is downloaded unless `download=True` is passed or
`download_audioseal_models()` is called.

**License check.** The weights were CC-BY-NC 4.0 (`LICENSE_weights` in the
GitHub repository) until commit `2956515` of 2024-04-02 ("Re-update of
license") removed that file and stated that the MIT license now covers the
weights. A first change on 2024-03-09 had been reverted on 2024-03-11. The
base files were uploaded to Hugging Face on 2024-03-11 (as `generator.pth`
and `detector.pth`, renamed on 2024-04-30) and their bytes have not changed
since. They are also byte-identical to the files first published on
2024-02-04 at `https://dl.fbaipublicfiles.com/audioseal/6edcf62f/`. In other
words, no separate set of "post-relicensing" weights exists: the current
checkpoints are the original ones, now under MIT by the licensor's
statement. The backend loads only the pinned Hugging Face files.

**Safe loading.** The checkpoints are pickles that also hold `omegaconf`
configuration objects, which `torch.load(weights_only=True)` refuses. The
backend therefore verifies the SHA-256 first, then reads the file with an
allowlist unpickler: tensors, plain containers and inert placeholders for the
configuration classes. Anything else, and the legacy non-zip format, is
refused. The model configuration comes from the adapter (copied from the
`audioseal` 0.2.1 cards), not from the file. A test checks that a checkpoint
with an embedded `os.mkdir` payload is refused without running it.

**Sample rate and channels.** The models work at 16 kHz mono. Embedding mixes
the channels down, resamples to 16 kHz in memory (windowed-sinc,
`provenance_kit.audio._resample`), computes the watermark, resamples it back,
scales it by `strength` and adds the same watermark to every channel, so a
later mono downmix keeps it. Detection mixes down and resamples the same way.
Long signals are processed in chunks of `chunk_seconds` (10 s). The models run
in eager mode; the backend disables the `torch.compile` path of audioseal's
vendored modules (which would need a C++ compiler).

**Error correction.** 16 bits leave no room for it (decision 8). The decision
is the detector's probability (`found` when it reaches 0.5); add `PayloadCodec`
check bits and read `payload.valid` before trusting the identifier.

**Alignment matters.** AudioSeal is sensitive to sample-level time shifts.
This is an upstream property, reproduced with the stock API: shifting the
16 kHz signal by 3 samples drops the detection probability from 1.0 to 0.06.
Encodings that record their encoder delay decode aligned: AAC in MP4/M4A
(edit list), MP3 written to a seekable file (LAME/Info tag), Ogg Opus
(pre-skip) and FLAC. Raw ADTS AAC and MP3 streamed to a pipe shift the audio
by the encoder delay; the opt-in alignment search (below) recovers them on
broadband content.

**Measured robustness** (8 s, 44.1 kHz stereo, four payloads each, LGPL ffmpeg
8.1 encoders; `tests/audio/test_watermark_integration.py`). Each cell:
detection probability range, bit accuracy, payloads verified.

| Encoding | Broadband content | Sparse tones |
|---|---|---|
| none (SNR 41.2 dB broadband, 26.8 dB tones) | 1.000, 1.000 | 1.000, 0.797 |
| AAC 64 kb/s in M4A | 1.000, 1.000, 4/4 | 1.000, 0.719, 0/4 |
| AAC 128 kb/s in M4A | 1.000, 1.000, 4/4 | 1.000, 0.797, 0/4 |
| 22.05 kHz, AAC 96 kb/s in M4A | 0.998 to 1.000, 1.000, 4/4 | 0.999 to 1.000, 0.781, 0/4 |
| MP3 file 64 kb/s | 0.999, 1.000, 4/4 | 0.999 to 1.000, 0.781, 0/4 |
| MP3 file 128 kb/s | 0.998 to 1.000, 1.000, 4/4 | 0.999 to 1.000, 0.781, 0/4 |
| MP3 file 96 kb/s, mono downmix | 0.999 to 1.000, 1.000, 4/4 | 0.999 to 1.000, 0.812, 0/4 |
| Ogg Opus 32 kb/s | 0.995 to 1.000, 1.000, 4/4 | 1.000, 0.688, 0/4 |
| Raw ADTS AAC 128 kb/s (delay not signaled) | 0.421 to 0.992, 0.594, 0/4 | 0.946 to 0.999, 0.547, 0/4 |
| MP3 128 kb/s to a pipe (delay not signaled) | 0.564 to 1.000, 0.609, 0/4 | 0.898 to 0.976, 0.672, 0/4 |
| unmarked | 0.000 | 0.247 |

Takeaways: with broadband content (speech-like signals, AudioSeal's domain)
and delay-signaling containers the payload survives intact. On sparse tonal
content the detection still works but the 16 bits do not. A misaligned
pipeline breaks both. `embed_verified` exists to catch exactly this in the
application's real pipeline.

**Alignment search (opt-in).** For audio that went through a pipeline that
does not record the encoder delay, `detect_watermark(..., alignment=AlignmentSearch())`
first searches for the number of leading samples to drop:

```python
from provenance_kit.audio.watermark import AlignmentSearch, detect_watermark

detection = detect_watermark(
    samples, rate, backend=backend, codec=codec, alignment=AlignmentSearch()
)
detection.offset_samples  # the samples dropped before detection
```

It tries the offsets `0, step_samples, ...` up to `max_offset_samples`
(defaults 2 and 2048, in samples at the signal's rate: 1,025 offsets), scores
the first `window_seconds` (0.5 s) after each, and keeps the offset whose 16
bits are the most confident (mean `|2p - 1|`). The detection probability
cannot pick the offset: it stays near 1 over a wide range of misalignments.
When no offset reaches `min_bit_confidence` (0.2), the signal is detected
unshifted. Cost: one detector call on 0.5 s of audio per offset, plus the
final detection: 20 to 40 s per search with the defaults on the reference CPU
(AudioSeal on CPU, 44.1 kHz input), linear in the number of offsets; narrow
the range when the delay is known.

Measured (8 s broadband test signal, 44.1 kHz stereo, payloads 1 and 777, the
default search; `test_alignment_search_recovers_unsignaled_delays` and
`test_alignment_search_on_unmarked_audio`, run with `-s`):

| Pipeline | Encoder delay | Offset found | Bits without search | Bits with search |
|---|---|---|---|---|
| Raw ADTS AAC 128 kb/s | 1024 | 1024 and 1024 | 12 and 8 of 16 | 16 of 16, verified (2/2) |
| MP3 128 kb/s to a pipe | 1105 (+1 from the decoder) | 1116 and 1106 | 10 and 9 of 16 | 16 of 16, verified (2/2) |

* Best excerpt bit confidence at the found offset: 0.36 to 0.44 on marked
  broadband audio, 0.04 to 0.05 on unmarked audio (broadband and tones, raw
  and after ADTS AAC), so unmarked audio stays unshifted and its detection
  probability unchanged. Without the confidence floor, the near-random best
  offset of the unmarked tones raised their detection probability from 0.25
  to 0.62 (above the 0.5 threshold), which is why the floor exists.
* On the sparse tonal content the bit confidence is not a reliable guide:
  the search picked offsets 78, 1102 and 1182 instead of 0, 1024 and about
  1106 for payload 1 (bit accuracy 0.625). The 16 bits do not survive on that
  content even when aligned (table above).
* Encodings that record the delay (M4A, MP3 files, Ogg Opus, FLAC) do not
  need the search.

**Thread safety.** `AudioSealBackend` can be shared between threads (model
calls are serialized). The eager-mode switch is a module-global flag in
audioseal, so other threads calling audioseal directly at the same time also
run eagerly.
