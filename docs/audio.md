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

audio = decode_pcm(data, sample_rate=16000, channels=1)  # resampled and downmixed (mean)
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
truncates on purpose instead.

**Decoded size (decompression bombs).** The duration (3 h), rate (192 kHz)
and channel (8) limits are separate, and their product is about 66 GB of
float32: a 16 KB FLAC of 20 s of 8-channel 192 kHz silence decodes to 123 MB
(a ratio of 7,700). `MediaLimits.max_decoded_audio_bytes` (default 1 GiB)
bounds the PCM that `decode_pcm` holds: duration x output rate x output
channels x 4 bytes. It is checked **before** ffmpeg starts, from the declared
duration (or the caller's `max_duration_seconds=`, whichever is shorter), and
again on the running byte count, for media that declares no duration or
lies; the error names `max_decoded_audio_bytes`. 1 GiB holds 46 minutes of
48 kHz stereo or 3 hours of 16 kHz mono: decode long media at a lower rate or
channel count, or raise the limit. `embed_verified` decodes the encoder's output the same way.
`embed_watermark` and `detect_watermark` take arrays the caller already
holds; the AudioSeal backend allocates two to three times their size again.
Fingerprinting streams its 16-bit PCM into `fpcalc` and holds none of it, so
only the duration bounds it.

**Downmixing** (`channels=` below the source's) uses ffmpeg's matrix
normalized so that no output sample can exceed full scale
(`-rematrix_maxval 1.0`, checked with the bundled ffmpeg 8.1): stereo to mono
is the mean of the two channels, 5.1 to stereo is scaled down likewise.
ffmpeg's float output does not normalize by default and summed stereo to mono
at -3 dB per channel, which raised identical channels by 3 dB: a 0.9-peak
sine came out at 1.27 and 42 % of its samples were clipped to 1.0, which
corrupted AudioSeal payload bits. The 16-bit output used for fingerprints
already normalizes; its fingerprints are unchanged (checked byte for byte on
stereo, 5.1 and mono inputs). Mono to stereo copies the channel at -3 dB
(ffmpeg's default).

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
shorter than about 2.7 s yields an empty fingerprint, and audio shorter than
about 4.6 s (fewer than 16 sub-fingerprints) never matches (see below).
`fingerprint_pcm` fingerprints already decoded audio.

**Serialized form** (`to_bytes`, big-endian): `b"PKAF"`, format version 1,
Chromaprint algorithm (2), 2 reserved bytes, duration as a double, count,
then the values as unsigned 32-bit integers.

**Comparison** is by bit error rate (BER) over the overlapping 32-bit
sub-fingerprints. All offsets are evaluated at once with FFT
cross-correlations of the 32 bit planes. The decision rule:

1. **Silence is not compared.** Chromaprint gives digital silence one
   constant value, 627964279 (`SILENCE_VALUE`); pairs where either side holds
   it are left out of both the bit count and the overlap (`overlap_items`
   counts compared pairs), as AcoustID's `acoustid_compare` skips it.
2. **Eligible offsets** have at least `min_overlap_seconds` (5 s) of compared
   pairs, or all the informative items of the shorter fingerprint if it is
   shorter, and **never fewer than 16** (`MIN_COMPARED_ITEMS`). 16 items span
   2 s of hops: a query needs at least 4.6 s of audio to match (Chromaprint's
   first item needs 2.7 s).
3. An eligible offset **passes** when its BER is at most `max_bit_error_rate`
   (0.20, `DEFAULT_MAX_BIT_ERROR_RATE`), **and** below half its chance error
   rate (`MAX_CHANCE_ERROR_FRACTION`, 0.5), **and** each side of the compared
   pairs holds at least 8 distinct values (`MIN_DISTINCT_VALUES`). The chance
   error rate (`AudioMatch.chance_bit_error_rate`) is the BER that two
   unrelated signals with the same bit statistics would show: the mean over
   bits of `p_q (1 - p_r) + p_r (1 - p_q)`, with `p` how often the bit is 1 on
   each side of the overlap. It is about 0.42 for music, 0.16 to 0.32 for
   stationary noise and 0 for a constant value: noise recordings share the
   bits that the noise statistics fix, and agree at a BER as low as a genuine
   match (0.15) without being related.
4. `is_match` is true when **any** offset passes; the reported offset is the
   passing one with the highest binomial z-score, `(0.5 - BER) *
   sqrt(compared bits)`, which prefers a long low-error overlap to a short
   one (for repetitive content). When none passes, the eligible offset with
   the highest z-score is reported with `is_match=False`. An upload that
   starts with the last seconds of a reference is therefore found even if a
   long unrelated overlap scores a higher z-score (before, the z-score chose
   first and the threshold decided after: a genuine 7.3 s overlap at BER
   0.015 lost to a long unrelated overlap at 0.325, so no match).

**Measured robustness** (60 s synthetic test songs, 44.1 kHz stereo, fpcalc
1.6.2, LGPL ffmpeg 8.1; `tests/audio/test_fingerprint_integration.py`, plus
120 s renders of the same generators for the clip rows):

| Variant | Offset found | Bit error rate | Matched |
|---|---|---|---|
| MP3 64 kb/s | 0 s | 0.037 to 0.052 | all |
| AAC 32 kb/s, mono, 22.05 kHz | 0 s | 0.065 to 0.088 | all |
| Opus 24 kb/s, trimmed 17.3 s to 37.3 s | 17.33 s | 0.045 to 0.092 | all |
| AAC 96 kb/s, trimmed 41.5 s to 49.5 s | 41.48 s | 0.001 to 0.007 | all |
| Volume x0.3, low-pass 4 kHz, MP3 | 0 s | 0.037 to 0.046 | all |
| AAC clips of 5, 6 and 8 s (3 positions x 7 songs each) | within 0.25 s | 0.039 to 0.094 | 63 of 63 |
| AAC clips of 3 to 4.5 s (3 to 15 items) | none | | 0 of 84 (below 16 items) |
| An upload made of the reference's last 10 s, then 120 s of another song | 79.98 s | 0.025 | yes (7.3 s overlap) |

Genuine matches stayed at or below 0.43 of their chance error rate (0.29 with
5 s of overlap or more).

**Measured false positives** (independent recordings; "before" is the rule
of the first implementation: z-score choice, then the threshold, minimum
overlap shrinking to the shorter fingerprint):

| Unrelated pair | Before | Now | BER / chance at the reported offset |
|---|---|---|---|
| Suite songs, 10 s clips taken 10 s in vs other 60 s songs (12 pairs) | 0 | 0 | BER 0.316 to 0.365 |
| Whole 60 s songs (12 pairs), whole 120 s songs (56 pairs) | 0, 0 | 0, 0 | |
| 30 s of pink noise vs a 60 s song | 0 | 0 | BER 0.435 |
| Unrelated music (NumPy generator `tests.audio.media.music`), 3 s clips vs a 6 min reference | 32 of 100 | 0 of 100 | no eligible offset |
| The same, 4 to 20 s clips (100 each) and whole 120 s tracks (30 pairs) | 0 | 0 | best BER 0.22 or more |
| 8 s silent intro + music vs other music + 8 s silent outro | match (BER 0.000) | no | 0.428 / 0.453 |
| White, blue, violet, velvet noise; hiss (-40 dBFS, high-passed); -60 dBFS noise floor | match (BER 0.15 to 0.16) | no | 0.89 to 0.92 of chance (0.16 to 0.18) |
| Pink, brown noise | no (BER 0.24, 0.30) | no | 0.91, 0.94 of chance |
| A steady tone (one value throughout, like hum) | match (BER 0.000) | no | chance 0.000 |

Stationary noise stays at 0.82 or more of its chance error rate at every
offset, against at most 0.43 for genuine matches.

**Content that is alike for a few seconds.** The suite's `song(k)` generator
renders slow glissandos whose frequency modulation grows with time; after
about 20 s the songs sweep through similar ranges, and windows of one song
resemble windows of another. Comparing a window taken anywhere in a 120 s
song with another whole song (60 windows each), the share reported as a
match was:

| Window | 5 s | 8 s | 10 s | 15 s | 20 s | 30 s, 45 s |
|---|---|---|---|---|---|---|
| Before | 48 % | 43 % | 27 % | 18 % | 7 % | 0 % |
| Now (threshold 0.20) | 38 % | 23 % | 13 % | 2 % | 2 % | 0 % |

With a stricter threshold (100 windows each, the new rule): 0.15 gave 10, 1,
2, 0 and 0 % for 5, 8, 10, 15 and 20 s windows, 0.12 gave 2 % for 5 s
windows and 0 % from 8 s, and 0.10 gave 0 %; every genuine variant above
(BER at most 0.094) still matched at 0.10. Chromaprint cannot tell apart
signals that sound alike over the compared span: for short queries against
content that may be similar (covers, loops, the same instrument and key),
pass `max_bit_error_rate=0.12` or require a longer `min_overlap_seconds`.

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
result.verified  # the payload survived the application's encoder
detection = detect_watermark(samples, rate, backend=backend, codec=codec)
if detection.verified:  # found AND the check bits match; `found` alone is not evidence
    identifier = detection.payload.identifier
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

**Error correction and evidence.** 16 bits leave no room for error
correction (decision 8). `found` is the detector's decision (its probability
reaches 0.5) and is **not evidence on its own**: the detector is fooled by
some unmarked periodic signals. Use a `PayloadCodec` with check bits and rely
on `AudioWatermarkDetection.verified`: true only when the watermark is found
**and** the check bits match (with 4 check bits random bits pass with
probability 1/16). Without a codec, or with a codec without check bits,
`verified` is always false.

Measured on unmarked signals (`detect_watermark` without search, codec with
4 check bits; `test_unmarked_click_trains_fool_the_detector_but_not_the_check_bits`
and a one-off script run on 2026-11-05):

| Unmarked signal | Detection probability | `found` | `verified` |
|---|---|---|---|
| Broadband and tonal test signals, 16 / 44.1 / 48 kHz (6) | 0.000 to 0.247 | 0 | 0 |
| White, pink, brown noise (3) | 0.000 | 0 | 0 |
| Sines at 50 Hz and 1 kHz, 100 Hz square wave, 3 rates (9) | 0.000 to 0.426 | 0 | 0 |
| One-sample click trains at 50 and 100 Hz, 3 rates, 4 phases (24) | 0.000 to 0.967 | 12 | 0 |

The click trains' detection probability depends on their phase (for 50 Hz at
44.1 kHz: 0.935, 0.933, 0.967 and 0.021 for the four phases). Earlier versions
of these docs gave 0.0 to 0.27 as the unmarked range: that holds for
broadband and tonal content only.

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

(Unmarked click trains score up to 0.967; see the table above.)

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
the first `window_seconds` (0.5 s) after each, and picks the offset whose 16
bits are the most confident (mean `|2p - 1|`). The detection probability
cannot pick the offset: it stays near 1 over a wide range of misalignments.
When no offset reaches `min_bit_confidence` (0.2), the signal is detected
unshifted.

**A shift must be corroborated.** Reaching the confidence floor is not
enough: an unmarked 100 Hz click train at 44.1 kHz reaches 0.205 at offset
380, its bits are the same in every window because the content repeats, and
dropping those 380 samples raised its detection probability from 0.136 to
0.948 (the first implementation reported it as found). The chosen offset is
therefore used only when:

* the codec has check bits and the payload read after the shift passes them
  (the click train's did not; the genuine shifts below all did); or
* there are no check bits and the unshifted signal is already found: the
  shift may then only refine its bits. Without check bits the search never
  turns "not found" into "found".

Otherwise the result is the unshifted detection, with `offset_samples=0`.
Requiring the bits of two disjoint windows to agree was measured and
rejected: the click train's windows agree on 16 of 16 bits, while the genuine
shifted marks agree on only 12 to 16 (0.5 s windows are noisy; the bits over
the whole signal are right). Pass a codec with check bits: a shifted mark
that is weakly detected unshifted (raw ADTS AAC, payload 777: 0.421) is
recovered only with them.

Cost: one detector call on 0.5 s of audio per offset, plus the detection of
the whole signal unshifted and, when an offset stands out, shifted: 20 to
40 s per search with the defaults on the reference CPU (AudioSeal on CPU,
44.1 kHz input), linear in the number of offsets; narrow the range when the
delay is known.

Measured (8 s broadband test signal, 44.1 kHz stereo, payloads 1 and 777, the
default search; `test_alignment_search_recovers_unsignaled_delays` and
`test_alignment_search_on_unmarked_audio`, run with `-s`):

| Pipeline | Encoder delay | Offset found | Bits without search | Bits with search |
|---|---|---|---|---|
| Raw ADTS AAC 128 kb/s | 1024 | 1024 and 1024 | 12 and 8 of 16 | 16 of 16, verified (2/2) |
| MP3 128 kb/s to a pipe | 1105 (+1 from the decoder) | 1116 and 1106 | 10 and 9 of 16 | 16 of 16, verified (2/2) |

* Best excerpt bit confidence at the found offset: 0.36 to 0.41 on marked
  broadband audio, 0.04 to 0.05 on unmarked broadband and tonal audio (raw
  and after ADTS AAC), 0.12 to 0.21 on unmarked click trains. The floor keeps
  the near-random best offset of unmarked tones from being considered
  (without it, their detection probability rose from 0.25 to 0.62); the
  corroboration rule above handles what reaches the floor. Measured on
  unmarked audio with the default search: broadband and the 100 Hz click
  train stay unshifted and not found (`test_alignment_search_on_unmarked_audio`).
* Unshifted detection probability of the four shifted marks above: 0.992,
  0.421 (ADTS) and 1.000, 0.564 (MP3 pipe); after the search, 1.000 and all
  four payloads pass the check bits.
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
