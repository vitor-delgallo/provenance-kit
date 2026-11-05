# Video: probing, frames, fingerprints and the VideoSeal watermark

`provenance_kit.video` reads untrusted video through the external `ffprobe`
and `ffmpeg` binaries, fingerprints it with perceptual frame hashes, and
embeds or detects Meta's VideoSeal invisible watermark on decoded frames.

| Module | What it does | Needs |
|---|---|---|
| `provenance_kit.video.probe` | `probe_video` → `VideoInfo` (duration, size, rotation, frame rate, codecs, stream counts) | `ffprobe` |
| `provenance_kit.video.frames` | `iter_frames` / `extract_frames`: RGB frames with timestamps, at a fixed rate, at key frames or all | `ffmpeg`, `ffprobe` |
| `provenance_kit.video.fingerprint` | `VideoFingerprint` (PDQ per sampled frame), `fingerprint_video`, `match_fingerprints` | binaries (PDQ is in the core install; extra `pdq-native` for the C++ backend) |
| `provenance_kit.video.watermark` | `VideoSealBackend`, `RepetitionCode`, `embed_watermark`, `detect_watermark`, `embed_verified` | extra `video-watermark` |

Binaries are located with `ExternalTools` (`PROVENANCE_KIT_FFMPEG`,
`PROVENANCE_KIT_FFPROBE`, then `PATH`); see [DEVELOPMENT.md](DEVELOPMENT.md).

## Probing and frames

```python
from provenance_kit.video.frames import FrameSampling, iter_frames
from provenance_kit.video.probe import probe_video

info = probe_video(data)  # data: bytes of an upload
for frame in iter_frames(data, fps=1.0, max_side=512):
    frame.index, frame.timestamp, frame.image  # image: uint8 (h, w, 3), RGB, read-only
```

**Untrusted input.** The container is detected from the bytes (MP4,
QuickTime, WebM, Matroska, AVI, MPEG-TS) and passed to the tools explicitly
(`-f mov`, ...), with `-protocol_whitelist pipe` so the media cannot make
ffmpeg open URLs or files. This plumbing (input staging, the mandatory
arguments, the defensive report parsing) is shared with the audio modules in
the core's private `provenance_kit._ffmpeg` module. ffprobe's JSON report is parsed defensively: wrong
types, `NaN`/`Infinity`, out-of-range values, cover-art streams and more than
64 streams are ignored or refused. Failures of the tools on bad media become
`MediaDecodeError`; timeouts kill the whole process tree.

**Pipes and temporary files.** Media goes to the tools through a pipe. The one
exception is an MP4/QuickTime file whose `moov` box (the sample tables) comes
after the `mdat` box (the media data): ffmpeg must seek back to the samples
after reading the tables, which a pipe cannot do. Such files (ffmpeg writes
them unless `-movflags +faststart` is given) are copied to a private temporary
file (`tempfiles.private_temp_file`, in `temp_dir` if given) for the duration
of the call. Fragmented MP4 and `moov`-first files stream through the pipe.

**Limits** (`MediaLimits`): the byte size first; then the declared dimensions,
duration and frame count from the probe; then the expected number of output
frames; and while decoding, ffmpeg runs with `-frames:v` and `-t` caps and the
frames and their timestamps are counted, so media that runs past
`max_frames` or `max_duration_seconds` raises `MediaLimitExceededError`
whatever its header said. Headers with placeholder values (an AVI written to a
pipe declares 2**30 frames) are refused by the limits.

**Sampling.** `FrameSampling.FPS` (default) uses ffmpeg's `fps` filter with
`round=up`, after rebasing timestamps to the first frame. The frame with
timestamp `index / fps` is the frame displayed at that instant (the last one
whose presentation time is at or before it), for constant and variable frame
rates alike, independent of the container's start offset; the test suite
checks this on lossless, frame-numbered videos at several rates, including a
variable-rate one. (With the filter's default `round=near`, the frame
labelled `t` was the one displayed at about `t + 0.5 / fps`: source frames
14, 44, 74, ... instead of 0, 30, 60, ... at 1 fps from 30 fps.)
`KEYFRAMES` decodes key frames only (`-skip_frame nokey`) and takes their
timestamps from an ffprobe pass. `ALL` returns every frame with nominal
timestamps (`index / frame_rate`, exact for constant frame rates).

**Rotation and mirroring.** The display transform (display matrix or legacy
`rotate` tag) is applied in the filter graph with the same filters ffmpeg's
own autorotation inserts, so frames come out upright. A display matrix with a
negative determinant also mirrors the picture (a phone's "flip" edit, or
`ffmpeg -display_hflip`): `VideoInfo.mirrored` is then true, and the frames
are flipped horizontally before `VideoInfo.rotation` is applied. The test
suite compares the frames with ffmpeg's own autorotation for rotations of 90
and 270 degrees, a horizontal flip, a vertical flip and the flips combined
with rotations. Sample aspect ratio is not applied.

**Memory.** One frame at a time; ffmpeg scales every frame to the fixed
output size, so a stream that changes resolution cannot inflate the records.

## Fingerprints

```python
from provenance_kit.video.fingerprint import (
    VideoFingerprint,
    fingerprint_video,
    match_fingerprints,
)

reference = fingerprint_video(original_bytes)  # 1 frame per second, PDQ
stored = reference.to_bytes()  # 33 bytes per second + 20-byte header
query = fingerprint_video(clip_bytes)
match = match_fingerprints(query, VideoFingerprint.from_bytes(stored))
match.is_match, match.offset_seconds, match.similarity
match.matched_seconds, match.distinct_frames  # the matched segment
match.query_start_seconds, match.query_end_seconds  # where it is in the query
match.reference_start_seconds, match.reference_end_seconds  # and in the reference
```

A `VideoFingerprint` is a sequence of PDQ hashes (256 bits, with their
quality) of frames sampled at a fixed rate; frames are scaled to 256 x 256
before hashing. Hashing goes through the `FrameHasher` Protocol; the default
is `provenance_kit.image.hashing.PdqHasher` (loaded lazily, so the module
imports without it).

**Serialized form** (`to_bytes`, big-endian): `b"PKVF"`, format version 1,
hash size in bytes, 2 reserved bytes, `fps` as a double, frame count, the
hashes, then one quality byte per hash. `from_bytes` validates everything and
raises `ValueError` on anything else. (Unchanged by the segment rule below:
it only changes how fingerprints are compared.)

**Matching: the matched segment.** For every temporal offset, the aligned
frame pairs are walked in time order; a pair *matches* when its Hamming
distance is at most `max_distance`, and frames with PDQ quality below
`min_quality` (flat or black frames) are skipped. A *segment* is a run of
matched pairs that may step over at most `max_gap_seconds` of unmatched pairs
at a time. Its *distinct frames* are its matched frames after collapsing
runs of near-identical consecutive samples (each within `static_distance`
bits of the first sample of its run) into one, on each side, keeping the
smaller count: a still end card is one distinct frame however long it is
shown. A segment is a match when it has at least `min_distinct_frames`
distinct frames, spans at least `min_matched_seconds` (first to last matched
sample, inclusive) and at least `min_match_fraction` of its compared pairs
match. The result is the best segment over all offsets (qualifying first,
then more distinct frames, more matched pairs, a higher fraction, a lower
mean distance, a smaller offset), with its position in both videos.

None of these thresholds depends on the length of either video, which fixes
two failures of the previous rule (best offset by matched frames, then "at
least 3 matched and half of the whole overlap"): 3 seconds of a shared end
card made two unrelated videos match (3 of 6 frames), and an exact 15-second
excerpt was missed when the query's other content overlapped the reference
at that offset (15 of 40). Cost: one Hamming distance per pair of frames,
computed one row at a time (memory proportional to the longer fingerprint);
`max_offset_seconds` restricts the offsets considered.

| Threshold | Default | Why |
|---|---|---|
| `max_distance` | 40 of 256 bits | re-encodes measure 6 to 10 (median); a trimmed clip off the sampling grid up to 34; unrelated frames about 128 (minimum 102) |
| `min_quality` | 50 | PDQ's guidance for featureless frames |
| `min_distinct_frames` | 3 | a still picture (end card, title, frozen frame) is one frame |
| `min_matched_seconds` | 5.0 | longer than typical platform end cards and logo stings |
| `min_match_fraction` | 0.5 | at least half of the segment's compared pairs match |
| `max_gap_seconds` | 2.0 | an occasional noisy sample or scene cut does not split a segment |
| `static_distance` | 16 bits | consecutive samples of a still within one encode: 0 to 2 bits (a still with heavy grain 4 to 14); moving content 40 or more |

**Measured robustness** (2026-11-05: 40 s of a fast-changing synthetic
source, 640x360, 1 frame per second, PDQ via the NumPy port, LGPL ffmpeg 8.1
encoders; `tests/video/test_fingerprint_integration.py`, run with `-s`):

| Variant | Offset found | Matched / compared | Distinct | Similarity | Match |
|---|---|---|---|---|---|
| VP9 at 150 kb/s | 0 s | 40 / 40 | 40 | 0.967 | yes |
| Scaled to 320x180, H.264 300 kb/s | 0 s | 40 / 40 | 40 | 0.971 | yes |
| 15 fps instead of 30 | 0 s | 39 / 40 | 39 | 0.956 | yes |
| Trimmed 12.4 s to 27.4 s | 12 s | 15 / 15 | 15 | 0.902 | yes |
| Trimmed 5 s to 20 s, VP9 480x270 | 5 s | 15 / 15 | 15 | 0.977 | yes |
| The first 15 s, then 25 s of other content (VP9) | 0 s | 15 / 15 | 15 | 0.981 | yes |
| 25 s of other content, then the first 15 s | -25 s | 15 / 15 | 15 | 0.975 | yes |
| Unrelated video | none | 0 matched | 0 | 0.539 | no |
| Unrelated videos sharing a 3 s still end card | the card | 3 / 3 | 1 | 0.995 | no |
| Same with an 8 s end card | the card | 8 / 8 | 1 | 0.993 | no |

False positives and negatives of the default rule, measured on 2026-11-05
with nine ffmpeg test sources (`life` with two seeds, `mandelbrot`,
`testsrc2`, `cellauto`, `gradients`, `sierpinski`, `zoneplate`, noisy
`smptehdbars`), MPEG-4 and VP9 at 250 kb/s:

| Case | Result |
|---|---|
| Unrelated 30 s videos, every ordered pair (72) | no match; not a single matched pair |
| Unrelated videos sharing a still end card of 3, 5, 10 or 30 s | no match (1 distinct frame) |
| Unrelated videos sharing an animated intro of 3 or 4 s | no match (shorter than 5 s) |
| Unrelated videos sharing an animated intro of 5 or 8 s | **match**, the segment is the intro |
| Excerpt of a 60 s video between 10 s of other content on each side, re-encoded, cut at a random 0.1 s position (8 trials each): 3 s / 4 s | not found (0/8, by design) |
| Same, 5 s | found 7/8 (one cut lost a sample to the sampling phase) |
| Same, 6, 8, 10 and 15 s | found 8/8, at the right offset |

* **Re-encoding and resizing**: robust (frames are hashed at a fixed size).
* **Frame-rate changes**: robust by construction, because sampling is by time.
* **Trimming and excerpts**: found at the right offset whatever surrounds the
  excerpt in either video, from 5 to 6 seconds of shared footage. A cut
  between two sampling instants is sampled up to `0.5 / fps` s away from the
  reference's grid; on fast motion that raises the distances (the 12.4 s
  row), which the default threshold allows for. Use `fps=2` for very fast
  content, or lower `min_matched_seconds` for shorter clips.
* **Shared boilerplate**: still cards (end screens, title cards, frozen
  frames) never make a match on their own, whatever their length. Moving
  boilerplate of 5 s or more (an animated logo sting, a channel intro) is
  identical footage and is reported as a match; the segment's times show
  where it is, so applications that see shared intros can require a longer
  `matched_seconds`, a minimum `query_coverage`, or ignore segments at the
  very start or end of both videos.
* **Slow content**: a segment needs 3 distinct pictures, so a near-static
  excerpt (a slideshow with fewer than 3 slides, a still interview shot that
  changes less than `static_distance` bits) is not a match on its own.
* **Not handled**: speed changes (a clip played at another speed matches only
  briefly), mirroring, rotation by an uploader, cropping, letterboxing,
  picture-in-picture and heavy overlays.

## VideoSeal watermark

```python
from provenance_kit import PayloadCodec
from provenance_kit.video.watermark import (
    RepetitionCode,
    VideoSealBackend,
    detect_watermark,
    embed_verified,
)

backend = VideoSealBackend(model_dir="/srv/models/videoseal")  # strength 0.5
code = RepetitionCode(copies=4)  # 64 data bits
codec = PayloadCodec(code.data_bits, version_bits=4, check_bits=12)
result = embed_verified(
    frames, codec.encode(42), encoder=my_encoder, backend=backend, repetition=code, codec=codec
)
result.verified, result.bit_errors, result.encoded  # bytes from my_encoder
```

**Model.** The official VideoSeal v1.0 TorchScript model, 256 bits
(ARCHITECTURE.md, decision 4); only PyTorch is needed.

| File | URL | Size | SHA-256 |
|---|---|---|---|
| `y_256b_img.jit` | <https://dl.fbaipublicfiles.com/videoseal/y_256b_img.jit> | 228,524,815 bytes | `5c7a4581c36fc6090aafdcfb3999123bae5172a4847f22e2da4e7fd1a39d1e1b` |

The URL is the one documented in the repository's
[docs/torchscript.md](https://github.com/facebookresearch/videoseal/blob/main/docs/torchscript.md);
upstream publishes no checksum, so the SHA-256 was computed on 2026-11-05
from the file served there (`Last-Modified: Wed, 02 Apr 2025`). The backend
verifies it before loading and raises `ModelLoadError` on a mismatch.
`model_dir` defaults to `$PROVENANCE_KIT_MODEL_DIR/videoseal` or, without
the variable, `provenance-kit/models/videoseal` in the per-user cache
directory (`provenance_kit.modelfiles.default_model_dir`), and is refused
inside `site-packages`; nothing is downloaded unless `download=True` is
passed or `download_videoseal_model()` is called (which keeps a valid file
and replaces a missing or invalid one). Pre-provisioned directories work
offline.

**Decoded media only** (decision 2): frames in, frames out; the application
encodes. `embed_verified` streams the watermarked frames into the caller's
encoder callable, decodes its output with `iter_frames` and detects, so the
check covers the real codec.

**How the model works.** Every 4th frame (`step_size`) goes through the
embedder at 256 x 256; its watermark is upscaled, attenuated by a perceptual
mask and repeated on the next frames. Detection scores 256 channel bits per
frame; `detect_watermark` averages the scores over all frames given. Channel 0
of the model's output (a detection mask) is unused by upstream for this model,
so detection relies on the payload statistics below.

**Repetition code** (decision 8). `RepetitionCode(copies=n)` places `n` copies
of a `256 // n`-bit payload in the channel, each XORed with a fixed
pseudo-random whitening mask, and decodes by soft majority vote (the scores of
the copies are summed). Because of the whitening, the copies of a bit agree on
unmarked content no more often than coin flips, which gives a calibrated
detection statistic: `p_value` is the probability that unmarked content's
copies agree among themselves at least as much (each data bit's hard
majority, whose distribution under the null hypothesis is exact).
`agreement` is the fraction of copy bits whose sign agrees with the decoded
(soft-vote) bit; a few confident copies can outvote many unsure ones, so it
can be lower than the majority agreement behind `p_value`. `found` means `p_value <= 1e-4` (configurable). On
seven unmarked synthetic sources with 2, 4 and 8 copies, the smallest p-value
measured was 0.055. Without a repetition code, a codec with check bits
decides; failing that, a weak heuristic on the score magnitude does.

**Batching and memory.** Frames are processed in batches of `batch_frames`
(16, a multiple of the step size); peak memory is roughly
`36 * width * height * batch_frames` bytes (about 1.2 GB for 16 frames of
1080p on CPU). Both `embed_frames` and `score_frames` stream.

**Strength.** `strength` is the model's blending factor (`scaling_w`). The
default is **0.5** (`VIDEOSEAL_DEFAULT_STRENGTH`); the model's built-in value
is 0.2 (`strength=None` keeps it). The default was chosen by measurement, on
the rule "0.5 if it keeps the PSNR at or above 40 dB on the textured test
content": it does (41.5 dB), and it is the setting that survives the codecs
below.

**Measured robustness** (measured on 2026-11-05: 320x240, 24 frames at 24
fps, 64-bit payload with 4 copies, LGPL ffmpeg 8.1 encoders `libopenh264` and
`libvpx-vp9`; `tests/video/test_watermark_integration.py`, run with `-s`).
PSNR is between the watermarked and the original frames, before encoding.
Each other cell: channel bit accuracy (256 bits), payload bit errors out of
64, outcome (`verified` = payload exact and check bits valid; `found` =
detected with `p_value <= 1e-4`).

| Content | Encoding | Strength 0.2 (model built-in) | Strength 0.5 (default) |
|---|---|---|---|
| Mandelbrot (textured) | PSNR | 49.2 dB | **41.5 dB** |
| Mandelbrot | none | 1.000, 0, verified | 1.000, 0, verified |
| Mandelbrot | H.264 1 Mb/s | 0.781, 4, found | **0.941, 0, verified** |
| Mandelbrot | H.264 400 kb/s | 0.609, 20, not found | 0.777, 4, found |
| Mandelbrot | VP9 CRF 35 | 0.727, 9, found | **0.902, 0, verified** |
| testsrc2 (flat areas) | PSNR | 52.5 dB | 44.7 dB |
| testsrc2 | none | 0.906, 0, verified | 0.992, 0, verified |
| testsrc2 | H.264 1 Mb/s | 0.613, 14, not found | 0.801, 5, found |
| testsrc2 | H.264 400 kb/s | 0.516, 32, not found | 0.566, 22, not found |
| testsrc2 | VP9 CRF 35 | 0.543, 23, not found | 0.730, 7, found |

Takeaways: robustness depends strongly on the content (texture helps, flat
synthetic areas hurt), on the strength and on the bit rate per pixel; small
frames such as 320x240 are a hard case because the model works at 256 x 256.
`openh264` is a weaker encoder than x264 at the same bit rate. Use
`embed_verified` with the real publishing pipeline, a repetition code and
check bits; lower the strength (0.2) only where the PSNR budget matters more
than robustness, and raise it when verification fails.

**Thread safety.** `VideoSealBackend` can be shared between threads (model
calls are serialized); strength and step size are fixed at construction.
