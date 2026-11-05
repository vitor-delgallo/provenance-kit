# Security policy

## Reporting a vulnerability

Please report vulnerabilities **privately**, not in public issues:

* use GitHub's private vulnerability reporting ("Report a vulnerability" under
  the repository's **Security** tab), or
* if that is unavailable, open an issue that asks for a private contact
  without describing the problem.

Include the affected version or commit, a description, and a minimal
reproduction. A reproduction file is welcome, but do not attach real personal
media. We aim to acknowledge reports within 5 working days and to agree on a
disclosure date with you; fixes are released as soon as they are ready and
credited unless you prefer otherwise.

Only the latest release (and `main`) receives security fixes while the project
is pre-1.0.

Vulnerabilities in a dependency (TrustMark, VideoSeal, AudioSeal, c2pa-python,
opentimestamps, PyTorch, FFmpeg, Chromaprint, ...) should be reported to that
project; tell us too if provenance-kit should mitigate it.

## Threat model (summary)

provenance-kit is designed to run inside a service that processes media
uploaded by people it does not trust. The design assumes:

**Attacker-controlled inputs.** Any media bytes, embedded metadata, C2PA
manifests, and OpenTimestamps proofs or calendar responses may be crafted.
The library defends with:

* resource limits (`MediaLimits`: bytes, pixels, dimensions, duration,
  frames, channels, sample rate, subprocess time) checked before full
  decoding, to resist decompression bombs and resource exhaustion;
* media type detection from content, never from names or extensions;
* external tools (ffmpeg, ffprobe, fpcalc) run without a shell, from absolute
  paths, with argument lists, input through pipes, restricted ffmpeg
  protocols, timeouts that kill the whole process tree, and capped output;
* every failure caused by input surfacing as a documented
  `ProvenanceKitError`, never as an arbitrary exception;
* messages and logs (INFO and above) that never contain media, payloads,
  keys or caller file paths;
* temporary files only when unavoidable, private, in a caller-chosen
  directory, and always deleted.

**Trusted inputs.** The application's configuration (tool paths, limits,
calendar URLs, signer implementation, model files it provisions) and the
Python environment are trusted. Model weights are loaded only from pinned
sources and verified by hash.

**Secrets.** C2PA signing goes through a pluggable signer so that private keys
can stay in a KMS or HSM; the library never logs or returns key material.

**Out of scope.** Watermarks and perceptual hashes are probabilistic signals:
a determined adversary can remove or forge invisible watermarks, and
perceptual hashes have false positives and negatives. Applications must treat
them as evidence, not proof. Denial of service through legitimately large
inputs is bounded by the limits the application chooses. Security of the
third-party decoders themselves (FFmpeg, Pillow, PyTorch) is their projects'
responsibility; provenance-kit reduces exposure (limits, pipes, protocol
restrictions, timeouts) but does not sandbox them. For hostile workloads, run
the worker with operating-system isolation (containers, seccomp, no network
except where needed).
