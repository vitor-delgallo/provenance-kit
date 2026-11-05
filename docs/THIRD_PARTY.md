# Third-party components and licenses

provenance-kit itself is MIT-licensed. Its optional backends depend on
third-party code, model weights and binaries with their own licenses. This
page records, for each one, the license of the **code** and of the **model
weights**, where that was verified, and what to watch for in a commercial
application.

Verified on 2026-11-05 against the official repositories, PyPI metadata and
model hosts linked below. This is an engineering summary, not legal advice:
have counsel review anything marked as a risk before you distribute a product.

## Summary

| Component | Used by (extra) | Code license | Weights license | Commercial risk |
|---|---|---|---|---|
| TrustMark (`trustmark` 0.9.2) | `image-watermark` | MIT | MIT | Low |
| VideoSeal (TorchScript model) | `video-watermark` | MIT | MIT | Low |
| AudioSeal (`audioseal` 0.2.1) | `audio-watermark` | MIT | MIT (since 2024-04-02) | Low; use current weights only |
| PDQ (`pdqhash` 0.2.8) | `image-hash` | MIT bindings + BSD-3-Clause PDQ | none | Low; no Linux wheels |
| imagehash (`ImageHash` 4.3.2) | `image-hash` | BSD-2-Clause | none | Low |
| c2pa-python (`c2pa-python` 0.38) | `c2pa` | MIT OR Apache-2.0 | none | Low |
| OpenTimestamps (`opentimestamps` 0.5.0) | `timestamps` | **LGPL-3.0-or-later** | none | **Medium: weak copyleft** |
| Chromaprint `fpcalc` 1.6.2 (binary) | `audio` | MIT source, **LGPL-2.1 as a whole** | none | **Medium if you ship the binary** |
| FFmpeg (binaries) | `video`, `audio` | **LGPL-2.1+ / LGPL-3.0+ or GPL, per build** | none | **Medium if you ship the binary; codec patents** |
| PyTorch (`torch`, `torchvision`) | all watermark extras | BSD-3-Clause (torch bundles permissive third-party code) | none | Low |

Nothing in the list is non-commercial today. The two copyleft items are weak
copyleft (LGPL) and only matter when you **distribute** the component (for
example inside a container image or an installer handed to customers); running
them on your own servers does not trigger the distribution conditions.

## Details

### TrustMark (Adobe)

* Package: [`trustmark` on PyPI](https://pypi.org/project/trustmark/) 0.9.2
  (sdist only; classifier "MIT License").
* Code: MIT. [LICENSE](https://github.com/adobe/trustmark/blob/main/LICENSE)
  is the MIT text preceded by an Adobe copyright notice.
* Weights: MIT. The README's License section states: "The MIT license covers
  both the code in this repo and the model files which are downloaded upon
  first use" ([README](https://github.com/adobe/trustmark#license)).
* Notes: the package downloads its checkpoints on first use from
  `https://cai-watermark.adobe.net/watermarking/trustmark-models/` into its own
  package directory and checks them with MD5. Pre-provision and pin the files
  in production. Transitive dependencies are permissive: `torchvision`
  (BSD-3-Clause), `lightning` (Apache-2.0), `omegaconf` (BSD-3-Clause),
  `einops` (MIT).

### VideoSeal (Meta)

* Source: [facebookresearch/videoseal](https://github.com/facebookresearch/videoseal).
  Also on PyPI as [`videoseal`](https://pypi.org/project/videoseal/) 1.0.1
  (July 2025, author "Facebook AI Research"), which lags the repository (no
  PixelSeal or ChunkySeal) and pulls about 40 heavy packages (transformers,
  tensorboard, opencv-python, pandas, decord, pycocotools, lpips, a pinned
  `timm==0.9.16`). The repository has no tags or releases.
* provenance-kit's `video-watermark` extra therefore depends on `torch` only
  and loads the official **TorchScript** model
  (`https://dl.fbaipublicfiles.com/videoseal/y_256b_img.jit`, 256-bit
  VideoSeal v1.0, documented in
  [docs/torchscript.md](https://github.com/facebookresearch/videoseal/blob/main/docs/torchscript.md)).
* Code: MIT ([LICENSE](https://github.com/facebookresearch/videoseal/blob/main/LICENSE)).
* Weights: MIT. The README states "All models, training code, and evaluation
  tools released under MIT license" and "The model is licensed under an MIT
  license" ([README](https://github.com/facebookresearch/videoseal#license)).
  This covers VideoSeal, PixelSeal and ChunkySeal; the third-party baseline
  models the repository can evaluate against are not covered and are not used.

### AudioSeal (Meta)

* Package: [`audioseal` on PyPI](https://pypi.org/project/audioseal/) 0.2.1
  (classifier "MIT License").
* Code: MIT ([LICENSE](https://github.com/facebookresearch/audioseal/blob/main/LICENSE)).
  The README credits code adapted from AudioCraft and Moshi.
* Weights: MIT. The README's update of 2024-04-02 says the license was changed
  "to full MIT license (including the license for the model weights)! Now you
  can use AudioSeal in commercial application too!"
  ([README](https://github.com/facebookresearch/audioseal#sparkles-key-updates)),
  and the [Hugging Face model card](https://huggingface.co/facebook/audioseal)
  declares `license: mit`.
* **Watch:** weights published before April 2024 were under a non-commercial
  license. Load only the current checkpoints from `facebook/audioseal`, pinned
  by hash.

### PDQ: `pdqhash`

* Package: [`pdqhash` on PyPI](https://pypi.org/project/pdqhash/) 0.2.8
  ([faustomorales/pdqhash-python](https://github.com/faustomorales/pdqhash-python)).
* Code: the Python bindings are MIT
  ([LICENSE](https://github.com/faustomorales/pdqhash-python/blob/master/LICENSE));
  the vendored PDQ C++ implementation is Meta's ThreatExchange code under the
  BSD-3-Clause "BSD License for ThreatExchange software"
  ([LICENSE](https://github.com/meta/ThreatExchange/blob/main/LICENSE)). No
  patent clause or PATENTS file.
* Weights: none (PDQ is a deterministic algorithm).
* **Packaging gap:** wheels exist for Windows and macOS (CPython 3.10 to 3.13)
  but **not for Linux**; the maintainer's README notes the Linux wheels are
  deleted because the manylinux build is broken. On Linux, pip builds it from
  the source distribution, which needs a C++11 compiler (Cython and NumPy are
  fetched as build requirements). There are no CPython 3.14 wheels. The last
  release is from May 2025. No maintained alternative with Linux wheels was
  found; note that the PyPI package named `pdq` is an unrelated Parquet tool.
  provenance-kit therefore also provides a NumPy port of Meta's BSD-licensed
  pure-Python reference implementation
  ([ThreatExchange/pdq/python](https://github.com/meta/ThreatExchange/tree/main/pdq/python)),
  validated against the reference test vectors in that repository; the
  ported code keeps Meta's BSD-3-Clause notice. `auto` uses `pdqhash` when it
  is installed and the port otherwise (ARCHITECTURE.md, decision 3).

### imagehash

* Package: [`ImageHash` on PyPI](https://pypi.org/project/ImageHash/) 4.3.2.
* Code: BSD-2-Clause
  ([LICENSE](https://github.com/JohannesBuchner/imagehash/blob/master/LICENSE)).
  Dependencies: `PyWavelets` (MIT AND BSD-3-Clause), `scipy` (BSD-3-Clause),
  NumPy, Pillow.
* Weights: none.

### c2pa-python

* Package: [`c2pa-python` on PyPI](https://pypi.org/project/c2pa-python/) 0.38
  ([contentauth/c2pa-python](https://github.com/contentauth/c2pa-python)).
* Code: dual MIT OR Apache-2.0 (PyPI license field; the repository ships
  [LICENSE-MIT](https://github.com/contentauth/c2pa-python/blob/main/LICENSE-MIT)
  and [LICENSE-APACHE](https://github.com/contentauth/c2pa-python/blob/main/LICENSE-APACHE)).
  The wheels bundle the native library built from
  [c2pa-rs](https://github.com/contentauth/c2pa-rs), also MIT OR Apache-2.0;
  its README notes that "some components and dependent crates are licensed
  under different terms". Python dependencies: `cryptography`
  (Apache-2.0 OR BSD-3-Clause), `requests` (Apache-2.0), `toml` (MIT).
* Weights: none.
* Notes: 0.38.1 and 0.38.2 have no Windows wheels; the resolver picks 0.38.0
  there. Signing certificates and trust lists come from your certificate
  authority and carry their own terms.

### OpenTimestamps

* Package: [`opentimestamps` on PyPI](https://pypi.org/project/opentimestamps/) 0.5.0
  ([python-opentimestamps](https://github.com/opentimestamps/python-opentimestamps)).
  Requires Python 3.11 or later.
* Code: **LGPL-3.0-or-later**
  ([LICENSE](https://github.com/opentimestamps/python-opentimestamps/blob/master/LICENSE),
  PyPI `License-Expression`). Its dependency `python-bitcoinlib` 0.12.2 is also
  LGPL-3.0-or-later (PyPI classifier); `pycryptodomex` is BSD-2-Clause and
  public domain.
* Weights: none.
* **Watch (weak copyleft):** importing an unmodified LGPL library from
  proprietary Python code is the use the LGPL is designed to allow. If you
  distribute it (for example in an image given to customers), include the
  license text, keep the library replaceable (a separate, unmodified package
  is) and make its source available; publish any modifications to it under the
  LGPL. LGPL-3.0 adds installation-information duties for consumer devices
  ("User Products"). Running it on your own servers is not distribution. If
  even that is unacceptable, the alternative is an in-house implementation of
  the (small, documented) proof format, which this project does not plan.

### Chromaprint `fpcalc`

* Binary: `fpcalc` 1.6.2 from the
  [Chromaprint releases](https://github.com/acoustid/chromaprint/releases/tag/v1.6.2).
* License: Chromaprint's own source is MIT, but "we include some parts of the
  FFmpeg library, which is licensed under the LGPL 2.1 license. As a whole,
  Chromaprint should be therefore considered to be licensed under the LGPL 2.1
  license" ([LICENSE.md](https://github.com/acoustid/chromaprint/blob/master/LICENSE.md)).
  The `fpcalc` binaries are statically linked with FFmpeg (`fpcalc -version`
  reports the bundled FFmpeg libraries), so treat the binary as LGPL-2.1 and
  check the FFT library it was built with if you redistribute it.
* Weights: none.
* provenance-kit only **executes** `fpcalc` as a separate process; it does not
  link to it. Shipping the binary means complying with the LGPL for it.
* **Watch:** the AcoustID lookup web service (not used by provenance-kit) has
  separate terms that restrict commercial use; local fingerprinting does not
  involve it.

### FFmpeg

* Binaries: `ffmpeg` and `ffprobe`. FFmpeg is LGPL-2.1-or-later by default,
  GPL-2.0-or-later when built with `--enable-gpl` (for example with x264 or
  x265), version 3 of either license with `--enable-version3`, and not
  redistributable with `--enable-nonfree`
  ([FFmpeg legal page](https://ffmpeg.org/legal.html)).
* The development binaries recorded in [DEVELOPMENT.md](DEVELOPMENT.md) are
  BtbN **LGPL** builds configured with `--enable-version3`, so they are
  LGPL-3.0-or-later (the bundled `LICENSE.txt` is the LGPL v3 text). Most Linux
  distribution packages and "full" Windows builds are **GPL** builds.
* Weights: none.
* provenance-kit only **executes** FFmpeg as a separate process; that does not
  make the library or your application a derivative work. Shipping FFmpeg
  binaries (for example in a container image you distribute) means complying
  with the license of that particular build; prefer an LGPL build.
* **Watch (patents):** decoding or encoding patented formats (H.264, HEVC, AAC
  and others) can require patent licenses in some jurisdictions,
  independently of the copyright license. The FFmpeg legal page leaves that
  responsibility to the user.

### PyTorch

* Packages: `torch` 2.15.0 and `torchvision` 0.29.2 (CPU builds in the
  development environment). `torch` declares
  `Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND
  BSD-3-Clause AND BSL-1.0 AND ISC AND MIT AND Zlib` (BSD-3-Clause for PyTorch
  itself plus bundled permissive components); `torchvision` is BSD-3-Clause.
* CUDA builds additionally bundle NVIDIA libraries under the NVIDIA software
  license; review it if you distribute CUDA images.

### Core dependencies

* NumPy: BSD-3-Clause (with bundled 0BSD, MIT, Zlib and CC0-1.0 components).
* Pillow: MIT-CMU (HPND).
* Transitive packages in the full development environment are permissive
  (MIT, BSD, Apache-2.0, PSF-2.0), except `certifi` and `tqdm`, which include
  MPL-2.0 (file-level copyleft: only modifications to those files must be
  shared).
