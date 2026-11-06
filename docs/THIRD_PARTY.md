# Third-party components and licenses

provenance-kit itself is MIT-licensed, except one ported file under
BSD-3-Clause (the PDQ NumPy port, below), so the package declares
`License-Expression: MIT AND BSD-3-Clause` and ships both texts (`LICENSE`
and `LICENSES/ThreatExchange-BSD-3-Clause.txt`) in the wheel and the source
distribution. Its optional backends depend on third-party code, model weights
and binaries with their own licenses. This
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
| PDQ (`pdqhash` 0.2.8) | `pdq-native` | MIT bindings + BSD-3-Clause PDQ | none | Low; no Linux wheels |
| imagehash (`ImageHash` 4.3.2) | `image-hash` | BSD-2-Clause | none | Low |
| PDQ NumPy port (in `provenance_kit.image`) | core install | BSD-3-Clause (Meta ThreatExchange; notice in the file, text in `LICENSES/`) | none | Low |
| PDQ test vectors (ThreatExchange) | tests only | BSD-3-Clause; images downloaded at test time, not redistributed | none | None |
| Aileron Regular (Pillow's default font) | core (`image.visible`) | CC0 (embedded in Pillow; not bundled by provenance-kit) | none | Low |
| HEIC/HEIF decoders (`pillow-heif`, `pi-heif`) | **not a dependency** | `pillow-heif` wheels **GPL-2.0**; `pi-heif` LGPL-3.0, discontinued | none | **High if bundled; see below** |
| c2pa-python (`c2pa-python` 0.38.0 to 0.38.2, per platform) | `c2pa` | MIT OR Apache-2.0 | none | Low |
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
* provenance-kit does not use that loader: `image.watermark` pins the size
  and SHA-256 of the config, encoder and decoder of variants Q, P, B and C
  (`MODEL_FILES`, taken on 2026-11-05; every file's MD5 matched the package's
  own `MODEL_CHECKSUMS`), loads them from a caller directory outside
  `site-packages`, and downloads only through `download_trustmark_models`.
  The watermark-remover and box-detector checkpoints are never fetched.

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
* Pinned file (verified 2026-11-05): `y_256b_img.jit`, 228,524,815 bytes,
  SHA-256 `5c7a4581c36fc6090aafdcfb3999123bae5172a4847f22e2da4e7fd1a39d1e1b`
  (computed from the file at the official URL, `Last-Modified` 2025-04-02;
  upstream publishes no checksum). Loaded with `torch.jit.load` after the hash
  check; never downloaded unless the caller asks.

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
* Checked 2026-11-05 (video/audio module work): the GitHub repository shipped
  `LICENSE_weights` (CC-BY-NC 4.0) until commit `2956515` of 2024-04-02
  ("Re-update of license") removed it; a first relicensing on 2024-03-09 had
  been reverted on 2024-03-11. The 16-bit base checkpoints used by
  provenance-kit (`generator_base.pth`, SHA-256
  `7a845b5fbe9364a63a3909d8ab3fe064d13a76ae4c2e983573e08c69b7b51748`;
  `detector_base.pth`, SHA-256
  `8a78e8a83584113523e161fc599fcab10fd0e94c04d2eb9d2fa1e9ec91ab69d9`) were
  uploaded to Hugging Face on 2024-03-11 and are byte-identical at every
  revision since, and also to the files first published on 2024-02-04 at
  `dl.fbaipublicfiles.com/audioseal/6edcf62f/`. There is no separate
  post-relicensing set of weights: the current files are the original ones,
  covered by MIT through the licensor's 2024-04-02 statement. provenance-kit
  pins them by SHA-256 and downloads them only from Hugging Face revision
  `3c19eba53390776cf2cc9ed5f6c9ac67ce72ecba` (model card `license: mit`).
  The checkpoints are pickles; they are read with an allowlist unpickler
  after the hash check (see `docs/audio.md`).

### PDQ: `pdqhash`

* Package: [`pdqhash` on PyPI](https://pypi.org/project/pdqhash/) 0.2.8
  ([faustomorales/pdqhash-python](https://github.com/faustomorales/pdqhash-python)).
* Code: the Python bindings are MIT
  ([LICENSE](https://github.com/faustomorales/pdqhash-python/blob/master/LICENSE));
  the vendored PDQ C++ implementation is Meta's ThreatExchange code under
  ThreatExchange's BSD-3-Clause license, titled "BSD License" / "For
  ThreatExchange software"
  ([LICENSE](https://github.com/facebook/ThreatExchange/blob/main/LICENSE)). No
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
  ported code keeps Meta's BSD-3-Clause notice. The port is the default
  backend and needs nothing beyond the core install; `pdqhash` is its own
  optional extra, `pdq-native`, so that no other extra needs a C++ compiler.
  `auto` uses `pdqhash` when it is installed and the port otherwise
  (ARCHITECTURE.md, decision 3).

### imagehash

* Package: [`ImageHash` on PyPI](https://pypi.org/project/ImageHash/) 4.3.2.
* Code: BSD-2-Clause
  ([LICENSE](https://github.com/JohannesBuchner/imagehash/blob/master/LICENSE)).
  Dependencies: `PyWavelets` (MIT AND BSD-3-Clause), `scipy` (BSD-3-Clause),
  NumPy, Pillow.
* Weights: none.

### PDQ NumPy port and test vectors

* `src/provenance_kit/image/_pdq_numpy.py` is a port of
  `pdq/python/pdqhashing/hasher/pdq_hasher.py` and `utils/matrix.py` from
  [facebook/ThreatExchange](https://github.com/facebook/ThreatExchange) at commit
  `a9a36d4552875c385c4b40f2be847db2b613ed44`, under ThreatExchange's
  BSD-3-Clause license, titled "BSD License" / "For ThreatExchange software"
  ([LICENSE](https://github.com/facebook/ThreatExchange/blob/main/LICENSE)).
  The full notice is kept in the module docstring, as the license requires
  for source redistribution. For binary redistribution, the license text is
  also shipped verbatim as `LICENSES/ThreatExchange-BSD-3-Clause.txt`
  (copied on 2026-11-05 from
  <https://raw.githubusercontent.com/facebook/ThreatExchange/main/LICENSE>,
  SHA-256 `68ecc6aafbd2a205a1077f86127030898f03091b7dae9d9017325a8702d8668f`),
  listed in `license-files`, so the wheel carries it under
  `.dist-info/licenses/` next to the MIT `LICENSE`, and the package's
  license expression is `MIT AND BSD-3-Clause`. Torben's median routine in
  that code is marked public domain.
* `tests/image/pdq_vectors.py` holds hash values only: hashes computed with
  the unmodified reference on images the tests generate, and the expected
  hashes published in ThreatExchange's `pdq_test.py`. The ThreatExchange
  images themselves (photos and a LabelMe subset whose own terms are not
  stated in the repository) are **not committed**; the integration test
  downloads them at the pinned commit and checks their SHA-256.

### Image formats in Pillow (AVIF, HEIC)

* AVIF decoding uses Pillow's own wheels: libavif 1.4.2 (BSD-2-Clause) with
  the dav1d decoder (BSD-2-Clause), listed in Pillow's bundled `LICENSE`.
  Pillow is MIT-CMU; its wheels do not include the GPL `libimagequant`
  (checked with `PIL.features`).
* **HEIC/HEIF has no permissively licensed decoder today**, so
  provenance-kit has no HEIF extra; `image.loading` accepts a decoder the
  application chooses (`openers=`). Checked on 2026-11-05:
  * [`pillow-heif`](https://pypi.org/project/pillow-heif/) 1.8.1: Python code
    BSD-3-Clause, but `LICENSES_bundled.txt` says "License for pillow-heif
    binary wheels: GPLv2, due to base library licenses": the wheels bundle
    libheif 1.23.7 and libde265 1.1.3 (LGPL-3.0) and **x265 4.3 (GPL-2.0)**,
    plus MinGW runtimes on Windows. Shipping it with a proprietary product
    means GPL obligations; review with counsel.
  * [`pi-heif`](https://pypi.org/project/pi-heif/) 1.4.0 (decode-only, LGPL
    wheels) is **discontinued**. According to the
    [pillow-heif README](https://github.com/bigcat88/pillow_heif#readme), its
    wheels bundle libheif 1.23.0, which is affected by a heap buffer overflow
    on crafted HEIC files. Do not use it for untrusted uploads.
  * HEVC (the codec inside HEIC) is patent-encumbered in some jurisdictions,
    independently of the copyright license.

### Fonts

* `image.visible` bundles no font. Without `font=`, it uses Pillow's
  `ImageFont.load_default(size)`, a subset of **Aileron Regular** embedded in
  Pillow (Basic Latin only), released by its author under CC0 1.0
  ([dotcolon.net/fonts/aileron](https://dotcolon.net/fonts/aileron)). Fonts
  passed by path carry their own licenses.

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
* Notes: the `c2pa` extra allows `>=0.38,<0.39`, and which release a
  resolver installs depends on the platform, because the later releases have
  fewer files (PyPI, checked 2026-11-05):

  | Release | Uploaded | Files |
  |---|---|---|
  | 0.38.0 | 2026-09-29 | wheels for Linux (manylinux_2_28 x86_64, aarch64), macOS and Windows (x64, arm64), and an sdist |
  | 0.38.1 | 2026-10-22 | wheels for Linux and macOS only; no Windows wheel, no sdist |
  | 0.38.2 | 2026-10-26 | macOS wheels only (x86_64, universal2); no sdist |

  So macOS resolves 0.38.2, Linux 0.38.1 and Windows 0.38.0. The module was
  developed and measured against 0.38.0 (Windows); the Linux CI integration
  job installs 0.38.1 and runs the C2PA integration tests against it (a
  missing or unloadable c2pa-python fails that job instead of skipping).
  Signing certificates and trust lists come from your certificate authority
  and carry their own terms.
* `cryptography` (installed 50.0.2) is also a direct dependency of the `c2pa`
  extra, for `LocalKeySigner`: Apache-2.0 OR BSD-3-Clause, no change in risk.
* Data the module refers to but does not ship: the C2PA soft binding
  algorithm list (algorithm identifiers such as `com.adobe.trustmark.Q` are
  names, used as data) and the C2PA trust list (load it yourself under its
  terms).

### OpenTimestamps

* Package: [`opentimestamps` on PyPI](https://pypi.org/project/opentimestamps/) 0.5.0
  ([python-opentimestamps](https://github.com/opentimestamps/python-opentimestamps)).
  Requires Python 3.11 or later.
* Code: **LGPL-3.0-or-later**
  ([LICENSE](https://github.com/opentimestamps/python-opentimestamps/blob/master/LICENSE),
  PyPI `License-Expression`). Its dependency `python-bitcoinlib` 0.12.2 is also
  LGPL-3.0-or-later (PyPI classifier). `pycryptodomex` (3.24.1 installed;
  3.24.2 released on 2026-11-05) is public domain (the code inherited from
  PyCrypto) and BSD-2-Clause, per the `LICENSE.rst` in its wheels. The
  repository's current
  [LICENSE.rst](https://github.com/Legrandin/pycryptodome/blob/master/LICENSE.rst)
  also has a "Third-party code" section: `src/mldsa-native` and
  `src/mlkem-native` (from the pq-code-package projects) are under
  "Apache-2.0 OR ISC OR MIT". That section is not in the 3.24.2 wheel yet;
  either way the terms are permissive.
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
* provenance-kit imports python-opentimestamps only for the proof format and
  its parser; the calendar HTTP client is its own (standard library), and
  no LGPL code is copied into this repository. The example proofs of the
  official client (an LGPL-3.0 repository without a separate data license)
  are not used as fixtures: the committed `tests/provenance/fixtures/*.ots`
  are generated by this project's own script. The test suite uses
  `python-bitcoinlib` (installed with python-opentimestamps) to build
  synthetic blocks.

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
  GPL-2.0-or-later when built with `--enable-gpl` (required, for example, to
  link libx264 or libx265), version 3 of either license with
  `--enable-version3` (also required to combine it with the Apache-2.0 or
  LGPL-3.0 libraries the file lists), and "unredistributable" with
  `--enable-nonfree`, as FFmpeg's own
  [LICENSE.md](https://github.com/FFmpeg/FFmpeg/blob/master/LICENSE.md)
  states (checked 2026-11-05).
* The development binaries recorded in [DEVELOPMENT.md](DEVELOPMENT.md) are
  BtbN **LGPL** builds configured with `--enable-version3`, so they are
  LGPL-3.0-or-later (the bundled `LICENSE.txt` is the LGPL v3 text). Most Linux
  distribution packages and "full" Windows builds are **GPL** builds (Ubuntu
  24.04's `ffmpeg` 6.1.1, for one, is configured with `--enable-gpl`).
* Versions in use: FFmpeg **n8.1.3-34-ga35157abb9** (2026-11-05) for the
  Windows development binaries, and **n8.1.3-27-g8dca1586f4** (2026-10-31,
  BtbN `autobuild-2026-10-31-12-59`, `linux64-lgpl-8.1`, pinned by SHA-256)
  in the Linux CI integration job. CI downloads and runs that build and
  distributes nothing.
* Weights: none.
* provenance-kit only **executes** FFmpeg as a separate process; that does not
  make the library or your application a derivative work. Shipping FFmpeg
  binaries (for example in a container image you distribute) means complying
  with the license of that particular build; prefer an LGPL build.
* **Watch (patents):** decoding or encoding patented formats (H.264, HEVC, AAC
  and others) can require patent licenses in some jurisdictions,
  independently of the copyright license. The patent FAQ of the
  [FFmpeg legal page](https://ffmpeg.org/legal.html) says the project cannot
  tell which algorithms are patented and that the answer depends on the
  jurisdiction.

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
