# Development

How to set up a development environment, run the checks and the tests. The
rules for writing code are in [ARCHITECTURE.md](ARCHITECTURE.md).

## Requirements

* Python 3.11 or later (the development environment uses 3.11; CI also tests 3.13).
* [uv](https://docs.astral.sh/uv/) to create the environment (pip works too).
* For integration tests: the `ffmpeg`, `ffprobe` and `fpcalc` binaries.
* On Linux, only for the optional `pdq-native` extra: a C++11 compiler
  (`build-essential` or equivalent) to build `pdqhash` from source, which
  publishes wheels for Windows and macOS only. Without it, PDQ runs on the
  built-in NumPy port and the `pdqhash` tests skip.

## Set up the environment

**Where uv is.** On the reference Windows machine uv is not installed
system-wide: the official uv 0.13.7 executable lives in the repository's
git-ignored `tools/` directory, at `tools/uv/uv.exe` (with `uvx.exe` and
`uvw.exe`). Run it by that path, for example `.\tools\uv\uv.exe venv ...` in
PowerShell or `./tools/uv/uv.exe venv ...` in Git Bash. The commands below say
`uv`; substitute the path if uv is not on your `PATH`. Its source and checksum
are in the table under "External binaries". Elsewhere, install uv from the
[uv releases](https://github.com/astral-sh/uv/releases) (a single executable)
or with the [installation guide](https://docs.astral.sh/uv/getting-started/installation/).

```sh
uv venv --python 3.11 .venv

# CPU-only PyTorch first, from the PyTorch CPU index (much smaller than the
# default CUDA build). Skip this on a machine where you want CUDA.
uv pip install --python .venv --index-url https://download.pytorch.org/whl/cpu torch torchvision

# The project in editable mode with every extra and the dev tools.
# PYTHONUTF8=1 is needed on Windows only: the trustmark 0.9.2 source
# distribution reads its README with the locale encoding and fails to build
# under cp1252 otherwise. "all" leaves out pdq-native (pdqhash needs a C++
# compiler on Linux); add it where it installs.
PYTHONUTF8=1 uv pip install --python .venv -e ".[all,pdq-native,dev]"
```

**`PYTHONUTF8=1` (Windows).** `trustmark` 0.9.2 is published as a source
distribution only, and its `setup.py` reads its README with the locale
encoding. Under the Windows default (cp1252) the build fails with
`UnicodeDecodeError: 'charmap' codec can't decode byte 0x90`. Python's UTF-8
mode fixes it. In PowerShell, run `$env:PYTHONUTF8 = "1"` before the last
command; in Git Bash or on Linux, prefix the command as shown (harmless where
it is not needed). For the core only, install `-e ".[dev]"`.

## External binaries

The tests look for each binary in this order: the environment variable
(`PROVENANCE_KIT_FFMPEG`, `PROVENANCE_KIT_FFPROBE`, `PROVENANCE_KIT_FPCALC`),
then the repository's `tools/` directory, then `PATH`. The library itself uses
the same variables through `ExternalTools.from_env()`.

`tools/` is git-ignored. Layout:

```
tools/
  ffmpeg/ffmpeg(.exe)   ffmpeg/ffprobe(.exe)   ffmpeg/LICENSE.txt
  fpcalc/fpcalc(.exe)
  uv/uv(.exe)           (optional: a local copy of uv)
```

The Windows binaries in the reference environment were downloaded on
2026-11-05 from these sources:

| Binary | Version | Source | SHA-256 |
|---|---|---|---|
| ffmpeg, ffprobe | n8.1.3-34-ga35157abb9 (2026-11-05), **LGPL** build | [BtbN/FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds) release `autobuild-2026-11-05-12-48`, asset `ffmpeg-n8.1.3-34-ga35157abb9-win64-lgpl-8.1.zip` (<https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2026-11-05-12-48/ffmpeg-n8.1.3-34-ga35157abb9-win64-lgpl-8.1.zip>) | zip `aeea3020a16e8d37aeac67df9ce3cdbf93b79f5ea96eddd26f4a79ba8154cc86` (matches the release's `checksums.sha256`); `ffmpeg.exe` `cc68aa459348b23ac5ba554534cedc8d83e2b60b87505a31291101f220a840a1`; `ffprobe.exe` `244c6842c1517211c43edbca716767517520ebb51687101ac8ebfd117f438444` |
| fpcalc | 1.6.2 (Chromaprint, 2026-10-15) | [acoustid/chromaprint releases](https://github.com/acoustid/chromaprint/releases/tag/v1.6.2), asset `chromaprint-fpcalc-1.6.2-windows-x86_64.zip` (<https://github.com/acoustid/chromaprint/releases/download/v1.6.2/chromaprint-fpcalc-1.6.2-windows-x86_64.zip>) | zip `b74369726676670af3ab0eadff45f960cf969fdf4a87a971ba37eaf8d530e874`; `fpcalc.exe` `1fff0e00387ae96eb461f9913511e3adfd4f634cf80c67c761fc88f69027251e` |
| uv | 0.13.7 | <https://github.com/astral-sh/uv/releases/download/0.13.7/uv-x86_64-pc-windows-msvc.zip> | zip `0cdc5b38fe5a8168ed6a25814f5ba43c21d313dd3187314fccb37b7cec46e021` (matches the published `.sha256`) |

BtbN keeps dated `autobuild-*` releases for a limited time only; if the link
above has expired, take the newest `win64-lgpl` build of the same FFmpeg
release branch and record its checksum. The LGPL variant is preferred over the
GPL one so that the binaries used for development carry the least restrictive
license (see [THIRD_PARTY.md](THIRD_PARTY.md)).

On Linux, use the static builds from the same projects, as CI does: BtbN's
`ffmpeg-n8.1-latest-linux64-lgpl-8.1.tar.xz` and Chromaprint's
`chromaprint-fpcalc-1.6.2-linux-x86_64.tar.gz` (SHA-256
`c968ad4a0821592a79b158fb2e1f46cbe0a22a0a1dbc0743c67d68e75d15d2bb`), and
point the `PROVENANCE_KIT_*` variables at them. Distribution packages
(`ffmpeg`, `libchromaprint-tools`) also work but are often older releases,
and distribution FFmpeg packages are usually GPL builds
([THIRD_PARTY.md](THIRD_PARTY.md)).

## Model files and other downloaded test data

The integration tests of the watermark backends need the model weights. They
use the library's own convention (`provenance_kit.modelfiles`): one
subdirectory per backend under a model root. The root is
`$PROVENANCE_KIT_MODEL_DIR` when that variable is set, otherwise the
repository's git-ignored `.cache/models/` (`tests.support.model_dir`):

```
.cache/
  models/
    trustmark/   trustmark_Q.yaml, encoder_Q.ckpt, decoder_Q.ckpt (65 MB; P, B, C optional)
    videoseal/   y_256b_img.jit (228 MB)
    audioseal/   generator_base.pth, detector_base.pth (94 MB)
  threatexchange-pdq/<commit>/   ThreatExchange PDQ test images
```

Missing files make the tests skip with the reason; `pytest --run-network`
downloads them once from the pinned URLs and verifies their size and
SHA-256. To provision them by hand, call `download_trustmark_models`,
`download_videoseal_models` and `download_audioseal_models` with
`".cache/models/<backend>"` as `model_dir` (same signature for the three;
TrustMark's other variants with `variant="P"` and so on). Never commit
anything under `.cache/`.

## Checks

Run from the repository root with the environment's executables (on Windows
they are under `.venv\Scripts\`):

```sh
ruff check src tests     # lint
ruff format --check .    # formatting, including Python blocks in Markdown ("ruff format ." fixes)
mypy                     # strict type check of src/ (configured in pyproject.toml)
```

## Tests

```sh
pytest -m "not integration and not network"   # unit tests: what CI runs
pytest -m integration                          # needs extras / binaries; skips what is missing
pytest --run-network                           # also runs tests that need internet
pytest                                         # everything available offline
```

* Skipped tests print their reason with `-rs` (enabled by default through
  `-ra`): it names the extra to install or the binary to configure.
* Warnings are errors (`filterwarnings = error`). If a third-party library
  emits a warning you cannot avoid, filter that specific warning in the test
  with `@pytest.mark.filterwarnings("ignore:...:DeprecationWarning")`.
* CI (`.github/workflows/ci.yml`) has two jobs. `checks` runs ruff, mypy
  and the unit tests on Ubuntu and Windows with Python 3.11 and 3.13, in an
  environment with the core and the dev tools only (`pip install -e
  ".[dev]"`): mypy must pass without the optional packages too, which is why
  `pyproject.toml` ignores their missing imports (including the typed
  `cryptography`; when it is installed, mypy still checks against its types).
  `integration` runs `pytest -m integration` on Ubuntu with Python 3.11 and
  3.13, with ffmpeg/ffprobe (BtbN's `linux64-lgpl` build of the 8.1 branch,
  checked against the release's `checksums.sha256`), fpcalc 1.6.2 (the
  `linux-x86_64` release asset, pinned by SHA-256) and the extras that need
  no model weights (`image-hash,video,audio,c2pa,timestamps`). The
  watermark tests skip there with their reason (no PyTorch, no weights), as
  do the `network` tests; run them locally.

## Known quirks found while setting up

* `fpcalc` reading from standard input (`fpcalc -json -`) cannot know the
  duration and reports `"duration": 0.0`; the fingerprint is identical to the
  one computed from a file. Take the duration from ffprobe or from the PCM
  sample count.
* `trustmark` downloads its model files on first use into its own package
  directory (`site-packages/trustmark/models/`) from
  `https://cai-watermark.adobe.net/`, verified with MD5 only, and prints
  progress with `print()`. Read-only or offline deployments need the files
  provisioned in advance.
* `c2pa-python` 0.38.1 has no Windows wheel and 0.38.2 has macOS wheels
  only, neither with an sdist: resolvers install 0.38.0 on Windows, 0.38.1
  on Linux and 0.38.2 on macOS (see [THIRD_PARTY.md](THIRD_PARTY.md)).
* `opentimestamps` 0.5.0 (the release that hardens proof parsing) requires
  Python 3.11, which is also this project's minimum.
* `ffmpeg -nostdin` still reads media from `-i pipe:0`. ffprobe has no
  `-nostdin` option and refuses it; it does not read the terminal anyway.
