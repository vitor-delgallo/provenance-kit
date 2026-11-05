"""VideoSeal with the real model and real codecs (bit accuracy after encoding).

The model (228 MB) is read from ``tests.support.model_dir`` (the git-ignored
``.cache/models/videoseal/``, or ``$PROVENANCE_KIT_MODEL_DIR/videoseal``). If
it is missing the tests skip, unless ``--run-network`` is given, in which
case it is downloaded there from the official URL and verified against the
pinned SHA-256. Run with ``-s`` to print the measured numbers; they are recorded in
``docs/video.md``.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from provenance_kit import PayloadCodec
from provenance_kit.bits import bit_error_rate
from provenance_kit.errors import ModelLoadError
from provenance_kit.tools import ExternalTools, Tool
from provenance_kit.video.frames import iter_frames
from provenance_kit.video.watermark import (
    VIDEOSEAL_DEFAULT_STRENGTH,
    VIDEOSEAL_MODEL,
    RepetitionCode,
    VideoSealBackend,
    detect_watermark,
    download_videoseal_models,
    embed_verified,
    embed_watermark,
)
from tests.support import SEED, find_tools, model_dir, require_module, require_tool
from tests.video.media import H264, MANDELBROT, TESTSRC2, VP9, encode_frames, psnr, raw_frames

pytestmark = pytest.mark.integration

MODEL_DIR = model_dir("videoseal")
W, H, RATE, COUNT = 320, 240, 24, 24
CODE = RepetitionCode(copies=4)
CODEC = PayloadCodec(CODE.data_bits, version_bits=4, check_bits=12)
CODECS = {
    "h264-1000k": [*H264[:2], "-b:v", "1000k", *H264[2:]],
    "h264-400k": [*H264[:2], "-b:v", "400k", *H264[2:]],
    "vp9-crf35": [*VP9[:2], "-crf", "35", "-b:v", "0", *VP9[2:]],
}
SOURCES = {"mandelbrot": MANDELBROT, "testsrc2": TESTSRC2}


def ensure_model(request: pytest.FixtureRequest) -> None:
    require_module("torch", extra="video-watermark")
    if (MODEL_DIR / VIDEOSEAL_MODEL.name).is_file():
        return
    if not request.config.getoption("--run-network"):
        pytest.skip(
            f"the VideoSeal model is not in {MODEL_DIR}: run with --run-network to download it"
        )
    download_videoseal_models(MODEL_DIR)


@pytest.fixture(scope="module")
def tools() -> ExternalTools:
    require_tool(Tool.FFMPEG)
    require_tool(Tool.FFPROBE)
    return find_tools()


@pytest.fixture(scope="module")
def backends(request) -> dict[float, VideoSealBackend]:
    ensure_model(request)
    return {
        0.2: VideoSealBackend(model_dir=MODEL_DIR, strength=None),  # the model's built-in value
        0.5: VideoSealBackend(model_dir=MODEL_DIR),  # the default
    }


@pytest.fixture(scope="module")
def sources(tools) -> dict[str, list[np.ndarray]]:
    return {
        name: raw_frames(tools, source, width=W, height=H, rate=RATE, count=COUNT)
        for name, source in SOURCES.items()
    }


def payload() -> str:
    return CODEC.encode(SEED % CODEC.max_identifier, version=1)


@pytest.mark.network
def test_download_verifies_the_pinned_model():
    require_module("torch", extra="video-watermark")
    directory = download_videoseal_models(MODEL_DIR)
    assert (directory / VIDEOSEAL_MODEL.name).stat().st_size == VIDEOSEAL_MODEL.size


def test_the_default_strength_and_the_upstream_one(backends):
    assert backends[0.5].strength == pytest.approx(VIDEOSEAL_DEFAULT_STRENGTH) == 0.5
    assert backends[0.2].strength == pytest.approx(0.2)  # strength=None: the model's own
    assert backends[0.2].step_size == backends[0.5].step_size == 4


def test_tampered_models_are_refused(tmp_path, request):
    ensure_model(request)
    fake = tmp_path / VIDEOSEAL_MODEL.name
    fake.write_bytes(b"not the model")
    with pytest.raises(ModelLoadError, match="SHA-256"):
        VideoSealBackend(model_dir=tmp_path)


@pytest.mark.parametrize("strength", [0.2, 0.5])
@pytest.mark.parametrize("source", sorted(SOURCES))
def test_raw_round_trip(backends, sources, source, strength):
    backend = backends[strength]
    frames = sources[source]
    marked = list(embed_watermark(frames, payload(), backend=backend, repetition=CODE))
    quality = psnr(np.stack(marked), np.stack(frames))
    detection = detect_watermark(marked, backend=backend, codec=CODEC, repetition=CODE)
    clean = detect_watermark(frames, backend=backend, codec=CODEC, repetition=CODE)
    print(
        f"\n{source} raw, strength {strength}: PSNR {quality:.1f} dB, "
        f"p={detection.p_value:.2g}, clean p={clean.p_value:.2g}"
    )
    # The default (0.5) was chosen because it keeps the PSNR above 40 dB on
    # the textured content (measured 41.5 dB; 44.7 dB on testsrc2).
    assert quality > 40
    assert detection.found
    assert detection.bits == payload()
    assert detection.payload is not None
    assert detection.payload.valid
    assert not clean.found


def channel_accuracy(backend: VideoSealBackend, encoded: bytes, tools) -> float:
    images = (frame.image for frame in iter_frames(encoded, sampling="all", tools=tools))
    scores = np.mean(list(backend.score_frames(images)), axis=0)
    read = "".join("1" if value > 0 else "0" for value in scores)
    return 1.0 - bit_error_rate(CODE.encode(payload()), read)


@pytest.mark.parametrize("strength", [0.2, 0.5])
@pytest.mark.parametrize("codec_name", sorted(CODECS))
@pytest.mark.parametrize("source", sorted(SOURCES))
def test_bit_accuracy_after_real_codecs(  # noqa: PLR0917 - pytest passes fixtures positionally
    backends, sources, tools, source, codec_name, strength
):
    backend = backends[strength]

    def encoder(frames: Iterator[np.ndarray]) -> bytes:
        return encode_frames(tools, list(frames), CODECS[codec_name], rate=RATE)

    result = embed_verified(
        sources[source],
        payload(),
        encoder=encoder,
        backend=backend,
        repetition=CODE,
        codec=CODEC,
        tools=tools,
    )
    accuracy = channel_accuracy(backend, result.encoded, tools)
    detection = result.detection
    print(
        f"\n{source} {codec_name} strength {strength}: {len(result.encoded) // 1024} KiB, "
        f"channel bit accuracy {accuracy:.3f}, payload errors {result.bit_errors}/64, "
        f"p={detection.p_value:.2g}, found={detection.found}, verified={result.verified}"
    )
    assert result.frames_embedded == COUNT
    assert detection.frames == COUNT
    assert accuracy > 0.5
    if source == "mandelbrot" and strength == 0.5:
        # The configuration documented as robust: textured content, strength
        # 0.5. Detected at every bit rate; exact at 1 Mb/s H.264 and VP9 CRF 35.
        assert detection.found
        if codec_name != "h264-400k":
            assert result.verified
