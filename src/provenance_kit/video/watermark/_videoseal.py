"""Meta VideoSeal adapter: the official TorchScript v1.0 model, 256 bits.

The model is ``y_256b_img.jit`` from the official download host
(:data:`VIDEOSEAL_MODEL`), documented in the VideoSeal repository's
``docs/torchscript.md``. It needs only PyTorch (extra ``video-watermark``);
the ``videoseal`` PyPI package is not used (ARCHITECTURE.md, decision 4).
"""

from __future__ import annotations

import threading
import warnings
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from provenance_kit._checks import check_int, check_number
from provenance_kit._logging import get_logger
from provenance_kit._optional import import_optional
from provenance_kit._typing import StrPath
from provenance_kit.bits import normalize_bits
from provenance_kit.device import parse_device, resolve_device
from provenance_kit.errors import ModelLoadError, WatermarkBackendError
from provenance_kit.modelfiles import (
    DEFAULT_DOWNLOAD_TIMEOUT_SECONDS,
    ModelFile,
    ensure_model_files,
    find_model_file,
    model_directory,
)
from provenance_kit.video.watermark.base import Frame

if TYPE_CHECKING:
    from pathlib import Path

    from provenance_kit.payload import PayloadCodec
    from provenance_kit.video.watermark.base import VideoWatermarkDetection
    from provenance_kit.video.watermark.repetition import RepetitionCode

__all__ = [
    "VIDEOSEAL_DEFAULT_STRENGTH",
    "VIDEOSEAL_MODEL",
    "VIDEOSEAL_PAYLOAD_BITS",
    "VideoSealBackend",
    "download_videoseal_models",
]

_log = get_logger(__name__)

#: The official VideoSeal v1.0 TorchScript model (256-bit image/video model).
#: SHA-256 computed on 2026-11-05 from the file served at the official URL
#: (``Last-Modified: Wed, 02 Apr 2025 07:48:57 GMT``, 228,524,815 bytes); the
#: upstream documentation publishes no checksum.
VIDEOSEAL_MODEL = ModelFile(
    name="y_256b_img.jit",
    url="https://dl.fbaipublicfiles.com/videoseal/y_256b_img.jit",
    sha256="5c7a4581c36fc6090aafdcfb3999123bae5172a4847f22e2da4e7fd1a39d1e1b",
    size=228_524_815,
)

#: Channel bits carried by the VideoSeal v1.0 model.
VIDEOSEAL_PAYLOAD_BITS = 256

#: The default watermark strength (the model's ``scaling_w``). The model's
#: built-in value is 0.2; on the test suite's synthetic 320x240 content, 0.5
#: keeps the PSNR at 41.5 dB (textured) and 44.7 dB (flat areas), against
#: 49.2 and 52.5 dB at 0.2, and raises the channel bit accuracy after H.264
#: at 1 Mb/s from 0.78 to 0.94 (textured) and from 0.61 to 0.80 (flat); see
#: ``docs/video.md``.
VIDEOSEAL_DEFAULT_STRENGTH = 0.5

_BACKEND = "videoseal"
_JIT_WARNING = r".*torch\.jit\.load.* is deprecated"


def download_videoseal_models(
    model_dir: StrPath | None = None,
    *,
    timeout_seconds: float = DEFAULT_DOWNLOAD_TIMEOUT_SECONDS,
) -> Path:
    """Downloads and verifies the VideoSeal model file (explicit, opt-in).

    Does nothing but verify when a valid file is already present; a missing
    or invalid file is fetched from the official URL
    (:func:`provenance_kit.modelfiles.ensure_model_files`). Same signature
    and return value as ``download_trustmark_models`` and
    ``download_audioseal_models``.

    Args:
        model_dir: Target directory. Defaults to
            :func:`provenance_kit.modelfiles.default_model_dir` for
            ``"videoseal"`` (``$PROVENANCE_KIT_MODEL_DIR/videoseal`` or the
            per-user cache). Created if needed.
        timeout_seconds: The network timeout of each read (default
            :data:`~provenance_kit.modelfiles.DEFAULT_DOWNLOAD_TIMEOUT_SECONDS`).

    Returns:
        The model directory (absolute), holding the verified
        ``y_256b_img.jit``.

    Raises:
        TypeError: If ``model_dir`` is not a path or ``timeout_seconds`` is
            not a number.
        ValueError: If the directory is inside ``site-packages`` or the
            timeout is not positive.
        ModelLoadError: If the download fails or the file fails its SHA-256
            check.
    """
    directory = model_directory(model_dir, _BACKEND)
    ensure_model_files([VIDEOSEAL_MODEL], directory, timeout_seconds=timeout_seconds)
    return directory


class VideoSealBackend:
    """VideoSeal invisible video watermark (implements ``VideoWatermarkBackend``).

    The model embeds 256 bits into frames: every ``step_size``-th frame goes
    through the embedder at 256 x 256 pixels, its watermark is upscaled,
    attenuated by a perceptual (JND) mask and repeated on the following
    frames. Detection scores every frame; average the scores over frames
    (as :func:`~provenance_kit.video.watermark.detect_watermark` does).

    Memory: frames are processed in batches of ``batch_frames``; a batch
    holds the frames as ``float32`` tensors at full resolution plus the
    model's 256 x 256 working copies, roughly ``36 * width * height *
    batch_frames`` bytes at peak (about 1.2 GB for 16 frames of 1080p).
    Lower ``batch_frames`` on small machines.

    Thread safety: safe to share between threads; model calls are serialized
    with an internal lock (iterators interleave between batches). The
    strength and step size are fixed at construction.

    Model location: the same options as the TrustMark and AudioSeal backends
    (``model_dir``, ``download``, ``download_timeout_seconds``), plus a
    single-file override (``model_path`` and ``sha256``) for weights kept
    elsewhere or converted by the application.

    Args:
        model_path: An explicit model file, verified against ``sha256``.
        model_dir: The directory holding ``y_256b_img.jit``. Defaults to
            ``$PROVENANCE_KIT_MODEL_DIR/videoseal`` or the per-user cache
            (:func:`provenance_kit.modelfiles.default_model_dir`); never
            inside ``site-packages``.
        download: Download the model from the official URL if it is missing.
            Off by default: the backend never touches the network unless
            asked. Not combinable with ``model_path``.
        download_timeout_seconds: The network timeout of each read during
            that download.
        sha256: The expected model digest; defaults to the pinned one.
        device: ``"cpu"`` (default), ``"cuda"``, ``"cuda:<n>"`` or ``"auto"``.
        strength: The blending factor of the watermark (the model's
            ``scaling_w``), at most 4. The default,
            :data:`VIDEOSEAL_DEFAULT_STRENGTH` (0.5), was chosen on measured
            PSNR and robustness (``docs/video.md``); higher is more robust
            and more visible. ``None`` keeps the model's built-in value
            (0.2, the upstream default).
        step_size: Embed every ``step_size``-th frame and repeat its
            watermark on the next ones (upstream default 4). ``None`` keeps
            the model's default.
        batch_frames: Frames per model call; must be a multiple of the step
            size so that the repetition pattern is the same as in one call.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an argument is out of range.
        BackendUnavailableError: If PyTorch is missing or the device is not
            available.
        ModelLoadError: If the model is missing, fails its checksum or does
            not load.
    """

    def __init__(
        self,
        *,
        model_path: StrPath | None = None,
        model_dir: StrPath | None = None,
        download: bool = False,
        download_timeout_seconds: float = DEFAULT_DOWNLOAD_TIMEOUT_SECONDS,
        sha256: str | None = None,
        device: str = "cpu",
        strength: float | None = VIDEOSEAL_DEFAULT_STRENGTH,
        step_size: int | None = None,
        batch_frames: int = 16,
    ) -> None:
        """Loads and verifies the model (see the class docstring)."""
        parse_device(device)
        check_number(
            "download_timeout_seconds",
            download_timeout_seconds,
            minimum=0.0,
            exclusive_minimum=True,
        )
        if strength is not None:
            check_number("strength", strength, minimum=0.0, maximum=4.0, exclusive_minimum=True)
        if step_size is not None:
            check_int("step_size", step_size, minimum=1, maximum=1024)
        check_int("batch_frames", batch_frames, minimum=1, maximum=4096)
        if step_size is not None and batch_frames % step_size:
            raise ValueError("batch_frames must be a multiple of step_size")
        torch = import_optional("torch", extra="video-watermark", feature="VideoSeal watermarking")
        self._torch = torch
        self._device = resolve_device(device, torch)
        path = find_model_file(
            VIDEOSEAL_MODEL,
            backend=_BACKEND,
            model_dir=model_dir,
            model_path=model_path,
            sha256=sha256,
            download=download,
            timeout_seconds=download_timeout_seconds,
        )
        try:
            with warnings.catch_warnings():
                # torch 2.10+ deprecates TorchScript; the official model ships
                # only in that format.
                warnings.filterwarnings("ignore", message=_JIT_WARNING, category=FutureWarning)
                model = torch.jit.load(str(path), map_location=self._device)
            model.eval()
            if strength is not None:
                model.blender.scaling_w = float(strength)
            if step_size is not None:
                model.step_size = int(step_size)
            self._strength = float(model.blender.scaling_w)
            self._step_size = int(model.step_size)
        except Exception as exc:
            raise ModelLoadError("the VideoSeal model could not be loaded") from exc
        if batch_frames % self._step_size:
            raise ValueError(f"batch_frames must be a multiple of step_size ({self._step_size})")
        self._model: Any = model
        self._batch_frames = batch_frames
        self._lock = threading.Lock()
        _log.debug(
            "VideoSeal loaded on %s (strength %.3f, step %d)",
            self._device,
            self._strength,
            self._step_size,
        )

    @property
    def payload_bits(self) -> int:
        """The channel bits the model carries: 256."""
        return VIDEOSEAL_PAYLOAD_BITS

    @property
    def strength(self) -> float:
        """The watermark blending factor in use."""
        return self._strength

    @property
    def step_size(self) -> int:
        """Every ``step_size``-th frame is embedded; the others repeat it."""
        return self._step_size

    @property
    def device(self) -> str:
        """The resolved compute device."""
        return self._device

    def embed_frames(self, frames: Iterable[Frame], bits: str) -> Iterator[Frame]:
        """Watermarks frames; see ``VideoWatermarkBackend.embed_frames``.

        Args:
            frames: RGB ``uint8`` frames shaped ``(height, width, 3)``.
            bits: Exactly 256 channel bits.

        Returns:
            An iterator over the watermarked frames (new arrays).

        Raises:
            TypeError: If a frame is not a NumPy array (on iteration).
            ValueError: If ``bits`` or a frame is malformed (on iteration for
                frames).
            WatermarkBackendError: If the model fails on the frames (on
                iteration).
        """
        channel = normalize_bits(bits, length=VIDEOSEAL_PAYLOAD_BITS)
        message = [float(c == "1") for c in channel]
        return self._embed(frames, message)

    def score_frames(self, frames: Iterable[Frame]) -> Iterator[npt.NDArray[np.float32]]:
        """Scores the 256 channel bits of each frame.

        Args:
            frames: RGB ``uint8`` frames shaped ``(height, width, 3)``.

        Returns:
            An iterator of ``float32`` arrays of 256 scores (the model's bit
            logits: positive means 1).

        Raises:
            TypeError: If a frame is not a NumPy array (on iteration).
            ValueError: If a frame is malformed (on iteration).
            WatermarkBackendError: If the model fails on the frames or
                returns an unexpected output shape (on iteration).
        """
        return self._score(frames)

    def detect(
        self,
        frames: Iterable[Frame],
        *,
        codec: PayloadCodec | None = None,
        repetition: RepetitionCode | None = None,
    ) -> VideoWatermarkDetection:
        """Detects the watermark: a shortcut for ``detect_watermark(..., backend=self)``.

        Args:
            frames: RGB ``uint8`` frames shaped ``(height, width, 3)``.
            codec: Decodes the payload bits, if given.
            repetition: The repetition code used when embedding, if any.

        Returns:
            The detection result.
        """
        from provenance_kit.video.watermark._api import detect_watermark  # noqa: PLC0415

        return detect_watermark(frames, backend=self, codec=codec, repetition=repetition)

    # ------------------------------------------------------------ internals

    def _embed(self, frames: Iterable[Frame], message: list[float]) -> Iterator[Frame]:
        torch = self._torch
        msg = torch.tensor([message], dtype=torch.float32, device=self._device)
        for batch in _batches(frames, self._batch_frames):
            stacked = np.stack(batch)
            try:
                with self._lock, torch.inference_mode():
                    x = torch.from_numpy(stacked).to(self._device)
                    x = x.permute(0, 3, 1, 2).float().div_(255.0)
                    marked = self._model.embed(x, msg, is_video=True)
                    out = (
                        marked.mul(255.0)
                        .round_()
                        .clamp_(0.0, 255.0)
                        .to(torch.uint8)
                        .permute(0, 2, 3, 1)
                        .contiguous()
                        .cpu()
                        .numpy()
                    )
            except MemoryError:
                raise
            except Exception as exc:
                raise WatermarkBackendError("VideoSeal embedding failed") from exc
            if out.shape != stacked.shape:
                raise WatermarkBackendError(
                    "the VideoSeal model returned an unexpected output shape"
                )
            yield from out

    def _score(self, frames: Iterable[Frame]) -> Iterator[npt.NDArray[np.float32]]:
        torch = self._torch
        for batch in _batches(frames, self._batch_frames):
            stacked = np.stack(batch)
            try:
                with self._lock, torch.inference_mode():
                    x = torch.from_numpy(stacked).to(self._device)
                    x = x.permute(0, 3, 1, 2).float().div_(255.0)
                    preds = self._model.detect(x, is_video=True)
                    scores = preds[:, 1:].float().cpu().numpy()
            except MemoryError:
                raise
            except Exception as exc:
                raise WatermarkBackendError("VideoSeal detection failed") from exc
            if scores.shape != (len(batch), VIDEOSEAL_PAYLOAD_BITS):
                raise WatermarkBackendError(
                    "the VideoSeal model returned an unexpected output shape"
                )
            yield from scores.astype(np.float32, copy=False)


def _batches(frames: Iterable[Frame], size: int) -> Iterator[list[Frame]]:
    """Groups frames into batches of ``size`` with a single shape each."""
    batch: list[Frame] = []
    shape: tuple[int, ...] | None = None
    for frame in frames:
        _check_frame(frame)
        if batch and frame.shape != shape:
            yield batch
            batch = []
        shape = frame.shape
        batch.append(frame)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def _check_frame(frame: object) -> None:
    if not isinstance(frame, np.ndarray):
        raise TypeError("frames must be NumPy arrays")
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frames must be uint8 arrays shaped (height, width, 3)")
    if frame.shape[0] < 1 or frame.shape[1] < 1:
        raise ValueError("frames must not be empty")
