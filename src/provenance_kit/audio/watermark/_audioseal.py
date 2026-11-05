"""Meta AudioSeal adapter: the 16-bit base generator and detector.

Checkpoints: ``generator_base.pth`` and ``detector_base.pth`` from the
``facebook/audioseal`` Hugging Face repository, pinned to a revision and by
SHA-256 (:data:`AUDIOSEAL_GENERATOR`, :data:`AUDIOSEAL_DETECTOR`). Both files
are MIT-licensed since the 2024-04-02 relicensing of the weights; see
``docs/THIRD_PARTY.md`` and ``docs/audio.md`` for the provenance check. The
streaming checkpoints of the same repository are not used.

Loading never runs an unrestricted unpickler: the checkpoints are verified
against their pinned hash first, then read with ``torch.load`` and a
restricted unpickler that allows only tensors, plain containers and the
configuration classes the checkpoints reference (which are replaced by inert
placeholders: the configuration is taken from this module, copied from the
``audioseal`` 0.2.1 model cards).
"""

from __future__ import annotations

import collections
import contextlib
import copy
import importlib
import itertools
import math
import pickle
import threading
import types
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from provenance_kit._checks import check_int, check_number
from provenance_kit._logging import get_logger
from provenance_kit._optional import import_optional
from provenance_kit._typing import StrPath
from provenance_kit.audio._resample import resample
from provenance_kit.audio.watermark.base import AudioScores, Samples
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

if TYPE_CHECKING:
    from provenance_kit.audio.watermark._api import AlignmentSearch
    from provenance_kit.audio.watermark.base import AudioWatermarkDetection
    from provenance_kit.payload import PayloadCodec

__all__ = [
    "AUDIOSEAL_DETECTOR",
    "AUDIOSEAL_GENERATOR",
    "AUDIOSEAL_PAYLOAD_BITS",
    "AUDIOSEAL_REVISION",
    "AUDIOSEAL_SAMPLE_RATE",
    "AudioSealBackend",
    "download_audioseal_models",
]

_log = get_logger(__name__)

#: The ``facebook/audioseal`` repository revision the checkpoints are pinned
#: to (2025-12-09; its model card declares ``license: mit``). The two base
#: files are byte-identical at every revision since their upload.
AUDIOSEAL_REVISION = "3c19eba53390776cf2cc9ed5f6c9ac67ce72ecba"

_HF = f"https://huggingface.co/facebook/audioseal/resolve/{AUDIOSEAL_REVISION}"

#: The 16-bit base generator (embedder).
AUDIOSEAL_GENERATOR = ModelFile(
    name="generator_base.pth",
    url=f"{_HF}/generator_base.pth",
    sha256="7a845b5fbe9364a63a3909d8ab3fe064d13a76ae4c2e983573e08c69b7b51748",
    size=58_805_980,
)

#: The 16-bit base detector.
AUDIOSEAL_DETECTOR = ModelFile(
    name="detector_base.pth",
    url=f"{_HF}/detector_base.pth",
    sha256="8a78e8a83584113523e161fc599fcab10fd0e94c04d2eb9d2fa1e9ec91ab69d9",
    size=34_667_641,
)

#: Payload bits carried by the base models.
AUDIOSEAL_PAYLOAD_BITS = 16

#: The sample rate the models work at; other rates are resampled in memory.
AUDIOSEAL_SAMPLE_RATE = 16_000

_BACKEND = "audioseal"
_MIN_TAIL_SECONDS = 1.0
_PROBABILITY_EPS = 1e-7

# Model configuration from audioseal 0.2.1's cards (audioseal_wm_16bits.yaml
# and audioseal_detector_16bits.yaml, MIT), matching the pinned checkpoints.
_SEANET: dict[str, Any] = {
    "activation": "ELU",
    "activation_params": {"alpha": 1.0},
    "causal": False,
    "channels": 1,
    "compress": 2,
    "dilation_base": 2,
    "dimension": 128,
    "disable_norm_outer_blocks": 0,
    "kernel_size": 7,
    "last_kernel_size": 7,
    "lstm": 2,
    "n_filters": 32,
    "n_residual_layers": 1,
    "norm": "weight_norm",
    "norm_params": {},
    "pad_mode": "constant",
    "ratios": [8, 5, 4, 2],
    "residual_kernel_size": 3,
    "true_skip": True,
}
_GENERATOR_CONFIG: dict[str, Any] = {
    "nbits": AUDIOSEAL_PAYLOAD_BITS,
    "seanet": _SEANET,
    "decoder": {"final_activation": None, "final_activation_params": None, "trim_right_ratio": 1.0},
}
_DETECTOR_CONFIG: dict[str, Any] = {
    "nbits": AUDIOSEAL_PAYLOAD_BITS,
    "seanet": _SEANET,
    "detector": {"output_dim": 32},
}

# Deprecation warnings that the upstream code triggers in current PyTorch
# releases (TorchScript decorators at import, weight_norm while loading).
_UPSTREAM_WARNINGS = (
    r".*torch\.jit\.\w+.* is deprecated",
    r".*torch\.nn\.utils\.weight_norm.* is deprecated",
)


def download_audioseal_models(
    model_dir: StrPath | None = None,
    *,
    timeout_seconds: float = DEFAULT_DOWNLOAD_TIMEOUT_SECONDS,
) -> Path:
    """Downloads and verifies the AudioSeal generator and detector (explicit, opt-in).

    Valid files already present are only verified; missing or invalid ones
    are fetched from the pinned Hugging Face revision
    (:func:`provenance_kit.modelfiles.ensure_model_files`). Same signature
    and return value as ``download_trustmark_models`` and
    ``download_videoseal_models``.

    Args:
        model_dir: Target directory. Defaults to
            :func:`provenance_kit.modelfiles.default_model_dir` for
            ``"audioseal"`` (``$PROVENANCE_KIT_MODEL_DIR/audioseal`` or the
            per-user cache). Created if needed.
        timeout_seconds: The network timeout of each read (default
            :data:`~provenance_kit.modelfiles.DEFAULT_DOWNLOAD_TIMEOUT_SECONDS`).

    Returns:
        The model directory (absolute), holding the verified
        ``generator_base.pth`` and ``detector_base.pth``.

    Raises:
        TypeError: If ``model_dir`` is not a path or ``timeout_seconds`` is
            not a number.
        ValueError: If the directory is inside ``site-packages`` or the
            timeout is not positive.
        ModelLoadError: If a download fails or a file fails its SHA-256 check.
    """
    directory = model_directory(model_dir, _BACKEND)
    ensure_model_files(
        (AUDIOSEAL_GENERATOR, AUDIOSEAL_DETECTOR), directory, timeout_seconds=timeout_seconds
    )
    return directory


class AudioSealBackend:
    """AudioSeal invisible audio watermark (implements ``AudioWatermarkBackend``).

    Embedding: the channels are mixed down to mono and resampled to 16 kHz,
    the generator computes the watermark (a low-level signal carrying 16
    bits), which is resampled back to the input rate, scaled by ``strength``
    and added to **every** channel. The same watermark in all channels
    survives a later mono downmix. Detection mixes down and resamples the
    same way and reports the fraction of 16 kHz samples the detector
    classifies as watermarked, plus the 16 bit probabilities (averaged over
    the whole signal).

    Alignment: the models are sensitive to sample-level time shifts (an
    upstream property, measured with the stock API too: a 3-sample shift at
    16 kHz drops the detection probability from 1.0 to 0.06). Encodings that
    record their encoder delay (AAC in MP4/M4A, MP3 files with the LAME/Info
    tag, Ogg Opus, FLAC) decode aligned and keep the mark; raw ADTS AAC and
    MP3 streamed to a pipe shift the audio by the encoder delay and weaken
    detection and bits. :func:`~provenance_kit.audio.watermark.embed_verified`
    checks the application's real pipeline for exactly this reason.

    Memory: long signals are processed in chunks of ``chunk_seconds`` at
    16 kHz (the models are not streamed; each chunk is watermarked and
    detected independently, which suits AudioSeal's sample-level design).

    Thread safety: safe to share between threads; model calls are serialized
    with an internal lock. The models run in eager mode: the backend
    disables the ``torch.compile`` path of audioseal's vendored modules
    (``audioseal.libs.moshi.utils.compile.no_compile``) during its calls,
    which needs no C++ compiler and keeps results deterministic. That switch
    is module-global in audioseal, so other threads calling audioseal
    directly at the same time also run eagerly.

    Model location: the same options as the TrustMark and VideoSeal backends
    (``model_dir``, ``download``, ``download_timeout_seconds``), plus a
    single-file override per checkpoint (``generator_path`` /
    ``generator_sha256`` and ``detector_path`` / ``detector_sha256``) for
    weights kept elsewhere.

    Args:
        model_dir: The directory holding ``generator_base.pth`` and
            ``detector_base.pth``. Defaults to
            ``$PROVENANCE_KIT_MODEL_DIR/audioseal`` or the per-user cache
            (:func:`provenance_kit.modelfiles.default_model_dir`); never
            inside ``site-packages``.
        generator_path: An explicit generator file instead.
        detector_path: An explicit detector file instead.
        download: Download missing files from the pinned Hugging Face
            revision. Off by default: the backend never touches the network
            unless asked. A checkpoint given by an explicit path is never
            downloaded.
        download_timeout_seconds: The network timeout of each read during
            that download.
        generator_sha256: The expected generator digest; defaults to the
            pinned one.
        detector_sha256: The expected detector digest; defaults to the
            pinned one.
        device: ``"cpu"`` (default), ``"cuda"``, ``"cuda:<n>"`` or ``"auto"``.
        strength: Scale of the added watermark (AudioSeal's ``alpha``; 1.0 is
            the trained level). Lower is quieter and less robust.
        chunk_seconds: Length of the chunks processed at once (at 16 kHz).
        frame_threshold: Per-sample probability above which the detector's
            output counts as "watermarked".

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an argument is out of range.
        BackendUnavailableError: If ``audioseal`` or PyTorch is missing, or the
            device is not available.
        ModelLoadError: If a checkpoint is missing, fails its checksum or does
            not load.
    """

    def __init__(
        self,
        *,
        model_dir: StrPath | None = None,
        generator_path: StrPath | None = None,
        detector_path: StrPath | None = None,
        download: bool = False,
        download_timeout_seconds: float = DEFAULT_DOWNLOAD_TIMEOUT_SECONDS,
        generator_sha256: str | None = None,
        detector_sha256: str | None = None,
        device: str = "cpu",
        strength: float = 1.0,
        chunk_seconds: float = 10.0,
        frame_threshold: float = 0.5,
    ) -> None:
        """Loads and verifies both models (see the class docstring)."""
        parse_device(device)
        self._strength = check_number(
            "strength", strength, minimum=0.0, maximum=10.0, exclusive_minimum=True
        )
        seconds = check_number("chunk_seconds", chunk_seconds, minimum=1.0, maximum=600.0)
        self._chunk = round(seconds * AUDIOSEAL_SAMPLE_RATE)
        self._frame_threshold = check_number(
            "frame_threshold", frame_threshold, minimum=0.0, maximum=1.0
        )
        check_number(
            "download_timeout_seconds",
            download_timeout_seconds,
            minimum=0.0,
            exclusive_minimum=True,
        )
        feature = "AudioSeal watermarking"
        torch = import_optional("torch", extra="audio-watermark", feature=feature)
        with _quiet_upstream():
            loader = import_optional("audioseal.loader", extra="audio-watermark", feature=feature)
            builder = import_optional("audioseal.builder", extra="audio-watermark", feature=feature)
        self._torch = torch
        self._device = resolve_device(device, torch)
        generator_file = find_model_file(
            AUDIOSEAL_GENERATOR,
            backend=_BACKEND,
            model_dir=model_dir,
            model_path=generator_path,
            sha256=generator_sha256,
            download=download and generator_path is None,
            timeout_seconds=download_timeout_seconds,
        )
        detector_file = find_model_file(
            AUDIOSEAL_DETECTOR,
            backend=_BACKEND,
            model_dir=model_dir,
            model_path=detector_path,
            sha256=detector_sha256,
            download=download and detector_path is None,
            timeout_seconds=download_timeout_seconds,
        )
        try:
            with _quiet_upstream():
                self._generator = _build(
                    torch,
                    loader,
                    create=builder.create_generator,
                    config_type=builder.AudioSealWMConfig,
                    config=_GENERATOR_CONFIG,
                    path=generator_file,
                    device=self._device,
                )
                self._detector = _build(
                    torch,
                    loader,
                    create=builder.create_detector,
                    config_type=builder.AudioSealDetectorConfig,
                    config=_DETECTOR_CONFIG,
                    path=detector_file,
                    device=self._device,
                )
        except ModelLoadError:
            raise
        except Exception as exc:
            raise ModelLoadError("the AudioSeal models could not be loaded") from exc
        self._no_compile = _no_compile_context()
        self._lock = threading.Lock()
        _log.debug("AudioSeal loaded on %s (strength %.3f)", self._device, self._strength)

    @property
    def payload_bits(self) -> int:
        """The payload bits the models carry: 16."""
        return AUDIOSEAL_PAYLOAD_BITS

    @property
    def sample_rate(self) -> int:
        """The rate the models work at: 16,000 Hz."""
        return AUDIOSEAL_SAMPLE_RATE

    @property
    def strength(self) -> float:
        """The watermark scale in use."""
        return self._strength

    @property
    def device(self) -> str:
        """The resolved compute device."""
        return self._device

    def embed(self, samples: Samples, sample_rate: int, bits: str) -> Samples:
        """Watermarks audio; see ``AudioWatermarkBackend.embed``.

        Args:
            samples: ``float32`` PCM shaped ``(channels, samples)``.
            sample_rate: The sample rate in hertz.
            bits: Exactly 16 payload bits.

        Returns:
            The watermarked PCM (a new array), clipped to ``[-1, 1]``.

        Raises:
            TypeError: If ``samples`` is not a NumPy float array.
            ValueError: If the shape, rate or bits are invalid.
            WatermarkBackendError: If the model fails on the signal.
        """
        audio = check_samples(samples)
        rate = check_int("sample_rate", sample_rate, minimum=1)
        payload = normalize_bits(bits, length=AUDIOSEAL_PAYLOAD_BITS)
        torch = self._torch
        message = torch.tensor([[int(c) for c in payload]], dtype=torch.int64, device=self._device)
        mono = resample(audio.mean(axis=0), rate, AUDIOSEAL_SAMPLE_RATE)
        watermark = np.zeros_like(mono)
        for start, stop in self._chunks(mono.size):
            segment = self._tensor(mono[start:stop])
            with self._run():
                out = self._generator.get_watermark(segment, message=message)
                watermark[start:stop] = out[0, 0].float().cpu().numpy()
        back = resample(watermark, AUDIOSEAL_SAMPLE_RATE, rate)
        added = np.zeros(audio.shape[1], dtype=np.float32)
        length = min(added.size, back.size)
        added[:length] = back[:length]
        marked = audio + np.float32(self._strength) * added[None, :]
        np.clip(marked, -1.0, 1.0, out=marked)
        return marked.astype(np.float32, copy=False)

    def score(self, samples: Samples, sample_rate: int) -> AudioScores:
        """Runs the detector; see ``AudioWatermarkBackend.score``.

        Args:
            samples: ``float32`` PCM shaped ``(channels, samples)``.
            sample_rate: The sample rate in hertz.

        Returns:
            The fraction of samples detected as watermarked and the 16 bit
            probabilities.

        Raises:
            TypeError: If ``samples`` is not a NumPy float array.
            ValueError: If the shape or rate is invalid.
            WatermarkBackendError: If the model fails on the signal or
                returns non-finite scores.
        """
        audio = check_samples(samples)
        rate = check_int("sample_rate", sample_rate, minimum=1)
        mono = resample(audio.mean(axis=0), rate, AUDIOSEAL_SAMPLE_RATE)
        positives = 0
        frames = 0
        logit_sum = np.zeros(AUDIOSEAL_PAYLOAD_BITS, dtype=np.float64)
        for start, stop in self._chunks(mono.size):
            segment = self._tensor(mono[start:stop])
            with self._run():
                result, message = self._detector(segment)
                probability = result[0, 1, :].double()
                count = int(probability.numel())
                positives += int((probability > self._frame_threshold).sum().item())
                bits = message[0].double().clamp(_PROBABILITY_EPS, 1.0 - _PROBABILITY_EPS)
                logits = torch_logit(bits).cpu().numpy()
            frames += count
            logit_sum += logits * count
        mean_logits = logit_sum / max(frames, 1)
        if not np.all(np.isfinite(mean_logits)):
            raise WatermarkBackendError("the AudioSeal detector returned non-finite scores")
        bit_probabilities = tuple(float(1.0 / (1.0 + math.exp(-v))) for v in mean_logits)
        return AudioScores(
            detection_probability=positives / max(frames, 1),
            bit_probabilities=bit_probabilities,
        )

    def detect(
        self,
        samples: Samples,
        sample_rate: int,
        *,
        codec: PayloadCodec | None = None,
        threshold: float = 0.5,
        alignment: AlignmentSearch | None = None,
    ) -> AudioWatermarkDetection:
        """Detects the watermark: a shortcut for ``detect_watermark(..., backend=self)``.

        Args:
            samples: ``float32`` PCM shaped ``(channels, samples)``.
            sample_rate: The sample rate in hertz.
            codec: Decodes the 16 payload bits, if given.
            threshold: Detection probability required for ``found``.
            alignment: An optional search over sample offsets (see
                :class:`~provenance_kit.audio.watermark.AlignmentSearch`).

        Returns:
            The detection result.
        """
        from provenance_kit.audio.watermark._api import detect_watermark  # noqa: PLC0415

        return detect_watermark(
            samples,
            sample_rate,
            backend=self,
            codec=codec,
            threshold=threshold,
            alignment=alignment,
        )

    # ------------------------------------------------------------ internals

    def _chunks(self, length: int) -> Iterator[tuple[int, int]]:
        if length == 0:
            return
        bounds = [*range(0, length, self._chunk), length]
        tail = bounds[-1] - bounds[-2]
        if len(bounds) > 2 and tail < _MIN_TAIL_SECONDS * AUDIOSEAL_SAMPLE_RATE:
            del bounds[-2]  # fold a short tail into the previous chunk
        yield from itertools.pairwise(bounds)

    def _tensor(self, segment: npt.NDArray[np.float32]) -> Any:
        return self._torch.from_numpy(np.ascontiguousarray(segment)).to(self._device)[None, None]

    @contextlib.contextmanager
    def _run(self) -> Iterator[None]:
        try:
            with self._lock, self._torch.inference_mode(), self._no_compile():
                yield
        except MemoryError:
            raise
        except Exception as exc:
            raise WatermarkBackendError("AudioSeal could not process this signal") from exc


def torch_logit(values: Any) -> Any:
    """``log(p / (1 - p))`` on a tensor (internal helper)."""
    return (values / (1.0 - values)).log()


def check_samples(samples: object) -> Samples:
    """Validates PCM samples and returns them as C-contiguous float32.

    Args:
        samples: A float NumPy array shaped ``(channels, samples)``.

    Returns:
        The samples as ``float32`` (a copy only when needed).

    Raises:
        TypeError: If ``samples`` is not a floating-point NumPy array.
        ValueError: If the shape is not ``(channels, samples)`` with at least
            one sample, or a value is not finite.
    """
    if not isinstance(samples, np.ndarray) or not np.issubdtype(samples.dtype, np.floating):
        raise TypeError("samples must be a floating-point NumPy array")
    if samples.ndim != 2 or samples.shape[0] < 1 or samples.shape[1] < 1:
        raise ValueError("samples must be shaped (channels, samples) with at least one sample")
    audio = np.ascontiguousarray(samples, dtype=np.float32)
    if not np.all(np.isfinite(audio)):
        raise ValueError("samples must be finite")
    return audio


@contextlib.contextmanager
def _quiet_upstream() -> Iterator[None]:
    with warnings.catch_warnings():
        for message in _UPSTREAM_WARNINGS:
            warnings.filterwarnings("ignore", message=message, category=FutureWarning)
            warnings.filterwarnings("ignore", message=message, category=DeprecationWarning)
        yield


def _no_compile_context() -> Any:
    try:
        module = importlib.import_module("audioseal.libs.moshi.utils.compile")
    except ImportError:
        return contextlib.nullcontext
    return getattr(module, "no_compile", contextlib.nullcontext)


def _build(
    torch: Any,
    loader: Any,
    *,
    create: Any,
    config_type: Any,
    config: dict[str, Any],
    path: Path,
    device: str,
) -> Any:
    parsed = loader.AudioSeal.parse_config(
        copy.deepcopy(config), config_type=config_type, nbits=AUDIOSEAL_PAYLOAD_BITS
    )
    model = create(parsed)
    state = _load_state_dict(torch, path)
    loader.load_state_dict(model, loader.convert_state_dict_for_scriptable_model(state))
    return model.to(device).eval()


class _Inert:
    """Placeholder for configuration objects in the checkpoints (ignored)."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    def __setstate__(self, state: object) -> None:
        del state


class _InertDict(dict[Any, Any]):
    """Placeholder for ``collections.defaultdict`` (its factory is dropped)."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        super().__init__()


_INERT_GLOBALS = frozenset(
    {
        ("omegaconf.dictconfig", "DictConfig"),
        ("omegaconf.listconfig", "ListConfig"),
        ("omegaconf.base", "ContainerMetadata"),
        ("omegaconf.base", "Metadata"),
        ("omegaconf.nodes", "AnyNode"),
        ("typing", "Any"),
    }
)


def _refuse_legacy(*args: object, **kwargs: object) -> Any:
    del args, kwargs
    raise pickle.UnpicklingError("legacy (non-zip) checkpoints are not supported")


def _load_state_dict(torch: Any, path: Path) -> dict[str, Any]:
    """Reads a verified checkpoint with an allowlist unpickler; returns its state dict."""
    allowed: dict[tuple[str, str], Any] = {
        ("collections", "OrderedDict"): collections.OrderedDict,
        ("collections", "defaultdict"): _InertDict,
        ("torch._utils", "_rebuild_tensor_v2"): torch._utils._rebuild_tensor_v2,
        ("torch._utils", "_rebuild_parameter"): torch._utils._rebuild_parameter,
        ("builtins", "int"): int,
        ("builtins", "float"): float,
        ("builtins", "list"): list,
        ("builtins", "dict"): dict,
        ("builtins", "tuple"): tuple,
    }

    class _AllowlistUnpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str) -> Any:
            if module == "__builtin__":
                module, name = "builtins", {"long": "int"}.get(name, name)
            target = allowed.get((module, name))
            if target is not None:
                return target
            if (module, name) in _INERT_GLOBALS:
                return _Inert
            raise pickle.UnpicklingError("the checkpoint references a type that is not allowed")

    restricted = types.ModuleType("provenance_kit_restricted_pickle")
    restricted.Unpickler = _AllowlistUnpickler  # type: ignore[attr-defined]
    restricted.load = _refuse_legacy  # type: ignore[attr-defined]
    try:
        checkpoint = torch.load(
            str(path), map_location="cpu", weights_only=False, pickle_module=restricted
        )
    except Exception as exc:
        raise ModelLoadError(f"checkpoint {path.name} could not be read") from exc
    state = checkpoint.get("model") if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()
    ):
        raise ModelLoadError(f"checkpoint {path.name} does not hold a state dict")
    return state
