"""Adobe TrustMark backend (internal).

The public names are re-exported by :mod:`provenance_kit.image.watermark`.
The adapter loads TrustMark's encoder and decoder from a model directory the
caller controls, verifies every file against a pinned SHA-256 before using
it, never downloads unless asked to, and keeps the upstream package quiet.
It reuses TrustMark's own embedding and BCH code (``trustmark`` 0.9.2) but
not its model loader, which downloads into ``site-packages`` with MD5 checks
and prints progress.
"""

from __future__ import annotations

import enum
import hashlib
import io
import math
import os
import site
import sys
import sysconfig
import threading
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from provenance_kit._logging import get_logger
from provenance_kit._optional import import_optional
from provenance_kit._typing import StrPath
from provenance_kit.detection import WatermarkDetection
from provenance_kit.device import parse_device, resolve_device
from provenance_kit.errors import ModelLoadError, ProvenanceKitError
from provenance_kit.image import _modes
from provenance_kit.tempfiles import private_temp_file

__all__ = [
    "MODEL_BASE_URL",
    "MODEL_DIR_ENV",
    "MODEL_FILES",
    "ModelFile",
    "TrustMarkBackend",
    "TrustMarkDetection",
    "TrustMarkEncoding",
    "TrustMarkVariant",
    "WatermarkBackendError",
    "default_model_dir",
    "download_trustmark_models",
]

_log = get_logger("provenance_kit.image.watermark")

_EXTRA = "image-watermark"
_FEATURE = "TrustMark watermarking"

#: Where :func:`download_trustmark_models` fetches the files (Adobe's CDN for
#: the Content Authenticity Initiative, the same host ``trustmark`` uses).
MODEL_BASE_URL = "https://cai-watermark.adobe.net/watermarking/trustmark-models/"

#: Environment variable naming the root directory for provenance-kit models;
#: TrustMark files go in its ``trustmark`` subdirectory.
MODEL_DIR_ENV = "PROVENANCE_KIT_MODEL_DIR"

_SECRET_LEN = 100  # bits the TrustMark models carry: data + BCH + 4 schema bits
_ECC_BITS = 96  # data + BCH; the last 4 bits name the schema
_MAX_STRENGTH = 10.0


class WatermarkBackendError(ProvenanceKitError):
    """The watermark model failed while processing an image.

    Raised when the model itself fails (for example a device error); the
    original exception is chained. Bad inputs raise ``TypeError`` or
    ``ValueError``; a missing package raises
    :class:`~provenance_kit.errors.BackendUnavailableError`.
    """


class TrustMarkVariant(enum.StrEnum):
    """TrustMark model variants (Adobe's naming).

    * ``Q``: the default; ResNet-50 decoder, typical PSNR about 43 dB.
    * ``P``: weighted towards perceptual quality, PSNR about 48 dB; always
      marks the centre square of the image.
    * ``B``: like ``Q`` with slightly more robustness, as in the 2023 paper.
    * ``C``: compact ResNet-18 decoder, PSNR about 39 dB.
    """

    Q = "Q"
    P = "P"
    B = "B"
    C = "C"


class TrustMarkEncoding(enum.StrEnum):
    """TrustMark's BCH error-correction modes.

    The model carries 100 bits: the payload, ``7 * t`` BCH parity bits over
    GF(2^7) correcting ``t`` bit errors, and 4 schema bits. Capacities, from
    ``trustmark/datalayer.py`` (``schemaCapacity`` and ``raw_payload_split``)
    in ``trustmark`` 0.9.2:

    ===============  ======  ============  ==================
    Mode             Schema  Payload bits  Correctable errors
    ===============  ======  ============  ==================
    ``BCH_SUPER``    0       40            8
    ``BCH_5``        1       61            5
    ``BCH_4``        2       68            4
    ``BCH_3``        3       75            3
    ===============  ======  ============  ==================
    """

    BCH_SUPER = "bch_super"
    BCH_5 = "bch_5"
    BCH_4 = "bch_4"
    BCH_3 = "bch_3"

    @property
    def schema(self) -> int:
        """TrustMark's schema number (its ``Encoding`` constant)."""
        return _ENCODINGS[self][0]

    @property
    def payload_bits(self) -> int:
        """Payload capacity in bits."""
        return _ENCODINGS[self][1]

    @property
    def correctable_bits(self) -> int:
        """Bit errors the BCH code corrects."""
        return _ENCODINGS[self][2]


_ENCODINGS: dict[TrustMarkEncoding, tuple[int, int, int]] = {
    TrustMarkEncoding.BCH_SUPER: (0, 40, 8),
    TrustMarkEncoding.BCH_5: (1, 61, 5),
    TrustMarkEncoding.BCH_4: (2, 68, 4),
    TrustMarkEncoding.BCH_3: (3, 75, 3),
}
_BY_SCHEMA = {schema: encoding for encoding, (schema, _, _) in _ENCODINGS.items()}


@dataclass(frozen=True, slots=True)
class ModelFile:
    """A pinned model file.

    Attributes:
        name: The file name in the model directory and on the CDN.
        size: Its size in bytes.
        sha256: Its SHA-256, lowercase hex.
    """

    name: str
    size: int
    sha256: str


_CONFIG_QPB = "43f37103f92efa8bd6b1c5902bb537cc12a981dc699fca19d9bb7de8c62d03d9"

#: The files each variant needs: (config, encoder, decoder). Hashes taken on
#: 2026-11-05 from the CDN; each file's MD5 matched ``MODEL_CHECKSUMS`` in
#: ``trustmark`` 0.9.2. The watermark remover and the box detector are not
#: used and not listed.
MODEL_FILES: Mapping[TrustMarkVariant, tuple[ModelFile, ModelFile, ModelFile]] = {
    TrustMarkVariant.Q: (
        ModelFile("trustmark_Q.yaml", 2009, _CONFIG_QPB),
        ModelFile(
            "encoder_Q.ckpt",
            17302074,
            "dc382c3f6b4fd568b27d6fbb763d6ffc2d2587126d84afe9d5ee95b4c5d99826",
        ),
        ModelFile(
            "decoder_Q.ckpt",
            47652460,
            "e3d9cea5406a26590735719f8f15cb10802b11852ae69047eaf4cf17214df781",
        ),
    ),
    TrustMarkVariant.P: (
        ModelFile("trustmark_P.yaml", 2009, _CONFIG_QPB),
        ModelFile(
            "encoder_P.ckpt",
            17301306,
            "659d427f72d16eea4e8fbda175ae72e3a78830dc6cf44b63642a5a4b28a2b4e2",
        ),
        ModelFile(
            "decoder_P.ckpt",
            47647180,
            "f5f1d570c889c5908c6e4a28c07dc90cf33990ebaf24037254ebe0b8849b1bdc",
        ),
    ),
    TrustMarkVariant.B: (
        ModelFile("trustmark_B.yaml", 2009, _CONFIG_QPB),
        ModelFile(
            "encoder_B.ckpt",
            17301658,
            "0bbdff257500879803ed0b931cfa3ba9819af971cb3384c2a369e20c57f92b49",
        ),
        ModelFile(
            "decoder_B.ckpt",
            47649788,
            "e3cab2862b7eeff22ddba0a25a230d6da79d9ecf8775e76340ac9ee0931152ed",
        ),
    ),
    TrustMarkVariant.C: (
        ModelFile(
            "trustmark_C.yaml",
            2047,
            "cf38887be05093c23c874ba999d38a70e1fbd2c67b18d388a64bc6446be661d2",
        ),
        ModelFile(
            "encoder_C.ckpt",
            17301658,
            "2eeeaf02178976f04d0996280466a22e53418fee8d7c063990b1a039aec0a50b",
        ),
        ModelFile(
            "decoder_C.ckpt",
            22520762,
            "ede8305d71b569a3a1c11cf51a76a7729a7ec2bc00e74b9a5d617458763b3a42",
        ),
    ),
}

# The configuration files name Python classes to instantiate. Only TrustMark's
# own modules are allowed, and only decoder architectures that do not fetch
# pretrained weights from the internet.
_ALLOWED_TARGET_PREFIX = "trustmark."
_ALLOWED_DECODER_ARCHS = frozenset({"resnet50", "resnet18"})
_QUIET_MODULES = (
    "trustmark.trustmark",
    "trustmark.model",
    "trustmark.unet",
    "trustmark.datalayer",
    "trustmark.denoise",
    "trustmark.bchecc",
)


@dataclass(frozen=True, slots=True, kw_only=True)
class TrustMarkDetection(WatermarkDetection):
    """A TrustMark detection result.

    ``found`` is true only when the BCH code decoded the bits *and* the
    schema bits name the backend's configured encoding. TrustMark gives no
    calibrated score, so ``confidence`` is ``None``.

    Attributes:
        encoding: The encoding of the detected mark (when found).
        corrected_bits: How many of the 96 coded bits the BCH code had to
            correct (when found): 0 is a clean read; values near
            ``encoding.correctable_bits`` mean the mark barely survived.
    """

    encoding: TrustMarkEncoding | None = None
    corrected_bits: int | None = None


# ---------------------------------------------------------------- model files


def default_model_dir() -> Path:
    """Returns the default TrustMark model directory.

    ``$PROVENANCE_KIT_MODEL_DIR/trustmark`` when the variable is set; else
    ``provenance-kit/models/trustmark`` in a per-user cache directory:
    ``%LOCALAPPDATA%`` on Windows, ``~/Library/Caches`` on macOS,
    ``$XDG_CACHE_HOME`` (default ``~/.cache``) elsewhere. Never inside
    ``site-packages``.

    Returns:
        The directory (it may not exist yet).
    """
    configured = os.environ.get(MODEL_DIR_ENV, "").strip()
    if configured:
        return Path(configured) / "trustmark"
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "provenance-kit" / "models" / "trustmark"


def _site_roots() -> set[Path]:
    candidates: list[str] = []
    paths = sysconfig.get_paths()
    candidates += [paths.get("purelib", ""), paths.get("platlib", "")]
    getsitepackages = getattr(site, "getsitepackages", None)
    if getsitepackages is not None:
        candidates += list(getsitepackages())
    user_site = getattr(site, "getusersitepackages", None)
    if user_site is not None:
        candidates.append(user_site())
    return {Path(c).resolve() for c in candidates if c}


def _model_dir(model_dir: StrPath | None) -> Path:
    directory = Path(model_dir) if model_dir is not None else default_model_dir()
    resolved = directory.resolve()
    for root in _site_roots():
        if resolved == root or root in resolved.parents:
            raise ValueError("the model directory must not be inside site-packages")
    return resolved


def _read_verified(directory: Path, spec: ModelFile) -> bytes:
    path = directory / spec.name
    try:
        with path.open("rb") as handle:
            data = handle.read(spec.size + 1)
    except FileNotFoundError:
        raise ModelLoadError(
            f"TrustMark model file {spec.name} is not in the model directory: provision it "
            "or call download_trustmark_models()"
        ) from None
    except OSError as exc:
        raise ModelLoadError(f"TrustMark model file {spec.name} could not be read") from exc
    if len(data) != spec.size or hashlib.sha256(data).hexdigest() != spec.sha256:
        raise ModelLoadError(f"TrustMark model file {spec.name} failed its SHA-256 check")
    return data


def _is_valid(directory: Path, spec: ModelFile) -> bool:
    try:
        _read_verified(directory, spec)
    except ModelLoadError:
        return False
    return True


class _HttpsOnlyRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(  # noqa: PLR0917 - signature fixed by urllib
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        if not newurl.lower().startswith("https://"):
            raise urllib.error.URLError("refusing a redirect away from HTTPS")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _fetch(url: str, size: int, timeout_seconds: float) -> bytes:
    opener = urllib.request.build_opener(_HttpsOnlyRedirects())
    request = urllib.request.Request(url, headers={"User-Agent": "provenance-kit"})  # noqa: S310 - pinned https URL
    try:
        with opener.open(request, timeout=timeout_seconds) as response:
            chunks: list[bytes] = []
            received = 0
            while received <= size:
                chunk = response.read(min(1 << 20, size + 1 - received))
                if not chunk:
                    break
                chunks.append(chunk)
                received += len(chunk)
    except (OSError, ValueError) as exc:  # URLError, HTTPError and timeouts are OSErrors
        raise ModelLoadError(f"TrustMark model download failed ({type(exc).__name__})") from exc
    return b"".join(chunks)


def download_trustmark_models(
    variant: TrustMarkVariant | str = TrustMarkVariant.Q,
    *,
    model_dir: StrPath | None = None,
    timeout_seconds: float = 120.0,
) -> Path:
    """Downloads a variant's model files into the model directory.

    This is the only function in the image module that uses the network.
    Files already present with the right SHA-256 are kept; others are
    fetched over HTTPS from :data:`MODEL_BASE_URL`, checked against
    :data:`MODEL_FILES` (size and SHA-256) and written atomically. Run it at
    deployment time (or provision the files by hand) and construct
    :class:`TrustMarkBackend` offline.

    Args:
        variant: The variant whose files to fetch (about 65 MB for Q, P and
            B, 40 MB for C).
        model_dir: The target directory; ``None`` uses
            :func:`default_model_dir`. Created if needed.
        timeout_seconds: Socket timeout for each request.

    Returns:
        The model directory.

    Raises:
        TypeError: If ``timeout_seconds`` is not a number.
        ValueError: If the variant is unknown, the directory is inside
            ``site-packages`` or the timeout is not positive.
        ModelLoadError: If a download fails or a file fails its check.
    """
    chosen = TrustMarkVariant(variant)
    timeout: object = timeout_seconds
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise TypeError("timeout_seconds must be a number")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout_seconds must be positive and finite")
    directory = _model_dir(model_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for spec in MODEL_FILES[chosen]:
        if _is_valid(directory, spec):
            continue
        _log.info("downloading TrustMark model file %s (%d bytes)", spec.name, spec.size)
        data = _fetch(MODEL_BASE_URL + spec.name, spec.size, float(timeout_seconds))
        if len(data) != spec.size or hashlib.sha256(data).hexdigest() != spec.sha256:
            raise ModelLoadError(f"downloaded TrustMark model file {spec.name} failed its check")
        with private_temp_file(
            directory=directory, prefix=".pk-", suffix=".part", data=data
        ) as tmp:
            os.replace(tmp, directory / spec.name)
    return directory


# ---------------------------------------------------------------- backend


def _quiet_print(*args: object, **kwargs: object) -> None:
    _log.debug("trustmark: %s", " ".join(str(a) for a in args)[:200])


def _check_targets(node: Any) -> None:
    """Refuses configuration nodes that would instantiate non-TrustMark code."""
    if isinstance(node, Mapping):
        target = node.get("target")
        if target is not None and not str(target).startswith(_ALLOWED_TARGET_PREFIX):
            raise ModelLoadError("TrustMark configuration names a class outside trustmark")
        for value in node.values():
            _check_targets(value)
    elif isinstance(node, (list, tuple)):
        for value in node:
            _check_targets(value)


class TrustMarkBackend:
    """Adobe TrustMark invisible image watermark (implements ``ImageWatermarkBackend``).

    Example::

        # Once, at deployment time (network):
        download_trustmark_models("Q", model_dir="/srv/models/trustmark")
        # In the worker (offline):
        backend = TrustMarkBackend(model_dir="/srv/models/trustmark")

    Model files: the constructor reads the variant's three files (config,
    encoder, decoder; see :data:`MODEL_FILES`) from ``model_dir``, checks
    each file's size and SHA-256, and builds the models from the verified
    bytes in memory, so a file swapped after the check is never loaded.
    Checkpoints load with ``torch.load(weights_only=True)``; configuration
    files may only name classes in the ``trustmark`` package. Nothing is
    downloaded unless ``download=True`` (or
    :func:`download_trustmark_models` is called).

    Output: ``trustmark`` prints progress with ``print()``; the backend
    replaces ``print`` in the ``trustmark`` modules with a DEBUG-level log
    call (a process-wide change to those modules only), so nothing reaches
    stdout or stderr.

    Cost (measured on a CPU): the first construction in a process takes
    about 5 s, mostly importing PyTorch, TorchVision and Lightning; another
    one about 0.6 s. The weights use about as much memory as the files
    (65 MB for Q). Embedding takes about 0.15 s at 512 x 384 and 0.7 s at
    3200 x 2000 (the model runs at 256 x 256; the full-size work is
    upsampling and adding the residual); extraction about 0.2 to 0.3 s at
    any size. Construct one backend per process and reuse it.

    Thread safety: safe to share between threads; model calls are
    serialized with an internal lock.

    Geometry (from TrustMark): images whose aspect ratio exceeds 2 are marked
    in their centre square only (variant ``P``: always the centre square);
    a later crop that removes that region removes the mark.

    Args:
        variant: The model variant (default ``Q``).
        encoding: The BCH mode; it fixes :attr:`payload_bits` (default
            ``BCH_5``, 61 bits).
        device: ``"cpu"`` (default), ``"cuda"``, ``"cuda:<index>"`` or
            ``"auto"``.
        model_dir: Where the model files are; ``None`` uses
            :func:`default_model_dir`. Never inside ``site-packages``.
        download: Fetch missing or invalid files first (network).
        download_timeout_seconds: Socket timeout for that download.

    Raises:
        ValueError: If an argument is invalid or ``model_dir`` is inside
            ``site-packages``.
        BackendUnavailableError: If the ``image-watermark`` extra is missing,
            or CUDA was requested and is not available.
        ModelLoadError: If a model file is missing, fails its check or cannot
            be loaded.
    """

    def __init__(
        self,
        *,
        variant: TrustMarkVariant | str = TrustMarkVariant.Q,
        encoding: TrustMarkEncoding | str = TrustMarkEncoding.BCH_5,
        device: str = "cpu",
        model_dir: StrPath | None = None,
        download: bool = False,
        download_timeout_seconds: float = 120.0,
    ) -> None:
        """Loads and verifies the models (see the class docstring)."""
        self._variant = TrustMarkVariant(variant)
        self._encoding = TrustMarkEncoding(encoding)
        requested_device = parse_device(device)
        directory = _model_dir(model_dir)
        torch = import_optional("torch", extra=_EXTRA, feature=_FEATURE)
        tm_module = import_optional("trustmark.trustmark", extra=_EXTRA, feature=_FEATURE)
        import_optional("trustmark.model", extra=_EXTRA, feature=_FEATURE)
        datalayer = import_optional("trustmark.datalayer", extra=_EXTRA, feature=_FEATURE)
        omegaconf = import_optional("omegaconf", extra=_EXTRA, feature=_FEATURE)
        for name in _QUIET_MODULES:
            module = sys.modules.get(name)
            if module is not None:
                module.print = _quiet_print  # type: ignore[attr-defined]
        self._device = resolve_device(requested_device, torch)

        if download:
            download_trustmark_models(
                self._variant, model_dir=directory, timeout_seconds=download_timeout_seconds
            )
        config_spec, encoder_spec, decoder_spec = MODEL_FILES[self._variant]
        config_text = _read_verified(directory, config_spec).decode("utf-8")
        parts = {}
        for part, spec in (("encoder", encoder_spec), ("decoder", decoder_spec)):
            parts[part] = _build_model(
                torch=torch,
                tm_module=tm_module,
                omegaconf=omegaconf,
                config_text=config_text,
                weights=_read_verified(directory, spec),
                part=part,
                device=self._device,
            )
        encoder, decoder = parts["encoder"], parts["decoder"]

        marker = object.__new__(tm_module.TrustMark)
        marker.device = self._device
        marker.model_type = self._variant.value
        marker.use_ECC = True
        marker.secret_len = _SECRET_LEN
        marker.ecc = datalayer.DataLayer(
            _SECRET_LEN, verbose=False, encoding_mode=self._encoding.schema
        )
        marker.enctyp = self._encoding.schema
        marker.concentrate_wm_region = tm_module.CONCENTRATE_WM_REGION
        is_p = self._variant is TrustMarkVariant.P
        marker.aspect_ratio_lim = 0 if is_p else tm_module.ASPECT_RATIO_LIM
        marker.model_resolution_enc = 256
        marker.model_resolution_dec = 224 if is_p else 245
        marker.model_resolution_remove = 256
        marker.model_resolution_max_detect = 1600
        marker.encoder = encoder
        marker.decoder = decoder
        marker.removal = None
        marker.detector = None
        self._marker = marker
        self._torch = torch
        self._datalayer = datalayer
        self._lock = threading.Lock()
        _log.debug(
            "TrustMark %s loaded on %s (%s)",
            self._variant.value,
            self._device,
            self._encoding.value,
        )

    # ------------------------------------------------------------ properties

    @property
    def payload_bits(self) -> int:
        """Payload capacity of the configured encoding."""
        return self._encoding.payload_bits

    @property
    def variant(self) -> TrustMarkVariant:
        """The model variant."""
        return self._variant

    @property
    def encoding(self) -> TrustMarkEncoding:
        """The BCH mode."""
        return self._encoding

    @property
    def device(self) -> str:
        """The resolved device (``"cpu"`` or ``"cuda[:N]"``)."""
        return self._device

    # ------------------------------------------------------------ operations

    def embed(self, image: Image.Image, bits: str, *, strength: float) -> Image.Image:
        """Embeds payload bits.

        Args:
            image: The image (any mode; it is converted to RGB, and an alpha
                channel is copied to the output unchanged).
            bits: Exactly :attr:`payload_bits` ``"0"``/``"1"`` characters.
            strength: TrustMark's residual multiplier (``WM_STRENGTH``),
                greater than 0 and at most 10; 1.0 is TrustMark's default.

        Returns:
            The watermarked image, same size, ``RGB`` (``RGBA`` when the
            input had alpha).

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If the bits or the strength are invalid, or the image
                is empty.
            WatermarkBackendError: If the model fails.
        """
        _check_image(image)
        _check_bits(bits, self.payload_bits)
        level = _check_strength(strength)
        rgb = _modes.to_rgb(image)
        try:
            with self._lock:
                marked = self._marker.encode(rgb, bits, MODE="binary", WM_STRENGTH=level)
        except MemoryError:
            raise
        except Exception as exc:
            raise WatermarkBackendError("TrustMark embedding failed") from exc
        result: Image.Image = marked if marked.mode == "RGB" else marked.convert("RGB")
        if _modes.has_alpha(image):
            result.putalpha(_modes.to_rgba(image).getchannel("A"))
        return result

    def extract(self, image: Image.Image) -> TrustMarkDetection:
        """Looks for a mark in the configured encoding.

        Args:
            image: The image (any mode; converted to RGB).

        Returns:
            The detection; ``found=False`` when there is no decodable mark
            in this backend's encoding.

        Raises:
            TypeError: If ``image`` is not a Pillow image.
            ValueError: If the image is empty.
            WatermarkBackendError: If the model fails.
        """
        _check_image(image)
        rgb = _modes.to_rgb(image)
        marker = self._marker
        try:
            with self._lock, self._torch.inference_mode():
                region = marker.get_the_image_for_processing(rgb)
                size = (marker.model_resolution_dec, marker.model_resolution_dec)
                small = region.resize(size, Image.Resampling.BILINEAR)
                # torchvision's ToTensor for 8-bit RGB: float32 / 255, channels first.
                pixels = np.asarray(small, dtype=np.float32) / np.float32(255.0)
                tensor = self._torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0)
                tensor = tensor.to(self._device) * 2.0 - 1.0
                raw = (marker.decoder.decoder(tensor) > 0).cpu().numpy()
            decoded, detected, schema = marker.ecc.decode_bitstream(raw, MODE="binary")[0]
        except MemoryError:
            raise
        except Exception as exc:
            raise WatermarkBackendError("TrustMark extraction failed") from exc
        if not detected or schema != self._encoding.schema:
            return TrustMarkDetection(found=False)
        expected = self._datalayer.DataLayer(
            _SECRET_LEN, verbose=False, encoding_mode=schema
        ).encode_binary([decoded])[0]
        corrected = int(np.count_nonzero(raw[0][:_ECC_BITS] != expected[:_ECC_BITS].astype(bool)))
        return TrustMarkDetection(
            found=True,
            bits=str(decoded),
            encoding=_BY_SCHEMA[schema],
            corrected_bits=corrected,
        )


def _build_model(
    *,
    torch: Any,
    tm_module: Any,
    omegaconf: Any,
    config_text: str,
    weights: bytes,
    part: str,
    device: str,
) -> Any:
    try:
        config = omegaconf.OmegaConf.create(config_text).model
        _check_targets(omegaconf.OmegaConf.to_container(config))
        arch = config.params.secret_decoder_config.params.arch
        if arch not in _ALLOWED_DECODER_ARCHS:
            raise ModelLoadError("TrustMark configuration names an unexpected decoder")
        # As trustmark's own loader does: keep only the part in use.
        for key in ("discriminator_config", "loss_config", "noise_config"):
            config.params[key].target = "trustmark.model.Identity"
        unused = "secret_decoder_config" if part == "encoder" else "secret_encoder_config"
        config.params[unused].target = "trustmark.model.Identity"
        model = tm_module.instantiate_from_config(config)
        state = torch.load(io.BytesIO(weights), map_location="cpu", weights_only=True)
        if isinstance(state, Mapping) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
    except ModelLoadError:
        raise
    except MemoryError:
        raise
    except Exception as exc:
        raise ModelLoadError(f"TrustMark {part} could not be loaded") from exc
    if missing or unexpected:
        raise ModelLoadError(f"TrustMark {part} weights do not match the model")
    model.requires_grad_(False)
    return model.to(device).eval()


def _check_image(image: object) -> None:
    if not isinstance(image, Image.Image):
        raise TypeError(f"image must be a PIL.Image.Image, got {type(image).__name__}")
    if image.width < 1 or image.height < 1:
        raise ValueError("cannot watermark an empty image")


def _check_bits(bits: object, length: int) -> None:
    if not isinstance(bits, str):
        raise TypeError("bits must be a str of '0' and '1'")
    if len(bits) != length or set(bits) - {"0", "1"}:
        raise ValueError(f"bits must be exactly {length} '0'/'1' characters")


def _check_strength(strength: object) -> float:
    if isinstance(strength, bool) or not isinstance(strength, (int, float)):
        raise TypeError("strength must be a number")
    value = float(strength)
    if not math.isfinite(value) or not 0.0 < value <= _MAX_STRENGTH:
        raise ValueError(f"strength must be greater than 0 and at most {_MAX_STRENGTH:g}")
    return value
