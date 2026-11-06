"""The adapter to c2pa-python (internal).

Everything that touches the ``c2pa`` package is here: the per-call settings
(network off, no thumbnails, explicit trust), the bridge between a
:class:`C2paSigner` and c2pa-python's callback signer, and the translation of
the library's exceptions into provenance-kit errors.

Settings applied to every call (c2pa-rs 0.91 names):

* ``verify.remote_manifest_fetch = false``: the library would otherwise fetch a
  manifest URL found inside an untrusted asset (server-side request forgery).
* ``core.allowed_network_hosts = []`` (or only the TSA host when signing with
  a TSA) and ``core.allow_redirects = false``: blocks every other request,
  including OCSP and ``did:web`` resolution.
* ``verify.ocsp_fetch = false``.
* ``builder.thumbnail.enabled = false``: no pixels are decoded, so image
  decompression bombs cannot reach the library's decoders.
* ``verify.verify_trust`` only when trust anchors are given.
"""

from __future__ import annotations

import io
import re
import urllib.parse
from dataclasses import dataclass
from types import ModuleType
from typing import Any

from provenance_kit._logging import get_logger
from provenance_kit._optional import import_optional
from provenance_kit.errors import MediaDecodeError, UnsupportedMediaError
from provenance_kit.provenance.c2pa._json import JsonObject, JsonValue
from provenance_kit.provenance.c2pa._manifest import IngredientSource
from provenance_kit.provenance.c2pa._signer import (
    C2paSigner,
    SignatureFormat,
    SigningAlgorithm,
    SigningError,
    check_signer,
    check_signing_certificate,
    normalize_signature,
)

__all__ = [
    "NativeReadResult",
    "SignatureBridge",
    "load_c2pa",
    "read_settings",
    "read_store",
    "sign_settings",
    "sign_with_c2pa",
    "tsa_host_pattern",
]

_log = get_logger(__name__)

_FEATURE = "C2PA manifests"
_ERROR_KIND = re.compile(r"^([A-Za-z]+)(?::|\s|$)")
_STATUS_CODE = re.compile(r"\b([a-z][A-Za-z0-9]*(?:\.[A-Za-z0-9_]+)+):")
_REMOTE_URL = re.compile(r"must fetch remote manifests from url (\S+)\s*$")
_REASONS = frozenset(
    {
        "Assertion",
        "AssertionNotFound",
        "Decoding",
        "Encoding",
        "Io",
        "Json",
        "Manifest",
        "ManifestNotFound",
        "NotSupported",
        "Other",
        "Remote",
        "RemoteManifest",
        "ResourceNotFound",
        "Signature",
        "Verify",
    }
)


def load_c2pa() -> ModuleType:
    """Imports c2pa-python.

    Raises:
        BackendUnavailableError: If it is not installed (extra ``c2pa``).
    """
    return import_optional("c2pa", extra="c2pa", feature=_FEATURE)


def tsa_host_pattern(tsa_url: str) -> str:
    """Validates a TSA URL and returns the host pattern that allows only it.

    Args:
        tsa_url: An ``http`` or ``https`` URL of an RFC 3161 time-stamp
            authority.

    Returns:
        ``scheme://host[:port]``, the c2pa-rs ``allowed_network_hosts`` entry.

    Raises:
        TypeError: If ``tsa_url`` is not a str.
        ValueError: If it is not an absolute http(s) URL with a host name or
            an IPv4 address, or it contains credentials or a fragment.
    """
    if not isinstance(tsa_url, str):
        raise TypeError("tsa_url must be a str")
    parts = urllib.parse.urlsplit(tsa_url)
    if parts.scheme not in ("http", "https"):
        raise ValueError("tsa_url must be an http or https URL")
    if parts.username is not None or parts.password is not None or parts.fragment:
        raise ValueError("tsa_url must not contain credentials or a fragment")
    host = parts.hostname
    if not host or ":" in host:
        raise ValueError("tsa_url needs a host name or an IPv4 address")
    try:
        port = parts.port
    except ValueError:
        raise ValueError("tsa_url has an invalid port") from None
    return f"{parts.scheme}://{host}" + (f":{port}" if port is not None else "")


def sign_settings(*, tsa_url: str | None) -> JsonObject:
    """Returns the c2pa-rs settings for signing."""
    hosts: list[JsonValue] = [tsa_host_pattern(tsa_url)] if tsa_url else []
    return {
        "verify": {
            "remote_manifest_fetch": False,
            "ocsp_fetch": False,
            "verify_trust": False,
            "verify_timestamp_trust": False,
            "verify_after_sign": True,
        },
        "core": {"allowed_network_hosts": hosts, "allow_redirects": False},
        "builder": {"thumbnail": {"enabled": False}},
    }


def read_settings(
    *,
    trust_anchors_pem: str | None,
    tsa_trust_anchors_pem: str | None,
) -> JsonObject:
    """Returns the c2pa-rs settings for reading untrusted assets."""
    settings: JsonObject = {
        "verify": {
            "remote_manifest_fetch": False,
            "ocsp_fetch": False,
            "verify_trust": trust_anchors_pem is not None,
            "verify_timestamp_trust": tsa_trust_anchors_pem is not None,
        },
        "core": {"allowed_network_hosts": [], "allow_redirects": False},
        "builder": {"thumbnail": {"enabled": False}},
    }
    anchors: list[JsonValue] = []
    if trust_anchors_pem is not None:
        anchors.append({"trust_anchors": trust_anchors_pem, "trust_kind": "manifest"})
    if tsa_trust_anchors_pem is not None:
        anchors.append({"trust_anchors": tsa_trust_anchors_pem, "trust_kind": "tsa"})
    if anchors:
        settings["trust"] = {"anchors": anchors}
    return settings


class SignatureBridge:
    """Adapts a :class:`C2paSigner` to c2pa-python's signing callback.

    The native library calls the bridge from inside ``Builder.sign``. An
    exception cannot cross the native frames, and c2pa-python logs the text
    of any exception that reaches its own wrapper. The bridge therefore
    catches everything, records it in :attr:`error` and returns an empty
    signature, which makes the native call fail; the caller then raises the
    recorded error. Signatures are checked and DER ECDSA is converted to
    ``r || s`` here.

    Attributes:
        algorithm: The signer's algorithm.
        certificate_chain_pem: The signer's chain.
        signature_format: The signer's declared format.
        error: The first exception raised while signing, if any.
        calls: How many times the native code asked for a signature.
    """

    def __init__(self, signer: C2paSigner) -> None:
        """Validates the signer's properties.

        Raises:
            TypeError: If a signer property has the wrong type.
            ValueError: If the chain or the format is unusable.
        """
        self._signer = signer
        algorithm, chain, signature_format = check_signer(signer)
        self.algorithm: SigningAlgorithm = algorithm
        self.certificate_chain_pem: str = chain
        self.signature_format: SignatureFormat = signature_format
        self.error: BaseException | None = None
        self.calls = 0

    def __call__(self, data: bytes) -> bytes:
        """Signs ``data`` for the native library; never raises."""
        self.calls += 1
        if self.error is not None:
            return b""
        try:
            signature = self._signer.sign(bytes(data))
            return normalize_signature(signature, self.algorithm, self.signature_format)
        except BaseException as exc:  # re-raised by raise_error() once native code returns
            self.error = exc
            return b""

    def raise_error(self) -> None:
        """Raises the recorded error, if any, as a provenance-kit error.

        Raises:
            SigningError: For an exception raised by the signer or a rejected
                signature.
            BaseException: ``KeyboardInterrupt``, ``SystemExit`` and
                ``MemoryError`` are re-raised unchanged.
        """
        error = self.error
        if error is None:
            return
        if isinstance(error, (KeyboardInterrupt, SystemExit, MemoryError)):
            raise error
        if isinstance(error, SigningError):
            raise error
        raise SigningError(f"the C2PA signer raised {type(error).__name__}") from error


def sign_with_c2pa(
    *,
    data: bytes,
    format_mime: str,
    definition: JsonObject,
    ingredients: tuple[IngredientSource, ...],
    bridge: SignatureBridge,
    tsa_url: str | None,
) -> bytes:
    """Signs an asset with c2pa-python and returns the signed bytes.

    Raises:
        BackendUnavailableError: If c2pa-python is not installed.
        ValueError: If the signer's chain is not strictly PEM certificates,
            or its leaf certificate lacks the ``digitalSignature`` key usage
            (checked here, because c2pa-rs 0.91 accepts a
            ``nonRepudiation``-only leaf). The library receives the chain
            re-encoded from the certificates checked, never the signer's
            text, so that both see the same leaf.
        MediaDecodeError: If the asset or an ingredient cannot be parsed.
        UnsupportedMediaError: If the library refuses the media type.
        SigningError: For every other failure (see :class:`SigningError`).
    """
    settings = sign_settings(tsa_url=tsa_url)
    c2pa = load_c2pa()
    chain = check_signing_certificate(bridge.certificate_chain_pem)
    algorithm = getattr(c2pa.C2paSigningAlg, bridge.algorithm.name)
    resources: list[Any] = []
    try:
        native_signer = c2pa.Signer.from_callback(bridge, algorithm, chain, tsa_url)
        resources.append(native_signer)
        context = c2pa.Context.from_dict(settings)
        resources.append(context)
        builder = c2pa.Builder(definition, context)
        resources.append(builder)
        for ingredient in ingredients:
            builder.add_ingredient_from_stream(
                ingredient.definition, ingredient.mime_type, io.BytesIO(ingredient.data)
            )
        destination = io.BytesIO()
        builder.sign(native_signer, format_mime, io.BytesIO(data), destination)
    except c2pa.C2paError as exc:
        bridge.raise_error()
        raise _signing_failure(str(exc)) from exc
    except Exception as exc:
        bridge.raise_error()
        raise SigningError(f"C2PA signing failed ({type(exc).__name__})") from exc
    finally:
        for resource in reversed(resources):
            _close(resource)
    bridge.raise_error()
    if bridge.calls == 0:
        raise SigningError("the C2PA library returned without requesting a signature")
    return destination.getvalue()


@dataclass(frozen=True, slots=True)
class NativeReadResult:
    """What c2pa-python returned for one asset (internal).

    Attributes:
        store_json: The manifest store report, or ``None``.
        remote_url: The remote manifest URL when the asset only references
            one (untrusted).
        unreadable_reason: The error category when a manifest store was found
            but could not be processed.
    """

    store_json: str | None = None
    remote_url: str | None = None
    unreadable_reason: str | None = None


def read_store(
    *,
    data: bytes,
    format_mime: str,
    manifest_data: bytes | None,
    settings: JsonObject,
) -> NativeReadResult:
    """Reads and validates the manifest store of an asset.

    Raises:
        BackendUnavailableError: If c2pa-python is not installed.
        MediaDecodeError: If the media container cannot be parsed.
        UnsupportedMediaError: If the library does not support the media.
    """
    c2pa = load_c2pa()
    context = c2pa.Context.from_dict(settings)
    reader: Any = None
    try:
        try:
            reader = c2pa.Reader(format_mime, io.BytesIO(data), manifest_data, context=context)
        except c2pa.C2paError.ManifestNotFound:
            return NativeReadResult()
        except c2pa.C2paError as exc:
            return _read_failure(str(exc), exc)
        except Exception as exc:
            raise MediaDecodeError(f"the C2PA reader failed ({type(exc).__name__})") from exc
        try:
            return NativeReadResult(store_json=reader.json())
        except c2pa.C2paError as exc:
            return _read_failure(str(exc), exc)
        except Exception as exc:
            raise MediaDecodeError(f"the C2PA reader failed ({type(exc).__name__})") from exc
    finally:
        if reader is not None:
            _close(reader)
        _close(context)


def _read_failure(message: str, exc: BaseException) -> NativeReadResult:
    kind = _error_kind(message)
    _log.debug("C2PA reader error (%s)", kind)
    if kind in ("Remote", "RemoteManifest"):
        match = _REMOTE_URL.search(message)
        return NativeReadResult(remote_url=match.group(1) if match else None)
    if "asset could not be parsed" in message or kind == "Io":
        raise MediaDecodeError("the media container could not be parsed") from exc
    if kind == "NotSupported":
        raise UnsupportedMediaError("the C2PA library does not support this media") from exc
    return NativeReadResult(unreadable_reason=kind)


def _signing_failure(message: str) -> Exception:
    kind = _error_kind(message)
    _log.debug("C2PA signing error (%s)", kind)
    if "asset could not be parsed" in message or kind == "Io":
        return MediaDecodeError("the asset or an ingredient could not be parsed")
    if kind == "NotSupported":
        return UnsupportedMediaError("the C2PA library does not support this media")
    if kind == "Verify":
        codes = sorted(set(_STATUS_CODE.findall(message)))
        detail = ", ".join(codes[:10]) if codes else "validation failed"
        return SigningError(f"the C2PA library rejected the manifest ({detail})")
    return SigningError(f"C2PA signing failed ({kind})")


def _error_kind(message: str) -> str:
    """Returns the library's error category (``Verify``, ``Signature``, ...)."""
    match = _ERROR_KIND.match(message)
    kind = match.group(1) if match else ""
    return kind if kind in _REASONS else "Other"


def _close(resource: Any) -> None:
    try:
        resource.close()
    except Exception:  # cleanup must not mask the result
        _log.debug("closing a C2PA object failed")
