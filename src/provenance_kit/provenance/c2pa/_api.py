"""``sign_asset`` and ``read_manifest`` (internal module; import from the package)."""

from __future__ import annotations

from provenance_kit._logging import get_logger
from provenance_kit._typing import BytesLike
from provenance_kit.limits import MediaLimits
from provenance_kit.mediatype import MediaType
from provenance_kit.provenance.c2pa._manifest import ManifestSpec, build_definition
from provenance_kit.provenance.c2pa._media import resolve_media_type
from provenance_kit.provenance.c2pa._native import (
    SignatureBridge,
    read_settings,
    read_store,
    sign_with_c2pa,
    tsa_host_pattern,
)
from provenance_kit.provenance.c2pa._report import (
    ManifestReport,
    parse_manifest_store,
    unreadable_report,
    unsigned_report,
)
from provenance_kit.provenance.c2pa._signer import C2paSigner

__all__ = ["read_manifest", "sign_asset"]

_log = get_logger(__name__)

_PEM_CERTIFICATE = "-----BEGIN CERTIFICATE-----"


def sign_asset(
    data: BytesLike,
    mime_type: str | MediaType,
    spec: ManifestSpec,
    signer: C2paSigner,
    *,
    tsa_url: str | None = None,
    limits: MediaLimits | None = None,
) -> bytes:
    """Embeds a signed C2PA manifest in an asset.

    The asset travels through in-memory streams only. The C2PA library parses
    the container to embed the manifest but decodes no pixels or samples
    (thumbnails are disabled), validates the new manifest right after signing,
    and makes no network request unless ``tsa_url`` is given.

    Args:
        data: The encoded asset (one of :data:`SUPPORTED_MEDIA_TYPES`).
        mime_type: The asset's MIME type. It must agree with the container
            detected from the bytes (any ISO BMFF brand is accepted for
            another; the bytes decide).
        spec: What to put in the manifest.
        signer: Signs the claim; the private key may live in a KMS or HSM.
        tsa_url: An RFC 3161 time-stamp authority to counter-sign the claim
            signature, for example a commercial TSA's URL. ``None`` (the
            default) adds no time-stamp and keeps the call offline; the
            manifest is then valid only while the signing certificate is.
            When given, the library contacts exactly this host (redirects
            disabled) and sends it a hash of the signature.
        limits: Size limits for the asset and the ingredients; ``None`` means
            ``MediaLimits()``.

    Returns:
        The asset with the manifest embedded.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``tsa_url`` is malformed or the signer's properties are
            unusable: a certificate chain that is not strictly PEM blocks
            labelled ``CERTIFICATE``, or (checked once the library is loaded)
            a block that is not a certificate or a leaf whose extensions
            cannot be parsed or whose key usage lacks ``digitalSignature``.
        BackendUnavailableError: If c2pa-python is not installed
            (extra ``c2pa``).
        MediaLimitExceededError: If the asset or an ingredient exceeds
            ``limits.max_bytes``.
        UnsupportedMediaError: If a media type is unrecognized, unsupported
            or contradicts the bytes.
        MediaDecodeError: If the asset or an ingredient cannot be parsed.
        SigningError: If the signer, the TSA or the C2PA library fails (the
            library refuses manifests that break the C2PA rules).
    """
    asset = _bytes("data", data)
    if not isinstance(spec, ManifestSpec):
        raise TypeError("spec must be a ManifestSpec")
    limits = _limits(limits)
    if tsa_url is not None:
        tsa_host_pattern(tsa_url)  # validates before any work
    bridge = SignatureBridge(signer)
    limits.check_bytes(len(asset))
    for ingredient in spec.ingredients:
        limits.check_bytes(len(ingredient.data))
    media_type = resolve_media_type(asset, mime_type)
    ingredient_types = tuple(
        resolve_media_type(ingredient.data, ingredient.mime_type) for ingredient in spec.ingredients
    )
    definition, ingredients = build_definition(spec, media_type, ingredient_types)
    signed = sign_with_c2pa(
        data=asset,
        format_mime=media_type.mime,
        definition=definition,
        ingredients=ingredients,
        bridge=bridge,
        tsa_url=tsa_url,
    )
    _log.debug(
        "signed a C2PA manifest (%s, %d -> %d bytes, %d ingredients)",
        media_type.mime,
        len(asset),
        len(signed),
        len(ingredients),
    )
    return signed


def read_manifest(
    data: BytesLike,
    mime_type: str | MediaType | None = None,
    *,
    manifest_data: BytesLike | None = None,
    trust_anchors_pem: str | None = None,
    tsa_trust_anchors_pem: str | None = None,
    limits: MediaLimits | None = None,
) -> ManifestReport:
    """Reads and validates the C2PA manifest of an untrusted asset.

    An asset without a manifest is a normal result
    (:attr:`ValidationState.UNSIGNED`). A manifest that fails validation, or
    that the library cannot parse at all, gives
    :attr:`ValidationState.INVALID` with the reasons in
    :attr:`ManifestReport.errors`. No network request is made: remote
    manifests, OCSP and ``did:web`` lookups are disabled.

    Args:
        data: The encoded asset (untrusted).
        mime_type: The declared MIME type, or ``None`` to use the type
            detected from the bytes. A declared type must agree with the
            detected container.
        manifest_data: A detached manifest store (a ``.c2pa`` sidecar, or a
            remote manifest the caller fetched) to validate against the
            asset instead of an embedded one.
        trust_anchors_pem: Root certificates (PEM) that signing certificates
            must chain to for :attr:`ValidationState.TRUSTED`, for example the
            C2PA trust list. ``None`` skips trust evaluation: valid manifests
            are reported as :attr:`ValidationState.VALID`.
        tsa_trust_anchors_pem: Root certificates (PEM) for time-stamp
            authorities. ``None`` validates time-stamp tokens without
            evaluating the TSA's trust.
        limits: Size limits; ``None`` means ``MediaLimits()``. They apply to
            ``data`` and ``manifest_data``. The library also caps a
            decompressed manifest at 32 MiB.

    Returns:
        The validation state, the active manifest's contents and the
        validation errors.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If a trust anchor bundle contains no PEM certificate.
        BackendUnavailableError: If c2pa-python is not installed
            (extra ``c2pa``).
        MediaLimitExceededError: If an input exceeds ``limits.max_bytes``.
        UnsupportedMediaError: If the content is not a supported media type
            or contradicts ``mime_type``.
        MediaDecodeError: If the media container cannot be parsed.
    """
    asset = _bytes("data", data)
    store = None if manifest_data is None else _bytes("manifest_data", manifest_data)
    for name, bundle in (
        ("trust_anchors_pem", trust_anchors_pem),
        ("tsa_trust_anchors_pem", tsa_trust_anchors_pem),
    ):
        if bundle is not None:
            if not isinstance(bundle, str):
                raise TypeError(f"{name} must be a str")
            if _PEM_CERTIFICATE not in bundle:
                raise ValueError(f"{name} must contain PEM certificates")
    limits = _limits(limits)
    limits.check_bytes(len(asset))
    if store is not None:
        limits.check_bytes(len(store))
    media_type = resolve_media_type(asset, mime_type)
    result = read_store(
        data=asset,
        format_mime=media_type.mime,
        manifest_data=store,
        settings=read_settings(
            trust_anchors_pem=trust_anchors_pem,
            tsa_trust_anchors_pem=tsa_trust_anchors_pem,
        ),
    )
    if result.store_json is not None:
        return parse_manifest_store(result.store_json)
    if result.unreadable_reason is not None:
        return unreadable_report(result.unreadable_reason)
    return unsigned_report(result.remote_url)


def _bytes(name: str, value: object) -> bytes:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError(f"{name} must be bytes-like, got {type(value).__name__}")
    return bytes(memoryview(value).cast("B"))


def _limits(limits: MediaLimits | None) -> MediaLimits:
    if limits is None:
        return MediaLimits()
    if not isinstance(limits, MediaLimits):
        raise TypeError("limits must be a MediaLimits or None")
    return limits
