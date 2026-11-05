"""C2PA manifests: sign them into media with a pluggable signer, read and validate them.

Backed by c2pa-python 0.38 (bindings to the c2pa-rs 0.91 native library,
C2PA claim version 2); install the ``c2pa`` extra. Importing this module does
not import c2pa-python: it loads on the first call that needs it.

Signing::

    from provenance_kit.provenance.c2pa import (
        ActionType,
        DigitalSourceType,
        GeneratorInfo,
        ManifestAction,
        ManifestSpec,
        TrainingMiningPolicy,
        sign_asset,
    )

    spec = ManifestSpec(
        claim_generator=GeneratorInfo("my-pipeline", "1.4.0"),
        actions=(
            ManifestAction(
                action=ActionType.CREATED, digital_source_type=DigitalSourceType.DIGITAL_CAPTURE
            ),
            ManifestAction(action=ActionType.PUBLISHED),
        ),
        authors=("Jane Doe",),
        training_mining=TrainingMiningPolicy.not_allowed(),
    )
    signed = sign_asset(jpeg_bytes, "image/jpeg", spec, signer)

``signer`` is any :class:`C2paSigner`: the private key can stay in a KMS or
HSM; :class:`LocalKeySigner` holds a PEM key in memory for development.

Reading untrusted media::

    report = read_manifest(upload_bytes, trust_anchors_pem=c2pa_trust_list)
    if report.state is ValidationState.TRUSTED and report.training_mining_not_allowed:
        ...

Security: every call disables remote manifest fetching, OCSP and all other
network access (except the TSA a caller names when signing), decodes no
pixels, enforces :class:`~provenance_kit.limits.MediaLimits` on the encoded
size, and detects the container from the bytes. ``read_manifest`` never
raises because a manifest is missing or invalid; see its documentation.

Thread safety: the functions are thread-safe; each call creates its own
c2pa-python objects. A signer shared between threads must be thread-safe.

See ``docs/provenance.md`` for the assertion labels, the specifications they
follow and the signature encodings.
"""

from __future__ import annotations

from provenance_kit.provenance.c2pa._api import read_manifest, sign_asset
from provenance_kit.provenance.c2pa._json import JsonValue
from provenance_kit.provenance.c2pa._manifest import (
    ActionType,
    DigitalSourceType,
    GeneratorInfo,
    Ingredient,
    IngredientRelationship,
    ManifestAction,
    ManifestSpec,
    SoftBinding,
    SoftBindingAlgorithm,
    TrainingMiningEntry,
    TrainingMiningKey,
    TrainingMiningPolicy,
    TrainingMiningUse,
)
from provenance_kit.provenance.c2pa._media import SUPPORTED_MEDIA_TYPES
from provenance_kit.provenance.c2pa._report import (
    MANIFEST_UNREADABLE,
    IngredientReport,
    ManifestAssertion,
    ManifestReport,
    SignerCertificateInfo,
    SoftBindingReport,
    ValidationState,
    ValidationStatus,
)
from provenance_kit.provenance.c2pa._signer import (
    C2paSigner,
    LocalKeySigner,
    SignatureFormat,
    SigningAlgorithm,
    SigningError,
    ecdsa_der_to_raw,
)

__all__ = [
    "MANIFEST_UNREADABLE",
    "SUPPORTED_MEDIA_TYPES",
    "ActionType",
    "C2paSigner",
    "DigitalSourceType",
    "GeneratorInfo",
    "Ingredient",
    "IngredientRelationship",
    "IngredientReport",
    "JsonValue",
    "LocalKeySigner",
    "ManifestAction",
    "ManifestAssertion",
    "ManifestReport",
    "ManifestSpec",
    "SignatureFormat",
    "SignerCertificateInfo",
    "SigningAlgorithm",
    "SigningError",
    "SoftBinding",
    "SoftBindingAlgorithm",
    "SoftBindingReport",
    "TrainingMiningEntry",
    "TrainingMiningKey",
    "TrainingMiningPolicy",
    "TrainingMiningUse",
    "ValidationState",
    "ValidationStatus",
    "ecdsa_der_to_raw",
    "read_manifest",
    "sign_asset",
]
