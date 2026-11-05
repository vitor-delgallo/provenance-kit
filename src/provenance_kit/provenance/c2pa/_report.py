"""The result of reading a C2PA manifest, and its parser (internal module).

The parser reads the JSON report that c2pa-python's ``Reader.json()``
produces. That JSON describes an untrusted manifest: every field is checked
for its type, and anything unexpected is skipped rather than trusted.
"""

from __future__ import annotations

import enum
import json
import re
from dataclasses import dataclass

from provenance_kit._logging import get_logger
from provenance_kit.provenance.c2pa._json import JsonValue
from provenance_kit.provenance.c2pa._manifest import (
    CAWG_METADATA_LABEL,
    SOFT_BINDING_LABEL,
    TRAINING_MINING_LABEL,
    GeneratorInfo,
    TrainingMiningEntry,
    TrainingMiningPolicy,
    TrainingMiningUse,
)

__all__ = [
    "MANIFEST_UNREADABLE",
    "IngredientReport",
    "ManifestAssertion",
    "ManifestReport",
    "SignerCertificateInfo",
    "SoftBindingReport",
    "ValidationState",
    "ValidationStatus",
    "parse_manifest_store",
]

_log = get_logger(__name__)

#: The validation status code that :func:`read_manifest` reports, with state
#: :attr:`ValidationState.INVALID`, when the C2PA library found a manifest
#: store but could not parse or verify it at all. It is namespaced so that it
#: cannot collide with a C2PA status code.
MANIFEST_UNREADABLE = "provenance_kit.manifest.unreadable"

_LEGACY_TRAINING_MINING_LABEL = "c2pa.training-mining"
_LEGACY_CREATIVE_WORK_LABEL = "stds.schema-org.CreativeWork"
_INSTANCE_SUFFIX = re.compile(r"__\d+$")
_VERSION_SUFFIX = re.compile(r"\.v\d+$")
_MAX_LIST = 10_000  # bounds what is copied out of one untrusted list


class ValidationState(enum.StrEnum):
    """The overall result of validating an asset's active manifest.

    The C2PA 2.x states (C2PA 2.3, section 14.3) are ``Invalid``, ``Valid``
    and ``Trusted``; this enum adds ``UNSIGNED`` for assets without a
    manifest.
    """

    UNSIGNED = "unsigned"
    """No manifest was found (or only a remote reference that was not fetched)."""
    INVALID = "invalid"
    """A manifest is present but failed validation; see the errors."""
    VALID = "valid"
    """Well-formed, hashes match and the signature verifies. The signer is
    not known to be trusted: no trust anchors were given, or the chain does not
    lead to one (then the errors include ``signingCredential.untrusted``)."""
    TRUSTED = "trusted"
    """Valid, and the signing certificate chains to a given trust anchor."""

    @property
    def is_valid(self) -> bool:
        """Whether the state is :attr:`VALID` or :attr:`TRUSTED`."""
        return self in (ValidationState.VALID, ValidationState.TRUSTED)


@dataclass(frozen=True, slots=True)
class ValidationStatus:
    """One validation status entry (C2PA 2.3, section 15.2).

    Attributes:
        code: The status code, for example ``assertion.dataHash.mismatch``.
        url: The JUMBF URI of the item the code is about, if given.
        explanation: The library's human-readable explanation, if given.
            **Untrusted**: it may quote labels or URIs from the manifest.
    """

    code: str
    url: str | None = None
    explanation: str | None = None


@dataclass(frozen=True, slots=True)
class ManifestAssertion:
    """An assertion of the active manifest, as the C2PA library decoded it.

    Attributes:
        label: The label as stored, including any version (``.v2``) or
            instance (``__1``) suffix.
        data: The decoded content (JSON-compatible values). Byte strings
            appear as base64 text. **Untrusted.**
        kind: ``"Json"`` for assertions stored as JSON boxes, else ``None``
            (CBOR).
    """

    label: str
    data: JsonValue
    kind: str | None = None

    @property
    def base_label(self) -> str:
        """The label without instance and version suffixes."""
        return _base_label(self.label)


@dataclass(frozen=True, slots=True)
class SignerCertificateInfo:
    """What the C2PA library reports about the claim signer.

    All fields are taken from the signing certificate embedded in the
    manifest. They identify the signer only as far as the validation state
    allows: they are meaningful for a :attr:`ValidationState.TRUSTED`
    manifest and merely claimed otherwise.

    Attributes:
        algorithm: The signature algorithm, for example ``"Es256"``.
        issuer: The issuer's organization name.
        common_name: The signer's common name.
        serial_number: The certificate serial number (decimal).
        signing_time: The time from the signature's time-stamp token, if the
            manifest was time-stamped (RFC 3339).
    """

    algorithm: str | None = None
    issuer: str | None = None
    common_name: str | None = None
    serial_number: str | None = None
    signing_time: str | None = None


@dataclass(frozen=True, slots=True)
class SoftBindingReport:
    """A ``c2pa.soft-binding`` assertion of the active manifest.

    Attributes:
        algorithm: The algorithm identifier (``alg``), if present.
        values: The ``value`` of each block. A text value is returned as is;
            a byte-string value appears base64-encoded, because the library's
            JSON report does not distinguish the two (see
            ``docs/provenance.md``). **Untrusted.**
    """

    algorithm: str | None
    values: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class IngredientReport:
    """An ingredient of the active manifest.

    Attributes:
        title: The ingredient's title. **Untrusted.**
        format: Its MIME type.
        relationship: ``parentOf``, ``componentOf`` or ``inputTo``.
        has_manifest: Whether the ingredient carried its own manifest.
    """

    title: str | None
    format: str | None
    relationship: str | None
    has_manifest: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class ManifestReport:
    """The result of :func:`~provenance_kit.provenance.c2pa.read_manifest`.

    Everything except :attr:`state` describes what the manifest **claims**;
    rely on it only when :attr:`state` is valid (and, to know who signed,
    :attr:`ValidationState.TRUSTED`).

    Attributes:
        state: The validation state.
        manifest_label: The active manifest's label (a URN).
        title: The claimed title.
        format: The claimed media type.
        claim_generator: The claim generator entries.
        assertions: The active manifest's assertions, decoded.
        signer: The claim signer's certificate details.
        errors: Validation failures of the active manifest (empty when
            valid, except ``signingCredential.untrusted`` with trust anchors).
        informational: Informational validation entries (for example
            ``signingCredential.ocsp.skipped``).
        authors: Creator names from ``cawg.metadata`` (``dc:creator``) and
            the deprecated ``stds.schema-org.CreativeWork`` assertion.
        training_mining: The ``cawg.training-mining`` policy (or the C2PA 1.x
            ``c2pa.training-mining`` one), if present.
        soft_bindings: The ``c2pa.soft-binding`` assertions.
        ingredients: The active manifest's ingredients.
        remote_manifest_url: When the asset only references a remote
            manifest, its URL. **Untrusted**: it is never fetched; fetch it
            yourself (with SSRF protections) and pass it as
            ``manifest_data`` if you need it.
        manifest_count: How many manifests the store holds.
    """

    state: ValidationState
    manifest_label: str | None = None
    title: str | None = None
    format: str | None = None
    claim_generator: tuple[GeneratorInfo, ...] = ()
    assertions: tuple[ManifestAssertion, ...] = ()
    signer: SignerCertificateInfo | None = None
    errors: tuple[ValidationStatus, ...] = ()
    informational: tuple[ValidationStatus, ...] = ()
    authors: tuple[str, ...] = ()
    training_mining: TrainingMiningPolicy | None = None
    soft_bindings: tuple[SoftBindingReport, ...] = ()
    ingredients: tuple[IngredientReport, ...] = ()
    remote_manifest_url: str | None = None
    manifest_count: int = 0

    @property
    def has_manifest(self) -> bool:
        """Whether a manifest was found (valid or not)."""
        return self.state is not ValidationState.UNSIGNED

    @property
    def training_mining_not_allowed(self) -> bool:
        """Whether the manifest declares all four standard uses ``notAllowed``.

        The four uses are AI training, generative AI training, data mining
        and AI inference. ``False`` when there is no training-mining
        assertion. Like every claim in the report, it binds the asset only if
        :attr:`state` is valid.
        """
        return self.training_mining is not None and self.training_mining.all_not_allowed

    def find_assertions(self, base_label: str) -> tuple[ManifestAssertion, ...]:
        """Returns the assertions with a base label, for example ``"c2pa.actions"``.

        Args:
            base_label: The label without version or instance suffix.

        Returns:
            The matching assertions, in manifest order.
        """
        return tuple(a for a in self.assertions if a.base_label == base_label)


def unsigned_report(remote_manifest_url: str | None = None) -> ManifestReport:
    """Returns the report for an asset without a manifest (internal)."""
    return ManifestReport(state=ValidationState.UNSIGNED, remote_manifest_url=remote_manifest_url)


def unreadable_report(reason: str) -> ManifestReport:
    """Returns the report for a manifest store that could not be processed (internal).

    Args:
        reason: A short, safe description (an error category, never input).
    """
    status = ValidationStatus(
        code=MANIFEST_UNREADABLE,
        explanation=f"the manifest store could not be parsed or verified ({reason})",
    )
    return ManifestReport(state=ValidationState.INVALID, errors=(status,))


def parse_manifest_store(text: str) -> ManifestReport:
    """Builds a report from the JSON of ``c2pa.Reader.json()`` (internal).

    Args:
        text: The JSON text.

    Returns:
        The report. JSON that cannot be decoded, or that has no active
        manifest, gives an :attr:`ValidationState.INVALID` report.
    """
    try:
        store = json.loads(text)
    except (ValueError, RecursionError):
        return unreadable_report("malformed JSON report")
    if not isinstance(store, dict):
        return unreadable_report("malformed JSON report")
    manifests = store.get("manifests")
    manifests = manifests if isinstance(manifests, dict) else {}
    if not manifests:
        return unsigned_report()
    label = _text(store.get("active_manifest"))
    active = manifests.get(label) if label is not None else None
    if not isinstance(active, dict):
        report = unreadable_report("no active manifest")
        return ManifestReport(
            state=report.state, errors=report.errors, manifest_count=len(manifests)
        )

    results = store.get("validation_results")
    active_results = results.get("activeManifest") if isinstance(results, dict) else None
    active_results = active_results if isinstance(active_results, dict) else {}
    errors = _statuses(active_results.get("failure"))
    if not active_results:  # older reports: a flat list of failures
        errors = _statuses(store.get("validation_status"))
    state = _state(store.get("validation_state"), errors)

    assertions = _assertions(active.get("assertions"))
    return ManifestReport(
        state=state,
        manifest_label=label,
        title=_text(active.get("title")),
        format=_text(active.get("format")),
        claim_generator=_generators(active.get("claim_generator_info")),
        assertions=assertions,
        signer=_signer(active.get("signature_info")),
        errors=errors,
        informational=_statuses(active_results.get("informational")),
        authors=_authors(assertions),
        training_mining=_training_mining(assertions),
        soft_bindings=_soft_bindings(assertions),
        ingredients=_ingredients(active.get("ingredients")),
        manifest_count=len(manifests),
    )


def _state(value: object, errors: tuple[ValidationStatus, ...]) -> ValidationState:
    states = {
        "trusted": ValidationState.TRUSTED,
        "valid": ValidationState.VALID,
        "invalid": ValidationState.INVALID,
    }
    state = states.get(value.lower()) if isinstance(value, str) else None
    if state is None:
        _log.debug("unknown C2PA validation state; reporting invalid")
        return ValidationState.INVALID  # fail closed
    if state is ValidationState.TRUSTED and errors:
        return ValidationState.VALID  # defensive: trust never coexists with failures
    return state


def _text(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _items(value: object) -> list[object]:
    return list(value[:_MAX_LIST]) if isinstance(value, list) else []


def _statuses(value: object) -> tuple[ValidationStatus, ...]:
    statuses = []
    for item in _items(value):
        if isinstance(item, dict) and isinstance(item.get("code"), str):
            statuses.append(
                ValidationStatus(
                    code=item["code"],
                    url=_text(item.get("url")),
                    explanation=_text(item.get("explanation")),
                )
            )
    return tuple(statuses)


def _assertions(value: object) -> tuple[ManifestAssertion, ...]:
    assertions = []
    for item in _items(value):
        if isinstance(item, dict) and isinstance(item.get("label"), str):
            data: JsonValue = item.get("data")
            assertions.append(
                ManifestAssertion(label=item["label"], data=data, kind=_text(item.get("kind")))
            )
    return tuple(assertions)


def _generators(value: object) -> tuple[GeneratorInfo, ...]:
    generators = []
    for item in _items(value):
        name = item.get("name") if isinstance(item, dict) else None
        if not isinstance(name, str) or not isinstance(item, dict):
            continue
        try:
            generators.append(GeneratorInfo(name, _text(item.get("version"))))
        except (TypeError, ValueError):
            continue
    return tuple(generators)


def _signer(value: object) -> SignerCertificateInfo | None:
    if not isinstance(value, dict):
        return None
    return SignerCertificateInfo(
        algorithm=_text(value.get("alg")),
        issuer=_text(value.get("issuer")),
        common_name=_text(value.get("common_name")),
        serial_number=_text(value.get("cert_serial_number")),
        signing_time=_text(value.get("time")),
    )


def _authors(assertions: tuple[ManifestAssertion, ...]) -> tuple[str, ...]:
    names: list[str] = []
    for assertion in assertions:
        data = assertion.data
        if not isinstance(data, dict):
            continue
        if assertion.base_label == CAWG_METADATA_LABEL:
            creator = data.get("dc:creator")
            if isinstance(creator, str):
                names.append(creator)
            else:
                names.extend(name for name in _items(creator) if isinstance(name, str))
        elif assertion.base_label == _LEGACY_CREATIVE_WORK_LABEL:
            for author in _items(data.get("author")):
                if isinstance(author, dict) and isinstance(author.get("name"), str):
                    names.append(author["name"])
    return tuple(name for name in names if name.strip())


def _training_mining(assertions: tuple[ManifestAssertion, ...]) -> TrainingMiningPolicy | None:
    entries: dict[str, TrainingMiningEntry] = {}
    for assertion in assertions:
        if assertion.base_label not in (TRAINING_MINING_LABEL, _LEGACY_TRAINING_MINING_LABEL):
            continue
        data = assertion.data
        raw_entries = data.get("entries") if isinstance(data, dict) else None
        if not isinstance(raw_entries, dict):
            continue
        for key, value in list(raw_entries.items())[:_MAX_LIST]:
            entry = _training_mining_entry(key, value)
            if entry is not None and key not in entries:  # the first declaration wins
                entries[key] = entry
    return TrainingMiningPolicy(tuple(entries.values())) if entries else None


def _training_mining_entry(key: str, value: JsonValue) -> TrainingMiningEntry | None:
    if not isinstance(value, dict):
        return None
    use = value.get("use")
    if not isinstance(use, str):
        return None
    try:
        return TrainingMiningEntry(key, TrainingMiningUse(use), _text(value.get("constraint_info")))
    except (TypeError, ValueError):
        return None


def _soft_bindings(assertions: tuple[ManifestAssertion, ...]) -> tuple[SoftBindingReport, ...]:
    reports = []
    for assertion in assertions:
        if assertion.base_label != SOFT_BINDING_LABEL or not isinstance(assertion.data, dict):
            continue
        values = tuple(
            block["value"]
            for block in _items(assertion.data.get("blocks"))
            if isinstance(block, dict) and isinstance(block.get("value"), str)
        )
        reports.append(SoftBindingReport(algorithm=_text(assertion.data.get("alg")), values=values))
    return tuple(reports)


def _ingredients(value: object) -> tuple[IngredientReport, ...]:
    reports = []
    for item in _items(value):
        if not isinstance(item, dict):
            continue
        reports.append(
            IngredientReport(
                title=_text(item.get("title")),
                format=_text(item.get("format")),
                relationship=_text(item.get("relationship")),
                has_manifest=isinstance(item.get("active_manifest"), str),
            )
        )
    return tuple(reports)


def _base_label(label: str) -> str:
    return _VERSION_SUFFIX.sub("", _INSTANCE_SUFFIX.sub("", label))
