"""The manifest specification and its translation to a c2pa-python definition.

The labels and schemas are the ones c2pa-python 0.38 (c2pa-rs 0.91, C2PA
claim version 2) writes and validates; ``docs/provenance.md`` cites the
specifications:

* ``c2pa.actions`` in the definition, written by the library as
  ``c2pa.actions.v2`` (C2PA 2.3, section 18.15);
* ``cawg.training-mining`` (CAWG Training and Data Mining Assertion 1.1),
  which replaced the ``c2pa.training-mining`` assertion removed in C2PA 2.0;
* ``cawg.metadata`` (CAWG Metadata Assertion 1.1) with ``dc:creator`` for
  authors, which replaces ``stds.schema-org.CreativeWork`` (deprecated in
  C2PA 2.0 and in c2pa-rs 0.91); ``dc:creator`` is not allowed in
  ``c2pa.metadata``;
* ``c2pa.soft-binding`` (C2PA 2.3, section 18.10).

Each assertion is listed either in the claim's ``created_assertions`` (the
signer attests to it) or in its ``gathered_assertions`` (the claim
generator received it from another part of the workflow, such as a person,
and it is not attributed to the signer), as :class:`AssertionPlacements`
says (C2PA 2.3, section 10.2.2). The library always lists the hard binding
(``c2pa.hash.*``) as created.
"""

from __future__ import annotations

import datetime as _dt
import enum
import re
from dataclasses import dataclass, field
from typing import TypeVar

from provenance_kit.mediatype import MediaType
from provenance_kit.provenance.c2pa._json import JsonObject, JsonValue

__all__ = [
    "ActionType",
    "AssertionPlacement",
    "AssertionPlacements",
    "DigitalSourceType",
    "GeneratorInfo",
    "Ingredient",
    "IngredientRelationship",
    "ManifestAction",
    "ManifestSpec",
    "SoftBinding",
    "SoftBindingAlgorithm",
    "TrainingMiningEntry",
    "TrainingMiningKey",
    "TrainingMiningPolicy",
    "TrainingMiningUse",
    "build_definition",
]

#: Assertion labels written by :func:`build_definition`.
ACTIONS_LABEL = "c2pa.actions"
TRAINING_MINING_LABEL = "cawg.training-mining"
CAWG_METADATA_LABEL = "cawg.metadata"
SOFT_BINDING_LABEL = "c2pa.soft-binding"
DUBLIN_CORE_NAMESPACE = "http://purl.org/dc/elements/1.1/"

_T = TypeVar("_T")

_MAX_TEXT = 4096
# Namespaced labels such as "c2pa.created" or "com.example.my-action".
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*(?:\.[A-Za-z0-9][A-Za-z0-9_-]*)+$")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


class ActionType(enum.StrEnum):
    """Standard C2PA actions (C2PA 2.3, section 18.15).

    Custom actions use a reverse-domain string such as ``"com.example.reviewed"``.
    """

    CREATED = "c2pa.created"
    OPENED = "c2pa.opened"
    EDITED = "c2pa.edited"
    EDITED_METADATA = "c2pa.edited.metadata"
    PUBLISHED = "c2pa.published"
    CONVERTED = "c2pa.converted"
    TRANSCODED = "c2pa.transcoded"
    RESIZED = "c2pa.resized"
    CROPPED = "c2pa.cropped"
    COLOR_ADJUSTMENTS = "c2pa.color_adjustments"
    FILTERED = "c2pa.filtered"
    ORIENTATION = "c2pa.orientation"
    REPACKAGED = "c2pa.repackaged"
    TRIMMED = "c2pa.trimmed"
    PLACED = "c2pa.placed"
    REMOVED = "c2pa.removed"
    REDACTED = "c2pa.redacted"
    TRANSLATED = "c2pa.translated"
    WATERMARKED_BOUND = "c2pa.watermarked.bound"
    """An invisible watermark linked to this manifest by a soft binding."""
    WATERMARKED_UNBOUND = "c2pa.watermarked.unbound"
    """An invisible watermark not linked to this manifest."""
    UNKNOWN = "c2pa.unknown"


class DigitalSourceType(enum.StrEnum):
    """IPTC digital source types, as C2PA uses them in ``digitalSourceType``."""

    DIGITAL_CAPTURE = "http://cv.iptc.org/newscodes/digitalsourcetype/digitalCapture"
    COMPUTATIONAL_CAPTURE = "http://cv.iptc.org/newscodes/digitalsourcetype/computationalCapture"
    NEGATIVE_FILM = "http://cv.iptc.org/newscodes/digitalsourcetype/negativeFilm"
    POSITIVE_FILM = "http://cv.iptc.org/newscodes/digitalsourcetype/positiveFilm"
    PRINT = "http://cv.iptc.org/newscodes/digitalsourcetype/print"
    HUMAN_EDITS = "http://cv.iptc.org/newscodes/digitalsourcetype/humanEdits"
    MINOR_HUMAN_EDITS = "http://cv.iptc.org/newscodes/digitalsourcetype/minorHumanEdits"
    COMPOSITE_WITH_TRAINED_ALGORITHMIC_MEDIA = (
        "http://cv.iptc.org/newscodes/digitalsourcetype/compositeWithTrainedAlgorithmicMedia"
    )
    ALGORITHMICALLY_ENHANCED = (
        "http://cv.iptc.org/newscodes/digitalsourcetype/algorithmicallyEnhanced"
    )
    DIGITAL_CREATION = "http://cv.iptc.org/newscodes/digitalsourcetype/digitalCreation"
    DIGITAL_ART = "http://cv.iptc.org/newscodes/digitalsourcetype/digitalArt"
    DATA_DRIVEN_MEDIA = "http://cv.iptc.org/newscodes/digitalsourcetype/dataDrivenMedia"
    TRAINED_ALGORITHMIC_MEDIA = (
        "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"
    )
    ALGORITHMIC_MEDIA = "http://cv.iptc.org/newscodes/digitalsourcetype/algorithmicMedia"
    SCREEN_CAPTURE = "http://cv.iptc.org/newscodes/digitalsourcetype/screenCapture"
    VIRTUAL_RECORDING = "http://cv.iptc.org/newscodes/digitalsourcetype/virtualRecording"
    COMPOSITE = "http://cv.iptc.org/newscodes/digitalsourcetype/composite"
    COMPOSITE_CAPTURE = "http://cv.iptc.org/newscodes/digitalsourcetype/compositeCapture"
    COMPOSITE_SYNTHETIC = "http://cv.iptc.org/newscodes/digitalsourcetype/compositeSynthetic"


class TrainingMiningUse(enum.StrEnum):
    """Whether a use is permitted (CAWG Training and Data Mining 1.1).

    CAWG TDM 1.1: "In the absence of additional information, ``constrained``
    shall be treated as equivalent to ``notAllowed``" (the constraints are
    free text that software cannot evaluate);
    :meth:`TrainingMiningPolicy.effective_use_of` applies that rule.
    """

    ALLOWED = "allowed"
    NOT_ALLOWED = "notAllowed"
    CONSTRAINED = "constrained"


class TrainingMiningKey(enum.StrEnum):
    """The standard entries of a ``cawg.training-mining`` assertion."""

    DATA_MINING = "cawg.data_mining"
    """Extracting text or data to find patterns, trends and correlations."""
    AI_INFERENCE = "cawg.ai_inference"
    """Using the asset as input to a trained AI/ML model."""
    AI_TRAINING = "cawg.ai_training"
    """Training non-generative AI/ML models (for example classifiers)."""
    AI_GENERATIVE_TRAINING = "cawg.ai_generative_training"
    """Training AI/ML models that generate new assets."""


class AssertionPlacement(enum.StrEnum):
    """Where the claim lists an assertion (C2PA 2.3, section 10.2.2)."""

    CREATED = "created"
    """In ``created_assertions``: made by the claim generator and attributed
    to the signer ("All created_assertions are attributed to the signer")."""
    GATHERED = "gathered"
    """In ``gathered_assertions``: provided to the claim generator by another
    component of the workflow, for example information entered by a person;
    part of the claim, but "not attributed to the signer"."""


@dataclass(frozen=True, slots=True, kw_only=True)
class AssertionPlacements:
    """The placement of each assertion that :class:`ManifestSpec` writes.

    The defaults follow who vouches for the content: the actions, the
    training and data mining policy and the soft binding describe what the
    signing application did or decided, so the signer attests to them
    (created); author names are supplied by people, so they are listed as
    gathered. Move an assertion to the other list when your workflow differs,
    for example ``authors=AssertionPlacement.CREATED`` when the signer has
    verified the authorship.

    Attributes:
        actions: ``c2pa.actions`` (default: created).
        authors: ``cawg.metadata`` with the authors (default: gathered).
        training_mining: ``cawg.training-mining`` (default: created).
        soft_binding: ``c2pa.soft-binding`` (default: created).

    Raises:
        TypeError: If a field is not an :class:`AssertionPlacement`.
    """

    actions: AssertionPlacement = AssertionPlacement.CREATED
    authors: AssertionPlacement = AssertionPlacement.GATHERED
    training_mining: AssertionPlacement = AssertionPlacement.CREATED
    soft_binding: AssertionPlacement = AssertionPlacement.CREATED

    def __post_init__(self) -> None:
        """Validates the fields."""
        for name in ("actions", "authors", "training_mining", "soft_binding"):
            if not isinstance(getattr(self, name), AssertionPlacement):
                raise TypeError(f"{name} must be an AssertionPlacement")


class IngredientRelationship(enum.StrEnum):
    """How an ingredient relates to the asset (C2PA 2.3, section 18.16)."""

    PARENT_OF = "parentOf"
    COMPONENT_OF = "componentOf"
    INPUT_TO = "inputTo"


class SoftBindingAlgorithm(enum.StrEnum):
    """Soft-binding algorithm identifiers from the C2PA soft binding algorithm list.

    The list is at https://github.com/c2pa-org/softbinding-algorithm-list.
    Any other registered identifier can be passed as a plain string.
    """

    TRUSTMARK_Q = "com.adobe.trustmark.Q"
    """TrustMark variant Q (list identifier 4)."""
    TRUSTMARK_C = "com.adobe.trustmark.C"
    """TrustMark variant C (list identifier 5)."""
    TRUSTMARK_P = "com.adobe.trustmark.P"
    """TrustMark variant P (list identifier 13)."""
    TRUSTMARK_F = "com.adobe.trustmark.F"
    """TrustMark variant F (list identifier 55)."""


@dataclass(frozen=True, slots=True)
class GeneratorInfo:
    """A software name and version (the C2PA ``claim-generator-info-map``).

    Used for the claim generator and for an action's software agent.

    Attributes:
        name: The product or tool name.
        version: Its version, if known.

    Raises:
        TypeError: If a field is not a str.
        ValueError: If ``name`` is empty or a field is too long.
    """

    name: str
    version: str | None = None

    def __post_init__(self) -> None:
        """Validates the fields."""
        _check_text("name", self.name)
        if self.version is not None:
            _check_text("version", self.version)

    def to_json(self) -> JsonObject:
        """Returns the JSON form used in the manifest definition."""
        info: JsonObject = {"name": self.name}
        if self.version is not None:
            info["version"] = self.version
        return info


@dataclass(frozen=True, slots=True, kw_only=True)
class ManifestAction:
    """One entry of the actions assertion.

    Attributes:
        action: A standard :class:`ActionType` or a custom reverse-domain name.
        when: When the action happened (timezone-aware); written in UTC as
            RFC 3339, to the millisecond.
        software_agent: The software that performed the action.
        digital_source_type: Required for :attr:`ActionType.CREATED`.
        description: A human-readable description.
        ingredient_ids: The ingredients the action used, by
            :attr:`Ingredient.ingredient_id`. Required for ``placed`` and
            ``removed``; an ``opened`` action without ids is linked to the
            parent ingredient automatically.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If a value is malformed, or ``when`` is naive.
    """

    action: ActionType | str
    when: _dt.datetime | None = None
    software_agent: GeneratorInfo | None = None
    digital_source_type: DigitalSourceType | str | None = None
    description: str | None = None
    ingredient_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validates the fields."""
        object.__setattr__(self, "action", _action_type(self.action))
        if self.when is not None:
            if not isinstance(self.when, _dt.datetime):
                raise TypeError("when must be a datetime")
            if self.when.tzinfo is None or self.when.utcoffset() is None:
                raise ValueError("when must be timezone-aware")
        if self.software_agent is not None and not isinstance(self.software_agent, GeneratorInfo):
            raise TypeError("software_agent must be a GeneratorInfo")
        if self.digital_source_type is not None:
            _check_source_type(self.digital_source_type)
        if self.description is not None:
            _check_text("description", self.description)
        ids = _typed_tuple("ingredient_ids", self.ingredient_ids, str)
        for ingredient_id in ids:
            _check_id(ingredient_id)
        object.__setattr__(self, "ingredient_ids", ids)

    def to_json(self, ingredient_ids: tuple[str, ...]) -> JsonObject:
        """Returns the JSON form, with the given resolved ingredient ids."""
        entry: JsonObject = {"action": str(self.action)}
        if self.when is not None:
            entry["when"] = _rfc3339(self.when)
        if self.software_agent is not None:
            entry["softwareAgent"] = self.software_agent.to_json()
        if self.digital_source_type is not None:
            entry["digitalSourceType"] = str(self.digital_source_type)
        if self.description is not None:
            entry["description"] = self.description
        if ingredient_ids:
            entry["parameters"] = {"ingredientIds": list(ingredient_ids)}
        return entry


@dataclass(frozen=True, slots=True)
class TrainingMiningEntry:
    """One entry of a training and data mining assertion.

    Attributes:
        key: A :class:`TrainingMiningKey` or a custom key.
        use: Whether the use is allowed.
        constraint_info: Free text describing the constraints, for
            :attr:`TrainingMiningUse.CONSTRAINED`.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If ``key`` or ``constraint_info`` is empty or too long.
    """

    key: TrainingMiningKey | str
    use: TrainingMiningUse
    constraint_info: str | None = None

    def __post_init__(self) -> None:
        """Validates the fields."""
        _check_text("key", self.key)
        if not isinstance(self.use, TrainingMiningUse):
            raise TypeError("use must be a TrainingMiningUse")
        if self.constraint_info is not None:
            _check_text("constraint_info", self.constraint_info)


@dataclass(frozen=True, slots=True)
class TrainingMiningPolicy:
    """The entries of a ``cawg.training-mining`` assertion.

    Attributes:
        entries: One entry per key, in order.

    Raises:
        TypeError: If an entry is not a :class:`TrainingMiningEntry`.
        ValueError: If there are no entries or a key repeats.
    """

    entries: tuple[TrainingMiningEntry, ...]

    def __post_init__(self) -> None:
        """Validates the entries."""
        entries = _typed_tuple("entries", self.entries, TrainingMiningEntry)
        if not entries:
            raise ValueError("a training-mining policy needs at least one entry")
        seen: set[str] = set()
        for entry in entries:
            if str(entry.key) in seen:
                raise ValueError("training-mining keys must be unique")
            seen.add(str(entry.key))
        object.__setattr__(self, "entries", entries)

    @classmethod
    def not_allowed(cls) -> TrainingMiningPolicy:
        """Returns a policy that refuses all four standard uses.

        Returns:
            ``notAllowed`` for AI training, generative AI training, data
            mining and AI inference.
        """
        return cls(tuple(TrainingMiningEntry(key, TrainingMiningUse.NOT_ALLOWED) for key in _KEYS))

    def use_of(self, key: TrainingMiningKey | str) -> TrainingMiningUse | None:
        """Returns the declared use for a key.

        A standard key also matches its C2PA 1.x spelling (``c2pa.`` instead
        of ``cawg.``), so that policies read from older manifests compare
        alike.

        Args:
            key: The entry key.

        Returns:
            The use as declared (``constrained`` stays ``constrained``), or
            ``None`` if the policy has no such entry.
        """
        wanted = {str(key)}
        if str(key).startswith("cawg."):
            wanted.add("c2pa." + str(key)[len("cawg.") :])
        for entry in self.entries:
            if str(entry.key) in wanted:
                return entry.use
        return None

    def effective_use_of(self, key: TrainingMiningKey | str) -> TrainingMiningUse | None:
        """Returns the use a consumer must apply for a key, per CAWG TDM 1.1.

        ``constrained`` becomes ``notAllowed`` ("in the absence of additional
        information, ``constrained`` shall be treated as equivalent to
        ``notAllowed``"): software cannot evaluate the free-text constraints,
        so it must not treat them as permission. Read
        :attr:`TrainingMiningEntry.constraint_info` and :meth:`use_of` when a
        person can.

        Args:
            key: The entry key (C2PA 1.x spellings match as in :meth:`use_of`).

        Returns:
            ``ALLOWED`` or ``NOT_ALLOWED``, or ``None`` if the policy has no
            such entry (the policy says nothing about that use).
        """
        use = self.use_of(key)
        if use is TrainingMiningUse.CONSTRAINED:
            return TrainingMiningUse.NOT_ALLOWED
        return use

    @property
    def all_not_allowed(self) -> bool:
        """Whether none of the four standard uses is permitted, per CAWG TDM 1.1.

        True when AI training, generative AI training, data mining and AI
        inference are each declared ``notAllowed`` or ``constrained`` (which
        CAWG says to treat as ``notAllowed``; see :meth:`effective_use_of`).
        False when a standard use is ``allowed`` or not declared at all.
        :attr:`all_declared_not_allowed` is the literal reading.
        """
        return all(self.effective_use_of(key) is TrainingMiningUse.NOT_ALLOWED for key in _KEYS)

    @property
    def all_declared_not_allowed(self) -> bool:
        """Whether all four standard uses are declared exactly ``notAllowed``.

        The literal view: a policy with a ``constrained`` use is not counted.
        For deciding whether a use is permitted, use :attr:`all_not_allowed`
        or :meth:`effective_use_of`.
        """
        return all(self.use_of(key) is TrainingMiningUse.NOT_ALLOWED for key in _KEYS)


_KEYS = (
    TrainingMiningKey.AI_TRAINING,
    TrainingMiningKey.AI_GENERATIVE_TRAINING,
    TrainingMiningKey.DATA_MINING,
    TrainingMiningKey.AI_INFERENCE,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class SoftBinding:
    """A soft binding that links an invisible watermark to the manifest.

    Attributes:
        algorithm: The watermark algorithm identifier from the C2PA soft
            binding algorithm list (for example
            :attr:`SoftBindingAlgorithm.TRUSTMARK_Q`).
        value: The algorithm-specific watermark value, as text. The value
            format is defined by the algorithm's entry in the list. For
            TrustMark, Adobe's reference example writes
            ``"<encoding>*<payload bits>"``. See "Soft binding value" in
            ``docs/provenance.md``: c2pa-python 0.38 can only write it as a
            CBOR text string, while the C2PA schema says byte string.
        timespan_ms: For audio or video, the ``(start, end)`` range in
            milliseconds from the start of the media that the watermark
            covers; ``None`` for the whole asset.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If a field is empty, malformed or out of range.
    """

    algorithm: SoftBindingAlgorithm | str
    value: str
    timespan_ms: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        """Validates the fields."""
        if not isinstance(self.algorithm, str):
            raise TypeError("algorithm must be a SoftBindingAlgorithm or a str")
        if not _NAME.match(self.algorithm) or len(self.algorithm) > 256:
            raise ValueError("algorithm must be a dotted identifier such as com.example.wm.1")
        _check_text("value", self.value)
        if self.timespan_ms is not None:
            span = _typed_tuple("timespan_ms", self.timespan_ms, int)
            if len(span) != 2 or any(isinstance(v, bool) for v in span):
                raise TypeError("timespan_ms must be a pair of ints")
            start, end = span
            if not 0 <= start <= end:
                raise ValueError("timespan_ms must satisfy 0 <= start <= end")
            object.__setattr__(self, "timespan_ms", (start, end))

    def to_json(self) -> JsonObject:
        """Returns the assertion data."""
        scope: JsonObject = {}
        if self.timespan_ms is not None:
            scope["timespan"] = {"start": self.timespan_ms[0], "end": self.timespan_ms[1]}
        return {"alg": str(self.algorithm), "blocks": [{"scope": scope, "value": self.value}]}


@dataclass(frozen=True, slots=True, kw_only=True)
class Ingredient:
    """An asset the new asset was made from (C2PA ingredient).

    The C2PA library reads the ingredient and, when it carries its own
    manifest, includes that manifest and its validation results.

    Attributes:
        data: The ingredient's encoded bytes (untrusted; size-checked against
            the signing call's limits).
        title: A human-readable name, for example a file name.
        mime_type: Its MIME type, or ``None`` to detect it from the bytes.
        relationship: How it relates to the new asset. At most one ingredient
            may be the parent.
        ingredient_id: The id that :attr:`ManifestAction.ingredient_ids`
            refer to; defaults to ``"ingredient-<index>"``.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If ``data`` is empty or a text field is malformed.
    """

    data: bytes = field(repr=False)
    title: str
    mime_type: str | MediaType | None = None
    relationship: IngredientRelationship = IngredientRelationship.PARENT_OF
    ingredient_id: str | None = None

    def __post_init__(self) -> None:
        """Validates the fields."""
        data: object = self.data
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("data must be bytes-like")
        object.__setattr__(self, "data", bytes(memoryview(data).cast("B")))
        if not self.data:
            raise ValueError("ingredient data must not be empty")
        _check_text("title", self.title)
        if self.mime_type is not None and not isinstance(self.mime_type, str):
            raise TypeError("mime_type must be a str, a MediaType or None")
        if not isinstance(self.relationship, IngredientRelationship):
            raise TypeError("relationship must be an IngredientRelationship")
        if self.ingredient_id is not None:
            _check_id(self.ingredient_id)


@dataclass(frozen=True, slots=True, kw_only=True)
class ManifestSpec:
    """What to put in a C2PA manifest.

    The manifest's ``format`` (``dc:format``) is the media type of the asset
    being signed, so it is not a field here. No thumbnail is added.

    Rules checked here (the C2PA library checks the rest when it signs):

    * the first action is ``c2pa.created`` or ``c2pa.opened``
      (C2PA 2.3, section 15.10.3.2);
    * ``c2pa.created`` has a ``digital_source_type``;
    * ``c2pa.opened`` needs a parent ingredient; there is at most one parent;
    * ``c2pa.placed`` and ``c2pa.removed`` name their ingredients;
    * ``c2pa.watermarked.bound`` needs a :attr:`soft_binding`.

    Attributes:
        claim_generator: The application that creates the manifest.
        actions: The actions, in order.
        title: The asset's title (``dc:title``), if any.
        authors: Author or creator names, written as ``dc:creator`` in a
            ``cawg.metadata`` assertion. By default the assertion is gathered
            (see :attr:`placements`): the names are human-supplied
            statements, not identity proofs (a CAWG identity assertion,
            signed by the named actor, is out of scope).
        training_mining: A ``cawg.training-mining`` policy, for example
            :meth:`TrainingMiningPolicy.not_allowed`; ``None`` adds none.
        soft_binding: A ``c2pa.soft-binding`` assertion linking an invisible
            watermark to this manifest.
        ingredients: The assets this one was made from.
        placements: Whether each assertion is created (attested by the
            signer) or gathered; see :class:`AssertionPlacements` for the
            defaults.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If a rule above is broken or a value is malformed.
    """

    claim_generator: GeneratorInfo
    actions: tuple[ManifestAction, ...]
    title: str | None = None
    authors: tuple[str, ...] = ()
    training_mining: TrainingMiningPolicy | None = None
    soft_binding: SoftBinding | None = None
    ingredients: tuple[Ingredient, ...] = ()
    placements: AssertionPlacements = field(default_factory=AssertionPlacements)

    def __post_init__(self) -> None:
        """Validates the specification."""
        if not isinstance(self.placements, AssertionPlacements):
            raise TypeError("placements must be an AssertionPlacements")
        if not isinstance(self.claim_generator, GeneratorInfo):
            raise TypeError("claim_generator must be a GeneratorInfo")
        object.__setattr__(self, "actions", _typed_tuple("actions", self.actions, ManifestAction))
        object.__setattr__(
            self, "ingredients", _typed_tuple("ingredients", self.ingredients, Ingredient)
        )
        object.__setattr__(self, "authors", _typed_tuple("authors", self.authors, str))
        if self.title is not None:
            _check_text("title", self.title)
        for author in self.authors:
            _check_text("author", author)
        if self.training_mining is not None:
            if not isinstance(self.training_mining, TrainingMiningPolicy):
                raise TypeError("training_mining must be a TrainingMiningPolicy or None")
            for entry in self.training_mining.entries:
                key = str(entry.key)
                if key.startswith("cawg.") and key not in _STANDARD_KEYS:
                    raise ValueError("the cawg. prefix is reserved for the standard keys")
        if self.soft_binding is not None and not isinstance(self.soft_binding, SoftBinding):
            raise TypeError("soft_binding must be a SoftBinding or None")
        self._check_rules()

    def _check_rules(self) -> None:
        """Checks the action and ingredient rules listed in the class docstring."""
        if not self.actions:
            raise ValueError("a manifest needs at least one action")
        ids = self.ingredient_ids()
        if len(set(ids)) != len(ids):
            raise ValueError("ingredient ids must be unique")
        parents = sum(
            ingredient.relationship is IngredientRelationship.PARENT_OF
            for ingredient in self.ingredients
        )
        if parents > 1:
            raise ValueError("a manifest can have at most one parent ingredient")
        if self.actions[0].action not in (ActionType.CREATED, ActionType.OPENED):
            raise ValueError("the first action must be c2pa.created or c2pa.opened")
        for action in self.actions:
            if set(action.ingredient_ids) - set(ids):
                raise ValueError("an action refers to an unknown ingredient id")
            if action.action is ActionType.CREATED and action.digital_source_type is None:
                raise ValueError("a c2pa.created action needs a digital_source_type")
            if action.action is ActionType.OPENED and not parents:
                raise ValueError("a c2pa.opened action needs a parent ingredient")
            if (
                action.action in (ActionType.PLACED, ActionType.REMOVED)
                and not action.ingredient_ids
            ):
                raise ValueError(f"a {action.action} action must name its ingredient_ids")
            if action.action is ActionType.WATERMARKED_BOUND and self.soft_binding is None:
                raise ValueError("a c2pa.watermarked.bound action needs a soft_binding")

    def ingredient_ids(self) -> tuple[str, ...]:
        """Returns each ingredient's id, with the defaults filled in."""
        return tuple(
            ingredient.ingredient_id or f"ingredient-{index}"
            for index, ingredient in enumerate(self.ingredients)
        )


_STANDARD_KEYS = frozenset(key.value for key in TrainingMiningKey)
_ACTIONS_BY_VALUE = {action.value: action for action in ActionType}


@dataclass(frozen=True, slots=True)
class IngredientSource:
    """An ingredient ready for ``Builder.add_ingredient_from_stream`` (internal)."""

    definition: JsonObject
    mime_type: str
    data: bytes = field(repr=False)


def build_definition(
    spec: ManifestSpec,
    media_type: MediaType,
    ingredient_types: tuple[MediaType, ...],
) -> tuple[JsonObject, tuple[IngredientSource, ...]]:
    """Translates a specification into a c2pa-python manifest definition.

    Args:
        spec: The specification.
        media_type: The type of the asset being signed (written as ``format``).
        ingredient_types: The resolved type of each ingredient, in order.

    Returns:
        The definition for ``c2pa.Builder`` and the ingredients to add.

    Raises:
        ValueError: If ``ingredient_types`` does not match the ingredients.
    """
    if len(ingredient_types) != len(spec.ingredients):
        raise ValueError("one media type is needed per ingredient")
    ids = spec.ingredient_ids()
    parent_ids = tuple(
        ingredient_id
        for ingredient_id, ingredient in zip(ids, spec.ingredients, strict=True)
        if ingredient.relationship is IngredientRelationship.PARENT_OF
    )
    actions: list[JsonValue] = []
    for action in spec.actions:
        linked = action.ingredient_ids
        if action.action is ActionType.OPENED and not linked:
            linked = parent_ids
        actions.append(action.to_json(linked))

    placements = spec.placements
    assertions: list[JsonValue] = [
        _assertion(ACTIONS_LABEL, {"actions": actions}, placements.actions)
    ]
    if spec.authors:
        assertions.append(
            _assertion(
                CAWG_METADATA_LABEL,
                {"@context": {"dc": DUBLIN_CORE_NAMESPACE}, "dc:creator": list(spec.authors)},
                placements.authors,
                kind="Json",
            )
        )
    if spec.training_mining is not None:
        entries: JsonObject = {}
        for entry in spec.training_mining.entries:
            value: JsonObject = {"use": entry.use.value}
            if entry.constraint_info is not None:
                value["constraint_info"] = entry.constraint_info
            entries[str(entry.key)] = value
        assertions.append(
            _assertion(TRAINING_MINING_LABEL, {"entries": entries}, placements.training_mining)
        )
    if spec.soft_binding is not None:
        assertions.append(
            _assertion(SOFT_BINDING_LABEL, spec.soft_binding.to_json(), placements.soft_binding)
        )

    definition: JsonObject = {
        "claim_generator_info": [spec.claim_generator.to_json()],
        "format": media_type.mime,
        "assertions": assertions,
    }
    if spec.title is not None:
        definition["title"] = spec.title
    sources = tuple(
        IngredientSource(
            definition={
                "title": ingredient.title,
                "relationship": ingredient.relationship.value,
                "label": ingredient_id,
            },
            mime_type=ingredient_type.mime,
            data=ingredient.data,
        )
        for ingredient_id, ingredient, ingredient_type in zip(
            ids, spec.ingredients, ingredient_types, strict=True
        )
    )
    return definition, sources


def _assertion(
    label: str,
    data: JsonObject,
    placement: AssertionPlacement,
    *,
    kind: str | None = None,
) -> JsonObject:
    # c2pa-rs (0.91) lists an assertion defined with "created": false in the
    # claim's gathered_assertions, and one with "created": true in
    # created_assertions (checked by the integration tests).
    created = placement is AssertionPlacement.CREATED
    assertion: JsonObject = {"label": label, "created": created, "data": data}
    if kind is not None:
        assertion["kind"] = kind
    return assertion


def _rfc3339(moment: _dt.datetime) -> str:
    utc = moment.astimezone(_dt.UTC)
    timespec = "milliseconds" if utc.microsecond else "seconds"
    return utc.replace(tzinfo=None).isoformat(timespec=timespec) + "Z"


def _action_type(value: object) -> ActionType | str:
    if isinstance(value, ActionType):
        return value
    if not isinstance(value, str):
        raise TypeError("action must be an ActionType or a str")
    if not _NAME.match(value):
        raise ValueError("a custom action must be a dotted name such as com.example.x")
    known = _ACTIONS_BY_VALUE.get(value)
    return value if known is None else known


def _check_source_type(value: object) -> None:
    if isinstance(value, DigitalSourceType):
        return
    if not isinstance(value, str):
        raise TypeError("digital_source_type must be a DigitalSourceType or a str")
    if not value.startswith(("http://", "https://")):
        raise ValueError("digital_source_type must be an absolute http(s) URI")
    _check_text("digital_source_type", value)


def _check_text(name: str, value: object) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a str")
    if not value.strip():
        raise ValueError(f"{name} must not be empty")
    if len(value) > _MAX_TEXT:
        raise ValueError(f"{name} must be at most {_MAX_TEXT} characters")
    if "\x00" in value:
        raise ValueError(f"{name} must not contain NUL characters")


def _check_id(value: object) -> None:
    if not isinstance(value, str):
        raise TypeError("ingredient ids must be str")
    if not _ID.match(value):
        raise ValueError("ingredient ids must be 1-128 letters, digits or . _ : -")


def _typed_tuple(name: str, value: object, item_type: type[_T]) -> tuple[_T, ...]:
    """Accepts a tuple (or list) whose items all have ``item_type``."""
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{name} must be a tuple")
    items = tuple(value)
    for item in items:
        if not isinstance(item, item_type):
            raise TypeError(f"{name} must contain {item_type.__name__} items")
    return items
