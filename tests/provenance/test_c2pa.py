"""Unit tests of the C2PA module: no c2pa-python, cryptography or network needed."""

from __future__ import annotations

import datetime as dt
import json
import sys

import pytest

from provenance_kit.errors import (
    BackendUnavailableError,
    MediaLimitExceededError,
    UnsupportedMediaError,
)
from provenance_kit.limits import MediaLimits
from provenance_kit.mediatype import MediaType
from provenance_kit.provenance.c2pa import (
    MANIFEST_UNREADABLE,
    SUPPORTED_MEDIA_TYPES,
    ActionType,
    AssertionPlacement,
    AssertionPlacements,
    DigitalSourceType,
    GeneratorInfo,
    Ingredient,
    IngredientRelationship,
    ManifestAction,
    ManifestSpec,
    SignatureFormat,
    SigningAlgorithm,
    SigningError,
    SoftBinding,
    SoftBindingAlgorithm,
    TrainingMiningEntry,
    TrainingMiningKey,
    TrainingMiningPolicy,
    TrainingMiningUse,
    ValidationState,
    ecdsa_der_to_raw,
    read_manifest,
    sign_asset,
)
from provenance_kit.provenance.c2pa._manifest import build_definition
from provenance_kit.provenance.c2pa._media import parse_media_type, resolve_media_type
from provenance_kit.provenance.c2pa._native import (
    SignatureBridge,
    _read_failure,
    _signing_failure,
    read_settings,
    sign_settings,
    tsa_host_pattern,
)
from provenance_kit.provenance.c2pa._report import parse_manifest_store
from provenance_kit.provenance.c2pa._signer import normalize_signature
from tests.support import HEAVY_MODULES, assert_imports_lightly

JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
FAKE_CHAIN = "-----BEGIN CERTIFICATE-----\nAAAA\n-----END CERTIFICATE-----\n"
CREATED = ManifestAction(
    action=ActionType.CREATED, digital_source_type=DigitalSourceType.DIGITAL_CAPTURE
)


def spec(**overrides: object) -> ManifestSpec:
    fields: dict[str, object] = {
        "claim_generator": GeneratorInfo("tests", "1.0"),
        "actions": (CREATED,),
    }
    fields.update(overrides)
    return ManifestSpec(**fields)  # type: ignore[arg-type]


class FakeSigner:
    """A signer that returns fixed bytes (or raises)."""

    def __init__(
        self,
        signature: object = b"\x01" * 64,
        *,
        algorithm: SigningAlgorithm = SigningAlgorithm.ES256,
        signature_format: SignatureFormat = SignatureFormat.RAW,
        chain: str = FAKE_CHAIN,
        error: BaseException | None = None,
    ) -> None:
        self.algorithm = algorithm
        self.certificate_chain_pem = chain
        self.signature_format = signature_format
        self._signature = signature
        self._error = error
        self.calls: list[bytes] = []

    def sign(self, data: bytes) -> bytes:
        self.calls.append(data)
        if self._error is not None:
            raise self._error
        return self._signature  # type: ignore[return-value]


def test_importing_the_module_is_light():
    assert_imports_lightly(
        ["provenance_kit.provenance.c2pa"],
        forbidden=(*HEAVY_MODULES, "cryptography"),
    )


# ------------------------------------------------------------------ DER


def der(r: int, s: int) -> bytes:
    def integer(value: int) -> bytes:
        body = value.to_bytes((value.bit_length() + 8) // 8 or 1, "big")
        return b"\x02" + bytes([len(body)]) + body

    content = integer(r) + integer(s)
    length = bytes([len(content)]) if len(content) < 128 else b"\x81" + bytes([len(content)])
    return b"\x30" + length + content


@pytest.mark.parametrize(
    ("algorithm", "width"),
    [(SigningAlgorithm.ES256, 32), (SigningAlgorithm.ES384, 48), (SigningAlgorithm.ES512, 66)],
)
def test_der_to_raw_pads_each_component(algorithm, width):
    r, s = 1, (1 << (8 * width - 1)) + 5  # s has its high bit set: DER adds a 00 byte
    raw = ecdsa_der_to_raw(der(r, s), algorithm)
    assert len(raw) == 2 * width
    assert int.from_bytes(raw[:width], "big") == r
    assert int.from_bytes(raw[width:], "big") == s


def test_der_to_raw_accepts_long_form_lengths_for_p521():
    big = (1 << 520) + 12345
    signature = der(big, big)
    assert signature[1] == 0x81
    assert ecdsa_der_to_raw(signature, SigningAlgorithm.ES512)[:66] == big.to_bytes(66, "big")


@pytest.mark.parametrize(
    "signature",
    [
        b"",
        b"\x31\x06\x02\x01\x01\x02\x01\x01",  # SET, not SEQUENCE
        der(1, 2) + b"\x00",  # trailing byte
        b"\x30\x07\x02\x01\x01\x02\x01\x01",  # wrong SEQUENCE length
        b"\x30\x07\x02\x02\x00\x01\x02\x01\x01",  # non-minimal INTEGER
        b"\x30\x06\x02\x01\x81\x02\x01\x01",  # negative INTEGER
        b"\x30\x06\x02\x01\x00\x02\x01\x01",  # r = 0
        b"\x30\x06\x04\x01\x01\x02\x01\x01",  # OCTET STRING, not INTEGER
        b"\x30\x81\x06\x02\x01\x01\x02\x01\x01",  # non-minimal long-form length
        b"\x30\x82\x00\x06\x02\x01\x01\x02\x01\x01",  # two length bytes
        b"\x30\x06\x02\x05\x01\x02\x01\x01",  # INTEGER runs past the end
    ],
)
def test_der_to_raw_rejects_malformed_input(signature):
    with pytest.raises(ValueError, match=r"DER ECDSA|out of range"):
        ecdsa_der_to_raw(signature, SigningAlgorithm.ES256)


def test_der_to_raw_rejects_components_wider_than_the_curve():
    with pytest.raises(ValueError, match="out of range"):
        ecdsa_der_to_raw(der(1 << 256, 1), SigningAlgorithm.ES256)


def test_der_to_raw_argument_checks():
    with pytest.raises(ValueError, match="not an ECDSA"):
        ecdsa_der_to_raw(der(1, 1), SigningAlgorithm.ED25519)
    with pytest.raises(TypeError):
        ecdsa_der_to_raw("30", SigningAlgorithm.ES256)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ecdsa_der_to_raw(der(1, 1), "es256")  # type: ignore[arg-type]


# ------------------------------------------------------------------ signer bridge


def test_bridge_converts_der_from_a_kms_like_signer():
    signer = FakeSigner(der(7, 9), signature_format=SignatureFormat.DER)
    bridge = SignatureBridge(signer)
    raw = bridge(b"to-be-signed")
    assert raw == (7).to_bytes(32, "big") + (9).to_bytes(32, "big")
    assert signer.calls == [b"to-be-signed"]
    bridge.raise_error()  # nothing recorded


def test_bridge_records_signer_exceptions_without_leaking_their_text():
    detail = "arn:aws:kms:eu-west-1:123456789012:key/secret-key-id"
    bridge = SignatureBridge(FakeSigner(error=RuntimeError(detail)))
    assert bridge(b"data") == b""
    assert bridge(b"again") == b""  # the signer is not called again
    with pytest.raises(SigningError) as info:
        bridge.raise_error()
    assert detail not in str(info.value)
    assert "RuntimeError" in str(info.value)
    assert isinstance(info.value.__cause__, RuntimeError)


def test_bridge_reraises_keyboard_interrupt_unchanged():
    bridge = SignatureBridge(FakeSigner(error=KeyboardInterrupt()))
    assert bridge(b"data") == b""
    with pytest.raises(KeyboardInterrupt):
        bridge.raise_error()


@pytest.mark.parametrize(
    ("signature", "algorithm", "signature_format", "message"),
    [
        ("not bytes", SigningAlgorithm.ES256, SignatureFormat.RAW, "not bytes"),
        (b"\x01" * 63, SigningAlgorithm.ES256, SignatureFormat.RAW, "64 bytes"),
        (der(1, 2), SigningAlgorithm.ES256, SignatureFormat.RAW, "64 bytes"),  # DER declared RAW
        (b"\x01" * 64, SigningAlgorithm.ES256, SignatureFormat.DER, "invalid DER"),
        (b"\x01" * 65, SigningAlgorithm.ED25519, SignatureFormat.RAW, "Ed25519"),
        (b"\x01" * 100, SigningAlgorithm.PS256, SignatureFormat.RAW, "256 to 1024"),
    ],
)
def test_bridge_rejects_unusable_signatures(signature, algorithm, signature_format, message):
    with pytest.raises(SigningError, match=message):
        normalize_signature(signature, algorithm, signature_format)


def test_normalize_signature_passes_valid_raw_signatures():
    assert normalize_signature(b"\x02" * 96, SigningAlgorithm.ES384, SignatureFormat.RAW)
    assert normalize_signature(b"\x02" * 64, SigningAlgorithm.ED25519, SignatureFormat.RAW)
    assert normalize_signature(bytearray(256), SigningAlgorithm.PS256, SignatureFormat.RAW)


@pytest.mark.parametrize(
    ("signer", "error"),
    [
        (FakeSigner(chain="not pem"), ValueError),
        (FakeSigner(chain=b"bytes"), TypeError),  # type: ignore[arg-type]
        (FakeSigner(algorithm="es256"), TypeError),  # type: ignore[arg-type]
        (
            FakeSigner(algorithm=SigningAlgorithm.ED25519, signature_format=SignatureFormat.DER),
            ValueError,
        ),
        (FakeSigner(signature_format="raw"), TypeError),  # type: ignore[arg-type]
    ],
)
def test_bridge_validates_the_signer(signer, error):
    with pytest.raises(error):
        SignatureBridge(signer)


def test_signing_algorithm_properties():
    assert SigningAlgorithm.ES512.ecdsa_component_bytes == 66
    assert SigningAlgorithm.PS384.hash_name == "sha384"
    assert SigningAlgorithm.ED25519.hash_name is None
    assert not SigningAlgorithm.PS256.is_ecdsa


# ------------------------------------------------------------------ manifest spec


def test_definition_uses_the_labels_the_library_validates():
    moment = dt.datetime(2026, 11, 5, 9, 30, tzinfo=dt.timezone(dt.timedelta(hours=-3)))
    manifest = spec(
        title="photo.jpg",
        actions=(
            ManifestAction(
                action=ActionType.CREATED,
                when=moment,
                software_agent=GeneratorInfo("camera-app", "2.1"),
                digital_source_type=DigitalSourceType.DIGITAL_CAPTURE,
            ),
            ManifestAction(action=ActionType.WATERMARKED_BOUND),
            ManifestAction(action="c2pa.published", description="first publication"),
        ),
        authors=("Jane Doe", "John Roe"),
        training_mining=TrainingMiningPolicy.not_allowed(),
        soft_binding=SoftBinding(algorithm=SoftBindingAlgorithm.TRUSTMARK_Q, value="1*0101"),
    )
    definition, ingredients = build_definition(manifest, MediaType.JPEG, ())
    assert ingredients == ()
    assert definition["claim_generator_info"] == [{"name": "tests", "version": "1.0"}]
    assert definition["format"] == "image/jpeg"
    assert definition["title"] == "photo.jpg"
    labels = [a["label"] for a in definition["assertions"]]  # type: ignore[index, union-attr]
    assert labels == ["c2pa.actions", "cawg.metadata", "cawg.training-mining", "c2pa.soft-binding"]
    assertions = {a["label"]: a for a in definition["assertions"]}  # type: ignore[index, union-attr]
    # Defaults: the signer attests to what the application did; the authors
    # are human-supplied and gathered (C2PA 2.3, section 10.2.2).
    assert {label: a["created"] for label, a in assertions.items()} == {
        "c2pa.actions": True,
        "cawg.metadata": False,
        "cawg.training-mining": True,
        "c2pa.soft-binding": True,
    }
    assert assertions["c2pa.actions"]["data"] == {
        "actions": [
            {
                "action": "c2pa.created",
                "when": "2026-11-05T12:30:00Z",
                "softwareAgent": {"name": "camera-app", "version": "2.1"},
                "digitalSourceType": "http://cv.iptc.org/newscodes/digitalsourcetype/digitalCapture",
            },
            {"action": "c2pa.watermarked.bound"},
            {"action": "c2pa.published", "description": "first publication"},
        ]
    }
    assert assertions["cawg.metadata"]["kind"] == "Json"
    assert assertions["cawg.metadata"]["data"] == {
        "@context": {"dc": "http://purl.org/dc/elements/1.1/"},
        "dc:creator": ["Jane Doe", "John Roe"],
    }
    assert assertions["cawg.training-mining"]["data"] == {
        "entries": {
            "cawg.ai_training": {"use": "notAllowed"},
            "cawg.ai_generative_training": {"use": "notAllowed"},
            "cawg.data_mining": {"use": "notAllowed"},
            "cawg.ai_inference": {"use": "notAllowed"},
        }
    }
    assert assertions["c2pa.soft-binding"]["data"] == {
        "alg": "com.adobe.trustmark.Q",
        "blocks": [{"scope": {}, "value": "1*0101"}],
    }
    json.dumps(definition)  # serializable as is


def test_assertion_placements_are_configurable_per_assertion():
    manifest = spec(
        authors=("Jane Doe",),
        training_mining=TrainingMiningPolicy.not_allowed(),
        soft_binding=SoftBinding(algorithm=SoftBindingAlgorithm.TRUSTMARK_Q, value="1*0101"),
        actions=(
            ManifestAction(
                action=ActionType.CREATED, digital_source_type=DigitalSourceType.DIGITAL_CAPTURE
            ),
            ManifestAction(action=ActionType.WATERMARKED_BOUND),
        ),
        placements=AssertionPlacements(
            actions=AssertionPlacement.GATHERED,
            authors=AssertionPlacement.CREATED,
            training_mining=AssertionPlacement.GATHERED,
            soft_binding=AssertionPlacement.GATHERED,
        ),
    )
    definition, _ = build_definition(manifest, MediaType.PNG, ())
    created = {a["label"]: a["created"] for a in definition["assertions"]}  # type: ignore[index, union-attr]
    assert created == {
        "c2pa.actions": False,
        "cawg.metadata": True,
        "cawg.training-mining": False,
        "c2pa.soft-binding": False,
    }


def test_assertion_placements_validation():
    defaults = AssertionPlacements()
    assert defaults.actions is AssertionPlacement.CREATED
    assert defaults.authors is AssertionPlacement.GATHERED
    assert defaults.training_mining is AssertionPlacement.CREATED
    assert defaults.soft_binding is AssertionPlacement.CREATED
    assert spec().placements == defaults
    with pytest.raises(TypeError):
        AssertionPlacements(authors="created")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        spec(placements={"authors": AssertionPlacement.CREATED})


def test_opened_action_links_the_parent_ingredient():
    manifest = spec(
        actions=(
            ManifestAction(action=ActionType.OPENED),
            ManifestAction(action=ActionType.PLACED, ingredient_ids=("logo",)),
        ),
        ingredients=(
            Ingredient(data=JPEG, title="parent.jpg"),
            Ingredient(
                data=PNG,
                title="logo.png",
                relationship=IngredientRelationship.COMPONENT_OF,
                ingredient_id="logo",
            ),
        ),
    )
    definition, ingredients = build_definition(
        manifest, MediaType.JPEG, (MediaType.JPEG, MediaType.PNG)
    )
    actions = definition["assertions"][0]["data"]["actions"]  # type: ignore[index, call-overload]
    assert actions[0]["parameters"] == {"ingredientIds": ["ingredient-0"]}
    assert actions[1]["parameters"] == {"ingredientIds": ["logo"]}
    assert [i.definition for i in ingredients] == [
        {"title": "parent.jpg", "relationship": "parentOf", "label": "ingredient-0"},
        {"title": "logo.png", "relationship": "componentOf", "label": "logo"},
    ]
    assert [i.mime_type for i in ingredients] == ["image/jpeg", "image/png"]
    assert "\\xff" not in repr(manifest.ingredients[0])  # media bytes stay out of repr


def test_timespan_soft_binding_and_constrained_training_mining():
    binding = SoftBinding(algorithm="com.example.audio-wm.1", value="abc", timespan_ms=(0, 1500))
    assert binding.to_json()["blocks"] == [
        {"scope": {"timespan": {"start": 0, "end": 1500}}, "value": "abc"}
    ]
    policy = TrainingMiningPolicy(
        (
            TrainingMiningEntry(TrainingMiningKey.AI_INFERENCE, TrainingMiningUse.ALLOWED),
            TrainingMiningEntry(
                "com.example.archive", TrainingMiningUse.CONSTRAINED, "research only"
            ),
        )
    )
    definition, _ = build_definition(spec(training_mining=policy), MediaType.PNG, ())
    data = definition["assertions"][1]["data"]  # type: ignore[index, call-overload]
    assert data["entries"]["com.example.archive"] == {
        "use": "constrained",
        "constraint_info": "research only",
    }
    assert not policy.all_not_allowed
    assert policy.use_of("cawg.ai_inference") is TrainingMiningUse.ALLOWED


def standard_policy(*uses: TrainingMiningUse) -> TrainingMiningPolicy:
    keys = (
        TrainingMiningKey.AI_TRAINING,
        TrainingMiningKey.AI_GENERATIVE_TRAINING,
        TrainingMiningKey.DATA_MINING,
        TrainingMiningKey.AI_INFERENCE,
    )
    return TrainingMiningPolicy(
        tuple(TrainingMiningEntry(key, use, "ask") for key, use in zip(keys, uses, strict=False))
    )


def test_constrained_uses_count_as_not_allowed():
    """CAWG TDM 1.1: "constrained shall be treated as equivalent to notAllowed"."""
    not_allowed, constrained, allowed = (
        TrainingMiningUse.NOT_ALLOWED,
        TrainingMiningUse.CONSTRAINED,
        TrainingMiningUse.ALLOWED,
    )
    every_use_constrained = standard_policy(*[constrained] * 4)
    assert every_use_constrained.all_not_allowed
    assert not every_use_constrained.all_declared_not_allowed  # the literal view
    assert every_use_constrained.effective_use_of("cawg.ai_training") is not_allowed
    assert every_use_constrained.use_of("cawg.ai_training") is constrained
    mixed = standard_policy(not_allowed, constrained, not_allowed, constrained)
    assert mixed.all_not_allowed
    assert not mixed.all_declared_not_allowed
    assert TrainingMiningPolicy.not_allowed().all_declared_not_allowed
    assert TrainingMiningPolicy.not_allowed().all_not_allowed
    assert not standard_policy(constrained, constrained, constrained, allowed).all_not_allowed
    assert not standard_policy(constrained, constrained, constrained).all_not_allowed  # 3 of 4
    assert every_use_constrained.effective_use_of("cawg.unknown") is None
    assert standard_policy(allowed).effective_use_of("cawg.ai_training") is allowed


def test_report_flag_treats_constrained_as_not_allowed():
    store = json.loads(store_json())
    store["manifests"]["urn:c2pa:1"]["assertions"] = [
        {
            "label": "cawg.training-mining",
            "data": {
                "entries": {
                    key: {"use": "constrained", "constraint_info": "contact the artist"}
                    for key in (
                        "cawg.ai_training",
                        "cawg.ai_generative_training",
                        "cawg.data_mining",
                        "cawg.ai_inference",
                    )
                }
            },
        }
    ]
    report = parse_manifest_store(json.dumps(store))
    assert report.training_mining_not_allowed
    assert report.training_mining is not None
    assert not report.training_mining.all_declared_not_allowed


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"actions": ()}, "at least one action"),
        ({"actions": (ManifestAction(action=ActionType.PUBLISHED),)}, "first action"),
        ({"actions": (ManifestAction(action=ActionType.CREATED),)}, "digital_source_type"),
        ({"actions": (ManifestAction(action=ActionType.OPENED),)}, "parent ingredient"),
        (
            {"actions": (CREATED, ManifestAction(action=ActionType.WATERMARKED_BOUND))},
            "soft_binding",
        ),
        (
            {"actions": (CREATED, ManifestAction(action=ActionType.PLACED))},
            "ingredient_ids",
        ),
        (
            {"actions": (CREATED, ManifestAction(action="c2pa.removed", ingredient_ids=("x",)))},
            "unknown ingredient",
        ),
        (
            {
                "ingredients": (
                    Ingredient(data=JPEG, title="a"),
                    Ingredient(data=JPEG, title="b"),
                )
            },
            "at most one parent",
        ),
        (
            {
                "ingredients": (
                    Ingredient(data=JPEG, title="a", ingredient_id="x"),
                    Ingredient(
                        data=JPEG,
                        title="b",
                        ingredient_id="x",
                        relationship=IngredientRelationship.INPUT_TO,
                    ),
                )
            },
            "unique",
        ),
        (
            {
                "training_mining": TrainingMiningPolicy(
                    (TrainingMiningEntry("cawg.something_new", TrainingMiningUse.ALLOWED),)
                )
            },
            "reserved",
        ),
        ({"authors": ("",)}, "must not be empty"),
        ({"title": "x" * 5000}, "at most"),
    ],
)
def test_spec_rules(overrides, message):
    with pytest.raises(ValueError, match=message):
        spec(**overrides)


@pytest.mark.parametrize(
    ("factory", "error"),
    [
        (lambda: ManifestAction(action="not a name"), ValueError),
        (lambda: ManifestAction(action=3), TypeError),
        (
            lambda: ManifestAction(action=ActionType.EDITED, when=dt.datetime(2026, 1, 1)),
            ValueError,
        ),
        (
            lambda: ManifestAction(action=ActionType.CREATED, digital_source_type="capture"),
            ValueError,
        ),
        (lambda: SoftBinding(algorithm="trustmark", value="x"), ValueError),
        (lambda: SoftBinding(algorithm="com.example.wm", value=""), ValueError),
        (
            lambda: SoftBinding(algorithm="com.example.wm", value="x", timespan_ms=(5, 1)),
            ValueError,
        ),
        (
            lambda: SoftBinding(algorithm="com.example.wm", value="x", timespan_ms=(True, 1)),
            TypeError,
        ),
        (lambda: Ingredient(data=b"", title="x"), ValueError),
        (lambda: Ingredient(data="text", title="x"), TypeError),
        (lambda: Ingredient(data=b"x", title="x", ingredient_id="bad id!"), ValueError),
        (lambda: GeneratorInfo(""), ValueError),
        (lambda: TrainingMiningPolicy(()), ValueError),
        (lambda: TrainingMiningEntry("k", "notAllowed"), TypeError),
        (lambda: ManifestSpec(claim_generator="x", actions=(CREATED,)), TypeError),
        (lambda: spec(actions=[CREATED, "c2pa.published"]), TypeError),
    ],
)
def test_field_validation(factory, error):
    with pytest.raises(error):
        factory()


def test_custom_action_strings_are_normalized_to_the_enum():
    assert ManifestAction(action="c2pa.edited").action is ActionType.EDITED
    assert ManifestAction(action="com.example.reviewed").action == "com.example.reviewed"
    assert spec(actions=[CREATED]).actions == (CREATED,)  # lists become tuples


# ------------------------------------------------------------------ media types


def test_media_type_resolution():
    assert resolve_media_type(JPEG, None) is MediaType.JPEG
    assert resolve_media_type(JPEG, "image/jpeg; charset=binary") is MediaType.JPEG
    assert resolve_media_type(JPEG, "IMAGE/JPG") is MediaType.JPEG
    with pytest.raises(UnsupportedMediaError, match="does not match"):
        resolve_media_type(JPEG, "image/png")
    with pytest.raises(UnsupportedMediaError, match="unrecognized"):
        resolve_media_type(b"plain text", None)
    with pytest.raises(UnsupportedMediaError, match="not supported"):
        resolve_media_type(b"BM" + b"\x00" * 4 + b"\x00" * 8 + (40).to_bytes(4, "little"), None)
    with pytest.raises(UnsupportedMediaError):
        parse_media_type("application/pdf")
    with pytest.raises(TypeError):
        parse_media_type(42)  # type: ignore[arg-type]


def test_iso_bmff_brands_are_interchangeable():
    mp4 = b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2" + b"\x00" * 32
    assert resolve_media_type(mp4, None) is MediaType.MP4
    assert resolve_media_type(mp4, "audio/mp4") is MediaType.M4A
    assert resolve_media_type(mp4, "video/quicktime") is MediaType.QUICKTIME


def test_id3_wrapped_flac_is_flac():
    # c2pa-rs embeds manifests in FLAC behind an ID3v2 tag.
    tag = b"ID3\x04\x00\x00" + (0).to_bytes(3, "big") + bytes([20]) + b"\x00" * 20
    flac = tag + b"fLaC" + b"\x00" * 40
    assert resolve_media_type(flac, None) is MediaType.FLAC
    mp3 = tag + b"\xff\xfb\x90\x00" + b"\x00" * 40
    assert resolve_media_type(mp3, None) is MediaType.MP3


def test_supported_media_types_are_images_video_and_audio():
    assert MediaType.WEBP in SUPPORTED_MEDIA_TYPES
    assert MediaType.MP4 in SUPPORTED_MEDIA_TYPES
    assert MediaType.WEBM not in SUPPORTED_MEDIA_TYPES  # not supported by c2pa-rs 0.91


# ------------------------------------------------------------------ settings


def test_read_settings_disable_the_network():
    settings = read_settings(trust_anchors_pem=None, tsa_trust_anchors_pem=None)
    assert settings["verify"] == {
        "remote_manifest_fetch": False,
        "ocsp_fetch": False,
        "verify_trust": False,
        "verify_timestamp_trust": False,
    }
    assert settings["core"] == {"allowed_network_hosts": [], "allow_redirects": False}
    assert "trust" not in settings


def test_read_settings_with_trust_anchors():
    settings = read_settings(trust_anchors_pem=FAKE_CHAIN, tsa_trust_anchors_pem=FAKE_CHAIN)
    assert settings["verify"]["verify_trust"] is True  # type: ignore[index, call-overload]
    assert settings["trust"] == {
        "anchors": [
            {"trust_anchors": FAKE_CHAIN, "trust_kind": "manifest"},
            {"trust_anchors": FAKE_CHAIN, "trust_kind": "tsa"},
        ]
    }


def test_sign_settings_allow_only_the_tsa_host():
    assert sign_settings(tsa_url=None)["core"]["allowed_network_hosts"] == []  # type: ignore[index, call-overload]
    settings = sign_settings(tsa_url="https://TSA.example.com:8443/rfc3161")
    assert settings["core"]["allowed_network_hosts"] == ["https://tsa.example.com:8443"]  # type: ignore[index, call-overload]
    assert settings["builder"] == {"thumbnail": {"enabled": False}}


@pytest.mark.parametrize(
    "url",
    [
        "ftp://tsa.example.com",
        "https://user:pw@tsa.example.com",
        "https://[::1]/tsa",
        "https://tsa.example.com:99999",
        "https:///nohost",
        "https://tsa.example.com/#frag",
    ],
)
def test_tsa_url_validation(url):
    with pytest.raises(ValueError):
        tsa_host_pattern(url)


def test_library_error_mapping():
    remote = _read_failure(
        "Remote: must fetch remote manifests from url http://10.0.0.1/m.c2pa", Exception()
    )
    assert remote.remote_url == "http://10.0.0.1/m.c2pa"
    assert (
        _read_failure("Decoding: claim could not be converted", Exception()).unreadable_reason
        == "Decoding"
    )
    with pytest.raises(Exception, match="could not be parsed"):
        _read_failure("Other: asset could not be parsed: Could not parse input JPEG", Exception())
    with pytest.raises(UnsupportedMediaError):
        _read_failure("NotSupported: type is unsupported", Exception())
    rejected = _signing_failure(
        "Verify: manifest failed validation with: failures:\n"
        "  assertion.action.malformed: first action must be created or opened (urn:x)"
    )
    assert isinstance(rejected, SigningError)
    assert "assertion.action.malformed" in str(rejected)
    assert "urn:x" not in str(rejected)
    assert str(_signing_failure("weird text")) == "C2PA signing failed (Other)"


# ------------------------------------------------------------------ report parsing


def store_json(**overrides: object) -> str:
    store: dict[str, object] = {
        "active_manifest": "urn:c2pa:1",
        "manifests": {
            "urn:c2pa:1": {
                "claim_generator_info": [{"name": "app", "version": "2"}, {"name": ""}, "x"],
                "title": "t.jpg",
                "assertions": [
                    {"label": "c2pa.actions.v2", "data": {"actions": [{"action": "c2pa.created"}]}},
                    {
                        "label": "cawg.metadata",
                        "kind": "Json",
                        "data": {"@context": {}, "dc:creator": ["Jane", 3, "  "]},
                    },
                    {
                        "label": "cawg.training-mining",
                        "data": {
                            "entries": {
                                "cawg.ai_training": {"use": "notAllowed"},
                                "cawg.ai_generative_training": {"use": "notAllowed"},
                                "cawg.data_mining": {"use": "notAllowed"},
                                "cawg.ai_inference": {"use": "notAllowed"},
                                "cawg.bogus": {"use": "maybe"},
                                "odd": "not a map",
                            }
                        },
                    },
                    {
                        "label": "c2pa.soft-binding__1",
                        "data": {"alg": "com.adobe.trustmark.Q", "blocks": [{"value": "v"}, 5]},
                    },
                    {"no": "label"},
                ],
                "ingredients": [
                    {
                        "title": "p.jpg",
                        "format": "image/jpeg",
                        "relationship": "parentOf",
                        "active_manifest": "urn:c2pa:0",
                    },
                    "junk",
                ],
                "signature_info": {
                    "alg": "Es256",
                    "issuer": "CA",
                    "common_name": "Signer",
                    "cert_serial_number": "12",
                    "time": "2026-11-05T12:00:00+00:00",
                },
            },
            "urn:c2pa:0": {},
        },
        "validation_results": {
            "activeManifest": {
                "success": [{"code": "claimSignature.validated"}],
                "informational": [{"code": "signingCredential.ocsp.skipped"}],
                "failure": [],
            }
        },
        "validation_state": "Valid",
    }
    store.update(overrides)
    return json.dumps(store)


def test_report_from_a_valid_store():
    report = parse_manifest_store(store_json())
    assert report.state is ValidationState.VALID
    assert report.state.is_valid
    assert report.has_manifest
    assert report.manifest_label == "urn:c2pa:1"
    assert report.title == "t.jpg"
    assert report.claim_generator == (GeneratorInfo("app", "2"),)
    assert report.authors == ("Jane",)
    assert report.training_mining_not_allowed
    assert report.soft_bindings[0].algorithm == "com.adobe.trustmark.Q"
    assert report.soft_bindings[0].values == ("v",)
    assert [a.base_label for a in report.assertions] == [
        "c2pa.actions",
        "cawg.metadata",
        "cawg.training-mining",
        "c2pa.soft-binding",
    ]
    assert report.find_assertions("c2pa.actions")[0].label == "c2pa.actions.v2"
    assert report.signer is not None
    assert report.signer.common_name == "Signer"
    assert report.signer.signing_time == "2026-11-05T12:00:00+00:00"
    assert report.ingredients[0].has_manifest
    assert report.ingredients[0].relationship == "parentOf"
    assert report.informational[0].code == "signingCredential.ocsp.skipped"
    assert report.errors == ()
    assert report.manifest_count == 2


def test_report_state_mapping_fails_closed():
    assert (
        parse_manifest_store(store_json(validation_state="Trusted")).state
        is ValidationState.TRUSTED
    )
    assert (
        parse_manifest_store(store_json(validation_state="Invalid")).state
        is ValidationState.INVALID
    )
    assert (
        parse_manifest_store(store_json(validation_state="Nope")).state is ValidationState.INVALID
    )
    assert parse_manifest_store(store_json(validation_state=None)).state is ValidationState.INVALID
    failing = store_json(
        validation_state="Trusted",
        validation_results={"activeManifest": {"failure": [{"code": "x.y", "url": 1}]}},
    )
    report = parse_manifest_store(failing)
    assert report.state is ValidationState.VALID
    assert report.errors[0].code == "x.y"
    assert report.errors[0].url is None


@pytest.mark.parametrize(
    "text",
    ["not json", "[]", '{"manifests": {"a": {}}, "active_manifest": "b"}', "[" * 100_000],
)
def test_report_from_hostile_json_is_invalid(text):
    report = parse_manifest_store(text)
    assert report.state is ValidationState.INVALID
    assert report.errors[0].code == MANIFEST_UNREADABLE


def test_report_without_manifests_is_unsigned():
    assert parse_manifest_store('{"manifests": {}}').state is ValidationState.UNSIGNED


def test_legacy_assertions_are_understood():
    text = store_json()
    store = json.loads(text)
    store["manifests"]["urn:c2pa:1"]["assertions"] = [
        {
            "label": "c2pa.training-mining",
            "data": {
                "entries": {
                    key: {"use": "notAllowed"}
                    for key in (
                        "c2pa.ai_training",
                        "c2pa.ai_generative_training",
                        "c2pa.data_mining",
                        "c2pa.ai_inference",
                    )
                }
            },
        },
        {
            "label": "stds.schema-org.CreativeWork",
            "data": {"author": [{"name": "Old Author"}, {"x": 1}]},
        },
    ]
    report = parse_manifest_store(json.dumps(store))
    assert report.training_mining_not_allowed
    assert report.authors == ("Old Author",)


# ------------------------------------------------------------------ public API guards


def block_module(monkeypatch, name: str) -> None:
    """Makes ``name`` and its already-imported submodules unimportable."""
    for module in [m for m in sys.modules if m == name or m.startswith(name + ".")]:
        monkeypatch.setitem(sys.modules, module, None)
    monkeypatch.setitem(sys.modules, name, None)


def test_sign_asset_checks_arguments_before_loading_the_library(monkeypatch):
    block_module(monkeypatch, "c2pa")
    signer = FakeSigner()
    with pytest.raises(TypeError):
        sign_asset("text", "image/jpeg", spec(), signer)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        sign_asset(JPEG, "image/jpeg", "spec", signer)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="tsa_url"):
        sign_asset(JPEG, "image/jpeg", spec(), signer, tsa_url="ftp://x")
    with pytest.raises(ValueError):
        sign_asset(JPEG, "image/jpeg", spec(), FakeSigner(chain="nope"))
    with pytest.raises(MediaLimitExceededError):
        sign_asset(JPEG, "image/jpeg", spec(), signer, limits=MediaLimits(max_bytes=10))
    with pytest.raises(UnsupportedMediaError):
        sign_asset(JPEG, "image/png", spec(), signer)
    with pytest.raises(BackendUnavailableError) as info:
        sign_asset(JPEG, "image/jpeg", spec(), signer)
    assert info.value.extra == "c2pa"


def test_read_manifest_checks_arguments_before_loading_the_library(monkeypatch):
    block_module(monkeypatch, "c2pa")
    with pytest.raises(TypeError):
        read_manifest(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="PEM"):
        read_manifest(JPEG, trust_anchors_pem="not a certificate")
    with pytest.raises(MediaLimitExceededError):
        read_manifest(JPEG, limits=MediaLimits(max_bytes=10))
    with pytest.raises(MediaLimitExceededError):
        read_manifest(JPEG, manifest_data=b"x" * 100, limits=MediaLimits(max_bytes=80))
    with pytest.raises(UnsupportedMediaError):
        read_manifest(b"%PDF-1.7 not media")
    with pytest.raises(BackendUnavailableError):
        read_manifest(JPEG)
