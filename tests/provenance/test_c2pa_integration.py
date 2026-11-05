"""Integration tests of the C2PA module against c2pa-python (extra ``c2pa``).

Keys and certificates are generated per test run (tests/provenance/_certs.py).
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime as dt
import io
import logging
import random
import subprocess
import wave
from pathlib import Path

import pytest

from provenance_kit.errors import MediaDecodeError, ProvenanceKitError, UnsupportedMediaError
from provenance_kit.mediatype import MediaType
from provenance_kit.provenance.c2pa import (
    SUPPORTED_MEDIA_TYPES,
    ActionType,
    AssertionPlacement,
    AssertionPlacements,
    DigitalSourceType,
    GeneratorInfo,
    Ingredient,
    LocalKeySigner,
    ManifestAction,
    ManifestSpec,
    SignatureFormat,
    SigningAlgorithm,
    SigningError,
    SoftBinding,
    SoftBindingAlgorithm,
    TrainingMiningPolicy,
    ValidationState,
    ecdsa_der_to_raw,
    read_manifest,
    sign_asset,
)
from tests.support import SEED

pytestmark = pytest.mark.integration

try:
    import c2pa
    import cryptography  # noqa: F401 - LocalKeySigner and the test PKI need it
except Exception:  # missing, or a native library that does not load
    pytest.skip('needs c2pa-python: pip install "provenance-kit[c2pa]"', allow_module_level=True)

from cryptography.hazmat.primitives import hashes, serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import ec  # noqa: E402
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature  # noqa: E402
from PIL import Image  # noqa: E402

from tests.provenance._calendar import MockCalendar  # noqa: E402
from tests.provenance._certs import make_c2pa_pki  # noqa: E402
from tests.provenance._cose import (  # noqa: E402
    assertion_cbor,
    claim_assertion_lists,
    claim_signature,
)

WHEN = dt.datetime(2026, 11, 5, 12, 0, tzinfo=dt.UTC)


@pytest.fixture(scope="module")
def pki():
    return make_c2pa_pki("es256")


@pytest.fixture(scope="module")
def signer(pki):
    return LocalKeySigner(pki.key_pem, pki.leaf_pem)


def image_bytes(fmt: str, size: tuple[int, int] = (64, 48)) -> bytes:
    rng = random.Random(SEED)
    image = Image.new("RGB", size)
    image.putdata([(rng.randrange(256), rng.randrange(256), 80) for _ in range(size[0] * size[1])])
    buffer = io.BytesIO()
    image.save(buffer, fmt)
    return buffer.getvalue()


def wav_bytes() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(8000)
        output.writeframes(bytes(range(256)) * 32)
    return buffer.getvalue()


def full_spec() -> ManifestSpec:
    return ManifestSpec(
        claim_generator=GeneratorInfo("provenance-kit-tests", "0.1"),
        title="asset",
        actions=(
            ManifestAction(
                action=ActionType.CREATED,
                when=WHEN,
                software_agent=GeneratorInfo("capture-tool", "3.2"),
                digital_source_type=DigitalSourceType.DIGITAL_CAPTURE,
            ),
            ManifestAction(action=ActionType.WATERMARKED_BOUND, when=WHEN),
            ManifestAction(action=ActionType.PUBLISHED, when=WHEN),
        ),
        authors=("Jane Doe",),
        training_mining=TrainingMiningPolicy.not_allowed(),
        soft_binding=SoftBinding(algorithm=SoftBindingAlgorithm.TRUSTMARK_Q, value="4*0110"),
    )


def simple_spec() -> ManifestSpec:
    return ManifestSpec(
        claim_generator=GeneratorInfo("provenance-kit-tests"),
        actions=(
            ManifestAction(
                action=ActionType.CREATED, digital_source_type=DigitalSourceType.DIGITAL_CREATION
            ),
        ),
    )


# ------------------------------------------------------------------ formats


def test_supported_media_types_match_the_library():
    builder = set(c2pa.Builder.get_supported_mime_types())
    reader = set(c2pa.Reader.get_supported_mime_types())
    for media_type in SUPPORTED_MEDIA_TYPES:
        assert media_type.mime in builder, media_type
        assert media_type.mime in reader, media_type


@pytest.mark.parametrize(
    ("fmt", "media_type"),
    [
        ("JPEG", MediaType.JPEG),
        ("PNG", MediaType.PNG),
        ("WEBP", MediaType.WEBP),
        ("GIF", MediaType.GIF),
        ("TIFF", MediaType.TIFF),
    ],
)
def test_sign_and_read_images(signer, pki, fmt, media_type):
    data = image_bytes(fmt)
    signed = sign_asset(data, media_type, full_spec(), signer)
    assert signed != data

    report = read_manifest(signed)
    assert report.state is ValidationState.VALID
    assert report.errors == ()
    assert report.title == "asset"
    # Created assertions first, then the gathered ones (the authors, by
    # default); no thumbnail assertion: thumbnails are disabled.
    assert [a.label for a in report.assertions] == [
        "c2pa.actions.v2",
        "cawg.training-mining",
        "c2pa.soft-binding",
        "cawg.metadata",
    ]
    actions = report.find_assertions("c2pa.actions")[0].data
    assert isinstance(actions, dict)
    assert [a["action"] for a in actions["actions"]] == [
        "c2pa.created",
        "c2pa.watermarked.bound",
        "c2pa.published",
    ]
    assert actions["actions"][0]["when"] == "2026-11-05T12:00:00Z"
    assert actions["actions"][0]["softwareAgent"] == {"name": "capture-tool", "version": "3.2"}
    assert report.authors == ("Jane Doe",)
    assert report.training_mining_not_allowed
    assert report.soft_bindings[0].algorithm == "com.adobe.trustmark.Q"
    assert report.soft_bindings[0].values == ("4*0110",)
    assert report.signer is not None
    assert report.signer.common_name == "provenance-kit test signer"
    assert report.signer.algorithm == "Es256"
    assert report.claim_generator[0].name == "provenance-kit-tests"

    trusted = read_manifest(signed, media_type, trust_anchors_pem=pki.root_pem)
    assert trusted.state is ValidationState.TRUSTED


def test_sign_and_read_avif_and_wav(signer):
    samples = [(wav_bytes(), MediaType.WAV)]
    with contextlib.suppress(KeyError, OSError):  # Pillow built without AVIF
        samples.append((image_bytes("AVIF"), MediaType.AVIF))
    for data, media_type in samples:
        signed = sign_asset(data, media_type.mime, simple_spec(), signer)
        assert read_manifest(signed).state is ValidationState.VALID, media_type


def flip_payload_byte(original: bytes, signed: bytes) -> bytes:
    """Flips a byte of the original media payload inside the signed file.

    The manifest has unhashed padding (zeros), so tampering must hit the media
    itself: the first distinctive 16-byte window from two thirds into the
    original that occurs exactly once in the signed file is flipped.
    """
    for start in range(len(original) * 2 // 3, len(original) - 16):
        window = original[start : start + 16]
        if len(set(window)) > 4 and signed.count(window) == 1:
            tampered = bytearray(signed)
            tampered[signed.index(window) + 8] ^= 0xFF
            return bytes(tampered)
    raise AssertionError("no distinctive payload window found")


def _ffmpeg_file(ffmpeg: Path, tmp_path: Path, args: list[str], name: str) -> bytes | None:
    output = tmp_path / name
    result = subprocess.run(
        [str(ffmpeg), "-nostdin", "-hide_banner", "-loglevel", "error", *args, str(output)],
        capture_output=True,
        check=False,
        timeout=120,
    )
    return output.read_bytes() if result.returncode == 0 else None


@pytest.mark.parametrize(
    "case",
    [
        (
            "v.mp4",
            ["-f", "lavfi", "-i", "testsrc=size=64x48:rate=5:duration=1", "-c:v", "mpeg4"],
            MediaType.MP4,
        ),
        (
            "v.mov",
            ["-f", "lavfi", "-i", "testsrc=size=64x48:rate=5:duration=1", "-c:v", "mpeg4"],
            MediaType.QUICKTIME,
        ),
        (
            "v.avi",
            ["-f", "lavfi", "-i", "testsrc=size=64x48:rate=5:duration=1", "-c:v", "mpeg4"],
            MediaType.AVI,
        ),
        (
            "a.m4a",
            ["-f", "lavfi", "-i", "sine=duration=1:sample_rate=8000", "-c:a", "aac"],
            MediaType.M4A,
        ),
        (
            "a.flac",
            ["-f", "lavfi", "-i", "sine=duration=1:sample_rate=8000", "-c:a", "flac"],
            MediaType.FLAC,
        ),
        (
            "a.mp3",
            ["-f", "lavfi", "-i", "sine=duration=1:sample_rate=8000", "-c:a", "libmp3lame"],
            MediaType.MP3,
        ),
    ],
)
def test_sign_and_read_video_and_audio(signer, ffmpeg_path, tmp_path, case):
    name, args, media_type = case
    data = _ffmpeg_file(ffmpeg_path, tmp_path, args, name)
    if data is None:
        pytest.skip(f"this ffmpeg build cannot write {name}")
    signed = sign_asset(data, media_type, simple_spec(), signer)
    report = read_manifest(signed)  # type detected from the bytes (FLAC behind ID3 included)
    assert report.state is ValidationState.VALID
    assert read_manifest(flip_payload_byte(data, signed)).state is ValidationState.INVALID


# ------------------------------------------------------------------ validation outcomes


def test_tampered_asset_is_invalid(signer):
    signed = sign_asset(image_bytes("JPEG"), "image/jpeg", full_spec(), signer)
    tampered = bytearray(signed)
    tampered[-20] ^= 0xFF  # in the entropy-coded image data, covered by the hash
    report = read_manifest(bytes(tampered))
    assert report.state is ValidationState.INVALID
    assert "assertion.dataHash.mismatch" in {e.code for e in report.errors}


def test_stripped_manifest_reads_as_unsigned(signer):
    signed = sign_asset(image_bytes("PNG"), "image/png", full_spec(), signer)
    reencoded = io.BytesIO()
    Image.open(io.BytesIO(signed)).save(reencoded, "PNG")
    report = read_manifest(reencoded.getvalue())
    assert report.state is ValidationState.UNSIGNED
    assert not report.has_manifest
    assert read_manifest(image_bytes("WEBP")).state is ValidationState.UNSIGNED


def test_foreign_trust_anchor_reports_untrusted_signer(signer):
    signed = sign_asset(image_bytes("JPEG"), "image/jpeg", simple_spec(), signer)
    other = make_c2pa_pki("es256")
    report = read_manifest(signed, trust_anchors_pem=other.root_pem)
    assert report.state is ValidationState.VALID
    assert "signingCredential.untrusted" in {e.code for e in report.errors}


def test_corrupted_assets_never_raise_undocumented_errors(signer):
    rng = random.Random(SEED)
    originals = [
        (sign_asset(image_bytes(fmt), mime, simple_spec(), signer), mime)
        for fmt, mime in (("JPEG", "image/jpeg"), ("PNG", "image/png"), ("WEBP", "image/webp"))
    ]
    states = set()
    for signed, mime in originals:
        for _ in range(60):
            data = bytearray(signed)
            if rng.random() < 0.3:
                data = data[: rng.randrange(8, len(data))]
            else:
                for _ in range(rng.randint(1, 8)):
                    data[rng.randrange(len(data))] ^= 1 << rng.randrange(8)
            try:
                states.add(read_manifest(bytes(data), mime).state)
            except (MediaDecodeError, UnsupportedMediaError):
                states.add("rejected")
    assert ValidationState.INVALID in states or "rejected" in states


def test_remote_manifest_references_are_not_fetched(signer, pki):
    with MockCalendar(mode="garbage") as canary:
        remote_url = f"{canary.url}/manifest.c2pa"
        native_signer = c2pa.Signer.from_callback(
            signer.sign, c2pa.C2paSigningAlg.ES256, pki.leaf_pem, None
        )
        definition = {
            "claim_generator_info": [{"name": "tests"}],
            "assertions": [
                {
                    "label": "c2pa.actions",
                    "data": {
                        "actions": [
                            {
                                "action": "c2pa.created",
                                "digitalSourceType": DigitalSourceType.DIGITAL_CREATION,
                            }
                        ]
                    },
                }
            ],
        }
        settings = {"builder": {"thumbnail": {"enabled": False}}}
        with c2pa.Context.from_dict(settings) as context:
            builder = c2pa.Builder(definition, context)
            builder.set_no_embed()
            builder.set_remote_url(remote_url)
            output = io.BytesIO()
            store = builder.sign(
                native_signer, "image/jpeg", io.BytesIO(image_bytes("JPEG")), output
            )
        asset = output.getvalue()

        report = read_manifest(asset)
        assert report.state is ValidationState.UNSIGNED
        assert report.remote_manifest_url == remote_url
        assert canary.requests == []  # the library did not fetch it

        # The caller may fetch it (with its own SSRF rules) and pass it in.
        sidecar = read_manifest(asset, manifest_data=store)
        assert sidecar.state is ValidationState.VALID


def test_parent_ingredient_with_its_own_manifest(signer):
    parent = sign_asset(image_bytes("JPEG"), "image/jpeg", full_spec(), signer)
    child_spec = ManifestSpec(
        claim_generator=GeneratorInfo("provenance-kit-tests"),
        actions=(
            ManifestAction(action=ActionType.OPENED),
            ManifestAction(action=ActionType.RESIZED),
            ManifestAction(action=ActionType.PUBLISHED),
        ),
        ingredients=(Ingredient(data=parent, title="original.jpg"),),
    )
    child = sign_asset(image_bytes("JPEG", (32, 24)), "image/jpeg", child_spec, signer)
    report = read_manifest(child)
    assert report.state is ValidationState.VALID
    assert report.manifest_count == 2
    assert report.ingredients[0].title == "original.jpg"
    assert report.ingredients[0].relationship == "parentOf"
    assert report.ingredients[0].has_manifest
    opened = report.find_assertions("c2pa.actions")[0].data["actions"][0]  # type: ignore[index, call-overload]
    assert opened["parameters"]["ingredients"]  # linked by hashed URI


PLACEMENTS = {
    "defaults": (
        AssertionPlacements(),
        ["c2pa.actions.v2", "cawg.training-mining", "c2pa.soft-binding"],
        ["cawg.metadata"],
    ),
    "inverted": (
        AssertionPlacements(
            actions=AssertionPlacement.GATHERED,
            authors=AssertionPlacement.CREATED,
            training_mining=AssertionPlacement.GATHERED,
            soft_binding=AssertionPlacement.GATHERED,
        ),
        ["cawg.metadata"],
        ["c2pa.actions.v2", "cawg.training-mining", "c2pa.soft-binding"],
    ),
}


@pytest.mark.parametrize("name", sorted(PLACEMENTS))
def test_created_and_gathered_assertions(signer, name):
    """c2pa-rs 0.91 writes and validates both lists of a v2 claim (C2PA 2.3, 10.2.2)."""
    placements, created, gathered = PLACEMENTS[name]
    spec = dataclasses.replace(full_spec(), placements=placements)
    signed = sign_asset(image_bytes("PNG"), "image/png", spec, signer)
    created_labels, gathered_labels = claim_assertion_lists(signed)
    # The library adds the hard binding, always as a created assertion.
    assert sorted(created_labels) == sorted([*created, "c2pa.hash.data"])
    assert sorted(gathered_labels) == sorted(gathered)
    report = read_manifest(signed)
    assert report.state is ValidationState.VALID, report.errors
    assert report.authors == ("Jane Doe",)
    assert report.training_mining_not_allowed


def test_soft_binding_value_is_written_as_text(signer):
    """c2pa-python 0.38 writes JSON-defined assertions as generic CBOR: no byte strings."""
    signed = sign_asset(image_bytes("PNG"), "image/png", full_spec(), signer)
    stored = assertion_cbor(signed, "c2pa.soft-binding")
    assert stored == {"alg": "com.adobe.trustmark.Q", "blocks": [{"scope": {}, "value": "4*0110"}]}


# ------------------------------------------------------------------ signature encodings


class FakeKms:
    """Stands in for a KMS: holds the key elsewhere and returns DER ECDSA signatures."""

    signature_format = SignatureFormat.DER

    def __init__(self, pki, algorithm=SigningAlgorithm.ES256, hash_class=hashes.SHA256) -> None:
        self._key = pki.key
        self.algorithm = algorithm
        self.certificate_chain_pem = pki.leaf_pem
        self._hash = hash_class
        self.calls = 0

    def sign(self, data: bytes) -> bytes:
        self.calls += 1
        return self._key.sign(data, ec.ECDSA(self._hash()))


@pytest.mark.parametrize(
    ("kind", "algorithm", "hash_class", "width"),
    [
        ("es256", SigningAlgorithm.ES256, hashes.SHA256, 32),
        ("es384", SigningAlgorithm.ES384, hashes.SHA384, 48),
        ("es512", SigningAlgorithm.ES512, hashes.SHA512, 66),
    ],
)
def test_kms_der_signatures_are_stored_as_raw_r_s(kind, algorithm, hash_class, width):
    pki = make_c2pa_pki(kind)
    kms = FakeKms(pki, algorithm, hash_class)
    signed = sign_asset(image_bytes("PNG"), "image/png", simple_spec(), kms)
    assert kms.calls == 1
    assert len(claim_signature(signed)) == 2 * width  # COSE: r || s, RFC 9053 section 2.1
    assert read_manifest(signed, trust_anchors_pem=pki.root_pem).state is ValidationState.TRUSTED


@pytest.mark.parametrize(("kind", "length"), [("es256", 64), ("ed25519", 64), ("rsa", 256)])
def test_local_key_signer_algorithms(kind, length):
    pki = make_c2pa_pki(kind)
    signer = LocalKeySigner(pki.key_pem, pki.leaf_pem)
    assert signer.signature_format is SignatureFormat.RAW
    signed = sign_asset(image_bytes("PNG"), "image/png", simple_spec(), signer)
    assert len(claim_signature(signed)) == length
    assert read_manifest(signed, trust_anchors_pem=pki.root_pem).state is ValidationState.TRUSTED


def test_library_callback_also_accepts_der_unconverted(pki):
    """Observation for decision 6 (c2pa-python 0.38 / c2pa-rs 0.91).

    The native signer normalizes a DER ECDSA signature returned by the
    callback to r || s itself, so both encodings validate. The adapter still
    converts, so that correctness does not depend on this leniency.
    """

    def der_callback(data: bytes) -> bytes:
        return pki.key.sign(data, ec.ECDSA(hashes.SHA256()))

    native_signer = c2pa.Signer.from_callback(
        der_callback, c2pa.C2paSigningAlg.ES256, pki.leaf_pem, None
    )
    definition = {
        "claim_generator_info": [{"name": "tests"}],
        "assertions": [
            {
                "label": "c2pa.actions",
                "data": {
                    "actions": [
                        {
                            "action": "c2pa.created",
                            "digitalSourceType": DigitalSourceType.DIGITAL_CREATION,
                        }
                    ]
                },
            }
        ],
    }
    with c2pa.Context.from_dict({"builder": {"thumbnail": {"enabled": False}}}) as context:
        output = io.BytesIO()
        c2pa.Builder(definition, context).sign(
            native_signer, "image/png", io.BytesIO(image_bytes("PNG")), output
        )
    assert len(claim_signature(output.getvalue())) == 64
    assert read_manifest(output.getvalue()).state is ValidationState.VALID


def test_der_converter_matches_cryptography(pki):
    for index in range(50):
        signature = pki.key.sign(b"message %d" % index, ec.ECDSA(hashes.SHA256()))
        r, s = decode_dss_signature(signature)
        assert ecdsa_der_to_raw(signature, SigningAlgorithm.ES256) == r.to_bytes(
            32, "big"
        ) + s.to_bytes(32, "big")


def test_signer_declaring_raw_but_returning_der_is_refused(pki):
    kms = FakeKms(pki)
    kms.signature_format = SignatureFormat.RAW  # wrong declaration
    with pytest.raises(SigningError, match="64 bytes"):
        sign_asset(image_bytes("JPEG"), "image/jpeg", simple_spec(), kms)


def test_signer_failure_does_not_leak_into_messages_or_logs(pki, caplog):
    detail = "kms-key-arn-1234567890"

    class BrokenKms(FakeKms):
        def sign(self, data: bytes) -> bytes:
            raise PermissionError(f"access denied for {detail}")

    caplog.set_level(logging.DEBUG)
    with pytest.raises(SigningError) as info:
        sign_asset(image_bytes("JPEG"), "image/jpeg", simple_spec(), BrokenKms(pki))
    assert detail not in str(info.value)
    assert isinstance(info.value.__cause__, PermissionError)
    assert all(detail not in record.getMessage() for record in caplog.records)


def test_manifest_rejected_by_the_library_is_a_signing_error(pki):
    no_eku = make_c2pa_pki("es256", with_eku=False)
    signer = LocalKeySigner(no_eku.key_pem, no_eku.leaf_pem)
    with pytest.raises(SigningError):
        sign_asset(image_bytes("JPEG"), "image/jpeg", simple_spec(), signer)


@pytest.mark.parametrize("key_usage", ["nonRepudiation", "none"])
def test_leaf_without_digital_signature_is_refused_for_every_signer(key_usage):
    """C2PA 2.3, section 14.5.1: claim-signing certificates assert digitalSignature.

    c2pa-rs 0.91 signs with a nonRepudiation-only leaf and reads the result
    back as trusted; the adapter refuses such a leaf before the library sees
    it, for KMS-style signers and for LocalKeySigner.
    """
    bad = make_c2pa_pki("es256", key_usage=key_usage)
    with pytest.raises(ValueError, match="digitalSignature"):
        LocalKeySigner(bad.key_pem, bad.leaf_pem)
    kms = FakeKms(bad)
    with pytest.raises(ValueError, match="digitalSignature"):
        sign_asset(image_bytes("JPEG"), "image/jpeg", simple_spec(), kms)
    assert kms.calls == 0


def test_corrupt_asset_or_ingredient_is_a_media_error(signer):
    corrupt = image_bytes("JPEG")[:40]
    with pytest.raises(MediaDecodeError):
        sign_asset(corrupt, "image/jpeg", simple_spec(), signer)
    spec = ManifestSpec(
        claim_generator=GeneratorInfo("tests"),
        actions=(ManifestAction(action=ActionType.OPENED),),
        ingredients=(Ingredient(data=corrupt, title="broken.jpg"),),
    )
    with pytest.raises(ProvenanceKitError):
        sign_asset(image_bytes("JPEG"), "image/jpeg", spec, signer)


# ------------------------------------------------------------------ LocalKeySigner


def not_a_key_pem() -> bytes:
    """A PEM block labelled as a private key whose body is not a key.

    The label is assembled at run time so that the file never contains the
    literal PEM marker, which secret scanners (GitHub push protection
    included) treat as a leaked key.
    """
    label = " ".join(("PRIVATE", "KEY"))
    return f"-----BEGIN {label}-----\nAAAA\n-----END {label}-----\n".encode()


def test_local_key_signer_checks_the_key_and_chain(pki):
    other = make_c2pa_pki("es256")
    with pytest.raises(ValueError, match="does not match"):
        LocalKeySigner(pki.key_pem, other.leaf_pem)
    with pytest.raises(ValueError, match="es256 only"):
        LocalKeySigner(pki.key_pem, pki.leaf_pem, algorithm=SigningAlgorithm.ES384)
    with pytest.raises(ValueError, match="private key"):
        LocalKeySigner(not_a_key_pem(), pki.leaf_pem)
    with pytest.raises(ValueError, match="certificate chain"):
        LocalKeySigner(pki.key_pem, "no certificates here")
    encrypted = pki.key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.BestAvailableEncryption(b"pass phrase"),
    )
    signer = LocalKeySigner(encrypted.decode(), pki.leaf_pem, password=b"pass phrase")
    assert repr(signer) == "LocalKeySigner(algorithm='es256')"
    rsa = make_c2pa_pki("rsa")
    assert (
        LocalKeySigner(rsa.key_pem, rsa.leaf_pem, algorithm=SigningAlgorithm.PS512).algorithm
        is SigningAlgorithm.PS512
    )


# ------------------------------------------------------------------ network (TSA)


@pytest.mark.network
def test_signing_with_a_public_tsa(signer):
    signed = sign_asset(
        image_bytes("JPEG"),
        "image/jpeg",
        simple_spec(),
        signer,
        tsa_url="http://timestamp.digicert.com",
    )
    report = read_manifest(signed)
    assert report.state is ValidationState.VALID
    assert report.signer is not None
    assert report.signer.signing_time is not None
