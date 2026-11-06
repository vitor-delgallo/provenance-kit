"""Unit tests of the signing-certificate checks: need cryptography only, not c2pa-python.

``check_signing_certificate`` and ``LocalKeySigner`` refuse a leaf that C2PA
does not allow to sign claims. They parse the chain themselves, so these
tests run wherever ``cryptography`` is installed (extra ``c2pa``), even when
c2pa-python's native library does not load.
"""

from __future__ import annotations

import pytest

from provenance_kit.provenance.c2pa import LocalKeySigner
from tests.support import require_module

require_module("cryptography", extra="c2pa")

from cryptography import x509  # noqa: E402

from provenance_kit.provenance.c2pa._signer import check_signing_certificate  # noqa: E402
from tests.provenance._certs import (  # noqa: E402
    leaf_with_raw_key_usage,
    make_c2pa_pki,
    pem_block,
)

DIGITAL_SIGNATURE = bytes.fromhex("03020780")  # BIT STRING, bit 0 set
NON_REPUDIATION = bytes.fromhex("03020640")  # BIT STRING, bit 1 set


@pytest.fixture(scope="module")
def good():
    return make_c2pa_pki("es256")


@pytest.fixture(scope="module")
def nonrep():
    return make_c2pa_pki("es256", key_usage="nonRepudiation")


def test_a_valid_chain_is_returned_in_canonical_form(good):
    chain = good.leaf_pem + good.root_pem
    assert check_signing_certificate(chain) == chain  # cryptography's own encoding
    messy = (
        "\r\n  " + chain.replace("\n", "\r\n").replace("-----\r\nMII", "-----\r\n\r\nMII") + "\n\n"
    )
    assert check_signing_certificate(messy) == chain


@pytest.mark.parametrize("key_usage", ["nonRepudiation", "none"])
def test_a_leaf_without_digital_signature_is_refused(key_usage):
    bad = make_c2pa_pki("es256", key_usage=key_usage)
    with pytest.raises(ValueError, match="digitalSignature"):
        check_signing_certificate(bad.leaf_pem)
    with pytest.raises(ValueError, match="digitalSignature"):
        LocalKeySigner(bad.key_pem, bad.leaf_pem)


@pytest.mark.parametrize("label", ["TRUSTED CERTIFICATE", "FOO", "X509 CERTIFICATE", "certificate"])
def test_only_certificate_blocks_are_accepted(good, nonrep, label):
    """R2-S03: cryptography skips blocks with other labels; c2pa-rs signs with them.

    A nonRepudiation-only leaf under another label, followed by a valid
    certificate, used to pass the check on the second block while the C2PA
    library used the first.
    """
    chain = pem_block(nonrep.leaf_pem, label) + good.leaf_pem
    with pytest.raises(ValueError, match="CERTIFICATE"):
        check_signing_certificate(chain)
    with pytest.raises(ValueError, match="CERTIFICATE"):
        LocalKeySigner(good.key_pem, good.leaf_pem + pem_block(good.root_pem, label))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda pem: "Bag Attributes\n" + pem,  # text before the first block
        lambda pem: pem + "trailing text\n",  # text after the last block
        lambda pem: pem.replace("-----\n", "-----\nProc-Type: 4,ENCRYPTED\n", 1),  # RFC 1421 header
        lambda pem: pem.replace("-----END CERTIFICATE-----", "-----END FOO-----"),  # mismatch
        lambda pem: pem.replace("-----END CERTIFICATE-----", ""),  # no end line
        lambda pem: pem[:40] + "*" + pem[41:],  # not base64
        lambda pem: pem[:40] + "é" + pem[41:],  # not ASCII
        lambda pem: pem.replace("MII", "MIJ", 1),  # base64, but not a certificate
        lambda _pem: "",
        lambda _pem: "   \n",
    ],
)
def test_malformed_chains_are_refused(good, mutate):
    with pytest.raises(ValueError, match="certificate chain"):
        check_signing_certificate(mutate(good.leaf_pem + good.root_pem))


def test_text_between_blocks_is_refused(good):
    with pytest.raises(ValueError, match="certificate chain"):
        check_signing_certificate(good.leaf_pem + "subject=CN = root\n" + good.root_pem)


@pytest.mark.parametrize(
    ("values", "accepted"),
    [
        ([(DIGITAL_SIGNATURE, True)], True),
        ([(DIGITAL_SIGNATURE, False)], True),  # C2PA asks for the bit, not for criticality
        ([(bytes.fromhex("030300ffff"), True)], True),  # extra bits, digitalSignature among them
        ([(NON_REPUDIATION, True)], False),
        ([(bytes.fromhex("030100"), True)], False),  # no bit set
        ([(bytes.fromhex("040100"), True)], False),  # an OCTET STRING, not a BIT STRING
        ([(DIGITAL_SIGNATURE, True), (NON_REPUDIATION, True)], False),  # R2-S07: two of them
        ([(NON_REPUDIATION, True), (DIGITAL_SIGNATURE, True)], False),
    ],
)
def test_hostile_key_usage_encodings_raise_value_error(values, accepted):
    """R2-S07: every extension-parsing failure is a ValueError, never cryptography's own type."""
    _, chain = leaf_with_raw_key_usage(values)
    if accepted:
        assert check_signing_certificate(chain) == chain
        return
    with pytest.raises(ValueError) as info:
        check_signing_certificate(chain)
    assert type(info.value) is ValueError


def test_duplicate_key_usage_is_a_value_error_for_every_signer():
    key_pem, chain = leaf_with_raw_key_usage([(DIGITAL_SIGNATURE, True), (DIGITAL_SIGNATURE, True)])
    with pytest.raises(x509.DuplicateExtension):  # what cryptography raises on its own
        _ = x509.load_pem_x509_certificate(chain.encode("ascii")).extensions
    with pytest.raises(ValueError, match="extensions") as info:
        check_signing_certificate(chain)
    assert type(info.value) is ValueError
    assert isinstance(info.value.__cause__, x509.DuplicateExtension)
    with pytest.raises(ValueError, match="extensions") as info:
        LocalKeySigner(key_pem, chain)
    assert type(info.value) is ValueError


def test_local_key_signer_keeps_the_canonical_chain(good):
    messy = good.leaf_pem.replace("\n", "\r\n") + "\n" + good.root_pem
    signer = LocalKeySigner(good.key_pem, messy)
    assert signer.certificate_chain_pem == good.leaf_pem + good.root_pem
