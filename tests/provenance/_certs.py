"""Test certificates, generated at test time with ``cryptography`` (never committed).

* :func:`make_c2pa_pki` builds a root CA and a C2PA claim-signing leaf with the
  profile c2pa-rs checks (C2PA 2.x, "Certificate profile"): X.509 v3, leaf not
  a CA, ``digitalSignature`` key usage (critical), extended key usages
  ``id-kp-emailProtection`` and ``id-kp-documentSigning``, authority and
  subject key identifiers (or, for negative tests, a leaf without the EKUs
  or without ``digitalSignature``).
* :func:`make_tls_contexts` builds a CA and a ``127.0.0.1`` server certificate
  for HTTPS tests.
* :func:`leaf_with_raw_key_usage` builds a certificate whose key usage
  extensions are given as raw DER, for encodings ``cryptography``'s builder
  refuses to produce (two key usage extensions, a value that is not a BIT
  STRING).
* :func:`pem_block` wraps a certificate in a PEM block with any label.

Import this module only after ``require_module("cryptography", ...)``.
"""

from __future__ import annotations

import base64
import datetime as dt
import ipaddress
import ssl
import textwrap
from dataclasses import dataclass
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519, rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID, ObjectIdentifier

ID_KP_DOCUMENT_SIGNING = ObjectIdentifier("1.3.6.1.5.5.7.3.36")


@dataclass(frozen=True)
class C2paPki:
    """A root CA and a claim-signing leaf."""

    key: object
    key_pem: bytes
    leaf_pem: str
    root_pem: str


def _name(common_name: str) -> x509.Name:
    return x509.Name(
        [
            x509.NameAttribute(NameOID.COMMON_NAME, common_name),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "provenance-kit tests"),
        ]
    )


def _key_usage(*, signing: bool, non_repudiation: bool = False) -> x509.KeyUsage:
    ca = not signing and not non_repudiation
    return x509.KeyUsage(
        digital_signature=signing,
        content_commitment=non_repudiation,
        key_encipherment=False,
        data_encipherment=False,
        key_agreement=False,
        key_cert_sign=ca,
        crl_sign=ca,
        encipher_only=False,
        decipher_only=False,
    )


def _pem(certificate: x509.Certificate) -> str:
    return certificate.public_bytes(serialization.Encoding.PEM).decode("ascii")


def _root(
    now: dt.datetime, common_name: str
) -> tuple[ec.EllipticCurvePrivateKey, x509.Certificate]:
    key = ec.generate_private_key(ec.SECP256R1())
    name = _name(common_name)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(days=1))
        .not_valid_after(now + dt.timedelta(days=30))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(_key_usage(signing=False), critical=True)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(key.public_key()), critical=False
        )
        .sign(key, hashes.SHA256())
    )
    return key, certificate


def make_c2pa_pki(
    kind: str = "es256", *, with_eku: bool = True, key_usage: str = "digitalSignature"
) -> C2paPki:
    """Returns a fresh root CA and a C2PA leaf with a ``kind`` key.

    ``kind`` is ``es256``, ``es384``, ``es512``, ``ed25519`` or ``rsa``.
    ``key_usage`` is ``digitalSignature`` (the C2PA profile),
    ``nonRepudiation`` (that bit only, no ``digitalSignature``) or ``none``
    (no key usage extension); the last two are for negative tests.
    """
    now = dt.datetime.now(dt.UTC)
    root_key, root = _root(now, "provenance-kit test root")
    if kind == "es256":
        key = ec.generate_private_key(ec.SECP256R1())
    elif kind == "es384":
        key = ec.generate_private_key(ec.SECP384R1())
    elif kind == "es512":
        key = ec.generate_private_key(ec.SECP521R1())
    elif kind == "ed25519":
        key = ed25519.Ed25519PrivateKey.generate()
    elif kind == "rsa":
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    else:
        raise ValueError(kind)
    builder = (
        x509.CertificateBuilder()
        .subject_name(_name("provenance-kit test signer"))
        .issuer_name(root.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(days=1))
        .not_valid_after(now + dt.timedelta(days=30))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(root_key.public_key()),
            critical=False,
        )
    )
    if key_usage == "digitalSignature":
        builder = builder.add_extension(_key_usage(signing=True), critical=True)
    elif key_usage == "nonRepudiation":
        builder = builder.add_extension(
            _key_usage(signing=False, non_repudiation=True), critical=True
        )
    elif key_usage != "none":
        raise ValueError(key_usage)
    if with_eku:
        builder = builder.add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.EMAIL_PROTECTION, ID_KP_DOCUMENT_SIGNING]),
            critical=False,
        )
    leaf = builder.sign(root_key, hashes.SHA256())
    key_pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    return C2paPki(key=key, key_pem=key_pem, leaf_pem=_pem(leaf), root_pem=_pem(root))


def make_tls_contexts(directory: Path) -> tuple[ssl.SSLContext, ssl.SSLContext]:
    """Returns ``(server_context, client_context)`` for ``https://127.0.0.1``.

    The server certificate is issued by a fresh CA that only the client
    context trusts. Key files are written to ``directory`` (a pytest
    ``tmp_path``) because ``ssl`` loads server keys from files.
    """
    now = dt.datetime.now(dt.UTC)
    root_key, root = _root(now, "provenance-kit TLS test root")
    key = ec.generate_private_key(ec.SECP256R1())
    certificate = (
        x509.CertificateBuilder()
        .subject_name(_name("127.0.0.1"))
        .issuer_name(root.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(days=1))
        .not_valid_after(now + dt.timedelta(days=30))
        .add_extension(
            x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]),
            critical=False,
        )
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(_key_usage(signing=True), critical=True)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        # Python 3.13's default context verifies with VERIFY_X509_STRICT,
        # which refuses an issued certificate without an authority key ID.
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(root_key.public_key()),
            critical=False,
        )
        .sign(root_key, hashes.SHA256())
    )
    cert_file = directory / "server-cert.pem"
    key_file = directory / "server-key.pem"
    cert_file.write_text(_pem(certificate), encoding="ascii")
    key_file.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    server = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server.load_cert_chain(cert_file, key_file)
    client = ssl.create_default_context(cadata=_pem(root))
    return server, client


def pem_block(certificate_pem: str, label: str) -> str:
    """Re-wraps the first certificate of ``certificate_pem`` under another PEM label."""
    certificate = x509.load_pem_x509_certificate(certificate_pem.encode("ascii"))
    der = certificate.public_bytes(serialization.Encoding.DER)
    body = "\n".join(textwrap.wrap(base64.b64encode(der).decode("ascii"), 64))
    return f"-----BEGIN {label}-----\n{body}\n-----END {label}-----\n"


#: Placeholder OIDs whose DER encoding has the length of id-ce-keyUsage (2.5.29.15).
_PLACEHOLDERS = (ObjectIdentifier("2.5.29.98"), ObjectIdentifier("2.5.29.99"))
_KEY_USAGE_OID_DER = bytes.fromhex("0603551d0f")


def _der_tlv(tag: int, body: bytes) -> bytes:
    if len(body) < 0x80:
        length = bytes([len(body)])
    else:
        raw = len(body).to_bytes((len(body).bit_length() + 7) // 8, "big")
        length = bytes([0x80 | len(raw)]) + raw
    return bytes([tag]) + length + body


def leaf_with_raw_key_usage(values: list[tuple[bytes, bool]]) -> tuple[bytes, str]:
    """Returns ``(key_pem, certificate_pem)``: a P-256 leaf with hand-encoded key usages.

    Each ``(extn_value, critical)`` becomes one key usage extension whose
    ``extnValue`` is the given DER (for example ``bytes.fromhex("03020780")``,
    a BIT STRING with ``digitalSignature``). The builder refuses duplicate or
    malformed extensions, so they are added under placeholder OIDs of the
    same length, patched to id-ce-keyUsage in the TBS certificate, and the
    certificate is signed again.
    """
    if not 1 <= len(values) <= len(_PLACEHOLDERS):
        raise ValueError("one or two key usage extensions")
    issuer_key = ec.generate_private_key(ec.SECP256R1())
    key = ec.generate_private_key(ec.SECP256R1())
    now = dt.datetime.now(dt.UTC)
    builder = (
        x509.CertificateBuilder()
        .subject_name(_name("provenance-kit raw key usage"))
        .issuer_name(_name("provenance-kit raw key usage"))
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(days=1))
        .not_valid_after(now + dt.timedelta(days=30))
    )
    for placeholder, (value, critical) in zip(_PLACEHOLDERS, values, strict=False):
        builder = builder.add_extension(
            x509.UnrecognizedExtension(placeholder, value), critical=critical
        )
    tbs = builder.sign(issuer_key, hashes.SHA256()).tbs_certificate_bytes
    for placeholder in _PLACEHOLDERS[: len(values)]:
        encoded = bytes.fromhex("0603551d") + bytes([int(placeholder.dotted_string.split(".")[-1])])
        assert tbs.count(encoded) == 1
        tbs = tbs.replace(encoded, _KEY_USAGE_OID_DER)
    signature = issuer_key.sign(tbs, ec.ECDSA(hashes.SHA256()))  # DER Ecdsa-Sig-Value
    ecdsa_with_sha256 = bytes.fromhex("300a06082a8648ce3d040302")
    der = _der_tlv(0x30, tbs + ecdsa_with_sha256 + _der_tlv(0x03, b"\x00" + signature))
    key_pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    return key_pem, _pem(x509.load_der_x509_certificate(der))
