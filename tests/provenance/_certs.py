"""Test certificates, generated at test time with ``cryptography`` (never committed).

* :func:`make_c2pa_pki` builds a root CA and a C2PA claim-signing leaf with the
  profile c2pa-rs checks (C2PA 2.x, "Certificate profile"): X.509 v3, leaf not
  a CA, ``digitalSignature`` key usage (critical), extended key usages
  ``id-kp-emailProtection`` and ``id-kp-documentSigning``, authority and
  subject key identifiers (or, for negative tests, a leaf without the EKUs
  or without ``digitalSignature``).
* :func:`make_tls_contexts` builds a CA and a ``127.0.0.1`` server certificate
  for HTTPS tests.

Import this module only after ``require_module("cryptography", ...)``.
"""

from __future__ import annotations

import datetime as dt
import ipaddress
import ssl
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
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
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
