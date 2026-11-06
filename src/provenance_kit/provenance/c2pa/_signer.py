"""Signer protocol, signature encodings and an in-process key signer."""

from __future__ import annotations

import base64
import binascii
import enum
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

from provenance_kit._optional import import_optional
from provenance_kit._typing import BytesLike
from provenance_kit.errors import ProvenanceKitError

if TYPE_CHECKING:
    from cryptography import x509
    from cryptography.hazmat.primitives.asymmetric.types import PrivateKeyTypes

__all__ = [
    "C2paSigner",
    "LocalKeySigner",
    "SignatureFormat",
    "SigningAlgorithm",
    "SigningError",
    "ecdsa_der_to_raw",
]

_PEM_CERTIFICATE = "-----BEGIN CERTIFICATE-----"
# A PEM block of any label; the label and the body are checked afterwards.
_PEM_BLOCK = re.compile(
    r"-----BEGIN (?P<begin>[^\r\n]*?)-----(?P<body>.*?)-----END (?P<end>[^\r\n]*?)-----",
    re.DOTALL,
)
_PEM_SPACE = " \t\r\n"
_PEM_SPACE_RE = re.compile(r"[ \t\r\n]+")
_CHAIN_FORMAT = (
    "the certificate chain must consist of PEM blocks labelled CERTIFICATE, "
    "separated by whitespace only"
)
# RSA moduli from 2048 bits (the C2PA minimum) to 8192 bits.
_RSA_SIGNATURE_BYTES = (256, 1024)


class SigningError(ProvenanceKitError):
    """A C2PA manifest could not be signed.

    Raised when the signer raises or returns an unusable signature, when the
    time-stamp authority fails, or when the C2PA library refuses the manifest
    (it validates every manifest right after signing it). The message names
    the failure only; the original exception, if any, is chained as
    ``__cause__``. Exceptions raised by a signer are never put into the
    message, because a KMS error can name keys or accounts.
    """


class SigningAlgorithm(enum.StrEnum):
    """A C2PA claim signature algorithm (COSE names, C2PA 2.3, section 13.2.1)."""

    ES256 = "es256"
    """ECDSA with P-256 and SHA-256."""
    ES384 = "es384"
    """ECDSA with P-384 and SHA-384."""
    ES512 = "es512"
    """ECDSA with P-521 and SHA-512."""
    PS256 = "ps256"
    """RSASSA-PSS with SHA-256, MGF1 with SHA-256, salt length 32."""
    PS384 = "ps384"
    """RSASSA-PSS with SHA-384, MGF1 with SHA-384, salt length 48."""
    PS512 = "ps512"
    """RSASSA-PSS with SHA-512, MGF1 with SHA-512, salt length 64."""
    ED25519 = "ed25519"
    """EdDSA with Ed25519 (pure, no pre-hash)."""

    @property
    def is_ecdsa(self) -> bool:
        """Whether this is one of the ECDSA algorithms."""
        return self in _ECDSA_COMPONENT_BYTES

    @property
    def ecdsa_component_bytes(self) -> int | None:
        """Bytes per ECDSA component (``r`` or ``s``), or ``None`` for non-ECDSA."""
        return _ECDSA_COMPONENT_BYTES.get(self)

    @property
    def hash_name(self) -> str | None:
        """The hash the signer applies to the data, or ``None`` for Ed25519."""
        return _HASH_NAMES.get(self)


_ECDSA_COMPONENT_BYTES = {
    SigningAlgorithm.ES256: 32,
    SigningAlgorithm.ES384: 48,
    SigningAlgorithm.ES512: 66,
}
_HASH_NAMES = {
    SigningAlgorithm.ES256: "sha256",
    SigningAlgorithm.ES384: "sha384",
    SigningAlgorithm.ES512: "sha512",
    SigningAlgorithm.PS256: "sha256",
    SigningAlgorithm.PS384: "sha384",
    SigningAlgorithm.PS512: "sha512",
}


class SignatureFormat(enum.StrEnum):
    """The encoding of the signature bytes a :class:`C2paSigner` returns.

    COSE (RFC 9053, section 2.1), and therefore C2PA, stores ECDSA signatures
    as the fixed-width concatenation ``r || s`` (IEEE P1363). KMS and HSM
    APIs usually return the ASN.1 DER ``Ecdsa-Sig-Value`` instead. A signer
    states which one it returns, and the adapter converts DER to ``r || s``
    before the signature reaches the C2PA library.

    Ed25519 (64 bytes, RFC 8032) and RSASSA-PSS (the modulus-sized octet
    string, RFC 8017) have a single encoding: their signers use ``RAW``.
    """

    RAW = "raw"
    """``r || s`` for ECDSA; the only encoding for Ed25519 and RSASSA-PSS."""
    DER = "der"
    """ASN.1 DER ``SEQUENCE { r INTEGER, s INTEGER }`` (ECDSA only)."""


class C2paSigner(Protocol):
    """What :func:`~provenance_kit.provenance.c2pa.sign_asset` needs from a signer.

    The private key never has to be in this process: :meth:`sign` may call a
    KMS or an HSM. Implementations must be safe to call from the thread that
    calls ``sign_asset``.

    What :meth:`sign` receives and must return:

    * ``data`` is the COSE ``Sig_structure`` to be signed (a few hundred bytes
      to a few kilobytes), **not** a digest. The signer hashes it with the
      algorithm's hash (:attr:`SigningAlgorithm.hash_name`); with a KMS, use
      its "sign a message" mode (for example ``MessageType=RAW`` with
      ``ECDSA_SHA_256``), or hash locally and use its digest mode.
    * The return value is the signature in :attr:`signature_format`. ECDSA
      signers may return DER (what KMSs return) or ``r || s``; Ed25519 and
      RSASSA-PSS signers return the raw signature. RSASSA-PSS must use MGF1
      with the same hash and a salt as long as the hash (RFC 8230).

    Attributes:
        algorithm: The signature algorithm; it must match the key.
        certificate_chain_pem: The signing certificate followed by any
            intermediate certificates, PEM-encoded, without the root: only
            blocks labelled exactly ``CERTIFICATE``, separated by whitespace
            (no other labels such as ``TRUSTED CERTIFICATE``, no explanatory
            text). The leaf must be a C2PA claim-signing certificate (a key
            usage extension with ``digitalSignature``, which ``sign_asset``
            checks itself; an allowed extended key usage such as
            ``id-kp-emailProtection`` or ``id-kp-documentSigning``; not a
            CA). ``sign_asset`` passes the C2PA library the chain re-encoded
            from the certificates it checked. It is embedded in every
            manifest and is therefore public.
        signature_format: The encoding :meth:`sign` returns.
    """

    @property
    def algorithm(self) -> SigningAlgorithm:
        """The signature algorithm."""
        ...

    @property
    def certificate_chain_pem(self) -> str:
        """The certificate chain, leaf first, PEM-encoded."""
        ...

    @property
    def signature_format(self) -> SignatureFormat:
        """The encoding of the signatures :meth:`sign` returns."""
        ...

    def sign(self, data: bytes) -> bytes:
        """Signs ``data`` (see the class docstring for the exact contract)."""
        ...


def ecdsa_der_to_raw(signature: BytesLike, algorithm: SigningAlgorithm) -> bytes:
    """Converts a DER ECDSA signature to the ``r || s`` form that COSE uses.

    KMS and HSM APIs (and most crypto libraries) return ECDSA signatures as
    ASN.1 DER ``Ecdsa-Sig-Value``; COSE and C2PA store them as the fixed-width
    big-endian concatenation of ``r`` and ``s`` (IEEE P1363). The DER input is
    parsed strictly: definite minimal lengths, positive minimal integers and
    no trailing bytes.

    Args:
        signature: The DER-encoded signature.
        algorithm: An ECDSA algorithm; it fixes the component width (32 bytes
            for ES256, 48 for ES384, 66 for ES512).

    Returns:
        ``r || s``, each left-padded with zeros to the component width.

    Raises:
        TypeError: If ``signature`` is not bytes-like or ``algorithm`` is not
            a :class:`SigningAlgorithm`.
        ValueError: If ``algorithm`` is not ECDSA, or the input is not a
            well-formed DER ECDSA signature whose components fit the width.
    """
    if not isinstance(signature, (bytes, bytearray, memoryview)):
        raise TypeError(f"signature must be bytes-like, got {type(signature).__name__}")
    if not isinstance(algorithm, SigningAlgorithm):
        raise TypeError("algorithm must be a SigningAlgorithm")
    width = algorithm.ecdsa_component_bytes
    if width is None:
        raise ValueError(f"{algorithm.value} is not an ECDSA algorithm")
    der = bytes(memoryview(signature).cast("B"))
    if len(der) < 8 or der[0] != 0x30:
        raise ValueError("not a DER ECDSA signature: expected a SEQUENCE")
    length, pos = _der_length(der, 1)
    if pos + length != len(der):
        raise ValueError("not a DER ECDSA signature: wrong SEQUENCE length or trailing bytes")
    r, pos = _der_integer(der, pos)
    s, pos = _der_integer(der, pos)
    if pos != len(der):
        raise ValueError("not a DER ECDSA signature: unexpected data after s")
    limit = 1 << (8 * width)
    for value in (r, s):
        if not 0 < value < limit:
            raise ValueError(f"ECDSA component out of range for {algorithm.value}")
    return r.to_bytes(width, "big") + s.to_bytes(width, "big")


def _der_length(der: bytes, pos: int) -> tuple[int, int]:
    if pos >= len(der):
        raise ValueError("not a DER ECDSA signature: truncated length")
    first = der[pos]
    pos += 1
    if first < 0x80:
        return first, pos
    if first != 0x81:  # an ECDSA signature never needs more than one length byte
        raise ValueError("not a DER ECDSA signature: unsupported length encoding")
    if pos >= len(der):
        raise ValueError("not a DER ECDSA signature: truncated length")
    length = der[pos]
    if length < 0x80:
        raise ValueError("not a DER ECDSA signature: non-minimal length")
    return length, pos + 1


def _der_integer(der: bytes, pos: int) -> tuple[int, int]:
    if pos >= len(der) or der[pos] != 0x02:
        raise ValueError("not a DER ECDSA signature: expected an INTEGER")
    length, pos = _der_length(der, pos + 1)
    end = pos + length
    if length == 0 or end > len(der):
        raise ValueError("not a DER ECDSA signature: bad INTEGER length")
    content = der[pos:end]
    if content[0] & 0x80:
        raise ValueError("not a DER ECDSA signature: negative INTEGER")
    if len(content) > 1 and content[0] == 0 and not content[1] & 0x80:
        raise ValueError("not a DER ECDSA signature: non-minimal INTEGER")
    return int.from_bytes(content, "big"), end


def check_signer(signer: C2paSigner) -> tuple[SigningAlgorithm, str, SignatureFormat]:
    """Validates a signer's declared properties (internal).

    Args:
        signer: The signer.

    Returns:
        Its algorithm, certificate chain and signature format.

    Raises:
        TypeError: If a property has the wrong type or ``sign`` is missing.
        ValueError: If the chain is not PEM certificate blocks (see
            :func:`split_pem_certificates`), or the format does not apply to
            the algorithm.
    """
    algorithm = signer.algorithm
    if not isinstance(algorithm, SigningAlgorithm):
        raise TypeError("signer.algorithm must be a SigningAlgorithm")
    chain = signer.certificate_chain_pem
    if not isinstance(chain, str):
        raise TypeError("signer.certificate_chain_pem must be a str")
    if _PEM_CERTIFICATE not in chain:
        raise ValueError("signer.certificate_chain_pem must contain PEM certificates")
    split_pem_certificates(chain)
    signature_format = signer.signature_format
    if not isinstance(signature_format, SignatureFormat):
        raise TypeError("signer.signature_format must be a SignatureFormat")
    if signature_format is SignatureFormat.DER and not algorithm.is_ecdsa:
        raise ValueError(f"DER signatures apply to ECDSA only, not {algorithm.value}")
    if not callable(getattr(signer, "sign", None)):
        raise TypeError("signer must have a sign(data) method")
    return algorithm, chain, signature_format


def split_pem_certificates(certificate_chain_pem: str) -> tuple[bytes, ...]:
    """Splits a PEM chain into DER certificates, strictly (internal).

    Every block must be labelled exactly ``CERTIFICATE`` and hold base64
    only (ASCII whitespace inside it is ignored); only ASCII whitespace may
    surround the blocks. Lenient parsers disagree on everything else:
    ``cryptography`` skips blocks with other labels (``TRUSTED
    CERTIFICATE``, ``FOO``) while c2pa-rs uses every block, so the leaf that
    was checked would not be the leaf that signs.

    Args:
        certificate_chain_pem: The chain, leaf first.

    Returns:
        The DER bytes of each block, in order (not parsed as certificates).

    Raises:
        ValueError: If the text is not one or more such blocks.
    """
    ders: list[bytes] = []
    position = 0
    for match in _PEM_BLOCK.finditer(certificate_chain_pem):
        if certificate_chain_pem[position : match.start()].strip(_PEM_SPACE):
            raise ValueError(_CHAIN_FORMAT)
        if match["begin"] != "CERTIFICATE" or match["end"] != "CERTIFICATE":
            raise ValueError(_CHAIN_FORMAT)
        body = _PEM_SPACE_RE.sub("", match["body"])
        try:
            der = base64.b64decode(body.encode("ascii"), validate=True)
        except (UnicodeError, binascii.Error):
            raise ValueError("the certificate chain could not be loaded (bad base64)") from None
        if not der:
            raise ValueError("the certificate chain could not be loaded (empty block)")
        ders.append(der)
        position = match.end()
    if certificate_chain_pem[position:].strip(_PEM_SPACE):
        raise ValueError(_CHAIN_FORMAT)
    if not ders:
        raise ValueError("the certificate chain is empty")
    return tuple(ders)


def check_signing_certificate(certificate_chain_pem: str) -> str:
    """Checks that a signer's leaf certificate may sign C2PA claims (internal).

    C2PA 2.3, section 14.5.1: certificates used to sign C2PA manifests "shall
    assert the digitalSignature bit" of the key usage extension. c2pa-rs 0.91
    refuses a leaf without the extension but accepts one that asserts only
    ``nonRepudiation``, and then reads the manifests it signed back as
    trusted. This check closes that gap before the library is called.

    The chain is parsed strictly (:func:`split_pem_certificates`) and
    returned re-encoded from the parsed certificates: pass the returned text,
    not the input, to the C2PA library, so that the certificate checked here
    is the one it signs with.

    Args:
        certificate_chain_pem: The chain, leaf first, PEM-encoded.

    Returns:
        The chain in canonical PEM (one ``CERTIFICATE`` block per
        certificate, 64-character lines, LF line endings).

    Raises:
        BackendUnavailableError: If ``cryptography`` is not installed (extra
            ``c2pa``).
        ValueError: If the chain is not strictly PEM certificates, a block is
            not a certificate, or the leaf has no key usage extension, one
            that cannot be parsed, or one without ``digitalSignature``.
    """
    certificates = _load_chain(certificate_chain_pem)
    _check_leaf_key_usage(certificates[0])
    return _canonical_pem(certificates)


def _load_chain(certificate_chain_pem: str) -> list[x509.Certificate]:
    _require_cryptography()
    from cryptography import x509  # noqa: PLC0415 - optional dependency

    certificates = []
    for der in split_pem_certificates(certificate_chain_pem):
        try:
            certificates.append(x509.load_der_x509_certificate(der))
        except MemoryError:
            raise
        except Exception as exc:  # ValueError, or a cryptography-specific type
            raise ValueError("the certificate chain could not be loaded") from exc
    return certificates


def _canonical_pem(certificates: list[x509.Certificate]) -> str:
    from cryptography.hazmat.primitives import serialization  # noqa: PLC0415

    return "".join(
        certificate.public_bytes(serialization.Encoding.PEM).decode("ascii")
        for certificate in certificates
    )


def _check_leaf_key_usage(leaf: x509.Certificate) -> None:
    from cryptography import x509  # noqa: PLC0415 - optional dependency

    try:
        usage = leaf.extensions.get_extension_for_class(x509.KeyUsage).value
    except x509.ExtensionNotFound:
        raise ValueError(
            "the signing certificate has no key usage extension; C2PA requires "
            "digitalSignature (C2PA 2.3, section 14.5.1)"
        ) from None
    except MemoryError:
        raise
    except Exception as exc:  # malformed or duplicate extensions (DuplicateExtension)
        raise ValueError("the signing certificate's extensions could not be parsed") from exc
    if not usage.digital_signature:
        raise ValueError(
            "the signing certificate's key usage lacks digitalSignature, which C2PA "
            "requires of claim-signing certificates (C2PA 2.3, section 14.5.1)"
        )


def normalize_signature(
    signature: object,
    algorithm: SigningAlgorithm,
    signature_format: SignatureFormat,
) -> bytes:
    """Checks a signer's output and converts it to the COSE encoding (internal).

    Args:
        signature: What the signer returned.
        algorithm: The signer's algorithm.
        signature_format: The signer's declared format.

    Returns:
        The signature as COSE stores it.

    Raises:
        SigningError: If the signature has the wrong type, length or encoding.
    """
    if not isinstance(signature, (bytes, bytearray, memoryview)):
        raise SigningError(f"the signer returned {type(signature).__name__}, not bytes")
    raw = bytes(memoryview(signature).cast("B"))
    width = algorithm.ecdsa_component_bytes
    if width is not None:
        if signature_format is SignatureFormat.DER:
            try:
                return ecdsa_der_to_raw(raw, algorithm)
            except ValueError as exc:
                raise SigningError(
                    f"the signer returned an invalid DER {algorithm.value} signature"
                ) from exc
        if len(raw) != 2 * width:
            raise SigningError(
                f"a raw {algorithm.value} signature must be {2 * width} bytes, got {len(raw)}"
            )
        return raw
    if algorithm is SigningAlgorithm.ED25519:
        if len(raw) != 64:
            raise SigningError(f"an Ed25519 signature must be 64 bytes, got {len(raw)}")
        return raw
    low, high = _RSA_SIGNATURE_BYTES
    if not low <= len(raw) <= high:
        raise SigningError(
            f"an {algorithm.value} signature must be {low} to {high} bytes, got {len(raw)}"
        )
    return raw


class LocalKeySigner:
    """A :class:`C2paSigner` that holds a private key in process memory.

    Meant for development, tests and small deployments. In production, prefer
    a signer that delegates to a KMS or HSM so that the key never enters the
    process. Requires the ``cryptography`` package (extra ``c2pa``).

    The algorithm defaults from the key: ES256, ES384 or ES512 for P-256,
    P-384 or P-521 keys, ED25519 for Ed25519 keys and PS256 for RSA keys (pass
    PS384 or PS512 to choose another hash). Signatures are returned as
    :attr:`SignatureFormat.RAW`. The constructor parses the chain strictly
    (only ``CERTIFICATE`` blocks, see :class:`C2paSigner`), checks that the
    first certificate holds the key's public key and asserts the
    ``digitalSignature`` key usage, and keeps the chain re-encoded from the
    parsed certificates (:attr:`certificate_chain_pem`).

    Thread safety: safe to share between threads.

    Args:
        private_key_pem: The PEM-encoded private key (PKCS#8 or traditional).
        certificate_chain_pem: The leaf certificate followed by any
            intermediates, PEM-encoded (see :class:`C2paSigner`).
        algorithm: The algorithm, or ``None`` to derive it from the key.
        password: The password of an encrypted private key.

    Raises:
        BackendUnavailableError: If ``cryptography`` is not installed.
        TypeError: If an argument has the wrong type.
        ValueError: If the key or the chain cannot be loaded (including a
            chain with blocks other than ``CERTIFICATE`` or text between
            them), the key type or size does not fit the algorithm, or the
            leaf certificate does not match the key, has extensions that
            cannot be parsed or lacks the ``digitalSignature`` key usage.
            Messages never contain key material.
    """

    __slots__ = ("_algorithm", "_chain", "_sign")

    def __init__(
        self,
        private_key_pem: bytes | str,
        certificate_chain_pem: str,
        *,
        algorithm: SigningAlgorithm | None = None,
        password: bytes | None = None,
    ) -> None:
        """Loads the key and the chain."""
        if isinstance(private_key_pem, str):
            private_key_pem = private_key_pem.encode("ascii", errors="strict")
        if not isinstance(private_key_pem, bytes):
            raise TypeError("private_key_pem must be bytes or str")
        if not isinstance(certificate_chain_pem, str):
            raise TypeError("certificate_chain_pem must be a str")
        if algorithm is not None and not isinstance(algorithm, SigningAlgorithm):
            raise TypeError("algorithm must be a SigningAlgorithm or None")
        if password is not None and not isinstance(password, bytes):
            raise TypeError("password must be bytes or None")

        _require_cryptography()
        from cryptography.hazmat.primitives import serialization  # noqa: PLC0415

        try:
            key = serialization.load_pem_private_key(private_key_pem, password=password)
        except (ValueError, TypeError) as exc:
            raise ValueError("the private key could not be loaded") from exc
        except Exception as exc:  # cryptography's UnsupportedAlgorithm
            raise ValueError(f"the private key could not be loaded ({type(exc).__name__})") from exc
        certificates = _load_chain(certificate_chain_pem)

        der = serialization.Encoding.DER
        spki = serialization.PublicFormat.SubjectPublicKeyInfo
        try:
            leaf_public = certificates[0].public_key().public_bytes(der, spki)
        except MemoryError:
            raise
        except Exception as exc:  # an unsupported or malformed public key
            raise ValueError(
                f"the first certificate's public key could not be loaded ({type(exc).__name__})"
            ) from exc
        if leaf_public != key.public_key().public_bytes(der, spki):
            raise ValueError("the first certificate of the chain does not match the private key")
        _check_leaf_key_usage(certificates[0])

        self._algorithm, self._sign = _key_signer(key, algorithm)
        self._chain = _canonical_pem(certificates)

    @property
    def algorithm(self) -> SigningAlgorithm:
        """The signature algorithm."""
        return self._algorithm

    @property
    def certificate_chain_pem(self) -> str:
        """The certificate chain, leaf first, in canonical PEM."""
        return self._chain

    @property
    def signature_format(self) -> SignatureFormat:
        """Always :attr:`SignatureFormat.RAW`."""
        return SignatureFormat.RAW

    def sign(self, data: bytes) -> bytes:
        """Signs ``data`` with the in-process key.

        Args:
            data: The bytes to sign.

        Returns:
            The signature in the COSE encoding (``r || s`` for ECDSA).
        """
        return self._sign(bytes(data))

    def __repr__(self) -> str:
        """Returns a representation without key material."""
        return f"LocalKeySigner(algorithm={self._algorithm.value!r})"


def _require_cryptography() -> None:
    import_optional("cryptography.hazmat.primitives", extra="c2pa", feature="LocalKeySigner")


def _key_signer(
    key: PrivateKeyTypes,
    algorithm: SigningAlgorithm | None,
) -> tuple[SigningAlgorithm, Callable[[bytes], bytes]]:
    from cryptography.hazmat.primitives import hashes  # noqa: PLC0415 - optional dependency
    from cryptography.hazmat.primitives.asymmetric import (  # noqa: PLC0415
        ec,
        ed25519,
        padding,
        rsa,
    )

    hash_classes: dict[str, type[hashes.SHA256 | hashes.SHA384 | hashes.SHA512]] = {
        "sha256": hashes.SHA256,
        "sha384": hashes.SHA384,
        "sha512": hashes.SHA512,
    }

    if isinstance(key, ec.EllipticCurvePrivateKey):
        by_curve = {
            "secp256r1": SigningAlgorithm.ES256,
            "secp384r1": SigningAlgorithm.ES384,
            "secp521r1": SigningAlgorithm.ES512,
        }
        expected = by_curve.get(key.curve.name)
        if expected is None:
            raise ValueError("C2PA supports the P-256, P-384 and P-521 curves only")
        if algorithm is not None and algorithm is not expected:
            raise ValueError(f"a {key.curve.name} key signs with {expected.value} only")
        ecdsa = ec.ECDSA(hash_classes[str(expected.hash_name)]())
        ec_key = key

        def sign_ecdsa(data: bytes) -> bytes:
            return ecdsa_der_to_raw(ec_key.sign(data, ecdsa), expected)

        return expected, sign_ecdsa

    if isinstance(key, ed25519.Ed25519PrivateKey):
        if algorithm not in (None, SigningAlgorithm.ED25519):
            raise ValueError("an Ed25519 key signs with ed25519 only")
        return SigningAlgorithm.ED25519, key.sign

    if isinstance(key, rsa.RSAPrivateKey):
        chosen = algorithm or SigningAlgorithm.PS256
        if chosen not in (SigningAlgorithm.PS256, SigningAlgorithm.PS384, SigningAlgorithm.PS512):
            raise ValueError("an RSA key signs with ps256, ps384 or ps512 only")
        if key.key_size < 2048:
            raise ValueError("C2PA requires RSA keys of at least 2048 bits")
        hash_class = hash_classes[str(chosen.hash_name)]
        pss = padding.PSS(mgf=padding.MGF1(hash_class()), salt_length=hash_class.digest_size)
        rsa_key = key

        def sign_pss(data: bytes) -> bytes:
            # bytes() keeps the declared type when cryptography's stubs are absent.
            return bytes(rsa_key.sign(data, pss, hash_class()))

        return chosen, sign_pss

    raise ValueError(f"unsupported private key type {type(key).__name__}")
