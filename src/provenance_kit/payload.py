"""Watermark payload codec: a format version and an identifier in N bits.

Invisible watermarks carry a small, fixed number of bits. The application
stores its records elsewhere and embeds only an unsigned integer
*identifier* that points at them, plus a small *format version* so that the
layout can change later without ambiguity. :class:`PayloadCodec` packs both
into a bit string of exactly the backend's capacity and unpacks it again.

Layout, most significant bit first::

    | version (version_bits) | identifier (identifier_bits) | check (check_bits) |

``check`` is optional (``check_bits=0`` by default). When present it holds the
leading bits of a SHA-256 digest of the version and identifier, so that random
bits (what a detector returns from an unmarked or damaged item) decode as
invalid with probability ``1 - 2**-check_bits``. It is an integrity check
against noise, not authentication: anyone can compute it.

Capacity trade-off: the bits are shared. Every bit given to the version
doubles the number of formats but halves the number of identifiers, and every
check bit halves the identifiers again::

    identifier_bits = total_bits - version_bits - check_bits
    distinct identifiers = 2 ** identifier_bits
    distinct versions    = 2 ** version_bits

Payload capacities of the supported backends (data bits available to this
codec, after the backend's own error correction):

=================================  ==========
Backend / mode                     total_bits
=================================  ==========
TrustMark ``BCH_SUPER``            40
TrustMark ``BCH_5`` (default)      61
TrustMark ``BCH_4``                68
TrustMark ``BCH_3``                75
AudioSeal (16-bit models)          16
VideoSeal v1.0 and PixelSeal       256
ChunkySeal                         1024
VideoSeal v0.0 (legacy)            96
=================================  ==========

TrustMark adds its own 4 schema bits outside these figures. AudioSeal and
VideoSeal apply no error correction: a few flipped bits are normal, so pair
them with ``check_bits`` or with backend-level redundancy.

Error contract:

* ``TypeError`` / ``ValueError``: caller mistakes. A wrong type, a negative
  identifier, a version out of range or not in ``versions``, an inconsistent
  codec configuration, or a bit sequence of the wrong length or with symbols
  other than 0 and 1 passed to :meth:`PayloadCodec.decode`.
* :class:`~provenance_kit.errors.PayloadError`: the identifier is valid but
  does not fit in this codec's capacity (the identifier space is exhausted),
  or :meth:`DecodedPayload.require_valid` was called on an invalid payload.
* :meth:`PayloadCodec.decode` never raises for a well-formed bit sequence of
  the right length, whatever its content: garbage decodes to a
  :class:`DecodedPayload` whose ``valid`` is false (when ``check_bits`` or
  ``versions`` make that detectable).
"""

from __future__ import annotations

import hashlib
from collections.abc import Collection
from dataclasses import dataclass, field

from provenance_kit.bits import BitsLike, int_to_bits, normalize_bits
from provenance_kit.errors import PayloadError

__all__ = ["MAX_TOTAL_BITS", "DecodedPayload", "PayloadCodec"]

#: The largest payload this codec accepts (ChunkySeal is 1024 bits).
MAX_TOTAL_BITS = 4096

_CHECK_DOMAIN = b"provenance-kit/payload-check/v1"
_MAX_CHECK_BITS = 256  # one SHA-256 digest


@dataclass(frozen=True, slots=True)
class DecodedPayload:
    """The fields unpacked from a payload bit string.

    Attributes:
        version: The format version read from the bits.
        identifier: The identifier read from the bits.
        check_ok: Whether the check bits match; ``None`` when the codec has no
            check bits.
        version_known: Whether ``version`` is one of the codec's ``versions``
            (always true when the codec accepts every version).
    """

    version: int
    identifier: int
    check_ok: bool | None
    version_known: bool

    @property
    def valid(self) -> bool:
        """True unless the check bits fail or the version is unknown.

        Without check bits and with every version accepted, any bits decode as
        valid: only the detector's own confidence can then reject garbage.
        """
        return self.check_ok is not False and self.version_known

    def require_valid(self) -> DecodedPayload:
        """Returns ``self`` if valid, else raises.

        Returns:
            This payload.

        Raises:
            PayloadError: If the payload is not :attr:`valid`.
        """
        if not self.valid:
            reason = "check bits do not match" if self.check_ok is False else "unknown version"
            raise PayloadError(f"invalid watermark payload: {reason}")
        return self


@dataclass(frozen=True, slots=True)
class PayloadCodec:
    """Packs a format version and an unsigned identifier into ``total_bits``.

    Example::

        codec = PayloadCodec(61, version_bits=4)  # TrustMark BCH_5
        bits = codec.encode(123_456, version=1)  # "0001..." (61 chars)
        decoded = codec.decode(bits)
        assert (decoded.version, decoded.identifier) == (1, 123_456)

    Attributes:
        total_bits: The backend's payload size in bits (1 to
            :data:`MAX_TOTAL_BITS`).
        version_bits: Bits reserved for the format version (may be 0).
        check_bits: Bits reserved for the integrity check (0 to 256).
        versions: The versions this application recognizes, stored as a
            ``frozenset``. ``None`` accepts every version that fits in
            ``version_bits``. Encoding a version outside the set is an error;
            decoding one gives ``version_known=False``.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If the configuration leaves no identifier bits or a
            version in ``versions`` does not fit.
    """

    total_bits: int
    version_bits: int = 4
    check_bits: int = 0
    versions: Collection[int] | None = field(default=None)

    def __post_init__(self) -> None:
        """Validates the configuration."""
        _check_int("total_bits", self.total_bits)
        _check_int("version_bits", self.version_bits)
        _check_int("check_bits", self.check_bits)
        if not 1 <= self.total_bits <= MAX_TOTAL_BITS:
            raise ValueError(f"total_bits must be between 1 and {MAX_TOTAL_BITS}")
        if self.version_bits < 0 or self.check_bits < 0:
            raise ValueError("version_bits and check_bits must not be negative")
        if self.check_bits > _MAX_CHECK_BITS:
            raise ValueError(f"check_bits must be at most {_MAX_CHECK_BITS}")
        if self.identifier_bits < 1:
            raise ValueError(
                "version_bits + check_bits must leave at least one identifier bit "
                f"(total_bits={self.total_bits})"
            )
        if self.versions is not None:
            if isinstance(self.versions, (str, bytes)) or not isinstance(self.versions, Collection):
                raise TypeError("versions must be a collection of ints")
            frozen = frozenset(self.versions)
            for version in frozen:
                self._check_version_range(version)
            if not frozen:
                raise ValueError("versions must not be empty; use None to accept every version")
            object.__setattr__(self, "versions", frozen)

    # ------------------------------------------------------------ capacity

    @property
    def identifier_bits(self) -> int:
        """Bits available to the identifier."""
        return self.total_bits - self.version_bits - self.check_bits

    @property
    def max_identifier(self) -> int:
        """The largest identifier that fits."""
        return (1 << self.identifier_bits) - 1

    @property
    def max_version(self) -> int:
        """The largest version that fits (0 when ``version_bits`` is 0)."""
        return (1 << self.version_bits) - 1

    @property
    def capacity(self) -> int:
        """The number of distinct identifiers, ``2 ** identifier_bits``."""
        return 1 << self.identifier_bits

    # ------------------------------------------------------------ codec

    def encode(self, identifier: int, *, version: int = 0) -> str:
        """Packs a version and an identifier into a bit string.

        Args:
            identifier: A non-negative integer up to :attr:`max_identifier`.
            version: The format version, up to :attr:`max_version` and in
                :attr:`versions` when that is set.

        Returns:
            A string of exactly :attr:`total_bits` ``"0"``/``"1"`` characters.
            Convert it with ``[int(c) for c in bits]`` for tensor backends.

        Raises:
            TypeError: If ``identifier`` or ``version`` is not an ``int``.
            ValueError: If ``identifier`` is negative, or ``version`` is out of
                range or not in :attr:`versions`.
            PayloadError: If ``identifier`` exceeds :attr:`max_identifier`.
        """
        _check_int("identifier", identifier)
        _check_int("version", version)
        if identifier < 0:
            raise ValueError("identifier must not be negative")
        self._check_version_range(version)
        if self.versions is not None and version not in self.versions:
            raise ValueError(f"version {version} is not one of the codec's versions")
        if identifier > self.max_identifier:
            raise PayloadError(
                f"identifier does not fit in {self.identifier_bits} bits "
                f"(maximum {self.max_identifier})"
            )
        body = self._body(version, identifier)
        return body + self._check(body)

    def decode(self, bits: BitsLike) -> DecodedPayload:
        """Unpacks a bit sequence produced by :meth:`encode` or by a detector.

        Args:
            bits: Exactly :attr:`total_bits` bits, as a string or an iterable
                of 0/1 values.

        Returns:
            The decoded fields. Never raises for well-formed input: check
            :attr:`DecodedPayload.valid`.

        Raises:
            TypeError: If ``bits`` is not a string or an iterable.
            ValueError: If ``bits`` has the wrong length or a symbol other than
                0 and 1.
        """
        text = normalize_bits(bits, length=self.total_bits)
        version = int(text[: self.version_bits], 2) if self.version_bits else 0
        end = self.version_bits + self.identifier_bits
        identifier = int(text[self.version_bits : end], 2)
        check_ok: bool | None = None
        if self.check_bits:
            check_ok = text[end:] == self._check(text[:end])
        version_known = self.versions is None or version in self.versions
        return DecodedPayload(
            version=version,
            identifier=identifier,
            check_ok=check_ok,
            version_known=version_known,
        )

    # ------------------------------------------------------------ internals

    def _body(self, version: int, identifier: int) -> str:
        head = int_to_bits(version, self.version_bits) if self.version_bits else ""
        return head + int_to_bits(identifier, self.identifier_bits)

    def _check(self, body: str) -> str:
        if not self.check_bits:
            return ""
        header = (
            self.total_bits.to_bytes(2, "big")
            + self.version_bits.to_bytes(2, "big")
            + self.check_bits.to_bytes(2, "big")
        )
        packed = int(body, 2).to_bytes((len(body) + 7) // 8, "big")
        digest = hashlib.sha256(_CHECK_DOMAIN + header + packed).digest()
        needed = (self.check_bits + 7) // 8
        value = int.from_bytes(digest[:needed], "big") >> (8 * needed - self.check_bits)
        return int_to_bits(value, self.check_bits)

    def _check_version_range(self, version: object) -> None:
        value = _check_int("version", version)
        if not 0 <= value <= self.max_version:
            raise ValueError(f"version must be between 0 and {self.max_version}")


def _check_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    return value
