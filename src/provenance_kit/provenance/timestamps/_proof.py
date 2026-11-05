"""Proof, inspection and verification types, and the opentimestamps adapter (internal)."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Protocol

from provenance_kit._optional import import_optional
from provenance_kit._typing import BytesLike
from provenance_kit.errors import MediaDecodeError, MediaLimitExceededError, UnsupportedMediaError

__all__ = [
    "DEFAULT_MAX_PROOF_BYTES",
    "AttestationCheck",
    "BitcoinAttestation",
    "BitcoinBlockHeader",
    "BitcoinBlockSource",
    "CalendarFailure",
    "InvalidProofError",
    "TimestampInfo",
    "TimestampProof",
    "TimestampVerification",
]

#: The largest serialized proof accepted, in bytes. Complete proofs from the
#: public calendars are 1 to 10 KiB; the cap bounds the memory a hostile proof
#: can make the parser allocate (an operation result can be up to 4 KiB).
DEFAULT_MAX_PROOF_BYTES = 64 * 1024

#: The ``.ots`` file header (python-opentimestamps ``DetachedTimestampFile``).
HEADER_MAGIC = b"\x00OpenTimestamps\x00\x00Proof\x00\xbf\x89\xe2\xe8\x84\xe8\x92\x94"
_MAJOR_VERSION = 1
_SHA256_TAG = 0x08
_DIGEST_OFFSET = len(HEADER_MAGIC) + 2  # magic, major version, hash op tag
_DIGEST_BYTES = 32
_FEATURE = "OpenTimestamps proofs"


class InvalidProofError(MediaDecodeError):
    """The bytes are not a well-formed OpenTimestamps proof."""


@dataclass(frozen=True, slots=True)
class CalendarFailure:
    """A calendar that did not contribute to an operation.

    Attributes:
        url: The calendar URL (a configured one when stamping; the URI named
            by the proof when upgrading, which is untrusted).
        reason: A short category such as ``"timeout"``, ``"http status 500"``,
            ``"response too large"``, ``"malformed response"`` or
            ``"not in the upgrade allow-list"``.
    """

    url: str
    reason: str


@dataclass(frozen=True, slots=True)
class TimestampProof:
    """A serialized OpenTimestamps proof for a SHA-256 digest.

    :attr:`data` is the standard detached ``.ots`` format that the
    ``opentimestamps-client`` (``ots``) reads and writes. Build one with
    :meth:`OpenTimestamper.stamp` or :meth:`from_bytes`; a proof is
    immutable and every operation returns a new one.

    Attributes:
        data: The ``.ots`` bytes.
        failed_calendars: The calendars that failed during the operation that
            produced this proof (not serialized, not compared).

    Raises:
        TypeError: If ``data`` is not bytes-like.
        InvalidProofError: If ``data`` lacks the ``.ots`` header.
        UnsupportedMediaError: If the proof is not for a SHA-256 digest.
    """

    data: bytes = field(repr=False)
    failed_calendars: tuple[CalendarFailure, ...] = field(default=(), compare=False)

    def __post_init__(self) -> None:
        """Checks the header and the digest algorithm."""
        data: object = self.data
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("proof data must be bytes-like")
        raw = bytes(memoryview(data).cast("B"))
        object.__setattr__(self, "data", raw)
        object.__setattr__(self, "failed_calendars", tuple(self.failed_calendars))
        if not raw.startswith(HEADER_MAGIC):
            raise InvalidProofError("not an OpenTimestamps proof (bad header)")
        if len(raw) <= _DIGEST_OFFSET + _DIGEST_BYTES:
            raise InvalidProofError("truncated OpenTimestamps proof")
        if raw[len(HEADER_MAGIC)] != _MAJOR_VERSION:
            raise UnsupportedMediaError("unsupported OpenTimestamps proof version")
        if raw[len(HEADER_MAGIC) + 1] != _SHA256_TAG:
            raise UnsupportedMediaError("only proofs for SHA-256 digests are supported")

    @classmethod
    def from_bytes(
        cls, data: BytesLike, *, max_proof_bytes: int = DEFAULT_MAX_PROOF_BYTES
    ) -> TimestampProof:
        """Parses and validates an ``.ots`` file (untrusted input).

        The proof is parsed completely with python-opentimestamps and stored
        re-serialized, so equal proofs have equal bytes. The size is checked
        first, before the library is needed.

        Args:
            data: The ``.ots`` bytes.
            max_proof_bytes: The size limit (the same name and default as
                :class:`OpenTimestamper`'s).

        Returns:
            The proof.

        Raises:
            TypeError: If ``data`` is not bytes-like.
            BackendUnavailableError: If opentimestamps is not installed
                (extra ``timestamps``).
            MediaLimitExceededError: If ``data`` exceeds ``max_proof_bytes``
                (``limit="max_proof_bytes"``).
            InvalidProofError: If the proof is malformed.
            UnsupportedMediaError: If it is not a SHA-256 proof.
        """
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("proof data must be bytes-like")
        raw = bytes(memoryview(data).cast("B"))
        detached = parse_detached(raw, max_bytes=max_proof_bytes)
        return cls(serialize_detached(detached))

    def to_bytes(self) -> bytes:
        """Returns the ``.ots`` bytes."""
        return self.data

    @property
    def digest(self) -> bytes:
        """The SHA-256 digest the proof is for."""
        return self.data[_DIGEST_OFFSET : _DIGEST_OFFSET + _DIGEST_BYTES]


@dataclass(frozen=True, slots=True)
class BitcoinAttestation:
    """A Bitcoin block-header attestation found in a proof.

    Attributes:
        height: The block height the attestation names (unverified).
        merkle_root: The value the block's merkle root must equal, in the
            internal byte order of the serialized block header (the reverse
            of the hex that ``bitcoind`` and block explorers display).
    """

    height: int
    merkle_root: bytes


@dataclass(frozen=True, slots=True)
class TimestampInfo:
    """What a proof contains (nothing here is verified).

    Attributes:
        digest: The SHA-256 digest the proof is for.
        pending_calendars: The calendar URIs of pending attestations, sorted
            and unique. **Untrusted**: they come from the proof.
        bitcoin_attestations: The Bitcoin attestations, sorted by height.
        other_attestations: How many attestations of other kinds (Litecoin,
            unknown) the proof holds; they are ignored.
    """

    digest: bytes
    pending_calendars: tuple[str, ...]
    bitcoin_attestations: tuple[BitcoinAttestation, ...]
    other_attestations: int

    @property
    def is_complete(self) -> bool:
        """Whether the proof holds at least one Bitcoin attestation."""
        return bool(self.bitcoin_attestations)


@dataclass(frozen=True, slots=True)
class BitcoinBlockHeader:
    """The two block-header fields that verification needs.

    Attributes:
        merkle_root: The 32-byte merkle root in internal byte order (as in
            the serialized header). Use :meth:`from_display_hex` for the hex
            that ``bitcoind getblockheader`` and explorers show.
        time: The header's ``nTime`` (Unix seconds).

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If ``merkle_root`` is not 32 bytes or ``time`` is negative.
    """

    merkle_root: bytes
    time: int

    def __post_init__(self) -> None:
        """Validates the fields."""
        if not isinstance(self.merkle_root, bytes):
            raise TypeError("merkle_root must be bytes")
        if len(self.merkle_root) != 32:
            raise ValueError("merkle_root must be 32 bytes")
        time: object = self.time
        if isinstance(time, bool) or not isinstance(time, int):
            raise TypeError("time must be an int")
        if self.time < 0:
            raise ValueError("time must not be negative")

    @classmethod
    def from_display_hex(cls, merkle_root_hex: str, time: int) -> BitcoinBlockHeader:
        """Builds a header from the merkle root as Bitcoin tools display it.

        Args:
            merkle_root_hex: The ``merkleroot`` field of ``bitcoind
                getblockheader`` (or an explorer's ``merkle_root``): 64 hex
                digits, most significant byte first.
            time: The header time (Unix seconds).

        Returns:
            The header, with the merkle root reversed into internal order.

        Raises:
            ValueError: If ``merkle_root_hex`` is not 64 hex digits.
        """
        text: object = merkle_root_hex
        if not isinstance(text, str) or len(text) != 64:
            raise ValueError("merkle_root_hex must be 64 hex digits")
        return cls(bytes.fromhex(text)[::-1], time)


class BitcoinBlockSource(Protocol):
    """Looks up Bitcoin block headers by height, for :meth:`OpenTimestamper.verify`.

    provenance-kit ships no implementation, so verification never touches
    the network by itself. Back it with your own node (``getblockhash`` then
    ``getblockheader``), a block explorer API or a header store; see
    ``docs/provenance.md``. Verification is only as trustworthy as the source:
    an explorer can lie, a node you run cannot.
    """

    def block_header(self, height: int) -> BitcoinBlockHeader | None:
        """Returns the header of the main-chain block at ``height``, or ``None`` if unknown."""
        ...


@dataclass(frozen=True, slots=True)
class AttestationCheck:
    """The verification of one Bitcoin attestation.

    Attributes:
        height: The attested block height.
        verified: Whether the block's merkle root matched.
        block_time: The block header time (Unix seconds) when verified.
        reason: Why it did not verify: ``"block not available"`` or
            ``"merkle root mismatch"``.
    """

    height: int
    verified: bool
    block_time: int | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class TimestampVerification:
    """The result of :meth:`OpenTimestamper.verify`.

    Attributes:
        digest_matches: Whether the proof is for the given digest.
        checks: One check per Bitcoin attestation, sorted by height. Empty
            when the digest does not match or the proof is still pending.
    """

    digest_matches: bool
    checks: tuple[AttestationCheck, ...]

    @property
    def verified(self) -> bool:
        """Whether the digest matches and at least one attestation verified."""
        return self.digest_matches and any(check.verified for check in self.checks)

    @property
    def attested_time(self) -> int | None:
        """The earliest verified block time (Unix seconds), or ``None``.

        The data existed before this time. Bitcoin header times may differ
        from real time by up to about two hours.
        """
        times = [c.block_time for c in self.checks if c.verified and c.block_time is not None]
        return min(times) if times else None


# ----------------------------------------------------------------- adapter


@dataclass(frozen=True, slots=True)
class OtsModules:
    """The python-opentimestamps names the adapter uses (internal)."""

    timestamp: ModuleType
    op: ModuleType
    notary: ModuleType
    serialize: ModuleType


def load_ots() -> OtsModules:
    """Imports python-opentimestamps.

    Raises:
        BackendUnavailableError: If it is not installed (extra ``timestamps``).
    """
    return OtsModules(
        timestamp=import_optional(
            "opentimestamps.core.timestamp", extra="timestamps", feature=_FEATURE
        ),
        op=import_optional("opentimestamps.core.op", extra="timestamps", feature=_FEATURE),
        notary=import_optional("opentimestamps.core.notary", extra="timestamps", feature=_FEATURE),
        serialize=import_optional(
            "opentimestamps.core.serialize", extra="timestamps", feature=_FEATURE
        ),
    )


def parse_detached(data: bytes, *, max_bytes: int) -> Any:
    """Parses ``.ots`` bytes into a ``DetachedTimestampFile``.

    Raises:
        MediaLimitExceededError: If ``data`` exceeds ``max_bytes``.
        InvalidProofError: If the proof is malformed.
        UnsupportedMediaError: If it is not a SHA-256 proof.
    """
    if len(data) > max_bytes:
        raise MediaLimitExceededError("max_proof_bytes", max_bytes, len(data))
    TimestampProof(data)  # header and digest algorithm
    ots = load_ots()
    try:
        context = ots.serialize.BytesDeserializationContext(data)
        detached = ots.timestamp.DetachedTimestampFile.deserialize(context)
    except Exception as exc:  # the parser raises many types for hostile input
        raise InvalidProofError(f"malformed OpenTimestamps proof ({type(exc).__name__})") from exc
    if not isinstance(detached.file_hash_op, ots.op.OpSHA256):
        raise UnsupportedMediaError("only proofs for SHA-256 digests are supported")
    return detached


def parse_timestamp(data: bytes, message: bytes) -> Any:
    """Parses a calendar's serialized ``Timestamp`` for ``message``.

    Raises:
        InvalidProofError: If the bytes are malformed or have trailing data.
    """
    ots = load_ots()
    try:
        context = ots.serialize.BytesDeserializationContext(data)
        timestamp = ots.timestamp.Timestamp.deserialize(context, message)
        context.assert_eof()
    except Exception as exc:
        raise InvalidProofError(f"malformed timestamp ({type(exc).__name__})") from exc
    return timestamp


def serialize_detached(detached: Any) -> bytes:
    """Serializes a ``DetachedTimestampFile`` to ``.ots`` bytes."""
    ots = load_ots()
    context = ots.serialize.BytesSerializationContext()
    detached.serialize(context)
    data: bytes = context.getbytes()
    return data


def iter_nodes(timestamp: Any) -> Iterator[Any]:
    """Yields every node of a ``Timestamp`` tree once, without recursion.

    Parsed proofs are trees; timestamps built in memory can share nodes.
    """
    stack = [timestamp]
    seen: set[int] = set()
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        yield node
        stack.extend(node.ops.values())
