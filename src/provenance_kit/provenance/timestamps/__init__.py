"""OpenTimestamps proofs for SHA-256 digests: stamp, upgrade, inspect, verify.

Backed by python-opentimestamps 0.5.0 (extra ``timestamps``; LGPL-3.0, see
``docs/THIRD_PARTY.md``) for the proof format and its parser. Importing this
module does not import it. The calendar HTTP client is provenance-kit's own
(standard library only), because calendar responses are untrusted.

Typical use::

    from provenance_kit.provenance.timestamps import OpenTimestamper, sha256_file

    ots = OpenTimestamper(min_calendars=2)  # the official client's calendars
    proof = ots.stamp(sha256_file("master.mov"))  # pending, a few hundred bytes
    store(proof.to_bytes())  # the standard .ots format

    # hours later (a background job):
    proof = ots.upgrade(TimestampProof.from_bytes(load()))
    if ots.info(proof).is_complete:
        result = ots.verify(proof, digest, block_source=my_bitcoin_node)
        assert result.verified, result.checks

Only the timestamps module talks to the network, and only to calendars: the
digest is nonced and hashed before it leaves the process, and
:meth:`OpenTimestamper.info` and :meth:`OpenTimestamper.verify` are offline.
Verification needs Bitcoin block headers from a :class:`BitcoinBlockSource`
that the application provides (its own node or an explorer).

Thread safety: :class:`OpenTimestamper` and the hash helpers are thread-safe.
"""

from __future__ import annotations

from provenance_kit.provenance.timestamps._hashing import (
    DEFAULT_CHUNK_BYTES,
    sha256_file,
    sha256_stream,
)
from provenance_kit.provenance.timestamps._proof import (
    DEFAULT_MAX_PROOF_BYTES,
    AttestationCheck,
    BitcoinAttestation,
    BitcoinBlockHeader,
    BitcoinBlockSource,
    CalendarFailure,
    InvalidProofError,
    TimestampInfo,
    TimestampProof,
    TimestampVerification,
)
from provenance_kit.provenance.timestamps._timestamper import (
    DEFAULT_CALENDARS,
    DEFAULT_MAX_RESPONSE_BYTES,
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_UPGRADE_ALLOWLIST,
    CalendarError,
    OpenTimestamper,
)

__all__ = [
    "DEFAULT_CALENDARS",
    "DEFAULT_CHUNK_BYTES",
    "DEFAULT_MAX_PROOF_BYTES",
    "DEFAULT_MAX_RESPONSE_BYTES",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_UPGRADE_ALLOWLIST",
    "AttestationCheck",
    "BitcoinAttestation",
    "BitcoinBlockHeader",
    "BitcoinBlockSource",
    "CalendarError",
    "CalendarFailure",
    "InvalidProofError",
    "OpenTimestamper",
    "TimestampInfo",
    "TimestampProof",
    "TimestampVerification",
    "sha256_file",
    "sha256_stream",
]
