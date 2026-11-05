"""Deterministic OpenTimestamps proofs, committed as test fixtures.

The fixtures in ``tests/provenance/fixtures/`` are produced by this module
(they are not copies of third-party proofs, so they carry this project's MIT
license). They have the structure of real proofs: the digest gets a nonce and
is hashed, a calendar prepends and hashes, and the complete proof continues
through a Bitcoin transaction whose ``OP_RETURN`` output holds the calendar
tip, the transaction id and the block's merkle path, ending in a
``BitcoinBlockHeaderAttestation``. The block is synthetic (built with
python-bitcoinlib), so the attestation verifies against
:data:`FIXTURE_HEADER`, not against the real chain.

Regenerate (only if the format intentionally changes)::

    python -m tests.provenance._ots_fixtures

The test suite checks that this module still produces the committed bytes,
which catches unintended serialization changes in python-opentimestamps.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

FIXTURES = Path(__file__).with_name("fixtures")
PENDING_FILE = FIXTURES / "pending.ots"
COMPLETE_FILE = FIXTURES / "complete.ots"

FIXTURE_DATA = b"provenance-kit fixture\n"
FIXTURE_DIGEST = hashlib.sha256(FIXTURE_DATA).digest()
CALENDAR_URI = "https://alice.calendar.example"
FIXTURE_HEIGHT = 900_000
FIXTURE_TIME = 1_750_000_000
_NONCE = bytes(range(16))
_CALENDAR_PREFIX = b"\x5a" * 8


def _calendar_tip():
    from opentimestamps.core.op import OpAppend, OpPrepend, OpSHA256
    from opentimestamps.core.timestamp import Timestamp

    root = Timestamp(FIXTURE_DIGEST)
    commitment = root.ops.add(OpAppend(_NONCE)).ops.add(OpSHA256())
    tip = commitment.ops.add(OpPrepend(_CALENDAR_PREFIX)).ops.add(OpSHA256())
    return root, tip


def _detached(root) -> bytes:
    from opentimestamps.core.op import OpSHA256
    from opentimestamps.core.serialize import BytesSerializationContext
    from opentimestamps.core.timestamp import DetachedTimestampFile

    context = BytesSerializationContext()
    DetachedTimestampFile(OpSHA256(), root).serialize(context)
    return context.getbytes()


def pending_proof() -> bytes:
    """A proof with one pending attestation."""
    from opentimestamps.core.notary import PendingAttestation

    root, tip = _calendar_tip()
    tip.attestations.add(PendingAttestation(CALENDAR_URI))
    return _detached(root)


def synthetic_block(tip: bytes):
    """A deterministic block whose second transaction commits to ``tip``."""
    from bitcoin.core import COIN, CBlock, COutPoint, CScript, CTransaction, CTxIn, CTxOut
    from bitcoin.core.script import OP_RETURN

    coinbase = CTransaction(
        [CTxIn(COutPoint(), CScript([FIXTURE_HEIGHT.to_bytes(3, "little")]))],
        [CTxOut(3 * COIN, CScript([b"\x51"]))],
    )
    commitment = CTransaction(
        [CTxIn(COutPoint(hashlib.sha256(b"funding").digest(), 0))],
        [CTxOut(0, CScript([OP_RETURN, tip]))],
    )
    other = CTransaction(
        [CTxIn(COutPoint(hashlib.sha256(b"other").digest(), 1))],
        [CTxOut(1000, CScript([b"\x52"]))],
    )
    transactions = [coinbase, commitment, other]
    merkle_root = CBlock(vtx=transactions).calc_merkle_root()
    return CBlock(
        nVersion=0x20000000,
        hashMerkleRoot=merkle_root,
        nTime=FIXTURE_TIME,
        nBits=0x1D00FFFF,
        vtx=transactions,
    )


def complete_proof() -> tuple[bytes, bytes]:
    """A proof with the pending attestation and a Bitcoin attestation.

    Returns:
        The proof and the block's merkle root (internal byte order).
    """
    from opentimestamps.bitcoin import make_timestamp_from_block
    from opentimestamps.core.notary import PendingAttestation

    root, tip = _calendar_tip()
    tip.attestations.add(PendingAttestation(CALENDAR_URI))
    block = synthetic_block(tip.msg)
    upgraded = make_timestamp_from_block(tip.msg, block, FIXTURE_HEIGHT)
    assert upgraded is not None
    tip.merge(upgraded)
    return _detached(root), bytes(block.hashMerkleRoot)


def write_fixtures() -> None:
    FIXTURES.mkdir(exist_ok=True)
    PENDING_FILE.write_bytes(pending_proof())
    COMPLETE_FILE.write_bytes(complete_proof()[0])


if __name__ == "__main__":
    write_fixtures()
