"""Integration tests of the timestamps module with python-opentimestamps (extra ``timestamps``).

A local mock calendar (tests/provenance/_calendar.py) serves real serialized
timestamps; nothing here needs the internet except the test marked
``network``.
"""

from __future__ import annotations

import hashlib
import random
import time

import pytest

from provenance_kit.errors import MediaLimitExceededError, UnsupportedMediaError
from provenance_kit.provenance.timestamps import (
    BitcoinBlockHeader,
    CalendarError,
    CalendarFailure,
    InvalidProofError,
    OpenTimestamper,
    TimestampProof,
)
from tests.support import SEED

pytestmark = pytest.mark.integration

try:
    import bitcoin  # noqa: F401 - python-bitcoinlib, a dependency of opentimestamps
    import opentimestamps  # noqa: F401
except Exception:
    pytest.skip(
        'needs opentimestamps: pip install "provenance-kit[timestamps]"', allow_module_level=True
    )

from opentimestamps.core.notary import PendingAttestation  # noqa: E402
from opentimestamps.core.op import OpAppend, OpSHA256  # noqa: E402
from opentimestamps.core.serialize import (  # noqa: E402
    BytesDeserializationContext,
    BytesSerializationContext,
)
from opentimestamps.core.timestamp import DetachedTimestampFile, Timestamp  # noqa: E402

from tests.provenance import _ots_fixtures as fixtures  # noqa: E402
from tests.provenance._calendar import BLOCK_HEIGHT, BLOCK_TIME, MockCalendar  # noqa: E402

DIGEST = hashlib.sha256(b"the asset").digest()


class BlockSource:
    """A BitcoinBlockSource backed by a dict."""

    def __init__(self, headers: dict[int, BitcoinBlockHeader]) -> None:
        self.headers = headers
        self.asked: list[int] = []

    def block_header(self, height: int) -> BitcoinBlockHeader | None:
        self.asked.append(height)
        return self.headers.get(height)


def timestamper(*urls: str, **kwargs: object) -> OpenTimestamper:
    """A client for the mock calendars; one answer suffices unless told otherwise.

    The library default (two calendars must answer) is exercised by
    ``test_min_calendars_and_failure_reporting``.
    """
    kwargs.setdefault("timeout_seconds", 2.0)
    kwargs.setdefault("min_calendars", 1)
    return OpenTimestamper(urls, allow_insecure_http=True, **kwargs)  # type: ignore[arg-type]


def detached_bytes(root: Timestamp) -> bytes:
    context = BytesSerializationContext()
    DetachedTimestampFile(OpSHA256(), root).serialize(context)
    return context.getbytes()


# ------------------------------------------------------------------ the full cycle


def test_stamp_upgrade_info_verify():
    with MockCalendar() as first, MockCalendar() as second:
        client = timestamper(first.url, second.url, min_calendars=2)
        proof = client.stamp(DIGEST)
        assert proof.failed_calendars == ()
        assert proof.digest == DIGEST
        info = client.info(proof)
        assert info.pending_calendars == tuple(sorted((first.url, second.url)))
        assert not info.is_complete

        # Only a nonced commitment leaves the process, never the digest.
        sent = [r.body for r in (*first.requests, *second.requests)]
        assert all(len(body) == 32 for body in sent)
        assert DIGEST not in sent
        assert sent[0] == sent[1]

        still_pending = client.upgrade(proof)
        assert still_pending.failed_calendars == ()
        assert not client.info(still_pending).is_complete

        block = first.confirm()
        upgraded = client.upgrade(proof)
        assert upgraded.failed_calendars == ()
        info = client.info(upgraded)
        assert info.is_complete
        assert [a.height for a in info.bitcoin_attestations] == [BLOCK_HEIGHT]
        assert info.pending_calendars == tuple(sorted((first.url, second.url)))

    source = BlockSource({BLOCK_HEIGHT: BitcoinBlockHeader(block.merkle_root, BLOCK_TIME)})
    result = client.verify(upgraded, DIGEST, source)
    assert result.verified
    assert result.attested_time == BLOCK_TIME
    assert source.asked == [BLOCK_HEIGHT]

    assert not client.verify(upgraded, hashlib.sha256(b"other").digest(), source).digest_matches
    missing = client.verify(upgraded, DIGEST, BlockSource({}))
    assert not missing.verified
    assert missing.checks[0].reason == "block not available"
    wrong = BlockSource({BLOCK_HEIGHT: BitcoinBlockHeader(b"\x00" * 32, BLOCK_TIME)})
    mismatch = client.verify(upgraded, DIGEST, wrong)
    assert not mismatch.verified
    assert mismatch.checks[0].reason == "merkle root mismatch"
    assert not client.verify(proof, DIGEST, source).verified  # pending proves nothing


def test_serialization_round_trip_is_the_standard_format():
    with MockCalendar() as calendar:
        client = timestamper(calendar.url)
        proof = client.stamp(DIGEST)
    data = proof.to_bytes()
    assert data.startswith(b"\x00OpenTimestamps\x00\x00Proof\x00")
    again = TimestampProof.from_bytes(data)
    assert again == TimestampProof(data)
    detached = DetachedTimestampFile.deserialize(BytesDeserializationContext(data))
    assert detached.file_digest == DIGEST


def test_min_calendars_and_failure_reporting():
    with MockCalendar() as good, MockCalendar(mode="status_500") as bad:
        proof = timestamper(good.url, bad.url).stamp(DIGEST)
        assert proof.failed_calendars == (CalendarFailure(bad.url, "http status 500"),)
        with pytest.raises(CalendarError) as info:  # the default: two must answer
            OpenTimestamper(
                (good.url, bad.url), allow_insecure_http=True, timeout_seconds=2.0
            ).stamp(DIGEST)
    assert info.value.succeeded == 1
    assert info.value.required == 2
    assert info.value.failures == (CalendarFailure(bad.url, "http status 500"),)


@pytest.mark.parametrize(
    ("mode", "reason"),
    [
        ("garbage", "malformed response"),
        ("trailing", "malformed response"),
        ("oversized", "response too large"),
        ("huge_declared", "response too large"),
        ("oversized_chunked", "response too large"),
        ("slow_body", "timeout"),
        ("slow_headers", "timeout"),
        ("redirect_other", "redirect refused"),
        ("redirect_https", "redirect refused"),
        ("redirect_loop", "too many redirects"),
    ],
)
def test_hostile_calendars_are_contained(mode, reason):
    with MockCalendar(mode=mode) as hostile, MockCalendar() as honest:
        started = time.monotonic()
        proof = timestamper(hostile.url, honest.url, timeout_seconds=1.0).stamp(DIGEST)
        assert time.monotonic() - started < 3.0
    assert proof.failed_calendars == (CalendarFailure(hostile.url, reason),)
    assert OpenTimestamper().info(proof).pending_calendars == (honest.url,)


def test_same_origin_redirect_is_followed_for_stamping():
    with MockCalendar(mode="redirect_same") as calendar:
        proof = timestamper(calendar.url).stamp(DIGEST)
    assert proof.failed_calendars == ()
    assert [r.path for r in calendar.requests] == ["/digest", "/r/digest"]


def test_https_calendar_with_a_custom_trust_store(tmp_path):
    pytest.importorskip("cryptography", reason='needs cryptography: "provenance-kit[c2pa]"')
    from tests.provenance._certs import make_tls_contexts

    server_context, client_context = make_tls_contexts(tmp_path)
    with MockCalendar(tls=server_context) as calendar:
        assert calendar.url.startswith("https://")
        proof = OpenTimestamper((calendar.url,), min_calendars=1, ssl_context=client_context).stamp(
            DIGEST
        )
        assert proof.failed_calendars == ()
        with pytest.raises(CalendarError) as info:  # the system store does not know the test CA
            OpenTimestamper((calendar.url,), min_calendars=1, timeout_seconds=2.0).stamp(DIGEST)
    assert info.value.failures[0].reason == "tls error"


# ------------------------------------------------------------------ untrusted proofs


def proof_pending_at(uri: str, *, count: int = 1) -> TimestampProof:
    root = Timestamp(DIGEST)
    for index in range(count):
        tip = root.ops.add(OpAppend(bytes([index]) * 16)).ops.add(OpSHA256())
        tip.attestations.add(PendingAttestation(uri))
    return TimestampProof.from_bytes(detached_bytes(root))


def test_upgrade_contacts_only_allowed_calendars():
    with MockCalendar(mode="garbage") as canary:
        proof = proof_pending_at(canary.url)  # an attacker-supplied calendar URI
        client = OpenTimestamper(timeout_seconds=1.0)  # defaults: https public calendars only
        upgraded = client.upgrade(proof)
        assert canary.requests == []
    assert upgraded.failed_calendars == (
        CalendarFailure(canary.url, "not in the upgrade allow-list"),
    )
    assert upgraded.data == proof.data


def test_upgrade_with_hostile_calendar_keeps_the_proof():
    with MockCalendar(mode="garbage") as calendar:
        proof = proof_pending_at(calendar.url)
        upgraded = timestamper(calendar.url).upgrade(proof)
    assert upgraded.failed_calendars == (CalendarFailure(calendar.url, "malformed response"),)
    assert upgraded.data == proof.data


def test_upgrade_caps_the_number_of_requests():
    with MockCalendar() as calendar:
        proof = proof_pending_at(calendar.url, count=40)
        upgraded = timestamper(calendar.url).upgrade(proof)
        assert len(calendar.requests) == 32
    reasons = [f.reason for f in upgraded.failed_calendars]
    assert reasons == ["too many pending attestations"] * 8


def test_hostile_proofs_raise_only_documented_errors():
    complete = fixtures.COMPLETE_FILE.read_bytes()
    rng = random.Random(SEED)
    outcomes = set()
    for _ in range(400):
        data = bytearray(complete)
        if rng.random() < 0.3:
            data = data[: rng.randrange(len(data))]
        else:
            for _ in range(rng.randint(1, 4)):
                data[rng.randrange(len(data))] = rng.randrange(256)
        try:
            proof = TimestampProof.from_bytes(bytes(data))
            OpenTimestamper().info(proof)
            outcomes.add("parsed")
        except (InvalidProofError, UnsupportedMediaError) as error:
            outcomes.add(type(error).__name__)
    assert "InvalidProofError" in outcomes


def test_proof_size_and_depth_limits():
    with pytest.raises(MediaLimitExceededError):
        TimestampProof.from_bytes(b"\x00" * 70_000)
    root = Timestamp(DIGEST)
    node = root
    for _ in range(300):  # deeper than the parser's recursion limit (256)
        node = node.ops.add(OpSHA256())
    node.attestations.add(PendingAttestation("https://deep.example"))
    with pytest.raises(InvalidProofError):
        TimestampProof.from_bytes(detached_bytes(root))


# ------------------------------------------------------------------ committed fixtures


def test_fixtures_are_reproducible():
    assert fixtures.pending_proof() == fixtures.PENDING_FILE.read_bytes()
    assert fixtures.complete_proof()[0] == fixtures.COMPLETE_FILE.read_bytes()


def test_fixture_contents_and_verification():
    client = OpenTimestamper()
    pending = TimestampProof.from_bytes(fixtures.PENDING_FILE.read_bytes())
    info = client.info(pending)
    assert info.digest == fixtures.FIXTURE_DIGEST
    assert info.pending_calendars == (fixtures.CALENDAR_URI,)
    assert not info.is_complete

    complete = TimestampProof.from_bytes(fixtures.COMPLETE_FILE.read_bytes())
    assert complete.data == fixtures.COMPLETE_FILE.read_bytes()  # canonical already
    info = client.info(complete)
    assert [a.height for a in info.bitcoin_attestations] == [fixtures.FIXTURE_HEIGHT]
    merkle_root = fixtures.complete_proof()[1]
    assert info.bitcoin_attestations[0].merkle_root == merkle_root
    source = BlockSource(
        {fixtures.FIXTURE_HEIGHT: BitcoinBlockHeader(merkle_root, fixtures.FIXTURE_TIME)}
    )
    result = client.verify(complete, hashlib.sha256(fixtures.FIXTURE_DATA).digest(), source)
    assert result.verified
    assert result.attested_time == fixtures.FIXTURE_TIME
    display = BitcoinBlockHeader.from_display_hex(merkle_root[::-1].hex(), fixtures.FIXTURE_TIME)
    assert display.merkle_root == merkle_root


# ------------------------------------------------------------------ network


@pytest.mark.network
def test_public_calendars():
    client = OpenTimestamper(timeout_seconds=10.0)
    digest = hashlib.sha256(random.Random().randbytes(32)).digest()
    proof = client.stamp(digest)
    info = client.info(proof)
    assert info.pending_calendars
    assert len(info.pending_calendars) + len(proof.failed_calendars) >= 1
    upgraded = client.upgrade(proof)  # a fresh stamp is still pending: calendars answer 404
    assert not client.info(upgraded).is_complete
    print("pending calendars:", info.pending_calendars)
    print("stamp failures:", proof.failed_calendars)
    print("upgrade failures:", upgraded.failed_calendars)
