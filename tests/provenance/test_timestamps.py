"""Unit tests of the timestamps module: no opentimestamps and no internet needed.

The calendar HTTP client is exercised against a local server
(tests/provenance/_calendar.py) in its hostile modes, which need no
python-opentimestamps.
"""

from __future__ import annotations

import hashlib
import io
import random
import sys
import time

import pytest

from provenance_kit.errors import (
    BackendUnavailableError,
    MediaLimitExceededError,
    UnsupportedMediaError,
)
from provenance_kit.provenance.timestamps import (
    DEFAULT_CALENDARS,
    DEFAULT_MIN_CALENDARS,
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_UPGRADE_ALLOWLIST,
    AttestationCheck,
    BitcoinAttestation,
    BitcoinBlockHeader,
    CalendarError,
    CalendarFailure,
    InvalidProofError,
    OpenTimestamper,
    TimestampInfo,
    TimestampProof,
    TimestampVerification,
    _http,
    sha256_file,
    sha256_stream,
)
from provenance_kit.provenance.timestamps._proof import HEADER_MAGIC
from tests.provenance._calendar import MockCalendar
from tests.support import HEAVY_MODULES, SEED, assert_imports_lightly

DIGEST = hashlib.sha256(b"provenance-kit").digest()


def proof_bytes(digest: bytes = DIGEST, *, version: int = 1, hash_tag: int = 0x08) -> bytes:
    return HEADER_MAGIC + bytes([version, hash_tag]) + digest + b"\x00" + b"\x01" * 9


def fetch(url: str, *, method: str = "GET", timeout: float = 3.0, cap: int = 10_000):
    return _http.request(
        method,
        url,
        body=b"x" * 32 if method == "POST" else None,
        headers={"Accept": "application/vnd.opentimestamps.v1", "User-Agent": "tests"},
        timeout_seconds=timeout,
        max_response_bytes=cap,
        allow_http=True,
        ssl_context=None,
    )


def test_importing_the_module_is_light():
    assert_imports_lightly(
        ["provenance_kit.provenance.timestamps"],
        forbidden=(*HEAVY_MODULES, "bitcoin", "Cryptodome"),
    )


# ------------------------------------------------------------------ defaults


def test_defaults_are_the_official_clients():
    # otsclient/cmds.py stamp_command and opentimestamps.calendar (0.5.0).
    assert DEFAULT_CALENDARS == (
        "https://a.pool.opentimestamps.org",
        "https://b.pool.opentimestamps.org",
        "https://a.pool.eternitywall.com",
        "https://ots.btc.catallaxy.com",
    )
    assert DEFAULT_UPGRADE_ALLOWLIST == (
        "https://*.calendar.opentimestamps.org",
        "https://*.calendar.eternitywall.com",
        "https://*.calendar.catallaxy.com",
    )
    assert DEFAULT_TIMEOUT_SECONDS == 5.0
    timestamper = OpenTimestamper()
    assert timestamper.calendars == DEFAULT_CALENDARS
    assert timestamper.min_calendars == DEFAULT_MIN_CALENDARS == 2


def test_a_single_calendar_needs_an_explicit_minimum():
    with pytest.raises(ValueError, match="min_calendars=1"):
        OpenTimestamper(("https://my-calendar.example",))
    assert OpenTimestamper(("https://my-calendar.example",), min_calendars=1).min_calendars == 1


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"calendars": "https://one.example"}, TypeError),
        ({"calendars": ()}, ValueError),
        ({"calendars": ("http://plain.example",)}, ValueError),
        ({"calendars": ("https://user:pw@c.example",)}, ValueError),
        ({"calendars": ("https://c.example/?q=1",)}, ValueError),
        ({"calendars": ("https://c.example:99999",)}, ValueError),
        ({"calendars": ("ftp://c.example",), "allow_insecure_http": True}, ValueError),
        ({"min_calendars": 0}, ValueError),
        ({"min_calendars": 5}, ValueError),
        ({"min_calendars": True}, TypeError),
        ({"timeout_seconds": 0}, ValueError),
        ({"timeout_seconds": float("inf")}, ValueError),
        ({"max_response_bytes": 0}, ValueError),
        ({"max_proof_bytes": 1.5}, TypeError),
        ({"upgrade_allowlist": "https://x.example"}, TypeError),
        ({"upgrade_allowlist": ("http://*.example",)}, ValueError),
        ({"user_agent": "bad\nagent"}, ValueError),
        ({"ssl_context": "context"}, TypeError),
        ({"allow_insecure_http": "yes"}, TypeError),
    ],
)
def test_configuration_is_validated(kwargs, error):
    with pytest.raises(error):
        OpenTimestamper(**kwargs)


def test_upgrade_allow_list_matching():
    timestamper = OpenTimestamper(("https://my-calendar.example:8443",), min_calendars=1)
    allowed = timestamper._upgrade_allowed
    assert allowed("https://alice.btc.calendar.opentimestamps.org")
    assert allowed("https://finney.calendar.eternitywall.com")
    assert allowed("https://my-calendar.example:8443")  # configured calendars are allowed
    assert not allowed("https://my-calendar.example")  # the port is part of the origin
    assert not allowed("https://calendar.opentimestamps.org")  # "*." needs a label
    assert not allowed("http://alice.btc.calendar.opentimestamps.org")
    assert not allowed("https://alice.btc.calendar.opentimestamps.org/other")
    assert not allowed("https://alice.btc.calendar.opentimestamps.org.evil.example")
    assert not allowed("https://127.0.0.1")
    assert not allowed("file:///etc/passwd")
    assert not allowed("not a url")


def block_module(monkeypatch, name: str) -> None:
    """Makes ``name`` and its already-imported submodules unimportable."""
    for module in [m for m in sys.modules if m == name or m.startswith(name + ".")]:
        monkeypatch.setitem(sys.modules, module, None)
    monkeypatch.setitem(sys.modules, name, None)


def test_stamp_validates_the_digest_before_loading_the_library(monkeypatch):
    block_module(monkeypatch, "opentimestamps")
    # A closed local port: nothing can leave the machine even if the guard failed.
    timestamper = OpenTimestamper(
        ("http://127.0.0.1:9",), min_calendars=1, allow_insecure_http=True
    )
    with pytest.raises(TypeError):
        timestamper.stamp("00" * 32)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="32 bytes"):
        timestamper.stamp(b"\x00" * 20)
    with pytest.raises(BackendUnavailableError) as info:
        timestamper.stamp(DIGEST)
    assert info.value.extra == "timestamps"
    with pytest.raises(BackendUnavailableError):
        TimestampProof.from_bytes(proof_bytes())


# ------------------------------------------------------------------ proof objects


def test_proof_header_checks():
    proof = TimestampProof(proof_bytes())
    assert proof.digest == DIGEST
    assert proof.to_bytes() == proof_bytes()
    assert "data" not in repr(proof)
    with pytest.raises(InvalidProofError):
        TimestampProof(b"not a proof")
    with pytest.raises(InvalidProofError):
        TimestampProof(HEADER_MAGIC + b"\x01\x08" + DIGEST)  # no timestamp
    with pytest.raises(UnsupportedMediaError):
        TimestampProof(proof_bytes(version=2))
    with pytest.raises(UnsupportedMediaError, match="SHA-256"):
        TimestampProof(proof_bytes(hash_tag=0x02))  # SHA-1
    with pytest.raises(TypeError):
        TimestampProof("text")  # type: ignore[arg-type]


def test_from_bytes_names_its_cap_like_the_timestamper():
    # The size is checked before python-opentimestamps is needed.
    with pytest.raises(MediaLimitExceededError) as info:
        TimestampProof.from_bytes(proof_bytes(), max_proof_bytes=10)
    assert info.value.limit == "max_proof_bytes"
    assert info.value.maximum == 10
    with pytest.raises(TypeError):
        TimestampProof.from_bytes(proof_bytes(), max_bytes=10)  # type: ignore[call-arg]


def test_block_header_and_results():
    header = BitcoinBlockHeader.from_display_hex("00" * 31 + "ff", 1_700_000_000)
    assert header.merkle_root == b"\xff" + b"\x00" * 31
    with pytest.raises(ValueError):
        BitcoinBlockHeader(b"\x00" * 31, 1)
    with pytest.raises(TypeError):
        BitcoinBlockHeader(b"\x00" * 32, 1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        BitcoinBlockHeader.from_display_hex("zz", 1)

    checks = (
        AttestationCheck(10, verified=False, reason="block not available"),
        AttestationCheck(11, verified=True, block_time=200),
        AttestationCheck(12, verified=True, block_time=100),
    )
    assert TimestampVerification(True, checks).verified
    assert TimestampVerification(True, checks).attested_time == 100
    assert not TimestampVerification(False, checks).verified
    assert TimestampVerification(True, checks[:1]).attested_time is None
    info = TimestampInfo(DIGEST, ("https://c.example",), (BitcoinAttestation(1, b"r"),), 0)
    assert info.is_complete


def test_calendar_error_message_names_hosts_and_reasons_only():
    error = CalendarError(
        (CalendarFailure("https://a.example/path", "timeout"),), succeeded=0, required=1
    )
    assert str(error) == "0 calendar(s) answered, 1 required (a.example: timeout)"
    assert error.failures[0].reason == "timeout"


# ------------------------------------------------------------------ hashing


def test_sha256_helpers(tmp_path):
    data = random.Random(SEED).randbytes(300_000)
    expected = hashlib.sha256(data).digest()
    assert sha256_stream(io.BytesIO(data), chunk_bytes=7_919) == expected
    path = tmp_path / "media.bin"
    path.write_bytes(data)
    assert sha256_file(path) == expected
    assert sha256_file(str(path), chunk_bytes=65_536) == expected
    with pytest.raises(TypeError):
        sha256_stream(io.StringIO("text"))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        sha256_stream(io.BytesIO(data), chunk_bytes=0)


# ------------------------------------------------------------------ HTTP client


def test_url_checks():
    assert _http.check_url("https://c.example", allow_http=False).hostname == "c.example"
    for url in ("http://c.example", "https://", "https://u@c.example", "https://c.example#x"):
        with pytest.raises(ValueError):
            _http.check_url(url, allow_http=False)
    with pytest.raises(TypeError):
        _http.check_url(b"https://c.example", allow_http=False)  # type: ignore[arg-type]


def test_successful_request_sends_the_protocol_headers():
    with MockCalendar(mode="garbage") as server:
        response = fetch(f"{server.url}/digest", method="POST")
    assert response.status == 200
    assert response.body.startswith(b"\xff\x00garbage")
    request = server.requests[0]
    assert request.headers["Accept"] == "application/vnd.opentimestamps.v1"
    assert request.headers["User-Agent"] == "tests"
    assert request.body == b"x" * 32


@pytest.mark.parametrize("mode", ["oversized", "huge_declared", "oversized_chunked"])
def test_oversized_responses_are_refused(mode):
    with MockCalendar(mode=mode) as server:
        server.max_response = 10_000
        with pytest.raises(_http.HttpFailure, match="too large"):
            fetch(f"{server.url}/digest", method="POST", cap=10_000)


def test_a_body_shorter_than_declared_is_refused():
    # http.client's read1 reports the early end of file as an empty read.
    with MockCalendar(mode="truncated") as server, pytest.raises(_http.HttpFailure, match="trunc"):
        fetch(f"{server.url}/digest", method="POST")


@pytest.mark.parametrize("mode", ["slow_body", "slow_headers"])
def test_slow_servers_hit_the_total_deadline(mode):
    # On Linux, a read blocked on a socket that the watchdog shuts down returns
    # end of file instead of raising: it must still be a timeout, not a
    # (truncated) response.
    with MockCalendar(mode=mode) as server:
        started = time.monotonic()
        with pytest.raises(_http.HttpFailure, match="timeout"):
            fetch(f"{server.url}/digest", method="POST", timeout=1.0)
        assert time.monotonic() - started < 3.0


def test_same_origin_redirects_are_followed():
    with MockCalendar(mode="redirect_same") as server:
        response = fetch(f"{server.url}/timestamp/{'ab' * 32}")
    assert response.status == 404  # the calendar's "pending" answer, after one redirect
    assert [r.path for r in server.requests] == [
        f"/timestamp/{'ab' * 32}",
        f"/r/timestamp/{'ab' * 32}",
    ]


@pytest.mark.parametrize(
    ("mode", "reason"),
    [
        ("redirect_other", "redirect refused"),  # 127.0.0.1 -> localhost
        ("redirect_https", "redirect refused"),  # scheme change
        ("redirect_loop", "too many redirects"),
    ],
)
def test_unsafe_redirects_are_refused(mode, reason):
    with MockCalendar(mode=mode) as server:
        with pytest.raises(_http.HttpFailure, match=reason):
            fetch(f"{server.url}/digest", method="POST")
        assert all(r.path in ("/digest",) for r in server.requests)


def test_error_statuses_are_returned():
    with MockCalendar(mode="status_500") as server:
        assert fetch(f"{server.url}/digest", method="POST").status == 500


def test_proxy_environment_is_ignored(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    with MockCalendar(mode="garbage") as server:
        assert fetch(f"{server.url}/digest", method="POST").status == 200


def test_unreachable_calendar_fails_cleanly():
    with MockCalendar(mode="garbage") as server:
        url = server.url
    with pytest.raises(_http.HttpFailure) as info:
        fetch(f"{url}/digest", method="POST", timeout=1.0)
    assert info.value.reason in ("connection failed", "timeout")
