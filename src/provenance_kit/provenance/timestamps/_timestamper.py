"""The OpenTimestamps client (internal module; import from the package)."""

from __future__ import annotations

import concurrent.futures
import fnmatch
import math
import os
import ssl
import time
import urllib.parse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from provenance_kit._logging import get_logger
from provenance_kit._typing import BytesLike
from provenance_kit.errors import ProvenanceKitError
from provenance_kit.provenance.timestamps import _http
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
    iter_nodes,
    load_ots,
    parse_detached,
    parse_timestamp,
    serialize_detached,
)

__all__ = [
    "DEFAULT_CALENDARS",
    "DEFAULT_MAX_RESPONSE_BYTES",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_UPGRADE_ALLOWLIST",
    "CalendarError",
    "OpenTimestamper",
]

_log = get_logger(__name__)

#: The calendars the official ``opentimestamps-client`` submits to when none
#: are given (``otsclient/cmds.py``, ``stamp_command``; also
#: ``opentimestamps.calendar.DEFAULT_AGGREGATORS`` in python-opentimestamps
#: 0.5.0). They are aggregators that forward to the public calendars.
DEFAULT_CALENDARS: tuple[str, ...] = (
    "https://a.pool.opentimestamps.org",
    "https://b.pool.opentimestamps.org",
    "https://a.pool.eternitywall.com",
    "https://ots.btc.catallaxy.com",
)

#: The calendars that :meth:`OpenTimestamper.upgrade` may contact by default:
#: python-opentimestamps 0.5.0 ``DEFAULT_CALENDAR_WHITELIST``, which the
#: official client also uses. ``*`` matches within the host name only.
DEFAULT_UPGRADE_ALLOWLIST: tuple[str, ...] = (
    "https://*.calendar.opentimestamps.org",
    "https://*.calendar.eternitywall.com",
    "https://*.calendar.catallaxy.com",
)

#: The per-calendar time limit of the official client (``--timeout``, 5 s).
DEFAULT_TIMEOUT_SECONDS = 5.0

#: The largest calendar response accepted (python-opentimestamps 0.5.0
#: ``MAX_RESPONSE_LENGTH``).
DEFAULT_MAX_RESPONSE_BYTES = 10_000

_ACCEPT = "application/vnd.opentimestamps.v1"
_NONCE_BYTES = 16  # what the official client appends for privacy
_MAX_PENDING_REQUESTS = 32
_MAX_ATTESTATION_CHECKS = 64
_MAX_WORKERS = 8
_GRACE_SECONDS = 0.5
# Header, digest, nonce and hash operations of a fresh proof, rounded up.
_STAMP_OVERHEAD = 128


class CalendarError(ProvenanceKitError):
    """Too few calendars answered a stamp request.

    Attributes:
        failures: The calendars that failed and why.
        succeeded: How many calendars answered.
        required: How many were required (``min_calendars``).
    """

    def __init__(
        self, failures: tuple[CalendarFailure, ...], *, succeeded: int, required: int
    ) -> None:
        """Initializes the error."""
        details = "; ".join(f"{_host(f.url)}: {f.reason}" for f in failures[:8])
        super().__init__(
            f"{succeeded} calendar(s) answered, {required} required"
            + (f" ({details})" if details else "")
        )
        self.failures = failures
        self.succeeded = succeeded
        self.required = required


@dataclass(frozen=True, slots=True)
class _AllowPattern:
    scheme: str
    host: str
    port: int | None
    path: str

    def matches(self, url: urllib.parse.SplitResult) -> bool:
        try:
            port = url.port
        except ValueError:
            return False
        return (
            url.scheme == self.scheme
            and url.path == self.path
            and url.hostname is not None
            and fnmatch.fnmatchcase(url.hostname, self.host)
            and port == self.port
        )


class OpenTimestamper:
    """Creates, upgrades, inspects and verifies OpenTimestamps proofs.

    Proofs commit to a SHA-256 digest (see :func:`sha256_file`). Like the
    official client, :meth:`stamp` appends a random 16-byte nonce and hashes
    again before submitting, so calendars never see the digest itself. Only
    that commitment is sent; media never leaves the process.

    Calendar responses are untrusted: requests use ``https`` only (unless
    ``allow_insecure_http``), have a total deadline, cap the response size,
    follow redirects only within the same origin and ignore proxy
    environment variables, and responses are parsed with
    python-opentimestamps' strict parser. :meth:`upgrade` contacts only the
    calendars that pending attestations name **and** that match the
    allow-list, because those URIs come from the (untrusted) proof.

    :meth:`info` and :meth:`verify` never use the network.

    Thread safety: safe to share between threads (the configuration is
    immutable; each call uses its own connections).

    Args:
        calendars: The calendar URLs to submit to (the official client's
            :data:`DEFAULT_CALENDARS` by default). They are also allowed for
            :meth:`upgrade`.
        min_calendars: How many calendars must answer for :meth:`stamp` to
            succeed. The default, 1, gives a proof as soon as one calendar
            answers; the official client requires 2 (``-m``), which survives
            the loss of one calendar. Use 2 or more in production.
        timeout_seconds: The deadline for one operation; calendars are
            contacted in parallel. 5 s is the official client's default.
        upgrade_allowlist: URL patterns (``*`` matches within the host) that
            :meth:`upgrade` may contact, in addition to ``calendars``.
        max_response_bytes: The largest calendar response accepted.
        max_proof_bytes: The largest proof parsed or produced.
        allow_insecure_http: Accept ``http`` calendars. For tests only.
        ssl_context: The TLS context; ``None`` uses the system trust store.
        user_agent: The ``User-Agent`` header.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If a URL is not ``https`` (or ``http`` when allowed),
            ``min_calendars`` is out of range or a limit is not positive.
    """

    def __init__(
        self,
        calendars: Sequence[str] = DEFAULT_CALENDARS,
        *,
        min_calendars: int = 1,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        upgrade_allowlist: Sequence[str] = DEFAULT_UPGRADE_ALLOWLIST,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
        max_proof_bytes: int = DEFAULT_MAX_PROOF_BYTES,
        allow_insecure_http: bool = False,
        ssl_context: ssl.SSLContext | None = None,
        user_agent: str = "provenance-kit",
    ) -> None:
        """Validates and stores the configuration."""
        if isinstance(calendars, str) or not isinstance(calendars, Sequence):
            raise TypeError("calendars must be a sequence of URLs")
        if isinstance(upgrade_allowlist, str) or not isinstance(upgrade_allowlist, Sequence):
            raise TypeError("upgrade_allowlist must be a sequence of URL patterns")
        if not isinstance(allow_insecure_http, bool):
            raise TypeError("allow_insecure_http must be a bool")
        self._allow_http = allow_insecure_http
        urls = tuple(dict.fromkeys(calendars))
        for url in urls:
            _http.check_url(url, allow_http=allow_insecure_http)
        if not urls:
            raise ValueError("at least one calendar is needed")
        required: object = min_calendars
        if isinstance(required, bool) or not isinstance(required, int):
            raise TypeError("min_calendars must be an int")
        if not 1 <= min_calendars <= len(urls):
            raise ValueError("min_calendars must be between 1 and the number of calendars")
        _check_positive("timeout_seconds", timeout_seconds)
        _check_positive_int("max_response_bytes", max_response_bytes)
        _check_positive_int("max_proof_bytes", max_proof_bytes)
        if ssl_context is not None and not isinstance(ssl_context, ssl.SSLContext):
            raise TypeError("ssl_context must be an ssl.SSLContext or None")
        agent: object = user_agent
        if not isinstance(agent, str) or not agent.isprintable():
            raise ValueError("user_agent must be a printable str")
        patterns = [_pattern(p, allow_http=allow_insecure_http) for p in upgrade_allowlist]
        patterns.extend(_pattern(url, allow_http=allow_insecure_http) for url in urls)

        self._calendars = urls
        self._min_calendars = min_calendars
        self._timeout = float(timeout_seconds)
        self._allowlist = tuple(patterns)
        self._max_response = max_response_bytes
        self._max_proof = max_proof_bytes
        self._ssl_context = ssl_context
        self._headers = {"Accept": _ACCEPT, "User-Agent": user_agent}

    @property
    def calendars(self) -> tuple[str, ...]:
        """The calendars :meth:`stamp` submits to."""
        return self._calendars

    @property
    def min_calendars(self) -> int:
        """How many calendars must answer a stamp request."""
        return self._min_calendars

    # ------------------------------------------------------------ network

    def stamp(self, digest: BytesLike) -> TimestampProof:
        """Submits a SHA-256 digest to the calendars and returns a pending proof.

        Args:
            digest: The 32-byte SHA-256 digest of the data.

        Returns:
            A proof with one pending attestation per calendar that answered;
            :attr:`TimestampProof.failed_calendars` lists the others. Upgrade
            it after a few hours to get Bitcoin attestations.

        Raises:
            TypeError: If ``digest`` is not bytes-like.
            ValueError: If ``digest`` is not 32 bytes.
            BackendUnavailableError: If opentimestamps is not installed
                (extra ``timestamps``).
            CalendarError: If fewer than ``min_calendars`` calendars answered.
        """
        message = _digest(digest)
        ots = load_ots()
        file_timestamp = ots.timestamp.Timestamp(message)
        nonced = file_timestamp.ops.add(ots.op.OpAppend(os.urandom(_NONCE_BYTES)))
        commitment = nonced.ops.add(ots.op.OpSHA256())

        jobs: list[tuple[str, Callable[[], Any]]] = [
            (url, _bind(self._submit, url, commitment.msg)) for url in self._calendars
        ]
        succeeded = 0
        failures: list[CalendarFailure] = []
        budget = self._max_proof - _STAMP_OVERHEAD
        for url, outcome in self._run_parallel(jobs):
            if isinstance(outcome, CalendarFailure):
                failures.append(outcome)
                continue
            size, timestamp = outcome
            if size > budget:
                failures.append(CalendarFailure(url, "proof would exceed max_proof_bytes"))
                continue
            budget -= size
            commitment.merge(timestamp)
            succeeded += 1
        _log.info(
            "OpenTimestamps stamp: %d of %d calendars answered",
            succeeded,
            len(self._calendars),
        )
        if succeeded < self._min_calendars:
            raise CalendarError(tuple(failures), succeeded=succeeded, required=self._min_calendars)
        detached = ots.timestamp.DetachedTimestampFile(ots.op.OpSHA256(), file_timestamp)
        data = serialize_detached(detached)
        return TimestampProof(data, failed_calendars=tuple(failures))

    def upgrade(self, proof: TimestampProof) -> TimestampProof:
        """Asks the calendars for the attestations a pending proof lacks.

        Each pending attestation names a calendar URI; only URIs that match
        the allow-list are contacted (others are reported as failures). A
        calendar that has not yet committed to Bitcoin answers "not found"
        and the attestation simply stays pending. Calendar failures never
        raise: they are listed in :attr:`TimestampProof.failed_calendars`.

        Args:
            proof: The proof (pending or partly complete).

        Returns:
            The proof with whatever the calendars returned merged in.

        Raises:
            TypeError: If ``proof`` is not a :class:`TimestampProof`.
            BackendUnavailableError: If opentimestamps is not installed.
            MediaLimitExceededError: If the proof exceeds ``max_proof_bytes``.
            InvalidProofError: If the proof is malformed.
        """
        detached = self._parse(proof)
        pending = self._pending(detached.timestamp)
        failures: list[CalendarFailure] = []
        jobs: list[tuple[str, Callable[[], Any]]] = []
        targets: list[Any] = []
        for index, (node, uri) in enumerate(pending):
            if index >= _MAX_PENDING_REQUESTS:
                failures.append(CalendarFailure(uri, "too many pending attestations"))
            elif not self._upgrade_allowed(uri):
                failures.append(CalendarFailure(uri, "not in the upgrade allow-list"))
            else:
                jobs.append((uri, _bind(self._fetch, uri, node.msg)))
                targets.append(node)
        budget = self._max_proof - len(proof.data)
        merged = 0
        for node, (uri, outcome) in zip(targets, self._run_parallel(jobs), strict=True):
            if isinstance(outcome, CalendarFailure):
                failures.append(outcome)
            elif outcome is not None:
                size, timestamp = outcome
                if size > budget:
                    failures.append(CalendarFailure(uri, "proof would exceed max_proof_bytes"))
                    continue
                budget -= size
                node.merge(timestamp)
                merged += 1
        _log.info(
            "OpenTimestamps upgrade: %d pending, %d upgraded, %d failed",
            len(pending),
            merged,
            len(failures),
        )
        return TimestampProof(serialize_detached(detached), failed_calendars=tuple(failures))

    # ------------------------------------------------------------ offline

    def info(self, proof: TimestampProof) -> TimestampInfo:
        """Lists a proof's pending calendars and Bitcoin attestations (offline).

        Args:
            proof: The proof.

        Returns:
            What the proof contains. Nothing is verified.

        Raises:
            TypeError: If ``proof`` is not a :class:`TimestampProof`.
            BackendUnavailableError: If opentimestamps is not installed.
            MediaLimitExceededError: If the proof exceeds ``max_proof_bytes``.
            InvalidProofError: If the proof is malformed.
        """
        detached = self._parse(proof)
        ots = load_ots()
        pending: set[str] = set()
        bitcoin: set[tuple[int, bytes]] = set()
        other = 0
        for node in iter_nodes(detached.timestamp):
            for attestation in node.attestations:
                if isinstance(attestation, ots.notary.PendingAttestation):
                    pending.add(str(attestation.uri))
                elif isinstance(attestation, ots.notary.BitcoinBlockHeaderAttestation):
                    bitcoin.add((int(attestation.height), bytes(node.msg)))
                else:
                    other += 1
        return TimestampInfo(
            digest=proof.digest,
            pending_calendars=tuple(sorted(pending)),
            bitcoin_attestations=tuple(
                BitcoinAttestation(height=height, merkle_root=root)
                for height, root in sorted(bitcoin)
            ),
            other_attestations=other,
        )

    def verify(
        self,
        proof: TimestampProof,
        digest: BytesLike,
        block_source: BitcoinBlockSource,
    ) -> TimestampVerification:
        """Checks a proof against a digest and the Bitcoin block headers (offline).

        The proof's operations are recomputed from ``digest`` by the parser,
        so every attestation's message is a commitment to the digest. Each
        Bitcoin attestation (at most 64, lowest heights first) is then
        checked against ``block_source``: the block's merkle root must equal
        the attestation's message. Pending attestations prove nothing and are
        ignored.

        Args:
            proof: The proof.
            digest: The SHA-256 digest the proof should be for.
            block_source: Supplies block headers by height.

        Returns:
            The checks; :attr:`TimestampVerification.verified` summarizes.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If ``digest`` is not 32 bytes.
            BackendUnavailableError: If opentimestamps is not installed.
            MediaLimitExceededError: If the proof exceeds ``max_proof_bytes``.
            InvalidProofError: If the proof is malformed.
            Exception: Whatever ``block_source`` raises is propagated.
        """
        expected = _digest(digest)
        info = self.info(proof)
        if info.digest != expected:
            return TimestampVerification(digest_matches=False, checks=())
        ots = load_ots()
        checks = []
        for attestation in info.bitcoin_attestations[:_MAX_ATTESTATION_CHECKS]:
            header = block_source.block_header(attestation.height)
            if header is None:
                checks.append(
                    AttestationCheck(attestation.height, False, reason="block not available")
                )
                continue
            if not isinstance(header, BitcoinBlockHeader):
                raise TypeError("block_source must return a BitcoinBlockHeader or None")
            try:
                block_time = ots.notary.BitcoinBlockHeaderAttestation(
                    attestation.height
                ).verify_against_blockheader(attestation.merkle_root, _HeaderView(header))
            except ots.notary.VerificationError:
                checks.append(
                    AttestationCheck(attestation.height, False, reason="merkle root mismatch")
                )
                continue
            checks.append(AttestationCheck(attestation.height, True, block_time=int(block_time)))
        return TimestampVerification(digest_matches=True, checks=tuple(checks))

    # ------------------------------------------------------------ helpers

    def _parse(self, proof: TimestampProof) -> Any:
        if not isinstance(proof, TimestampProof):
            raise TypeError("proof must be a TimestampProof")
        return parse_detached(proof.data, max_bytes=self._max_proof)

    def _pending(self, timestamp: Any) -> list[tuple[Any, str]]:
        ots = load_ots()
        pending = []
        seen: set[tuple[str, bytes]] = set()
        for node in iter_nodes(timestamp):
            for attestation in node.attestations:
                if isinstance(attestation, ots.notary.PendingAttestation):
                    key = (str(attestation.uri), bytes(node.msg))
                    if key not in seen:
                        seen.add(key)
                        pending.append((node, key[0]))
        pending.sort(key=lambda item: item[1])
        return pending

    def _upgrade_allowed(self, uri: str) -> bool:
        try:
            parts = _http.check_url(uri, allow_http=self._allow_http)
        except (TypeError, ValueError):
            return False
        return any(pattern.matches(parts) for pattern in self._allowlist)

    def _submit(self, url: str, commitment: bytes) -> tuple[int, Any]:
        response = self._request("POST", urllib.parse.urljoin(url, "digest"), body=commitment)
        if response.status != 200:
            raise _http.HttpFailure(f"http status {response.status}")
        return len(response.body), parse_timestamp(response.body, commitment)

    def _fetch(self, uri: str, commitment: bytes) -> tuple[int, Any] | None:
        path = "timestamp/" + commitment.hex()
        response = self._request("GET", urllib.parse.urljoin(uri, path), body=None)
        if response.status == 404:
            return None  # not committed to Bitcoin yet
        if response.status != 200:
            raise _http.HttpFailure(f"http status {response.status}")
        return len(response.body), parse_timestamp(response.body, commitment)

    def _request(self, method: str, url: str, *, body: bytes | None) -> _http.HttpResponse:
        return _http.request(
            method,
            url,
            body=body,
            headers=self._headers,
            timeout_seconds=self._timeout,
            max_response_bytes=self._max_response,
            allow_http=self._allow_http,
            ssl_context=self._ssl_context,
        )

    def _run_parallel(self, jobs: list[tuple[str, Callable[[], Any]]]) -> list[tuple[str, Any]]:
        """Runs the jobs in threads; failures become :class:`CalendarFailure`."""
        if not jobs:
            return []
        deadline = time.monotonic() + self._timeout + _GRACE_SECONDS
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=min(_MAX_WORKERS, len(jobs)), thread_name_prefix="pk-ots"
        )
        try:
            futures = [(url, executor.submit(job)) for url, job in jobs]
            concurrent.futures.wait(
                [future for _, future in futures], timeout=max(0.0, deadline - time.monotonic())
            )
        finally:
            # Requests end at their own deadline; never wait longer than ours.
            executor.shutdown(wait=False, cancel_futures=True)
        results: list[tuple[str, Any]] = []
        for url, future in futures:
            if not future.done() or future.cancelled():
                results.append((url, CalendarFailure(url, "timeout")))
                continue
            error = future.exception()
            if error is None:
                results.append((url, future.result()))
            elif isinstance(error, _http.HttpFailure):
                results.append((url, CalendarFailure(url, error.reason)))
            elif isinstance(error, InvalidProofError):
                results.append((url, CalendarFailure(url, "malformed response")))
            elif isinstance(error, (KeyboardInterrupt, SystemExit, MemoryError)):
                raise error
            else:
                _log.debug("calendar request failed (%s)", type(error).__name__)
                results.append((url, CalendarFailure(url, "request failed")))
        return results


@dataclass(frozen=True, slots=True)
class _HeaderView:
    """The attribute names python-opentimestamps reads from a block header."""

    header: BitcoinBlockHeader

    @property
    def hashMerkleRoot(self) -> bytes:  # python-bitcoinlib's attribute name
        return self.header.merkle_root

    @property
    def nTime(self) -> int:  # python-bitcoinlib's attribute name
        return self.header.time


def _bind(function: Callable[[str, bytes], Any], url: str, message: bytes) -> Callable[[], Any]:
    def job() -> Any:
        return function(url, message)

    return job


def _pattern(value: str, *, allow_http: bool) -> _AllowPattern:
    if not isinstance(value, str):
        raise TypeError("allow-list entries must be str")
    parts = _http.check_url(value.replace("*", "wildcard-x"), allow_http=allow_http)
    host = urllib.parse.urlsplit(value).hostname or ""
    return _AllowPattern(scheme=parts.scheme, host=host, port=parts.port, path=parts.path)


def _host(url: str) -> str:
    return urllib.parse.urlsplit(url).hostname or "?"


def _digest(value: BytesLike) -> bytes:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError(f"digest must be bytes-like, got {type(value).__name__}")
    digest = bytes(memoryview(value).cast("B"))
    if len(digest) != 32:
        raise ValueError(f"a SHA-256 digest is 32 bytes, got {len(digest)}")
    return digest


def _check_positive(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")


def _check_positive_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
