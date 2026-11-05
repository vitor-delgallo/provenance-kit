"""A local OpenTimestamps calendar for tests, with hostile behaviors.

``MockCalendar`` speaks the calendar protocol that the official client uses:

* ``POST /digest`` with a commitment: answers a serialized ``Timestamp`` (built
  with python-opentimestamps) that appends a nonce, hashes, and ends in a
  ``PendingAttestation`` naming the calendar's own URL;
* ``GET /timestamp/<hex>``: answers 404 until :meth:`confirm` is called, then
  the path from the commitment to a Bitcoin block-header attestation. The
  block is built with python-bitcoinlib: a coinbase plus a transaction whose
  ``OP_RETURN`` output carries the commitment, exactly the structure real
  calendars produce, so the proof goes commitment -> transaction -> txid ->
  merkle path -> merkle root.

The ``mode`` attribute switches to a hostile behavior (no python-opentimestamps
needed for those): ``garbage``, ``trailing``, ``oversized``,
``oversized_chunked``, ``huge_declared``, ``truncated``, ``slow_body``,
``slow_headers``, ``status_500``, ``redirect_other``, ``redirect_same``, ``redirect_loop`` and
``redirect_https``.
"""

from __future__ import annotations

import hashlib
import http.server
import os
import ssl
import threading
import time
from dataclasses import dataclass, field

BLOCK_HEIGHT = 840_000
BLOCK_TIME = 1_713_571_767  # any fixed header time


@dataclass
class Block:
    """A synthetic Bitcoin block that commits to calendar tips."""

    height: int
    merkle_root: bytes
    time: int


@dataclass
class Recorded:
    method: str
    path: str
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b""


class _Server(http.server.ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False
    calendar: MockCalendar


class _Handler(http.server.BaseHTTPRequestHandler):
    server: _Server
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:
        pass

    def _record(self, body: bytes = b"") -> None:
        self.server.calendar.requests.append(
            Recorded(self.command, self.path, dict(self.headers.items()), body)
        )

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        self._record(body)
        calendar = self.server.calendar
        if calendar.hostile(self, body):
            return
        path = self.path[2:] if self.path.startswith("/r/") else self.path
        if path != "/digest":
            self._send(404, b"not found")
            return
        self._send(200, calendar.stamp_response(body))

    def do_GET(self) -> None:
        self._record()
        calendar = self.server.calendar
        if calendar.hostile(self, b""):
            return
        path = self.path[2:] if self.path.startswith("/r/") else self.path
        prefix = "/timestamp/"
        if not path.startswith(prefix):
            self._send(404, b"not found")
            return
        try:
            commitment = bytes.fromhex(path[len(prefix) :])
        except ValueError:
            self._send(400, b"bad commitment")
            return
        response = calendar.upgrade_response(commitment)
        if response is None:
            self._send(404, b"Pending confirmation in Bitcoin blockchain")
        else:
            self._send(200, response)

    def _send(self, status: int, body: bytes, headers: dict[str, str] | None = None) -> None:
        self.send_response(status)
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class MockCalendar:
    """A calendar server on 127.0.0.1 (use as a context manager)."""

    def __init__(self, *, tls: ssl.SSLContext | None = None, mode: str = "normal") -> None:
        self.mode = mode
        self.requests: list[Recorded] = []
        self.max_response = 10_000
        self._tls = tls
        self._tips: dict[bytes, bytes] = {}  # commitment -> merkle tip
        self._lock = threading.Lock()
        self._confirmed = False
        self.block: Block | None = None
        self._block_stamps: dict[bytes, object] = {}
        self._server = _Server(("127.0.0.1", 0), _Handler)
        self._server.calendar = self
        if tls is not None:
            self._server.socket = tls.wrap_socket(self._server.socket, server_side=True)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> MockCalendar:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    @property
    def url(self) -> str:
        scheme = "https" if self._tls is not None else "http"
        return f"{scheme}://127.0.0.1:{self.port}"

    # ------------------------------------------------------------ protocol

    def stamp_response(self, message: bytes) -> bytes:
        from opentimestamps.core.notary import PendingAttestation
        from opentimestamps.core.op import OpAppend, OpSHA256
        from opentimestamps.core.timestamp import Timestamp

        stamp = Timestamp(message)
        tip = stamp.ops.add(OpAppend(os.urandom(8))).ops.add(OpSHA256())
        tip.attestations.add(PendingAttestation(self.url))
        with self._lock:
            self._tips[tip.msg] = tip.msg
        return _serialize(stamp)

    def confirm(self) -> Block:
        """Commits every tip received so far to a synthetic Bitcoin block."""
        from bitcoin.core import (
            COIN,
            COutPoint,
            CScript,
            CTransaction,
            CTxIn,
            CTxOut,
            b2lx,
        )
        from bitcoin.core import CBlock as _CBlock
        from bitcoin.core.script import OP_RETURN
        from opentimestamps.bitcoin import make_timestamp_from_block

        with self._lock:
            tips = list(self._tips)
        coinbase = CTransaction(
            [CTxIn(COutPoint(), CScript([BLOCK_HEIGHT.to_bytes(3, "little")]))],
            [CTxOut(50 * COIN, CScript([b"\x51"]))],
        )
        commitments = [
            CTransaction(
                [CTxIn(COutPoint(hashlib.sha256(tip).digest(), 0))],
                [CTxOut(0, CScript([OP_RETURN, tip]))],
            )
            for tip in tips
        ]
        unrooted = _CBlock(vtx=[coinbase, *commitments])
        block = _CBlock(
            nVersion=0x20000000,
            hashMerkleRoot=unrooted.calc_merkle_root(),
            nTime=BLOCK_TIME,
            nBits=0x1D00FFFF,
            vtx=[coinbase, *commitments],
        )
        stamps = {}
        for tip in tips:
            stamp = make_timestamp_from_block(tip, block, BLOCK_HEIGHT)
            assert stamp is not None, "commitment not found in the block"
            stamps[tip] = stamp
        with self._lock:
            self._block_stamps = stamps
            self._confirmed = True
            self.block = Block(BLOCK_HEIGHT, bytes(block.hashMerkleRoot), BLOCK_TIME)
        assert len(b2lx(block.hashMerkleRoot)) == 64
        return self.block

    def upgrade_response(self, commitment: bytes) -> bytes | None:
        with self._lock:
            stamp = self._block_stamps.get(commitment) if self._confirmed else None
        return None if stamp is None else _serialize(stamp)

    # ------------------------------------------------------------ hostility

    def hostile(self, handler: _Handler, body: bytes) -> bool:  # noqa: PLR0912 - a branch per mode
        mode = self.mode
        if mode == "normal":
            return False
        if mode == "garbage":
            handler._send(200, b"\xff\x00garbage, not a timestamp" * 3)
        elif mode == "trailing":
            handler._send(200, self.stamp_response(body) + b"\x00trailing")
        elif mode == "oversized":
            handler._send(200, b"\x00" * (self.max_response + 1))
        elif mode == "huge_declared":
            handler.send_response(200)
            handler.send_header("Content-Length", str(10**12))
            handler.end_headers()
            handler.wfile.write(b"\x00" * 16)
        elif mode == "oversized_chunked":
            handler.send_response(200)
            handler.send_header("Transfer-Encoding", "chunked")
            handler.end_headers()
            chunk = b"\x00" * 4096
            for _ in range(10):
                handler.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
            handler.wfile.write(b"0\r\n\r\n")
        elif mode == "truncated":
            handler.send_response(200)
            handler.send_header("Content-Length", "1000")
            handler.send_header("Connection", "close")
            handler.end_headers()
            handler.wfile.write(b"\x00" * 10)
            handler.close_connection = True
        elif mode == "slow_body":
            handler.send_response(200)
            handler.send_header("Content-Length", "1000")
            handler.end_headers()
            _trickle(handler, b"\x00" * 1000)
        elif mode == "slow_headers":
            handler.wfile.write(b"HTTP/1.1 200 OK\r\n")
            _trickle(handler, b"X-Slow: " + b"a" * 1000 + b"\r\n")
        elif mode == "status_500":
            handler._send(500, b"internal error")
        elif mode == "redirect_other":
            handler._send(302, b"", {"Location": f"http://localhost:{self.port}{handler.path}"})
        elif mode == "redirect_same":
            if handler.path.startswith("/r/"):
                return False
            handler._send(307, b"", {"Location": "/r" + handler.path})
        elif mode == "redirect_loop":
            handler._send(307, b"", {"Location": handler.path})
        elif mode == "redirect_https":
            handler._send(302, b"", {"Location": f"https://127.0.0.1:{self.port}/digest"})
        else:
            raise AssertionError(f"unknown mode {mode}")
        return True


def _trickle(handler: _Handler, data: bytes, *, delay: float = 0.1, limit: float = 10.0) -> None:
    started = time.monotonic()
    for byte in data:
        if time.monotonic() - started > limit:
            return
        try:
            handler.wfile.write(bytes([byte]))
            handler.wfile.flush()
        except OSError:
            return
        time.sleep(delay)


def _serialize(stamp: object) -> bytes:
    from opentimestamps.core.serialize import BytesSerializationContext

    context = BytesSerializationContext()
    stamp.serialize(context)  # type: ignore[attr-defined]
    return context.getbytes()
