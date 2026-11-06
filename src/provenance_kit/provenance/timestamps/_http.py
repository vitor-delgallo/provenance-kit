"""A minimal HTTP client for untrusted calendar servers (internal).

Only the standard library is used (``http.client``). What it enforces:

* ``https`` only, unless the caller explicitly allows ``http`` (tests);
* a deadline for the whole exchange, every blocking step included, so that
  the calling thread (a non-daemon pool thread of the timestamper) is free
  shortly after it:

  - DNS: ``getaddrinfo`` has no timeout, so it runs in a daemon thread that
    the request stops waiting for at the deadline (at most
    ``_MAX_LOOKUPS`` such lookups run at once);
  - connect: each address of the host gets only the time that is left;
  - TLS handshake: bounded as a whole by the socket timeout, which is the
    time left when it starts;
  - headers and body: socket timeouts bound each read, and a watchdog shuts
    down every socket of the exchange at the deadline, so a server that
    trickles bytes cannot hold the call longer. A socket created after the
    watchdog fired is refused (the exchange is then a timeout), so the
    watchdog cannot be spent before the socket exists. Once it has fired the
    exchange is a timeout, whatever the read returned: on Linux a read
    blocked on a socket that is shut down returns end of file instead of
    raising, which must not pass for a complete body;
* a body shorter than its ``Content-Length`` is refused (``http.client``'s
  ``read1`` reports such an early end of file as an empty read);
* a response size cap, checked against ``Content-Length`` and while reading;
* redirects only to the same scheme, host and port, at most a few hops;
* no proxies, cookies or credentials. Environment proxy variables are
  ignored.
"""

from __future__ import annotations

import contextlib
import http.client
import socket
import ssl
import threading
import time
import urllib.parse
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

__all__ = ["HttpFailure", "HttpResponse", "check_url", "request"]

_REDIRECTS = frozenset({301, 302, 303, 307, 308})
_MAX_REDIRECTS = 3
_CHUNK = 16 * 1024
# Name lookups still running in their daemon threads, process-wide. A lookup
# the caller stopped waiting for keeps its slot until the resolver answers,
# so a slow resolver cannot pile up threads without bound.
_MAX_LOOKUPS = 32
_LOOKUP_SLOTS = threading.BoundedSemaphore(_MAX_LOOKUPS)

# One getaddrinfo() result: family, type, protocol, canonical name, address.
_AddressInfo: TypeAlias = tuple[socket.AddressFamily, socket.SocketKind, int, str, Any]


class HttpFailure(Exception):
    """A request failed; :attr:`reason` is a short, log-safe category.

    Attributes:
        reason: For example ``"timeout"``, ``"http status 500"``,
            ``"response too large"``, ``"truncated response"`` or
            ``"redirect refused"``.
    """

    def __init__(self, reason: str) -> None:
        """Initializes the failure."""
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class HttpResponse:
    """A complete response (internal)."""

    status: int
    body: bytes


def check_url(url: str, *, allow_http: bool) -> urllib.parse.SplitResult:
    """Validates a calendar URL.

    Args:
        url: The URL.
        allow_http: Whether ``http`` is acceptable.

    Returns:
        The parsed URL.

    Raises:
        TypeError: If ``url`` is not a str.
        ValueError: If the scheme is not allowed, the host is missing, or the
            URL has credentials, a query or a fragment.
    """
    if not isinstance(url, str):
        raise TypeError("calendar URLs must be str")
    parts = urllib.parse.urlsplit(url)
    schemes = ("https", "http") if allow_http else ("https",)
    if parts.scheme not in schemes:
        raise ValueError(
            "calendar URLs must use https" + (" or http" if allow_http else "") + f": {url!r}"
        )
    if not parts.hostname:
        raise ValueError(f"calendar URL has no host: {url!r}")
    if parts.username is not None or parts.password is not None:
        raise ValueError("calendar URLs must not contain credentials")
    if parts.query or parts.fragment:
        raise ValueError(f"calendar URLs must not have a query or fragment: {url!r}")
    try:
        _ = parts.port
    except ValueError:
        raise ValueError(f"calendar URL has an invalid port: {url!r}") from None
    return parts


def request(
    method: str,
    url: str,
    *,
    body: bytes | None,
    headers: dict[str, str],
    timeout_seconds: float,
    max_response_bytes: int,
    allow_http: bool,
    ssl_context: ssl.SSLContext | None,
) -> HttpResponse:
    """Performs one request with a total deadline and a size cap.

    Args:
        method: ``"GET"`` or ``"POST"``.
        url: The absolute URL (validated with :func:`check_url`).
        body: The request body, if any.
        headers: Request headers.
        timeout_seconds: The deadline for the whole exchange, redirects
            included.
        max_response_bytes: The largest body accepted.
        allow_http: Whether ``http`` URLs are acceptable.
        ssl_context: The TLS context; ``None`` uses the system trust store
            with hostname checking.

    Returns:
        The final response (any status other than a redirect).

    Raises:
        HttpFailure: On any network, protocol, size, redirect or time failure.
    """
    deadline = time.monotonic() + timeout_seconds
    current = check_url(url, allow_http=allow_http)
    origin = (current.scheme, current.hostname, current.port)
    for _ in range(_MAX_REDIRECTS + 1):
        status, location, payload = _exchange(
            method,
            current,
            body=body,
            headers=headers,
            deadline=deadline,
            max_response_bytes=max_response_bytes,
            ssl_context=ssl_context,
        )
        if status not in _REDIRECTS:
            return HttpResponse(status=status, body=payload)
        if not location:
            raise HttpFailure("redirect without a location")
        target = urllib.parse.urlsplit(urllib.parse.urljoin(current.geturl(), location))
        try:
            same_origin = (target.scheme, target.hostname, target.port) == origin
        except ValueError:
            same_origin = False
        if not same_origin or target.username is not None or target.password is not None:
            raise HttpFailure("redirect refused")
        if status == 303:
            method, body = "GET", None
        current = target
    raise HttpFailure("too many redirects")


def _exchange(
    method: str,
    url: urllib.parse.SplitResult,
    *,
    body: bytes | None,
    headers: dict[str, str],
    deadline: float,
    max_response_bytes: int,
    ssl_context: ssl.SSLContext | None,
) -> tuple[int, str | None, bytes]:
    remaining = _remaining(deadline)
    host = str(url.hostname)
    connection: http.client.HTTPConnection
    tls: ssl.SSLContext | None = None
    if url.scheme == "https":
        tls = ssl_context if ssl_context is not None else ssl.create_default_context()
        connection = http.client.HTTPSConnection(host, url.port, timeout=remaining, context=tls)
    else:
        connection = http.client.HTTPConnection(host, url.port, timeout=remaining)
    watchdog = _Watchdog(deadline)
    try:
        watchdog.start()
        # Connected here rather than by http.client, so that every step has
        # the deadline; http.client then uses the socket as it is.
        connection.sock = _connect(
            host, connection.port, tls=tls, deadline=deadline, watchdog=watchdog
        )
        target = (url.path or "/") + (f"?{url.query}" if url.query else "")
        connection.request(method, target, body=body, headers=headers)
        response = connection.getresponse()
        location = response.getheader("Location")
        declared = response.getheader("Content-Length")
        if (
            declared is not None
            and declared.strip().isdigit()
            and int(declared) > max_response_bytes
        ):
            raise HttpFailure("response too large")
        payload = _read_capped(response, deadline, max_response_bytes)
        if watchdog.fired:  # the deadline passed: the body may have been cut short
            raise HttpFailure("timeout")
        if response.length:  # bytes still declared, but the connection ended
            raise HttpFailure("truncated response")
        return response.status, location, payload
    except HttpFailure:
        raise
    except TimeoutError:
        raise HttpFailure("timeout") from None
    except ssl.SSLError:
        if watchdog.fired or time.monotonic() >= deadline:
            raise HttpFailure("timeout") from None
        raise HttpFailure("tls error") from None
    except (OSError, http.client.HTTPException, ValueError):
        if watchdog.fired or time.monotonic() >= deadline:
            raise HttpFailure("timeout") from None
        raise HttpFailure("connection failed") from None
    finally:
        watchdog.cancel()
        connection.close()


def _remaining(deadline: float) -> float:
    """Returns the time left, or raises the timeout when there is none."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise HttpFailure("timeout")
    return remaining


def _connect(
    host: str,
    port: int,
    *,
    tls: ssl.SSLContext | None,
    deadline: float,
    watchdog: _Watchdog,
) -> socket.socket:
    """Resolves, connects and (for HTTPS) handshakes, all within the deadline."""
    error: OSError | None = None
    for family, kind, proto, _, address in _resolve(host, port, deadline):
        sock = socket.socket(family, kind, proto)
        try:
            watchdog.watch(sock)
            sock.settimeout(_remaining(deadline))
            sock.connect(address)
        except OSError as exc:
            sock.close()
            error = exc
            continue
        except BaseException:
            sock.close()
            raise
        break
    else:
        _remaining(deadline)  # a timeout if the addresses used up the time
        raise error if error is not None else HttpFailure("connection failed")
    try:
        with contextlib.suppress(OSError):  # as http.client does
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        if tls is not None:
            sock.settimeout(_remaining(deadline))  # caps the whole handshake
            sock = tls.wrap_socket(sock, server_hostname=host)
            watchdog.watch(sock)
        sock.settimeout(_remaining(deadline))
    except BaseException:
        sock.close()
        raise
    return sock


def _resolve(host: str, port: int, deadline: float) -> Sequence[_AddressInfo]:
    """Looks a host up in a daemon thread and waits for it until the deadline."""
    if not _LOOKUP_SLOTS.acquire(timeout=_remaining(deadline)):
        raise HttpFailure("timeout")
    outcome: list[Sequence[_AddressInfo] | Exception] = []
    done = threading.Event()

    def lookup() -> None:
        try:
            outcome.append(socket.getaddrinfo(host, port, type=socket.SOCK_STREAM))
        except Exception as exc:  # handed to the waiting request below
            outcome.append(exc)
        finally:
            _LOOKUP_SLOTS.release()
            done.set()

    try:
        threading.Thread(target=lookup, name="pk-dns", daemon=True).start()
    except BaseException:
        _LOOKUP_SLOTS.release()
        raise
    if not done.wait(_remaining(deadline)):
        raise HttpFailure("timeout")
    result = outcome[0]
    if isinstance(result, (OSError, UnicodeError)):  # unknown host, bad IDNA name
        raise HttpFailure("connection failed") from None
    if isinstance(result, Exception):
        raise result
    return result


class _Watchdog:
    """Shuts down every socket of one exchange when its deadline passes."""

    def __init__(self, deadline: float) -> None:
        self._lock = threading.Lock()
        self._sockets: list[socket.socket] = []
        self._fired = threading.Event()
        self._timer = threading.Timer(max(0.0, deadline - time.monotonic()), self._fire)
        self._timer.daemon = True

    @property
    def fired(self) -> bool:
        """Whether the deadline has passed."""
        return self._fired.is_set()

    def start(self) -> None:
        """Starts the timer."""
        self._timer.start()

    def cancel(self) -> None:
        """Stops the timer (the exchange is over)."""
        self._timer.cancel()

    def watch(self, sock: socket.socket) -> None:
        """Registers a socket; after the deadline, refuses it with a timeout."""
        with self._lock:
            if self._fired.is_set():
                raise HttpFailure("timeout")
            self._sockets.append(sock)

    def _fire(self) -> None:
        with self._lock:
            self._fired.set()
            sockets = list(self._sockets)
        for sock in sockets:
            # socket.socket.shutdown, not SSLSocket.shutdown: the TLS object
            # must stay intact while another thread may be reading through it.
            with contextlib.suppress(OSError):
                socket.socket.shutdown(sock, socket.SHUT_RDWR)


def _read_capped(response: http.client.HTTPResponse, deadline: float, cap: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        if time.monotonic() >= deadline:
            raise HttpFailure("timeout")
        chunk = response.read1(_CHUNK)
        if not chunk:
            return b"".join(chunks)
        total += len(chunk)
        if total > cap:
            raise HttpFailure("response too large")
        chunks.append(chunk)
