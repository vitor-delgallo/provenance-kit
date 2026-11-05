"""A minimal HTTP client for untrusted calendar servers (internal).

Only the standard library is used (``http.client``). What it enforces:

* ``https`` only, unless the caller explicitly allows ``http`` (tests);
* a deadline for the whole exchange (connect, headers and body). Socket
  timeouts bound each read; a watchdog shuts the socket down at the deadline,
  so a server that trickles bytes cannot hold the call longer. Once the
  watchdog has fired the exchange is a timeout, whatever the read returned:
  on Linux a read blocked on a socket that is shut down returns end of file
  instead of raising, which must not pass for a complete body;
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
from dataclasses import dataclass

__all__ = ["HttpFailure", "HttpResponse", "check_url", "request"]

_REDIRECTS = frozenset({301, 302, 303, 307, 308})
_MAX_REDIRECTS = 3
_CHUNK = 16 * 1024


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
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise HttpFailure("timeout")
    host = str(url.hostname)
    port = url.port
    connection: http.client.HTTPConnection
    if url.scheme == "https":
        context = ssl_context if ssl_context is not None else ssl.create_default_context()
        connection = http.client.HTTPSConnection(host, port, timeout=remaining, context=context)
    else:
        connection = http.client.HTTPConnection(host, port, timeout=remaining)
    # The watchdog covers the TLS handshake too: it shuts down whatever socket
    # the connection holds when the deadline passes.
    fired = threading.Event()
    watchdog = threading.Timer(max(0.0, remaining), _abort, (connection, fired))
    watchdog.daemon = True
    try:
        watchdog.start()
        connection.connect()
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
        if fired.is_set():  # the deadline passed: the body may have been cut short
            raise HttpFailure("timeout")
        if response.length:  # bytes still declared, but the connection ended
            raise HttpFailure("truncated response")
        return response.status, location, payload
    except HttpFailure:
        raise
    except TimeoutError:
        raise HttpFailure("timeout") from None
    except ssl.SSLError:
        raise HttpFailure("tls error") from None
    except (OSError, http.client.HTTPException, ValueError):
        if fired.is_set() or time.monotonic() >= deadline:
            raise HttpFailure("timeout") from None
        raise HttpFailure("connection failed") from None
    finally:
        watchdog.cancel()
        connection.close()


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


def _abort(connection: http.client.HTTPConnection, fired: threading.Event) -> None:
    """Shuts the socket down so that a blocked read returns (watchdog)."""
    fired.set()
    sock = connection.sock
    if sock is None:
        return
    # socket.socket.shutdown, not SSLSocket.shutdown: the TLS object must stay
    # intact while another thread may be reading through it.
    with contextlib.suppress(OSError):
        socket.socket.shutdown(sock, socket.SHUT_RDWR)
