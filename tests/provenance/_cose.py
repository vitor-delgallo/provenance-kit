"""Just enough JUMBF and CBOR to read the claim signature of a signed PNG (tests only).

Used to confirm which signature encoding ends up in the COSE_Sign1 structure
(ARCHITECTURE.md decision 6). PNG keeps the C2PA manifest store in a ``caBX``
chunk; the store is a JUMBF superbox whose ``c2pa.signature`` box holds a
CBOR ``COSE_Sign1`` (tag 18): ``[protected, unprotected, payload, signature]``.
"""

from __future__ import annotations

from collections.abc import Iterator


def png_manifest_store(png: bytes) -> bytes:
    """Returns the JUMBF manifest store of a signed PNG."""
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    pos = 8
    while pos + 8 <= len(png):
        length = int.from_bytes(png[pos : pos + 4], "big")
        kind = png[pos + 4 : pos + 8]
        if kind == b"caBX":
            return png[pos + 8 : pos + 8 + length]
        pos += 12 + length
    raise AssertionError("no caBX chunk")


def _boxes(data: bytes) -> Iterator[tuple[bytes, bytes]]:
    pos = 0
    while pos + 8 <= len(data):
        size = int.from_bytes(data[pos : pos + 4], "big")
        kind = data[pos + 4 : pos + 8]
        header = 8
        if size == 1:
            size = int.from_bytes(data[pos + 8 : pos + 16], "big")
            header = 16
        elif size == 0:
            size = len(data) - pos
        yield kind, data[pos + header : pos + size]
        pos += size


def jumbf_cbor_boxes(store: bytes) -> dict[str, bytes]:
    """Maps each JUMBF label to the content of its CBOR box."""
    found: dict[str, bytes] = {}

    def walk(data: bytes) -> None:
        label = None
        for kind, payload in _boxes(data):
            if kind == b"jumd":
                label = payload[17:].split(b"\x00", 1)[0].decode("utf-8", "replace")
            elif kind == b"jumb":
                walk(payload)
            elif kind == b"cbor" and label is not None:
                found.setdefault(label, payload)

    walk(store)
    return found


def cbor_decode(data: bytes, pos: int = 0) -> tuple[object, int]:  # noqa: PLR0911
    """Decodes one CBOR item (definite lengths only)."""
    initial = data[pos]
    major, info = initial >> 5, initial & 31
    pos += 1
    if info < 24:
        value = info
    elif info in (24, 25, 26, 27):
        size = 1 << (info - 24)
        value = int.from_bytes(data[pos : pos + size], "big")
        pos += size
    else:
        raise ValueError("indefinite lengths are not supported")
    if major == 0:
        return value, pos
    if major == 1:
        return -1 - value, pos
    if major in (2, 3):
        chunk = data[pos : pos + value]
        return (bytes(chunk) if major == 2 else chunk.decode()), pos + value
    if major == 4:
        items = []
        for _ in range(value):
            item, pos = cbor_decode(data, pos)
            items.append(item)
        return items, pos
    if major == 5:
        mapping = {}
        for _ in range(value):
            key, pos = cbor_decode(data, pos)
            item, pos = cbor_decode(data, pos)
            mapping[key] = item
        return mapping, pos
    if major == 6:
        item, pos = cbor_decode(data, pos)
        return ("tag", value, item), pos
    return {20: False, 21: True, 22: None}.get(value, ("simple", value)), pos


def claim_signature(png: bytes) -> bytes:
    """Returns the signature bytes stored in the active claim's COSE_Sign1."""
    boxes = jumbf_cbor_boxes(png_manifest_store(png))
    cose, _ = cbor_decode(boxes["c2pa.signature"])
    assert isinstance(cose, tuple)
    assert cose[1] == 18  # COSE_Sign1 tag
    signature = cose[2][3]
    assert isinstance(signature, bytes)
    return signature


def assertion_cbor(png: bytes, label: str) -> object:
    """Decodes a CBOR assertion of a signed PNG by its JUMBF label."""
    value, _ = cbor_decode(jumbf_cbor_boxes(png_manifest_store(png))[label])
    return value
