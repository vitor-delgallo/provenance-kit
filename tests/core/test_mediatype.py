from __future__ import annotations

import io
import random
import struct

import pytest
from PIL import Image

from provenance_kit.errors import UnsupportedMediaError
from provenance_kit.mediatype import (
    SNIFF_BYTES,
    MediaKind,
    MediaType,
    require_media_type,
    sniff_media_type,
)
from tests.support import SEED


def ftyp(major: bytes, *compatible: bytes) -> bytes:
    body = major + b"\x00\x00\x00\x00" + b"".join(compatible)
    return struct.pack(">I", 8 + len(body)) + b"ftyp" + body + b"\x00" * 32


def ebml(doctype: bytes) -> bytes:
    return b"\x1a\x45\xdf\xa3\x9f\x42\x86\x81\x01\x42\x82\x84" + doctype + b"\x00" * 32


SAMPLES = {
    MediaType.JPEG: b"\xff\xd8\xff\xe0\x00\x10JFIF\x00",
    MediaType.PNG: b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR",
    MediaType.GIF: b"GIF89a\x01\x00\x01\x00",
    MediaType.WEBP: b"RIFF\x24\x00\x00\x00WEBPVP8 ",
    MediaType.TIFF: b"II*\x00\x08\x00\x00\x00",
    MediaType.BMP: b"BM" + b"\x00" * 8 + b"\x36\x00\x00\x00" + b"\x28\x00\x00\x00",
    MediaType.HEIC: ftyp(b"heic", b"mif1", b"heic"),
    MediaType.HEIF: ftyp(b"mif1", b"mif1", b"miaf"),
    MediaType.AVIF: ftyp(b"avif", b"mif1", b"miaf", b"MA1B"),
    MediaType.JXL: b"\x00\x00\x00\x0cJXL \r\n\x87\n\x00\x00",
    MediaType.MP4: ftyp(b"isom", b"isom", b"iso2", b"avc1", b"mp41"),
    MediaType.QUICKTIME: ftyp(b"qt  ", b"qt  "),
    MediaType.M4A: ftyp(b"M4A ", b"M4A ", b"mp42", b"isom"),
    MediaType.WEBM: ebml(b"webm"),
    MediaType.MATROSKA: ebml(b"matroska"),
    MediaType.AVI: b"RIFF\x00\x00\x00\x00AVI LIST",
    MediaType.MPEG_TS: (b"\x47" + b"\x00" * 187) * 3,
    MediaType.WAV: b"RIFF\x24\x08\x00\x00WAVEfmt ",
    MediaType.MP3: b"ID3\x04\x00\x00\x00\x00\x00\x00",
    MediaType.AAC: b"\xff\xf1\x50\x80\x02\x1f\xfc",
    MediaType.FLAC: b"fLaC\x00\x00\x00\x22",
    MediaType.OGG: b"OggS\x00\x02\x00\x00",
    MediaType.AIFF: b"FORM\x00\x00\x00\x00AIFFCOMM",
}


def test_every_media_type_has_a_sample():
    assert set(SAMPLES) == set(MediaType)


@pytest.mark.parametrize(("expected", "head"), list(SAMPLES.items()))
def test_sniff_recognizes_each_container(expected, head):
    assert sniff_media_type(head) is expected


def test_mp3_frame_sync_without_id3():
    assert sniff_media_type(b"\xff\xfb\x90\x64\x00\x00") is MediaType.MP3


def id3_tag(payload_size: int, *, footer: bool = False) -> bytes:
    """An ID3v2.4 tag header (synchsafe size) followed by a zero payload."""
    synchsafe = bytes((payload_size >> shift) & 0x7F for shift in (21, 14, 7, 0))
    flags = 0x10 if footer else 0x00
    tag = b"ID3\x04\x00" + bytes([flags]) + synchsafe + b"\x00" * payload_size
    return tag + (b"3DI\x04\x00" + bytes([flags]) + synchsafe if footer else b"")


def test_flac_behind_an_id3v2_tag_is_flac():
    # c2pa-rs embeds C2PA manifests in FLAC behind an ID3v2 tag.
    assert sniff_media_type(id3_tag(20) + b"fLaC" + b"\x00" * 40) is MediaType.FLAC
    assert sniff_media_type(id3_tag(20, footer=True) + b"fLaC\x00\x00") is MediaType.FLAC
    # A manifest-sized tag: the marker lies far beyond SNIFF_BYTES.
    big = id3_tag(200_000) + b"fLaC" + b"\x00" * 40
    assert sniff_media_type(big) is MediaType.FLAC
    assert require_media_type(memoryview(big), allowed={MediaType.FLAC}) is MediaType.FLAC
    # Only the head was passed: the marker is unknown, so the tag says MP3.
    assert sniff_media_type(big[:SNIFF_BYTES]) is MediaType.MP3


def test_mp3_behind_an_id3v2_tag_stays_mp3():
    assert sniff_media_type(id3_tag(20) + b"\xff\xfb\x90\x00" + b"\x00" * 40) is MediaType.MP3
    assert sniff_media_type(id3_tag(20)) is MediaType.MP3  # nothing after the tag
    # A size that is not synchsafe is not trusted.
    broken = b"ID3\x04\x00\x00\x00\x00\x00\x94" + b"\x00" * 20 + b"fLaC"
    assert sniff_media_type(broken) is MediaType.MP3


def test_rf64_and_aifc():
    assert sniff_media_type(b"RF64\xff\xff\xff\xffWAVEds64") is MediaType.WAV
    assert sniff_media_type(b"FORM\x00\x00\x00\x00AIFCFVER") is MediaType.AIFF


@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "GIF", "WEBP", "TIFF", "BMP"])
def test_real_images_from_pillow(fmt):
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (200, 10, 10)).save(buffer, format=fmt)
    detected = sniff_media_type(buffer.getvalue())
    assert detected is not None
    assert detected.name == fmt
    assert detected.kind is MediaKind.IMAGE


def test_kind_and_mime():
    assert MediaType.MP4.kind is MediaKind.VIDEO
    assert MediaType.M4A.kind is MediaKind.AUDIO
    assert MediaType.PNG.mime == "image/png"


@pytest.mark.parametrize(
    "head",
    [
        b"",
        b"\x00",
        b"hello world, this is plain text",
        b"%PDF-1.7",
        ftyp(b"crx ", b"crx "),
        b"\x00\x00\x00\x08ftyp",  # box too small
        b"\x1a\x45\xdf\xa3\x00\x00",  # EBML without a doc type
        b"RIFF\x00\x00\x00\x00XXXX",
        b"BM\x00\x00",
        b"\x47" + b"\x00" * 200,  # one TS packet only
    ],
)
def test_unknown_or_unsupported_content_returns_none(head):
    assert sniff_media_type(head) is None


def test_sniff_never_raises_for_random_bytes():
    rand = random.Random(SEED)
    for _ in range(3000):
        size = rand.choice([0, 1, 2, 3, 4, 8, 12, 16, 64, 400, 5000])
        sniff_media_type(bytes(rand.getrandbits(8) for _ in range(size)))


def test_sniff_accepts_bytearray_and_memoryview():
    head = SAMPLES[MediaType.PNG]
    assert sniff_media_type(bytearray(head)) is MediaType.PNG
    assert sniff_media_type(memoryview(head)) is MediaType.PNG


def test_sniff_refuses_non_bytes():
    with pytest.raises(TypeError):
        sniff_media_type("GIF89a")  # type: ignore[arg-type]


def test_require_media_type():
    assert require_media_type(SAMPLES[MediaType.PNG], kinds={MediaKind.IMAGE}) is MediaType.PNG
    with pytest.raises(UnsupportedMediaError, match="unrecognized"):
        require_media_type(b"plain text")
    with pytest.raises(UnsupportedMediaError, match="image/png is not accepted"):
        require_media_type(SAMPLES[MediaType.PNG], allowed={MediaType.JPEG})
    with pytest.raises(UnsupportedMediaError, match="video media"):
        require_media_type(SAMPLES[MediaType.MP4], kinds={MediaKind.AUDIO})
