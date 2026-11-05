# Provenance: C2PA manifests and OpenTimestamps proofs

Two modules under `provenance_kit.provenance`:

* [`c2pa`](#c2pa-manifests): sign C2PA manifests into media with a pluggable
  signer (the private key can stay in a KMS or HSM), and read and validate
  them. Extra `c2pa`.
* [`timestamps`](#opentimestamps-proofs): OpenTimestamps proofs for SHA-256
  digests: stamp, upgrade, inspect, verify. Extra `timestamps`.

Everything here was checked on 2026-11-05 against c2pa-python 0.38.0
(c2pa-rs 0.91.0) on Windows and python-opentimestamps 0.5.0, by reading their
source and by the tests in `tests/provenance/`. Sources are linked where a
decision depends on them. The `c2pa` extra resolves to a different 0.38
release per platform (0.38.0 on Windows, 0.38.1 on Linux, 0.38.2 on macOS;
see the table below); the Linux CI integration job runs the same tests
against 0.38.1, and 0.38.2 has not been run.

## C2PA manifests

### Versions

| Component | Version | Notes |
|---|---|---|
| c2pa-python | 0.38.0 (Windows); 0.38.1 in Linux CI | 0.38.1 has Linux and macOS wheels only, 0.38.2 macOS wheels only, neither an sdist ([THIRD_PARTY.md](THIRD_PARTY.md#c2pa-python)) |
| c2pa-rs (bundled native library) | 0.91.0 | writes claim version 2; `spec_versions.rs`: generator 2.2.0, validator 2.4.0 |
| C2PA specification cited | [2.3](https://spec.c2pa.org/specifications/specifications/2.3/specs/C2PA_Specification.html) (December 2025) | section numbers below refer to it |
| CAWG assertions | [Training and Data Mining 1.1](https://cawg.io/training-and-data-mining/1.1/), [Metadata 1.1](https://cawg.io/metadata/1.1/) | both DIF ratified, 16 May 2025 |

### Supported media types

`SUPPORTED_MEDIA_TYPES`: the containers that `provenance_kit.mediatype`
detects and that the library can both sign and read. A test checks the set
against `c2pa.Builder.get_supported_mime_types()` and
`c2pa.Reader.get_supported_mime_types()`, and the integration tests sign and
validate each one.

| Kind | MIME types |
|---|---|
| Images | `image/jpeg`, `image/png`, `image/gif`, `image/webp`, `image/tiff` (also DNG), `image/heic`, `image/heif`, `image/avif`, `image/jxl` |
| Video | `video/mp4`, `video/quicktime`, `video/x-msvideo` |
| Audio | `audio/wav`, `audio/mpeg`, `audio/flac`, `audio/mp4` |

The library supports more (PDF, SVG, office documents, EPUB); they are out of
scope for a media toolkit. It does not support WebM, Matroska, Ogg, AAC
(ADTS), AIFF or BMP. The type is always detected from the bytes; a declared
MIME type must agree with it (any ISO BMFF brand is accepted for another,
because c2pa-rs uses one handler for all of them). c2pa-rs embeds manifests in
FLAC behind an ID3v2 tag, so a signed FLAC starts with `ID3`;
`provenance_kit.mediatype` looks past the tag and still detects FLAC (so do
the audio modules, which decode signed FLAC as FLAC).

### Signing

```python
from provenance_kit.provenance.c2pa import (
    ActionType,
    DigitalSourceType,
    GeneratorInfo,
    ManifestAction,
    ManifestSpec,
    SoftBinding,
    SoftBindingAlgorithm,
    TrainingMiningPolicy,
    sign_asset,
)

spec = ManifestSpec(
    claim_generator=GeneratorInfo("my-pipeline", "1.4.0"),
    title="photo.jpg",
    actions=(
        ManifestAction(
            action=ActionType.CREATED,
            when=uploaded_at,
            digital_source_type=DigitalSourceType.DIGITAL_CAPTURE,
        ),
        ManifestAction(action=ActionType.WATERMARKED_BOUND, when=processed_at),
        ManifestAction(
            action=ActionType.PUBLISHED,
            when=published_at,
            software_agent=GeneratorInfo("my-pipeline", "1.4.0"),
        ),
    ),
    authors=("Jane Doe",),
    training_mining=TrainingMiningPolicy.not_allowed(),
    soft_binding=SoftBinding(algorithm=SoftBindingAlgorithm.TRUSTMARK_Q, value=wm_value),
)
signed = sign_asset(jpeg_bytes, "image/jpeg", spec, signer)  # tsa_url=None
```

What `sign_asset` does:

* checks the size against `MediaLimits.max_bytes` (asset and ingredients),
  detects the container from the bytes, and validates the specification
  (`ValueError` for the rules listed in `ManifestSpec`);
* builds the manifest definition and signs it through in-memory streams; the
  library parses the container but decodes no pixels or samples, because
  thumbnails are disabled;
* lets the library validate the new manifest (c2pa-rs `verify_after_sign`); a
  manifest that breaks a C2PA rule raises `SigningError` with the validation
  codes, for example `assertion.action.malformed`;
* lists each assertion it writes in the claim's `created_assertions` or
  `gathered_assertions`, as `ManifestSpec.placements` says (see "Created and
  gathered assertions" below).

`format` (`dc:format`) is taken from the asset's media type. Claim version 2
makes `dc:title` and `dc:format` optional and c2pa-rs does not store
`dc:format` in v2 claims, so `ManifestReport.format` is `None` for manifests
written by current libraries.

### The signer and signature encodings (decision 6)

`C2paSigner` is a Protocol with `algorithm`, `certificate_chain_pem`,
`signature_format` and `sign(data) -> bytes`. `sign` receives the COSE
`Sig_structure` to be signed (not a digest), and the adapter passes it to
c2pa-python's `Signer.from_callback`. `LocalKeySigner` implements the
Protocol with a PEM key in memory, for development and tests.

| Algorithm | COSE stores (RFC 9053, RFC 8230) | A signer may return | Adapter |
|---|---|---|---|
| ES256 / ES384 / ES512 | `r \|\| s`, 64 / 96 / 132 bytes | `SignatureFormat.DER` (what KMSs return) or `RAW` | converts DER with `ecdsa_der_to_raw`, then checks the length |
| ED25519 | 64 bytes (RFC 8032) | `RAW` only | checks the length |
| PS256 / PS384 / PS512 | the modulus-sized octet string; MGF1 with the same hash, salt length = hash length | `RAW` only | checks 256 to 1024 bytes |

**Finding (confirmed by `tests/provenance/test_c2pa_integration.py`).** The
signature that ends up in the `COSE_Sign1` is `r || s` for ECDSA: the tests
read it back from the manifest store. c2pa-rs 0.91 also accepts a DER
signature from the callback and normalizes it to `r || s` itself
(`test_library_callback_also_accepts_der_unconverted`). The adapter does not
rely on that leniency: signers state their format, DER is converted before
the library sees it, and a signer that declares `RAW` but returns DER is
refused (`SigningError`). Ed25519 and RSASSA-PSS signatures pass unchanged.

Other properties of the bridge:

* an exception raised by `sign` cannot cross the native frames, and
  c2pa-python's own wrapper logs the exception text (`logger.error`). The
  adapter catches everything first and raises `SigningError` after the native
  call returns, with only the exception type in the message (a KMS error can
  name keys or accounts); the original is chained as `__cause__`.
  `KeyboardInterrupt`, `SystemExit` and `MemoryError` are re-raised as is;
* c2pa-python silently truncates a signature longer than the space it
  reserved; the adapter checks lengths, so that cannot happen.

**Certificates.** `certificate_chain_pem` is the signing certificate followed
by intermediates, without the root. c2pa-rs checks most of the C2PA
certificate profile (section 14.5.1; c2pa-rs `crypto/cose/certificate_profile.rs`):
X.509 v3; ECDSA P-256/P-384/P-521, RSA of at least 2048 bits or Ed25519; a key
usage extension; an extended key usage it accepts (for example
`id-kp-emailProtection` 1.3.6.1.5.5.7.3.4 or `id-kp-documentSigning`
1.3.6.1.5.5.7.3.36) and not `anyExtendedKeyUsage`; not a CA; an authority key
identifier. A leaf without an accepted EKU, or without a key usage extension,
is refused at signing time.

The specification also says that claim-signing certificates "shall assert the
digitalSignature bit". c2pa-rs 0.91 does not enforce that bit. Measured on
2026-11-05 with a KMS-style signer before provenance-kit added its own check:
a leaf whose key usage asserts only `nonRepudiation` signs, and the library
reads the manifest back as `TRUSTED` when its root is a trust anchor (a
`keyAgreement`-only leaf and a leaf without the extension are refused).
provenance-kit closes the gap on its side: `LocalKeySigner` refuses such a
leaf when it is constructed, and `sign_asset` checks the leaf of every signer
(KMS signers included) once the library is loaded, before any signature is
requested; both raise `ValueError` naming `digitalSignature`. **On reading,
the gap remains**: the library's reports do not expose the signing
certificate (only its issuer, common name and serial number), so
`read_manifest` cannot re-check its key usage, and a manifest signed elsewhere
with such a leaf is reported `TRUSTED` if it chains to your anchors. Trust
lists such as the C2PA trust list only admit CAs that issue conforming
certificates, which bounds the exposure. The tests generate the PKI at test
time (`tests/provenance/_certs.py`).

**Time-stamp authority.** `tsa_url` (default `None`) names an RFC 3161
time-stamp authority that counter-signs the claim signature. Without it the
call is offline, and validators check the signing certificate against the
current time, so the manifest stops validating once the certificate expires.
With it, the library sends the TSA a hash of the signature and validators
check the certificate at the time-stamp's time. Only that host is reachable
during the call (`core.allowed_network_hosts`), with redirects disabled. Use
a TSA on the C2PA TSA trust list if consumers should trust the time. Plain
`http` is accepted because RFC 3161 responses are signed.

### Assertions written

| Spec field | Label (as stored) | Source and notes |
|---|---|---|
| actions | `c2pa.actions` in the definition, stored as `c2pa.actions.v2` | Section 18.15. The first action must be `c2pa.created` or `c2pa.opened` (section 15.10.3.2); `c2pa.created` needs a `digitalSourceType` (IPTC URIs, `DigitalSourceType`); `c2pa.opened`, `c2pa.placed` and `c2pa.removed` need ingredients (the module links `opened` to the parent). `softwareAgent` is a claim-generator-info map in v2; `when` is RFC 3339 (CBOR tag 0). |
| authors | `cawg.metadata`, JSON box, `{"@context": {"dc": "http://purl.org/dc/elements/1.1/"}, "dc:creator": [...]}` | [CAWG Metadata 1.1](https://cawg.io/metadata/1.1/): label `cawg.metadata`, a single JSON box, `@context` required, no field restrictions. `stds.schema-org.CreativeWork` "is no longer part of the C2PA Technical Specification" and is deprecated in c2pa-rs 0.91 (removal planned for 0.92), which points to the CAWG identity and metadata assertions instead. `dc:creator` is not in the `c2pa.metadata` allow-list (C2PA metadata annex; c2pa-rs `assertions/metadata.rs`). c2pa-rs validates that every prefix is declared in `@context`. |
| training_mining | `cawg.training-mining`, `{"entries": {"cawg.ai_training": {"use": "notAllowed"}, ...}}` | [CAWG Training and Data Mining 1.1](https://cawg.io/training-and-data-mining/1.1/): "SHALL have a label of `cawg.training-mining`"; standard keys `cawg.data_mining`, `cawg.ai_inference`, `cawg.ai_training`, `cawg.ai_generative_training`; uses `allowed`, `notAllowed`, `constrained` (+ `constraint_info`). The `c2pa.training-mining` assertion of C2PA 1.4 was removed in C2PA 2.0, and c2pa-rs 0.91 has no label for it (`assertions/labels.rs`): the assertion is stored as given and not validated by the library. |
| soft_binding | `c2pa.soft-binding`, `{"alg": ..., "blocks": [{"scope": {...}, "value": ...}]}` | Section 18.10. `alg` comes from the [soft binding algorithm list](https://github.com/c2pa-org/softbinding-algorithm-list): TrustMark is `com.adobe.trustmark.Q` (list identifier 4), `.C` (5), `.P` (13) and `.F` (55); the list has no entry for TrustMark variant B. c2pa-rs 0.91 checks the structure only ("Starting with C2PA 2.4 we no longer require soft binding alg to be on the official list", `claim.rs`). A `c2pa.watermarked.bound` action requires a soft binding (section 18.15.5; c2pa-rs enforces it). |
| ingredients | `c2pa.ingredient.v3` | Section 18.16. The ingredient's own manifest, if any, is embedded and validated. |

**Soft binding value: a known deviation.** The schema says `"value": bstr`
(section 18.10.2). c2pa-python 0.38 builds manifests from a JSON definition,
and c2pa-rs converts the JSON of every assertion it has no typed support for
into CBOR without byte strings (`builder.rs`:
`c2pa_cbor::value::to_value(serde_json::Value)`; there is no typed path for
`c2pa.soft-binding`). The value is therefore written as a CBOR **text**
string (checked by `test_soft_binding_value_is_written_as_text`). This is what
Adobe's own TrustMark example produces too: it writes
`"<encoding>*<payload bits>"` as text through `c2patool`
([`c2pa_watermark_example.py`](https://github.com/adobe/trustmark/blob/main/c2pa/c2pa_watermark_example.py)).
c2pa-rs reads it (its `SoftBinding` decoder accepts text through
`serde_bytes`) and validation passes, but a strict validator could reject
it. `SoftBinding.value` is a `str` for that reason. On reading, the library's
JSON report renders byte strings as base64 text, so a text value and a
byte-string value cannot be told apart in `SoftBindingReport.values`.

**Not covered.** CAWG identity assertions (`cawg.identity`, signed by the
named actor), thumbnails, redactions, update manifests, remote or sidecar
publication when signing (reading sidecars is supported).

### Created and gathered assertions

A version 2 claim lists its assertions in two places (C2PA 2.3, section
10.2.2, "Fields"):

* `created_assertions`: assertions made by the claim generator. "All
  created_assertions are attributed to the signer as the Trust Model is
  rooted in the trust of the signer." The field is mandatory and, in a
  standard manifest, includes a hard binding (section 10.3.2.1); c2pa-rs adds
  the hard binding (`c2pa.hash.data`, `c2pa.hash.bmff.v3`, ...) itself, always
  as created.
* `gathered_assertions`: assertions "provided to the claim generator by other
  components in the workflow". "By putting an assertion into this list, the
  claim generator is declaring that the assertion is part of the claim, but
  it was not sourced from the claim generator and is not attributed to the
  signer." The specification's example: "assertions containing information
  entered by a human actor would be listed in gathered_assertions."

Both lists are covered by the signature: a gathered assertion is tamper-evident
like a created one; the difference is who vouches for its content.
`ManifestSpec.placements` (`AssertionPlacements`) sets the list for each
assertion the module writes. The defaults:

| Assertion | Default | Why |
|---|---|---|
| actions (`c2pa.actions`) | `CREATED` | what the signing application did |
| training and data mining (`cawg.training-mining`) | `CREATED` | the signer's policy decision |
| soft binding (`c2pa.soft-binding`) | `CREATED` | the watermark the signing application embedded |
| authors (`cawg.metadata`) | `GATHERED` | names supplied by people, not verified by the signer |

```python
from provenance_kit.provenance.c2pa import AssertionPlacement, AssertionPlacements

spec = ManifestSpec(
    ...,
    authors=("Jane Doe",),
    placements=AssertionPlacements(authors=AssertionPlacement.CREATED),  # verified authorship
)
```

c2pa-python 0.38 supports both lists: an assertion defined with
`"created": false` is written to `gathered_assertions` (c2pa-rs 0.91). The
integration tests (`test_created_and_gathered_assertions`) sign with the
defaults and with every placement inverted, decode the claim to check both
lists, and validate the result (state `VALID`). `ManifestReport` does not
say which list an assertion came from.

### Reading and validation

```python
report = read_manifest(upload_bytes, trust_anchors_pem=c2pa_trust_list_pem)
report.state  # ValidationState
report.errors  # failure codes of the active manifest
report.training_mining_not_allowed  # no standard use permitted (constrained counts as notAllowed)
```

**Training and data mining policy.** CAWG TDM 1.1: "In the absence of
additional information, `constrained` shall be treated as equivalent to
`notAllowed`." `training_mining_not_allowed` and
`TrainingMiningPolicy.all_not_allowed` follow that rule: they are true when
AI training, generative AI training, data mining and AI inference are each
`notAllowed` or `constrained`, and false when a standard use is `allowed`,
undeclared, or there is no policy. For one use, `effective_use_of(key)` returns
`ALLOWED`, `NOT_ALLOWED` (also for `constrained`) or `None` (undeclared).
The literal views remain: `use_of(key)` returns the declared value and
`all_declared_not_allowed` counts only `notAllowed`; the free-text
`constraint_info` is in `training_mining.entries`.

| `ValidationState` | Meaning |
|---|---|
| `UNSIGNED` | no manifest (also an asset that only references a remote manifest; see below) |
| `INVALID` | a manifest is present but fails validation (section 14.3), or the store cannot be parsed at all (`errors` holds `MANIFEST_UNREADABLE`) |
| `VALID` | well-formed, hashes match, signature verifies. The signer is not known to be trusted: no anchors were given, or the chain does not lead to one (`errors` then holds `signingCredential.untrusted`) |
| `TRUSTED` | valid, and the signing certificate chains to `trust_anchors_pem` |

The asset is untrusted. Only documented errors escape: `MediaLimitExceededError`,
`UnsupportedMediaError` (unrecognized, unsupported or contradicting the
declared type), `MediaDecodeError` (the container cannot be parsed) and
`BackendUnavailableError`. Every assertion, string and URL in the report is a
claim of the manifest, meaningful only when the state is valid. Tampering
with the media payload gives `assertion.dataHash.mismatch`; re-encoding
(which strips the manifest) gives `UNSIGNED`.

Settings applied to every library call (c2pa-rs 0.91 names, `_native.py`):

| Setting | Value | Why |
|---|---|---|
| `verify.remote_manifest_fetch` | `false` | the default `true` makes the library fetch a manifest URL found **inside** the untrusted asset (server-side request forgery); a test with a local canary server confirms no request is made |
| `core.allowed_network_hosts` | `[]` (only the TSA host when signing with one) | an empty list blocks every request of the library, including OCSP and `did:web` resolution for CAWG identity assertions |
| `core.allow_redirects` | `false` | |
| `verify.ocsp_fetch` | `false` | |
| `verify.verify_trust` | only with `trust_anchors_pem` | trust anchors use `trust.anchors` with `trust_kind: "manifest"` (`"tsa"` for `tsa_trust_anchors_pem`) |
| `builder.thumbnail.enabled` | `false` | no pixel decoding inside the library |

When an asset only references a remote manifest, the report is `UNSIGNED`
with `remote_manifest_url` set (untrusted). Fetch it yourself, with your own
SSRF protections and size limits, and pass the bytes as `manifest_data`; the
same argument validates a `.c2pa` sidecar. The library also caps a
decompressed manifest at 32 MiB (`core.max_decompressed_manifest_size_in_mb`).

For trust, use the [C2PA trust list](https://github.com/c2pa-org/conformance-public/tree/main/trust-list)
or your own roots. Legacy assertions are still understood when reading:
`c2pa.training-mining` (C2PA 1.x, `c2pa.` keys) and the authors of
`stds.schema-org.CreativeWork`.

## OpenTimestamps proofs

### Defaults (decision 7)

| Setting | Default | Source |
|---|---|---|
| Calendars | `https://a.pool.opentimestamps.org`, `https://b.pool.opentimestamps.org`, `https://a.pool.eternitywall.com`, `https://ots.btc.catallaxy.com` | the official client adds exactly these when no calendar is given ([`otsclient/cmds.py`](https://github.com/opentimestamps/opentimestamps-client/blob/master/otsclient/cmds.py), `stamp_command`); identical to `DEFAULT_AGGREGATORS` in python-opentimestamps 0.5.0 `opentimestamps/calendar.py` |
| Upgrade allow-list | `https://*.calendar.opentimestamps.org`, `https://*.calendar.eternitywall.com`, `https://*.calendar.catallaxy.com` | `DEFAULT_CALENDAR_WHITELIST` in python-opentimestamps 0.5.0, which the official client uses for upgrades ([`otsclient/args.py`](https://github.com/opentimestamps/opentimestamps-client/blob/master/otsclient/args.py)) |
| Timeout | 5 s per operation (calendars in parallel) | the client's `--timeout` default, 5 |
| `min_calendars` | **2** (`DEFAULT_MIN_CALENDARS`) | the client's `-m` default ([`otsclient/args.py`](https://github.com/opentimestamps/opentimestamps-client/blob/master/otsclient/args.py)): the proof survives the loss of one calendar. Like the client (`create_timestamp` in `cmds.py`), more than the number of calendars is refused, so a single custom calendar needs `min_calendars=1` |
| Response size | 10,000 bytes | python-opentimestamps `MAX_RESPONSE_LENGTH` |
| Proof size | 64 KiB | this library (see below) |

On 2026-11-05 the four aggregators answered with pending attestations for
`alice.btc.calendar.opentimestamps.org`, `bob.btc.calendar.opentimestamps.org`,
`finney.calendar.eternitywall.com` and `btc.calendar.catallaxy.com`, all
covered by the default allow-list (the `network` test prints them).

### Usage

```python
from provenance_kit.provenance.timestamps import OpenTimestamper, TimestampProof, sha256_file

ots = OpenTimestamper()  # the official client's calendars; 2 must answer
digest = sha256_file("master.mov")  # store it with the proof: verify needs it
proof = ots.stamp(digest)  # pending; proof.failed_calendars lists failures
data = proof.to_bytes()  # standard .ots, readable by `ots info`

proof = ots.upgrade(TimestampProof.from_bytes(data))  # hours later
info = ots.info(proof)  # pending calendars, Bitcoin heights
result = ots.verify(proof, digest, block_source)  # offline, against your block headers
```

`block_source` is your `BitcoinBlockSource` (see "Verifying against
Bitcoin" below).

Like the official client, `stamp` appends a random 16-byte nonce to the
digest and hashes again: calendars only ever see that commitment.
`upgrade` merges what calendars return and never raises for calendar
failures; a calendar that has not committed yet answers 404 and the
attestation stays pending. `info` and `verify` never use the network.

### Untrusted calendars and proofs

The HTTP client is provenance-kit's own (standard library only), not
python-opentimestamps' `RemoteCalendar`, so that every limit is enforced:

* `https` only (`allow_insecure_http=True` is for tests); an optional
  `ssl_context` (for a private CA), otherwise the system trust store;
* one deadline for the whole exchange: socket timeouts bound each read and a
  watchdog shuts the socket down at the deadline, so a server that trickles
  headers or body cannot hold the call (tested on Windows and Linux: there a
  read blocked on a shut-down socket returns end of file instead of raising,
  and an exchange the watchdog cut is still a `timeout`, never a short body);
* a body shorter than its `Content-Length` is a `truncated response` failure;
* response size cap, checked against `Content-Length` and while reading
  (chunked responses included);
* redirects only within the same scheme, host and port, at most 3; proxy
  environment variables are ignored;
* responses are parsed with python-opentimestamps' parser
  (`Timestamp.deserialize` with `assert_eof`); any parser error becomes a
  `malformed response` failure;
* `upgrade` contacts a pending attestation's URI only if it matches the
  allow-list (or is one of the configured calendars): the URI comes from the
  proof, which may come from an attacker (a test proves a canary server is
  never contacted). At most 32 requests per call.

Proofs are untrusted input as well. `TimestampProof.from_bytes` caps the size
before parsing (`max_proof_bytes=`, the same keyword and default as
`OpenTimestamper`'s; 64 KiB: complete proofs from the public calendars are 1 to
10 KiB, and an operation result can be up to 4 KiB, so fan-out in a hostile
proof multiplies memory by a few hundred at most), relies on the parser's
recursion limit (256) and raises `InvalidProofError` (a `MediaDecodeError`)
or `UnsupportedMediaError` (not SHA-256) only. `verify` checks at most 64
Bitcoin attestations.

### Verifying against Bitcoin

`verify(proof, digest, block_source)` checks that the proof is for `digest`
(the parser recomputes every operation from it) and then, for each Bitcoin
attestation, asks `block_source.block_header(height)` for the header and
compares its merkle root with the attestation's message. The result's
`attested_time` is the earliest verified block time; Bitcoin header times can
be off by about two hours.

provenance-kit ships no `BitcoinBlockSource`, so the default never touches the
network. `BitcoinBlockHeader.merkle_root` is in the header's internal byte
order; tools display it reversed, hence `BitcoinBlockHeader.from_display_hex`.

A source backed by your own node (`bitcoind` JSON-RPC):

```python
import base64, json, urllib.request
from provenance_kit.provenance.timestamps import BitcoinBlockHeader


class NodeBlockSource:
    def __init__(self, url, user, password):
        self._url = url
        self._auth = "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()

    def _call(self, method, *params):
        body = json.dumps({"jsonrpc": "1.0", "id": 0, "method": method, "params": params})
        request = urllib.request.Request(self._url, body.encode(), {"Authorization": self._auth})
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.load(response)["result"]

    def block_header(self, height):
        try:
            block_hash = self._call("getblockhash", height)
        except Exception:
            return None
        header = self._call("getblockheader", block_hash)
        return BitcoinBlockHeader.from_display_hex(header["merkleroot"], header["time"])
```

With a block explorer that offers an Esplora-style API, `GET
/api/block-height/<height>` returns the block hash and `GET /api/block/<hash>`
returns `merkle_root` (display hex) and `timestamp`. An explorer can lie; a
node you run cannot, and a header store you filled from your node is the
cheapest option for repeated verification.

### Hashing

`sha256_file(path)` and `sha256_stream(stream)` hash in 1 MiB chunks (the
chunk size python-opentimestamps uses), so any size works in constant memory.

### Test fixtures

`tests/provenance/fixtures/pending.ots` and `complete.ots` are generated by
`tests/provenance/_ots_fixtures.py` (deterministic, MIT like the rest of the
repository), and a test checks that the generator still produces the
committed bytes. They have the structure of real proofs: nonce, calendar
operations, a Bitcoin transaction with an `OP_RETURN` commitment, the merkle
path and a block-header attestation; the block is synthetic (python-bitcoinlib),
so they verify against the header the generator defines, not the real chain.
The official client's `examples/` proofs were not copied: they live in an
LGPL-3.0 repository without a separate data license.

The integration tests run a local calendar (`tests/provenance/_calendar.py`)
that serves real serialized timestamps built with python-opentimestamps and,
on demand, hostile responses: garbage, trailing bytes, oversized (with and
without `Content-Length`, chunked), slow headers and body, HTTP errors,
cross-origin, scheme-changing and looping redirects. One opt-in test
(`--run-network`) stamps against the public calendars; another signs a C2PA
manifest with a public TSA.
