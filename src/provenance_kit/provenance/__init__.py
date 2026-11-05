"""Provenance records: C2PA manifests and OpenTimestamps proofs.

Planned subpackages (see ``docs/ARCHITECTURE.md`` for ownership):

* ``provenance_kit.provenance.c2pa``: sign manifests with a pluggable signer
  (the private key may live in a KMS or HSM) and read/validate them
  (extra ``c2pa``).
* ``provenance_kit.provenance.timestamps``: stamp SHA-256 digests on several
  OpenTimestamps calendars, upgrade pending proofs, inspect proofs
  (extra ``timestamps``).

This file stays import-light: it must not import the subpackages eagerly.
"""
