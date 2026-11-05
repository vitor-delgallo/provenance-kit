# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Until 1.0, minor versions may change the API.

## [Unreleased]

### Added

- Project foundation: `src/` layout, hatchling build, optional-dependency
  extras per capability, ruff, mypy `--strict`, pytest markers
  (`integration`, `network`) and CI on Linux and Windows with Python 3.11
  and 3.13. Requires Python 3.11 or later.
- `provenance_kit.errors`: exception hierarchy (`ProvenanceKitError`,
  `BackendUnavailableError`, `MediaError` and subclasses,
  `ExternalToolError`, `ExternalToolTimeoutError`, `ModelLoadError`,
  `PayloadError`).
- `provenance_kit.limits.MediaLimits`: validated resource limits for untrusted
  media, with documented defaults.
- `provenance_kit.tools`: `ExternalTools` binary lookup and a shell-free
  runner (`run_tool`, `stream_tool`) with pipes, timeouts, process-tree kill
  and output caps.
- `provenance_kit.tempfiles`: private temporary files and directories that are
  always deleted.
- `provenance_kit.payload`: `PayloadCodec` packing a format version, an
  identifier and optional check bits into a watermark payload.
- `provenance_kit.mediatype`: media type detection from leading bytes.
- `provenance_kit.bits`, `provenance_kit.matching`,
  `provenance_kit.detection`, `provenance_kit.device`: shared bit helpers,
  Hamming matching, the common watermark detection result and device
  selection.
- Documentation: architecture contract, development guide, third-party
  license review, security policy.
