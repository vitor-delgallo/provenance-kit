"""Exception hierarchy shared by every provenance-kit module.

The rules (see ``docs/ARCHITECTURE.md``, "Error handling"):

* Caller mistakes (a wrong type, an out-of-range argument, an inconsistent
  configuration) raise the built-in ``TypeError`` or ``ValueError``. They are
  programming errors and are not part of this hierarchy.
* Problems caused by the environment or by the media itself raise a subclass of
  :class:`ProvenanceKitError`. Code that processes untrusted media can catch
  :class:`ProvenanceKitError` and be sure that nothing else escapes for a bad
  input.
* Detection functions do not raise when nothing is found: they return a result
  that says so.

Messages are safe to log. They never contain media bytes, watermark payloads,
keys, full subprocess output or the caller's file paths. Structured attributes
carry only small, non-sensitive values (names, counts, limits, exit codes),
except where an attribute is documented otherwise.

Modules may define more specific exceptions in their own namespace, but each
must subclass the closest class defined here.
"""

from __future__ import annotations

__all__ = [
    "BackendUnavailableError",
    "ExternalToolError",
    "ExternalToolTimeoutError",
    "MediaDecodeError",
    "MediaError",
    "MediaLimitExceededError",
    "ModelLoadError",
    "PayloadError",
    "ProvenanceKitError",
    "UnsupportedMediaError",
    "WatermarkBackendError",
]


class ProvenanceKitError(Exception):
    """Base class of every exception raised on purpose by provenance-kit."""


class BackendUnavailableError(ProvenanceKitError):
    """A backend cannot run because an optional dependency or binary is missing.

    The message names the extra to install (``pip install
    "provenance-kit[<extra>]"``) or the binary to configure. Use the
    :meth:`missing_extra` and :meth:`missing_binary` constructors so that the
    wording stays consistent across modules.

    Attributes:
        extra: The optional-dependency extra that provides the backend, if any.
        binary: The external binary that is missing, if any.
    """

    def __init__(
        self,
        message: str,
        *,
        extra: str | None = None,
        binary: str | None = None,
    ) -> None:
        """Initializes the error.

        Args:
            message: A human-readable explanation, safe to log.
            extra: The extra that provides the missing dependency.
            binary: The name of the missing external binary.
        """
        super().__init__(message)
        self.extra = extra
        self.binary = binary

    @classmethod
    def missing_extra(
        cls,
        feature: str,
        extra: str,
        *,
        reason: str | None = None,
    ) -> BackendUnavailableError:
        """Builds the error for a missing optional Python dependency.

        Args:
            feature: What the caller tried to use, for example
                ``"TrustMark watermarking"``.
            extra: The extra that installs the dependency, for example
                ``"image-watermark"``.
            reason: An optional short detail, for example the module that failed
                to import. Must not contain paths or secrets.

        Returns:
            The error, ready to raise.
        """
        message = f'{feature} requires the "{extra}" extra: pip install "provenance-kit[{extra}]"'
        if reason:
            message = f"{message} ({reason})"
        return cls(message, extra=extra)

    @classmethod
    def missing_binary(
        cls,
        binary: str,
        *,
        feature: str | None = None,
        hint: str | None = None,
    ) -> BackendUnavailableError:
        """Builds the error for a missing external binary.

        Args:
            binary: The binary name, for example ``"ffmpeg"``.
            feature: What the caller tried to use, if it helps the reader.
            hint: How to configure the binary. Defaults to a pointer at
                :class:`provenance_kit.tools.ExternalTools`.

        Returns:
            The error, ready to raise.
        """
        subject = f"{feature} requires the {binary} binary" if feature else f"{binary} binary"
        if hint is None:
            hint = (
                f"install it on PATH, pass ExternalTools({binary}=...), or set "
                f"PROVENANCE_KIT_{binary.upper()}"
            )
        return cls(f"{subject} was not found: {hint}", binary=binary)


class MediaError(ProvenanceKitError):
    """The media cannot be processed. Base class of the media errors."""


class MediaDecodeError(MediaError):
    """The media is corrupt, truncated or cannot be decoded."""


class UnsupportedMediaError(MediaError):
    """The media type, codec or layout is valid but not supported here.

    Raised, for example, when the bytes are a video but an image was expected,
    or when the detected container is not one the backend accepts.
    """


class MediaLimitExceededError(MediaError):
    """The media exceeds a configured limit.

    Usually a :class:`~provenance_kit.limits.MediaLimits` field; some
    components have their own size settings (for example the proof size cap
    of the timestamps module) and report them the same way. Raised before
    full decoding whenever the size can be known up front, so that
    decompression bombs are refused cheaply.

    Attributes:
        limit: The name of the exceeded setting: a ``MediaLimits`` field such
            as ``"max_pixels"``, or a component setting or argument such as
            ``"max_proof_bytes"`` (named in the raising function's
            documentation).
        maximum: The configured limit.
        actual: The measured or declared value that exceeded it, if known.
    """

    def __init__(
        self,
        limit: str,
        maximum: float,
        actual: float | None = None,
    ) -> None:
        """Initializes the error.

        Args:
            limit: The name of the exceeded setting (a ``MediaLimits`` field
                or a component setting such as ``"max_proof_bytes"``).
            maximum: The configured limit.
            actual: The measured or declared value, if known.
        """
        if actual is None:
            message = f"media exceeds {limit}={maximum:g}"
        else:
            message = f"media exceeds {limit}: {actual:g} > {maximum:g}"
        super().__init__(message)
        self.limit = limit
        self.maximum = maximum
        self.actual = actual


class ExternalToolError(ProvenanceKitError):
    """An external binary failed, or its output could not be used.

    The message is sanitized: it names the tool and the failure (exit code,
    signal, output limit) but never includes media content or the tool's
    standard error. A truncated standard error is logged at DEBUG level only.

    Attributes:
        tool: The tool name, for example ``"ffmpeg"``.
        returncode: The process exit code, or ``None`` when the process did not
            exit normally (killed, failed to start, output limit exceeded).
        stderr_tail: The last bytes of the tool's standard error, capped by the
            runner. **Untrusted**: it may echo metadata from the media. Use it
            only to classify the failure; never log it above DEBUG level and
            never put it in a message.
    """

    def __init__(
        self,
        tool: str,
        message: str,
        *,
        returncode: int | None = None,
        stderr_tail: bytes = b"",
    ) -> None:
        """Initializes the error.

        Args:
            tool: The tool name.
            message: A sanitized description of the failure.
            returncode: The exit code, if the process exited.
            stderr_tail: The capped tail of standard error (untrusted).
        """
        super().__init__(f"{tool}: {message}")
        self.tool = tool
        self.returncode = returncode
        self.stderr_tail = stderr_tail

    def __repr__(self) -> str:
        """Returns a representation without the untrusted standard error."""
        return f"{type(self).__name__}({self.args[0]!r}, returncode={self.returncode!r})"


class ExternalToolTimeoutError(ExternalToolError):
    """An external binary exceeded its timeout and was killed.

    Attributes:
        timeout_seconds: The timeout that was exceeded.
    """

    def __init__(
        self,
        tool: str,
        timeout_seconds: float,
        *,
        stderr_tail: bytes = b"",
    ) -> None:
        """Initializes the error.

        Args:
            tool: The tool name.
            timeout_seconds: The timeout that was exceeded.
            stderr_tail: The capped tail of standard error (untrusted).
        """
        super().__init__(
            tool,
            f"timed out after {timeout_seconds:g} s and was killed",
            stderr_tail=stderr_tail,
        )
        self.timeout_seconds = timeout_seconds


class ModelLoadError(ProvenanceKitError):
    """Model weights could not be downloaded, verified or loaded.

    Raised, for example, when a downloaded checkpoint fails its checksum or
    when a model file cannot be deserialized. A *missing package* is a
    :class:`BackendUnavailableError` instead, and a model that loaded but
    fails while it runs is a :class:`WatermarkBackendError`.
    """


class WatermarkBackendError(ProvenanceKitError):
    """A watermark model failed while it ran, or returned malformed output.

    Raised by the image, video and audio watermark modules and by their
    backends (TrustMark, VideoSeal, AudioSeal) in two cases:

    * the model fails during embedding or detection (for example a PyTorch
      device or runtime error); the original exception is chained as
      ``__cause__`` and never named in the message;
    * a backend returns output that breaks its Protocol: a payload or score
      vector of the wrong length or with non-finite values, or an image,
      frame or signal of another size or type than the input.

    Problems before the model runs are reported differently: weights that
    cannot be downloaded, verified or loaded raise :class:`ModelLoadError`, a
    missing package :class:`BackendUnavailableError`, and bad arguments
    ``TypeError`` or ``ValueError``. ``MemoryError`` propagates unchanged.
    """


class PayloadError(ProvenanceKitError):
    """A watermark payload cannot be built or is not valid.

    Raised by :class:`~provenance_kit.payload.PayloadCodec` when an identifier
    does not fit in the codec's capacity, and by
    :meth:`~provenance_kit.payload.DecodedPayload.require_valid` when a decoded
    payload fails validation. Wrong types and other caller mistakes raise
    ``TypeError`` or ``ValueError`` instead.
    """
