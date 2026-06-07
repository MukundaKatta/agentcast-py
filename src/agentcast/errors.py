"""CastError -- raised when cast() exhausts its retries."""

from __future__ import annotations

from typing import List


class CastError(Exception):
    """Raised by :func:`agentcast.cast` when retries are exhausted.

    Carries the full attempt history so the caller can debug why it failed.

    Attributes:
        attempts: List of dicts ``{"text": str, "parsed": Any, "error": str}``.
        last_error: Validation error message from the final attempt.
        last_text: Raw text the model returned on the final attempt.
        last_parsed: Value that was extracted (or ``None`` if extraction failed).
    """

    def __init__(self, message: str, attempts: List[dict]) -> None:
        super().__init__(message)
        self.name = "CastError"
        self.attempts = list(attempts)
        last = self.attempts[-1] if self.attempts else None
        self.last_error = last.get("error") if last else None
        self.last_text = last.get("text") if last else None
        self.last_parsed = last.get("parsed") if last else None
