"""cast() -- validate an LLM call's output, retry on failure."""

from __future__ import annotations

import asyncio
import inspect
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, List, Optional, Union

from .errors import CastError
from .extract import extract_json

LlmFn = Callable[[List[dict]], Union[Awaitable[str], str]]
Validator = Callable[[Any], dict]
OnAttempt = Callable[[dict], None]


@dataclass
class Attempt:
    """One attempt at validating the LLM's response.

    Returned inside :class:`CastError.attempts`.
    """

    text: str
    parsed: Any
    error: str

    def to_dict(self) -> dict:
        return {"text": self.text, "parsed": self.parsed, "error": self.error}


_ASKED_FOR_JSON_RE = re.compile(r"\bonly\s+json\b|reply\s+with\s+json", re.IGNORECASE)


async def cast(
    *,
    llm: LlmFn,
    validate: Validator,
    prompt: str,
    system: Optional[str] = None,
    max_retries: int = 2,
    on_attempt: Optional[OnAttempt] = None,
) -> Any:
    """Get a typed/validated structured value out of any LLM call.

    Flow:

    1. Call ``llm(messages)`` to get text.
    2. Extract JSON from the text (handles fences and prose-wrapped JSON).
    3. Run ``validate(value)`` on the extracted JSON.
    4. If validation passes -> return validated value.
    5. If validation fails -> append the error to messages as feedback and
       retry, up to ``max_retries`` times. After exhausting retries, raise
       :class:`CastError`.

    BYO-LLM and BYO-validator: pass ``llm`` and ``validate`` callables.
    Adapters for pydantic / shape-spec / predicate live in
    :mod:`agentcast.adapters`.

    ``llm`` may be sync or async; both are supported.
    """
    if not callable(llm):
        raise TypeError("cast: llm must be a callable (messages) -> str (sync or async)")
    if not callable(validate):
        raise TypeError(
            "cast: validate must be a callable (value) -> "
            "{'valid': bool, 'value': ... | 'error': str}"
        )
    if not isinstance(prompt, str) or not prompt:
        raise TypeError("cast: prompt must be a non-empty string")
    if not isinstance(max_retries, int) or max_retries < 0:
        raise TypeError("cast: max_retries must be a non-negative int")
    if on_attempt is not None and not callable(on_attempt):
        raise TypeError("cast: on_attempt must be callable (or None)")

    messages: List[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": _append_json_instruction(prompt)})

    attempts: List[dict] = []

    for attempt_index in range(max_retries + 1):
        result = llm(messages)
        if inspect.isawaitable(result):
            text = await result
        else:
            text = result

        if not isinstance(text, str):
            raise TypeError(
                "cast: llm() must return a string (got " + type(text).__name__ + "). "
                "If your LLM SDK returns a richer object, unwrap the text content first."
            )

        parsed = extract_json(text)

        if parsed is None:
            err = "No JSON could be extracted from the response."
            attempts.append({"text": text, "parsed": None, "error": err})
            if on_attempt:
                on_attempt({"attempt": attempt_index + 1, "text": text, "parsed": None, "error": err})
            _push_feedback(messages, text, err)
            continue

        v = validate(parsed)
        if isinstance(v, dict) and v.get("valid") is True:
            return v.get("value", parsed)

        err = (
            v.get("error")
            if isinstance(v, dict) and isinstance(v.get("error"), str)
            else "Validation failed (no error message provided)."
        )
        attempts.append({"text": text, "parsed": parsed, "error": err})
        if on_attempt:
            on_attempt({"attempt": attempt_index + 1, "text": text, "parsed": parsed, "error": err})
        _push_feedback(messages, text, err)

    last_err = attempts[-1].get("error") if attempts else "(unknown)"
    plural = "" if len(attempts) == 1 else "s"
    raise CastError(
        "cast: failed validation after "
        + str(len(attempts))
        + " attempt"
        + plural
        + ". Last error: "
        + str(last_err),
        attempts,
    )


def _append_json_instruction(prompt: str) -> str:
    if _ASKED_FOR_JSON_RE.search(prompt):
        return prompt
    return prompt + "\n\nRespond with ONLY valid JSON. No prose, no markdown fences."


def _push_feedback(messages: List[dict], assistant_text: str, error: str) -> None:
    messages.append({"role": "assistant", "content": assistant_text})
    messages.append(
        {
            "role": "user",
            "content": (
                "Your previous response did not match the required shape. "
                "Error: " + error + "\n\n"
                "Try again. Respond with ONLY valid JSON that fixes the error above."
            ),
        }
    )
