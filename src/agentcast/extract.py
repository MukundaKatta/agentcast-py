"""Pull JSON out of a possibly-prosed LLM response.

Strategies, in order:

1. Try the whole text as JSON.
2. Look for a fenced `````json ... ````` block
   (also accepts plain triple-backtick fences).
3. Find the largest balanced ``{...}`` or ``[...]`` substring.

Returns the parsed value, or ``None`` if no parseable JSON was found.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

_FENCE_RE = re.compile(r"```(?:json|JSON|Json)?\s*\n?([\s\S]*?)\n?```", re.MULTILINE)


def extract_json(text: str) -> Optional[Any]:
    """Extract a JSON value from arbitrary LLM text.

    Returns the parsed value, or ``None`` if extraction failed.
    """
    if not isinstance(text, str):
        return None
    trimmed = text.strip()
    if not trimmed:
        return None

    # 1. Whole text
    parsed = _try_parse(trimmed)
    if parsed is not _UNPARSEABLE:
        return parsed

    # 2. Fenced code block
    fenced = _extract_fenced(trimmed)
    if fenced is not None:
        parsed = _try_parse(fenced)
        if parsed is not _UNPARSEABLE:
            return parsed

    # 3. Largest balanced substring
    balanced = _extract_largest_balanced(trimmed)
    if balanced is not None:
        parsed = _try_parse(balanced)
        if parsed is not _UNPARSEABLE:
            return parsed

    return None


_UNPARSEABLE = object()


def _try_parse(s: str):
    try:
        return json.loads(s)
    except Exception:
        return _UNPARSEABLE


def _extract_fenced(text: str) -> Optional[str]:
    m = _FENCE_RE.search(text)
    return m.group(1).strip() if m else None


def _extract_largest_balanced(text: str) -> Optional[str]:
    best = None
    n = len(text)
    for i in range(n):
        ch = text[i]
        if ch != "{" and ch != "[":
            continue
        end = _find_matching(text, i)
        if end == -1:
            continue
        candidate = text[i : end + 1]
        if best is None or len(candidate) > len(best):
            best = candidate
    return best


def _find_matching(text: str, start: int) -> int:
    open_ch = text[start]
    close_ch = "}" if open_ch == "{" else "]"
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if in_string:
            if ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return i
    return -1
