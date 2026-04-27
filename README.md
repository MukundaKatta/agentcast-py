# agentcast-py

[![PyPI](https://img.shields.io/pypi/v/agentcast-py.svg)](https://pypi.org/project/agentcast-py/)
[![Python](https://img.shields.io/pypi/pyversions/agentcast-py.svg)](https://pypi.org/project/agentcast-py/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Structured output for any LLM call.** Validate the response, retry with the validation error as feedback, raise `CastError` if all retries fail. BYO LLM, BYO validator. Zero runtime dependencies.

Python port of [@mukundakatta/agentcast](https://github.com/MukundaKatta/agentcast).

## Install

```bash
pip install agentcast-py
# pydantic adapter is optional:
pip install "agentcast-py[pydantic]"
```

## Usage

```python
import asyncio
from anthropic import AsyncAnthropic
from pydantic import BaseModel
from agentcast import cast, adapters, CastError

client = AsyncAnthropic()

async def llm(messages):
    resp = await client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=messages,
    )
    return resp.content[0].text

class Diagnosis(BaseModel):
    code: str
    severity: str
    notes: str

async def main():
    out = await cast(
        llm=llm,
        validate=adapters.pydantic(Diagnosis),
        prompt="Triage: server is on fire. Output a Diagnosis JSON object.",
        max_retries=3,
    )
    print(out)  # validated dict

asyncio.run(main())
```

If the LLM returns prose, fenced JSON, or near-misses, `cast()` extracts the JSON, validates it, and retries with the validation error in the chat history -- so the next attempt has the spec. After `max_retries` failures, raises `CastError` carrying every attempt for postmortem.

## Sync LLMs work too

```python
def sync_llm(messages):
    return openai_client.chat.completions.create(...).choices[0].message.content

out = await cast(llm=sync_llm, validate=schema, prompt="...", max_retries=2)
```

`cast()` is async (so you can await your async LLM), but the `llm` callable can be sync or async.

## API

### `cast(*, llm, validate, prompt, system=None, max_retries=2, on_attempt=None) -> Any`

Async. Returns the validated value (whatever the validator's `value` field is). Raises `CastError` on exhausted retries.

### `extract_json(text) -> Any | None`

Sync. Pulls the largest valid JSON value out of arbitrary text -- whole text, fenced block, or balanced substring. Returns `None` if nothing parseable.

### `adapters.shape(spec)`

Tiny built-in shape checker. Same as the sibling `agentvet`. `{"field": "str|int|...|str?"}`.

### `adapters.fn(predicate, error_builder?)`

Predicate adapter.

### `adapters.pydantic(model_cls)`

Wraps a pydantic v2 `BaseModel`. Requires `pip install agentcast-py[pydantic]`.

### `CastError`

Carries `attempts: list[dict]`, `last_error`, `last_text`, `last_parsed` for postmortem.

## API differences from the JS sibling

* `cast()` is `async def` (Python's natural pattern for LLM calls).
* `llm` may be sync or async; both are auto-detected.
* `adapters.pydantic` replaces `adapters.zod` as the natural Python adapter.
* Keyword args throughout (`max_retries=`, `on_attempt=`).
* Validators return Python dicts: `{"valid": bool, "value": ... | "error": str}`.

See the JS sibling's [README](https://github.com/MukundaKatta/agentcast) for the full design notes.
