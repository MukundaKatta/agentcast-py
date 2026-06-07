"""Tests for ``agentcast.cast``."""

from __future__ import annotations

import asyncio

import pytest

from agentcast import CastError, adapters, cast


def _run(coro):
    """Compat helper -- run a coroutine in a fresh loop without pytest-asyncio."""
    return asyncio.new_event_loop().run_until_complete(coro)


def test_cast_returns_validated_value_on_first_try():
    schema = adapters.shape({"name": "str", "age": "int"})

    async def llm(messages):
        return '{"name": "Alice", "age": 30}'

    out = _run(
        cast(llm=llm, validate=schema, prompt="Give me a person.", max_retries=0)
    )
    assert out == {"name": "Alice", "age": 30}


def test_cast_unwraps_fenced_json():
    schema = adapters.shape({"x": "int"})

    async def llm(messages):
        return 'Sure!\n```json\n{"x": 7}\n```'

    out = _run(cast(llm=llm, validate=schema, prompt="Give me x.", max_retries=0))
    assert out == {"x": 7}


def test_cast_retries_with_validation_error_then_succeeds():
    schema = adapters.shape({"name": "str"})
    calls = {"n": 0}

    async def llm(messages):
        calls["n"] += 1
        if calls["n"] == 1:
            return '{"nope": 1}'  # missing 'name'
        # On retry, the error must be in the conversation history.
        history = " ".join(m["content"] for m in messages if m["role"] == "user")
        assert "missing required field 'name'" in history
        return '{"name": "Bob"}'

    out = _run(cast(llm=llm, validate=schema, prompt="Person.", max_retries=2))
    assert out == {"name": "Bob"}
    assert calls["n"] == 2


def test_cast_raises_cast_error_when_retries_exhausted():
    schema = adapters.shape({"name": "str"})

    async def llm(messages):
        return '{"nope": 1}'

    with pytest.raises(CastError) as exc:
        _run(cast(llm=llm, validate=schema, prompt="Person.", max_retries=1))
    err = exc.value
    assert len(err.attempts) == 2  # 1 initial + 1 retry
    assert err.last_error is not None
    assert err.last_text == '{"nope": 1}'


def test_cast_raises_when_no_json_extractable_then_retries():
    schema = adapters.shape({"x": "int"})
    seq = ["sorry I cannot help", '{"x": 5}']
    i = {"n": 0}

    async def llm(messages):
        text = seq[i["n"]]
        i["n"] += 1
        return text

    out = _run(cast(llm=llm, validate=schema, prompt="x?", max_retries=2))
    assert out == {"x": 5}


def test_cast_invalid_llm_type_raises():
    with pytest.raises(TypeError):
        _run(cast(llm="not callable", validate=lambda v: {"valid": True}, prompt="x"))  # type: ignore[arg-type]


def test_cast_invalid_validate_type_raises():
    async def llm(_):
        return "{}"

    with pytest.raises(TypeError):
        _run(cast(llm=llm, validate="not callable", prompt="x"))  # type: ignore[arg-type]


def test_cast_invalid_prompt_raises():
    async def llm(_):
        return "{}"

    with pytest.raises(TypeError):
        _run(cast(llm=llm, validate=lambda v: {"valid": True}, prompt=""))


def test_cast_negative_max_retries_raises():
    async def llm(_):
        return "{}"

    with pytest.raises(TypeError):
        _run(
            cast(
                llm=llm, validate=lambda v: {"valid": True}, prompt="x", max_retries=-1
            )
        )


def test_cast_on_attempt_called_for_each_failure():
    schema = adapters.shape({"x": "int"})
    seen = []

    async def llm(_):
        return '{"nope": 1}'

    with pytest.raises(CastError):
        _run(
            cast(
                llm=llm,
                validate=schema,
                prompt="x?",
                max_retries=2,
                on_attempt=lambda info: seen.append(info["attempt"]),
            )
        )
    assert seen == [1, 2, 3]


def test_cast_appends_json_instruction_idempotently():
    schema = adapters.shape({"x": "int"})
    captured = {"prompt": None}

    async def llm(messages):
        captured["prompt"] = messages[-1]["content"]
        return '{"x": 1}'

    _run(cast(llm=llm, validate=schema, prompt="hi", max_retries=0))
    assert "Respond with ONLY valid JSON" in captured["prompt"]

    captured["prompt"] = None
    _run(
        cast(
            llm=llm,
            validate=schema,
            prompt="hi -- reply with json",
            max_retries=0,
        )
    )
    # Idempotent: didn't double-add.
    assert captured["prompt"].count("Respond with ONLY valid JSON") == 0


def test_cast_supports_sync_llm():
    """``llm`` may be a plain (non-async) callable returning a str."""
    schema = adapters.shape({"x": "int"})

    def sync_llm(messages):
        return '{"x": 5}'

    out = _run(cast(llm=sync_llm, validate=schema, prompt="x?", max_retries=0))
    assert out == {"x": 5}


def test_cast_with_pydantic_adapter():
    pyd = pytest.importorskip("pydantic")

    class Person(pyd.BaseModel):
        name: str
        age: int

    async def llm(_):
        return '{"name": "Alice", "age": 30}'

    out = _run(
        cast(
            llm=llm,
            validate=adapters.pydantic(Person),
            prompt="A person.",
            max_retries=0,
        )
    )
    assert out == {"name": "Alice", "age": 30}
