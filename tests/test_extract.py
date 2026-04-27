"""Tests for ``agentcast.extract_json``."""

from __future__ import annotations

from agentcast import extract_json


def test_extract_whole_text_object():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_whole_text_array():
    assert extract_json("[1, 2, 3]") == [1, 2, 3]


def test_extract_fenced_json_block():
    text = "Here you go:\n```json\n{\"a\": 1}\n```\nThanks!"
    assert extract_json(text) == {"a": 1}


def test_extract_fenced_block_without_lang_tag():
    text = "Output:\n```\n{\"x\": 2}\n```"
    assert extract_json(text) == {"x": 2}


def test_extract_largest_balanced_substring_in_prose():
    text = "Here is the data: {\"name\": \"Alice\", \"items\": [1, 2, 3]} -- enjoy."
    assert extract_json(text) == {"name": "Alice", "items": [1, 2, 3]}


def test_extract_returns_none_on_garbage():
    assert extract_json("nothing to see here") is None


def test_extract_returns_none_on_empty_string():
    assert extract_json("") is None


def test_extract_returns_none_on_non_string():
    assert extract_json(None) is None  # type: ignore[arg-type]
    assert extract_json(123) is None  # type: ignore[arg-type]


def test_extract_picks_largest_when_multiple_candidates():
    # Two JSON-shaped substrings; the longer should win.
    text = "first {\"a\": 1} then {\"b\": 2, \"more\": [1, 2, 3, 4]}"
    assert extract_json(text) == {"b": 2, "more": [1, 2, 3, 4]}


def test_extract_handles_strings_with_braces_inside():
    text = 'Sure: {"motto": "do {} better", "n": 1}'
    assert extract_json(text) == {"motto": "do {} better", "n": 1}
