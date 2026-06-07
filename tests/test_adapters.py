"""Tests for ``agentcast.adapters`` (shape / fn / pydantic)."""

from __future__ import annotations

import pytest

from agentcast import adapters


def test_shape_accepts_valid_object():
    v = adapters.shape({"name": "str", "age": "int"})
    assert v({"name": "Alice", "age": 30}) == {
        "valid": True,
        "value": {"name": "Alice", "age": 30},
    }


def test_shape_rejects_non_object():
    v = adapters.shape({"name": "str"})
    res = v([1, 2, 3])
    assert res["valid"] is False
    assert "expected an object" in res["error"]


def test_shape_reports_missing_required_field():
    v = adapters.shape({"name": "str", "age": "int"})
    res = v({"name": "Alice"})
    assert res["valid"] is False
    assert "missing required field 'age'" in res["error"]


def test_shape_optional_field_may_be_absent():
    v = adapters.shape({"name": "str", "email": "str?"})
    assert v({"name": "Alice"})["valid"] is True


def test_shape_optional_field_is_type_checked_when_present():
    v = adapters.shape({"email": "str?"})
    res = v({"email": 123})
    assert res["valid"] is False
    assert "field 'email' should be str" in res["error"]


def test_shape_reports_wrong_type_with_description():
    v = adapters.shape({"age": "int"})
    res = v({"age": "thirty"})
    assert res["valid"] is False
    assert "field 'age' should be int, got str" in res["error"]


def test_shape_accepts_js_synonyms():
    v = adapters.shape(
        {"name": "string", "n": "number", "items": "array", "meta": "object"}
    )
    assert v({"name": "x", "n": 1.5, "items": [], "meta": {}})["valid"] is True


def test_shape_bool_is_not_int():
    # bools must not satisfy an int/number spec.
    v = adapters.shape({"age": "int"})
    res = v({"age": True})
    assert res["valid"] is False
    assert "got bool" in res["error"]


def test_shape_int_is_not_bool():
    v = adapters.shape({"flag": "bool"})
    res = v({"flag": 1})
    assert res["valid"] is False


def test_shape_unknown_type_name_never_matches():
    v = adapters.shape({"x": "widget"})
    res = v({"x": "anything"})
    assert res["valid"] is False


def test_shape_rejects_non_mapping_spec():
    with pytest.raises(TypeError):
        adapters.shape(["not", "a", "mapping"])


def test_fn_passes_when_predicate_true():
    v = adapters.fn(lambda x: x > 0)
    assert v(5) == {"valid": True, "value": 5}


def test_fn_fails_with_default_error():
    v = adapters.fn(lambda x: x > 0)
    res = v(-1)
    assert res["valid"] is False
    assert res["error"] == "value did not pass predicate"


def test_fn_fails_with_static_error_string():
    v = adapters.fn(lambda x: False, "must be positive")
    assert v(0)["error"] == "must be positive"


def test_fn_fails_with_callable_error_builder():
    v = adapters.fn(lambda x: False, lambda val: f"bad value: {val}")
    assert v(42)["error"] == "bad value: 42"


def test_fn_rejects_non_callable_predicate():
    with pytest.raises(TypeError):
        adapters.fn("not callable")


def test_pydantic_rejects_non_model():
    with pytest.raises(TypeError):
        adapters.pydantic(object)


def test_pydantic_reports_validation_errors():
    pyd = pytest.importorskip("pydantic")

    class Person(pyd.BaseModel):
        name: str
        age: int

    v = adapters.pydantic(Person)
    res = v({"name": "Alice"})  # missing age
    assert res["valid"] is False
    assert "age" in res["error"]
