"""Validator adapters for ``cast()``.

Same shape as ``agentvet.adapters`` so you can swap them. Zero deps for
``shape``/``fn``; ``pydantic`` is an optional extra.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Union


class _Adapters:
    """Namespace object so callers do ``adapters.shape(...)`` etc."""

    @staticmethod
    def shape(spec: Mapping[str, str]) -> Callable[[Any], dict]:
        """Tiny built-in shape checker.

        Spec format::

            {"name": "str", "age": "int", "email": "str?"}

        Required by default. Suffix with ``?`` for optional. Accepts both
        Python type names (``"str"``, ``"int"``, ``"list"``, ``"dict"``) and
        JS sibling synonyms (``"string"``, ``"number"``, ``"array"``,
        ``"object"``).
        """
        if not isinstance(spec, Mapping):
            raise TypeError("adapters.shape: spec must be a mapping (dict)")

        def validator(value):
            if not isinstance(value, Mapping):
                return {"valid": False, "error": "expected an object"}
            errors = []
            for key, type_spec in spec.items():
                optional = type_spec.endswith("?")
                base_type = type_spec[:-1] if optional else type_spec
                present = key in value
                if not present:
                    if not optional:
                        errors.append("missing required field '" + key + "'")
                    continue
                if not _matches_type(value[key], base_type):
                    errors.append(
                        "field '"
                        + key
                        + "' should be "
                        + base_type
                        + ", got "
                        + _describe(value[key])
                    )
            if errors:
                return {"valid": False, "error": "; ".join(errors)}
            return {"valid": True, "value": value}

        return validator

    @staticmethod
    def fn(
        predicate: Callable[[Any], bool],
        error_builder: Union[str, Callable[[Any], str]] = "value did not pass predicate",
    ) -> Callable[[Any], dict]:
        """Predicate adapter for ad-hoc validation."""
        if not callable(predicate):
            raise TypeError("adapters.fn: predicate must be callable")

        def validator(value):
            if predicate(value):
                return {"valid": True, "value": value}
            err = error_builder(value) if callable(error_builder) else error_builder
            return {"valid": False, "error": err}

        return validator

    @staticmethod
    def pydantic(model_cls) -> Callable[[Any], dict]:
        """Adapter for pydantic v2 ``BaseModel`` subclasses.

        Requires ``pip install agentcast-py[pydantic]``.
        """
        if not hasattr(model_cls, "model_validate"):
            raise TypeError(
                "adapters.pydantic: model_cls must be a pydantic v2 BaseModel "
                "(missing model_validate). Install pydantic >= 2."
            )

        def validator(value):
            try:
                model = model_cls.model_validate(value)
            except Exception as exc:
                msgs = []
                if hasattr(exc, "errors"):
                    try:
                        for e in exc.errors():
                            loc = ".".join(str(p) for p in e.get("loc", [])) or "<root>"
                            msgs.append(loc + ": " + e.get("msg", "invalid"))
                    except Exception:
                        msgs = [str(exc)]
                if not msgs:
                    msgs = [str(exc)]
                return {"valid": False, "error": "; ".join(msgs)}
            try:
                out = model.model_dump()
            except Exception:
                out = model
            return {"valid": True, "value": out}

        return validator


# Singleton exposed as ``agentcast.adapters``.
adapters = _Adapters()


# --- helpers --------------------------------------------------------------

_SYNONYMS = {
    "string": str,
    "str": str,
    "number": (int, float),
    "int": int,
    "integer": int,
    "float": float,
    "boolean": bool,
    "bool": bool,
    "array": list,
    "list": list,
    "object": dict,
    "dict": dict,
}


def _matches_type(value, type_name: str) -> bool:
    expected = _SYNONYMS.get(type_name)
    if expected is None:
        return False
    if expected is bool:
        return isinstance(value, bool)
    if expected in (int, float):
        if isinstance(value, bool):
            return False
        return isinstance(value, expected)
    if expected is dict:
        return isinstance(value, dict)
    return isinstance(value, expected)


def _describe(value) -> str:
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, list):
        return "list"
    return type(value).__name__
