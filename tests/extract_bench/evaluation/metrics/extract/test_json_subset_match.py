from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from extract_bench.evaluation.metrics.extract.json_subset_match import (
    _is_nullable_numeric_field,
    json_subset_match_score,
)

_NULLABLE_NUMERIC_SCHEMA: dict[str, Any] = {
    "anyOf": [{"type": "number"}, {"type": "null"}],
    "default": None,
}
_NULLABLE_NUMERIC_TYPE_LIST: dict[str, Any] = {"type": ["number", "null"]}
_PLAIN_NUMERIC_SCHEMA: dict[str, Any] = {"type": "number"}


def _wrap_schema_with_numeric_field(field_schema: Mapping[str, Any]) -> dict[str, Any]:
    return {"type": "object", "properties": {"amount": field_schema}}


def test_is_nullable_numeric_field_recognizes_anyof_shape() -> None:
    assert _is_nullable_numeric_field(_NULLABLE_NUMERIC_SCHEMA) is True
    assert _is_nullable_numeric_field({"anyOf": [{"type": "null"}, {"type": "number"}]}) is True
    assert _is_nullable_numeric_field(_NULLABLE_NUMERIC_TYPE_LIST) is True
    assert _is_nullable_numeric_field({"type": ["null", "number"]}) is True


def test_is_nullable_numeric_field_rejects_other_shapes() -> None:
    assert _is_nullable_numeric_field(_PLAIN_NUMERIC_SCHEMA) is False
    assert _is_nullable_numeric_field({"anyOf": [{"type": "string"}, {"type": "null"}]}) is False
    assert _is_nullable_numeric_field({"type": ["number", "null", "string"]}) is False
    assert _is_nullable_numeric_field({"anyOf": [{"type": "number"}, {"type": "null"}, {"type": "string"}]}) is False
    assert _is_nullable_numeric_field(None) is False
    assert _is_nullable_numeric_field({}) is False


def test_null_equals_zero_on_nullable_numeric() -> None:
    schema = _wrap_schema_with_numeric_field(_NULLABLE_NUMERIC_SCHEMA)
    assert json_subset_match_score(expected={"amount": None}, actual={"amount": 0.0}, data_schema=schema) == 1.0
    assert json_subset_match_score(expected={"amount": 0.0}, actual={"amount": None}, data_schema=schema) == 1.0
    assert json_subset_match_score(expected={"amount": None}, actual={"amount": 0}, data_schema=schema) == 1.0
    schema_type_list = _wrap_schema_with_numeric_field(_NULLABLE_NUMERIC_TYPE_LIST)
    assert (
        json_subset_match_score(expected={"amount": None}, actual={"amount": 0.0}, data_schema=schema_type_list) == 1.0
    )


def test_null_not_equals_zero_on_non_nullable_numeric() -> None:
    schema = _wrap_schema_with_numeric_field(_PLAIN_NUMERIC_SCHEMA)
    assert json_subset_match_score(expected={"amount": None}, actual={"amount": 0.0}, data_schema=schema) == 0.0


def test_nonzero_still_mismatches_on_nullable_numeric() -> None:
    schema = _wrap_schema_with_numeric_field(_NULLABLE_NUMERIC_SCHEMA)
    assert json_subset_match_score(expected={"amount": None}, actual={"amount": 1.5}, data_schema=schema) == 0.0


def test_no_schema_falls_back_to_strict_equality() -> None:
    assert json_subset_match_score(expected={"amount": None}, actual={"amount": 0.0}) == 0.0
