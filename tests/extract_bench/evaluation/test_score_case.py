"""Tests for single-case extraction scoring.

Builds a real dataset directory on disk — a document, a ``.test.json`` in the
v0.2 ``_field_rules`` shape, and a prediction — so the loader's rule
normalization is exercised rather than stubbed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from extract_bench.evaluation.score_case import (
    load_extract_test_case,
    score_extract_prediction,
)

_SCHEMA = {
    "type": "object",
    "properties": {"manager": {"type": "string"}, "total": {"type": "number"}},
}


def _write_case(directory: Path, test_config: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "filing.pdf").write_bytes(b"%PDF-1.4 stub")
    test_json = directory / "filing.test.json"
    test_json.write_text(json.dumps(test_config))
    return test_json


def _config(**overrides) -> dict:
    config = {
        "data_schema": _SCHEMA,
        "expected_output": {"manager": "Leonteq Securities AG", "total": 1234.5},
        "_field_rules": {
            "manager": {"comparator": "case_insensitive"},
            "total": {"comparator": "numeric"},
        },
    }
    config.update(overrides)
    return config


def test_field_rules_dict_is_normalized_by_the_loader(tmp_path: Path) -> None:
    case = load_extract_test_case(_write_case(tmp_path / "case", _config()))

    assert {rule.field_path for rule in case.get_extract_field_rules()} == {"manager", "total"}
    assert case.data_schema == _SCHEMA


def test_exact_prediction_scores_one(tmp_path: Path) -> None:
    case = load_extract_test_case(_write_case(tmp_path / "case", _config()))

    result = score_extract_prediction(case, {"manager": "Leonteq Securities AG", "total": 1234.5})

    assert result["scorable"] is True
    assert result["metrics"]["extract_unified_value_f1"] == pytest.approx(1.0)
    assert result["value_f1_metadata"]["expected_cells"] == 2


def test_a_wrong_value_lowers_f1(tmp_path: Path) -> None:
    case = load_extract_test_case(_write_case(tmp_path / "case", _config()))

    result = score_extract_prediction(case, {"manager": "Someone Else", "total": 1234.5})

    assert result["metrics"]["extract_unified_value_f1"] < 1.0
    assert result["value_f1_metadata"]["v_correct"] == 1


def test_a_case_with_nothing_to_compare_is_unscorable(tmp_path: Path) -> None:
    test_json = _write_case(tmp_path / "case", {"data_schema": _SCHEMA, "expected_output": {}, "test_rules": []})
    case = load_extract_test_case(test_json)

    result = score_extract_prediction(case, {"manager": "anything"})

    assert result["scorable"] is False
    assert result["metrics"] == {}


def test_source_document_is_found_beside_the_test_json(tmp_path: Path) -> None:
    case = load_extract_test_case(_write_case(tmp_path / "case", _config()))

    assert case.file_path.name == "filing.pdf"


def test_a_missing_source_document_is_reported(tmp_path: Path) -> None:
    directory = tmp_path / "case"
    directory.mkdir()
    test_json = directory / "filing.test.json"
    test_json.write_text(json.dumps(_config()))

    with pytest.raises(ValueError, match="No source document"):
        load_extract_test_case(test_json)
