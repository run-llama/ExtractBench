"""Annotation loading and native evaluation of row identity hints."""

import json
from datetime import datetime
from pathlib import Path

import pytest

from extract_bench.evaluation.evaluators.extract import ExtractEvaluator
from extract_bench.evaluation.metrics.extract import array_record_match_metric
from extract_bench.schemas.extract_output import ExtractOutput
from extract_bench.schemas.pipeline_io import InferenceRequest, InferenceResult
from extract_bench.schemas.product import ProductType
from extract_bench.test_cases.identity_hints import top_level_identity_fields
from extract_bench.test_cases.loader import load_test_cases

SCHEMA = {
    "type": "object",
    "properties": {
        "rows": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "amount": {"type": "number"}},
            },
        },
    },
}
EXPECTED = {"rows": [{"id": "A", "amount": 1}, {"id": "B", "amount": 2}]}
ACTUAL = {"rows": [{"id": "B", "amount": 3}, {"id": "A", "amount": 4}]}


def write_case(directory: Path, hints: object) -> None:
    (directory / "doc.pdf").touch()
    (directory / "doc.test.json").write_text(
        json.dumps({"data_schema": SCHEMA, "expected_output": EXPECTED, "_identity_hints": hints})
    )


@pytest.mark.parametrize("hint", ["id", ["id"], {"fields": ["id"]}])
@pytest.mark.parametrize("cap", [1, 100_000_000])
def test_loaded_hints_reach_array_matcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hint: object, cap: int
) -> None:
    write_case(tmp_path, {"rows": hint})
    (case,) = load_test_cases(str(tmp_path), product_type="extract")
    assert case.identity_hints == {"rows": hint}
    assert case.data_schema == SCHEMA
    monkeypatch.setattr(array_record_match_metric, "_MAX_RESIDUAL_ASSIGNMENT_CELLS", cap)
    now = datetime.now()
    result = InferenceResult(
        request=InferenceRequest(
            example_id="doc", source_file_path=str(tmp_path / "doc.pdf"), product_type=ProductType.EXTRACT
        ),
        pipeline_name="fixture",
        product_type=ProductType.EXTRACT,
        raw_output={},
        output=ExtractOutput(example_id="doc", pipeline_name="fixture", extracted_data=ACTUAL),
        started_at=now,
        completed_at=now,
        latency_in_ms=1,
    )
    evaluator = ExtractEvaluator()
    hinted = {m.metric_name: m for m in evaluator.compute_metrics(result, case).metrics}
    plain = {
        m.metric_name: m
        for m in evaluator.compute_metrics(result, case.model_copy(update={"identity_hints": None})).metrics
    }
    assert hinted["array_record_f1"].metadata["correct"] == 2
    assert hinted["array_record_f1"].value == 0.5
    assert plain["array_record_f1"].value == (0.0 if cap == 1 else 0.5)


@pytest.mark.parametrize(
    "hints",
    [
        [],
        "rows",
        {"rows": []},
        {"rows": [1]},
        {"rows": ""},
        {"rows": {"fields": ["id"], "typo": True}},
        {"": ["id"]},
        {"rows..nested": ["id"]},
    ],
)
def test_loader_rejects_malformed_hints(tmp_path: Path, hints: object) -> None:
    write_case(tmp_path, hints)
    with pytest.raises(ValueError, match="_identity_hints"):
        load_test_cases(str(tmp_path), product_type="extract")


def test_nested_hints_are_preserved_but_not_forwarded_as_top_level(tmp_path: Path) -> None:
    hints = {"rows": ["id"], "rows.children": {"fields": ["code", "date"]}}
    write_case(tmp_path, hints)
    (case,) = load_test_cases(str(tmp_path), product_type="extract")
    assert case.identity_hints == hints
    assert top_level_identity_fields(case.identity_hints) == {"rows": ("id",)}


@pytest.mark.parametrize("hints", [None, {}])
def test_optional_hints(tmp_path: Path, hints: object) -> None:
    write_case(tmp_path, hints)
    (case,) = load_test_cases(str(tmp_path), product_type="extract")
    assert top_level_identity_fields(case.identity_hints) == {}
