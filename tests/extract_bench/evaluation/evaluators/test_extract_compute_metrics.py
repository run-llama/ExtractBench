"""Contract tests for the downstream-harness entry point into extract scoring.

A harness that keeps its own ``EvaluationResult`` model calls
``compute_metrics`` instead of ``evaluate``. These tests pin the two
properties it relies on: the metrics are the same ones ``evaluate`` reports,
and the normalized inputs come back so the harness's own metrics score the
same data the headline numbers did.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel

from extract_bench.evaluation.evaluators.extract import (
    ExtractEvaluator,
    ExtractMetricBundle,
    ExtractScoringInputs,
    is_extract_test_case,
)
from extract_bench.schemas.extract_output import ExtractOutput
from extract_bench.schemas.pipeline_io import InferenceRequest, InferenceResult
from extract_bench.schemas.product import ProductType
from extract_bench.test_cases.schema import ExtractFieldTestRule, ExtractTestCase


class HarnessProductType(Enum):
    """A harness's own product enum: same values, distinct members."""

    EXTRACT = "extract"


class HarnessExtractTestCase(ExtractTestCase):
    """A harness test case carrying fields this package knows nothing about."""

    identity_hints: dict[str, Any] | None = None


class HarnessInferenceResult(InferenceResult):
    """Mirrors how a harness redeclares ``product_type`` with its own enum."""

    product_type: HarnessProductType  # type: ignore[assignment]


def _inference_result(
    extracted_data: dict[str, Any],
    *,
    cls: type[InferenceResult] = InferenceResult,
) -> InferenceResult:
    now = datetime.now()
    product_type: Any = HarnessProductType.EXTRACT if cls is HarnessInferenceResult else ProductType.EXTRACT
    return cls(
        request=InferenceRequest(
            example_id="doc",
            source_file_path="/tmp/doc.pdf",
            product_type=ProductType.EXTRACT,
        ),
        pipeline_name="fake",
        product_type=product_type,
        raw_output={"job_id": "job-1"},
        output=ExtractOutput(
            example_id="doc",
            pipeline_name="fake",
            extracted_data=extracted_data,
            field_citations=[],
        ),
        started_at=now,
        completed_at=now,
        latency_in_ms=1,
    )


def _test_case(cls: type[ExtractTestCase] = ExtractTestCase, **extra: Any) -> ExtractTestCase:
    return cls(
        test_id="group/doc",
        group="group",
        file_path="/tmp/doc.pdf",
        schema={"type": "object", "properties": {"po_number": {"type": "string"}}},
        expected_output={"po_number": "PO-1"},
        test_rules=[
            {
                "type": "extract_field",
                "field_path": "po_number",
                "expected_value": "PO-1",
                "evidence": [{"value": "PO-1"}],
            }
        ],
        **extra,
    )


def test_compute_metrics_matches_evaluate() -> None:
    evaluator = ExtractEvaluator()
    inference_result = _inference_result({"po_number": "PO-1"})
    test_case = _test_case()

    bundle = evaluator.compute_metrics(inference_result, test_case)
    result = evaluator.evaluate(inference_result, test_case)

    assert isinstance(bundle, ExtractMetricBundle)
    assert [(m.metric_name, m.value) for m in bundle.metrics] == [(m.metric_name, m.value) for m in result.metrics]
    assert [(m.metric_name, m.value) for m in bundle.diagnostic_metrics] == [
        (m.metric_name, m.value) for m in result.diagnostic_metrics
    ]


def test_compute_metrics_returns_normalized_inputs() -> None:
    bundle = ExtractEvaluator().compute_metrics(_inference_result({"po_number": "PO-1"}), _test_case())

    assert isinstance(bundle.inputs, ExtractScoringInputs)
    # The scored data, not the raw provider payload: a harness adding its own
    # metrics must see the same post-unwrap shape the package metrics saw.
    assert bundle.inputs.extracted_data == {"po_number": "PO-1"}
    assert bundle.inputs.expected_output == {"po_number": "PO-1"}
    assert [rule.field_path for rule in bundle.inputs.field_rules] == ["po_number"]


def test_compute_metrics_accepts_a_harness_test_case_subclass() -> None:
    test_case = _test_case(HarnessExtractTestCase, identity_hints={"owners": ["name"]})

    bundle = ExtractEvaluator().compute_metrics(_inference_result({"po_number": "PO-1"}), test_case)

    assert any(m.metric_name == "accuracy" for m in bundle.metrics)


def test_can_evaluate_accepts_a_harness_product_enum() -> None:
    # A harness declares ``product_type`` with its own enum, whose EXTRACT
    # member is a distinct object from this package's. Only the value matches,
    # and that has to be enough or the harness can never be scored here.
    inference_result = _inference_result({"po_number": "PO-1"}, cls=HarnessInferenceResult)
    assert inference_result.product_type is HarnessProductType.EXTRACT

    assert ExtractEvaluator().can_evaluate(inference_result, _test_case()) is True
    assert any(
        m.metric_name == "accuracy" for m in ExtractEvaluator().compute_metrics(inference_result, _test_case()).metrics
    )


class ForeignExtractTestCase(BaseModel):
    """A harness case class that extends the harness's own base, not this one.

    This is the shape that matters: the downstream harness has its own test-case
    hierarchy, so it never inherits from ``ExtractTestCase``.
    """

    test_id: str
    data_schema: dict[str, Any]
    eval_data_schema: dict[str, Any]
    expected_output: dict[str, Any] | None
    test_rules: list[dict[str, Any]] | None

    def get_extract_field_rules(self) -> list[Any]:
        return [ExtractFieldTestRule.model_validate(rule) for rule in self.test_rules or []]


def test_structural_guard_accepts_a_foreign_test_case() -> None:
    schema = {"type": "object", "properties": {"po_number": {"type": "string"}}}
    test_case = ForeignExtractTestCase(
        test_id="group/doc",
        data_schema=schema,
        eval_data_schema=schema,
        expected_output={"po_number": "PO-1"},
        test_rules=[
            {
                "type": "extract_field",
                "field_path": "po_number",
                "expected_value": "PO-1",
                "evidence": [{"value": "PO-1"}],
            }
        ],
    )
    assert not isinstance(test_case, ExtractTestCase)
    assert is_extract_test_case(test_case)

    bundle = ExtractEvaluator().compute_metrics(_inference_result({"po_number": "PO-1"}), test_case)  # type: ignore[arg-type]
    assert {m.metric_name for m in bundle.metrics} >= {"accuracy"}


def test_structural_guard_rejects_a_non_extract_case() -> None:
    class ParseLikeCase(BaseModel):
        test_id: str = "group/doc"

    assert not is_extract_test_case(ParseLikeCase())
