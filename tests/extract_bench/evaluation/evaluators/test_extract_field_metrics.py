"""Tests for ExtractEvaluator metrics that still fire on v0.2 gold."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from extract_bench.evaluation.evaluators.extract import ExtractEvaluator
from extract_bench.schemas.evaluation import EvaluationResult, MetricValue
from extract_bench.schemas.extract_output import ExtractOutput, FieldCitation
from extract_bench.schemas.pipeline_io import InferenceRequest, InferenceResult
from extract_bench.schemas.product import ProductType
from extract_bench.test_cases.schema import ExtractTestCase


def _make_inference_result(
    extracted_data: dict[str, Any],
    *,
    field_citations: list[FieldCitation] | None = None,
) -> InferenceResult:
    now = datetime.now()
    return InferenceResult(
        request=InferenceRequest(
            example_id="doc",
            source_file_path="/tmp/doc.pdf",
            product_type=ProductType.EXTRACT,
        ),
        pipeline_name="fake",
        product_type=ProductType.EXTRACT,
        raw_output={},
        output=ExtractOutput(
            example_id="doc",
            pipeline_name="fake",
            extracted_data=extracted_data,
            field_citations=field_citations or [],
        ),
        started_at=now,
        completed_at=now,
        latency_in_ms=1,
    )


def _make_inference_result_list(extracted_data: list[Any]) -> InferenceResult:
    now = datetime.now()
    return InferenceResult(
        request=InferenceRequest(
            example_id="doc",
            source_file_path="/tmp/doc.pdf",
            product_type=ProductType.EXTRACT,
        ),
        pipeline_name="fake",
        product_type=ProductType.EXTRACT,
        raw_output={},
        output=ExtractOutput(
            example_id="doc",
            pipeline_name="fake",
            extracted_data=extracted_data,  # type: ignore[arg-type]
            field_citations=[],
        ),
        started_at=now,
        completed_at=now,
        latency_in_ms=1,
    )


def _make_test_case(
    *,
    expected_output: dict[str, Any] | None = None,
    test_rules: list[dict[str, Any]] | None = None,
    data_schema: dict[str, Any] | None = None,
) -> ExtractTestCase:
    return ExtractTestCase(
        test_id="group/doc",
        group="group",
        file_path="/tmp/doc.pdf",
        schema=data_schema or {"type": "object"},
        expected_output=expected_output,
        test_rules=test_rules,
    )


def _v02_rule(field_path: str, expected_value: Any, **extra: Any) -> dict[str, Any]:
    return {
        "type": "extract_field",
        "field_path": field_path,
        "expected_value": expected_value,
        "evidence": [{"value": expected_value}],
        **extra,
    }


def _metrics_by_name(metrics: list[MetricValue]) -> dict[str, MetricValue]:
    return {m.metric_name: m for m in metrics}


def _diagnostic_metrics_by_name(result: EvaluationResult) -> dict[str, MetricValue]:
    return _metrics_by_name(result.diagnostic_metrics)


def test_parse_rules_do_not_enter_extract_metrics() -> None:
    test_case = _make_test_case(
        expected_output={"po_number": "PO-1"},
        test_rules=[
            _v02_rule("po_number", "PO-1"),
            {
                "type": "table",
                "cell": "[yes]",
                "top_heading": "Vehicle 1",
                "left_heading": "Navigation",
            },
        ],
    )

    result = ExtractEvaluator().evaluate(_make_inference_result({"po_number": "PO-1"}), test_case)
    metrics = _metrics_by_name(result.metrics)
    assert "rule_pass_rate" not in metrics
    assert "rule_table_pass_rate" not in metrics
    assert metrics["extract_evidence_value_pass_rate"].value == 1.0


def test_evaluator_emits_confidence_scoped_metrics_from_field_citations() -> None:
    tc = _make_test_case(
        expected_output={"invoice_id": "INV-1", "total": 100},
        test_rules=[
            _v02_rule("invoice_id", "INV-1"),
            _v02_rule("total", 100),
        ],
    )
    ir = _make_inference_result(
        {"invoice_id": "INV-1", "total": 99},
        field_citations=[
            FieldCitation(field_path="invoice_id", page=1, confidence=0.99),
            FieldCitation(field_path="total", page=1, confidence=0.97),
        ],
    )

    result = ExtractEvaluator().evaluate(ir, tc)
    metrics = _metrics_by_name(result.metrics)

    assert "confidence_scoped_auc" in metrics
    assert metrics["confidence_scoped_precision_at_0_95"].value == 0.5
    assert metrics["confidence_scoped_coverage_at_0_95"].value == 1.0
    assert metrics["confidence_scoped_precision_at_0_99"].value == 1.0
    assert metrics["confidence_scoped_coverage_at_0_99"].value == 0.5
    assert metrics["confidence_scoped_auc"].metadata["fields_with_confidence"] == 2


def test_evaluator_derives_confidence_rows_from_expected_output_without_rules() -> None:
    tc = _make_test_case(expected_output={"invoice_id": "INV-1", "total": 100})
    ir = _make_inference_result(
        {"invoice_id": "INV-1", "total": 99},
        field_citations=[
            FieldCitation(field_path="invoice_id", page=1, confidence=0.99),
            FieldCitation(field_path="total", page=1, confidence=0.97),
        ],
    )

    result = ExtractEvaluator().evaluate(ir, tc)
    metrics = _metrics_by_name(result.metrics)

    assert "confidence_scoped_auc" in metrics
    assert metrics["confidence_scoped_auc"].metadata["total_fields"] == 2
    assert metrics["confidence_scoped_precision_at_0_99"].value == 1.0


def test_confidence_scoped_metrics_expand_structured_array_rules_to_leaves() -> None:
    tc = _make_test_case(
        expected_output={
            "line_items": [
                {"sku": "A", "amount": 100},
                {"sku": "B", "amount": 200},
            ]
        },
        test_rules=[
            {
                "type": "extract_field",
                "field_path": "line_items",
                "comparator": {"sku": "case_insensitive", "amount": "number"},
                "structural": "match_by:sku",
                "evidence": [
                    {"page": 1, "value": {"sku": "A", "amount": 100}},
                    {"page": 1, "value": {"sku": "B", "amount": 200}},
                ],
            }
        ],
    )
    ir = _make_inference_result(
        {
            "line_items": [
                {"sku": "A", "amount": 200},
                {"sku": "B", "amount": 100},
            ]
        },
        field_citations=[
            FieldCitation(field_path="line_items[0].sku", page=1, confidence=0.99),
            FieldCitation(field_path="line_items[0].amount", page=1, confidence=0.99),
            FieldCitation(field_path="line_items[1].sku", page=1, confidence=0.99),
            FieldCitation(field_path="line_items[1].amount", page=1, confidence=0.99),
        ],
    )

    result = ExtractEvaluator().evaluate(ir, tc)
    metrics = _metrics_by_name(result.metrics)

    assert metrics["confidence_scoped_auc"].metadata["total_fields"] == 4
    assert metrics["confidence_scoped_auc"].metadata["fields_with_confidence"] == 4
    assert metrics["confidence_scoped_precision_at_0_95"].value == 0.5
    assert metrics["confidence_scoped_coverage_at_0_95"].value == 1.0
    rows = metrics["confidence_scoped_auc"].metadata["field_rows_sample"]
    assert {row["field_path"] for row in rows} == {
        "line_items[0].sku",
        "line_items[0].amount",
        "line_items[1].sku",
        "line_items[1].amount",
    }


def test_top_level_field_accuracy_diagnostics_remain() -> None:
    tc = _make_test_case(expected_output={"po_number": "PO-1"})
    ir = _make_inference_result({"po_number": "PO-1"})
    result = ExtractEvaluator().evaluate(ir, tc)
    metrics = _metrics_by_name(result.metrics)
    diagnostics = _diagnostic_metrics_by_name(result)
    assert "field_accuracy_po_number" not in metrics
    assert "field_accuracy_po_number" in diagnostics
    assert diagnostics["field_accuracy_po_number"].value == 1.0
    assert "extract_field_value_pass_rate" not in metrics
    assert "field_accuracy[po_number]" not in diagnostics


def test_list_unwrap_applied_on_list_rooted_prediction() -> None:
    tc = _make_test_case(
        expected_output={"personnel": [{"name": "Alice"}], "client_id": "C-1"},
        test_rules=[
            _v02_rule("personnel[0].name", "Alice"),
            _v02_rule("client_id", "C-1"),
        ],
    )
    ir = _make_inference_result_list([{"name": "Alice"}])
    result = ExtractEvaluator().evaluate(ir, tc)
    metrics = _metrics_by_name(result.metrics)

    assert "extract_list_unwrap_applied" not in metrics
    evidence = metrics["extract_evidence_value_pass_rate"]
    assert evidence.metadata.get("skipped_field_paths") == ["client_id"]
    assert evidence.metadata["total"] == 1
    assert evidence.value == 1.0


def test_list_unwrap_flattens_per_table_row_document_wrappers() -> None:
    tc = _make_test_case(
        expected_output={
            "personnel": [{"name": "Alice", "net_pay": 100}, {"name": "Bob", "net_pay": 200}],
            "client_id": "C-1",
        },
        test_rules=[
            _v02_rule("personnel[0].name", "Alice"),
            _v02_rule("personnel[0].net_pay", 100),
            _v02_rule("personnel[1].name", "Bob"),
            _v02_rule("personnel[1].net_pay", 200),
            _v02_rule("client_id", "C-1"),
        ],
    )
    ir = _make_inference_result_list(
        [
            {"client_id": "C-1", "personnel": [{"name": "Alice", "net_pay": 100}]},
            {"client_id": "C-1", "personnel": [{"name": "Bob", "net_pay": 200}]},
        ]
    )
    result = ExtractEvaluator().evaluate(ir, tc)
    metrics = _metrics_by_name(result.metrics)
    evidence = metrics["extract_evidence_value_pass_rate"]
    assert evidence.metadata.get("skipped_field_paths") == []
    assert evidence.metadata["total"] == 5
    assert evidence.value == 1.0


def test_list_unwrap_keeps_duplicate_wrapper_rows_as_precision_loss() -> None:
    tc = _make_test_case(
        expected_output={"personnel": [{"name": "Alice"}]},
        test_rules=[_v02_rule("personnel[0].name", "Alice")],
        data_schema={
            "type": "object",
            "properties": {
                "personnel": {
                    "type": "array",
                    "items": {"type": "object", "properties": {"name": {"type": "string"}}},
                }
            },
        },
    )
    ir = _make_inference_result_list(
        [
            {"personnel": [{"name": "Alice"}]},
            {"personnel": [{"name": "Alice"}]},
        ]
    )
    result = ExtractEvaluator().evaluate(ir, tc)
    metrics = _metrics_by_name(result.metrics)
    unified = metrics["extract_unified_value_precision"]
    assert unified.metadata["tp"] == 1
    assert unified.metadata["predicted_cells"] == 2
    assert unified.metadata["expected_cells"] == 1
    assert unified.value == 0.5
    assert metrics["extract_unified_value_recall"].value == 1.0


def test_list_unwrap_not_applied_on_dict_rooted_prediction() -> None:
    tc = _make_test_case(
        expected_output={"personnel": [{"name": "Alice"}], "client_id": "C-1"},
        test_rules=[
            _v02_rule("personnel[0].name", "Alice"),
            _v02_rule("client_id", "C-1"),
        ],
    )
    ir = _make_inference_result({"personnel": [{"name": "Alice"}], "client_id": "C-1"})
    result = ExtractEvaluator().evaluate(ir, tc)
    metrics = _metrics_by_name(result.metrics)
    evidence = metrics["extract_evidence_value_pass_rate"]
    assert evidence.metadata.get("skipped_field_paths") == []
    assert evidence.metadata["total"] == 2


def test_list_unwrap_ambiguous_multi_array_not_applied() -> None:
    tc = _make_test_case(
        expected_output={},
        test_rules=[
            _v02_rule("personnel[0].name", "Alice"),
            _v02_rule("transactions[0].amount", 50),
        ],
    )
    ir = _make_inference_result_list([{"name": "Alice"}])
    result = ExtractEvaluator().evaluate(ir, tc)
    metrics = _metrics_by_name(result.metrics)
    assert "extract_list_unwrap_applied" not in metrics
    assert metrics["extract_evidence_value_pass_rate"].metadata.get("skipped_field_paths") == []


def test_list_unwrap_scores_multi_array_wrapper_rows() -> None:
    tc = _make_test_case(
        expected_output={
            "account_number": "123",
            "checks_paid": [{"amount": 10}],
            "electronic_debits_bank_debits": [{"amount": 20}],
        },
        test_rules=[
            _v02_rule("account_number", "123"),
            _v02_rule("checks_paid[0].amount", 10),
            _v02_rule("electronic_debits_bank_debits[0].amount", 20),
        ],
    )
    ir = _make_inference_result_list(
        [
            {"account_number": "123", "checks_paid": [{"amount": 10}], "electronic_debits_bank_debits": []},
            {"account_number": "123", "checks_paid": [], "electronic_debits_bank_debits": [{"amount": 20}]},
        ]
    )
    result = ExtractEvaluator().evaluate(ir, tc)
    metrics = _metrics_by_name(result.metrics)
    evidence = metrics["extract_evidence_value_pass_rate"]
    assert evidence.metadata["total"] == 3
    assert evidence.value == 1.0


def test_list_unwrap_scores_singleton_scalar_document_list() -> None:
    tc = _make_test_case(
        expected_output={"tax_year": "2024", "employee_name": "Jane Doe"},
        test_rules=[
            _v02_rule("tax_year", "2024"),
            _v02_rule("employee_name", "Jane Doe"),
        ],
    )
    ir = _make_inference_result_list([{"tax_year": "2024", "employee_name": "Jane Doe"}])
    result = ExtractEvaluator().evaluate(ir, tc)
    metrics = _metrics_by_name(result.metrics)
    evidence = metrics["extract_evidence_value_pass_rate"]
    assert evidence.value == 1.0
    assert evidence.metadata["total"] == 2


def test_list_unwrap_skips_case_only_alias_rules_from_schema() -> None:
    tc = _make_test_case(
        expected_output={
            "Employee": [{"Name": "Alice"}],
            "employee": [{"Name": "Alice"}],
        },
        data_schema={"type": "object", "properties": {"Employee": {"type": "array"}}},
        test_rules=[
            _v02_rule("Employee[0].Name", "Alice"),
            _v02_rule("employee[0].Name", "Alice"),
        ],
    )
    ir = _make_inference_result_list([{"Employee": [{"Name": "Alice"}]}])
    result = ExtractEvaluator().evaluate(ir, tc)
    metrics = _metrics_by_name(result.metrics)
    evidence = metrics["extract_evidence_value_pass_rate"]
    assert evidence.metadata.get("skipped_field_paths") == ["employee[0].Name"]
    assert evidence.metadata["total"] == 1
    assert evidence.value == 1.0


def test_accuracy_uses_match_by_identity_keys_for_row_pairing() -> None:
    rows = [
        {"item_no": "0001", "description": "ALPHA", "amount": 100},
        {"item_no": "0002", "description": "BRAVO", "amount": 200},
        {"item_no": "0003", "description": "CHARLIE", "amount": 300},
    ]
    test_case = _make_test_case(
        expected_output={"line_items": rows},
        test_rules=[
            {
                "type": "extract_field",
                "field_path": "line_items",
                "structural": "match_by:item_no",
                "comparator": {"item_no": "case_insensitive", "amount": "number"},
                "evidence": [{"page": 1, "value": row, "coarse": True} for row in rows],
            }
        ],
    )
    reversed_result = ExtractEvaluator().evaluate(
        _make_inference_result({"line_items": list(reversed(rows))}), test_case
    )
    accuracy = _metrics_by_name(reversed_result.metrics)["accuracy"]
    assert accuracy.value == 1.0
    assert accuracy.metadata["identity_paired_paths"] == ["line_items"]

    dropped_result = ExtractEvaluator().evaluate(_make_inference_result({"line_items": rows[1:]}), test_case)
    assert abs(_metrics_by_name(dropped_result.metrics)["accuracy"].value - 2 / 3) < 1e-9


def test_accuracy_without_match_by_falls_back_to_assignment_pairing() -> None:
    rows = [
        {"description": "ALPHA", "amount": 100},
        {"description": "BRAVO", "amount": 200},
    ]
    test_case = _make_test_case(expected_output={"line_items": rows})
    result = ExtractEvaluator().evaluate(_make_inference_result({"line_items": list(reversed(rows))}), test_case)
    assert _metrics_by_name(result.metrics)["accuracy"].value == 1.0

    wrong_rows = [
        {"description": "CHARLIE", "amount": 700},
        {"description": "DELTA", "amount": 800},
    ]
    result = ExtractEvaluator().evaluate(_make_inference_result({"line_items": wrong_rows}), test_case)
    assert _metrics_by_name(result.metrics)["accuracy"].value < 1.0
