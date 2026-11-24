"""Confidence summaries use the cells scored by Value F1."""

from pathlib import Path
from typing import Any

import pytest

from extract_bench.evaluation.grounded_confidence_payloads import field_citation_lookup, serialize_confidence_row
from extract_bench.evaluation.metrics.extract.confidence_scoped.scoring import build_confidence_field_rows
from extract_bench.evaluation.metrics.extract.confidence_scoped.summary import (
    compute_confidence_scoped_metrics,
    pooled_leaderboard_confidence,
)
from extract_bench.evaluation.metrics.extract.unified_evidence_metric import ValueCell, compute_unified_evidence_metrics
from extract_bench.evaluation.runner import EvaluationRunner
from extract_bench.schemas.evaluation import EvaluationResult, MetricValue, RunStat
from extract_bench.schemas.extract_output import FieldCitation

ROW_SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "address": {"type": "string"}},
}


def _confidence_metrics(
    expected: dict[str, Any],
    extracted: dict[str, Any],
    citations: list[FieldCitation],
    schema: dict[str, Any],
) -> dict[str, MetricValue]:
    cells: list[ValueCell] = []
    compute_unified_evidence_metrics(
        expected,
        extracted,
        [],
        data_schema=schema,
        value_cells=cells,
    )
    return {
        metric.metric_name: metric
        for metric in compute_confidence_scoped_metrics(
            extracted_data=extracted,
            expected_output=expected,
            field_rules=[],
            field_citations=citations,
            data_schema=schema,
            value_cells=cells,
        )
    }


def test_confidence_uses_the_array_verdict_and_counts_it_once() -> None:
    expected = {"record": {"amount": 10}, "labels": ["A", "B"]}
    extracted = {"record": {"amount": 9}, "labels": ["A", "B"]}
    schema = {
        "type": "object",
        "properties": {
            "record": {"type": "object", "properties": {"amount": {"type": "number"}}},
            "labels": {"type": "array", "items": {"type": "string"}},
        },
    }
    metrics = _confidence_metrics(
        expected,
        extracted,
        [
            FieldCitation(field_path=path, page=1, confidence=0.9)
            for path in ("record", "record.amount", "labels", "labels[0]", "labels[1]")
        ],
        schema,
    )

    carrier = metrics["confidence_scoped_auc"]
    assert {row["predicted_path"]: row["correct"] for row in carrier.metadata["field_rows_sample"]} == {
        "record.amount": False,
        "labels": True,
    }
    assert carrier.metadata["full_gt"]["expected_leaf_count"] == 2
    assert metrics["confidence_full_gt_recall"].value == 0.5
    assert metrics["confidence_signal_coverage"].value == 1.0


def test_empty_array_can_have_a_confidence_verdict() -> None:
    extracted = {"labels": []}
    metrics = _confidence_metrics(
        extracted,
        extracted,
        [FieldCitation(field_path="labels", page=1, confidence=0.9)],
        {"properties": {"labels": {"type": "array", "items": {"type": "string"}}}},
    )
    assert metrics["confidence_full_gt_recall"].value == 1.0


def test_confidence_rejoins_citations_on_the_original_wrapped_path() -> None:
    extracted = {"value": {"amount": 10}}
    metrics = _confidence_metrics(
        extracted,
        extracted,
        [FieldCitation(field_path="value.amount", page=1, confidence=0.9)],
        {"properties": {"value": {"properties": {"amount": {"type": "number"}}}}},
    )
    (row,) = metrics["confidence_scoped_auc"].metadata["field_rows_sample"]
    assert row["predicted_path"] == row["matched_gt_path"] == "value.amount"
    assert row["correct"]
    assert metrics["confidence_full_gt_recall"].value == 1.0
    assert metrics["confidence_signal_coverage"].value == 1.0


def test_object_confidence_covers_only_emitted_cells_with_their_own_verdicts() -> None:
    extracted = {"record": {"amount": 9, "note": None}}
    metrics = _confidence_metrics(
        {"record": {"amount": 10, "note": None, "missing": "required"}},
        extracted,
        [FieldCitation(field_path="record", page=1, confidence=0.8)],
        {
            "properties": {
                "record": {
                    "properties": {
                        "amount": {"type": "number"},
                        "note": {"type": "null"},
                        "missing": {"type": "string"},
                    },
                },
            },
        },
    )
    rows = metrics["confidence_scoped_auc"].metadata["field_rows_sample"]
    assert {row["predicted_path"]: (row["confidence"], row["correct"]) for row in rows} == {
        "record.amount": (0.8, False),
        "record.note": (0.8, True),
    }
    assert metrics["confidence_signal_coverage"].value == 1.0
    assert metrics["confidence_full_gt_recall"].value == pytest.approx(1 / 3)


def test_duplicate_rows_keep_value_f1_assignments_when_confidence_changes() -> None:
    repeated = {"name": "NAME ON FILE", "address": "ADDRESS ON FILE"}
    anchors = [{"name": "ALPHA", "address": "1 A STREET"}, {"name": "BETA", "address": "2 B STREET"}]
    expected = {"rows": [repeated] * 201 + anchors}
    extracted = {"rows": anchors[::-1] + [repeated] * 202}
    schema = {"properties": {"rows": {"type": "array", "items": ROW_SCHEMA}}}

    verdicts = []
    for scores in (range(len(extracted["rows"])), reversed(range(len(extracted["rows"])))):
        metrics = _confidence_metrics(
            expected,
            extracted,
            [
                FieldCitation(field_path=f"rows[{index}].{key}", page=1, confidence=score / len(extracted["rows"]))
                for index, score in enumerate(scores)
                for key in ("name", "address")
            ],
            schema,
        )
        metadata = metrics["confidence_scoped_auc"].metadata
        assert metadata["total_fields"] == 408
        assert metadata["correct_fields"] == 406
        assert metadata["full_gt"]["unique_leaf_covered"] == 406
        verdicts.append(
            {row["predicted_path"]: (row["matched_gt_path"], row["correct"]) for row in metadata["field_rows_sample"]}
        )
    assert verdicts[0] == verdicts[1]


def test_element_confidence_uses_minimum_on_the_aligned_wrapped_array() -> None:
    expected = {"value": {"rows": [{"id": "A", "labels": [1, 2, 3]}, {"id": "B", "labels": [4]}]}}
    extracted = {"value": {"rows": [{"id": "B", "labels": [4]}, {"id": "A", "labels": [1, 4, 3, 2]}]}}
    schema = {
        "properties": {
            "rows": {
                "type": "array",
                "items": {
                    "properties": {
                        "id": {"type": "string"},
                        "labels": {"type": "array", "items": {"type": "number"}},
                    },
                },
            },
        },
    }
    metrics = _confidence_metrics(
        expected,
        extracted,
        [
            FieldCitation(field_path=f"value.rows[{row}].labels[{index}]", page=1, confidence=score)
            for row, index, score in [(0, 0, 0.8), (1, 0, 0.99), (1, 1, 0.1), (1, 2, 0.98), (1, 3, 0.97)]
        ],
        schema,
    )
    rows = metrics["confidence_scoped_auc"].metadata["field_rows_sample"]
    assert {row["predicted_path"]: (row["matched_gt_path"], row["confidence"], row["correct"]) for row in rows} == {
        "value.rows[0].labels": ("value.rows[1].labels", 0.8, True),
        "value.rows[1].labels": ("value.rows[0].labels", 0.1, False),
    }
    assert metrics["confidence_signal_coverage"].value == 0.5
    assert metrics["confidence_signal_coverage"].metadata["scored_leaf_count"] == 2
    assert metrics["confidence_full_gt_recall"].value == 0.25


@pytest.mark.parametrize(
    ("native", "children", "expected_score"),
    [
        (0.95, [(0, 0.2), (1, 0.4)], 0.95),
        (0.0, [(0, 0.2), (1, 0.4)], 0.0),
        (None, [(0, 0.9), (0, 0.1), (1, 0.8)], 0.8),
        (float("nan"), [(0, 0.7), (1, None), (2, 0.01)], 0.7),
        (None, [(0, float("nan")), (1, None), (2, 0.01)], None),
    ],
    ids=["native-wins", "native-zero-wins", "duplicate-element", "partial-and-out-of-range", "no-valid-elements"],
)
def test_array_confidence_respects_native_scores_and_valid_emitted_elements(
    native: float | None,
    children: list[tuple[int, float | None]],
    expected_score: float | None,
) -> None:
    extracted = {"labels": ["A", "B"]}
    citations = [FieldCitation(field_path="labels", page=1, confidence=native)] + [
        FieldCitation(field_path=f"labels[{index}]", page=1, confidence=score) for index, score in children
    ]
    metrics = _confidence_metrics(
        extracted,
        extracted,
        citations,
        {"properties": {"labels": {"type": "array", "items": {"type": "string"}}}},
    )
    rows = metrics["confidence_scoped_auc"].metadata["field_rows_sample"]
    assert [(row["predicted_path"], row["confidence"]) for row in rows] == (
        [] if expected_score is None else [("labels", expected_score)]
    )
    assert metrics["confidence_signal_coverage"].value == float(expected_score is not None)


@pytest.mark.parametrize("wrapped", [False, True], ids=["plain", "value-wrapper"])
def test_nearest_object_confidence_preserves_child_and_array_scores(wrapped: bool) -> None:
    extracted = {
        "record": {
            "address": {"city": "Boston", "zip": "02101"},
            "labels": ["A", "B"],
            "rows": [{"name": "first"}, {"name": "second"}],
        },
    }
    schema = {
        "properties": {
            "record": {
                "properties": {
                    "address": {"properties": {"city": {"type": "string"}, "zip": {"type": "string"}}},
                    "labels": {"type": "array", "items": {"type": "string"}},
                    "rows": {"type": "array", "items": {"properties": {"name": {"type": "string"}}}},
                },
            },
        },
    }
    prefix = "value." if wrapped else ""
    if wrapped:
        extracted = {"value": extracted}
    metrics = _confidence_metrics(
        extracted,
        extracted,
        [
            FieldCitation(field_path=prefix + path, page=1, confidence=score)
            for path, score in [
                ("record", 0.8),
                ("record.address", 0.4),
                ("record.address.city", 0.0),
                ("record.labels[0]", 0.9),
                ("record.labels[1]", 0.2),
                ("record.rows", 0.99),
                ("record.rows[0]", 0.3),
                ("record.rows[1].name", float("nan")),
            ]
        ],
        schema,
    )
    rows = metrics["confidence_scoped_auc"].metadata["field_rows_sample"]
    assert {row["predicted_path"].removeprefix(prefix): row["confidence"] for row in rows} == {
        "record.address.city": 0.0,
        "record.address.zip": 0.4,
        "record.labels": 0.2,
        "record.rows[0].name": 0.3,
        "record.rows[1].name": 0.8,
    }
    assert metrics["confidence_signal_coverage"].value == 1.0


@pytest.mark.parametrize("direct_evidence", [False, True], ids=["no-array-citation", "unscored-array-citation"])
def test_derived_array_confidence_keeps_evidence_on_the_predicted_path(direct_evidence: bool) -> None:
    expected = {"rows": [{"id": "A", "labels": [1]}, {"id": "B", "labels": [2]}]}
    extracted = {"rows": list(reversed(expected["rows"]))}
    schema = {
        "properties": {
            "rows": {
                "type": "array",
                "items": {
                    "properties": {
                        "id": {"type": "string"},
                        "labels": {"type": "array", "items": {"type": "number"}},
                    },
                },
            },
        },
    }
    citations = [
        FieldCitation(field_path="rows[0].labels[0]", page=1, confidence=0.8, reference_text="Element text"),
        FieldCitation(field_path="rows[1].labels", page=2, confidence=0.6, reference_text="Other prediction"),
    ]
    if direct_evidence:
        citations.append(FieldCitation(field_path="rows[0].labels", page=3, reference_text="Own array text"))
    original = [citation.model_dump() for citation in citations]
    cells: list[ValueCell] = []
    compute_unified_evidence_metrics(expected, extracted, [], data_schema=schema, value_cells=cells)
    rows = build_confidence_field_rows(
        extracted_data=extracted,
        expected_output=expected,
        field_rules=[],
        field_citations=citations,
        data_schema=schema,
        value_cells=cells,
    )
    row = next(row for row in rows if row.predicted_path == "rows[0].labels")
    assert row.matched_gt_path == "rows[1].labels"
    serialized = serialize_confidence_row("doc", 0, row, field_citation_lookup(citations))
    assert serialized["confidence"] == 0.8
    assert serialized["citation"] == (citations[-1].model_dump() if direct_evidence else None)
    assert serialized["cited_text"] == ("Own array text" if direct_evidence else "")
    assert [citation.model_dump() for citation in citations] == original


def test_wrapper_object_confidence_keeps_child_evidence() -> None:
    extracted = {"value": {"amount": 10, "name": "A"}}
    schema = {"properties": {"amount": {"type": "number"}, "name": {"type": "string"}}}
    citations = [
        FieldCitation(field_path="value", page=1, confidence=0.8, reference_text="Parent-only text"),
        FieldCitation(field_path="value.amount", page=2, reference_text="10"),
        FieldCitation(field_path="value.name", page=2, confidence=0.0, reference_text="A"),
    ]
    cells: list[ValueCell] = []
    compute_unified_evidence_metrics(extracted, extracted, [], data_schema=schema, value_cells=cells)
    rows = build_confidence_field_rows(
        extracted_data=extracted,
        expected_output=extracted,
        field_rules=[],
        field_citations=citations,
        data_schema=schema,
        value_cells=cells,
    )
    assert {row.predicted_path: row.confidence for row in rows} == {"value.amount": 0.8, "value.name": 0.0}
    row = next(row for row in rows if row.predicted_path == "value.amount")
    serialized = serialize_confidence_row("doc", 0, row, field_citation_lookup(citations, "value."))
    assert serialized["citation"] == citations[1].model_dump()
    assert serialized["cited_text"] == "10"
    metrics = _confidence_metrics(extracted, extracted, citations, schema)
    assert metrics["confidence_signal_coverage"].value == 1.0


@pytest.mark.parametrize("array", [False, True], ids=["duplicate-scalar", "array-elements"])
def test_coverage_retains_semantics_of_all_contributing_confidence_signals(array: bool) -> None:
    extracted = {"answer": [1, 2] if array else 1}
    citations = [
        FieldCitation(
            field_path=f"answer[{index}]" if array else "answer",
            page=1,
            confidence=score,
            metadata={"confidence_semantics": semantics},
        )
        for index, (score, semantics) in enumerate([(0.6, "categorical_verification"), (0.8, "ordinal_1_5")])
    ]
    field_schema = {"type": "array", "items": {"type": "number"}} if array else {"type": "number"}
    metrics = _confidence_metrics(extracted, extracted, citations, {"properties": {"answer": field_schema}})
    assert metrics["confidence_signal_coverage"].metadata["confidence_semantics"] == [
        "categorical_verification",
        "ordinal_1_5",
    ]
    (row,) = metrics["confidence_scoped_auc"].metadata["field_rows_sample"]
    assert row["confidence"] == (0.6 if array else 0.8)


def test_inherited_confidence_keeps_semantics_from_equivalent_wrapped_paths() -> None:
    extracted = {"value": {"record": {"amount": 10}}}
    metrics = _confidence_metrics(
        extracted,
        extracted,
        [
            FieldCitation(field_path=path, page=1, confidence=score, metadata={"confidence_semantics": semantics})
            for path, score, semantics in [
                ("record", 0.8, "ordinal_1_5"),
                ("value.record", 0.6, "categorical_verification"),
            ]
        ],
        {"properties": {"record": {"properties": {"amount": {"type": "number"}}}}},
    )
    assert metrics["confidence_signal_coverage"].metadata["confidence_semantics"] == [
        "categorical_verification",
        "ordinal_1_5",
    ]
    (row,) = metrics["confidence_scoped_auc"].metadata["field_rows_sample"]
    assert row["confidence"] == 0.8


def test_leaderboard_pools_fields_and_ties_unscored_with_zero(tmp_path: Path) -> None:
    docs = [
        _confidence_metrics(
            {"name": "A", "address": "X"},
            {"name": "A", "address": "wrong"},
            [FieldCitation(field_path="address", page=1, confidence=0.0)],
            ROW_SCHEMA,
        ),
        _confidence_metrics(
            {"name": "B"},
            {"name": "B"},
            [FieldCitation(field_path="name", page=1, confidence=1.0)],
            ROW_SCHEMA,
        ),
    ]
    results = [
        EvaluationResult(
            test_id=str(i),
            example_id=str(i),
            pipeline_name="test",
            product_type="extract",
            success=True,
            metrics=list(metrics.values()),
            stats=[RunStat(name="latency_ms", value=1_800_000, unit="ms")],
        )
        for i, metrics in enumerate(docs)
    ]
    pooled = EvaluationRunner(tmp_path)._aggregate_metrics(results)
    assert pooled["Confidence_Coverage"] == pytest.approx(200 / 3)
    # One high correct field and a bottom tie (one error, one correct).
    assert pooled["Confidence_AUGRC"] == pytest.approx((0 + 0.5 + 1) / 9)

    # Late results leave both pooled metrics.
    # The document exactly at the cutoff still contributes its scored correct cell.
    results[0].stats = [RunStat(name="latency_ms", value=1_800_001, unit="ms")]
    pooled = EvaluationRunner(tmp_path)._aggregate_metrics(results)
    assert pooled["Confidence_Coverage"] == 100.0
    assert pooled["Confidence_AUGRC"] == 0.0


def test_omitted_blank_gets_rank_only_in_represented_record() -> None:
    schema = {"type": "object", "properties": {"rows": {"type": "array", "items": ROW_SCHEMA}}}
    metric = _confidence_metrics(
        {"rows": [{"name": "A", "address": None}, {"name": "B", "address": None}]},
        {"rows": [{"name": "A", "address": "wrong"}, {"name": "B"}]},
        [],
        schema,
    )["confidence_signal_coverage"]
    # The omitted blank adds a fourth ranked cell. Only the wrong address is an error.
    assert metric.metadata["emitted_leaf_count"] == 3
    assert pooled_leaderboard_confidence([(metric, 1000)]) == {"Confidence_Coverage": 0.0, "Confidence_AUGRC": 2.5 / 16}
    dropped = _confidence_metrics(
        {"rows": [{"name": "A", "address": None}, {"name": "B", "address": None}]},
        {"rows": [{"name": "A", "address": "wrong"}]},
        [],
        schema,
    )["confidence_signal_coverage"]
    assert pooled_leaderboard_confidence([(dropped, 1000)])["Confidence_AUGRC"] == 1.5 / 4


def test_partial_rank_metadata_cannot_publish_partial_augrc() -> None:
    current = _confidence_metrics(
        {"name": "A"},
        {"name": "A"},
        [FieldCitation(field_path="name", page=1, confidence=0.5)],
        ROW_SCHEMA,
    )["confidence_signal_coverage"]
    historical = MetricValue(
        metric_name="confidence_signal_coverage",
        value=0.0,
        metadata={"coverage_scope": "emitted_value_f1_cells", "emitted_leaf_count": 2, "scored_leaf_count": 0},
    )
    assert pooled_leaderboard_confidence([(current, 1000), (historical, 1000)]) == {"Confidence_Coverage": 100 / 3}
    assert pooled_leaderboard_confidence([(current, 1000), (historical, None)]) == {}
    assert pooled_leaderboard_confidence([(current, 1_800_001)]) == {}
