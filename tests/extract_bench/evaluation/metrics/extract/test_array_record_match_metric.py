from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from extract_bench.evaluation.evaluators.extract import ExtractEvaluator
from extract_bench.evaluation.metrics.extract.array_record_match_metric import (
    _UNHASHABLE,
    DEFAULT_FUZZY_FIELD_THRESHOLDS,
    ArrayRecordMatchMetric,
    _cell_key,
    _intern_field,
    array_item_properties,
    array_subfield_names,
    cell_match,
    compute_array_record_match_counts,
    is_array_schema,
    mismatch_cost_matrix,
)
from extract_bench.schemas.extract_output import ExtractOutput
from extract_bench.schemas.pipeline_io import InferenceRequest, InferenceResult
from extract_bench.schemas.product import ProductType
from extract_bench.test_cases.schema import ExtractTestCase


def _schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "account": {"type": ["string", "null"]},
            "transactions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "date": {"type": ["string", "null"]},
                        "description": {"type": ["string", "null"]},
                        "amount": {"type": "object"},
                    },
                },
            },
        },
    }


def _expected() -> dict:
    return {
        "account": "1234",
        "transactions": [
            {"date": "2026-01-01", "description": "Coffee Shop", "amount": {"amount": 4.5, "currency": "USD"}},
            {"date": "2026-01-02", "description": "Grocery Store", "amount": {"amount": 21.0, "currency": "USD"}},
        ],
    }


def _metric_by_name(actual: dict) -> dict[str, float]:
    values = ArrayRecordMatchMetric().compute(expected=_expected(), actual=actual, data_schema=_schema())
    return {metric.metric_name: metric.value for metric in values}


def test_array_subfield_names_reads_items_through_anyof() -> None:
    wrapped = {
        "anyOf": [
            {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "sku": {"type": "string"},
                        "qty": {"type": "integer"},
                    },
                },
            },
            {"type": "null"},
        ]
    }
    assert array_item_properties(wrapped) == {
        "sku": {"type": "string"},
        "qty": {"type": "integer"},
    }
    assert array_subfield_names(wrapped) == ["sku", "qty"]
    assert is_array_schema(wrapped) is True


def test_array_record_scores_ref_item_schema() -> None:
    """``items: {$ref: #/$defs/...}`` must still expand columns (same inliner as unified)."""
    schema = {
        "type": "object",
        "$defs": {
            "Line": {
                "type": "object",
                "properties": {
                    "sku": {"type": "string"},
                    "qty": {"type": "integer"},
                },
            }
        },
        "properties": {
            "lines": {
                "type": "array",
                "items": {"$ref": "#/$defs/Line"},
            }
        },
    }
    expected = {"lines": [{"sku": "A", "qty": 1}, {"sku": "B", "qty": 2}]}
    actual = {"lines": [{"sku": "B", "qty": 2}, {"sku": "A", "qty": 1}]}
    counts = compute_array_record_match_counts(expected=expected, actual=actual, data_schema=schema)
    assert counts is not None
    assert counts.correct == 4
    assert counts.expected_total == 4
    assert counts.predicted_total == 4


def test_is_array_schema_ignores_gold_and_reads_combinators() -> None:
    assert is_array_schema({}) is False
    assert is_array_schema({"type": "string"}) is False
    assert is_array_schema({"type": "array"}) is True
    # A list in gold does not make an untyped/string field an array.
    assert is_array_schema({}) is False


def test_array_record_match_is_order_insensitive() -> None:
    actual = {
        "account": "1234",
        "transactions": list(reversed(_expected()["transactions"])),
    }

    metrics = _metric_by_name(actual)

    assert metrics["array_record_accuracy"] == 1.0
    assert metrics["array_record_precision"] == 1.0
    assert metrics["array_record_f1"] == 1.0


def test_array_record_exact_peel_handles_large_shifted_array() -> None:
    rows = [{"id": str(i), "value": f"value-{i}"} for i in range(30_000)]
    expected = {"rows": rows}
    actual = {"rows": [{"id": "extra", "value": "extra"}, *rows]}
    schema = {
        "type": "object",
        "properties": {
            "rows": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": ["string", "null"]},
                        "value": {"type": ["string", "null"]},
                    },
                },
            }
        },
    }

    counts = compute_array_record_match_counts(expected=expected, actual=actual, data_schema=schema)

    assert counts is not None
    assert counts.correct == 60_000
    assert counts.expected_total == 60_000
    assert counts.predicted_total == 60_002


def test_array_record_accuracy_matches_extend_recall_denominator_for_extra_rows() -> None:
    actual = {
        "account": "1234",
        "transactions": [
            *_expected()["transactions"],
            {"date": "2026-01-03", "description": "Extra Row", "amount": {"amount": 99.0, "currency": "USD"}},
        ],
    }

    metrics = _metric_by_name(actual)

    assert metrics["array_record_accuracy"] == 1.0
    assert metrics["array_record_precision"] == pytest.approx(7 / 10)
    assert metrics["array_record_f1"] < 1.0
    assert metrics["array_record_row_count_ratio"] == 1.5


def test_array_record_ignores_reserved_provenance_key() -> None:
    # A reserved _provenance key (top-level object AND each array record) is source-
    # attribution metadata, never an extracted cell — it must not change any value
    # count, so a perfect prediction carrying it still scores 1.0.
    exp = _expected()
    perfect_with_prov = {
        **exp,
        "_provenance": {"page": 1},
        "transactions": [{**r, "_provenance": {"page": 2}} for r in exp["transactions"]],
    }
    metrics = _metric_by_name(perfect_with_prov)
    assert metrics["array_record_accuracy"] == 1.0
    assert metrics["array_record_precision"] == 1.0
    assert metrics["array_record_recall"] == 1.0
    assert metrics["array_record_f1"] == 1.0


def test_array_record_match_penalizes_missing_rows() -> None:
    actual = {
        "account": "1234",
        "transactions": [_expected()["transactions"][0]],
    }

    metrics = _metric_by_name(actual)

    assert metrics["array_record_accuracy"] == pytest.approx(4 / 7)
    assert metrics["array_record_recall"] == pytest.approx(4 / 7)
    assert metrics["array_record_precision"] == 1.0


def test_array_record_match_applies_configured_fuzzy_fields() -> None:
    actual = {
        "account": "1234",
        "transactions": [
            {"date": "2026-01-01", "description": "Coffee-Shop", "amount": {"amount": 4.5, "currency": "USD"}},
            {"date": "2026-01-02", "description": "Grocery Store", "amount": {"amount": 21.0, "currency": "USD"}},
        ],
    }

    metrics = _metric_by_name(actual)

    assert metrics["array_record_accuracy"] == 1.0


def test_array_record_fuzzy_fields_use_full_assignment() -> None:
    schema = {
        "type": "object",
        "properties": {
            "rows": {
                "type": "array",
                "items": {"type": "object", "properties": {"court": {"type": ["string", "null"]}}},
            }
        },
    }
    expected = {"rows": [{"court": "apple"}, {"court": "apples"}]}
    actual = {"rows": [{"court": "apple"}, {"court": "applez"}]}

    counts = compute_array_record_match_counts(expected=expected, actual=actual, data_schema=schema)

    assert counts is not None
    assert counts.correct == 2
    assert counts.expected_total == 2


def test_array_record_match_normalizes_dates() -> None:
    """Differing date formats for the same date must count as a match,
    aligned with json_subset_match's normalize_dates behavior."""
    actual = {
        "account": "1234",
        "transactions": [
            {"date": "Jan 1, 2026", "description": "Coffee Shop", "amount": {"amount": 4.5, "currency": "USD"}},
            {"date": "01/02/2026", "description": "Grocery Store", "amount": {"amount": 21.0, "currency": "USD"}},
        ],
    }

    metrics = _metric_by_name(actual)

    assert metrics["array_record_accuracy"] == 1.0
    assert metrics["array_record_precision"] == 1.0


def test_array_record_match_keeps_punctuation_spacing_significant_by_default() -> None:
    # The default cell equality is shared by every extract dataset, so
    # whitespace around punctuation stays significant here. Datasets that want
    # this leniency opt in per field via the 'punctuation_spacing' normalizer
    # (unified evidence metric).
    fuzzy = DEFAULT_FUZZY_FIELD_THRESHOLDS

    assert not cell_match("R. L. Underwood", "R.L.Underwood", "operator", fuzzy_field_thresholds=fuzzy)
    assert not cell_match("P.O. Box 978", "P.O.Box 978", "address", fuzzy_field_thresholds=fuzzy)
    assert not cell_match("Total Depth", "TotalDepth", "field", fuzzy_field_thresholds=fuzzy)
    assert not cell_match("1,000", "1000", "amount", fuzzy_field_thresholds=fuzzy)
    assert cell_match("R. L.\nUnderwood", "R. L. Underwood", "operator", fuzzy_field_thresholds=fuzzy)


def test_cell_match_zero_is_not_null() -> None:
    fuzzy: dict[str, float] = {}
    assert cell_match(0, None, "n", fuzzy_field_thresholds=fuzzy) is False
    assert cell_match(None, 0, "n", fuzzy_field_thresholds=fuzzy) is False
    assert cell_match(0, 0, "n", fuzzy_field_thresholds=fuzzy) is True
    assert cell_match(None, None, "n", fuzzy_field_thresholds=fuzzy) is True


def test_array_record_match_date_normalization_can_be_disabled() -> None:
    actual = {
        "account": "1234",
        "transactions": [
            {"date": "Jan 1, 2026", "description": "Coffee Shop", "amount": {"amount": 4.5, "currency": "USD"}},
            {"date": "2026-01-02", "description": "Grocery Store", "amount": {"amount": 21.0, "currency": "USD"}},
        ],
    }

    counts = compute_array_record_match_counts(
        expected=_expected(),
        actual=actual,
        data_schema=_schema(),
        normalize_dates=False,
    )

    assert counts is not None
    # The "Jan 1, 2026" cell no longer matches GT "2026-01-01" when
    # normalization is off; everything else still matches.
    assert counts.correct == counts.expected_total - 1


def test_array_record_match_does_not_normalize_id_like_strings() -> None:
    """Pure digit strings (account numbers, IDs) must pass through date
    normalization untouched and still compare exactly."""
    counts = compute_array_record_match_counts(
        expected=_expected(),
        actual=_expected(),
        data_schema=_schema(),
    )

    assert counts is not None
    assert counts.correct == counts.expected_total


def test_array_record_match_returns_none_when_no_top_level_array_records() -> None:
    counts = compute_array_record_match_counts(
        expected={"account": "1234"},
        actual={"account": "1234"},
        data_schema={"type": "object", "properties": {"account": {"type": "string"}}},
    )

    assert counts is None


def test_extract_evaluator_emits_array_record_metrics() -> None:
    test_case = ExtractTestCase(
        test_id="longarray/example",
        group="longarray",
        file_path=Path("example.pdf"),
        data_schema=_schema(),
        expected_output=_expected(),
    )
    inference_result = InferenceResult(
        request=InferenceRequest(
            example_id="example",
            source_file_path="example.pdf",
            product_type=ProductType.EXTRACT,
        ),
        pipeline_name="candidate",
        product_type=ProductType.EXTRACT,
        raw_output={},
        output=ExtractOutput(
            example_id="example",
            pipeline_name="candidate",
            extracted_data={
                "account": "1234",
                "transactions": list(reversed(_expected()["transactions"])),
            },
        ),
        started_at=datetime.now(),
        completed_at=datetime.now(),
        latency_in_ms=1,
    )

    result = ExtractEvaluator().evaluate(inference_result, test_case)
    metrics = {metric.metric_name: metric.value for metric in result.metrics}

    # Both metrics are order-invariant: reversed-but-correct rows score 1.0.
    assert metrics["accuracy"] == 1.0
    assert metrics["array_record_accuracy"] == 1.0


def _jsonable_intern_key(key: object) -> object:
    """Tuples -> lists so intern keys round-trip through ``json.dumps``."""
    if isinstance(key, tuple):
        return [_jsonable_intern_key(part) for part in key]
    return key


def test_cell_key_interns_json_containers_with_exact_nested_equality() -> None:
    """Opaque JSON arrays and objects intern; nested strings keep exact ``==``."""
    assert _cell_key(["Moscow", "Kyiv"]) is not _UNHASHABLE
    assert _cell_key(["Moscow", "Kyiv"]) == _cell_key(["Moscow", "Kyiv"])
    assert _cell_key(["Moscow", "Kyiv"]) != _cell_key(["Kyiv", "Moscow"])
    assert _cell_key(["Moscow "]) != _cell_key(["Moscow"])
    assert _cell_key("Moscow") != _cell_key(["Moscow"])
    assert _cell_key(["Moscow"]) != _cell_key(("Moscow",))

    assert _cell_key({"city": "Moscow"}) is not _UNHASHABLE
    assert _cell_key({"b": 1, "a": 2}) == _cell_key({"a": 2, "b": 1})
    assert _cell_key({"city": "Moscow "}) != _cell_key({"city": "Moscow"})
    assert _cell_key({}) != _cell_key([])
    assert _cell_key([{"city": "Moscow"}]) == _cell_key([{"city": "Moscow"}])
    assert _cell_key([{"city": "Moscow"}]) != _cell_key([{"city": "Kyiv"}])

    dumped = json.dumps(_jsonable_intern_key(_cell_key({"b": 1, "a": ["x", "y"]})))
    assert json.loads(dumped) == ["d", [[["v", "a"], ["l", [["v", "x"], ["v", "y"]]]], [["v", "b"], ["v", 1]]]]


def test_intern_field_list_column_matches_pairwise_mismatch_cost() -> None:
    actual = [{"addr": ["a", "b"]}, {"addr": ["c"]}]
    expected = [{"addr": ["c"]}, {"addr": ["a", "b"]}]
    interned = _intern_field(actual, expected, "addr")
    assert interned is not None
    cost = mismatch_cost_matrix(actual, expected, subfields=["addr"], fuzzy_field_thresholds={})
    assert cost[0, 1] == 0
    assert cost[1, 0] == 0
    assert cost[0, 0] == 1
    assert cost[1, 1] == 1


def test_intern_field_dict_column_matches_pairwise_mismatch_cost() -> None:
    actual = [{"amount": {"amount": 4.5, "currency": "USD"}}, {"amount": {"amount": 1.0, "currency": "EUR"}}]
    expected = [{"amount": {"currency": "EUR", "amount": 1.0}}, {"amount": {"currency": "USD", "amount": 4.5}}]
    interned = _intern_field(actual, expected, "amount")
    assert interned is not None
    cost = mismatch_cost_matrix(actual, expected, subfields=["amount"], fuzzy_field_thresholds={})
    assert cost[0, 1] == 0
    assert cost[1, 0] == 0
    assert cost[0, 0] == 1
    assert cost[1, 1] == 1


def test_array_record_precision_stays_bounded_when_null_scalars_are_omitted() -> None:
    """An omitted GT-null scalar matches as an implicit null, so it must also
    count as a prediction -- otherwise correct exceeds predicted_total and
    precision runs past 1.0 (omission would outscore an explicit null)."""
    schema = _schema()
    for name in ("note_a", "note_b", "note_c"):
        schema["properties"][name] = {"type": ["string", "null"]}
    expected = {**_expected(), "note_a": None, "note_b": None, "note_c": None}
    # Model omits the null scalars entirely and gets the account wrong.
    actual = {"account": "9999", "transactions": _expected()["transactions"]}

    counts = compute_array_record_match_counts(
        expected=expected, actual=actual, data_schema=schema, fuzzy_field_thresholds=None
    )

    assert counts is not None
    assert counts.predicted_total >= counts.correct
    # 4 scalars (account wrong, 3 implicit nulls correct) + 6 array cells.
    assert counts.predicted_total == 10
    assert counts.correct == 9
