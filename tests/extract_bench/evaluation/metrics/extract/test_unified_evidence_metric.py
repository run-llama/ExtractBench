"""Tests for the unified evidence metric (array_record + OR-values + grounding)."""

from __future__ import annotations

from typing import Any

from extract_bench.evaluation.metrics.extract import unified_evidence_metric
from extract_bench.evaluation.metrics.extract.array_record_match_metric import (
    ArrayRecordMatchMetric,
)
from extract_bench.evaluation.metrics.extract.unified_evidence_metric import (
    build_rule_indexes,
    compute_unified_evidence_metrics,
    index_citations,
    iou_xywh,
    lookup,
    path_leaf,
)
from extract_bench.test_cases.schema import ExtractFieldTestRule, FieldEvidence


def test_path_leaf_strips_parent_and_trailing_index() -> None:
    assert path_leaf("account") == "account"
    assert path_leaf("holdings[0].security") == "security"
    assert path_leaf("entities[0].aliases[1].name") == "name"
    assert path_leaf("rows[2]") == "rows"


def _schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "as_of": {"type": ["string", "null"]},
            "holdings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "security": {"type": ["string", "null"]},
                        "coupon": {"type": ["number", "null"]},
                        "note": {"type": ["string", "null"]},
                    },
                },
            },
        },
    }


def _rule(
    path: str,
    *values: Any,
    page: int | None = None,
    bbox: list[float] | None = None,
    normalizers: list[str] | None = None,
) -> ExtractFieldTestRule:
    ev = [FieldEvidence(value=v, page=page, bbox=bbox) for v in values] or [FieldEvidence(value=None)]
    return ExtractFieldTestRule(field_path=path, evidence=ev, normalizers=normalizers or [])


def _leaf_rules_single(expected: dict[str, Any]) -> list[ExtractFieldTestRule]:
    """One single-evidence rule per leaf, value == expected (no alt, no bbox)."""
    rules: list[ExtractFieldTestRule] = [_rule("as_of", expected.get("as_of"))]
    for i, row in enumerate(expected["holdings"]):
        for key in ("security", "coupon", "note"):
            rules.append(_rule(f"holdings[{i}].{key}", row.get(key)))
    return rules


def _val(metrics: list[Any], name: str) -> float | None:
    return next((m.value for m in metrics if m.metric_name == name), None)


# --------------------------------------------------------------- reduction
def test_value_metrics_equal_array_record_on_single_evidence() -> None:
    expected = {
        "as_of": "2024-01-01",
        "holdings": [
            {"security": "AAA", "coupon": 5.0, "note": None},
            {"security": "BBB", "coupon": 6.0, "note": "x"},
        ],
    }
    actual = {  # one cell wrong, one row reordered -> exercises Hungarian + a miss
        "as_of": "2024-01-01",
        "holdings": [
            {"security": "BBB", "coupon": 6.0, "note": "x"},
            {"security": "AAA", "coupon": 9.9, "note": None},
        ],
    }
    arr = ArrayRecordMatchMetric(normalize_dates=True).compute(expected=expected, actual=actual, data_schema=_schema())
    uni = compute_unified_evidence_metrics(expected, actual, _leaf_rules_single(expected), [], _schema())
    assert _val(uni, "extract_unified_value_f1") == _val(arr, "array_record_f1")
    assert _val(uni, "extract_unified_value_recall") == _val(arr, "array_record_recall")
    assert _val(uni, "extract_unified_value_precision") == _val(arr, "array_record_precision")


def test_reserved_provenance_key_is_not_scored() -> None:
    # A reserved _provenance key (top-level + per record) is attribution metadata, not
    # an extracted cell. A perfect prediction carrying it must still score 1.0 — the
    # value metric must not count it in either the precision or recall denominator.
    expected = {
        "as_of": "2024-01-01",
        "holdings": [
            {"security": "AAA", "coupon": 5.0, "note": None},
            {"security": "BBB", "coupon": 6.0, "note": "x"},
        ],
    }
    with_prov = {
        **expected,
        "_provenance": {"page": 1},
        "holdings": [{**r, "_provenance": {"page": 2 + i}} for i, r in enumerate(expected["holdings"])],
    }
    uni = compute_unified_evidence_metrics(expected, with_prov, _leaf_rules_single(expected), [], _schema())
    assert _val(uni, "extract_unified_value_precision") == 1.0
    assert _val(uni, "extract_unified_value_recall") == 1.0
    assert _val(uni, "extract_unified_value_f1") == 1.0


# ------------------------------------------------------------ OR-acceptance
def test_or_acceptable_alternate_value_passes() -> None:
    expected = {"as_of": None, "holdings": [{"security": "Acme Inc.", "coupon": 5.0, "note": None}]}
    actual = {"as_of": None, "holdings": [{"security": "Acme Incorporated", "coupon": 5.0, "note": None}]}
    # array_record sees a mismatch on `security`; the unified metric accepts the
    # alternate evidence value, so its recall is strictly higher.
    rules = _leaf_rules_single(expected)
    rules.append(_rule("holdings[0].security", "Acme Inc.", "Acme Incorporated"))
    arr = ArrayRecordMatchMetric().compute(expected=expected, actual=actual, data_schema=_schema())
    uni = compute_unified_evidence_metrics(expected, actual, rules, [], _schema())
    assert _val(uni, "extract_unified_value_recall") > _val(arr, "array_record_recall")
    assert _val(uni, "extract_unified_value_recall") == 1.0


def test_alternate_evidence_values_use_full_assignment() -> None:
    schema = {
        "type": "object",
        "properties": {
            "rows": {
                "type": "array",
                "items": {"type": "object", "properties": {"name": {"type": ["string", "null"]}}},
            }
        },
    }
    expected = {"rows": [{"name": "apple"}, {"name": "apples"}]}
    actual = {"rows": [{"name": "apple"}, {"name": "applez"}]}
    rules = [
        _rule("rows[0].name", "apple", "applez"),
        _rule("rows[1].name", "apples", "apple"),
    ]

    uni = compute_unified_evidence_metrics(expected, actual, rules, [], schema)

    assert _val(uni, "extract_unified_value_recall") == 1.0


def test_optional_terminal_punctuation_normalizer_is_field_scoped() -> None:
    expected = {"as_of": "1093' feet"}
    actual = {"as_of": "1093' feet."}
    schema = {"type": "object", "properties": {"as_of": {"type": ["string", "null"]}}}

    strict = compute_unified_evidence_metrics(expected, actual, [_rule("as_of", "1093' feet")], [], schema)
    normalized = compute_unified_evidence_metrics(
        expected,
        actual,
        [_rule("as_of", "1093' feet", normalizers=["optional_terminal_punctuation"])],
        [],
        schema,
    )

    assert _val(strict, "extract_unified_value_f1") == 0.0
    assert _val(normalized, "extract_unified_value_f1") == 1.0


def test_optional_terminal_punctuation_normalizer_does_not_drop_internal_punctuation() -> None:
    expected = {"as_of": "1,000 feet"}
    actual = {"as_of": "1000 feet."}
    schema = {"type": "object", "properties": {"as_of": {"type": ["string", "null"]}}}

    metrics = compute_unified_evidence_metrics(
        expected,
        actual,
        [_rule("as_of", "1,000 feet", normalizers=["optional_terminal_punctuation"])],
        [],
        schema,
    )

    assert _val(metrics, "extract_unified_value_f1") == 0.0


def test_optional_terminal_punctuation_normalizer_participates_in_array_alignment() -> None:
    schema = {
        "type": "object",
        "properties": {
            "rows": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "depth": {"type": ["string", "null"]},
                        "name": {"type": ["string", "null"]},
                    },
                },
            }
        },
    }
    expected = {"rows": [{"depth": "1093' feet", "name": "A"}, {"depth": "1073 feet", "name": "B"}]}
    actual = {"rows": [{"depth": "1073 feet.", "name": "B"}, {"depth": "1093' feet.", "name": "A"}]}
    rules = [
        _rule("rows[0].depth", "1093' feet", normalizers=["optional_terminal_punctuation"]),
        _rule("rows[0].name", "A"),
        _rule("rows[1].depth", "1073 feet", normalizers=["optional_terminal_punctuation"]),
        _rule("rows[1].name", "B"),
    ]

    metrics = compute_unified_evidence_metrics(expected, actual, rules, [], schema)

    assert _val(metrics, "extract_unified_value_f1") == 1.0


# -------------------------------------------------------- truncation / order
def test_truncation_penalized_no_vacuous_null_pass() -> None:
    # 3 GT rows, 2 of which have a null `note`. Prediction returns only 1 row.
    # The dropped rows' null cells must NOT pass: recall ~= 1/3, not inflated.
    expected = {
        "as_of": None,
        "holdings": [
            {"security": "AAA", "coupon": 1.0, "note": None},
            {"security": "BBB", "coupon": 2.0, "note": None},
            {"security": "CCC", "coupon": 3.0, "note": None},
        ],
    }
    actual = {"as_of": None, "holdings": [{"security": "AAA", "coupon": 1.0, "note": None}]}
    uni = compute_unified_evidence_metrics(expected, actual, _leaf_rules_single(expected), [], _schema())
    recall = _val(uni, "extract_unified_value_recall")
    assert recall is not None and 0.30 <= recall <= 0.40  # ~ 1 of 3 rows, not vacuously high


def test_reordered_rows_pass_without_match_by() -> None:
    expected = {
        "as_of": None,
        "holdings": [{"security": s, "coupon": float(i), "note": None} for i, s in enumerate("ABCDE")],
    }
    actual = {"as_of": None, "holdings": list(reversed(expected["holdings"]))}
    uni = compute_unified_evidence_metrics(expected, actual, _leaf_rules_single(expected), [], _schema())
    assert _val(uni, "extract_unified_value_recall") == 1.0  # Hungarian re-aligns; no match_by rule used


def test_unified_exact_peel_handles_large_shifted_array() -> None:
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

    uni = compute_unified_evidence_metrics(expected, actual, [], [], schema)

    assert _val(uni, "extract_unified_value_recall") == 1.0
    assert _val(uni, "extract_unified_value_precision") == 60_000 / 60_002


def test_over_extraction_lowers_precision() -> None:
    expected = {"as_of": None, "holdings": [{"security": "AAA", "coupon": 1.0, "note": None}]}
    actual = {
        "as_of": None,
        "holdings": [
            {"security": "AAA", "coupon": 1.0, "note": None},
            {"security": "ZZZ", "coupon": 9.0, "note": "junk"},
        ],
    }
    uni = compute_unified_evidence_metrics(expected, actual, _leaf_rules_single(expected), [], _schema())
    assert _val(uni, "extract_unified_value_recall") == 1.0
    assert _val(uni, "extract_unified_value_precision") < 1.0  # extra row penalized


def test_bare_object_recurses_to_child_cells() -> None:
    schema = {
        "type": "object",
        "properties": {
            "vendor": {
                "type": "object",
                "properties": {
                    "name": {"type": ["string", "null"]},
                    "city": {"type": ["string", "null"]},
                },
            }
        },
    }
    expected = {"vendor": {"name": "Acme", "city": "Austin"}}
    actual = {"vendor": {"name": "Acme", "city": "Dallas"}}
    rules = [_rule("vendor.name", "Acme"), _rule("vendor.city", "Austin")]
    uni = compute_unified_evidence_metrics(expected, actual, rules, [], schema)
    # Two cells, one miss: recall 0.5, precision 0.5 (not one opaque 0).
    assert _val(uni, "extract_unified_value_recall") == 0.5
    assert _val(uni, "extract_unified_value_precision") == 0.5


def test_omitted_object_is_null_children_not_one_miss() -> None:
    schema = {
        "type": "object",
        "properties": {
            "vendor": {
                "anyOf": [
                    {
                        "type": "object",
                        "properties": {
                            "name": {"type": ["string", "null"]},
                            "city": {"type": ["string", "null"]},
                        },
                    },
                    {"type": "null"},
                ]
            }
        },
    }
    expected = {"vendor": {"name": "Acme", "city": "Austin"}}
    actual: dict[str, Any] = {}
    rules = [_rule("vendor.name", "Acme"), _rule("vendor.city", "Austin")]
    uni = compute_unified_evidence_metrics(expected, actual, rules, [], schema)
    assert _val(uni, "extract_unified_value_recall") == 0.0
    # No field ``default`` → omit is not an implicit null prediction.
    explicit_nulls = compute_unified_evidence_metrics(
        expected, {"vendor": {"name": None, "city": None}}, rules, [], schema
    )
    omitted_meta = next(m.metadata for m in uni if m.metric_name == "extract_unified_value_recall")
    explicit_meta = next(m.metadata for m in explicit_nulls if m.metric_name == "extract_unified_value_recall")
    assert omitted_meta["expected_cells"] == 2
    assert omitted_meta["predicted_cells"] == 0
    assert explicit_meta["predicted_cells"] == 2


def test_omitted_object_uses_schema_field_defaults() -> None:
    schema = {
        "type": "object",
        "$defs": {
            "Vendor": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "default": "Acme"},
                    "city": {"type": "string", "default": None},
                },
            }
        },
        "properties": {
            "vendor": {"anyOf": [{"$ref": "#/$defs/Vendor"}, {"type": "null"}], "default": None},
        },
    }
    expected = {"vendor": {"name": "Acme", "city": "Austin"}}
    actual: dict[str, Any] = {}
    rules = [_rule("vendor.name", "Acme"), _rule("vendor.city", "Austin")]
    uni = compute_unified_evidence_metrics(expected, actual, rules, [], schema)
    # name matches the explicit schema default; city default is null ≠ Austin.
    assert _val(uni, "extract_unified_value_recall") == 0.5
    assert _val(uni, "extract_unified_value_precision") == 0.5
    meta = next(m.metadata for m in uni if m.metric_name == "extract_unified_value_recall")
    assert meta["expected_cells"] == 2
    assert meta["predicted_cells"] == 2


def test_null_object_matches_omitted_and_explicit_null() -> None:
    schema = {
        "type": "object",
        "properties": {"vendor": {"type": ["object", "null"], "default": None}},
    }
    expected = {"vendor": None}
    rules = [_rule("vendor", None)]
    omitted = compute_unified_evidence_metrics(expected, {}, rules, [], schema)
    explicit = compute_unified_evidence_metrics(expected, {"vendor": None}, rules, [], schema)
    assert _val(omitted, "extract_unified_value_f1") == 1.0
    assert _val(explicit, "extract_unified_value_f1") == 1.0


def test_nested_object_scalar_array_is_one_opaque_cell() -> None:
    # Schema-typed primitive arrays on nested objects are one opaque cell (same
    # as a string field), not a Hungarian table and not a skip. Disagreeing
    # lists are 0/0; matching lists are 1/1. Per-element scoring of ["a","b"] vs
    # ["a","c"] would be 0.5, so these bounds also pin "one cell".
    schema = {
        "type": "object",
        "properties": {
            "vendor": {
                "type": "object",
                "properties": {
                    "tags": {
                        "type": "array",
                        "items": {"type": ["string", "null"]},
                    }
                },
            }
        },
    }
    expected = {"vendor": {"tags": ["a", "b"]}}
    rules = [_rule("vendor.tags", ["a", "b"])]
    miss = compute_unified_evidence_metrics(expected, {"vendor": {"tags": ["a", "c"]}}, rules, [], schema)
    hit = compute_unified_evidence_metrics(expected, expected, rules, [], schema)
    assert _val(miss, "extract_unified_value_recall") == 0.0
    assert _val(miss, "extract_unified_value_precision") == 0.0
    assert _val(hit, "extract_unified_value_recall") == 1.0
    assert _val(hit, "extract_unified_value_precision") == 1.0


def test_object_inside_array_row_recurses_to_child_cells() -> None:
    schema = {
        "type": "object",
        "properties": {
            "rows": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": ["string", "null"]},
                        "addr": {
                            "type": "object",
                            "properties": {
                                "city": {"type": ["string", "null"]},
                                "zip": {"type": ["string", "null"]},
                            },
                        },
                    },
                },
            }
        },
    }
    expected = {"rows": [{"id": "1", "addr": {"city": "A", "zip": "1"}}]}
    actual = {"rows": [{"id": "1", "addr": {"city": "A", "zip": "2"}}]}
    rules = [
        _rule("rows[0].id", "1"),
        _rule("rows[0].addr.city", "A"),
        _rule("rows[0].addr.zip", "1"),
    ]
    uni = compute_unified_evidence_metrics(expected, actual, rules, [], schema)
    arr = ArrayRecordMatchMetric().compute(expected=expected, actual=actual, data_schema=schema)
    # id + city match, zip misses: 2/3. array_record still treats addr as one cell (1/2).
    assert _val(uni, "extract_unified_value_recall") == 2 / 3
    assert _val(uni, "extract_unified_value_precision") == 2 / 3
    assert _val(arr, "array_record_recall") == 0.5
    assert _val(uni, "extract_unified_value_recall") > _val(arr, "array_record_recall")


# ----------------------------------------------------- nested object arrays
def test_object_array_subfield_gets_per_child_credit() -> None:
    # `tags` is a list of objects. array_record scores it as one
    # opaque cell (whole list mismatches -> 1 of 2 subfields correct -> 0.5);
    # the unified metric recurses, so only tags[1].name is wrong -> 4 of 5
    # cells -> 0.8. This is the depth array_record cannot give.
    schema = {
        "type": "object",
        "properties": {
            "holdings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "security": {"type": ["string", "null"]},
                        "tags": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": ["string", "null"]},
                                    "kind": {"type": ["string", "null"]},
                                },
                            },
                        },
                    },
                },
            }
        },
    }
    expected = {"holdings": [{"security": "AAA", "tags": [{"name": "x", "kind": "a"}, {"name": "y", "kind": "b"}]}]}
    actual = {"holdings": [{"security": "AAA", "tags": [{"name": "x", "kind": "a"}, {"name": "WRONG", "kind": "b"}]}]}
    rules = [
        _rule("holdings[0].security", "AAA"),
        _rule("holdings[0].tags[0].name", "x"),
        _rule("holdings[0].tags[0].kind", "a"),
        _rule("holdings[0].tags[1].name", "y"),
        _rule("holdings[0].tags[1].kind", "b"),
    ]
    arr = ArrayRecordMatchMetric().compute(expected=expected, actual=actual, data_schema=schema)
    uni = compute_unified_evidence_metrics(expected, actual, rules, [], schema)
    assert _val(arr, "array_record_recall") == 0.5  # tags is one opaque cell that mismatches
    assert _val(uni, "extract_unified_value_recall") == 0.8  # recursed: only tags[1].name wrong
    assert _val(uni, "extract_unified_value_recall") > _val(arr, "array_record_recall")


def test_empty_gt_object_array_expands_predicted_children() -> None:
    """Empty (or null) gold object-array still expands invented predicted rows.

    Schema says ``equipment_adjustments`` is an array of objects. Gold is ``[]``;
    the prediction invents two rows / four child values. Those must be four
    precision misses, not one opaque cell.
    """
    schema = {
        "type": "object",
        "properties": {
            "vehicles": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "year": {"type": ["string", "null"]},
                        "equipment_adjustments": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": ["string", "null"]},
                                    "amount": {"type": ["number", "null"]},
                                },
                            },
                        },
                    },
                },
            }
        },
    }
    expected_empty = {"vehicles": [{"year": "2020", "equipment_adjustments": []}]}
    expected_null = {"vehicles": [{"year": "2020", "equipment_adjustments": None}]}
    actual = {
        "vehicles": [
            {
                "year": "2020",
                "equipment_adjustments": [
                    {"name": "A", "amount": 1},
                    {"name": "B", "amount": 2},
                ],
            }
        ]
    }
    for expected in (expected_empty, expected_null):
        uni = compute_unified_evidence_metrics(expected, actual, [], [], schema)
        assert _val(uni, "extract_unified_value_recall") == 1.0
        assert _val(uni, "extract_unified_value_precision") == 0.2


def test_gold_list_does_not_make_a_string_field_an_array() -> None:
    """Eval shape comes from schema, not the gold value's Python type."""
    schema = {"type": "object", "properties": {"note": {"type": ["string", "null"]}}}
    expected = {"note": [{"text": "a"}, {"text": "b"}]}
    actual = {"note": [{"text": "a"}, {"text": "b"}]}
    uni = compute_unified_evidence_metrics(expected, actual, [], [], schema)
    # One opaque string-field cell (the lists compare equal), not 2 Hungarian rows.
    assert _val(uni, "extract_unified_value_recall") == 1.0
    assert _val(uni, "extract_unified_value_precision") == 1.0


def test_gold_dict_does_not_make_a_string_field_an_object() -> None:
    schema = {"type": "object", "properties": {"note": {"type": ["string", "null"]}}}
    expected = {"note": {"text": "a"}}
    actual = {"note": {"text": "b"}}
    uni = compute_unified_evidence_metrics(expected, actual, [], [], schema)
    # One opaque miss, not a child-cell recurse into ``text``.
    assert _val(uni, "extract_unified_value_recall") == 0.0
    assert _val(uni, "extract_unified_value_precision") == 0.0


def test_object_wrapping_array_of_records_with_nested_object_arrays() -> None:
    """Root object → child object → array of records → nested object-array.

    Inner service rows are reordered; Hungarian at that depth still pairs them.
    One paid miss: 6 leaves, 5 correct.
    """
    schema = {
        "type": "object",
        "properties": {
            "packet": {
                "type": "object",
                "properties": {
                    "title": {"type": ["string", "null"]},
                    "claims": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "claim_id": {"type": ["string", "null"]},
                                "services": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "code": {"type": ["string", "null"]},
                                            "paid": {"type": ["number", "null"]},
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            }
        },
    }
    expected = {
        "packet": {
            "title": "EOB",
            "claims": [
                {
                    "claim_id": "C1",
                    "services": [{"code": "A", "paid": 1.0}, {"code": "B", "paid": 2.0}],
                }
            ],
        }
    }
    actual = {
        "packet": {
            "title": "EOB",
            "claims": [
                {
                    "claim_id": "C1",
                    "services": [{"code": "B", "paid": 2.0}, {"code": "A", "paid": 9.0}],
                }
            ],
        }
    }
    rules = [
        _rule("packet.title", "EOB"),
        _rule("packet.claims[0].claim_id", "C1"),
        _rule("packet.claims[0].services[0].code", "A"),
        _rule("packet.claims[0].services[0].paid", 1.0),
        _rule("packet.claims[0].services[1].code", "B"),
        _rule("packet.claims[0].services[1].paid", 2.0),
    ]
    uni = compute_unified_evidence_metrics(expected, actual, rules, [], schema)
    arr = ArrayRecordMatchMetric().compute(expected=expected, actual=actual, data_schema=schema)
    assert _val(uni, "extract_unified_value_recall") == 5 / 6
    assert _val(uni, "extract_unified_value_precision") == 5 / 6
    # No top-level array, so array_record does not emit. Unified still walks the
    # nested claims/services arrays.
    assert arr == []


def test_array_records_with_nested_object_and_nested_object_array() -> None:
    """Array of records whose columns mix a nested object and a nested object-array."""
    schema = {
        "type": "object",
        "properties": {
            "rows": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": ["string", "null"]},
                        "addr": {
                            "type": "object",
                            "properties": {
                                "city": {"type": ["string", "null"]},
                                "geo": {
                                    "type": "object",
                                    "properties": {
                                        "lat": {"type": ["string", "null"]},
                                        "lon": {"type": ["string", "null"]},
                                    },
                                },
                            },
                        },
                        "tags": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {"name": {"type": ["string", "null"]}},
                            },
                        },
                    },
                },
            }
        },
    }
    expected = {
        "rows": [
            {
                "id": "1",
                "addr": {"city": "A", "geo": {"lat": "1", "lon": "2"}},
                "tags": [{"name": "x"}, {"name": "y"}],
            }
        ]
    }
    actual = {
        "rows": [
            {
                "id": "1",
                "addr": {"city": "A", "geo": {"lat": "1", "lon": "WRONG"}},
                "tags": [{"name": "y"}, {"name": "x"}],
            }
        ]
    }
    rules = [
        _rule("rows[0].id", "1"),
        _rule("rows[0].addr.city", "A"),
        _rule("rows[0].addr.geo.lat", "1"),
        _rule("rows[0].addr.geo.lon", "2"),
        _rule("rows[0].tags[0].name", "x"),
        _rule("rows[0].tags[1].name", "y"),
    ]
    uni = compute_unified_evidence_metrics(expected, actual, rules, [], schema)
    arr = ArrayRecordMatchMetric().compute(expected=expected, actual=actual, data_schema=schema)
    # id, city, lat match; lon misses; tags Hungarian-match both names.
    assert _val(uni, "extract_unified_value_recall") == 5 / 6
    assert _val(uni, "extract_unified_value_precision") == 5 / 6
    # array_record: id matches; addr and tags are opaque mismatches → 1/3.
    assert _val(arr, "array_record_recall") == 1 / 3
    assert _val(uni, "extract_unified_value_recall") > _val(arr, "array_record_recall")


def test_hungarian_pairs_rows_by_nested_object_children() -> None:
    """Same opaque scalars, nested objects that only partially overlap.

    Whole-dict pairing cost is 1 for every gold/pred pair (no issuer dict is
    identical), so Hungarian can follow list order and score children under the
    wrong partner: 2 scalar hits, 0 nested hits → 2/6. Child-level cost prefers
    the crossed pairing (each row matches city, misses zip) → 4/6.
    """
    schema = {
        "type": "object",
        "properties": {
            "holdings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "class": {"type": ["string", "null"]},
                        "issuer": {
                            "type": "object",
                            "properties": {
                                "city": {"type": ["string", "null"]},
                                "zip": {"type": ["string", "null"]},
                            },
                        },
                    },
                },
            }
        },
    }
    expected = {
        "holdings": [
            {"class": "equity", "issuer": {"city": "NYC", "zip": "10001"}},
            {"class": "equity", "issuer": {"city": "SF", "zip": "94105"}},
        ]
    }
    actual = {
        "holdings": [
            {"class": "equity", "issuer": {"city": "SF", "zip": "00000"}},
            {"class": "equity", "issuer": {"city": "NYC", "zip": "00000"}},
        ]
    }
    uni = compute_unified_evidence_metrics(expected, actual, [], [], schema)
    assert _val(uni, "extract_unified_value_recall") == 4 / 6
    assert _val(uni, "extract_unified_value_precision") == 4 / 6
    arr = ArrayRecordMatchMetric().compute(expected=expected, actual=actual, data_schema=schema)
    # array_record still treats issuer as one opaque cell: list-order pairing,
    # class matches, issuer dicts all miss → 2/4.
    assert _val(arr, "array_record_recall") == 2 / 4


def test_pairing_walks_schema_keys_that_contain_dots() -> None:
    """A key named ``a.b`` is one field, not a nested ``a`` then ``b``.

    Same opaque scalar, nested objects that only overlap on that dotted key.
    Splitting the path on ``.`` would read missing children and list-order pair
    (2/6). Segment walks pair on ``a.b`` (4/6).
    """
    schema = {
        "type": "object",
        "properties": {
            "holdings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "class": {"type": ["string", "null"]},
                        "issuer": {
                            "type": "object",
                            "properties": {
                                "a.b": {"type": ["string", "null"]},
                                "zip": {"type": ["string", "null"]},
                            },
                        },
                    },
                },
            }
        },
    }
    expected = {
        "holdings": [
            {"class": "equity", "issuer": {"a.b": "NYC", "zip": "10001"}},
            {"class": "equity", "issuer": {"a.b": "SF", "zip": "94105"}},
        ]
    }
    actual = {
        "holdings": [
            {"class": "equity", "issuer": {"a.b": "SF", "zip": "00000"}},
            {"class": "equity", "issuer": {"a.b": "NYC", "zip": "00000"}},
        ]
    }
    uni = compute_unified_evidence_metrics(expected, actual, [], [], schema)
    assert _val(uni, "extract_unified_value_recall") == 4 / 6
    assert _val(uni, "extract_unified_value_precision") == 4 / 6


def test_nested_object_array_alts_participate_in_parent_pairing() -> None:
    """Alts on nested object-array leaves must affect parent Hungarian cost.

    Parent scalars tie. Without alts, list-order nested cost is worse than the
    crossed pairing, so the solver swaps and gold[1] misses applez (3/4). With
    alts in the nested cost, list order is free and both names match (4/4).
    """
    schema = {
        "type": "object",
        "properties": {
            "rows": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "cls": {"type": ["string", "null"]},
                        "loc": {
                            "type": "object",
                            "properties": {
                                "aliases": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {"name": {"type": ["string", "null"]}},
                                    },
                                }
                            },
                        },
                    },
                },
            }
        },
    }
    expected = {
        "rows": [
            {"cls": "x", "loc": {"aliases": [{"name": "apple"}]}},
            {"cls": "x", "loc": {"aliases": [{"name": "apples"}]}},
        ]
    }
    actual = {
        "rows": [
            {"cls": "x", "loc": {"aliases": [{"name": "applez"}]}},
            {"cls": "x", "loc": {"aliases": [{"name": "apple"}]}},
        ]
    }
    rules = [
        _rule("rows[0].cls", "x"),
        _rule("rows[0].loc.aliases[0].name", "apple", "applez"),
        _rule("rows[1].cls", "x"),
        _rule("rows[1].loc.aliases[0].name", "apples", "apple"),
    ]
    uni = compute_unified_evidence_metrics(expected, actual, rules, [], schema)
    assert _val(uni, "extract_unified_value_recall") == 1.0
    assert _val(uni, "extract_unified_value_precision") == 1.0


# ------------------------------------------------------------- grounding
def _bbox_rules(expected: dict[str, Any], page: int, bbox: list[float]) -> list[ExtractFieldTestRule]:
    rules = [_rule("as_of", expected.get("as_of"))]
    for i, row in enumerate(expected["holdings"]):
        for key in ("security", "coupon", "note"):
            rules.append(_rule(f"holdings[{i}].{key}", row.get(key), page=page, bbox=bbox))
    return rules


def test_grounded_requires_matching_bbox() -> None:
    expected = {"as_of": None, "holdings": [{"security": "AAA", "coupon": 1.0, "note": "n"}]}
    actual = {"as_of": None, "holdings": [{"security": "AAA", "coupon": 1.0, "note": "n"}]}
    gt_box = [0.1, 0.1, 0.2, 0.05]
    rules = _bbox_rules(expected, page=1, bbox=gt_box)
    # Citation on the right page/box -> grounded passes.
    good_cit = [{"field_path": f"holdings[0].{k}", "page": 1, "bbox": gt_box} for k in ("security", "coupon", "note")]
    uni_good = compute_unified_evidence_metrics(expected, actual, rules, good_cit, _schema())
    assert _val(uni_good, "extract_unified_grounded_recall") == 1.0
    # Citation on the wrong page -> value still right, grounded fails.
    bad_cit = [{"field_path": f"holdings[0].{k}", "page": 2, "bbox": gt_box} for k in ("security", "coupon", "note")]
    uni_bad = compute_unified_evidence_metrics(expected, actual, rules, bad_cit, _schema())
    assert _val(uni_bad, "extract_unified_value_recall") == 1.0
    assert _val(uni_bad, "extract_unified_grounded_recall") == 0.0


def test_indexers_emit_validated_xywh_tuples() -> None:
    box = [0.1, 0.2, 0.3, 0.4, 99.0]
    rules = [_rule("as_of", "x", page=1, bbox=box)]
    _, ev_boxes, _, *_ = build_rule_indexes(rules)
    pred_boxes, _ = index_citations([{"field_path": "as_of", "page": 1, "bbox": box}])
    assert ev_boxes["as_of"] == [(1, (0.1, 0.2, 0.3, 0.4))]
    assert pred_boxes["as_of"] == [(1, (0.1, 0.2, 0.3, 0.4))]


def test_iou_xywh_clamps_identical_non_dyadic_boxes_to_one() -> None:
    box = (0.1, 0.2, 0.3, 0.4)
    assert iou_xywh(box, box) == 1.0
    assert iou_xywh(box, (0.9, 0.9, 0.05, 0.05)) == 0.0


def test_iou_xywh_identical_malformed_boxes_are_not_perfect_hits() -> None:
    nan = float("nan")
    inf = float("inf")
    assert iou_xywh((nan, nan, nan, nan), (nan, nan, nan, nan)) == 0.0
    assert iou_xywh((0.0, 0.0, inf, inf), (0.0, 0.0, inf, inf)) == 0.0
    assert iou_xywh((0.1, 0.2, 0.0, 0.4), (0.1, 0.2, 0.0, 0.4)) == 0.0
    assert iou_xywh((0.1, 0.2, 0.3, 0.0), (0.1, 0.2, 0.3, 0.0)) == 0.0


def test_indexers_drop_non_finite_or_non_positive_bboxes() -> None:
    nan = float("nan")
    inf = float("inf")
    rules = [
        _rule("as_of", "x", page=1, bbox=[0.1, 0.2, nan, 0.4]),
        _rule("holdings[0].security", "AAA", page=1, bbox=[0.1, 0.2, inf, 0.4]),
        _rule("holdings[0].coupon", 1.0, page=1, bbox=[0.1, 0.2, 0.0, 0.4]),
        _rule("holdings[0].note", "n", page=1, bbox=[0.1, 0.2, 0.3, -0.1]),
    ]
    _, ev_boxes, _, *_ = build_rule_indexes(rules)
    pred_boxes, _ = index_citations(
        [
            {"field_path": "as_of", "page": 1, "bbox": [0.1, 0.2, nan, 0.4]},
            {"field_path": "holdings[0].security", "page": 1, "bbox": [0.1, 0.2, inf, 0.4]},
            {"field_path": "holdings[0].coupon", "page": 1, "bbox": [0.1, 0.2, 0.0, 0.4]},
            {"field_path": "holdings[0].note", "page": 1, "bbox": [0.1, 0.2, 0.3, -0.1]},
        ]
    )
    assert ev_boxes == {}
    assert pred_boxes == {}


def test_no_citations_yields_zero_grounded() -> None:
    # GT carries bboxes (grounding IS applicable) but the prediction emits no
    # citation: a real grounding miss -> grounded F1 is emitted as 0.0.
    expected = {"as_of": None, "holdings": [{"security": "AAA", "coupon": 1.0, "note": "n"}]}
    actual = {"as_of": None, "holdings": [{"security": "AAA", "coupon": 1.0, "note": "n"}]}
    rules = _bbox_rules(expected, page=1, bbox=[0.1, 0.1, 0.2, 0.05])
    uni = compute_unified_evidence_metrics(expected, actual, rules, [], _schema())
    assert _val(uni, "extract_unified_value_f1") == 1.0
    assert _val(uni, "extract_unified_grounded_f1") == 0.0


def test_no_gt_bbox_omits_grounded_metrics() -> None:
    # When the ground truth carries NO evidence bbox, grounding is undefined:
    # the *_grounded_* metrics are omitted entirely (so the runner excludes the
    # document from the grounded average) even if the prediction emits boxes.
    # The *_value_* metrics are unaffected. This is the fix for a dataset whose
    # GT lacks bboxes scoring a misleading ~0 grounded F1 instead of being
    # excluded.
    expected = {"as_of": None, "holdings": [{"security": "AAA", "coupon": 1.0, "note": "n"}]}
    actual = {"as_of": None, "holdings": [{"security": "AAA", "coupon": 1.0, "note": "n"}]}
    rules = _leaf_rules_single(expected)  # no bbox on any GT evidence
    cits = [
        {"field_path": f"holdings[0].{k}", "page": 1, "bbox": [0.1, 0.1, 0.2, 0.05]}
        for k in ("security", "coupon", "note")
    ]
    uni = compute_unified_evidence_metrics(expected, actual, rules, cits, _schema())
    assert _val(uni, "extract_unified_value_f1") == 1.0
    assert _val(uni, "extract_unified_grounded_f1") is None
    assert _val(uni, "extract_unified_grounded_precision") is None
    assert _val(uni, "extract_unified_grounded_recall") is None


def test_sparse_gt_bbox_does_not_punish_unannotated_claims() -> None:
    # Sparsely annotated GT: only ONE cell carries an evidence bbox while the
    # rest are value-only (the sec_13f shape: one cover field annotated, 30k
    # value-only cells). A pipeline that cites EVERY cell must not have its
    # grounded precision divided by all those ungradeable claims -- only the
    # claim on the bbox-bearing cell is gradeable. Correct grounding there
    # means grounded P/R/F1 == 1.0.
    expected = {"as_of": "2024-01-01", "holdings": [{"security": "AAA", "coupon": 1.0, "note": "n"}]}
    actual = {"as_of": "2024-01-01", "holdings": [{"security": "AAA", "coupon": 1.0, "note": "n"}]}
    gt_box = [0.1, 0.1, 0.2, 0.05]
    rules = _leaf_rules_single(expected)  # value-only everywhere...
    rules.append(_rule("as_of", "2024-01-01", page=1, bbox=gt_box))  # ...except the one cover scalar
    cits = [{"field_path": "as_of", "page": 1, "bbox": gt_box}] + [
        {"field_path": f"holdings[0].{k}", "page": 1, "bbox": [0.5, 0.5, 0.1, 0.05]}
        for k in ("security", "coupon", "note")
    ]
    uni = compute_unified_evidence_metrics(expected, actual, rules, cits, _schema())
    assert _val(uni, "extract_unified_grounded_recall") == 1.0
    assert _val(uni, "extract_unified_grounded_precision") == 1.0  # 3 unannotated claims excluded, not counted wrong
    assert _val(uni, "extract_unified_grounded_f1") == 1.0
    meta = next(m.metadata for m in uni if m.metric_name == "extract_unified_grounded_f1")
    assert meta["grounded_expected_cells"] == 1
    assert meta["grounded_pred_claims"] == 1


def test_extra_predicted_rows_claims_stay_out_of_grounded_precision() -> None:
    # An extra predicted row has no GT counterpart, so its citation bboxes are
    # ungradeable: they must not enter the grounded precision denominator (they
    # still hurt VALUE precision as always).
    expected = {"as_of": None, "holdings": [{"security": "AAA", "coupon": 1.0, "note": "n"}]}
    actual = {
        "as_of": None,
        "holdings": [{"security": "AAA", "coupon": 1.0, "note": "n"}, {"security": "ZZZ", "coupon": 9.0, "note": "z"}],
    }
    gt_box = [0.1, 0.1, 0.2, 0.05]
    rules = _bbox_rules(expected, page=1, bbox=gt_box)
    cits = [{"field_path": f"holdings[{j}].{k}", "page": 1, "bbox": gt_box} for j in (0, 1) for k in ("security",)]
    uni = compute_unified_evidence_metrics(expected, actual, rules, cits, _schema())
    meta = next(m.metadata for m in uni if m.metric_name == "extract_unified_grounded_f1")
    assert meta["grounded_pred_claims"] == 1  # only the claim aligned to GT row 0's bbox-bearing cell
    assert _val(uni, "extract_unified_grounded_precision") == 1.0
    assert _val(uni, "extract_unified_value_precision") < 1.0  # the extra row still costs value precision


# --------------------------------------------------------------- page grounding
def _page_rules(expected: dict[str, Any], page: int) -> list[ExtractFieldTestRule]:
    """One rule per leaf carrying a page but NO bbox (page-only grounding)."""
    rules = [_rule("as_of", expected.get("as_of"))]
    for i, row in enumerate(expected["holdings"]):
        for key in ("security", "coupon", "note"):
            rules.append(_rule(f"holdings[{i}].{key}", row.get(key), page=page))
    return rules


def test_page_correct_but_bbox_wrong_nests_between_value_and_grounded() -> None:
    # The core of the page family: a citation on the RIGHT page but with a
    # non-overlapping bbox is value-correct and page-correct, yet bbox-wrong.
    # So value_f1 == page_f1 == 1.0 while grounded_f1 == 0.0, and the three
    # nest: value_f1 >= page_f1 >= grounded_f1.
    expected = {"as_of": None, "holdings": [{"security": "AAA", "coupon": 1.0, "note": "n"}]}
    actual = {"as_of": None, "holdings": [{"security": "AAA", "coupon": 1.0, "note": "n"}]}
    rules = _bbox_rules(expected, page=1, bbox=[0.1, 0.1, 0.2, 0.05])
    cits = [
        {"field_path": f"holdings[0].{k}", "page": 1, "bbox": [0.8, 0.8, 0.1, 0.05]}  # right page, disjoint box
        for k in ("security", "coupon", "note")
    ]
    uni = compute_unified_evidence_metrics(expected, actual, rules, cits, _schema())
    vf1 = _val(uni, "extract_unified_value_f1")
    pf1 = _val(uni, "extract_unified_page_f1")
    gf1 = _val(uni, "extract_unified_grounded_f1")
    assert vf1 == 1.0
    assert pf1 == 1.0
    assert gf1 == 0.0
    assert vf1 >= pf1 >= gf1


def test_page_only_gt_emits_page_metrics_but_omits_grounded() -> None:
    # GT evidence carries a page but no bbox: the page family is defined (and
    # passes) while the bbox/grounded family is undefined and omitted -- page is
    # the coarser, more widely-applicable signal.
    expected = {"as_of": None, "holdings": [{"security": "AAA", "coupon": 1.0, "note": "n"}]}
    actual = {"as_of": None, "holdings": [{"security": "AAA", "coupon": 1.0, "note": "n"}]}
    rules = _page_rules(expected, page=3)
    cits = [{"field_path": f"holdings[0].{k}", "page": 3} for k in ("security", "coupon", "note")]  # page-only cites
    uni = compute_unified_evidence_metrics(expected, actual, rules, cits, _schema())
    assert _val(uni, "extract_unified_page_recall") == 1.0
    assert _val(uni, "extract_unified_page_precision") == 1.0
    assert _val(uni, "extract_unified_page_f1") == 1.0
    assert _val(uni, "extract_unified_grounded_f1") is None  # no GT bbox -> grounded undefined
    meta = next(m.metadata for m in uni if m.metric_name == "extract_unified_page_f1")
    assert meta["page_expected_cells"] == 3
    assert meta["page_pred_claims"] == 3


def test_wrong_page_fails_page_metric() -> None:
    # Value right, but the citation names a page the GT evidence never claims:
    # page_recall drops to 0 while value_recall stays 1.
    expected = {"as_of": None, "holdings": [{"security": "AAA", "coupon": 1.0, "note": "n"}]}
    actual = {"as_of": None, "holdings": [{"security": "AAA", "coupon": 1.0, "note": "n"}]}
    rules = _page_rules(expected, page=1)
    cits = [{"field_path": f"holdings[0].{k}", "page": 7} for k in ("security", "coupon", "note")]
    uni = compute_unified_evidence_metrics(expected, actual, rules, cits, _schema())
    assert _val(uni, "extract_unified_value_recall") == 1.0
    assert _val(uni, "extract_unified_page_recall") == 0.0


def test_no_gt_page_omits_page_metrics() -> None:
    # When the GT carries NO page anywhere, page grounding is undefined: the
    # *_page_* metrics are omitted (excluded from the dataset average), mirroring
    # how *_grounded_* is omitted when the GT carries no bbox.
    expected = {"as_of": None, "holdings": [{"security": "AAA", "coupon": 1.0, "note": "n"}]}
    actual = {"as_of": None, "holdings": [{"security": "AAA", "coupon": 1.0, "note": "n"}]}
    rules = _leaf_rules_single(expected)  # no page, no bbox on any GT evidence
    cits = [{"field_path": f"holdings[0].{k}", "page": 1} for k in ("security", "coupon", "note")]
    uni = compute_unified_evidence_metrics(expected, actual, rules, cits, _schema())
    assert _val(uni, "extract_unified_value_f1") == 1.0
    assert _val(uni, "extract_unified_page_f1") is None
    assert _val(uni, "extract_unified_page_precision") is None
    assert _val(uni, "extract_unified_page_recall") is None


def test_nested_grounding_survives_outer_array_reorder() -> None:
    # Regression for the gt/pred path bug: when the outer object-array is
    # reordered, nested object-array citations must be looked up at the *matched
    # predicted* index, not the GT index. With correct citations on the actual
    # prediction paths, grounded recall must stay 1.0.
    schema = {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": ["string", "null"]},
                        "aliases": {
                            "type": "array",
                            "items": {"type": "object", "properties": {"name": {"type": ["string", "null"]}}},
                        },
                    },
                },
            }
        },
    }
    box_a = [0.1, 0.1, 0.2, 0.05]
    box_b = [0.5, 0.5, 0.2, 0.05]
    expected = {
        "entities": [
            {"id": "E1", "aliases": [{"name": "alpha"}]},
            {"id": "E2", "aliases": [{"name": "beta"}]},
        ]
    }
    actual = {"entities": list(reversed(expected["entities"]))}  # E2 first, E1 second
    rules = [
        ExtractFieldTestRule(field_path="entities[0].id", evidence=[FieldEvidence(value="E1", page=1, bbox=box_a)]),
        ExtractFieldTestRule(
            field_path="entities[0].aliases[0].name", evidence=[FieldEvidence(value="alpha", page=1, bbox=box_a)]
        ),
        ExtractFieldTestRule(field_path="entities[1].id", evidence=[FieldEvidence(value="E2", page=1, bbox=box_b)]),
        ExtractFieldTestRule(
            field_path="entities[1].aliases[0].name", evidence=[FieldEvidence(value="beta", page=1, bbox=box_b)]
        ),
    ]
    # Citations sit at the PREDICTED paths: entities[0]=E2 -> box_b, entities[1]=E1 -> box_a.
    cits = [
        {"field_path": "entities[0].id", "page": 1, "bbox": box_b},
        {"field_path": "entities[0].aliases[0].name", "page": 1, "bbox": box_b},
        {"field_path": "entities[1].id", "page": 1, "bbox": box_a},
        {"field_path": "entities[1].aliases[0].name", "page": 1, "bbox": box_a},
    ]
    uni = compute_unified_evidence_metrics(expected, actual, rules, cits, schema)
    assert _val(uni, "extract_unified_value_recall") == 1.0
    assert _val(uni, "extract_unified_grounded_recall") == 1.0  # would be 0.5 with the gt/pred path bug


# --------------------------------------------------------------- degenerate
def test_non_dict_inputs_return_empty() -> None:
    assert compute_unified_evidence_metrics(["a"], {"x": 1}, [], [], {}) == []
    assert compute_unified_evidence_metrics({"x": 1}, "nope", [], [], {}) == []


# ------------------------------------------------------- evaluator wiring
def test_extract_evaluator_emits_unified_metrics() -> None:
    """End-to-end: the metric flows through ExtractEvaluator and matches array_record."""
    from datetime import datetime
    from pathlib import Path

    from extract_bench.evaluation.evaluators.extract import ExtractEvaluator
    from extract_bench.schemas.extract_output import ExtractOutput, FieldCitation
    from extract_bench.schemas.pipeline_io import InferenceRequest, InferenceResult
    from extract_bench.schemas.product import ProductType
    from extract_bench.test_cases.schema import ExtractTestCase

    expected = {"as_of": None, "holdings": [{"security": "AAA", "coupon": 1.0, "note": "n"}]}
    box = [0.1, 0.1, 0.2, 0.05]
    test_case = ExtractTestCase(
        test_id="longarray/example",
        group="longarray",
        file_path=Path("example.pdf"),
        data_schema=_schema(),
        expected_output=expected,
        test_rules=[r.model_dump() for r in _bbox_rules(expected, page=1, bbox=box)],
    )
    inference_result = InferenceResult(
        request=InferenceRequest(example_id="ex", source_file_path="example.pdf", product_type=ProductType.EXTRACT),
        pipeline_name="candidate",
        product_type=ProductType.EXTRACT,
        raw_output={},
        output=ExtractOutput(
            example_id="ex",
            pipeline_name="candidate",
            extracted_data=expected,
            field_citations=[
                FieldCitation(field_path=f"holdings[0].{k}", page=1, bbox=box) for k in ("security", "coupon", "note")
            ],
        ),
        started_at=datetime.now(),
        completed_at=datetime.now(),
        latency_in_ms=1,
    )
    metrics = {m.metric_name: m.value for m in ExtractEvaluator().evaluate(inference_result, test_case).metrics}
    assert metrics["extract_unified_value_f1"] == metrics["array_record_f1"] == 1.0
    assert metrics["extract_unified_grounded_recall"] == 1.0  # citations match the evidence boxes


# ----------------------------------------- exact-row peel is pairing-safe only
# The exact-row peel preserves array_record's value score (pairing-independent:
# correct == n_pairs*k - total_cost). But the unified score is *pairing-sensitive*
# whenever it recurses into nested object arrays or scores per-cell grounding, since
# both key off the matched predicted index. There the peel could select a
# different (equal-cost) pairing than the full assignment and shift grounded /
# nested true positives, so it must fall back to full assignment.
_FLAT_PEEL_SCHEMA: dict[str, Any] = {
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
_NESTED_PEEL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "rows": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": ["string", "null"]},
                    "kids": {
                        "type": "array",
                        "items": {"type": "object", "properties": {"v": {"type": ["string", "null"]}}},
                    },
                },
            },
        }
    },
}


def _spy_on_peel(monkeypatch: Any) -> list[list[str]]:
    """Record the cost-subfields each exact-row peel call aligned on."""
    calls: list[list[str]] = []
    original = unified_evidence_metric.peel_exact_row_matches

    def _spy(
        act_rows: Any,
        exp_rows: Any,
        *,
        subfields: Any,
        fuzzy_field_thresholds: Any,
        field_schemas: Any = None,
    ) -> Any:
        calls.append(list(subfields))
        return original(
            act_rows,
            exp_rows,
            subfields=subfields,
            fuzzy_field_thresholds=fuzzy_field_thresholds,
            field_schemas=field_schemas,
        )

    monkeypatch.setattr(unified_evidence_metric, "peel_exact_row_matches", _spy)
    return calls


def test_exact_peel_used_for_flat_ungrounded_arrays(monkeypatch: Any) -> None:
    calls = _spy_on_peel(monkeypatch)
    rows = [{"id": "1", "value": "a"}, {"id": "2", "value": "b"}]
    compute_unified_evidence_metrics({"rows": rows}, {"rows": rows}, [], [], _FLAT_PEEL_SCHEMA)
    assert ["id", "value"] in calls, "flat un-grounded arrays must keep the fast exact-row peel"


def test_exact_peel_skipped_for_object_array_subfields(monkeypatch: Any) -> None:
    calls = _spy_on_peel(monkeypatch)
    rows = [{"id": "A", "kids": [{"v": "x"}]}, {"id": "B", "kids": [{"v": "y"}]}]
    compute_unified_evidence_metrics({"rows": rows}, {"rows": rows}, [], [], _NESTED_PEEL_SCHEMA)
    # The outer array (identity cell "id") has a nested object array, so its
    # alignment must use full assignment -- the peel must not see ["id"].
    assert ["id"] not in calls, "outer array with object-array subfields must use full assignment"
    # The inner flat "kids" arrays are opaque-cell-only, so they still peel.
    assert ["v"] in calls, "flat inner sub-arrays should still use the fast peel"


def test_exact_peel_skipped_when_cell_grounding_present(monkeypatch: Any) -> None:
    calls = _spy_on_peel(monkeypatch)
    schema = {
        "type": "object",
        "properties": {
            "rows": {
                "type": "array",
                "items": {"type": "object", "properties": {"id": {"type": ["string", "null"]}}},
            }
        },
    }
    rows = [{"id": "A"}, {"id": "B"}]
    box = [0.0, 0.0, 10.0, 10.0]
    rules = [
        _rule("rows[0].id", "A", page=1, bbox=box),
        _rule("rows[1].id", "B", page=1, bbox=box),
    ]
    citations = [
        {"field_path": "rows[0].id", "page": 1, "bbox": box},
        {"field_path": "rows[1].id", "page": 1, "bbox": box},
    ]
    compute_unified_evidence_metrics({"rows": rows}, {"rows": rows}, rules, citations, schema)
    assert calls == [], "per-cell grounding makes scoring pairing-sensitive; must use full assignment"


def test_object_array_full_assignment_scores_perfect_match() -> None:
    # Sanity: the full-assignment fallback still scores a perfect nested match.
    rows = [{"id": "A", "kids": [{"v": "x"}]}, {"id": "B", "kids": [{"v": "y"}]}]
    uni = compute_unified_evidence_metrics({"rows": rows}, {"rows": rows}, [], [], _NESTED_PEEL_SCHEMA)
    assert _val(uni, "extract_unified_value_recall") == 1.0
    assert _val(uni, "extract_unified_value_precision") == 1.0


def _scalar_schema(field: str, typ: str = "string") -> dict[str, Any]:
    return {"type": "object", "properties": {field: {"type": [typ, "null"]}}}


def test_null_equals_false_normalizer_accepts_blank_checkbox_both_ways() -> None:
    schema = {
        "type": "object",
        "properties": {"swr37_bw_hearing": {"type": ["boolean", "null"], "default": None}},
    }

    # GT False, model omitted the field entirely.
    omitted = compute_unified_evidence_metrics(
        {"swr37_bw_hearing": False},
        {},
        [_rule("swr37_bw_hearing", False, normalizers=["null_equals_false"])],
        [],
        schema,
    )
    # GT null, model asserted False.
    asserted = compute_unified_evidence_metrics(
        {"swr37_bw_hearing": None},
        {"swr37_bw_hearing": False},
        [_rule("swr37_bw_hearing", None, normalizers=["null_equals_false"])],
        [],
        schema,
    )
    strict = compute_unified_evidence_metrics(
        {"swr37_bw_hearing": False},
        {},
        [_rule("swr37_bw_hearing", False)],
        [],
        schema,
    )

    assert _val(omitted, "extract_unified_value_f1") == 1.0
    assert _val(asserted, "extract_unified_value_f1") == 1.0
    assert _val(strict, "extract_unified_value_f1") == 0.0


def test_null_equals_false_normalizer_rejects_true_and_zero() -> None:
    schema = _scalar_schema("swr37_bw_hearing", "boolean")

    true_vs_null = compute_unified_evidence_metrics(
        {"swr37_bw_hearing": True},
        {},
        [_rule("swr37_bw_hearing", True, normalizers=["null_equals_false"])],
        [],
        schema,
    )
    zero_vs_false = compute_unified_evidence_metrics(
        {"swr37_bw_hearing": False},
        {"swr37_bw_hearing": 0},
        [_rule("swr37_bw_hearing", False, normalizers=["null_equals_false"])],
        [],
        schema,
    )

    assert _val(true_vs_null, "extract_unified_value_f1") == 0.0
    # 0 == False in Python; the normalizer must not launder ints, but the base
    # ``==`` cell match already treats 0 as False, so this stays a pass there.
    assert _val(zero_vs_false, "extract_unified_value_f1") == 1.0


def test_case_insensitive_normalizer_is_field_scoped() -> None:
    schema = _scalar_schema("time_point")

    strict = compute_unified_evidence_metrics(
        {"time_point": "INITIAL"},
        {"time_point": "Initial"},
        [_rule("time_point", "INITIAL")],
        [],
        schema,
    )
    normalized = compute_unified_evidence_metrics(
        {"time_point": "INITIAL"},
        {"time_point": "Initial"},
        [_rule("time_point", "INITIAL", normalizers=["case_insensitive"])],
        [],
        schema,
    )

    assert _val(strict, "extract_unified_value_f1") == 0.0
    assert _val(normalized, "extract_unified_value_f1") == 1.0


def test_phone_digits_normalizer_matches_formatting_variants_only() -> None:
    schema = _scalar_schema("phone")
    rule = [_rule("phone", "( 713 ) 372-2430", normalizers=["phone_digits"])]

    same_digits = compute_unified_evidence_metrics(
        {"phone": "( 713 ) 372-2430"}, {"phone": "713-372-2430"}, rule, [], schema
    )
    other_digits = compute_unified_evidence_metrics(
        {"phone": "( 713 ) 372-2430"}, {"phone": "(713) 372-2431"}, rule, [], schema
    )
    area_code_only = compute_unified_evidence_metrics(
        {"phone": "( 512 )"},
        {"phone": "512"},
        [_rule("phone", "( 512 )", normalizers=["phone_digits"])],
        [],
        schema,
    )

    assert _val(same_digits, "extract_unified_value_f1") == 1.0
    assert _val(other_digits, "extract_unified_value_f1") == 0.0
    assert _val(area_code_only, "extract_unified_value_f1") == 1.0


def test_lenient_date_normalizer_joins_split_preprinted_years() -> None:
    schema = _scalar_schema("p2_date_well_plugged")
    rule = [_rule("p2_date_well_plugged", "1955-05-11", normalizers=["lenient_date"])]

    split_year = compute_unified_evidence_metrics(
        {"p2_date_well_plugged": "1955-05-11"}, {"p2_date_well_plugged": "May 11 , 19 55"}, rule, [], schema
    )
    two_digit_year = compute_unified_evidence_metrics(
        {"p2_date_well_plugged": "1955-05-11"}, {"p2_date_well_plugged": "May 11, 55"}, rule, [], schema
    )
    wrong_day = compute_unified_evidence_metrics(
        {"p2_date_well_plugged": "1955-05-11"}, {"p2_date_well_plugged": "May 12, 19 55"}, rule, [], schema
    )

    assert _val(split_year, "extract_unified_value_f1") == 1.0
    assert _val(two_digit_year, "extract_unified_value_f1") == 1.0
    assert _val(wrong_day, "extract_unified_value_f1") == 0.0


def test_lenient_date_normalizer_does_not_equate_different_values() -> None:
    """Rewrites are guarded: no century expansion without a day, no split-year
    join without a preceding day, and unparseable rewrites never match."""
    schema = _scalar_schema("d")

    def _f1(gt: str, pred: str) -> float | None:
        rule = [_rule("d", gt, normalizers=["lenient_date"])]
        return _val(
            compute_unified_evidence_metrics({"d": gt}, {"d": pred}, rule, [], schema), "extract_unified_value_f1"
        )

    # A day-only value must not be read as a 2-digit year ("May 11" != May 2011).
    assert _f1("May 11", "May 2011") == 0.0
    # "19" here is the day, not a split decade: "June 19 44" is June 19, 1944.
    assert _f1("1944-06-19", "June 19 44") == 1.0
    assert _f1("June 19 44", "June 1944") == 0.0
    # Non-date text must never match through a fabricated rewrite.
    assert _f1("Permit 12-34", "Permit 12-19 34") == 0.0


def test_optional_terminal_punctuation_normalizer_strips_one_char_only() -> None:
    schema = _scalar_schema("f")

    def _f1(gt: str, pred: str) -> float | None:
        rule = [_rule("f", gt, normalizers=["optional_terminal_punctuation"])]
        return _val(
            compute_unified_evidence_metrics({"f": gt}, {"f": pred}, rule, [], schema), "extract_unified_value_f1"
        )

    assert _f1("item 5.", "item 5") == 1.0
    # Only ONE terminal char is optional; a punctuation run is real content.
    assert _f1("item 5.", "item 5;,,.") == 0.0
    # Punctuation-only values must not collapse to '' and match each other.
    assert _f1(".", ";") == 0.0


def test_punctuation_spacing_normalizer_is_opt_in() -> None:
    schema = _scalar_schema("address")
    gt = {"address": "P.O. Box 978"}
    pred = {"address": "P.O.Box 978"}

    strict = compute_unified_evidence_metrics(gt, pred, [_rule("address", "P.O. Box 978")], [], schema)
    lenient = compute_unified_evidence_metrics(
        gt, pred, [_rule("address", "P.O. Box 978", normalizers=["punctuation_spacing"])], [], schema
    )
    different = compute_unified_evidence_metrics(
        gt,
        {"address": "P.O. Box 979"},
        [_rule("address", "P.O. Box 978", normalizers=["punctuation_spacing"])],
        [],
        schema,
    )

    assert _val(strict, "extract_unified_value_f1") == 0.0
    assert _val(lenient, "extract_unified_value_f1") == 1.0
    assert _val(different, "extract_unified_value_f1") == 0.0


def test_omitted_scalar_without_default_is_not_implicit_null() -> None:
    """A missing key with no schema ``default`` is absent, not predicted null."""
    schema = {
        "type": "object",
        "properties": {"a": {"type": ["string", "null"]}, "b": {"type": ["string", "null"]}},
    }
    rules = [_rule("a", "x"), _rule("b", None)]
    omitted = compute_unified_evidence_metrics({"a": "x", "b": None}, {}, rules, [], schema)
    explicit = compute_unified_evidence_metrics({"a": "x", "b": None}, {"a": None, "b": None}, rules, [], schema)
    omitted_meta = next(m.metadata for m in omitted if m.metric_name == "extract_unified_value_recall")
    explicit_meta = next(m.metadata for m in explicit if m.metric_name == "extract_unified_value_recall")
    assert omitted_meta["expected_cells"] == 2
    assert omitted_meta["predicted_cells"] == 0
    assert explicit_meta["predicted_cells"] == 2


def test_omitted_scalar_with_null_default_matches_explicit_null() -> None:
    schema = {
        "type": "object",
        "properties": {
            "a": {"type": ["string", "null"], "default": None},
            "b": {"type": ["string", "null"], "default": None},
        },
    }
    rules = [_rule("a", "x"), _rule("b", None)]
    omitted = compute_unified_evidence_metrics({"a": "x", "b": None}, {}, rules, [], schema)
    explicit = compute_unified_evidence_metrics({"a": "x", "b": None}, {"a": None, "b": None}, rules, [], schema)
    assert _val(omitted, "extract_unified_value_precision") == 0.5
    assert _val(omitted, "extract_unified_value_precision") == _val(explicit, "extract_unified_value_precision")
    assert _val(omitted, "extract_unified_value_recall") == _val(explicit, "extract_unified_value_recall")


def test_lookup_present_key_wins_including_explicit_null() -> None:
    schema = {"type": ["string", "null"], "default": "x"}
    assert lookup({"a": None}, "a", schema) == (True, None)
    assert lookup({"a": "y"}, "a", schema) == (True, "y")
    assert lookup({}, "a", schema) == (True, "x")
    assert lookup({}, "a", {"type": "string"}) == (False, None)
    assert lookup({}, "a", {"default": None}) == (True, None)


def test_explicit_null_vs_omit_without_default_is_one_sided() -> None:
    schema = {"type": "object", "properties": {"a": {"type": ["string", "null"]}}}
    rules = [_rule("a", None)]
    omit_pred = compute_unified_evidence_metrics({"a": None}, {}, rules, [], schema)
    omit_gold = compute_unified_evidence_metrics({}, {"a": None}, rules, [], schema)
    both = compute_unified_evidence_metrics({"a": None}, {"a": None}, rules, [], schema)
    omit_pred_meta = next(m.metadata for m in omit_pred if m.metric_name == "extract_unified_value_recall")
    omit_gold_meta = next(m.metadata for m in omit_gold if m.metric_name == "extract_unified_value_recall")
    assert omit_pred_meta["expected_cells"] == 1
    assert omit_pred_meta["predicted_cells"] == 0
    assert omit_gold_meta["expected_cells"] == 0
    assert omit_gold_meta["predicted_cells"] == 1
    assert _val(both, "extract_unified_value_f1") == 1.0


def test_omitted_scalar_with_non_null_default_is_a_real_value() -> None:
    schema = {"type": "object", "properties": {"name": {"type": "string", "default": "Acme"}}}
    match = compute_unified_evidence_metrics({"name": "Acme"}, {}, [_rule("name", "Acme")], [], schema)
    miss = compute_unified_evidence_metrics({"name": "Beta"}, {}, [_rule("name", "Beta")], [], schema)
    assert _val(match, "extract_unified_value_f1") == 1.0
    assert _val(miss, "extract_unified_value_recall") == 0.0
    assert _val(miss, "extract_unified_value_precision") == 0.0
    miss_meta = next(m.metadata for m in miss if m.metric_name == "extract_unified_value_recall")
    assert miss_meta["expected_cells"] == 1
    assert miss_meta["predicted_cells"] == 1


def test_both_sides_omit_defaulted_root_field() -> None:
    """Schema names, not instance keys, are the cell universe (plus lookup)."""
    schema = {"type": "object", "properties": {"name": {"type": "string", "default": "Acme"}}}
    uni = compute_unified_evidence_metrics({}, {}, [_rule("name", "Acme")], [], schema)
    assert _val(uni, "extract_unified_value_f1") == 1.0
    meta = next(m.metadata for m in uni if m.metric_name == "extract_unified_value_recall")
    assert meta["expected_cells"] == 1
    assert meta["predicted_cells"] == 1


def test_both_sides_omit_root_field_without_default_is_not_a_cell() -> None:
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    uni = compute_unified_evidence_metrics({}, {}, [_rule("name", "Acme")], [], schema)
    assert uni == []


def test_nested_object_child_omit_uses_field_default() -> None:
    schema = {
        "type": "object",
        "properties": {
            "vendor": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "default": "Acme"},
                    "city": {"type": ["string", "null"]},
                },
            }
        },
    }
    expected = {"vendor": {"name": "Acme", "city": "Austin"}}
    actual = {"vendor": {}}
    rules = [_rule("vendor.name", "Acme"), _rule("vendor.city", "Austin")]
    uni = compute_unified_evidence_metrics(expected, actual, rules, [], schema)
    meta = next(m.metadata for m in uni if m.metric_name == "extract_unified_value_recall")
    # name defaulted to Acme (match); city absent (recall-only, not a null prediction).
    assert meta["expected_cells"] == 2
    assert meta["predicted_cells"] == 1
    assert _val(uni, "extract_unified_value_recall") == 0.5
    assert _val(uni, "extract_unified_value_precision") == 1.0


def test_in_row_object_child_omit_uses_schema_default() -> None:
    schema = {
        "type": "object",
        "properties": {
            "rows": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": ["string", "null"]},
                        "addr": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string", "default": "Austin"},
                                "zip": {"type": ["string", "null"]},
                            },
                        },
                    },
                },
            }
        },
    }
    expected = {"rows": [{"id": "1", "addr": {"city": "Austin", "zip": "1"}}]}
    actual = {"rows": [{"id": "1", "addr": {}}]}
    rules = [_rule("rows[0].id", "1"), _rule("rows[0].addr.city", "Austin"), _rule("rows[0].addr.zip", "1")]
    uni = compute_unified_evidence_metrics(expected, actual, rules, [], schema)
    meta = next(m.metadata for m in uni if m.metric_name == "extract_unified_value_recall")
    # id match + city default match; zip absent (recall-only).
    assert meta["expected_cells"] == 3
    assert meta["predicted_cells"] == 2
    assert _val(uni, "extract_unified_value_recall") == 2 / 3
    assert _val(uni, "extract_unified_value_precision") == 1.0


def test_supported_normalizers_match_schema_vocabulary() -> None:
    from extract_bench.test_cases.schema import EXTRACT_FIELD_NORMALIZERS

    assert unified_evidence_metric.SUPPORTED_NORMALIZERS == EXTRACT_FIELD_NORMALIZERS


def test_giant_grounded_array_skips_grounding_value_exact_and_peels(monkeypatch: Any) -> None:
    """A grounded flat array over the cell threshold: value stays bit-exact, the
    grounded metrics are withheld (grounded_incomplete), and it takes the peel
    instead of building the multi-GB grounded matrix. Nothing below the
    threshold changes. This is the memory escape hatch for oklahoma-scale docs.
    """
    box = [0.0, 0.0, 10.0, 10.0]
    expected = {
        "as_of": None,
        "holdings": [{"security": f"S{i}", "coupon": float(i), "note": f"n{i}"} for i in range(6)],
    }
    actual = {  # one reorder + one wrong cell so value scoring is non-trivial
        "as_of": None,
        "holdings": [{"security": f"S{i}", "coupon": float(i), "note": f"n{i}"} for i in (1, 0, 2, 3, 4, 5)],
    }
    actual["holdings"][2]["coupon"] = 999.0
    rules = _bbox_rules(expected, page=1, bbox=box)
    cits = [
        {"field_path": f"holdings[{i}].{k}", "page": 1, "bbox": box}
        for i in range(6)
        for k in ("security", "coupon", "note")
    ]

    # Full-matrix reference (skip disabled): grounding present, value computed.
    full = compute_unified_evidence_metrics(expected, actual, rules, cits, _schema())
    assert _val(full, "extract_unified_grounded_f1") is not None

    # Force the skip by lowering the threshold below this array's cell count
    # (6 GT x 6 pred = 36), and confirm it takes the peel, not the full matrix.
    monkeypatch.setattr(unified_evidence_metric, "_GROUNDED_MAX_CELLS", 1)
    calls = _spy_on_peel(monkeypatch)
    skipped = compute_unified_evidence_metrics(expected, actual, rules, cits, _schema())

    assert ["security", "coupon", "note"] in calls, "giant grounded array must fall back to the peel"
    # Value metrics identical to the full-matrix reference.
    for name in (
        "extract_unified_value_precision",
        "extract_unified_value_recall",
        "extract_unified_value_f1",
    ):
        assert _val(skipped, name) == _val(full, name), name
    # Grounded metrics withheld entirely for the document.
    assert _val(skipped, "extract_unified_grounded_f1") is None
    assert _val(skipped, "extract_unified_grounded_precision") is None
    assert _val(skipped, "extract_unified_grounded_recall") is None
    # And the value metadata records why.
    vf1 = next(m for m in skipped if m.metric_name == "extract_unified_value_f1")
    assert vf1.metadata["grounded_incomplete"] is True


def test_giant_threshold_does_not_touch_nested_or_below_threshold(monkeypatch: Any) -> None:
    """The skip must not fire for arrays with nested object arrays (peel would shift
    nested TP), nor below the threshold."""
    box = [0.0, 0.0, 10.0, 10.0]
    expected = {"as_of": None, "holdings": [{"security": "A", "coupon": 1.0, "note": "n"}]}
    rules = _bbox_rules(expected, page=1, bbox=box)
    cits = [{"field_path": f"holdings[0].{k}", "page": 1, "bbox": box} for k in ("security", "coupon", "note")]
    # Below threshold (default 100M): grounding kept.
    kept = compute_unified_evidence_metrics(expected, expected, rules, cits, _schema())
    assert _val(kept, "extract_unified_grounded_f1") == 1.0
    assert next(m for m in kept if m.metric_name == "extract_unified_value_f1").metadata["grounded_incomplete"] is False

    # An object-array nested over the threshold must NOT skip (stays grounded via
    # full assignment) because the peel there could shift nested TP.
    monkeypatch.setattr(unified_evidence_metric, "_GROUNDED_MAX_CELLS", 1)
    nested = {"rows": [{"id": "A", "kids": [{"v": "x"}]}]}
    nrules = [
        _rule("rows[0].id", "A", page=1, bbox=box),
        _rule("rows[0].kids[0].v", "x", page=1, bbox=box),
    ]
    ncits = [
        {"field_path": "rows[0].id", "page": 1, "bbox": box},
        {"field_path": "rows[0].kids[0].v", "page": 1, "bbox": box},
    ]
    out = compute_unified_evidence_metrics(nested, nested, nrules, ncits, _NESTED_PEEL_SCHEMA)
    # Outer array has an object-array subfield -> not eligible for the skip -> grounding kept.
    assert _val(out, "extract_unified_grounded_f1") == 1.0


_HEADLINE_UNIFIED_PREFIXES = (
    "extract_unified_value_",
    "extract_unified_page_",
    "extract_unified_grounded_",
)


def _is_headline_unified(name: str) -> bool:
    return name.startswith(_HEADLINE_UNIFIED_PREFIXES) and "_word_" not in name and "_structural_" not in name
