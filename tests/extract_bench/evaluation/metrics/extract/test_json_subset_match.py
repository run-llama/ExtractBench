from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from extract_bench.evaluation.metrics.extract.json_subset_match import (
    _is_nullable_numeric_field,
    json_subset_match_score,
    normalize_date_string,
    normalize_field_aliases,
)


def test_normalize_date_string_handles_weekday_prefix_with_periods() -> None:
    assert normalize_date_string("Mon. Jan. 02 2023") == "2023-01-02"
    assert normalize_date_string("Fri. Dec. 29 2023") == "2023-12-29"


def test_normalize_date_string_tolerates_comma_spacing() -> None:
    assert normalize_date_string("March 27 ,1956") == "1956-03-27"
    assert normalize_date_string("March 28,1956") == "1956-03-28"


def test_json_subset_match_scores_weekday_date_equivalence() -> None:
    score = json_subset_match_score(
        {"attendance_records": [{"date": "2023-01-02"}]},
        {"attendance_records": [{"date": "Mon. Jan. 02 2023"}]},
    )

    assert score == 1.0


# Regression tests: missing arrays/dicts must be weighted by the full leaf
# count of `expected`, not weight=1. Otherwise a pipeline that drops a whole
# claims array can score the same as one that extracts it.


def test_missing_top_level_array_weighted_by_full_leaf_count() -> None:
    """A 15-claim array dropped entirely must outweigh a 1-leaf scalar field."""
    expected = {
        "scalar_field": "x",
        "claims": [{"a": 1, "b": 2, "c": 3} for _ in range(15)],
    }
    actual = {"scalar_field": "x"}

    score = json_subset_match_score(expected, actual, weighted=True)
    # 1 leaf right out of 46 total => ~0.022. Pre-fix this was ~0.5 because
    # the missing claims array was treated as a single weight=1 leaf.
    assert score < 0.05, f"expected score <0.05, got {score}"


def test_missing_array_same_as_empty_array() -> None:
    """An absent key and `[]` should score identically when expected is non-empty."""
    expected = {"claims": [{"a": 1, "b": 2}]}
    score_missing = json_subset_match_score(expected, {}, weighted=True)
    score_empty = json_subset_match_score(expected, {"claims": []}, weighted=True)
    assert score_missing == score_empty == 0.0


def test_missing_nested_dict_weighted_by_subtree() -> None:
    """A dropped nested dict must be weighted by its leaf count."""
    expected = {
        "id": "x",
        "patient": {
            "first_name": "A",
            "last_name": "B",
            "address": {"city": "X", "zip": "1"},
        },
    }
    actual = {"id": "x"}
    score = json_subset_match_score(expected, actual, weighted=True)
    # 1/5 = 0.20
    assert 0.18 < score < 0.22, f"expected ~0.20, got {score}"


def test_partial_array_drop_weights_each_missing_item_recursively() -> None:
    """Dropping the tail of an array must penalize by per-item leaves."""
    expected = {
        "claims": [
            {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5},
            {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5},
            {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5},
        ]
    }
    actual = {"claims": [{"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}]}

    score = json_subset_match_score(expected, actual, weighted=True)
    # 5/15 = ~0.33. Pre-fix this would be ~0.78 because each missing tail
    # item only contributed weight=1.
    assert 0.30 < score < 0.40, f"expected ~0.33, got {score}"


def test_eob_pattern_dropped_claims_scores_low() -> None:
    """Reproducer for the agentic-vs-continuous gh_internal puzzle.

    When a pipeline returns payment_details correctly but doesn't emit
    claims at all on a doc that has 15 claims with ~10 fields each, the
    score should reflect that ~150 claim-leaves are missing, not weight=1.
    """
    expected = {
        "payment_details": [
            {
                "check": {"check_number": "X", "check_amount": 100, "check_date": "2026-01-01"},
                "payer": {"payer_name": "P", "payer_phone": "555"},
                "payee": {"payee_name": "Q"},
            }
        ],
        "claims": [
            {
                "claim_number": f"C{i}",
                "patient_name": f"P{i}",
                "total_paid": i * 10.0,
                "total_submitted": i * 12.0,
                "claim_from_date": "2026-01-01",
                "claim_to_date": "2026-01-31",
                "patient_account_number": f"A{i}",
                "plan_type": "PPO",
                "claim_status": "PAID",
                "patient_responsibility": 0.0,
            }
            for i in range(15)
        ],
    }
    actual = {"payment_details": expected["payment_details"]}
    score = json_subset_match_score(expected, actual, weighted=True)
    # PD has 7 leaves, claims has 15 × 10 = 150 leaves. 7/157 = ~0.045.
    # Pre-fix this scored ~0.875 (PD perfect / claims weight=1).
    assert score < 0.10, f"dropped 15-claim array should score <0.10, got {score}"


# Identity-keyed list pairing: when every expected record carries an
# identity field (claim_number, payment_id, etc.) the scorer must pair
# records by identity, not by index. Catches the failure mode where the
# extractor returns the right records but in a slightly shifted order
# (e.g. missed one in the middle, off-by-one for everything after that)
# and strict index pairing punishes every subsequent pair.


def test_identity_pairing_recovers_index_shifted_claims() -> None:
    """Same records, shifted by one mid-list — should score nearly 1.0, not
    ~0 the way strict index pairing penalizes the off-by-one cascade."""
    expected = {
        "claims": [
            {"claim_number": "A1", "amount": 100.0},
            {"claim_number": "A2", "amount": 200.0},
            {"claim_number": "A3", "amount": 300.0},
            {"claim_number": "A4", "amount": 400.0},
        ]
    }
    actual = {
        "claims": [
            {"claim_number": "A1", "amount": 100.0},
            # A2 dropped — extractor missed it
            {"claim_number": "A3", "amount": 300.0},
            {"claim_number": "A4", "amount": 400.0},
        ]
    }

    score = json_subset_match_score(expected, actual, weighted=True)
    # 3/4 records perfectly matched by identity → ~0.75. Pre-fix this would
    # have scored each shifted pair as a per-field mismatch, dragging the
    # score way down because every claim_number after A1 was scored against
    # the wrong GT record.
    assert score >= 0.7, f"identity-paired score should be >=0.7, got {score}"


def test_identity_pairing_penalizes_wrong_identity() -> None:
    """If the extractor returns a record with a completely different
    claim_number, identity pairing must NOT give it credit for any field
    overlap with the GT record at the same index."""
    expected = {"claims": [{"claim_number": "REAL", "amount": 100.0}]}
    actual = {"claims": [{"claim_number": "WRONG", "amount": 100.0}]}

    score = json_subset_match_score(expected, actual, weighted=True)
    # Pre-fix this scored ~0.5 because amount happened to overlap. With
    # identity pairing the WRONG record can't pair with REAL, so the GT
    # record counts as fully missed.
    assert score < 0.1, f"wrong-identity record should score <0.1, got {score}"


def test_identity_pairing_falls_back_to_assignment_when_no_identity() -> None:
    """Lists whose elements don't carry an identity field pair by optimal
    assignment — workloads that score arrays of strings or anonymous dicts
    still get full credit for exact matches."""
    expected = {"chunks": ["a", "b", "c"]}
    actual = {"chunks": ["a", "b", "c"]}
    assert json_subset_match_score(expected, actual, weighted=True) == 1.0

    expected2 = {"rows": [{"col_a": 1, "col_b": 2}, {"col_a": 3, "col_b": 4}]}
    actual2 = {"rows": [{"col_a": 1, "col_b": 2}, {"col_a": 3, "col_b": 4}]}
    assert json_subset_match_score(expected2, actual2, weighted=True) == 1.0


# Order-invariant pairing: lists without identity fields must be paired by
# optimal assignment (Hungarian), not by index, mirroring
# ArrayRecordMatchMetric. Shuffled-but-correct extractions should not be
# penalized for row order.


def test_anonymous_record_list_is_order_invariant() -> None:
    expected = {
        "rows": [
            {"col_a": 1, "col_b": "x"},
            {"col_a": 2, "col_b": "y"},
            {"col_a": 3, "col_b": "z"},
        ]
    }
    actual = {"rows": list(reversed(expected["rows"]))}

    assert json_subset_match_score(expected, actual, weighted=True) == 1.0


def test_scalar_list_is_order_invariant() -> None:
    assert json_subset_match_score({"chunks": ["a", "b", "c"]}, {"chunks": ["c", "a", "b"]}) == 1.0


def test_order_invariant_pairing_recovers_dropped_middle_row() -> None:
    """A dropped middle row must cost only that row, not cascade an
    off-by-one penalty onto every subsequent index-paired row."""
    expected = {"rows": [{"a": i, "b": f"r{i}"} for i in range(4)]}
    actual = {"rows": [expected["rows"][0], expected["rows"][2], expected["rows"][3]]}

    score = json_subset_match_score(expected, actual, weighted=True)
    # 3 of 4 rows match perfectly, each row has 2 leaves: 6/8 = 0.75.
    assert score == 0.75, f"expected 0.75, got {score}"


def test_tied_assignment_prefers_index_order() -> None:
    """When every pairing is a near-miss (uniform approximate cost matrix),
    index order must win the tie deterministically. Pins the eps*|i-j|
    tie-break: without it, which optimal assignment wins is an undocumented
    scipy implementation detail, and a scipy upgrade could silently swap
    aligned near-miss rows for crossed ones, dropping the Levenshtein
    partial credit with no code change."""
    expected = {"rows": [{"text": "alpha bravo"}, {"text": "charlie delta"}]}
    # Aligned rows with one-character typos: no leaf matches exactly, so the
    # approximate cost is 1 for every pairing — a fully tied matrix.
    actual = {"rows": [{"text": "alpha brivo"}, {"text": "charlie delte"}]}

    score = json_subset_match_score(expected, actual, weighted=True)
    # Index-aligned pairing keeps ~0.92 Levenshtein partial credit; the
    # crossed pairing would score near zero.
    assert score > 0.8, f"aligned near-miss rows should keep partial credit, got {score}"


def test_order_invariant_pairing_does_not_reward_wrong_values() -> None:
    """Optimal assignment must not inflate scores when values are wrong —
    a fully mismatched list still scores low."""
    expected = {"rows": [{"a": 1}, {"a": 2}]}
    actual = {"rows": [{"a": 7}, {"a": 8}]}

    score = json_subset_match_score(expected, actual, weighted=True)
    assert score < 0.5, f"expected <0.5, got {score}"


def test_identity_pairing_only_engages_when_all_expected_have_identity() -> None:
    """If some expected records carry an identity and some don't, fall
    back to index pairing so partial-identity datasets aren't silently
    rescored. Conservative default."""
    expected = {
        "claims": [
            {"claim_number": "A", "amount": 1.0},
            {"amount": 2.0},  # no identity
        ]
    }
    # Identical actual: should score 1.0 via either pairing.
    actual = {
        "claims": [
            {"claim_number": "A", "amount": 1.0},
            {"amount": 2.0},
        ]
    }
    assert json_subset_match_score(expected, actual, weighted=True) == 1.0


def test_declared_identity_keys_recover_reordered_rows() -> None:
    expected = {
        "line_items": [
            {"item_no": "0001", "description": "ALPHA", "amount": 100},
            {"item_no": "0002", "description": "BRAVO", "amount": 200},
            {"item_no": "0003", "description": "CHARLIE", "amount": 300},
        ]
    }
    actual = {"line_items": list(reversed(expected["line_items"]))}
    keys = {("line_items",): ["item_no"]}
    assert json_subset_match_score(expected, actual, identity_keys_by_path=keys) == 1.0
    # The assignment-pairing fallback also recovers pure reorders.
    assert json_subset_match_score(expected, actual) == 1.0

    # Where declared identity is stricter than assignment pairing: a row
    # carrying the wrong identity gets no credit for overlapping fields,
    # while assignment pairing still grants partial credit.
    wrong_identity = {"line_items": [{"item_no": "9999", "description": "ALPHA", "amount": 100}]}
    expected_one = {"line_items": [expected["line_items"][0]]}
    assert json_subset_match_score(expected_one, wrong_identity, identity_keys_by_path=keys) == 0.0
    assert json_subset_match_score(expected_one, wrong_identity) > 0.5


def test_declared_identity_keys_scope_dropped_row_penalty() -> None:
    expected = {
        "line_items": [
            {"item_no": "0001", "description": "ALPHA", "amount": 100},
            {"item_no": "0002", "description": "BRAVO", "amount": 200},
            {"item_no": "0003", "description": "CHARLIE", "amount": 300},
        ]
    }
    actual = {"line_items": expected["line_items"][1:]}
    keys = {("line_items",): ["item_no"]}
    score = json_subset_match_score(expected, actual, identity_keys_by_path=keys)
    # Two of three rows fully recovered; only the dropped row's leaves are lost.
    assert abs(score - 2 / 3) < 1e-9


def test_declared_identity_composite_keys_with_null_cells() -> None:
    expected = {
        "holdings": [
            {"cusip": "AAA", "put_call": None, "value": 10},
            {"cusip": "AAA", "put_call": "PUT", "value": 20},
        ]
    }
    # Provider omits null keys and reorders; composite identity still pairs.
    actual = {
        "holdings": [
            {"cusip": "AAA", "put_call": "PUT", "value": 20},
            {"cusip": "AAA", "value": 10},
        ]
    }
    keys = {("holdings",): ["cusip", "put_call"]}
    assert json_subset_match_score(expected, actual, identity_keys_by_path=keys) == 1.0


def test_declared_identity_duplicate_rows_pair_in_order() -> None:
    expected = {"rows": [{"sku": "X", "qty": 1}, {"sku": "X", "qty": 2}]}
    actual = {"rows": [{"sku": "X", "qty": 1}, {"sku": "X", "qty": 2}]}
    keys = {("rows",): ["sku"]}
    assert json_subset_match_score(expected, actual, identity_keys_by_path=keys) == 1.0


def test_declared_identity_keys_apply_to_nested_paths() -> None:
    expected = {"report": {"rows": [{"id": "A", "v": 1}, {"id": "B", "v": 2}]}}
    actual = {"report": {"rows": [{"id": "B", "v": 2}, {"id": "A", "v": 1}]}}
    keys = {("report", "rows"): ["id"]}
    assert json_subset_match_score(expected, actual, identity_keys_by_path=keys) == 1.0


def test_declared_identity_falls_back_for_non_dict_rows() -> None:
    # Scalars under a declared path cannot carry identity; order-invariant
    # assignment pairing applies, so reorders still score 1.0 and wrong
    # values are still penalized.
    expected = {"codes": ["A", "B", "C"]}
    keys = {("codes",): ["item_no"]}
    assert json_subset_match_score(expected, {"codes": ["A", "B", "C"]}, identity_keys_by_path=keys) == 1.0
    assert json_subset_match_score(expected, {"codes": ["C", "B", "A"]}, identity_keys_by_path=keys) == 1.0
    assert json_subset_match_score(expected, {"codes": ["X", "Y", "Z"]}, identity_keys_by_path=keys) < 1.0


def test_declared_identity_tolerates_identity_surface_drift() -> None:
    """Surface-form drift in an identity cell (trailing punctuation, numeric
    formatting) must still pair the row — aligned with the v0.2 evidence
    alignment's stable_value_key normalization — not hard-zero it. The
    drifted cell itself still pays its per-field score after pairing."""
    keys = {("line_items",): ["item_no"]}

    expected = {"line_items": [{"item_no": "0001", "description": "ALPHA", "amount": 100}]}
    actual = {"line_items": [{"item_no": "0001.", "description": "ALPHA", "amount": 100}]}
    score = json_subset_match_score(expected, actual, identity_keys_by_path=keys)
    # Row pairs; only item_no pays Levenshtein drift: (0.8 + 1 + 1) / 3.
    assert score > 0.9, f"trailing-punctuation drift should not unpair the row, got {score}"

    expected = {"line_items": [{"item_no": 100, "description": "ALPHA", "amount": 5}]}
    actual = {"line_items": [{"item_no": "100.00", "description": "ALPHA", "amount": 5}]}
    score = json_subset_match_score(expected, actual, identity_keys_by_path=keys)
    # Row pairs; the int-vs-str item_no cell scores 0: (0 + 1 + 1) / 3.
    assert score > 0.6, f"numeric-format drift should not unpair the row, got {score}"

    # Genuinely different identities must still hard-zero (declared-identity
    # strictness is about wrong identity, not drifted identity).
    expected = {"line_items": [{"item_no": "0001", "description": "ALPHA", "amount": 100}]}
    actual = {"line_items": [{"item_no": "0002", "description": "ALPHA", "amount": 100}]}
    assert json_subset_match_score(expected, actual, identity_keys_by_path=keys) == 0.0


def test_assignment_pairing_falls_back_to_index_beyond_pair_cap(monkeypatch: Any) -> None:
    """Beyond the bounded pair budget (mirroring the v0.2 adapter's pass-2
    cap) the n×m assignment is skipped and index pairing applies, so
    pathological very-long arrays don't stall the evaluator."""
    import extract_bench.evaluation.metrics.extract.json_subset_match as jsm

    expected = {"rows": [{"a": i} for i in range(3)]}
    actual = {"rows": [{"a": i} for i in reversed(range(3))]}

    monkeypatch.setattr(jsm, "_ASSIGNMENT_MAX_PAIRS", 4)
    capped = json_subset_match_score(expected, actual, weighted=True)
    assert capped < 1.0, f"beyond the cap, index pairing should penalize reorder, got {capped}"

    monkeypatch.setattr(jsm, "_ASSIGNMENT_MAX_PAIRS", 250_000)
    assert json_subset_match_score(expected, actual, weighted=True) == 1.0


def test_normalize_date_string_probe_is_comma_scoped() -> None:
    """The comma-respacing probe must not widen what counts as a date: strings
    without a comma are left exactly as before (doubled spaces included)."""
    assert normalize_date_string("March  27 1956") == "March  27 1956"
    assert normalize_date_string("March 27 1956") == "1956-03-27"


# Input immutability. The evaluator scores the same `expected_output` object
# several times (overall accuracy, per-field accuracy, per-schema-group
# accuracy, the evidence metrics), and its ground-truth normalization only
# shallow-copies, so payment_details entries are shared across those calls. Any
# in-place rewrite inside the alias normalization silently changes what every
# later call is scored against.

_MUTATION_REPRO_EXPECTED: dict[str, Any] = {
    "payment_details": [{"check": {"check_number": "12345", "check_amount": 100.0}}]
}
# check_number in the wrong place and wrong-valued: must score 0 every time.
_MUTATION_REPRO_ACTUAL: dict[str, Any] = {
    "payment_details": [{"check_number": "99999", "check": {"check_amount": 100.0}}]
}


def test_scoring_does_not_mutate_expected() -> None:
    """`expected` must come back byte-identical — the nested check_number lift
    used to pop the key out of the caller's ground truth."""
    expected = copy.deepcopy(_MUTATION_REPRO_EXPECTED)
    actual = copy.deepcopy(_MUTATION_REPRO_ACTUAL)
    before = copy.deepcopy(expected)
    actual_before = copy.deepcopy(actual)

    json_subset_match_score(expected, actual)

    assert expected == before
    assert actual == actual_before


def test_repeated_scoring_of_same_objects_is_stable() -> None:
    """Two consecutive scores of the same objects must agree. Pre-fix the first
    call stripped `check.check_number` from the ground truth, so the second call
    saw a GT with no check_number at all and scored the wrong prediction 1.0."""
    expected = copy.deepcopy(_MUTATION_REPRO_EXPECTED)
    actual = copy.deepcopy(_MUTATION_REPRO_ACTUAL)

    first = json_subset_match_score(expected, actual)
    second = json_subset_match_score(expected, actual)

    assert first == second, f"repeated scoring drifted: {first} -> {second}"
    assert first == 0.0, f"wrong check_number should score 0.0, got {first}"


def test_scoring_does_not_mutate_expected_for_checks_plural_shape() -> None:
    """Same guarantee for the legacy `checks: [<dict>]` shape, whose entry dict
    is also aliased out of the caller's input before the check_number lift."""
    expected = {"payment_details": [{"checks": [{"check_number": "12345", "check_amount": 100.0}]}]}
    actual = {"payment_details": [{"check_number": "99999", "check": {"check_amount": 100.0}}]}
    before = copy.deepcopy(expected)

    first = json_subset_match_score(expected, actual)
    second = json_subset_match_score(expected, actual)

    assert expected == before
    assert first == second


# normalize_field_aliases: the back-compat rewrites applied to both sides
# before comparison. Exercised directly so a change to one alias shape can't
# ride in behind an end-to-end score assertion.


def test_normalize_field_aliases_renames_rendering_provider_npi() -> None:
    normalized = normalize_field_aliases(
        {"claims": [{"rendering_provider": {"rendering_provider_identification_number": "1234567893"}}]}
    )
    assert normalized == {"claims": [{"rendering_provider": {"provider_npi": "1234567893"}}]}


def test_normalize_field_aliases_rewrites_payment_details_shapes() -> None:
    normalized = normalize_field_aliases(
        {
            "payment_details": [
                {
                    "checks": [{"check_number": "12345"}],
                    "check_amount": 100.0,
                    "check_date": "2026-01-01",
                    "payment_method": "CHECK",
                    "provider_id": "1234567893",
                    "provider_name": "ACME",
                    "provider_address": "1 Main St",
                }
            ]
        }
    )
    assert normalized == {
        "payment_details": [
            {
                "check": {
                    "check_amount": 100.0,
                    "check_date": "2026-01-01",
                    "payment_method": "CHECK",
                },
                "check_number": "12345",
                "payee": {
                    "payee_npi": "1234567893",
                    "payee_name": "ACME",
                    "payee_address": {"payee_address_one": "1 Main St"},
                },
            }
        ]
    }


def test_normalize_field_aliases_prefers_top_level_check_number() -> None:
    """An already-lifted top-level check_number wins over the nested fallback,
    and the nested copy is dropped so it isn't scored twice."""
    normalized = normalize_field_aliases(
        {"payment_details": [{"check_number": "TOP", "check": {"check_number": "NESTED", "check_amount": 5.0}}]}
    )
    assert normalized == {"payment_details": [{"check_number": "TOP", "check": {"check_amount": 5.0}}]}


def test_normalize_field_aliases_maps_non_npi_provider_id_to_tin() -> None:
    normalized = normalize_field_aliases({"payment_details": [{"provider_id": "12-3456789"}]})
    assert normalized == {"payment_details": [{"payee": {"payee_tin": "12-3456789"}}]}


def test_normalize_field_aliases_leaves_unrelated_values_untouched() -> None:
    payload = {"claims": [{"claim_number": "A1", "amount": 100.0}], "notes": None, "tags": ["x", "y"]}
    assert normalize_field_aliases(payload) == payload


def test_empty_expected_list_against_populated_actual() -> None:
    """Weighted (the default) keeps subset semantics — an empty expected list
    makes no claim about what the extractor returned. Unweighted mirrors
    upstream scoring and penalizes the over-extraction."""
    expected: dict[str, Any] = {"claims": []}
    actual: dict[str, Any] = {"claims": [{"claim_number": "A1"}, {"claim_number": "A2"}]}

    assert json_subset_match_score(expected, actual, weighted=True) == 1.0
    assert json_subset_match_score(expected, actual, weighted=False) == 0.0
    # Both empty still matches in either mode.
    assert json_subset_match_score(expected, {"claims": []}, weighted=False) == 1.0


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
