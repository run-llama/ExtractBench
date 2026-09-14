"""Unit tests for the list-root unwrap helper used by per_table_row pipelines."""

from __future__ import annotations

from extract_bench.evaluation.metrics.extract.list_unwrap import (
    infer_array_field,
    normalize_list_prediction,
    unwrap_list_prediction,
)
from extract_bench.test_cases.schema import ExtractFieldTestRule


def _rule(field_path: str, value: str | int | float | bool | None = "x") -> ExtractFieldTestRule:
    return ExtractFieldTestRule(field_path=field_path, expected_value=value)


def test_dict_rooted_prediction_is_passthrough() -> None:
    """Per_doc predictions already have a dict root — unwrap must be a no-op."""
    rules = [_rule("personnel[0].name", "Alice"), _rule("client_id", "C-1")]
    extracted = {"personnel": [{"name": "Alice"}], "client_id": "C-1"}

    result, applied, skipped = unwrap_list_prediction(extracted, rules)

    assert result is extracted
    assert applied is False
    assert skipped == []


def test_list_rooted_prediction_all_rules_same_array_prefix() -> None:
    """All rules under one array — wrap, no skipped scalars."""
    rules = [
        _rule("personnel[0].name", "Alice"),
        _rule("personnel[1].name", "Bob"),
        _rule("personnel[0].net_pay", 100),
    ]
    extracted = [{"name": "Alice", "net_pay": 100}, {"name": "Bob", "net_pay": 200}]

    result, applied, skipped = unwrap_list_prediction(extracted, rules)

    assert applied is True
    assert skipped == []
    assert result == {"personnel": extracted}


def test_wrapper_per_row_prediction_is_flattened_to_array_prefix() -> None:
    """per_table_row can emit document-shaped wrappers; flatten their array field."""
    rules = [
        _rule("personnel[0].name", "Alice"),
        _rule("personnel[1].name", "Bob"),
        _rule("personnel[0].net_pay", 100),
        _rule("client_id", "C-1"),
    ]
    extracted = [
        {"client_id": "C-1", "run_number": "R-1", "personnel": [{"name": "Alice", "net_pay": 100}]},
        {"client_id": "C-1", "run_number": "R-1", "personnel": [{"name": "Bob", "net_pay": 200}]},
    ]

    result, applied, skipped = unwrap_list_prediction(extracted, rules)

    assert applied is True
    assert skipped == []
    assert result == {
        "client_id": "C-1",
        "run_number": "R-1",
        "personnel": [
            {"name": "Alice", "net_pay": 100},
            {"name": "Bob", "net_pay": 200},
        ],
    }


def test_wrapper_per_row_prediction_extends_multi_row_wrappers() -> None:
    """Flatten all rows if a wrapper carries more than one array item."""
    rules = [_rule("personnel[0].name", "Alice"), _rule("personnel[1].name", "Bob")]
    extracted = [
        {"personnel": [{"name": "Alice"}, {"name": "Bob"}]},
        {"personnel": []},
    ]

    result, applied, skipped = unwrap_list_prediction(extracted, rules)

    assert applied is True
    assert skipped == []
    assert result == {"personnel": [{"name": "Alice"}, {"name": "Bob"}]}


def test_attendance_rules_wrapper_rows_are_flattened() -> None:
    """Attendance v0.5 shape: one large array plus document-level scalar rules."""
    rules = [
        _rule("attendance_records[0].date", "2026-01-01"),
        _rule("attendance_records[0].clock_on", "09:00"),
        _rule("attendance_records[0].attendance_type", "Work"),
        _rule("attendance_records[1].date", "2026-01-02"),
        _rule("attendance_records[1].clock_off", "17:00"),
        _rule("employee_name", "Jane Doe"),
        _rule("period_start_date", "2026-01-01"),
        _rule("period_end_date", "2026-01-31"),
    ]
    extracted = [
        {
            "employee_name": "Jane Doe",
            "period_start_date": "2026-01-01",
            "period_end_date": "2026-01-31",
            "attendance_records": [{"date": "2026-01-01", "clock_on": "09:00", "attendance_type": "Work"}],
        },
        {
            "employee_name": "Jane Doe",
            "period_start_date": "2026-01-01",
            "period_end_date": "2026-01-31",
            "attendance_records": [{"date": "2026-01-02", "clock_off": "17:00"}],
        },
    ]

    result, applied, skipped = unwrap_list_prediction(extracted, rules)

    assert applied is True
    assert skipped == []
    assert result == {
        "employee_name": "Jane Doe",
        "period_start_date": "2026-01-01",
        "period_end_date": "2026-01-31",
        "attendance_records": [
            {"date": "2026-01-01", "clock_on": "09:00", "attendance_type": "Work"},
            {"date": "2026-01-02", "clock_off": "17:00"},
        ],
    }


def test_mixed_wrapper_shape_merges_available_arrays_and_scalars() -> None:
    """Wrapper merge keeps rows that are present and preserves scalar fields."""
    rules = [_rule("personnel[0].name", "Alice")]
    extracted = [
        {"personnel": [{"name": "Alice"}]},
        {"name": "Bob"},
    ]

    result, applied, skipped = unwrap_list_prediction(extracted, rules)

    assert applied is True
    assert skipped == []
    assert result == {"personnel": [{"name": "Alice"}], "name": "Bob"}


def test_list_rooted_prediction_skips_scalar_rules() -> None:
    """Scalar rules (no array index) are reported as skipped, not scored."""
    rules = [
        _rule("personnel[0].name", "Alice"),
        _rule("personnel[1].name", "Bob"),
        _rule("client_id", "C-1"),
        _rule("buyer.company", "Acme"),
    ]
    extracted = [{"name": "Alice"}, {"name": "Bob"}]

    result, applied, skipped = unwrap_list_prediction(extracted, rules)

    assert applied is True
    assert set(skipped) == {"client_id", "buyer.company"}
    assert result == {"personnel": extracted}


def test_list_rooted_prediction_multiple_array_prefixes_is_ambiguous() -> None:
    """Rules split across two top-level arrays — cannot safely unwrap."""
    rules = [
        _rule("personnel[0].name", "Alice"),
        _rule("transactions[0].amount", 50),
    ]
    extracted = [{"name": "Alice"}, {"name": "Bob"}]

    result, applied, skipped = unwrap_list_prediction(extracted, rules)

    assert result is extracted
    assert applied is False
    assert skipped == []


def test_multi_array_wrapper_prediction_merges_all_arrays() -> None:
    """DataSnipper bank statements can emit wrappers with several arrays."""
    rules = [
        _rule("checks_paid[0].amount", 10),
        _rule("electronic_debits_bank_debits[0].amount", 20),
        _rule("account_number", "123"),
    ]
    extracted = [
        {
            "account_number": "123",
            "checks_paid": [{"amount": 10}],
            "electronic_debits_bank_debits": [],
        },
        {
            "account_number": "123",
            "checks_paid": [],
            "electronic_debits_bank_debits": [{"amount": 20}],
        },
    ]

    result, applied, skipped = unwrap_list_prediction(extracted, rules)

    assert applied is True
    assert skipped == []
    assert result == {
        "account_number": "123",
        "checks_paid": [{"amount": 10}],
        "electronic_debits_bank_debits": [{"amount": 20}],
    }


def test_singleton_scalar_doc_list_is_unwrapped() -> None:
    """Scalar-only per-doc responses can arrive wrapped in a singleton list."""
    rules = [_rule("tax_year", "2024"), _rule("employee_name", "Jane Doe")]
    extracted = [{"tax_year": "2024", "employee_name": "Jane Doe"}]

    result, applied, skipped = unwrap_list_prediction(extracted, rules)

    assert applied is True
    assert skipped == []
    assert result == {"tax_year": "2024", "employee_name": "Jane Doe"}


def test_case_only_alias_is_skipped_using_schema_canonical_key() -> None:
    """If schema only defines Employee, lowercase duplicate rules are aliases."""
    rules = [
        _rule("Employee[0].Name", "Alice"),
        _rule("employee[0].Name", "Alice"),
    ]
    extracted = [{"Employee": [{"Name": "Alice"}]}]

    normalized = normalize_list_prediction(
        extracted,
        rules,
        data_schema={"type": "object", "properties": {"Employee": {"type": "array"}}},
    )

    assert normalized.applied is True
    assert normalized.mode == "wrapper_merge"
    assert normalized.alias_skipped_field_paths == ["employee[0].Name"]
    assert normalized.extracted_data == {"Employee": [{"Name": "Alice"}]}


def test_list_rooted_prediction_all_rules_scalar_unwraps_singleton_doc() -> None:
    """Scalar-only rules can score against a singleton document list."""
    rules = [_rule("client_id", "C-1"), _rule("buyer.company", "Acme")]
    extracted = [{"anything": 1}]

    result, applied, skipped = unwrap_list_prediction(extracted, rules)

    assert result == {"anything": 1}
    assert applied is True
    assert skipped == []


def test_empty_rule_list_is_passthrough() -> None:
    """No rules → nothing to infer from; leave the prediction untouched."""
    extracted: list[dict[str, int]] = [{"a": 1}]

    result, applied, skipped = unwrap_list_prediction(extracted, [])

    assert result is extracted
    assert applied is False
    assert skipped == []


def test_infer_array_field_single_prefix() -> None:
    rules = [_rule("personnel[0].name"), _rule("personnel[1].net_pay")]
    assert infer_array_field(rules) == "personnel"


def test_infer_array_field_no_array_rooted_returns_none() -> None:
    rules = [_rule("client_id"), _rule("buyer.company")]
    assert infer_array_field(rules) is None


def test_infer_array_field_multiple_prefixes_returns_none() -> None:
    rules = [_rule("personnel[0].name"), _rule("transactions[0].amount")]
    assert infer_array_field(rules) is None
