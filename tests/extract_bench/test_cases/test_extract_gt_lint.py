"""Tests for the fail-closed extract ground-truth lint.

Each defect class reproduces something the public corpus actually shipped
(ExtractBench issue #5) so the gate is pinned to real failures, not
hypothetical ones.
"""

from __future__ import annotations

import copy
from typing import Any

from extract_bench.test_cases.extract_gt_lint import LintFinding, lint_extract_sidecar


def _schema() -> dict[str, Any]:
    nullable = lambda t: {"anyOf": [{"type": t}, {"type": "null"}], "default": None}  # noqa: E731
    return {
        "type": "object",
        "title": "FundReport",
        "properties": {
            "fund_name": nullable("string"),
            "total": nullable("number"),
            "holdings": {
                "anyOf": [{"type": "array", "items": {"$ref": "#/$defs/Holding"}}, {"type": "null"}],
                "default": None,
            },
        },
        "$defs": {
            "Holding": {
                "type": "object",
                "properties": {"cusip": nullable("string"), "amount": nullable("number")},
            }
        },
    }


def _payload() -> dict[str, Any]:
    return {
        "tags": ["source:synthetic"],
        "data_schema": _schema(),
        "expected_output": {"fund_name": "Acme", "total": 12.5, "holdings": [{"cusip": "A1", "amount": 1}]},
        "_field_rules": {
            "fund_name": {"evidence": [{"page": 1, "bbox": [0.1, 0.1, 0.2, 0.05], "quote": "Acme", "value": "Acme"}]},
            "total": {"evidence": [{"page": 1, "value": 12.5}], "comparator": "number"},
            "holdings": {"structural": "match_by:cusip", "evidence": []},
            "holdings[0].cusip": {"evidence": [{"page": 2, "value": "A1"}]},
            "holdings[0].amount": {"evidence": [{"page": 2, "value": 1}]},
        },
        "_eval_row_identity": {"holdings": {"identity_key": "cusip"}},
    }


def _codes(findings: list[LintFinding]) -> list[str]:
    return [finding.code for finding in findings]


def test_clean_sidecar_has_no_findings() -> None:
    assert lint_extract_sidecar(_payload()) == []


def test_missing_schema_is_the_only_finding() -> None:
    assert _codes(lint_extract_sidecar({"expected_output": {}})) == ["schema.missing"]


# --- issue #5 item 1: expected_output keys the schema never declares -------


def test_undeclared_output_key_is_flagged() -> None:
    payload = _payload()
    payload["expected_output"]["report_title"] = "Annual Report"
    findings = lint_extract_sidecar(payload)
    assert [(f.code, f.path) for f in findings] == [("output.unknown_field", "expected_output/report_title")]


def test_undeclared_key_inside_an_array_row_is_flagged() -> None:
    payload = _payload()
    payload["expected_output"]["holdings"][0]["sector"] = "Energy"
    assert [f.path for f in lint_extract_sidecar(payload)] == ["expected_output/holdings[0]/sector"]


# --- issue #5 item 2: a number leaf shipped as a string --------------------


def test_string_in_number_leaf_is_flagged() -> None:
    payload = _payload()
    payload["expected_output"]["total"] = "2029"
    findings = lint_extract_sidecar(payload)
    assert _codes(findings) == ["output.type_mismatch"]
    assert findings[0].path == "expected_output/total"
    assert "expected number/null, got str" in findings[0].message


def test_null_leaf_is_not_a_type_mismatch() -> None:
    payload = _payload()
    payload["expected_output"]["total"] = None
    assert lint_extract_sidecar(payload) == []


# --- issue #5 item 3: the rule container --------------------------------


def _as_legacy_list(field_rules: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"type": "extract_field", "field_path": path, **body} for path, body in field_rules.items()]


def test_legacy_list_container_is_flagged_unless_allowed() -> None:
    payload = _payload()
    payload["test_rules"] = _as_legacy_list(payload.pop("_field_rules"))
    assert _codes(lint_extract_sidecar(payload)) == ["rules.legacy_container"]
    assert lint_extract_sidecar(payload, allow_legacy_rules=True) == []


def test_both_containers_is_flagged() -> None:
    payload = _payload()
    payload["test_rules"] = _as_legacy_list(payload["_field_rules"])
    assert _codes(lint_extract_sidecar(payload)) == ["rules.both_containers"]


def test_non_object_container_is_flagged() -> None:
    payload = _payload()
    payload["_field_rules"] = _as_legacy_list(payload["_field_rules"])
    assert _codes(lint_extract_sidecar(payload)) == ["rules.not_object"]


def test_legacy_entries_are_still_path_checked_when_allowed() -> None:
    payload = _payload()
    rules = _as_legacy_list(payload.pop("_field_rules"))
    rules.append({"type": "extract_field", "field_path": "holdings[0].nope", "evidence": []})
    rules.append({"type": "extract_field", "evidence": []})
    payload["test_rules"] = rules
    assert _codes(lint_extract_sidecar(payload, allow_legacy_rules=True)) == [
        "rules.entry_missing_path",
        "rules.path_not_in_schema",
    ]


# --- rule bodies and paths ----------------------------------------------


def test_rule_path_must_resolve_in_schema() -> None:
    payload = _payload()
    payload["_field_rules"]["holdings[0].nope"] = {"evidence": []}
    findings = lint_extract_sidecar(payload)
    assert [(f.code, f.path) for f in findings] == [("rules.path_not_in_schema", "_field_rules['holdings[0].nope']")]


def test_rule_path_must_resolve_in_output() -> None:
    payload = _payload()
    payload["_field_rules"]["holdings[5].cusip"] = {"evidence": []}
    assert _codes(lint_extract_sidecar(payload)) == ["rules.path_not_in_output"]


def test_null_ancestor_in_output_counts_as_resolved() -> None:
    payload = _payload()
    payload["expected_output"]["holdings"] = None
    for path in ("holdings[0].cusip", "holdings[0].amount"):
        payload["_field_rules"][path] = {"evidence": []}
    assert lint_extract_sidecar(payload) == []


def test_rule_body_is_validated_against_the_bench_model() -> None:
    payload = _payload()
    payload["_field_rules"]["fund_name"]["evidence"][0]["page"] = 0
    findings = lint_extract_sidecar(payload)
    assert _codes(findings) == ["rules.invalid"]
    assert findings[0].message.startswith("evidence.0.page")


def test_non_extract_rule_type_is_flagged() -> None:
    payload = _payload()
    payload["_field_rules"]["fund_name"]["type"] = "present"
    assert _codes(lint_extract_sidecar(payload)) == ["rules.entry_type"]


def test_no_rule_container_at_all_is_legal() -> None:
    payload = _payload()
    del payload["_field_rules"]
    del payload["_eval_row_identity"]
    assert lint_extract_sidecar(payload) == []


# --- row identity placement ---------------------------------------------


def test_identity_inside_data_schema_is_flagged() -> None:
    payload = _payload()
    payload["data_schema"]["repeated_structure"] = {"holdings": {"identity_key": "cusip"}}
    findings = lint_extract_sidecar(payload)
    assert [(f.code, f.path) for f in findings] == [("identity.in_schema", "data_schema.repeated_structure")]


def test_a_property_named_repeated_structure_is_not_identity() -> None:
    payload = _payload()
    payload["data_schema"]["properties"]["repeated_structure"] = {"anyOf": [{"type": "string"}, {"type": "null"}]}
    assert lint_extract_sidecar(payload) == []


def test_legacy_identity_key_is_flagged() -> None:
    payload = _payload()
    payload["_repeated_structure"] = copy.deepcopy(payload["_eval_row_identity"])
    assert _codes(lint_extract_sidecar(payload)) == ["identity.legacy_key"]


def test_identity_key_must_be_a_row_property() -> None:
    payload = _payload()
    payload["_eval_row_identity"] = {"holdings": {"identity_key": ["cusip", "isin"]}}
    findings = lint_extract_sidecar(payload)
    assert _codes(findings) == ["identity.key_missing"]
    assert "'isin'" in findings[0].message


def test_identity_path_must_be_an_array() -> None:
    payload = _payload()
    payload["_eval_row_identity"] = {"fund_name": {"identity_key": "x"}, "nope": {"identity_key": "x"}}
    assert _codes(lint_extract_sidecar(payload)) == ["identity.not_array", "identity.array_missing"]


def test_findings_render_as_one_line_each() -> None:
    assert LintFinding("a.b", "x", "why").render() == "a.b x: why"
