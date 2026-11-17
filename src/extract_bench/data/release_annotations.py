"""Filter dataset records at the release boundary."""

import json
from collections.abc import Collection
from typing import Any

from extract_bench.test_cases.schema import ExtractFieldBbox, ExtractFieldTestRule, FieldEvidence

OPTIONAL_SIDECAR_FIELDS = {
    "repeated_structure": "_eval_row_identity",
    "_identity_hints": "_identity_hints",
}

_FIELD_RULES = "field_rules"
_DATASET_COLUMNS = frozenset(
    {
        "id",
        "category",
        "pdf",
        "data_schema",
        "expected_output",
        _FIELD_RULES,
        "tags",
        *OPTIONAL_SIDECAR_FIELDS,
    }
)
_RULE_LIST_ITEMS = {
    "evidence": FieldEvidence.model_fields.keys(),
    "bboxes": ExtractFieldBbox.model_fields.keys(),
}


def _remove_extra_keys(record: dict[str, Any], supported: Collection[str]) -> bool:
    extra_keys = record.keys() - supported
    for key in extra_keys:
        del record[key]
    return len(extra_keys) > 0


def sanitize_release_row(row: dict[str, Any]) -> bool:
    """Remove undeclared record, rule and citation keys without rewriting scored values."""
    changed = _remove_extra_keys(row, _DATASET_COLUMNS)
    encoded_rules = row.get(_FIELD_RULES)
    if encoded_rules is None or encoded_rules == "":
        return changed
    rules = json.loads(encoded_rules)
    if rules is None:
        return changed
    entries = rules.values() if isinstance(rules, dict) else rules
    rules_changed = False
    for rule in entries:
        rules_changed |= _remove_extra_keys(rule, ExtractFieldTestRule.model_fields.keys())
        for field, supported in _RULE_LIST_ITEMS.items():
            citations = rule.get(field)
            if citations is not None:
                for citation in citations:
                    rules_changed |= _remove_extra_keys(citation, supported)
    if rules_changed:
        row[_FIELD_RULES] = json.dumps(rules, ensure_ascii=False)
    return changed or rules_changed
