"""Fail-closed lint for extract ground-truth sidecars (``<stem>.test.json``).

Every extract GT family (synthetic long lists, the schema pipeline, extract-first
forms) writes its own sidecar, and each one has shipped a shape the bench only
papered over later. The public corpus audit (ExtractBench issue #5) found four
docs whose ``expected_output`` carried keys their own ``data_schema`` never
declared, one ``number`` leaf shipped as the string ``"2029"``, and 45 docs on
the legacy ``test_rules`` list instead of the ``_field_rules`` dict. Each of
those is decidable from the payload alone, so this module decides it once, at
write time, instead of after the corpus has shipped.

A payload passes when:

1. ``expected_output`` conforms to ``data_schema`` — no undeclared key, no
   scalar of a type the schema does not allow. These are the same two gates the
   code-gen extractor applies to provider output (``table_codegen/schema_utils``);
   they were simply never pointed at the ground truth.
2. Field rules live in exactly one container, the ``_field_rules`` dict keyed by
   field path. Every entry validates as an ``ExtractFieldTestRule`` and its path
   resolves in both the schema and the ground truth (a null ancestor in the
   ground truth counts as resolved: the leaf is vacuously absent).
3. Row identity rides beside the schema as ``_eval_row_identity``. It must not
   sit inside ``data_schema`` (that object reaches every provider) and must not
   use the inert legacy spelling ``_repeated_structure`` (nothing reads it).

The lint is pure — parsed payload in, findings out — so the same function gates
a writer (refuse the file), a tree (``extract-bench dataset lint``) and an export.
Evidence values are deliberately NOT compared to ``expected_output``: an evidence
entry is an *accepted reading* and may legitimately differ from the canonical
value, which is why the loader's drift check is a warning and not a gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from extract_bench.inference.providers.extract.table_codegen.schema_utils import (
    effective_schema,
    invalid_output_values,
    resolve_refs,
    unknown_output_fields,
)
from extract_bench.test_cases.extract_field_paths import parse_field_path
from extract_bench.test_cases.schema import ExtractFieldTestRule

__all__ = [
    "FIELD_RULES_KEY",
    "LEGACY_ROW_IDENTITY_KEY",
    "LEGACY_RULES_KEY",
    "ROW_IDENTITY_KEY",
    "SCHEMA_IDENTITY_KEY",
    "LintFinding",
    "lint_extract_sidecar",
]

FIELD_RULES_KEY = "_field_rules"
LEGACY_RULES_KEY = "test_rules"
ROW_IDENTITY_KEY = "_eval_row_identity"
LEGACY_ROW_IDENTITY_KEY = "_repeated_structure"
SCHEMA_IDENTITY_KEY = "repeated_structure"

# Schema keys whose *children* are user-named maps (property names, $defs
# names), not schema nodes — a ``repeated_structure`` key there is a field name.
_NAME_MAP_KEYS = frozenset({"properties", "$defs", "definitions"})


@dataclass(frozen=True)
class LintFinding:
    """One defect: a stable dotted ``code``, where it is, and why it fails."""

    code: str
    path: str
    message: str

    def render(self) -> str:
        return f"{self.code} {self.path}: {self.message}"


def lint_extract_sidecar(payload: dict[str, Any], *, allow_legacy_rules: bool = False) -> list[LintFinding]:
    """Return every finding for one parsed sidecar; an empty list means it ships.

    ``allow_legacy_rules`` tolerates the ``test_rules`` list container — for
    migration tooling that reads legacy files, never for a writer.
    """
    schema = payload.get("data_schema")
    if not isinstance(schema, dict):
        return [LintFinding("schema.missing", "data_schema", "must be a JSON Schema object")]
    resolved: dict[str, Any] = resolve_refs(schema)

    findings: list[LintFinding] = []
    _scan_schema_identity(schema, "data_schema", False, findings)

    output = payload.get("expected_output")
    if not isinstance(output, dict):
        findings.append(LintFinding("output.missing", "expected_output", "must be an object"))
        output = None
    else:
        findings.extend(
            LintFinding("output.unknown_field", f"expected_output{path}", "key is not declared by data_schema")
            for path in unknown_output_fields(output, resolved)
        )
        for message in invalid_output_values(output, resolved):
            where, _, detail = message.partition(": ")
            findings.append(LintFinding("output.type_mismatch", f"expected_output{where.rstrip('/')}", detail))

    rules, container_findings = _rules_container(payload, allow_legacy_rules)
    findings.extend(container_findings)
    findings.extend(_lint_rules(rules, resolved, output))
    findings.extend(_lint_row_identity(payload, resolved))
    return findings


# ---------------------------------------------------------------------------
# rules


def _rules_container(
    payload: dict[str, Any], allow_legacy_rules: bool
) -> tuple[list[tuple[str, Any]], list[LintFinding]]:
    """Every (field_path, rule body) pair, whichever container holds them, plus
    findings about the container itself."""
    field_rules = payload.get(FIELD_RULES_KEY)
    legacy_rules = payload.get(LEGACY_RULES_KEY)
    findings: list[LintFinding] = []
    if field_rules is not None and legacy_rules is not None:
        findings.append(
            LintFinding(
                "rules.both_containers",
                LEGACY_RULES_KEY,
                f"sidecar carries both {FIELD_RULES_KEY} and {LEGACY_RULES_KEY}; keep the dict, drop the list",
            )
        )
    if field_rules is not None:
        if not isinstance(field_rules, dict):
            findings.append(
                LintFinding(
                    "rules.not_object",
                    FIELD_RULES_KEY,
                    f"must be an object keyed by field_path, got {type(field_rules).__name__}",
                )
            )
            return [], findings
        return list(field_rules.items()), findings
    if legacy_rules is None:
        return [], findings
    if not allow_legacy_rules:
        findings.append(
            LintFinding(
                "rules.legacy_container",
                LEGACY_RULES_KEY,
                f"legacy list container; rules ship as the {FIELD_RULES_KEY} dict keyed by field_path",
            )
        )
    if not isinstance(legacy_rules, list):
        findings.append(
            LintFinding("rules.not_list", LEGACY_RULES_KEY, f"must be a list, got {type(legacy_rules).__name__}")
        )
        return [], findings
    rules: list[tuple[str, Any]] = []
    for index, rule in enumerate(legacy_rules):
        where = f"{LEGACY_RULES_KEY}[{index}]"
        if not isinstance(rule, dict):
            findings.append(LintFinding("rules.entry_not_object", where, f"got {type(rule).__name__}"))
            continue
        field_path = rule.get("field_path")
        if not isinstance(field_path, str) or not field_path:
            findings.append(LintFinding("rules.entry_missing_path", where, "rule has no field_path"))
            continue
        rules.append((field_path, {k: v for k, v in rule.items() if k != "field_path"}))
    return rules, findings


def _lint_rules(
    rules: list[tuple[str, Any]], resolved_schema: dict[str, Any], output: dict[str, Any] | None
) -> list[LintFinding]:
    findings: list[LintFinding] = []
    for field_path, body in rules:
        where = f"{FIELD_RULES_KEY}[{field_path!r}]"
        if not isinstance(body, dict):
            findings.append(LintFinding("rules.entry_not_object", where, f"got {type(body).__name__}"))
            continue
        rule_type = body.get("type")
        if rule_type not in (None, "extract_field"):
            findings.append(
                LintFinding(
                    "rules.entry_type", where, f"{FIELD_RULES_KEY} holds extract_field rules only, got {rule_type!r}"
                )
            )
            continue
        try:
            ExtractFieldTestRule.model_validate({**body, "type": "extract_field", "field_path": field_path})
        except ValidationError as exc:
            first = exc.errors()[0]
            loc = ".".join(str(part) for part in first.get("loc", ()))
            findings.append(LintFinding("rules.invalid", where, f"{loc}: {first.get('msg')}"))
            continue
        try:
            tokens = parse_field_path(field_path)
        except ValueError as exc:
            findings.append(LintFinding("rules.path_invalid", where, str(exc)))
            continue
        if _schema_node_at(resolved_schema, tokens) is None:
            findings.append(
                LintFinding("rules.path_not_in_schema", where, "field_path does not resolve in data_schema")
            )
            continue  # a path the schema never declares cannot be in the GT either; report the root cause once
        if output is not None and not _output_has_path(output, tokens):
            findings.append(
                LintFinding("rules.path_not_in_output", where, "field_path does not resolve in expected_output")
            )
    return findings


def _output_has_path(output: dict[str, Any], tokens: list[str | int]) -> bool:
    """True when the path exists in the ground truth, or dead-ends at a null
    ancestor (an absent optional object/array whose leaves are vacuously null)."""
    cursor: Any = output
    for token in tokens:
        if cursor is None:
            return True
        if isinstance(token, int):
            if not isinstance(cursor, list) or not 0 <= token < len(cursor):
                return False
            cursor = cursor[token]
        else:
            if not isinstance(cursor, dict) or token not in cursor:
                return False
            cursor = cursor[token]
    return True


# ---------------------------------------------------------------------------
# schema navigation


def _schema_types(node: dict[str, Any]) -> list[str]:
    raw = node.get("type")
    return [t for t in (raw if isinstance(raw, list) else [raw]) if isinstance(t, str)]


def _schema_node_at(resolved_schema: dict[str, Any], tokens: list[str | int]) -> dict[str, Any] | None:
    """The effective schema node a field path lands on, or None when the schema
    declares nothing there. Integer tokens descend into ``items``; string tokens
    into ``properties`` or a schema-valued ``additionalProperties``."""
    node = effective_schema(resolved_schema)
    for token in tokens:
        if isinstance(token, int):
            items = node.get("items")
            if not isinstance(items, dict):
                return None
            node = effective_schema(items)
            continue
        properties = node.get("properties") or {}
        if token in properties:
            node = effective_schema(properties[token])
        elif isinstance(node.get("additionalProperties"), dict):
            node = effective_schema(node["additionalProperties"])
        else:
            return None
    return node


# ---------------------------------------------------------------------------
# row identity


def _scan_schema_identity(node: Any, pointer: str, inside_name_map: bool, findings: list[LintFinding]) -> None:
    """Flag a ``repeated_structure`` block anywhere in the schema tree, skipping
    the user-named maps where that string would be a field name."""
    if isinstance(node, dict):
        for key, value in node.items():
            child = f"{pointer}.{key}"
            if not inside_name_map and key == SCHEMA_IDENTITY_KEY:
                findings.append(
                    LintFinding(
                        "identity.in_schema",
                        child,
                        f"row identity must ship beside the schema as {ROW_IDENTITY_KEY}, not inside data_schema",
                    )
                )
                continue
            _scan_schema_identity(value, child, not inside_name_map and key in _NAME_MAP_KEYS, findings)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _scan_schema_identity(value, f"{pointer}[{index}]", False, findings)


def _lint_row_identity(payload: dict[str, Any], resolved_schema: dict[str, Any]) -> list[LintFinding]:
    findings: list[LintFinding] = []
    if LEGACY_ROW_IDENTITY_KEY in payload:
        findings.append(
            LintFinding(
                "identity.legacy_key",
                LEGACY_ROW_IDENTITY_KEY,
                f"inert legacy key that no consumer reads; move the block to {ROW_IDENTITY_KEY} or drop it",
            )
        )
    block = payload.get(ROW_IDENTITY_KEY)
    if block is None:
        return findings
    if not isinstance(block, dict):
        findings.append(LintFinding("identity.not_object", ROW_IDENTITY_KEY, f"got {type(block).__name__}"))
        return findings
    for array_path, spec in block.items():
        where = f"{ROW_IDENTITY_KEY}[{array_path!r}]"
        if not isinstance(spec, dict):
            findings.append(LintFinding("identity.entry_not_object", where, f"got {type(spec).__name__}"))
            continue
        try:
            node = _schema_node_at(resolved_schema, parse_field_path(array_path))
        except ValueError as exc:
            findings.append(LintFinding("identity.path_invalid", where, str(exc)))
            continue
        if node is None:
            findings.append(LintFinding("identity.array_missing", where, "path does not resolve in data_schema"))
            continue
        if "array" not in _schema_types(node):
            findings.append(LintFinding("identity.not_array", where, "path is not an array in data_schema"))
            continue
        keys = spec.get("identity_key")
        if keys is None:
            continue
        items = node.get("items")
        row_properties = (effective_schema(items).get("properties") or {}) if isinstance(items, dict) else {}
        for key in [keys] if isinstance(keys, str) else list(keys):
            if key not in row_properties:
                findings.append(
                    LintFinding(
                        "identity.key_missing", where, f"identity_key {key!r} is not a property of the row object"
                    )
                )
    return findings
