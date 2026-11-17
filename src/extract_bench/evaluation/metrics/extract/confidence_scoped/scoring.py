from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from extract_bench.evaluation.metrics.extract.array_record_match_metric import unwrap_value
from extract_bench.evaluation.metrics.extract.unified_evidence_metric import ValueCell
from extract_bench.schemas.extract_output import FieldCitation
from extract_bench.test_cases.schema import ExtractFieldTestRule

from .paths import iter_field_values, path_without_indices, schema_path_exists
from .types import ConfidenceFieldRow


def clamp_confidence(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return max(0.0, min(1.0, float(value))) if math.isfinite(value) else None


def _citation_reference_text(citation: FieldCitation) -> str:
    if citation.reference_text is not None:
        return citation.reference_text
    metadata = citation.metadata
    if isinstance(metadata, dict):
        for key in ("reference_text", "matching_text", "text"):
            value = metadata.get(key)
            if isinstance(value, str):
                return value
        entries = metadata.get("citation")
        if isinstance(entries, list) and len(entries) > 0 and isinstance(entries[0], dict):
            return str(entries[0].get("matching_text") or "")
    return ""


def value_path_prefix(data: Any) -> str:
    """Translate Value F1 cell paths back to the corresponding input paths."""
    return "value." if unwrap_value(data) is not data else ""


def build_confidence_field_rows(
    *,
    extracted_data: Any,
    expected_output: Any,
    field_rules: Sequence[ExtractFieldTestRule],
    field_citations: Sequence[FieldCitation],
    data_schema: dict[str, Any] | None,
    value_cells: Sequence[ValueCell],
) -> list[ConfidenceFieldRow]:
    """Join confidence signals to the cells Value F1 actually graded."""
    if len(value_cells) == 0 or not any(clamp_confidence(c.confidence) is not None for c in field_citations):
        return []
    pred_prefix = value_path_prefix(extracted_data)
    gt_prefix = value_path_prefix(expected_output)
    by_path = {cell.pred_path: cell for cell in value_cells if cell.pred_path is not None}
    verified = {rule.field_path for rule in field_rules if rule.verified}
    predictions = dict(iter_field_values(extracted_data, include_containers=True))
    expected = dict(iter_field_values(expected_output, include_containers=True))
    rows: dict[str, ConfidenceFieldRow] = {}
    for citation in field_citations:
        score = clamp_confidence(citation.confidence)
        cell_path = citation.field_path.removeprefix(pred_prefix)
        cell = by_path.get(cell_path)
        if score is None or cell is None:
            continue
        path = pred_prefix + cell_path
        if path not in predictions:
            continue
        current = rows.get(path)
        if current is not None and current.confidence is not None and current.confidence >= score:
            continue
        gt_path = gt_prefix + cell.gt_path if cell.gt_path is not None else None
        field_path = path if gt_path is None else gt_path
        rows[path] = ConfidenceFieldRow(
            field_path=field_path,
            field_pattern=path_without_indices(field_path),
            predicted_path=path,
            expected_value=expected.get(gt_path) if gt_path is not None else None,
            predicted_value=predictions[path],
            confidence=score,
            correct=cell.value_correct,
            comparison_score=float(cell.value_correct),
            comparison_mode="value_f1",
            verified=gt_path in verified,
            alignment_mode="value_f1" if gt_path is not None else "unmatched_prediction",
            alignment_score=float(gt_path is not None),
            cited_text=_citation_reference_text(citation),
            schema_valid=data_schema is None or len(data_schema) == 0 or schema_path_exists(data_schema, path),
            matched_gt_path=gt_path if gt_path is not None else "",
        )
    return list(rows.values())
