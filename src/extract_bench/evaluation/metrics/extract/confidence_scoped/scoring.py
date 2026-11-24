from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence, Set
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


def confidence_signals_by_path(
    extracted_data: Any,
    field_citations: Sequence[FieldCitation],
    value_cells: Sequence[ValueCell],
) -> dict[str, tuple[float, FieldCitation]]:
    """Resolve confidence for emitted cells, retaining the source citation.

    An opaque scalar array uses its native confidence when available, otherwise
    the minimum available element confidence. Duplicate citations for one element
    keep the existing maximum-score policy before aggregation. Source citations
    remain unchanged; their geometry is not promoted to the array. A cell with
    no direct or element-derived score inherits its nearest scored object ancestor.
    """
    prefix = value_path_prefix(extracted_data)
    predictions = dict(iter_field_values(extracted_data, include_containers=True))
    signals: dict[str, tuple[float, FieldCitation]] = {}
    for citation in field_citations:
        score = clamp_confidence(citation.confidence)
        path = citation.field_path
        if path not in predictions:
            path = prefix + path.removeprefix(prefix)
        if score is None or path not in predictions:
            continue
        current = signals.get(path)
        if current is None or score > current[0]:
            signals[path] = (score, citation)

    array_paths = {
        prefix + cell.pred_path
        for cell in value_cells
        if cell.pred_path is not None
        and isinstance(value := predictions.get(prefix + cell.pred_path), list)
        and all(not isinstance(element, (dict, list)) for element in value)
    }
    derived: dict[str, tuple[float, FieldCitation]] = {}
    for path, signal in signals.items():
        parent, bracket, index = path.rpartition("[")
        if bracket != "[" or not index.endswith("]") or not index[:-1].isdigit():
            continue
        if parent not in array_paths or parent in signals:
            continue
        current = derived.get(parent)
        if current is None or signal[0] < current[0]:
            derived[parent] = signal
    resolved = signals | derived
    for cell in value_cells:
        if cell.pred_path is None:
            continue
        path = prefix + cell.pred_path
        if path in resolved or path not in predictions:
            continue
        # Walk actual ancestors, bounded by path depth rather than the number
        # of scored objects. Array containers do not supply object confidence.
        ancestor = path
        while "." in ancestor:
            ancestor = ancestor.rsplit(".", 1)[0]
            if ancestor in signals and isinstance(predictions.get(ancestor), dict):
                resolved[path] = signals[ancestor]
                break
    return resolved


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
    # Confidence may come from another path; evidence text must still belong
    # to the emitted cell, including citations with no confidence of their own.
    cited_text_by_path: dict[str, str] = {}
    for citation in field_citations:
        if len(text := _citation_reference_text(citation)) > 0:
            cited_text_by_path.setdefault(citation.field_path.removeprefix(pred_prefix), text)
    rows: dict[str, ConfidenceFieldRow] = {}
    for signal_path, (score, citation) in confidence_signals_by_path(
        extracted_data,
        field_citations,
        value_cells,
    ).items():
        cell_path = signal_path.removeprefix(pred_prefix)
        cell = by_path.get(cell_path)
        if cell is None:
            continue
        path = pred_prefix + cell_path
        if path not in predictions:
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
            cited_text=(
                _citation_reference_text(citation)
                if citation.field_path.removeprefix(pred_prefix) == cell_path
                else cited_text_by_path.get(cell_path, "")
            ),
            schema_valid=data_schema is None or len(data_schema) == 0 or schema_path_exists(data_schema, path),
            matched_gt_path=gt_path if gt_path is not None else "",
        )
    return list(rows.values())


def augrc_groups(
    value_cells: Sequence[ValueCell],
    emitted_paths: Set[str],
    signals: Mapping[str, tuple[float, FieldCitation]],
    pred_prefix: str,
    expected_output: Any,
) -> list[tuple[float, int, int]]:
    """Count evaluated fields and errors at each exact score in one document."""
    counts: dict[float, list[int]] = defaultdict(lambda: [0, 0])
    represented_records: set[str] = set()
    omitted: set[str] = set()
    for cell in value_cells:
        path = pred_prefix + cell.pred_path if cell.pred_path is not None else None
        if path is None or path not in emitted_paths:
            if cell.gt_path is not None:
                omitted.add(cell.gt_path)
            continue
        score = signals[path][0] if path in signals else 0.0
        counts[score][0] += 1
        counts[score][1] += int(not cell.value_correct)
        if cell.gt_path is not None:
            record = cell.gt_path
            while "]" in record:
                parent, _, _ = record.rpartition("]")
                key = parent + "]"
                if key in represented_records:
                    break
                represented_records.add(key)
                record = parent

    # Credit a missing blank within a represented record. An entirely absent
    # document or dropped array record contributes no synthetic ranked fields.
    if len(omitted) > 0 and len(counts) > 0:
        gt_prefix = value_path_prefix(expected_output)
        for gt_path, value in iter_field_values(expected_output):
            path = gt_path.removeprefix(gt_prefix)
            if path not in omitted or not (value is None or isinstance(value, str) and value.strip() == ""):
                continue
            record, bracket, _ = path.rpartition("]")
            if bracket == "" or record + "]" in represented_records:
                counts[0.0][0] += 1
    return [(score, total, errors) for score, (total, errors) in counts.items()]
