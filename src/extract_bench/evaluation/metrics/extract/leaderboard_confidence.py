"""Pooled confidence metrics for ExtractBench leaderboard submissions."""

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence, Set
from typing import Any

from extract_bench.evaluation.metrics.extract.unified_evidence_metric import ValueCell
from extract_bench.schemas.evaluation import MetricValue
from extract_bench.schemas.extract_output import FieldCitation

from .confidence_scoped.paths import iter_field_values
from .confidence_scoped.scoring import value_path_prefix

AUGRC_GROUPS = "augrc_groups"


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


def pooled_leaderboard_confidence(coverage_metrics: Iterable[MetricValue]) -> dict[str, float]:
    """Pool document count metadata across the splits represented by one CSV row."""
    returned = scored = 0
    groups: dict[float, list[int]] = defaultdict(lambda: [0, 0])
    complete_ranks = True
    for metric in coverage_metrics:
        data = metric.metadata
        if data.get("coverage_scope") != "emitted_value_f1_cells":
            if AUGRC_GROUPS not in data and data.get("emitted_leaf_count", 0) > 0:
                # Older scalar-leaf counters cannot be mixed with Value F1 cells.
                return {}
            continue
        returned += data["emitted_leaf_count"]
        scored += data["scored_leaf_count"]
        if AUGRC_GROUPS not in data:
            complete_ranks = False
            continue
        for score, total, errors in data[AUGRC_GROUPS]:
            groups[score][0] += total
            groups[score][1] += errors

    result = {"Confidence_Coverage": 100 * scored / returned} if returned > 0 else {}
    if complete_ranks and len(groups) > 0:
        # At rank j within a tie of size m, the expected new errors are j*e/m.
        # Summing over j=1..m gives e*(m+1)/2; keep twice the sum integral.
        prior_errors = twice_integral = n = 0
        for score in sorted(groups, reverse=True):
            total, errors = groups[score]
            twice_integral += 2 * total * prior_errors + errors * (total + 1)
            prior_errors += errors
            n += total
        result["Confidence_AUGRC"] = twice_integral / (2 * n * n)
    return result
