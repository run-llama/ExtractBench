"""Field grounding metrics for extract pipeline outputs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from extract_bench.evaluation.metrics.field_grounding.core import (
    BBox,
    ValueComparison,
    compute_standard_iou_metrics,
)
from extract_bench.evaluation.metrics.field_grounding.evidence_comparator import (
    compare_evidence_value,
    parse_match_by_keys,
    stable_value_key,
)
from extract_bench.evaluation.metrics.field_grounding.value_compare import (
    compare_array_against_rule,
    compare_value_against_rule,
)
from extract_bench.schemas.evaluation import MetricValue
from extract_bench.test_cases.extract_field_paths import get_path, parse_field_path
from extract_bench.test_cases.schema import ExtractFieldTestRule, iter_rule_evidence

_MISSING = object()


def compute_extract_field_grounding_metrics(
    *,
    extracted_data: Any,
    field_rules: list[ExtractFieldTestRule],
    field_citations: list[Any],
    data_schema: dict[str, Any] | None = None,
    skip_field_paths: Iterable[str] = (),
) -> list[MetricValue]:
    """Compute v0.2 evidence metrics for extract outputs.

    ``skip_field_paths`` lists rule ``field_path`` values that are known not
    to be scorable against the current ``extracted_data`` shape (typically
    scalar rules excluded after a per_table_row list-unwrap). They are
    dropped from the evidence-metric denominators.

    ``data_schema`` is accepted for call-site compatibility; v0.2 value matching
    reads comparators off each rule rather than the schema object.
    """
    if not field_rules:
        return []

    v02_rules = [rule for rule in field_rules if rule.evidence is not None]
    return _compute_v02_evidence_metrics(
        v02_rules,
        extracted_data,
        field_citations,
        skip_field_paths=skip_field_paths,
    )


@dataclass(frozen=True)
class _V02FamilyCitations:
    """Citation-derived facts shared by every rule of one field pattern.

    All of these depend only on the rule's field pattern, and long-list docs
    repeat the same pattern across thousands of leaf rules — computing them
    per rule made the evidence loop quadratic in family size (40+ minutes on
    a 37k-rule document).
    """

    group: str
    page_qual: bool
    cit_pages: frozenset[int]
    bbox_cit_pages: frozenset[int]
    pred_boxes_by_page: dict[int, list[BBox]]
    exact_count: int
    coarse_count: int
    descendant_count: int


def _build_v02_family_citations(
    rule_pattern: tuple[str | None, ...],
    citations_by_pattern: dict[tuple[str | None, ...], list[Any]],
) -> _V02FamilyCitations:
    exact_cits = citations_by_pattern.get(rule_pattern, [])
    # Coarse parent prefix-walk: take only the NEAREST non-empty parent
    # prefix's citations, not every ancestor. The nearest parent is the
    # most specific cite a pipeline could plausibly mean for this leaf;
    # broader top-level cites would over-credit M2a coverage.
    coarse_cits: list[Any] = []
    for prefix_len in range(len(rule_pattern) - 1, 0, -1):
        candidates = citations_by_pattern.get(rule_pattern[:prefix_len], [])
        if candidates:
            coarse_cits = candidates
            break

    # Descendant walk: for array-shaped rules whose pattern is a strict
    # prefix of any per-leaf citation pattern, include those leaf citations.
    # Without this, an array rule with pattern ``('grants',)`` never sees
    # pipeline citations like ``('grants', None, 'recipient_name')`` even
    # though the pipeline IS correctly grounding every row's leaf — the
    # pipeline just doesn't emit a synthetic citation at the array level.
    # Pipelines that DO emit array-level citations contribute via
    # ``exact_cits`` and additionally via this walk when leaf citations
    # also exist; the two are unioned, not gated.
    descendant_cits: list[Any] = []
    for pattern, cits in citations_by_pattern.items():
        if len(pattern) > len(rule_pattern) and pattern[: len(rule_pattern)] == rule_pattern:
            descendant_cits.extend(cits)

    group = ".".join("[]" if token is None else token for token in rule_pattern)

    cit_pages: set[int | None] = set()
    for cits in (exact_cits, coarse_cits, descendant_cits):
        cit_pages.update(_as_int(getattr(cit, "page", None)) for cit in cits)
    cit_pages.discard(None)

    bbox_cit_pages: set[int] = set()
    for cits in (exact_cits, descendant_cits):
        for cit in cits:
            if getattr(cit, "bbox", None) is None:
                continue
            page = _as_int(getattr(cit, "page", None))
            if page is not None:
                bbox_cit_pages.add(page)

    pred_boxes_by_page: dict[int, list[BBox]] = defaultdict(list)
    for citation in exact_cits:
        page = _as_int(getattr(citation, "page", None))
        bbox = _as_xywh(getattr(citation, "bbox", None))
        if page is not None and bbox is not None:
            pred_boxes_by_page[page].append(BBox(page=page, bbox=bbox, group=group))

    return _V02FamilyCitations(
        group=group,
        page_qual=bool(exact_cits or coarse_cits or descendant_cits),
        cit_pages=frozenset(page for page in cit_pages if page is not None),
        bbox_cit_pages=frozenset(bbox_cit_pages),
        pred_boxes_by_page=dict(pred_boxes_by_page),
        exact_count=len(exact_cits),
        coarse_count=len(coarse_cits),
        descendant_count=len(descendant_cits),
    )


def _compute_v02_evidence_metrics(
    field_rules: list[ExtractFieldTestRule],
    extracted_data: Any,
    field_citations: list[Any],
    *,
    skip_field_paths: Iterable[str] = (),
) -> list[MetricValue]:
    """v0.2 evidence-list value and page-grounded metrics.

    Pass/fail does not depend on bbox IoU. For bbox-bearing evidence entries we
    still compute alignment diagnostics so dashboards can analyze localization
    quality separately from value/page correctness.

    Rules with ``evidence_required=False`` are excluded from denominators. If a
    rule has ``evidence=[]`` while evidence is required, emitting no value (or
    null) passes value-only grading; emitting any concrete value fails because
    there is no accepted evidence value to match.
    """
    skip_set = set(skip_field_paths)
    eligible = [
        rule
        for rule in field_rules
        if rule.evidence_required and not _has_stray_tag(rule) and rule.field_path not in skip_set and rule.verified
    ]
    if not eligible:
        return []

    citations_by_pattern: dict[tuple[str | None, ...], list[Any]] = defaultdict(list)
    for citation in field_citations:
        cit_field_path = getattr(citation, "field_path", None)
        if not cit_field_path:
            continue
        if _as_int(getattr(citation, "page", None)) is None:
            continue
        cit_pattern = _field_pattern(cit_field_path)
        if cit_pattern is None:
            continue
        citations_by_pattern[cit_pattern].append(citation)

    # Row identity alignment for match_by array families: per-row leaf rules
    # are graded against the predicted row with the matching identity, not the
    # row at the same position, so reordered or dropped rows fail only
    # themselves. Built from the full rule list — the parent rule contributes
    # alignment context even when it is itself unverified/ungraded.
    row_alignments = _build_match_by_alignments(field_rules, extracted_data)

    value_pass = 0
    page_pass = 0
    page_pass_covered = 0
    page_qualified = 0
    bbox_qualified = 0
    bbox_iou_sum = 0.0
    bbox_iou_count = 0
    bbox_covered_pass = 0
    bbox_iou_pass_sum = 0.0
    # Denominator for the bbox_* metrics: leaves whose GT evidence carries a
    # bounding box. Grounding is undefined for a leaf with no GT bbox, so such
    # leaves are excluded from the bbox metrics (and a doc with none emits no
    # bbox metric at all) instead of being scored 0 and averaged in.
    bbox_gt_total = 0

    rule_results: list[dict[str, Any]] = []
    family_citations: dict[tuple[str | None, ...], _V02FamilyCitations] = {}

    for rule in eligible:
        rule_pattern = _field_pattern(rule.field_path)
        if rule_pattern is None:
            continue
        evidence_entries = iter_rule_evidence(rule)

        family = family_citations.get(rule_pattern)
        if family is None:
            family = _build_v02_family_citations(rule_pattern, citations_by_pattern)
            family_citations[rule_pattern] = family

        page_qual = family.page_qual
        # bbox coverage is only meaningful when the predicted bbox is on a page
        # that matches some evidence page. A bbox emitted on the wrong page is
        # not "the correct bbox at the correct position" — it's a wrong-page
        # bbox, which should not count as covered. We require both:
        #   1. an exact-path or descendant-path citation with a non-null bbox,
        #   2. that citation's page intersects the evidence page set.
        # If no evidence carries a page, we cannot validate; leaf is uncovered.
        ev_pages = {ev.page for ev in evidence_entries if ev.page is not None}
        # A leaf is grounding-gradeable only when its GT evidence carries a bbox.
        gt_has_bbox = any(
            ev.page is not None and ev.bbox is not None and _as_xywh(ev.bbox) is not None for ev in evidence_entries
        )
        if gt_has_bbox:
            bbox_gt_total += 1
        bbox_qual = bool(ev_pages) and not ev_pages.isdisjoint(family.bbox_cit_pages)
        page_qualified += int(page_qual)
        if gt_has_bbox:
            bbox_qualified += int(bbox_qual)

        predictions = _prediction_values_for_v02_rule(extracted_data, rule, row_alignments)
        value_comparisons = [_compare_v02_prediction(rule, prediction) for prediction in predictions]
        best_value_comparison = _best_comparison(value_comparisons)
        empty_evidence_null_pass = not evidence_entries and (
            not predictions or all(prediction is None for prediction in predictions)
        )
        value_pass_for_rule = bool(best_value_comparison and best_value_comparison.passed) or empty_evidence_null_pass
        value_pass += int(value_pass_for_rule)

        # Page-pass logic. The metric "is the value at the right page" is
        # vacuously true when there's no positive page claim to verify.
        # No positive page claim means ``evidence_with_value`` is empty —
        # which covers two cases:
        #   1. Evidence list is itself empty (no entries at all).
        #   2. Evidence list is non-empty but every entry has ``value=None``
        #      (the gold asserts "no value here", and a null assertion
        #      doesn't bind to a page).
        # In both cases, when ``value_pass`` succeeds, page_pass passes
        # vacuously. The positive-evidence path only runs when at least one
        # entry has a non-null value; mixed evidence lists score against
        # the value-bearing entries only.
        evidence_with_value = [ev for ev in evidence_entries if ev.value is not None]
        page_pass_for_rule = False
        if value_pass_for_rule and not evidence_with_value:
            # Vacuous: no positive page claim. Pass on any value_pass.
            page_pass_for_rule = True
        elif page_qual and value_pass_for_rule and evidence_with_value:
            page_pass_for_rule = any(ev.page in family.cit_pages for ev in evidence_with_value)

        if page_pass_for_rule:
            page_pass += 1
            if page_qual:
                page_pass_covered += 1

        bbox_iou = _best_v02_bbox_iou(evidence_entries, family.pred_boxes_by_page, family.group)
        if bbox_iou is not None:
            bbox_iou_sum += bbox_iou
            bbox_iou_count += 1

        # Value-gated bbox metrics. A bbox is only meaningful when the value is
        # also right; a wrong value with a perfectly placed bbox is still a
        # wrong extraction. Both metrics use ``total`` as denominator so wrong
        # values, missing bboxes, and wrong-page bboxes all contribute 0.
        bbox_covered_pass_for_rule = bool(value_pass_for_rule and bbox_qual)
        bbox_iou_pass_for_rule = float(bbox_iou) if (value_pass_for_rule and bbox_iou is not None) else 0.0
        if gt_has_bbox:
            bbox_covered_pass += int(bbox_covered_pass_for_rule)
            bbox_iou_pass_sum += bbox_iou_pass_for_rule

        rule_results.append(
            {
                "field_path": rule.field_path,
                "path": rule.field_path,
                "evidence_count": len(evidence_entries),
                "value_pass": value_pass_for_rule,
                "page_pass": page_pass_for_rule,
                "page_qualified": page_qual,
                "bbox_qualified": bbox_qual,
                "bbox_iou": bbox_iou,
                "bbox_covered_pass": bbox_covered_pass_for_rule,
                "bbox_iou_pass": bbox_iou_pass_for_rule,
                "exact_citation_count": family.exact_count,
                "coarse_citation_count": family.coarse_count,
                "descendant_citation_count": family.descendant_count,
                "mode": getattr(best_value_comparison, "mode", "null_empty" if empty_evidence_null_pass else "missing"),
                "reason": (
                    "pass" if value_pass_for_rule else getattr(best_value_comparison, "reason", "missing_prediction")
                ),
            }
        )

    total = len(eligible)
    base_meta: dict[str, Any] = {
        "total": total,
        "verified_only": True,
        "rule_results": rule_results,
        "skipped_field_paths": sorted(skip_set),
        "match_by_row_alignment": {family: len(family_map) for family, family_map in row_alignments.items()},
    }

    metrics: list[MetricValue] = []

    metrics.append(
        MetricValue(
            metric_name="extract_evidence_value_pass_rate",
            value=value_pass / total if total > 0 else 0.0,
            metadata={
                **base_meta,
                "tp": value_pass,
                "fp": total - value_pass,
                "fn": 0,
                "denominator": "graded_verified_v02_rules",
            },
        )
    )
    metrics.append(
        MetricValue(
            metric_name="extract_evidence_page_pass_rate",
            value=page_pass / total if total > 0 else 0.0,
            metadata={
                **base_meta,
                "tp": page_pass,
                "fp": total - page_pass,
                "fn": 0,
                "denominator": "all_verified_rules",
            },
        )
    )
    metrics.append(
        MetricValue(
            metric_name="extract_evidence_page_covered_pass_rate",
            value=page_pass_covered / page_qualified if page_qualified > 0 else 0.0,
            metadata={
                **base_meta,
                "tp": page_pass_covered,
                "fp": page_qualified - page_pass_covered,
                "fn": 0,
                "denominator": "page_qualified_rules",
                "covered_total": page_qualified,
            },
        )
    )
    # bbox_* metrics are grounding metrics: they are defined only over leaves
    # whose GT evidence carries a bounding box (``bbox_gt_total``). A document
    # with no bbox-bearing GT emits none of them, so it is excluded from their
    # averages instead of being scored 0 and dragging the dataset number down.
    # A leaf whose GT *does* carry a bbox but whose prediction emits none still
    # counts as a 0 within these denominators -- a real grounding miss.
    if bbox_gt_total > 0:
        metrics.append(
            MetricValue(
                metric_name="extract_evidence_bbox_IOU_alignment",
                value=bbox_iou_sum / bbox_iou_count if bbox_iou_count > 0 else 0.0,
                metadata={
                    **base_meta,
                    "tp": 0,
                    "fp": 0,
                    "fn": 0,
                    "diagnostic": True,
                    "bbox_iou_count": bbox_iou_count,
                    "bbox_gt_total": bbox_gt_total,
                    "definition": "diagnostic_mean_best_iou_for_bbox_bearing_evidence",
                },
            )
        )
        metrics.append(
            MetricValue(
                metric_name="extract_evidence_bbox_coverage",
                value=bbox_qualified / bbox_gt_total,
                metadata={
                    **base_meta,
                    "tp": bbox_qualified,
                    "fp": bbox_gt_total - bbox_qualified,
                    "fn": 0,
                    "denominator": "gt_bbox_bearing_rules",
                    "bbox_gt_total": bbox_gt_total,
                    "definition": "per_leaf_binary_any_matching_citation_with_bbox",
                },
            )
        )
        # Value-gated bbox metrics. ``bbox_covered_pass_rate`` is the binary joint
        # of value-pass AND bbox-coverage (the bbox is on a page that matches some
        # evidence page). ``bbox_iou_pass_rate`` is the IoU score when value passes
        # else 0, averaged over the bbox-bearing leaves. Both penalize wrong
        # values, missing bboxes, and wrong-page bboxes.
        metrics.append(
            MetricValue(
                metric_name="extract_evidence_bbox_covered_pass_rate",
                value=bbox_covered_pass / bbox_gt_total,
                metadata={
                    **base_meta,
                    "tp": bbox_covered_pass,
                    "fp": bbox_gt_total - bbox_covered_pass,
                    "fn": 0,
                    "denominator": "gt_bbox_bearing_rules",
                    "bbox_gt_total": bbox_gt_total,
                    "definition": "per_leaf_binary_value_pass_and_bbox_qualified",
                },
            )
        )
        metrics.append(
            MetricValue(
                metric_name="extract_evidence_bbox_IOU_pass_rate",
                value=bbox_iou_pass_sum / bbox_gt_total,
                metadata={
                    **base_meta,
                    "tp": 0,
                    "fp": 0,
                    "fn": 0,
                    "denominator": "gt_bbox_bearing_rules",
                    "bbox_gt_total": bbox_gt_total,
                    "bbox_iou_pass_sum": bbox_iou_pass_sum,
                    "definition": "per_leaf_iou_when_value_pass_else_zero",
                },
            )
        )
    return metrics



def _get_field_value(extracted_data: Any, field_path: str) -> Any:
    try:
        tokens = parse_field_path(field_path)
    except ValueError:
        return _MISSING
    return get_path(extracted_data, tokens, default=_MISSING)



def _has_stray_tag(rule: ExtractFieldTestRule) -> bool:
    tags = {tag.casefold() for tag in rule.tags}
    return "stray" in tags or "no_value" in tags or any(tag.endswith(":stray") for tag in tags)


def compute_failure_field_denominator(field_rules: Iterable[ExtractFieldTestRule]) -> int:
    """Count rules eligible under the v0.2 evidence-metric filter, matching
    ``_compute_v02_evidence_metrics``.

    Exposed for the evaluation runner: when inference fails entirely on a
    doc, the runner synthesises a zero-scored EvaluationResult so the doc
    still counts in macro averages. But the synthesised metrics have no
    ``tp/fp/fn`` metadata, so the *micro* aggregator silently drops them
    (0/0 contribution) and pipelines that crash on a subset of docs end up
    ranked higher than they deserve. The runner uses this denominator to
    project the failure as ``tp=0, fp=N, fn=0`` for pooled pass-rate
    metrics, making the doc count in micro too.
    """
    return sum(1 for rule in field_rules if rule.evidence_required and rule.verified and not _has_stray_tag(rule))


def _field_pattern(field_path: str) -> tuple[str | None, ...] | None:
    """Return a path pattern with array indices wildcarded.

    Exact index matching is too brittle for table extraction: if a provider
    skips one row, all later rows shift and would falsely fail. The source export's
    text metrics are field-family metrics, so `rows[3].amount` and
    `rows[4].amount` are compared within the same `rows[].amount` pool.
    """
    try:
        tokens = parse_field_path(field_path)
    except ValueError:
        return None
    return tuple(None if isinstance(token, int) else token for token in tokens)


# Pass-2 alignment is quadratic in the worst case; bound the work so a
# pathological doc (thousands of GT rows, none exactly matching) degrades to
# positional fallback instead of stalling evaluation.
_ALIGNMENT_PASS2_MAX_PAIRS = 250_000
_IDENTITY_COMBO_CAP = 16


def _build_match_by_alignments(
    field_rules: list[ExtractFieldTestRule],
    extracted_data: Any,
) -> dict[str, dict[int, int]]:
    """Map GT row indices to predicted row indices for ``match_by`` families.

    Exact-index leaf lookup is too brittle for long lists: one dropped or
    reordered row shifts every later index and falsely fails every subsequent
    per-row leaf rule (the same brittleness ``_field_pattern`` documents for
    the legacy field-family metrics). When an array's parent rule declares
    ``match_by:<keys>`` semantics — rows are identified by key, order is
    irrelevant — leaf rules are graded against the predicted row carrying the
    matching identity rather than the row at the same position.

    Returns ``{family_root_path: {gt_index: pred_index}}``. GT rows absent
    from a family map have no surviving counterpart in the prediction; their
    leaf rules grade as missing predictions — a genuine miss scoped to that
    row instead of an index-shift cascade. Families whose parent rule is not
    ``match_by`` (ordered/set/multiset) are not realigned.
    """
    alignments: dict[str, dict[int, int]] = {}
    for parent in field_rules:
        structural = parent.structural or ""
        if not structural.startswith("match_by:"):
            continue
        keys = parse_match_by_keys(structural.split(":", 1)[1])
        if not keys:
            continue
        pred_rows = _get_field_value(extracted_data, parent.field_path)
        if pred_rows is _MISSING or not isinstance(pred_rows, list):
            continue
        try:
            root_tokens = parse_field_path(parent.field_path)
        except ValueError:
            continue
        identities, row_indices = _collect_gt_row_identities(field_rules, parent, root_tokens, keys)
        if not row_indices:
            continue
        alignments[parent.field_path] = _align_family_rows(identities, row_indices, pred_rows, keys, parent.comparator)
    return alignments


def _collect_gt_row_identities(
    field_rules: list[ExtractFieldTestRule],
    parent: ExtractFieldTestRule,
    root_tokens: list[str | int],
    keys: list[str],
) -> tuple[dict[int, dict[str, list[Any]]], set[int]]:
    """Gather per-row identity-key values for one array family.

    Primary source is the per-leaf rules (``root[i].key``), whose evidence may
    carry multiple OR-acceptable surface forms. Cells with no leaf rule fall
    back to the parent rule's evidence row dict, which is index-aligned with
    ``expected_output`` by construction in the GT builder.
    """
    depth = len(root_tokens)
    identities: dict[int, dict[str, list[Any]]] = {}
    row_indices: set[int] = set()
    for rule in field_rules:
        try:
            tokens = parse_field_path(rule.field_path)
        except ValueError:
            continue
        if len(tokens) <= depth or tokens[:depth] != root_tokens:
            continue
        row_index = tokens[depth]
        if not isinstance(row_index, int):
            continue
        row_indices.add(row_index)
        if len(tokens) != depth + 2:
            continue
        leaf = tokens[depth + 1]
        if not isinstance(leaf, str) or leaf not in keys:
            continue
        variants = identities.setdefault(row_index, {}).setdefault(leaf, [])
        for entry in iter_rule_evidence(rule):
            if entry.value not in variants:
                variants.append(entry.value)
    for row_index, entry in enumerate(iter_rule_evidence(parent)):
        row_value = entry.value
        if not isinstance(row_value, dict):
            continue
        row_identity = identities.setdefault(row_index, {})
        for key in keys:
            if key not in row_identity and key in row_value:
                row_identity[key] = [row_value[key]]
    return identities, row_indices


def _identity_combos(row_identity: dict[str, list[Any]], keys: list[str]) -> list[tuple[str, ...]]:
    combos: list[tuple[str, ...]] = [()]
    for key in keys:
        combos = [combo + (stable_value_key(variant),) for combo in combos for variant in row_identity[key]][
            :_IDENTITY_COMBO_CAP
        ]
    return combos


def _align_family_rows(
    identities: dict[int, dict[str, list[Any]]],
    row_indices: set[int],
    pred_rows: list[Any],
    keys: list[str],
    comparator: Any,
) -> dict[int, int]:
    used: set[int] = set()
    aligned: dict[int, int] = {}

    # Pass 1: exact normalized identity tuples, GT order claims duplicate
    # identities in prediction order so relative order is preserved.
    pred_index: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for pred_pos, row in enumerate(pred_rows):
        if isinstance(row, dict):
            pred_index[tuple(stable_value_key(row.get(key)) for key in keys)].append(pred_pos)
    pending: list[int] = []
    for row_index in sorted(row_indices):
        row_identity = identities.get(row_index)
        if not row_identity or any(key not in row_identity for key in keys):
            pending.append(row_index)
            continue
        matched = False
        for combo in _identity_combos(row_identity, keys):
            for pred_pos in pred_index.get(combo, []):
                if pred_pos in used:
                    continue
                aligned[row_index] = pred_pos
                used.add(pred_pos)
                matched = True
                break
            if matched:
                break
        if not matched:
            pending.append(row_index)

    # Pass 2: comparator-based matching for leftovers — catches surface-form
    # drift the exact pass misses (case, punctuation, near-miss strings).
    leftover_preds = [
        pred_pos for pred_pos in range(len(pred_rows)) if pred_pos not in used and isinstance(pred_rows[pred_pos], dict)
    ]
    if pending and leftover_preds and len(pending) * len(leftover_preds) <= _ALIGNMENT_PASS2_MAX_PAIRS:
        still_pending: list[int] = []
        for row_index in pending:
            row_identity = identities.get(row_index) or {}
            known_keys = [key for key in keys if key in row_identity]
            matched_pred: int | None = None
            if known_keys:
                for pred_pos in leftover_preds:
                    if pred_pos in used:
                        continue
                    pred_row = pred_rows[pred_pos]
                    if all(
                        any(
                            compare_evidence_value(
                                variant,
                                pred_row.get(key),
                                comparator.get(key) if isinstance(comparator, dict) else None,
                            ).passed
                            for variant in row_identity[key]
                        )
                        for key in known_keys
                    ):
                        matched_pred = pred_pos
                        break
            if matched_pred is None:
                still_pending.append(row_index)
            else:
                aligned[row_index] = matched_pred
                used.add(matched_pred)
        pending = still_pending

    # Pass 3: positional fallback. A GT row whose identity never matched keeps
    # today's exact-index behavior when that position is still unclaimed, so a
    # mis-extracted identity cell degrades to the legacy semantics instead of
    # grading the whole row as missing.
    for row_index in pending:
        if 0 <= row_index < len(pred_rows) and row_index not in used:
            aligned[row_index] = row_index
            used.add(row_index)
    return aligned


def _prediction_values_for_v02_rule(
    source: Any,
    rule: ExtractFieldTestRule,
    alignments: dict[str, dict[int, int]] | None = None,
) -> list[Any]:
    """Return exact-path predictions for v0.2 rules, preserving arrays/objects.

    When ``alignments`` covers the rule's array family, the first array index
    is translated from GT row space to predicted row space; an unaligned GT
    row resolves to a missing prediction.
    """
    if alignments:
        try:
            tokens = parse_field_path(rule.field_path)
        except ValueError:
            return []
        for position, token in enumerate(tokens):
            if not isinstance(token, int):
                continue
            family = ".".join(str(part) for part in tokens[:position])
            family_map = alignments.get(family)
            if family_map is not None:
                mapped = family_map.get(token)
                if mapped is None:
                    return []
                tokens[position] = mapped
            # Only the first array level carries match_by row identity.
            break
        value = get_path(source, tokens, default=_MISSING)
    else:
        value = _get_field_value(source, rule.field_path)
    if value is _MISSING:
        return []
    return [value]


def _compare_v02_prediction(rule: ExtractFieldTestRule, prediction: Any) -> ValueComparison:
    if rule.structural is not None or isinstance(prediction, list):
        return compare_array_against_rule(rule, prediction)
    return compare_value_against_rule(
        rule,
        prediction,
        source_kind="structured_value_no_citation_text",
    )


def _best_comparison(comparisons: Iterable[ValueComparison]) -> ValueComparison | None:
    best: ValueComparison | None = None
    for comparison in comparisons:
        if (
            best is None
            or (comparison.passed and not best.passed)
            or (comparison.passed == best.passed and comparison.score > best.score)
        ):
            best = comparison
    return best


def _best_v02_bbox_iou(
    evidence_entries: list[Any],
    pred_boxes_by_page: dict[int, list[BBox]],
    group: str,
) -> float | None:
    if not pred_boxes_by_page:
        return None
    best: float | None = None
    for ev in evidence_entries:
        if getattr(ev, "page", None) is None or getattr(ev, "bbox", None) is None:
            continue
        normalized = _as_xywh(ev.bbox)
        if normalized is None:
            continue
        # Predictions on other pages always score IoU 0.0 (the metric scopes
        # rectangles by page), so only same-page candidates need the full
        # computation and 0.0 is the floor once any (evidence, prediction)
        # pair exists.
        if best is None:
            best = 0.0
        gt = [BBox(page=ev.page, bbox=normalized, group=group)]
        for pred in pred_boxes_by_page.get(ev.page, ()):
            summary = compute_standard_iou_metrics(gt, [pred])
            if summary.iou > best:
                best = summary.iou
    return best


def _as_xywh(value: Any) -> tuple[float, float, float, float] | None:
    if value is None or len(value) != 4:
        return None
    x, y, w, h = value
    x_f = float(x)
    y_f = float(y)
    w_f = float(w)
    h_f = float(h)
    if w_f <= 0.0 or h_f <= 0.0:
        return None
    return (x_f, y_f, w_f, h_f)


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


