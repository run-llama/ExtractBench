"""Order-insensitive record matching metric for long extract arrays."""

from __future__ import annotations

import re
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeGuard

import numpy as np
from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment

from extract_bench.evaluation.metrics.extract.json_subset_match import (
    normalize_date_string,
)
from extract_bench.inference.providers.extract.table_codegen.schema_utils import (
    resolve_refs,
    schema_items,
    schema_properties,
)
from extract_bench.schemas.evaluation import MetricValue

# Above this many residual cells (unmatched_actual x unmatched_expected, after the
# exact-row peel) the dense assignment matrix is skipped to bound eval memory.
# Matches unified_evidence's _GROUNDED_MAX_CELLS; a 77.4k-row array would need
# ~67 GB otherwise and OOM-kill the runner.
_MAX_RESIDUAL_ASSIGNMENT_CELLS = 100_000_000

_PUNCT_RE = re.compile(r"[^\w\s]", re.U)
_WS_RE = re.compile(r"\s+", re.U)

# Reserved output keys that carry attribution metadata, not schema content, and must
# NEVER be scored as extracted cells. Kept in sync with the codegen provider's
# ``PROVENANCE_KEY`` (table_codegen/schema_utils.py): a record may carry
# ``_provenance: {"page": N}`` for source attribution, which the validity gate tolerates
# and these value metrics ignore (nested object-array subfields already come from the schema /
# expected rows, so the only leak was the top-level union walk over actual keys).
RESERVED_OUTPUT_KEYS: frozenset[str] = frozenset({"_provenance"})

DEFAULT_FUZZY_FIELD_THRESHOLDS: dict[str, float] = {
    # Matches the public LongArray-Extract reference scorer.
    "description": 0.95,
    "court": 0.85,
}

# Intern / Hungarian column id: a JSON key on a raw row (`str`) or a flattened
# pairing path (`tuple[str, ...]`). A singleton tuple is not a JSON key.
type CellKey = str | tuple[str, ...]


@dataclass(frozen=True)
class ArrayRecordMatchCounts:
    correct: int
    expected_total: int
    predicted_total: int
    expected_rows: int
    predicted_rows: int
    arrays_scored: int
    scalar_fields_scored: int

    @property
    def false_positive(self) -> int:
        return max(self.predicted_total - self.correct, 0)

    @property
    def false_negative(self) -> int:
        return max(self.expected_total - self.correct, 0)


@dataclass(frozen=True)
class _RowAssignment:
    pairs: list[tuple[int, int]]
    unmatched_actual_indices: list[int]
    unmatched_expected_indices: list[int]


def _normalize_text(value: str) -> str:
    return _WS_RE.sub(" ", _PUNCT_RE.sub(" ", value)).strip().lower()


def unwrap_value(value: Any) -> Any:
    if isinstance(value, dict) and set(value) == {"value"}:
        return value["value"]
    return value


def normalize_dates_deep(value: Any) -> Any:
    """Canonicalize date-like strings to ISO format everywhere in a JSON value.

    Applied once to both sides up front (not per cell pair) so the n×m
    assignment cost matrix stays cheap. `normalize_date_string` only rewrites
    strings that parse as a whole date, so IDs, amounts, and descriptions
    that merely contain digits pass through untouched.
    """
    if isinstance(value, str):
        return normalize_date_string(value)
    if isinstance(value, dict):
        return {k: normalize_dates_deep(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize_dates_deep(v) for v in value]
    return value


def normalize_ws(value: str) -> str:
    """Collapse internal whitespace runs so only inter-word spacing matters.

    Multi-line cells (postal addresses, wrapped headers) carry hard line breaks
    whose position depends on render column width, not content, so a prediction
    that re-wraps the same text is correct.

    Deliberately whitespace-RUN-only: whitespace hugging punctuation ("3, 4" vs
    "3,4") stays significant here because this helper defines the default cell
    equality (and ``_cell_key`` interning) for EVERY extract dataset. Datasets
    that want punctuation-spacing leniency opt in per field via the
    ``punctuation_spacing`` normalizer (unified_evidence_metric).
    """
    return _WS_RE.sub(" ", value).strip()


def cell_match[K: CellKey](
    expected: Any,
    actual: Any,
    field: K,
    *,
    fuzzy_field_thresholds: Mapping[K, float],
    field_schema: Any = None,
) -> bool:
    threshold = fuzzy_field_thresholds.get(field)
    if threshold is not None and isinstance(expected, str) and isinstance(actual, str):
        expected_norm = _normalize_text(expected)
        actual_norm = _normalize_text(actual)
        return expected_norm == actual_norm or (
            len(expected_norm) > 0
            and len(actual_norm) > 0
            and fuzz.ratio(expected_norm, actual_norm) >= threshold * 100.0
        )
    if isinstance(expected, str) and isinstance(actual, str):
        return normalize_ws(expected) == normalize_ws(actual)
    # JSON scalars compare with ``==`` (``0 != None``).
    return bool(expected == actual)


_UNHASHABLE = object()


def _eq_key(value: Any) -> Any:
    """Hashable snapshot of Python ``==`` for intern (not the string-cell rule).

    JSON containers freeze to tagged tuples so intern keys stay plain hashable
    values (no FrozenDict) and JSON-dump after replacing tuples with lists:
    lists as ``("l", items...)`` in order, dicts as ``("d", sorted (key, value)
    pairs)`` because ``dict ==`` is key-order independent. Nested strings stay
    exact: ``["Moscow "]`` and ``["Moscow"]`` are different keys. The top-level
    string branch of ``_cell_key`` still whitespace-folds. Anything that cannot
    freeze (mixed incomparable dict keys, a set, ...) stays ``_UNHASHABLE`` so
    that column falls back to pairwise compare.
    """
    if isinstance(value, list):
        parts = tuple(_eq_key(item) for item in value)
        if any(part is _UNHASHABLE for part in parts):
            return _UNHASHABLE
        return ("l", parts)
    if isinstance(value, dict):
        items: list[tuple[Any, Any]] = []
        for key, item in value.items():
            frozen_key = _eq_key(key)
            frozen_item = _eq_key(item)
            if frozen_key is _UNHASHABLE or frozen_item is _UNHASHABLE:
                return _UNHASHABLE
            items.append((frozen_key, frozen_item))
        try:
            items.sort()
        except TypeError:
            return _UNHASHABLE
        return ("d", tuple(items))
    try:
        hash(value)
    except TypeError:
        return _UNHASHABLE
    return ("v", value)


def _cell_key(
    value: Any,
    field_schema: Any = None,
) -> Any:
    """Hashable interning key that mirrors ``cell_match`` for non-fuzzy fields.

    Two cells share a key iff ``cell_match`` returns True (strings compared
    whitespace-insensitively, everything else by ``==``). Strings are tagged
    apart from non-strings so a normalized string can never collide with a
    look-alike scalar, exactly as ``cell_match`` keeps its str/str branch
    distinct from the ``==`` fallback. List and dict cells freeze to tagged
    tuples of exact nested values so opaque JSON arrays and objects intern
    the same way scalars already do. Returns ``_UNHASHABLE`` when a cell still
    cannot be interned -> caller scores that column pairwise.

    ``field_schema`` matches ``cell_match``'s keyword; intern keys come from
    the value (whitespace-folded strings, tagged scalars, frozen JSON).
    """
    if isinstance(value, str):
        return ("s", normalize_ws(value))
    if isinstance(value, (list, dict)):
        return _eq_key(value)
    try:
        hash(value)
    except TypeError:
        return _UNHASHABLE
    return ("v", value)


def _intern_field[K: CellKey](
    actual_list: Sequence[Any],
    expected_list: Sequence[Any],
    field: K,
    field_schema: Any = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Map each row's ``field`` cell to an int id shared across both sides.

    Equal cells (per ``cell_match``) receive the same id, so a single
    broadcasted integer ``!=`` reproduces the per-pair mismatch test while
    normalizing each cell once, not once per pair. Returns ``None`` if any
    cell cannot be interned so the caller scores that column pairwise.
    """
    keymap: dict[Any, int] = {}
    out: list[np.ndarray] = []
    for rows in (actual_list, expected_list):
        ids = np.empty(len(rows), dtype=np.int64)
        for idx, row in enumerate(rows):
            key = _cell_key(row.get(field), field_schema) if isinstance(row, dict) else _cell_key(None, field_schema)
            if key is _UNHASHABLE:
                return None
            slot = keymap.get(key)
            if slot is None:
                slot = len(keymap)
                keymap[key] = slot
            ids[idx] = slot
        out.append(ids)
    return out[0], out[1]


def _row_match_key[K: CellKey](
    row: Any,
    subfields: Sequence[K],
    field_schemas: Mapping[K, Any] | None = None,
) -> tuple[Any, ...] | None:
    """Hashable full-row key whose equality implies zero assignment cost."""
    row_dict = row if isinstance(row, dict) else {}
    parts: list[Any] = []
    schemas: Mapping[K, Any] = field_schemas if field_schemas is not None else {}
    for field in subfields:
        value = row_dict.get(field)
        cell_key = _cell_key(value, schemas.get(field))
        if cell_key is _UNHASHABLE:
            return None
        parts.append(cell_key)
    return tuple(parts)


def _can_peel_exact_rows[K: CellKey](
    subfields: Sequence[K],
    fuzzy_field_thresholds: Mapping[K, float],
) -> bool:
    return all(fuzzy_field_thresholds.get(field) is None for field in subfields)


def peel_exact_row_matches[K: CellKey](
    actual_list: Sequence[Any],
    expected_list: Sequence[Any],
    *,
    subfields: Sequence[K],
    fuzzy_field_thresholds: Mapping[K, float],
    field_schemas: Mapping[K, Any] | None = None,
) -> _RowAssignment:
    """Pre-align zero-cost rows before building an expensive residual matrix.

    Long-array predictions are often mostly exact with a few inserted, dropped,
    or shifted rows. Removing exact full-row matches first preserves the value
    score while shrinking the dense Hungarian problem to the true residual.
    """
    if not _can_peel_exact_rows(subfields, fuzzy_field_thresholds):
        return _RowAssignment(
            pairs=[],
            unmatched_actual_indices=list(range(len(actual_list))),
            unmatched_expected_indices=list(range(len(expected_list))),
        )

    actual_by_key: defaultdict[tuple[Any, ...], deque[int]] = defaultdict(deque)
    for actual_idx, row in enumerate(actual_list):
        key = _row_match_key(row, subfields, field_schemas)
        if key is not None:
            actual_by_key[key].append(actual_idx)

    pairs: list[tuple[int, int]] = []
    matched_actual: set[int] = set()
    unmatched_expected_indices: list[int] = []
    for expected_idx, row in enumerate(expected_list):
        key = _row_match_key(row, subfields, field_schemas)
        bucket = actual_by_key.get(key) if key is not None else None
        if bucket is not None and len(bucket) > 0:
            actual_idx = bucket.popleft()
            pairs.append((actual_idx, expected_idx))
            matched_actual.add(actual_idx)
        else:
            unmatched_expected_indices.append(expected_idx)

    unmatched_actual_indices = [idx for idx in range(len(actual_list)) if idx not in matched_actual]
    return _RowAssignment(
        pairs=pairs,
        unmatched_actual_indices=unmatched_actual_indices,
        unmatched_expected_indices=unmatched_expected_indices,
    )


def mismatch_cost_matrix[K: CellKey](
    actual_list: Sequence[Any],
    expected_list: Sequence[Any],
    *,
    subfields: Sequence[K],
    fuzzy_field_thresholds: Mapping[K, float],
    field_schemas: Mapping[K, Any] | None = None,
) -> np.ndarray:
    """Vectorized ``(n_actual, n_expected)`` matrix of mismatched-subfield counts.

    Bit-identical to the per-pair ``sum(not cell_match(...))`` build but without
    the n*m*k Python loop: each exact-match subfield is interned to ints and
    compared with one broadcasted ``!=`` (each cell normalized once, not once per
    pair). Fuzzy fields and unhashable cells fall back to the original pairwise
    compare for that column only. The matrix is ``int16`` (counts are bounded by
    ``len(subfields)``) -- a 4x smaller cost matrix than the old ``float64`` and
    the change that lifts the giant-array build time and OOM.
    """
    na, ne = len(actual_list), len(expected_list)
    cost = np.zeros((na, ne), dtype=np.int16)
    if na == 0 or ne == 0:
        return cost
    for field in subfields:
        field_schema = (field_schemas or {}).get(field)
        interned = None
        if fuzzy_field_thresholds.get(field) is None:
            interned = _intern_field(actual_list, expected_list, field, field_schema)
        if interned is not None:
            act_ids, exp_ids = interned
            cost += act_ids[:, None] != exp_ids[None, :]
            continue
        # Fuzzy threshold or unhashable cell: original per-pair compare, this column only.
        for i, actual_row in enumerate(actual_list):
            actual_dict = actual_row if isinstance(actual_row, dict) else {}
            actual_value = actual_dict.get(field)
            cost_row = cost[i]
            for j, expected_row in enumerate(expected_list):
                expected_dict = expected_row if isinstance(expected_row, dict) else {}
                if not cell_match(
                    expected_dict.get(field),
                    actual_value,
                    field,
                    fuzzy_field_thresholds=fuzzy_field_thresholds,
                    field_schema=field_schema,
                ):
                    cost_row[j] += 1
    return cost


def _assign_rows(
    actual_list: list[Any],
    expected_list: list[Any],
    *,
    subfields: Sequence[str],
    fuzzy_field_thresholds: Mapping[str, float],
    field_schemas: Mapping[str, Any] | None = None,
) -> list[tuple[int, int]]:
    assignment = peel_exact_row_matches(
        actual_list,
        expected_list,
        subfields=subfields,
        fuzzy_field_thresholds=fuzzy_field_thresholds,
        field_schemas=field_schemas,
    )
    pairs = list(assignment.pairs)
    if len(assignment.unmatched_actual_indices) == 0 or len(assignment.unmatched_expected_indices) == 0:
        return pairs

    residual_actual = [actual_list[idx] for idx in assignment.unmatched_actual_indices]
    residual_expected = [expected_list[idx] for idx in assignment.unmatched_expected_indices]
    # Memory ceiling on the residual assignment matrix (mirrors the
    # unified_evidence value-matrix guard). After the exact-row peel, a divergent
    # giant array leaves a residual ~= the full array; the dense int16 cost
    # matrix plus scipy's internal float64 copy is tens of GB at ~50k+ rows (a
    # 77,400-row array needs ~67 GB) and OOM-kills the eval runner. Over the cell
    # budget, skip the residual assignment: those rows stay unmatched (scored as
    # misses) rather than crashing the whole run -- a peel-only lower bound.
    if len(residual_actual) * len(residual_expected) > _MAX_RESIDUAL_ASSIGNMENT_CELLS:
        return pairs
    cost = mismatch_cost_matrix(
        residual_actual,
        residual_expected,
        subfields=subfields,
        fuzzy_field_thresholds=fuzzy_field_thresholds,
        field_schemas=field_schemas,
    )
    row_ind, col_ind = linear_sum_assignment(cost)
    pairs.extend(
        (
            assignment.unmatched_actual_indices[int(actual_idx)],
            assignment.unmatched_expected_indices[int(expected_idx)],
        )
        for actual_idx, expected_idx in zip(row_ind, col_ind, strict=True)
    )
    return pairs


def is_value_sequence(value: Any) -> TypeGuard[Sequence[Any]]:
    """List-like JSON arrays, not ``str`` / ``bytes`` (those are Sequences too)."""
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def as_rows(value: Any) -> list[Any]:
    return list(value) if is_value_sequence(value) else []


def is_array_schema(field_schema: Any) -> bool:
    """True when JSON Schema describes an array (incl. ``anyOf`` / ``oneOf`` / ``allOf``).

    Combinator-wrapped arrays (``anyOf: [{type: array, items: ...}, {type: null}]``)
    have no top-level ``type: array``; walk the same combinators as ``schema_items``.
    Gold/pred values are not consulted — a list in the extraction does not make
    a string field an array.
    """
    if not isinstance(field_schema, Mapping):
        return False
    raw = field_schema.get("type")
    types = raw if isinstance(raw, list) else [raw]
    if "array" in types or isinstance(field_schema.get("items"), dict):
        return True
    for key in ("anyOf", "oneOf", "allOf"):
        for alt in field_schema.get(key) or []:
            if is_array_schema(alt):
                return True
    return False


def array_item_properties(field_schema: Any) -> Mapping[str, Any]:
    """Column schemas for an array of objects: ``items.properties``, combinators flattened.

    ``field_schema`` is the array node (possibly ``anyOf`` [array, null]). Same
    ``schema_items`` / ``schema_properties`` helpers as extract GT lint. ``$ref``
    is not followed; callers inline with ``resolve_refs`` first. The mapping is
    not owned by the caller; do not mutate it.
    """
    if not isinstance(field_schema, Mapping):
        return {}
    return schema_properties(schema_items(field_schema))


def array_subfield_names(field_schema: Mapping[str, Any]) -> Sequence[str]:
    """Column names for one array of objects, from ``items.properties`` only."""
    return list(array_item_properties(field_schema).keys())


def _score_array(
    expected_rows: Any,
    actual_rows: Any,
    *,
    subfields: Sequence[str],
    fuzzy_field_thresholds: Mapping[str, float],
    field_schemas: Mapping[str, Any] | None = None,
) -> tuple[int, int, int, int, int]:
    expected_list = as_rows(expected_rows)
    actual_list = as_rows(actual_rows)
    expected_total = len(expected_list) * len(subfields)
    predicted_total = len(actual_list) * len(subfields)

    if len(expected_list) == 0 or len(subfields) == 0 or len(actual_list) == 0:
        return 0, expected_total, predicted_total, len(expected_list), len(actual_list)

    correct = 0
    for actual_idx, expected_idx in _assign_rows(
        actual_list,
        expected_list,
        subfields=subfields,
        fuzzy_field_thresholds=fuzzy_field_thresholds,
        field_schemas=field_schemas,
    ):
        actual_dict = actual_list[actual_idx] if isinstance(actual_list[actual_idx], Mapping) else {}
        expected_dict = expected_list[expected_idx] if isinstance(expected_list[expected_idx], Mapping) else {}
        correct += sum(
            cell_match(
                expected_dict.get(field),
                actual_dict.get(field),
                field,
                fuzzy_field_thresholds=fuzzy_field_thresholds,
                field_schema=(field_schemas or {}).get(field),
            )
            for field in subfields
        )

    return (
        correct,
        expected_total,
        predicted_total,
        len(expected_list),
        len(actual_list),
    )


def compute_array_record_match_counts(
    expected: Any,
    actual: Any,
    data_schema: Mapping[str, Any] | None = None,
    fuzzy_field_thresholds: Mapping[str, float] | None = None,
    normalize_dates: bool = True,
) -> ArrayRecordMatchCounts | None:
    """Compute order-insensitive record matching counts for top-level arrays.

    The denominator follows the public LongArray-Extract scorer: every expected
    scalar field is one data point, and every expected array row subfield is one
    data point. Array row order is ignored through optimal one-to-one matching.

    When ``normalize_dates`` is set (the default), date-like strings on both
    sides are canonicalized to ISO format before comparison, aligned with the
    generic ``accuracy`` metric (json_subset_match).
    """
    expected = unwrap_value(expected)
    actual = unwrap_value(actual)
    if not isinstance(expected, Mapping) or not isinstance(actual, Mapping):
        return None
    if normalize_dates:
        expected = normalize_dates_deep(expected)
        actual = normalize_dates_deep(actual)

    resolved = resolve_refs(data_schema) if isinstance(data_schema, Mapping) else {}
    schema_props = schema_properties(resolved)
    fuzzy = dict(DEFAULT_FUZZY_FIELD_THRESHOLDS if fuzzy_field_thresholds is None else fuzzy_field_thresholds)

    correct = 0
    expected_total = 0
    predicted_total = 0
    expected_rows = 0
    predicted_rows = 0
    arrays_scored = 0
    scalar_fields_scored = 0

    for field in sorted((set(expected) | set(actual)) - RESERVED_OUTPUT_KEYS):
        field_schema = schema_props.get(field, {})
        expected_value = expected.get(field)
        actual_value = actual.get(field)
        if is_array_schema(field_schema):
            subfield_names = array_subfield_names(field_schema)
            if len(subfield_names) == 0:
                continue
            field_schemas = array_item_properties(field_schema)
            (
                array_correct,
                array_expected_total,
                array_predicted_total,
                array_expected_rows,
                array_predicted_rows,
            ) = _score_array(
                expected_value,
                actual_value,
                subfields=subfield_names,
                fuzzy_field_thresholds=fuzzy,
                field_schemas=field_schemas,
            )
            correct += array_correct
            expected_total += array_expected_total
            predicted_total += array_predicted_total
            expected_rows += array_expected_rows
            predicted_rows += array_predicted_rows
            arrays_scored += 1
        else:
            expected_total += 1
            # An omitted key is an implicit null prediction and always counts in
            # the precision denominator (an omitted GT-null field matches via
            # ``cell_match(None, None)``, so gating on ``field in actual`` let
            # correct exceed predicted_total and precision run past 1.0).
            predicted_total += 1
            scalar_fields_scored += 1
            if cell_match(
                expected_value,
                actual_value,
                field,
                fuzzy_field_thresholds=fuzzy,
                field_schema=field_schema,
            ):
                correct += 1

    if arrays_scored == 0 or expected_total <= 0:
        return None

    return ArrayRecordMatchCounts(
        correct=correct,
        expected_total=expected_total,
        predicted_total=predicted_total,
        expected_rows=expected_rows,
        predicted_rows=predicted_rows,
        arrays_scored=arrays_scored,
        scalar_fields_scored=scalar_fields_scored,
    )


class ArrayRecordMatchMetric:
    """Metric bundle for long-array extraction quality."""

    def __init__(
        self,
        fuzzy_field_thresholds: Mapping[str, float] | None = None,
        normalize_dates: bool = True,
    ):
        self._fuzzy_field_thresholds = fuzzy_field_thresholds
        self._normalize_dates = normalize_dates

    def compute(self, expected: Any, actual: Any, **kwargs: Any) -> list[MetricValue]:
        data_schema = kwargs.get("data_schema")
        fuzzy_field_thresholds = kwargs.get("fuzzy_field_thresholds", self._fuzzy_field_thresholds)
        normalize_dates = kwargs.get("normalize_dates", self._normalize_dates)
        counts = compute_array_record_match_counts(
            expected=expected,
            actual=actual,
            data_schema=data_schema,
            fuzzy_field_thresholds=fuzzy_field_thresholds,
            normalize_dates=normalize_dates,
        )
        if counts is None:
            return []

        accuracy = counts.correct / counts.expected_total if counts.expected_total else 0.0
        precision = counts.correct / counts.predicted_total if counts.predicted_total else 0.0
        recall = accuracy
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        row_count_ratio = counts.predicted_rows / counts.expected_rows if counts.expected_rows else 0.0

        common_metadata = {
            "correct": counts.correct,
            "expected_total": counts.expected_total,
            "predicted_total": counts.predicted_total,
            "expected_rows": counts.expected_rows,
            "predicted_rows": counts.predicted_rows,
            "arrays_scored": counts.arrays_scored,
            "scalar_fields_scored": counts.scalar_fields_scored,
            "fuzzy_field_thresholds": dict(
                DEFAULT_FUZZY_FIELD_THRESHOLDS if fuzzy_field_thresholds is None else fuzzy_field_thresholds
            ),
            "normalize_dates": normalize_dates,
        }
        prf_metadata = {
            **common_metadata,
            "tp": counts.correct,
            "fp": counts.false_positive,
            "fn": counts.false_negative,
        }

        return [
            MetricValue(
                metric_name="array_record_accuracy",
                value=accuracy,
                metadata={
                    **common_metadata,
                    "passed": counts.correct,
                    "total": counts.expected_total,
                    "denominator": "expected_data_points",
                },
            ),
            MetricValue(
                metric_name="array_record_precision",
                value=precision,
                metadata=prf_metadata,
            ),
            MetricValue(
                metric_name="array_record_recall",
                value=recall,
                metadata=prf_metadata,
            ),
            MetricValue(metric_name="array_record_f1", value=f1, metadata=prf_metadata),
            MetricValue(
                metric_name="array_record_row_count_ratio",
                value=row_count_ratio,
                metadata={
                    **common_metadata,
                    "count": counts.predicted_rows,
                },
            ),
        ]
