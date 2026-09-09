"""Unified evidence + array-record extract metric.

One metric that subsumes both `array_record_*` (keyless Hungarian row matching,
order-insensitive, value-aware) and the v0.2 `extract_evidence_*` family
(OR-acceptable evidence values + page/bbox grounding) -- with NONE of the
`match_by`-rule dependence that makes the evidence metric cascade on dropped or
reordered rows.

Design
------
The value scorer is ``array_record``'s keyless Hungarian row assignment with
three additions:

1. **OR-acceptable values.** A ground-truth cell passes if the prediction
   matches its ``expected_output`` value *or any* alternate value declared in
   that leaf's ``evidence[]``. (When a leaf has a single evidence value equal to
   the expected value, this is just ``array_record``.)
2. **Object and object-array recursion.** A list-of-objects subfield (e.g. OFAC
   ``entities[].aliases``) is recursed into and aligned with another Hungarian
   pass, scoring each of its fields, instead of being compared as one opaque
   cell. Bare objects (root, nested outside arrays, and objects on array rows)
   are recursed into and scored per child cell. **Which comparison runs**
   (Hungarian object-array, object child walk, or opaque scalar) is decided
   by JSON Schema, including combinators — not by whether gold or pred is a
   list or dict. A list in an extraction does not make a string field an
   array. A missing key uses that field's JSON Schema ``default`` when the
   keyword is present; without ``default`` the key is absent (one-sided miss),
   not an implicit ``null``. ``lookup`` fills values; it does not pick the
   scorer. The same lookup is used at the root, inside nested objects, and on
   object columns of array rows after Hungarian pairing. Opaque array cells
   stay ``array_record``'s ``.get``. Row-pairing cost uses those same children
   (and nested object-arrays) so Hungarian maximizes the cells F1 actually
   scores. Scalar lists stay opaque.
3. **Grounding (bbox + page).** Two parallel sets of counters, both value-gated
   and both nesting under the value score. A cell is *grounded-correct* when the
   value matches AND the prediction's citation has a bbox that overlaps an
   evidence bbox on the same page (IoU >= threshold) -- a value-gated
   **bbox-IoU** metric, emitted as ``*_grounded_*``. A cell is *page-correct*
   when the value matches AND the citation's page is in the evidence's page set
   -- the coarser **page** check (no IoU), emitted as ``*_page_*``. Page is a
   per-cell superset of bbox (a box match requires equal pages, and page
   evidence/claims are supersets of bbox evidence/claims), so the three F1s nest:
   ``value_f1 >= page_f1 >= grounded_f1``. Together ``*_grounded_*`` and
   ``*_page_*`` replace the old ``extract_evidence_bbox_*`` and
   ``extract_evidence_page_*`` families as one nested precision/recall/F1 trio.
   Grounding is only defined where the GT carries the corresponding annotation
   (a page for ``*_page_*``, a bbox for ``*_grounded_*``), on
   BOTH sides of the P/R pair: recall's denominator is the GT cells that carry
   an evidence bbox, and precision's denominator is the predicted citation-bbox
   claims on cells *aligned to such a GT cell*. A claim on a cell whose GT has
   no bbox (or on an extra predicted row with no GT counterpart) is ungradeable
   -- it can never be verified right or wrong -- so it is excluded from the
   denominator rather than counted as a miss. Without this, a sparsely
   annotated document (e.g. one bbox-bearing cover field among 30k value-only
   cells) pins precision to ~0 for every pipeline that cites everything. A
   document whose GT has no evidence bbox at all emits **no** ``*_grounded_*``
   metric (excluded from the dataset average), rather than a 0.0 that would
   punish a document that simply cannot be graded for grounding.

On a document with no object-array subfields, no object-valued cells, and no
alternate evidence values, the ``*_value_*`` metrics are bit-identical to
``array_record_*`` -- same alignment, subfields, cost matrix, and denominators
-- with no ``match_by`` rule anywhere. Object-array subfields and objects
recursed as child cells
make the value metrics diverge from array_record (per-child credit); the
``*_grounded_*`` metrics
are the other new signal (0 for pipelines that emit no citations *on documents
whose GT carries bboxes*; omitted entirely on documents whose GT has none).
Truncation
drops recall exactly as in ``array_record``: an unaligned ground-truth row
contributes 0 correct against a full denominator, so there is no vacuous null
pass.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, SupportsFloat

import numpy as np
from scipy.optimize import linear_sum_assignment

from extract_bench.evaluation.metrics.extract.array_record_match_metric import (
    DEFAULT_FUZZY_FIELD_THRESHOLDS,
    RESERVED_OUTPUT_KEYS,
    array_item_properties,
    array_subfield_names,
    as_rows,
    cell_match,
    is_array_schema,
    mismatch_cost_matrix,
    normalize_dates_deep,
    normalize_ws,
    peel_exact_row_matches,
    unwrap_value,
)
from extract_bench.evaluation.metrics.extract.json_subset_match import normalize_date_string
from extract_bench.inference.providers.extract.table_codegen.schema_utils import (
    resolve_refs,
    schema_items,
    schema_properties,
)
from extract_bench.schemas.evaluation import MetricValue
from extract_bench.test_cases.schema import ExtractFieldTestRule, iter_rule_evidence

# Opt-in per-field normalizers (``_field_rules[path].normalizers``). Each is a
# fail->pass-only fallback applied after the exact cell match misses, so setting
# one can never turn a passing cell into a failure. They exist for scanned-form
# fields where the printed template makes one strict reading unfair (preprinted
# units, checkbox blanks, typewriter case, split preprinted years).
_OPTIONAL_TERMINAL_PUNCT = "optional_terminal_punctuation"
_CASE_INSENSITIVE = "case_insensitive"
_NULL_EQUALS_FALSE = "null_equals_false"
_PHONE_DIGITS = "phone_digits"
_LENIENT_DATE = "lenient_date"
_PUNCTUATION_SPACING = "punctuation_spacing"
# Above this many cells (GT rows x predicted rows) in a single flat array, the
# grounded assignment matrix -- an int16 cost matrix plus scipy's internal
# float64 copy -- is several GB, and a memory-limited eval runner OOMs building
# it. For such an array we SKIP grounding (its bbox metrics), which lets the
# value score take the exact-row peel that is provably value-identical for
# opaque un-grounded cells (the pairing the full matrix would have chosen only
# matters for grounded/nested TP, which we are not scoring). The document then
# emits NO grounded metric at all (``grounded_incomplete``), rather than a
# partial grounded score. Only arrays with no nested object or object-array
# subfields qualify, so the peel
# never shifts nested TP. ~100M cells keeps the matrix
# under ~1 GB while sparing every ordinary document.
_GROUNDED_MAX_CELLS = 100_000_000
# Everything ``configured_cell_match`` handles; kept equal to the schema's
# EXTRACT_FIELD_NORMALIZERS vocabulary (guarded by a unit test) so a name can't
# validate at load time yet silently no-op here.
SUPPORTED_NORMALIZERS: frozenset[str] = frozenset(
    {
        _OPTIONAL_TERMINAL_PUNCT,
        _CASE_INSENSITIVE,
        _NULL_EQUALS_FALSE,
        _PHONE_DIGITS,
        _LENIENT_DATE,
        _PUNCTUATION_SPACING,
    }
)
# One optional terminal punctuation char, not a whole run: "item 5." forgives
# the dot, but "item 5;,,." stays distinct from "item 5".
_TERMINAL_PUNCT_RE = re.compile(r"\s*[.,;:]\s*$")
_PUNCT_SPACING_RE = re.compile(r"\s*([,.;:])\s*", re.U)
_NON_DIGIT_RE = re.compile(r"\D+")
_DIGIT_RE = re.compile(r"\d")
# Preprinted decade split: a filer types "55" after the printed "19", so
# transcriptions render "19 55". Only joined when a digit (a day) already
# appears before the pair -- "June 19 44" is a day 19 plus a 2-digit year,
# not a split 1944.
_SPLIT_YEAR_RE = re.compile(r"\b(19|20)\s+(\d{2})\b")
_TRAILING_2DIGIT_YEAR_RE = re.compile(r"^(.*[ ,/-])(\d{2})\s*$")
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Validated COCO xywh after ingest. JSON / Pydantic still carry list[float].
type BBox = tuple[float, float, float, float]
type PageBBox = tuple[int, BBox]
type BoxIndex = dict[str, list[PageBBox]]


def _validated_bbox(raw: Sequence[SupportsFloat]) -> BBox | None:
    """Four finite COCO xywh floats with positive size, else None."""
    if len(raw) < 4:
        return None
    box = (float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))
    _, _, w, h = box
    if not all(map(math.isfinite, box)) or w <= 0 or h <= 0:
        return None
    return box


@lru_cache(maxsize=8192)
def _lenient_date_probes(value: str) -> frozenset[str]:
    """Date readings of a scanned-form value: split-year joined, and 2-digit
    trailing years expanded into both centuries ("June 17, 43" -> 1943/2043).

    Guardrails against equating different values:
    - Both rewrites require a digit (a day) before the year token, so a lone
      trailing pair ("May 11") is never century-expanded.
    - A rewritten probe only counts when it parses to an ISO date; unparseable
      rewrites are discarded rather than matched as raw strings ("Permit 12-34"
      vs "Permit 12-19 34" must not intersect on a fabricated "Permit 12-1934").
    """
    probes = {normalize_date_string(value)}
    rewrites: list[str] = []
    m = _SPLIT_YEAR_RE.search(value)
    if m and _DIGIT_RE.search(value[: m.start()]):
        rewrites.append(_SPLIT_YEAR_RE.sub(r"\1\2", value))
    for base in (value, *rewrites):
        m = _TRAILING_2DIGIT_YEAR_RE.match(base)
        if m and _DIGIT_RE.search(m.group(1)):
            rewrites += [f"{m.group(1)}19{m.group(2)}", f"{m.group(1)}20{m.group(2)}"]
    for rewrite in rewrites:
        normalized = normalize_date_string(rewrite)
        if _ISO_DATE_RE.match(normalized):
            probes.add(normalized)
    return frozenset(probes)


def configured_cell_match(
    expected: Any,
    actual: Any,
    field: str,
    *,
    fuzzy: dict[str, float],
    normalizers: tuple[str, ...] | None,
) -> bool:
    """``cell_match`` plus the field's opt-in normalizer fallbacks."""
    if cell_match(expected, actual, field, fuzzy_field_thresholds=fuzzy):
        return True
    if not normalizers:
        return False
    if _NULL_EQUALS_FALSE in normalizers:
        # A blank checkbox reads as False or as an omitted/null field with equal
        # right; identity checks keep 0/"" from sneaking in via == coercion.
        if (expected is False or expected is None) and (actual is False or actual is None):
            return True
    if not isinstance(expected, str) or not isinstance(actual, str):
        return False
    if _OPTIONAL_TERMINAL_PUNCT in normalizers:
        expected_norm = _TERMINAL_PUNCT_RE.sub("", expected.strip())
        actual_norm = _TERMINAL_PUNCT_RE.sub("", actual.strip())
        # Both sides must survive the strip: punctuation-only values ('.' vs
        # ';') would otherwise collapse to '' == '' and match each other.
        if expected_norm and actual_norm:
            if cell_match(expected_norm, actual_norm, field, fuzzy_field_thresholds=fuzzy):
                return True
    if _PUNCTUATION_SPACING in normalizers:
        expected_norm = _PUNCT_SPACING_RE.sub(r"\1", expected)
        actual_norm = _PUNCT_SPACING_RE.sub(r"\1", actual)
        if cell_match(expected_norm, actual_norm, field, fuzzy_field_thresholds=fuzzy):
            return True
    if _CASE_INSENSITIVE in normalizers:
        if normalize_ws(expected).casefold() == normalize_ws(actual).casefold():
            return True
    if _PHONE_DIGITS in normalizers:
        expected_digits = _NON_DIGIT_RE.sub("", expected)
        actual_digits = _NON_DIGIT_RE.sub("", actual)
        if len(expected_digits) >= 10 and len(actual_digits) >= 10:
            if expected_digits[-10:] == actual_digits[-10:]:
                return True
        elif expected_digits and expected_digits == actual_digits:
            return True
    if _LENIENT_DATE in normalizers:
        if _lenient_date_probes(expected) & _lenient_date_probes(actual):
            return True
    return False


@dataclass
class _Counts:
    """Pooled tp/denominator counts for value and grounded scoring."""

    v_correct: int = 0  # value matches over aligned cells (tp for precision AND recall)
    expected: int = 0  # GT leaf cells (value recall denominator)
    predicted: int = 0  # predicted leaf cells (value precision denominator)
    g_correct: int = 0  # value+grounding correct cells (grounded tp)
    g_expected: int = 0  # GT cells whose evidence carries a bbox (grounded recall denom)
    # Predicted citation-bbox claims on cells aligned to a bbox-bearing GT cell
    # (grounded precision denom). Claims on cells whose GT has no bbox -- or on
    # extra predicted rows with no GT counterpart -- are ungradeable and stay out.
    g_claims: int = 0
    # Page-level counterpart of the grounded (bbox) counters: a cell is
    # page-correct when the value matches AND the prediction cites a page in the
    # GT evidence's page set. Coarser than bbox (no IoU), so page is a superset:
    # every bbox-bearing cell also carries a page, and a bbox match implies a
    # page match -- hence p_correct >= g_correct and p_expected >= g_expected,
    # giving value_f1 >= page_f1 >= grounded_f1.
    p_correct: int = 0  # value+page correct cells (page tp)
    p_expected: int = 0  # GT cells whose evidence carries a page (page recall denom)
    p_claims: int = 0  # predicted page claims on cells aligned to a page-bearing GT cell (page precision denom)
    expected_rows: int = 0
    predicted_rows: int = 0
    grounded_incomplete: bool = False  # a giant array skipped grounding -> emit no grounded metric
    value_incomplete: bool = False  # a giant flat array skipped the residual value matrix -> value is a lower bound


def path_leaf(path: str) -> str:
    """Last field name on an extract path, with a trailing array index stripped.

    ``holdings[0].security`` → ``security``; ``rows[2]`` → ``rows``. Used as the
    ``field`` key for fuzzy / cell-match lookup when scoring a scalar at ``path``.
    """
    return path.rsplit(".", 1)[-1].split("[", 1)[0]


def lookup(obj: Mapping[str, Any], key: str, field_schema: Mapping[str, Any] | Any) -> tuple[bool, Any]:
    """Value of ``key`` on ``obj``, or the field's schema ``default`` if omitted.

    A present key (including an explicit ``null``) wins. A missing key uses
    ``default`` iff that keyword is present (including ``default: null``).
    A missing key with no ``default`` keyword is absent — not filled with
    ``null``.
    """
    if key in obj:
        return True, obj[key]
    if isinstance(field_schema, Mapping) and "default" in field_schema:
        return True, field_schema["default"]
    return False, None


def is_object_array_schema(field_schema: Any) -> bool:
    """True when JSON Schema describes an array of objects (incl. combinators)."""
    if not isinstance(field_schema, Mapping):
        return False
    if len(array_item_properties(field_schema)) > 0:
        return True
    items = schema_items(field_schema)
    if not isinstance(items, Mapping):
        return False
    raw = items.get("type")
    types = raw if isinstance(raw, list) else [raw]
    return "object" in types


def is_object_schema(field_schema: Any) -> bool:
    """True when JSON Schema describes an object (incl. combinators)."""
    if not isinstance(field_schema, Mapping):
        return False
    if len(schema_properties(field_schema)) > 0:
        return True
    raw = field_schema.get("type")
    types = raw if isinstance(raw, list) else [raw]
    if "object" in types:
        return True
    for key in ("anyOf", "oneOf", "allOf"):
        for alt in field_schema.get(key) or []:
            if is_object_schema(alt):
                return True
    return False


def is_object_array_subfield(field_schema: Any) -> bool:
    """True when the field schema is a nested object-array (Hungarian at the next depth)."""
    return is_object_array_schema(field_schema)


def is_object_subfield(field_schema: Any) -> bool:
    """True when the field schema is a dict (scored via ``_score_node``)."""
    return is_object_schema(field_schema)


type PairingPath = tuple[str, ...]


def _read_segments(
    row: Any,
    *,
    segments: Sequence[str],
) -> Any:
    """Walk successive mapping keys; a missing step is an implicit ``None`` cell.

    Keys are not split, so a schema field named ``a.b`` is one step, not two.
    """
    cur: Any = row if isinstance(row, Mapping) else {}
    for part in segments:
        if not isinstance(cur, Mapping):
            return None
        cur = cur.get(part)
    return cur


def _gt_rel_path(segments: Sequence[str]) -> str:
    """Join key segments with ``.`` for ``field_path`` lookup (alts / normalizers)."""
    return ".".join(segments)


def _collect_pairing_fields(
    *,
    schema: Any,
    prefix: PairingPath,
) -> tuple[Sequence[tuple[PairingPath, Any]], Sequence[tuple[PairingPath, Any]]]:
    """Opaque pairing cells and nested object-arrays under ``schema``.

    Same descent as ``Scorer._score_node``: object-arrays are a nested Hungarian
    unit; scalar arrays and scalars stay one opaque cell; objects expand to
    properties so pairing cost counts the children F1 will score, not a 0/1
    ``==`` on the whole dict.
    """
    if is_array_schema(schema):
        if is_object_array_schema(schema):
            return [], [(prefix, schema)]
        return [(prefix, schema)], []
    if is_object_schema(schema):
        props = schema_properties(schema)
        if len(props) == 0:
            return [(prefix, schema)], []
        opaque: list[tuple[PairingPath, Any]] = []
        arrays: list[tuple[PairingPath, Any]] = []
        for key, child in props.items():
            child_opaque, child_arrays = _collect_pairing_fields(schema=child, prefix=(*prefix, key))
            opaque.extend(child_opaque)
            arrays.extend(child_arrays)
        return opaque, arrays
    return [(prefix, schema)], []


def _item_pairing_fields(
    *,
    item_sch: Mapping[str, Any],
    subfield_names: Sequence[str],
) -> tuple[Sequence[PairingPath], Mapping[PairingPath, Any], Sequence[tuple[PairingPath, Any]]]:
    opaque: list[tuple[PairingPath, Any]] = []
    arrays: list[tuple[PairingPath, Any]] = []
    for name in subfield_names:
        child_opaque, child_arrays = _collect_pairing_fields(schema=item_sch.get(name, {}), prefix=(name,))
        opaque.extend(child_opaque)
        arrays.extend(child_arrays)
    return [path for path, _ in opaque], dict(opaque), arrays


def _flatten_pairing_rows(
    rows: Sequence[Any],
    *,
    paths: Sequence[PairingPath],
) -> Sequence[Mapping[PairingPath, Any]]:
    return [{path: _read_segments(row, segments=path) for path in paths} for row in rows]


def _fuzzy_for_paths(
    paths: Sequence[PairingPath],
    *,
    fuzzy: Mapping[str, float],
) -> Mapping[PairingPath, float]:
    out: dict[PairingPath, float] = {}
    for path in paths:
        leaf = path[-1] if path else ""
        if leaf in fuzzy:
            out[path] = fuzzy[leaf]
    return out


def _value_leaf_count(
    value: Any,
    *,
    schema: Any,
) -> int:
    """How many F1 leaves ``_count_subtree`` would emit for an unmatched value."""
    if is_array_schema(schema):
        if is_object_array_schema(schema):
            item_sch = array_item_properties(schema)
            names = array_subfield_names(schema)
            total = 0
            for row in as_rows(value):
                rd = row if isinstance(row, Mapping) else {}
                for name in names:
                    total += _value_leaf_count(rd.get(name), schema=item_sch.get(name, {}))
            return total
        return 1
    if is_object_schema(schema):
        props = schema_properties(schema)
        if len(props) == 0:
            return 1
        rd = value if isinstance(value, Mapping) else {}
        return sum(_value_leaf_count(rd.get(key), schema=child) for key, child in props.items())
    return 1


def _row_leaf_count(
    row: Any,
    *,
    item_sch: Mapping[str, Any],
    names: Sequence[str],
) -> int:
    rd = row if isinstance(row, Mapping) else {}
    return sum(_value_leaf_count(rd.get(name), schema=item_sch.get(name, {})) for name in names)


def iou_xywh(a: BBox, b: BBox) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    if not all(map(math.isfinite, (ax, ay, aw, ah, bx, by, bw, bh))) or aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    # Reconstructing width as (x+w)-x is a few ulps off for values like 0.1,
    # so the raw ratio can exceed 1.0 even for identical boxes.
    # min(1.0, nan) is 1.0 in Python, so non-finite ratios must not clamp to a hit.
    ratio = inter / union
    return min(1.0, ratio) if math.isfinite(ratio) else 0.0


class Scorer:
    def __init__(
        self,
        *,
        alt_values: dict[str, list[Any]],
        evidence_boxes: BoxIndex,
        evidence_pages: dict[str, set[int]],
        normalizers: dict[str, tuple[str, ...]],
        pred_boxes: BoxIndex,
        pred_pages: dict[str, set[int]],
        fuzzy: dict[str, float],
        iou_threshold: float,
    ) -> None:
        self._alt = alt_values
        self._ev_boxes = evidence_boxes
        self._ev_pages = evidence_pages
        self._normalizers = normalizers
        self._pred_boxes = pred_boxes
        self._pred_pages = pred_pages
        self._fuzzy = fuzzy
        self._iou = iou_threshold

    def _configured_cell_match(self, gt_path: str, expected: Any, actual: Any, field: str) -> bool:
        return configured_cell_match(
            expected, actual, field, fuzzy=self._fuzzy, normalizers=self._normalizers.get(gt_path)
        )

    def _gold_needs_configured_match(
        self,
        *,
        exp_rows: Sequence[Any],
        pairing_paths: Sequence[PairingPath],
        gt_field: str,
    ) -> bool:
        """True when this level's opaque pairing cells have a distinct alt or a normalizer.

        Nested object-arrays detect this for themselves; do not inherit a parent flag.
        """
        if not pairing_paths or (not self._alt and not self._normalizers):
            return False
        for i, erow in enumerate(exp_rows):
            for path in pairing_paths:
                gt_path = f"{gt_field}[{i}].{_gt_rel_path(path)}"
                if self._normalizers.get(gt_path):
                    return True
                alt = self._alt.get(gt_path)
                if not alt:
                    continue
                canon = _read_segments(erow, segments=path)
                if any(value != canon for value in alt):
                    return True
        return False

    def _object_array_pair_cost(
        self,
        exp_val: Any,
        act_val: Any,
        *,
        schema: Mapping[str, Any],
        gt_path: str,
    ) -> int:
        """F1-aligned mismatch for one nested object-array under an already-paired parent."""
        exp_rows = as_rows(exp_val)
        act_rows = as_rows(act_val)
        item_sch = array_item_properties(schema)
        names = array_subfield_names(schema)
        if len(names) == 0:
            return 0
        if len(exp_rows) == 0 or len(act_rows) == 0:
            return sum(_row_leaf_count(row, item_sch=item_sch, names=names) for row in exp_rows) + sum(
                _row_leaf_count(row, item_sch=item_sch, names=names) for row in act_rows
            )
        inner_paths, _, _ = _item_pairing_fields(item_sch=item_sch, subfield_names=names)
        inner = self._pairing_cost_matrix(
            act_rows,
            exp_rows,
            item_sch=item_sch,
            subfield_names=names,
            gt_field=gt_path,
            use_alts=self._gold_needs_configured_match(exp_rows=exp_rows, pairing_paths=inner_paths, gt_field=gt_path),
        )
        row_ind, col_ind = linear_sum_assignment(inner)
        paired = int(inner[row_ind, col_ind].sum())
        matched_exp = {int(i) for i in col_ind}
        matched_act = {int(j) for j in row_ind}
        unmatched = sum(
            _row_leaf_count(exp_rows[i], item_sch=item_sch, names=names)
            for i in range(len(exp_rows))
            if i not in matched_exp
        ) + sum(
            _row_leaf_count(act_rows[j], item_sch=item_sch, names=names)
            for j in range(len(act_rows))
            if j not in matched_act
        )
        return paired + unmatched

    def _pairing_cost_matrix(
        self,
        act_rows: Sequence[Any],
        exp_rows: Sequence[Any],
        *,
        item_sch: Mapping[str, Any],
        subfield_names: Sequence[str],
        gt_field: str,
        use_alts: bool,
    ) -> np.ndarray:
        """``(n_actual, n_expected)`` leaf-mismatch cost, matching later F1 recursion.

        Nested objects contribute each scored child (not one 0/1 dict compare).
        Nested object-arrays add their own Hungarian leaf cost. Scalar arrays
        stay one opaque cell.
        """
        paths, path_schemas, array_fields = _item_pairing_fields(item_sch=item_sch, subfield_names=subfield_names)
        na, ne = len(act_rows), len(exp_rows)
        path_fuzzy = _fuzzy_for_paths(paths, fuzzy=self._fuzzy)
        if use_alts:
            cost = np.zeros((na, ne), dtype=np.int32)
            exp_cands: list[list[list[Any]]] = []
            for i, erow in enumerate(exp_rows):
                row_cands = []
                for path in paths:
                    canon = _read_segments(erow, segments=path)
                    alt = self._alt.get(f"{gt_field}[{i}].{_gt_rel_path(path)}")
                    cand = [canon]
                    if alt:
                        cand += [v for v in alt if v != canon and v not in cand]
                    row_cands.append(cand)
                exp_cands.append(row_cands)
            for j, arow in enumerate(act_rows):
                avals = [_read_segments(arow, segments=path) for path in paths]
                for i, cands in enumerate(exp_cands):
                    cost[j, i] = sum(
                        not any(
                            self._configured_cell_match(
                                f"{gt_field}[{i}].{_gt_rel_path(paths[si])}",
                                cv,
                                avals[si],
                                paths[si][-1] if paths[si] else "",
                            )
                            for cv in cands[si]
                        )
                        for si in range(len(paths))
                    )
        elif paths:
            cost = mismatch_cost_matrix(
                _flatten_pairing_rows(act_rows, paths=paths),
                _flatten_pairing_rows(exp_rows, paths=paths),
                subfields=paths,
                fuzzy_field_thresholds=path_fuzzy,
                field_schemas=path_schemas,
            ).astype(np.int32, copy=False)
        else:
            cost = np.zeros((na, ne), dtype=np.int32)
        if array_fields:
            extra = np.zeros((na, ne), dtype=np.int32)
            for j, arow in enumerate(act_rows):
                for i, erow in enumerate(exp_rows):
                    extra[j, i] = sum(
                        self._object_array_pair_cost(
                            _read_segments(erow, segments=path),
                            _read_segments(arow, segments=path),
                            schema=array_schema,
                            gt_path=f"{gt_field}[{i}].{_gt_rel_path(path)}",
                        )
                        for path, array_schema in array_fields
                    )
            cost = cost + extra
        return cost

    def _value_match(self, gt_path: str, canonical: Any, actual: Any, field: str) -> bool:
        """OR-acceptable: the prediction matches the expected value or any evidence value."""
        for candidate in (canonical, *self._alt.get(gt_path, ())):
            if self._configured_cell_match(gt_path, candidate, actual, field):
                return True
        return False

    def _box_match(self, gt_path: str, pred_path: str) -> bool:
        gt_boxes = self._ev_boxes.get(gt_path)
        pred = self._pred_boxes.get(pred_path)
        if not gt_boxes or not pred:
            return False
        return any(gp == pp and iou_xywh(gb, pb) >= self._iou for gp, gb in gt_boxes for pp, pb in pred)

    def _note_grounded_expected(self, path: str, c: _Counts) -> None:
        if self._ev_boxes.get(path):
            c.g_expected += 1

    def _note_grounded_claim(self, gt_path: str, pred_path: str, c: _Counts) -> None:
        if not self._pred_boxes.get(pred_path):
            return
        if self._ev_boxes.get(gt_path):
            c.g_claims += 1

    def _page_match(self, gt_path: str, pred_path: str) -> bool:
        """A cited page for pred_path lands in the GT evidence's page set for
        gt_path. Coarser than ``_box_match`` (no IoU) and implied by it: a box
        match requires the pages to be equal, so every box match is a page
        match."""
        gt_pages = self._ev_pages.get(gt_path)
        pred_pages = self._pred_pages.get(pred_path)
        if not gt_pages or not pred_pages:
            return False
        return not gt_pages.isdisjoint(pred_pages)

    def _score_cell(
        self, gt_path: str, pred_path: str, canonical: Any, actual: Any, field: str, c: _Counts, ground: bool = True
    ) -> bool:
        """Score one aligned cell (value + page + grounding). Caller owns the denominators.

        ``ground=False`` scores value only, for cells in a giant array whose
        grounding-family scoring was skipped (their g_expected/g_claims and
        p_expected/p_claims were not counted either, so g_correct/p_correct must
        not be counted here or the grounded/page precision/recall would be
        computed against a truncated denominator).
        """
        matched = self._value_match(gt_path, canonical, actual, field)
        if matched:
            c.v_correct += 1
            if ground:
                if self._box_match(gt_path, pred_path):
                    c.g_correct += 1
                if self._page_match(gt_path, pred_path):
                    c.p_correct += 1
        return matched

    def _count_subtree(self, path: str, value: Any, schema: Mapping[str, Any], c: _Counts, *, expected: bool) -> None:
        """Count a one-sided subtree's leaves (recall-only or precision-only miss).

        Used for unmatched rows and for a key present on only one side with no
        schema ``default``. Do not route these through ``_score_node``: there is
        no partner value, so every leaf is a miss.
        """
        if is_array_schema(schema):
            if is_object_array_schema(schema):
                subfield_names = array_subfield_names(schema)
                item_sch = array_item_properties(schema)
                for i, row in enumerate(as_rows(value)):
                    rd = row if isinstance(row, Mapping) else {}
                    for s in subfield_names:
                        self._count_subtree(f"{path}[{i}].{s}", rd.get(s), item_sch.get(s, {}), c, expected=expected)
                return
        elif is_object_schema(schema):
            props = schema_properties(schema)
            if len(props) > 0:
                rd = value if isinstance(value, Mapping) else {}
                for key, child_schema in props.items():
                    child = f"{path}.{key}" if path else key
                    in_side, child_val = lookup(rd, key, child_schema)
                    self._count_subtree(child, child_val if in_side else None, child_schema, c, expected=expected)
                return
        if expected:
            c.expected += 1
            self._note_grounded_expected(path, c)
            if self._ev_pages.get(path):
                c.p_expected += 1
        else:
            # Extra predicted subtree: no GT counterpart exists, so its citation
            # bboxes are ungradeable and do not enter the grounded precision
            # denominator (see _Counts.g_claims).
            c.predicted += 1

    def _count_unmatched_nested(
        self,
        base: str,
        row: Mapping[str, Any],
        item_sch: Mapping[str, Any],
        nested_names: Sequence[str],
        c: _Counts,
        *,
        expected: bool,
    ) -> None:
        for s in nested_names:
            child_schema = item_sch.get(s, {})
            in_side, child_val = lookup(row, s, child_schema)
            self._count_subtree(f"{base}.{s}", child_val if in_side else None, child_schema, c, expected=expected)

    def _score_array(
        self, gt_field: str, pred_field: str, exp_rows: Any, act_rows: Any, schema: Mapping[str, Any], c: _Counts
    ) -> None:
        # gt_field / pred_field are separate base paths: ground-truth cells (alt
        # values, evidence boxes) key off gt_field; predicted cells (citations)
        # key off pred_field. They differ once the outer array is reordered, so
        # threading both is what keeps nested grounding correct after a remap.
        exp_rows = as_rows(exp_rows)
        act_rows = as_rows(act_rows)
        subfield_names = array_subfield_names(schema)
        if len(subfield_names) == 0:
            return
        item_sch = array_item_properties(schema)
        # List-of-objects subfields recurse with another Hungarian pass.
        # Dict-valued subfields recurse via ``_score_node`` (same as root).
        # Scalars and scalar lists stay opaque cells.
        object_array_names = [s for s in subfield_names if is_object_array_subfield(item_sch.get(s, {}))]
        object_names = [
            s for s in subfield_names if s not in object_array_names and is_object_subfield(item_sch.get(s, {}))
        ]
        cell_names = [s for s in subfield_names if s not in object_array_names and s not in object_names]
        # Opaque-cell denominators: array_record-exact (rows x cell subfields).
        # Object and object-array fields are counted by the recursion below.
        c.expected += len(exp_rows) * len(cell_names)
        c.predicted += len(act_rows) * len(cell_names)
        c.expected_rows += len(exp_rows)
        c.predicted_rows += len(act_rows)
        # A flat array whose full grounded matrix would be multi-GB: skip its
        # grounding-family scoring (bbox AND page). Leaving has_ev_page/
        # has_pred_page False routes the value score to the exact-row peel below
        # (value-identical for opaque cells), and not counting
        # g_expected/g_claims/p_expected/p_claims here keeps the grounded and
        # page denominators consistent with the skipped g_correct/p_correct.
        # ``not object_array_names and not object_names`` guarantees no nested recursion, so the peel cannot
        # shift nested TP. The memory win only lands on the plain peel path: an
        # array with genuine alternate values (``multi``) or field normalizers
        # still builds the full candidate matrix below regardless of ``giant`` --
        # there grounding is withheld but the allocation is unchanged. The
        # threshold bounds the FULL matrix; the peel's residual matrix over
        # unmatched rows can still grow if few rows match exactly, so treat it as
        # a heuristic, not a hard ceiling.
        giant = (
            len(exp_rows) > 0
            and len(act_rows) > 0
            and len(cell_names) > 0
            and len(object_array_names) == 0
            and len(object_names) == 0
            and len(exp_rows) * len(act_rows) > _GROUNDED_MAX_CELLS
        )
        # Page presence drives both the grounded-family denominators and the
        # pairing-sensitive branch selection below. It is a SUPERSET of bbox
        # presence (every bbox-bearing evidence/citation also carries a page), so
        # probing pages alone detects any dropped grounding -- bbox or page-only.
        has_ev_page = False
        has_pred_page = False
        if giant:
            # Existence-only probe (early-exit) so we still know whether any
            # grounding-family scoring was actually dropped -- only then is the
            # document's grounded/page score incomplete and must be withheld
            # entirely. Pages are the superset, so this covers dropped bboxes too.
            dropped_grounding = any(
                self._ev_pages.get(f"{gt_field}[{i}].{s}") for i in range(len(exp_rows)) for s in cell_names
            ) or any(self._pred_pages.get(f"{pred_field}[{j}].{s}") for j in range(len(act_rows)) for s in cell_names)
            if dropped_grounding:
                c.grounded_incomplete = True
        else:
            for i in range(len(exp_rows)):
                for s in cell_names:
                    gt_path = f"{gt_field}[{i}].{s}"
                    self._note_grounded_expected(gt_path, c)
                    if self._ev_pages.get(gt_path):
                        c.p_expected += 1
                        has_ev_page = True
            # Flag-only scan: g_claims/p_claims are counted per aligned pair
            # below, and only for cells whose GT side carries a bbox/page
            # (ungradeable claims stay out of the precision denominator). The flag
            # reflects ANY predicted page (the superset of any predicted bbox) so
            # the pairing-sensitive branch selection below covers both families.
            for j in range(len(act_rows)):
                for s in cell_names:
                    if self._pred_pages.get(f"{pred_field}[{j}].{s}"):
                        has_pred_page = True
                        break
                if has_pred_page:
                    break
        # Object-array subfields are counted by the recursion below, per matched
        # pair (and as one-sided misses for unmatched rows).
        match_for_exp: dict[int, int] = {}
        if len(exp_rows) > 0 and len(act_rows) > 0:
            # Align on the same leaves F1 scores: opaque cells plus nested-object
            # children (not a 0/1 ``==`` on the whole object). Object-arrays add
            # their nested Hungarian cost. Scalar arrays stay one opaque cell.
            pairing_paths, _, _ = _item_pairing_fields(item_sch=item_sch, subfield_names=subfield_names)
            # Peel still keys original row dicts by schema column name (not nested
            # segment tuples). Pairing-sensitive branches use pairing_paths.
            cost_names: Sequence[str] = cell_names if cell_names else subfield_names
            fuzzy = self._fuzzy
            # Distinct alts / normalizers on *this* level's opaque cells expand the
            # zero-cost graph, so skip the exact-row peel. Nested object-arrays
            # detect configured match for themselves when they add their cost.
            if self._gold_needs_configured_match(exp_rows=exp_rows, pairing_paths=pairing_paths, gt_field=gt_field):
                cost = self._pairing_cost_matrix(
                    act_rows,
                    exp_rows,
                    item_sch=item_sch,
                    subfield_names=subfield_names,
                    gt_field=gt_field,
                    use_alts=True,
                )
                row_ind, col_ind = linear_sum_assignment(cost)
                match_for_exp = {int(i): int(j) for j, i in zip(row_ind, col_ind, strict=True)}
            elif len(object_array_names) > 0 or len(object_names) > 0 or (has_ev_page and has_pred_page):
                # Pairing-sensitive scoring: nested object arrays and in-row objects
                # recurse on the matched predicted index, and per-cell grounding +
                # page grading key off it too. Among equal-cost optima the exact-row peel could
                # pick a different pairing than the full assignment and shift
                # grounded / page / nested TP, so fall back to full assignment to
                # preserve the exact tie-break. ``has_ev_page``/``has_pred_page``
                # are the bbox supersets, so this branch also covers every
                # bbox-grounded array.
                cost = self._pairing_cost_matrix(
                    act_rows,
                    exp_rows,
                    item_sch=item_sch,
                    subfield_names=subfield_names,
                    gt_field=gt_field,
                    use_alts=False,
                )
                row_ind, col_ind = linear_sum_assignment(cost)
                match_for_exp = {int(i): int(j) for j, i in zip(row_ind, col_ind, strict=True)}
            else:
                # Opaque-cell-only, un-grounded: the value score is pairing-
                # independent (n_pairs*k - total_cost), so peeling exact rows is
                # safe. Reuse the vectorized interned matrix on the residual rows.
                assignment = peel_exact_row_matches(
                    act_rows, exp_rows, subfields=cost_names, fuzzy_field_thresholds=fuzzy
                )
                match_for_exp = {int(i): int(j) for j, i in assignment.pairs}
                if len(assignment.unmatched_actual_indices) > 0 and len(assignment.unmatched_expected_indices) > 0:
                    residual_actual = [act_rows[idx] for idx in assignment.unmatched_actual_indices]
                    residual_expected = [exp_rows[idx] for idx in assignment.unmatched_expected_indices]
                    # Memory ceiling on the residual assignment matrix (parallel to
                    # the _GROUNDED_MAX_CELLS grounding guard above). After the
                    # exact-row peel, a flat array whose prediction diverges leaves
                    # a residual ~= the full array; the dense int16 cost matrix plus
                    # scipy's internal float64 copy is tens of GB at ~50k+ rows and
                    # OOMs the eval runner (a 77,400-row array needs ~67 GB). Over
                    # the cell budget, skip the residual assignment: those rows stay
                    # unmatched (scored as value misses below) and the value score is
                    # flagged ``value_incomplete`` -- a lower bound, not a crash.
                    if len(residual_actual) * len(residual_expected) > _GROUNDED_MAX_CELLS:
                        c.value_incomplete = True
                    else:
                        cost = mismatch_cost_matrix(
                            residual_actual,
                            residual_expected,
                            subfields=cost_names,
                            fuzzy_field_thresholds=fuzzy,
                        )
                        row_ind, col_ind = linear_sum_assignment(cost)
                        match_for_exp.update(
                            {
                                int(assignment.unmatched_expected_indices[int(expected_idx)]): int(
                                    assignment.unmatched_actual_indices[int(actual_idx)]
                                )
                                for actual_idx, expected_idx in zip(row_ind, col_ind, strict=True)
                            }
                        )
        matched_act = set(match_for_exp.values())
        for i, erow in enumerate(exp_rows):
            ed = erow if isinstance(erow, Mapping) else {}
            mj = match_for_exp.get(i)
            if mj is not None:
                ad = act_rows[mj] if isinstance(act_rows[mj], Mapping) else {}
                for s in cell_names:
                    gt_path = f"{gt_field}[{i}].{s}"
                    pred_path = f"{pred_field}[{mj}].{s}"
                    if not giant:
                        self._note_grounded_claim(gt_path, pred_path, c)
                        if self._ev_pages.get(gt_path) and self._pred_pages.get(pred_path):
                            c.p_claims += 1
                    self._score_cell(gt_path, pred_path, ed.get(s), ad.get(s), s, c, ground=not giant)
                for s in object_array_names:  # recurse with the matched predicted index, not the GT index
                    self._score_array(
                        f"{gt_field}[{i}].{s}", f"{pred_field}[{mj}].{s}", ed.get(s), ad.get(s), item_sch.get(s, {}), c
                    )
                for s in object_names:
                    child_schema = item_sch.get(s, {})
                    in_e, child_ev = lookup(ed, s, child_schema)
                    in_a, child_av = lookup(ad, s, child_schema)
                    self._score_pair(
                        f"{gt_field}[{i}].{s}",
                        f"{pred_field}[{mj}].{s}",
                        in_e,
                        child_ev,
                        in_a,
                        child_av,
                        child_schema,
                        c,
                    )
            else:  # unmatched GT row: cell subfields already counted; recurse nested as recall misses
                self._count_unmatched_nested(
                    f"{gt_field}[{i}]", ed, item_sch, object_array_names + object_names, c, expected=True
                )
        for j, arow in enumerate(act_rows):  # extra predicted rows: nested fields are precision misses
            if j in matched_act:
                continue
            ad = arow if isinstance(arow, Mapping) else {}
            self._count_unmatched_nested(
                f"{pred_field}[{j}]", ad, item_sch, object_array_names + object_names, c, expected=False
            )

    def _score_scalar_cell(self, gt_path: str, pred_path: str, ev: Any, av: Any, c: _Counts) -> None:
        leaf = path_leaf(gt_path)
        c.expected += 1
        self._note_grounded_expected(gt_path, c)
        self._note_grounded_claim(gt_path, pred_path, c)
        if self._ev_pages.get(gt_path):
            c.p_expected += 1
            if self._pred_pages.get(pred_path):
                c.p_claims += 1
        self._score_cell(gt_path, pred_path, ev, av, leaf, c)
        c.predicted += 1

    def _score_pair(
        self,
        gt_path: str,
        pred_path: str,
        in_e: bool,
        ev: Any,
        in_a: bool,
        av: Any,
        schema: Mapping[str, Any],
        c: _Counts,
    ) -> None:
        # Presence after schema ``default``. The scorer itself is still chosen
        # from ``schema`` in ``_score_node`` / ``_count_subtree``, not from
        # whether ``ev``/``av`` is a list or dict.
        if in_e and in_a:
            self._score_node(gt_path, pred_path, ev, av, schema, c)
        elif in_e:
            self._count_subtree(gt_path, ev, schema, c, expected=True)
        elif in_a:
            self._count_subtree(pred_path, av, schema, c, expected=False)
        # neither: no-op — do not treat "both missing" as a precision claim.

    def _score_node(
        self, gt_path: str, pred_path: str, ev: Any, av: Any, schema: Mapping[str, Any], c: _Counts
    ) -> None:
        # Schema-only dispatch: gold/pred Python types do not select Hungarian
        # vs object-walk vs scalar. ``as_rows`` / ``{}`` only coerce a mismatched
        # instance once that scorer is already chosen.
        if is_array_schema(schema):
            # List-of-objects: Hungarian table. Array of primitives (string[] /
            # number[] / …, incl. null combinators): one opaque cell, same as a
            # string field. Empty items.properties is the primitive-array shape,
            # not a skip — that used to drop nested lists like vendor.tags.
            if is_object_array_schema(schema):
                self._score_array(gt_path, pred_path, ev, av, schema, c)
            else:
                self._score_scalar_cell(gt_path, pred_path, ev, av, c)
            return
        if is_object_schema(schema):
            props = schema_properties(schema)
            ed = ev if isinstance(ev, Mapping) else {}
            ad = av if isinstance(av, Mapping) else {}
            keys = list(props.keys())
            if len(keys) == 0:
                self._score_scalar_cell(gt_path, pred_path, ev, av, c)
                return
            for key in keys:
                child_gt = f"{gt_path}.{key}" if gt_path else key
                child_pred = f"{pred_path}.{key}" if pred_path else key
                child_schema = props.get(key, {})
                in_e, child_ev = lookup(ed, key, child_schema)
                in_a, child_av = lookup(ad, key, child_schema)
                self._score_pair(child_gt, child_pred, in_e, child_ev, in_a, child_av, child_schema, c)
            return
        self._score_scalar_cell(gt_path, pred_path, ev, av, c)

    def score_root(
        self, expected: Mapping[str, Any], actual: Mapping[str, Any], schema_props: Mapping[str, Any]
    ) -> _Counts:
        c = _Counts()
        # Schema names first: a defaulted field omitted on both sides must still
        # compare. Instance extras (keys not in the schema) stay in the union so
        # a hallucinated key is still a scalar cell — shape of that cell is
        # still schema (empty → opaque), not the Python type of the value.
        names = (set(schema_props) | set(expected) | set(actual)) - RESERVED_OUTPUT_KEYS
        for name in sorted(names):
            schema = schema_props.get(name, {})
            in_e, ev = lookup(expected, name, schema)
            in_a, av = lookup(actual, name, schema)
            self._score_pair(name, name, in_e, ev, in_a, av, schema, c)
        return c


def build_rule_indexes(
    field_rules: list[ExtractFieldTestRule],
) -> tuple[
    dict[str, list[Any]],
    BoxIndex,
    dict[str, set[int]],
    dict[str, tuple[str, ...]],
]:
    """field_path -> alternate values; -> evidence boxes; -> evidence pages; -> normalizers.

    ``pages`` is a superset of the pages implied by ``boxes``: it holds every
    page any evidence entry carries, even entries with no bbox, so a page-only
    annotation still makes a cell page-gradeable.
    """
    alt: dict[str, list[Any]] = {}
    boxes: BoxIndex = {}
    pages: dict[str, set[int]] = {}
    normalizers: dict[str, tuple[str, ...]] = {}
    for rule in field_rules:
        path = rule.field_path
        if rule.normalizers:
            normalizers[path] = tuple(rule.normalizers)
        entries = iter_rule_evidence(rule)
        vals = [e.value for e in entries if e.value is not None]
        if vals:
            alt.setdefault(path, []).extend(vals)
        for e in entries:
            if e.page is not None:
                pages.setdefault(path, set()).add(int(e.page))
                parsed = _validated_bbox(e.bbox) if e.bbox is not None else None
                if parsed is not None:
                    boxes.setdefault(path, []).append((int(e.page), parsed))
    return alt, boxes, pages, normalizers


def index_citations(
    field_citations: list[Any],
) -> tuple[BoxIndex, dict[str, set[int]]]:
    """field_path -> citation boxes; field_path -> citation pages.

    A page-only citation (page present, bbox missing) is indexed for pages but
    not for boxes, so it can satisfy a page match without ever being credited as
    a bbox match.
    """
    boxes: BoxIndex = {}
    pages: dict[str, set[int]] = {}
    for cit in field_citations or []:
        path = cit.get("field_path") if isinstance(cit, dict) else getattr(cit, "field_path", None)
        page = cit.get("page") if isinstance(cit, dict) else getattr(cit, "page", None)
        bbox = cit.get("bbox") if isinstance(cit, dict) else getattr(cit, "bbox", None)
        if not path or page is None:
            continue
        pages.setdefault(path, set()).add(int(page))
        parsed = _validated_bbox(bbox) if isinstance(bbox, (list, tuple)) else None
        if parsed is not None:
            boxes.setdefault(path, []).append((int(page), parsed))
    return boxes, pages


def _prf(tp: int, denom_p: int, denom_r: int) -> tuple[float, float, float]:
    precision = tp / denom_p if denom_p else 0.0
    recall = tp / denom_r if denom_r else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


def compute_unified_evidence_metrics(
    expected_output: Any,
    extracted_data: Any,
    field_rules: list[ExtractFieldTestRule],
    field_citations: list[Any] | None = None,
    data_schema: dict[str, Any] | None = None,
    *,
    fuzzy_field_thresholds: Mapping[str, float] | None = None,
    normalize_dates: bool = True,
    bbox_iou_threshold: float = 0.5,
) -> list[MetricValue]:
    """Value + grounded precision/recall/F1 under keyless Hungarian alignment.

    Returns an empty list when either side is not a dict (no array structure to
    score), matching ``array_record``'s ``counts is None`` guard.
    """
    expected = unwrap_value(expected_output)
    actual = unwrap_value(extracted_data)
    if not isinstance(expected, Mapping) or not isinstance(actual, Mapping):
        return []
    if normalize_dates:
        expected = normalize_dates_deep(expected)
        actual = normalize_dates_deep(actual)

    fuzzy = dict(DEFAULT_FUZZY_FIELD_THRESHOLDS if fuzzy_field_thresholds is None else fuzzy_field_thresholds)
    alt, ev_boxes, ev_pages, normalizers = build_rule_indexes(field_rules)
    if normalize_dates:
        alt = {p: normalize_dates_deep(v) for p, v in alt.items()}
    pred_boxes, pred_pages = index_citations(field_citations or [])
    # Same inliner extract GT lint uses: root ``#/$defs/X`` (and ``definitions``).
    resolved = resolve_refs(data_schema) if isinstance(data_schema, Mapping) else {}
    schema_props = schema_properties(resolved)

    c = Scorer(
        alt_values=alt,
        evidence_boxes=ev_boxes,
        evidence_pages=ev_pages,
        normalizers=normalizers,
        pred_boxes=pred_boxes,
        pred_pages=pred_pages,
        fuzzy=fuzzy,
        iou_threshold=bbox_iou_threshold,
    ).score_root(expected, actual, schema_props)
    if c.expected == 0 and c.predicted == 0:
        return []

    vp, vr, vf1 = _prf(c.v_correct, c.predicted, c.expected)
    meta = {
        "v_correct": c.v_correct,
        "expected_cells": c.expected,
        "predicted_cells": c.predicted,
        "g_correct": c.g_correct,
        "grounded_expected_cells": c.g_expected,
        "grounded_pred_claims": c.g_claims,
        "p_correct": c.p_correct,
        "page_expected_cells": c.p_expected,
        "page_pred_claims": c.p_claims,
        "expected_rows": c.expected_rows,
        "predicted_rows": c.predicted_rows,
        "bbox_iou_threshold": bbox_iou_threshold,
        "grounded_incomplete": c.grounded_incomplete,
        "value_incomplete": c.value_incomplete,
    }
    metrics = [
        MetricValue(metric_name="extract_unified_value_precision", value=vp, metadata={**meta, "tp": c.v_correct}),
        MetricValue(metric_name="extract_unified_value_recall", value=vr, metadata={**meta, "tp": c.v_correct}),
        MetricValue(metric_name="extract_unified_value_f1", value=vf1, metadata=meta),
    ]
    # Page grounding is the coarse counterpart of the bbox grounding below:
    # value + correct-page, no IoU. It is emitted under the SAME
    # bbox-bearing-only guard shape as grounded (``c.p_expected`` counts GT cells
    # whose evidence carries a page; ``grounded_incomplete`` withholds it when a
    # giant array skipped pairing-sensitive scoring), and it sits between value
    # and grounded so the three F1s nest: value_f1 >= page_f1 >= grounded_f1.
    # (Page is a per-cell superset of bbox -- box match implies page match and
    # page evidence/claims are supersets of bbox evidence/claims -- so a
    # document that emits a grounded metric always emits a page metric too.)
    if c.p_expected > 0 and not c.grounded_incomplete:
        pp, pr, pf1 = _prf(c.p_correct, c.p_claims, c.p_expected)
        metrics += [
            MetricValue(metric_name="extract_unified_page_precision", value=pp, metadata={**meta, "tp": c.p_correct}),
            MetricValue(metric_name="extract_unified_page_recall", value=pr, metadata={**meta, "tp": c.p_correct}),
            MetricValue(metric_name="extract_unified_page_f1", value=pf1, metadata=meta),
        ]
    # Grounding is only defined where the ground truth carries bounding boxes.
    # A document whose GT has NO evidence bbox (``c.g_expected == 0``) cannot be
    # scored for grounding at all, so we emit NO ``*_grounded_*`` metric for it
    # -- rather than a 0.0 that the runner would average in as if the pipeline
    # had failed to ground. This makes the dataset-level grounded P/R/F1 an
    # average over bbox-bearing documents only (mirroring how the runner already
    # excludes docs that don't emit a metric). Within a bbox-bearing document the
    # same principle applies per cell: precision's denominator (``g_claims``)
    # only counts citation-bbox claims on cells aligned to a bbox-bearing GT
    # cell, so a sparsely annotated GT (few bbox cells among many value-only
    # cells) grades only what it can verify instead of pinning precision to ~0.
    # A document whose GT *does* carry boxes but whose prediction emits none on
    # those cells still scores 0.0 here (``g_claims``/``g_correct == 0``): that
    # is a real grounding miss and the 0.0 is kept.
    # ``grounded_incomplete`` means at least one giant array's grounding was
    # skipped to bound eval memory, so any grounded score would be partial and
    # misleading. Withhold all grounded metrics for the document instead (the
    # value metrics above are unaffected -- they are exact).
    if c.g_expected > 0 and not c.grounded_incomplete:
        gp, gr, gf1 = _prf(c.g_correct, c.g_claims, c.g_expected)
        metrics += [
            MetricValue(
                metric_name="extract_unified_grounded_precision", value=gp, metadata={**meta, "tp": c.g_correct}
            ),
            MetricValue(metric_name="extract_unified_grounded_recall", value=gr, metadata={**meta, "tp": c.g_correct}),
            MetricValue(metric_name="extract_unified_grounded_f1", value=gf1, metadata=meta),
        ]
    return metrics
