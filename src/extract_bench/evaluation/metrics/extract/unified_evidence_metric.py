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
   scorer. The same lookup is used at the root, inside nested objects, on
   object columns of array rows after Hungarian pairing, on opaque array
   cells, and when reading nested leaves for pairing cost. A missing key is
   equal only to another missing key, unless ``default`` is present (then
   missing also equals that default). Explicit ``0`` is never equal to
   explicit ``null``. Row-pairing cost uses those same children (and nested
   object-arrays) so Hungarian maximizes the cells F1 actually scores.
   Scalar lists stay opaque.
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
   ``*_page_*`` are one nested precision/recall/F1 trio.
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

   How a citation is matched against a cell's evidence boxes is selected by
   ``bbox_match_mode``. The default ``"envelope"`` reads the entries as an OR
   of ANDs: a ``coarse`` entry is the envelope of the precise entries it
   encloses on its page (a value printed on several lines carries one entry
   per line plus their union, flagged coarse), and a citation matches the
   envelope only when it covers it -- one box at IoU >= threshold, or several
   boxes (one per line) whose union is -- AND touches every enclosed line. A
   citation of one line of a paragraph-long value is not grounded. A precise
   entry outside every envelope, and a coarse entry enclosing no precise
   entry, is matched by one citation box at IoU >= threshold, so GT without
   coarse envelopes grades exactly as before. ``"any"`` is that original flat
   OR for every entry (any citation box against any evidence box).

On a document with no object-array subfields, no object-valued cells, no
alternate evidence values, and no omit/default split, the ``*_value_*``
metrics match ``array_record_*`` alignment. ``array_record`` reads
omitted keys with ``.get`` (implicit ``null``); this scorer uses ``lookup``.
Object-array subfields and objects recursed as child cells make the value
metrics diverge from array_record (per-child credit); the
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
# One gradeable location under ``bbox_match_mode="envelope"``: the page, the box
# a citation must cover, and the precise boxes it encloses (the lines of a value
# printed on several lines) that the citation must all touch. Empty parts = a
# plain single-box IoU target.
type EvidenceTarget = tuple[int, BBox, tuple[BBox, ...]]
type TargetIndex = dict[str, Sequence[EvidenceTarget]]

# ``envelope``: OR of ANDs -- a coarse evidence entry is the envelope of the
# precise entries it encloses and a citation must cover all of it (default).
# ``any``: the original flat OR -- any citation box against any evidence box.
BBOX_MATCH_MODES: frozenset[str] = frozenset({"envelope", "any"})
DEFAULT_BBOX_MATCH_MODE = "envelope"
# A precise entry is a part of a coarse entry on the same page when it lies
# inside it. Envelopes are the union of their lines, so containment is 1.0 up
# to 5-decimal rounding on stored boxes. 0.99 admits that rounding without
# treating a nearby sibling line as enclosed.
ENVELOPE_PART_CONTAINMENT = 0.99
# Gold may record the envelope twice: once as the coarse union and once as a
# precise box over the same lines. That precise twin is folded into the coarse
# target: it is not a part (a per-line citation could never cover it) and not a
# target of its own (against it alone, one of two equal rows would reach IoU
# 0.5 and re-admit the single-line citation the envelope rejects).
ENVELOPE_DUPLICATE_IOU = 0.9
# A part is touched by a citation box that covers at least this fraction of it.
ENVELOPE_PART_COVERAGE = 0.5


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


class _Absent:
    """Sentinel for a key that is missing and has no schema ``default``.

    Distinct from JSON ``null`` so pairing can treat missing==missing and
    missing!=null. A singleton so interned cost matrices hash it stably.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "ABSENT"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Absent)

    def __hash__(self) -> int:
        return 0xAB5E11D


ABSENT = _Absent()


def _read_segments(
    row: Any,
    *,
    segments: Sequence[str],
    item_sch: Mapping[str, Any],
) -> Any:
    """Walk successive mapping keys, applying ``lookup`` at each step.

    Keys are not split, so a schema field named ``a.b`` is one step, not two.
    Omitted keys use JSON Schema ``default`` when present, matching F1.
    A missing key with no ``default`` stops the walk as ``ABSENT``, not
    ``None`` (explicit null). A present non-object (including explicit
    ``null``) is coerced to ``{}`` before the next step, the same way
    ``_score_node`` walks children.
    """
    cur: Any = row if isinstance(row, Mapping) else {}
    field_schema: Any = {}
    for i, part in enumerate(segments):
        if i == 0:
            field_schema = item_sch.get(part, {})
        else:
            props = schema_properties(field_schema) if isinstance(field_schema, Mapping) else {}
            field_schema = props.get(part, {})
        if not isinstance(cur, Mapping):
            cur = {}
        present, cur = lookup(cur, part, field_schema)
        if not present:
            return ABSENT
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
    item_sch: Mapping[str, Any],
) -> Sequence[Mapping[PairingPath, Any]]:
    return [{path: _read_segments(row, segments=path, item_sch=item_sch) for path in paths} for row in rows]


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
                    child_schema = item_sch.get(name, {})
                    in_side, child_val = lookup(rd, name, child_schema)
                    if in_side:
                        total += _value_leaf_count(child_val, schema=child_schema)
            return total
        return 1
    if is_object_schema(schema):
        props = schema_properties(schema)
        if len(props) == 0:
            return 1
        rd = value if isinstance(value, Mapping) else {}
        total = 0
        for key, child in props.items():
            in_side, child_val = lookup(rd, key, child)
            if in_side:
                total += _value_leaf_count(child_val, schema=child)
        return total
    return 1


def _row_leaf_count(
    row: Any,
    *,
    item_sch: Mapping[str, Any],
    names: Sequence[str],
) -> int:
    rd = row if isinstance(row, Mapping) else {}
    total = 0
    for name in names:
        child_schema = item_sch.get(name, {})
        in_side, child_val = lookup(rd, name, child_schema)
        if in_side:
            total += _value_leaf_count(child_val, schema=child_schema)
    return total


def _finite_positive(*boxes: BBox) -> bool:
    """Every coordinate finite and every box of positive size."""
    return all(math.isfinite(v) for box in boxes for v in box) and all(box[2] > 0 and box[3] > 0 for box in boxes)


def _inter_xywh(a: BBox, b: BBox) -> float:
    """Intersection area; 0 when either box is non-finite or has no size."""
    if not _finite_positive(a, b):
        return 0.0
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    return ix * iy


def iou_xywh(a: BBox, b: BBox) -> float:
    if not _finite_positive(a, b):
        return 0.0
    inter = _inter_xywh(a, b)
    union = a[2] * a[3] + b[2] * b[3] - inter
    if union <= 0:
        return 0.0
    # Reconstructing width as (x+w)-x is a few ulps off for values like 0.1,
    # so the raw ratio can exceed 1.0 even for identical boxes.
    # min(1.0, nan) is 1.0 in Python, so non-finite ratios must not clamp to a hit.
    ratio = inter / union
    return min(1.0, ratio) if math.isfinite(ratio) else 0.0


def contained_fraction_xywh(inner: BBox, outer: BBox) -> float:
    """Fraction of ``inner``'s area inside ``outer``; 0 for a degenerate or non-finite box."""
    if not _finite_positive(inner, outer):
        return 0.0
    ratio = _inter_xywh(inner, outer) / (inner[2] * inner[3])
    return min(1.0, ratio) if math.isfinite(ratio) else 0.0


def union_xywh(boxes: Sequence[BBox]) -> BBox | None:
    """Smallest xywh box enclosing every finite box; None when there is none."""
    finite = [b for b in boxes if _finite_positive(b)]
    if len(finite) == 0:
        return None
    x0 = min(b[0] for b in finite)
    y0 = min(b[1] for b in finite)
    x1 = max(b[0] + b[2] for b in finite)
    y1 = max(b[1] + b[3] for b in finite)
    return (x0, y0, x1 - x0, y1 - y0)


def _flat_targets(boxes: BoxIndex) -> TargetIndex:
    """Every box a target of its own (no envelope structure): ``envelope`` mode then equals ``any``."""
    return {path: [(page, box, ()) for page, box in entries] for path, entries in boxes.items()}


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
        evidence_targets: TargetIndex | None = None,
        bbox_match_mode: str = DEFAULT_BBOX_MATCH_MODE,
    ) -> None:
        if bbox_match_mode not in BBOX_MATCH_MODES:
            raise ValueError(f"unknown bbox_match_mode {bbox_match_mode!r}; supported: {sorted(BBOX_MATCH_MODES)}")
        self._alt = alt_values
        self._ev_boxes = evidence_boxes
        self._ev_pages = evidence_pages
        self._ev_targets = _flat_targets(evidence_boxes) if evidence_targets is None else evidence_targets
        self._mode = bbox_match_mode
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
        item_sch: Mapping[str, Any],
    ) -> bool:
        """True when this level's opaque pairing cells have a distinct alt or a normalizer.

        Nested object-arrays check their own leaves.
        """
        if len(pairing_paths) == 0 or (len(self._alt) == 0 and len(self._normalizers) == 0):
            return False
        for i, erow in enumerate(exp_rows):
            for path in pairing_paths:
                gt_path = f"{gt_field}[{i}].{_gt_rel_path(path)}"
                normalizers = self._normalizers.get(gt_path)
                if normalizers is not None and len(normalizers) > 0:
                    return True
                alt = self._alt.get(gt_path)
                if alt is None or len(alt) == 0:
                    continue
                canon = _read_segments(erow, segments=path, item_sch=item_sch)
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
            use_alts=self._gold_needs_configured_match(
                exp_rows=exp_rows, pairing_paths=inner_paths, gt_field=gt_path, item_sch=item_sch
            ),
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
                    canon = _read_segments(erow, segments=path, item_sch=item_sch)
                    alt = self._alt.get(f"{gt_field}[{i}].{_gt_rel_path(path)}")
                    cand = [canon]
                    if alt:
                        cand += [v for v in alt if v != canon and v not in cand]
                    row_cands.append(cand)
                exp_cands.append(row_cands)
            for j, arow in enumerate(act_rows):
                avals = [_read_segments(arow, segments=path, item_sch=item_sch) for path in paths]
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
                _flatten_pairing_rows(act_rows, paths=paths, item_sch=item_sch),
                _flatten_pairing_rows(exp_rows, paths=paths, item_sch=item_sch),
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
                            _read_segments(erow, segments=path, item_sch=item_sch),
                            _read_segments(arow, segments=path, item_sch=item_sch),
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

    def _gt_targets(self, gt_path: str) -> Sequence[EvidenceTarget] | None:
        return self._ev_targets.get(gt_path)

    def _target_hit(self, target: EvidenceTarget, pred: list[PageBBox]) -> bool:
        return evidence_target_hit(target, pred, iou_threshold=self._iou)

    def _box_match(self, gt_path: str, pred_path: str) -> bool:
        pred = self._pred_boxes.get(pred_path)
        if pred is None:
            return False
        if self._mode == "envelope":
            targets = self._gt_targets(gt_path)
            if targets is None:
                return False
            return any(self._target_hit(t, pred) for t in targets)
        gt_boxes = self._ev_boxes.get(gt_path)
        if gt_boxes is None:
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

        ``ground=False`` scores value only: giant arrays skip grounding-family
        counters (g_expected/g_claims and p_expected/p_claims), and this keeps
        g_correct/p_correct on the same skipped set.
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

    def _count_subtree(
        self, path: str, value: Any, schema: Mapping[str, Any], c: _Counts, *, expected: bool, ground: bool = True
    ) -> None:
        """Count a one-sided subtree's leaves (recall-only or precision-only miss).

        Unmatched rows and a key present on only one side have no partner, so
        every leaf ``lookup`` reports present is a miss. Children ``lookup``
        reports absent (no ``default``) are not cells.
        """
        if is_array_schema(schema):
            if is_object_array_schema(schema):
                subfield_names = array_subfield_names(schema)
                item_sch = array_item_properties(schema)
                for i, row in enumerate(as_rows(value)):
                    rd = row if isinstance(row, Mapping) else {}
                    for s in subfield_names:
                        child_schema = item_sch.get(s, {})
                        in_side, child_val = lookup(rd, s, child_schema)
                        if in_side:
                            self._count_subtree(
                                f"{path}[{i}].{s}", child_val, child_schema, c, expected=expected, ground=ground
                            )
                return
        elif is_object_schema(schema):
            props = schema_properties(schema)
            if len(props) > 0:
                rd = value if isinstance(value, Mapping) else {}
                for key, child_schema in props.items():
                    child = f"{path}.{key}" if len(path) > 0 else key
                    in_side, child_val = lookup(rd, key, child_schema)
                    if in_side:
                        self._count_subtree(child, child_val, child_schema, c, expected=expected, ground=ground)
                return
        if expected:
            c.expected += 1
            if ground:
                self._note_grounded_expected(path, c)
                pages = self._ev_pages.get(path)
                if pages is not None and len(pages) > 0:
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
        ground: bool = True,
    ) -> None:
        for s in nested_names:
            child_schema = item_sch.get(s, {})
            in_side, child_val = lookup(row, s, child_schema)
            if in_side:
                self._count_subtree(f"{base}.{s}", child_val, child_schema, c, expected=expected, ground=ground)

    def _score_array(
        self,
        gt_field: str,
        pred_field: str,
        exp_rows: Any,
        act_rows: Any,
        schema: Mapping[str, Any],
        c: _Counts,
        *,
        ground: bool = True,
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
        # Opaque, object, and object-array cells are counted via ``lookup`` /
        # ``_score_pair`` below (omit with no ``default`` is not a cell).
        c.expected_rows += len(exp_rows)
        c.predicted_rows += len(act_rows)
        # A flat array whose full grounded matrix would be multi-GB: skip its
        # grounding-family scoring (bbox AND page). Leaving has_ev_page/
        # has_pred_page False routes the value score to the exact-row peel below
        # (value-identical for opaque cells), and not counting
        # g_expected/g_claims/p_expected/p_claims here keeps the grounded and
        # page denominators consistent with the skipped g_correct/p_correct.
        # ``len(object_array_names) == 0 and len(object_names) == 0`` guarantees
        # no nested recursion, so the peel cannot
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
        score_ground = ground and not giant
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
            dropped_grounding = False
            for i in range(len(exp_rows)):
                for s in cell_names:
                    pages = self._ev_pages.get(f"{gt_field}[{i}].{s}")
                    if pages is not None and len(pages) > 0:
                        dropped_grounding = True
                        break
                if dropped_grounding:
                    break
            if not dropped_grounding:
                for j in range(len(act_rows)):
                    for s in cell_names:
                        pages = self._pred_pages.get(f"{pred_field}[{j}].{s}")
                        if pages is not None and len(pages) > 0:
                            dropped_grounding = True
                            break
                    if dropped_grounding:
                        break
            if dropped_grounding:
                c.grounded_incomplete = True
        else:
            for i in range(len(exp_rows)):
                for s in cell_names:
                    gt_path = f"{gt_field}[{i}].{s}"
                    pages = self._ev_pages.get(gt_path)
                    if pages is not None and len(pages) > 0:
                        has_ev_page = True
                        break
                if has_ev_page:
                    break
            # Flag-only scan: g_claims/p_claims are counted per aligned pair
            # below, and only for cells whose GT side carries a bbox/page
            # (ungradeable claims stay out of the precision denominator). The flag
            # reflects ANY predicted page (the superset of any predicted bbox) so
            # the pairing-sensitive branch selection below covers both families.
            for j in range(len(act_rows)):
                for s in cell_names:
                    pages = self._pred_pages.get(f"{pred_field}[{j}].{s}")
                    if pages is not None and len(pages) > 0:
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
            pairing_paths, path_schemas, _ = _item_pairing_fields(item_sch=item_sch, subfield_names=subfield_names)
            path_fuzzy = _fuzzy_for_paths(pairing_paths, fuzzy=self._fuzzy)
            # Distinct alts / normalizers on *this* level's opaque cells expand the
            # zero-cost graph, so skip the exact-row peel. Nested object-arrays
            # detect configured match for themselves when they add their cost.
            if self._gold_needs_configured_match(
                exp_rows=exp_rows, pairing_paths=pairing_paths, gt_field=gt_field, item_sch=item_sch
            ):
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
                # safe. Flatten through ``lookup`` so omit/default/null match F1.
                flat_act = list(_flatten_pairing_rows(act_rows, paths=pairing_paths, item_sch=item_sch))
                flat_exp = list(_flatten_pairing_rows(exp_rows, paths=pairing_paths, item_sch=item_sch))
                assignment = peel_exact_row_matches(
                    flat_act,
                    flat_exp,
                    subfields=pairing_paths,
                    fuzzy_field_thresholds=path_fuzzy,
                    field_schemas=path_schemas,
                )
                match_for_exp = {int(i): int(j) for j, i in assignment.pairs}
                if len(assignment.unmatched_actual_indices) > 0 and len(assignment.unmatched_expected_indices) > 0:
                    residual_actual = [flat_act[idx] for idx in assignment.unmatched_actual_indices]
                    residual_expected = [flat_exp[idx] for idx in assignment.unmatched_expected_indices]
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
                            subfields=pairing_paths,
                            fuzzy_field_thresholds=path_fuzzy,
                            field_schemas=path_schemas,
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
                for s in subfield_names:
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
                        ground=score_ground,
                    )
            else:
                self._count_unmatched_nested(
                    f"{gt_field}[{i}]",
                    ed,
                    item_sch,
                    subfield_names,
                    c,
                    expected=True,
                    ground=score_ground,
                )
        for j, arow in enumerate(act_rows):
            if j in matched_act:
                continue
            ad = arow if isinstance(arow, Mapping) else {}
            self._count_unmatched_nested(
                f"{pred_field}[{j}]",
                ad,
                item_sch,
                subfield_names,
                c,
                expected=False,
                ground=score_ground,
            )

    def _score_scalar_cell(
        self, gt_path: str, pred_path: str, ev: Any, av: Any, c: _Counts, *, ground: bool = True
    ) -> None:
        leaf = path_leaf(gt_path)
        c.expected += 1
        if ground:
            self._note_grounded_expected(gt_path, c)
            self._note_grounded_claim(gt_path, pred_path, c)
            ev_pages = self._ev_pages.get(gt_path)
            if ev_pages is not None and len(ev_pages) > 0:
                c.p_expected += 1
                pred_pages = self._pred_pages.get(pred_path)
                if pred_pages is not None and len(pred_pages) > 0:
                    c.p_claims += 1
        self._score_cell(gt_path, pred_path, ev, av, leaf, c, ground=ground)
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
        *,
        ground: bool = True,
    ) -> None:
        # ``in_e``/``in_a`` are ``lookup`` presence (schema ``default`` already
        # applied). ``_score_node`` / ``_count_subtree`` pick the scorer from
        # ``schema``.
        if in_e and in_a:
            self._score_node(gt_path, pred_path, ev, av, schema, c, ground=ground)
        elif in_e:
            self._count_subtree(gt_path, ev, schema, c, expected=True, ground=ground)
        elif in_a:
            self._count_subtree(pred_path, av, schema, c, expected=False, ground=ground)
        # neither: both absent is not a cell.

    def _score_node(
        self,
        gt_path: str,
        pred_path: str,
        ev: Any,
        av: Any,
        schema: Mapping[str, Any],
        c: _Counts,
        *,
        ground: bool = True,
    ) -> None:
        # Schema type chooses Hungarian vs object-walk vs scalar. ``as_rows`` /
        # ``{}`` coerce a mismatched instance after that choice.
        if is_array_schema(schema):
            # List-of-objects: Hungarian table. Primitive arrays (string[] /
            # number[] / …, including null combinators and empty
            # items.properties) are one opaque cell, the same as a string field.
            if is_object_array_schema(schema):
                self._score_array(gt_path, pred_path, ev, av, schema, c, ground=ground)
            else:
                self._score_scalar_cell(gt_path, pred_path, ev, av, c, ground=ground)
            return
        if is_object_schema(schema):
            props = schema_properties(schema)
            ed = ev if isinstance(ev, Mapping) else {}
            ad = av if isinstance(av, Mapping) else {}
            keys = list(props.keys())
            if len(keys) == 0:
                self._score_scalar_cell(gt_path, pred_path, ev, av, c, ground=ground)
                return
            for key in keys:
                child_gt = f"{gt_path}.{key}" if len(gt_path) > 0 else key
                child_pred = f"{pred_path}.{key}" if len(pred_path) > 0 else key
                child_schema = props.get(key, {})
                in_e, child_ev = lookup(ed, key, child_schema)
                in_a, child_av = lookup(ad, key, child_schema)
                self._score_pair(child_gt, child_pred, in_e, child_ev, in_a, child_av, child_schema, c, ground=ground)
            return
        self._score_scalar_cell(gt_path, pred_path, ev, av, c, ground=ground)

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


def build_evidence_targets(field_rules: Sequence[ExtractFieldTestRule]) -> TargetIndex:
    """field_path -> evidence targets for ``bbox_match_mode="envelope"``.

    Per rule and page, a ``coarse`` entry is an envelope whose parts are the
    precise entries with at least ENVELOPE_PART_CONTAINMENT of their area
    inside it (the lines of a value printed on several lines); a precise
    entry enclosed by no envelope is a target of its own with no parts. A
    precise entry that is essentially the envelope itself (IoU >=
    ENVELOPE_DUPLICATE_IOU) is folded into it. Every path indexed by
    ``build_rule_indexes`` gets at least one target, so the grounded
    denominators do not depend on the mode.
    """
    items: dict[str, list[tuple[int, BBox, bool]]] = {}
    for rule in field_rules:
        for e in iter_rule_evidence(rule):
            if e.page is None or e.bbox is None:
                continue
            parsed = _validated_bbox(e.bbox)
            if parsed is None:
                continue
            items.setdefault(rule.field_path, []).append((int(e.page), parsed, bool(e.coarse)))
    return {path: group_evidence_targets(entries) for path, entries in items.items()}


def group_evidence_targets(entries: Sequence[tuple[int, BBox, bool]]) -> Sequence[EvidenceTarget]:
    """Group coarse/precise evidence boxes on a page into envelope targets.

    Per page, a ``coarse`` entry is an envelope whose parts are the precise
    entries with at least ENVELOPE_PART_CONTAINMENT of their area inside it; a
    precise entry enclosed by no envelope is a target of its own with no parts.
    A precise entry that is essentially the envelope itself (IoU >=
    ENVELOPE_DUPLICATE_IOU) is folded into it.
    """
    out: list[EvidenceTarget] = []
    for page in dict.fromkeys(p for p, _, _ in entries):
        precise = [b for p, b, coarse in entries if p == page and not coarse]
        enclosed: set[int] = set()
        for p, box, coarse in entries:
            if p != page or not coarse:
                continue
            parts: list[BBox] = []
            for i, pb in enumerate(precise):
                if contained_fraction_xywh(pb, box) < ENVELOPE_PART_CONTAINMENT:
                    continue
                enclosed.add(i)
                if iou_xywh(pb, box) < ENVELOPE_DUPLICATE_IOU:
                    parts.append(pb)
            out.append((page, box, tuple(parts)))
        out.extend((page, b, ()) for i, b in enumerate(precise) if i not in enclosed)
    return out


def evidence_target_hit(target: EvidenceTarget, pred: Sequence[PageBBox], *, iou_threshold: float) -> bool:
    """One evidence target against citation boxes (``envelope`` mode).

    A target with no parts (a single-line value; a coarse box enclosing no
    precise entry) is hit by one citation box at IoU >= threshold, exactly
    as in ``any`` mode. An envelope with parts is hit when the citation
    covers it -- one box at IoU >= threshold, or several boxes (one per
    line) whose union is -- AND every part is touched by some citation box
    (>= ENVELOPE_PART_COVERAGE of the part's area). A citation of one line
    of a value printed on several lines therefore never hits its envelope,
    even when that line alone reaches IoU 0.5 (two equally wide rows).
    """
    page, box, parts = target
    boxes = [pb for pp, pb in pred if pp == page]
    if len(boxes) == 0:
        return False
    if any(iou_xywh(box, pb) >= iou_threshold for pb in boxes):
        covered = True
    elif len(parts) > 0:
        touching = [pb for pb in boxes if _inter_xywh(box, pb) > 0.0]
        union = union_xywh(touching) if len(touching) > 0 else None
        covered = union is not None and iou_xywh(box, union) >= iou_threshold
    else:
        covered = False
    if not covered:
        return False
    return all(any(contained_fraction_xywh(part, pb) >= ENVELOPE_PART_COVERAGE for pb in boxes) for part in parts)


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
    bbox_match_mode: str = DEFAULT_BBOX_MATCH_MODE,
) -> list[MetricValue]:
    """Value + grounded precision/recall/F1 under keyless Hungarian alignment.

    ``bbox_match_mode`` selects how a citation is graded against a cell's
    evidence boxes: ``"envelope"`` (default; a coarse entry must be covered
    whole, see the module docstring) or ``"any"`` (any box against any box).

    Returns an empty list when either side is not a dict (no array structure to
    score), matching ``array_record``'s ``counts is None`` guard.
    """
    if bbox_match_mode not in BBOX_MATCH_MODES:
        raise ValueError(f"unknown bbox_match_mode {bbox_match_mode!r}; supported: {sorted(BBOX_MATCH_MODES)}")
    expected = unwrap_value(expected_output)
    actual = unwrap_value(extracted_data)
    if not isinstance(expected, Mapping) or not isinstance(actual, Mapping):
        return []
    if normalize_dates:
        expected = normalize_dates_deep(expected)
        actual = normalize_dates_deep(actual)

    fuzzy = dict(DEFAULT_FUZZY_FIELD_THRESHOLDS if fuzzy_field_thresholds is None else fuzzy_field_thresholds)
    alt, ev_boxes, ev_pages, normalizers = build_rule_indexes(field_rules)
    ev_targets = build_evidence_targets(field_rules) if bbox_match_mode == "envelope" else {}
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
        evidence_targets=ev_targets if bbox_match_mode == "envelope" else None,
        bbox_match_mode=bbox_match_mode,
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
        "bbox_match_mode": bbox_match_mode,
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
