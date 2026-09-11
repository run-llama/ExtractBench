# Changelog

All notable changes to `extract-bench` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

Pulls the extract scorer into alignment with the internal LlamaCloud benchmark
harness ahead of the first PyPI release.

### Scoring (changes evaluation numbers)
- Unified extract F1: a key omitted from a prediction resolves to the JSON
  Schema `default` when the property declares one; without a default it is a
  one-sided miss rather than an implicit `null` prediction. Omitted, explicit
  `null` and `0` are three distinct cells (`0 != null`; an omitted key equals
  `0` only under `"default": 0`). Defaulted fields omitted on both sides count
  as matches.
- Unified extract F1: nested primitive arrays (`tags: ["a", "b"]`) score as one
  opaque cell instead of being dropped from the cell count.
- Unified extract F1: Hungarian row pairing for object arrays counts nested
  object children and nested object arrays in the assignment cost, and reads
  nested leaves through the same schema-default lookup as scoring, so rows are
  aligned on the full leaf set F1 actually grades.
- Unified grounding: envelope match mode (`bbox_match_mode="envelope"`, the
  default). A `coarse` evidence entry is graded as an AND over the precise
  boxes it encloses on its page; a citation must cover the whole envelope, not
  one of its lines. Identical to the previous behaviour on ground truth with no
  coarse entries.
- Unified grounding: non-finite or non-positive predicted boxes score as
  misses instead of propagating `NaN` through IoU.
- JSON subset match: ground truth is no longer mutated during scoring (a second
  pass over the same objects saw a stripped nested `check_number`). Unweighted
  mode scores an empty expected list against a non-empty actual list as 0.
- Removed the extract-side metrics that cannot fire on the published ground
  truth (every sidecar is v0.2 `_field_rules`): the legacy element-pass-rate /
  loc / attr / `extract_avg_iou*` family, bare `precision` / `recall` / `f1` /
  `iou` / `bbox_recall`, `record_*`, `null_hallucination_rate`,
  `extract_field_value_pass_rate`, `schema_field_accuracy_*`, extract-side
  `rule_pass_rate` / `rule_array_*`, and `association_f1`. `extract_unified_*`,
  `extract_evidence_*`, `accuracy`, `array_record_*` and
  `confidence_scoped_*` are unchanged.

### Added
- `FieldEvidence.layer` names the bbox geometry when the same location is
  annotated more than once (`word`, `structural`, `checkbox`, ...), so ground
  truth written for the internal harness loads unchanged. Unknown names are
  rejected at load time. The headline grounded F1 ORs every box regardless of
  layer; per-layer metrics are not emitted.
- `deepseek_v4_1_flash_extract_oneshot_structured_output_file` pipeline:
  DeepSeek-V4.1-Flash over page images with thinking disabled
  (`DEEPSEEK_API_KEY`).
- `test_rules` entries typed `layout_line` / `layout_word` load as layout rules
  with the matching granularity.
