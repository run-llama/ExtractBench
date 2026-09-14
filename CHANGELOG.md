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
- Evaluation failures count. A document whose evaluator worker crashes or
  times out becomes one zero-scored row for that pipeline, with
  `evaluation_worker_error` or `evaluation_timeout` in the metric metadata,
  instead of being dropped from the denominator. Failure rows are keyed by
  `(pipeline, test_id)`, so a retried inference that logged one `_errors.json`
  entry per attempt is penalized once, not once per attempt.
- The per-worker evaluation timeout now fires. It was written against
  `as_completed`, which only yields finished futures, so it could never elapse
  and one hung document held a run open indefinitely. The pool is drained with
  a progress-based stall window (8 minutes with nothing completing), stalled
  workers are terminated, and queued work is re-submitted to a fresh pool. The
  guard is enabled only when every selected ground-truth sidecar is at most
  1 MiB, so citation-heavy long documents are never cut off.

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
- `extract-bench dataset lint <dir>`: fail-closed gate over every extract
  `.test.json` in a tree. Ground truth must conform to its `data_schema`, field
  rules must live in one `_field_rules` container with resolvable paths, and row
  identity must sit beside the schema as `_eval_row_identity`. Exit 1 on any
  finding, exit 2 when the tree holds no extract sidecar.
- `extract-bench evaluation score_case <test.json> <prediction.json>`: score a
  single prediction against one test case and print the value-F1 metrics as
  JSON, for interactive harnesses that do not produce a results directory.
- `confidence_signal_coverage` metric: the share of emitted scalar leaves that
  carry a finite confidence score, with per-document and aggregate counts in the
  grounded-confidence payloads.
- Non-LlamaCloud providers receive the document under a random basename
  (`report-3f9a….pdf`) staged from the same bytes, so a dataset filename cannot
  leak what a document is to a hosted model or an agent's prompt. The saved
  request keeps the benchmark path. Recorded in `_metadata.json` as
  `randomize_external_filenames`; a pipeline opts out with
  `randomize_external_filename: false`.
- `Provider.recompute_cost(raw_output)`: a provider-agnostic re-pricing seam
  called by `inference renormalize`, so a corrected rate table re-prices saved
  runs without new inference. Implemented for the OpenAI Responses and Codex
  providers.
- Pricing rows for `gpt-5.5`, `gpt-5.6-sol` / `-terra` / `-luna`,
  `gemini-3.5-flash-lite` and `gemini-3.1-pro`. The Codex `gpt-5.6-sol` row is
  corrected from 5.00/0.50/30.00 to 4.00/0.40/20.00 per million tokens.
- Codex provider: `retry_empty_output` treats an all-null `output.json` as a
  transient failure and retries; `extra_instructions` appends pipeline-specific
  lines to the task prompt.
- Claude Code provider: `non_zdr: true` authenticates with
  `ANTHROPIC_NON_ZDR_API_KEY` for models unavailable under zero-data-retention.
- `glm_deepinfra_extract` provider and the
  `glm_5_3_flash_deepinfra_extract_oneshot_structured_output_file` pipeline:
  GLM-5.3-Flash served by DeepInfra over page images (`DEEPINFRA_API_KEY`). The
  z.ai pipeline is unchanged and remains the leaderboard row.

### Fixed
- Extend: a schema property named `id` (reserved by Extend) is aliased in the
  submitted schema and restored in the result, and an array whose items are an
  empty object is wrapped like a primitive array. Both shapes previously failed
  at processor creation.

### Fixed
- Dataset discovery keeps one input per stem and directory. A sibling `.png`
  saved beside `<stem>.pdf` used to collide on `test_id` and on the shared
  `.test.json`, so which twin was scored depended on iteration order.
- Per-document artifact directories (`<stem>.parse/`, `<stem>.pdf.images/`,
  `<stem>.v2.screenshots/`) no longer flip a flat dataset into grouped mode.
- Dataset names split into base and version at the last version-like segment,
  so `extract/short/v0.2` files under `extract/short` instead of `extract`.
- `evaluation run` pointed directly at a pipeline output root (one holding
  `_metadata.json`) treats it as that pipeline rather than scanning its document
  groups as if they were pipelines. `*.result.json` files inside
  `<document>.images/` artifact bundles are no longer scored as documents.
