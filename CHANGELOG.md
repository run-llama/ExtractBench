# Changelog

Notable changes to `llama-extract-bench`. This project follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Evaluation

- Layout diagnostics use ParseBench's attribution scorer, including rotated box overlap and zero for missing reading-order indices. Import helpers from `parse_bench.evaluation.metrics.attribution`. [#130]
- Field confidence grading excludes object and array citations and pairs identical rows one-to-one. Rescore cached runs with container confidence before comparing. [#128]
- Array-record metrics accept `_identity_hints` for large row-matching problems. [#77]
- Fixed citation field paths for LlamaExtract and Reducto. [#105]
- Preserve unknown citation pages as `null` and read page numbers and dimensions from nested citation metadata. [#107]

### Running benchmarks

- Use `--shuffle_seed` for reproducible submission order and `--max_concurrent_pages` to cap concurrent PDF pages. [#74] [#75]
- Built-in providers load on the first `create_provider` call. [#122]
- Download and run ExtractBench Pro through the CLI. [#81]
- Run ExtractBench and Pro as [Harbor tasks](docs/harbor.md) with the paper's coding-agent prompt. [#89]
- Pro renamed six challenge splits. Cached runs with the original split names can be reused. Harbor task IDs now use document IDs. [#101]

### Pipelines and dependencies

- LlamaExtract leaderboard pipelines use Extract 2.6. The Cost Effective and Agentic `_standard_bbox` variants were removed. [#86] [#87]
- LlamaExtract results retain API latency and state transitions. Agentic Plus Max Effort uses reported billing when available. [#76] [#114]
- Added Parseur AI Vision, Extend Operator, Gemini 4 Argon, and more Codex pipelines. The Extend Performance and Max Context pipelines now use Review Agent. [#85] [#90] [#98] [#104] [#124]
- Require ParseBench 1.0.7 or later for nullable citation pages. Layout average precision uses ParseBench, removing the direct scikit-learn dependency. [#107] [#116]

### Leaderboards

- Costs are computed by averaging each document's USD per page, with equal weighting given to all documents. Use recorded totals for full-run budgets. [#108]
- Pro reports scores with and without the 30-minute scoring cutoff. [#91]
- Leaderboard CSVs include pipeline and reasoning settings. Submissions set `include_in_rankings` explicitly. [#113] [#120]
- Published confidence AUGRC, acceptance curves, and threshold counts where available. [#126] [#127] [#128]

## [1.0.0] - 2026-10-01

### Packaging and APIs

- Install from PyPI as `llama-extract-bench`. Use the `extract-bench` CLI and `extract_bench` Python package.
- Added per-provider extras and `extract-bench version`. The `runners` extra installs all provider SDKs.
- Shared schemas and helpers with `parse-bench>=1.0.2` while keeping their ExtractBench import paths.
- `extract_bench.extensions` exposes registration hooks for providers, pipelines, products, output models, and layout adapters.
- `ExtractEvaluator.compute_metrics()` exposes metrics and normalized scoring inputs. Compatible extraction test cases can use their own model classes.
- `compute_unified_evidence_metrics(..., graded_cells=...)` exposes per-cell grounding verdicts.
- Added `ProviderTerminalCheckpointError`, `CloseOnceClient`, `format_field_path`, `field_citations_from_raw`, and `ToolResultWithImages` for provider integrations.

### Evaluation

- Unified Value F1 applies declared schema defaults to omitted fields, scores nested primitive arrays as single cells, and uses nested fields when matching rows.
- Grounding requires citations to cover the full coarse evidence envelope. Invalid predicted boxes count as misses.
- JSON subset scoring preserves ground truth across runs. In unweighted mode, a nonempty prediction against an empty expected list scores zero.
- Removed legacy extract metrics. Use unified Value F1 and grounded F1 for the published task.
- Evaluator crashes and timeouts count as one zero per pipeline and document. Stalled workers restart for datasets with sidecars up to 1 MiB.
- Added `confidence_signal_coverage` to report the share of emitted scalar fields with finite confidence scores.
- Added `FieldEvidence.layer` for evidence geometry and support for `layout_line` and `layout_word` rules.

### CLI and providers

- Added `dataset lint` to validate extraction test cases and `evaluation score_case` to score a single prediction.
- Dataset discovery deduplicates same-stem inputs, skips artifact directories, and recognizes versioned dataset paths. `evaluation run` accepts a pipeline output root directly.
- External providers receive randomized filenames by default. Set `randomize_external_filename: false` to opt out.
- `inference renormalize` can reprice saved OpenAI Responses and Codex results.
- Added DeepSeek V4.1 Flash and DeepInfra GLM-5.3 Flash pipelines and updated OpenAI and Gemini pricing.
- Codex can retry all-null output and accept pipeline-specific prompt instructions. Claude Code supports `non_zdr: true`.
- Fixed Extend schema submission for properties named `id` and arrays of empty objects.
- Removed the `tqdm` dependency.

[Unreleased]: https://github.com/run-llama/ExtractBench/compare/v1.0.0...main
[1.0.0]: https://github.com/run-llama/ExtractBench/tree/v1.0.0
[#74]: https://github.com/run-llama/ExtractBench/pull/74
[#75]: https://github.com/run-llama/ExtractBench/pull/75
[#76]: https://github.com/run-llama/ExtractBench/pull/76
[#77]: https://github.com/run-llama/ExtractBench/pull/77
[#81]: https://github.com/run-llama/ExtractBench/pull/81
[#85]: https://github.com/run-llama/ExtractBench/pull/85
[#86]: https://github.com/run-llama/ExtractBench/pull/86
[#87]: https://github.com/run-llama/ExtractBench/pull/87
[#89]: https://github.com/run-llama/ExtractBench/pull/89
[#90]: https://github.com/run-llama/ExtractBench/pull/90
[#91]: https://github.com/run-llama/ExtractBench/pull/91
[#98]: https://github.com/run-llama/ExtractBench/pull/98
[#101]: https://github.com/run-llama/ExtractBench/pull/101
[#104]: https://github.com/run-llama/ExtractBench/pull/104
[#105]: https://github.com/run-llama/ExtractBench/pull/105
[#107]: https://github.com/run-llama/ExtractBench/pull/107
[#108]: https://github.com/run-llama/ExtractBench/pull/108
[#113]: https://github.com/run-llama/ExtractBench/pull/113
[#114]: https://github.com/run-llama/ExtractBench/pull/114
[#116]: https://github.com/run-llama/ExtractBench/pull/116
[#120]: https://github.com/run-llama/ExtractBench/pull/120
[#122]: https://github.com/run-llama/ExtractBench/pull/122
[#124]: https://github.com/run-llama/ExtractBench/pull/124
[#126]: https://github.com/run-llama/ExtractBench/pull/126
[#127]: https://github.com/run-llama/ExtractBench/pull/127
[#128]: https://github.com/run-llama/ExtractBench/pull/128
[#130]: https://github.com/run-llama/ExtractBench/pull/130
