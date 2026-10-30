# ExtractBench

**ExtractBench** &nbsp;
[![Website](https://img.shields.io/badge/Website-extractbench.ai-blue)](https://www.extractbench.ai/)
[![arXiv](https://img.shields.io/badge/arXiv-2607.29677-b31b1b.svg)](https://arxiv.org/abs/2607.29677)
[![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/llamaindex/ExtractBench)
[![Papers with Code:ExtractBench](https://paperswithcode.co/api/v1/papers/2607.29677/leaderboard-badge.svg?eval=35030&live=1)](https://paperswithcode.co/api/v1/papers/2607.29677/leaderboard-badge-link?eval=35030)
<br>
**ExtractBench Pro** &nbsp;
[![Website](https://img.shields.io/badge/Website-extractbench.ai%2Fpro-blue)](https://www.extractbench.ai/pro/)
[![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/llamaindex/ExtractBenchPro)

**ExtractBench** is a benchmark for **schema-guided extraction** from enterprise documents. Given a document and a user-defined JSON Schema, a system must return schema-valid JSON with the correct values, *every* record of each repeated structure, missing fields marked `null` rather than invented, and source evidence for each value.

A schema defines one extraction task, and all documents of that type share it: one invoice schema covers invoices from every vendor, however different each one looks. Enterprises write a new schema for almost every workflow, so a system cannot be tuned to a fixed template. It has to handle schemas and documents it has never seen. And because agents increasingly act on extracted values before anyone reviews them, one truncated schedule or one invented value becomes a wrong payment or a wrong decision. The benchmark therefore scores completeness and traceability.

This repository hosts two benchmarks. They share the task, the evaluator, the metrics, and the harness; they differ in their documents and in what makes a value hard to extract.

| Benchmark | What makes it hard | Size | Documents | Dataset | Details |
|---|---|---|---|---|---|
| **ExtractBench** | Long lists, needles in long documents, dense forms, scans, and handwriting | 370 documents, 4,869 pages, 67 document types | Real public records, plus synthetic long lists rendered from real layouts | [llamaindex/ExtractBench](https://huggingface.co/datasets/llamaindex/ExtractBench) | [docs/extractbench.md](docs/extractbench.md) · [launch post](https://www.llamaindex.ai/blog/introducing-extractbench) |
| **ExtractBench Pro** | Values read off charts and drawings, counted under house rules, or decoded with house vocabulary | 100 documents, 2,622 pages, 86 document types | Synthetic, derived from real long-tail documents | [llamaindex/ExtractBenchPro](https://huggingface.co/datasets/llamaindex/ExtractBenchPro) | [docs/extractbench-pro.md](docs/extractbench-pro.md) · [launch post](https://www.llamaindex.ai/blog/introducing-extractbench-pro) |

<p align="center">
  <img src="docs/extractbench_teaser.png" alt="ExtractBench: JSON Schema and document in; JSON plus evidence out; scored on value F1, grounding F1, challenge tags, and cost" width="100%">
</p>

## Leaderboards

[ExtractBench](#extractbench-leaderboard) · [ExtractBench Pro](#extractbench-pro-leaderboard) · [Submit results](#submitting-results)

Models and prices reflect each provider's official documentation; each system uses its recommended configuration.

Scores are percentages; costs are USD. The best score in each metric is **bold**; the second-best distinct score is <u>underlined</u>.

Report costs using each provider's official public prices intended to remain in effect indefinitely, excluding discounts, free credits, and other promotions. Permanent changes to those prices change reported costs; temporary offers do not.

<details>
<summary><strong>Metric details</strong></summary>

- **Value F1:** F1 of extracted values, averaged across documents. Each document has equal weight, regardless of its length.
- **Grounding F1:** requires both a correct value and correct evidence: the predicted box must overlap an accepted evidence box at IoU 0.5 or higher (word level). Grounding is evaluated on the subset with verified ground-truth evidence for the corresponding metric.
- **% Perfect:** percentage of documents with 100% Value F1.

In the official benchmark, documents that fail or exceed a 30-minute limit score zero. However, due to the additional thinking required to handle Pro documents, we also maintain an unofficial leaderboard where completed outputs count regardless of elapsed time.

</details>

### ExtractBench leaderboard

<!-- LEADERBOARD:START -->
**Unified value F1** — rows are ordered by Overall Value F1, then lower cost.

| Rank | Provider | Category | Overall | Short | Medium | Long | ¢ / Page |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | LlamaExtract Agentic Plus | LlamaExtract | **96.17** | <u>97.02</u> | 94.30 | <u>94.65</u> | 7.50¢ |
| 2 | Reducto Deep Extract | Specialized APIs | <u>96.08</u> | 96.84 | **94.80** | 92.85 | 6.35¢ |
| 3 | Pulse (Effort) | Specialized APIs | 96.07 | 96.83 | 94.70 | 93.24 | 10.50¢ |
| 4 | LlamaExtract Agentic | LlamaExtract | 95.65 | 96.06 | <u>94.79</u> | **94.73** | 3.12¢ |
| 5 | Codex (GPT-6.1 Sol Evidence) | Coding Agents | 95.31 | 97.01 | 92.96 | 85.46 | 7.13¢ |
| 6 | Claude Code (Sonnet 5.5 Evidence) | Coding Agents | 94.68 | 96.22 | 92.97 | 83.69 | 5.69¢ |
| 7 | Codex (GPT-6 Sol Evidence) | Coding Agents | 94.57 | 95.99 | 92.42 | 87.25 | 9.80¢ |
| 8 | Pulse (Non-effort) | Specialized APIs | 94.51 | 95.39 | 92.49 | 93.31 | 3.00¢ |
| 9 | Codex (GPT-6.1 Terra Evidence) | Coding Agents | 94.28 | 96.26 | 91.86 | 81.21 | 4.68¢ |
| 10 | LlamaExtract Cost-Effective | LlamaExtract | 94.21 | 95.49 | 91.58 | 90.86 | 1.00¢ |

Top 10 of 52 systems — full table in [leaderboard.csv](leaderboard.csv).
<!-- LEADERBOARD:END -->

<!-- GROUNDING:START -->
**Grounding F1** — rows are ordered by overall word-level grounding F1.

<table>
  <thead>
    <tr><th rowspan="2">Rank</th><th rowspan="2">Provider</th><th colspan="4">Word-level grounding F1</th><th colspan="4">Page-level grounding F1</th></tr>
    <tr><th align="right">Overall</th><th align="right">Short</th><th align="right">Medium</th><th align="right">Long</th><th align="right">Overall</th><th align="right">Short</th><th align="right">Medium</th><th align="right">Long</th></tr>
  </thead>
  <tbody>
    <tr><td align="right">1</td><td>Codex (GPT-6.1 Sol Evidence)</td><td align="right"><strong>87.61</strong></td><td align="right"><strong>89.03</strong></td><td align="right"><u>84.38</u></td><td align="right"><strong>85.70</strong></td><td align="right"><strong>89.37</strong></td><td align="right"><strong>89.63</strong></td><td align="right"><u>89.25</u></td><td align="right">86.80</td></tr>
    <tr><td align="right">2</td><td>Codex (GPT-6.1 Terra Evidence)</td><td align="right"><u>85.69</u></td><td align="right"><u>88.13</u></td><td align="right">81.32</td><td align="right">77.21</td><td align="right">87.42</td><td align="right"><u>88.49</u></td><td align="right">86.10</td><td align="right">81.15</td></tr>
    <tr><td align="right">3</td><td>LlamaExtract Agentic Plus</td><td align="right">83.25</td><td align="right">83.22</td><td align="right">83.44</td><td align="right"><u>82.84</u></td><td align="right"><u>88.26</u></td><td align="right">87.97</td><td align="right">88.48</td><td align="right"><u>90.61</u></td></tr>
    <tr><td align="right">4</td><td>LlamaExtract Agentic</td><td align="right">82.33</td><td align="right">81.39</td><td align="right"><strong>84.94</strong></td><td align="right">81.63</td><td align="right">88.20</td><td align="right">86.95</td><td align="right"><strong>90.33</strong></td><td align="right"><strong>92.59</strong></td></tr>
    <tr><td align="right">5</td><td>LlamaExtract Cost-Effective</td><td align="right">79.80</td><td align="right">80.99</td><td align="right">77.52</td><td align="right">76.16</td><td align="right">86.28</td><td align="right">86.97</td><td align="right">84.60</td><td align="right">86.41</td></tr>
    <tr><td align="right">6</td><td>Codex (GPT-6 Sol Evidence)</td><td align="right">77.11</td><td align="right">77.61</td><td align="right">75.53</td><td align="right">78.31</td><td align="right">86.50</td><td align="right">87.20</td><td align="right">85.12</td><td align="right">84.82</td></tr>
    <tr><td align="right">7</td><td>Claude Code (Opus 5.5 Evidence)</td><td align="right">71.55</td><td align="right">73.57</td><td align="right">65.54</td><td align="right">74.87</td><td align="right">76.95</td><td align="right">79.27</td><td align="right">71.36</td><td align="right">76.47</td></tr>
    <tr><td align="right">8</td><td>Codex (GPT-6 Luna Evidence)</td><td align="right">65.70</td><td align="right">62.91</td><td align="right">71.11</td><td align="right">74.36</td><td align="right">83.93</td><td align="right">84.41</td><td align="right">83.47</td><td align="right">80.17</td></tr>
    <tr><td align="right">9</td><td>Claude Code (Sonnet 5.5 Evidence)</td><td align="right">64.61</td><td align="right">64.59</td><td align="right">63.18</td><td align="right">71.22</td><td align="right">75.99</td><td align="right">77.93</td><td align="right">71.50</td><td align="right">74.98</td></tr>
    <tr><td align="right">10</td><td>Reducto Extract (v4)</td><td align="right">54.97</td><td align="right">60.53</td><td align="right">49.35</td><td align="right">16.30</td><td align="right">75.28</td><td align="right">87.47</td><td align="right">57.38</td><td align="right">18.59</td></tr>
  </tbody>
</table>

Top 10 of 52 systems — full table in [leaderboard.csv](leaderboard.csv).
<!-- GROUNDING:END -->

### ExtractBench Pro leaderboard

<!-- PRO_LEADERBOARD:START -->
Rows are ordered by % Perfect, then Value F1, then lower cost.

| Rank | Provider | Category | % Perfect | Value F1 | Grounding F1 | ¢ / Page |
|---:|---|---|---:|---:|---:|---:|
| 1 | LlamaExtract Agentic Plus (Max Effort) | LlamaExtract | **61** | <u>94.32</u> | 57.02 | 12.71¢ |
| 2 | Codex GPT-6 Astra | Coding Agents | <u>58</u> | **94.53** | **70.97** | 95.20¢ |
| 3 | Claude Code Opus 5.5 | Coding Agents | 51 | 74.87 | 45.29 | 41.20¢ |
| 4 | LlamaExtract Agentic Plus | LlamaExtract | 38 | 77.70 | 60.39 | 7.50¢ |
| 5 | Extend Operator | Specialized APIs | 34 | 73.87 | 57.75 | 9.88¢ |
| 6 | Claude Code Sonnet 5.5 | Coding Agents | 32 | 73.62 | 40.47 | 19.98¢ |
| 7 | Gemini CLI (Gemini 4 Argon Medium) | Coding Agents | 30 | 42.98 | 33.90 | 36.16¢ |
| 8 | Codex GPT-6.1 Terra Medium | Coding Agents | 27 | 74.19 | 60.16 | 4.06¢ |
| 9 | LlamaExtract Agentic | LlamaExtract | 20 | 71.32 | 58.01 | 3.12¢ |
| 10 | Pulse Effort | Specialized APIs | 18 | 70.53 | 18.74 | 10.50¢ |

Top 10 of 16 systems — full table in [leaderboard-pro.csv](leaderboard-pro.csv). Scores with no time limit are also available there.
<!-- PRO_LEADERBOARD:END -->

### Submitting results

Open a PR updating [leaderboard.csv](leaderboard.csv) or [leaderboard-pro.csv](leaderboard-pro.csv), with your pipeline configuration and evaluation reports. Scores are percentages; costs are USD per page. Regenerate the tables with `uv run python scripts/update_readme.py`.

The rows each CSV needs are described in [ExtractBench: submitting results](docs/extractbench.md#submitting-results) and [ExtractBench Pro: submitting results](docs/extractbench-pro.md#submitting-results). For Pro, include the overall (`all`) row and all six challenge rows, with scores for both time-limit views.

<details>
<summary><strong>Inclusion criteria</strong></summary>

1. The model or API needs to be publicly accessible, either via open weights or a self-serve API that any user can sign up for.
2. The benchmark run needs to finish within a reasonable time (roughly single-digit hours).
3. We can adjust concurrency based on the provider's recommended settings, but providers should not require custom framework changes, so the evaluation stays fair across models.

</details>

## Quick Start

**Prerequisites:** Create a `.env` file with the API key for the extraction system you want to evaluate (see [Configuration](#configuration)).

```bash
# Install from PyPI (pick the extras for the systems you want to run)
pip install "llama-extract-bench[runners]"    # every provider SDK
pip install "llama-extract-bench[llamaextract]" # or just one, e.g. llamaextract / openai / anthropic / google

# Or, from a checkout of this repo
uv sync --extra runners

# Quick test run on ExtractBench (6 documents — good for trying things out)
# (drop the `uv run` prefix if you installed from PyPI)
uv run extract-bench run llamaextract_agentic --test

# Full ExtractBench run (replace with any pipeline name, see "Pipelines" below)
uv run extract-bench run llamaextract_agentic

# Full ExtractBench Pro run
uv run extract-bench run llamaextract_agentic --dataset pro

# View interactive reports in your browser
uv run extract-bench serve llamaextract_agentic
uv run extract-bench serve output-pro/llamaextract_agentic
```

> [!WARNING]
> A full run calls a metered API. ExtractBench (370 documents / 4,869 pages) costs roughly
> **$10 to $1,677**, and ExtractBench Pro (100 documents / 2,622 pages) roughly **$26 to $2,496**,
> depending on which system you evaluate. Start with `--test` on ExtractBench, which runs 6 documents
> for cents on any hosted system. Pro has no `--test` subset; run one challenge with `--group` instead.
> Per-system estimates: [ExtractBench](docs/extractbench.md#cost-of-a-full-run), [ExtractBench Pro](docs/extractbench-pro.md#cost-of-a-full-run).

## Datasets

Both datasets are hosted on Hugging Face and download automatically the first time you run a pipeline. Pass `--dataset pro` to any `run`, `download`, or `status` command to use ExtractBench Pro.

| | ExtractBench | ExtractBench Pro |
|---|---|---|
| Hugging Face | [`llamaindex/ExtractBench`](https://huggingface.co/datasets/llamaindex/ExtractBench) | [`llamaindex/ExtractBenchPro`](https://huggingface.co/datasets/llamaindex/ExtractBenchPro) |
| `--dataset` | `extractbench` (default) | `pro` |
| Local data | `data/` | `data-pro/` |
| Results | `output/` | `output-pro/` |
| `--group` values | `short`, `medium`, `long` | `document_conventions`, `house_vocabulary`, `house_counting_rules`, `drawn_relations`, `charts_inside_documents`, `long_chart_sheets` |
| `--test` subset | 6 documents | none |
| Splits, tags, sources, ground truth | [docs/extractbench.md](docs/extractbench.md) | [docs/extractbench-pro.md](docs/extractbench-pro.md) |

To manage the data manually:

```bash
# ExtractBench
uv run extract-bench download            # full dataset
uv run extract-bench download --test     # 6-document test dataset
uv run extract-bench status              # check readiness and show summary statistics

# ExtractBench Pro
uv run extract-bench download --dataset pro
uv run extract-bench status --dataset pro
```

Each download writes a `<stem>.pdf` and `<stem>.test.json` per document and records the Hugging Face revision in a manifest. Re-running `download` resumes an incomplete download or refreshes a changed dataset.

## Pipelines

A **pipeline** is an extraction system or configuration you want to evaluate. Run `uv run extract-bench pipelines` for the live list, or see [docs/pipelines.md](docs/pipelines.md). The pipeline names behind each leaderboard row are listed in [ExtractBench: leaderboard pipelines](docs/extractbench.md#leaderboard-pipelines) and [ExtractBench Pro: leaderboard pipelines](docs/extractbench-pro.md#leaderboard-pipelines).

> [!NOTE]
> Coding-agent pipelines execute local shell commands against benchmark documents. Run them in a container or VM.

The parse and layout-detection rosters inherited from [ParseBench](https://github.com/run-llama/ParseBench) are still registered and runnable by name; list them with `extract-bench pipelines --parse`, `--layout`, or `--all`. They are hidden from the default listing because this benchmark scores extraction; the two-stage extract pipelines use them internally as their parse stage.

## Metrics

Both benchmarks use the same deterministic evaluator: one metric for value accuracy and two for grounding. We score the grounding metrics only on fields that carry verified box ground truth.

- **Unified value F1** — whether the extracted *values* match the expected output, under one definition for scalar fields and arrays of records. Each output is flattened into cells, one per scalar field and per aligned record subfield, and a cell is correct when it matches its expected counterpart after normalization. Precision, recall, and F1 are computed over these cells per document; slices report unweighted document means.
- **Word-level grounding F1** — a field is grounded-correct only when its value is accepted *and* its predicted box overlaps any accepted box on its evidence list, at IoU 0.5.
- **Page-level grounding F1** — the same rule against the cited source page instead of the box: a field counts only when its value is accepted and the page it cites is correct.

<details>
<summary><strong>How the unified value F1 is computed</strong></summary>

- **Array alignment.** A repeated structure compares as an unordered set of records: records pair by a globally optimal one-to-one assignment (the Hungarian algorithm) that minimizes mismatched cells. Unmatched expected records lower recall, surplus predictions lower precision.
- **Normalization.** Matching is deterministic and mostly exact: dates in eight written formats canonicalize to ISO form, strings compare exactly after whitespace collapsing, and everything else uses plain equality, with no numeric tolerance and no LLM judge.
- **Missing values.** An omitted key scores as an explicit `null`, every scalar field enters both denominators, and a correct `null` on a blank field is credited. Only repeated records move precision and recall apart, so a gap between them means records were dropped or invented.

</details>

Failed and missing documents score zero rather than being dropped, so a pipeline cannot raise its average by erroring out on the documents it finds hardest.

## Usage

### Running the Benchmark

The `run` command runs inference, evaluates against ground truth, and generates reports. Add `--dataset pro` to run ExtractBench Pro:

```bash
# Evaluate an extraction system on the whole benchmark
uv run extract-bench run <pipeline_name>
uv run extract-bench run <pipeline_name> --dataset pro

# Evaluate a single split (ExtractBench) or challenge (Pro) only
uv run extract-bench run <pipeline_name> --group short
uv run extract-bench run <pipeline_name> --dataset pro --group house_vocabulary

# Skip calling the extraction system — just re-evaluate existing results
uv run extract-bench run <pipeline_name> --skip_inference

# Control how many documents are processed in parallel
uv run extract-bench run <pipeline_name> --max_concurrent 10

# Run on the small ExtractBench test dataset only
uv run extract-bench run <pipeline_name> --test
```

For a reproducible submission order (`--shuffle_seed`), a cap on concurrent source pages (`--max_concurrent_pages`), and row identity hints in annotations, see [docs/advanced-usage.md](docs/advanced-usage.md).

### Viewing & Comparing Results

Pipeline names resolve to `output/<name>`; pass `output-pro/<name>` paths for Pro results.

```bash
# View reports in your browser (needed because browsers block PDF rendering from file:// URLs)
uv run extract-bench serve <pipeline_name>
uv run extract-bench serve output-pro/<pipeline_name>

# Compare two extraction systems side-by-side
uv run extract-bench compare <pipeline_a> <pipeline_b>
uv run extract-bench compare output-pro/<pipeline_a> output-pro/<pipeline_b>

# Generate a leaderboard across all evaluated systems
uv run extract-bench leaderboard
uv run extract-bench leaderboard --output_dir output-pro

# Leaderboard for specific systems only
uv run extract-bench leaderboard llamaextract_agentic llamaextract_cost_effective
```

<details>
<summary><strong>Advanced Subcommands</strong></summary>

For fine-grained control over individual steps:

```bash
# Run inference only (call the extraction system, don't evaluate)
uv run extract-bench inference run <pipeline_name> data/short --output_dir output

# Run evaluation only (on existing inference results)
uv run extract-bench evaluation run output/<pipeline_name> --test_cases_dir data

# Generate detailed HTML report from evaluation results
uv run extract-bench analysis generate_report --evaluation_dir ./output/<pipeline_name>
```

</details>

<details>
<summary><strong>Evaluating Your Own System</strong></summary>

To add a new extraction system, use [Claude Code](https://claude.ai/code):

```bash
/integrate-pipeline <name> <API docs or SDK link>
```

This creates the provider, registers the pipeline, and updates docs. The skill definition lives in [`.claude/commands/integrate-pipeline.md`](.claude/commands/integrate-pipeline.md) and can be adapted for other AI coding agents.

</details>

### Harbor

[ExtractBench](https://huggingface.co/datasets/llamaindex/ExtractBench) and
[ExtractBench Pro](https://huggingface.co/datasets/llamaindex/ExtractBenchPro) are
available as Harbor tasks. With Harbor 0.24.0+, Docker, Git LFS, and your model's
API credentials configured, these commands download the tasks and run an agent:

```bash
git lfs install
harbor run --repo https://huggingface.co/datasets/llamaindex/ExtractBench -p tasks \
  -a codex -m <model_name> --allow-agent-host <model_api_host>
harbor run --repo https://huggingface.co/datasets/llamaindex/ExtractBenchPro -p tasks \
  -a codex -m <model_name> --allow-agent-host <model_api_host>
```

Replace `<model_name>` and `<model_api_host>` with your model and its API hostname. Add
`-l 1 -n 1` to try a single task. See [Harbor usage](docs/harbor.md) for downloading
without running, local runs, and scoring details.

## Configuration

### API Keys

Each pipeline calls a specific system's API. You only need the key for the system you want to evaluate. Add it to a `.env` file at the project root (see [.env.example](.env.example) for the full list):

```bash
# Only add the keys you need. For example, to evaluate LlamaExtract:
LLAMA_CLOUD_API_KEY=...

# To evaluate OpenAI-based pipelines:
OPENAI_API_KEY=...

# To evaluate Anthropic-based pipelines (including claude_code_extract_*):
ANTHROPIC_API_KEY=...

# To evaluate Google-based pipelines:
GOOGLE_API_KEY=...

# Codex coding-agent pipelines authenticate the codex CLI:
CODEX_API_KEY=...
```

ExtractBench does not use LLM-as-a-judge; all value scoring is deterministic. Your keys only ever call the extraction system you are evaluating.

### CLI Reference

| Command | Description |
|---------|-------------|
| `extract-bench run` | Evaluate an extraction system end-to-end (inference + evaluation + reports) |
| `extract-bench download` | Download original ExtractBench; add `--dataset pro` for Pro |
| `extract-bench status` | Check dataset readiness (`--dataset pro` for Pro) |
| `extract-bench pipelines` | List extraction pipelines (`--parse`, `--layout`, `--all` for the rest) |
| `extract-bench compare` | Compare results from two systems side-by-side |
| `extract-bench leaderboard` | Generate a leaderboard across all evaluated systems |
| `extract-bench serve` | View HTML reports in your browser (with PDF rendering support) |

Advanced subcommands: `inference`, `evaluation`, `analysis`, `pipeline`, `data`

<details>
<summary><strong>Output Structure</strong></summary>

ExtractBench results go to `output/`, with one directory per split. Pro results go to `output-pro/`, with one directory per challenge.

```
output/                                      # output-pro/ for ExtractBench Pro
├── _leaderboard.html                        # Cross-pipeline leaderboard
└── <pipeline_name>/
    ├── short/                               # one directory per split or challenge
    │   ├── *.result.json                    # Inference results
    │   ├── _evaluation_report.json          # Evaluation summary
    │   ├── _evaluation_report_detailed.html # Interactive detailed report
    │   ├── _evaluation_results.csv          # Per-example CSV
    │   └── _evaluation_report.md            # Markdown summary
    ├── medium/  (same structure)
    ├── long/    (same structure)
    ├── _errors.json                         # Per-document inference failures
    └── _metadata.json                       # Run metadata
```

</details>

<details>
<summary><strong>Project Structure</strong></summary>

```
src/extract_bench/
├── cli.py                           # Fire CLI entry point
├── pipeline/cli.py                  # End-to-end pipeline orchestration
├── data/
│   ├── download.py                  # HuggingFace dataset download
│   └── cli.py                       # Data management CLI
├── inference/
│   ├── runner.py                    # Batch inference with concurrency
│   ├── pipelines/                   # Pipeline registry (extract, parse, layout)
│   └── providers/                   # Provider implementations per product type
├── evaluation/
│   ├── runner.py                    # Parallel evaluation + failure penalties
│   ├── evaluators/                  # Product-specific evaluators
│   ├── metrics/extract/             # Unified value F1, grounding, record matching
│   └── reports/                     # CSV, HTML, markdown export
├── analysis/
│   ├── detailed_report.py           # Interactive per-split HTML report
│   └── comparison.py                # Pipeline comparison
├── test_cases/
│   ├── loader.py                    # Load test cases (JSONL or sidecar .test.json)
│   └── schema.py                    # TestCase types (Extract, Parse, LayoutDetection)
└── schemas/
    ├── pipeline_io.py               # InferenceRequest, InferenceResult
    ├── evaluation.py                # EvaluationResult, EvaluationSummary
    └── product.py                   # ProductType enum
```

</details>

## Citation

```bibtex
@misc{zhang2026extractbenchbenchmarkschemaguidedenterprise,
  title={ExtractBench: A Benchmark for Schema-Guided Enterprise Document Extraction},
  author={Boyang Zhang and Adrian Lyjak and Eli Stewart and Zhaoqi Li and Simon Suo},
  year={2026},
  eprint={2607.29677},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2607.29677},
}
```

## Links

- **ExtractBench dataset**: [llamaindex/ExtractBench](https://huggingface.co/datasets/llamaindex/ExtractBench) · [details](docs/extractbench.md) · [launch post](https://www.llamaindex.ai/blog/introducing-extractbench)
- **ExtractBench Pro dataset**: [llamaindex/ExtractBenchPro](https://huggingface.co/datasets/llamaindex/ExtractBenchPro) · [details](docs/extractbench-pro.md) · [launch post](https://www.llamaindex.ai/blog/introducing-extractbench-pro) · [website](https://www.extractbench.ai/pro/)
- **Code**: [run-llama/ExtractBench](https://github.com/run-llama/ExtractBench)
- **ParseBench**: [run-llama/ParseBench](https://github.com/run-llama/ParseBench) — A Document Parsing Benchmark for AI Agents

## License

The code and both datasets (ExtractBench and ExtractBench Pro) are released under the [Apache 2.0 License](LICENSE).
