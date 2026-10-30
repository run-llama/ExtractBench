# ExtractBench Pro

Details specific to ExtractBench Pro. Installation, download, usage, metrics and configuration shared with ExtractBench are in the [README](../README.md).

- **Dataset:** [`llamaindex/ExtractBenchPro`](https://huggingface.co/datasets/llamaindex/ExtractBenchPro)
- **Website:** [extractbench.ai/pro](https://www.extractbench.ai/pro/)
- **Launch post:** [Introducing ExtractBench Pro](https://www.llamaindex.ai/blog/introducing-extractbench-pro)
- **Leaderboard:** [README](../README.md#extractbench-pro-leaderboard), full table with per-challenge rows in [leaderboard-pro.csv](../leaderboard-pro.csv)

## Overview

ExtractBench Pro is the hard companion to ExtractBench. The task is the same: given a full document and a user-written schema, return a schema-valid JSON object with the correct, exhaustive values and evidence for each one. The difference is what makes a value hard. In Pro, many values cannot be copied from the page. They have to be read off a chart or a drawing, counted under the document's own rules, or decoded with its own vocabulary.

Pro contains **100 documents (2,622 pages, 1–83 pages each)** across 86 document types, with 144,141 expected values. The documents are derived from real long-tail documents and grouped into six challenges.

## Challenges

Each document belongs to exactly one challenge: the one a naive reader most often gets wrong in it. A document can also contain values of other challenges, and values that are read directly from the page.

| Challenge (`--group`) | What it tests | Documents | Pages | Expected values |
|---|---|---:|---:|---:|
| `document_conventions` | One part of the document, such as a legend, rule or amendment, changes how the rest must be read | 22 | 875 | 121,287 |
| `house_vocabulary` | The organization's own definitions of its jargon decide the value | 14 | 606 | 15,242 |
| `house_counting_rules` | Counts and sums computed under definitions from the schema and the document | 12 | 225 | 409 |
| `drawn_relations` | All or part of the value depends on a line in a diagram rather than a printed value | 20 | 352 | 2,349 |
| `charts_inside_documents` | Values read off charts, as the range between the tick marks the value falls in | 25 | 557 | 1,347 |
| `long_chart_sheets` | Hundreds of values on one extremely long, narrow chart sheet, such as a well log or a time-location chart | 7 | 7 | 3,507 |
| **Total** | | **100** | **2,622** | **144,141** |

The document types in each challenge are listed on the [dataset card](https://huggingface.co/datasets/llamaindex/ExtractBenchPro#challenge-subsets).

## Capture, length, and domain

Each document is also tagged by how most of its pages were captured, by length, and by business domain. These axes are independent of the challenge.

| Perception | Documents | What most pages look like |
|---|---:|---|
| `digital` | 25 | Born-digital pages with a text layer |
| `image` | 8 | Clean page images without scanner noise |
| `scan` | 44 | Light or medium scan |
| `heavy_scan` | 16 | Heavy or extreme scan |
| `handwritten` | 7 | Hand-filled forms or handwritten values |

- **Length:** `short` (10 pages or fewer) 8 documents, `medium` (11–50 pages) 81 documents, `long` (more than 50 pages) 11 documents.
- **Domain:** D1 finance (27), D2 energy (19), D3 government (16), D5 supply chain (9), D6 healthcare (13), D7 legal (3), D8 real estate (13), using the [ExtractBench domain codes](extractbench.md#tag-axes).

Seven of the eight short documents are single plotter sheets 13 to 17 feet long (well logs, a borehole core log, a track chart, and time-location charts), scanned or photocopied, with no text layer; each asks for 390 to 715 values. Charts and drawings in other documents are often page images too, so their values cannot be read from a text layer. Some scanned documents carry a text layer that does not match the printed page; the printed page is authoritative.

## Sources and ground truth

Every document is synthetic: its values are fictional and fixed before the page is rendered, so the expected output is exact by construction. Each value's scoring rule lists its accepted readings with their source locations (`page`, `bbox`, `quote`, `value`), and any accepted reading counts as correct.

Grounding F1 (box level) and page-level grounding F1 are computed on the 81 documents with verified ground-truth evidence. Value F1 and % Perfect use all 100 documents.

Both leaderboard views use the same runs and document denominators. The **30-minute limit** view gives zero to documents taking longer than 1,800 seconds. The **no time limit** view scores all completed outputs regardless of elapsed time. Failed extractions, including attempts that timed out without a usable output, score zero in both views. Removing the scoring cutoff does not recover an output from a terminated attempt. Costs are shared between the two views.

## Local layout

`extract-bench download --dataset pro` writes one directory per challenge under `data-pro/`, with a `<stem>.pdf` and `<stem>.test.json` for each document. Results go to `output-pro/<pipeline_name>/<challenge>/`. Both are kept separate from the original ExtractBench `data/` and `output/`.

Pro has no `--test` subset. To try a pipeline on a smaller slice, run a single challenge with `--group`; `long_chart_sheets` is the smallest by page count (7 documents, 7 pages).

## Leaderboard pipelines

| Pipeline name | Leaderboard name |
|---------------|------------------|
| `codex_code_extract_gpt_6_astra_xhigh_evidence` | Codex GPT-6 Astra |
| `claude_code_extract_opus_5_5_evidence` | Claude Code Opus 5.5 |
| `llamaextract_agentic_plus` | LlamaExtract Agentic Plus |
| `extend_extract_operator` | Extend Operator |
| `claude_code_extract_sonnet_5_5_evidence` | Claude Code Sonnet 5.5 |
| `gemini_cli_extract_gemini_4_argon_medium_thinking_vision_evidence` | Gemini CLI (Gemini 4 Argon Medium) |
| `llamaextract_agentic` | LlamaExtract Agentic |
| `pulse_schema_effort` | Pulse Effort |
| `codex_code_extract_gpt_6_1_sol_low_evidence` | Codex GPT-6.1 Sol |
| `reducto_deep_extract` | Reducto Deep |
| `datalab_parse_accurate_extract_accurate` | Datalab Accurate |
| `llamaextract_cost_effective` | LlamaExtract Cost Effective |
| `codex_code_extract_gpt_6_luna_medium_evidence_prompt_opt` | Codex GPT-6 Luna |

See [pipelines.md](pipelines.md) for registered configurations.

The **LlamaExtract Agentic Plus (Max Effort)** leaderboard entry uses `tier="agentic_plus"`, `effort="max"`, `version="latest"`, citations and confidence scores enabled, and no custom Parse configuration. It was evaluated on the development API on October 30, 2026; the resolved release version was not recorded, and production parity was not measured. The [aggregate run evidence](agentic-plus-max-effort-results.json) records all six source runs, their configurations, and unrounded scores.

All 100 documents completed within 30 minutes, so the official and no-time-limit scores agree. Value F1 and % Perfect weight all 100 documents equally; grounding metrics weight the 81 documents with evidence equally. Cost is total billed credits at $1.25 per 1,000 credits, including Parse, divided by the 2,622 source pages: $333.33 / 2,622 = $0.127128 per page. Max effort uses [model-usage billing with an Agentic Plus page-charge minimum](https://developers.llamaindex.ai/llamaparse/general/pricing/), so its cost varies by document.

## Cost of a full run

Costs use each provider's official listed price.

| Category | One full run | Examples |
|---|---:|---|
| LlamaExtract | $26 – $333 | Cost Effective ~$26, Agentic ~$82, Agentic Plus ~$197, Agentic Plus (Max Effort) ~$333 |
| Specialized APIs | $162 – $413 | Reducto Deep ~$162, Extend Operator ~$259, Pulse Effort ~$275, Datalab Accurate ~$413 |
| Coding agents | $30 – $2,496 | Codex GPT-6 Luna ~$30, Codex GPT-6.1 Sol ~$363, Claude Code Sonnet 5.5 ~$524, Claude Code Opus 5.5 ~$1,080, Codex GPT-6 Astra ~$2,496 |

These figures apply the reported mean cost per page to 2,622 pages. Per-system prices are in [leaderboard-pro.csv](../leaderboard-pro.csv) (`Cost_Per_Page`, in dollars per page).

## Submitting results

Add seven rows per system to [leaderboard-pro.csv](../leaderboard-pro.csv): the overall row (`Challenge` = `all`) and one row for each of the six challenges. Each row carries `Documents`, `Grounding_Documents`, `Cost_Per_Page`, and both score views:

| Metric | 30-minute limit | No time limit |
|---|---|---|
| % Perfect | `Perfect_Documents` | `Perfect_Documents_No_Time_Limit` |
| Value F1 | `Value_F1` | `Value_F1_No_Time_Limit` |
| Grounding F1 | `Word_Grounding` | `Word_Grounding_No_Time_Limit` |
| Page-level grounding F1 | `Page_Grounding` | `Page_Grounding_No_Time_Limit` |

Scores are percentages. Use an empty cell for an unreported metric; zero means a measured zero. Then follow the shared steps in [Submitting results](../README.md#submitting-results).
