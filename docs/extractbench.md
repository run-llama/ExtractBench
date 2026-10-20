# ExtractBench

Details specific to the original ExtractBench benchmark. Installation, download, usage, metrics and configuration shared with ExtractBench Pro are in the [README](../README.md).

- **Dataset:** [`llamaindex/ExtractBench`](https://huggingface.co/datasets/llamaindex/ExtractBench)
- **Paper:** [arXiv 2607.29677](https://arxiv.org/abs/2607.29677)
- **Launch post:** [Introducing ExtractBench](https://www.llamaindex.ai/blog/introducing-extractbench)
- **Leaderboard:** [README](../README.md#extractbench-leaderboard), full table in [leaderboard.csv](../leaderboard.csv)

## Overview

The benchmark covers **370 documents (4,869 pages)** across 8 business domains and 67 document types, each type with its own schema. Every document is tagged along five independent axes: task challenge, perception challenge, table structure, length, and business domain, so a low score can be traced to its cause.

## Splits

The benchmark is split by document length, with one JSONL row per (document, schema) test case plus the source PDFs:

| Split | File | Documents | Pages | Length |
|-------|------|----------:|------:|--------|
| **Short** | `short.jsonl` | 252 | 615 | ≤10 pages |
| **Medium** | `medium.jsonl` | 98 | 2,438 | 11–50 pages |
| **Long** | `long.jsonl` | 20 | 1,816 | >50 pages |
| **Total** | | **370** | **4,869** | |

Run one split with `--group short`, `--group medium`, or `--group long`. The `--test` subset has 6 documents: 3 short, 2 medium, and 1 long.

## Domains

The benchmark spans 8 business domains and 67 document types: finance and fund holdings, energy-sector regulatory forms, government procurement and customs, auto valuation, supply chain, healthcare remittance, legal and bankruptcy filings, and real estate.

## Tag axes

**What each task challenge tests:**

- **T1: long-list completeness.** Recover *every* record of a repeated structure that can span many pages. Typical failures are truncation, duplicated or merged rows, hallucinated records, and values attached to the wrong record.
- **T2: needle-in-haystack.** Find a small number of requested facts in a long document. T2 has few target records but many plausible mentions, only one of which is canonical; failures are missed targets, wrong occurrences, and unnormalized paraphrases.
- **T3: dense documents.** Fill many fields from a document dense with labels, blanks, checkboxes, handwriting, and scan artifacts. The characteristic failure is over-extraction, inventing a value for a field that is actually blank, compounded by missed checkboxes and mislabeled fields.

A document can carry more than one task challenge.

The other axes are tagged independently of the task challenge:

- **Table structure** — S1 merged headers, S2 header not at top / pivoted, S3 cross-page table, S4 enormous table, S5 table within a cell.
- **Perception challenge** — P1 rotated or image-only capture, P2 scanned page images, and P3 handwriting. 38 documents are degraded re-captures of documents that also appear clean, so capture degradation is a paired measurement on the same documents.
- **Business domain** — D1 finance (145), D2 energy (98), D3 government (49), D4 automotive (27), D5 supply chain (20), D6 healthcare (15), D7 legal (10), D8 real estate (6).

## Sources

All documents come from public records: SEC and regulatory filings, government procurement and customs forms, court and agency exhibits (including tax forms such as W-2, 1040, K-1, and 1099-B), Texas Railroad Commission energy filings, and published business documents. 325 are real; 45 are synthetic long lists rendered from real layouts. PDF metadata has been stripped from every file.

## Ground truth

Ground truth uses a method matched to each source: adjudicated agreement across independent extraction systems for real documents, values fixed before rendering for synthetic long lists, and human-verified values and boxes for forms. Each field's ground truth is an *evidence list* — the expected value plus any alternate acceptable readings, each with its source location — and scoring accepts a match against any listed reading.

## Leaderboard pipelines

The 14 paper baselines and their pipeline names:

| Pipeline name | Name in paper |
|---------------|---------------|
| `llamaextract_agentic_plus` | LlamaExtract Agentic Plus |
| `llamaextract_agentic` | LlamaExtract Agentic |
| `llamaextract_cost_effective` | LlamaExtract Cost-Effective |
| `reducto_deep_extract` | Reducto Deep Extract |
| `extend_extract_max` | Extend (Max Context) |
| `datalab_parse_accurate_extract_balanced` | Datalab (Accurate + Balanced) |
| `codex_code_extract_gpt_5_5_low` | Codex (GPT-5.5) |
| `claude_code_extract_opus_4_8` | Claude Code (Opus 4.8) |
| `qwen3_6_35b_a3b_fp8_vllm_extract_oneshot_structured_output_file` | Qwen3.6 35B (self-hosted) |
| `gemma4_26b_vllm_extract_oneshot_structured_output_file` | Gemma4 26B (self-hosted) |
| `nuextract3_extract` | NuExtract3 (self-hosted) |
| `lift_extract` | Lift Datalab 9B (self-hosted) |
| `gemini_3_5_flash_extract_oneshot_structured_output_file` | Google Gemini 3.5 Flash |
| `openai_gpt_5_4_nano_extract_oneshot_structured_output_file` | OpenAI GPT-5.4 Nano |

The four self-hosted pipelines need an endpoint you run yourself; see [.env.example](../.env.example). Other configurations of the same systems are registered too (`reducto_extract`, `extend_extract`, `codex_code_extract_gpt_5_5_high`, the two-stage parse baselines, and more). Run `uv run extract-bench pipelines` for the full roster, or see [pipelines.md](pipelines.md).

The three LlamaExtract leaderboard pipelines select Extract v2.6 with citations enabled and no custom Parse configuration. All three return word-level citation boxes.

## Cost of a full run

Costs use each provider's official listed price.

| Category | One full run | Examples |
|---|---:|---|
| Commercial VLM | $10 – $49 | GPT-5.4 Nano ~$10, Gemini 3.5 Flash ~$49 |
| LlamaExtract | $49 – $395 | Cost-Effective ~$49, Agentic ~$152, Agentic Plus ~$395 |
| Specialized APIs | $170 – $487 | Datalab ~$170, Reducto Deep Extract ~$309, Extend ~$487 |
| Coding agents | $787 – $1,355 | Claude Code (Opus 4.8) ~$787, Codex (GPT-5.5) ~$1,355 |
| Self-hosted open weights | GPU time only | Qwen3.6 35B, Gemma4 26B, NuExtract3, Lift Datalab 9B |

These figures apply the reported mean cost per page to 4,869 pages. Per-system, per-split prices are in [leaderboard.csv](../leaderboard.csv) (`Cost_Per_Page`, `Cost_Short`, `Cost_Medium`, `Cost_Long`, in dollars per page).

## Results layout

```
output/<pipeline_name>/
├── short/      # one directory per split
├── medium/
└── long/
```

See "Output Structure" under [Configuration](../README.md#configuration) for the files in each split directory.

## Submitting results

Add one row per system to [leaderboard.csv](../leaderboard.csv). Each row carries value F1 (`Overall`, `Short`, `Medium`, `Long`), precision and recall per split (`P_*`, `R_*`), word- and page-level grounding F1 overall and per split, cost per page overall and per split, and latency per document per split. Then follow the shared steps in [Submitting results](../README.md#submitting-results).
