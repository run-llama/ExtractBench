# Jev + LiteParse extraction framework

The frozen framework reached **43.7753% development value F1 on 12 documents**, at **$0.127376676** recorded API cost. Four rounds of three variants are complete. This is a development result; full-corpus, holdout, and leaderboard acceptance are not claimed here. See [RESULTS.md](RESULTS.md) for complete results, evaluator distinctions, limitations, and final-report placeholders.

Jev selects among explicit source-value, region, column, and row alternatives. Python assembles the requested JSON; no generative model fills missing values. LiteParse 2.14.4 is the only PDF parser, including Markdown table structure and word geometry.

## Frozen final route

The `router` variant chooses `anchors` when the parsed source contains at least 100 non-separator Markdown table rows and the public schema contains an array of objects. Otherwise it chooses `compact`. The rule uses source structure and schema, not filenames, document identities, labels, or schema-hash lookup. It was selected on development data and needs final evaluation.

`anchors` preserves Markdown empty cells and wrapped values, falls back to generic record anchors, and batches row repair. `compact` uses localized source retrieval with concise field questions. Nested arrays inside array records remain unsupported; correct values absent from source candidates cannot be recovered. The complete frozen dependencies and PDF/schema hashes are recorded in [freeze.json](freeze.json), created at 2026-09-22 14:18:26 UTC.

## Four experiment rounds

| Round | Three variants |
|---|---|
| 1 | `fields`: source candidates; `hierarchical`: region then span; `tables`: contiguous-table columns |
| 2 | `geometric`: word positions; `localized`: local questions; `multipage`: full-document table enumeration |
| 3 | `hybrid`: source-size routing; `boundaries`: start/end spans; `rowrepair`: batched irregular-row repair |
| 4 | `compact`: concise scalar questions; `consensus`: candidate union; `anchors`: Markdown tables and record boundaries |

The final router was freshly validated after correcting blank-cell repair. Its development artifacts are preserved in `results/router_development/`; historical variant outputs retain their original results. Source snapshots are retained under `source_snapshots/`. Markdown cache format `jev-markdown-words-v2` augments existing source text and geometry; the two output modes were checked for identical plain text/geometry on the 55-page development source.

## Evaluation protocol

`manifest.json` reuses 12 development and 12 schema-disjoint designated holdout documents. The prior Luna study inspected the holdout labels: this is a holdout for Jev selection, not an unseen dataset for the broader project. The development set is a coverage sample, not a representative estimate of the 370-document leaderboard. Full-corpus reporting must disclose the 12-document development overlap.

Inference receives only parsed source and public schema. Predictions are saved before labels are loaded by the evaluator; failures score zero. Local metrics and separately pinned upstream rescoring are reported independently. Upstream revision is `11937d102cad9d9b815044888d55e7ce343d3924`; dataset revision is `f6180e917a050a84582e6366cff85b7dc1e84e58`. Do not compare incomplete rescore averages with full 12-document means.

## Setup and execution

```bash
.venv/bin/pip install -e '.[jev]'
# Set OPENROUTER_API_KEY in the environment or repository .env.
.venv/bin/python learning/jev/run.py router --partition dev
.venv/bin/python learning/jev/run.py router --partition holdout
.venv/bin/python learning/jev/run.py router --partition full --workers 3
```

These runner commands can make paid requests when matching saved results do not exist. Held-out/full execution requires the matching frozen source, model, and PDF/schema hashes. Results are written under `results/<variant>/`; saved matching artifacts are reused. The registered production provider is `jev_liteparse`, defaulting to `router`, and requires an extraction request with a schema and local source PDF. Its configurable artifact directory defaults to `output/jev_liteparse`.

The working experiment request was `typesafe/jev-1.13` at `/api/alpha/decisions`, observed as `typesafe/jev-1.13-20260917`. The initial `typesafe/jev-latest` request returned HTTP 400. Experiment-time listed input pricing was $0.042/M tokens, with zero output charge.

## Budget and artifacts

Study workers share `api/budget.sqlite` with a $9 cap. Requests reserve a conservative amount before execution, then settle against reported provider cost. Failed or unknown attempts retain reservations. Transient failures retry only their batch, up to three attempts, with separate reservations. The client batches at 24 questions or roughly 26 KB and defaults to a 2,000-call/1,800-second document guard. Do not delete the ledger to restart a run. Separate production artifact directories have separate ledgers.

API archives omit authentication headers but contain source questions and decisions. Parsed documents, raw API responses, and benchmark artifacts are local study outputs. A $0.000014532 smoke probe predates the ledger. Report final settled spend separately from unresolved reservations, diagnostics, and this probe. See [RESULTS.md](RESULTS.md) for production limitations and the pending final-evaluation checklist.
