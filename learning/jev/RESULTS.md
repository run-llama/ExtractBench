# Jev + LiteParse: budget-limited benchmark results

The frozen framework scored **42.0072% value F1 across 370 cases** under the
study budget. It produced 355 completed predictions; **15 long documents remain
budget failures and count as zero**. This is a budget-limited measurement, not
a completed inference run or an accepted leaderboard entry.

Four rounds evaluated three variants each. A source/schema router was selected
and frozen before final evaluation. It scored 43.7753% on 12 development cases
and 41.8274% on the designated 12-case selection holdout. The full benchmark
includes the development cases; excluding those 12 yields 42.1866% on 358 cases.
Even perfect predictions for the 15 remaining cases could raise the full score
only to 46.0613%. Accuracy remains well below the leading leaderboard systems.

## Budget-limited corpus results

| Split | Cases | Budget failures | Value F1 (%) | Recorded API cost |
|---|---:|---:|---:|---:|
| Short | 252 | 0 | 47.4334 | $2.579566 |
| Medium | 98 | 0 | 35.8503 | $3.798022 |
| Long | 20 | 15 | 3.8067 | $0.905579 |
| Overall | 370 | 15 | **42.0072** | **$7.283167** |

Overall precision is 49.0176% and recall 42.4919%. Official word grounding is
8.2029% over 239 metric-bearing rows; page grounding is 27.4118% over 295.
The official runner pads observed grounding metrics with zero for every failed
document, including failures without grounding annotations. This report follows
that behavior. The cost-per-page headline is **$0.00458134**, the unweighted mean
of each document's cost divided by its page count, not total cost / 4,869 pages.
It includes paid work on the failed documents. Latency is not submitted because
parsing was cached, citation conversion was offline, and some jobs were resumed.

The native archive roundtrips through the unchanged official evaluation runner:
355 successes, 15 failures, zero skipped; all 20 aggregate value/grounding means
match exactly. All 1,040 repository tests pass, including 50 Jev-focused tests.

[Full report](final/report.json), [per-document results](final/per_document.csv),
[native predictions](final/native_results.jsonl.gz), and
[roundtrip verification](final/native_roundtrip.json) contain the auditable data.

## Evaluation protocol

The primary number is the arithmetic mean of per-document `extract_unified_value_f1`, multiplied by 100. Failures contribute zero; documents, not fields or pages, receive equal weight. The development set comprises eight short, three medium, and one long document. It is a selected coverage sample, not a representative estimate of the 370-document benchmark.

Inference receives the public schema and LiteParse source output. It does not receive benchmark labels. Predictions are persisted before the evaluator loads labels. Twelve variants were implemented in four rounds of three; a source/schema router was then assembled and freshly validated on the same 12 development documents. Consequently, the development result is selection-biased. The full-corpus measurement includes those 12 development documents.

The designated 12-document holdout is schema-disjoint from development. The previous Luna study inspected its labels: it is withheld from this Jev selection process, but is not an unseen dataset for the broader project. The holdout results were not used for further quality tuning.

## Completed development experiments

These figures are recomputed from saved per-document artifacts, rather than the older summary files. Every local row below includes all 12 development documents. The upstream column is shown only for complete 12-document rescores available when this report was prepared.

| Round | Variant | Local value F1 (%) | Upstream value F1 (%) | Local failures | Recorded API cost |
|---|---|---:|---:|---:|---:|
| 1 | fields | 38.1994 | 38.1994 | 0 | $0.312761820 |
| 1 | hierarchical | 39.9879 | 39.9879 | 1 | $0.442601250 |
| 1 | tables | 38.1984 | 38.3251 | 0 | $0.140700168 |
| 2 | geometric | 33.4334 | 33.4334 | 1 | $0.181002108 |
| 2 | localized | 39.0046 | 39.0046 | 0 | $0.086933028 |
| 2 | multipage | 32.5528 | 32.6390 | 0 | $0.132771870 |
| 3 | hybrid | 34.5498 | 34.5498 | 0 | $0.117169164 |
| 3 | boundaries | 33.5240 | 33.5240 | 0 | $0.101497452 |
| 3 | rowrepair | 39.0041 | 39.0110 | 0 | $0.403477578 |
| 4 | compact | 40.0260 | 40.0260 | 0 | $0.079932552 |
| 4 | consensus | 39.5863 | 39.5863 | 0 | $0.097213242 |
| 4 | anchors | 35.5522 | 35.5522 | 0 | $0.239457036 |
| Final development validation | router | **43.7753** | **43.7753** | **0** | **$0.127376676** |

The local worktree is based on commit `5d04ac6d62a98c6864ac37c21a832eacdfd6240e`; separately pinned upstream evaluation uses `11937d102cad9d9b815044888d55e7ce343d3924`. The small `tables` and `multipage` differences demonstrate why these evaluator columns must remain distinct. All 12 variants and the final-development router now have complete 12-document upstream rescores; `rowrepair` also has a small local/upstream difference (39.0041 versus 39.0110). The preserved final-development artifacts are in `results/router_development/`. Historical round artifacts retain their original predictions and costs. The published `development/` archive contains all 156 development predictions and metrics. One early `localized` source hash has no retained code snapshot; this provenance gap is recorded in `development/summary.json`. The final frozen framework has complete source hashes.

Grounded metrics require a separate statement of eligible-document and citation coverage denominators. The upstream summarizer's `word_f1` field averages only records with `extract_unified_grounded_f1` available; it is not a zero-filled full-corpus mean and is not reported here as a leaderboard score.

## Framework and route selection

Jev makes typed decisions over explicit alternatives; Python copies source spans, converts supported primitive values, and assembles JSON. No generative LLM supplies missing values. LiteParse **2.14.4** is the only PDF parser, providing source text, word boxes, and Markdown.

Round 1 compared field candidates, hierarchical source selection, and contiguous-table column mapping. Round 2 tested word geometry, localized questions, and enumeration of tables across pages. Round 3 tested source-size routing, explicit text boundaries, and batched repair of irregular rows. Round 4 compared compact questions, candidate-set union, and Markdown tables with record anchors.

Two findings informed the final route. Concise questions improved ordinary-source decisions while preserving schema constraints. LiteParse Markdown retained empty columns and joined wrapped cell text that plain-text/geometry assembly had lost. However, using the table engine everywhere overproduced records on some short documents. The router therefore uses **anchors only when the source has at least 100 non-separator Markdown table rows and the schema contains a reachable array of objects**; all other sources use compact extraction. The row count includes printed headers. Routing uses no filenames, document IDs, labels, or schema-hash lookup. The threshold and engines were selected using development evidence, so their measured generalization is reported above.

Anchors prefers Markdown tables, falls back to generic identifier/date/numeric record boundaries, and reuses column decisions when explicit headers match. It retains all source pages and iterates all candidate rows; candidate generation and final row decisions can still miss records. Compact retains the localized extractor's source selection and its contiguous-table array fallback. Thus the final router does not guarantee complete multi-table coverage on every document below the threshold.

Before fresh final-development validation, a confirmed assembly bug was fixed: a known blank cell in a complete row now remains null instead of triggering neighboring-value repair. Historical rowrepair/anchors outputs were preserved. Citation caching subsequently changed performance only; old/new outputs matched in an offline repeated-value comparison and all citation tests passed. Previous source snapshots and the freeze hashes identify these versions.

## Production constraints and limitations

The registered `jev_liteparse` provider defaults to `router`. It accepts extraction requests with an explicit public JSON schema, a local source PDF, and `OPENROUTER_API_KEY`. The pinned parser extra is available through `pip install -e '.[jev]'`. Source parsing/cache and API artifacts are written beneath the configurable artifact directory. The provider rejects an empty root extraction, but does not perform a complete JSON Schema validation of every emitted value.

The observed working model request is `typesafe/jev-1.13`, resolving to `typesafe/jev-1.13-20260917`, through OpenRouter `/api/alpha/decisions`. The initial `typesafe/jev-latest` probe failed. Experiment-time listed pricing was $0.042 per million input tokens and zero output charge; these are recorded observations, not a guarantee of future availability or pricing.

The frozen client splits batches at 24 questions or roughly 26 KB of serialized state/questions, rejects payloads over 100 KB, and uses a 90-second HTTP timeout. It permits up to three attempts for a failed transient batch, reserving each attempt separately; permanent failures propagate. Default per-document guards are 2,000 requests and 1,800 seconds. These are operational guards, not guarantees that every large document finishes. The shared $9 ledger is scoped to an artifact directory; separate directories do not share a global cap. The provider does not itself enforce the experimental source/data freeze checks; the study runner does.

Arrays nested inside array records are unsupported by the table leaf mapper: their properties can be omitted, and arrays of arrays can become empty. Arrays beneath ordinary objects are supported. Missing candidate values cannot be generated by Jev; retrieval limits, parser errors, and row classification remain recall bottlenecks. Repair can still select an incorrect neighboring value when a row is genuinely ambiguous. There is no universal support for all JSON Schema composition/validation semantics.

Citations restrict boxes to local evidence and preserve page-only attribution for ambiguous same-page occurrences. Ambiguous source pages receive no citation. Numeric signs and whole-word matching are checked. This conservative strategy leaves some correct values ungrounded, particularly normalized or merged text. Citation confidence is not a calibrated document-level correctness guarantee.

## Budget-only continuation

The initial shared $9 study cap included all development experiments and interrupted
18 of the 370 final documents. The other 352 produced completed predictions.
The continuation raised the cumulative study ceiling to $9.75, within the initial
$9.823701469 available credits. It selects only budget-interrupted documents,
replays their 961 previously paid API attempts in exact request-hash order,
and requests only the remaining decisions. An offline pass through all 18
frozen extractors verified this replay before any continuation calls were made.
Three interrupted documents completed; 15 remained budget-limited at the new ceiling.
Successful document predictions were not rerun. All original and new charges
remain included in per-document and study totals.

The extraction implementation, source/schema hashes, and requested model remain
frozen. The operational continuation starts a fresh 1,800-second allowance for
each interrupted document, while old and new attempts together remain subject
to the 2,000-call limit. These timing changes are disclosed; resumed document
latency is unavailable and leaderboard latency is left blank. Immutable original
failure records, replay hashes, wrapper hashes, and outcomes are retained under
`operational_resumes/`. The normal provider's cumulative default cap remains $9
for a fresh artifact directory.

## Freeze and spend reconciliation

[freeze.json](freeze.json) records source hashes, document PDF/schema hashes, LiteParse version, model IDs, and these revisions:

- Dataset: `f6180e917a050a84582e6366cff85b7dc1e84e58`.
- Upstream evaluator: `11937d102cad9d9b815044888d55e7ce343d3924`.
- Selection: four rounds of three variants, followed by a source/schema-only router, with no subsequent tuning.

The study made 49,355 API attempts. All 49,354 responses with model metadata
resolved to `typesafe/jev-1.13-20260917`; the remaining attempt was the historical
oversized-request failure. Every archived request hash and settled cost reconciles.

- Reported provider charges, including the pre-ledger probe: **$9.746107644**.
- Unresolved conservative reservation: **$0.003493014**.
- Conservative total study accounting: **$9.749600658**.
- Development and diagnostics ledger accounting: **$2.466419088**.
- Final-corpus API accounting, including all interrupted/resumed work: **$7.283167038**.

Full-corpus costs and call counts exactly equal ledger entries after the frozen
run boundary (call 11,449). See [finance reconciliation](final/finance.json) and
[API audit](final/api_audit.json). No additional paid calls are active. Completing
the remaining long-document attempts requires additional OpenRouter credit.
The draft submission will explicitly label the current measurement as budget-limited;
maintainer acceptance is pending.

## Submission status

[Draft leaderboard PR #61](https://github.com/run-llama/ExtractBench/pull/61)
was opened on September 22, 2026. It proposes an explicitly budget-limited row;
acceptance and the decision on funding the remaining attempts are pending.
