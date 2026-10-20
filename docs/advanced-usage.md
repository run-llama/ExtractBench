# Advanced usage

Options that apply to both ExtractBench and ExtractBench Pro. Basic usage is in the [README](../README.md#usage).

## Reproducible submission order

Pass `--shuffle_seed 0` to `extract-bench run`, `pipeline run`, or `inference run`
to shuffle test cases after sorting their IDs. The same seed and selected IDs give
all pipelines the same planned order, independent of filesystem discovery order.
Omit the flag to preserve discovery order. Each inference output directory saves
`_submission_order.json` with the seed and ordered IDs; cached results are still
skipped normally. Parallel requests can complete in a different order.

## Limit concurrent source pages

With the `local` (or `runners`) extra installed, add `--max_concurrent_pages 100`
alongside `--max_concurrent 8` to bound both documents and source PDF pages.
The default is no page limit. This option requires readable, unencrypted PDFs;
all source page counts are checked before any submissions.

Admission follows input order (or the seeded shuffle). It waits for the next
PDF to fit rather than backfilling smaller PDFs. A PDF larger than the page
budget runs alone. Capacity stays reserved through provider processing,
polling, and retries; cached documents consume no capacity. Document errors
are recorded and subsequent documents continue. These limits bound local
attempts: timed-out remote jobs may still be running after capacity is released.

Each run saves `_admission_plan.json` (limits and source page counts) and
`_admission_events.jsonl` (timestamped admissions, releases, and active counts).
Resuming archives the previous plan and event log under `_admission_history/`
before creating the current session's files, including when cached documents
are skipped. These describe client admission, not backend worker capacity.

## Row identity hints

For `.test.json` annotations, `_identity_hints` can declare row identity beside `data_schema`, for example `"_identity_hints": {"holdings": ["cusip"]}`. Array-record metrics use these hints to partition top-level arrays only when the residual assignment exceeds 100 million cells after exact-row matching. Each bucket keeps the same cap; smaller assignments remain keyless. Hints are evaluation-only and are not added to the schema sent to providers.
