# Contributing

## Submitting benchmark results

Open a pull request in [run-llama/ExtractBench](https://github.com/run-llama/ExtractBench/pulls) with scores and enough evidence to reproduce them. Both new systems and updates to existing systems are welcome.

| Benchmark | Result file | Evaluation set |
|---|---|---|
| ExtractBench | [leaderboard.csv](leaderboard.csv) | All 370 documents |
| ExtractBench Pro | [leaderboard-pro.csv](leaderboard-pro.csv) | All 100 documents across six challenges |

Use a distinct provider/configuration name when changing the model, harness, or effort setting. For an update to an existing configuration, include the previous and new run dates and describe what changed.

### Run and score

Install your provider extra and configure its API key as described in the [README](README.md#configuration). To add a provider, see [Evaluating Your Own System](README.md#usage).

For original ExtractBench:

```bash
uv run extract-bench run your_pipeline
```

For Pro, use a 30-minute deadline per document and no timeout retries:

```bash
uv run extract-bench download --dataset pro
uv run extract-bench inference run your_pipeline data-pro --output_dir output-pro --per_file_timeout 1800 --timeout_retries 0
uv run extract-bench run your_pipeline --dataset pro --skip_inference
uv run extract-bench serve output-pro/your_pipeline
```

Replace `your_pipeline` with a registered pipeline name. Inference calls your provider's paid API. The second `run` command evaluates the saved predictions without calling that API again. Execution failures, missing outputs, and documents that exceed the deadline remain in the denominator with zero scores. Do not select the best output from repeated attempts. Disclose any infrastructure recovery and preserve failed-attempt records.

### Update the leaderboard

1. Update the appropriate CSV. For Pro, include one `all` row and one row for each of the six challenges; see the [column definitions](docs/pro-leaderboard.md#csv-format). Use the scorer's per-document metrics, not rule pass rate. All scores are percentages and costs are USD per page. Leave unavailable metrics or costs blank.
2. Run `uv run python scripts/update_readme.py` to regenerate the README tables.
3. Run `uv run python scripts/update_readme.py --check` and `git diff --check`.
4. Include the following in your PR:
   - Dataset repository and revision from `data/.extract_bench_manifest.json` or `data-pro/.extract_bench_manifest.json`, and the harness commit/version.
   - Provider/model version, harness, effort settings, prompts, tools, concurrency, timeout and retry settings, and evaluation date. Include the pipeline implementation/configuration or a reproducible script.
   - Evaluation reports (`_evaluation_report.json` for each split), per-document predictions/citations, and execution errors, as attachments or a durable downloadable artifact. Keep API keys and account credentials out of artifacts.
   - Document counts, failures/timeouts, and how costs were measured, including failed attempts and any missing usage. Distinguish measured usage from an estimate based on published prices.

Use the [benchmark result PR template](https://github.com/run-llama/ExtractBench/compare?expand=1&template=benchmark-result.md) when opening your PR. Maintainers review evidence before merging leaderboard updates. The benchmark websites are updated separately; a CSV edit does not automatically publish a website change.
