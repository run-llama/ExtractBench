# Harbor datasets

The exporter adds `tasks/<case-id>/` to the existing ExtractBench and
ExtractBenchPro Hugging Face repositories. It generates tasks from their public
JSONL rows and PDFs. It preserves the original JSONL bytes and dataset-card
configuration, and adds `rl-environment` and `harbor` discovery tags.

Each task uses the shared Claude Code/Codex prompt published in
[ExtractBench Appendix C.1](https://arxiv.org/html/2607.29677v1#A3.SS1), with
`./document.pdf` substituted for the staged document filename. The complete
case schema is inserted without modifying its fields or descriptions.

## Generate a pinned export

Run from an ExtractBench checkout with `uv sync --extra dev` completed:

```bash
uv run python -m extract_bench.harbor.export \
  --dataset extractbench \
  --revision FULL_HF_DATASET_COMMIT_SHA \
  --scorer-revision FULL_EXTRACTBENCH_GIT_COMMIT_SHA \
  --output output/harbor/ExtractBench

uv run python -m extract_bench.harbor.export \
  --dataset pro \
  --revision FULL_HF_PRO_COMMIT_SHA \
  --scorer-revision FULL_EXTRACTBENCH_GIT_COMMIT_SHA \
  --output output/harbor/ExtractBenchPro
```

Both revisions must be full commit hashes. The scorer revision must be reachable
from the public `run-llama/ExtractBench` repository. The verifier image installs
that exact revision and `parse-bench==1.0.4`; the small Harbor adapter is bundled
with each task. There is no copied scoring implementation.

The Hugging Face cache makes downloads resumable. `--snapshot PATH` uses an
already downloaded snapshot instead. The output is an overlay for the same
dataset repository: existing `docs/`, thumbnails, JSONL configuration and other
files remain in the repository. The exporter never prunes existing source data.
Rerunning repairs incomplete task files and writes the completed manifest last.
If source cases were removed, use a fresh staging directory; stale task folders
are rejected rather than silently retained.

`harbor-manifest.json` records source/scorer revisions, JSONL SHA-256 hashes,
the original ID for every task, and every generated task file's hash. Directory
names are bounded lowercase slugs with a source-ID hash to avoid collisions.

## Run and validate

Install Harbor 0.24.0 or later and start Docker. Remote `--repo` runs also require
Git LFS; install it and run `git lfs install` before the first remote run so the
source PDFs are downloaded. For a local packaging smoke test:

```bash
harbor run -p output/harbor/ExtractBench/tasks -a oracle -l 1 -n 1
harbor run -p output/harbor/ExtractBenchPro/tasks -a oracle -l 1 -n 1
```

The oracle copies the already public reference output. It makes no model calls
and verifies the container, output transfer and evaluator wiring, not whether
an extraction agent can solve the case. `-a nop` exercises missing output, which
must receive zero. Real extraction runs use the desired agent and model, e.g.:

```bash
harbor run --repo https://huggingface.co/datasets/llamaindex/ExtractBench -p tasks -a codex -m MODEL
harbor run --repo https://huggingface.co/datasets/llamaindex/ExtractBenchPro -p tasks -a codex -m MODEL
```

`uv run python scripts/smoke_harbor.py` runs both agents on one real case from
each dataset and checks the trial rewards, including Harbor's per-trial errors
(a job process can exit successfully despite a failed trial). The `Harbor smoke`
PR check runs this on a clean Docker runner. Its reduced fixtures are only for
testing and must not be published as full datasets.

These remote commands become usable when the generated tasks are published.
For evaluation, pin a dataset repository commit with Harbor's `@COMMIT` syntax
and preserve the Harbor job configuration alongside the export manifest.

## Scoring and isolation

The agent image contains only the document, schema, PDF/OCR tools and Python.
`tests/` and `solution/` are outside its Docker build context. The separate
verifier receives only the declared `/app/output.json` artifact; its reference
and grading code are in its own image. The verifier calls
`ExtractEvaluator.compute_metrics`, including its existing normalization,
acceptable-value and row-matching logic.

`/logs/verifier/reward.txt` contains `extract_unified_value_f1` in [0, 1].
`metrics.json` contains the value precision/recall/F1 and perfect-document flag.
Average Value F1 across all cases gives macro Value F1; average the perfect flags
to obtain the perfect-document fraction. Keep failed trials in those denominators.
Malformed, missing or non-object outputs score zero. A reference or evaluator
error fails grading visibly and does not produce a reward.

`feedback.json` attributes that same reward to scored fields and table cells.
It records `reference_path`, `output_path`, `outcome`, and `f1_credit` for each
cell. The outcomes are `matched`, `value_differs`, `reference_only`, and
`prediction_only`. One-sided cells have a null path on the absent side; an
unmatched row produces an entry for each of its scored leaves. Primitive
arrays remain single cells, following the evaluator. Reference values are not
included in this feedback.

For example, when table rows are reordered, a matching cell may be reported as:

```json
{
  "reference_path": "items[0].amount",
  "output_path": "items[3].amount",
  "outcome": "matched",
  "f1_credit": 0.02
}
```

The scorer emits these records during the same assignment that computes Value
F1. Each match receives `2 / (expected_cells + predicted_cells)` credit; other
cells receive zero. The credits sum to the document reward, within floating
point precision. `counts` includes the expected, predicted, and matched totals;
one-sided cells contribute to the denominator. Credit explains the scored
submission, not the counterfactual reward change from editing a field: an edit
can change both the denominator and row assignment. This is output attribution,
not credit for reasoning steps or tokens.

Paths address the **normalized scoring inputs**: schema defaults, list
normalization, and EOB layout reconciliation can change the representation from
the submitted JSON. A defaulted field can therefore receive credit even if
omitted from the submitted file. Acceptable values and normalizers use the
existing evaluator's comparison rules. The report preserves `value_incomplete`
when the scorer limits an oversized assignment; feedback is then an explanation
of that lower-bound score. Invalid submissions produce an empty cell list and
an error status. A failed grader removes stale feedback and produces no reward.

The published prompt does not ask for citations, so this lane does not report
grounding quality. It does not impose a new schema-validation penalty beyond
the existing evaluator. Timeouts are 1,200 seconds for original ExtractBench
(the paper's coding-agent protocol) and 1,800 seconds for Pro (its released
benchmark limit). Verifier time is separate from agent time.

Before publishing, validate all task configurations, compare representative
predictions through the normal evaluator and verifier, run the oracle and nop
smokes, and load every staged JSONL split with Hugging Face `datasets`. Review
the two Hugging Face diffs: only `tasks/`, `harbor-manifest.json`, the README
tags/Harbor section, and an LFS rule for task PDFs in `.gitattributes` should change.

After validation and review, authenticate with `hf auth login`, then run:

```bash
uv run python -m extract_bench.harbor.publish output/harbor/ExtractBench
uv run python -m extract_bench.harbor.publish output/harbor/ExtractBenchPro
```

The command verifies task hashes and opens a Hugging Face dataset PR using
`create_pr=True` and a `parent_commit` guard against the pinned source revision.
Only `README.md`, `.gitattributes`, `harbor-manifest.json`, and `tasks/**` are
uploaded; original JSONL/PDF source files are never uploaded or replaced.
