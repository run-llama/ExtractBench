# Maintaining Harbor exports

For downloading and running the published tasks, see [Harbor usage](harbor.md).
This guide covers regenerating and publishing the official dataset overlays.

The exporter reconstructs Harbor tasks from ExtractBench-format JSONL rows and
PDFs, preserving source data and recording source/scorer revisions and file
hashes. It is tooling for maintainers and users preparing their own compatible
data; running the published datasets does not require it. The publish command
is restricted to the two official Hugging Face repositories.

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
Pro task names use only the document identifier below the split, including for
the hash, so moving a document between challenges does not rename its task.
The source case ID, split, and challenge tags remain in task metadata.
Duplicate document identifiers across Pro splits are rejected before export.

## Validate an export

Run local packaging checks without model calls:

```bash
harbor run -p output/harbor/ExtractBench/tasks -a oracle -l 1 -n 1
harbor run -p output/harbor/ExtractBenchPro/tasks -a oracle -l 1 -n 1
```

The oracle copies the public reference output, checking packaging and grading.
Use `-a nop` to check that a missing output scores zero.

`uv run --with pymupdf==1.26.6 python scripts/smoke_harbor.py` resolves the current
HF `main` revisions on every run. It exports and validates every case: source
JSONL preservation, PDF readability, schema/reference agreement, perfect
reference rewards, and cell-credit accounting. It builds a wheel from the
current checkout and substitutes only that package source in the production
verifier Dockerfile.
When the HF revision contains published Harbor tasks, their hashes, coverage,
inputs, and references are also checked against the current HF rows, so a data
update without matching task regeneration fails.
One case per dataset also runs oracle/nop through Harbor, including checks for
per-trial errors (a job can exit successfully despite a failed trial). The oracle
also checks that its allowed model API host is reachable without credentials,
while Hugging Face, GitHub, and a direct public IP are blocked during execution.

The `Harbor smoke` workflow runs on relevant PRs/main changes and on
manual dispatch. Its artifacts record resolved HF revisions, the tested wheel
hash, per-case results, and container results. Exact wheel/reference hashes
permit interrupted local audits to resume. Use `--extractbench-revision` and
`--pro-revision` to reproduce a reported revision; defaults track `main`.
These checks establish data/export/scorer consistency, not an independent
annotation review against the PDFs. Smoke container tasks install a local wheel
and must not be published as dataset tasks.

## Publish an export

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

## Canonical Pro split-name migration

All six original split names are replaced by `interpretation_rules`, `custom_terminology`, `conditional_aggregation`, `relational_diagrams`, `charts_and_maps`, and `long_single_pages`. `scripts/rename_pro_splits.py` stages the renamed JSONL/PDF paths and records the complete case mapping plus unchanged PDF SHA-256 hashes. It never publishes. Verify source row payloads and PDFs, then review the data commit before regenerating Harbor against that commit. The HF migration must remove the original split files, PDF paths, and superseded task directories; an ordinary overlay upload alone does not perform those deletions.

The new Pro task directories omit the challenge component from both the slug and its hash. All 100 task names change, including the Interpretation rules tasks. No aliases are provided for the original Harbor names. Validate the same PDFs, schemas, prompt, reference answers, verifier, and scorer pin before releasing the new packages.

Prepare an unpublished Hub candidate from the validated export:

```bash
uv run --with harbor==0.24.0 python scripts/stage_pro_hub.py \
  --export <validated-canonical-export> \
  --output <fresh-hub-staging-directory> \
  --hf-revision <full-reviewed-HF-commit>
```

The script adds explicit `llamaindex/<canonical-task-id>` package names with schema 1.4, verifies the execution settings, and writes content-pinned task references to `dataset.toml`. Review it before any separate `harbor publish` operation. Publishing the candidate changes the Hub release and requires release approval.
